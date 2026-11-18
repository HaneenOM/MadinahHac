"""P48: isolated zero-shot experiment. No production imports in inference mode.

prepare/score use the baseline environment; pilot/infer use the isolated venv.
No fitting, automatic blending, submission writing, or model selection.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/prompt48_chronos'
DESIGN = dict(model='amazon/chronos-bolt-small', revision='772f3d25d38aec6d914c8949dab4462e2d46f5d8',
              device='cpu', dtype='float32',
              max_context=2048, prediction_length=15, quantile=.5, batch_size=32,
              threads=4, minimum_observed=2, training='none; zero-shot',
              missing='NaN mask on unchanged one-minute grid; no imputation',
              context='same recording and camera, first minute through now inclusive',
              fallback='unchanged paired accepted-method prediction',
              selection='One frozen model/settings; no label-based tuning or blending')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x, indent=2), encoding='utf-8')

def context(frame, unit, first, now):
    import numpy as np
    import pandas as pd
    grid = pd.RangeIndex(max(first, now-DESIGN['max_context']+1), now+1)
    s = frame[unit] if unit in frame else pd.Series(dtype=float)
    return s.reindex(grid).to_numpy(dtype=np.float32)

def cached_forecast(setting, cases, labels, feeds):
    """Prove compatibility of old task functions rather than whole HIDDEN-changed file."""
    import pandas as pd
    import evaluation_artifacts as a
    a.validated_entries(setting, allow_legacy=True)
    folder = ROOT / f'outputs/prompt38_{setting}'
    m = json.loads((folder/'run_manifest.json').read_text())
    assert m['inputs'] == a.context()['inputs']
    # All inference code and reference code other than task file must match.
    paths = ['src/pipeline.py']
    paths += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'t05_forecast').glob('*.py')]
    for p in paths:
        assert sha(ROOT/p) == m['code'][p.replace('/', '\\')], p
    def fns(s):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    for path,names in [('src/baseline_tasks.py',('condition_forecast','forecast_value')),
                       ('src/reliable_evaluation.py',('forecast_predictions','prefix_record'))]:
        expected = m['code'][path.replace('/', '\\')]
        old = None
        for rev in subprocess.check_output(['git','log','--format=%H','--',path],cwd=ROOT,text=True).splitlines():
            b = subprocess.check_output(['git','show',f'{rev}:{path}'],cwd=ROOT)
            if expected in [hashlib.sha256(x).hexdigest() for x in (b,b.replace(b'\n',b'\r\n'))]:
                old = b.decode(); break
        if old is None:
            # Old run code was not preserved in Git. Recompute inference only;
            # never pretend a successful asset load proves prediction provenance.
            import reliable_evaluation as ev
            import baseline_tasks as tasks
            from t05_forecast.core import load_recordings
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                fresh,_=ev.forecast_predictions(cases,feeds,load_recordings(ROOT/'data/raw'),
                    setting,a.validated_entries(setting,allow_legacy=True),tasks.default_config()['forecast'])
            fresh.to_csv(OUT/f'{setting}_fresh_t05.csv',index=False)
            return fresh,dict(predictions='fresh isolated inference; old prediction code hash unavailable',
                reused_covariances=True,t05_fits=0,legacy_provenance='incomplete; conditional diagnostic only')
        current = (ROOT/path).read_text()
        for name in names: assert fns(old)[name] == fns(current)[name],name
    saved = pd.read_csv(folder/'predictions.csv')
    saved = saved[saved.kind.eq('forecast')]
    assert set(saved.case_id) == set(cases.case_id)
    oldlabels = pd.read_csv(folder/'labels.csv').set_index('case_id').truth
    pd.testing.assert_series_equal(labels.set_index('case_id').truth.sort_index(),
                                   oldlabels.reindex(labels.case_id).sort_index(), check_names=False)
    return saved, dict(path=str(folder.relative_to(ROOT)), prediction_sha256=sha(folder/'predictions.csv'),
                       task_ast_verified=True, legacy_provenance='incomplete; conditional diagnostic only')

def prepare():
    import numpy as np
    import pandas as pd
    sys.path[:0] = [str(ROOT), str(ROOT/'src')]
    import baseline_tasks as tasks
    import reliable_evaluation as ev
    from pipeline import load, table
    from evaluate_baseline_task import fixed_benchmark
    manifest = json.loads((ROOT/'models/current_baseline/manifest.json').read_text())
    assert sha(ROOT/'outputs/submission_hidden_bgd_v1.csv') == manifest['accepted_submission_sha256']
    protected = [p for p in (ROOT/'outputs').rglob('submission*.csv')]
    protected += list((ROOT/'src').glob('*.py')) + list((ROOT/'models').rglob('*.*'))
    protected += [ROOT/'notebooks/current_baseline.ipynb']
    dump(OUT/'protected.json', {str(p.relative_to(ROOT)):sha(p) for p in protected})
    dump(OUT/'design.json', DESIGN)
    train,test,_,_ = load(ROOT/'data/raw')
    c,l,f = fixed_benchmark(train,test,prefer_saved=False)
    known = json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())
    assert ev.case_fingerprints(c,l) == known['historical']
    settings = {'historical':(c,l,f)}
    c,l = ev.released_cases(test)
    settings['representative'] = c,l,table(test[test.id.isna() & test.people.notna()])
    c,l = ev.im_cases(test)
    settings['im-proxy'] = c,l,settings['representative'][2]
    requests = {}; arrays = {}; cache = {}; coverage = []
    for setting,(c,l,feeds) in settings.items():
        assert ev.case_fingerprints(c,l) == known[setting]
        c = c[c.kind.eq('forecast')].copy()
        l = l[l.case_id.isin(c.case_id)]
        c = ev.annotate_cases(c,feeds)
        c['horizon_band'] = pd.cut(c.horizon,[0,5,10,15],labels=['1-5','6-10','11-15']).astype(str)
        if setting != 'im-proxy': base,cache[setting] = cached_forecast(setting,c,l,feeds)
        else: base,_ = ev.im_predictions(c,feeds,tasks.default_config()['forecast'])
        base = base[['case_id','prediction','reference']].rename(columns={'prediction':'baseline'})
        frame = c.merge(base,on='case_id',validate='one_to_one')
        reqkeys = []
        for r in frame.itertuples():
            key = f'{setting}|{r.recording}|{r.unit}|{int(r.now)}'
            reqkeys.append(key)
            if key in requests: continue
            first = int(test[test.recording.eq(r.recording)].minute_of_day.min()) if setting!='historical' else int(train[train.recording.eq(r.recording)].minute_of_day.min())
            x = context(feeds[r.recording],r.unit,first,int(r.now))
            # Future poisoning and recording isolation: context function must not read past now.
            poisoned = feeds[r.recording].copy(); poisoned.loc[poisoned.index>r.now] = 1e8
            np.testing.assert_array_equal(x,context(poisoned,r.unit,first,int(r.now)))
            assert len(x) == min(DESIGN['max_context'],int(r.now)-first+1)
            idx = f'x{len(requests):05d}'
            arrays[idx] = x
            n = int(np.isfinite(x).sum())
            requests[key] = dict(key=key,array=idx,setting=setting,recording=r.recording,unit=r.unit,
                now=int(r.now),context_length=len(x),observed=n,missing_fraction=float(np.isnan(x).mean()),
                latest_age=int(len(x)-1-np.where(np.isfinite(x))[0][-1]) if n else None,
                fallback_reason='fewer_than_two_observations' if n<2 else '')
        frame['request_key'] = reqkeys
        frame.to_csv(OUT/f'{setting}_cases.csv',index=False)
        l.to_csv(OUT/f'{setting}_labels.csv',index=False)
    # Actual target availability is audited without predictions or answers.
    actual = test[test.kind.eq('forecast')]
    feeds = settings['representative'][2]
    for r in actual.itertuples():
        first = int(test[test.recording.eq(r.recording)].minute_of_day.min())
        x = context(feeds[r.recording],r.unit,first,int(r.now_minute_of_day))
        coverage.append(dict(recording=r.recording,unit=r.unit,context_length=len(x),observed=int(np.isfinite(x).sum()),
                             missing_fraction=float(np.isnan(x).mean())))
    pd.DataFrame(coverage).to_csv(OUT/'actual_context_coverage.csv',index=False)
    np.savez_compressed(OUT/'contexts.npz',**arrays)
    dump(OUT/'requests.json',list(requests.values()))
    dump(OUT/'cache_verification.json',cache)
    print(json.dumps(dict(requests=len(requests),contexts_sha256=sha(OUT/'contexts.npz'),cache=cache),indent=2))

def infer(pilot=False):
    import importlib.metadata as md
    import numpy as np
    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download
    torch.set_num_threads(DESIGN['threads'])
    started=time.perf_counter()
    req=json.loads((OUT/'requests.json').read_text())
    data=np.load(OUT/'contexts.npz')
    # Freeze the downloaded snapshot before inference; full runs require pilot success.
    pin=OUT/'model_pin.json'
    if pin.exists(): model_path=str(ROOT/json.loads(pin.read_text())['snapshot'])
    else:
        model_path=snapshot_download(DESIGN['model'],revision=DESIGN['revision'],cache_dir=str(OUT/'hub'),allow_patterns=['config.json','model.safetensors'])
        dump(pin,dict(model=DESIGN['model'],revision=Path(model_path).name,snapshot=str(Path(model_path).relative_to(ROOT)),
                      weights_sha256=sha(Path(model_path)/'model.safetensors')))
    assert Path(model_path).name==DESIGN['revision']
    assert sha(Path(model_path)/'model.safetensors')==json.loads(pin.read_text())['weights_sha256']
    pipeline=BaseChronosPipeline.from_pretrained(model_path,device_map='cpu',torch_dtype=torch.float32)
    load_seconds=time.perf_counter()-started
    valid=[r for r in req if not r['fallback_reason']]
    if pilot:
        # Input metadata only: lengths/missingness extremes plus ordinary contexts.
        chosen={r['key']:r for r in sorted(valid,key=lambda r:r['context_length'])[:4]}
        for r in sorted(valid,key=lambda r:r['context_length'])[-4:]: chosen[r['key']]=r
        for r in sorted(valid,key=lambda r:r['missing_fraction'],reverse=True)[:4]: chosen[r['key']]=r
        valid=list(chosen.values())
    else:
        assert json.loads((OUT/'pilot.json').read_text())['passed']
    rows=[];inference_start=time.perf_counter()
    for start in range(0,len(valid),DESIGN['batch_size']):
        batch=valid[start:start+DESIGN['batch_size']]
        xs=[torch.tensor(data[r['array']]) for r in batch]
        # 1.5.x names this argument context. Quantile .5 is the MAE-aligned point estimate.
        q=pipeline.predict(context=xs,prediction_length=15).cpu().numpy()
        qi=list(pipeline.quantiles).index(.5)
        for r,ys in zip(batch,q[:,qi,:]):
            if len(ys)!=15 or not np.isfinite(ys).all():
                reason='invalid_model_output'; values=[None]*15
            else: reason='';values=np.maximum(ys,0).astype(float).tolist()
            rows.append(dict(key=r['key'],prediction=values,fallback_reason=reason))
    infer_seconds=time.perf_counter()-inference_start
    if pilot:
        # Verify horizon position and native internal/trailing NaN support structurally.
        passed=all(not r['fallback_reason'] for r in rows)
        estimate=infer_seconds/max(len(rows),1)*len(req)
        passed=passed and estimate<1200
        dump(OUT/'pilot.json',dict(passed=passed,requests=len(rows),load_seconds=load_seconds,
              inference_seconds=infer_seconds,estimated_full_seconds=estimate,
              input_metadata=[r for r in valid],outputs=rows))
        if not passed: raise RuntimeError('Pilot failed correctness/runtime gate')
    else:
        rows += [dict(key=r['key'],prediction=[None]*15,fallback_reason=r['fallback_reason']) for r in req if r['fallback_reason']]
        dump(OUT/'chronos_predictions.json',rows)
        dump(OUT/'inference_manifest.json',dict(design=DESIGN,requests=len(rows),load_seconds=load_seconds,
            inference_seconds=infer_seconds,context_sha256=sha(OUT/'contexts.npz'),source_sha256=sha(__file__),
            versions={p:md.version(p) for p in ('torch','transformers','chronos-forecasting','numpy','accelerate','huggingface-hub')}))
    print(json.dumps(dict(pilot=pilot,requests=len(rows),load_seconds=load_seconds,inference_seconds=infer_seconds)))

def score():
    import numpy as np
    import pandas as pd
    preds={r['key']:r for r in json.loads((OUT/'chronos_predictions.json').read_text())}
    requests=pd.DataFrame(json.loads((OUT/'requests.json').read_text())).rename(columns={'key':'request_key'})
    outputs=[]
    for setting in ('representative','historical','im-proxy'):
        c=pd.read_csv(OUT/f'{setting}_cases.csv')
        raw=[];reasons=[]
        for r in c.itertuples():
            p=preds[r.request_key];raw.append(p['prediction'][int(r.horizon)-1]);reasons.append(p['fallback_reason'])
        c['raw_chronos']=pd.to_numeric(pd.Series(raw),errors='coerce')
        c['fallback_reason']=reasons
        c['alternative']=c.raw_chronos.fillna(c.baseline)
        c['changed']=c.raw_chronos.notna() & c.alternative.ne(c.baseline)
        c=c.merge(requests.drop(columns=['setting','recording','unit','now']),on='request_key',validate='many_to_one')
        c=c.merge(pd.read_csv(OUT/f'{setting}_labels.csv'),on='case_id',validate='one_to_one')
        c['paired']=np.isfinite(c[['baseline','alternative','reference','truth']]).all(axis=1)
        for col in ('baseline','alternative','reference'): c[col+'_error']=(c[col]-c.truth).abs()
        c.to_csv(OUT/f'{setting}_scored.csv',index=False)
        def metric(g):
            p=g[g.paired];b=p.baseline_error.mean();v=p.alternative_error.mean();ref=p.reference_error.mean()
            return dict(cases=len(g),paired=len(p),unique_targets=g.target_id.nunique(),paired_unique_targets=p.target_id.nunique(),
                origins=g[['recording','now']].drop_duplicates().shape[0],
                windows=g.window_id.nunique(),paired_windows=p.window_id.nunique(),
                model_rows=int(g.raw_chronos.notna().sum()),fallback_rows=int(g.raw_chronos.isna().sum()),
                changed=int(g.changed.sum()),baseline_mae=b,alternative_mae=v,reference_mae=ref,
                baseline_ratio=b/ref if ref else None,alternative_ratio=v/ref if ref else None,
                gain_percent=100*(b-v)/b if b else None)
        summary=[]
        for dims in ([],['day'],['day','unit'],['day','horizon_band'],['day','unit','horizon_band'],['recording','window_id']):
            groups=[((),c)] if not dims else c.groupby(dims,dropna=False)
            for keys,g in groups:
                keys=keys if isinstance(keys,tuple) else (keys,)
                summary.append(dict(setting=setting,group='/'.join(dims) or 'all',**dict(zip(dims,keys)),**metric(g)))
        pd.DataFrame(summary).to_csv(OUT/f'{setting}_summary.csv',index=False)
        changed=metric(c[c.changed]); outputs.append(dict(setting=setting,all=metric(c),changed=changed))
        sensitivity=[]
        for day,g in [('all',c),*list(c.groupby('day'))]:
            gains=g[g.paired].groupby('window_id').apply(lambda q:float((q.baseline_error-q.alternative_error).sum()))
            best=gains.sort_values(ascending=False).index.tolist()
            for k in (0,1,2):
                sensitivity.append(dict(day=day,remove_best=k,removed=best[:k],**metric(g[~g.window_id.isin(best[:k])])) )
            for w in gains.index:
                sensitivity.append(dict(day=day,remove_best='leave_one',removed=[w],**metric(g[g.window_id.ne(w)])))
        dump(OUT/f'{setting}_sensitivity.json',sensitivity)
        per_camera=[]
        for (day,unit),g in c.groupby(['day','unit']):
            gains=g[g.paired].groupby('window_id').apply(lambda q:float((q.baseline_error-q.alternative_error).sum()))
            best=gains.sort_values(ascending=False).index.tolist()
            for k in (0,1,2):
                per_camera.append(dict(day=day,unit=unit,remove_best=k,
                    winning_windows=int((gains>0).sum()),removed=best[:k],**metric(g[~g.window_id.isin(best[:k])])) )
        dump(OUT/f'{setting}_camera_sensitivity.json',per_camera)
    protected=json.loads((OUT/'protected.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in protected.items())
    dump(OUT/'verification.json',dict(protected_files=len(protected),all_unchanged=True,
        accepted_submission_sha256=sha(ROOT/'outputs/submission_hidden_bgd_v1.csv'),
        context_sha256=sha(OUT/'contexts.npz'),source_sha256=sha(__file__),
        chronos_fits=0,t05_fits=0,submission_created=False,
        context_checks='Each unique context passed whole-table future poisoning and exact minute-grid length checks',
        assessment='All dates examined before; no fresh validation; no model/feature search'))
    dump(OUT/'results.json',outputs)
    print(json.dumps(outputs,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','pilot','infer','score'])
    mode=parser.parse_args().mode
    OUT.mkdir(parents=True,exist_ok=True)
    if mode=='prepare': prepare()
    elif mode=='score': score()
    else: infer(mode=='pilot')
