"""P59 feasibility audit and one frozen local-matching comparison; no model fits."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import subprocess

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'src', ROOT/'noran/experiments'):
    sys.path.insert(0,str(path))
import numpy as np
import pandas as pd
from pipeline import load, locate_data, table
from prayer_experiments import timetable
from noran.tracks.p59_local_match import model as M


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/p59')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if (out/'result.json').exists():
        raise FileExistsError('Use a new output directory; previous results are preserved')
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    deadline=started+480
    manifest=json.loads((ROOT/'models/version7/manifest.json').read_text(encoding='utf-8'))
    protected={p:sha(ROOT/p) for p in manifest['assets']}
    for name,expected in manifest['inputs'].items():
        assert sha(ROOT/'data/raw'/name)==expected
    for path,expected in manifest['assets'].items():
        assert protected[path]==expected
    refpath=ROOT/'outputs/version7/reference/submission_v7_board_0.724.csv'
    assert sha(refpath)==manifest['accepted_submission_sha256']
    train,test,_,_=load(locate_data(None))
    schedule=timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)
    spec=importlib.util.spec_from_file_location('p59_cases',ROOT/'noran/experiments/prompt54_hf.py')
    P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    saved=json.loads((ROOT/'outputs/p58_verified/verification.json').read_text())['hashes']
    query_rows,match_rows,selected_rows=[],[],[]
    bundles={}
    for which in ('fixed','mirror','testday','actual'):
        assert time.perf_counter()<deadline
        cases,feeds,labels=P.protocol(which,train,test)
        if which!='actual':
            cache=ROOT/'outputs/p58_verified'/which/'predictions.csv'
            assert sha(cache)==saved[str(cache.relative_to(ROOT))], 'Baseline cache changed'
            baseline=pd.read_csv(cache)
            assert baseline.case_id.is_unique and set(cases.case_id)==set(baseline.case_id)
            masks={rec:list(M.C.HIDDEN_WITH[g.unit.iloc[0]]) for rec,g in cases.groupby('recording')}
        else:
            baseline=None
            masks={'20260918_1100':['cam_B','cam_G'],'20260919_1100':['cam_D']}
        library=train.copy()
        for rec,hidden in masks.items():
            library=library[~(library.recording.eq(rec)&library.unit.isin(hidden))]
            assert not (library.recording.eq(rec)&library.unit.isin(hidden)).any()
        perday={day:M.library(library[library.day.ne(day)],schedule) for day in cases.day.unique()}
        selections={}
        checks=0
        for r in cases.drop_duplicates(['recording','unit','now']).itertuples():
            assert time.perf_counter()<deadline
            model=perday[r.day]
            selected,candidates=M.select(r,feeds[r.recording],model)
            key=(r.recording,r.unit,int(r.now))
            selections[key]=selected
            for kind,modified in [('hidden',feeds[r.recording].copy()),('post_now',feeds[r.recording].copy())]:
                if kind=='hidden':
                    for unit in M.C.HIDDEN_WITH[r.unit]: modified[unit]=1e8
                else:
                    modified.loc[modified.index>int(r.now)]=1e8
                again,_=M.select(r,modified,model)
                assert selected==again, (which,r.case_id,kind)
                checks+=1
            for candidate in candidates:
                match_rows.append(dict(protocol=which,query_recording=r.recording,unit=r.unit,now=int(r.now),**candidate))
            for candidate in selected:
                selected_rows.append(dict(protocol=which,query_recording=r.recording,unit=r.unit,now=int(r.now),**candidate))
            query_rows.append(dict(protocol=which,recording=r.recording,unit=r.unit,now=int(r.now),
                candidates=len(candidates),source_recordings=len({c['recording'] for c in candidates}),
                source_dates=len({c['day'] for c in candidates}),source_events=len({c['event'] for c in candidates}),
                accepted_events=len(selected),supported=len(selected)>=M.SPEC['neighbors'],
                best_distance=min([c['distance'] for c in candidates],default=np.nan),
                selected_distance=float(np.mean([c['distance'] for c in selected])) if selected else np.nan,
                overlap_min=min([c['min_pairs'] for c in selected],default=0)))
        bundles[which]=(cases,feeds,labels,baseline,perday,selections,checks)
        print(f'{which}: feasibility complete, {len(selections)} camera windows',flush=True)
    queries=pd.DataFrame(query_rows)
    queries.to_csv(out/'feasibility_by_window.csv',index=False)
    pd.DataFrame(match_rows).to_csv(out/'eligible_segments.csv',index=False)
    pd.DataFrame(selected_rows).to_csv(out/'selected_matches.csv',index=False)
    feasibility=queries.groupby(['protocol','unit']).agg(windows=('now','size'),supported=('supported','sum'),
          candidates_mean=('candidates','mean'),events_max=('source_events','max'),events_mean=('source_events','mean'),
          best_shape_mean=('best_distance','mean'),selected_shape_mean=('selected_distance','mean'),overlap_min=('overlap_min','min')).reset_index()
    feasibility.to_csv(out/'feasibility_summary.csv',index=False)
    print(feasibility.to_string(index=False),flush=True)
    # Do not proceed unless multiple-event support exists for both assessment and
    # actual competition requests. This gate uses context only, never answers.
    feasible=bool(queries[queries.protocol.eq('actual')].supported.any() and
                  queries[queries.protocol.ne('actual')].supported.any())
    results=[]
    if feasible:
        for which in ('fixed','mirror','testday'):
            cases,feeds,labels,base,models,selections,checks=bundles[which]
            folder=out/which;folder.mkdir(exist_ok=True)
            base=base.set_index('case_id')
            rows=[]
            for r in cases.itertuples():
                assert time.perf_counter()<deadline
                baseline=float(base.at[r.case_id,'baseline'])
                prediction=baseline
                selected=selections[(r.recording,r.unit,int(r.now))]
                used=False
                reason='insufficient_context_events'
                asr,maghrib=int(schedule[r.day][2]),int(schedule[r.day][3])
                if asr+55<=int(r.now)<=maghrib-35:
                    reason='v7_afternoon_fallback'
                elif len(selected)>=M.SPEC['neighbors']:
                    # Target continuations are accessed here, after all matching
                    # has been selected and persisted independently of labels.
                    analog=M.continuation(r,selected,models[r.day])
                    p=dict(M.v7.PARAMS)
                    v7model=dict(models[r.day],params=p)
                    original=M.v7.analog_value(feeds[r.recording].loc[:int(r.now)],r.unit,int(r.now),
                        int(r.minute-r.now),str(r.day),v7model,schedule)
                    if np.isfinite(analog) and np.isfinite(original):
                        prediction=max(0.,baseline+0.8*(analog-original))
                        used=True;reason='matched'
                    else: reason='continuation_or_v7_analog_unavailable'
                rows.append(dict(case_id=r.case_id,day=r.day,recording=r.recording,unit=r.unit,minute=int(r.minute),
                    now=int(r.now),horizon=int(r.minute-r.now),baseline=baseline,prediction=prediction,
                    reference=float(base.at[r.case_id,'reference']),used=used,changed=prediction!=baseline,reason=reason))
            detail=pd.DataFrame(rows)
            assert np.isfinite(detail[['prediction','baseline','reference']]).all().all() and (detail.prediction>=0).all()
            detail.to_csv(folder/'predictions.csv',index=False)
            d=detail.merge(labels,on='case_id',validate='one_to_one')
            for col in ('baseline','prediction','reference'): d['err_'+col]=(d[col]-d.truth).abs()
            d['band']=pd.cut(d.horizon,[0,5,10,15],labels=['1-5','6-10','11-15'])
            d['window']=d.recording+'|'+d.unit+'|'+d.now.astype(str)
            d.to_csv(folder/'scored.csv',index=False)
            for name,keys in [('recording_camera',['recording','unit']),('horizon',['recording','unit','band']),('windows',['recording','unit','now'])]:
                g=d.groupby(keys,observed=True).agg(n=('truth','size'),baseline_mae=('err_baseline','mean'),
                    new_mae=('err_prediction','mean'),reference_mae=('err_reference','mean'),
                    changed=('changed','sum'),matched=('used','sum'),windows=('window','nunique')).reset_index()
                g['baseline_norm']=g.baseline_mae/g.reference_mae
                g['new_norm']=g.new_mae/g.reference_mae
                g.to_csv(folder/(name+'.csv'),index=False)
            days=d.groupby('day').agg(base=('err_baseline','mean'),new=('err_prediction','mean'),ref=('err_reference','mean'))
            w=d.groupby('window').agg(base=('err_baseline','sum'),new=('err_prediction','sum'),n=('truth','size'))
            w['gain']=w.base-w.new
            sensitivity={}
            for count in (0,1,2):
                keep=w.sort_values('gain',ascending=False).iloc[count:]
                sensitivity[str(count)]=float((keep.base.sum()-keep.new.sum())/keep.n.sum())
            result=dict(protocol=which,rows=len(d),changed=int(d.changed.sum()),matched=int(d.used.sum()),
                fallback=int((~d.used).sum()),camera_windows=d.window.nunique(),
                temporal_windows=len(d.drop_duplicates(['recording','now'])),
                baseline_mae=float(d.err_baseline.mean()),new_mae=float(d.err_prediction.mean()),
                baseline_norm=float((days.base/days.ref).mean()),new_norm=float((days.new/days.ref).mean()),
                gain_remove_best_windows=sensitivity,mask_postnow_selection_checks=checks,
                fallback_reasons=d.reason.value_counts().to_dict())
            (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            results.append(result)
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip()
    for path,digest in protected.items(): assert sha(ROOT/path)==digest
    assert sha(refpath)==manifest['accepted_submission_sha256']
    result=dict(status='complete' if feasible else 'stopped_insufficient_context_support',
        feasible=feasible,recipe=M.SPEC,results=results,runtime_seconds=round(time.perf_counter()-started,2),
        candidate_created=False,independent_validation=False,
        selection_invariance_checks=sum(b[-1] for b in bundles.values()),
        reference_sha256=sha(refpath),protected_assets=len(protected),
        protected_tracked_files_changed=False)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
