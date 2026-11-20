"""P61 reproducible isolated development and joint-exclusion assessment."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, importlib.util, json, os, sys, time
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'noran/experiments'): sys.path.insert(0,str(p))
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import prompt56 as P
from pipeline import load,locate_data,table
import baseline_tasks as tasks

OUT = ROOT/'outputs/prompt61'
M = P.mod('noran/tracks/p61_missing/model.py','p61_missing')
START = 1795111765. # 2026-11-19 18:09:25 UTC
ASSESSMENT_DAYS = ('2026-09-11','2026-09-12')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def check_time():
    if time.time()>START+4800: raise TimeoutError('Final ten minutes reserved')


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protected.json'
    if not path.exists():
        paths=[]
        for folder in ('configs','models','notebooks','src','t05_forecast','data/raw','noran/tracks'):
            paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'p61_' not in str(p))
        paths.append(ROOT/'outputs/version7/reference/submission_v7_board_0.724.csv')
        save(path,{str(p.relative_to(ROOT)):sha(p) for p in paths})
    protected=json.loads(path.read_text())
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    manifest=json.loads((ROOT/'models/version7/manifest.json').read_text())
    assert sha(ROOT/'outputs/version7/reference/submission_v7_board_0.724.csv')==manifest['accepted_submission_sha256']
    for p,h in manifest['inputs'].items(): assert sha(ROOT/'data/raw'/p)==h
    session=OUT/'session.json'
    if not session.exists(): save(session,dict(start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(START)),
        deadline_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(START+5400)),
        experiment_cutoff_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(START+4800)),protected_count=len(protected),
        previous_work='P54 rejected timing; P56 G/D smoothing exploratory, convex missing and I-M forecast rejected; P58 slope rejected; P59 unsupported fixed/mirror; P60 diagnosis only',
        numerical_reproduction='3158 unequal float values; max forecast 6.681995e-6, hidden 4.4872853e-5, missing 7.958e-13; not counted as gain'))
    return len(protected)


def inputs():
    train,test,_,_=load(locate_data(None))
    old=train[~train.day.isin(ASSESSMENT_DAYS)].copy()
    rel=test[test.id.isna()&test.people.notna()].copy()
    return train,test,old,rel,dict(config=tasks.default_config()['missing'],test_released=rel)


def crossfit_v2(rows,training,config):
    """Never use the pseudo-source's answers in its generated calibration."""
    rows=rows.copy()
    for source,g in rows.groupby('source'):
        cal=tasks.fit_missing(training[training.recording.ne(source)],config)
        rows.loc[g.index,'v2']=M.v7.stack.v2_values(g,cal,config)
    return rows


def pseudo(old,ctx):
    # The full training and release frames are in the key: no stale scope reuse.
    digest=hashlib.sha256(pd.util.hash_pandas_object(old,index=True).values.tobytes()+
        pd.util.hash_pandas_object(ctx['test_released'],index=True).values.tobytes()+
        (ROOT/'noran/tracks/missing/model.py').read_bytes()).hexdigest()
    path=OUT/f'pseudo_{digest[:16]}.pkl'
    if path.exists(): return pd.read_pickle(path)
    tick=time.perf_counter()
    rows=M.v7.pseudo_rows(old,ctx['test_released'],ctx['config'])
    assert not rows.source.str.startswith(('20260911','20260912')).any()
    rows.to_pickle(path)
    save(OUT/'pseudo_scope.json',dict(sha256=sha(path),cache_key=digest,rows=len(rows),
        sources=rows.groupby('source').size().to_dict(),seconds=time.perf_counter()-tick,
        excluded_dates=ASSESSMENT_DAYS))
    return rows


def summarize(d):
    d=d.copy()
    for c in ('v7','trial','reference'): d['e_'+c]=(d[c]-d.truth).abs()
    d['gain']=d.e_v7-d.e_trial
    daily=d.groupby('day')[['e_v7','e_trial','e_reference']].mean()
    events=d.groupby('event').gain.sum().sort_values(ascending=False)
    sensitivity=[]
    for n in (0,1,2):
        rem=d[~d.event.isin(events.head(n).index)]
        sensitivity.append(dict(best_events_removed=n,mae_gain=float(rem.gain.mean())))
    return dict(rows=len(d),v7_mae=d.e_v7.mean(),trial_mae=d.e_trial.mean(),
        v7_ratio=(daily.e_v7/daily.e_reference).mean(),trial_ratio=(daily.e_trial/daily.e_reference).mean(),
        changed=int((abs(d.trial-d.v7)>1e-8).sum()),events=len(events),
        wins=int((events>1e-8).sum()),losses=int((events< -1e-8).sum()),sensitivity=sensitivity),d


def development():
    check_time();initialize();started=time.perf_counter()
    if (OUT/'development_result.json').exists(): raise FileExistsError('Development already complete')
    _,_,old,rel,ctx=inputs();rows=pseudo(old,ctx)
    config_path=getattr(M,'CONFIG_PATH',ROOT/'noran/configs/prompt61.json')
    save(OUT/'recipe_lock.json',dict(config=sha(config_path),config_path=str(config_path.relative_to(ROOT)),
        model=sha(M.__file__),model_path=str(Path(M.__file__).relative_to(ROOT)),
        selection='fixed 50% blend; recording-held-out pseudo development; training-only gate added after first aggregate fold, before camera results or assessment'))
    parts=[];folds=[]
    for source in sorted(rows.source.unique()):
        check_time();tick=time.perf_counter()
        fit_rows=rows[rows.source.ne(source)].copy().reset_index(drop=True)
        training=old[old.recording.ne(source)].copy()
        fit_rows=crossfit_v2(fit_rows,training,ctx['config'])
        holdout=rows[rows.source.eq(source)].copy().reset_index(drop=True)
        model=M.from_rows(fit_rows,training,ctx)
        baseline=M.baseline_values(holdout,model['baseline'])
        extra=(M.nonlinear_values(holdout,model['nonlinear'],baseline) if hasattr(M,'direct') else
               M.nonlinear_values(holdout,model['nonlinear']))
        d=holdout[['source','unit','minute0','a','b','m','span','n_left','n_right','truth','base']].copy()
        d['v7']=baseline;d['trial']=(1-M.SETTINGS['blend'])*baseline+M.SETTINGS['blend']*extra
        d['nonlinear']=extra;d['reference']=d.base;d['day']=source[:8]
        # minute0+a/b identifies the physical gap; shifts/windows are dependent variants.
        d['event']=source+'|'+(d.minute0+d.a).astype(str)+'|'+(d.minute0+d.b).astype(str)
        parts.append(d)
        result,scored=summarize(d)
        folds.append(dict(source=source,seconds=time.perf_counter()-tick,**result))
        print('DEVELOPMENT',json.dumps(folds[-1],default=str),flush=True)
    pred=pd.concat(parts,ignore_index=True);summary,scored=summarize(pred)
    scored.to_csv(OUT/'development_scored.csv',index=False)
    by_camera=scored.groupby('unit').agg(n=('truth','size'),v7=('e_v7','mean'),trial=('e_trial','mean'),gain=('gain','mean'))
    by_camera.to_csv(OUT/'development_camera.csv')
    for source,g in scored.groupby('source'):
        g.groupby('unit').agg(n=('truth','size'),v7=('e_v7','mean'),trial=('e_trial','mean'),gain=('gain','mean')).to_csv(OUT/f'development_{source}_camera.csv')
    result=dict(**summary,folds=folds,runtime_seconds=time.perf_counter()-started,
        overlap_warning='Pseudo rows overlap heavily; source recording is primary unit. Event variants are not independent samples.')
    save(OUT/'development_result.json',result)
    print(json.dumps(result,default=str,indent=2),flush=True)


def assessment():
    check_time();initialize();started=time.perf_counter()
    if (OUT/'assessment_result.json').exists(): raise FileExistsError('Assessment already complete')
    selected=json.loads((OUT/'selection.json').read_text())['selected_units']
    if not selected: raise RuntimeError('No training-supported camera scope to assess')
    train,test,old,rel,ctx=inputs();rows=pseudo(old,ctx)
    model=M.from_rows(rows,old,ctx)
    fit_seconds=time.perf_counter()-started
    results=[];check_count=0
    for which in ('fixed','mirror'):
        cases,feeds,labels=P.protocols(which,'missing',train,test)
        poisoned=train.copy()
        for rec,t in feeds.items():
            allowed={(u,int(m)) for u in t for m in t.index[t[u].notna()]}
            indices=poisoned.index[poisoned.recording.eq(rec)]
            bad=[idx for idx in indices if (poisoned.at[idx,'unit'],int(poisoned.at[idx,'minute_of_day'])) not in allowed]
            poisoned.loc[bad,'people']=1e8
        _,altered_feeds,_=P.protocols(which,'missing',poisoned,test)
        for rec in feeds: pd.testing.assert_frame_equal(feeds[rec],altered_feeds[rec])
        pd.testing.assert_frame_equal(old,poisoned[~poisoned.day.isin(ASSESSMENT_DAYS)])
        predictions=[]
        for r in cases.itertuples():
            t=feeds[r.recording]
            v=M.v7.predict(r,t,model['baseline'],ctx)
            ref,old_value,_=tasks.missing_value(t,r.unit,int(r.minute),model['baseline']['calibration'],ctx['config'])
            v=float(v) if np.isfinite(v) else old_value
            p_all=M.predict(r,t,model,ctx)
            p=p_all if r.unit in selected else v
            raw=M.inference_rows(r,t)
            copied=M.baseline_values(raw,model['baseline'])[0]
            np.testing.assert_allclose(copied,v,atol=1e-8,rtol=0)
            bounds=M.v7.gap_context(t,r.unit,int(r.minute))
            predictions.append(dict(case_id=r.case_id,day=r.day,recording=r.recording,unit=r.unit,
                minute=r.minute,v7=v,trial=p,trial_unrestricted=p_all,reference=ref,gap_start=int(bounds[0]),gap_end=int(bounds[1]),
                event=f'{r.recording}|{bounds[0]}|{bounds[1]}'))
            if r.Index%20==0:
                # Observable cells only; taint unknown cells then restore the release mask.
                altered=t.copy();unknown=altered.isna();altered=altered.fillna(1e8).mask(unknown)
                np.testing.assert_allclose(p_all,M.predict(r,altered,model,ctx),atol=1e-8,rtol=0)
                check_count+=1
        d=pd.DataFrame(predictions)
        d.to_csv(OUT/f'{which}_predictions.csv',index=False) # prediction freeze precedes label join
        d=d.merge(labels,on='case_id',validate='one_to_one')
        result,scored=summarize(d)
        scored.to_csv(OUT/f'{which}_scored.csv',index=False)
        scored.groupby(['recording','unit']).agg(n=('truth','size'),v7=('e_v7','mean'),trial=('e_trial','mean'),gain=('gain','mean')).to_csv(OUT/f'{which}_camera.csv')
        scored.groupby('event').agg(n=('truth','size'),v7=('e_v7','mean'),trial=('e_trial','mean'),gain=('gain','sum')).to_csv(OUT/f'{which}_events.csv')
        results.append(dict(protocol=which,selected_units=selected,**result))
        print('ASSESSMENT',json.dumps(results[-1],default=str),flush=True)
    save(OUT/'assessment_result.json',dict(results=results,fit_seconds=fit_seconds,checks=check_count,
        runtime_seconds=time.perf_counter()-started,baseline='Unmodified V7 refitted with both Sep 11/12 absent; identical training scope to candidate',
        limitation='Fixed/mirror share physical answers, inspected in earlier sessions. Neither is fresh independent validation.'))
    initialize()


def selection():
    """Training-only applicability gate, frozen before either assessment run."""
    if (OUT/'assessment_result.json').exists(): raise RuntimeError('Cannot select after assessment')
    d=pd.read_csv(OUT/'development_scored.csv')
    audit=[]
    for unit,g in d.groupby('unit'):
        sources=g.groupby('source').agg(v7=('e_v7','mean'),trial=('e_trial','mean'))
        gains=sources.v7-sources.trial
        events=g.groupby('event').gain.sum().sort_values(ascending=False)
        rem=g[~g.event.isin(events.head(2).index)]
        use=len(sources)>=2 and (gains>0).sum()>=np.ceil(2*len(sources)/3) and gains.mean()>0 and rem.gain.mean()>0
        audit.append(dict(unit=unit,source_recordings=len(sources),positive_sources=int((gains>0).sum()),
            equal_source_gain=gains.mean(),gain=g.gain.mean(),gain_without_best_two=rem.gain.mean(),selected=bool(use)))
    result=dict(protocol='Preassessment training-only gate: >=2 sources, >=2/3 positive sources, positive equal-source gain and after top-two physical gap removal',
        selected_units=[r['unit'] for r in audit if r['selected']],audit=audit,
        development_sha256=sha(OUT/'development_scored.csv'))
    save(OUT/'selection.json',result);print(json.dumps(result,indent=2,default=str))


def main():
    global OUT,START,M
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['develop','assess','select','check'])
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--method',choices=['direct','residual','weighted','adapted'],default='direct')
    args=parser.parse_args()
    chosen=args.output_dir.resolve()
    if chosen!=OUT:
        OUT=chosen
        session_path=next((p/'session.json' for p in (OUT,*OUT.parents) if (p/'session.json').exists()),None)
        if session_path is not None:
            import datetime
            START=datetime.datetime.fromisoformat(json.loads(session_path.read_text())['start_utc']).timestamp()
        else: START=time.time()
    if args.method=='residual': M=P.mod('noran/tracks/p61_missing_residual/model.py','p61_residual')
    elif args.method=='weighted': M=P.mod('noran/tracks/p61_missing_weighted/model.py','p61_weighted')
    elif args.method=='adapted': M=P.mod('noran/tracks/p61_missing_adapted/model.py','p61_adapted')
    with threadpool_limits(limits=2):
        if args.action=='develop':development()
        elif args.action=='assess':assessment()
        elif args.action=='select':selection()
        else: print('Protected unchanged',initialize())


if __name__=='__main__':main()
