"""Bounded four-mission session: isolated hidden smoothing / convex missing trials.
No T05 execution. Existing HF evidence is read-only. Outputs under outputs/prompt56.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,os,sys,time
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'src',ROOT/'noran'):
    sys.path.insert(0,str(path))
os.environ['HF_HOME']=str(ROOT/'outputs/version7/huggingface')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONUTF8']='1'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import baseline_tasks as tasks
from pipeline import load,locate_data,table,CAMERAS
from prayer_experiments import timetable

OUT=ROOT/'outputs/prompt56'
START=1795087020.0  # 2026-11-19 11:17:00 UTC, conservative start before first inspection.
DEADLINE=START+3600

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_time(reserve=600):
    if time.time()>DEADLINE-reserve:raise TimeoutError('Experiment cutoff reached; final 10 minutes reserved.')

def mod(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    session=OUT/'session.json'
    if not session.exists():session.write_text(json.dumps(dict(start_utc='2026-11-19T11:17:00Z',
        deadline_utc='2026-11-19T12:17:00Z',experiment_cutoff_utc='2026-11-19T12:07:00Z',
        prompt55_found=False),indent=2))
    protected=OUT/'protected.json'
    if not protected.exists():
        paths=[]
        for directory in ('configs','notebooks','models','data/raw','src','noran/tracks','outputs/version7','outputs/noran','outputs/prompt54_hf'):
            paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                         and 'huggingface' not in p.parts and 'p56_' not in str(p.relative_to(ROOT)))
        protected.write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},indent=2))
    saved=json.loads(protected.read_text());wrong=[p for p,h in saved.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    assert not wrong,f'Protected files changed: {wrong}'
    paths=['noran/configs/prompt56.json','noran/tracks/p56_missing_convex/model.py','noran/tracks/p56_hidden_smooth/model.py']
    hashes={p:sha(ROOT/p) for p in paths};lock=OUT/'recipe_lock.json'
    if lock.exists():assert json.loads(lock.read_text())['hashes']==hashes,'Locked recipe changed'
    else:lock.write_text(json.dumps(dict(hashes=hashes,locked_time=time.time(),development='fixed',assessment=['mirror','testday']),indent=2))
    return len(saved)

def setup():
    train,test,sample,_=load(locate_data(None))
    return train,test,sample,timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)

def protocols(which,mission,train,test):
    rows,labels,feeds=[],[],{}
    if which in ('fixed','mirror'):
        maps=({'20260918_1100':'20260911_1100','20260919_1100':'20260912_1100'} if which=='fixed' else
              {'20260918_1100':'20260912_1100','20260919_1100':'20260911_1100'})
        for source,rec in maps.items():
            truth=train[train.recording.eq(rec)];day=truth.day.iloc[0]
            mask=test[test.recording.eq(source)&test.unit.isin(CAMERAS)].copy();mask['recording']=rec
            feed=mask[mask.id.isna()][['recording','unit','minute_of_day']].merge(truth,on=['recording','unit','minute_of_day'],how='left')
            feeds[rec]=table(feed[feed.people.notna()])[rec]
            target=mask[mask.id.notna()&mask.kind.eq(mission)].drop(columns='people').merge(
                truth[['recording','unit','minute_of_day','people']],on=['recording','unit','minute_of_day'],how='inner')
            for r in target.itertuples():
                m=int(r.minute_of_day);key=f'{rec}_{r.unit}_{m}'
                rows.append(dict(case_id=key,day=day,recording=rec,kind=mission,unit=r.unit,minute=m))
                labels.append(dict(case_id=key,truth=float(r.people)))
    elif which in ('testday','testday_strict') and mission=='hidden':
        for rec,hidden in {'20260918_1100':['cam_D'],'20260919_1100':['cam_B','cam_G']}.items():
            released=test[test.recording.eq(rec)&test.id.isna()&test.people.notna()]
            feeds[rec]=table(released[~released.unit.isin(hidden)])[rec]
            for r in released[released.unit.isin(hidden)].itertuples():
                m=int(r.minute_of_day);key=f'{rec}_{r.unit}_{m}'
                rows.append(dict(case_id=key,day=r.day,recording=rec,kind=mission,unit=r.unit,minute=m))
                labels.append(dict(case_id=key,truth=float(r.people)))
    else:raise ValueError((which,mission))
    cases=pd.DataFrame(rows);assert cases.case_id.is_unique
    return cases,feeds,pd.DataFrame(labels)

def ctx_for(day,which,mission,train,test,schedule,cases):
    training=train[train.day.ne(day)].copy()
    released=test[test.id.isna()&test.people.notna()].copy()
    if which in ('testday','testday_strict'):
        rec=cases.recording.iloc[0];hidden=cases.unit.unique()
        released=released[~(released.recording.eq(rec)&released.unit.isin(hidden))]
        if which=='testday_strict':
            # Stronger audit: exclude every pseudo-assessment answer on both
            # recordings from every fit, not only the current held-out fold.
            all_mask=(released.recording.eq('20260918_1100')&released.unit.eq('cam_D'))|(
                released.recording.eq('20260919_1100')&released.unit.isin(['cam_B','cam_G']))
            released=released[~all_mask]
    # Whole assessment date excluded from historical training. For testday,
    # current-day released donors are allowed, but pseudo-hidden answers are not.
    assert not training.day.eq(day).any()
    if which in ('testday','testday_strict'):
        assert not (released.recording.eq(rec)&released.unit.isin(hidden)).any()
    return training,dict(schedule=schedule,config=tasks.default_config()[mission],mission=mission,test_released=released)

def score(d,mission,which):
    d=d.copy()
    for col in ('v7','trial','reference'):d['e_'+col]=(d[col]-d.truth).abs()
    d['gain']=d.e_v7-d.e_trial;d['changed']=(d.v7-d.trial).abs()>1e-8
    if mission=='missing':
        d['window']=d.recording+'|'+d.unit+'|'+d.gap_start.astype(str)+'|'+d.gap_end.astype(str)
        d['event']=d.recording+'|'+d.gap_start.astype(str)+'|'+d.gap_end.astype(str)
    else:
        d['window']=d.recording+'|'+d.unit+'|'+d.block.astype(str)
        d['event']=d.recording+'|'+d.block.astype(str)
    daily=d.groupby('day')[['e_v7','e_trial','e_reference']].mean()
    result=dict(mission=mission,case_set=which,rows=len(d),v7_mae=float(d.e_v7.mean()),trial_mae=float(d.e_trial.mean()),
        v7_ratio=float((daily.e_v7/daily.e_reference).mean()),trial_ratio=float((daily.e_trial/daily.e_reference).mean()),
        changed=int(d.changed.sum()),fallback=int(d.fallback.sum()))
    for label,keys in dict(recording=['recording','unit'],window=['recording','unit','window']).items():
        g=d.groupby(keys).agg(n=('truth','size'),windows=('window','nunique'),v7_mae=('e_v7','mean'),
            trial_mae=('e_trial','mean'),reference_mae=('e_reference','mean'),gain=('gain','mean'),changed=('changed','sum'))
        g['v7_ratio']=g.v7_mae/g.reference_mae;g['trial_ratio']=g.trial_mae/g.reference_mae
        g.to_csv(OUT/f'{mission}_{which}_by_{label}.csv')
        if label=='recording':print(g.to_string(),flush=True)
    w=d.groupby('window').gain.sum();e=d.groupby('event').gain.sum().sort_values(ascending=False)
    result.update(windows=len(w),event_windows=len(e),wins=int((w>1e-8).sum()),losses=int((w< -1e-8).sum()))
    result['remove_best_events']=[]
    for n in (0,1,2):
        rem=d[~d.event.isin(e.head(n).index)];daily=rem.groupby('day')[['e_v7','e_trial','e_reference']].mean()
        result['remove_best_events'].append(dict(n=n,mae_gain=float(rem.gain.mean()),
            ratio_gain=float(((daily.e_v7-daily.e_trial)/daily.e_reference).mean())))
    d.to_csv(OUT/f'{mission}_{which}_scored.csv',index=False)
    (OUT/f'{mission}_{which}_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def collect(mission,which):
    initialize();check_time()
    path=OUT/f'{mission}_{which}_predictions.csv'
    if path.exists():print('Cache exists:',path);return
    started=time.perf_counter();train,test,sample,schedule=setup()
    cases,feeds,labels=protocols(which,mission,train,test)
    source=(test if which.startswith('testday') else train).copy()
    # Independently taint every withheld answer before constructing the feed.
    # The released inference tables must be identical, including missing-minute positions.
    source=train.copy() if not which.startswith('testday') else test.copy()
    for rec,g in cases.groupby('recording'):
        if mission=='hidden':
            mask=source.recording.eq(rec)&source.unit.isin(g.unit.unique())
        else:
            keys=set(zip(g.unit,g.minute))
            mask=source.recording.eq(rec)&pd.Series(list(zip(source.unit,source.minute_of_day)),index=source.index).isin(keys)
        source.loc[mask,'people']=1e8
    _,tainted,_=protocols(which,mission,train if which.startswith('testday') else source,source if which.startswith('testday') else test)
    for rec,t in feeds.items():pd.testing.assert_frame_equal(t,tainted[rec])
    trial=mod('noran/tracks/p56_'+('missing_convex' if mission=='missing' else 'hidden_smooth')+'/model.py','p56_trial_'+mission)
    rows,folds=[],[];leaks=0;leak_checks=0
    with threadpool_limits(limits=2):
        for day,today in cases.groupby('day'):
            check_time(900)
            training,ctx=ctx_for(day,which,mission,train,test,schedule,today)
            tick=time.perf_counter()
            if mission=='missing':
                key=hashlib.sha256(pd.util.hash_pandas_object(training,index=True).values.tobytes()+
                    pd.util.hash_pandas_object(ctx['test_released'],index=True).values.tobytes()).hexdigest()[:16]
                cache=OUT/f'pseudo_{key}.pkl'
                if cache.exists():pseudo=pd.read_pickle(cache)
                else:
                    pseudo=trial.v7.pseudo_rows(training,ctx['test_released'],ctx['config']);pseudo.to_pickle(cache)
                model=trial.from_rows(training,ctx,pseudo)
                (OUT/f'missing_{which}_{day}_weights.json').write_text(json.dumps(model['weight_audit'],indent=2))
            else:model=trial.fit(training,day,ctx)
            fit_seconds=time.perf_counter()-tick;tick=time.perf_counter()
            fold_checks=0
            for rec,group in today.groupby('recording'):
                t=feeds[rec]
                if mission=='hidden':
                    # Precompute all eligible minute predictions, not only labeled targets.
                    for unit,unit_cases in group.groupby('unit'):
                        donors=[c for c in t if c in CAMERAS and c not in trial.MASK[unit]]
                        visible=t[donors].notna().sum(axis=1)>=2
                        seq={}
                        for m in t.index.union(pd.Index(unit_cases.minute.unique())):
                            case=SimpleNamespace(day=day,recording=rec,kind='hidden',unit=unit,minute=int(m))
                            value=trial.v7.predict(case,t,model,ctx)
                            seq[int(m)]=float(value)
                        series=pd.Series(seq,dtype=float).sort_index()
                        eligible=series.where(visible.reindex(series.index).fillna(False))
                        center=np.log1p(series)
                        left=np.log1p(eligible.reindex(series.index-1).set_axis(series.index)).fillna(center)
                        right=np.log1p(eligible.reindex(series.index+1).set_axis(series.index)).fillna(center)
                        smoothed=np.expm1((2*center+left+right)/4)
                        seq_path=OUT/f'hidden_{which}_{rec}_{unit}_sequence.csv'
                        pd.DataFrame(dict(v7=series,trial=smoothed)).to_csv(seq_path,index_label='minute')
                        base=tasks.fit_hidden(training,ctx['config'],schedule)
                        for r in unit_cases.itertuples():
                            ref,old,_=tasks.hidden_value(t,unit,int(r.minute),base,ctx['config'],day=day,schedule=schedule)
                            raw=float(series[r.minute]);new=float(smoothed[r.minute])
                            rows.append(dict(**r._asdict(),v7=raw if np.isfinite(raw) else old,trial=new if np.isfinite(new) else old,
                                reference=ref,fallback=not np.isfinite(new),block=int(np.searchsorted(np.flatnonzero(np.diff(t.index)>1),
                                    np.searchsorted(t.index,r.minute),side='left'))))
                        # Full-module masking check, including smoothing; one representative per visible block.
                        chosen=unit_cases.iloc[::25]
                        for r in chosen.itertuples():
                            altered=t.copy()
                            for c in trial.MASK[unit]:altered[c]=1e8
                            expected=float(smoothed[r.minute]);actual=trial.predict(r,altered,model,ctx)
                            leaks+=int(not np.isclose(expected,actual,atol=1e-8,rtol=0,equal_nan=True));leak_checks+=1;fold_checks+=1
                else:
                    for r in group.itertuples():
                        v=trial.v7.predict(r,t,model['baseline'],ctx);p=trial.predict(r,t,model,ctx)
                        ref,old,_=tasks.missing_value(t,r.unit,int(r.minute),model['baseline']['calibration'],ctx['config'],day=day)
                        bounds=trial.v7.gap_context(t,r.unit,int(r.minute));start,end=bounds[:2] if bounds else (r.minute,r.minute)
                        fallback=not np.isfinite(p)
                        rows.append(dict(**r._asdict(),v7=float(v) if np.isfinite(v) else old,
                            trial=float(p) if np.isfinite(p) else old,reference=ref,fallback=fallback,
                            gap_start=int(start),gap_end=int(end)))
                    # Poison every unknown cell; restore mask before prediction, and verify target answers never enter features.
                    for r in group.iloc[::15].itertuples():
                        altered=t.copy();bound=trial.v7.gap_context(t,r.unit,int(r.minute))
                        if bound:
                            altered.loc[(altered.index>bound[0])&(altered.index<bound[1]),r.unit]=np.nan
                        expected=trial.predict(r,t,model,ctx);actual=trial.predict(r,altered,model,ctx)
                        leaks+=int(not np.isclose(expected,actual,atol=1e-8,rtol=0,equal_nan=True));leak_checks+=1;fold_checks+=1
            folds.append(dict(day=day,fit_seconds=fit_seconds,inference_checks_seconds=time.perf_counter()-tick,
                training_days=sorted(training.day.unique()),test_context_days=sorted(ctx['test_released'].day.unique()),checks=fold_checks))
            print(mission,which,day,'complete',time.perf_counter()-started,flush=True)
    assert leaks==0
    d=pd.DataFrame(rows).drop(columns='Index',errors='ignore');assert len(d)==len(cases)
    assert np.isfinite(d[['v7','trial','reference']]).all().all()
    d.to_csv(path,index=False)
    # Assessment labels join after all predictions have been saved.
    d=d.merge(labels,on='case_id',validate='one_to_one');score(d,mission,which)
    (OUT/f'{mission}_{which}_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,
        folds=folds,leak_checks=leak_checks,leak_failures=leaks),indent=2))
    initialize()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','check']);parser.add_argument('--mission',choices=['hidden','missing'])
    parser.add_argument('--set',default='fixed',choices=['fixed','mirror','testday','testday_strict']);args=parser.parse_args()
    if args.action=='check':print('Protected:',initialize())
    else:collect(args.mission,args.set)
if __name__=='__main__':main()
