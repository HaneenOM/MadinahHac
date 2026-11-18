"""Manual per-task evaluation through the SAME functions as the baseline notebook.

Always writes a separate directory; fixed historical benchmark files are read-only.
Forecast covariance caches are load-only. No T05 fitting is permitted here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pipeline import load,table,pseudo
from prayer_experiments import timetable
import baseline_tasks as tasks


def fixed_benchmark(train,test,*,prefer_saved=True):
    local=ROOT/'outputs/local_score'
    if prefer_saved and all((local/f).exists() for f in ('cases.csv','labels.csv','released_feed.csv','case_hashes.json')):
        from current_baseline import sha256
        for name,value in json.loads((local/'case_hashes.json').read_text()).items():assert sha256(local/name)==value
        return pd.read_csv(local/'cases.csv'),pd.read_csv(local/'labels.csv'),table(pd.read_csv(local/'released_feed.csv'))
    # A fresh clone can reconstruct the exact same masks from authorized raw data.
    cases=[];labels=[];feeds={}
    for name,feed,targets in pseudo(train,test,None):
        day=train.loc[train.recording.eq(name),'day'].iloc[0]
        feeds.update(table(feed))
        for r in targets.itertuples():
            now=int(r.now_minute_of_day) if pd.notna(r.now_minute_of_day) else None
            key=f'{name}_{r.unit}_{int(r.minute_of_day)}'
            cases.append(dict(case_id=key,day=day,recording=name,kind=r.kind,unit=r.unit,
                minute=int(r.minute_of_day),now=now,horizon=int(r.minute_of_day)-now if now is not None else None))
            labels.append(dict(case_id=key,truth=float(r.people)))
    c=pd.DataFrame(cases).sort_values(['day','kind','unit','minute'])
    assert c.groupby('kind').size().to_dict()==dict(forecast=2036,hidden=843,**{'hidden-future':482},missing=358)
    return c,pd.DataFrame(labels),feeds


def run(task,config,out,*,data_dir=None,allow_legacy_provenance=False,hidden_protocol='strict'):
    if task=='hidden':
        from hidden_evaluation import run as hidden_run
        return hidden_run(config,out,data_dir=data_dir,protocol=hidden_protocol)
    if task=='forecast':
        # Forecast evaluation must pass the provenance/context gate; never load
        # arbitrary legacy NPZ files by name or refit on a cache miss.
        if data_dir is not None and Path(data_dir).resolve()!=(ROOT/'data/raw').resolve():
            raise ValueError('Use the documented data/raw release for sealed forecast evaluation')
        from reliable_evaluation import run as reliable_run
        return reliable_run('historical',out,allow_legacy_provenance,config,task='forecast')
    started=time.perf_counter();out=Path(out)
    # Protect previous evaluations as well as submissions: choose a new directory.
    if (out/'predictions.csv').exists():raise FileExistsError(f'Choose a new --output-dir: {out}')
    out.mkdir(parents=True,exist_ok=True)
    train,test,_,_=load(data_dir or ROOT/'data/raw')
    if task=='forecast-im':return evaluate_im(test,config['forecast'],out,started)
    schedule=timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)
    cases,labels,feeds=fixed_benchmark(train,test)
    selected=cases[cases.kind.eq(task)]
    rows=[];checks=0;t05={}
    with threadpool_limits(limits=4):
        if task=='forecast':
            import t05_local_forecast as local
            from t05_forecast.pipeline import load_params
            def load_only(cache,key,datasets,**kwargs):
                path=cache/f'params_{key}.npz'
                if not path.exists():raise FileNotFoundError(
                    f'Historical held-out forecast cache missing: {path}. Production assets cannot replace it. See docs/teammate_baseline.md.')
                return load_params(path)
            local.fit_cached=load_only
            t05,checks=local.predict(cases,feeds)
        for day,today in selected.groupby('day'):
            training=train[train.day.ne(day)].copy();assert not training.day.eq(day).any()
            if task=='hidden':models=tasks.fit_hidden(training,config[task],schedule)
            elif task=='hidden-future':models=tasks.fit_hidden_future(training,schedule,config[task])
            elif task=='missing':models=tasks.fit_missing(training,config[task])
            else:models=None
            for r in today.itertuples():
                t=feeds[r.recording];minute=int(r.minute)
                if task=='forecast':
                    from pipeline import feature
                    reference=feature(t,r.unit,int(r.now),int(r.horizon))[0]
                    value=tasks.forecast_value(reference,t05[r.case_id],config[task])
                elif task=='hidden':
                    reference,value,_=tasks.hidden_value(t,r.unit,minute,models,config[task],day=day,schedule=schedule)
                    altered=t.copy()
                    import v2_hidden
                    for unit in v2_hidden.PATTERNS[r.unit]:altered[unit]=1e8
                    _,check,_=tasks.hidden_value(altered,r.unit,minute,models,config[task],day=day,schedule=schedule)
                    np.testing.assert_allclose(value,check,rtol=0,atol=1e-10);checks+=1
                elif task=='hidden-future':
                    reference,value,_=tasks.hidden_future_value(t,r.unit,int(r.now),int(r.horizon),day,schedule,models,config[task])
                    altered=t.copy();altered.loc[altered.index>int(r.now)]=1e8
                    _,check,_=tasks.hidden_future_value(altered,r.unit,int(r.now),int(r.horizon),day,schedule,models,config[task])
                    np.testing.assert_allclose(value,check,rtol=0,atol=1e-10);checks+=1
                else:reference,value,_=tasks.missing_value(t,r.unit,minute,models,config[task])
                rows.append(dict(case_id=r.case_id,day=day,unit=r.unit,minute=minute,
                    prediction=value,reference=reference))
    detail=pd.DataFrame(rows)
    assert len(detail)==len(selected) and np.isfinite(detail[['prediction','reference']]).all().all()
    detail.to_csv(out/'predictions.csv',index=False)
    scored=detail.merge(labels,on='case_id',validate='one_to_one')
    scored['error']=(scored.prediction-scored.truth).abs();scored['reference_error']=(scored.reference-scored.truth).abs()
    scored.to_csv(out/'scored.csv',index=False)
    for groups,name in [(['day'],'date'),(['day','unit'],'date_camera')]:
        report=scored.groupby(groups).agg(targets=('case_id','size'),mae=('error','mean'),reference_mae=('reference_error','mean')).reset_index()
        report['normalized_mae']=report.mae/report.reference_mae
        report.to_csv(out/f'by_{name}.csv',index=False)
    cached=ROOT/'outputs/local_score/predictions.csv'
    comparison=None
    if cached.exists():
        paired=detail.merge(pd.read_csv(cached)[['case_id','prediction']],on='case_id',validate='one_to_one',suffixes=('','_saved'))
        comparison=float((paired.prediction-paired.prediction_saved).abs().max())
    info=dict(task=task,targets=len(detail),t05_fits=0,cutoff_or_hidden_checks=checks,
        max_difference_from_saved_baseline=comparison,runtime_seconds=time.perf_counter()-started,
        config=config,limitations='Already examined dates; fold-specific reference and forecast sample96 fitting proxy. I-M absent from this fixed benchmark.')
    (out/'verification.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(pd.read_csv(out/'by_date.csv').to_string(index=False))
    print(json.dumps({k:v for k,v in info.items() if k!='config'},indent=2))
    return info


def evaluate_im(test,config,out,started):
    """Released-target rolling proxy; target is always strictly after now.

    Same context/predict functions as production; no fitting on test counts.
    Cases overlap and are not independent validation events.
    """
    released=test[test.id.isna()&test.people.notna()&test.unit.str[-1].isin(list('IJKLM'))]
    first=test.groupby('recording').minute_of_day.min().to_dict()
    rows=[];labels=[]
    for (name,unit),g in released.groupby(['recording','unit']):
        s=g.set_index('minute_of_day').people.sort_index();day=g.day.iloc[0]
        for minute,truth in s.items():
            for h in range(1,16):
                now=int(minute)-h
                if now<first[name]:continue
                base,value,eligible=tasks.im_forecast_value(s,now,config)
                key=f'{name}:{unit}:{int(minute)}:{h}'
                rows.append(dict(case_id=key,day=day,unit=unit,horizon=h,now=now,
                    prediction=value,reference=base,eligible=eligible))
                labels.append(dict(case_id=key,truth=truth))
    detail=pd.DataFrame(rows);detail.to_csv(out/'predictions.csv',index=False)
    scored=detail.merge(pd.DataFrame(labels),on='case_id',validate='one_to_one')
    scored=scored[scored.reference.notna() & scored.prediction.notna()].copy()
    scored['error']=(scored.prediction-scored.truth).abs();scored['reference_error']=(scored.reference-scored.truth).abs()
    scored.to_csv(out/'scored.csv',index=False)
    for groups,name in [(['day'],'date'),(['day','unit'],'date_camera'),(['horizon'],'horizon')]:
        report=scored.groupby(groups).agg(targets=('case_id','size'),mae=('error','mean'),reference_mae=('reference_error','mean')).reset_index()
        report['normalized_mae']=report.mae/report.reference_mae
        report.to_csv(out/f'by_{name}.csv',index=False)
    info=dict(task='forecast-im',cases=len(detail),scoreable=len(scored),eligible=int(scored.eligible.sum()),
        models_fitted=0,runtime_seconds=time.perf_counter()-started,
        limitations='Overlapping released-target rolling proxy on already examined test dates, no true I-M target labels; no independent validation.')
    (out/'verification.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2));return info


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=[*tasks.DEFAULT_CONFIG,'forecast-im'],required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--allow-legacy-provenance',action='store_true')
    parser.add_argument('--hidden-protocol',choices=['strict','release_matched'],default='strict')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8')) if args.config else tasks.default_config()
    run(args.task,cfg,args.output_dir,allow_legacy_provenance=args.allow_legacy_provenance,hidden_protocol=args.hidden_protocol)
