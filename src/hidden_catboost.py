"""Reusable frozen P42 CatBoost recipe; no artifact/cache/submission dependency."""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from pipeline import table
DONORS=[f'cam_{u}' for u in 'ACDEFH']
CAMERAS=('cam_B','cam_G')
SPECS={'counts_d4':dict(temporal=False,depth=4,l2_leaf_reg=10.)}
PARAMS=dict(loss_function='MAE',eval_metric='MAE',iterations=500,learning_rate=.05,random_seed=42,thread_count=4,border_count=64,random_strength=1.,bootstrap_type='Bernoulli',subsample=.8,nan_mode='Min',allow_writing_files=False,verbose=False)

def features(feeds,cases,schedule,temporal):
    """Exact central values and actual observed neighbours, independently per recording.

    Reindexing creates NaNs only. No interpolation/forward-fill of COUNT values.
    Before/after windows exclude the central minute. Ages are metadata only.
    """
    pieces=[]
    for rec,g in cases.groupby('recording',sort=False):
        feed=feeds[rec].reindex(columns=DONORS).copy()
        assert not set(feed.columns)&set(CAMERAS)
        start=min(int(feed.index.min()),int(g.minute.min()))
        end=max(int(feed.index.max()),int(g.minute.max()))
        grid=feed.reindex(np.arange(start,end+1))
        raw=grid.reindex(g.minute.to_numpy()).reset_index(drop=True)
        result=pd.DataFrame(index=np.arange(len(g)))
        total=raw.sum(axis=1,min_count=1)
        for donor in DONORS:
            result[donor+'_count']=raw[donor]
            result[donor+'_log']=np.log1p(raw[donor].clip(lower=0))
            result[donor+'_share']=raw[donor]/total.where(total.ge(1.))
            result[donor+'_seen']=raw[donor].notna().astype(float)
            if temporal:
                series=grid[donor]
                past=series.shift(1).rolling(5,min_periods=1)
                future=series.iloc[::-1].shift(1).rolling(5,min_periods=1)
                columns={'before_mean':past.mean(),'after_mean':future.mean().iloc[::-1],
                    'before_n':past.count(),'after_n':future.count().iloc[::-1]}
                observed=pd.Series(np.where(series.notna(),grid.index,np.nan),index=grid.index)
                # Only timestamps are propagated, never people readings.
                columns['before_age']=pd.Series(grid.index,index=grid.index)-observed.shift(1).ffill()
                columns['after_age']=observed.shift(-1).bfill()-pd.Series(grid.index,index=grid.index)
                for suffix,values in columns.items():
                    if suffix.endswith('age'):values=values.where(values.le(5))
                    result[donor+'_'+suffix]=values.reindex(g.minute.to_numpy()).to_numpy()
        result['visible_total']=total
        result['available']=raw.notna().sum(axis=1)
        minute=g.minute.to_numpy(float)
        result['minute']=minute
        result['clock_sin']=np.sin(2*np.pi*minute/1440)
        result['clock_cos']=np.cos(2*np.pi*minute/1440)
        result['friday']=[float(pd.Timestamp(d).day_of_week==4) for d in g.day]
        result['dhuhr_offset']=[m-schedule[d][1] for m,d in zip(minute,g.day)]
        result['nearest_prayer']=[m-schedule[d][np.argmin(abs(schedule[d]-m))] for m,d in zip(minute,g.day)]
        result.index=g.case_id.to_numpy();pieces.append(result)
    return pd.concat(pieces).loc[cases.case_id].astype(float)

def dataset(training,schedule):
    """Fit rows and features derive only from explicitly supplied released table."""
    feeds=table(training);rows=[]
    for rec,g in training.groupby('recording'):
        available=feeds[rec].reindex(columns=DONORS).notna().sum(axis=1)
        for r in g[g.unit.isin(CAMERAS)].itertuples():
            if available.get(r.minute_of_day,0)>=2:
                rows.append(dict(case_id=f'fit:{rec}:{r.unit}:{int(r.minute_of_day)}',recording=rec,day=r.day,
                    unit=r.unit,minute=int(r.minute_of_day),truth=float(r.people)))
    cases=pd.DataFrame(rows)
    # Targets are NEVER present in feed given to feature construction.
    safe={rec:t.reindex(columns=DONORS) for rec,t in feeds.items()}
    x=features(safe,cases,schedule,True)
    return cases,x

def columns(x,spec):
    if SPECS[spec]['temporal']:return list(x.columns)
    return [c for c in x if not any(s in c for s in ('_before_','_after_'))]

def augmentation(x,meta):
    dropped=x.copy();rng=np.random.default_rng(42)
    mask=rng.random((len(x),len(DONORS)))<.2
    for j,u in enumerate(DONORS):
        lost=mask[:,j]
        for c in x:
            if c.startswith(u+'_'):dropped.loc[lost,c]=0. if c.endswith('_seen') or c.endswith('_n') else np.nan
    total=dropped[[u+'_count' for u in DONORS]].sum(axis=1,min_count=1)
    dropped['visible_total']=total
    dropped['available']=dropped[[u+'_seen' for u in DONORS]].sum(axis=1)
    for u in DONORS:dropped[u+'_share']=dropped[u+'_count']/total.where(total.ge(1.))
    keep=dropped.available.ge(2).to_numpy()
    xx=pd.concat([x,dropped.iloc[np.flatnonzero(keep)]],ignore_index=True)
    y=meta.truth.to_numpy(float);w=np.where(pd.to_datetime(meta.day).dt.day_of_week.eq(4),3.,1.)
    return xx,np.concatenate([y,y[keep]]),np.concatenate([w,w[keep]])

def train_model(meta,x,unit,spec,iterations=500,validation=None):
    keep=meta.unit.eq(unit).to_numpy();m=meta.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    z=x.iloc[np.flatnonzero(keep)][columns(x,spec)].reset_index(drop=True)
    z,y,w=augmentation(z,m)
    params=dict(PARAMS,iterations=int(iterations),**{k:v for k,v in SPECS[spec].items() if k!='temporal'})
    model=CatBoostRegressor(**params)
    extra={}
    if validation is not None:
        vm,vx=validation;valid=vm.unit.eq(unit).to_numpy()
        extra=dict(eval_set=Pool(vx.iloc[np.flatnonzero(valid)][list(z.columns)],vm.loc[valid,'truth']),
            early_stopping_rounds=50,use_best_model=True)
    model.fit(Pool(z,y,weight=w),**extra)
    return model,dict(unit=unit,spec=spec,original_rows=len(m),augmented_rows=len(y),
        records=sorted(m.recording.unique()),dates=sorted(m.day.unique()),iterations=model.tree_count_,features=list(z.columns))

