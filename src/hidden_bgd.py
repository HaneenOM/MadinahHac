"""Accepted 0.800 HIDDEN recipes, with caller-controlled fitting sources."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from pipeline import table
import v2_hidden
from v2_missing import interp_at
import hidden_catboost as cb
import hidden_reconstruction_trial as nonlinear

def released_rows(frame):
    if frame is None:return None
    if 'id' in frame:frame=frame[frame.id.isna()&frame.people.notna()]
    return frame.loc[frame.unit.isin([f'cam_{u}' for u in 'ABCDEFGH'])&frame.people.notna(),
        ['recording','day','unit','minute_of_day','people']].copy()

def fit(training,released,schedule,config):
    old=released_rows(training);extra=released_rows(released)
    expanded=pd.concat([old,extra],ignore_index=True) if extra is not None else old
    keys=['recording','unit','minute_of_day']
    assert not expanded.duplicated(keys).any()
    meta,x=cb.dataset(expanded,schedule)
    model,bmanifest=cb.train_model(meta,x,'cam_B','counts_d4',config['catboost_iterations'])
    frames=table(expanded)
    full=pd.concat(frames.values(),keys=frames.keys(),names=['recording','minute_of_day'])
    valid=full.cam_G.notna()&full.reindex(columns=cb.DONORS).notna().sum(axis=1).ge(2)
    raw=full.loc[valid].reindex(columns=cb.DONORS);y=full.loc[valid,'cam_G'].to_numpy()
    med=raw.median().fillna(0.);design=v2_hidden._design(raw,cb.DONORS,med)
    augmented=[design]+[v2_hidden._design(raw,cb.DONORS,med,dropout=True,seed=i) for i in range(3)]
    day_map=expanded.groupby('recording').day.first().to_dict()
    weights=np.array([3. if pd.Timestamp(day_map[r]).day_of_week==4 else 1. for r in raw.index.get_level_values(0)])
    g=Ridge(alpha=config['alpha']).fit(pd.concat(augmented,ignore_index=True),np.tile(y,4),sample_weight=np.tile(weights,4))
    d,_=nonlinear.fit_new(old,schedule)
    return dict(B=model,G=g,G_med=med,D=d['cam_D'],manifest=dict(
        B=bmanifest,G=dict(examples=len(raw),records=sorted(raw.index.get_level_values(0).unique())),
        D=dict(records=sorted(old.recording.unique()),source='caller training only, no released expansion'),
        training_records=sorted(old.recording.unique()),extra_records=sorted(extra.recording.unique()) if extra is not None else []))

def predict(models,targets,feeds,references,legacy,config,schedule):
    result={};component={};used={}
    for (rec,unit),g in targets.groupby(['recording','unit'],sort=True):
        t=feeds[rec];minutes=g.minute_of_day.to_numpy(int)
        if unit=='cam_B':
            cases=g.copy();cases['case_id']=g.id;cases['minute']=g.minute_of_day
            x=cb.features(feeds,cases,schedule,False)
            fitted=np.maximum(0.,models['B'].predict(x));ok=x.available.ge(2).to_numpy()
            anchor=np.array([legacy[key] for key in g.id]);value=np.maximum(0.,anchor+config['catboost_weight']*(fitted-anchor))
        elif unit=='cam_G':
            raw=pd.DataFrame([[interp_at(t[u],int(m)) if u in t else np.nan for u in cb.DONORS] for m in minutes],
                index=pd.MultiIndex.from_arrays([g.recording,minutes],names=['recording','minute_of_day']),columns=cb.DONORS)
            fitted=np.maximum(0.,models['G'].predict(v2_hidden._design(raw,cb.DONORS,models['G_med'])))
            ok=raw.notna().sum(axis=1).ge(2).to_numpy();anchor=np.array([references[key] for key in g.id])
            value=np.maximum(0.,anchor+config['g_weight']*(fitted-anchor))
        else:
            x,available=nonlinear.make_features(t.reindex(index=minutes),unit,g.day.tolist(),schedule)
            fitted=np.maximum(0.,models['D'].predict(x));ok=available.ge(2).to_numpy();value=fitted
        for key,v,f,u in zip(g.id,value,fitted,ok):
            result[key]=float(v) if u else legacy[key];component[key]=float(f);used[key]=bool(u)
    return result,component,used
