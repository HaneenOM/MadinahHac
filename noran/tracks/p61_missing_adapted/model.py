"""Unsupervised input-distribution adaptation of V7 missing stack weights."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('p61_weighted_for_adapt',ROOT/'noran/tracks/p61_missing_weighted/model.py')
weighted=importlib.util.module_from_spec(spec);spec.loader.exec_module(weighted)
direct=weighted.direct;v7=direct.v7
CONFIG_PATH=ROOT/'noran/configs/prompt61_adapted.json'
SETTINGS=json.loads(CONFIG_PATH.read_text())
baseline_values=direct.baseline_values
inference_rows=direct.inference_rows
FEATURES=['log_level','span','n_left','n_right','fraction','own_4','spl_30','partner0_4','partner1_4']


def design(rows):
    f=direct.features(rows)[FEATURES].replace(-999.,np.nan)
    return np.c_[f.fillna(0.).to_numpy(),f.isna().to_numpy().astype(float)]


def density(training,query):
    x,q=design(training),design(query)
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),.05)
    xx=np.clip((x-mean)/std,-20.,20.);qq=np.clip((q-mean)/std,-20.,20.)
    classifier=LogisticRegression(C=1.,class_weight='balanced',max_iter=300,random_state=6104)
    classifier.fit(np.r_[xx,qq],np.r_[np.zeros(len(xx)),np.ones(len(qq))])
    prob=classifier.predict_proba(xx)[:,1]
    return np.clip(prob/np.maximum(1-prob,1e-8),*SETTINGS['density_ratio_bounds'])


def adapt(query,model):
    baseline=model['baseline'];rows=model['rows']
    alternate={u:{s:dict(m,w=m['w'].copy()) for s,m in subsets.items()} for u,subsets in baseline['weights'].items()}
    for unit,g in query.groupby('unit'):
        if unit not in alternate:continue
        setting=v7.SPEC[unit]
        # M's borrowed pool has no direct matching camera; keep V7 there.
        if setting.get('pool'):continue
        frame=rows[rows.unit.eq(unit)].copy()
        if len(frame)<200:continue
        weight=density(frame,g)
        y=frame.truth.to_numpy(float)-frame.base.to_numpy(float)
        scale=np.maximum(frame.base.to_numpy(float),5.) if setting.get('relative') else np.ones(len(frame))
        for subset,sub in alternate[unit].items():
            if not sub['columns']:continue
            x=frame[sub['columns']].to_numpy(float)-frame.base.to_numpy(float)[:,None]
            ok=np.isfinite(x).all(axis=1)
            sub['w']=weighted.solve(x[ok]/scale[ok,None],y[ok]/scale[ok],weight[ok],v7.RIDGE,setting.get('loss','l2'))
    return dict(baseline,weights=alternate)


def from_rows(rows,training,context):
    baseline=direct.baseline_from_rows(rows,training,context)
    return dict(baseline=baseline,nonlinear=dict(baseline=baseline,rows=rows.copy()),cache={})


def nonlinear_values(rows,model,baseline=None):
    return direct.baseline_values(rows,adapt(rows,model))


def fit(training,day,context):
    return from_rows(v7.pseudo_rows(training,context['test_released'],context['config']),training,context)


def recording_query(t,unit):
    if unit not in t:return pd.DataFrame()
    known=t.index[t[unit].notna()].to_numpy(int)
    parts=[]
    for a,b in zip(known[:-1],known[1:]):
        if 1<b-a<=30:
            # Every unavailable bounded minute, independent of assessment ID or label.
            for minute in range(int(a)+1,int(b)):
                row=inference_rows(type('Gap',(),dict(unit=unit,minute=minute))(),t)
                if len(row):parts.append(row)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def predict(case,t,model,context):
    unit=case.unit
    key=(unit,hashlib.sha256(pd.util.hash_pandas_object(t,index=True).values.tobytes()+str(tuple(t.columns)).encode()).hexdigest())
    if key not in model['cache']:
        query=recording_query(t,unit)
        model['cache'][key]=adapt(query,model['nonlinear']) if len(query) else model['baseline']
    base=v7.predict(case,t,model['baseline'],context)
    extra=v7.predict(case,t,model['cache'][key],context)
    if not np.isfinite(base) or not np.isfinite(extra):return np.nan
    return float((1-SETTINGS['blend'])*base+SETTINGS['blend']*extra)
