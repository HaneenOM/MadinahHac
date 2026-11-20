"""V7 candidates with duplicate-aware, equal-date fitting; no new features."""
from pathlib import Path
import importlib.util
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('p61_direct_for_weighted',ROOT/'noran/tracks/p61_missing/model.py')
direct=importlib.util.module_from_spec(spec);spec.loader.exec_module(direct)
v7=direct.v7
CONFIG_PATH=ROOT/'noran/configs/prompt61_weighted.json'
SETTINGS=json.loads(CONFIG_PATH.read_text())
baseline_values=direct.baseline_values
inference_rows=direct.inference_rows


def sample_weights(frame):
    d=frame.copy()
    d['absolute_minute']=d.minute0+d.m
    multiplicity=d.groupby(['source','unit','absolute_minute']).source.transform('size').to_numpy(float)
    w=1./multiplicity
    dates=d.source.str[:8]
    for day in dates.unique():
        mask=dates.eq(day).to_numpy();w[mask]/=w[mask].sum()
    return w*len(w)/w.sum()


def solve(x,y,w,ridge,loss):
    penalty=ridge*((x*x)*w[:,None]).sum()/w.sum()/x.shape[1]+1e-9
    k=x.shape[1]
    coef=np.linalg.solve(x.T@(x*w[:,None])/w.sum()+penalty*np.eye(k),x.T@(y*w)/w.sum())
    if loss=='l1':
        for _ in range(25):
            residual=np.maximum(abs(y-x@coef),.05*np.average(abs(y),weights=w)+1e-9)
            rw=w/residual
            coef=np.linalg.solve(x.T@(x*rw[:,None])/w.sum()+penalty*(rw.sum()/w.sum())*np.eye(k),x.T@(y*rw)/w.sum())
    return coef


def from_rows(rows,training,context):
    baseline=direct.baseline_from_rows(rows,training,context)
    alternate={u:{subset:dict(m,w=m['w'].copy()) for subset,m in subsets.items()} for u,subsets in baseline['weights'].items()}
    audit=[]
    for unit,subsets in alternate.items():
        setting=v7.SPEC[unit]
        frame=rows[rows.unit.isin(setting['pool'])] if setting.get('pool') else rows[rows.unit.eq(unit)]
        y=frame.truth.to_numpy(float)-frame.base.to_numpy(float)
        scale=np.maximum(frame.base.to_numpy(float),5.) if setting.get('relative') else np.ones(len(frame))
        weight=sample_weights(frame)
        for subset,m in subsets.items():
            if not m['columns']:continue
            x=frame[m['columns']].to_numpy(float)-frame.base.to_numpy(float)[:,None]
            ok=np.isfinite(x).all(axis=1)
            m['w']=solve(x[ok]/scale[ok,None],y[ok]/scale[ok],weight[ok],v7.RIDGE,setting.get('loss','l2'))
        audit.append(dict(unit=unit,rows=len(frame),dates=sorted(frame.source.str[:8].unique()),
            distinct_readings=len(frame.assign(absolute_minute=frame.minute0+frame.m).drop_duplicates(['source','unit','absolute_minute']))))
    return dict(baseline=baseline,nonlinear=dict(baseline,weights=alternate),weight_audit=audit)


def nonlinear_values(rows,model,baseline=None):
    return direct.baseline_values(rows,model)


def fit(training,day,context):
    return from_rows(v7.pseudo_rows(training,context['test_released'],context['config']),training,context)


def predict(case,t,model,context):
    base=v7.predict(case,t,model['baseline'],context)
    extra=v7.predict(case,t,model['nonlinear'],context)
    if not np.isfinite(base) or not np.isfinite(extra):return np.nan
    return float((1-SETTINGS['blend'])*base+SETTINGS['blend']*extra)
