"""Camera-specific donor level correction; V7 timing and analog weights unchanged."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
for path in (ROOT,ROOT/'src'):sys.path.insert(0,str(path))
spec=importlib.util.spec_from_file_location('p61_hf_original',ROOT/'noran/tracks/hf_final/model.py')
v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
BASE,C=v7.base,v7.base.C
SETTINGS=json.loads((ROOT/'noran/configs/prompt61_hf_level.json').read_text())
UNITS=('cam_B','cam_G','cam_D')


def inputs(causal,unit,now,day,an,p,ref,schedule):
    cams=[c for c in C.CAMS if c in causal and c not in C.HIDDEN_WITH[unit]]
    wins={c:C.window(causal,c,now,p['K']) for c in cams}
    sw={c:C.window(causal,c,now,p['SHIFT_WINDOW']) for c in p['SHIFT_CAMS'] if c in cams}
    delta=C.best_shift(sw,an,list(sw),day,schedule,ref,p['STRENGTH'],p['SHIFT_MAX'],p['SHIFT_PENALTY'])
    logs={}
    for c in cams:
        ratio,_=C.analog_ratio(*wins[c],an,c,day,schedule,ref,min_pairs=p['MIN_PAIRS'],strength=p['STRENGTH'],delta=delta)
        if np.isfinite(ratio) and ratio>0:logs[c]=np.log(ratio)
    if len(logs)<p['MIN_CAMS']:return None
    mean=float(np.mean(list(logs.values())))
    # A sum-zero deviation vector learns relative donor response. The uniform
    # component remains exactly the V7 beta*mean; unavailable donor deviation=0.
    x=np.array([logs.get(c,mean)-mean for c in C.CAMS])
    return x,mean,float(delta)


def examples(training,schedule):
    ref=C.reference_prayers(schedule);p=BASE.PARAMS
    frames=BASE.full_tables(training);days=training.groupby('recording').day.first().to_dict()
    analogs={r:C.Analog(t,days[r],schedule,ref,smooth=p['S'],strength=p['STRENGTH']) for r,t in frames.items()}
    rows=[];xx=[]
    for rec,t in frames.items():
        day=days[rec]
        for now in range(int(t.index.min())+19,int(t.index.max())-4,SETTINGS['origin_step']):
            causal=t.loc[:now]
            for unit in UNITS:
                if unit not in t:continue
                for donor,an in analogs.items():
                    if an.day==day or unit not in an.smooth:continue
                    made=inputs(causal,unit,now,day,an,p,ref,schedule)
                    if made is None:continue
                    x,mean,delta=made
                    for h in SETTINGS['horizons']:
                        truth=t[unit].get(now+h,np.nan)
                        tau=C.align([now+h],day,schedule,ref,p['STRENGTH'])+delta
                        target=float(an.at(an.smooth[unit],tau)[0])
                        if not np.isfinite(truth) or not np.isfinite(target) or min(truth,target)<=0:continue
                        baseline=float(np.exp(p['BETA']*mean)*target)
                        xx.append(x)
                        rows.append(dict(source=rec,day=day,analog=donor,analog_day=an.day,unit=unit,now=now,horizon=h,
                            truth=float(truth),base=baseline,residual=float(np.log(truth/baseline)),
                            same=int(pd.Timestamp(day).day_of_week==an.weekday)))
    return pd.DataFrame(rows),np.asarray(xx)


def fit_rows(rows,x):
    models={}
    for unit,g in rows.groupby('unit'):
        idx=g.index.to_numpy();a=x[idx];y=g.residual.to_numpy()
        # Pair recordings define the fitting support; overlapping origins and
        # horizons are never counted as independent recording replicates.
        count=g.groupby(['source','analog']).source.transform('size').to_numpy(float)
        w=np.maximum(g.base.to_numpy(),1)*(1+15*g.same.to_numpy())/count
        w=w/w.mean();coef=np.zeros(a.shape[1])
        for _ in range(10):
            rw=w/np.maximum(np.abs(y-a@coef),.1)
            coef=np.linalg.solve(a.T@(a*rw[:,None])/len(y)+SETTINGS['ridge']*rw.mean()*np.eye(a.shape[1]),a.T@(rw*y)/len(y))
        models[unit]=coef
    return models


def fit(training,day,context):
    baseline=v7.fit(training,day,context)
    rows,x=examples(training,context['schedule'])
    return dict(baseline=baseline,weights=fit_rows(rows,x),fit_rows=len(rows))


def analog_pair(rows,x,weights):
    correction=np.zeros(len(rows))
    for unit,g in rows.groupby('unit'):
        if unit in weights:correction[g.index]=x[g.index]@weights[unit]
    return rows.base.to_numpy()*np.exp(SETTINGS['strength']*np.clip(correction,-SETTINGS['max_log_correction'],SETTINGS['max_log_correction']))


def predict(case,t,model,context):
    baseline=model['baseline'];p=baseline['params'];now=int(case.now)
    raw=v7.predict(case,t,baseline,context)
    if not np.isfinite(raw) or case.unit not in model['weights']:return raw
    causal=t.loc[:now].drop(columns=list(C.HIDDEN_WITH[case.unit]),errors='ignore')
    tau=C.align([int(case.minute)],case.day,context['schedule'],baseline['ref_pr'],p['STRENGTH'])
    old_num=new_num=den=0.
    for an in baseline['analogs'].values():
        if case.unit not in an.smooth:continue
        made=inputs(causal,case.unit,now,case.day,an,p,baseline['ref_pr'],context['schedule'])
        if made is None:continue
        x,mean,delta=made
        target=float(an.at(an.smooth[case.unit],tau+delta)[0])
        if not np.isfinite(target):continue
        w=p['SAME_WEIGHT'] if an.weekday==pd.Timestamp(case.day).day_of_week else 1.
        old=float(np.exp(p['BETA']*mean)*target)
        correction=SETTINGS['strength']*np.clip(x@model['weights'][case.unit],-SETTINGS['max_log_correction'],SETTINGS['max_log_correction'])
        old_num+=w*old;new_num+=w*old*np.exp(correction);den+=w
    if den==0:return raw
    return float(max(0.,raw+p['ANALOG_WEIGHT']*(new_num-old_num)/den))
