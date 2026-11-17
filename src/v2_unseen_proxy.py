"""Mask released I-M observations to test a local, camera-agnostic gap method."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline import ROOT,load,locate_data,table
import v2_missing

train,test,_,_=load(locate_data(None))
released=test[test.id.isna() & test.people.notna()]
all_tables=table(released)
holdouts=[]
masked={rec:t.copy() for rec,t in all_tables.items()}
for rec,t in all_tables.items():
    for unit in [f'cam_{c}' for c in 'IJKLM']:
        if unit not in t:
            continue
        s=t[unit].dropna()
        chunks=[]; current=[]
        for minute in s.index:
            if current and minute-current[-1]>1:
                chunks.append(current); current=[]
            current.append(int(minute))
        if current: chunks.append(current)
        for chunk in chunks:
            if len(chunk)<25:
                continue
            for start in range(10,len(chunk)-11,45):
                window=chunk[start:start+8]
                if len(window)<8:
                    continue
                for minute in window:
                    holdouts.append((rec,unit,minute,float(t.at[minute,unit])))
                    masked[rec].at[minute,unit]=np.nan

# All pseudo-labels are removed before fitting any local coefficient.
calibration=v2_missing.fit_calibration(masked)
rows=[]
for rec,unit,minute,truth in holdouts:
    t=masked[rec]
    linear=v2_missing.predict(t,unit,minute,calibration,'linear')
    single=v2_missing.predict(t,unit,minute,calibration,'single',.5)
    blend=v2_missing.predict(t,unit,minute,calibration,'blend_full',1.)
    if not np.isfinite(linear[0]):
        continue
    for method,p in [('linear',linear),('single',single),('blend_full',blend)]:
        rows.append(dict(recording=rec,unit=unit,minute=minute,truth=truth,
                         method=method,prediction=p[0],reference=linear[0],
                         gap_length=p[2],donor_available=p[1]))
detail=pd.DataFrame(rows)
detail.to_csv(ROOT/'outputs/v2_unseen_proxy_predictions.csv',index=False)
detail['error']=(detail.truth-detail.prediction).abs()
detail['reference_error']=(detail.truth-detail.reference).abs()
summary=detail.groupby(['recording','unit','method']).agg(n=('error','size'),
        mae=('error','mean'),reference_mae=('reference_error','mean')).reset_index()
summary['normalized_mae']=summary.mae/summary.reference_mae.replace(0,np.nan)
summary.to_csv(ROOT/'outputs/v2_unseen_proxy_scores.csv',index=False)
print('Masked I-M observations:',len(holdouts))
print(summary.to_string(index=False))
