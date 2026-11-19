"""Pooled normalized residual forecast for I-M; fit only on explicitly released rows.
One fixed Ridge recipe; no T05. Each evaluation date is entirely excluded from fitting.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from im_forecast_fallback_trial import context as last_context,predict as v7_predict
UNITS=[f'cam_{c}' for c in 'IJKLM']

def features(series,now,horizon):
    past=series.loc[:int(now)].dropna()
    if len(past)<3:return None
    ref,base,ok=v7_predict(dict(now=int(now),**last_context(series,int(now))))
    if not np.isfinite(base):return None
    scale=max(float(base),5.)
    short=past.loc[int(now)-4:int(now)];long=past.loc[int(now)-9:int(now)]
    if short.empty:short=past.tail(1)
    if long.empty:long=past.tail(1)
    def slope(s):
        if len(s)<3:return 0.
        x=s.index.to_numpy(float);x=x-x.mean()
        return float(x@s.to_numpy(float)/(x@x+1e-9))
    s5,s10=slope(short),slope(long)
    x=[horizon/15,(past.iloc[-1]-base)/scale,(short.median()-base)/scale,
       (long.mean()-base)/scale,horizon*s5/scale,horizon*s10/scale,
       float(short.std(ddof=0))/scale,len(long)/10]
    return np.asarray(x,float),float(base),scale,float(ref)

def examples(rows):
    features_out,labels,metadata=[],[],[]
    for (rec,unit),g in rows[rows.unit.isin(UNITS)&rows.people.notna()].groupby(['recording','unit']):
        s=g.set_index('minute_of_day').people.sort_index()
        for minute,truth in s.items():
            for h in range(1,16):
                now=int(minute)-h;made=features(s,now,h)
                if made is None:continue
                x,base,scale,ref=made
                features_out.append(x);labels.append((float(truth)-base)/scale)
                metadata.append(dict(recording=rec,day=g.day.iloc[0],unit=unit,minute=int(minute),now=now,horizon=h,
                    v7=base,reference=ref,scale=scale,truth=float(truth)))
    return np.asarray(features_out),np.asarray(labels),pd.DataFrame(metadata)

def fit(training,day,context):
    cols=['recording','day','unit','minute_of_day','people']
    rows=pd.concat([training[cols],context.get('test_released',training.iloc[:0])[cols]],ignore_index=True)
    rows=rows.drop_duplicates(['recording','unit','minute_of_day'])
    if day is not None:rows=rows[rows.day.ne(day)]
    x,y,meta=examples(rows)
    if len(y)<100:return None
    model=Ridge(alpha=10.).fit(x,y)
    return dict(model=model,training_rows=len(y),training_recordings=sorted(meta.recording.unique()))

def predict(case,t,model,context):
    if model is None or case.unit not in UNITS or case.unit not in t:return np.nan
    made=features(t[case.unit],int(case.now),int(case.minute)-int(case.now))
    if made is None:return np.nan
    x,base,scale,_=made
    return float(max(0.,base+scale*model['model'].predict(x[None])[0]))
