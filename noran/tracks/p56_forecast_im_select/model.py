"""Training-only blocked selection of when to apply pooled I-M residuals.
The assessment recording is excluded from regression and applicability selection.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('p56_im_original',ROOT/'noran/tracks/p56_forecast_im/model.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
UNITS=base.UNITS;features=base.features;examples=base.examples

def fit(training,day,context):
    cols=['recording','day','unit','minute_of_day','people']
    rows=pd.concat([training[cols],context.get('test_released',training.iloc[:0])[cols]],ignore_index=True)
    rows=rows.drop_duplicates(['recording','unit','minute_of_day'])
    if day is not None:rows=rows[rows.day.ne(day)]
    assert day is None or not rows.day.eq(day).any()
    x,y,meta=examples(rows)
    if len(y)<100:return None
    event={}
    for rec,g in rows.groupby('recording'):
        minutes=np.sort(g.minute_of_day.unique());blocks=np.cumsum(np.r_[True,np.diff(minutes)>1])
        event.update({(rec,int(m)):f'{rec}|{int(b)}' for m,b in zip(minutes,blocks)})
    meta['event']=[event[(r.recording,r.minute)] for r in meta.itertuples()]
    oof=np.zeros(len(meta))
    for e in sorted(meta.event.unique()):
        hold=meta.event.eq(e).to_numpy()
        if (~hold).sum()<100:continue
        model=Ridge(alpha=10.).fit(x[~hold],y[~hold])
        oof[hold]=model.predict(x[hold])
    meta['gain']=np.abs(y*meta.scale)-np.abs((y-oof)*meta.scale)
    selected=[];audit=[]
    for unit,g in meta.groupby('unit'):
        w=g.groupby('event').gain.sum().sort_values(ascending=False)
        remain=g[~g.event.isin(w.head(1).index)]
        gain=float(g.gain.mean());robust=float(remain.gain.mean()) if len(remain) else -np.inf
        use=len(w)>=4 and gain>0 and robust>0
        if use:selected.append(unit)
        audit.append(dict(unit=unit,windows=len(w),gain=gain,gain_remove_best=robust,selected=use))
    return dict(model=Ridge(alpha=10.).fit(x,y),training_rows=len(y),
                training_recordings=sorted(meta.recording.unique()),eligible_units=selected,selection_audit=audit)

def predict(case,t,model,context):
    if model is None or case.unit not in UNITS or case.unit not in t:return np.nan
    made=features(t[case.unit],int(case.now),int(case.minute)-int(case.now))
    if made is None:return np.nan
    x,old,scale,_=made
    if case.unit not in model['eligible_units']:return old
    return float(max(0.,old+scale*model['model'].predict(x[None])[0]))
