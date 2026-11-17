"""Causal released-observation proxy for forecast on unseen I-M cameras."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import ROOT, FEATURES, feature, fit_forecast, load, locate_data, table
from prayer_experiments import (fit_forecast_variant, predict_forecast_variant,
                                timetable)


def main():
    train,test,_,recordings=load(locate_data(None))
    schedule=timetable(ROOT/'data/raw/prayer_times_madinah.csv',train,test)
    train_t=table(train)
    baseline=fit_forecast(train_t,'log')[0]
    prayer=fit_forecast_variant(train_t,recordings.set_index('recording').day.to_dict(),
                                schedule,'prayer')
    released=table(test[test.id.isna() & test.people.notna()])
    rows=[]
    for rec,t in released.items():
        day=rec[:4]+'-'+rec[4:6]+'-'+rec[6:8]
        for unit in [f'cam_{x}' for x in 'IJKLM']:
            if unit not in t: continue
            s=t[unit].dropna()
            for minute,truth in s.items():
                if int(minute)%8: continue
                for h in (5,10,15):
                    now=int(minute)-h
                    f=feature(t,unit,now,h)
                    if not np.isfinite(f[0]): continue
                    base=max(0.,float(np.expm1(np.log1p(f[0])+
                        baseline.predict(pd.DataFrame([f],columns=FEATURES))[0])))
                    pray=predict_forecast_variant(t,unit,now,h,day,schedule,prayer,'prayer')
                    for variant,pred in [('v2',base),('prayer',pray)]:
                        rows.append(dict(day=day,recording=rec,unit=unit,
                            minute=minute,horizon=h,truth=truth,variant=variant,
                            prediction=pred,reference=f[0]))
    detail=pd.DataFrame(rows)
    detail.to_csv(ROOT/'outputs/v3_unseen_forecast_proxy_predictions.csv',index=False)
    detail['error']=(detail.truth-detail.prediction).abs()
    detail['ref_error']=(detail.truth-detail.reference).abs()
    summary=detail.groupby(['day','unit','variant']).agg(n=('error','size'),
        mae=('error','mean'),reference_mae=('ref_error','mean')).reset_index()
    summary['normalized_mae']=summary.mae/summary.reference_mae
    summary.to_csv(ROOT/'outputs/v3_unseen_forecast_proxy_scores.csv',index=False)
    print(summary.to_string(index=False))
    print('Overall:',detail.groupby('variant').agg(n=('error','size'),
        mae=('error','mean'),reference_mae=('ref_error','mean')).to_string())


if __name__=='__main__': main()
