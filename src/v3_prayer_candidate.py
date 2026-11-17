"""Apply supported prayer forecasts and Saturday hidden-future predictions."""
from __future__ import annotations

import csv
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import ROOT, load, locate_data, table
from v2_candidate import HF_WEIGHT
from prayer_experiments import (fit_direct_variant, fit_forecast_variant,
    predict_direct_variant, predict_forecast_variant, timetable)

PRAYER_FORECAST = True
PRAYER_HF_WEEKDAYS = ('Sat',)


def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main(output_path: str | None = None):
    train,test,sample,recordings=load(locate_data(None))
    out=ROOT/'outputs'
    schedule=timetable(ROOT/'data/raw/prayer_times_madinah.csv',train,test)
    v2=read(out/'submission_v2.csv')
    v2_original={r['id']:r['people'] for r in v2}
    v1=read(out/'submission_v1_0.987.csv')
    assert len(v2)==len(v1)==len(sample)==4457
    assert [r['id'] for r in v2]==[r['id'] for r in v1]==sample.id.tolist()
    old_ref={r['id']:float(r['people']) for r in v1}
    rec_day=recordings.set_index('recording').day.to_dict()
    fitted=fit_direct_variant(table(train),rec_day,schedule,'prayer')
    forecast_fit=fit_forecast_variant(table(train),rec_day,schedule,'prayer')
    released=table(test[test.id.isna() & test.people.notna()])
    targets=test[test.id.notna()].set_index('id')
    changed=Counter(); invariance=0
    for row in v2:
        r=targets.loc[row['id']]
        if not ((r.kind=='forecast' and PRAYER_FORECAST) or
                (r.kind=='hidden-future' and r.weekday in PRAYER_HF_WEEKDAYS)):
            continue
        now=int(r.now_minute_of_day); minute=int(r.minute_of_day); h=minute-now
        t=released[r.recording]; day=r.day
        if r.kind=='forecast':
            fitted_pred=predict_forecast_variant(t,r.unit,now,h,day,schedule,forecast_fit,'prayer')
            if not np.isfinite(fitted_pred):
                continue  # retain the exact v2 fallback
        else:
            fitted_pred=predict_direct_variant(t,r.unit,now,h,day,schedule,fitted,'prayer')
        altered=t.copy(); altered.loc[altered.index>now]=1e6
        if r.kind=='forecast':
            check=predict_forecast_variant(altered,r.unit,now,h,day,schedule,forecast_fit,'prayer')
        else:
            check=predict_direct_variant(altered,r.unit,now,h,day,schedule,fitted,'prayer')
        assert np.isclose(fitted_pred,check,atol=1e-10)
        invariance+=1
        prediction=(max(0.,old_ref[row['id']]+HF_WEIGHT*(fitted_pred-old_ref[row['id']]))
                    if r.kind=='hidden-future' else fitted_pred)
        assert np.isfinite(prediction)
        value=str(float(prediction))
        if value!=row['people']:
            row['people']=value; changed[r.kind]+=1
    path=Path(output_path) if output_path else out/'submission_v3_prayer.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','people']);w.writeheader();w.writerows(v2)
    saved=read(path)
    assert len(saved)==4457 and [r['id'] for r in saved]==sample.id.tolist()
    assert len({r['id'] for r in saved})==4457
    assert all(np.isfinite(float(r['people'])) and float(r['people'])>=0 for r in saved)
    by_kind=Counter(targets.loc[r['id']].kind for r in saved
                    if v2_original[r['id']]!=r['people'])
    assert by_kind==changed
    print('Post-cutoff candidate checks:',invariance)
    print('Re-read candidate:',len(saved),'changed by task:',dict(by_kind))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-path',help='Write a new submission path instead of the default v3 file')
    args=parser.parse_args()
    main(args.output_path)
