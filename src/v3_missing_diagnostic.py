"""Mask contiguous released I-M runs, keeping observed endpoints."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import ROOT, load, locate_data, table
import v2_missing


def main():
    _, test, _, _ = load(locate_data(None))
    released = test[test.id.isna() & test.people.notna()]
    tables = table(released)
    masked = {rec:t.copy() for rec,t in tables.items()}
    gaps = []
    labels = []
    for rec,t in tables.items():
        for unit in [f'cam_{c}' for c in 'IJKLM']:
            if unit not in t: continue
            s = t[unit].dropna()
            chunks=[]; chunk=[]
            for minute in s.index:
                if chunk and minute-chunk[-1]>1:
                    chunks.append(chunk); chunk=[]
                chunk.append(int(minute))
            if chunk: chunks.append(chunk)
            # Non-overlapping, 4-10 minute artificial outages, with a released
            # endpoint on both sides and a buffer before the next outage.
            for chunk in chunks:
                if len(chunk)<8: continue
                pos=1
                while pos+5<len(chunk):
                    length=min(10,len(chunk)-pos-2)
                    if length<4: break
                    window=chunk[pos:pos+length]
                    left=chunk[pos-1]; right=chunk[pos+length]
                    gap_id=f'{rec}:{unit}:{left}-{right}'
                    gaps.append(dict(gap_id=gap_id,recording=rec,unit=unit,
                                     start=left,end=right,length=length))
                    for minute in window:
                        labels.append((gap_id,rec,unit,minute,float(t.at[minute,unit])))
                        masked[rec].at[minute,unit]=np.nan
                    pos+=length+2
    # Calibrate on the other released recording only. No pseudo-label or
    # observation from the evaluated date can fit its correction.
    calibrations={rec:v2_missing.fit_calibration({other:t for other,t in tables.items()
                                                  if other!=rec}) for rec in tables}
    rows=[]
    for gap_id,rec,unit,minute,truth in labels:
        t=masked[rec]
        calibration=calibrations[rec]
        linear=v2_missing.predict(t,unit,minute,calibration,'linear',0.)
        donor=v2_missing.predict(t,unit,minute,calibration,'blend_full',1.)
        if not np.isfinite(linear[0]): continue
        for method,p in [('endpoint',linear),('donor_correction',donor)]:
            rows.append(dict(gap_id=gap_id,recording=rec,day=str(rec)[:4]+'-'+str(rec)[4:6]+'-'+str(rec)[6:8],
                             unit=unit,minute=minute,truth=truth,method=method,
                             prediction=p[0],donor_available=p[1],gap_length=p[2]))
    detail=pd.DataFrame(rows)
    detail.to_csv(ROOT/'outputs/v3_missing_masked_predictions.csv',index=False)
    pd.DataFrame(gaps).to_csv(ROOT/'outputs/v3_missing_masked_gaps.csv',index=False)
    detail['error']=(detail.truth-detail.prediction).abs()
    summary=detail.groupby(['day','recording','unit','method']).agg(
        n=('error','size'),mae=('error','mean'),gaps=('gap_id','nunique'),
        donor_covered=('donor_available',lambda x:int((x>0).sum()))).reset_index()
    summary.to_csv(ROOT/'outputs/v3_missing_masked_scores.csv',index=False)
    print('Distinct gaps/cameras/dates/scored observations:',
          detail.gap_id.nunique(),detail.unit.nunique(),detail.day.nunique(),
          detail[detail.method.eq('endpoint')].shape[0])
    print('Gaps by length:',pd.DataFrame(gaps).length.value_counts().sort_index().to_dict())
    print(summary.to_string(index=False))
    print('Overall:',detail.groupby('method').agg(n=('error','size'),mae=('error','mean'),
              donor_covered=('donor_available',lambda x:int((x>0).sum()))).to_string())


if __name__=='__main__': main()
