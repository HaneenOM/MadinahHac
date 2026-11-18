"""Prompt 36: saved-prediction Friday B/G review, no model fit or new predictions.

One observable hypothesis only: absence of the preferred reference donor at
now may identify recurring errors. No threshold search or candidate is tested.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT,table,last_pair
from v2_experiments import direct_feature
from v2_hidden import REFERENCE_DONOR

OUT=ROOT/'outputs/friday_hidden_future_diagnostic'
FIXED=ROOT/'outputs/local_score'
OLD=ROOT/'outputs/friday_arrival_trial'
UNITS=['cam_B','cam_G']
DONORS=[f'cam_{c}' for c in 'ACDEFH']


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(g):
    ref=g.reference_error.mean()
    return dict(targets=len(g),origins=g['now'].nunique(),mae=g.error.mean(),reference_mae=ref,
        normalized=g.error.mean()/ref if ref>0 else np.nan,bias=g.signed.mean(),
        preferred_absent_targets=int(g.preferred_absent.sum()),
        preferred_absent_origins=g[g.preferred_absent]['now'].nunique(),
        exact_donors_min=int(g.exact_donors.min()),recent_donors_min=int(g.recent_donors.min()))


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/'outputs').rglob('submission*.csv'))+list((ROOT/'src').glob('*.py'))
    paths+=list((ROOT/'t05_forecast').glob('*.py'))+list((ROOT/'models/current_baseline').rglob('*.*'))
    paths+=list(FIXED.rglob('*.*'))+list(OLD.glob('*.*'))
    paths+=list((ROOT/'outputs/friday_hidden_diagnostic').glob('*.*'))
    paths+=list((ROOT/'data/raw').glob('*.csv'))+[ROOT/'notebooks/current_baseline.ipynb']
    protected={str(p):sha(p) for p in paths if p.is_file() and p.resolve()!=Path(__file__).resolve()}
    for name,value in json.loads((FIXED/'case_hashes.json').read_text()).items():assert sha(FIXED/name)==value
    spec=dict(hypothesis='Preferred reference donor absent exactly at now causes recurring Friday B/G hidden-future errors',
        observable='existing preferred-donor availability, no error-derived thresholds; each camera/origin counted once as support',
        predictions='reuse fixed accepted baseline and date-excluded secondary baselines from earlier Friday arrival trial',
        no_fit=True,no_candidate=True,no_tuning=True)
    (OUT/'design.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    meta=pd.read_csv(ROOT/'data/raw/recordings.csv').set_index('recording')
    cases=pd.read_csv(FIXED/'cases.csv');cases=cases[cases.kind.eq('hidden-future') & cases.unit.isin(UNITS)]
    cache=pd.read_csv(FIXED/'predictions.csv')[['case_id','prediction','reference']]
    primary=cases.merge(cache,on='case_id',validate='one_to_one')
    primary=primary.merge(pd.read_csv(FIXED/'labels.csv'),on='case_id',validate='one_to_one')
    secondary=pd.read_csv(OLD/'secondary_cases.csv')
    previous=pd.read_csv(OLD/'scored.csv')[['case_id','baseline','reference','truth']].rename(columns={'baseline':'prediction'})
    secondary=secondary.merge(previous,on='case_id',validate='one_to_one')
    assert len(primary)==322 and len(secondary)==84
    # Saved primary values independently reproduce the historical Friday trial.
    match=primary.merge(previous,on='case_id',validate='one_to_one',suffixes=('','_old'))
    for c in ['prediction','reference','truth']:np.testing.assert_allclose(match[c],match[c+'_old'],rtol=0,atol=1e-9)
    fits=json.loads((OLD/'verification.json').read_text())['fit_details']
    for f in fits:assert f['day'] not in f['fit_dates']
    detail=pd.concat([primary,secondary],ignore_index=True)
    detail['now']=detail['now'].astype(int);detail['horizon']=detail.minute-detail['now']
    assert detail.horizon.between(1,15).all() and detail.case_id.is_unique
    feeds=table(pd.read_csv(FIXED/'released_feed.csv'))
    # Reconstruct secondary released INPUTS only, using the same metadata mask.
    raw=train[train.recording.eq('20260904_1105')]
    mask=test[test.recording.eq('20260918_1100') & test.id.isna() & ~test.unit.isin(UNITS)]
    visible=mask[['unit','minute_of_day']].merge(raw[['recording','unit','minute_of_day','people']],on=['unit','minute_of_day'],how='inner')
    feeds.update(table(visible))
    metadata=[];checks=0
    for (name,unit,now),g in detail.groupby(['recording','unit','now']):
        feed=feeds[name]
        assert all(u not in feed or feed[u].isna().all() for u in UNITS)
        prefix=feed.drop(columns=UNITS,errors='ignore').loc[:now]
        exact=prefix.reindex(index=[now],columns=DONORS).iloc[0]
        preferred=REFERENCE_DONOR[unit]
        recent=0;age=np.nan;slope=np.nan
        for donor in DONORS:
            if donor not in prefix:continue
            last,donor_age,donor_slope=last_pair(prefix[donor],now,30)
            recent+=int(np.isfinite(last))
            if donor==preferred:age=donor_age;slope=donor_slope
        hidden=feed.copy()
        for u in UNITS:hidden[u]=1e8+np.arange(len(hidden))
        future=feed.copy();future.loc[future.index>now]=1e9
        for r in g.itertuples():
            original=np.array(direct_feature(feed,unit,now,int(r.horizon)))
            for altered in [hidden,future]:
                np.testing.assert_allclose(original,direct_feature(altered,unit,now,int(r.horizon)),rtol=0,atol=0,equal_nan=True)
                checks+=1
        metadata.append(dict(recording=name,unit=unit,now=now,
            preferred_donor=preferred,preferred_absent=not np.isfinite(exact.get(preferred,np.nan)),
            preferred_last_age=age,preferred_slope=slope,
            exact_donors=int(exact.notna().sum()),recent_donors=recent))
    detail=detail.merge(pd.DataFrame(metadata),on=['recording','unit','now'],validate='many_to_one')
    detail['window_id']=detail.recording+':'+detail['now'].astype(str)
    detail['horizon_band']=pd.cut(detail.horizon,[0,5,10,15],labels=['1-5','6-10','11-15']).astype(str)
    detail['signed']=detail.prediction-detail.truth;detail['error']=detail.signed.abs()
    detail['reference_error']=(detail.reference-detail.truth).abs()
    for name,g in detail.groupby('recording'):
        intervals=g.groupby('now').minute.agg(['min','max']).sort_index()
        assert (intervals['min'].iloc[1:].to_numpy()>intervals['max'].iloc[:-1].to_numpy()).all()
    detail.to_csv(OUT/'scored_diagnostic.csv',index=False)
    for groups,suffix in [(['day','recording','unit'],'recording_camera'),
        (['day','recording','unit','horizon_band'],'horizon'),
        (['day','recording','unit','window_id'],'window'),
        (['day','recording','unit','preferred_absent'],'preferred_availability')]:
        rows=[]
        for key,g in detail.groupby(groups):
            key=key if isinstance(key,tuple) else (key,)
            rows.append(dict(zip(groups,key))|summarize(g))
        pd.DataFrame(rows).to_csv(OUT/f'by_{suffix}.csv',index=False)
    coverage=[];loo=[]
    for (name,unit),g in detail.groupby(['recording','unit']):
        source=train[train.recording.eq(name)];first=int(source.minute_of_day.min());last=int(source.minute_of_day.max())
        expected=test[test.recording.eq('20260918_1100')&test.kind.eq('hidden-future')&test.unit.eq(unit)&test.minute_of_day.between(first,last)]
        coverage.append(dict(recording=name,unit=unit,recording_first=first,recording_last=last,
            metadata_targets=len(expected),known_targets=len(g),coverage=len(g)/len(expected),
            metadata_origins=expected.now_minute_of_day.nunique(),known_origins=g['now'].nunique(),
            E_training_rows=int(source.unit.eq('cam_E').sum())))
        for window in g.window_id.unique():
            kept=g[g.window_id.ne(window)]
            loo.append(dict(recording=name,unit=unit,omitted_window=window,**summarize(kept)))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(loo).to_csv(OUT/'leave_one_window_out.csv',index=False)
    # Rank camera/origin summaries, never adjacent target rows as independent events.
    windows=pd.read_csv(OUT/'by_window.csv')
    windows.sort_values('mae',ascending=False).to_csv(OUT/'largest_camera_windows.csv',index=False)
    prior=pd.read_csv(OLD/'scores.csv');prior[prior.window.eq('all') & prior.unit.isin(UNITS)].to_csv(OUT/'previous_arrival_trial.csv',index=False)
    assert all(sha(p)==value for p,value in protected.items())
    result=dict(candidate_tested=False,models_fitted=0,t05_fits=0,new_predictions_generated=0,
        targets=len(detail),dates=detail.day.nunique(),recordings=detail.recording.nunique(),
        independent_target_intervals=detail[['recording','now']].drop_duplicates().shape[0],
        camera_origins=detail[['recording','unit','now']].drop_duplicates().shape[0],
        separate_hidden_and_future_feature_checks=checks,protected_files=len(protected),
        protected_files_unchanged=True,protected_sha256=protected,runtime_seconds=time.perf_counter()-started)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'by_recording_camera.csv').round(6).to_string(index=False))
    print(pd.read_csv(OUT/'by_preferred_availability.csv').round(6).to_string(index=False))
    print(pd.read_csv(OUT/'coverage.csv').round(6).to_string(index=False))
    print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))


if __name__=='__main__':main()
