"""Local-only alternative to the current donor/Ridge hidden reconstructor."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, load, locate_data, table
from v2_missing import interp_at
from hidden_candidate import HIDDEN_CANDIDATE
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import v2_hidden

OUT=ROOT/'outputs/hidden_reconstruction_trial'
FIXED=ROOT/'outputs/local_score'
BASE=ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
UNITS=[f'cam_{c}' for c in 'ABCDEFGH']
PARAMS=dict(loss='absolute_error',max_iter=75,learning_rate=.07,max_leaf_nodes=7,
            max_depth=3,min_samples_leaf=25,l2_regularization=15.,early_stopping=False,random_state=26)
DESIGN=dict(family='HistGradientBoostingRegressor',params=PARAMS,
    limitation='Global donor ratios/linear Ridge blend underestimates high B/D and overestimates low D even when donors are present; test nonlinear dependence on simultaneous visible-camera composition.',
    inputs='Exact same-minute visible A-H log levels, shares, seen indicators, total/availability, clock and scheduled prayer time. No target-camera history, interpolation or template.',
    augmentation='One fixed 20% donor dropout copy of training rows; no evaluation-mask fitting.',
    evaluation='Fixed 843 Sep11/12 cases; separately Sep4/5 released targets on a fixed five-minute grid. Entire evaluated DATE excluded from every fit/preprocessing step.',
    fallback='Unchanged baseline if fewer than two exact visible donors or no trained unit model.',
    acceptance='>=1% mean MAE gain on BOTH primary dates; >=3 winning windows/date; all leave-one-window-out gains positive; no camera MAE regression above max(0.5 person,10% baseline); positive mean gain on each additional date and >=2 additional recording wins.')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_cases():
    for name,expected in json.loads((FIXED/'case_hashes.json').read_text()).items():
        assert sha(FIXED/name)==expected
    cases=pd.read_csv(FIXED/'cases.csv').query("kind=='hidden'").copy()
    feed=pd.read_csv(FIXED/'released_feed.csv')
    cases['window_id']=cases.recording+':'+(cases.minute//30*30).astype(str)
    keys=pd.MultiIndex.from_frame(feed[['recording','unit','minute_of_day']])
    targets=pd.MultiIndex.from_arrays([cases.recording,cases.unit,cases.minute])
    assert not targets.isin(keys).any() and len(cases)==843
    return cases,table(feed)


def inspect():
    OUT.mkdir(parents=True,exist_ok=True)
    cases,feeds=fixed_cases()
    baseline=pd.read_csv(FIXED/'predictions.csv').query("kind=='hidden'")[['case_id','prediction','reference']]
    detail=cases.merge(baseline,on='case_id',validate='one_to_one')
    availability=[]
    for r in detail.itertuples():
        frame=feeds[r.recording].drop(columns=list(v2_hidden.PATTERNS[r.unit]),errors='ignore')
        donors=[c for c in frame if c.startswith('cam_')]
        exact=frame.reindex(index=[int(r.minute)],columns=donors).iloc[0]
        primary=v2_hidden.REFERENCE_DONOR[r.unit]
        availability.append(dict(case_id=r.case_id,exact_donors=int(exact.notna().sum()),
            interpolated_donors=sum(np.isfinite(interp_at(frame[c],int(r.minute))) for c in donors),
            primary_exact=bool(pd.notna(exact.get(primary,np.nan)))))
    detail=detail.merge(pd.DataFrame(availability),on='case_id',validate='one_to_one')
    detail=detail.merge(pd.read_csv(FIXED/'labels.csv'),on='case_id',validate='one_to_one')
    detail['error']=(detail.prediction-detail.truth).abs()
    detail['reference_error']=(detail.reference-detail.truth).abs()
    detail['crowd']=pd.cut(detail.truth,[-1,20,40,np.inf],labels=['0-20','20-40','over40']).astype(str)
    detail['signed_error']=detail.prediction-detail.truth
    for groups,name in [(['recording','unit'],'camera'),(['recording','unit','window_id'],'window'),
                        (['unit','exact_donors','primary_exact'],'availability'),(['unit','crowd'],'crowd')]:
        result=detail.groupby(groups).agg(targets=('case_id','size'),windows=('window_id','nunique'),
            mae=('error','mean'),reference_mae=('reference_error','mean'),bias=('signed_error','mean')).reset_index()
        result['normalized']=result.mae/result.reference_mae
        result.to_csv(OUT/f'baseline_by_{name}.csv',index=False)
    detail.nlargest(25,'error').to_csv(OUT/'largest_current_errors.csv',index=False)
    detail.to_csv(OUT/'baseline_inspection.csv',index=False)
    print(pd.read_csv(OUT/'baseline_by_camera.csv').to_string(index=False))
    print('Crowd groups:');print(pd.read_csv(OUT/'baseline_by_crowd.csv').to_string(index=False))
    print('Worst camera/windows:');print(pd.read_csv(OUT/'baseline_by_window.csv').nlargest(8,'mae').to_string(index=False))


def schedule():
    frame=pd.read_csv(ROOT/'data/raw/prayer_times_madinah.csv')
    assert frame.day.is_unique and frame.columns.tolist()==['day','fajr','dhuhr','asr','maghrib','isha']
    for col in frame.columns[1:]:assert frame[col].str.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d').all()
    return {r.day:np.array([int(getattr(r,c)[:2])*60+int(getattr(r,c)[3:]) for c in frame.columns[1:]])
            for r in frame.itertuples()}


def make_features(frame,unit,days,timetable):
    # Reindexing removes BOTH B and G for either Friday target, before ANY
    # aggregation. Evaluation hidden readings can never enter preprocessing.
    donors=[u for u in UNITS if u not in v2_hidden.PATTERNS[unit]]
    x=frame.reindex(columns=donors).astype(float)
    assert not set(x.columns).intersection(v2_hidden.PATTERNS[unit])
    available=x.notna().sum(axis=1)
    total=x.sum(axis=1,min_count=1)
    result=pd.DataFrame(index=frame.index)
    for c in donors:
        result[c+'_log']=np.log1p(x[c].clip(lower=0))
        result[c+'_share']=x[c]/total.replace(0,np.nan)
        result[c+'_seen']=x[c].notna().astype(float)
    minute=np.asarray(frame.index.get_level_values(-1),int)
    result['visible_total_log']=np.log1p(total.clip(lower=0))
    result['available']=available
    result['clock_sin']=np.sin(2*np.pi*minute/1440)
    result['clock_cos']=np.cos(2*np.pi*minute/1440)
    result['friday']=[int(pd.Timestamp(day).day_of_week==4) for day in days]
    result['dhuhr_offset']=[int(m-timetable[d][1]) for m,d in zip(minute,days)]
    result['nearest_prayer']=[int(m-timetable[d][np.argmin(np.abs(timetable[d]-m))]) for m,d in zip(minute,days)]
    return result,available


def fit_new(train,timetable):
    tables=table(train)
    full=pd.concat(tables.values(),keys=tables.keys(),names=['recording','minute_of_day'])
    rec_day=train.drop_duplicates('recording').set_index('recording').day.to_dict()
    models={};manifest=[]
    for unit,mask in v2_hidden.PATTERNS.items():
        donors=[u for u in UNITS if u not in mask]
        if unit not in full:continue
        valid=full[unit].notna() & full.reindex(columns=donors).notna().sum(axis=1).ge(2)
        raw=full.loc[valid].reindex(columns=donors)
        if len(raw)<100:continue
        days=[rec_day[n] for n in raw.index.get_level_values(0)]
        x,_=make_features(raw,unit,days,timetable)
        rng=np.random.default_rng(26)
        dropped=raw.mask(rng.random(raw.shape)<.2)
        augmented,seen=make_features(dropped,unit,days,timetable)
        keep=seen.ge(2)
        design=pd.concat([x,augmented[keep]],ignore_index=True)
        target=full.loc[valid,unit].to_numpy(float)
        y=np.concatenate([target,target[keep.to_numpy()]])
        model=HistGradientBoostingRegressor(**PARAMS).fit(design,y)
        models[unit]=model
        manifest.append(dict(unit=unit,training_rows=len(raw),augmented_rows=len(design),
                             features=x.shape[1],training_recordings=sorted(raw.index.get_level_values(0).unique())))
    return models,manifest


def extra_cases(train):
    case_rows=[];labels=[];feeds={}
    for name,g in train[train.day.isin(['2026-09-04','2026-09-05'])].groupby('recording'):
        day=g.day.iloc[0];masked={'cam_B','cam_G'} if pd.Timestamp(day).day_of_week==4 else {'cam_D'}
        feeds.update(table(g[~g.unit.isin(masked)]))
        first=int(g.minute_of_day.min())
        targets=g[g.unit.isin(masked)&g.people.notna()&((g.minute_of_day-first)%5==0)]
        for r in targets.itertuples():
            minute=int(r.minute_of_day);case_id=f'extra:{name}:{r.unit}:{minute}'
            case_rows.append(dict(case_id=case_id,day=day,recording=name,unit=r.unit,minute=minute,
                                  window_id=f'{name}:{minute//30*30}',scope='additional'))
            labels.append(dict(case_id=case_id,truth=float(r.people)))
    return pd.DataFrame(case_rows),pd.DataFrame(labels),feeds


def predict_fold(cases,feeds,training,timetable):
    baseline_models=v2_hidden.fit(table(training))
    new_models,manifest=fit_new(training,timetable)
    rows=[];checks=0
    for (name,unit),targets in cases.groupby(['recording','unit']):
        frame=feeds[name]
        minutes=targets.minute.to_numpy(int)
        # No interpolation: each feature is read exactly at its target minute.
        raw=frame.reindex(index=minutes)
        days=targets.day.tolist()
        x,available=make_features(raw,unit,days,timetable)
        altered=frame.copy()
        # Change EVERY hidden value in the whole evaluation recording, adding
        # missing hidden columns as well. Neither candidate nor baseline may move.
        for c in v2_hidden.PATTERNS[unit]:
            altered[c]=np.arange(len(altered),dtype=float)*1e6+1e9
        x2,available2=make_features(altered.reindex(index=minutes),unit,days,timetable)
        pd.testing.assert_frame_equal(x,x2);pd.testing.assert_series_equal(available,available2)
        candidate=np.maximum(new_models[unit].predict(x),0.) if unit in new_models else np.full(len(x),np.nan)
        candidate2=np.maximum(new_models[unit].predict(x2),0.) if unit in new_models else candidate
        np.testing.assert_allclose(candidate,candidate2,rtol=0,atol=0,equal_nan=True)
        for i,r in enumerate(targets.itertuples()):
            minute=int(r.minute)
            reference,_=v2_hidden.predict(frame,unit,minute,baseline_models,'single')
            method,weight=HIDDEN_CANDIDATE[unit]
            fitted,_=v2_hidden.predict(frame,unit,minute,baseline_models,method)
            baseline=max(0.,reference+weight*(fitted-reference))
            ref2,_=v2_hidden.predict(altered,unit,minute,baseline_models,'single')
            fit2,_=v2_hidden.predict(altered,unit,minute,baseline_models,method)
            np.testing.assert_allclose([reference,baseline],[ref2,max(0.,ref2+weight*(fit2-ref2))],rtol=0,atol=1e-10)
            use=bool(available.iloc[i]>=2 and np.isfinite(candidate[i]))
            value=float(candidate[i]) if use else baseline
            rows.append(dict(case_id=r.case_id,day=r.day,recording=name,unit=unit,minute=minute,
                window_id=r.window_id,scope=r.scope,baseline=baseline,reference=reference,candidate=value,
                exact_donors=int(available.iloc[i]),used_new_method=use))
            # All donor inputs correspond to released feed values, at the exact
            # minute. Hidden columns have already been removed by construction.
            for donor in [u for u in UNITS if u not in v2_hidden.PATTERNS[unit]]:
                value=raw.iloc[i].get(donor,np.nan)
                if np.isfinite(value):assert donor in frame and float(frame.loc[minute,donor])==value
            checks+=1
    return pd.DataFrame(rows),manifest,checks


def summarize(scored,prefix):
    frame=scored.copy()
    for col in ('baseline','candidate','reference'):frame[col+'_error']=(frame[col]-frame.truth).abs()
    for groups,suffix in [(['day','recording','unit'],'recording_camera'),(['day','unit'],'camera'),
                           (['day'],'date'),(['day','recording','unit','window_id'],'camera_window'),
                           (['day','recording','window_id'],'window'),(['day','unit','used_new_method'],'coverage')]:
        result=frame.groupby(groups).agg(targets=('case_id','size'),windows=('window_id','nunique'),
            timestamps=('minute','nunique'),new_predictions=('used_new_method','sum'),
            baseline_mae=('baseline_error','mean'),candidate_mae=('candidate_error','mean'),
            reference_mae=('reference_error','mean')).reset_index()
        result['baseline_normalized']=result.baseline_mae/result.reference_mae.replace(0,np.nan)
        result['candidate_normalized']=result.candidate_mae/result.reference_mae.replace(0,np.nan)
        result['gain']=result.baseline_mae-result.candidate_mae
        result.to_csv(OUT/f'{prefix}_by_{suffix}.csv',index=False)
    loo=[]
    for groups,kind in [(['day'],'date'),(['day','unit'],'camera')]:
        for key,group in frame.groupby(groups):
            key=key if isinstance(key,tuple) else (key,)
            for window in group.window_id.unique():
                kept=group[group.window_id.ne(window)]
                loo.append(dict(kind=kind,day=key[0],unit=key[1] if len(key)>1 else 'all',omitted_window=window,
                    targets=len(kept),baseline_mae=kept.baseline_error.mean(),candidate_mae=kept.candidate_error.mean(),
                    gain=(kept.baseline_error-kept.candidate_error).mean()))
    pd.DataFrame(loo).to_csv(OUT/f'{prefix}_leave_one_window_out.csv',index=False)
    frame.to_csv(OUT/f'{prefix}_scored.csv',index=False)


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'baseline_inspection.csv').exists():inspect()
    protected_paths=list((ROOT/'outputs').glob('submission*.csv'))+list((ROOT/'t05_forecast').glob('*.py'))
    protected_paths+=list((ROOT/'src').glob('*.py'))+[FIXED/n for n in ('cases.csv','labels.csv','released_feed.csv','predictions.csv','score.json')]
    protected={str(p):sha(p) for p in protected_paths}
    (OUT/'design.json').write_text(json.dumps(DESIGN,indent=2),encoding='utf-8')
    train,_,_,_=load(locate_data(None)); timetable=schedule()
    primary,feeds=fixed_cases();primary['scope']='fixed'
    extra,extra_labels,extra_feeds=extra_cases(train)
    all_cases=pd.concat([primary[['case_id','day','recording','unit','minute','window_id','scope']],extra],ignore_index=True)
    all_feeds={**feeds,**extra_feeds}
    outputs=[];manifests=[];checks=0
    with threadpool_limits(limits=4):
        for day,cases in all_cases.groupby('day'):
            training=train[train.day.ne(day)].copy()
            assert day not in set(training.day) and not set(cases.recording).intersection(training.recording)
            # Even changes to evaluation hidden values in the ORIGINAL training
            # dataframe cannot alter any fitting/preprocessing or donor selection.
            mutated=train.copy();hidden={'cam_B','cam_G'} if pd.Timestamp(day).day_of_week==4 else {'cam_D'}
            mutated.loc[mutated.day.eq(day)&mutated.unit.isin(hidden),'people']=1e9
            pd.testing.assert_frame_equal(training,mutated[mutated.day.ne(day)])
            pred,manifest,ncheck=predict_fold(cases,all_feeds,training,timetable)
            outputs.append(pred);checks+=ncheck
            manifests.append(dict(heldout_day=day,excluded_recordings=sorted(cases.recording.unique()),
                                  training_recordings=sorted(training.recording.unique()),models=manifest))
    predictions=pd.concat(outputs,ignore_index=True)
    assert len(predictions)==len(all_cases) and predictions.case_id.is_unique
    assert np.isfinite(predictions[['baseline','candidate','reference']]).all().all()
    predictions.to_csv(OUT/'predictions.csv',index=False)
    (OUT/'fit_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    saved=pd.read_csv(FIXED/'predictions.csv').set_index('case_id')
    fixed_pred=predictions[predictions.scope.eq('fixed')].set_index('case_id')
    np.testing.assert_allclose(fixed_pred.baseline,saved.loc[fixed_pred.index,'prediction'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(fixed_pred.reference,saved.loc[fixed_pred.index,'reference'],rtol=0,atol=1e-9)
    labels=pd.concat([pd.read_csv(FIXED/'labels.csv')[['case_id','truth']],extra_labels])
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    for scope,prefix in [('fixed','fixed'),('additional','additional')]:summarize(scored[scored.scope.eq(scope)],prefix)
    days=pd.read_csv(OUT/'fixed_by_date.csv');cameras=pd.read_csv(OUT/'fixed_by_camera.csv')
    windows=pd.read_csv(OUT/'fixed_by_window.csv');loo=pd.read_csv(OUT/'fixed_leave_one_window_out.csv').query("kind=='date'")
    winning=windows.assign(win=windows.gain.gt(0)).groupby('day').win.sum()
    additional_days=pd.read_csv(OUT/'additional_by_date.csv')
    additional_rec=pd.read_csv(OUT/'additional_by_recording_camera.csv')
    rec_gains=additional_rec.groupby('recording').gain.mean()
    camera_checks=pd.concat([cameras.assign(evaluation='fixed'),additional_rec.assign(evaluation='additional')],ignore_index=True)
    bad=camera_checks[camera_checks.candidate_mae-camera_checks.baseline_mae>np.maximum(.5,camera_checks.baseline_mae*.1)]
    accepted=bool((days.gain>=days.baseline_mae*.01).all() and loo.gain.gt(0).all()
        and winning.ge(3).all() and bad.empty and additional_days.gain.gt(0).all() and rec_gains.gt(0).sum()>=2)
    assert all(sha(path)==expected for path,expected in protected.items())
    report=dict(accepted=accepted,status='needs isolated full benchmark and submission verification' if accepted else 'rejected',
        model_family='HistGradientBoostingRegressor',model_fits=sum(len(m['models']) for m in manifests),t05_fits=0,
        fixed_cases=len(fixed_pred),additional_cases=int(predictions.scope.eq('additional').sum()),
        invariance_and_visible_mask_checks=checks,baseline_reproduced_atol=1e-9,
        winning_windows=winning.to_dict(),major_camera_regressions=bad.fillna('').to_dict('records'),
        protected_files_unchanged=True,candidate_submission_created=False,
        elapsed_seconds=time.perf_counter()-started,protected_sha256=protected)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(cameras.to_string(index=False));print('Additional recordings:');print(additional_rec.to_string(index=False))
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
    if args.inspect:inspect()
    else:main()
