"""One independent nonlinear forecast model, one held-out evaluation pass."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from pipeline import ROOT
from short_term_change_trial import actual_cases
from t05_forecast.core import RELEASED, load_recordings

OUT=ROOT/'outputs/independent_forecast_trial'
WEEKLY=ROOT/'outputs/weekly_forecast_diagnostic'
BASE=ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
UNITS=[f'cam_{c}' for c in 'ABCDEFGH']
META=['case_id','day','recording','unit','now','minute','horizon','window_id']
MODEL=dict(loss='absolute_error',learning_rate=.05,max_iter=100,max_leaf_nodes=15,
           max_depth=4,min_samples_leaf=40,l2_regularization=10.,early_stopping=False,random_state=25)
DESIGN=dict(model=MODEL,target='log1p(target) - log1p(last own reading)',
            own_recency_minutes=5,training_origin_step=30,prior_week_template=False,
            missing='Native NaN splits plus explicit availability, ages and point counts',
            movement='Training 75th percentile of absolute observed 10-minute log slope; >=3 points spanning >=5 minutes',
            future_change='Scoring-only abrupt category above training 90th percentile absolute log change',
            acceptance='>=3% MAE reduction on BOTH dates, positive gains after leaving out every origin, >=3 winning origins/date, no >=30-case/>=5-origin camera or horizon group regressing more than max(1 person,10% baseline).',
            selection='Fixed a priori; no tuning, no early-stopping validation, no T05 predictions used as model features or training targets.')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths_to_protect():
    paths=list((ROOT/'outputs').glob('submission*.csv'))+list((ROOT/'src').glob('*.py'))
    paths+=list((ROOT/'t05_forecast').glob('*.py'))+[ROOT/'t05_forecast_config.json']
    paths += [WEEKLY/n for n in ('cases.csv','labels.csv','predictions.csv','case_hashes.json','verification.json')]
    paths += list((ROOT/'data/raw').glob('*.csv'))
    return paths


def prayer_minutes():
    frame=pd.read_csv(ROOT/'data/raw/prayer_times_madinah.csv')
    assert frame.columns.tolist()==['day','fajr','dhuhr','asr','maghrib','isha'] and frame.day.is_unique
    for col in frame.columns[1:]:
        assert frame[col].str.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d').all()
    return {r.day:np.array([int(getattr(r,col)[:2])*60+int(getattr(r,col)[3:])
                           for col in frame.columns[1:]]) for r in frame.itertuples()}


def summary_of_prefix(rec,unit,now):
    if unit not in rec.units:return dict(level=np.nan,age=np.nan,slope=np.nan,n=0,span=0,present=0)
    col=rec.col(unit); end=max(0,min(rec.n_minutes,int(now)-rec.first+1))
    yy=rec.y[:end,col]; tt=rec.minutes[:end]
    keep=(rec.status[:end,col]==RELEASED)&np.isfinite(yy)
    t,y=tt[keep],yy[keep]
    if not len(t):return dict(level=np.nan,age=np.nan,slope=np.nan,n=0,span=0,present=0)
    age=int(now-t[-1]); recent=t>=now-10; rt,ry=t[recent],np.log1p(y[recent])
    span=int(np.ptp(rt)) if len(rt) else 0
    slope=np.nan
    if len(rt)>=3 and span>=5:
        x=rt-rt.mean(); slope=float(np.dot(x,ry)/np.dot(x,x))
    return dict(level=float(np.log1p(y[-1])) if age<=5 else np.nan,
                age=min(age,120),slope=slope,n=len(rt),span=span,present=int(age<=5))


def context_features(rec,unit,now,prayers):
    # Every camera reader slices at now BEFORE selecting or aggregating cells.
    all_context={u:summary_of_prefix(rec,u,now) for u in UNITS}
    own=all_context[unit]
    if not own['present']:return None
    col=rec.col(unit); end=int(now)-rec.first+1
    yy=rec.y[:end,col]; tt=rec.minutes[:end]
    valid=(rec.status[:end,col]==RELEASED)&np.isfinite(yy)
    t,y=tt[valid],np.log1p(yy[valid])
    def lag_level(lag):
        candidates=np.flatnonzero((t<=now-lag)&(t>=now-lag-5))
        return float(y[candidates[-1]]) if len(candidates) else np.nan
    def recent_mean(width):
        selected=y[t>=now-width]
        return float(selected.mean()) if len(selected) else np.nan
    short=t>=now-3
    slope3=np.nan
    if short.sum()>=2 and np.ptp(t[short])>0:
        x=t[short]-t[short].mean(); slope3=float(np.dot(x,y[short])/np.dot(x,x))
    pair=np.nan; interval=np.nan
    if len(t)>=2 and t[-2]>=now-10:
        interval=int(t[-1]-t[-2]); pair=float((y[-1]-y[-2])/interval)
    nearest=prayers[np.argmin(np.abs(prayers-now))]
    result=dict(own_level=own['level'],own_age=own['age'],own_n10=own['n'],own_span10=own['span'],
                own_slope10=own['slope'],own_slope3=slope3,own_pair_slope=pair,own_interval=interval,
                own_lag5=lag_level(5),own_lag10=lag_level(10),own_mean5=recent_mean(5),own_mean15=recent_mean(15),
                camera_index=UNITS.index(unit),friday=int(rec.weekday=='Fri'),
                clock_sin=float(np.sin(now*2*np.pi/1440)),clock_cos=float(np.cos(now*2*np.pi/1440)),
                nearest_prayer_now=int(now-nearest),dhuhr_now=int(now-prayers[1]))
    for camera,ctx in all_context.items():
        # The target camera is kept in the same slots across all targets.
        for key in ('level','age','slope','present'):
            result[f'{camera}_{key}']=ctx[key]
    result['available_other_cameras']=sum(ctx['present'] for camera,ctx in all_context.items() if camera!=unit)
    return result


def extract(cases,recs,timetable):
    rows=[]; checks=0
    for (name,unit,now),targets in cases.groupby(['recording','unit','now'],sort=False):
        rec=recs[name]; now=int(now); prayers=timetable[rec.day]
        context=context_features(rec,unit,now,prayers)
        altered=rec.copy(); altered.y[altered.minutes>now]=1e9
        other=context_features(altered,unit,now,prayers)
        if context is None:assert other is None
        else:
            assert context.keys()==other.keys()
            np.testing.assert_allclose(list(context.values()),list(other.values()),rtol=0,atol=0,equal_nan=True)
        checks+=len(targets)
        if context is None:continue
        for r in targets.itertuples(index=False):
            minute=int(r.minute); nearest=prayers[np.argmin(np.abs(prayers-minute))]
            rows.append(dict(**r._asdict(),**context,horizon_feature=int(r.horizon),
                             nearest_prayer_target=int(minute-nearest),dhuhr_target=int(minute-prayers[1]),
                             prayer_between=int(((prayers>now)&(prayers<=minute)).any())))
    return pd.DataFrame(rows),checks


def prepare():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    protected={str(p):sha(p) for p in paths_to_protect()}
    recs=load_recordings(ROOT/'data/raw'); timetable=prayer_minutes()
    assert all(r.day in timetable for r in recs.values())
    fixed=pd.read_csv(WEEKLY/'cases.csv'); cached=pd.read_csv(WEEKLY/'predictions.csv')
    pd.testing.assert_frame_equal(fixed[META],cached[META],check_dtype=False)
    assert len(fixed)==1566 and fixed.case_id.is_unique
    for name,expected in json.loads((WEEKLY/'case_hashes.json').read_text()).items():assert sha(WEEKLY/name)==expected
    audit=json.loads((WEEKLY/'verification.json').read_text())
    assert 'directly and indirectly' in audit['fitting'] and cached.t05.notna().all()
    # Only genuinely earlier training recordings contribute supervised examples.
    training_recs={n:r for n,r in recs.items() if r.split=='train' and r.day<'2026-09-18'}
    assert len(training_recs)==6 and len({r.day for r in training_recs.values()})==4
    cases=[]; labels=[]
    for name,rec in training_recs.items():
        for now in range(rec.first+15,int(rec.minutes[-1]),DESIGN['training_origin_step']):
            for unit in UNITS:
                if unit not in rec.units:continue
                col=rec.col(unit)
                for h in range(1,16):
                    minute=now+h; row=minute-rec.first
                    if row>=rec.n_minutes or rec.status[row,col]!=RELEASED or not np.isfinite(rec.y[row,col]):continue
                    case_id=f'{name}:{unit}:{now}:{h}'
                    cases.append(dict(case_id=case_id,day=rec.day,recording=name,unit=unit,now=now,
                                      minute=minute,horizon=h,window_id=f'{name}:{now}'))
                    labels.append(dict(case_id=case_id,truth=float(rec.y[row,col])))
    train,ntrain=extract(pd.DataFrame(cases),training_recs,timetable)
    recent,nrecent=extract(fixed[META],recs,timetable)
    requests=actual_cases(recs); actual,nactual=extract(requests,recs,timetable)
    features=[c for c in train if c not in META]
    assert set(features)==set(recent.columns)-set(META)==set(actual.columns)-set(META)
    assert not set(train.recording).intersection(recent.recording)
    assert len(train)>1000 and set(train.unit)==set(UNITS)
    assert np.isfinite(recent[features].to_numpy()).any(axis=1).all()
    for name,frame in [('training_features.csv',train),('recent_features.csv',recent),('actual_features.csv',actual)]:
        frame.to_csv(OUT/name,index=False)
    training_labels=pd.DataFrame(labels).set_index('case_id').loc[train.case_id].reset_index()
    training_labels.to_csv(OUT/'training_labels.csv',index=False)
    # Bounds and subgroup definitions are determined from training only.
    change=np.log1p(training_labels.truth.to_numpy())-train.own_level.to_numpy()
    threshold=float(np.nanquantile(np.abs(train.own_slope10),.75))
    plan=dict(design=DESIGN,features=features,training_recordings=sorted(training_recs),
              training_cases=len(train),training_windows=train.window_id.nunique(),
              movement_threshold=threshold,abrupt_threshold=float(np.quantile(np.abs(change),.90)),
              log_change_bound=float(np.quantile(np.abs(change),.995)),
              recent_fixed_cases=len(fixed),recent_eligible=len(recent),actual_requests=len(requests),actual_eligible=len(actual),
              input_invariance_checks=ntrain+nrecent+nactual,protected_sha256=protected,
              feasibility='Feasible: independent earlier-recording training, identical fixed targets with clean cached T05, no template or T05 input to new model.',
              preparation_seconds=time.perf_counter()-started)
    for filename in ('training_features.csv','training_labels.csv','recent_features.csv','actual_features.csv'):
        plan.setdefault('prepared_sha256',{})[filename]=sha(OUT/filename)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in plan.items() if k not in ('protected_sha256','prepared_sha256','features','design')},indent=2))


def predict(frame,model,plan):
    frame=frame.copy()
    delta=model.predict(frame[plan['features']])
    bound=plan['log_change_bound']
    frame['independent']=np.maximum(np.expm1(frame.own_level+np.clip(delta,-bound,bound)),0.)
    frame['recent_context']=np.where((frame.own_n10>=3)&(frame.own_span10>=5),'rich','limited')
    s=frame.own_slope10; threshold=plan['movement_threshold']
    frame['movement']=np.where(s.isna(),'insufficient_trend',np.where(s>threshold,'rapid_rise',np.where(s < -threshold,'rapid_fall','quieter')))
    assert np.isfinite(frame.independent).all() and frame.independent.ge(0).all()
    return frame


def summarize(frame,plan):
    frame=frame.copy()
    frame['horizon_band']=pd.cut(frame.horizon,[0,5,10,15],labels=['1-5','6-10','11-15']).astype(str)
    frame['baseline_error']=(frame.t05-frame.truth).abs()
    frame['independent_error']=(frame.independent-frame.truth).abs()
    # Future abruptness is a scorer-only diagnostic, NEVER a feature or gate.
    frame['actual_change']=np.where((np.log1p(frame.truth)-frame.own_level).abs()>plan['abrupt_threshold'],'abrupt','other')
    for groups,suffix in [(['day'],'date'),(['day','unit'],'camera'),(['day','horizon_band'],'horizon'),
                           (['day','window_id'],'window'),(['day','movement'],'movement'),
                           (['day','recent_context'],'context'),(['day','actual_change'],'actual_change')]:
        result=frame.groupby(groups).agg(targets=('case_id','size'),windows=('window_id','nunique'),
            timestamps=('minute','nunique'),baseline_mae=('baseline_error','mean'),independent_mae=('independent_error','mean')).reset_index()
        result['gain']=result.baseline_mae-result.independent_mae
        result['sparse']=result.targets.lt(30)|result.windows.lt(5)|result.timestamps.lt(30)
        result.to_csv(OUT/f'by_{suffix}.csv',index=False)
    rows=[]
    for day,group in frame.groupby('day'):
        for window in group.window_id.unique():
            kept=group[group.window_id.ne(window)]
            rows.append(dict(day=day,omitted_window=window,targets=len(kept),windows=kept.window_id.nunique(),
                baseline_mae=kept.baseline_error.mean(),independent_mae=kept.independent_error.mean(),
                gain=(kept.baseline_error-kept.independent_error).mean()))
    pd.DataFrame(rows).to_csv(OUT/'leave_one_window_out.csv',index=False)
    frame.to_csv(OUT/'scored.csv',index=False)


def run():
    started=time.perf_counter()
    if not (OUT/'plan.json').exists():prepare()
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['design']==DESIGN,'Prepared design changed; prepare again BEFORE evaluation'
    for filename,expected in plan['prepared_sha256'].items():assert sha(OUT/filename)==expected
    for filename,expected in plan['protected_sha256'].items():assert sha(filename)==expected
    training=pd.read_csv(OUT/'training_features.csv'); labels=pd.read_csv(OUT/'training_labels.csv')
    assert training.case_id.tolist()==labels.case_id.tolist()
    y=np.log1p(labels.truth.to_numpy())-training.own_level.to_numpy()
    weights=1/training.groupby(['recording','unit','now']).case_id.transform('size').to_numpy()
    weights*=len(weights)/weights.sum()
    with threadpool_limits(limits=4):
        model=HistGradientBoostingRegressor(**MODEL)
        model.fit(training[plan['features']],y,sample_weight=weights)
        recent=predict(pd.read_csv(OUT/'recent_features.csv'),model,plan)
        actual=predict(pd.read_csv(OUT/'actual_features.csv'),model,plan)
    cached=pd.read_csv(WEEKLY/'predictions.csv')[['case_id','t05']]
    recent=recent.merge(cached,on='case_id',validate='one_to_one')
    assert len(recent)==plan['recent_eligible']
    recent.to_csv(OUT/'recent_predictions.csv',index=False)
    actual.to_csv(OUT/'actual_prediction_diagnostic.csv',index=False)
    # Evaluation answers only enter AFTER every prediction is saved.
    scored=recent.merge(pd.read_csv(WEEKLY/'labels.csv'),on='case_id',validate='one_to_one')
    assert len(scored)==len(recent)
    summarize(scored,plan)
    days=pd.read_csv(OUT/'by_date.csv'); windows=pd.read_csv(OUT/'by_window.csv')
    loo=pd.read_csv(OUT/'leave_one_window_out.csv')
    major=pd.concat([pd.read_csv(OUT/'by_camera.csv'),pd.read_csv(OUT/'by_horizon.csv')])
    major=major[major.targets.ge(30)&major.windows.ge(5)]
    bad=major[major.independent_mae-major.baseline_mae>np.maximum(1.,major.baseline_mae*.1)]
    winning=windows.assign(win=windows.gain.gt(0)).groupby('day').win.sum()
    accepted=bool(len(days)==2 and (days.gain>=days.baseline_mae*.03).all() and loo.gain.gt(0).all()
                  and winning.ge(3).all() and bad.empty)
    coverage=pd.read_csv(WEEKLY/'cases.csv').groupby('day').size().rename('fixed_cases').to_frame()
    coverage=coverage.join(recent.groupby('day').agg(eligible=('case_id','size'),windows=('window_id','nunique')))
    coverage.to_csv(OUT/'coverage.csv')
    actual_base=actual.merge(pd.read_csv(BASE).rename(columns={'id':'case_id','people':'baseline'}),on='case_id',validate='one_to_one')
    would_change=int((actual_base.independent-actual_base.baseline).abs().gt(1e-9).sum())
    candidate=None
    if accepted:
        candidate=ROOT/'outputs/submission_t05_independent_ah_v1.csv'
        if candidate.exists():raise FileExistsError(f'Refusing overwrite: {candidate}')
        with BASE.open(newline='',encoding='utf-8-sig') as handle:original=list(csv.DictReader(handle))
        sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv').id.tolist()
        assert len(original)==4457 and [r['id'] for r in original]==sample
        updates=actual.set_index('case_id').independent.to_dict()
        intended=set(actual_cases(load_recordings(ROOT/'data/raw')).case_id)
        assert set(updates).issubset(intended)
        rows=[dict(r) for r in original]
        for row in rows:
            if row['id'] in updates:row['people']=repr(float(updates[row['id']]))
        with candidate.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(rows)
        reread=pd.read_csv(candidate)
        assert reread.id.tolist()==sample and reread.id.is_unique and np.isfinite(reread.people).all() and reread.people.ge(0).all()
        assert all(a['people']==b['people'] for a,b in zip(original,rows) if a['id'] not in intended)
    for filename,expected in plan['protected_sha256'].items():assert sha(filename)==expected
    result=dict(accepted=accepted,model_fits=1,t05_fits=0,features=len(plan['features']),
                training_cases=len(training),evaluation_cases=len(recent),actual_predictable=len(actual),
                actual_total=plan['actual_requests'],actual_would_change=would_change,
                winning_windows=winning.to_dict(),major_regressions=bad.fillna('').to_dict('records'),
                candidate=str(candidate) if candidate else None,protected_files_unchanged=True,
                input_invariance_checks=plan['input_invariance_checks'],training_evaluation_seconds=time.perf_counter()-started,
                preparation_seconds=plan['preparation_seconds'])
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(days.to_string(index=False));print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:run()
