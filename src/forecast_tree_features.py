"""P49 shared features; no production defaults changed and no estimator fitting.

Weekly profiles use earlier TRAIN dates only. Completion coefficients may fit
other released dates, but never the feature row's date or its outer held-out
date. Prediction inputs are a same-recording prefix, not future donor counts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from independent_forecast_trial import context_features, prayer_minutes
from t05_forecast.template import build_template, complete_template
from t05_forecast.core import RELEASED, ABSENT

UNITS=[f'cam_{c}' for c in 'ABCDEFGH']

def weekly(rec, sources, excluded_dates):
    allowed=[r for r in sources.values() if r.day not in set(excluded_dates)|{rec.day}]
    past=[r for r in allowed if r.split=='train' and r.day<rec.day and r.weekday==rec.weekday]
    long=[r for r in past if r.n_minutes>=600]
    target=rec.copy();target.y[:]=np.nan;target.truth=None
    base=build_template(target,long or past,lam=.8)
    # Coefficients, not target-time visible donor observations, complete the curve.
    coeff=[r for r in allowed if r.split=='train' and r.n_minutes>=600]
    tmpl,report=complete_template(target,base,coeff,lam=.8)
    assert not ({r.day for r in past+coeff} & (set(excluded_dates)|{rec.day}))
    return tmpl,dict(recording=rec.name,excluded_dates=sorted(set(excluded_dates)|{rec.day}),
        profile_sources=[r.name for r in long or past],coefficient_sources=[r.name for r in coeff],
        finite_cells=int(np.isfinite(tmpl).sum()),completion=report)

def masked_training_record(rec,test):
    """Metadata-only transplant of released-test availability, at actual clock times.

    Synthetic training answers remain legal supervised targets; they are absent
    from each input prefix. Outer assessment-date answers never enter this pool.
    """
    out=rec.copy();out.truth=None
    pattern=test[test.day.eq('2026-09-18' if rec.weekday=='Fri' else '2026-09-19')]
    hidden=['cam_B','cam_G'] if rec.weekday=='Fri' else ['cam_D']
    for u in hidden:
        if u in out.units:
            out.y[:,out.col(u)]=np.nan;out.status[:,out.col(u)]=ABSENT
    for r in pattern[pattern.id.notna()].itertuples():
        i=int(r.minute_of_day)-out.first
        if 0<=i<out.n_minutes and r.unit in out.units:
            out.y[i,out.col(r.unit)]=np.nan;out.status[i,out.col(r.unit)]=ABSENT
    return out

def training_cases(recs):
    rows=[]
    for rec in recs.values():
        hidden=['cam_B','cam_G'] if rec.weekday=='Fri' else ['cam_D']
        for now in range(rec.first+15,int(rec.minutes[-1]),30):
            for u in UNITS:
                if u not in rec.units or u in hidden:continue
                for h in range(1,16):
                    i=now+h-rec.first
                    if i>=rec.n_minutes or not np.isfinite(rec.y[i,rec.col(u)]):continue
                    rows.append(dict(case_id=f'{rec.name}:{u}:{now}:{h}',recording=rec.name,day=rec.day,
                        unit=u,now=now,minute=now+h,horizon=h,truth=float(rec.y[i,rec.col(u)])))
    return pd.DataFrame(rows)

def feature_rows(cases,recordings,templates,prayers=None,*,check=False):
    prayers=prayers or prayer_minutes();rows=[];checks=0
    for (name,u,now),g in cases.groupby(['recording','unit','now'],sort=False):
        rec=recordings[name];now=int(now);p=prayers[rec.day];tmpl=templates[name]
        own=context_features(rec,u,now,p)
        if check:
            poisoned=rec.copy();poisoned.y[poisoned.minutes>now]=1e8
            poisoned.y[(poisoned.status!=RELEASED)&(poisoned.minutes[:,None]<=now)]=1e9
            q=context_features(poisoned,u,now,p)
            assert (own is None)==(q is None)
            if own is not None:np.testing.assert_allclose(list(own.values()),list(q.values()),rtol=0,atol=0,equal_nan=True)
            checks+=1
        if own is None:continue
        col=rec.col(u);last_minute=now-int(own['own_age']);last_log=own['own_level']
        own=own.copy()
        own['template_now_log']=float(tmpl[now-rec.first,col])
        own['template_last_log']=float(tmpl[last_minute-rec.first,col])
        own['last_departure_log']=last_log-own['template_last_log']
        for c in UNITS:own['is_'+c]=float(c==u)
        for width in (5,15,30):
            ix=(rec.minutes>=now-width+1)&(rec.minutes<=now)&(rec.status[:,col]==RELEASED)
            yy=rec.y[ix,col];tt=tmpl[ix,col]
            good=np.isfinite(yy)&np.isfinite(tt)
            own[f'departure_mean{width}']=float(np.mean(np.log1p(yy[good])-tt[good])) if good.any() else np.nan
            own[f'observed_n{width}']=int(np.isfinite(yy).sum())
            own[f'missing_fraction{width}']=1-own[f'observed_n{width}']/min(width,now-rec.first+1)
        for c in UNITS:
            value=float(tmpl[now-rec.first,rec.col(c)]) if c in rec.units else np.nan
            own[c+'_template_now_log']=value
            own[c+'_departure_now_log']=own[c+'_level']-value
        for r in g.itertuples():
            t=int(r.minute);z=float(tmpl[t-rec.first,col])
            anchor=float(np.expm1(z)) if np.isfinite(z) else float(np.expm1(last_log))
            d=dict(case_id=r.case_id,recording=name,day=rec.day,unit=u,now=now,minute=t,horizon=int(r.horizon),
                **own,horizon_feature=int(r.horizon),target_clock_sin=float(np.sin(t*2*np.pi/1440)),
                target_clock_cos=float(np.cos(t*2*np.pi/1440)),template_target_log=z,
                template_change_log=z-own['template_now_log'],template_available=int(np.isfinite(z)),
                nearest_prayer_target=int(t-p[np.argmin(abs(p-t))]),dhuhr_target=int(t-p[1]),
                prayer_between=int(((p>now)&(p<=t)).any()),anchor=anchor)
            rows.append(d)
        if check:
            # Exercise ALL added feature readers, not only the reused context helper.
            other,_=feature_rows(g,{name:poisoned},templates,prayers,check=False)
            actual=pd.DataFrame(rows[-len(g):])
            columns=feature_columns(actual)
            np.testing.assert_allclose(actual[columns].to_numpy(),other[columns].to_numpy(),
                                       rtol=0,atol=0,equal_nan=True)
    return pd.DataFrame(rows),checks

META=['case_id','recording','day','unit','now','minute','horizon']
def feature_columns(frame):return [c for c in frame if c not in META]
