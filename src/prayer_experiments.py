"""Paired date-held-out prayer-time tests; no target labels enter test feeds."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pipeline import ROOT, FEATURES, feature, fit_forecast, load, locate_data, predict_one, table
from v2_experiments import DIRECT_COLS, DONORS, cases, direct_feature, fit_direct
import v2_hidden

PRAYERS = ('fajr', 'dhuhr', 'asr', 'maghrib', 'isha')
EXTRA = ['friday', 'prayer_now', 'prayer_target', 'prayer_between']
_FORECAST_CACHE = {}


def timetable(path: Path, train: pd.DataFrame, test: pd.DataFrame):
    raw = pd.read_csv(path, dtype=str)
    assert list(raw) == ['day', *PRAYERS]
    assert raw.day.is_unique and raw.notna().all().all()
    assert raw.day.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all()
    assert pd.to_datetime(raw.day, format='%Y-%m-%d', errors='raise').notna().all()
    assert all(raw[p].str.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d').all() for p in PRAYERS)
    dates = set(train.day) | set(test.day)
    assert set(raw.day) == dates
    for name, df in [('train', train), ('test', test)]:
        joined = df[['day']].merge(raw, on='day', how='left', validate='many_to_one')
        assert len(joined) == len(df) and joined[list(PRAYERS)].notna().all().all()
        print(f'Prayer join {name}: {len(df)} -> {len(joined)} rows', flush=True)
    schedule = {r.day: np.array([int(getattr(r, p)[:2])*60 + int(getattr(r, p)[3:])
                                  for p in PRAYERS]) for r in raw.itertuples()}
    print('Prayer dates:', sorted(schedule), flush=True)
    return schedule


def schedule_features(day: str, now: int, h: int, schedule: dict):
    prayer = schedule[day]
    target = now + h
    # Signed value is minute minus the nearest scheduled prayer minute.
    near_now = int(now - prayer[np.argmin(np.abs(prayer-now))])
    near_target = int(target - prayer[np.argmin(np.abs(prayer-target))])
    between = int(np.any((prayer > now) & (prayer <= target)))
    return [int(pd.Timestamp(day).day_of_week == 4), near_now, near_target, between]


def extras(day, now, h, schedule, variant):
    z = schedule_features(day, now, h, schedule)
    return z[:1] if variant == 'clock' else z


def fit_direct_variant(tables, rec_day, schedule, variant):
    rows, labels = [], []
    for rec, t in tables.items():
        day = rec_day[rec]
        for unit in v2_hidden.PATTERNS:
            if unit not in t:
                continue
            y = t[unit]
            for now in range(int(t.index.min())+20, int(t.index.max())-15, 8):
                base = direct_feature(t, unit, now, 0)
                for h in range(1, 16):
                    truth = y.get(now+h, np.nan)
                    if np.isfinite(truth):
                        x = base.copy(); x[1] = h
                        rows.append(x + extras(day, now, h, schedule, variant))
                        labels.append(float(truth))
    cols = DIRECT_COLS + (EXTRA[:1] if variant == 'clock' else EXTRA)
    X = pd.DataFrame(rows, columns=cols)
    used = X.columns[X.notna().any()].tolist()
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=100,
        learning_rate=.06, max_leaf_nodes=12, min_samples_leaf=40, random_state=27)
    model.fit(X[used], labels)
    return model, used, cols


def fit_forecast_variant(tables, rec_day, schedule, variant):
    key=tuple(sorted(tables))
    if key not in _FORECAST_CACHE:
        rows, labels, extra_rows = [], [], []
        for rec, t in tables.items():
            day = rec_day[rec]
            for unit in t:
                s = t[unit].dropna()
                if len(s) < 30:
                    continue
                for now in range(int(s.index.min())+20, int(s.index.max())-15, 8):
                    for h in range(1, 16):
                        y = s.get(now+h, np.nan)
                        if np.isfinite(y):
                            f = feature(t, unit, now, h)
                            if np.isfinite(f[0]):
                                rows.append(f)
                                extra_rows.append(schedule_features(day,now,h,schedule))
                                labels.append(float(y))
        _FORECAST_CACHE[key]=(rows,labels,extra_rows)
    rows, labels, extra_rows = _FORECAST_CACHE[key]
    cols = FEATURES + (EXTRA[:1] if variant == 'clock' else EXTRA)
    n_extra=1 if variant=='clock' else 4
    X = pd.DataFrame([r+e[:n_extra] for r,e in zip(rows,extra_rows)], columns=cols)
    y = np.asarray(labels)
    target = np.log1p(y)-np.log1p(X['last'].to_numpy())
    weight = np.maximum(X['last'].to_numpy(), 1)
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=120,
        learning_rate=.06, max_leaf_nodes=15, min_samples_leaf=40, random_state=12)
    model.fit(X, target, sample_weight=weight)
    return model, cols


def predict_direct_variant(t, unit, now, h, day, schedule, fitted, variant):
    model, used, cols = fitted
    x = direct_feature(t, unit, now, h) + extras(day, now, h, schedule, variant)
    return max(0., float(model.predict(pd.DataFrame([x], columns=cols)[used])[0]))


def predict_forecast_variant(t, unit, now, h, day, schedule, fitted, variant):
    model, cols = fitted
    f = feature(t, unit, now, h)
    if not np.isfinite(f[0]):
        return np.nan
    x = f + extras(day, now, h, schedule, variant)
    residual = float(model.predict(pd.DataFrame([x], columns=cols))[0])
    return max(0., float(np.expm1(np.log1p(f[0])+residual)))


def forecast_cases(train, test):
    templates = {'Fri': test[test.recording.eq('20260918_1100')],
                 'Sat': test[test.recording.eq('20260919_1100')]}
    out = []
    for rec, g in train.groupby('recording'):
        mask = templates[g.weekday.iloc[0]].copy()
        mask = mask[mask.kind.eq('forecast')].copy()
        mask['recording'] = rec
        released = templates[g.weekday.iloc[0]].copy()
        released['recording'] = rec
        released = released[released.id.isna()][['recording','unit','minute_of_day']]
        feed = released.merge(g[['recording','unit','minute_of_day','people']],
                              on=['recording','unit','minute_of_day'], how='inner')
        targets = mask.drop(columns='people').merge(
            g[['recording','unit','minute_of_day','people']],
            on=['recording','unit','minute_of_day'], how='inner')
        if len(targets):
            out.append((g.day.iloc[0], rec, table(feed).get(rec, pd.DataFrame()), targets))
    return out


def score(detail, groups):
    x = detail.copy()
    x['error'] = (x.truth-x.prediction).abs()
    x['reference_error'] = (x.truth-x.reference).abs()
    s = x.groupby(groups, observed=True, dropna=False).agg(
        n=('error','size'), mae=('error','mean'), reference_mae=('reference_error','mean')).reset_index()
    s['normalized_mae'] = s.mae/s.reference_mae.replace(0, np.nan)
    return s


def evaluate_hf(train, test, schedule, out, rec_day):
    prior = pd.read_csv(out/'v2_selected_validation_predictions.csv')
    prior = prior[prior.kind.eq('hidden-future')].set_index(['recording','unit','minute'])
    rows = []; invariance = 0
    for day in sorted(train.day.unique()):
        fold = [c for c in cases(train,test) if c[0] == day]
        if not fold: continue
        fit_t = table(train[train.day.ne(day)])
        baseline = fit_direct(fit_t)
        alt = {v:fit_direct_variant(fit_t,rec_day,schedule,v) for v in ('clock','prayer')}
        for _, rec, t, targets in fold:
            for r in targets[targets.kind.eq('hidden-future')].itertuples():
                minute = int(r.minute_of_day); now = int(r.now_minute_of_day); h = minute-now
                key = (rec,r.unit,minute)
                if key not in prior.index: continue
                old = prior.loc[key]
                if isinstance(old,pd.DataFrame): raise AssertionError('duplicate validation target')
                # Reproduce v2 direct model and the selected 75% blend exactly.
                raw = direct_feature(t,r.unit,now,h)
                p = float(baseline[0].predict(pd.DataFrame([raw],columns=DIRECT_COLS)[baseline[1]])[0])
                assert np.isclose(p,old.direct,atol=1e-8),(key,p,old.direct)
                predictions = {'v2':float(old.candidate)}
                changed = t.copy(); changed.loc[changed.index>now] = 1e6
                for variant in ('clock','prayer'):
                    q = predict_direct_variant(t,r.unit,now,h,day,schedule,alt[variant],variant)
                    q2 = predict_direct_variant(changed,r.unit,now,h,day,schedule,alt[variant],variant)
                    assert np.isclose(q,q2,atol=1e-10)
                    invariance += 1
                    predictions[variant] = float(old.reference + .75*(q-old.reference))
                near = int(np.min(np.abs(schedule[day]-minute)) <= 15)
                for variant,pred in predictions.items():
                    rows.append(dict(task='hidden-future',fold=day,recording=rec,unit=r.unit,
                        minute=minute,horizon=h,near_prayer=near,truth=float(r.people),
                        reference=float(old.reference),variant=variant,prediction=pred))
        print('Hidden-future fold',day,'done',flush=True)
    print('Hidden-future post-cutoff invariance:',invariance,flush=True)
    return pd.DataFrame(rows)


def evaluate_forecast(train, test, schedule, out, rec_day):
    rows=[]; invariance=0
    for day in sorted(train.day.unique()):
        fold=[c for c in forecast_cases(train,test) if c[0]==day]
        if not fold: continue
        fit_t=table(train[train.day.ne(day)])
        alt={v:fit_forecast_variant(fit_t,rec_day,schedule,v) for v in ('clock','prayer')}
        base_rows,base_labels,_=_FORECAST_CACHE[tuple(sorted(fit_t))]
        base_X=pd.DataFrame(base_rows,columns=FEATURES)
        base_y=np.asarray(base_labels)
        baseline=HistGradientBoostingRegressor(loss='absolute_error',max_iter=120,
            learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=40,random_state=12)
        baseline.fit(base_X,np.log1p(base_y)-np.log1p(base_X['last'].to_numpy()),
                     sample_weight=np.maximum(base_X['last'].to_numpy(),1))
        for _,rec,t,targets in fold:
            for r in targets.itertuples():
                minute=int(r.minute_of_day); now=int(r.now_minute_of_day); h=minute-now
                f=feature(t,r.unit,now,h)
                if not np.isfinite(f[0]): continue
                ref=float(f[0]); base=float(max(0.,np.expm1(np.log1p(ref)+
                    float(baseline.predict(pd.DataFrame([f],columns=FEATURES))[0]))))
                predictions={'v2':base}
                changed=t.copy(); changed.loc[changed.index>now]=1e6
                for variant in ('clock','prayer'):
                    q=predict_forecast_variant(t,r.unit,now,h,day,schedule,alt[variant],variant)
                    q2=predict_forecast_variant(changed,r.unit,now,h,day,schedule,alt[variant],variant)
                    assert np.isclose(q,q2,atol=1e-10)
                    invariance+=1; predictions[variant]=q
                near=int(np.min(np.abs(schedule[day]-minute))<=15)
                for variant,pred in predictions.items():
                    rows.append(dict(task='forecast',fold=day,recording=rec,unit=r.unit,
                        minute=minute,horizon=h,near_prayer=near,truth=float(r.people),
                        reference=ref,variant=variant,prediction=pred))
        print('Forecast fold',day,'done',flush=True)
    print('Forecast post-cutoff invariance:',invariance,flush=True)
    return pd.DataFrame(rows)


def save(detail,out,task):
    assert detail.groupby(['recording','unit','minute']).variant.nunique().eq(3).all()
    detail.to_csv(out/f'v3_{task}_predictions.csv',index=False)
    detail['horizon_band']=pd.cut(detail.horizon,[0,5,10,15],labels=['1-5','6-10','11-15'])
    for groups,name in [(['variant'],'overall'),(['fold','recording','variant'],'by_date'),
                        (['horizon_band','variant'],'by_horizon'),
                        (['near_prayer','variant'],'near_prayer'),
                        (['fold','near_prayer','variant'],'date_near_prayer')]:
        score(detail,groups).to_csv(out/f'v3_{task}_{name}.csv',index=False)
    print(task,score(detail,['fold','recording','variant']).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--task',choices=['hidden-future','forecast','all'],default='all')
    args=parser.parse_args()
    train,test,_,recordings=load(locate_data(None)); out=ROOT/'outputs'
    schedule=timetable(ROOT/'data/raw/prayer_times_madinah.csv',train,test)
    rec_day=recordings.set_index('recording').day.to_dict()
    if args.task in ('hidden-future','all'):
        save(evaluate_hf(train,test,schedule,out,rec_day),out,'hidden_future')
    if args.task in ('forecast','all'):
        save(evaluate_forecast(train,test,schedule,out,rec_day),out,'forecast')


if __name__=='__main__': main()
