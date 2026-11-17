"""One prespecified ridge change model; isolated, cached T05 is never refitted."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pipeline import ROOT
from t05_forecast.core import ABSENT, FORECAST, RELEASED, load_recordings

OUT = ROOT / 'outputs/short_term_change_trial'
WEEKLY = ROOT / 'outputs/weekly_forecast_diagnostic'
BASE = ROOT / 'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
UNITS = [f'cam_{c}' for c in 'ABCDEFGH']
SPEC = dict(family='Ridge', mean_loss_l2=.1, blend_weight=.25,
            max_reading_age=5, context_minutes=10, min_context_points=3,
            min_context_span=5, training_origin_step=30,
            model_change_rate_bound_quantile=.99, blend_innovation_bound_quantile=.95,
            rapid_threshold_quantile=.75, features='horizon-scaled level, 3/10-minute slopes, last pair change, age, last interval, short-context availability, time sin/cos, Friday, camera intercepts',
            decision='Blend must gain >=1% on both recent dates; every leave-one-window-out gain positive; >=3 winning windows/date; camera/horizon groups with >=30 cases and >=5 origins cannot regress by more than max(1 person,10% baseline MAE).',
            design='Prespecified without inspecting September 18/19 labels; no tuning or T05 fitting.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context(rec, unit, now):
    """Read a prefix only: future targets cannot enter feature computation."""
    col = rec.col(unit)
    end = max(0, min(rec.n_minutes, int(now)-rec.first+1))
    y = rec.y[:end, col]
    t = rec.minutes[:end]
    valid = np.isfinite(y) & (rec.status[:end, col] == RELEASED)
    t, y = t[valid], y[valid]
    if len(t) == 0 or now-t[-1] > SPEC['max_reading_age']:
        return None
    recent = t >= now-SPEC['context_minutes']
    rt, ry = t[recent], y[recent]
    if len(rt) < SPEC['min_context_points'] or np.ptp(rt) < SPEC['min_context_span']:
        return None
    def slope(tt, yy):
        x = tt-tt.mean()
        return float(np.dot(x, yy)/np.dot(x, x))
    short = rt >= now-3
    short_ok = int(short.sum() >= 2)
    s3 = slope(rt[short], ry[short]) if short_ok else 0.
    s10 = slope(rt, ry)
    dt = int(t[-1]-t[-2])
    return dict(last=float(y[-1]), s3=s3, s10=s10,
                pair=float((y[-1]-y[-2])/dt), age=int(now-t[-1]),
                interval=dt, short_ok=short_ok)


def extract(cases, recs, check=True):
    rows, checks = [], 0
    for (name, unit, now), targets in cases.groupby(['recording', 'unit', 'now'], sort=False):
        rec = recs[name]
        ctx = context(rec, unit, int(now))
        if check:
            altered = rec.copy(); altered.y[altered.minutes > now] = 1e9
            assert ctx == context(altered, unit, int(now))
            checks += len(targets)
        if ctx is None:
            continue
        for r in targets.itertuples(index=False):
            rows.append(dict(**r._asdict(), **ctx, friday=int(rec.weekday == 'Fri')))
    return pd.DataFrame(rows), checks


def training_data(recs, excluded_day=None):
    eligible = {n: r for n, r in recs.items() if r.split == 'train' and r.day != excluded_day}
    cases, labels = [], []
    for name, rec in eligible.items():
        for now in range(rec.first+15, int(rec.minutes[-1]), SPEC['training_origin_step']):
            for unit in UNITS:
                if unit not in rec.units:
                    continue
                col = rec.col(unit)
                for horizon in range(1, 16):
                    minute = now+horizon; row = minute-rec.first
                    if row >= rec.n_minutes or rec.status[row, col] != RELEASED or not np.isfinite(rec.y[row, col]):
                        continue
                    case_id = f'{name}:{unit}:{now}:{horizon}'
                    cases.append(dict(case_id=case_id, day=rec.day, recording=name, unit=unit,
                                      now=now, minute=minute, horizon=horizon,
                                      window_id=f'{name}:{now}'))
                    labels.append(dict(case_id=case_id, truth=float(rec.y[row, col])))
    features, checks = extract(pd.DataFrame(cases), eligible)
    # Training answers are joined only after all training inputs are built.
    frame = features.merge(pd.DataFrame(labels), on='case_id', validate='one_to_one')
    assert excluded_day is None or not frame.day.eq(excluded_day).any()
    return frame, checks


def matrix(frame, scales):
    scale = frame.unit.map(scales).to_numpy(float)
    h = frame.horizon.to_numpy(float)
    f = h/15.
    angle = frame.now.to_numpy(float)*2*np.pi/1440.
    return np.column_stack([
        f, frame['last']/scale*f,
        frame.s3/scale*h, frame.s10/scale*h, frame.pair/scale*h,
        frame.age/5*f, np.minimum(frame.interval, 10)/10*f,
        frame.short_ok*f, np.sin(angle)*f, np.cos(angle)*f, frame.friday*f,
        *[(frame.unit.eq(unit)*f).to_numpy(float) for unit in UNITS[1:]],
    ])


def fit(frame):
    # Camera scales and both bounds are calculated exclusively from training.
    scales = frame.groupby('unit')['last'].median().clip(lower=5).to_dict()
    scale = frame.unit.map(scales).to_numpy(float)
    delta = (frame.truth-frame['last']).to_numpy()/scale
    rates = np.abs(delta/frame.horizon.to_numpy())
    # Equal total weight per camera/origin, regardless of available horizons.
    weight = 1/frame.groupby(['recording', 'unit', 'now']).case_id.transform('size').to_numpy()
    weight *= len(weight)/weight.sum()
    scaler = StandardScaler(with_mean=False)
    x = scaler.fit_transform(matrix(frame, scales), sample_weight=weight)
    model = Ridge(alpha=SPEC['mean_loss_l2']*weight.sum(), fit_intercept=False)
    model.fit(x, delta, sample_weight=weight)
    return dict(model=model, scaler=scaler, scales=scales,
                rate_bound=float(np.quantile(rates, SPEC['model_change_rate_bound_quantile'])),
                blend_bound=float(np.quantile(rates, SPEC['blend_innovation_bound_quantile'])),
                rapid_threshold=float(np.quantile(np.abs(frame.s10.to_numpy()/scale), SPEC['rapid_threshold_quantile'])))


def predict(frame, fitted):
    frame = frame.copy()
    scale = frame.unit.map(fitted['scales']).to_numpy(float)
    h = frame.horizon.to_numpy(float)
    delta = fitted['model'].predict(fitted['scaler'].transform(matrix(frame, fitted['scales'])))
    bound = fitted['rate_bound']*h
    frame['change_prediction'] = np.maximum(frame['last']+np.clip(delta, -bound, bound)*scale, 0.)
    cap = fitted['blend_bound']*h*scale
    frame['blend_prediction'] = np.maximum(frame.t05+SPEC['blend_weight']*np.clip(
        frame.change_prediction-frame.t05, -cap, cap), 0.)
    trend = frame.s10.to_numpy()/scale
    threshold = fitted['rapid_threshold']
    frame['movement'] = np.where(trend > threshold, 'rapid_rise',
                                 np.where(trend < -threshold, 'rapid_fall', 'quieter'))
    assert np.isfinite(frame[['t05', 'change_prediction', 'blend_prediction']]).all().all()
    return frame


def summarize(scored, prefix):
    frame = scored.copy()
    frame['horizon_band'] = pd.cut(frame.horizon, [0,5,10,15], labels=['1-5','6-10','11-15']).astype(str)
    for method in ('t05', 'change_prediction', 'blend_prediction'):
        frame[f'{method}_error'] = (frame[method]-frame.truth).abs()
    for groups, suffix in [(['day'], 'date'), (['day','unit'], 'camera'),
                           (['day','horizon_band'], 'horizon'), (['day','window_id'], 'window'),
                           (['day','movement'], 'movement')]:
        result = frame.groupby(groups).agg(targets=('case_id','size'), windows=('window_id','nunique'),
            timestamps=('minute','nunique'), t05_mae=('t05_error','mean'),
            change_mae=('change_prediction_error','mean'), blend_mae=('blend_prediction_error','mean')).reset_index()
        result['blend_gain'] = result.t05_mae-result.blend_mae
        result['sparse'] = result.targets.lt(30) | result.windows.lt(5) | result.timestamps.lt(30)
        result.to_csv(OUT / f'{prefix}_by_{suffix}.csv', index=False)
    leave_out = []
    for day, group in frame.groupby('day'):
        for window in group.window_id.unique():
            kept = group[group.window_id.ne(window)]
            leave_out.append(dict(day=day, omitted_window=window, targets=len(kept), windows=kept.window_id.nunique(),
                t05_mae=kept.t05_error.mean(), change_mae=kept.change_prediction_error.mean(),
                blend_mae=kept.blend_prediction_error.mean(),
                blend_gain=(kept.t05_error-kept.blend_prediction_error).mean()))
    pd.DataFrame(leave_out).to_csv(OUT / f'{prefix}_leave_one_window_out.csv', index=False)
    frame.to_csv(OUT / f'{prefix}_scored.csv', index=False)


def actual_cases(recs):
    rows = []
    for name, rec in recs.items():
        if rec.split != 'test':
            continue
        for row, col in zip(*np.nonzero(rec.status == FORECAST)):
            if rec.units[col] not in UNITS:
                continue
            minute, now = int(rec.minutes[row]), int(rec.now[row,col])
            rows.append(dict(case_id=rec.ids[(int(row),int(col))], day=rec.day, recording=name,
                             unit=rec.units[col], now=now, minute=minute, horizon=minute-now,
                             window_id=f'{name}:{now}'))
    return pd.DataFrame(rows)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert BASE.exists()
    protected_paths = list((ROOT/'outputs').glob('submission*.csv'))
    protected_paths += [WEEKLY/n for n in ('cases.csv','labels.csv','predictions.csv','benchmark_predictions.csv')]
    protected_paths += list((ROOT/'src').glob('t05*.py'))+list((ROOT/'t05_forecast').glob('*.py'))
    protected_paths += [ROOT/'outputs/local_score'/n for n in ('cases.csv','labels.csv','released_feed.csv')]
    protected = {str(p): sha(p) for p in protected_paths}
    for directory in (WEEKLY, ROOT/'outputs/local_score'):
        for name, expected in json.loads((directory/'case_hashes.json').read_text()).items():
            assert sha(directory/name) == expected
    verification = json.loads((WEEKLY/'verification.json').read_text())
    assert 'directly and indirectly' in verification['fitting']
    # Save the design before any evaluation labels are opened.
    (OUT/'design.json').write_text(json.dumps(SPEC, indent=2), encoding='utf-8')
    recs = load_recordings(ROOT/'data/raw')
    training, checks = training_data(recs)
    with threadpool_limits(limits=4):
        fitted = fit(training)
        training.to_csv(OUT/'training_cases_and_labels.csv', index=False)
        plan = dict(recordings=sorted(training.recording.unique()), days=sorted(training.day.unique()),
                    cases=len(training), windows=training.window_id.nunique(), scales=fitted['scales'],
                    rate_bound=fitted['rate_bound'], blend_bound=fitted['blend_bound'],
                    rapid_threshold=fitted['rapid_threshold'], ridge_alpha=fitted['model'].alpha,
                    features=matrix(training, fitted['scales']).shape[1])
        (OUT/'training_manifest.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
        np.savez(OUT/'ridge_parameters.npz', coef=fitted['model'].coef_, feature_scale=fitted['scaler'].scale_)
        recent_cases = pd.read_csv(WEEKLY/'cases.csv')
        features, ncheck = extract(recent_cases, recs); checks += ncheck
        cached = pd.read_csv(WEEKLY/'predictions.csv')[['case_id','t05']]
        inputs = features.merge(cached, on='case_id', validate='one_to_one')
        inputs = inputs[inputs.t05.notna()].copy()
        recent = predict(inputs, fitted)
        recent.to_csv(OUT/'recent_predictions.csv', index=False)
        assert not set(recent.recording).intersection(training.recording)
        # Actual application coverage and predictions also precede scoring.
        requested = actual_cases(recs)
        actual_features, ncheck = extract(requested, recs); checks += ncheck
        base = pd.read_csv(BASE).rename(columns={'id':'case_id','people':'t05'})
        actual = predict(actual_features.merge(base, on='case_id', validate='one_to_one'), fitted)
        actual.to_csv(OUT/'actual_prediction_diagnostic.csv', index=False)
        # Older stress uses the saved, masked feed, never raw training answers.
        feed = pd.read_csv(ROOT/'outputs/local_score/released_feed.csv')
        older_cache = pd.read_csv(WEEKLY/'benchmark_predictions.csv')
        old_recs = {}
        for name in older_cache.recording.unique():
            rec = recs[name].copy(); rec.y[:]=np.nan; rec.status[:]=ABSENT; rec.truth=None
            for r in feed[feed.recording.eq(name)].itertuples():
                row, col = int(r.minute_of_day)-rec.first, rec.col(r.unit)
                rec.y[row,col]=r.people; rec.status[row,col]=RELEASED
            old_recs[name] = rec
        old_predictions = []
        for day, targets in older_cache.groupby('day'):
            old_training, ncheck = training_data(recs, excluded_day=day); checks += ncheck
            old_model = fit(old_training)
            old_inputs, ncheck = extract(targets.drop(columns=[c for c in targets if c not in recent_cases.columns and c!='t05']), old_recs)
            checks += ncheck
            old_predictions.append(predict(old_inputs, old_model))
        older = pd.concat(old_predictions, ignore_index=True)
        older.to_csv(OUT/'older_predictions.csv', index=False)
    # All evaluation predictions are saved before any evaluation label merge.
    recent_scored = recent.merge(pd.read_csv(WEEKLY/'labels.csv'), on='case_id', validate='one_to_one')
    older_scored = older.merge(pd.read_csv(ROOT/'outputs/local_score/labels.csv'), on='case_id', validate='one_to_one')
    summarize(recent_scored, 'recent'); summarize(older_scored, 'older')
    days = pd.read_csv(OUT/'recent_by_date.csv')
    windows = pd.read_csv(OUT/'recent_by_window.csv')
    loo = pd.read_csv(OUT/'recent_leave_one_window_out.csv')
    major = pd.concat([pd.read_csv(OUT/'recent_by_camera.csv'), pd.read_csv(OUT/'recent_by_horizon.csv')])
    major = major[major.targets.ge(30) & major.windows.ge(5)]
    major_bad = major[major.blend_mae-major.t05_mae > np.maximum(1., major.t05_mae*.1)]
    winning = windows.assign(win=windows.blend_gain.gt(0)).groupby('day').win.sum()
    accepted = bool(len(days)==2 and (days.blend_gain >= days.t05_mae*.01).all()
                    and loo.blend_gain.gt(0).all() and winning.ge(3).all() and major_bad.empty)
    actual_changed = int((actual.blend_prediction-actual.t05).abs().gt(1e-9).sum())
    candidate_path = None
    if accepted:
        candidate_path = ROOT/'outputs/submission_t05_short_term_change_v1.csv'
        if candidate_path.exists():
            raise FileExistsError(f'Refusing to overwrite existing submission: {candidate_path}')
        with BASE.open(newline='', encoding='utf-8-sig') as handle:
            original_rows = list(csv.DictReader(handle))
        changes = actual.set_index('case_id').blend_prediction.to_dict()
        sample_ids = pd.read_csv(ROOT/'data/raw/sample_submission.csv').id.tolist()
        assert len(original_rows)==4457 and [r['id'] for r in original_rows]==sample_ids
        rows = [dict(r) for r in original_rows]
        for r in rows:
            if r['id'] in changes:
                r['people'] = repr(float(changes[r['id']]))
        with candidate_path.open('w', newline='', encoding='utf-8') as handle:
            writer=csv.DictWriter(handle, fieldnames=['id','people']); writer.writeheader(); writer.writerows(rows)
        reread=pd.read_csv(candidate_path)
        assert reread.id.tolist()==sample_ids and reread.id.is_unique
        assert np.isfinite(reread.people).all() and reread.people.ge(0).all()
        assert all(old['people']==new['people'] for old,new in zip(original_rows,rows) if old['id'] not in changes)
    assert all(sha(p)==expected for p,expected in protected.items())
    coverage = recent_cases.groupby('day').size().rename('fixed_cases').to_frame().join(
        recent.groupby('day').agg(eligible_cases=('case_id','size'), eligible_windows=('window_id','nunique')))
    coverage.to_csv(OUT/'coverage.csv')
    report = dict(accepted=accepted, t05_fits=0, ridge_fits=3, input_invariance_checks=checks,
                  actual_AH_requests=len(requested), actual_eligible=len(actual),
                  hypothetical_changed_rows=actual_changed, submission_created=str(candidate_path) if candidate_path else None,
                  protected_files_unchanged=True, clean_T05_cache='weekly diagnostic directly/indirectly excluded fits',
                  winning_windows=winning.to_dict(),
                  major_regressions=[{k:v for k,v in row.items()
                                      if not (isinstance(v, float) and not np.isfinite(v))}
                                     for row in major_bad.to_dict('records')],
                  elapsed_seconds=time.perf_counter()-started, protected_sha256=protected)
    (OUT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(days.to_string(index=False)); print('Older stress:'); print(pd.read_csv(OUT/'older_by_date.csv').to_string(index=False))
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'}, indent=2))


if __name__=='__main__':
    main()
