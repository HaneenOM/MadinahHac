"""Fixed, date-held-out local score for the current four-task candidate."""
from __future__ import annotations

import hashlib
import json
import sys
import traceback

import numpy as np
import pandas as pd

from pipeline import ROOT, FEATURES, feature, fit_forecast, load, locate_data, pseudo, table
from prayer_experiments import (fit_direct_variant, fit_forecast_variant,
                                predict_direct_variant, predict_forecast_variant,
                                timetable)
from v2_candidate import HF_WEIGHT, HIDDEN, MISSING_WEIGHT
from v2_experiments import fit_direct, predict_hf
from v3_prayer_candidate import PRAYER_FORECAST, PRAYER_HF_WEEKDAYS
import v2_hidden
import v2_missing

OUT = ROOT / 'outputs' / 'local_score'
DATES = ('2026-09-11', '2026-09-12')
KINDS = ('forecast', 'hidden', 'hidden-future', 'missing')


def case_hashes(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def initialize_cases(train, test):
    """Write immutable masks, released observations, and separate labels once."""
    paths = [OUT / x for x in ('cases.csv', 'released_feed.csv', 'labels.csv')]
    if all(p.exists() for p in paths):
        fingerprints = OUT / 'case_hashes.json'
        actual = case_hashes(paths)
        if fingerprints.exists():
            assert json.loads(fingerprints.read_text(encoding='utf-8')) == actual, \
                'Fixed case, feed, or label file changed'
        else:
            fingerprints.write_text(json.dumps(actual, indent=2), encoding='utf-8')
        return
    if any(p.exists() for p in paths):
        raise RuntimeError('Fixed case files are incomplete; restore all three together')
    case_rows, feeds, labels = [], [], []
    for rec, feed, targets in pseudo(train, test, None):
        day = train.loc[train.recording.eq(rec), 'day'].iloc[0]
        assert day in DATES
        feed = feed[['recording', 'unit', 'minute_of_day', 'people']].copy()
        feed.insert(0, 'day', day)
        feeds.append(feed)
        for r in targets.itertuples():
            minute = int(r.minute_of_day)
            now = int(r.now_minute_of_day) if pd.notna(r.now_minute_of_day) else None
            horizon = minute-now if now is not None else None
            if r.kind in ('forecast', 'hidden-future'):
                assert 1 <= horizon <= 15
            case_id = f'{rec}_{r.unit}_{minute}'
            case_rows.append(dict(case_id=case_id, day=day, recording=rec,
                                  kind=r.kind, unit=r.unit, minute=minute,
                                  now=now, horizon=horizon,
                                  template_target_id=r.id))
            labels.append(dict(case_id=case_id, truth=float(r.people)))
    cases = pd.DataFrame(case_rows).sort_values(['day', 'kind', 'unit', 'minute'])
    feed = pd.concat(feeds, ignore_index=True).sort_values(
        ['day', 'recording', 'unit', 'minute_of_day'])
    truth = pd.DataFrame(labels).sort_values('case_id')
    assert cases.case_id.is_unique and truth.case_id.is_unique
    assert not feed.duplicated(['recording', 'unit', 'minute_of_day']).any()
    assert set(cases.day) == set(DATES) and set(cases.kind) == set(KINDS)
    assert cases.groupby('kind').size().to_dict() == {
        'forecast': 2036, 'hidden': 843, 'hidden-future': 482, 'missing': 358}
    fixed_hf = pd.read_csv(ROOT / 'outputs' / 'hidden_future_benchmark' / 'cases.csv')
    current_hf = cases[cases.kind.eq('hidden-future')]
    assert set(current_hf.case_id) == set(fixed_hf.case_id)
    assert current_hf[['case_id', 'now', 'minute', 'horizon']].merge(
        fixed_hf[['case_id', 'now', 'minute', 'horizon']], on='case_id',
        validate='one_to_one', suffixes=('', '_fixed')).pipe(
            lambda x: len(x) == 482 and all(np.array_equal(x[c].to_numpy(),
                                                          x[f'{c}_fixed'].to_numpy())
                                         for c in ('now', 'minute', 'horizon')))
    # Never store target answers in the case or released-feed files.
    OUT.mkdir(parents=True, exist_ok=True)
    cases.to_csv(paths[0], index=False)
    feed.to_csv(paths[1], index=False)
    truth.to_csv(paths[2], index=False)
    (OUT / 'case_hashes.json').write_text(json.dumps(case_hashes(paths), indent=2),
                                          encoding='utf-8')


def load_cases():
    cases = pd.read_csv(OUT / 'cases.csv')
    feed = pd.read_csv(OUT / 'released_feed.csv')
    assert cases.case_id.is_unique
    assert cases.groupby(['day', 'kind']).size().to_dict() == {
        ('2026-09-11', 'forecast'): 936,
        ('2026-09-11', 'hidden'): 560,
        ('2026-09-11', 'hidden-future'): 322,
        ('2026-09-11', 'missing'): 163,
        ('2026-09-12', 'forecast'): 1100,
        ('2026-09-12', 'hidden'): 283,
        ('2026-09-12', 'hidden-future'): 160,
        ('2026-09-12', 'missing'): 195}
    assert not feed.duplicated(['recording', 'unit', 'minute_of_day']).any()
    return cases, table(feed)


def fit_fold(train, recordings, day, schedule):
    fit = train[train.day.ne(day)]
    assert day not in set(fit.day)
    fit_tables = table(fit)
    rec_day = recordings.set_index('recording').day.to_dict()
    return dict(
        hidden=v2_hidden.fit(fit_tables),
        direct=fit_direct(fit_tables),
        direct_prayer=fit_direct_variant(fit_tables, rec_day, schedule, 'prayer')
            if (('Sat' if day == DATES[1] else 'Fri') in PRAYER_HF_WEEKDAYS) else None,
        forecast_prayer=fit_forecast_variant(fit_tables, rec_day, schedule, 'prayer')
            if PRAYER_FORECAST else fit_forecast(fit_tables, 'log')[0],
        missing=v2_missing.fit_calibration(fit_tables))


def predict_case(r, t, models, schedule):
    unit = r.unit
    minute = int(r.minute)
    if r.kind == 'forecast':
        now = int(r.now); horizon = minute-now
        reference = feature(t, unit, now, horizon)[0]  # persistence
        if PRAYER_FORECAST:
            prediction = predict_forecast_variant(t, unit, now, horizon, r.day,
                                                  schedule, models['forecast_prayer'], 'prayer')
        else:
            f = feature(t, unit, now, horizon)
            residual = float(models['forecast_prayer'].predict(pd.DataFrame([f], columns=FEATURES))[0])
            prediction = max(0., float(np.expm1(np.log1p(reference)+residual)))
    elif r.kind == 'hidden':
        reference = v2_hidden.predict(t, unit, minute, models['hidden'], 'single')[0]
        method, weight = HIDDEN[unit]
        fitted = v2_hidden.predict(t, unit, minute, models['hidden'], method)[0]
        prediction = reference + weight * (fitted-reference)
    elif r.kind == 'hidden-future':
        now = int(r.now); horizon = minute-now
        reference = predict_hf(t, unit, now, horizon, models['hidden'],
                               models['direct'], 'fixed')
        weekday = 'Sat' if r.day == DATES[1] else 'Fri'
        if weekday in PRAYER_HF_WEEKDAYS:
            fitted = predict_direct_variant(t, unit, now, horizon, r.day,
                                            schedule, models['direct_prayer'], 'prayer')
        else:
            fitted = predict_hf(t, unit, now, horizon, models['hidden'],
                                models['direct'], 'direct')
        prediction = reference + HF_WEIGHT * (fitted-reference)
    else:
        reference = v2_missing.predict(t, unit, minute, models['missing'], 'linear')[0]
        prediction = reference
        weight = MISSING_WEIGHT.get(unit, 0.)
        if weight:
            method = 'blend_full' if weight == 1. else 'blend'
            fitted, available, gap = v2_missing.predict(
                t, unit, minute, models['missing'], method, weight)
            if np.isfinite(fitted) and gap >= 6 and available > 0:
                prediction = fitted
    if not (np.isfinite(reference) and np.isfinite(prediction) and prediction >= 0):
        raise RuntimeError(f'{r.case_id}: nonfinite reference or current prediction')
    return float(reference), float(prediction)


def make_predictions(train, recordings, cases, feeds, schedule):
    rows = []
    checks = 0
    for day in DATES:
        models = fit_fold(train, recordings, day, schedule)
        today = cases[cases.day.eq(day)]
        for r in today.itertuples():
            t = feeds[r.recording]
            reference, prediction = predict_case(r, t, models, schedule)
            if r.kind in ('forecast', 'hidden-future'):
                # Forecasting paths must be invariant to all later camera data.
                altered = t.copy()
                altered.loc[altered.index > int(r.now)] = 1e6
                ref2, pred2 = predict_case(r, altered, models, schedule)
                assert np.isclose(reference, ref2, rtol=0, atol=1e-10)
                assert np.isclose(prediction, pred2, rtol=0, atol=1e-10)
                checks += 1
            rows.append(dict(case_id=r.case_id, day=day, kind=r.kind,
                             reference=reference, prediction=prediction))
        print(f'Fresh predictions for {day}: {len(today)}', flush=True)
    detail = pd.DataFrame(rows)
    assert len(detail) == len(cases) and detail.case_id.is_unique
    detail.to_csv(OUT / 'predictions.csv', index=False)
    return detail, checks


def score(detail, checks):
    # Target labels enter only after all predictions have been saved.
    labels = pd.read_csv(OUT / 'labels.csv')
    scored = detail.merge(labels, on='case_id', how='inner', validate='one_to_one')
    assert len(scored) == len(detail) == len(labels)
    scored['absolute_error'] = (scored.truth-scored.prediction).abs()
    scored['reference_absolute_error'] = (scored.truth-scored.reference).abs()
    scored.to_csv(OUT / 'scored_predictions.csv', index=False)
    per_task = scored.groupby(['day', 'kind']).agg(
        predictions=('absolute_error', 'size'),
        mae_people=('absolute_error', 'mean'),
        reference_mae_people=('reference_absolute_error', 'mean')).reset_index()
    assert per_task.groupby('day').kind.nunique().eq(4).all()
    assert (per_task.reference_mae_people > 0).all()
    per_task['normalized_mae'] = per_task.mae_people / per_task.reference_mae_people
    per_task.to_csv(OUT / 'scores_by_date_task.csv', index=False)
    date_score = per_task.groupby('day').normalized_mae.mean().reset_index(name='date_score')
    date_score.to_csv(OUT / 'scores_by_date.csv', index=False)
    overall = float(date_score.date_score.mean())
    report = dict(status='complete', local_score=overall, date_scores=date_score.to_dict('records'),
                  total_predictions=len(scored), post_cutoff_checks=checks)
    (OUT / 'score.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(per_task.to_string(index=False))
    print(date_score.to_string(index=False))
    print(f'OVERALL LOCAL SCORE: {overall:.6f} (complete; {checks} post-cutoff checks)')
    return overall


def main():
    train, test, _, recordings = load(locate_data(None))
    initialize_cases(train, test)
    cases, feeds = load_cases()
    schedule = timetable(ROOT/'data/raw/prayer_times_madinah.csv', train, test)
    detail, checks = make_predictions(train, recordings, cases, feeds, schedule)
    score(detail, checks)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        blocker = f'{type(exc).__name__}: {exc}'
        (OUT / 'score.json').write_text(json.dumps(dict(status='incomplete', blocker=blocker),
                                                 indent=2), encoding='utf-8')
        print(f'INCOMPLETE LOCAL SCORE — blocker: {blocker}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
