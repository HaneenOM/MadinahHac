"""One prespecified Friday Dhuhr timing trial; labels enter only after prediction."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from pipeline import ROOT, load, locate_data, table
from local_score import OUT as LOCAL, initialize_cases, load_cases
from prayer_experiments import timetable
from v2_candidate import HF_WEIGHT
from v2_experiments import DIRECT_COLS, cases as transplant, direct_feature, fit_direct, predict_hf
import v2_hidden

OUT = ROOT / 'outputs/friday_arrival_trial'
WINDOWS = ('60-90', '30-60', '0-30', 'outside')
COLS = DIRECT_COLS + ['friday_dhuhr_now', 'friday_dhuhr_target']


def window(offset):
    for low, high, name in ((-90, -60, '60-90'), (-60, -30, '30-60'), (-30, 0, '0-30')):
        if low <= offset < high:
            return name
    return 'outside'


def timing(day, now, h, schedule):
    # Friday only; NaN on Saturday. These two features are fixed before scoring.
    if pd.Timestamp(day).day_of_week != 4:
        return [np.nan, np.nan]
    dhuhr = int(schedule[day][1])
    return [now-dhuhr, now+h-dhuhr]


def fit_candidate(tables, rec_day, schedule, baseline):
    rows, labels = [], []
    # Identical sampling and labels to fit_direct; clone its estimator settings.
    for rec, t in tables.items():
        day = rec_day[rec]
        for unit in v2_hidden.PATTERNS:
            if unit not in t:
                continue
            for now in range(int(t.index.min())+20, int(t.index.max())-15, 8):
                base = direct_feature(t, unit, now, 0)
                for h in range(1, 16):
                    truth = t[unit].get(now+h, np.nan)
                    if np.isfinite(truth):
                        x = base.copy(); x[1] = h
                        rows.append(x + timing(day, now, h, schedule))
                        labels.append(float(truth))
    X = pd.DataFrame(rows, columns=COLS)
    used = X.columns[X.notna().any()].tolist()
    model = clone(baseline[0])
    assert model.get_params() == baseline[0].get_params()
    model.fit(X[used], labels)
    return model, used, len(labels)


def pattern_summary(train, schedule):
    # Called only after saving/scoring predictions; these held-out counts never
    # select features or enter the fold's model fit.
    rows = []
    friday = train[train.weekday.eq('Fri') & train.unit.isin(['cam_B', 'cam_G'])]
    for (day, rec, unit), g in friday.groupby(['day', 'recording', 'unit']):
        g = g[g.people.notna()].sort_values('minute_of_day').copy()
        g['offset'] = g.minute_of_day-int(schedule[day][1])
        g['window'] = g.offset.map(window)
        for name in WINDOWS:
            part = g[g.window.eq(name)]
            rows.append(dict(day=day, recording=rec, unit=unit, window=name,
                             observations=len(part),
                             first_offset=part.offset.min(), last_offset=part.offset.max(),
                             mean_people=part.people.mean(), median_people=part.people.median(),
                             first5_mean=part.people.head(5).mean(),
                             last5_mean=part.people.tail(5).mean()))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'observed_patterns.csv', index=False)
    print('Observed Friday counts (descriptive, after scoring):')
    print(result.to_string(index=False))


def main():
    OUT.mkdir(exist_ok=True)
    submissions = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ROOT / 'outputs').glob('submission*.csv')}
    train, test, _, _ = load(locate_data(None))
    schedule = timetable(ROOT / 'data/raw/prayer_times_madinah.csv', train, test)
    initialize_cases(train, test)
    fixed, feeds = load_cases()
    fixed = fixed[fixed.kind.eq('hidden-future') & fixed.day.eq('2026-09-11')].copy()
    assert len(fixed) == 322 and set(fixed.unit) == {'cam_B', 'cam_G'}
    # Transplant the same release mask to the other Friday without exposing
    # answers to feature construction or prediction.
    secondary_cases, secondary_labels = [], []
    for day, rec, feed, targets in transplant(train[train.day.eq('2026-09-04')], test):
        feeds[rec] = feed
        for r in targets[targets.kind.eq('hidden-future')].itertuples():
            case_id = f'{rec}_{r.unit}_{int(r.minute_of_day)}'
            secondary_cases.append(dict(case_id=case_id, day=day, recording=rec,
                                        unit=r.unit, minute=int(r.minute_of_day),
                                        now=int(r.now_minute_of_day)))
            secondary_labels.append(dict(case_id=case_id, truth=float(r.people)))
    secondary = pd.DataFrame(secondary_cases)
    secondary.to_csv(OUT / 'secondary_cases.csv', index=False)
    pd.DataFrame(secondary_labels).to_csv(OUT / 'secondary_labels.csv', index=False)
    evaluation = pd.concat([fixed[['case_id', 'day', 'recording', 'unit', 'minute', 'now']],
                            secondary], ignore_index=True)
    rows, fits = [], []
    checks = 0
    for day, today in evaluation.groupby('day'):
        fitting = train[train.day.ne(day)]
        assert day not in set(fitting.day)
        tables = table(fitting)
        rec_day = fitting.drop_duplicates('recording').set_index('recording').day.to_dict()
        hidden = v2_hidden.fit(tables)
        baseline = fit_direct(tables)
        candidate, used, n_train = fit_candidate(tables, rec_day, schedule, baseline)
        fits.append(dict(day=day, fit_dates=sorted(fitting.day.unique()),
                         training_examples=n_train, estimator=baseline[0].get_params()))
        for r in today.itertuples():
            now = int(r.now); minute = int(r.minute); h = minute-now
            t = feeds[r.recording]
            assert 1 <= h <= 15 and r.unit in ('cam_B', 'cam_G')
            assert all(c not in t or t[c].isna().all() for c in ('cam_B', 'cam_G'))
            def predict(feed):
                ref = predict_hf(feed, r.unit, now, h, hidden, baseline, 'fixed')
                old = predict_hf(feed, r.unit, now, h, hidden, baseline, 'direct')
                x = direct_feature(feed, r.unit, now, h) + timing(day, now, h, schedule)
                new = max(0., float(candidate.predict(pd.DataFrame([x], columns=COLS)[used])[0]))
                return [ref, max(0., ref+HF_WEIGHT*(old-ref)),
                        max(0., ref+HF_WEIGHT*(new-ref))]
            values = predict(t)
            altered = t.copy(); altered.loc[altered.index > now] = 1e6
            np.testing.assert_allclose(values, predict(altered), rtol=0, atol=1e-10)
            checks += 1
            rows.append(dict(case_id=r.case_id, day=day, unit=r.unit, minute=minute,
                             window=window(minute-int(schedule[day][1])),
                             reference=values[0], baseline=values[1], candidate=values[2]))
    predictions = pd.DataFrame(rows)
    predictions.to_csv(OUT / 'predictions.csv', index=False)
    # Scoring starts only after every prediction is saved.
    labels = pd.concat([pd.read_csv(LOCAL / 'labels.csv'),
                        pd.read_csv(OUT / 'secondary_labels.csv')], ignore_index=True)
    scored = predictions.merge(labels, on='case_id', validate='one_to_one')
    assert len(scored) == len(predictions)
    current = pd.read_csv(LOCAL / 'predictions.csv')
    verified = predictions[predictions.day.eq('2026-09-11')].merge(
        current[['case_id', 'reference', 'prediction']], on='case_id', validate='one_to_one',
        suffixes=('', '_current'))
    assert len(verified) == 322
    np.testing.assert_allclose(verified.baseline, verified.prediction, rtol=0, atol=1e-9)
    np.testing.assert_allclose(verified.reference, verified.reference_current, rtol=0, atol=1e-9)
    scored.to_csv(OUT / 'scored.csv', index=False)
    summaries = []
    for day, today in scored.groupby('day'):
        for unit in ('cam_B', 'cam_G', 'combined'):
            g = today if unit == 'combined' else today[today.unit.eq(unit)]
            for name in ('all', *WINDOWS):
                part = g if name == 'all' else g[g.window.eq(name)]
                ref = (part.truth-part.reference).abs().mean()
                old = (part.truth-part.baseline).abs().mean()
                new = (part.truth-part.candidate).abs().mean()
                summaries.append(dict(day=day, unit=unit, window=name, targets=len(part),
                                      reference_mae=ref, baseline_mae=old, candidate_mae=new,
                                      baseline_normalized=old/ref, candidate_normalized=new/ref))
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / 'scores.csv', index=False)
    print(result.to_string(index=False))
    pattern_summary(train, schedule)
    assert submissions == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT / 'outputs').glob('submission*.csv')}
    report = dict(post_cutoff_checks=checks, baseline_verified_cases=len(verified),
                  fit_details=fits, saved_submissions_unchanged=True,
                  features=COLS[-2:], window_definition='[-90,-60), [-60,-30), [-30,0); target minute',
                  secondary_independence='September 4 was already used in v3 prayer experiments')
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Post-cutoff checks: {checks}; all saved submissions unchanged')


if __name__ == '__main__':
    main()
