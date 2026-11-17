"""Paired prayer-feature trial on the immutable local hidden-future cases."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import ROOT, load, locate_data, table
from local_score import DATES, OUT, load_cases
from prayer_experiments import (fit_direct_variant, predict_direct_variant,
                                schedule_features, timetable)
from v2_candidate import HF_WEIGHT
from v2_experiments import fit_direct, predict_hf
import v2_hidden


def main():
    train, test, _, recordings = load(locate_data(None))
    schedule = timetable(ROOT / 'data/raw/prayer_times_madinah.csv', train, test)
    cases, feeds = load_cases()
    cases = cases[cases.kind.eq('hidden-future')].copy()
    assert len(cases) == 482
    saved = pd.read_csv(ROOT / 'outputs/hidden_future_benchmark/cases.csv')
    assert set(saved.case_id) == set(cases.case_id)
    assert cases.groupby('day').size().to_dict() == {DATES[0]: 322, DATES[1]: 160}
    rec_day = recordings.set_index('recording').day.to_dict()
    rows = []
    checks = 0
    for day in DATES:
        fit = train[train.day.ne(day)]
        assert day not in set(fit.day)
        fit_tables = table(fit)
        hidden = v2_hidden.fit(fit_tables)
        direct = fit_direct(fit_tables)
        clock = fit_direct_variant(fit_tables, rec_day, schedule, 'clock')
        prayer = fit_direct_variant(fit_tables, rec_day, schedule, 'prayer')
        for r in cases[cases.day.eq(day)].itertuples():
            t = feeds[r.recording]
            now = int(r.now); h = int(r.horizon)
            assert int(r.minute) == now + h and 1 <= h <= 15
            assert r.unit in (('cam_B', 'cam_G') if day == DATES[0] else ('cam_D',))
            assert r.unit not in t or int(r.minute) not in t.index or not np.isfinite(t.at[int(r.minute), r.unit])
            def values(feed):
                ref = predict_hf(feed, r.unit, now, h, hidden, direct, 'fixed')
                base = predict_hf(feed, r.unit, now, h, hidden, direct, 'direct')
                control = predict_direct_variant(feed, r.unit, now, h, day,
                                                 schedule, clock, 'clock')
                changed = predict_direct_variant(feed, r.unit, now, h, day,
                                                 schedule, prayer, 'prayer')
                return [ref, *[max(0., ref + HF_WEIGHT * (x-ref))
                               for x in (base, control, changed)]]
            vals = values(t)
            altered = t.copy()
            altered.loc[altered.index > now] = 1e6
            np.testing.assert_allclose(vals, values(altered), rtol=0, atol=1e-10)
            checks += 1
            near = abs(schedule_features(day, now, h, schedule)[2]) <= 15
            rows.append(dict(case_id=r.case_id, day=day, unit=r.unit,
                             near_prayer=near, reference=vals[0],
                             v2=vals[1], clock=vals[2], prayer=vals[3]))
    trial = ROOT / 'outputs/hidden_future_prayer_trial'
    trial.mkdir(exist_ok=True)
    predictions = pd.DataFrame(rows)
    assert predictions.case_id.is_unique and len(predictions) == 482
    predictions.to_csv(trial / 'predictions.csv', index=False)
    labels = pd.read_csv(OUT / 'labels.csv')
    scored = predictions.merge(labels, on='case_id', validate='one_to_one')
    assert len(scored) == 482
    baseline = pd.read_csv(ROOT / 'outputs/hidden_future_benchmark/predictions.csv')
    check = scored[['case_id', 'v2', 'reference']].merge(
        baseline[['case_id', 'prediction', 'reference']], on='case_id',
        validate='one_to_one', suffixes=('', '_local'))
    np.testing.assert_allclose(check.v2, check.prediction, rtol=0, atol=1e-9)
    np.testing.assert_allclose(check.reference, check.reference_local, rtol=0, atol=1e-9)
    scored.to_csv(trial / 'scored.csv', index=False)
    summary = []
    for (day, near), group in scored.groupby(['day', 'near_prayer']):
        ref_mae = (group.truth-group.reference).abs().mean()
        for variant in ('v2', 'clock', 'prayer'):
            mae = (group.truth-group[variant]).abs().mean()
            summary.append(dict(day=day, window='within_15_min' if near else 'outside_15_min',
                                variant=variant, targets=len(group), mae_people=mae,
                                reference_mae_people=ref_mae, normalized_mae=mae/ref_mae))
    for day, group in scored.groupby('day'):
        ref_mae = (group.truth-group.reference).abs().mean()
        for variant in ('v2', 'clock', 'prayer'):
            mae = (group.truth-group[variant]).abs().mean()
            summary.append(dict(day=day, window='all', variant=variant,
                                targets=len(group), mae_people=mae,
                                reference_mae_people=ref_mae, normalized_mae=mae/ref_mae))
    result = pd.DataFrame(summary).sort_values(['day', 'window', 'variant'])
    result.to_csv(trial / 'scores.csv', index=False)
    print(result.to_string(index=False))
    print('Post-cutoff invariance checks:', checks)


if __name__ == '__main__':
    main()
