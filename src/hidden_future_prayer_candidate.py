"""Add Saturday prayer-time hidden-future predictions to the hidden candidate."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from hidden_candidate import read_submission
from pipeline import ROOT, load, locate_data, table
from prayer_experiments import fit_direct_variant, predict_direct_variant, timetable
from v2_candidate import HF_WEIGHT
from v2_experiments import fit_direct, predict_hf
import v2_hidden


def build(output_path: Path | None = None, *, data_dir=None, baseline_path=None,
          reference_path=None, timetable_path=None):
    train, test, sample, recordings = load(Path(data_dir) if data_dir else locate_data(None))
    ids = sample.id.tolist()
    out = ROOT / 'outputs'
    base = read_submission(Path(baseline_path) if baseline_path else out / 'submission_t05_hidden_v1.csv', ids)
    base_values = {r['id']: float(r['people']) for r in base}
    v1 = read_submission(Path(reference_path) if reference_path else out / 'submission_v1_0.987.csv', ids)
    reference = {r['id']: float(r['people']) for r in v1}
    schedule = timetable(Path(timetable_path) if timetable_path else ROOT / 'data/raw/prayer_times_madinah.csv', train, test)
    fit_tables = table(train)
    hidden = v2_hidden.fit(fit_tables)
    direct = fit_direct(fit_tables)
    rec_day = recordings.set_index('recording').day.to_dict()
    prayer = fit_direct_variant(fit_tables, rec_day, schedule, 'prayer')
    released = table(test[test.id.isna() & test.people.notna()])
    targets = test[test.id.notna()]
    result = [r.copy() for r in base]
    by_id = {r['id']: r for r in result}
    changed = Counter()
    checks = 0
    for r in targets[targets.kind.eq('hidden-future')].itertuples():
        t = released[r.recording]
        now = int(r.now_minute_of_day)
        horizon = int(r.minute_of_day) - now
        assert 1 <= horizon <= 15
        original = predict_hf(t, r.unit, now, horizon, hidden, direct, 'direct')
        expected = max(0., reference[r.id] + HF_WEIGHT * (original-reference[r.id]))
        if abs(expected-base_values[r.id]) > 1e-9:
            raise AssertionError(f'Base hidden-future row differs from v2: {r.id}')
        if r.weekday != 'Sat':
            continue
        fitted = predict_direct_variant(t, r.unit, now, horizon, r.day,
                                        schedule, prayer, 'prayer')
        altered = t.copy()
        altered.loc[altered.index > now] = 1e6
        check = predict_direct_variant(altered, r.unit, now, horizon, r.day,
                                       schedule, prayer, 'prayer')
        np.testing.assert_allclose(fitted, check, rtol=0, atol=1e-10)
        checks += 1
        prediction = max(0., reference[r.id] + HF_WEIGHT * (fitted-reference[r.id]))
        if abs(prediction-base_values[r.id]) > 1e-9:
            by_id[r.id]['people'] = str(float(prediction))
            changed[r.day] += 1
    path = output_path or out / 'submission_t05_hidden_hidden_future_v1.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'people'])
        writer.writeheader()
        writer.writerows(result)
    reread = read_submission(path, ids)
    kind = targets.set_index('id').kind.to_dict()
    day = targets.set_index('id').day.to_dict()
    altered_ids = [r['id'] for r in reread if abs(float(r['people'])-base_values[r['id']]) > 1e-9]
    assert len(altered_ids) == sum(changed.values())
    assert all(kind[id_] == 'hidden-future' and day[id_] == '2026-09-19' for id_ in altered_ids)
    assert len(targets[targets.kind.eq('hidden-future')]) == 484
    print(f'Wrote {path}: {len(reread)} IDs; changed by date: {dict(changed)}; '
          f'post-cutoff checks: {checks}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=Path)
    args = parser.parse_args()
    build(args.output_path)
