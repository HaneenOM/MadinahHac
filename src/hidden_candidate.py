"""Targeted hidden blend trial for cameras B and D."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import ROOT, load, locate_data, table
from v2_candidate import HIDDEN
import v2_hidden

HIDDEN_CANDIDATE = {**HIDDEN,
                    'cam_B': ('dropout', 0.75),
                    'cam_D': ('ridge', 0.50)}


def read_submission(path: Path, sample_ids: list[str]) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'people']:
            raise ValueError(f'{path}: expected id,people')
        rows = list(reader)
    if len(rows) != 4457 or [r['id'] for r in rows] != sample_ids:
        raise ValueError(f'{path}: target IDs or order differ from sample')
    if len({r['id'] for r in rows}) != 4457:
        raise ValueError(f'{path}: duplicate target IDs')
    if not all(np.isfinite(float(r['people'])) and float(r['people']) >= 0 for r in rows):
        raise ValueError(f'{path}: invalid prediction')
    return rows


def build(output_path: Path | None = None):
    train, test, sample, _ = load(locate_data(None))
    sample_ids = sample.id.tolist()
    baseline_path = ROOT / 'outputs' / 'submission_t05_integrated.csv'
    baseline = read_submission(baseline_path, sample_ids)
    base_by_id = {r['id']: r['people'] for r in baseline}
    v1 = read_submission(ROOT / 'outputs' / 'submission_v1_0.987.csv', sample_ids)
    reference = {r['id']: float(r['people']) for r in v1}
    models = v2_hidden.fit(table(train))
    released = table(test[test.id.isna() & test.people.notna()])
    targets = test[test.id.notna()]
    changed = Counter()
    baseline_mismatch = Counter()
    output = [r.copy() for r in baseline]
    by_id = {r['id']: r for r in output}
    for r in targets[targets.kind.eq('hidden')].itertuples():
        t = released[r.recording]
        minute = int(r.minute_of_day)
        method, old_weight = HIDDEN[r.unit]
        fitted, _ = v2_hidden.predict(t, r.unit, minute, models, method)
        old = max(0., reference[r.id] + old_weight * (fitted-reference[r.id]))
        if not np.isfinite(old) or abs(old-float(base_by_id[r.id])) > 1e-9:
            baseline_mismatch[r.unit] += 1
            continue
        _, new_weight = HIDDEN_CANDIDATE[r.unit]
        if new_weight != old_weight:
            new = max(0., reference[r.id] + new_weight * (fitted-reference[r.id]))
            if not np.isfinite(new):
                raise ValueError(f'{r.id}: nonfinite hidden candidate')
            if abs(new-float(base_by_id[r.id])) > 1e-9:
                by_id[r.id]['people'] = str(float(new))
                changed[r.unit] += 1
    if baseline_mismatch:
        raise AssertionError(f'Integrated hidden baseline not reproduced: {dict(baseline_mismatch)}')
    path = output_path or ROOT / 'outputs' / 'submission_t05_hidden_v1.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'people'])
        writer.writeheader()
        writer.writerows(output)
    saved = read_submission(path, sample_ids)
    kinds = targets.set_index('id').kind.to_dict()
    altered = [r['id'] for r in saved if abs(float(r['people'])-float(base_by_id[r['id']])) > 1e-9]
    assert len(altered) == sum(changed.values())
    assert all(kinds[id_] == 'hidden' for id_ in altered)
    assert len(targets[targets.kind.eq('hidden')]) == 849
    print(f'Wrote {path}: {len(saved)} IDs; changed hidden predictions: {len(altered)}/849; '
          f'by camera: {dict(changed)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=Path)
    args = parser.parse_args()
    build(args.output_path)
