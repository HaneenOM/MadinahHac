"""One fixed-case trial: increase the v2 donor blend for hidden B and D."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, load, locate_data, table
from v2_candidate import HIDDEN
from v2_missing import interp_at
import v2_hidden

OUT = ROOT / 'outputs' / 'hidden_blend_trial'
CANDIDATE_WEIGHT = {'cam_B': 0.75, 'cam_D': 0.50}


def main():
    train, _, _, _ = load(locate_data(None))
    fixed = ROOT / 'outputs' / 'local_score'
    cases = pd.read_csv(fixed / 'cases.csv').query("kind == 'hidden'")
    released = pd.read_csv(fixed / 'released_feed.csv')
    feed_keys = pd.MultiIndex.from_frame(released[['recording', 'unit', 'minute_of_day']])
    target_keys = pd.MultiIndex.from_arrays([cases.recording, cases.unit, cases.minute])
    assert not target_keys.isin(feed_keys).any(), 'Hidden answers appear in the released feed'
    feeds = table(released)
    rows = []
    for day, group in cases.groupby('day'):
        fit = train[train.day.ne(day)]
        assert day not in set(fit.day)
        models = v2_hidden.fit(table(fit))
        for r in group.itertuples():
            t, minute = feeds[r.recording], int(r.minute)
            method, base_weight = HIDDEN[r.unit]
            reference, available = v2_hidden.predict(t, r.unit, minute, models, 'single')
            donor_value, _ = v2_hidden.predict(t, r.unit, minute, models, method)
            candidate_weight = CANDIDATE_WEIGHT.get(r.unit, base_weight)
            top_donor = models[r.unit]['pairs'][0][1]
            top_value = interp_at(t[top_donor], minute) if top_donor in t else np.nan
            rows.append(dict(case_id=r.case_id, day=day, unit=r.unit,
                             reference=reference,
                             baseline=reference + base_weight * (donor_value-reference),
                             candidate=reference + candidate_weight * (donor_value-reference),
                             available=available, top_donor=top_donor,
                             top_available=bool(np.isfinite(top_value))))
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(rows)
    assert len(pred) == len(cases) == 843 and pred.case_id.is_unique
    assert np.isfinite(pred[['reference', 'baseline', 'candidate']]).all().all()
    pred.to_csv(OUT / 'predictions.csv', index=False)

    # Labels remain separate until every candidate prediction is saved.
    labels = pd.read_csv(fixed / 'labels.csv')
    scored = pred.merge(labels, on='case_id', how='inner', validate='one_to_one')
    assert len(scored) == len(pred)
    for column in ('reference', 'baseline', 'candidate'):
        scored[f'{column}_error'] = (scored[column]-scored.truth).abs()
    scored['crowd_band'] = pd.cut(scored.truth, [-1, 20, 50, 100, np.inf],
                                  labels=['0-20', '21-50', '51-100', '>100'])
    scored.to_csv(OUT / 'scored_predictions.csv', index=False)
    summary = scored.groupby(['day', 'unit']).agg(
        n=('case_id', 'size'),
        baseline_mae=('baseline_error', 'mean'),
        candidate_mae=('candidate_error', 'mean'),
        reference_mae=('reference_error', 'mean')).reset_index()
    summary['baseline_normalized'] = summary.baseline_mae / summary.reference_mae
    summary['candidate_normalized'] = summary.candidate_mae / summary.reference_mae
    summary['mae_change'] = summary.candidate_mae-summary.baseline_mae
    summary.to_csv(OUT / 'by_date_camera.csv', index=False)
    availability = scored.groupby(['day', 'unit', 'top_available']).agg(
        n=('case_id', 'size'), baseline_mae=('baseline_error', 'mean'),
        candidate_mae=('candidate_error', 'mean')).reset_index()
    availability.to_csv(OUT / 'by_donor_availability.csv', index=False)
    crowd = scored.groupby(['day', 'unit', 'crowd_band'], observed=True).agg(
        n=('case_id', 'size'), baseline_mae=('baseline_error', 'mean'),
        candidate_mae=('candidate_error', 'mean')).reset_index()
    crowd.to_csv(OUT / 'by_crowd_level.csv', index=False)
    top = scored.nlargest(20, 'baseline_error')
    top.to_csv(OUT / 'largest_baseline_errors.csv', index=False)
    print(summary.to_string(index=False))
    print('\nBy reference donor availability:')
    print(availability.to_string(index=False))
    print('\nBy crowd level:')
    print(crowd.to_string(index=False))


if __name__ == '__main__':
    main()
