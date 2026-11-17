"""Date-held-out A-H missing diagnostics on the immutable local masks."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from pipeline import ROOT, load, locate_data, table
from local_score import OUT as LOCAL, initialize_cases, load_cases
from missing_median_trial_model import baseline, predict as candidate_predict
import v2_missing

OUT = ROOT / 'outputs/missing_targeted_trial'


def metadata(t, unit, minute, calibration):
    ctx = v2_missing.gap_context(t, unit, minute)
    assert ctx is not None
    start, end, frac, linear = ctx
    series = t[unit].dropna()
    left, right = float(series.loc[start]), float(series.loc[end])
    observed = pd.concat([series.loc[start-5:start], series.loc[end:end+5]])
    spread = float(observed.max()-observed.min())
    sharp = spread >= max(10., .5*max((left+right)/2, 1.))
    corrections = []
    for donor, deviation in v2_missing.deviations(t, unit, minute, ctx).items():
        if (unit, donor) in calibration:
            slope, before, after, _ = calibration[(unit, donor)]
            if before > after:
                corrections.append((before-after, donor, slope*deviation))
    corrections.sort(reverse=True)
    top = corrections[:4]
    raw = np.array([x[2] for x in top])
    return dict(start=start, end=end, gap_length=end-start-1,
                gap_id=f'{unit}:{start}-{end}', endpoint_left=left, endpoint_right=right,
                endpoint_jump=abs(right-left), observed_spread=spread, sharp=sharp,
                donor_candidates=len(corrections),
                donor_positive=int((raw > 0).sum()), donor_negative=int((raw < 0).sum()),
                donor_disagreement=bool((raw > 0).any() and (raw < 0).any()))


def score(detail):
    labels = pd.read_csv(LOCAL / 'labels.csv')
    scored = detail.merge(labels, on='case_id', validate='one_to_one')
    assert len(scored) == len(detail) == 358
    current = pd.read_csv(LOCAL / 'predictions.csv')
    checked = scored.merge(current[['case_id', 'prediction', 'reference']], on='case_id',
                           validate='one_to_one', suffixes=('', '_current'))
    assert len(checked) == 358
    np.testing.assert_allclose(checked.baseline, checked.prediction, rtol=0, atol=1e-9)
    np.testing.assert_allclose(checked.reference, checked.reference_current, rtol=0, atol=1e-9)
    scored['baseline_error'] = (scored.truth-scored.baseline).abs()
    scored['reference_error'] = (scored.truth-scored.reference).abs()
    if 'candidate' in scored:
        scored['candidate_error'] = (scored.truth-scored.candidate).abs()
    scored.to_csv(OUT / 'scored.csv', index=False)
    outages = scored.groupby(['day', 'recording', 'unit', 'gap_id']).agg(
        targets=('case_id', 'size'), gap_length=('gap_length', 'first'), sharp=('sharp', 'first'),
        endpoint_jump=('endpoint_jump', 'first'), observed_spread=('observed_spread', 'first'),
        baseline_mae=('baseline_error', 'mean'), reference_mae=('reference_error', 'mean'),
        max_error=('baseline_error', 'max'), total_error=('baseline_error', 'sum'),
        donor_disagreement_fraction=('donor_disagreement', 'mean')).reset_index()
    outages.sort_values('total_error', ascending=False).to_csv(OUT / 'outages.csv', index=False)
    scored.sort_values('baseline_error', ascending=False).head(30).to_csv(OUT / 'largest_errors.csv', index=False)
    scored['gap_band'] = pd.cut(scored.gap_length, [0, 5, 10, 20, 30], labels=['1-5', '6-10', '11-20', '21-30']).astype(str)
    for groups, name in [(['day'], 'by_date'), (['unit'], 'by_camera'),
                         (['gap_band', 'sharp'], 'by_length_sharp'),
                         (['donor_disagreement'], 'by_donor_disagreement')]:
        result = scored.groupby(groups).agg(targets=('case_id', 'size'), outages=('gap_id', 'nunique'),
                                             baseline_mae=('baseline_error', 'mean'),
                                             reference_mae=('reference_error', 'mean')).reset_index()
        result['normalized_mae'] = result.baseline_mae/result.reference_mae
        result.to_csv(OUT / f'{name}.csv', index=False)
        print(name); print(result.to_string(index=False))
    print('Largest outages by total absolute error:')
    print(outages.sort_values('total_error', ascending=False).head(15).to_string(index=False))
    if 'candidate' in scored:
        results = []
        for (day, unit), g in [*((key, g) for key, g in scored.groupby(['day', 'unit'])),
                               *(( (day, 'all'), g) for day, g in scored.groupby('day'))]:
            old = g.baseline_error.mean(); new = g.candidate_error.mean()
            ref = g.reference_error.mean()
            results.append(dict(day=day, unit=unit, targets=len(g), outages=g.gap_id.nunique(),
                                baseline_mae=old, candidate_mae=new,
                                baseline_normalized=old/ref, candidate_normalized=new/ref))
        result = pd.DataFrame(results)
        result.to_csv(OUT / 'candidate_scores.csv', index=False)
        print('Paired candidate scores:'); print(result.to_string(index=False))
        per_gap = scored.groupby(['day', 'unit', 'gap_id']).agg(
            targets=('case_id', 'size'), baseline_mae=('baseline_error', 'mean'),
            candidate_mae=('candidate_error', 'mean'),
            baseline_total=('baseline_error', 'sum'), candidate_total=('candidate_error', 'sum')).reset_index()
        per_gap['improvement_total'] = per_gap.baseline_total-per_gap.candidate_total
        per_gap['improved'] = per_gap.improvement_total.gt(1e-9)
        per_gap['regressed'] = per_gap.improvement_total.lt(-1e-9)
        per_gap.to_csv(OUT / 'candidate_by_outage.csv', index=False)
        print('Gains across outages:')
        print(per_gap.groupby('day').agg(outages=('gap_id', 'size'), improved=('improved', 'sum'),
                                         regressed=('regressed', 'sum'), net_gain=('improvement_total', 'sum')).to_string())
        print(per_gap.sort_values('improvement_total', ascending=False).head(8).to_string(index=False))
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnose', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'outputs').glob('submission*.csv')}
    train, test, _, _ = load(locate_data(None))
    initialize_cases(train, test)
    cases, feeds = load_cases()
    cases = cases[cases.kind.eq('missing') & cases.unit.isin([f'cam_{c}' for c in 'ABCDEFGH'])]
    assert len(cases) == 358
    rows = []
    endpoint_checks = set()
    for day, today in cases.groupby('day'):
        fitting = train[train.day.ne(day)]
        assert day not in set(fitting.day)
        calibration = v2_missing.fit_calibration(table(fitting))
        for r in today.itertuples():
            minute = int(r.minute); t = feeds[r.recording]
            assert r.unit not in t or minute not in t.index or not np.isfinite(t.at[minute, r.unit])
            reference, prediction, count = baseline(t, r.unit, minute, calibration)
            meta = metadata(t, r.unit, minute, calibration)
            meta['gap_id'] = f'{r.recording}:{meta["gap_id"]}'
            extra = {}
            if not args.diagnose:
                extra['candidate'] = candidate_predict(t, r.unit, minute, calibration)
                for endpoint in (meta['start'], meta['end']):
                    assert candidate_predict(t, r.unit, endpoint, calibration) == t.at[endpoint, r.unit]
                    endpoint_checks.add((r.recording, r.unit, endpoint))
            rows.append(dict(case_id=r.case_id, day=day, recording=r.recording,
                             unit=r.unit, minute=minute, reference=reference,
                             baseline=prediction, applied_donors=count, **meta, **extra))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'predictions.csv', index=False)
    score(detail)
    assert protected == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT / 'outputs').glob('submission*.csv')}
    (OUT / 'verification.json').write_text(json.dumps(dict(
        baseline_verified_cases=len(detail), endpoints_checked=len(endpoint_checks),
        fit_excludes_evaluated_date=True, saved_submissions_unchanged=True,
        decision='Retain v2: Friday regression and Saturday gain concentrated in one outage',
        candidate='Median of v2 correction at minute-1, minute, minute+1; zero at endpoints'),
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
