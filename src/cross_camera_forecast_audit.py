"""Prompt 22: audit existing cross-camera conditioning, without fitting a correction.

The ablation removes donor observations at inference only. It is NOT a proposed
replacement or a residual correction. All parameters must already be cached by
the fully excluded weekly diagnostic; this command is forbidden to fit models.
"""
from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import weekly_forecast_diagnostic as weekly
from t05_forecast import gp
from t05_forecast.core import load_recordings
from t05_forecast.pipeline import FRI, SAT, load_params, make_template
from t05_forecast.submit import member_settings, settings
from t05_forecast.template import slope

ROOT = weekly.ROOT
OUT = ROOT / 'outputs/cross_camera_forecast_audit'


def cached_only(cache, key, datasets, **kwargs):
    path = cache / f'params_{key}.npz'
    if not path.is_file():
        raise RuntimeError(f'Required clean cache is missing; refusing to refit: {path}')
    return load_params(path)


def summaries(frame, prefix, ablation=True):
    frame = frame.copy()
    frame['horizon_band'] = pd.cut(frame.horizon, [0, 5, 10, 15],
                                          labels=['1-5', '6-10', '11-15']).astype(str)
    frame['baseline_error'] = (frame.t05-frame.truth).abs()
    frame['reference_error'] = (frame.latest-frame.truth).abs()
    if ablation:
        frame['own_only_error'] = (frame.own_only-frame.truth).abs()
    # Normalization always uses a reference on EXACTLY the same targets. A
    # five-minute persistence reference is undefined outside the common subset.
    for subset, selected in [('all', frame), ('reference_eligible', frame[frame.latest.notna()])]:
        for groups, suffix in [(['day'], 'date'), (['day', 'unit'], 'camera'),
                               (['day', 'horizon_band'], 'horizon'),
                               (['day', 'window_id'], 'window')]:
            rows = []
            for key, g in selected.groupby(groups):
                key = key if isinstance(key, tuple) else (key,)
                row = dict(zip(groups, key))
                row.update(targets=len(g), windows=g.window_id.nunique(),
                           target_timestamps=g.minute.nunique(), baseline_mae=g.baseline_error.mean())
                if ablation:
                    row.update(own_only_mae=g.own_only_error.mean(),
                               donor_removal_delta=g.own_only_error.mean()-g.baseline_error.mean(),
                               affected=int((g.t05-g.own_only).abs().gt(1e-9).sum()))
                if subset == 'reference_eligible':
                    ref = g.reference_error.mean()
                    row.update(reference_mae=ref, baseline_normalized=row['baseline_mae']/ref if ref else None)
                    if ablation:
                        row['own_only_normalized'] = row['own_only_mae']/ref if ref else None
                row['sparse'] = len(g) < 30 or g.minute.nunique() < 30 or g.window_id.nunique() < 5
                rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / f'{prefix}_{subset}_by_{suffix}.csv', index=False)
    if ablation:
        rows = []
        for subset, selected in [('all', frame), ('reference_eligible', frame[frame.latest.notna()])]:
            for day, group in selected.groupby('day'):
                for window in group.window_id.unique():
                    g = group[group.window_id.ne(window)]
                    rows.append(dict(subset=subset, day=day, omitted_window=window,
                                     targets=len(g), windows=g.window_id.nunique(),
                                     baseline_mae=g.baseline_error.mean(), own_only_mae=g.own_only_error.mean(),
                                     donor_removal_delta=(g.own_only_error-g.baseline_error).mean()))
        pd.DataFrame(rows).to_csv(OUT / f'{prefix}_leave_one_window_out.csv', index=False)
    return frame


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = list((ROOT / 'outputs').glob('submission*.csv'))
    protected_paths += [weekly.OUT / f for f in ('cases.csv', 'labels.csv', 'predictions.csv',
                                                'benchmark_predictions.csv')]
    protected = {str(p): weekly.digest(p) for p in protected_paths}
    expected = json.loads((weekly.OUT / 'case_hashes.json').read_text())
    for name, sha in expected.items():
        assert weekly.digest(weekly.OUT / name) == sha
    recs = load_recordings(ROOT / 'data/raw')
    members = member_settings(settings(json.loads((ROOT / 't05_forecast_config.json').read_text())))
    cases = pd.read_csv(weekly.OUT / 'cases.csv')  # no labels read until predictions saved
    cached = pd.read_csv(weekly.OUT / 'predictions.csv').set_index('case_id')
    weekly.fit_cached = cached_only  # never train, even if a cache is absent
    fields, rows, checks = {}, [], 0
    with threadpool_limits(limits=4):
        for name in (FRI, SAT):
            source = recs[name]
            today = cases[cases.recording.eq(name)]
            full_values, own_values = [], []
            for member in members:
                with contextlib.redirect_stdout(io.StringIO()):
                    params = weekly.fitted_params(recs, name, member, fields)
                tmpl = make_template(recs, name, member['lam'])
                changed = source.copy(); changed.y[:] = 1e6
                altered = dict(recs); altered[name] = changed
                np.testing.assert_allclose(tmpl, make_template(altered, name, member['lam']),
                                           rtol=0, atol=1e-12, equal_nan=True)
                dslope = slope(tmpl)
                full, own = {}, {}
                for now, targets in today.groupby('now'):
                    masked = weekly.masked_at_now(source, targets, int(now))
                    altered_source = source.copy()
                    altered_source.y[altered_source.minutes > now] = 1e6
                    altered_mask = weekly.masked_at_now(altered_source, targets, int(now))
                    np.testing.assert_allclose(masked.y, altered_mask.y, rtol=0, atol=0, equal_nan=True)
                    assert np.array_equal(masked.status, altered_mask.status)
                    obs = gp.observed(masked, tmpl, dslope)
                    assert (obs['t'] <= now).all()
                    rr = np.array([int(r.minute)-source.first for r in targets.itertuples()])
                    cc = np.array([source.col(r.unit) for r in targets.itertuples()])
                    target = gp.cells(masked, rr, cc, dslope)
                    mean = gp.predict(params, obs, target)
                    values = np.maximum(np.expm1(tmpl[rr, cc]+mean), 0.)
                    for case_id, value in zip(targets.case_id, values):
                        full[case_id] = value
                    for unit, unit_targets in targets.groupby('unit'):
                        col = source.col(unit)
                        keep = obs['c'] == gp.UNIT_INDEX[unit]
                        own_obs = {k: obs[k][keep] for k in ('t', 'c', 'd', 'r')}
                        ur = unit_targets.minute.to_numpy(int)-source.first
                        uc = np.full(len(ur), col)
                        target = gp.cells(masked, ur, uc, dslope)
                        means = gp.predict(params, own_obs, target)
                        values = np.maximum(np.expm1(tmpl[ur, uc]+means), 0.)
                        for case_id, value in zip(unit_targets.case_id, values):
                            own[case_id] = value
                    checks += len(targets)
                full_values.append(np.array([full[c] for c in today.case_id]))
                own_values.append(np.array([own[c] for c in today.case_id]))
            baseline = weekly.finite_mean(full_values)
            own_only = weekly.finite_mean(own_values)
            np.testing.assert_allclose(baseline, cached.loc[today.case_id, 't05'], rtol=0, atol=1e-9)
            for r, value, own_value in zip(today.itertuples(index=False), baseline, own_only):
                rows.append(dict(**r._asdict(), t05=value, own_only=own_value,
                                 latest=cached.loc[r.case_id, 'latest']))
            print(f'{source.day}: completed {len(today)} cases; no model fitting', flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'predictions.csv', index=False)
    labels = pd.read_csv(weekly.OUT / 'labels.csv')
    scored = detail.merge(labels, on='case_id', validate='one_to_one')
    assert len(scored) == len(cases)
    summaries(scored, 'recent').to_csv(OUT / 'scored.csv', index=False)
    # Separate older stress test, reusing the previous verified predictions.
    older = pd.read_csv(weekly.OUT / 'benchmark_scored.csv')
    summaries(older, 'older', ablation=False)
    assert all(weekly.digest(Path(p)) == sha for p, sha in protected.items())
    coverage = pd.read_csv(weekly.OUT / 'input_coverage_summary.csv')
    coverage.to_csv(OUT / 'input_coverage_summary.csv', index=False)
    report = dict(decision='No residual correction trained: proposed cross-camera observations already condition T05.',
                  ablation='Inference-only donor removal; not a correction candidate or replacement.',
                  recent_cases=len(detail), member_cutoff_checks=checks, fitted_models=0,
                  baseline_matches_clean_weekly_predictions_atol=1e-9,
                  exclusion='Each evaluation recording excluded directly and from fitting-template completion; cached clean parameters only.',
                  normalization='MAE / five-minute persistence MAE on identical reference-eligible cases; undefined on other cases.',
                  actual_AH_forecast_rows=int(coverage.loc[coverage.source.eq('real_requests'), 'cases'].sum()),
                  proposed_changed_submission_rows=0, submission_generated=False,
                  all_saved_submissions_and_fixed_cases_unchanged=True,
                  elapsed_seconds=time.perf_counter()-started,
                  protected_sha256=protected)
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(pd.read_csv(OUT / 'recent_all_by_date.csv').to_string(index=False))
    print(pd.read_csv(OUT / 'recent_reference_eligible_by_date.csv').to_string(index=False))
    print(f'Elapsed: {report["elapsed_seconds"]:.1f}s; decision: no redundant correction model.')


if __name__ == '__main__':
    main()
