"""T05 forecast ensemble on the fixed, date-held-out local-score cases.

Fitting follows the teammate's T05 local benchmark: 96 released cells from
the other test recording, with the evaluated training day excluded. Prediction
uses the same six ensemble members and strict cutoff path as the submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline import ROOT
from t05_forecast import gp
from t05_forecast.core import ABSENT, FORECAST, RELEASED, load_recordings
from t05_forecast.pipeline import fit_cached
from t05_forecast.submit import fit_kwargs, member_settings, settings
from t05_forecast.template import build_template, complete_template, slope

OUT = ROOT / 'outputs' / 'local_score'
NAMES = {'2026-09-11': '20260911_1100', '2026-09-12': '20260912_1100'}
FIT_SOURCE = {'20260911_1100': ('20260919_1100', '20260912_1100'),
              '20260912_1100': ('20260918_1100', '20260911_1100')}


def masked_recording(original, feed, targets):
    rec = original.copy()
    rec.truth = None
    rec.ids = {}
    rec.y[:] = np.nan
    rec.status[:] = ABSENT
    rec.now[:] = -1
    for unit in feed.columns:
        c = rec.col(unit)
        series = feed[unit].dropna()
        rows = (series.index.to_numpy(dtype=int) - rec.first).astype(int)
        assert ((rows >= 0) & (rows < rec.n_minutes)).all()
        rec.y[rows, c] = series.to_numpy(float)
        rec.status[rows, c] = RELEASED
    for r in targets.itertuples():
        row, col = int(r.minute) - rec.first, rec.col(r.unit)
        assert not np.isfinite(rec.y[row, col]), r.case_id
        rec.status[row, col] = FORECAST
        rec.now[row, col] = int(r.now)
    return rec


def heldout_params(recs, heldout, member, *, allow_legacy_provenance=False):
    """Load only after compatibility checks; NEVER optimize on a cache miss.

    Old scripts must explicitly acknowledge incomplete legacy provenance.
    Use reliable_evaluation.py for the supported evaluation workflow.
    """
    from evaluation_artifacts import validated_entries,ASSETS
    from t05_forecast.pipeline import load_params
    entries=validated_entries('historical',allow_legacy_provenance)
    return load_params(ASSETS/'historical'/entries[heldout,member['key']]['filename'])


def predict(cases, feeds):
    cfg = json.loads((ROOT / 't05_forecast_config.json').read_text(encoding='utf-8'))
    members = member_settings(settings(cfg))
    recs = load_recordings(ROOT / 'data' / 'raw')
    values = {}
    cutoff_checks = 0
    for day, name in NAMES.items():
        targets = cases[cases.day.eq(day) & cases.kind.eq('forecast')]
        rec = masked_recording(recs[name], feeds[name], targets)
        train_recs = [r for r in recs.values() if r.split == 'train' and r.name != name]
        assert all(r.day != day for r in train_recs)
        prior = [r for r in train_recs if r.weekday == rec.weekday]
        others = [r for r in train_recs if r.weekday != rec.weekday]
        if not prior:
            raise RuntimeError(f'T05 template for {day} has no same-weekday training source')
        predictions = []
        for i, member in enumerate(members, 1):
            print(f'T05 held-out {day}: member {i}/{len(members)}', flush=True)
            tmpl = build_template(rec, prior, lam=member['lam'])
            tmpl, _ = complete_template(rec, tmpl, others, lam=member['lam'])
            params = heldout_params(recs, name, member)
            predictions.append((tmpl, slope(tmpl), params))
        from baseline_tasks import condition_forecast
        mean = condition_forecast(rec, predictions)
        for r in targets.itertuples():
            row, col = int(r.minute) - rec.first, rec.col(r.unit)
            values[r.case_id] = float(mean[row, col])
        # The T05 strict path filters observations to <= now. Verify that the
        # actual observations passed to each cutoff are unchanged if all later
        # released camera values are perturbed.
        observed = np.argwhere(rec.status == RELEASED)
        for now in targets['now'].dropna().unique():
            earlier = observed[rec.first + observed[:, 0] <= int(now)]
            altered = rec.y.copy()
            altered[rec.minutes > int(now), :] = 1e6
            assert np.array_equal(rec.y[earlier[:, 0], earlier[:, 1]],
                                  altered[earlier[:, 0], earlier[:, 1]])
        cutoff_checks += len(targets)
    assert len(values) == int(cases.kind.eq('forecast').sum())
    return values, cutoff_checks
