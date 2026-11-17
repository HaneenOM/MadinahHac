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


def heldout_params(recs, heldout, member):
    """Fit/cache T05's fixed 96-cell validation proxy without heldout labels."""
    other, prior = FIT_SOURCE[heldout]
    assert heldout not in (other, prior)
    source = recs[other]
    tmpl = build_template(source, [recs[prior]], lam=member['lam'])
    field = gp.observed(source, tmpl, slope(tmpl), clip=1.5)
    if len(field['t']) < 96:
        raise RuntimeError(f'T05 fitting source {other} has only {len(field["t"])} usable cells')
    pick = np.linspace(0, len(field['t']) - 1, 96, dtype=int)
    sample = {k: field[k][pick] for k in ('t', 'c', 'd', 'r')}
    key = f'heldout_{heldout}_sample96_{member["key"]}'
    return fit_cached(OUT / 't05_fit_cache', key, [sample], **fit_kwargs(member))


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
            predictions.append(gp.predict_recording(rec, tmpl, slope(tmpl), params, strict=True))
        stack = np.stack(predictions)
        counts = np.isfinite(stack).sum(axis=0)
        mean = np.divide(np.nansum(stack, axis=0), counts,
                         out=np.full(rec.y.shape, np.nan), where=counts > 0)
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
