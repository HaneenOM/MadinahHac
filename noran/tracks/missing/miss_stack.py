"""Combining the candidate fills: weights fitted on pseudo-gaps, plus row-wise copies of the two existing methods
(the accepted v2 filler and the partner-ratio example) so that all three can be compared on the same pseudo-gaps.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

MIN_LEVEL = 5.0


def v2_values(d: pd.DataFrame, calibration: dict, config: dict) -> np.ndarray:
    """The accepted method (src/v2_missing.py 'blend' through baseline_tasks.missing_value) computed from the
    dv_ columns of pseudo-gap rows. check_v2_copy.py verifies this copy against the saved accepted values."""
    out = d.base.to_numpy(float).copy()
    for unit, g in d.groupby("unit"):
        weight = config["weights"].get(unit, 0.0)
        if not weight:
            continue
        donors = []
        for column in [c for c in g.columns if c.startswith("dv_")]:
            key = (unit, "cam_" + column[3:])
            if key in calibration and isinstance(calibration[key], tuple):
                slope, baseline, residual, _ = calibration[key]
                if baseline - residual > 0:
                    donors.append((baseline - residual, "cam_" + column[3:], slope, column))
        donors.sort(reverse=True)
        n = len(g)
        taken = np.zeros(n, int)
        strength = np.zeros(n)
        total = np.zeros(n)
        for improvement, _, slope, column in donors:
            x = g[column].to_numpy(float)
            use = np.isfinite(x) & (taken < 4)
            total[use] += improvement * slope * x[use]
            strength[use] += improvement
            taken[use] += 1
        base = g.base.to_numpy(float)
        correction = total / (strength + 1e-9)
        limit = 0.5 * np.maximum(base, 5.0)
        correction = np.clip(correction, -limit, limit)
        value = np.maximum(0.0, base + weight * correction)
        ok = (taken > 0) & (g.span.to_numpy() >= config["min_gap"])
        out[d.index.get_indexer(g.index)] = np.where(ok, value, base)
    return out


def example_values(d: pd.DataFrame, models: dict, fallback: np.ndarray, max_rel: float = 0.5) -> np.ndarray:
    """The partner-ratio example (src/noran_missing.py with side=4) computed from the lg_ columns; rows without a
    usable partner keep `fallback` (the accepted value), as noran/tracks/_example does."""
    out = np.asarray(fallback, float).copy()
    for unit, g in d.groupby("unit"):
        partners = models["partners"].get(unit, [])
        if not partners or unit not in ("cam_A", "cam_E", "cam_F"):
            continue
        base = g.base.to_numpy(float)
        scale = np.maximum(base, MIN_LEVEL)
        feats = {p: (g[f"lg_{p[-1]}"].to_numpy(float) - base) / scale if f"lg_{p[-1]}" in g else np.full(len(g), np.nan)
                 for p in partners}
        value = np.full(len(g), np.nan)
        avail = np.stack([np.isfinite(feats[p]) for p in partners], axis=1)
        for pattern in {tuple(r) for r in avail.tolist()}:
            used = tuple(p for p, flag in zip(partners, pattern) if flag)
            rows = (avail == np.array(pattern)).all(axis=1)
            key = (unit, used)
            if not used or key not in models["coef"]:
                continue
            x = np.stack([feats[p][rows] for p in used], axis=1)
            rel = np.clip(x @ models["coef"][key]["w"], -max_rel, max_rel)
            value[rows] = np.maximum(0.0, base[rows] + scale[rows] * rel)
        idx = d.index.get_indexer(g.index)
        out[idx] = np.where(np.isfinite(value), value, out[idx])
    return out


def _solve(x: np.ndarray, y: np.ndarray, ridge: float, loss: str) -> np.ndarray:
    k = x.shape[1]
    penalty = (ridge * (x * x).mean() + 1e-9) * np.eye(k)        # a column that never departs from the line gets weight 0
    w = np.linalg.solve(x.T @ x / len(y) + penalty, x.T @ y / len(y))
    if loss == "l1":                                   # iteratively reweighted least squares for the absolute error
        for _ in range(25):
            r = np.maximum(np.abs(y - x @ w), 0.05 * np.abs(y).mean() + 1e-9)
            xw = x / r[:, None]
            w = np.linalg.solve(xw.T @ x / len(y) + penalty * (1 / r).mean(), xw.T @ y / len(y))
    return w


def fit_weights(d: pd.DataFrame, own: str | None, ratios: tuple, extra: tuple = (), ridge: float = 1e-3,
                loss: str = "l2", min_rows: int = 200, relative: bool = False) -> dict:
    """Weights of (candidate - base) for one camera, one set per usable subset of partner fills.

    own    column of the own-level candidate (or None), always available,
    ratios columns of the partner ratio fills, each may be NaN,
    extra  further always-available columns (e.g. the accepted value).
    Returns {subset of ratio columns: dict(columns=[...], w=array, n=rows)}; the empty subset is the no-partner model.
    """
    base = d.base.to_numpy(float)
    y = d.truth.to_numpy(float) - base
    scale = np.maximum(base, MIN_LEVEL) if relative else np.ones(len(d))
    always = [c for c in ((own,) if own else ()) + tuple(extra)]
    models = {}
    for r in range(len(ratios), -1, -1):
        for subset in itertools.combinations(ratios, r):
            columns = always + list(subset)
            if not columns:
                models[subset] = dict(columns=[], w=np.zeros(0), n=int(len(d)))
                continue
            x = d[columns].to_numpy(float) - base[:, None]
            ok = np.isfinite(x).all(axis=1)
            if ok.sum() < min_rows:
                continue
            w = _solve(x[ok] / scale[ok, None], y[ok] / scale[ok], ridge, loss)
            models[subset] = dict(columns=columns, w=w, n=int(ok.sum()))
    return models


def apply_weights(d: pd.DataFrame, models: dict, ratios: tuple, max_rel: float | None = None) -> np.ndarray:
    """Value per row: the model of the largest usable partner subset. NaN when no model applies."""
    base = d.base.to_numpy(float)
    out = np.full(len(d), np.nan)
    done = np.zeros(len(d), bool)
    for subset in sorted(models, key=len, reverse=True):
        m = models[subset]
        if not m["columns"]:
            out[~done] = base[~done]
            done[:] = True
            continue
        x = d[m["columns"]].to_numpy(float) - base[:, None]
        ok = ~done & np.isfinite(x).all(axis=1)
        departure = x[ok] @ m["w"]
        if max_rel is not None:
            limit = max_rel * np.maximum(base[ok], MIN_LEVEL)
            departure = np.clip(departure, -limit, limit)
        out[ok] = np.maximum(0.0, base[ok] + departure)
        done |= ok
    return out


def fit_spec(d: pd.DataFrame, spec: dict, ridge: float = 1e-3, default_loss: str = "l2") -> dict:
    """{camera: weights} for every camera of `spec` that has pseudo-gap rows in d (see miss_settings.py for the keys)."""
    models = {}
    for unit, s in spec.items():
        rows = d[d.unit.isin(s["pool"])] if s.get("pool") else d[d.unit.eq(unit)]
        columns = [c for c in ((s.get("own"),) + tuple(s.get("ratios", ())) + tuple(s.get("extra", ()))) if c]
        if len(rows) < 200 or any(c not in rows for c in columns):
            continue
        models[unit] = fit_weights(rows, s.get("own"), tuple(s.get("ratios", ())), tuple(s.get("extra", ())), ridge=ridge,
                                   loss=s.get("loss", default_loss), relative=bool(s.get("relative", False)))
    return models


def apply_spec(d: pd.DataFrame, models: dict, spec: dict, max_rel: float | None = None) -> np.ndarray:
    """Value per row; NaN for cameras without fitted weights."""
    out = np.full(len(d), np.nan)
    for unit, g in d.groupby("unit"):
        if unit in models:
            out[d.index.get_indexer(g.index)] = apply_weights(g, models[unit], tuple(spec[unit].get("ratios", ())), max_rel=max_rel)
    return out


def blend_with_accepted(units, value: np.ndarray, accepted: np.ndarray, shrink: dict) -> np.ndarray:
    """accepted + g * (value - accepted) with g per camera (1 when the camera is not named); NaN stays NaN."""
    g = np.array([shrink.get(u, 1.0) for u in units])
    return np.asarray(accepted, float) + g * (np.asarray(value, float) - np.asarray(accepted, float))
