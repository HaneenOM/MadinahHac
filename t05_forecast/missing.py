"""Direct filler for drop-outs (the "missing" kind).

Reference: a straight line between the camera's readings on both sides of the gap.
Here that line is bent by how the other cameras moved away from their own straight
lines during the same minutes (in log scale). The weights are fitted, for each gap,
on pseudo-gaps of the same length cut from other recordings, using exactly the
cameras that are available during that gap.
"""
from __future__ import annotations

import numpy as np

from .core import MISSING, Recording


def gap_runs(rec: Recording) -> list:
    """(col, first_row, last_row, left_row, right_row) for every drop-out with readings on both sides."""
    runs = []
    for c in range(len(rec.units)):
        rows = np.nonzero(rec.status[:, c] == MISSING)[0]
        if rows.size == 0:
            continue
        known = np.nonzero(np.isfinite(rec.y[:, c]))[0]
        splits = np.nonzero(np.diff(rows) > 1)[0] + 1
        for seg in np.split(rows, splits):
            left, right = known[known < seg[0]], known[known > seg[-1]]
            if left.size and right.size:
                runs.append((c, int(seg[0]), int(seg[-1]), int(left[-1]), int(right[0])))
    return runs


def _pseudo_rows(z: np.ndarray, c: int, donors: list, span: int) -> tuple:
    """Training rows from one source table: target deviation and donor deviations for every placement."""
    n = z.shape[0]
    xs, ys = [], []
    inner = np.arange(1, span)
    w = inner / span
    for a in range(0, n - span):
        b = a + span
        if not (np.isfinite(z[a, c]) and np.isfinite(z[b, c])):
            continue
        ends = z[[a, b]][:, donors]
        if not np.isfinite(ends).all():
            continue
        t = a + inner
        y = z[t, c] - (z[a, c] * (1 - w) + z[b, c] * w)
        x = z[t][:, donors] - (ends[0][None, :] * (1 - w[:, None]) + ends[1][None, :] * w[:, None])
        ok = np.isfinite(y)
        if ok.any():
            xs.append(np.where(np.isfinite(x[ok]), x[ok], 0.0))
            ys.append(y[ok])
    if not xs:
        return np.zeros((0, len(donors))), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys)


def fill_missing_direct(rec: Recording, sources: list, ridge: float = 0.02, min_rows: int = 150,
                        max_span: int = 40) -> np.ndarray:
    """[T, U] predictions for drop-out cells (NaN where the method cannot be applied)."""
    out = np.full(rec.y.shape, np.nan)
    z = np.log1p(rec.y)
    src_tables = []
    for s in sources:
        zs = np.log1p(s.y)
        src_tables.append((s, zs))
    for c, first, last, a, b in gap_runs(rec):
        span = b - a
        if span > max_span:
            continue
        unit = rec.units[c]
        rows = np.arange(first, last + 1)
        w = (rows - a) / span
        loglin = z[a, c] * (1 - w) + z[b, c] * w
        cand = [d for d in range(len(rec.units)) if d != c and np.isfinite(z[a, d]) and np.isfinite(z[b, d])
                and np.isfinite(z[rows, d]).mean() >= 0.5]
        if not cand:
            continue
        use = [(s, zs) for s, zs in src_tables if unit in s.units and np.isfinite(zs[:, s.col(unit)]).sum() >= 60]
        if not use:
            continue
        # donors must exist (with data) in every source used
        donors = [d for d in cand if all(rec.units[d] in s.units and np.isfinite(zs[:, s.col(rec.units[d])]).sum() >= 60 for s, zs in use)]
        if not donors:
            continue
        xs, ys = [], []
        for s, zs in use:
            x, y = _pseudo_rows(zs, s.col(unit), [s.col(rec.units[d]) for d in donors], span)
            xs.append(x)
            ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
        if len(y) < min_rows:
            continue
        coef = np.linalg.solve(x.T @ x + ridge * len(y) * max(x.var(), 1e-6) * np.eye(len(donors)), x.T @ y)
        ends = z[[a, b]][:, donors]
        feat = z[rows][:, donors] - (ends[0][None, :] * (1 - w[:, None]) + ends[1][None, :] * w[:, None])
        feat = np.where(np.isfinite(feat), feat, 0.0)
        out[rows, c] = np.maximum(np.expm1(loglin + feat @ coef), 0.0)
    return out
