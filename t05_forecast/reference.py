"""The organizers' reference predictions (the sample submission, score 1.0), rebuilt.

forecast       last released value of the camera, at most 5 minutes back from `now`
missing        straight line between the readings on both sides of the gap
hidden         one visible camera, scaled; the camera and the scale are fixed on the training days
hidden-future  the hidden reference's value at `now`

They are needed to score local validation the same way the leaderboard does.
"""
from __future__ import annotations

import numpy as np

from .core import FORECAST, HIDDEN, HIDDEN_FUTURE, MISSING, RELEASED, Recording


def last_value(y: np.ndarray, back: int = 5) -> np.ndarray:
    """[T, U] most recent finite value within `back` minutes (inclusive of the current minute)."""
    out = np.full(y.shape, np.nan)
    for j in range(back, -1, -1):
        sh = np.full(y.shape, np.nan)
        sh[j:] = y[: y.shape[0] - j]
        out = np.where(np.isfinite(sh), sh, out)
    return out


def ref_forecast(rec: Recording) -> np.ndarray:
    out = np.full(rec.y.shape, np.nan)
    last = last_value(rec.y, 5)
    for r, c in zip(*np.nonzero(rec.status == FORECAST)):
        out[r, c] = last[rec.now[r, c] - rec.first, c]
    return out


def ref_missing(rec: Recording) -> np.ndarray:
    out = np.full(rec.y.shape, np.nan)
    n = rec.n_minutes
    for c in range(len(rec.units)):
        rows = np.nonzero(rec.status[:, c] == MISSING)[0]
        if rows.size == 0:
            continue
        known = np.nonzero(np.isfinite(rec.y[:, c]))[0]
        for r in rows:
            left = known[known < r]
            right = known[known > r]
            if left.size and right.size:
                a, b = left[-1], right[0]
                w = (r - a) / (b - a)
                out[r, c] = rec.y[a, c] * (1 - w) + rec.y[b, c] * w
            elif left.size:
                out[r, c] = rec.y[left[-1], c]
            elif right.size:
                out[r, c] = rec.y[right[0], c]
    return out


def donor_table(train_recs: list, units: list) -> dict:
    """For every ordered camera pair (target, donor): the scale and the fit on the training days.

    scale = mean(target) / mean(donor) over the minutes where both were recorded;
    fit = mean absolute error of scale * donor against the target on those minutes.
    """
    table = {}
    for tgt in units:
        for don in units:
            if tgt == don:
                continue
            xs, ys = [], []
            for rec in train_recs:
                if tgt in rec.units and don in rec.units:
                    a, b = rec.y[:, rec.col(tgt)], rec.y[:, rec.col(don)]
                    ok = np.isfinite(a) & np.isfinite(b)
                    xs.append(b[ok]); ys.append(a[ok])
            if not xs or sum(len(x) for x in xs) < 30:
                continue
            x, y = np.concatenate(xs), np.concatenate(ys)
            scale = y.mean() / x.mean()
            table[(tgt, don)] = {"scale": float(scale), "mae": float(np.abs(y - scale * x).mean()), "n": int(len(x))}
    return table


def donor_ranking(table: dict, target: str, visible: list) -> list:
    """Visible donors for a hidden camera, best fit on the training days first."""
    cands = [(table[(target, d)]["mae"], d) for d in visible if (target, d) in table]
    return [d for _, d in sorted(cands)]


def ref_hidden(rec: Recording, table: dict, ranking: dict | None = None) -> np.ndarray:
    """Hidden and hidden-future references for every hidden camera of the recording."""
    out = np.full(rec.y.shape, np.nan)
    hidden_cols = sorted({c for _, c in zip(*np.nonzero((rec.status == HIDDEN) | (rec.status == HIDDEN_FUTURE)))})
    visible = [u for i, u in enumerate(rec.units) if i not in hidden_cols and (rec.status[:, i] == RELEASED).any()]
    for c in hidden_cols:
        tgt = rec.units[c]
        order = (ranking or {}).get(tgt) or donor_ranking(table, tgt, visible)
        scaled = np.full(rec.n_minutes, np.nan)          # scaled donor value at each minute, first available donor
        for d in reversed(order):
            v = rec.y[:, rec.col(d)] * table[(tgt, d)]["scale"]
            scaled = np.where(np.isfinite(v), v, scaled)
        for r in np.nonzero(rec.status[:, c] == HIDDEN)[0]:
            out[r, c] = scaled[r]
        held = last_value(scaled[:, None], 5)[:, 0]     # value at `now` (most recent within 5 minutes)
        for r in np.nonzero(rec.status[:, c] == HIDDEN_FUTURE)[0]:
            out[r, c] = held[rec.now[r, c] - rec.first]
    return out


def reference(rec: Recording, table: dict, ranking: dict | None = None) -> np.ndarray:
    """All four references in one [T, U] array (NaN outside holes)."""
    out = ref_forecast(rec)
    m = ref_missing(rec)
    h = ref_hidden(rec, table, ranking)
    out = np.where(np.isfinite(out), out, m)
    out = np.where(np.isfinite(out), out, h)
    return out
