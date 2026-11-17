"""Weekly template: the same camera on the same weekday in other weeks.

The template for a target recording is a smoothed log curve per camera, read at
the clock time that sits at the same distance from the prayers (see prayer.py).
It is a fitted profile of other days, not an input taken from the target day.
"""
from __future__ import annotations

import numpy as np

from .core import Recording
from .prayer import warp_minutes


def smooth_log(y: np.ndarray, lam: float, reach: int = 2, max_gap: int = 20) -> np.ndarray:
    """Two-sided exponentially weighted mean of log1p(y) along time, per column.

    lam = 0 keeps the raw values; larger lam smooths more (weight lam^|k| at k minutes).
    A minute is filled only if a reading exists within `reach` minutes; remaining
    interior gaps of up to `max_gap` minutes are bridged by a straight line.
    """
    z = np.log1p(y)
    n, u = z.shape
    width = 0 if lam <= 0 else int(np.ceil(np.log(0.01) / np.log(lam)))
    fin = np.isfinite(z)
    z0 = np.where(fin, z, 0.0)
    num, den, near = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)
    for k in range(-width, width + 1):
        wk = 1.0 if k == 0 else lam ** abs(k)
        src = slice(max(0, k), n + min(0, k))
        dst = slice(max(0, -k), n - max(0, k))
        num[dst] += wk * z0[src]
        den[dst] += wk * fin[src]
        if abs(k) <= reach:
            near[dst] += fin[src]
    out = np.where((den > 0) & (near > 0), num / np.maximum(den, 1e-12), np.nan)
    for j in range(u):
        col = out[:, j]
        idx = np.nonzero(np.isfinite(col))[0]
        if idx.size < 2:
            continue
        for g in np.nonzero(np.diff(idx) > 1)[0]:
            a, b = idx[g], idx[g + 1]
            if b - a - 1 <= max_gap:
                col[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [col[a], col[b]])
    return out


def sample_at(curve: np.ndarray, first: int, minutes: np.ndarray) -> np.ndarray:
    """Read a per-minute curve at fractional clock minutes (straight line between minutes)."""
    pos = np.asarray(minutes, dtype=float) - first
    lo = np.floor(pos + 1e-9).astype(int)
    frac = np.clip(pos - lo, 0.0, 1.0)
    out = np.full(len(pos), np.nan)
    inside = (lo >= 0) & (lo < len(curve))
    hi = np.minimum(lo + 1, len(curve) - 1)
    a = np.where(inside, curve[np.clip(lo, 0, len(curve) - 1)], np.nan)
    b = np.where(inside, curve[np.clip(hi, 0, len(curve) - 1)], np.nan)
    exact = frac < 1e-6
    out = np.where(exact, a, a * (1 - frac) + b * frac)
    out[~inside] = np.nan
    return out


def build_template(target: Recording, sources: list, lam: float = 0.5, use_warp: bool = True,
                   align_levels: bool = True) -> np.ndarray:
    """[T, U] log-space template for the target recording from other recordings.

    Sources are averaged where several cover the same minute. With `align_levels`
    every further source is first moved to the level of the source that covers the
    most minutes (per camera), so the template does not jump where coverage changes.
    """
    curves = []
    for s in sources:
        if s.name == target.name:
            continue
        zs = smooth_log(s.y, lam)
        src_minutes = warp_minutes(target.minutes, target.day, s.day) if use_warp else target.minutes.astype(float)
        # the shift can push the first or last few minutes just outside the source day: hold the edge value there
        lo, hi = float(s.first), float(s.first + s.n_minutes - 1)
        near = (src_minutes >= lo - 10) & (src_minutes <= hi + 10)
        src_minutes = np.where(near, np.clip(src_minutes, lo, hi), src_minutes)
        arr = np.full(target.y.shape, np.nan)
        for j, unit in enumerate(target.units):
            if unit in s.units:
                arr[:, j] = sample_at(zs[:, s.col(unit)], s.first, src_minutes)
        curves.append(arr)
    if not curves:
        return np.full(target.y.shape, np.nan)
    stack = np.stack(curves)                                   # [S, T, U]
    if align_levels and len(curves) > 1:
        cover = np.isfinite(stack).sum(axis=1)                 # [S, U]
        for j in range(stack.shape[2]):
            main = int(np.argmax(cover[:, j]))
            for s in range(stack.shape[0]):
                if s == main:
                    continue
                both = np.isfinite(stack[s, :, j]) & np.isfinite(stack[main, :, j])
                if both.sum() >= 20:
                    stack[s, :, j] += np.mean(stack[main, both, j] - stack[s, both, j])
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(stack).sum(axis=0)
        tmpl = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
    return tmpl


def complete_template(target: Recording, tmpl: np.ndarray, rows_from: list, lam: float = 0.8, alpha: float = 0.05,
                      min_rows: int = 50) -> tuple:
    """Fill template cells that have no earlier-week curve, using the other cameras' templates.

    For such a camera a ridge regression of its smoothed log level on the other cameras'
    smoothed log levels is fitted on the recordings in `rows_from`. The camera's template is
    that regression applied to the other cameras' templates. Returns (template, report).
    """
    out = tmpl.copy()
    smoothed = [(r, smooth_log(r.y, lam, max_gap=0)) for r in rows_from if r.name != target.name]
    report = {}
    for j, unit in enumerate(target.units):
        need = ~np.isfinite(tmpl[:, j])
        if not need.any():
            continue
        use = [(r, z) for r, z in smoothed if unit in r.units and np.isfinite(z[:, r.col(unit)]).sum() >= 30]
        if not use:
            continue
        donors = []
        for i, d in enumerate(target.units):
            if d == unit or np.isfinite(tmpl[need, i]).mean() < 0.5:
                continue
            if all(d in r.units and np.isfinite(z[:, r.col(d)]).sum() >= 30 for r, z in use):
                donors.append(d)
        if not donors:
            continue
        xs, ys = [], []
        for r, z in use:
            cols = [r.col(d) for d in donors]
            ok = np.isfinite(z[:, r.col(unit)]) & np.isfinite(z[:, cols]).all(axis=1)
            xs.append(z[np.ix_(ok, cols)])
            ys.append(z[ok, r.col(unit)])
        x, y = np.concatenate(xs), np.concatenate(ys)
        if len(y) < min_rows:
            continue
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-9
        xn, y0 = (x - mu) / sd, y.mean()
        coef = np.linalg.solve(xn.T @ xn + alpha * len(y) * np.eye(len(donors)), xn.T @ (y - y0))
        dcols = [target.col(d) for d in donors]
        td = tmpl[:, dcols]
        rows = need & np.isfinite(td).all(axis=1)
        out[rows, j] = y0 + ((td[rows] - mu) / sd) @ coef
        report[unit] = {"donors": "".join(d[-1] for d in donors), "rows": int(len(y)),
                        "r2": float(1 - np.var(y - y0 - xn @ coef) / np.var(y)), "filled": int(rows.sum())}
    return out, report


def slope(template: np.ndarray, half: int = 2) -> np.ndarray:
    """Per-minute slope of the template (log units per minute), by central difference."""
    out = np.full(template.shape, np.nan)
    n = template.shape[0]
    out[half:n - half] = (template[2 * half:] - template[: n - 2 * half]) / (2.0 * half)
    out = np.where(np.isfinite(out), out, 0.0)
    return out
