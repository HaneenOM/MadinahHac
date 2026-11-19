"""Building blocks of the missing track: trust-aware smoothers, candidate fills and pseudo-gaps that copy
the real release pattern of the test recordings.

Release pattern found in data/raw/test.csv (describe_gaps.py, show_strips.py): each test recording is released
in windows of roughly 20 to 30 minutes; a window is followed by the forecast targets and then by nothing.
A camera's outage sits inside a window, so the readings on its left are limited by the window start
(1 to 14 readings, the same limit for every camera) while the right side has 7 or more readings.

Candidate fills for a target minute m inside a gap bounded by readings at a (left) and b (right):
  base    straight line between the readings at a and b (the organizers' reference),
  own     a random-walk + reading-noise smoother of the camera's own log counts (kriging with a linear variogram
          and a nugget `lam` = reading-noise variance / random-walk variance per minute). lam = 0 is the straight
          line in log scale; a larger lam averages more readings on each side. A side with one reading
          automatically receives less weight than a side with many,
  spl     the same with an integrated random walk (a smoothing spline): it continues the trend of each side,
  ratio   partner reading at m times the target/partner ratio, where the log ratio is smoothed across the gap
          by the random-walk smoother on the minutes where both cameras are visible,
  legacy  the ratio fill of src/noran_missing.py (ratio of sums over 4 minutes on each side, straight line between),
  dev     departure of another camera from its own straight line between a and b (the input of the accepted method).
Nothing here reads a value of the target camera inside its gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 12                      # readings further than this from the gap are ignored
LAM_OWN = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
LAM_SPL = (3.0, 30.0, 300.0, 3000.0, 30000.0)
LAM_RATIO = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
LEGACY_SIDE = 4
PARTNERS = {"cam_A": ("cam_E", "cam_F"), "cam_E": ("cam_A", "cam_F"), "cam_F": ("cam_A", "cam_E"),
            "cam_B": ("cam_G",), "cam_G": ("cam_B",),
            # cameras that exist only on the test days: strongest same-minute pairs in the released test rows
            "cam_J": ("cam_K",), "cam_K": ("cam_J",), "cam_M": ("cam_K",), "cam_I": ("cam_L",), "cam_L": ("cam_I",)}

_WEIGHTS: dict = {}


def _cached(key, build):
    hit = _WEIGHTS.get(key)
    if hit is None:
        if len(_WEIGHTS) > 500000:
            _WEIGHTS.clear()
        hit = _WEIGHTS[key] = build()
    return hit


def krige_weights(obs: np.ndarray, query: np.ndarray, lam: float) -> np.ndarray:
    """[len(query), len(obs)] weights of the best linear estimate of the level of a random walk observed with
    noise (flat prior on the level). Each row sums to 1."""
    obs = np.asarray(obs, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)

    def build():
        n = len(obs)
        if n == 1:
            return np.ones((len(query), 1))
        system = np.zeros((n + 1, n + 1))
        system[:n, :n] = -0.5 * np.abs(obs[:, None] - obs[None, :]) + (lam + 1e-9) * np.eye(n)
        system[:n, n] = 1.0
        system[n, :n] = 1.0
        rhs = np.vstack([-0.5 * np.abs(obs[:, None] - query[None, :]), np.ones((1, len(query)))])
        return np.linalg.solve(system, rhs)[:n].T

    return _cached(("rw", obs.tobytes(), query.tobytes(), float(lam)), build)


def spline_weights(obs: np.ndarray, query: np.ndarray, lam: float) -> np.ndarray:
    """Same for an integrated random walk observed with noise (flat prior on level and slope): a smoothing spline.
    Falls back to the random-walk weights when fewer than 3 readings exist."""
    obs = np.asarray(obs, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    n = len(obs)
    if n < 3:
        return krige_weights(obs, query, 0.0)

    def build():
        t, q = obs.astype(float), query.astype(float)
        system = np.zeros((n + 2, n + 2))
        system[:n, :n] = np.abs(t[:, None] - t[None, :]) ** 3 / 12.0 + (lam + 1e-9) * np.eye(n)
        system[:n, n] = 1.0
        system[n, :n] = 1.0
        system[:n, n + 1] = t
        system[n + 1, :n] = t
        rhs = np.vstack([np.abs(t[:, None] - q[None, :]) ** 3 / 12.0, np.ones((1, len(q))), q[None, :]])
        return np.linalg.solve(system, rhs)[:n].T

    return _cached(("irw", obs.tobytes(), query.tobytes(), float(lam)), build)


def value_at(x: np.ndarray, m: int, tolerance: int = 2) -> float:
    """Reading at position m, or a short interpolation when readings exist within `tolerance` on both sides
    (the same rule as interp_at in src/v2_missing.py)."""
    if m < 0 or m >= len(x):
        return np.nan
    if np.isfinite(x[m]):
        return float(x[m])
    lo = m - 1
    while lo >= 0 and m - lo <= tolerance and not np.isfinite(x[lo]):
        lo -= 1
    hi = m + 1
    while hi < len(x) and hi - m <= tolerance and not np.isfinite(x[hi]):
        hi += 1
    if lo < 0 or hi >= len(x) or m - lo > tolerance or hi - m > tolerance:
        return np.nan
    f = (m - lo) / (hi - lo)
    return float(x[lo] * (1 - f) + x[hi] * f)


def _side_ratio(xu: np.ndarray, xp: np.ndarray, lo: int, hi: int) -> float:
    lo, hi = max(lo, 0), min(hi, len(xu) - 1)
    if hi < lo:
        return np.nan
    xs, ps = xu[lo:hi + 1], xp[lo:hi + 1]
    ok = np.isfinite(xs) & np.isfinite(ps)
    if not ok.any() or ps[ok].sum() <= 0:
        return np.nan
    return float(xs[ok].sum() / ps[ok].sum())


def gap_candidates(x: dict, unit: str, partners: tuple, a: int, b: int, ms: np.ndarray,
                   lam_own: tuple = LAM_OWN, lam_spl: tuple = LAM_SPL, lam_ratio: tuple = LAM_RATIO,
                   window: int = WINDOW, legacy: bool = False, donors: tuple = (), shifts: tuple = (),
                   lam_shift: float = 4.0) -> dict:
    """Candidate fills (in people) for positions `ms` strictly between a and b.

    x : {camera: array of visible readings on a common minute axis, NaN where nothing is visible}.
    Returns {'base', ('own', lam), ('spl', lam), ('ratio', partner, lam), ('legacy', partner), ('dev', donor),
             ('n', partner) -> (common readings left, right)}.
    """
    xu = x[unit]
    ms = np.asarray(ms, dtype=np.int64)
    lo, hi = max(0, a - window), min(len(xu) - 1, b + window)
    idx = np.arange(lo, hi + 1)
    ok = np.isfinite(xu[idx]) & ((idx <= a) | (idx >= b))
    obs = idx[ok]
    zu = np.log1p(xu[obs])
    f = (ms - a) / (b - a)
    out = {"base": xu[a] * (1 - f) + xu[b] * f, "n_left": int((obs <= a).sum()), "n_right": int((obs >= b).sum())}
    for lam in lam_own:
        out[("own", lam)] = np.expm1(krige_weights(obs - a, ms - a, lam) @ zu)
    for lam in lam_spl:
        out[("spl", lam)] = np.expm1(np.clip(spline_weights(obs - a, ms - a, lam) @ zu, 0.0, 12.0))
    for p in partners:
        xp = x.get(p)
        if xp is None:
            continue
        common = obs[np.isfinite(xp[obs])]
        pm = np.array([value_at(xp, int(m)) for m in ms])
        out[("n", p)] = (int((common <= a).sum()), int((common >= b).sum()))
        if len(common) == 0 or not np.isfinite(pm).any():
            continue
        d = np.log1p(xu[common]) - np.log1p(xp[common])
        for lam in lam_ratio:
            out[("ratio", p, lam)] = np.expm1(np.log1p(pm) + krige_weights(common - a, ms - a, lam) @ d)
        for k in shifts:                                 # the partner read k minutes later (k > 0) or earlier (k < 0)
            xs = np.full(len(xp), np.nan)
            if k > 0:
                xs[:-k] = xp[k:]
            else:
                xs[-k:] = xp[:k]
            common_k = obs[np.isfinite(xs[obs])]
            pk = np.array([value_at(xs, int(m)) for m in ms])
            if len(common_k) == 0 or not np.isfinite(pk).any():
                continue
            dk = np.log1p(xu[common_k]) - np.log1p(xs[common_k])
            out[("ratio", f"{p}{k:+d}", lam_shift)] = np.expm1(np.log1p(pk) + krige_weights(common_k - a, ms - a, lam_shift) @ dk)
        if legacy:
            hidden = xu.copy()
            hidden[a + 1:b] = np.nan
            left = _side_ratio(hidden, xp, a - LEGACY_SIDE + 1, a)
            right = _side_ratio(hidden, xp, b, b + LEGACY_SIDE - 1)
            out[("legacy", p)] = pm * (left * (1 - f) + right * f)
    for donor in donors:
        xd = x.get(donor)
        if xd is None or donor == unit:
            continue
        da, db = value_at(xd, a), value_at(xd, b)
        if np.isfinite(da) and np.isfinite(db):
            out[("dev", donor)] = np.array([value_at(xd, int(m)) for m in ms]) - (da * (1 - f) + db * f)
    return out


# ------------------------------------------------------------------ release pattern of the test recordings

def release_windows(test: pd.DataFrame) -> list:
    """Every released window of the test recordings: which cameras are released / are missing targets, per minute."""
    windows = []
    for rec, g in test.groupby("recording"):
        lo, hi = int(g.minute_of_day.min()), int(g.minute_of_day.max())
        units = sorted(g.unit.unique())
        rel = {u: np.zeros(hi - lo + 1, bool) for u in units}
        mis = {u: np.zeros(hi - lo + 1, bool) for u in units}
        released = g[g.id.isna() & g.people.notna()]
        for u, gg in released.groupby("unit"):
            rel[u][gg.minute_of_day.to_numpy().astype(int) - lo] = True
        for u, gg in g[g.id.notna() & g.kind.eq("missing")].groupby("unit"):
            mis[u][gg.minute_of_day.to_numpy().astype(int) - lo] = True
        any_rel = np.any(np.stack(list(rel.values())), axis=0)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], any_rel.astype(int), [0]])))
        for start, stop in zip(edges[::2], edges[1::2]):
            windows.append(dict(recording=rec, w0=lo + int(start), w1=lo + int(stop) - 1,
                                rel={u: rel[u][start:stop].copy() for u in units},
                                mis={u: mis[u][start:stop].copy() for u in units}))
    return windows


def dense(t: pd.DataFrame) -> tuple:
    """(first minute, {camera: array over every minute first..last, NaN where no reading})."""
    lo, hi = int(t.index.min()), int(t.index.max())
    full = t.reindex(range(lo, hi + 1))
    return lo, {u: full[u].to_numpy(float) for u in full.columns}


def gap_rows(visible: dict, truth: np.ndarray, unit: str, targets: np.ndarray, partners: tuple, meta: dict,
             legacy: bool = True, donors: tuple = (), **grids) -> list:
    """Rows (one per target minute with a known truth) for the gaps that contain `targets` in `visible[unit]`."""
    xu = visible[unit]
    known = np.flatnonzero(np.isfinite(xu))
    rows = []
    if len(known) < 2:
        return rows
    pos = np.searchsorted(known, targets)
    inside = (pos > 0) & (pos < len(known)) & np.isfinite(truth[targets])
    inside[inside] &= ~np.isfinite(xu[targets[inside]])
    pairs = sorted(set(zip(known[pos[inside] - 1].tolist(), known[pos[inside]].tolist())))
    for a, b in pairs:
        if b - a > 30:
            continue
        ms = targets[inside & (targets > a) & (targets < b)]
        cand = gap_candidates(visible, unit, partners, a, b, ms, legacy=legacy, donors=donors, **grids)
        for j, m in enumerate(ms):
            row = dict(meta, unit=unit, a=a, b=b, m=int(m), span=b - a - 1, n_targets=len(ms),
                       pos=int(min(m - a, b - m)), n_left=cand["n_left"], n_right=cand["n_right"],
                       truth=float(truth[m]), base=float(cand["base"][j]))
            for key, value in cand.items():
                if not isinstance(key, tuple):
                    continue
                if key[0] == "own":
                    row[f"own_{key[1]:g}"] = float(value[j])
                elif key[0] == "spl":
                    row[f"spl_{key[1]:g}"] = float(value[j])
                elif key[0] == "ratio":
                    row[f"r{key[1][4:]}_{key[2]:g}"] = float(value[j])          # rE_4, or rE+1_4 for a shifted partner
                elif key[0] == "legacy":
                    row[f"lg_{key[1][-1]}"] = float(value[j])
                elif key[0] == "dev":
                    row[f"dv_{key[1][-1]}"] = float(value[j])
                elif key[0] == "n":
                    row[f"cl_{key[1][-1]}"], row[f"cr_{key[1][-1]}"] = value
            rows.append(row)
    return rows


def transplant_cases(source: pd.DataFrame, name: str, windows: list, step: int = 3, units: tuple | None = None,
                     partners: dict = PARTNERS, skip_near: int = 0, only_shift: int | None = None,
                     legacy: bool = True, with_donors: bool = True, **grids) -> pd.DataFrame:
    """Pseudo-gaps on a fully released table: every test window (its release and outage masks of all cameras)
    is laid over the table at every `step`-th shift. skip_near > 0 leaves out shifts closer than that to 0
    (the unshifted placement is the team's local case set)."""
    lo, x = dense(source)
    n = len(next(iter(x.values())))
    rows = []
    for w in windows:
        width = w["w1"] - w["w0"] + 1
        targets_of = {u: np.flatnonzero(m) for u, m in w["mis"].items() if m.any() and u in x and (units is None or u in units)}
        if not targets_of or width > n:
            continue
        for start in range((w["w0"] - lo) % step, n - width + 1, step):       # shift 0 stays on the grid
            shift = lo + start - w["w0"]
            if (skip_near and abs(shift) < skip_near) or (only_shift is not None and shift != only_shift):
                continue
            visible = {u: np.where(w["rel"][u], x[u][start:start + width], np.nan) for u in x if u in w["rel"]}
            for u, tg in targets_of.items():
                meta = dict(source=name, window=f"{w['recording'][4:8]}_{w['w0']}", shift=shift, minute0=lo + start)
                rows += gap_rows(visible, x[u][start:start + width], u, tg, partners.get(u, ()), meta, legacy=legacy,
                                 donors=tuple(visible) if with_donors else (), **grids)
    return pd.DataFrame(rows)


def outage_patterns(windows: list) -> list:
    """(readings of the camera before the outage inside its window, outage span) of every real outage."""
    found = set()
    for w in windows:
        for u, mis in w["mis"].items():
            if not mis.any():
                continue
            vis = np.flatnonzero(w["rel"][u])
            for m in np.flatnonzero(mis):
                left, right = vis[vis < m], vis[vis > m]
                if len(left) and len(right):
                    a, b = int(left[-1]), int(right[0])
                    run = 1
                    while a - run >= 0 and w["rel"][u][a - run]:
                        run += 1
                    found.add((min(run, 14), b - a - 1))
    return sorted(found)


def window_cases(windows: list, feed: dict, patterns: list, units: tuple | None = None, partners: dict = PARTNERS,
                 min_right: int = 4, step: int = 3, legacy: bool = True, with_donors: bool = True, **grids) -> pd.DataFrame:
    """Pseudo-gaps inside the released windows of the test recordings themselves.

    feed : {recording: minute x camera table of released readings}. A camera that is released without a break in a
    window receives pseudo-gaps with the (left readings, span) patterns of the real outages at every `step`-th
    position; everything earlier than `left` readings before the gap is blanked for all cameras (the real window
    start is common to all cameras). Otherwise the other cameras keep their real visibility.
    """
    rows = []
    for w in windows:
        t = feed[w["recording"]]
        full = t.reindex(range(w["w0"], w["w1"] + 1))
        x = {u: full[u].to_numpy(float) for u in full.columns}
        for u in x:
            if (units is not None and u not in units) or w["mis"].get(u, np.zeros(1, bool)).any():
                continue
            known = np.flatnonzero(np.isfinite(x[u]))
            if len(known) < 8 or (np.diff(known) != 1).any():
                continue
            first, last = int(known[0]), int(known[-1])
            for left, span in patterns:
                for offset in range(left, last - first + 1, step):
                    a = first + offset - 1
                    b = a + span + 1
                    if b + min_right - 1 > last:
                        break
                    visible = {v: xv.copy() for v, xv in x.items()}
                    for xv in visible.values():
                        xv[:a - left + 1] = np.nan                  # the window starts `left` readings before the gap
                    visible[u][a + 1:b] = np.nan
                    meta = dict(source=w["recording"], window=f"{w['recording'][4:8]}_{w['w0']}", shift=offset - left,
                                minute0=w["w0"], pattern=f"{left}_{span}")
                    rows += gap_rows(visible, x[u], u, np.arange(a + 1, b), partners.get(u, ()), meta, legacy=legacy,
                                     donors=tuple(visible) if with_donors else (), **grids)
    return pd.DataFrame(rows)


def norm_score(d: pd.DataFrame, column: str) -> float:
    """Summed absolute error of `column` / summed absolute error of the straight line."""
    return float((d[column] - d.truth).abs().sum() / (d.base - d.truth).abs().sum())
