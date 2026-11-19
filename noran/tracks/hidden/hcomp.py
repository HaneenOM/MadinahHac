"""Components of the hidden-camera estimate. Imported by model.py and by the experiment scripts.

Nothing here reads answers. Inputs are: the fitting rows handed over by the harness (training dates other than
the held-out one), the released rows of the two test recordings (allowed by the host), the prayer timetable and,
at prediction time, the released feed of the recording WITHOUT the columns of its hidden cameras.

Members (all in log1p scale)
  jn      joint normal over [raw log hidden camera, smoothed log donors], fitted by EM so that rows with blank
          cameras count; the value is the conditional mean given the donors visible at that minute.
  carry   jn plus the smoothed residual that the same regression left on reference recordings (other days on
          which the hidden camera was seen) at the prayer-aligned minute: the time-of-day scale between the hidden
          camera and its donors, taken first from the same weekday.
  gbm     gradient boosting on smoothed donor logs relative to the crowd level, with a prayer-aligned clock.

Donors are smoothed over neighbouring minutes and, when lag_max > 0, read at the lead or lag (in minutes) that
correlates best with the hidden camera on the fitting recordings.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

A_H = [f"cam_{c}" for c in "ABCDEFGH"]
PATTERNS = {"cam_B": ("cam_B", "cam_G"), "cam_G": ("cam_B", "cam_G"), "cam_D": ("cam_D",)}
CANON = np.array([288, 738, 946, 1110, 1200], float)      # timetable of 11 Sep, the common clock for the boosting

DEFAULT = dict(
    half=6, decay=0.8, min_weight=0.5,                    # smoothing of the donors (minutes each side, weight decay^|k|)
    ref_half=7, ref_decay=0.85, ref_min_weight=1.5,       # smoothing of the residual on reference recordings
    lam=0.05,                                             # relative ridge on the donor covariance
    w0=0.5, w_same=1.0, w_cross=0.25,                     # weights: plain regression, same-weekday reference, other-weekday reference
    em_iters=60,
    gbm=dict(loss="absolute_error", max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=30,
             l2_regularization=1.0, early_stopping=False, random_state=11),
    use_test_rows=True,
    lag_max=0,                                            # donors read at their best lead/lag, found on the fitting recordings; 0 = off
    same_day_weight=1.0,                                  # boosting: weight of fitting rows on the target's weekday (1 = no preference)
    gbm_base="level",
    bags=1,                                               # boosting: number of block-bootstrap fits that are averaged (1 = one plain fit)
    bag_block=30,                                         # minutes per bootstrap block
    seed=11,
    edge=0,                                               # a reference recording is extended by this many minutes at both ends (its first/last residual is held)                                     # boosting target: raw - crowd level ('level') or raw - joint normal ('jn')
)


# ------------------------------------------------------------------ basic tools
def to_grid(t: pd.DataFrame) -> pd.DataFrame:
    """Minute x camera table on a gap-free minute index, cameras A to H only (absent cameras are all NaN)."""
    idx = np.arange(int(t.index.min()), int(t.index.max()) + 1)
    return t.reindex(index=idx, columns=A_H).astype(float)


def smooth(X: np.ndarray, half: int, decay: float, min_weight: float) -> np.ndarray:
    """Two-sided weighted mean over the readings that exist; NaN where too little support."""
    X = np.asarray(X, float)
    one = X.ndim == 1
    if one:
        X = X[:, None]
    k = decay ** np.abs(np.arange(-half, half + 1))
    ok = np.isfinite(X)
    filled = np.where(ok, X, 0.0)
    num = np.empty_like(filled)
    den = np.empty_like(filled)
    for j in range(X.shape[1]):
        num[:, j] = np.convolve(filled[:, j], k, mode="same")
        den[:, j] = np.convolve(ok[:, j].astype(float), k, mode="same")
    out = np.where(den >= min_weight, num / np.where(den > 0, den, 1.0), np.nan)
    return out[:, 0] if one else out


def shift_to(minutes: np.ndarray, today: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Map today's minutes onto another day's clock, anchored at the five prayers."""
    return np.rint(minutes + np.interp(minutes, today, np.asarray(other, float) - np.asarray(today, float))).astype(int)


def is_friday(day: str) -> bool:
    return pd.Timestamp(day).day_of_week == 4


def em_normal(Z: np.ndarray, iters: int = 60):
    """Mean and covariance of a normal fitted to rows with missing entries (EM)."""
    Z = np.asarray(Z, float)
    Z = Z[np.isfinite(Z).any(axis=1)]
    n, p = Z.shape
    mu = np.nanmean(Z, axis=0)
    sigma = np.diag(np.nanvar(Z, axis=0) + 1e-6)
    ok = np.isfinite(Z)
    codes = ok @ (1 << np.arange(p))
    groups = [(np.flatnonzero(codes == c)) for c in np.unique(codes)]
    for _ in range(iters):
        s1 = np.zeros(p)
        s2 = np.zeros((p, p))
        for rows in groups:
            o = np.flatnonzero(ok[rows[0]])
            m = np.flatnonzero(~ok[rows[0]])
            x = Z[rows].copy()
            if len(m):
                b = sigma[np.ix_(m, o)] @ np.linalg.inv(sigma[np.ix_(o, o)] + 1e-8 * np.eye(len(o)))
                x[:, m] = mu[m] + (x[:, o] - mu[o]) @ b.T
                c = sigma[np.ix_(m, m)] - b @ sigma[np.ix_(o, m)]
                s2[np.ix_(m, m)] += len(rows) * c
            s1 += x.sum(axis=0)
            s2 += x.T @ x
        mu = s1 / n
        sigma = s2 / n - np.outer(mu, mu)
        sigma = (sigma + sigma.T) / 2
    return mu, sigma


# ------------------------------------------------------------------ lead / lag of the donors
def estimate_lags(recs: list, h: int, donors: list, lag_max: int) -> np.ndarray:
    """For each donor, the shift (minutes, positive = donor read later) with the highest correlation to the hidden
    camera, on smoothed log counts minus a 61-minute rolling mean, pooled over the long training recordings."""
    lags = np.zeros(len(A_H), int)
    if lag_max <= 0:
        return lags
    shifts = np.arange(-lag_max, lag_max + 1)
    for d in donors:
        num = np.zeros(len(shifts))
        sa = np.zeros(len(shifts))
        sb = np.zeros(len(shifts))
        for r in recs:
            if r["test"] or len(r["minutes"]) < 120:
                continue
            hp = pd.DataFrame(r["smooth"][:, [h, d]])
            hp = (hp - hp.rolling(61, center=True, min_periods=20).mean()).to_numpy()
            for k, sft in enumerate(shifts):
                b = np.roll(hp[:, 1], -sft)
                if sft > 0:
                    b[-sft:] = np.nan
                if sft < 0:
                    b[:-sft] = np.nan
                ok = np.isfinite(hp[:, 0]) & np.isfinite(b)
                if ok.sum() < 60:
                    continue
                a0 = hp[ok, 0] - hp[ok, 0].mean()
                b0 = b[ok] - b[ok].mean()
                num[k] += a0 @ b0
                sa[k] += a0 @ a0
                sb[k] += b0 @ b0
        good = (sa > 0) & (sb > 0)
        if good.any():
            corr = np.where(good, num / np.sqrt(np.where(good, sa * sb, 1.0)), -np.inf)
            lags[d] = int(shifts[int(np.argmax(corr))])
    return lags


def shifted(S: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Each column read at minute + lag; where that minute has no support, the next smaller shift is used."""
    if not np.any(lags):
        return S
    out = np.full_like(S, np.nan)
    n = len(S)
    for j, lag in enumerate(lags):
        lag = int(lag)
        step = int(np.sign(lag))
        col = np.full(n, np.nan)
        for sft in ([0] if lag == 0 else list(range(lag, 0, -step)) + [0]):
            moved = np.full(n, np.nan)
            if sft > 0:
                moved[:n - sft] = S[sft:, j]
            elif sft < 0:
                moved[-sft:] = S[:n + sft, j]
            else:
                moved = S[:, j]
            col = np.where(np.isfinite(col), col, moved)
        out[:, j] = col
    return out


# ------------------------------------------------------------------ fitting
def _recordings(training: pd.DataFrame, context: dict, cfg: dict):
    """Every fitting recording as a dict: name, day, raw log grid (minute x A..H)."""
    frames = [training[["recording", "day", "unit", "minute_of_day", "people"]]]
    if cfg["use_test_rows"]:
        extra = context["test_released"]
        frames.append(extra[extra.unit.isin(A_H)][["recording", "day", "unit", "minute_of_day", "people"]])
    out = []
    for number, frame in enumerate(frames):
        for rec, g in frame.groupby("recording"):
            grid = to_grid(g.pivot(index="minute_of_day", columns="unit", values="people").sort_index())
            out.append(dict(name=rec, day=g.day.iloc[0], minutes=grid.index.to_numpy(), raw=np.log1p(grid.to_numpy()),
                            test=number == 1))
    return out


def fit_core(training: pd.DataFrame, day, context: dict, cfg: dict | None = None) -> dict:
    cfg = {**DEFAULT, **(cfg or {})}
    schedule = context["schedule"]
    recs = _recordings(training, context, cfg)
    for r in recs:
        r["smooth"] = smooth(r["raw"], cfg["half"], cfg["decay"], cfg["min_weight"])
        r["friday"] = is_friday(r["day"])
        r["prayers"] = np.asarray(schedule[r["day"]], float)
    units = {}
    for unit, hidden in PATTERNS.items():
        h = A_H.index(unit)
        donors = [i for i, c in enumerate(A_H) if c not in hidden]
        lags = estimate_lags(recs, h, donors, cfg["lag_max"])
        S_of = {r["name"]: shifted(r["smooth"], lags) for r in recs}
        Z = []
        for r in recs:
            block = np.column_stack([r["raw"][:, h], S_of[r["name"]][:, donors]])
            keep = np.isfinite(block[:, 1:]).sum(axis=1) >= 2
            Z.append(block[keep])
        mu, sigma = em_normal(np.vstack(Z), cfg["em_iters"])
        refs = [r for r in recs if np.isfinite(r["raw"][:, h]).sum() >= 20]
        # the weekday the estimate is needed for: the held-out date locally; on the real test days B and G are
        # hidden on the Friday and D on the Saturday
        target_friday = is_friday(day) if day is not None else unit != "cam_D"
        units[unit] = dict(h=h, donors=donors, mu=mu, sigma=sigma, refs=refs, coef={}, resid={}, lags=lags, S=S_of,
                           target_friday=target_friday)
        units[unit]["gbm"] = _fit_gbm(units[unit], recs, cfg)
    return dict(cfg=cfg, units=units, schedule=schedule, cache={}, held_out=day)


def _coef(info: dict, subset: tuple, lam: float):
    """Regression of the hidden camera on a subset of donors, read off the joint normal."""
    if subset not in info["coef"]:
        mu, sigma = info["mu"], info["sigma"]
        idx = [1 + info["donors"].index(i) for i in subset]
        sss = sigma[np.ix_(idx, idx)]
        beta = np.linalg.solve(sss + lam * np.diag(np.diag(sss)), sigma[0, idx])
        info["coef"][subset] = (mu[0] - beta @ mu[idx], beta)
    return info["coef"][subset]


def _regress(info: dict, S: np.ndarray, lam: float) -> np.ndarray:
    """Conditional mean for every row of a smoothed grid (all eight columns), by pattern of visible donors."""
    donors = info["donors"]
    ok = np.isfinite(S[:, donors])
    out = np.full(len(S), np.nan)
    codes = ok @ (1 << np.arange(len(donors)))
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        subset = tuple(d for d, o in zip(donors, ok[rows[0]]) if o)
        if not subset:
            out[rows] = info["mu"][0]
            continue
        a, beta = _coef(info, subset, lam)
        out[rows] = a + S[np.ix_(rows, list(subset))] @ beta
    return out


def _residual_profile(info: dict, ref: dict, subset: tuple, cfg: dict) -> np.ndarray:
    """Smoothed residual of the subset regression on a reference recording, per minute of that recording."""
    key = (ref["name"], subset)
    if key not in info["resid"]:
        a, beta = _coef(info, subset, cfg["lam"])
        fitted = a + info["S"][ref["name"]][:, list(subset)] @ beta    # NaN where a donor of the subset is absent
        res = ref["raw"][:, info["h"]] - fitted
        info["resid"][key] = smooth(res, cfg["ref_half"], cfg["ref_decay"], cfg["ref_min_weight"])
    return info["resid"][key]


def _level_features(S: np.ndarray, donors: list, mu_donors: np.ndarray, minutes: np.ndarray, prayers: np.ndarray, friday: bool):
    centred = S[:, donors] - mu_donors
    count = np.isfinite(centred).sum(axis=1)
    level = np.where(count > 0, np.nansum(centred, axis=1) / np.maximum(count, 1), np.nan)
    canon = minutes + np.interp(minutes, prayers, CANON - prayers)
    nearest = np.array([m - prayers[np.argmin(np.abs(prayers - m))] for m in minutes], float)
    X = np.column_stack([centred - level[:, None], level, canon, nearest, np.full(len(S), float(friday))])
    return X, level


def _fit_gbm(info: dict, recs: list, cfg: dict):
    donors, h = info["donors"], info["h"]
    mu_d = info["mu"][1:]
    X, y, w, blocks = [], [], [], []
    for number, r in enumerate(recs):
        S = info["S"][r["name"]]
        keep = np.isfinite(r["raw"][:, h]) & (np.isfinite(S[:, donors]).sum(axis=1) >= 2)
        if keep.sum() == 0:
            continue
        feats, level = _level_features(S, donors, mu_d, r["minutes"].astype(float), r["prayers"], r["friday"])
        base = _regress(info, S, cfg["lam"]) if cfg["gbm_base"] == "jn" else level
        X.append(feats[keep])
        y.append(r["raw"][keep, h] - base[keep])
        w.append(np.full(int(keep.sum()), cfg["same_day_weight"] if r["friday"] == info["target_friday"] else 1.0))
        blocks.append(number * 10000 + r["minutes"][keep] // cfg["bag_block"])
    if not X:
        return None
    X, y, w, blocks = np.vstack(X), np.concatenate(y), np.concatenate(w), np.concatenate(blocks)
    if cfg["bags"] <= 1:
        return [HistGradientBoostingRegressor(**cfg["gbm"]).fit(X, y, sample_weight=w)]
    rng = np.random.default_rng(cfg["seed"] + 1000 * h)
    ids, inverse = np.unique(blocks, return_inverse=True)
    models = []
    for b in range(cfg["bags"]):
        counts = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inverse].astype(float)
        use = counts > 0
        params = {**cfg["gbm"], "random_state": cfg["gbm"]["random_state"] + b}
        models.append(HistGradientBoostingRegressor(**params).fit(X[use], y[use], sample_weight=(w * counts)[use]))
    return models


# ------------------------------------------------------------------ prediction
def components(t: pd.DataFrame, unit: str, day: str, model: dict) -> pd.DataFrame:
    """All members for every minute of the feed (log1p scale). The hidden columns are dropped first."""
    cfg, info = model["cfg"], model["units"][unit]
    visible = t.drop(columns=[c for c in PATTERNS[unit] if c in t.columns])
    grid = to_grid(visible)
    raw = np.log1p(grid.to_numpy())
    key = (unit, day, hashlib.sha1(np.nan_to_num(raw, nan=-1.0).tobytes()).hexdigest(), int(grid.index[0]))
    if key in model["cache"]:
        return model["cache"][key]
    minutes = grid.index.to_numpy()
    donors = info["donors"]
    S = shifted(smooth(raw, cfg["half"], cfg["decay"], cfg["min_weight"]), info["lags"])
    prayers = np.asarray(model["schedule"][day], float)
    friday = is_friday(day)
    jn = _regress(info, S, cfg["lam"])
    ok = np.isfinite(S[:, donors])
    total = cfg["w0"] * jn
    weight = np.full(len(S), cfg["w0"])
    same_w = np.zeros(len(S))
    for ref in info["refs"]:
        w_ref = cfg["w_same"] if ref["friday"] == friday else cfg["w_cross"]
        if w_ref <= 0:
            continue
        there = shift_to(minutes.astype(float), prayers, ref["prayers"]) - int(ref["minutes"][0])
        inside = (there >= -cfg["edge"]) & (there < len(ref["minutes"]) + cfg["edge"])
        if not inside.any():
            continue
        pos = np.where(inside, np.clip(there, 0, len(ref["minutes"]) - 1), 0)
        ok_ref = np.isfinite(info["S"][ref["name"]][pos][:, donors]) & inside[:, None]
        common = ok & ok_ref
        codes = common @ (1 << np.arange(len(donors)))
        for code in np.unique(codes[inside]):
            rows = np.flatnonzero((codes == code) & inside)
            subset = tuple(d for d, o in zip(donors, common[rows[0]]) if o)
            if len(subset) < 2:
                continue
            profile = _residual_profile(info, ref, subset, cfg)[pos[rows]]
            a, beta = _coef(info, subset, cfg["lam"])
            value = a + S[np.ix_(rows, list(subset))] @ beta + profile
            good = np.isfinite(value)
            total[rows[good]] += w_ref * value[good]
            weight[rows[good]] += w_ref
            if ref["friday"] == friday:
                same_w[rows[good]] += 1
    carry = total / np.where(weight > 0, weight, np.nan)
    carry = np.where(np.isfinite(carry), carry, jn)
    out = pd.DataFrame(dict(jn=jn, carry=carry, same_refs=same_w, donors=ok.sum(axis=1)), index=minutes)
    if info.get("gbm") is not None:
        feats, level = _level_features(S, donors, info["mu"][1:], minutes.astype(float), prayers, friday)
        out["gbm"] = np.mean([m.predict(feats) for m in info["gbm"]], axis=0) + (jn if cfg["gbm_base"] == "jn" else level)
        out.loc[out.donors < 2, "gbm"] = np.nan
    out.loc[out.donors < 1, ["jn", "carry"]] = np.nan
    model["cache"][key] = out
    return out
