"""Building blocks of the hidden-future track: prayer-aligned time, week-analog scaling, smoothed hidden level.

Everything here reads the feed only at or before `now` (the caller passes the causal slice).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAMS = [f"cam_{c}" for c in "ABCDEFGH"]
HIDDEN_WITH = {"cam_B": ("cam_B", "cam_G"), "cam_G": ("cam_B", "cam_G"), "cam_D": ("cam_D",)}


# ---------------------------------------------------------------- prayer-aligned time
def reference_prayers(schedule: dict) -> np.ndarray:
    return np.median(np.stack([schedule[d] for d in sorted(schedule)]), axis=0)


def align(minute, day: str, schedule: dict, ref: np.ndarray, strength=(0., 0., 0., 1., 1.)):
    """Clock minute -> aligned minute. The offset to the reference timetable is interpolated between prayers.
    `strength` says how much of each prayer's timetable shift the crowd follows
    (fajr, dhuhr, asr, maghrib, isha): Maghrib and Isha crowds follow the timetable, Dhuhr and Asr keep the clock."""
    pr = np.asarray(schedule[day], float)
    offset = (ref - pr) * np.asarray(strength, float)
    return np.asarray(minute, float) + np.interp(np.asarray(minute, float), pr, offset)


class Analog:
    """One training recording on the aligned integer-minute grid."""

    def __init__(self, table: pd.DataFrame, day: str, schedule: dict, ref: np.ndarray, smooth: int = 5, strength=(0., 0., 0., 1., 1.)):
        self.day = day
        self.weekday = int(pd.Timestamp(day).day_of_week)
        minutes = np.asarray(table.index, float)
        a = align(minutes, day, schedule, ref, strength)
        self.first, self.last = int(np.ceil(a.min())), int(np.floor(a.max()))
        grid = np.arange(self.first, self.last + 1, dtype=float)
        pos = np.interp(grid, a, np.arange(len(minutes), dtype=float))
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(minutes) - 1)
        frac = pos - lo
        self.raw, self.smooth, self.smooth5 = {}, {}, {}
        for c in table.columns:
            v = table[c].to_numpy(float)
            g = np.where(frac < 1e-9, v[lo], (1 - frac) * v[lo] + frac * v[hi])
            self.raw[c] = g
            s = pd.Series(g)
            self.smooth[c] = s.rolling(smooth, center=True, min_periods=max(2, smooth // 2)).mean().to_numpy()
            self.smooth5[c] = s.rolling(5, center=True, min_periods=2).mean().to_numpy()

    def at(self, series: np.ndarray, tau) -> np.ndarray:
        """Linear read of a grid series at aligned times `tau` (NaN outside the recording or next to a gap)."""
        tau = np.atleast_1d(np.asarray(tau, float))
        pos = tau - self.first
        out = np.full(tau.shape, np.nan)
        ok = (pos >= 0) & (pos <= len(series) - 1)
        lo = np.floor(pos[ok]).astype(int)
        hi = np.minimum(lo + 1, len(series) - 1)
        f = pos[ok] - lo
        out[ok] = np.where(f < 1e-9, series[lo], (1 - f) * series[lo] + f * series[hi])
        return out


def window(causal: pd.DataFrame, cam: str, now: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Minutes and readings of `cam` in [now-k+1, now] (finite readings only)."""
    if cam not in causal.columns:
        return np.empty(0), np.empty(0)
    s = causal[cam].loc[now - k + 1:now].dropna()
    return s.index.to_numpy(float), s.to_numpy(float)


def analog_ratio(minutes: np.ndarray, values: np.ndarray, analog: Analog, cam: str, day: str, schedule, ref,
                 min_pairs: int = 3, strength=(0., 0., 0., 1., 1.), delta: float = 0.0):
    """Today's level of a visible camera over a window divided by the analog recording's level at the same aligned minutes
    (the analog is read `delta` minutes later)."""
    if len(minutes) < min_pairs or cam not in analog.raw:
        return np.nan, 0
    other = analog.at(analog.raw[cam], align(minutes, day, schedule, ref, strength) + delta)
    ok = np.isfinite(other)
    if ok.sum() < min_pairs or other[ok].sum() <= 0:
        return np.nan, 0
    return float(values[ok].sum() / other[ok].sum()), int(ok.sum())


def ridge_values(causal: pd.DataFrame, now: int, k: int, info) -> np.ndarray:
    """v2 multi-camera hidden model (ridge) at each of the last k minutes that has at least two visible cameras."""
    import v2_hidden
    frame = causal.reindex(index=range(now - k + 1, now + 1), columns=info["donors"])
    ok = frame.notna().sum(axis=1).ge(2).to_numpy()
    if not ok.any():
        return np.empty(0)
    X = v2_hidden._design(frame[ok], info["donors"], info["med"])
    return np.maximum(0., info["ridge"].predict(X))


def shape_mismatch(minutes: np.ndarray, values: np.ndarray, analog: Analog, cam: str, tau: np.ndarray, delta: float, min_pairs: int = 5):
    """Mean absolute difference between log(today) and log(analog, 5-minute mean read `delta` minutes later) after removing
    the mean difference: how badly the analog's shape fits today's last minutes for one camera. `tau` = aligned `minutes`."""
    if len(minutes) < min_pairs or cam not in analog.smooth5:
        return np.nan
    other = analog.at(analog.smooth5[cam], tau + delta)
    ok = np.isfinite(other) & (other > 0) & (values > 0)
    if ok.sum() < min_pairs:
        return np.nan
    diff = np.log(values[ok]) - np.log(other[ok])
    return float(np.abs(diff - diff.mean()).mean())


def best_shift(wins: dict, analog: Analog, cams, day: str, schedule, ref, strength, max_shift: int, penalty: float) -> int:
    """Shift (whole minutes, |shift| <= max_shift) at which the analog's visible cameras `cams` fit today's windows best.
    0 when the cameras do not have enough readings. `penalty` is added per minute of shift."""
    taus = {c: align(wins[c][0], day, schedule, ref, strength) for c in cams if c in wins and len(wins[c][0])}
    base = {c: shape_mismatch(*wins[c], analog, c, taus[c], 0.0) for c in taus}
    use = [c for c, v in base.items() if np.isfinite(v)]
    if not use:
        return 0
    best, best_cost = 0, float(np.mean([base[c] for c in use]))
    for d in range(-max_shift, max_shift + 1):
        if d == 0:
            continue
        costs = [shape_mismatch(*wins[c], analog, c, taus[c], float(d)) for c in use]
        if not np.all(np.isfinite(costs)):
            continue
        cost = float(np.mean(costs)) + penalty * abs(d)
        if cost < best_cost - 1e-12:
            best, best_cost = d, cost
    return best
