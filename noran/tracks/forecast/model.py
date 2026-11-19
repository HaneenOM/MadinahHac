"""Forecast for cameras A to H: the accepted T05 model run on a weekly template whose timing is re-estimated at every `now`.

The accepted model reads last week's curve of the same camera at the clock time given by the prayer timetable and
models today's deviation from it with a Gaussian process. Its timing term (about 3 minutes sd, about 15 minutes memory,
linear in the template's slope) cannot follow a crowd that runs several minutes early or late through a whole event.

Here, for every forecast moment `now`:
  1. the delay of today against the template (minutes, positive = today runs later) is estimated from the readings of
     cameras A, E and F at or before `now` (estimate_delay_ls: a curve match over the last 45 minutes with a free level
     per camera and a normal prior on the delay),
  2. a positive delay is dropped when last week's curve climbs steeply within the next 15 minutes (rise_gate),
  3. the template of every camera is read `delay` minutes earlier (shift_template) and its slope is recomputed,
  4. the same deviation model with the same fitted covariances conditions on the readings at or before `now`
     (forecast_at_now); the memory of its timing term is set to 120 minutes and its sd to 3 minutes by hand
     (nothing is refitted).
Nothing after `now` is read: every function receives `now` and cuts the recording itself.

estimate_delay_gp (the deviation model's own timing term as the estimator), shift_reach, level_ref and
build_template_custom(keep=...) are alternatives that were tested and are switched off in DEFAULTS.

Used by eval_fixed.py, eval_rolling.py and make_test_predictions.py in this folder.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _folder in (ROOT, ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

from t05_forecast import gp  # noqa: E402
from t05_forecast.core import FORECAST, RELEASED, Recording  # noqa: E402
from t05_forecast.prayer import PRAYERS, prayer_minutes  # noqa: E402
from t05_forecast.template import complete_template, sample_at, slope, smooth_log  # noqa: E402

NEW_CAMERAS = "IJKLM"          # no earlier week exists: these keep the accepted rule
TEMPLATE_KEEP = PRAYERS         # prayers at which the template follows the timetable shift (all four = accepted)

# method: "none" = the accepted model (delay 0), "ls" or "gp" = which estimator sets the delay.
# How each number was chosen is in the track report: the estimator's numbers are first guesses that a sensitivity run on
# the selection sets left in place; rise_gate, sd_delta and ell_delta were chosen on the selection sets.
DEFAULTS = dict(
    method="ls",
    max_shift=12.0,        # largest delay considered and applied, minutes either way
    gain=1.0,              # share of the estimated delay that is applied
    # --- estimator "ls": direct curve match
    shift_cameras="AEF",   # cameras whose readings set the delay (they watch the same crowd and carry the error)
    window=45,             # minutes of today's readings before `now` that are compared with the template
    step=0.5,              # grid step, minutes
    prior_sd=4.0,          # prior sd of the delay around prior_mean, minutes
    prior_mean=0.0,
    noise_sd=0.12,         # sd of a reading around the shifted template, log units, after the level is removed
    eff=3.0,               # neighbouring minutes are not independent: this many minutes count as one reading
    level_sd=0.25,         # prior sd of the level difference between today and the template inside the window
    min_rows=8,            # fewest readings a camera needs in the window to take part
    estimate="mean",       # posterior mean over the grid ("map" = its highest point)
    level_ref=0,           # 0 = level prior centred on 0; N = centred on the camera's median deviation in the readings
                           #     between N minutes back and the start of the window (tested, no gain)
    # --- gate: a positive delay (today late) is not applied when the template of the big cameras climbs by more than
    #     this many log units within the next 15 minutes (the surge after a prayer started on time). None = no gate.
    rise_gate=0.15,
    # --- reach of the delay back in time: None = the whole day is read at the delay; (a, b) = full delay for the last
    #     b minutes, none before now - a, straight line in between (tested, no difference)
    shift_reach=None,
    # --- estimator "gp" (tested, unstable): the deviation model's own timing term
    est_sd=4.0, est_ell=120.0, est_iter=2, est_window=0,
    # --- deviation model used for the forecast itself: None keeps the fitted value
    sd_delta=3.0,          # sd of the timing term, minutes (fitted production values: 3.0 to 3.4, one member 1.8)
    ell_delta=120.0,       # memory of the timing term, minutes (fitted production values: 8 to 18)
)
ACCEPTED = dict(method="none", sd_delta=None, ell_delta=None)       # these three settings give the accepted model


# ---------------------------------------------------------------- template

def shift_template(base: np.ndarray, delay) -> np.ndarray:
    """[T, U] template read `delay` minutes earlier: out[t] = base[t - delay] (straight line between minutes, edges held).

    delay may be one number or one number per minute. Positive = today's events come later than the template says.
    """
    n = base.shape[0]
    pos = np.clip(np.arange(n, dtype=float) - np.asarray(delay, dtype=float), 0.0, n - 1.0)
    pos = np.broadcast_to(pos, (n,))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    f = (pos - lo)[:, None]
    with np.errstate(invalid="ignore"):
        out = base[lo] * (1.0 - f) + base[hi] * f
    exact = (pos - lo) < 1e-9                      # keep cells whose upper neighbour is missing
    out[exact] = base[lo[exact]]
    return out


def warp_custom(minutes, target_day: str, source_day: str, keep: tuple = PRAYERS):
    """The timetable shift of prayer.warp_minutes, applied only at the prayers named in `keep` (0 at the others)."""
    tp, sp = prayer_minutes(target_day), prayer_minutes(source_day)
    anchors = [tp[p] for p in PRAYERS]
    shifts = [(sp[p] - tp[p]) if p in keep else 0.0 for p in PRAYERS]
    minutes = np.asarray(minutes, dtype=float)
    return minutes + np.interp(minutes, anchors, shifts)


def build_template_custom(target: Recording, sources: list, lam: float, keep: tuple = PRAYERS) -> np.ndarray:
    """template.build_template with the timetable shift limited to the prayers in `keep`."""
    curves = []
    for s in sources:
        if s.name == target.name:
            continue
        zs = smooth_log(s.y, lam)
        src = warp_custom(target.minutes, target.day, s.day, keep)
        lo, hi = float(s.first), float(s.first + s.n_minutes - 1)
        near = (src >= lo - 10) & (src <= hi + 10)
        src = np.where(near, np.clip(src, lo, hi), src)
        arr = np.full(target.y.shape, np.nan)
        for j, unit in enumerate(target.units):
            if unit in s.units:
                arr[:, j] = sample_at(zs[:, s.col(unit)], s.first, src)
        curves.append(arr)
    if not curves:
        return np.full(target.y.shape, np.nan)
    stack = np.stack(curves)
    if len(curves) > 1:
        cover = np.isfinite(stack).sum(axis=1)
        for j in range(stack.shape[2]):
            main = int(np.argmax(cover[:, j]))
            for s in range(stack.shape[0]):
                both = np.isfinite(stack[s, :, j]) & np.isfinite(stack[main, :, j])
                if s != main and both.sum() >= 20:
                    stack[s, :, j] += np.mean(stack[main, both, j] - stack[s, both, j])
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(stack).sum(axis=0)
        return np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)


def production_template(recs: dict, name: str, lam: float, keep: tuple = PRAYERS) -> np.ndarray:
    """pipeline.make_template for a test recording (same sources, same filling of gaps and of cameras I to M)."""
    from t05_forecast.pipeline import OTHER_TEST_DAY, sources_for
    rec = recs[name]
    tmpl = build_template_custom(rec, sources_for(recs, name), lam, keep)
    long_train = [r for r in recs.values() if r.split == "train" and r.n_minutes >= 600 and r.name != name]
    tmpl, _ = complete_template(rec, tmpl, long_train, lam=lam)
    if name in OTHER_TEST_DAY:
        tmpl, _ = complete_template(rec, tmpl, [recs[OTHER_TEST_DAY[name]]], lam=lam)
    return tmpl


def heldout_template(recs: dict, name: str, rec: Recording, lam: float, keep: tuple = PRAYERS) -> np.ndarray:
    """The template src/t05_local_forecast.py builds for a training day: other training recordings only."""
    train_recs = [r for r in recs.values() if r.split == "train" and r.name != name]
    prior = [r for r in train_recs if r.weekday == rec.weekday]
    others = [r for r in train_recs if r.weekday != rec.weekday]
    tmpl = build_template_custom(rec, prior, lam, keep)
    tmpl, _ = complete_template(rec, tmpl, others, lam=lam)
    return tmpl


# ---------------------------------------------------------------- delay of today against the template

def _edit(params: gp.Params, sd=None, ell=None) -> gp.Params:
    if sd is None and ell is None:
        return params
    return dataclasses.replace(params, s2_delta=float(params.s2_delta if sd is None else sd ** 2),
                               ell_delta=float(params.ell_delta if ell is None else ell))


def _observed(rec: Recording, tmpl: np.ndarray, sl: np.ndarray, now: int, window: int = 0) -> dict:
    """Released cells at or before `now` that have a template, with their deviation from it."""
    ok = (rec.status == RELEASED) & np.isfinite(rec.y) & np.isfinite(tmpl) & (rec.minutes <= now)[:, None]
    if window:
        ok &= (rec.minutes > now - window)[:, None]
    rows, cols = np.nonzero(ok)
    out = gp.cells(rec, rows, cols, sl)
    out["r"] = np.log1p(rec.y[rows, cols]) - tmpl[rows, cols]
    return out


def estimate_delay_gp(rec: Recording, base: np.ndarray, params: gp.Params, now: int, cfg: dict | None = None) -> dict:
    """Delay at `now` from the deviation model's own timing term, given readings at or before `now`.

    The model says: deviation = (time offset) x (template slope) + level parts + noise, one time offset shared by all
    cameras. With a prior sd `est_sd` and memory `est_ell` for that offset, its posterior mean at `now` follows from
    the readings. Because a delay of several minutes is not small against the width of a peak, the template is then
    read at the estimated delay and the step is repeated (`est_iter` times), keeping the prior centred on delay 0.
    """
    c = {**DEFAULTS, **(cfg or {})}
    p = _edit(params, c["est_sd"], c["est_ell"])
    s, steps = 0.0, []
    for _ in range(int(c["est_iter"])):
        tmpl = shift_template(base, s) if s != 0.0 else base
        sl = slope(tmpl)
        obs = _observed(rec, tmpl, sl, now, int(c["est_window"]))
        if len(obs["t"]) < 5:
            break
        k = p.kernel(obs["t"], obs["c"], obs["d"], obs["t"], obs["c"], obs["d"])
        k[np.diag_indices_from(k)] += p.sig2[obs["c"]] + 1e-8
        cf = sla.cho_factor(k, lower=True, check_finite=False)
        # the offset process has prior mean s in the shifted frame (total delay = s - offset, prior mean 0)
        alpha = sla.cho_solve(cf, obs["r"] - s * obs["d"], check_finite=False)
        u = obs["t"] - (p.tau[obs["c"]] if p.tau is not None else 0.0)
        cross = p.s2_delta * obs["d"] * np.exp(-np.abs(float(now) - u) / p.ell_delta)
        s = float(np.clip(-(cross @ alpha), -c["max_shift"], c["max_shift"]))
        steps.append(s)
    return dict(now=int(now), raw=s, shift=float(np.clip(c["gain"] * s, -c["max_shift"], c["max_shift"])), steps=steps)


def estimate_delay_ls(rec: Recording, base: np.ndarray, now: int, cfg: dict | None = None) -> dict:
    """Delay at `now` from a direct curve match, readings at or before `now` only.

    For every delay s on a grid, today's log readings of the last `window` minutes are compared with the template
    read s minutes earlier. Per camera the level difference is free but pulled toward zero (level_sd). The cameras'
    costs are added, a normal prior keeps s near `prior_mean`, and the posterior mean is returned.
    """
    c = {**DEFAULTS, **(cfg or {})}
    grid = np.arange(-c["max_shift"], c["max_shift"] + 1e-9, c["step"])
    r0 = int(now) - rec.first
    logp = -((grid - c["prior_mean"]) ** 2) / (2.0 * c["prior_sd"] ** 2)
    out = dict(now=int(now), shift=float(c["prior_mean"]) * c["gain"], raw=float(c["prior_mean"]), sd=float(c["prior_sd"]),
               cameras=0, rows=0)
    if r0 < 0:
        return out
    rows = np.arange(max(0, r0 - int(c["window"]) + 1), min(r0, rec.n_minutes - 1) + 1)       # rows <= now only
    used, total = 0, 0
    n = base.shape[0]
    for j, unit in enumerate(rec.units):
        if unit[-1] not in c["shift_cameras"]:
            continue
        y = rec.y[rows, j]
        ok = (rec.status[rows, j] == RELEASED) & np.isfinite(y)
        if ok.sum() < c["min_rows"]:
            continue
        rr, z = rows[ok], np.log1p(y[ok])
        pos = np.clip(rr[:, None].astype(float) - grid[None, :], 0.0, n - 1.0)                # [rows, S]
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        f = pos - lo
        t = base[lo, j] * (1 - f) + base[hi, j] * f
        good = np.isfinite(t).all(axis=1)
        if good.sum() < c["min_rows"]:
            continue
        t, z = t[good], z[good]
        e = z[:, None] - t
        m = len(z) / c["eff"]                                                                 # effective readings
        mean_e = e.mean(axis=0)
        ss = ((e - mean_e) ** 2).mean(axis=0) * m / c["noise_sd"] ** 2
        centre = 0.0
        if c["level_ref"]:
            old_rows = np.arange(max(0, r0 - int(c["level_ref"]) + 1), rows[0])
            if old_rows.size:
                yo = rec.y[old_rows, j]
                oko = (rec.status[old_rows, j] == RELEASED) & np.isfinite(yo) & np.isfinite(base[old_rows, j])
                if oko.sum() >= 5:
                    centre = float(np.median(np.log1p(yo[oko]) - base[old_rows[oko], j]))
        lvl = (mean_e - centre) ** 2 / (c["noise_sd"] ** 2 / m + c["level_sd"] ** 2)          # level ~ N(centre, level_sd^2)
        logp = logp - 0.5 * (ss + lvl)
        used += 1
        total += len(z)
    if used == 0:
        return out
    w = np.exp(logp - logp.max())
    w = w / w.sum()
    mean = float((w * grid).sum())
    raw = mean if c["estimate"] == "mean" else float(grid[int(np.argmax(w))])
    out.update(shift=c["gain"] * raw, raw=raw, sd=float(np.sqrt((w * (grid - mean) ** 2).sum())), cameras=used, rows=total)
    return out


def estimate_delay(rec: Recording, members: list, now: int, cfg: dict | None = None) -> dict:
    """The delay applied at `now`. Read from the first member's template and covariance, used for all members."""
    c = {**DEFAULTS, **(cfg or {})}
    base, params = members[0]
    if c["method"] == "gp":
        out = estimate_delay_gp(rec, base, params, now, c)
    elif c["method"] == "ls":
        out = estimate_delay_ls(rec, base, now, c)
    else:
        return dict(now=int(now), shift=0.0, raw=0.0)
    out["rise"] = template_rise(rec, base, now, c["shift_cameras"])
    out["gated"] = bool(c["rise_gate"] is not None and out["shift"] > 0 and np.isfinite(out["rise"]) and out["rise"] > c["rise_gate"])
    if out["gated"]:
        out["shift"] = 0.0
    return out


def template_rise(rec: Recording, base: np.ndarray, now: int, cameras: str = "AEF", horizon: int = 15) -> float:
    """Mean over the big cameras of the template's climb from `now` to its highest point in the next `horizon` minutes
    (log units). It is a property of last week's curve and the clock only; no reading of today enters."""
    r0 = int(now) - rec.first
    cols = [j for j, u in enumerate(rec.units) if u[-1] in cameras]
    seg = base[max(r0, 0):min(r0 + horizon + 1, base.shape[0])][:, cols]
    seg = seg[:, np.isfinite(seg).all(axis=0)] if seg.size else seg
    if seg.size == 0 or seg.shape[0] < 2:
        return float("nan")
    return float(np.mean(seg.max(axis=0) - seg[0]))


# ---------------------------------------------------------------- deviation model on the re-timed template

def forecast_at_now(rec: Recording, members: list, now: int, cfg: dict | None = None, shift=None) -> tuple:
    """Values for the forecast holes of `rec` whose `now` is this one. Returns ([T, U] array, info).

    members : list of (base template [T, U], fitted covariance) - the accepted ensemble.
    shift   : None = estimate the delay from the readings at or before `now`; a number = use it.
    With method "none" (or shift=0) and no timing edit this is the accepted model, value for value.
    Only cells at or before `now` are read. Cameras I to M stay NaN (the caller keeps the accepted rule for them).
    """
    c = {**DEFAULTS, **(cfg or {})}
    now = int(now)
    info = estimate_delay(rec, members, now, c) if shift is None else dict(now=now, shift=float(shift), raw=float(shift))
    s = float(info["shift"])
    delay = s
    if s != 0.0 and c["shift_reach"] is not None:
        far, near = c["shift_reach"]
        delay = s * np.clip((rec.minutes - (now - far)) / float(far - near), 0.0, 1.0)
    target = (rec.status == FORECAST) & (rec.now == now)
    preds = []
    for base, params in members:
        tmpl = shift_template(base, delay) if s != 0.0 else base
        sl = slope(tmpl)
        p = _edit(params, c["sd_delta"], c["ell_delta"])
        obs = _observed(rec, tmpl, sl, now)
        trow, tcol = np.nonzero(target & np.isfinite(tmpl))
        out = np.full(rec.y.shape, np.nan)
        if trow.size:
            mean = gp.predict(p, obs, gp.cells(rec, trow, tcol, sl))
            out[trow, tcol] = np.maximum(np.expm1(tmpl[trow, tcol] + mean), 0.0)
        for j, unit in enumerate(rec.units):
            if unit[-1] in NEW_CAMERAS:
                out[:, j] = np.nan
        preds.append(out)
    stack = np.stack(preds)
    count = np.isfinite(stack).sum(axis=0)
    mean = np.divide(np.nansum(stack, axis=0), count, out=np.full(rec.y.shape, np.nan), where=count > 0)
    return mean, info


def forecast_recording(rec: Recording, members: list, cfg: dict | None = None, shift=None) -> tuple:
    """All forecast holes of a recording, one `now` at a time. Returns ([T, U] array, list of info per now)."""
    out = np.full(rec.y.shape, np.nan)
    infos = []
    fc = rec.status == FORECAST
    for now in np.unique(rec.now[fc]):
        p, info = forecast_at_now(rec, members, int(now), cfg, shift=shift)
        mask = fc & (rec.now == now)
        out[mask] = p[mask]
        infos.append(info)
    return out, infos
