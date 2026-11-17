"""Today's deviation from the weekly template, modelled as a Gaussian process over (camera, minute).

deviation r = log1p(people) - template.  Its covariance between two cells is a sum of
  * a day part       (constant over the day: "today is busier than last week"),
  * time parts       that fade with the time distance (rough or smooth in time), each with
                     its own camera-to-camera matrix (cameras that watch the same crowd move together),
  * a timing part    (today's events run a few minutes early or late: the deviation is then
                     proportional to the template's slope),
  * a same-minute part shared between cameras, and independent reading noise per camera.
Each camera also has a delay in minutes: a crowd wave reaches some cameras a few minutes
after others, so time distances are measured after shifting each camera by its delay.

Any hole is predicted by conditioning on the deviations that may be used for it
(all released cells for hidden and missing holes; only cells at or before `now`
for forecast and hidden-future holes). The same model serves all four kinds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .core import ALL_UNITS, FORECAST, HIDDEN, HIDDEN_FUTURE, MISSING, RELEASED, Recording

UNIT_INDEX = {u: i for i, u in enumerate(ALL_UNITS)}
SQRT3 = math.sqrt(3.0)


@dataclass
class Params:
    ells: np.ndarray        # [K] time scales in minutes
    B: np.ndarray           # [K, U, U] camera-to-camera matrices
    B0: np.ndarray          # [U, U] day part
    sig2: np.ndarray        # [U] reading noise variance (independent per camera)
    s2_delta: float         # timing part: variance of the time offset (minutes^2)
    ell_delta: float
    Bn: np.ndarray | None = None        # [U, U] same-minute part shared between cameras
    tau: np.ndarray | None = None       # [U] delay of each camera in minutes
    kinds: tuple = field(default_factory=tuple)   # per time part: "ou" (rough) or "m32" (smooth)

    def kernel(self, ta, ca, da, tb, cb, db) -> np.ndarray:
        ua = ta - (self.tau[ca] if self.tau is not None else 0.0)
        ub = tb - (self.tau[cb] if self.tau is not None else 0.0)
        dt = np.abs(ua[:, None] - ub[None, :])
        ci, cj = ca[:, None], cb[None, :]
        k = self.B0[ci, cj].copy()
        for i, ell in enumerate(self.ells):
            kind = self.kinds[i] if self.kinds else "ou"
            if kind == "m32":
                x = SQRT3 * dt / ell
                k += self.B[i][ci, cj] * (1.0 + x) * np.exp(-x)
            else:
                k += self.B[i][ci, cj] * np.exp(-dt / ell)
        if self.s2_delta > 0:
            k += self.s2_delta * da[:, None] * db[None, :] * np.exp(-dt / self.ell_delta)
        if self.Bn is not None:
            k += self.Bn[ci, cj] * (ta[:, None] == tb[None, :])
        return k


def predict(p: Params, obs: dict, tgt: dict, want_var: bool = False):
    """Posterior mean (and variance) of the deviation at target cells given observed cells."""
    if len(obs["t"]) == 0:
        mean = np.zeros(len(tgt["t"]))
        return (mean, np.full(len(tgt["t"]), np.nan)) if want_var else mean
    koo = p.kernel(obs["t"], obs["c"], obs["d"], obs["t"], obs["c"], obs["d"])
    koo[np.diag_indices_from(koo)] += p.sig2[obs["c"]] + 1e-8
    cf = sla.cho_factor(koo, lower=True, check_finite=False)
    alpha = sla.cho_solve(cf, obs["r"], check_finite=False)
    kxo = p.kernel(tgt["t"], tgt["c"], tgt["d"], obs["t"], obs["c"], obs["d"])
    mean = kxo @ alpha
    if not want_var:
        return mean
    v = sla.solve_triangular(cf[0], kxo.T, lower=True, check_finite=False)
    prior = p.B0[tgt["c"], tgt["c"]] + sum(p.B[i][tgt["c"], tgt["c"]] for i in range(len(p.ells))) + p.s2_delta * tgt["d"] ** 2
    if p.Bn is not None:
        prior = prior + p.Bn[tgt["c"], tgt["c"]]
    return mean, np.maximum(prior - (v * v).sum(axis=0), 0.0)


def cells(rec: Recording, rows: np.ndarray, cols: np.ndarray, slope: np.ndarray) -> dict:
    return {
        "t": rec.first + rows.astype(float),
        "c": np.array([UNIT_INDEX[rec.units[c]] for c in cols], dtype=int),
        "d": slope[rows, cols].astype(float),
    }


def observed(rec: Recording, tmpl: np.ndarray, slope: np.ndarray, clip: float | None = None) -> dict:
    """Released cells that have a template, with their deviation from it."""
    ok = (rec.status == RELEASED) & np.isfinite(rec.y) & np.isfinite(tmpl)
    rows, cols = np.nonzero(ok)
    out = cells(rec, rows, cols, slope)
    r = np.log1p(rec.y[rows, cols]) - tmpl[rows, cols]
    if clip is not None:
        r = np.clip(r, -clip, clip)
    out["r"] = r
    out["rows"], out["cols"] = rows, cols
    return out


def predict_recording(rec: Recording, tmpl: np.ndarray, slope: np.ndarray, p: Params, strict: bool = True,
                      shrink: dict | None = None, return_dev: bool = False):
    """[T, U] predicted people for every hole that has a template (NaN elsewhere).

    strict=True  : forecast and hidden-future holes use only cells at or before their `now`
                   (the competition rule). strict=False exists for experiments only.
    shrink       : optional {status code: factor} applied to the predicted deviation.
    """
    obs = observed(rec, tmpl, slope)
    pred = np.full(rec.y.shape, np.nan)
    dev = np.full(rec.y.shape, np.nan)
    shrink = shrink or {}

    def fill(mask: np.ndarray, sub: dict) -> None:
        rows, cols = np.nonzero(mask & np.isfinite(tmpl))
        if rows.size == 0:
            return
        tgt = cells(rec, rows, cols, slope)
        mean = predict(p, sub, tgt)
        fac = np.array([shrink.get(int(s), 1.0) for s in rec.status[rows, cols]])
        dev[rows, cols] = mean
        pred[rows, cols] = np.maximum(np.expm1(tmpl[rows, cols] + fac * mean), 0.0)

    fill((rec.status == MISSING) | (rec.status == HIDDEN), obs)
    timed = (rec.status == FORECAST) | (rec.status == HIDDEN_FUTURE)
    if not strict:
        fill(timed, obs)
    else:
        keys = ("t", "c", "d", "r")
        for now in np.unique(rec.now[timed]):
            keep = obs["t"] <= now
            fill(timed & (rec.now == now), {k: obs[k][keep] for k in keys})
    return (pred, dev) if return_dev else pred


# ---------------------------------------------------------------- fitting (PyTorch)

def fit(datasets: list, steps: int = 160, lr: float = 0.05, rank: int = 2, ells=(4.0, 45.0), kinds=None,
        use_delta: bool = True, use_same_minute: bool = True, use_delay: bool = True,
        n_units: int = len(ALL_UNITS), seed: int = 0, verbose: bool = True, reg: float = 1e-3,
        threads: int | None = None) -> Params:
    """Maximum-likelihood fit of the covariance on a list of deviation fields (one per recording)."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    dtype = torch.float64
    k = len(ells)
    kinds = tuple(kinds) if kinds else tuple("ou" for _ in ells)
    log_ell = torch.nn.Parameter(torch.log(torch.tensor(ells, dtype=dtype)))
    load = torch.zeros(k, n_units, rank, dtype=dtype)
    load[:, :, 0] = 0.08
    load += 0.02 * torch.randn(k, n_units, rank, dtype=dtype)
    load = torch.nn.Parameter(load)
    log_d = torch.nn.Parameter(torch.full((k, n_units), math.log(0.06), dtype=dtype))
    log_a0 = torch.nn.Parameter(torch.tensor(math.log(0.10), dtype=dtype))
    log_d0 = torch.nn.Parameter(torch.full((n_units,), math.log(0.08), dtype=dtype))
    log_sig = torch.nn.Parameter(torch.full((n_units,), math.log(0.10), dtype=dtype))
    log_sdel = torch.nn.Parameter(torch.tensor(math.log(1.5), dtype=dtype))
    log_ldel = torch.nn.Parameter(torch.tensor(math.log(90.0), dtype=dtype))
    load_n = torch.nn.Parameter(0.03 + 0.01 * torch.randn(n_units, rank, dtype=dtype))
    tau = torch.nn.Parameter(torch.zeros(n_units, dtype=dtype))
    params = [log_ell, load, log_d, log_a0, log_d0, log_sig]
    params += [log_sdel, log_ldel] if use_delta else []
    params += [load_n] if use_same_minute else []
    groups = [{"params": params, "lr": lr}]
    if use_delay:
        groups.append({"params": [tau], "lr": 3 * lr})
    opt = torch.optim.Adam(groups)
    all_params = params + ([tau] if use_delay else [])

    data = []
    for ds in datasets:
        t = torch.tensor(ds["t"], dtype=dtype)
        data.append({
            "t": t,
            "same": (t[:, None] == t[None, :]).to(dtype),
            "c": torch.tensor(ds["c"], dtype=torch.long),
            "dd": torch.tensor(np.outer(ds["d"], ds["d"]), dtype=dtype),
            "r": torch.tensor(ds["r"], dtype=dtype),
        })
    n_total = sum(len(d["r"]) for d in data)

    def components():
        ell = torch.exp(log_ell).clamp(1.0, 400.0)
        bk = load @ load.transpose(1, 2) + torch.diag_embed(torch.exp(2 * log_d))
        b0 = torch.exp(2 * log_a0) + torch.diag(torch.exp(2 * log_d0))
        bn = load_n @ load_n.T
        return ell, bk, b0, bn, tau.clamp(-15.0, 15.0)

    def nll() -> "torch.Tensor":
        ell, bk, b0, bn, tau_c = components()
        total = torch.zeros((), dtype=dtype)
        for d in data:
            ci, cj = d["c"][:, None], d["c"][None, :]
            u = d["t"] - tau_c[d["c"]] if use_delay else d["t"]
            dt = (u[:, None] - u[None, :]).abs()
            cov = b0[ci, cj]
            for i in range(k):
                if kinds[i] == "m32":
                    x = SQRT3 * dt / ell[i]
                    cov = cov + bk[i][ci, cj] * (1.0 + x) * torch.exp(-x)
                else:
                    cov = cov + bk[i][ci, cj] * torch.exp(-dt / ell[i])
            if use_delta:
                cov = cov + torch.exp(2 * log_sdel) * d["dd"] * torch.exp(-dt / torch.exp(log_ldel).clamp(5.0, 600.0))
            if use_same_minute:
                cov = cov + bn[ci, cj] * d["same"]
            cov = cov + torch.diag(torch.exp(2 * log_sig)[d["c"]] + 1e-6)
            chol = torch.linalg.cholesky(cov)
            alpha = torch.cholesky_solve(d["r"][:, None], chol)[:, 0]
            total = total + 0.5 * (d["r"] * alpha).sum() + torch.log(torch.diagonal(chol)).sum()
        return total / n_total + reg * ((load ** 2).sum() + (load_n ** 2).sum()) + 1e-5 * (tau ** 2).sum()

    best, best_state = float("inf"), None
    for step in range(steps):
        opt.zero_grad()
        loss = nll()
        loss.backward()
        opt.step()
        val = float(loss.detach())
        if val < best:
            best, best_state = val, [p.detach().clone() for p in all_params]
        if verbose and (step % 40 == 0 or step == steps - 1):
            print(f"    fit step {step:4d}  nll/cell {val:.4f}", flush=True)
    with torch.no_grad():
        for p, s in zip(all_params, best_state):
            p.copy_(s)
        ell, bk, b0, bn, tau_c = components()
        return Params(
            ells=ell.numpy().copy(), B=bk.numpy().copy(), B0=b0.numpy().copy(),
            sig2=torch.exp(2 * log_sig).numpy().copy(),
            s2_delta=float(torch.exp(2 * log_sdel)) if use_delta else 0.0,
            ell_delta=float(torch.exp(log_ldel).clamp(5.0, 600.0)) if use_delta else 1.0,
            Bn=bn.numpy().copy() if use_same_minute else None,
            tau=tau_c.numpy().copy() if use_delay else np.zeros(n_units),
            kinds=kinds,
        )


def describe(p: Params) -> str:
    kinds = p.kinds or tuple("ou" for _ in p.ells)
    lines = [f"time parts: " + ", ".join(f"{kd} {ell:.1f} min" for kd, ell in zip(kinds, p.ells))
             + f"; timing sd {math.sqrt(p.s2_delta):.2f} min over {p.ell_delta:.0f} min"]
    if p.tau is not None:
        lines.append("delay (min):           " + " ".join(f"{u[-1]}:{p.tau[i]:+.1f}" for u, i in UNIT_INDEX.items()))
    lines.append("day sd per camera:     " + " ".join(f"{u[-1]}:{math.sqrt(p.B0[i, i]):.2f}" for u, i in UNIT_INDEX.items()))
    for k, ell in enumerate(p.ells):
        lines.append(f"sd of part {kinds[k]:>3} {ell:5.1f}:  " + " ".join(f"{u[-1]}:{math.sqrt(p.B[k][i, i]):.2f}" for u, i in UNIT_INDEX.items()))
    lines.append("reading noise sd:      " + " ".join(f"{u[-1]}:{math.sqrt(p.sig2[i]):.2f}" for u, i in UNIT_INDEX.items()))
    if p.Bn is not None:
        lines.append("same-minute shared sd: " + " ".join(f"{u[-1]}:{math.sqrt(p.Bn[i, i]):.2f}" for u, i in UNIT_INDEX.items()))
    return "\n".join(lines)


def correlations(p: Params, units: list) -> str:
    """Camera-to-camera correlation implied by each part (for reading the fit)."""
    import pandas as pd
    idx = [UNIT_INDEX[u] for u in units]
    names = [u[-1] for u in units]
    blocks = []
    parts = [(f"time part {ell:.0f} min", p.B[k]) for k, ell in enumerate(p.ells)]
    if p.Bn is not None:
        parts.append(("same minute (incl. own noise)", p.Bn + np.diag(p.sig2)))
    for label, mat in parts:
        m = mat[np.ix_(idx, idx)]
        sd = np.sqrt(np.diag(m))
        blocks.append(f"correlation, {label}:\n" + pd.DataFrame(m / np.outer(sd, sd), index=names, columns=names).round(2).to_string())
    return "\n".join(blocks)
