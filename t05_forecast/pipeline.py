"""End-to-end steps shared by validation and by the submission builder."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import gp
from .core import Recording
from .template import build_template, complete_template, slope

FRI, SAT = "20260918_1100", "20260919_1100"
P11, P12 = "20260911_1100", "20260912_1100"
OTHER_TEST_DAY = {FRI: SAT, SAT: FRI}


def sources_for(recs: dict, target_name: str) -> list:
    """Earlier recordings of the same weekday; the full-length day alone when there is one."""
    tgt = recs[target_name]
    same = [r for n, r in recs.items() if r.weekday == tgt.weekday and n != target_name and r.split == "train"]
    long_days = [r for r in same if r.n_minutes >= 600]
    return long_days or same


def make_template(recs: dict, name: str, lam: float, verbose: bool = False) -> np.ndarray:
    """Weekly template for one recording, with gaps and the new cameras I-M filled in."""
    rec = recs[name]
    tmpl = build_template(rec, sources_for(recs, name), lam=lam)
    long_train = [r for r in recs.values() if r.split == "train" and r.n_minutes >= 600 and r.name != name]
    tmpl, rep1 = complete_template(rec, tmpl, long_train, lam=lam)             # gaps of cameras A-H
    rep2 = {}
    if name in OTHER_TEST_DAY:                                                  # cameras I-M: fitted on the other test day
        tmpl, rep2 = complete_template(rec, tmpl, [recs[OTHER_TEST_DAY[name]]], lam=lam)
    if verbose:
        for unit, r in {**rep1, **rep2}.items():
            print(f"  {name} {unit}: template filled for {r['filled']} minutes from cameras {r['donors']} "
                  f"(fit on {r['rows']} rows, R2 {r['r2']:.2f})", flush=True)
    return tmpl


def deviation_fields(recs: dict, names: tuple, lam: float, clip: float = 1.5, verbose: bool = False) -> tuple:
    """Templates, their slopes and the observed deviation field for each named recording."""
    tm, sl, field = {}, {}, {}
    for name in names:
        tm[name] = make_template(recs, name, lam, verbose=verbose)
        sl[name] = slope(tm[name])
        field[name] = gp.observed(recs[name], tm[name], sl[name], clip=clip)
    return tm, sl, field


def save_params(p: gp.Params, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ells=p.ells, B=p.B, B0=p.B0, sig2=p.sig2, s2_delta=p.s2_delta, ell_delta=p.ell_delta,
             Bn=p.Bn if p.Bn is not None else np.zeros((0, 0)),
             tau=p.tau if p.tau is not None else np.zeros(p.B0.shape[0]),
             kinds=np.array(list(p.kinds) or ["ou"] * len(p.ells)))


def load_params(path: str | Path) -> gp.Params:
    d = np.load(path)
    bn = d["Bn"]
    return gp.Params(ells=d["ells"], B=d["B"], B0=d["B0"], sig2=d["sig2"], s2_delta=float(d["s2_delta"]),
                     ell_delta=float(d["ell_delta"]), Bn=bn if bn.size else None, tau=d["tau"],
                     kinds=tuple(str(x) for x in d["kinds"]))


def fit_cached(cache_dir: str | Path, key: str, datasets: list, **fit_kwargs) -> gp.Params:
    """Fit the deviation model, or load it if this exact configuration was fitted before."""
    path = Path(cache_dir) / f"params_{key}.npz"
    if path.exists():
        return load_params(path)
    p = gp.fit(datasets, **fit_kwargs)
    save_params(p, path)
    return p
