"""Hidden camera with the hedge the independent checker asked for: noran/tracks/hidden_final/model.py, and for
camera B the mean with the accepted baseline value in the minutes where camera D stands far above the others.

Why (checker's report on the hidden track, 18 Nov 2026): when camera D stands far above what cameras A, C, F and H
suggest, the hidden track was 5 to 7 people low for B on both local Fridays, while the accepted baseline, which
leans on D more, did better there. The largest such block of the real Friday is 20:56 to 21:14, where D reads
73% above last week and the other cameras 7 to 26% below. The hedge is insurance for such blocks, not a measured
gain: it helps on the Friday analogs and costs a little on the Saturday analog.

    D excess = smoothed log D - regression of smoothed log D on smoothed log A, C, F, H
               (regression fitted on the training rows handed to fit(); only visible cameras are read)
    camera B, D excess > HEDGE_CUT:   value = (1 - HEDGE_SHARE) * hidden_final + HEDGE_SHARE * accepted baseline

Contract: fit / predict as in noran/track_eval.py.

Run (repository root):
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/hidden_hedged/model.py --set fixed --label hidden_hedged
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = Path(__file__).resolve().parents[1]
HEDGE_UNIT = "cam_B"
HEDGE_CUT = float(os.environ.get("NORAN_HEDGE_CUT", "0.3"))        # D excess (log scale) above which the hedge applies
HEDGE_SHARE = float(os.environ.get("NORAN_HEDGE_SHARE", "0.5"))    # share of the accepted baseline value
OTHERS = ["cam_A", "cam_C", "cam_F", "cam_H"]


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


final = _load("hidden_final_four_members", TRACKS / "hidden_final" / "model.py")
hcomp = final.base.hcomp
tasks = final.cat.tasks


def _smoothed(table: pd.DataFrame) -> pd.DataFrame:
    grid = hcomp.to_grid(table)
    return pd.DataFrame(hcomp.smooth(np.log1p(grid.to_numpy()), 6, 0.8, 0.5), index=grid.index, columns=grid.columns)


def _fit_excess(training: pd.DataFrame):
    """Regression of smoothed log D on smoothed log A, C, F, H over the training recordings handed over."""
    parts = []
    for _, g in training.groupby("recording"):
        t = g.pivot(index="minute_of_day", columns="unit", values="people").sort_index()
        parts.append(_smoothed(t).iloc[::5])
    pool = pd.concat(parts).dropna(subset=OTHERS + ["cam_D"])
    if len(pool) < 50:
        return None
    x = np.column_stack([np.ones(len(pool)), pool[OTHERS].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, pool["cam_D"].to_numpy(), rcond=None)
    return beta


def d_excess(t: pd.DataFrame, minute: int, model) -> float:
    """D excess at one minute of the feed (NaN when D or one of A, C, F, H has no reading nearby)."""
    beta = model["excess_beta"]
    if beta is None or "cam_D" not in t.columns or len(t) < 13:
        return np.nan
    visible = t.reindex(columns=["cam_D"] + OTHERS)
    key = (len(visible), hashlib.sha1(np.nan_to_num(visible.to_numpy(float), nan=-1.0).tobytes()).hexdigest())
    if key not in model["excess_cache"]:
        s = _smoothed(visible)
        model["excess_cache"][key] = s["cam_D"] - (beta[0] + s[OTHERS].to_numpy() @ beta[1:])
    return float(model["excess_cache"][key].get(int(minute), np.nan))


def fit(training, day, context):
    return dict(final=final.fit(training, day, context), excess_beta=_fit_excess(training), excess_cache={})


def predict(case, t, model, context):
    value = final.predict(case, t, model["final"], context)
    if case.unit != HEDGE_UNIT or not np.isfinite(value):
        return value
    excess = d_excess(t, int(case.minute), model)
    if not (np.isfinite(excess) and excess > HEDGE_CUT):
        return value
    _, accepted, _ = tasks.hidden_value(t, case.unit, int(case.minute), model["final"]["cat"]["base"], context["config"],
                                        day=str(case.day), schedule=context["schedule"])
    if not np.isfinite(accepted):
        return value
    return float((1 - HEDGE_SHARE) * value + HEDGE_SHARE * accepted)
