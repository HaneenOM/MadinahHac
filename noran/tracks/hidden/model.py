"""Hidden-camera track: value of a camera that is hidden for the whole recording, at a released minute.

Run (repository root):
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/hidden/model.py --set mirror --label hidden
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/hidden/model.py --set fixed  --label hidden
    .venv/Scripts/python.exe -X utf8 noran/track_predict.py hidden noran/tracks/hidden/model.py --label hidden

The value is a blend, in log1p scale, of independent estimates of the hidden camera's level (see hcomp.py):
  carry   regression on smoothed donors (joint normal, EM) plus the smoothed residual of the same regression on
          the days the camera was seen, at the prayer-aligned minute (same weekday first),
  gbm     boosting on smoothed donors relative to the crowd level with a prayer-aligned clock (bootstrap average),
  tree    the team's tree model on same-minute donors (teamtree.py), fitted with the released test rows as well,
  base    the accepted baseline value (optional weight).
Fitting rows: the training rows handed over by the harness plus the released rows of the two test recordings.
At prediction the columns of the hidden cameras are dropped before anything is computed.
How the settings were chosen is written in the track's report (selection on the mirror set and on 4/5 Sep).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for folder in (HERE, ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baseline_tasks as tasks  # noqa: E402
import hcomp  # noqa: E402
import teamtree  # noqa: E402

CFG = dict(lag_max=6, bags=16, seed=11, edge=10)        # everything else: hcomp.DEFAULT
WEIGHTS = dict(carry=1.0, gbm=1.0, tree=1.0, base=0.0)  # log-scale blend weights
COLS = ["recording", "day", "unit", "minute_of_day", "people"]


def fit(training, day, context, cfg=None, weights=None):
    cfg = {**CFG, **(cfg or {})}
    weights = {**WEIGHTS, **(weights or {})}
    core = hcomp.fit_core(training, day, context, cfg)
    rows = training[COLS]
    if cfg.get("use_test_rows", True):
        rows = pd.concat([rows, context["test_released"][COLS]], ignore_index=True)
    model = dict(core=core, weights=weights, tree=teamtree.fit(rows, context["schedule"]) if weights["tree"] else None,
                 tree_cache={}, base=None)
    if weights["base"]:
        model["base"] = tasks.fit_hidden(training, context["config"], context["schedule"])
    return model


def members(case, t, model, context) -> dict:
    """Every member's value (people) for one case. Reads only the visible cameras of the feed."""
    unit, minute, day = case.unit, int(case.minute), str(case.day)
    out = {}
    comp = hcomp.components(t, unit, day, model["core"])
    if minute in comp.index:
        row = comp.loc[minute]
        out["carry"] = float(np.expm1(row["carry"]))
        out["gbm"] = float(np.expm1(row["gbm"])) if "gbm" in row else np.nan
        out["jn"] = float(np.expm1(row["jn"]))
    if model["tree"] is not None:
        visible = t.drop(columns=[c for c in hcomp.PATTERNS[unit] if c in t.columns]).reindex(columns=hcomp.A_H)
        key = (unit, day, hashlib.sha1(np.nan_to_num(visible.to_numpy(float), nan=-1.0).tobytes()).hexdigest(), len(visible))
        if key not in model["tree_cache"]:
            model["tree_cache"][key] = teamtree.predict(model["tree"], visible, unit, day, context["schedule"])
        series = model["tree_cache"][key]
        out["tree"] = float(series.get(minute, np.nan))
    if model["base"] is not None:
        _, value, _ = tasks.hidden_value(t, unit, minute, model["base"], context["config"], day=day, schedule=context["schedule"])
        out["base"] = float(value)
    return out


def predict(case, t, model, context):
    day = getattr(case, "day", None)
    if day is None or str(day) not in context["schedule"] or case.unit not in hcomp.PATTERNS:
        return np.nan                                   # no date or not a hidden camera: keep the accepted baseline value
    values = members(case, t, model, context)
    total, wsum = 0.0, 0.0
    for name, w in model["weights"].items():
        v = values.get(name, np.nan)
        if w > 0 and np.isfinite(v):
            total += w * np.log1p(max(v, 0.0))
            wsum += w
    if wsum <= 0:
        return np.nan                                   # no visible donor: keep the accepted baseline value
    return float(max(0.0, np.expm1(total / wsum)))
