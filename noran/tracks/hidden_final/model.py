"""Hidden camera, combined version: the hidden track (noran/tracks/hidden/model.py) plus the teammate's CatBoost
recipe (noran/tracks/cat/model.py) as a fourth member, only for the cameras where it helped.

    value = expm1((1 - s) * log1p(hidden track) + s * log1p(CatBoost))        s = CAT_SHARE[camera]

Why these shares (noran/blend_members.py hidden:hidden_main cat_w100, 18 Nov 2026): the hidden track averages three
members in log scale, so a fourth member with equal weight has s = 0.25. With s = 0.25 camera D improved on all
three case sets, camera B improved on two and stayed level on the third, and camera G got worse on all three.
So CatBoost joins for B and D and stays out for G.

Contract: fit / predict as in noran/track_eval.py. Hidden columns are never read (both parts drop them).
The version with the checker's hedge for camera B is noran/tracks/hidden_hedged/model.py; it wraps this file.

Run (repository root):
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/hidden_final/model.py --set fixed --label hidden_final
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TRACKS = Path(__file__).resolve().parents[1]
CAT_SHARE = {"cam_B": 0.25, "cam_D": 0.25, "cam_G": 0.0}


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("hidden_track_three_members", TRACKS / "hidden" / "model.py")
cat = _load("hidden_track_catboost_member", TRACKS / "cat" / "model.py")


def fit(training, day, context):
    units = [unit for unit, share in CAT_SHARE.items() if share > 0]
    return dict(base=base.fit(training, day, context),
                cat=cat.fit(training, day, context, weight=1.0, units=units))


def predict(case, t, model, context):
    value = base.predict(case, t, model["base"], context)
    share = CAT_SHARE.get(case.unit, 0.0)
    if share <= 0 or not np.isfinite(value):
        return value
    extra = cat.raw_value(case, t, model["cat"], context)
    if not np.isfinite(extra):
        return value
    return float(max(0.0, np.expm1((1 - share) * np.log1p(max(value, 0.0)) + share * np.log1p(max(extra, 0.0)))))
