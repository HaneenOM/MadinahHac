"""Hidden camera, version 3 of the combined model.

    step 1  noran/tracks/hidden_final      the hidden track's three members, plus CatBoost for cameras B and D
    step 2  week-analog member             noran/tracks/hidden_analog, joined with the weight of one equal member
    step 3  hedge for camera B             noran/tracks/hidden_hedged: mean with the accepted baseline value in the
                                           minutes where camera D stands far above cameras A, C, F and H

Step 2, why (noran/blend_members.py hidden_final hidden_analog, 18 Nov 2026):
  the week-analog is the hidden camera's own curve from the same weekday one week earlier, scaled by today's level of
  the visible cameras. As one more equal member (share 0.2 where there are four members, 0.25 where there are three)
  it lowered the team metric on all three case sets (fixed 0.7157 to 0.6976, mirror 0.6646 to 0.6588, test day 0.5695
  to 0.5654). Per camera: D and G improved on all three sets. B improved on the two sets where D is visible, which is
  B's real situation (9 of 12 released blocks better on each), and got worse on the test-day set, where D is absent
  and the level has to come from cameras that follow B less.

Step 3, why: see noran/tracks/hidden_hedged/model.py (the independent checker's request).

Contract: fit / predict as in noran/track_eval.py. Hidden columns are never read.

Run (repository root):
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/hidden_v3/model.py --set fixed --label hidden_v3
Environment switches for tests: NORAN_V3_ANALOG=0 turns step 2 off, NORAN_V3_HEDGE=0 turns step 3 off.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

TRACKS = Path(__file__).resolve().parents[1]
ANALOG_SHARE = {"cam_B": 0.2, "cam_D": 0.2, "cam_G": 0.25}       # one equal member among five (B, D) or four (G)
ANALOG_ON = os.environ.get("NORAN_V3_ANALOG", "1") != "0"
HEDGE_ON = os.environ.get("NORAN_V3_HEDGE", "1") != "0"


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hedged = _load("hidden_hedged_for_v3", TRACKS / "hidden_hedged" / "model.py")
analog = _load("hidden_analog_for_v3", TRACKS / "hidden_analog" / "model.py")
final = hedged.final


def fit(training, day, context):
    return dict(final=final.fit(training, day, context),
                analog=analog.fit(training, day, context) if ANALOG_ON else None,
                excess_beta=hedged._fit_excess(training) if HEDGE_ON else None, excess_cache={})


def predict(case, t, model, context):
    value = final.predict(case, t, model["final"], context)
    if not np.isfinite(value):
        return value
    share = ANALOG_SHARE.get(case.unit, 0.0)
    if model["analog"] is not None and share > 0:
        extra = analog.predict(case, t, model["analog"], context)
        if np.isfinite(extra):
            value = float(max(0.0, np.expm1((1 - share) * np.log1p(max(value, 0.0)) + share * np.log1p(max(extra, 0.0)))))
    if HEDGE_ON and case.unit == hedged.HEDGE_UNIT:
        excess = hedged.d_excess(t, int(case.minute), model)
        if np.isfinite(excess) and excess > hedged.HEDGE_CUT:
            _, accepted, _ = hedged.tasks.hidden_value(t, case.unit, int(case.minute), model["final"]["cat"]["base"],
                                                       context["config"], day=str(case.day), schedule=context["schedule"])
            if np.isfinite(accepted):
                value = float((1 - hedged.HEDGE_SHARE) * value + hedged.HEDGE_SHARE * accepted)
    return value
