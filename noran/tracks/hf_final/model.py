"""Hidden-future, combined version: the week-analog model of noran/tracks/hidden_future/model.py, with the one
change its independent checker recommended.

Change: for origins (`now`) in the late afternoon, from 55 minutes after Asr to 35 minutes before Maghrib,
the accepted value is kept. In that slot the crowd sits on a plateau and last week's curve added error in
every evidence set the checker looked at (fixed set, mirror set, dense pseudo-cases on the long days, and the
released rows of the two test recordings with a camera hidden). The rule was found after looking at the
fixed set, so it is recorded as post-hoc; its independent support is the test-day released rows.

Contract: fit / predict as in noran/track_eval.py. Only readings at or before `now` are read.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1] / "hidden_future" / "model.py"
AFTER_ASR, BEFORE_MAGHRIB = 55, 35          # minutes; schedule[day] = [fajr, dhuhr, asr, maghrib, isha]


def _load_base():
    if str(BASE.parent) not in sys.path:
        sys.path.insert(0, str(BASE.parent))
    spec = importlib.util.spec_from_file_location("hf_track_week_analog", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def fit(training, day, context):
    return base.fit(training, day, context)


def predict(case, t, model, context):
    prayers = context["schedule"][case.day]
    asr, maghrib = int(prayers[2]), int(prayers[3])
    if asr + AFTER_ASR <= int(case.now) <= maghrib - BEFORE_MAGHRIB:
        return np.nan                              # late-afternoon plateau: keep the accepted value
    return base.predict(case, t, model, context)
