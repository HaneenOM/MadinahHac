"""Hidden camera, week-analog member: the hidden camera's own curve from the earlier recordings (same weekday
first), aligned to today's prayer times and scaled by today's level of the visible cameras around the minute.

It reuses the hidden-future model's analog (noran/tracks/hidden_future/model.py, analog_value) at horizon 0.
The hidden mission may read every released reading, so the level window is centred on the minute:
the last 10 minutes up to minute + AHEAD.

Meant as one more member for noran/tracks/hidden_final (test: noran/blend_members.py hidden_final hidden_analog).
Contract: fit / predict as in noran/track_eval.py. Hidden columns are never read.

Run (repository root):
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/hidden_analog/model.py --set fixed --label hidden_analog
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TRACKS = Path(__file__).resolve().parents[1]
AHEAD = 5                                         # minutes after the target minute that the level window reaches


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf = _load("hidden_future_week_analog", TRACKS / "hidden_future" / "model.py")


def fit(training, day, context):
    return hf.fit(training, day, context, params=dict(ANALOG_WEIGHT=1.0))


def predict(case, t, model, context):
    minute = int(case.minute)
    end = min(minute + AHEAD, int(t.index.max()))
    value = hf.analog_value(t.loc[:end], case.unit, end, minute - end, str(case.day), model, context["schedule"])
    return float(max(0.0, value)) if np.isfinite(value) else np.nan
