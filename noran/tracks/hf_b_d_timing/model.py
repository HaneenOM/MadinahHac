"""Exploratory B-only timing transfer from its existing reference donor D.

V7 amplitude, smoothing, blend, maximum shift, penalty and plateau rule stay intact.
D's trailing visible shape selects the analog's timing shift for B. No hidden
history or future inference observations are used. G and D stay on V7.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / 'src'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location('prompt54_original_hf', ROOT / 'noran/tracks/hf_final/model.py')
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

def fit(training, day, context):
    return v7.fit(training, day, context)

def supported(case, t, model, context):
    if case.unit != 'cam_B':
        return False
    now = int(case.now)
    prayers = context['schedule'][case.day]
    if int(prayers[2])+55 <= now <= int(prayers[3])-35:
        return False
    # best_shift's minimum-pairs requirement; otherwise keep V7's timing.
    minutes, _ = v7.base.C.window(t.loc[:now], 'cam_D', now, model['params']['SHIFT_WINDOW'])
    return len(minutes) >= 5

def predict(case, t, model, context):
    if not supported(case, t, model, context):
        return v7.predict(case, t, model, context)
    adjusted = dict(model, params=dict(model['params'], SHIFT_CAMS=('cam_D',)))
    return v7.predict(case, t, adjusted, context)
