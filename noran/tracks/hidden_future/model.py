"""Hidden-future track: week-analog curve of the hidden camera, scaled by today's visible cameras, blended with the accepted model.

Idea
  Every camera repeats its curve on the same weekday a week later. For a hidden camera and an origin `now`:
  1. each training recording is an "analog". Its clock is aligned to today's prayer timetable;
  2. today's level against the analog = geometric mean, over the visible cameras, of
     (camera's readings in the last K minutes at or before now) / (the analog's readings at the same aligned minutes),
     shrunk towards 1 (power BETA);
  3. forecast for now+h = that level ratio x the analog's own curve of the hidden camera (centred S-minute mean) at the
     aligned minute of now+h. Same-weekday analogs get SAME_WEIGHT times the weight of other-weekday analogs;
  4. the result is blended with the accepted hidden-future value (tree model 75% + reference 25%).
  Where no analog covers the target minute the accepted value is kept (NaN is returned).

Only readings at or before `now` are read: the feed is cut to t.loc[:now] before anything else.
Settings were chosen on dense pseudo-cases of the non-fixed recording x mask combinations (see tune_*.py), not on the fixed set.

Harness:
  .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden-future noran/tracks/hidden_future/model.py --set mirror --label hidden_future
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baseline_tasks as tasks  # noqa: E402
import hf_core as C  # noqa: E402

PARAMS = dict(
    K=10,                 # minutes of visible-camera history used for the level ratio
    S=9,                  # centred smoothing (minutes) of the analog's hidden-camera curve
    BETA=0.8,             # shrink of the level ratio towards 1
    SAME_WEIGHT=16.0,     # weight of a same-weekday analog against an other-weekday analog
    STRENGTH=(1., 1., 1., 1., 1.),   # share of each prayer's timetable shift used in the time alignment
    ANALOG_WEIGHT=0.8,    # blend: ANALOG_WEIGHT x analog + (1 - ANALOG_WEIGHT) x accepted value
    MIN_PAIRS=3,          # paired minutes a camera needs inside the window
    MIN_CAMS=3,           # visible cameras an analog needs for its level ratio
    SHIFT_MAX=4,          # timing shift searched per analog, in minutes either way (0 = off)
    SHIFT_WINDOW=20,      # minutes of history of the large cameras used to find the shift
    SHIFT_CAMS=("cam_A", "cam_E", "cam_F"),
    SHIFT_PENALTY=0.01,   # added to the shape mismatch per minute of shift
)


def full_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for rec, g in frame[frame.people.notna()].groupby("recording"):
        t = g.pivot(index="minute_of_day", columns="unit", values="people").sort_index()
        out[rec] = t.reindex(range(int(t.index.min()), int(t.index.max()) + 1))
    return out


def fit(training, day, context, params=None):
    p = dict(PARAMS, **(params or {}))
    schedule = context["schedule"]
    ref_pr = C.reference_prayers(schedule)
    rec_day = training.groupby("recording").day.first().to_dict()
    analogs = {rec: C.Analog(t, rec_day[rec], schedule, ref_pr, smooth=p["S"], strength=p["STRENGTH"])
               for rec, t in full_tables(training).items()}
    accepted = tasks.fit_hidden_future(training, schedule, context["config"]) if p["ANALOG_WEIGHT"] < 1 else None
    return dict(params=p, analogs=analogs, ref_pr=ref_pr, accepted=accepted)


def analog_value(causal: pd.DataFrame, unit: str, now: int, h: int, day: str, model, schedule) -> float:
    """Week-analog forecast of `unit` at now+h from readings at or before now (NaN when no analog covers it)."""
    p = model["params"]
    ref_pr = model["ref_pr"]
    weekday = int(pd.Timestamp(day).day_of_week)
    cams = [c for c in C.CAMS if c in causal.columns and c not in C.HIDDEN_WITH[unit]]
    wins = {c: C.window(causal, c, now, p["K"]) for c in cams}
    tau = C.align([now + h], day, schedule, ref_pr, p["STRENGTH"])
    shift_wins = ({c: C.window(causal, c, now, p["SHIFT_WINDOW"]) for c in p["SHIFT_CAMS"] if c in cams}
                  if p["SHIFT_MAX"] > 0 else {})
    num = den = 0.0
    for an in model["analogs"].values():
        if unit not in an.smooth:
            continue
        delta = float(C.best_shift(shift_wins, an, list(shift_wins), day, schedule, ref_pr, p["STRENGTH"],
                                   p["SHIFT_MAX"], p["SHIFT_PENALTY"])) if shift_wins else 0.0
        target = float(an.at(an.smooth[unit], tau + delta)[0])
        if not np.isfinite(target):
            continue
        logs = []
        for c in cams:
            ratio, _ = C.analog_ratio(*wins[c], an, c, day, schedule, ref_pr, min_pairs=p["MIN_PAIRS"], strength=p["STRENGTH"],
                                      delta=delta)
            if np.isfinite(ratio) and ratio > 0:
                logs.append(np.log(ratio))
        if len(logs) < max(1, p["MIN_CAMS"]):
            continue
        w = p["SAME_WEIGHT"] if an.weekday == weekday else 1.0
        num += w * float(np.exp(p["BETA"] * np.mean(logs))) * target
        den += w
    return num / den if den > 0 else np.nan


def predict(case, t, model, context):
    p = model["params"]
    unit, now = case.unit, int(case.now)
    h = int(case.minute) - now
    causal = t.loc[:now]                                   # nothing after `now` is read below this line
    value = analog_value(causal, unit, now, h, case.day, model, context["schedule"])
    if not np.isfinite(value):
        return np.nan                                      # keep the accepted baseline value
    if p["ANALOG_WEIGHT"] >= 1:
        return max(0.0, value)
    _, accepted, _ = tasks.hidden_future_value(causal, unit, now, h, case.day, context["schedule"], model["accepted"], context["config"])
    return max(0.0, p["ANALOG_WEIGHT"] * value + (1 - p["ANALOG_WEIGHT"]) * accepted)
