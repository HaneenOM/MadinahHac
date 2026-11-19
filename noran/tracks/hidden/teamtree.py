"""The team's tree model for a hidden camera, copied from src/hidden_reconstruction_trial.py (make_features, fit_new,
PARAMS) so that this track does not depend on that script's imports. Logic and settings are unchanged; the only
difference is that fit() accepts any frame of fitting rows (this track adds the released test rows).

Inputs of the model: exact same-minute visible A-H log levels, shares of the visible total, seen flags, clock,
Friday flag and scheduled prayer offsets. One 20% donor-dropout copy of the fitting rows. No reading of the hidden
cameras of the pattern ever enters a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

UNITS = [f"cam_{c}" for c in "ABCDEFGH"]
PATTERNS = {"cam_B": {"cam_B", "cam_G"}, "cam_G": {"cam_B", "cam_G"}, "cam_D": {"cam_D"}}
PARAMS = dict(loss="absolute_error", max_iter=75, learning_rate=.07, max_leaf_nodes=7,
              max_depth=3, min_samples_leaf=25, l2_regularization=15., early_stopping=False, random_state=26)


def make_features(frame: pd.DataFrame, unit: str, days, timetable):
    donors = [u for u in UNITS if u not in PATTERNS[unit]]
    x = frame.reindex(columns=donors).astype(float)
    available = x.notna().sum(axis=1)
    total = x.sum(axis=1, min_count=1)
    result = pd.DataFrame(index=frame.index)
    for c in donors:
        result[c + "_log"] = np.log1p(x[c].clip(lower=0))
        result[c + "_share"] = x[c] / total.replace(0, np.nan)
        result[c + "_seen"] = x[c].notna().astype(float)
    minute = np.asarray(frame.index.get_level_values(-1), int)
    result["visible_total_log"] = np.log1p(total.clip(lower=0))
    result["available"] = available
    result["clock_sin"] = np.sin(2 * np.pi * minute / 1440)
    result["clock_cos"] = np.cos(2 * np.pi * minute / 1440)
    result["friday"] = [int(pd.Timestamp(day).day_of_week == 4) for day in days]
    result["dhuhr_offset"] = [int(m - timetable[d][1]) for m, d in zip(minute, days)]
    result["nearest_prayer"] = [int(m - timetable[d][np.argmin(np.abs(timetable[d] - m))]) for m, d in zip(minute, days)]
    return result, available


def fit(rows: pd.DataFrame, timetable) -> dict:
    """rows: recording, day, unit, minute_of_day, people (cameras outside A-H are ignored)."""
    rows = rows[rows.unit.isin(UNITS) & rows.people.notna()]
    tables = {rec: g.pivot(index="minute_of_day", columns="unit", values="people").sort_index() for rec, g in rows.groupby("recording")}
    full = pd.concat(tables.values(), keys=tables.keys(), names=["recording", "minute_of_day"])
    rec_day = rows.drop_duplicates("recording").set_index("recording").day.to_dict()
    models = {}
    for unit, mask in PATTERNS.items():
        donors = [u for u in UNITS if u not in mask]
        if unit not in full:
            continue
        valid = full[unit].notna() & full.reindex(columns=donors).notna().sum(axis=1).ge(2)
        raw = full.loc[valid].reindex(columns=donors)
        if len(raw) < 100:
            continue
        days = [rec_day[n] for n in raw.index.get_level_values(0)]
        x, _ = make_features(raw, unit, days, timetable)
        rng = np.random.default_rng(26)
        dropped = raw.mask(rng.random(raw.shape) < .2)
        augmented, seen = make_features(dropped, unit, days, timetable)
        keep = seen.ge(2)
        design = pd.concat([x, augmented[keep]], ignore_index=True)
        target = full.loc[valid, unit].to_numpy(float)
        y = np.concatenate([target, target[keep.to_numpy()]])
        models[unit] = HistGradientBoostingRegressor(**PARAMS).fit(design, y)
    return models


def predict(models: dict, visible: pd.DataFrame, unit: str, day: str, timetable) -> pd.Series:
    """Value for every minute of the feed (people). `visible` must already be without the hidden cameras."""
    if unit not in models:
        return pd.Series(np.nan, index=visible.index)
    x, available = make_features(visible, unit, [day] * len(visible), timetable)
    value = np.maximum(models[unit].predict(x), 0.0)
    return pd.Series(np.where(available.to_numpy() >= 2, value, np.nan), index=visible.index)
