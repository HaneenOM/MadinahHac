"""Hidden camera, CatBoost member: the teammate's P42/P43 recipe (src/friday_catboost_trial.py on origin/main),
written once for any hidden camera so it can be scored on the shared harness.

Recipe (unchanged from P43 for camera B): CatBoost regressor, MAE loss, depth 4, l2 10, learning rate 0.05,
269 trees, Bernoulli subsample 0.8, seed 42; features at the same minute from the visible cameras only
(count, log count, share of the visible total, seen flag), the visible total, the number of visible cameras,
the clock and the distance to Dhuhr and to the nearest prayer, a Friday flag. Friday rows weigh 3. One extra
copy of the rows with 20% of the visible cameras dropped. The value is used when at least two visible cameras
have a reading at that minute.

What is general here: the visible cameras are A to H minus the hidden pattern (B and G hidden together, D alone),
so the same recipe also gives a value for G and for D.

    value = accepted + WEIGHT * (catboost - accepted)          WEIGHT 0.5 as in P43, 1.0 = CatBoost alone

Contract: fit / predict as in noran/track_eval.py. Fitting rows: the training rows handed over plus the released
rows of the two test recordings. Hidden columns are dropped before any feature is computed.

Run (repository root):
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/cat/model.py --set fixed --label cat_w50
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor, Pool  # noqa: E402

import baseline_tasks as tasks  # noqa: E402
from pipeline import table  # noqa: E402

A_H = [f"cam_{c}" for c in "ABCDEFGH"]
PATTERNS = {"cam_B": ("cam_B", "cam_G"), "cam_G": ("cam_B", "cam_G"), "cam_D": ("cam_D",)}
COLS = ["recording", "day", "unit", "minute_of_day", "people"]
PARAMS = dict(loss_function="MAE", eval_metric="MAE", learning_rate=0.05, random_seed=42, thread_count=4,
              border_count=64, random_strength=1.0, bootstrap_type="Bernoulli", subsample=0.8, nan_mode="Min",
              allow_writing_files=False, verbose=False, depth=4, l2_leaf_reg=10.0)
WEIGHT = float(os.environ.get("NORAN_CAT_WEIGHT", "0.5"))     # share of CatBoost in the value (P43: 0.5)
ITERATIONS = int(os.environ.get("NORAN_CAT_TREES", "269"))
FRIDAY_WEIGHT = 3.0
UNITS = tuple(u for u in os.environ.get("NORAN_CAT_UNITS", "cam_B,cam_G,cam_D").split(",") if u)


def donors_of(unit: str) -> list:
    return [c for c in A_H if c not in PATTERNS[unit]]


def features(feed: pd.DataFrame, minutes, day: str, schedule, donors: list) -> pd.DataFrame:
    """Same-minute features from the visible cameras. Missing readings stay missing: nothing is filled in."""
    minutes = np.asarray(minutes, dtype=int)
    raw = feed.reindex(columns=donors).reindex(minutes).reset_index(drop=True)
    out = pd.DataFrame(index=np.arange(len(minutes)))
    total = raw.sum(axis=1, min_count=1)
    for donor in donors:
        out[donor + "_count"] = raw[donor]
        out[donor + "_log"] = np.log1p(raw[donor].clip(lower=0))
        out[donor + "_share"] = raw[donor] / total.where(total.ge(1.0))
        out[donor + "_seen"] = raw[donor].notna().astype(float)
    out["visible_total"] = total
    out["available"] = raw.notna().sum(axis=1)
    m = minutes.astype(float)
    prayers = np.asarray(schedule[str(day)], dtype=float)
    out["minute"] = m
    out["clock_sin"] = np.sin(2 * np.pi * m / 1440)
    out["clock_cos"] = np.cos(2 * np.pi * m / 1440)
    out["friday"] = float(pd.Timestamp(day).day_of_week == 4)
    out["dhuhr_offset"] = m - prayers[1]
    out["nearest_prayer"] = [x - prayers[np.argmin(np.abs(prayers - x))] for x in m]
    return out.astype(float)


def augment(x: pd.DataFrame, y: np.ndarray, days, donors: list):
    """The rows once as they are and once with 20% of the visible cameras dropped (rows keeping two or more)."""
    dropped = x.copy()
    mask = np.random.default_rng(42).random((len(x), len(donors))) < 0.2
    for j, donor in enumerate(donors):
        lost = mask[:, j]
        for c in x.columns:
            if c.startswith(donor + "_"):
                dropped.loc[lost, c] = 0.0 if c.endswith("_seen") else np.nan
    total = dropped[[d + "_count" for d in donors]].sum(axis=1, min_count=1)
    dropped["visible_total"] = total
    dropped["available"] = dropped[[d + "_seen" for d in donors]].sum(axis=1)
    for donor in donors:
        dropped[donor + "_share"] = dropped[donor + "_count"] / total.where(total.ge(1.0))
    keep = dropped.available.ge(2).to_numpy()
    xx = pd.concat([x, dropped.iloc[np.flatnonzero(keep)]], ignore_index=True)
    w = np.where(pd.to_datetime(pd.Series(days)).dt.day_of_week.eq(4), FRIDAY_WEIGHT, 1.0)
    return xx, np.concatenate([y, y[keep]]), np.concatenate([w, w[keep]])


def fit_unit(rows: pd.DataFrame, unit: str, schedule, iterations: int = ITERATIONS):
    donors = donors_of(unit)
    feeds = table(rows)
    parts, truth, days = [], [], []
    for rec, g in rows[rows.unit.eq(unit)].groupby("recording"):
        day = str(g.day.iloc[0])
        minutes = g.minute_of_day.to_numpy(int)
        x = features(feeds[rec], minutes, day, schedule, donors)
        ok = x.available.ge(2).to_numpy()
        parts.append(x.iloc[np.flatnonzero(ok)])
        truth.append(g.people.to_numpy(float)[ok])
        days.extend([day] * int(ok.sum()))
    if not parts:
        return None
    x = pd.concat(parts, ignore_index=True)
    y = np.concatenate(truth)
    xx, yy, ww = augment(x, y, days, donors)
    model = CatBoostRegressor(**dict(PARAMS, iterations=int(iterations)))
    model.fit(Pool(xx, yy, weight=ww))
    return dict(model=model, donors=donors, columns=list(xx.columns), rows=len(x), days=sorted(set(days)))


def fit(training, day, context, weight=None, units=None):
    rows = pd.concat([training[COLS], context["test_released"][COLS]], ignore_index=True)
    units = UNITS if units is None else units
    members = {unit: fit_unit(rows, unit, context["schedule"]) for unit in units if unit in PATTERNS}
    base = tasks.fit_hidden(training, context["config"], context["schedule"])
    return dict(members=members, base=base, weight=WEIGHT if weight is None else float(weight), cache={})


def raw_value(case, t, model, context) -> float:
    """CatBoost value alone (people); NaN when fewer than two visible cameras have a reading at that minute."""
    unit, minute, day = case.unit, int(case.minute), str(case.day)
    member = model["members"].get(unit)
    if member is None:
        return np.nan
    visible = t.drop(columns=[c for c in PATTERNS[unit] if c in t.columns]).reindex(columns=member["donors"])
    key = (unit, day, len(visible), hashlib.sha1(np.nan_to_num(visible.to_numpy(float), nan=-1.0).tobytes()).hexdigest())
    if key not in model["cache"]:
        minutes = visible.index.to_numpy(int)
        x = features(visible, minutes, day, context["schedule"], member["donors"])[member["columns"]]
        value = np.maximum(0.0, member["model"].predict(x))
        value[~x.available.ge(2).to_numpy()] = np.nan
        model["cache"][key] = pd.Series(value, index=minutes)
    return float(model["cache"][key].get(minute, np.nan))


def predict(case, t, model, context):
    day = getattr(case, "day", None)
    if day is None or str(day) not in context["schedule"] or case.unit not in PATTERNS:
        return np.nan
    cat = raw_value(case, t, model, context)
    if not np.isfinite(cat):
        return np.nan                                   # keep the accepted value
    if model["weight"] >= 1.0:
        return float(cat)
    _, anchor, _ = tasks.hidden_value(t, case.unit, int(case.minute), model["base"], context["config"],
                                      day=str(day), schedule=context["schedule"])
    return float(max(0.0, anchor + model["weight"] * (cat - anchor)))
