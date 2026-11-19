"""Hidden camera, TabICL v2 member (library test, 18 Nov 2026).

TabICL v2 (tabicl 2.2.0, checkpoint tabicl-regressor-v2-20260212.ckpt, read from the local Hugging Face cache) is a
pretrained model for tables: it is not trained here. "Fitting" hands it the fitting rows as context, and the value
of a case is its predicted median (predict(X, output_type="median")).

Variants (environment variable NORAN_TABICL_VARIANT)
  A   the features and fitting rows of the CatBoost recipe (noran/tracks/cat/model.py): same-minute count, log
      count, share and seen flag of every visible camera, the visible total, the number of visible cameras, the
      clock, the distance to Dhuhr and to the nearest prayer, a Friday flag; the rows once as they are and once
      with 20% of the visible cameras dropped. TabICL takes no row weights, so the Friday weight of 3 is not used.
  B   A plus four columns taken from the other days of the same weekday on which the hidden camera was seen
      (on the real test days: the day one week earlier), at the prayer-aligned minute (hcomp.shift_to):
        ref_own      smoothed log count of the hidden camera on that day,
        ref_change   mean over the visible cameras of (smoothed log count today - smoothed log count that day),
        ref_carried  ref_own + ref_change,   ref_carried_count = expm1(ref_carried),
        ref_n        number of reference days that gave a value.
      The columns are blank where no such day covers the minute.

Other settings (environment variables)
  NORAN_TABICL_TARGET     log | count      the model predicts log1p(people) turned back with expm1 (default), or people
  NORAN_TABICL_ESTIMATORS number of TabICL ensemble members (default 8, the library's own default)
  NORAN_TABICL_MAXROWS    largest number of context rows; more rows are thinned at random with seed 42 (default 6000)
  NORAN_TABICL_AUG        1 = with the 20% dropout copy (default), 0 = without
  NORAN_TABICL_UNITS      cameras that get a value (default cam_B,cam_G,cam_D)

Contract: fit / predict as in noran/track_eval.py. The value is TabICL's median alone; NaN (keep the accepted value)
when fewer than two visible cameras have a reading at that minute. Fitting rows: the training rows handed over plus
the released rows of the two test recordings. The columns of the hidden cameras are dropped before any feature is
computed, and reference days never include the day of the case.

Run (repository root):
    export PYTHONPATH="$(pwd -W)"; export HF_HUB_OFFLINE=1
    .venv_lab/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/lib_hidden/model.py --set mirror --label lib_hidden_A_count
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRACKS = HERE.parent
for folder in (TRACKS / "hidden", ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from tabicl import TabICLRegressor  # noqa: E402

import hcomp  # noqa: E402  (smooth, shift_to, to_grid: prayer alignment as in the hidden track)
from pipeline import table  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cat = _load("lib_hidden_cat_recipe", TRACKS / "cat" / "model.py")       # features() and augment() of the CatBoost recipe

A_H = cat.A_H
PATTERNS = cat.PATTERNS
COLS = cat.COLS
VARIANT = os.environ.get("NORAN_TABICL_VARIANT", "A").upper()
TARGET = os.environ.get("NORAN_TABICL_TARGET", "log").lower()      # log scored better on the mirror set (0.6609 against 0.6730)
ESTIMATORS = int(os.environ.get("NORAN_TABICL_ESTIMATORS", "8"))
MAXROWS = int(os.environ.get("NORAN_TABICL_MAXROWS", "6000"))
AUG = os.environ.get("NORAN_TABICL_AUG", "1") == "1"
UNITS = tuple(u for u in os.environ.get("NORAN_TABICL_UNITS", "cam_B,cam_G,cam_D").split(",") if u)
THREADS = 4
SMOOTH = dict(half=6, decay=0.8, min_weight=0.5)          # as hcomp.DEFAULT for the donors
REF_COLUMNS = ["ref_own", "ref_change", "ref_carried", "ref_carried_count", "ref_n"]
TIMES = dict(context_rows={}, predict_seconds={}, predict_rows={})


def checkpoint_path() -> str:
    """The TabICL regressor checkpoint already on disk. Nothing is downloaded."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache("jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt")
    if not isinstance(path, str) or not Path(path).exists():
        raise FileNotFoundError("TabICL checkpoint tabicl-regressor-v2-20260212.ckpt is not in the local Hugging Face cache")
    return path


# ------------------------------------------------------------------ reference days (variant B)
def recordings(rows: pd.DataFrame, schedule) -> list:
    """Every fitting recording on a gap-free minute grid: raw and smoothed log counts of cameras A to H."""
    out = []
    rows = rows[rows.unit.isin(A_H) & rows.people.notna()]
    for rec, g in rows.groupby("recording"):
        day = str(g.day.iloc[0])
        grid = hcomp.to_grid(g.pivot(index="minute_of_day", columns="unit", values="people").sort_index())
        raw = np.log1p(grid.to_numpy().clip(min=0))
        out.append(dict(name=rec, day=day, friday=hcomp.is_friday(day), minutes=grid.index.to_numpy(int), raw=raw,
                        smooth=hcomp.smooth(raw, **SMOOTH), prayers=np.asarray(schedule[day], float)))
    return out


def reference_features(unit: str, day: str, minutes, visible: pd.DataFrame, recs: list, schedule) -> pd.DataFrame:
    """The four reference columns for the given minutes of one day. `visible` holds today's visible cameras only."""
    minutes = np.asarray(minutes, dtype=int)
    donors = [i for i, c in enumerate(A_H) if c not in PATTERNS[unit]]
    h = A_H.index(unit)
    grid = hcomp.to_grid(visible.drop(columns=[c for c in PATTERNS[unit] if c in visible.columns]))
    today = hcomp.smooth(np.log1p(grid.to_numpy().clip(min=0)), **SMOOTH)
    today = pd.DataFrame(today, index=grid.index).reindex(minutes).to_numpy()[:, donors]
    prayers = np.asarray(schedule[str(day)], float)
    friday = hcomp.is_friday(str(day))
    own_all, change_all, carried_all = [], [], []
    for ref in recs:
        if ref["day"] == str(day) or ref["friday"] != friday or np.isfinite(ref["raw"][:, h]).sum() < 20:
            continue
        there = hcomp.shift_to(minutes.astype(float), prayers, ref["prayers"]) - int(ref["minutes"][0])
        inside = (there >= 0) & (there < len(ref["minutes"]))
        if not inside.any():
            continue
        pos = np.where(inside, there, 0)
        own = np.where(inside, ref["smooth"][pos, h], np.nan)
        diff = today - np.where(inside[:, None], ref["smooth"][pos][:, donors], np.nan)
        count = np.isfinite(diff).sum(axis=1)
        change = np.where(count >= 2, np.nansum(diff, axis=1) / np.maximum(count, 1), np.nan)
        own_all.append(own)
        change_all.append(change)
        carried_all.append(own + change)
    out = pd.DataFrame(np.nan, index=np.arange(len(minutes)), columns=REF_COLUMNS)
    out["ref_n"] = 0.0
    if own_all:
        own_all, change_all, carried_all = np.vstack(own_all), np.vstack(change_all), np.vstack(carried_all)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)       # a minute that no reference day covers
            out["ref_own"] = np.nanmean(own_all, axis=0)
            out["ref_change"] = np.nanmean(change_all, axis=0)
            out["ref_carried"] = np.nanmean(carried_all, axis=0)
        out["ref_carried_count"] = np.expm1(out["ref_carried"])
        out["ref_n"] = np.isfinite(carried_all).sum(axis=0).astype(float)
    return out


# ------------------------------------------------------------------ fitting rows
def design(rows: pd.DataFrame, unit: str, schedule, variant: str, aug: bool, recs: list | None):
    """Context table of one hidden camera: features and the camera's count, as in the CatBoost recipe."""
    donors = cat.donors_of(unit)
    feeds = table(rows)
    parts, truth, days = [], [], []
    for rec, g in rows[rows.unit.eq(unit)].groupby("recording"):
        day = str(g.day.iloc[0])
        minutes = g.minute_of_day.to_numpy(int)
        x = cat.features(feeds[rec], minutes, day, schedule, donors)
        if variant == "B":
            x = pd.concat([x, reference_features(unit, day, minutes, feeds[rec], recs, schedule)], axis=1)
        ok = x.available.ge(2).to_numpy()
        parts.append(x.iloc[np.flatnonzero(ok)])
        truth.append(g.people.to_numpy(float)[ok])
        days.extend([day] * int(ok.sum()))
    if not parts:
        return None
    x = pd.concat(parts, ignore_index=True)
    y = np.concatenate(truth)
    if aug:
        base_cols = [c for c in x.columns if c not in REF_COLUMNS]
        xx, yy, _ = cat.augment(x[base_cols], y, days, donors)          # same rows and seed as the CatBoost recipe
        if variant == "B":
            # the dropped copy keeps the reference columns of its own row: rebuild which rows augment() kept
            mask = np.random.default_rng(42).random((len(x), len(donors))) < 0.2
            seen = np.column_stack([x[d + "_seen"].to_numpy() > 0 for d in donors]) & ~mask
            keep = seen.sum(axis=1) >= 2
            assert int(keep.sum()) == len(xx) - len(x), "dropout copy does not line up with the CatBoost recipe"
            assert np.allclose(yy[len(x):], y[keep]), "dropout copy does not line up with the CatBoost recipe"
            ref = pd.concat([x[REF_COLUMNS], x[REF_COLUMNS].iloc[np.flatnonzero(keep)]], ignore_index=True)
            xx = pd.concat([xx.reset_index(drop=True), ref], axis=1)
        x, y = xx, yy
    return x.astype(float), y


def fit_unit(rows: pd.DataFrame, unit: str, schedule, variant: str, recs):
    made = design(rows, unit, schedule, variant, AUG, recs)
    if made is None:
        return None
    x, y = made
    if len(x) > MAXROWS:
        pick = np.sort(np.random.default_rng(42).choice(len(x), MAXROWS, replace=False))
        x, y = x.iloc[pick].reset_index(drop=True), y[pick]
    target = np.log1p(np.maximum(y, 0.0)) if TARGET == "log" else y
    model = TabICLRegressor(n_estimators=ESTIMATORS, random_state=42, device="cpu", n_jobs=THREADS,
                            model_path=checkpoint_path(), allow_auto_download=False)
    model.fit(x.to_numpy(float), target.astype(float))
    TIMES["context_rows"][unit] = int(len(x))
    return dict(model=model, donors=cat.donors_of(unit), columns=list(x.columns), rows=int(len(x)))


def fit(training, day, context, units=None, variant=None):
    variant = VARIANT if variant is None else variant
    rows = pd.concat([training[COLS], context["test_released"][COLS]], ignore_index=True)
    recs = recordings(rows, context["schedule"]) if variant == "B" else None
    units = UNITS if units is None else units
    members = {unit: fit_unit(rows, unit, context["schedule"], variant, recs) for unit in units if unit in PATTERNS}
    return dict(members=members, recs=recs, variant=variant, cache={})


def raw_value(case, t, model, context) -> float:
    """TabICL median (people) for one case; NaN when fewer than two visible cameras have a reading at that minute."""
    unit, minute, day = case.unit, int(case.minute), str(case.day)
    member = model["members"].get(unit)
    if member is None:
        return np.nan
    visible = t.drop(columns=[c for c in PATTERNS[unit] if c in t.columns]).reindex(columns=member["donors"])
    key = (unit, day, len(visible), hashlib.sha1(np.nan_to_num(visible.to_numpy(float), nan=-1.0).tobytes()).hexdigest())
    if key not in model["cache"]:
        minutes = visible.index.to_numpy(int)
        x = cat.features(visible, minutes, day, context["schedule"], member["donors"])
        if model["variant"] == "B":
            x = pd.concat([x, reference_features(unit, day, minutes, visible, model["recs"], context["schedule"])], axis=1)
        x = x[member["columns"]]
        usable = x.available.ge(2).to_numpy()
        value = np.full(len(x), np.nan)
        if usable.any():
            started = time.perf_counter()
            torch.set_num_threads(THREADS)
            out = np.asarray(member["model"].predict(x.to_numpy(float)[usable], output_type="median"), float)
            TIMES["predict_seconds"][unit] = round(time.perf_counter() - started, 1)
            TIMES["predict_rows"][unit] = int(usable.sum())
            if TARGET == "log":
                out = np.expm1(out)
            value[usable] = np.maximum(0.0, out)
        model["cache"][key] = pd.Series(value, index=minutes)
    return float(model["cache"][key].get(minute, np.nan))


def predict(case, t, model, context):
    day = getattr(case, "day", None)
    if day is None or str(day) not in context["schedule"] or case.unit not in PATTERNS:
        return np.nan
    return raw_value(case, t, model, context)
