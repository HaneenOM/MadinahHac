"""Evaluation helpers for the hidden track (NOT used by model.py at prediction time).

Scenarios
  mirror   the harness's swapped-mask cases (11 Sep with D hidden, 12 Sep with B and G hidden), read from
           outputs/noran/mirror (cases, feed, saved baseline and reference). Selection set.
  pseudo   4 Sep and 5 Sep under BOTH real masks, taken at the same clock minutes (the 04:26 recording is
           shifted by +420 minutes so that it falls inside the mask; it is reported apart as 'morning').
           Selection set. Reference and accepted baseline are rebuilt with the team's own functions
           (baseline_tasks.fit_hidden / hidden_value) fitted without the held-out date.
  fixed    the team's local test (outputs/local_score). Confirmation only.

Answers are joined only inside score(); a method never sees them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import contextlib  # noqa: E402
import io  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baseline_tasks as tasks  # noqa: E402
import v2_hidden  # noqa: E402
from pipeline import CAMERAS, load, locate_data, table  # noqa: E402
from prayer_experiments import timetable  # noqa: E402

LOCAL = ROOT / "outputs/local_score"
MIRROR = ROOT / "outputs/noran/mirror"
FRI_MASK, SAT_MASK = "20260918_1100", "20260919_1100"
HIDDEN_OF = {FRI_MASK: ("cam_B", "cam_G"), SAT_MASK: ("cam_D",)}


def load_all():
    with contextlib.redirect_stdout(io.StringIO()):
        train, test, sample, _ = load(locate_data(None))
        schedule = timetable(ROOT / "models/current_baseline/prayer_times_madinah.csv", train, test)
    released = test[test.id.isna() & test.people.notna()][["recording", "day", "weekday", "unit", "minute", "minute_of_day", "people"]].copy()
    context = {"schedule": schedule, "test_released": released, "config": tasks.default_config()["hidden"], "mission": "hidden"}
    return train, test, sample, schedule, context


def saved_set(which: str):
    base = LOCAL if which == "fixed" else MIRROR
    cases = pd.read_csv(base / "cases.csv")
    cases = cases[cases.kind.eq("hidden")].copy()
    feeds = table(pd.read_csv(base / "released_feed.csv"))
    saved = pd.read_csv(base / "predictions.csv")
    saved = saved[saved.kind.eq("hidden")][["case_id", "reference", "prediction"]].rename(columns={"prediction": "baseline"})
    cases = cases.merge(saved, on="case_id", validate="one_to_one")
    labels = pd.read_csv(base / "labels.csv")
    labels = labels[labels.case_id.isin(cases.case_id)]
    cases["scenario"] = which + ":" + cases.day.str[-2:] + ":" + np.where(cases.unit.eq("cam_D"), "D", "BG")
    return cases, feeds, labels


def pseudo_set(train, test, schedule):
    """4 and 5 Sep under both real masks. Returns cases (with reference and baseline), feeds keyed by a
    scenario-specific recording name, and labels."""
    case_rows, labels, feeds = [], [], {}
    hidden_models = {}
    for rec in ("20260904_1105", "20260905_0426", "20260905_1124", "20260905_1900"):
        truth = train[train.recording.eq(rec)]
        day = truth.day.iloc[0]
        shift = 420 if rec.endswith("0426") else 0
        if day not in hidden_models:
            hidden_models[day] = tasks.fit_hidden(train[train.day.ne(day)], tasks.default_config()["hidden"], schedule)
        for mask_rec in (FRI_MASK, SAT_MASK):
            mask = test[test.recording.eq(mask_rec) & test.unit.isin(CAMERAS)].copy()
            mask["minute_of_day"] = mask.minute_of_day - shift
            hidden = HIDDEN_OF[mask_rec]
            name = f"{rec}|{'BG' if len(hidden) == 2 else 'D'}"
            released = mask[mask.id.isna()][["unit", "minute_of_day"]]
            feed = released.merge(truth[["unit", "minute_of_day", "people"]], on=["unit", "minute_of_day"], how="inner")
            assert not feed.unit.isin(hidden).any()
            t = feed.pivot(index="minute_of_day", columns="unit", values="people").sort_index()
            feeds[name] = t
            targets = mask[mask.id.notna() & mask.kind.eq("hidden")][["unit", "minute_of_day"]].merge(
                truth[["unit", "minute_of_day", "people"]], on=["unit", "minute_of_day"], how="inner")
            tag = ("morning" if shift else "pseudo") + ":" + day[-2:] + ("am" if shift else "") + \
                  (rec[-4:] if day.endswith("05") and not shift else "") + ":" + ("BG" if len(hidden) == 2 else "D")
            for r in targets.itertuples():
                minute = int(r.minute_of_day)
                reference, baseline, _ = tasks.hidden_value(t, r.unit, minute, hidden_models[day],
                                                            tasks.default_config()["hidden"], day=day, schedule=schedule)
                case_id = f"{name}|{r.unit}|{minute}"
                case_rows.append(dict(case_id=case_id, day=day, recording=name, kind="hidden", unit=r.unit, minute=minute,
                                      now=np.nan, horizon=np.nan, reference=reference, baseline=baseline, scenario=tag))
                labels.append(dict(case_id=case_id, truth=float(r.people)))
    return pd.DataFrame(case_rows), feeds, pd.DataFrame(labels)


def selection_sets(train, test, schedule, with_fixed=False):
    parts = [saved_set("mirror"), pseudo_set(train, test, schedule)]
    if with_fixed:
        parts.append(saved_set("fixed"))
    cases = pd.concat([p[0] for p in parts], ignore_index=True)
    feeds = {}
    for p in parts:
        for k, v in p[1].items():
            key = k if k not in feeds else k + "#fixed"
            feeds[key] = v
    # the fixed set and the mirror set share recording names: give fixed cases their own feed key
    if with_fixed:
        fixed = cases.scenario.str.startswith("fixed")
        cases.loc[fixed, "feed_key"] = cases.loc[fixed, "recording"] + "#fixed"
    if "feed_key" not in cases:
        cases["feed_key"] = np.nan
    cases["feed_key"] = cases.feed_key.fillna(cases.recording)
    labels = pd.concat([p[2] for p in parts], ignore_index=True)
    return cases, feeds, labels


def run(module, cases, feeds, train, context, fit_cache=None):
    """Same contract as noran/track_eval.py: fit once per held-out date, predict each case."""
    out = []
    fit_cache = {} if fit_cache is None else fit_cache
    for day, today in cases.groupby("day"):
        if day not in fit_cache:
            fit_cache[day] = module.fit(train[train.day.ne(day)].copy(), day, context)
        model = fit_cache[day]
        for r in today.itertuples():
            real_rec = r.recording.split("|")[0]
            case = r._replace(recording=real_rec) if hasattr(r, "_replace") else r
            value = module.predict(case, feeds[r.feed_key], model, context)
            out.append(float(value) if value is not None and np.isfinite(value) else float(r.baseline))
    return np.maximum(0.0, np.asarray(out))


def score(cases, labels, columns):
    """Normalized MAE (MAE / reference MAE) per scenario and camera for each prediction column."""
    d = cases.merge(labels, on="case_id", validate="one_to_one")
    rows = []
    for (scenario, unit), g in d.groupby(["scenario", "unit"], sort=False):
        ref = float((g.reference - g.truth).abs().mean())
        row = dict(scenario=scenario, unit=unit[-1], n=len(g), ref_mae=ref)
        for c in columns:
            row[c] = float((g[c] - g.truth).abs().mean()) / ref
        rows.append(row)
    return pd.DataFrame(rows)


def summary(table_, columns):
    """Mean of the normalized scores over groups of scenarios; the harness metric for mirror and fixed."""
    out = {}
    t = table_.copy()
    t["family"] = t.scenario.str.split(":").str[0]
    for fam, g in t.groupby("family", sort=False):
        out[fam + " (mean of camera-days)"] = g[columns].mean()
    return pd.DataFrame(out).T


def harness_metric(cases, labels, column, which):
    """Exactly the harness metric: mean over the two dates of pooled MAE / pooled reference MAE."""
    d = cases[cases.scenario.str.startswith(which)].merge(labels, on="case_id", validate="one_to_one")
    vals = []
    for _, g in d.groupby("day"):
        vals.append((g[column] - g.truth).abs().mean() / (g.reference - g.truth).abs().mean())
    return float(np.mean(vals))
