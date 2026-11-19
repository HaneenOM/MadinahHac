"""Missing track: gap filling for every camera (A to M).

Run from the repository root:
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py missing noran/tracks/missing/model.py --set mirror --label missing
    .venv/Scripts/python.exe -X utf8 noran/track_eval.py missing noran/tracks/missing/model.py --set fixed --label missing
    .venv/Scripts/python.exe -X utf8 noran/track_predict.py missing noran/tracks/missing/model.py --label missing

Method (details in miss_gaplib.py, choices in miss_settings.py)
  value = line + sum of weight_k * (candidate_k - line), bounded to 50% of max(line, 5), where the line is the
  organizers' reference and the candidates are
    * the camera's own level from a random-walk (or spline) smoother of its log counts on both sides of the gap,
    * for each partner camera: the partner's reading at that minute (and one minute earlier / later for A and E)
      times the target/partner ratio, the log ratio smoothed across the gap,
    * for B and C: the accepted v2 value.
  For A to F the result is then blended with the accepted value: accepted + 0.75 * (stack - accepted).
  The smoothers weight each side by how much it can be trusted: a side with a single reading (a gap that starts
  right after the window opens) counts less than a side with many readings.
  The weights are fitted on pseudo-gaps that copy the real release pattern: every released window of the two test
  recordings (release and outage masks of all cameras) laid over the training recordings at every 3rd shift, plus
  pseudo-gaps cut inside the released test windows (the host allows fitting on released test rows). Cameras I to M
  are fitted on the released test rows only.

Inputs of a value: released readings of the case's recording (any minute, as the rules allow for `missing`),
weights fitted on the training rows passed to fit() and on released test rows. data/raw/test.csv is read in fit()
only for the positions of released rows and missing targets (it holds no answers). The held-out date is never loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for folder in (ROOT, ROOT / "src", HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import baseline_tasks as tasks  # noqa: E402
import miss_gaplib as gaplib  # noqa: E402
import miss_stack as stack  # noqa: E402
from pipeline import locate_data, table  # noqa: E402
from miss_settings import GRIDS, MAX_REL, RIDGE, SHRINK, SPEC  # noqa: E402
from v2_missing import gap_context  # noqa: E402

MARGIN = gaplib.WINDOW + 5
USES_V2 = {unit for unit, s in SPEC.items() if "v2" in s.get("extra", ())}


def pseudo_rows(training: pd.DataFrame, test_released: pd.DataFrame, config: dict, step: int = 3) -> pd.DataFrame:
    """Pseudo-gap rows for fitting: test windows over each training recording + gaps inside released test windows."""
    test = pd.read_csv(locate_data(None) / "test.csv")        # release pattern and target positions only
    windows = gaplib.release_windows(test)
    day_of = training.groupby("recording").day.first().to_dict()
    parts = []
    for name, t in table(training).items():
        d = gaplib.transplant_cases(t, name, windows, step=step, legacy=False, **GRIDS)
        if len(d):
            d["day"] = day_of[name]
            parts.append(d)
    feed = table(test_released)
    d = gaplib.window_cases([w for w in windows if w["recording"] in feed], feed, gaplib.outage_patterns(windows),
                            legacy=False, **GRIDS)
    d["day"] = "test"
    parts.append(d)
    rows = pd.concat(parts, ignore_index=True)
    # accepted value as a candidate: for a training day, from a calibration that leaves that day out
    rows["v2"] = np.nan
    for day in rows.day.unique():
        calibration = tasks.fit_missing(training if day == "test" else training[training.day.ne(day)], config)
        mask = rows.day.eq(day)
        rows.loc[mask, "v2"] = stack.v2_values(rows[mask].reset_index(drop=True), calibration, config)
    return rows


def fit(training, day, context):
    config = context["config"]
    rows = pseudo_rows(training, context["test_released"], config)
    return dict(weights=stack.fit_spec(rows, SPEC, ridge=RIDGE), calibration=tasks.fit_missing(training, config),
                config=config, fit_rows=int(len(rows)))


def predict(case, t, model, context):
    unit, minute = case.unit, int(case.minute)
    if unit not in model["weights"]:
        return np.nan                                          # keep the accepted value
    bounds = gap_context(t, unit, minute)
    if bounds is None:
        return np.nan
    start, end = bounds[0], bounds[1]
    lo = start - MARGIN
    local = t.reindex(range(lo, end + MARGIN + 1))
    visible = {u: local[u].to_numpy(float) for u in local.columns}
    rows = gaplib.gap_rows(visible, np.zeros(len(local)), unit, np.array([minute - lo]), gaplib.PARTNERS.get(unit, ()),
                           {}, legacy=False, **GRIDS)
    if len(rows) != 1:
        return np.nan
    row = pd.DataFrame(rows)
    accepted = np.nan
    if unit in USES_V2 or SHRINK.get(unit, 1.0) != 1.0:
        accepted = tasks.missing_value(t, unit, minute, model["calibration"], model["config"])[1]
        row["v2"] = accepted
    for column in SPEC[unit].get("ratios", ()):
        if column not in row:
            row[column] = np.nan
    value = stack.apply_spec(row, model["weights"], SPEC, max_rel=MAX_REL)[0]
    if np.isfinite(value) and SHRINK.get(unit, 1.0) != 1.0:
        value = stack.blend_with_accepted([unit], [value], [accepted], SHRINK)[0]
    return float(value) if np.isfinite(value) else np.nan
