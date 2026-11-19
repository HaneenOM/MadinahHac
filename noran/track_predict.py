"""Run a track module (same fit/predict functions as noran/track_eval.py) on the two REAL test recordings.

Usage (repository root):
    .venv/Scripts/python.exe noran/track_predict.py <mission> <path/to/module.py> [--label NAME]

fit() is called once with every training date and day=None. predict() is called for each real target of the
mission with the released feed of its recording. Targets for which predict() returns NaN keep the accepted
baseline value. No answers exist for these targets, so nothing is scored: the script reports how far the
values move from the accepted baseline file and from the reference, per day and camera.

Output: outputs/noran/tracks/<label>/test_predictions.csv with columns id, people, used
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src", ROOT / "noran"):
    sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from pipeline import load, locate_data, table  # noqa: E402
from prayer_experiments import timetable  # noqa: E402
from track_eval import MISSIONS, load_module, make_context  # noqa: E402

BASELINE = ROOT / "outputs/noran/baseline_local/submission_baseline_local.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mission", choices=MISSIONS)
    parser.add_argument("module", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    mission, label = args.mission, args.label or args.module.stem
    out = ROOT / "outputs/noran/tracks" / label
    out.mkdir(parents=True, exist_ok=True)

    train, test, sample, _ = load(locate_data(None))
    schedule = timetable(ROOT / "models/current_baseline/prayer_times_madinah.csv", train, test)
    feeds = table(test[test.id.isna() & test.people.notna()])
    targets = test[test.id.notna() & test.kind.eq(mission)].copy()
    context = make_context(train, test, schedule, mission)
    module = load_module(args.module.resolve())
    base = pd.read_csv(BASELINE).set_index("id").people
    reference = sample.set_index("id").people

    rows = []
    with threadpool_limits(limits=2):
        model = module.fit(train.copy(), None, context)
        for r in targets.itertuples():
            now = int(r.now_minute_of_day) if pd.notna(r.now_minute_of_day) else None
            case = pd.Series(dict(case_id=r.id, day=r.day, recording=r.recording, kind=r.kind, unit=r.unit,
                                  minute=int(r.minute_of_day), now=now,
                                  horizon=(int(r.minute_of_day) - now) if now is not None else None))
            value = module.predict(case, feeds[r.recording], model, context)
            value = float(value) if value is not None else np.nan
            used = bool(np.isfinite(value))
            rows.append(dict(id=r.id, day=r.day, unit=r.unit, used=used,
                             people=max(0.0, value) if used else float(base[r.id])))
    d = pd.DataFrame(rows)
    assert len(d) == len(targets) and np.isfinite(d.people).all() and (d.people >= 0).all()
    d[["id", "people", "used"]].to_csv(out / "test_predictions.csv", index=False)
    d["baseline"] = d.id.map(base)
    d["reference"] = d.id.map(reference)
    summary = d.groupby(["day", "unit"]).agg(targets=("id", "size"), used=("used", "sum"), mean_reference=("reference", "mean"),
                                             mean_baseline=("baseline", "mean"), mean_new=("people", "mean"),
                                             move_from_baseline=("people", lambda s: float(np.abs(s - d.loc[s.index, "baseline"]).mean())),
                                             move_from_reference=("people", lambda s: float(np.abs(s - d.loc[s.index, "reference"]).mean())))
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string())
    print("wrote", out / "test_predictions.csv", f"({int(d.used.sum())} of {len(d)} targets changed)")


if __name__ == "__main__":
    main()
