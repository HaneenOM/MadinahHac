"""Checks on a built submission file before it is handed over.

1. Shape: the ids and their order equal data/raw/sample_submission.csv, every value is finite and not negative.
2. Mission by mission: the values in the file equal the values each model gives when it is run on its own
   (outputs of noran/track_predict.py, and the forecast blend file).
3. Moves against the accepted baseline file, per day and mission, and the ten largest single moves.

Run from the repository root:
    .venv/Scripts/python.exe -X utf8 noran/check_candidate.py v5_all_four_plus_catboost_bd \
        --hidden hidden_final --hidden-future hf_final --missing missing \
        --forecast outputs/noran/forecast_blend/w0.5/test_values.csv
Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs/noran/baseline_local/submission_baseline_local.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--hidden")
    parser.add_argument("--hidden-future", dest="hidden_future")
    parser.add_argument("--missing")
    parser.add_argument("--forecast")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    path = ROOT / "outputs/noran/candidates" / args.name / f"submission_{args.name}.csv"
    sub = pd.read_csv(path)
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    meta = test[test.id.notna()].set_index("id")
    failures = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"[{'ok' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    check("columns are id, people", list(sub.columns) == ["id", "people"])
    check("ids and order equal the sample file", sub.id.tolist() == sample.id.tolist(), f"{len(sub)} rows")
    check("values finite and not negative", bool(np.isfinite(sub.people).all() and (sub.people >= 0).all()),
          f"min {sub.people.min():.3f}, max {sub.people.max():.1f}")
    check("every id is a test target", bool(sub.id.isin(meta.index).all()))
    values = sub.set_index("id").people
    kind = meta.kind.reindex(values.index)

    sources = {"hidden": args.hidden, "hidden-future": args.hidden_future, "missing": args.missing}
    for mission, label in sources.items():
        if not label:
            continue
        own = pd.read_csv(ROOT / "outputs/noran/tracks" / label / "test_predictions.csv").set_index("id").people
        ids = kind.index[kind.eq(mission)]
        gap = (values.reindex(ids) - own.reindex(ids)).abs()
        check(f"{mission}: file equals the model run on its own ({label})",
              bool(len(own) == len(ids) and gap.notna().all() and gap.max() <= args.tolerance),
              f"{len(ids)} targets, largest difference {gap.max():.2e}")
    if args.forecast:
        own = pd.read_csv(ROOT / args.forecast).set_index("id").people
        ids = kind.index[kind.eq("forecast")]
        gap = (values.reindex(ids) - own.reindex(ids)).abs()
        check("forecast: file equals the forecast value file",
              bool(len(own) == len(ids) and gap.notna().all() and gap.max() <= args.tolerance),
              f"{len(ids)} targets, largest difference {gap.max():.2e}")

    base = pd.read_csv(BASELINE).set_index("id").people.reindex(values.index)
    frame = pd.DataFrame({"day": meta.day.reindex(values.index), "kind": kind, "unit": meta.unit.reindex(values.index),
                          "base": base, "new": values})
    frame["move"] = (frame.new - frame.base).abs()
    print()
    print("moves against the accepted baseline file")
    summary = frame.groupby(["day", "kind"]).agg(targets=("move", "size"), changed=("move", lambda s: int((s > 1e-9).sum())),
                                                 mean_base=("base", "mean"), mean_new=("new", "mean"), mean_move=("move", "mean"))
    print(summary.round(3).to_string())
    print()
    print("ten largest single moves")
    print(frame.sort_values("move", ascending=False).head(10).round(2).to_string())
    print()
    print("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
