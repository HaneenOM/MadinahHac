"""One entry point for a candidate version: local score on the team's fixed test, and the submission file.

A candidate is a small JSON file of task settings laid over the accepted baseline settings
(baseline_tasks.DEFAULT_CONFIG). Everything runs through the same task functions as the baseline notebook.

Usage (repository root):
    .venv/Scripts/python.exe noran/candidate.py score  noran/configs/<name>.json
    .venv/Scripts/python.exe noran/candidate.py build  noran/configs/<name>.json
    .venv/Scripts/python.exe noran/candidate.py both   noran/configs/<name>.json

score -> outputs/noran/local_score/<name>/  (scores_by_date_task.csv, score.json, predictions.csv)
build -> outputs/noran/candidates/<name>/submission_<name>.csv (+ verification.json, task_config.json)

The team's own record in outputs/local_score/ is only read (fixed cases, labels and the forecast fit cache).
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import baseline_tasks as tasks  # noqa: E402

KINDS = ("forecast", "hidden", "hidden-future", "missing")


def merged_config(path: Path) -> dict:
    """Baseline settings with the candidate's settings laid over them (one level deep per task)."""
    config = tasks.default_config()
    override = json.loads(Path(path).read_text(encoding="utf-8"))
    for task, values in override.items():
        if task.startswith("_"):
            continue
        config.setdefault(task, {})
        config[task].update(copy.deepcopy(values))
    return config


def _score_set(config: dict, name: str, which: str) -> dict:
    """Score every mission on one case set ('fixed' = the team's own, 'mirror' = masks swapped)."""
    import local_score
    from pipeline import load, locate_data, table
    from prayer_experiments import timetable

    out = ROOT / "outputs/noran/local_score" / (name if which == "fixed" else f"{name}__mirror")
    out.mkdir(parents=True, exist_ok=True)
    train, test, _, _ = load(locate_data(None))
    schedule = timetable(ROOT / "models/current_baseline/prayer_times_madinah.csv", train, test)
    if which == "fixed":
        from t05_local_forecast import predict as predict_t05_forecast
        cases, feeds = local_score.load_cases()
        t05_values, _ = predict_t05_forecast(cases, feeds)
        labels = pd.read_csv(ROOT / "outputs/local_score/labels.csv")
        accepted = pd.read_csv(ROOT / "outputs/local_score/predictions.csv")
        override_key = "local_values_csv"
    else:
        base = ROOT / "outputs/noran/mirror"
        cases = pd.concat([pd.read_csv(base / "cases.csv"), pd.read_csv(base / "forecast_cases.csv")], ignore_index=True)
        feeds = table(pd.read_csv(base / "released_feed.csv"))
        fc = pd.read_csv(base / "forecast_predictions.csv")
        t05_values = {r.case_id: (float(r.prediction) if r.source == "t05_model" else np.nan) for r in fc.itertuples()}
        labels = pd.concat([pd.read_csv(base / "labels.csv"), pd.read_csv(base / "forecast_labels.csv")], ignore_index=True)
        accepted = pd.concat([pd.read_csv(base / "predictions.csv"), fc[["case_id", "day", "kind", "reference", "prediction"]]], ignore_index=True)
        override_key = "mirror_values_csv"
    csv_path = config["forecast"].get(override_key)
    if csv_path:                                    # a forecast track's values on these cases: columns case_id, value
        override = pd.read_csv(ROOT / csv_path)
        override = override[np.isfinite(override.value)]
        unknown = set(override.case_id) - set(t05_values)
        assert not unknown, f"{len(unknown)} forecast case ids are not in the {which} set"
        t05_values.update(dict(zip(override.case_id, override.value.astype(float))))
        print(f"forecast [{which}]: {len(override)} values taken from {csv_path}")
    rows = []
    for day in local_score.DATES:
        models = local_score.fit_fold(train, day, schedule, config)
        for r in cases[cases.day.eq(day)].itertuples():
            reference, prediction, source = local_score.predict_case(r, feeds[r.recording], models, t05_values, schedule)
            rows.append(dict(case_id=r.case_id, day=day, kind=r.kind, unit=r.unit, reference=reference,
                             prediction=prediction, source=source))
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "predictions.csv", index=False)
    scored = detail.merge(labels, on="case_id", validate="one_to_one")     # answers enter only after predictions are saved
    scored = scored[np.isfinite(scored.reference)]
    scored["absolute_error"] = (scored.truth - scored.prediction).abs()
    scored["reference_absolute_error"] = (scored.truth - scored.reference).abs()
    scored.to_csv(out / "scored_predictions.csv", index=False)
    per_task = scored.groupby(["day", "kind"]).agg(predictions=("absolute_error", "size"),
                                                   mae_people=("absolute_error", "mean"),
                                                   reference_mae_people=("reference_absolute_error", "mean")).reset_index()
    per_task["normalized_mae"] = per_task.mae_people / per_task.reference_mae_people
    per_task.to_csv(out / "scores_by_date_task.csv", index=False)
    by_task = per_task.groupby("kind").normalized_mae.mean()
    overall = float(per_task.groupby("day").normalized_mae.mean().mean())
    acc = accepted.merge(labels, on="case_id")
    acc = acc[np.isfinite(acc.reference)]
    acc_task = (acc.assign(e=(acc.truth - acc.prediction).abs(), r=(acc.truth - acc.reference).abs())
                .groupby(["day", "kind"]).agg(e=("e", "mean"), r=("r", "mean")))
    baseline = float((acc_task.e / acc_task.r).groupby("day").mean().mean())
    table_ = per_task.pivot(index="kind", columns="day", values="normalized_mae").reindex(KINDS)
    table_["mean"] = table_.mean(axis=1)
    print(f"--- {which} case set")
    print(table_.round(4).to_string())
    print(f"{which.upper()} SCORE of {name}: {overall:.6f}   (accepted baseline: {baseline:.6f})")
    report = dict(name=name, case_set=which, local_score=overall, baseline_local_score=baseline,
                  by_task={k: float(by_task[k]) for k in KINDS}, config=config)
    (out / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def score(config: dict, name: str) -> float:
    """Local score on the team's fixed set, plus the mirror set when it has been built."""
    fixed = _score_set(config, name, "fixed")
    if (ROOT / "outputs/noran/mirror/forecast_predictions.csv").exists():
        mirror = _score_set(config, name, "mirror")
        both = (fixed["local_score"] + mirror["local_score"]) / 2
        both_base = (fixed["baseline_local_score"] + mirror["baseline_local_score"]) / 2
        print()
        print(f"SUMMARY {name}: fixed {fixed['local_score']:.4f} (accepted {fixed['baseline_local_score']:.4f}) | "
              f"mirror {mirror['local_score']:.4f} (accepted {mirror['baseline_local_score']:.4f}) | "
              f"mean of both {both:.4f} (accepted {both_base:.4f})")
        for kind in KINDS:
            print(f"   {kind:14s} fixed {fixed['by_task'][kind]:.4f}   mirror {mirror['by_task'][kind]:.4f}")
    return fixed["local_score"]


def build(config: dict, name: str) -> Path:
    import baseline_workflow as workflow
    from current_baseline import ASSETS

    out = ROOT / "outputs/noran/candidates" / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protected = workflow.snapshot_protected()
    project = workflow.load_project(ROOT / "data/raw")
    train, feeds, references = project["train"], project["feeds"], project["references"]
    schedule = workflow.load_schedule(project)

    forecast_models = tasks.load_forecast(ROOT / "data/raw", ASSETS, project["frozen_config"])
    forecast = workflow.forecast_output(project, forecast_models, config["forecast"])
    test_csv = config["forecast"].get("test_values_csv")
    if test_csv:                                    # a forecast track's values for the real targets: columns id, people
        override = pd.read_csv(ROOT / test_csv).set_index("id").people
        override = override[np.isfinite(override) & (override >= 0)]
        hit = forecast.id.isin(override.index)
        assert hit.sum() == len(override), "forecast override holds ids that are not forecast targets"
        new = forecast.loc[hit, "id"].map(override).astype(float)
        moved = (new - forecast.loc[hit, "people"]).abs() > 1e-9
        forecast.loc[new.index[moved], "people"] = new[moved]
        forecast.loc[new.index[moved], "people_text"] = [str(float(v)) for v in new[moved]]
        print(f"forecast: {int(moved.sum())} test values taken from {test_csv}")

    hidden_targets = workflow.task_targets(project, "hidden")
    hidden_models = tasks.fit_hidden(train, config["hidden"], schedule)
    hidden_raw = tasks.predict_hidden(hidden_models, hidden_targets, feeds, references, config["hidden"], schedule)
    hidden = workflow.hidden_output(project, hidden_raw, config["hidden"])
    if config["hidden"].get("track_module"):        # the legacy writer keeps old text for unchanged blend weights
        raw = hidden_raw.set_index("id").people.astype(float)
        new = hidden.id.map(raw)
        moved = (new - hidden.people).abs() > 1e-9
        hidden.loc[moved, "people"] = new[moved]
        hidden.loc[moved, "people_text"] = [str(float(v)) for v in new[moved]]

    hf_targets = workflow.task_targets(project, "hidden-future")
    hf_models = tasks.fit_hidden_future(train, schedule, config["hidden-future"])
    hf_raw = tasks.predict_hidden_future(hf_models, hf_targets, feeds, references, config["hidden-future"], schedule)
    hidden_future = workflow.hidden_future_output(project, hf_raw, hf_models, config["hidden-future"])

    missing_targets = workflow.task_targets(project, "missing")
    calibration = tasks.fit_missing(train, config["missing"])
    missing_raw = tasks.predict_missing(calibration, missing_targets, feeds, references, config["missing"])
    missing = workflow.missing_output(project, missing_raw)

    (out / "task_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    filename = f"submission_{name}.csv"
    if (out / filename).exists():
        (out / filename).replace(out / f"previous_{filename}")
    predictions = {"forecast": forecast, "hidden": hidden, "hidden-future": hidden_future, "missing": missing}
    workflow.save_combined(project, predictions, config, out, verify_accepted=False, filename=filename,
                           protected=protected, started=started)
    new = pd.read_csv(out / filename)
    base_path = ROOT / "outputs/noran/baseline_local/submission_baseline_local.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        meta = project["targets"].set_index("id").loc[new.id]
        diff = (new.people.to_numpy() - base.people.to_numpy())
        changed = np.abs(diff) > 1e-9
        summary = (pd.DataFrame({"day": meta.day.to_numpy(), "kind": meta.kind.to_numpy(), "changed": changed, "move": np.abs(diff)})
                   .groupby(["day", "kind"]).agg(targets=("changed", "size"), changed=("changed", "sum"), mean_move=("move", "mean")))
        print("difference from the accepted baseline file:")
        print(summary.round(3).to_string())
    print("submission file:", out / filename)
    return out / filename


def main() -> None:
    action, path = sys.argv[1], Path(sys.argv[2])
    name = path.stem
    config = merged_config(path)
    with threadpool_limits(limits=4):
        if action in ("score", "both"):
            score(config, name)
        if action in ("build", "both"):
            build(config, name)


if __name__ == "__main__":
    main()
