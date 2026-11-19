"""Run the accepted baseline exactly as notebooks/current_baseline.ipynb does and save the result as a
separately named local file (the notebook refuses to save when the hash differs from the accepted one).

Usage (from the repository root):
    .venv/Scripts/python.exe noran/run_baseline_local.py [output_dir]

Writes <output_dir>/submission_baseline_local.csv, task_config.json and verification.json.
Nothing in outputs/current_baseline_manual or models/ is modified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(folder))

from threadpoolctl import threadpool_limits  # noqa: E402

import baseline_tasks as tasks  # noqa: E402
import baseline_workflow as workflow  # noqa: E402
from current_baseline import ASSETS  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/noran/baseline_local"
    out.mkdir(parents=True, exist_ok=True)
    threadpool_limits(limits=4)
    started = time.perf_counter()
    protected = workflow.snapshot_protected()
    project = workflow.load_project(ROOT / "data/raw")
    train, feeds = project["train"], project["feeds"]
    config = tasks.default_config()
    schedule = workflow.load_schedule(project)
    references = project["references"]

    forecast_models = tasks.load_forecast(ROOT / "data/raw", ASSETS, project["frozen_config"])
    forecast_predictions = workflow.forecast_output(project, forecast_models, config["forecast"])

    hidden_targets = workflow.task_targets(project, "hidden")
    hidden_models = tasks.fit_hidden(train, config["hidden"], schedule)
    hidden_raw = tasks.predict_hidden(hidden_models, hidden_targets, feeds, references, config["hidden"], schedule)
    hidden_predictions = workflow.hidden_output(project, hidden_raw, config["hidden"])

    hf_targets = workflow.task_targets(project, "hidden-future")
    hf_models = tasks.fit_hidden_future(train, schedule, config["hidden-future"])
    hf_raw = tasks.predict_hidden_future(hf_models, hf_targets, feeds, references, config["hidden-future"], schedule)
    hf_predictions = workflow.hidden_future_output(project, hf_raw, hf_models, config["hidden-future"])

    missing_targets = workflow.task_targets(project, "missing")
    missing_calibration = tasks.fit_missing(train, config["missing"])
    missing_raw = tasks.predict_missing(missing_calibration, missing_targets, feeds, references, config["missing"])
    missing_predictions = workflow.missing_output(project, missing_raw)

    (out / "task_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    task_predictions = {"forecast": forecast_predictions, "hidden": hidden_predictions,
                        "hidden-future": hf_predictions, "missing": missing_predictions}
    verification = workflow.save_combined(project, task_predictions, config, out, verify_accepted=False,
                                          filename="submission_baseline_local.csv", protected=protected,
                                          started=started)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
