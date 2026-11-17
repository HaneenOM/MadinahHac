"""Build T05 forecasts with freshly generated v2 predictions for other tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import locate_data  # noqa: E402
from v2_candidate import build as build_v2  # noqa: E402
from t05_forecast.submit import build as build_t05  # noqa: E402

KINDS = ("forecast", "hidden", "hidden-future", "missing")


def checked_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "people"]:
        raise ValueError(f"{path}: expected exactly id,people")
    if len(df) != 4457 or not df.id.is_unique or df.id.tolist() != sample.id.tolist():
        raise ValueError(f"{path}: target IDs or order do not match the 4,457-row sample")
    if not np.isfinite(df.people).all() or (df.people < 0).any():
        raise ValueError(f"{path}: predictions must be finite and nonnegative")
    return df.set_index("id")


def compare(label: str, actual: pd.DataFrame, expected: pd.DataFrame, kinds: pd.Series) -> dict:
    difference = (actual.people - expected.people).abs()
    report = {
        "maximum_absolute_difference": float(difference.max()),
        "mismatches_over_1e-9": int((difference > 1e-9).sum()),
        "by_task": {},
    }
    for kind in KINDS:
        values = difference.loc[kinds.index[kinds == kind]]
        report["by_task"][kind] = {
            "targets": len(values),
            "maximum_absolute_difference": float(values.max()),
            "mismatches_over_1e-9": int((values > 1e-9).sum()),
        }
    print(f"{label}: max={report['maximum_absolute_difference']:.17g}, "
          f"mismatches >1e-9={report['mismatches_over_1e-9']}", flush=True)
    for kind, row in report["by_task"].items():
        print(f"  {kind}: {row['targets']} targets, max={row['maximum_absolute_difference']:.17g}, "
              f"mismatches={row['mismatches_over_1e-9']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, help="T05 supplied submission.csv to compare")
    parser.add_argument("--t05-base", type=Path, help="T05 supplied v2 base CSV to compare")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/submission_t05_integrated.csv")
    args = parser.parse_args()

    data_dir = locate_data(None)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    target = pd.read_csv(data_dir / "test.csv").dropna(subset=["id"])
    kinds = target.set_index("id").loc[sample.id, "kind"]
    generated_v2_path = ROOT / "outputs/t05_v2_regenerated.csv"
    build_v2(generated_v2_path)
    generated_v2 = checked_submission(generated_v2_path, sample)
    reports = {}
    saved_v2 = checked_submission(ROOT / "outputs/submission_v2.csv", sample)
    reports["v2_saved"] = compare("Regenerated v2 versus saved v2", generated_v2, saved_v2, kinds)
    if reports["v2_saved"]["mismatches_over_1e-9"]:
        raise AssertionError("Regenerated v2 differs from saved v2; see task breakdown above")
    if args.t05_base:
        package_base = checked_submission(args.t05_base, sample)
        reports["v2_t05_base"] = compare("Regenerated v2 versus T05 base", generated_v2, package_base, kinds)
        if reports["v2_t05_base"]["mismatches_over_1e-9"]:
            raise AssertionError("Regenerated v2 differs from T05 base; see task breakdown above")

    cfg = json.loads((ROOT / "t05_forecast_config.json").read_text(encoding="utf-8"))
    cfg["base_submission"] = str(generated_v2_path)
    work_dir = ROOT / "outputs/t05_work"
    build_t05(data_dir, ROOT / "outputs/t05_cache", work_dir, cfg, plots=False)
    generated = checked_submission(work_dir / "submission.csv", sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated.reset_index().to_csv(args.output, index=False)
    reread = checked_submission(args.output, sample)
    if not np.allclose(generated.people, reread.people, rtol=0, atol=0):
        raise AssertionError("Saved output differs from generated predictions")
    print(f"Wrote {args.output}: {len(reread)} predictions", flush=True)

    if args.reference:
        supplied = checked_submission(args.reference, sample)
        reports["t05_supplied"] = compare("Integrated versus supplied T05", reread, supplied, kinds)
    report_path = ROOT / "outputs/t05_integration_verification.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Verification saved to {report_path}", flush=True)
    if args.reference and reports["t05_supplied"]["mismatches_over_1e-9"]:
        raise AssertionError("Integrated output differs from supplied T05; see task breakdown above")


if __name__ == "__main__":
    main()
