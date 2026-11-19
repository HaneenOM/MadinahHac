"""Assemble a submission file from an existing one by swapping in the values of one or more missions.

Each mission's values come from a model run on its own (outputs of noran/track_predict.py), so the result is the
same file noran/candidate.py build would write, without refitting the missions that did not change.
Rows that are not swapped keep their text exactly as it is in the source file.

Run from the repository root:
    .venv/Scripts/python.exe -X utf8 noran/assemble.py <new name> --from <existing name> \
        [--hidden LABEL] [--hidden-future LABEL] [--missing LABEL] [--forecast path/to/values.csv]
LABEL = a folder under outputs/noran/tracks/ holding test_predictions.csv (id, people).
Output: outputs/noran/candidates/<new name>/submission_<new name>.csv and assembly.json (what came from where).
Check it afterwards with noran/check_candidate.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "outputs/noran/candidates"
TRACKS = ROOT / "outputs/noran/tracks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--from", dest="source", required=True)
    parser.add_argument("--hidden")
    parser.add_argument("--hidden-future", dest="hidden_future")
    parser.add_argument("--missing")
    parser.add_argument("--forecast")
    args = parser.parse_args()

    source = CANDIDATES / args.source / f"submission_{args.source}.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and list(rows[0].keys()) == ["id", "people"], "source file must have the columns id, people"
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    kind = test[test.id.notna()].set_index("id").kind
    swaps = {"hidden": args.hidden, "hidden-future": args.hidden_future, "missing": args.missing}
    record = {"name": args.name, "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "swapped": {}}
    new_text = {}
    for mission, label in swaps.items():
        if not label:
            continue
        path = TRACKS / label / "test_predictions.csv"
        values = pd.read_csv(path).set_index("id").people.astype(float)
        ids = kind.index[kind.eq(mission)]
        assert set(values.index) == set(ids), f"{label}: ids do not match the {mission} targets"
        assert np.isfinite(values).all() and (values >= 0).all(), f"{label}: values must be finite and not negative"
        new_text.update({i: str(float(v)) for i, v in values.items()})
        record["swapped"][mission] = {"from": str(path.relative_to(ROOT)), "targets": int(len(ids)),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.forecast:
        path = ROOT / args.forecast
        values = pd.read_csv(path).set_index("id").people.astype(float)
        ids = kind.index[kind.eq("forecast")]
        assert set(values.index) == set(ids), "forecast file: ids do not match the forecast targets"
        assert np.isfinite(values).all() and (values >= 0).all()
        new_text.update({i: str(float(v)) for i, v in values.items()})
        record["swapped"]["forecast"] = {"from": args.forecast, "targets": int(len(ids)),
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert new_text, "nothing to swap: name at least one mission"

    changed = 0
    for row in rows:
        text = new_text.get(row["id"])
        if text is not None and abs(float(text) - float(row["people"])) > 1e-12:
            row["people"] = text
            changed += 1
    out = CANDIDATES / args.name
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"submission_{args.name}.csv"
    assert not target.exists(), f"{target} exists already: choose a new name"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "people"])
        writer.writeheader()
        writer.writerows(rows)
    record.update(rows=len(rows), rows_changed=changed, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out / "assembly.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"written: {target}  ({changed} of {len(rows)} rows changed against {args.source})")


if __name__ == "__main__":
    main()
