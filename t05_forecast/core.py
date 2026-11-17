"""Core data structures for the Madinah crowd table.

One `Recording` holds a minute-by-camera table for one stretch of footage:
released values, what kind of hole each empty cell is, and (for forecast-type
holes) the minute `now` after which nothing may be used as an input.

Only numpy and pandas are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# cell status codes
ABSENT, RELEASED, FORECAST, HIDDEN, HIDDEN_FUTURE, MISSING = 0, 1, 2, 3, 4, 5
KIND_CODE = {"forecast": FORECAST, "hidden": HIDDEN, "hidden-future": HIDDEN_FUTURE, "missing": MISSING}
KIND_NAME = {v: k for k, v in KIND_CODE.items()}
KINDS = ["forecast", "hidden", "hidden-future", "missing"]
ALL_UNITS = [f"cam_{c}" for c in "ABCDEFGHIJKLM"]


@dataclass
class Recording:
    name: str
    day: str
    weekday: str
    first: int                      # minute of day of row 0
    units: list                     # camera labels, column order
    y: np.ndarray                   # [T, U] released people counts, NaN elsewhere
    status: np.ndarray              # [T, U] cell status code
    now: np.ndarray                 # [T, U] minute-of-day of `now` for forecast / hidden-future holes, -1 elsewhere
    truth: np.ndarray | None = None # [T, U] true values of holes when known (validation only)
    ids: dict = field(default_factory=dict)  # (row, col) -> submission id (test only)
    split: str = "train"

    @property
    def n_minutes(self) -> int:
        return self.y.shape[0]

    @property
    def minutes(self) -> np.ndarray:
        return self.first + np.arange(self.n_minutes)

    def col(self, unit: str) -> int:
        return self.units.index(unit)

    def holes(self, kind: str | None = None) -> list:
        """List of (row, col) of holes, optionally of one kind."""
        if kind is None:
            mask = self.status >= FORECAST
        else:
            mask = self.status == KIND_CODE[kind]
        return list(zip(*np.nonzero(mask)))

    def copy(self) -> "Recording":
        return Recording(self.name, self.day, self.weekday, self.first, list(self.units), self.y.copy(),
                         self.status.copy(), self.now.copy(), None if self.truth is None else self.truth.copy(),
                         dict(self.ids), self.split)


def load_recordings(data_dir: str | Path) -> dict:
    """Read the official csv files into a dict of Recording objects keyed by recording name."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    meta = pd.read_csv(data_dir / "recordings.csv").set_index("recording")
    recs = {}

    def hhmm(text: str) -> int:
        h, m = str(text).split(":")
        return int(h) * 60 + int(m)

    for split, df in (("train", train), ("test", test)):
        for name, g in df.groupby("recording"):
            first, last = hhmm(meta.loc[name, "first"]), hhmm(meta.loc[name, "last"])
            first = min(first, int(g["minute_of_day"].min()))
            last = max(last, int(g["minute_of_day"].max()))
            n = last - first + 1
            units = [u for u in ALL_UNITS if u in set(g["unit"])]
            ucol = {u: i for i, u in enumerate(units)}
            y = np.full((n, len(units)), np.nan)
            status = np.zeros((n, len(units)), dtype=np.int8)
            now = np.full((n, len(units)), -1, dtype=np.int32)
            ids = {}
            rows = (g["minute_of_day"].to_numpy() - first).astype(int)
            cols = g["unit"].map(ucol).to_numpy()
            people = g["people"].to_numpy(float)
            if split == "train":
                y[rows, cols] = people
                status[rows, cols] = RELEASED
            else:
                kinds = g["kind"].to_numpy()
                gid = g["id"].to_numpy()
                nowm = g["now_minute_of_day"].to_numpy()
                for r, c, p, k, i, nm in zip(rows, cols, people, kinds, gid, nowm):
                    if isinstance(k, str):
                        status[r, c] = KIND_CODE[k]
                        ids[(int(r), int(c))] = i
                        if k in ("forecast", "hidden-future"):
                            now[r, c] = int(nm)
                    else:
                        y[r, c] = p
                        status[r, c] = RELEASED
            recs[name] = Recording(name, str(meta.loc[name, "day"]), str(meta.loc[name, "weekday"]), first, units,
                                   y, status, now, None, ids, split)
    return recs


def submission_frame(recs: dict, preds: dict, sample_path: str | Path) -> pd.DataFrame:
    """Turn {recording: [T, U] prediction array} into a submission in the sample's row order."""
    sample = pd.read_csv(sample_path)
    values = {}
    for name, rec in recs.items():
        if rec.split != "test":
            continue
        p = preds[name]
        for (r, c), gid in rec.ids.items():
            values[gid] = p[r, c]
    missing = [i for i in sample["id"] if i not in values]
    assert not missing, f"{len(missing)} ids without a prediction, e.g. {missing[:3]}"
    out = pd.DataFrame({"id": sample["id"], "people": [values[i] for i in sample["id"]]})
    assert np.isfinite(out["people"]).all(), "non-finite prediction"
    out["people"] = out["people"].clip(lower=0.0)
    return out
