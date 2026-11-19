"""Shared helpers for the hidden-future track experiments (data tables, pseudo-cases, scoring).

Pseudo-cases: every training recording is replayed under both hidden patterns (B and G hidden together, D hidden)
with an origin every few minutes; the models are fitted without the replayed date. The two combinations that are
the team's fixed local test (11 Sep with B and G hidden, 12 Sep with D hidden) are flagged `fixed_combo` and are
left out of every selection step.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "src", ROOT / "noran"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CAMS = [f"cam_{c}" for c in "ABCDEFGH"]
MASKS = {"BG": ["cam_B", "cam_G"], "D": ["cam_D"]}
MASK_OF = {"cam_B": "BG", "cam_G": "BG", "cam_D": "D"}
FIXED_COMBOS = {("20260911_1100", "BG"), ("20260912_1100", "D")}
SCRATCH = Path(r"C:/Users/naser/AppData/Local/Temp/claude/6aedb350c0a1319d/473d99bd-5dcd-4883-8899-8744860e022d/scratchpad/hf")


def load_train() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw/train.csv")


def load_schedule() -> dict:
    raw = pd.read_csv(ROOT / "data/raw/prayer_times_madinah.csv", dtype=str)
    return {r.day: np.array([int(getattr(r, p)[:2]) * 60 + int(getattr(r, p)[3:])
                             for p in ("fajr", "dhuhr", "asr", "maghrib", "isha")]) for r in raw.itertuples()}


def tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """recording -> minute x camera table on the full minute range of the recording (NaN where no reading)."""
    out = {}
    for rec, g in frame[frame.people.notna()].groupby("recording"):
        t = g.pivot(index="minute_of_day", columns="unit", values="people").sort_index()
        out[rec] = t.reindex(range(int(t.index.min()), int(t.index.max()) + 1))
    return out


def rec_days(frame: pd.DataFrame) -> dict[str, str]:
    return frame.groupby("recording").day.first().to_dict()


def band(h) -> str:
    return "1-5" if h <= 5 else ("6-10" if h <= 10 else "11-15")


def score(detail: pd.DataFrame, columns: list[str], by: list[str]) -> pd.DataFrame:
    """MAE of each prediction column divided by the reference MAE, per group."""
    rows = []
    for key, g in detail.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ref = float((g.truth - g.reference).abs().mean())
        row = dict(zip(by, key), n=len(g), ref_mae=ref)
        for c in columns:
            row[c] = float((g.truth - g[c]).abs().mean()) / ref
        rows.append(row)
    return pd.DataFrame(rows)


def summary(detail: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Selection table: one line per recording x hidden pattern, plus pooled lines.
    'select' = mean of the non-fixed combinations (what settings may be chosen on);
    'fixed_dense' = mean of the two fixed combinations (report only)."""
    s = score(detail, columns, ["recording", "mask"])
    s["fixed_combo"] = [(r.recording, r.mask) in FIXED_COMBOS for r in s.itertuples()]
    lines = [s]
    for name, part in (("select: long", s[~s.fixed_combo & s.recording.isin(["20260911_1100", "20260912_1100"])]),
                       ("select: short", s[~s.fixed_combo & ~s.recording.isin(["20260911_1100", "20260912_1100"])]),
                       ("select: all", s[~s.fixed_combo]),
                       ("fixed_dense", s[s.fixed_combo])):
        if len(part):
            row = {"recording": name, "mask": "", "n": int(part.n.sum()), "ref_mae": float(part.ref_mae.mean()), "fixed_combo": ""}
            row.update({c: float(part[c].mean()) for c in columns})
            lines.append(pd.DataFrame([row]))
    return pd.concat(lines, ignore_index=True)
