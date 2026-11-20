"""P58 exploratory HIDDEN-FUTURE correction from reconstructed target history.

The V7 HIDDEN v3 estimator reconstructs each of the last 15 minutes from the
masked released feed. A robust slope over the real minute coordinates is used
as a small, fixed trend correction to the unchanged V7 HIDDEN-FUTURE forecast.
No second-stage model is fitted. Both component models are fitted only on the
training rows supplied by the date-held-out evaluation harness.
"""
from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TRACKS = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf = _load("p58_v7_hf_final", TRACKS / "hf_final" / "model.py")
hidden = _load("p58_v7_hidden_v3", TRACKS / "hidden_v3" / "model.py")
HISTORY_MINUTES = 15
TREND_SHARE = 0.5


def fit(training, day, context):
    # For the paired fixed/mirror protocol, remove the other assessment day's
    # jointly masked target camera answers as well as the held-out date.
    if os.environ.get("P58_CASE_SET") in ("fixed", "mirror"):
        other_day = "2026-09-12" if str(day) == "2026-09-11" else "2026-09-11"
        if os.environ["P58_CASE_SET"] == "fixed":
            other_mask = ["cam_D"] if other_day == "2026-09-12" else ["cam_B", "cam_G"]
        else:
            other_mask = ["cam_B", "cam_G"] if other_day == "2026-09-12" else ["cam_D"]
        training = training[~(training.day.eq(other_day) & training.unit.isin(other_mask))].copy()
    import baseline_tasks as tasks
    hidden_context = dict(context, config=tasks.default_config()["hidden"], mission="hidden")
    return {
        "hf": hf.fit(training, day, context),
        "hidden": hidden.fit(training, day, hidden_context),
        "hidden_context": hidden_context,
        "history_cache": {},
    }


def predict(case, t, model, context):
    now = int(case.now)
    horizon = int(case.minute) - now
    causal = t.loc[:now]
    baseline = hf.predict(case, causal, model["hf"], context)
    if not np.isfinite(baseline) or horizon <= 0:
        return baseline

    # Keep the full minute grid; missing reconstructed values remain missing.
    masked = causal.drop(columns=["cam_B", "cam_G"] if case.unit in ("cam_B", "cam_G") else ["cam_D"], errors="ignore")
    key = (getattr(case, "recording", ""), case.unit, str(case.day), now,
           hashlib.sha256(masked.to_numpy().tobytes() + masked.index.to_numpy().tobytes()).hexdigest())
    if key not in model["history_cache"]:
        minutes = np.arange(now - 14, now + 1, dtype=int)
        # HIDDEN's centred convolution needs at least 13 rows. Pad only with
        # missing rows, never data from another recording or after now.
        reconstruction_feed = masked.reindex(range(min(int(masked.index.min()), now - 14), now + 1))
        values = []
        for minute in minutes:
            donors = reconstruction_feed.reindex(columns=[f"cam_{c}" for c in "ABCDEFGH"]).loc[minute].notna().sum()
            if donors < 2:
                values.append(np.nan)
                continue
            hist_case = SimpleNamespace(unit=case.unit, minute=int(minute), day=case.day)
            value = hidden.predict(hist_case, reconstruction_feed, model["hidden"], model["hidden_context"])
            values.append(float(value) if value is not None and np.isfinite(value) else np.nan)
        model["history_cache"][key] = (minutes, values)
    minutes, values = model["history_cache"][key]

    valid = np.isfinite(values)
    if valid.sum() < 5:
        return baseline
    x = minutes[valid].astype(float)
    y = np.asarray(values, dtype=float)[valid]
    # Median pairwise slope is robust to noisy reconstructed points. Pairwise
    # denominators preserve the actual minute spacing.
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(len(x)) for j in range(i + 1, len(x))]
    slope = float(np.median(slopes))
    return float(max(0.0, baseline + TREND_SHARE * slope * horizon))
