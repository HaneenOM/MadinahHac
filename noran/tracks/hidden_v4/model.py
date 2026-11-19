"""Hidden camera, version 4: version 3 (noran/tracks/hidden_v3) plus the TabICL v2 member of the library test
(noran/tracks/lib_hidden/model.py, variant A, log target), joined per camera in log scale:

    value = expm1((1 - s) * log1p(hidden_v3) + s * log1p(TabICL median))        s = SHARE[camera]

Shares B 0.33, D 0.20, G 0: the library test chose them on top of hidden_final (B and D won on every case set,
G lost on two). Re-checked on top of hidden_v3 with noran/blend_members.py hidden_v3 lib_hidden_A_log (18 Nov 2026):
B and D still win on all three sets at these shares, G still loses.
A case with no TabICL value (fewer than two visible cameras at that minute) keeps the version 3 value.

Needs the lab environment (.venv_lab: tabicl 2.2.0, model file in the Hugging Face cache).
Run (repository root):
    export PYTHONPATH="$(pwd -W)"; export HF_HUB_OFFLINE=1
    .venv_lab/Scripts/python.exe -X utf8 noran/track_eval.py hidden noran/tracks/hidden_v4/model.py --set fixed --label hidden_v4
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("NORAN_TABICL_VARIANT", "A")
os.environ.setdefault("NORAN_TABICL_TARGET", "log")
TRACKS = Path(__file__).resolve().parents[1]
SHARE = {"cam_B": 0.33, "cam_D": 0.20, "cam_G": 0.0}
if os.environ.get("NORAN_HIDDEN_V4_SHARE"):    # set by notebooks/version7.ipynb section 5 (configs/version7.json)
    SHARE = {unit: float(s) for unit, s in json.loads(os.environ["NORAN_HIDDEN_V4_SHARE"]).items()}


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = _load("hidden_v3_for_v4", TRACKS / "hidden_v3" / "model.py")
tabicl = _load("lib_hidden_tabicl_for_v4", TRACKS / "lib_hidden" / "model.py")


def fit(training, day, context):
    units = [u for u, s in SHARE.items() if s > 0]
    return dict(v3=v3.fit(training, day, context), tabicl=tabicl.fit(training, day, context, units=units))


def predict(case, t, model, context):
    value = v3.predict(case, t, model["v3"], context)
    share = SHARE.get(case.unit, 0.0)
    if share <= 0 or not np.isfinite(value):
        return value
    extra = tabicl.raw_value(case, t, model["tabicl"], context)
    if not np.isfinite(extra):
        return value
    return float(max(0.0, np.expm1((1 - share) * np.log1p(max(value, 0.0)) + share * np.log1p(max(extra, 0.0)))))
