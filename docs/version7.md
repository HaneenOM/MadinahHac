# Version 7: Run and Modify

Version 7 scored **0.724** on the public Kaggle board (user reported, uploaded 18 Nov 2026). Its local scores are
0.7038 on the team's fixed test set and 0.7284 on the mirror set (11 and 12 Sep with the hidden cameras swapped).
Local scores come from days whose answers the team holds; they are not a forecast of the board.

| File | Role |
|---|---|
| `notebooks/version7.ipynb` | Entry point: eight sections, one settings, run and summary group per mission |
| `scripts/run_version7_notebook.py` | Runs the notebook without an editor |
| `configs/version7.json` | Editable settings the notebook reads |
| `models/version7/manifest.json` | Input and asset hashes, package versions, pretrained models, accepted fingerprint |
| `models/version7/README.md` | What the frozen assets and pretrained models are |
| `requirements_version7.txt` | Packages of the version 7 environment |
| `noran/` | Mission code, build and check scripts, and the analyses behind each choice |

## Running

Python 3.12 on Windows, CPU only. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv_lab
.\.venv_lab\Scripts\python.exe -m pip install -r requirements_version7.txt
.\.venv_lab\Scripts\python.exe scripts\run_version7_notebook.py
```

Place the official `train.csv`, `test.csv`, `recordings.csv` and `sample_submission.csv` in `data/raw/`, or open
`notebooks/version7.ipynb` with the `.venv_lab` kernel, then Restart Kernel and Run All. Section 1 downloads the
two pretrained models once (about 570 MB) at the revisions in the manifest; every later step reads them offline.

The notebook stops when an input file, a frozen asset, a package version or a model file differs from
`models/version7/manifest.json`, and when the final file differs from the accepted SHA-256:

```
2eced7c2ea9765d93d798fc0130126a523e51e69b1fb4ac5bf6d5783d81cc88c
```

Outputs in `outputs/version7/`: `submission_version7.csv`, `task_config.json` (the settings used),
`verification.json` (checks, SHA-256 and the time of every step), `logs/` (one log per step) and, for the script
run, `version7_executed.ipynb`. Intermediate files of each mission are under `outputs/noran/`. If the uploaded
file is copied to `outputs/version7/reference/submission_v7_board_0.724.csv`, section 8 also compares every value
by ID.

Measured on 19 Nov 2026 from an empty `outputs/` folder with only the four competition files in `data/raw/`:
Run All took 781 seconds (about 13 minutes) on a laptop CPU with `.venv_lab` for every step, and the file matched
the fingerprint with zero unequal values against the uploaded file. The longest steps are the hidden model (245 s),
Chronos-2 (150 s plus 56 s for the isolated run), the base build (140 s) and the timing-shift forecast (91 s).

## Recipes and where to change them

| Mission | Version 7 recipe | Settings in `configs/version7.json` | Code |
|---|---|---|---|
| FORECAST | (1 - `track_weight`) x accepted T05 forecast + `track_weight` x timing-shift forecast; for `chronos_cameras`, (1 - `chronos_weight`) x that value + `chronos_weight` x Chronos-2. Values: 0.5, 0.25, cameras B, C, D, G, H. Cameras I to M keep the accepted rule. | `forecast` | `noran/tracks/forecast/model.py` (timing shift), `noran/tracks/lib_forecast/run_lib.py` (Chronos-2 variants), `noran/blend_forecast.py`, `noran/tracks/lib_forecast/verify/make_others_blend.py` |
| HIDDEN | Version 3 (five estimates blended in log scale: ridge regression plus last week's residual at the prayer-aligned minute, a boosted model, the team's tree model, the teammate's CatBoost for B and D, and the camera's own curve from last week scaled to today; B averaged with the baseline where D stands far above the other cameras) joined with TabICL v2 in log scale, share 0.33 for B and 0.20 for D. | `hidden` (`tabicl_share`, `track_module`, `label`) | `noran/tracks/hidden_v4/model.py`, which loads `hidden_v3`, `hidden_hedged`, `hidden_final`, `hidden_analog`, `hidden_future`, `cat`, `hidden` and `lib_hidden` |
| HIDDEN-FUTURE | The camera's curve from the same weekday last week, aligned to today's prayer times and scaled by today's level of the visible cameras over the last 10 minutes; the baseline value is kept in the late afternoon. | `hidden-future` (`track_module`, `label`) | `noran/tracks/hf_final/model.py`, `noran/tracks/hidden_future/model.py` (constants at the top) |
| MISSING | The straight line across the gap, corrected by the partner camera's readings inside the gap and the camera's own level smoothed from both sides; stacking weights fitted on gaps cut in the same pattern as the real ones. | `missing` (`track_module`, `label`) | `noran/tracks/missing/model.py` |

Every mission module has `fit(training, day, context)` and `predict(case, t, model, context)`. `predict` returns
a value for a target it changes or NaN to keep the accepted baseline value. Models are fitted when the notebook
runs; no fitted model is loaded from disk. The only frozen assets are the prayer timetable and the twelve T05
covariance files in `models/current_baseline/`.

`src/baseline_tasks.py` holds the team's task functions with optional `track_module` hooks for each mission and an
optional partner-ratio filler for missing (`src/noran_missing.py`). Neither is active unless its setting is given,
so `notebooks/current_baseline.ipynb` gives the same file as before. `noran/blend_forecast.py` and
`make_others_blend.py` accept `--test-only`, which writes the test values without the fixed and mirror evidence
files; the formulas are the same either way.

The method and the evidence behind each choice are in `noran/HOW_IT_WORKS.md`.

## Evaluation after a change

Each mission section of the notebook ends with its local scoring commands. For a mission module:

```powershell
$env:PYTHONPATH = (Get-Location).Path; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONUTF8 = "1"
.\.venv_lab\Scripts\python.exe -X utf8 noran\track_eval.py hidden noran\tracks\hidden_v4\model.py --set mirror --label hidden_v4
```

`--set` is `fixed` (the team's test: 11 Sep with the Friday mask, 12 Sep with the Saturday mask), `mirror` (masks
swapped) or `testday` (the two real test days with a released camera hidden; hidden and hidden-future only).
`noran/candidate.py score noran/configs/v7_tabicl_chronos.json` scores the whole file on the fixed and mirror sets.
The fixed set reads `outputs/local_score/`, written by `src/local_score.py` (`scripts/run_local_score.ps1`); that
step was not rerun for this package.

Version 7 per mission on the fixed set: forecast 0.8358, hidden 0.6803, hidden-future 0.5931, missing 0.7060.
A change was kept only when it won on every case set; the record of each test is in the mission folder.

## Contributing

1. `git switch -c task/hidden-update` and change one mission: its settings in `configs/version7.json`, or a copy of
   its module folder under `noran/tracks/` with `track_module` and `label` pointing at the copy.
2. Run that mission's evaluation commands on every case set and compare with the numbers above.
3. In section 8 set `VERIFY_ACCEPTED = False`, then Restart Kernel and Run All. The new file and its settings are
   in `outputs/version7/`.
4. Review `git diff` and commit the source and settings. Data and outputs stay out of Git. Update the fingerprint
   in the manifest only for a file accepted as the next version, never to hide a difference.

## Not included

- Data, value files, logs and model weights (outside Git; the weights download in section 1).
- Experiments made after version 7 (versions 8 and 9 and the later test rounds).
