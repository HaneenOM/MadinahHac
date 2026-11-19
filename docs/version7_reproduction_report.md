# Version 7 local reproduction — 19 November 2026

Environment setup succeeded. Exact accepted-file reproduction **failed**; the original notebook correctly stopped at its accepted SHA-256 assertion. The user-confirmed public Kaggle score of **0.724** belongs to the supplied reference file, not this generated file. No Kaggle upload was performed.

Repository root: `C:\Users\96655\Downloads\MadinahHac_version7_code\MadinahHac`.

All six requested entry-point files were inspected. No applicable AGENTS.md was found in the repository or inspected ancestor directories. All four official input files were already present and matched the manifest; no files were copied from or modified in the old project. The 13 frozen assets and supplied accepted reference also matched their documented hashes before execution, and again afterward.

## Setup

- Created a new `.venv_lab` with the available Python **3.12.10**. The manifest records **3.12.14**; that patch-version difference remains.
- Installed `requirements_version7.txt` without substitutions or changes. All manifest package versions matched; `pip check` passed. Installed packages are recorded in `outputs/version7/environment_freeze.txt`.
- Downloaded and verified Chronos-2 at `29ec3766d36d6f73f0696f85560a422f50e8498c` and TabICL at `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`. All three documented model-file hashes matched.
- Used an isolated cache at `outputs/version7/huggingface`. Its `refs/main` entries point to the verified pinned snapshots because the existing mission loaders resolve the default revision offline. No model loader or settings were changed.
- Ran the original `scripts/run_version7_notebook.py` with `VERIFY_ACCEPTED = True`, `HF_HUB_OFFLINE=1`, and the dedicated interpreter. T05 used frozen assets; other models followed normal fitting steps.
- There were no previous generated outputs before this run, only the preserved reference. No modeling experiments, algorithm changes, weight changes, uploads, commits, or pushes were performed.

## Result

Generated file: `C:\Users\96655\Downloads\MadinahHac_version7_code\MadinahHac\outputs\version7\submission_version7.csv`.

There are 4,457 rows and unique full IDs, with the exact sample order and ID set. Predictions are finite and nonnegative. Independent CSV comparison found **3,158 differing prediction strings**, **3,158 differing decimal values**, and **3,158 differing floating-point values** by full ID.

| Task | Rows | Unequal numeric values | Maximum absolute difference |
|---|---:|---:|---:|
| forecast | 2,660 | 1,941 | 0.000006681995 |
| hidden | 849 | 842 | 0.000044872853 |
| hidden-future | 484 | 0 | 0 |
| missing | 464 | 375 | 0.0000000000007958 |

Generated SHA-256:
`f5ab51ecd76287544afbadd12b2d230752e01b929575e6cb37b6436506d209da`

Accepted reference SHA-256, unchanged:
`2eced7c2ea9765d93d798fc0130126a523e51e69b1fb4ac5bf6d5783d81cc88c`

The largest difference is for `20260918_1100_cam_B_2010`. All hidden-future predictions and forecast cameras I–M match numerically. Forecast cameras A/E/F and the missing task differ only around double-precision rounding scale; the larger differences occur in the pretrained-model paths (Chronos forecast cameras B/C/D/G/H and hidden cameras B/D).

This pattern is consistent with numerical backend/CPU differences, but does **not** prove the cause. The original machine's CPU/backend configuration is not in the manifest, and Python's patch version also differs. Current platform/backend details are saved in `outputs/version7/runtime_platform.json`. No alternate CPU modes, package versions, or model settings were tried. Exact reproduction remains unresolved; changing the accepted fingerprint or copying reference predictions into the result would conceal the mismatch and was not done.

The runner log spans approximately **11 minutes 4 seconds**; its recorded computation steps total **656.4 seconds**. Model download and hash verification took **47.15 seconds**, separate from dependency installation. The runner does not save `version7_executed.ipynb` when execution raises an assertion; its logs, generated CSV, settings, and verification JSON remain available.

Evidence: `outputs/version7/verification.json`, `independent_verification.json`, `pretrained_verification.json`, `differences_by_full_id.csv`, `difference_by_task_camera.csv`, and `runner.log`.

## Next run

From this repository root in PowerShell:

```powershell
.\scripts\run_version7_reproduction.ps1
```

This helper copies previous generated Version 7 outputs and `outputs/noran` into a timestamped `outputs/version7_runs/before_*` archive, sets the project cache and interpreter, and invokes the original notebook runner. It preserves accepted verification and propagates failures. Its PowerShell syntax was checked; a second full notebook run was not performed. With the current unresolved numerical differences, another run may fail the same accepted-output assertion.
