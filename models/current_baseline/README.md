# Accepted baseline assets

PROMPT45: current public score is **0.800, user reported**. The expected output
hash in `manifest.json` is now
`342af77dd40e4120b589485ecf02bf25bf3434650e81473692328903bcf67a4a`.
The twelve covariance assets below are unchanged. B/G/D HIDDEN models are rebuilt
from the tracked recipes `src/hidden_bgd.py` and `src/hidden_catboost.py` and the
authorized released data, not loaded from ignored caches or submission CSVs.
`manifest_0806.json` and `configs/baseline_0806.json` preserve the previous version.
The remaining text below describes the original T05 asset provenance.

These twelve `.npz` files are the fitted T05 GP covariance parameters from the
accepted production cache (`outputs/t05_cache/params_without_{fri,sat}_*.npz`).
They contain `ells`, `B`, `B0`, `sig2`, `s2_delta`, `ell_delta`, `Bn`, `tau`,
and kernel names. They contain no recording observations or target predictions.
Six ensemble configurations each have separate Friday and Saturday parameters.
Loading these files avoids optimization and dependence on an untracked cache.
The tracked `t05_forecast` implementation still constructs templates and computes
fresh predictions from the supplied competition inputs at runtime.

`manifest.json` records the original input hashes, individual asset hashes,
configuration, numerical package versions, and accepted submission SHA-256:

`c7e0a0785fee013449b30a1041abdf939a58461bb5f359c5294e69b2c8ec6cd5`

The reference file is `submission_t05_hidden_hidden_future_im_fallback_v1.csv`;
its 0.806 public Kaggle score is user reported. No submission CSV is bundled.
The nonlinear D-only trial is not an asset of this baseline.

`prayer_times_madinah.csv` is the six-date timetable supplied for earlier
experiments, source <https://www.noor-deen.com/prayer-times/medina/>, local
`Asia/Riyadh` times. Dhuhr is the timetable time, not a sermon or iqama time.
The notebook uses this copy rather than an untracked local timetable.

The four raw competition files must be obtained separately through authorized
access and placed in `data/raw/`. See the root README and
`notebooks/current_baseline.ipynb`.
