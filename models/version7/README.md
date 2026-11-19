# Version 7 Assets

Version 7 scored **0.724** on the public Kaggle board (user reported). `manifest.json` records what the file was
built and verified with, and `notebooks/version7.ipynb` stops when any of it differs:

- the SHA-256 of the four competition files (`train.csv`, `test.csv`, `recordings.csv`, `sample_submission.csv`);
- the SHA-256 of the frozen assets it reads, all in `models/current_baseline/`: the prayer timetable
  (`prayer_times_madinah.csv`, source <https://www.noor-deen.com/prayer-times/medina/>, `Asia/Riyadh` times) and the
  twelve fitted T05 covariance files used by the forecast (`t05/params_without_{fri,sat}_*.npz`; they hold kernel
  parameters, no observations and no predictions);
- the package versions of the version 7 environment (`requirements_version7.txt`);
- the pretrained models, by repository, revision and file SHA-256:

| Model | Repository and file | Licence | Used for |
|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` (`config.json`, `model.safetensors`), chronos-forecasting 2.3.2 | Apache-2.0 | Forecast, cameras B, C, D, G, H, share 0.25 |
| TabICL v2 | `jingang/TabICL` (`tabicl-regressor-v2-20260212.ckpt`), tabicl 2.2.0 | BSD-3-Clause | Hidden, cameras B (0.33) and D (0.20) |

- the accepted submission SHA-256 `2eced7c2ea9765d93d798fc0130126a523e51e69b1fb4ac5bf6d5783d81cc88c`.

No fitted mission model is stored: the hidden, hidden-future and missing models are fitted from the competition
files every run, and Chronos-2 and TabICL v2 are used without training. The model weights are not in Git; section 1
of the notebook downloads them once into the Hugging Face cache. No submission file is bundled.

The original build ran its library steps in `.venv_lab` and the other steps in the team's `.venv`; both
environments hold the same NumPy, pandas, SciPy, scikit-learn, PyTorch and CatBoost versions. The verified notebook
run used `.venv_lab` for every step.
