# Madinah Ethical AI Competition forecasting project

This project uses the supplied 2026 release. Its four CSV files match the
`MDA-Ethical-AI-KAUST-Academy.zip` in Downloads byte for byte. The originals
remain in the project root; `data/raw/` contains working copies. The official
starter is preserved as `notebooks/starter-official.ipynb`.

## Collaborator setup

### Actual CatBoost Friday HIDDEN trial (PROMPT 42)

[The P42 report](reports/prompt42_friday_catboost_ar.md) documents actual
CatBoost 1.2.10 training with MAE, date-aware development, inner early stopping,
three small feature/depth configurations and separate B/G decisions. B improves
the fixed strict/release-matched MAE by 5.04%/22.11%, but development Friday
regresses and its strict gain disappears without its two best windows. G does
not consistently beat P40. Neither replacement was adopted; no submission was
created. CatBoost is optional in `requirements-catboost.txt`; production and
notebook defaults do not import it. The isolated runner is
`src/friday_catboost_trial.py`, with results in `outputs/prompt42_friday_catboost/`.

### Friday HIDDEN normalized-target trial (PROMPT 41)

[The P41 report](reports/prompt41_friday_ratios_ar.md) distinguishes existing
constant donor ratios and previously tested donor shares from a new normalized
target Ridge. Date-aware development excludes September 11 entirely. The trial
was rejected: neither camera passed development, G did not improve on P40, and
B's release-matched gain disappeared after removing its two best windows.
No new submission or production/default change was made. The standalone script
is `src/friday_ratio_trial.py`; diagnostics and separate strict/release-matched
tables are saved in `outputs/prompt41_friday_ratios/`.

### Friday-only HIDDEN candidate (PROMPT 40)

[The P40 report](reports/prompt40_friday_hidden_ar.md) documents a separate
candidate changing only 280 Friday HIDDEN G rows. A small date-held-out
development comparison selected released-row dropout Ridge with Friday fitting
weight 3 and the existing 0.5 G blend. B stays unchanged because development did
not consistently support a replacement. Fixed Friday G MAE improves 1.61% in
the strict benchmark and 1.25% in the separate release-matched setting, including
positive gains after removing the two best windows. These already-examined
dates provide limited evidence, not a predicted Kaggle score.

`outputs/prompt40_friday_hidden/submission_friday_hidden_v1.csv` is separate from
the protected baseline and D-only candidate. All 4,177 other text values,
including every Saturday row, are unchanged. Nothing is enabled in production
or the default notebook; no automatic Kaggle upload is performed.

### Released-row fitting rule and one bounded audit (PROMPT 39)

The user supplied the competition-specific rule permitting model fitting on
any genuinely released row, including after a gap's now; forecast/hidden-future
observation inputs still stop at each own now. Hidden/missing may use the whole
released table. This does not establish an official scoring formula.
See [the P39 audit and experiment](reports/prompt39_released_training_ar.md).
Production already uses released test information in T05 fitting/template
completion; the older statement below about no fitting on test is historical.
The three other production task fitters receive train only.

One separate HIDDEN experiment added released A–H rows to the existing Ridge
family, with unchanged features/settings/blends. It was rejected: Friday
improved, Saturday worsened under both strict date-held-out and a distinct
globally masked release-matched protocol. No candidate submission was generated,
and no production/notebook/model-asset changes or T05 refits were made.

### Local evaluation: supported commands (PROMPT 38)

Use [the documented evaluation protocol](docs/local_evaluation.md) and
`src/reliable_evaluation.py` for new evaluations. It separates the fixed
September 11–12 low-prior-week-coverage stress test, the fixed September 18–19
released-target A–H diagnostic, and the overlapping I–M proxy. It runs current
task implementations, never optimizes T05, and reports coverage and paired
reference MAE separately. The project aggregate is **not confirmed as the
official Kaggle metric**; missing official rule text is explicitly listed.

`models/local_evaluation/` bundles 24 small evaluation covariances separately
from production. Their original training provenance is incomplete. Forecast
runs require `--allow-legacy-provenance` and remain labeled conditional; input,
code, settings, reconstructed field and asset fingerprints are checked before
reuse. Without the flag, with changed fingerprints, or with missing assets the
run stops. No automatic 12-model refit occurs. The old full-score command and
experiment commands below are historical records, not the recommended workflow.

### Current accepted baseline: one notebook

Open **`notebooks/current_baseline.ipynb`** for the accepted T05-based pipeline
whose historical public Kaggle score was **0.806 (user reported)**. This is
the teammate entry point; the older commands and experiment sections below
document the project's history.

After cloning, run from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the authorized competition `train.csv`, `test.csv`, `recordings.csv`,
and `sample_submission.csv` in `data/raw/`. Open the notebook in VS Code,
select `.venv\Scripts\python.exe` as its Python kernel, and choose **Run All**.
Alternatively, execute that same notebook without a notebook editor:

```powershell
.\.venv\Scripts\python.exe scripts\run_current_baseline_notebook.py
```

The new output is `outputs/current_baseline_manual/submission_current_baseline.csv`.
`verification.json`, editable `task_config.json`, and (for the command above)
`current_baseline_executed.ipynb` are in the same directory. Existing submissions
are preserved. Every run generates fresh predictions. The notebook now has eight
numbered Arabic sections with separate configuration, loading/training and
prediction cells for each task. See [the manual contribution guide](docs/teammate_baseline.md)
for source functions, evaluation commands and saved-model limitations.

The accepted path includes T05 forecast, the accepted I–M fallback, the accepted
hidden donor/Ridge weights, Saturday prayer hidden-future, and v2 missing.
**The separate nonlinear D-only candidate is disabled.** No rejected experiment,
diagnostic evaluation, hyperparameter search, or T05 retraining runs here.

Reproducibility dependencies previously missing from a clone are now supplied in
`models/current_baseline/`: the twelve small accepted T05 parameter sets, the
supplied prayer timetable, and a manifest of hashes and numerical package
versions. These parameter files contain fitted covariance parameters, not raw
competition observations or submission predictions. The required reference for
the other three tasks is recovered directly from the official sample: the first
submission retained those exact sample values. No previous submission or local
`outputs/t05_cache` is required. Optional path arguments were added to the accepted
builders without changing their prediction formulas or default commands.

The notebook checks the four input hashes, frozen assets, all 4,457 sample IDs in
order, unique IDs, and finite nonnegative predictions. It requires the final
SHA-256 to match the accepted fingerprint in the manifest. If the local saved
`outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv` exists, it also
compares every value by ID and the file hash. A mismatch stops the notebook;
cross-platform numerical differences are not silently accepted as an exact
reproduction. The raw competition inputs and all generated outputs remain
ignored by Git; commit the new notebook, source changes, and model assets before
the teammate clones this setup. No commit or push is performed by the notebook.

The tracked timetable is the supplied September 2026 timetable from
<https://www.noor-deen.com/prayer-times/medina/>, timezone `Asia/Riyadh`.
Its Dhuhr entry is a scheduled timetable reference, without assuming a sermon
or iqama time. The baseline notebook uses this frozen copy, so an additional
local timetable download is unnecessary.

Verified locally on November 17, 2026: Run All took 82.35 seconds, generated
4,457 valid rows, and had **zero unequal values by ID** against the saved
0.806 reference (maximum absolute difference **0**). Both files have SHA-256
`c7e0a0785fee013449b30a1041abdf939a58461bb5f359c5294e69b2c8ec6cd5`.
A second execution from a clean source export containing only repository
source/assets plus the four authorized raw inputs also matched that fingerprint
in 50.53 seconds, with no old submissions, local model cache, or untracked
experiment scripts present. Both executions used the installed pinned numerical
dependencies; a fresh network installation into a new environment was not run.
No T05 model was refitted. The verification reports are respectively
`outputs/current_baseline/verification.json` and
`outputs/current_baseline_clean_checkout/outputs/current_baseline/verification.json`.

The repository tracks source code, PowerShell benchmark commands, dependencies,
documentation, frozen baseline model assets, and the official starter notebook.
Raw competition CSVs, local prayer timetable copies, executed notebooks,
benchmark results, and submissions are
ignored by Git. After cloning, place the four competition CSVs in `data/raw/`
from the authorized local copies. For the legacy experiment commands below,
also place `prayer_times_madinah.csv` there; the new baseline notebook instead
uses its bundled timetable. Keep any Kaggle submission files in `outputs/` locally.

## Run on Windows PowerShell

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\prepare_notebook.py
.\.venv\Scripts\python.exe src\run_starter.py
.\.venv\Scripts\python.exe src\pipeline.py
.\.venv\Scripts\python.exe src\checks.py
.\.venv\Scripts\python.exe src\starter_validation.py
.\.venv\Scripts\python.exe src\scaled_donor_experiment.py
.\.venv\Scripts\python.exe src\direct_hidden_future_experiment.py
.\.venv\Scripts\python.exe src\report.py
```

The dependencies are CPU compatible. Installation needs network access once;
the experiment itself runs offline. `--data-dir PATH` overrides `data/raw/`.
The data locator also searches `/kaggle/input`; Kaggle output defaults to
`/kaggle/working`. Use `--output-dir PATH` to change the output directory.
`--validate-only` runs evaluation without writing the submission.

## Inputs and outputs

The loader checks schema, target IDs, blank target counts, nonnegative counts,
timestamp ordering, and duplicate recording/unit/minute keys before fitting.
It discovers the input folder containing `train.csv`, `test.csv`, and
`sample_submission.csv`, and also loads `recordings.csv`. The current release
has 13,443 training rows, 8,611 test rows, 4,154 released test observations,
and 4,457 targets. The target kinds are 2,660 forecast, 849 hidden, 484
hidden-future, and 464 missing. Train cameras are A–H; test cameras are A–M.

`outputs/reference_submission.csv` is the supplied Kaggle sample. The locally
executed notebook writes `outputs/starter_submission.csv` and
`notebooks/starter-executed.ipynb`. The selected pipeline writes
`outputs/submission.csv`, `outputs/validation_predictions.csv`,
`outputs/validation_scores.csv`, `outputs/selection.json`, and
`outputs/cold_start_proxy.csv`. The report command adds
`validation_experiment_by_kind.csv`, `validation_experiment_by_camera.csv`,
`validation_experiment_by_horizon.csv`, and `submission_by_camera.csv`. The
validation experiment files show the tested proxies for the three kinds that
retain sample values. The final CSV is re-read and checked for sample
order, unique IDs, finite nonnegative predictions, and the exact row count.

## Methods

The official starter pivots each recording into a minute-by-camera table. It
uses the last reading within five minutes, a 15-minute camera mean, and a
five-minute table mean, all in log space, plus the horizon. A histogram
gradient booster with absolute-error loss predicts change from the last log
count. It fits horizons 5, 15, and 30, although scored forecast horizons are
1–15. It uses the supplied sample values for the other three kinds.

This pipeline fits on all horizons 1–15. Forecast comparisons are persistence,
a damped recent slope, and pooled histogram boosters for count residuals and
log residuals. The features use actual minute offsets: the most recent value,
its age, 5/15-minute means and slopes, donor-camera recent levels and slope,
camera availability, minute of day, and horizon. It has no camera-ID feature,
so unseen I–M can use the pooled model when they have recent readings. If
recent readings are absent, the submitted value falls back to the supplied
reference.

For hidden cameras, a Ridge model trained on overlapping training-camera
counts estimates the target from concurrent donors. Hidden-future compares
holding that estimate at `now` with forecasting donor values to the target
minute before applying the same Ridge model. Missing compares endpoint linear
interpolation with a donor-movement adjustment that leaves the target's
endpoint trajectory intact. The submitted method uses the log-residual forecast
model. For hidden, hidden-future, and missing, it retains the supplied sample
reference values because no candidate showed a reliable improvement against
a defensible local reference. The experiments remain in the output files.

For forecast and hidden-future, the complete recording table is sliced at
`now` before features, donor estimates, or prediction. Hidden and missing may
use the whole released feed. The leakage check replaces every value after
`now` and asserts unchanged forecasts. Released target labels are never
inserted into the feed.

## Validation and limitations

The folds hold out whole dates, not individual recordings. The September 11
and 12 training recordings have the same 615-minute clock span as the two test
recordings. Their labels are evaluated through transplanted test masks for
the corresponding weekday, retaining only originally known training counts.
This mirrors whole-camera hidden masks, rolling forecast cutoffs, and missing
gaps while preserving natural missingness. Earlier dates supply fit data.

Reference methods in local scoring are **approximations**: persistence for
forecast, Ridge donors for hidden, a fixed donor estimate for hidden-future,
and endpoint interpolation for missing. The exact organizer reference
formulas are not supplied in the notebook, so normalized local errors are
not Kaggle scores. `validation_scores.csv` reports each candidate's MAE,
reference MAE, ratio, and count by fold, recording, and kind. If a local
reference MAE is zero, its ratio is reported as blank. Some target masks have
no known counterpart in training, especially I–M; their accuracy cannot be
measured directly. `cold_start_proxy.csv` withholds H from fitting and keeps
only every fifth H reading to probe an unseen camera with sparse context.

No model is fitted on released test observations. That optional experiment
needs a separate validation protocol that exactly mirrors adding unlabeled
released rows to fitting while withholding pseudo-target labels. No external
features are used.

## Measured results from this run

Forecast MAE on 2,036 known pseudo-targets: persistence 30.71, official
starter method 28.10, and selected log-residual model 22.95. By held-out
date, the selected model had MAE 28.88 on September 11 and 17.91 on
September 12; the starter had 37.23 and 20.32. Count-residual boosting had
33.45 and 17.26, so the log model's advantage is date dependent.

The tested local proxies had MAEs hidden 6.19 (842 targets), hidden-future
6.51 (482), and missing 6.95 (358). These are experiment results, not
measured errors for the supplied sample values used in the final submission.
The equal-kind score cannot be estimated faithfully because the exact
organizer reference formulas and corresponding validation predictions are
unavailable. None of these figures is a Kaggle score.

On overlapping hidden-camera validation targets, strongest scaled donor
versus Ridge MAE was 6.25 versus 6.78 on September 11, and 7.12 versus
6.91 on September 12. Ridge covers more targets but is not selected. The
direct horizon-aware hidden-future model had MAE 6.69 versus 5.35 for the
fixed estimate on September 11, and 7.44 versus 8.87 on September 12.
Its weighted result was worse. Donor-movement missing interpolation was worse
than linear on both dates.

The 236 post-cutoff perturbation comparisons passed. The cold-start proxy
withheld camera H from fitting and retained every fifth H observation; on
160 forecast targets, count-residual boosting had 3.68 MAE, versus 4.02 for
persistence and 4.32 for damped trend. This is a proxy for I–M, not direct
validation on those cameras. `submission_by_camera.csv` reports output
counts and means for each test camera; labels for I–M are unavailable.

To try a new experiment, add a variant in `predict_one`, include it in the
appropriate `variants` list in `validate`, and rerun `src/pipeline.py`. The
output detail and per-fold score files show whether it helps across dates.

## Iteration 2: hidden, hidden-future, and missing

The user reported a **0.987 public Kaggle score** for the first submission.
That file is preserved byte for byte as `outputs/submission_v1_0.987.csv`.
The score is historical context; no v2 file has been uploaded or scored.

Run the second iteration after the setup commands above:

```powershell
.\.venv\Scripts\python.exe src\v2_reference_audit.py
.\.venv\Scripts\python.exe src\v2_experiments.py
.\.venv\Scripts\python.exe src\v2_unseen_proxy.py
.\.venv\Scripts\python.exe src\v2_candidate.py
.\.venv\Scripts\python.exe src\v2_report.py
.\.venv\Scripts\python.exe src\v2_leakage.py
```

The candidate is `outputs/submission_v2.csv`. `v2_candidate.py` requires the
preserved first submission and checks its IDs before combining predictions.
It writes the 4,457 rows in sample order, re-reads the file, checks finite
nonnegative counts and unique IDs, and compares every forecast value as text
with the first file. All 2,660 forecast values match exactly. Changes are:
849 hidden, 484 hidden-future, and 296 missing. The remaining 168 missing
predictions retain the sample reference, including all 89 I–M missing targets.

### Reference audit and validation design

The supplied sample's missing values match time-aware endpoint interpolation
within 0.0005 people, consistent with rounding to three decimals. For hidden,
the sample uses donor D for B, E for G, and B for D, scaled by the ratio of
historical mean counts on overlapping observations. Where the designated
donor is released, this reconstructs the sample within 0.0005. Hidden-future
holds the scaled donor's last available reading at `now`; this also matches
the sample to rounding. One hidden D target has a donor fallback we could not
reconstruct exactly. The local reference uses a historical-median fallback
when no donor is available, so that case remains approximate.

Validation holds out entire dates. For each held-out training recording, it
transplants the corresponding real Friday or Saturday test-release mask onto
the same minute-of-day keys and scores only originally labeled counts. Friday
removes B and G together; Saturday removes D. The released feed also carries
the real simultaneous forecast and missing masks. September 11 and 12 are
the main 615-minute folds, matching the test recording duration. The shorter
September 4 and 5 recordings are stress tests. All coefficients, donor
selection, imputation values, and model parameters are fitted without that
fold's date. `v2_coverage.csv` reports known target coverage by recording.

The hidden candidates are scaled single donor, multi-camera Ridge, and Ridge
trained with random donor dropout. Friday B/G predictions blend 50% dropout
model with the exact sample reference; Saturday D blends 25% ordinary Ridge.
The hidden-future candidates are a fixed donor estimate, damped forecasts of
available donors passed through Ridge, and a direct horizon-aware histogram
booster. The direct model receives only donor values through each gap's `now`
and is blended 75% with the sample reference. The donor forecast path fits no
downstream model on generated forecasts, so no in-fold generated forecasts
enter model fitting. Missing candidates use historical pair calibration for
each donor's deviation from its own straight line between the gap endpoints.
The target's endpoint interpolation plus a weighted donor-deviation blend is
used for A/B/C/E/G, at half strength for F, and only for gaps of at least six
missing minutes with a usable donor. D/H and I–M retain the sample values.

### Held-out results

MAE is in people, and the ratio uses reference MAE on the **same targets**.
The selected blend results on the two main folds are:

| Kind | Sep 11 reference → v2 | Sep 12 reference → v2 | Combined count | Combined reference → v2 |
|---|---:|---:|---:|---:|
| hidden | 5.07 → 4.45 | 7.73 → 7.43 | 843 | 5.97 → 5.45 (0.914) |
| hidden-future | 5.66 → 3.87 | 9.95 → 6.44 | 482 | 7.08 → 4.73 (0.667) |
| missing | 8.60 → 6.55 | 5.57 → 4.19 | 358 | 6.95 → 5.27 (0.758) |

The full per-recording, per-camera, hidden-future horizon-band, and missing
gap-length/donor-availability results are in `outputs/v2_scores_by_recording.csv`,
`v2_scores_by_camera.csv`, `v2_scores_horizon.csv`, and
`v2_scores_missing_gap.csv`. The selected blends and exact same-target
reference comparisons are in `v2_selected_scores_by_recording.csv`,
`v2_selected_scores_by_camera.csv`, and `v2_selected_validation_predictions.csv`.
The date-and-camera breakdowns are in `v2_scores_by_date_camera.csv` and
`v2_selected_scores_by_date_camera.csv`; selected horizon and gap breakdowns
are in `v2_selected_scores_horizon.csv` and
`v2_selected_scores_missing_gap.csv`. Hidden donor availability, weekday, and
time-of-day results are in `v2_hidden_conditions.csv` and
`v2_selected_hidden_conditions.csv`.

On the short stress recordings, hidden-future direct prediction regressed on
both Saturday segments (37 and 29 targets); the 75% blend ratios were 1.78
and 1.16. These segments are 134 and 165 minutes rather than the test's 615.
The September 4 missing stress group has only 15 known targets; the candidate
kept interpolation there because its gaps did not meet the selected gate.
Camera H and D missing corrections did not improve consistently and were
excluded. These are material limits on extrapolating the long-fold gains.

For unseen I–M, `v2_unseen_proxy.py` removes 40 released observations in
contiguous eight-minute blocks, then fits local donor calibration without
those labels. The donor correction helped four of five tiny camera/recording
groups but regressed on one; coverage is too small to replace I–M references.
No true I–M target labels are available. `v2_leakage.py` perturbs all
post-`now` observations and checks every hidden-future prediction path.

The next useful experiment is a broader I–M released-observation proxy with
more distinct outage windows and an out-of-fold calibration of donor
deviations. It would test whether the 89 untouched I–M missing targets can
be improved without relying on five very small proxy groups.

## Iteration 3: prayer timetable and I–M gap diagnostic

The user reported a **0.826 public Kaggle score** for `outputs/submission_v2.csv`.
Its byte-identical comparison copy is `outputs/submission_v2_0.826.csv` (SHA-256
`efe8f3c1d62a3bc2df0ac0bd6f79c1b3ee7e3db25980fa33d45b1c26450d6d27`).
All figures below are **local held-out MAE**, not leaderboard scores.

The supplied `data/raw/prayer_times_madinah.csv` has one unique row for every
training and test date: September 4, 5, 11, 12, 18, and 19, 2026. Source:
[Noor Deen, Medina prayer times](https://www.noor-deen.com/prayer-times/medina/),
September 2026 timetable, local `Asia/Riyadh` time. The given `dhuhr` column is
the timetable's Dhuhr time. A Friday sermon, congregational start, or iqama
time would require separate sourced data. The experiment checks the schema,
unique ISO dates, `HH:MM` syntax, full date coverage, and many-to-one joins
preserving all 13,443 training and 8,611 test rows.

Run the held-out experiments, separate missing proxy, and candidate builder:

```powershell
.\.venv\Scripts\python.exe src\prayer_experiments.py --task hidden-future
.\.venv\Scripts\python.exe src\prayer_experiments.py --task forecast
.\.venv\Scripts\python.exe src\v3_missing_diagnostic.py
.\.venv\Scripts\python.exe src\v3_unseen_forecast_proxy.py
.\.venv\Scripts\python.exe src\v3_prayer_candidate.py
```

The v2 direct hidden-future and log-residual forecast models are reproduced
with the same training rows, labels, settings, and 75% hidden-future blend.
Both already use minute of day. The clock control adds a Friday indicator.
The prayer variant adds signed minutes from the nearest scheduled prayer at
`now` and the target (minute minus prayer minute), plus a flag for a prayer in
`(now, target]`. All observed camera features are cut off at `now`; target
time and prayer times are known in advance. Modified predictions are checked
against perturbations of all post-cutoff camera values. Score files report
paired overall, held-out date, horizon band, and target-within-15-minutes-of-
prayer results. Normalized MAE divides by same-target local reference MAE.

For hidden-future, paired local MAE across 632 known targets is 5.280 for v2,
5.266 for clock, and 4.807 for prayer. On the long Friday September 11
recording (322 targets), prayer is slightly worse: 3.918 versus 3.872 v2 and
3.838 clock. On long Saturday September 12 (160), prayer improves to 5.496
versus 6.445 v2 and 6.666 clock. On short Saturday recordings, prayer
improves 7.961 to 5.209 at 11:24 (37 targets) and 7.628 to 7.586 at 19:00
(29 targets), though both remain worse than the local fixed-donor reference.
September 4 Friday stress improves 6.466 to 5.764 (84 targets). Prayer
improves all three horizon bands overall. Within 15 minutes of a prayer, it
improves 4.628 to 3.966 on 135 targets; on the other 497, 5.457 to 5.035.

The separate I–M diagnostic masks 70 nonoverlapping contiguous gaps of 4–10
minutes, keeping observed endpoints, across five cameras and both released
test dates. It scores exactly the same 575 released observations under
endpoint interpolation and the existing full donor correction. Donor
calibration uses only the other released date, never the pseudo-labels.
Endpoint MAE is 8.541; donor correction is 8.563, with a usable correction
on 460 observations. It helps J and L on both dates but hurts K on both;
M has no usable calibrated donor. The next experiment should calibrate on
additional independently labeled dates, then validate by held-out date and
camera before replacing any of the 89 I–M `missing` reference values.

Forecast paired local MAE across 2,593 known pseudo-targets is 21.649 v2,
21.968 clock, and 19.564 prayer (normalized MAE 0.753, 0.764, 0.680).
Prayer beats both controls on every recording. On the long Friday and
Saturday dates it reduces 28.877 to 26.729 (936 targets) and 17.908 to
16.905 (1,100 targets). The three short stress recordings improve from
14.092 to 12.753, 18.694 to 10.475, and 18.607 to 15.043. It improves
each horizon band: 1–5 minutes 14.275 to 13.109, 6–10 minutes 22.422 to
20.257, and 11–15 minutes 28.154 to 25.241. For 525 targets within 15
minutes of a scheduled prayer, it improves 14.612 to 11.337; outside that
window (2,068 targets), 23.435 to 21.652. The original long-fold v2
forecast scores reproduce `validation_scores.csv` exactly.

The v3 builder applies prayer features to forecast and to Saturday
hidden-future only. The Friday hidden-future regression argues for retaining
v2 there. All hidden and missing predictions stay exactly as v2. The builder
re-reads the 4,457-row output and checks the sample ID order, unique IDs,
finite nonnegative values, and changed-prediction counts by task. Its output
is `outputs/submission_v3_prayer.csv`; its Kaggle score is unknown until the
user submits it.

For the 555 I–M forecast targets, held-out training labels do not exist. An
extra causal proxy scores 309 released I–M observation/horizon pairs (5, 10,
and 15 minutes, sampled every eight target minutes) over both test dates.
The prayer model improves over v2 on those same pairs: MAE 22.176 versus
24.275, although both are worse than persistence at 20.823. One of ten
date/camera groups (Saturday I) regresses slightly. This supports the
relative prayer change but suggests a separately validated persistence or
shrinkage fallback for unseen-camera forecasts as the next accuracy test.
The candidate changed all 2,660 forecast rows and 162 Saturday hidden-future
rows; the other 1,635 rows match v2 byte for byte. It passed 2,822
post-cutoff invariance checks on the changed paths.

## Repeatable v2 hidden-future benchmark

From the project root in PowerShell, run:

```powershell
.\scripts\run_hidden_future_benchmark.ps1
```

This focused benchmark reuses the v2 mask transplant, fixed-donor reference,
direct horizon-aware model, and selected 75% direct / 25% reference blend.
It holds out September 11 and September 12 separately from fitting, masks B
and G together on Friday and D on Saturday, and scores only known target
counts at horizons 1–15 minutes. Camera features are limited to observations
through each case's `now`; all post-cutoff observations are perturbed as an
invariance check. No prayer-time features are used in this baseline.

`outputs/hidden_future_benchmark/cases.csv` fixes the 482 evaluated target
gaps and their known labels. `released_feed.csv` fixes the available camera
observations under those masks. The same directory contains
`predictions.csv` and `scores_by_date.csv`. The fresh predictions reproduce
all 482 archived v2 values and pass 964 post-cutoff checks.

| Held-out date | Predictions | v2 MAE (people) | Fixed-donor reference MAE | Normalized MAE |
|---|---:|---:|---:|---:|
| 2026-09-11 | 322 | 3.872144 | 5.660323 | 0.684085 |
| 2026-09-12 | 160 | 6.444831 | 9.952099 | 0.647585 |

Normalized MAE is v2 MAE divided by the fixed-donor reference MAE on the
same targets. These are local validation figures, not Kaggle scores.

## One-command local score for the current model

From the project root in PowerShell:

```powershell
.\scripts\run_local_score.ps1
```

The first run fixes cases from the September 11 and 12 recordings under the
real Friday/Saturday release masks. The 482 hidden-future gaps match the
saved benchmark above exactly. The case and released-feed files contain no
masked answers; `labels.csv` stores those separately and is read only after
fresh predictions are saved. SHA-256 hashes guard these three files against
later changes. Each run fits without the evaluated date and predicts anew
using the T05 six-member forecast ensemble, the tested hidden B/D blend, and
the v2 hidden-future and missing selections. For date-held-out T05 fitting, the command uses the
teammate's fixed 96-cell validation proxy from the other test recording;
the scored training day is excluded. The T05 prediction path uses only camera
values through `now`; the v2 hidden-future predictions are also checked by
perturbing all later values.
The command does not write a Kaggle submission.

For each date and task, normalized MAE is the model MAE divided by the local
reference MAE on exactly the same targets: persistence for forecast,
single-donor for hidden, fixed-donor for hidden-future, and endpoint
interpolation for missing. The four ratios receive equal weight within each
date; the two date scores receive equal weight overall. A failed task makes
the result `incomplete` with a blocker instead of printing an overall score.

The current complete **overall local score is 0.793817**, with date scores
0.793990 (September 11) and 0.793644 (September 12), based on 3,719 fixed
targets. Of the 2,036 forecast targets, 703 received a T05 model prediction
and 1,333 used the reference fallback. The integrated T05 + v2 baseline
scored 0.799727 on these same cases; its results are saved as
`prior_t05_v2_score.json` and `prior_t05_v2_scores_by_date_task.csv` in the
same directory. The earlier v3 local score was 0.754722 and is saved as
`outputs/local_score/prior_v3_score.json`. The eight current task/date results are in
`outputs/local_score/scores_by_date_task.csv`; `scores_by_date.csv`,
`predictions.csv`, `scored_predictions.csv`, and `score.json` are in the same
directory. This local score is separate from all public Kaggle scores.

## T05 forecast with v2 other tasks

`scripts/run_t05_integrated.ps1` regenerates the v2 submission from the
project's `src/v2_candidate.py` and the preserved v1 baseline, then fits or
loads the T05 ensemble and predicts the forecast rows. It uses those fresh v2
predictions for hidden, hidden-future, and missing. It writes
`outputs/submission_t05_integrated.csv`, leaving the saved v2 and v3 submissions
untouched. The exact T05 model settings are in `t05_forecast_config.json`;
the model code is in `t05_forecast/`. PyTorch and SciPy are required alongside
the existing dependencies. Fitted parameters are cached under
`outputs/t05_cache/`; deleting that directory makes the next run refit them.

From the project root in PowerShell, with the required packages installed in
the selected Python environment:

```powershell
.\scripts\run_t05_integrated.ps1
```

Set `MADINAH_PYTHON` to a Python executable if it differs from the local
Python 3.12 installation used by the other project scripts. Pass
`--reference PATH` to compare every prediction by ID with the supplied T05
submission, and `--t05-base PATH` to check the regenerated v2 values against
the base file in the T05 package. The verification summary is written to
`outputs/t05_integration_verification.json`.

The integration run reproduced the v2 base exactly across 4,457 targets.
Against the supplied T05 `submission.csv`, the maximum absolute difference
was `4.3200998334214091e-12`, with zero differences above `1e-9` in every
task. The package's `0.808` public Kaggle score is historical; this comparison
only establishes numerical reproduction and is not a new Kaggle score.

## Hidden B/D blend trial

Run the one-change held-out experiment with:

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path + [IO.Path]::PathSeparator + (Join-Path (Resolve-Path .).Path '.vendor'); py -3.12 src\hidden_blend_trial.py
```

It reuses `outputs/local_score/cases.csv` and `released_feed.csv`, fits v2
without the evaluated date, saves predictions, and only then reads the separate
labels. The baseline weights are 0.50 for B and G (dropout Ridge) and 0.25 for
D (ordinary Ridge). The trial changes B to 0.75 and D to 0.50; G stays at
0.50. The result by held-out camera was:

| Date and camera | Targets | Baseline MAE | Candidate MAE | Baseline normalized | Candidate normalized |
|---|---:|---:|---:|---:|---:|
| Sep 11 B | 284 | 6.004 | 5.795 | 0.860873 | 0.830938 |
| Sep 11 G | 276 | 2.856 | 2.856 | 0.916672 | 0.916672 |
| Sep 12 D | 283 | 7.431 | 7.227 | 0.960740 | 0.934333 |

The trial outputs include predictions, date/camera scores, donor availability,
crowd-level groups, and the 20 largest baseline errors in
`outputs/hidden_blend_trial/`. B and D improve whether their preferred donor
is present or absent. Some small high-crowd groups regress (B: 28 targets in
51–100; D: 6 targets in 51–100). The complete local score improves by
0.005910. `src/hidden_candidate.py` generates
`outputs/submission_t05_hidden_v1.csv` from the verified integrated submission;
it reproduced the baseline hidden rows within 1e-9 and changed 569 of 849
hidden predictions, leaving every other task identical.

## Saturday hidden-future prayer trial

The timetable source and schema are documented above. `timetable()` validates
six unique dates, `HH:MM` values, and joins preserving all 13,443 training
and 8,611 test rows. The paired experiment reuses the 482 fixed cases and
separate labels, excludes the evaluated date from fitting, and changes only
the direct model's prayer features. Both variants use the same training
examples, estimator settings, 75% blend, and reference. The existing direct
features already contain clock minute and horizon; the clock control adds
only the Friday flag. The prayer variant adds signed minutes to the nearest
scheduled prayer at `now` and the target, plus a prayer-between flag. Camera
inputs are cut off at `now`; all 482 trial invariance checks pass.

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path + [IO.Path]::PathSeparator + (Join-Path (Resolve-Path .).Path '.vendor'); py -3.12 src\hidden_future_prayer_trial.py
.\scripts\run_local_score.ps1
$env:PYTHONPATH = (Resolve-Path .).Path + [IO.Path]::PathSeparator + (Join-Path (Resolve-Path .).Path '.vendor'); py -3.12 src\hidden_future_prayer_candidate.py
```

The trial saves predictions before reading labels. Its files are in
`outputs/hidden_future_prayer_trial/`. Local held-out results:

| Date | Targets | v2 MAE / normalized | Clock MAE / normalized | Prayer MAE / normalized |
|---|---:|---:|---:|---:|
| Sep 11 Friday | 322 | 3.872 / 0.684085 | 3.838 / 0.678079 | 3.918 / 0.692174 |
| Sep 12 Saturday | 160 | 6.445 / 0.647585 | 6.666 / 0.669832 | 5.496 / 0.552231 |

Within 15 minutes of a scheduled prayer, Friday has 64 targets (v2 MAE
3.170, prayer 2.993) and Saturday has 30 (v2 6.287, prayer 4.704).
Outside those windows, Friday has 258 (v2 4.046, prayer 4.147) and
Saturday has 130 (v2 6.481, prayer 5.679). The earlier v3 exploration
also evaluated September 5 Saturday: prayer improved the 11:24 recording
(37 targets, 7.961 to 5.209 MAE) and the 19:00 recording (29 targets,
7.628 to 7.586). Those recordings were already used to choose the prayer
features, so they are supportive but not independent validation.

The current local benchmark applies prayer features to Saturday
hidden-future only. Its score is **0.781898**, versus **0.793817** before
this change (lower is better); Saturday hidden-future normalized error is
0.552231 versus 0.647585, while Friday remains 0.684085. The candidate
`outputs/submission_t05_hidden_hidden_future_v1.csv` is built from
`submission_t05_hidden_v1.csv`; it contains all 4,457 IDs in sample order
and changes only 162 September 19 hidden-future values. All 484 base
hidden-future rows were reproduced from the v2 method within 1e-9 before
building it. The historical **0.808 public Kaggle score** belongs to the
original T05 submission, not this local candidate.

## Friday early-arrival hypothesis (local experiment)

Run ` .\scripts\run_friday_arrival_trial.ps1 ` from the project root. Results
are saved in `outputs/friday_arrival_trial/`: `scores.csv`,
`observed_patterns.csv`, `predictions.csv`, `scored.csv`, and
`verification.json`. The current full baseline was rerun and reproduced
**0.7818975063504505**. The Friday experiment is not adopted; Saturday
prayer features and all production prediction paths remain as before.

The prespecified candidate adds only `now - Friday Dhuhr` and
`target - Friday Dhuhr` to the v2 direct features (NaN on non-Friday
training rows). Dhuhr is the supplied local timetable reference, not an
observed sermon, iqama, or movement time: 12:21 on September 4 and 12:18 on
September 11. Training rows, targets, estimator parameters, reference, and
75% direct blend match v2. Each evaluated date is entirely excluded from
fitting. B and G are masked together. Predictions are saved before scoring;
the count-pattern analysis runs only after prediction and scoring. All 406
post-cutoff checks pass (322 fixed September 11 cases plus 84 September 4
cases). SHA-256 checks confirm every saved submission is unchanged.

| Date | Camera | Targets | v2 MAE | Candidate MAE | v2 normalized | Candidate normalized |
|---|---|---:|---:|---:|---:|---:|
| Sep 11 | B | 164 | 5.161389 | 5.456324 | 0.639326 | 0.675858 |
| Sep 11 | G | 158 | 2.533940 | 2.596658 | 0.802937 | 0.822810 |
| Sep 11 | Both | 322 | 3.872144 | 4.053134 | 0.684085 | 0.716061 |
| Sep 4 | B | 56 | 7.334129 | 6.270531 | 0.585430 | 0.500531 |
| Sep 4 | G | 28 | 4.728633 | 4.497429 | 1.172519 | 1.115190 |

Window boundaries are target-minus-Dhuhr in [-90,-60), [-60,-30), and
[-30,0); all remaining times are outside. Per-camera/window MAE,
normalized errors and zero-count windows are included in `scores.csv`.
September 11 has no targets in the earliest window, 15 per camera in
30-60 minutes, and only one per camera in 0-30 minutes. Outside these
windows, B regresses 5.437018 to 5.802020 (148 cases), and G regresses
2.635686 to 2.700984 (142). The early B gain is insufficient overall.
September 4 was already used in the earlier v3 prayer experiment, so its
improvement is not fresh independent validation. Its 84 cases have no
60-90 minute targets either.

Observed mean counts (observation counts in parentheses), computed after
the paired prediction run:

| Date | Camera | 60-90 min before | 30-60 min before | 0-30 min before |
|---|---|---:|---:|---:|
| Sep 11 | B | 18.801 (18) | 13.949 (30) | 10.639 (30) |
| Sep 11 | G | 9.373 (18) | 8.114 (30) | 7.620 (30) |
| Sep 4 | B | 17.099 (5) | 15.065 (30) | 8.704 (30) |
| Sep 4 | G | unavailable (0) | 5.542 (10) | 5.547 (30) |

The earliest window is partial: September 11 covers only 78-61 minutes
before Dhuhr; September 4 B covers only 65-61 minutes. September 4 G starts
40 minutes before Dhuhr. Later windows are complete except that G 30-60
window. These readings show no consistent count rise approaching Dhuhr.
Counts in a camera view alone cannot establish the direction of movement
toward the mosque. The limited coverage and conflicting held-out results
support retaining the current Friday method and full local score 0.781898.

## I-M forecast fallback trial

Rerun the fixed experiment and build/check its separate candidate with:

```powershell
.\scripts\run_im_forecast_fallback_trial.ps1 --build-candidate
```

The current full benchmark was rerun and reproduced **0.7818975063504505**.
That benchmark has A-H forecast cases, so it does not measure this I-M
change. The experiment reuses the causal rolling-origin idea from
`v3_unseen_forecast_proxy.py`, now with all released I-M target minutes and
every horizon 1-15 instead of every eighth minute and only 5/10/15.
Cases and their last-three-reading contexts are fixed in `cases.csv`;
answers are in a separate `labels.csv`, read only after predictions are
saved. These files and results are in `outputs/im_forecast_fallback_trial/`.
The estimator-free alternative needs no fitting on either test recording.

T05 explicitly sets I-M forecast model values to NaN via
`forecast_reference_units`, then uses `ref_forecast`: the latest finite
reading within five minutes of `now`, inclusive. A reading older than five
minutes, or no previous reading, yields NaN; no additional numeric fallback
exists in this path, and submission validation would reject that value.
This is not unlimited-age persistence. Actual T05 fallback values were
reproduced for all 555 real requests and matched the saved base within 1e-9.
All 555 have at least three readings through `now`; none have zero, one or
two. I-L each have 60 requests on each date; M has 15 Friday and 60 Saturday.

Before looking at errors, input coverage was used to choose one rule:
if the three most recent readings are all at most five minutes old, use
`0.5 * (latest + median(last_three))`; otherwise retain T05's fallback.
All 555 real requests satisfy this rule, with the latest reading aged 0-1
minutes. It changes 405 predictions; the remaining 150 equal the fallback.
The rule and coverage counts are saved in `verification.json` and
`coverage_by_date_camera.csv`.

Of 13,035 fixed pseudo-forecast pairs from 869 distinct released target
readings, 7,121 have a finite actual T05 fallback (813 distinct readings).
The other 5,914 are explicitly unscoreable in the paired experiment; neither
method invents a value for them. The alternative has enough context on
5,587 pairs and changes 4,028. On all scoreable pairs, MAE falls from
11.017690 to 10.584249 (3.93%); on changed pairs, 11.283910 to 10.517640
(6.79%). Every date/camera group improves:

| Date | Camera | All cases | T05 MAE | Candidate MAE | Changed cases | T05 MAE on changed | Candidate MAE on changed |
|---|---|---:|---:|---:|---:|---:|---:|
| Sep 18 | I | 859 | 10.121 | 9.714 | 467 | 10.995 | 10.247 |
| Sep 18 | J | 949 | 11.755 | 11.532 | 547 | 11.894 | 11.506 |
| Sep 18 | K | 913 | 13.447 | 12.294 | 559 | 12.711 | 10.828 |
| Sep 18 | L | 634 | 12.526 | 12.146 | 392 | 13.052 | 12.436 |
| Sep 18 | M | 114 | 3.153 | 3.016 | 37 | 3.220 | 2.799 |
| Sep 19 | I | 682 | 10.182 | 9.717 | 373 | 9.883 | 9.033 |
| Sep 19 | J | 740 | 10.764 | 10.324 | 432 | 11.595 | 10.841 |
| Sep 19 | K | 775 | 14.170 | 13.786 | 357 | 14.967 | 14.133 |
| Sep 19 | L | 784 | 11.203 | 11.135 | 473 | 11.118 | 11.005 |
| Sep 19 | M | 671 | 5.000 | 4.669 | 391 | 5.557 | 4.990 |

| Horizon | All cases | T05 MAE | Candidate MAE | Changed cases | T05 MAE on changed | Candidate MAE on changed |
|---|---:|---:|---:|---:|---:|---:|
| 1-5 | 3426 | 9.585 | 9.124 | 2013 | 9.852 | 9.068 |
| 6-10 | 2227 | 11.765 | 11.285 | 1215 | 11.937 | 11.057 |
| 11-15 | 1468 | 13.228 | 12.929 | 800 | 13.895 | 13.347 |

`scores.csv` contains the full date x camera x horizon breakdown, for all,
changed, and real-context-matched cases. At horizons 11-15 on Saturday,
J regresses 10.520 to 10.639 (145 cases), K 17.054 to 17.330 (146), and
L 16.122 to 16.478 (146). There is no post-hoc camera/horizon tuning.

Coverage is not identical to the real requests. Real requests have 185
targets in each horizon band, whereas validation favors shorter horizons.
Restricting validation to the same freshness range as the real requests
(latest <=1 minute, all three <=5) leaves 5,457 pairs from 700 distinct
readings, improving 11.094993 to 10.554867 with gains in all ten
date/camera groups. Friday M is weakly covered: 44 such pairs from only
12 readings, with none at horizons 11-15. Saturday M has 495 matched pairs.
The pairs overlap across horizons and come from only two dates; they are
not thousands of independent events. Actual forecast target labels remain
unknown. `coverage_by_horizon_band.csv`, `coverage_by_context_age.csv`, and
`coverage_by_hour.csv` document the input-distribution differences.

The measured consistency supports retaining this as a separate candidate,
`outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv`, based on
`submission_t05_hidden_hidden_future_v1.csv`. It preserves all 4,457 IDs
in sample order, finite nonnegative values, and changes only 405 I-M
forecast predictions (225 Friday, 180 Saturday). A-H forecast, hidden,
hidden-future and missing values are copied verbatim. All preexisting
submissions retain their SHA-256 hashes. All 13,590 cutoff checks pass
(13,035 pseudo cases plus 555 actual requests). Candidate verification is
saved as `candidate_verification.json`; rerunning checks an existing
candidate rather than overwriting it. Suggested commit message:
`Add recency-gated median fallback candidate for I-M forecasts`.

## A-H missing correction median trial (rejected)

```powershell
.\scripts\run_missing_targeted_trial.ps1
```

The current complete local benchmark was rerun and reproduced
**0.7818975063504505**. The experiment reuses the 358 fixed missing targets
from September 11 and 12, with exactly the same released feed and reference.
Each evaluated date is excluded from calibration. Predictions are saved
before reading the separate labels; all 358 baseline values match the
current v2 predictions within 1e-9. Files are in
`outputs/missing_targeted_trial/`.

V2 starts with time-aware endpoint interpolation. For gaps of at least six
minutes, it applies a calibrated donor correction on A/B/C/E/F/G when useful
donors are available. Pair slopes are trained on historical deviation
windows, retaining pairs whose training absolute error improves. It averages
up to four donors using their training improvements as weights, limits the
correction to half the interpolated count (minimum scale 5), and uses the
camera weight: A/B/C/E/G 1.0, F 0.5, D/H 0.0. D/H, short gaps and unavailable
donors retain interpolation. Donor deviations are anchored to both observed
ends of the gap.

The fixed targets represent 38 distinct outages: 17 Friday and 21 Saturday.
Baseline MAE by camera is A 9.904 (6 outages), B 2.323 (3), C 3.469 (6),
D 4.205 (3), E 8.354 (6), F 6.853 (5), G 1.447 (3), and H 2.872 (6).
Largest outage totals come from Friday A 16:26-16:42 (15 masked minutes,
MAE 16.570), Friday E 20:05-20:19 (13, 12.447), Friday A 19:06-19:18
(11, 10.660), and Saturday E 16:29-16:38 (8, 13.352). The largest individual
error is Saturday E at minute 993, 31.994 people.

For descriptive grouping, a sharp observed change means the range of the
target's released counts in the five minutes around both endpoints is at
least max(10, 50% of the endpoints' mean). No hidden count enters this flag.
The binary flag is a relative threshold; high absolute changes can remain
below it at high crowd levels. Group coverage and baseline error:

| Gap length | Sharp | Target minutes | Distinct outages | v2 MAE |
|---|---|---:|---:|---:|
| 1-5 | No | 5 | 1 | 0.874 |
| 6-10 | No | 142 | 18 | 4.761 |
| 6-10 | Yes | 54 | 7 | 6.132 |
| 11-20 | No | 107 | 8 | 3.650 |
| 11-20 | Yes | 50 | 4 | 9.658 |

Several large errors coincided with isolated donor-correction spikes:
Friday A minute 1051 had a +27.482 correction when interpolation error was
only 2.010. One targeted alternative therefore takes the median of the v2
correction at minute-1, minute and minute+1, adds it to the unchanged linear
trend, and retains the v2 correction limits. Corrections at both observed
endpoints are zero; all 76 endpoint checks pass. Calibration, camera weights,
and masks are identical; no alternative settings were searched.

| Held-out date | Targets / outages | v2 MAE | Median trial MAE | v2 normalized | Trial normalized |
|---|---:|---:|---:|---:|---:|
| Sep 11 | 163 / 17 | 6.549358 | 7.151343 | 0.761470 | 0.831460 |
| Sep 12 | 195 / 21 | 4.191951 | 4.173008 | 0.752627 | 0.749226 |

Friday has 5 improved, 6 regressed and 6 unchanged outages. Saturday has
9 improved, 9 regressed and 3 unchanged. Saturday's total absolute-error
gain is only 3.694; its best outage alone (E 16:29-16:38) contributes
18.596, so excluding it reverses the gain to a loss of 14.902. The largest
Friday regression is E 20:05-20:19, MAE 12.447 to 18.744. Median filtering
also removes useful rapid donor movement, so this experiment is rejected.

`candidate_scores.csv` gives date/camera scores; `candidate_by_outage.csv`
shows the concentration of gains; `outages.csv`, `by_length_sharp.csv`, and
`largest_errors.csv` provide the diagnosis. Production code and all saved
submissions are unchanged (SHA-256 checked). The retained combined candidate
is `outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv`; the
complete local score remains 0.781898. No missing candidate CSV is generated.

## A-H weekly-template forecast diagnostic

The user-confirmed public Kaggle score is **0.806** for
`outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv`. This
diagnostic preserves that file, every other saved submission, and the
production predictor. Rerun with:

```powershell
.\scripts\run_weekly_forecast_diagnostic.ps1
```

The diagnostic uses the same rolling-origin principle as the earlier I-M
proxy. To obtain distinct forecast windows, origins are fixed every 30
minutes starting at 11:15; every released A-H value 1-15 minutes later is a
target. There are 752 cases in 16 origins on September 18, and 814 in 15
origins on September 19. Labels and case metadata are separate files in
`outputs/weekly_forecast_diagnostic/`. Each prediction recording contains
only released observations through its `now`, with all later values removed.
Predictions are saved before labels enter scoring. B/G have no released
Friday targets and D has none on Saturday; those camera/date combinations
cannot be evaluated by this proxy.

T05's actual A-H template selects the full-length same-weekday historical
recording (September 11 for September 18, September 12 for September 19).
It smooths log1p counts using exponential distance weights, bridges short
historical gaps, and warps clock time between solar-computed Dhuhr/Asr/
Maghrib/Isha anchors. The shift reads the prior recording about 2.49-7.42
minutes later over these recordings. These are the existing T05 computed
anchors, not the supplied prayer CSV. Alignment of levels across historical
sources is already implemented but has no multi-source effect when only the
full-length source is selected. Missing template cells can be completed
using regressions fitted on other eligible historical recordings. The
six-member ensemble has five lam=0.8 templates and one lam=0.7; the standalone
template comparison averages their people-scale predictions just as the
forecast ensemble does.

An exclusion audit found an indirect dependency in the saved T05 fits:
the other test day's I-M template completion can be trained using the
evaluated recording's released A-H/I-M observations. Those observations are
now pseudo-targets in this local diagnostic. The original frozen T05 is
therefore reported explicitly as an audit, alongside a diagnostic-only
full refit with the evaluated recording excluded from template fitting and
covariance fitting. That refit keeps all estimator settings (six members,
140 steps) and excludes 374/405 indirectly dependent fitting cells. Its
cache is isolated under this diagnostic's `fit_cache/`; the submitted cache
is untouched. `fit_exclusion_audit.csv` documents the fitting cell counts.

The latest-reading comparator uses T05's inclusive five-minute recency rule.
Three-way comparisons are paired only where every method is finite. The
frozen submitted T05 audit on the common cases is:

| Date | Common targets | Origins | Frozen T05 MAE | Weekly template MAE | Latest MAE |
|---|---:|---:|---:|---:|---:|
| Sep 18 | 403 | 9 | 17.155 | 23.810 | 26.691 |
| Sep 19 | 531 | 9 | 11.230 | 16.765 | 23.779 |

On all primary cases, frozen T05 predicts 752/814 using the model, with
zero fallbacks and MAE 18.967/13.420. The reduction to 403/531 common cases
comes from stale/unavailable latest readings. `released_all_t05.csv` reports
all-case source counts and MAE; `released_by_date.csv`, `released_by_horizon.csv`,
and `released_by_camera_horizon.csv` report paired comparisons. The frozen
audit is additionally preserved under `frozen_audit/`.

At each `now`, level divergence is log1p(latest) minus log1p(template) at the
latest observation's timestamp. Fixed bands are within a 1.25 ratio, between
1.25 and 2, and over 2. Trend compares slopes fitted to observed log counts
and the template at those same observed times in the preceding ten minutes;
it requires at least three points spanning five minutes. Slopes below 0.005
log units/minute are weak/flat. Diagnostics never use target counts. Groups
with fewer than 30 target cells or clock timestamps, or five camera-windows/origins are
flagged. Cases can share clock timestamps across cameras; reports show both
target timestamps and origin counts so these are not treated as independent.

The original frozen T05 audit beats persistence both within 25% of the
template and at moderate divergence, on both dates. Severe divergence has
only 15 common cases from one Saturday camera-window. Friday's flat-trend
group appears worse than the template, but removing the single 11:15 origin
reverses that comparison. `released_window_sensitivity.csv` and
`released_camera_origin_scores.csv` record these concentration checks.
`five_examples.csv` contains five largest T05 errors from distinct recording/
origin windows, with all three predictions and observed target counts.

The September 11-12 fixed benchmark is scored separately and reproduced
without overwriting its outputs: all-case forecast MAE 37.231/17.281,
model counts 241/462 and fallback counts 695/638. Its existing local fallback
uses a 30-minute last-reading rule; the paired latest comparator still uses
five minutes. Only 241/462 cases have the historical template and recent
reading required for the paired comparison. Prior same-weekday recordings
are short and cover discontinuous parts of the day. On that common subset,
T05/template/latest MAEs are 22.432/20.564/43.370 Friday and
10.993/10.498/18.828 Saturday. The template is better at horizons 11-15
on these older checks, while T05 is better on both newer recordings;
this coverage difference limits transfer of a blanket template-weight rule.

Input coverage also differs between the released-target proxy and the real
requests. All 975 Friday and 1,130 Saturday A-H forecast requests have both
a five-minute latest value and enough ten-minute history for the trend
diagnostic. The pseudo cases have only 403/531 recent latest values and
311/417 complete trend contexts. Among those context-matched cases, the
original frozen T05 still beats template/latest overall: 17.510 versus
20.818/24.760 Friday and 11.331 versus 15.175/24.569 Saturday.
`input_coverage_summary.csv` and `released_matched_input_by_horizon.csv`
document this. Saturday's context-matched 11-15 minute group favors the
template, but almost all the gain comes from one evening origin; removing
that origin removes the evidence for a broad long-horizon rule.

The five examples are chosen from different recording/origin windows and
explicitly flag whether their input context matches the real requests.
Two conspicuous A failures at 18:15 have insufficient recent trend history,
unlike the actual forecast requests, so they cannot alone justify a new
fallback policy. The severe-divergence example is Saturday G at 18:15:
latest 36.683, with subsequent latest/T05 MAE 28.848/1.852, based on only
one origin. Levels diverging from the template do not by themselves show
that the template's influence should be reduced.

One exploratory next experiment is a small bounded contribution from a
damped ten-minute observed trend for A/E forecasts at horizons 6-10,
retaining T05 where the context is insufficient. In the original frozen
diagnostic, when recent and template trends agree, these cameras lag the
observed direction on 80% of 20 Friday cases and 95.5% of 22 Saturday cases,
from only three origins per date. This is a hypothesis for a paired trial,
not a measured gain from a new predictor. The older Saturday check has the
opposite signed bias in 10 cases from two origins; it uses short prior
recordings and the existing 96-cell fitting proxy rather than the full
primary fit. That is a concrete failure risk to check before adoption.
No trend correction or new submission is implemented in this diagnostic.

The completed fully excluded refit confirms the paired ranking. The
following table reports the exact frozen current T05 and the clean refit
separately, on identical three-way-common targets:

| Date | Horizon | Targets | Current T05 MAE | Clean-fit T05 MAE | Weekly MAE | Latest MAE |
|---|---|---:|---:|---:|---:|---:|
| Sep 18 | 1-5 | 194 | 16.917 | 16.547 | 24.125 | 20.785 |
| Sep 18 | 6-10 | 149 | 19.717 | 18.409 | 24.576 | 30.146 |
| Sep 18 | 11-15 | 60 | 11.564 | 10.956 | 20.893 | 37.210 |
| Sep 19 | 1-5 | 237 | 9.992 | 9.751 | 18.731 | 12.533 |
| Sep 19 | 6-10 | 185 | 11.590 | 10.962 | 15.778 | 26.806 |
| Sep 19 | 11-15 | 109 | 13.313 | 11.784 | 14.165 | 43.096 |

The clean refit's common-case MAEs are 16.402953/10.590168; its all-case
MAEs are 20.402136/14.354977, worse than the frozen model's all-case MAEs
18.966670/13.419983. Thus this fit-isolation control is not adopted as a
production replacement. On real-context-matched cases, clean MAEs are
16.886214/10.544087. Moderate level divergence still favors T05 over
latest on both days: current 24.151 versus 34.283 (167 cases) and 10.779
versus 24.105 (207 cases); clean-fit values are 22.740 and 9.920. The
proposed A/E medium-horizon trend experiment remains exploratory: the clean
fit also lags observed direction on the same 80%/95.5% of those 20/22 cases,
with signed mean lag 20.492/37.472 people. No prospective gain is claimed.

The 11-15 minute groups have only four origins and 12/19 distinct clock
timestamps, and are flagged sparse. `released_by_direction.csv` and
`released_by_signed_level_trend.csv` distinguish rising/falling trends and
levels above/below the template. Camera-level results with target and origin
counts are in `released_by_camera.csv` and `released_by_camera_horizon.csv`.
`verification.json` confirms 9,396 primary member/cutoff checks and 2,036
benchmark checks; all preexisting submission SHA-256 hashes are unchanged.
The first run fits twelve diagnostic models in full and can take about an
hour on this machine; subsequent runs reuse those fit parameters and
generate the diagnostic predictions again. `--summarize` rebuilds reports
from saved diagnostic predictions, while `--frozen-only` is a separate audit
of the original cache and does not meet the full fit-isolation condition.
This comparison is diagnostic rather than a causal template ablation: T05
also differs from persistence in its donor and temporal components.

### Prompt 22: audit of cross-camera forecast information

Run `powershell -ExecutionPolicy Bypass -File .\scripts\run_cross_camera_forecast_audit.ps1`.
Results are in `outputs/cross_camera_forecast_audit/`. This command requires
the clean weekly diagnostic caches and refuses to fit if any cache is missing.
It does not alter any predictor, fixed cases, labels, cache, or submission.

Inspection of `t05_forecast/gp.py` found no missing cross-camera observation
input: `observed` collects all released, finite cells with finite templates;
`predict_recording(strict=True)` limits them to `t <= now`. The GP conditions on
their entire log-count deviation histories. Learned camera covariance matrices,
multiple temporal kernels, camera delays, template-slope timing covariance, and
(in configured members) same-minute covariance already transfer information
between cameras. Missing camera inputs are omitted, not filled with future
values. Levels and recent changes are represented by those histories, although
this does not guarantee optimal donor weighting or nonlinear calibration.

Accordingly **no redundant residual model was trained**. Instead, an
inference-only donor-removal ablation measures the information already used:
keep the same template and cached parameters, but condition each target only
on its own camera. This ablation is not a proposed correction or submission.
The command recomputes the clean T05 predictions and checks agreement with the
weekly diagnostic to 1e-9 before scoring. It excludes the evaluation recording
both directly and through fitting-template completion. Production/frozen
parameters retain the indirect dependency documented above and are not used
to claim clean held-out accuracy. No new fit of any ensemble member is run.

| Date | All cases / windows | Clean T05 MAE | Own-camera-only MAE |
|---|---:|---:|---:|
| Sep 18 | 752 / 16 | 20.402136 | 19.838485 |
| Sep 19 | 814 / 15 | 14.354977 | 14.427517 |

For normalized errors, the denominator is five-minute persistence MAE on the
**identical reference-eligible targets**, rather than dropping different cases
per method. All-case normalized errors are undefined when that reference is
unavailable. The frozen submitted model's all-case MAEs remain 18.966670 and
13.419983; the clean fit is an isolation control, not a production update.

| Date | Eligible cases / windows | T05 MAE / normalized | Own-only MAE / normalized |
|---|---:|---:|---:|
| Sep 18 | 403 / 9 | 16.402953 / 0.614544 | 15.872825 / 0.594683 |
| Sep 19 | 531 / 9 | 10.590168 / 0.445351 | 11.567719 / 0.486460 |

Removing donors changes all 1,566 pseudo predictions, demonstrating active
cross-camera conditioning. On reference-eligible cases the donor-removal MAE
delta remains negative for Friday (-0.949 to -0.226) and positive for Saturday
(+0.519 to +1.289) after omitting each window in turn. On all cases, Friday
remains negative (-0.884 to -0.247), while Saturday changes sign (-0.321 to
+0.813). Thus blanket donor removal does not improve both dates robustly.
Friday A/E benefit on the reference-eligible subset, while Saturday A/E worsen;
the mechanism is weighting/calibration, not an absence of cross-camera inputs.

Separate older stress results, reusing the verified Sep 11/12 predictions,
are MAE 37.231374 / 17.281041, normalized 0.873513 / 0.840029, on 936 / 1,100
cases with 11 origins each. No donor-removal run or learned correction was
performed on these older cases. Historical template/model coverage is limited
(241 / 462 model predictions; remaining cases use the benchmark fallback).
These scores are not pooled with the recent dates or the Kaggle score.

Files `recent_{all,reference_eligible}_by_{date,camera,horizon,window}.csv`
contain counts, unique windows/timestamps, MAE and eligible normalized errors;
`recent_leave_one_window_out.csv` lists every omitted-window result. Matching
`older_*.csv` files contain the separate stress baseline. Long-horizon
reference-eligible subsets have only four windows and are flagged sparse.
Released pseudo cases contain no Friday B/G or Saturday D answers; they do not
validate those camera/day combinations. All 2,105 real A-H forecast requests
have fresh own-camera and recent-trend context, versus 934 / 728 of the 1,566
pseudo cases, respectively. This coverage mismatch limits transfer claims.

Decision: retain the accepted 0.806 submission; reject adding another model
merely to supply cross-camera histories already represented by T05. A future
experiment could test conservative calibration of existing donor influence,
but this ablation does not establish a transferable improvement. No candidate
submission was generated and zero submission rows were changed. The run took
22.2 seconds, fitted zero models, passed 9,396 member/cutoff checks and template
perturbation checks, and verified SHA-256 preservation of existing submissions,
fixed cases, labels and saved weekly predictions. See `verification.json`.

### Prompt 23: bounded short-term change model (rejected)

Run `powershell -ExecutionPolicy Bypass -File .\scripts\run_short_term_change_trial.ps1`.
Reports are in `outputs/short_term_change_trial/`. Production prediction code
and every saved submission remain unchanged. The protected submission is
`outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv`, with a
user-reported historical public Kaggle score of 0.806. No Kaggle score is
predicted by this experiment.

T05 already conditions on cross-camera observations; the new hypothesis is a
different response function for short-term change, not additional camera
information. One Ridge model predicts `(target - latest)/training_camera_scale`
using 18 horizon-scaled own-camera features: level, 3/10-minute and last-pair
slopes, reading age/interval, short-context availability, time sin/cos, Friday,
and camera offsets. Its standardized mean-loss L2 penalty is 0.1, fixed without
evaluation-label tuning. Six eligible training recordings on four dates give
6,508 examples in 63 nonoverlapping target windows. Camera scales are training
medians; the 99th training quantile of absolute per-minute normalized change
bounds the predicted change. A prespecified 25% blend with T05 caps its
innovation using the corresponding 95th training quantile. There is no model
search or blend-weight tuning. Rapid rise/fall groups use the training 75th
quantile of absolute normalized 10-minute slope (0.041155/minute), not future
target changes.

Eligibility requires an own-camera reading no older than five minutes and
at least three readings over a span of five minutes within the last ten.
Recent cases and labels are unchanged. Only 311/752 Friday and 417/814 Saturday
fixed cases qualify, each covering eight distinct 30-minute origins with
nonoverlapping 1-15 minute target windows. Cached clean T05 predictions from
the weekly diagnostic exclude each evaluation recording both directly and
through fitting-template completion. These isolated-fit baseline scores differ
from the frozen production cache; production is not replaced. The change
model fits only the six released training recordings, never Sep 18/19. Every
feature reads a recording prefix ending at `now`; evaluation labels are joined
only after recent, older and actual-request predictions have been saved.

| Date | Cases / windows | Isolated T05 MAE | Change model MAE | Bounded blend MAE |
|---|---:|---:|---:|---:|
| Sep 18 | 311 / 8 | 16.886214 | 24.663127 | 16.619071 |
| Sep 19 | 417 / 8 | 10.544087 | 21.121632 | 12.626780 |

The blend improves Friday by 0.267 people (1.58%) but worsens Saturday by
2.083 (19.75%). Friday wins four of eight windows; leaving out each window
gives gains from -0.571 to +1.137, so the gain is not robust. Saturday wins
five small windows but loses heavily at 20:15 (91 cases, blend MAE 33.491
versus 24.475); its leave-one-window-out gain stays negative, -2.497 to
-0.147. Camera F on Friday regresses 19.067 to 22.516; Saturday A/E regress
28.006 to 34.512 and 19.375 to 25.362. Saturday horizons 6-10 regress
11.039 to 13.419; 11-15 regress 13.059 to 18.822 (only three origins).

| Date / known-at-now movement | Cases / windows | T05 MAE | Change MAE | Blend MAE |
|---|---:|---:|---:|---:|
| Sep 18 rapid rise | 4 / 1 | 13.572 | 3.523 | 11.997 |
| Sep 18 rapid fall | 38 / 4 | 3.389 | 4.102 | 3.212 |
| Sep 18 quieter | 269 / 8 | 18.842 | 27.882 | 18.582 |
| Sep 19 rapid rise | 67 / 4 | 3.497 | 4.468 | 3.680 |
| Sep 19 rapid fall | 34 / 4 | 4.503 | 4.945 | 4.429 |
| Sep 19 quieter | 316 / 8 | 12.688 | 26.393 | 15.406 |

Rapid movement groups have at most four origins and the Friday rise group
only four examples, so they cannot establish a general improvement. These
are categories of recent observed movement, not labels of future rises/falls.
Separate older stress folds exclude the entire evaluated training date from
Ridge fitting and use the saved masked observation feed and T05 predictions:
Sep 11 (925 eligible, 11 origins), T05/change/blend MAE
37.622657/45.845940/38.279161; Sep 12 (1,100, 11 origins),
17.281041/19.832240/16.548677. Older T05 template/model coverage is limited;
its fallback is preserved. The older results are not pooled with Sep 18/19.

Decision: reject. The approach fails the prespecified requirement of at least
1% gain on both recent dates, positive gain after leaving out every window,
at least three winning windows/day, and no major camera/horizon regression
greater than max(1 person, 10% of baseline MAE). All 2,105 actual A-H forecast
requests have eligible input context and would change under the tested blend,
but this is coverage, not evidence of improved accuracy. No submission was
created. Features/predictions, per-date/camera/horizon/movement/window MAEs,
all leave-one-window-out results, training manifest and frozen design are
saved. The successful command took about 3.6 seconds including startup
(1.56 seconds inside the script), fit one primary and two small stress Ridge
folds, fit zero T05 members, passed 20,681 future-perturbation feature checks,
and verified SHA-256 preservation of submissions, fixed files and T05 source.

### Prompt 24: causal online forecast bias correction (rejected)

Run `powershell -ExecutionPolicy Bypass -File .\scripts\run_online_bias_trial.ps1`.
Results are in `outputs/online_bias_trial/`. No production predictor or existing
submission is modified. The protected 0.806 Kaggle score is user-reported and
historical; the following numbers are local MAEs, not Kaggle score estimates.

One rule was fixed before evaluation labels were read: add half the median
of the same camera's eligible errors from the preceding 90 minutes, requiring
at least ten distinct released target timestamps and two original forecast
origins. Cap its magnitude at max(1 person, 10% of that camera's median count
on eligible training recordings). The 90-minute memory spans three intervals
of the existing 30-minute diagnostic grid. No thresholds, shrinkage, caps or
memory were selected using Sep 18/19 errors; no second rule or search was run.
Older stress caps exclude the evaluated training date.

Earlier forecasts are reconstructed exactly at their original cutoffs with
load-only cached parameters. All 1,566 clean and frozen recent predictions
match the saved diagnostic to absolute tolerance 1e-9. The clean baseline
excludes each evaluated recording directly and from fitting-template
completion; it is an isolation control, not a replacement for production
parameters. Frozen production-style past predictions are used only to
estimate actual-request correction coverage/application.

At each current `now`, the history reader first takes the released recording
prefix strictly BEFORE `now`. A past error requires
`past_origin < past_target < current_now`, the original cell status RELEASED,
a finite reading, the same recording and camera, and a finite causal T05
prediction. No masked hidden label or future target is used. Previously scored
pseudo targets can contribute later only because their ORIGINAL recording
status is RELEASED and their timestamp has already passed. Error histories
are read from released data, never from the evaluation labels file. Saved
per-row traces include past IDs, count, origins and latest eligible target
timestamp. Predictions are saved before joining separate scoring labels.

| Date / subset | Targets | Origins | Unchanged isolated T05 MAE | Bias-corrected MAE |
|---|---:|---:|---:|---:|
| Sep 18 all | 752 | 16 | 20.402136 | 20.894802 |
| Sep 18 corrected | 492 | 12 | 21.041859 | 21.794877 |
| Sep 18 fallback | 260 | 6 | 19.191584 | 19.191584 |
| Sep 19 all | 814 | 15 | 14.354977 | 14.867586 |
| Sep 19 corrected | 454 | 10 | 16.725226 | 17.644308 |
| Sep 19 fallback | 360 | 8 | 11.365830 | 11.365830 |

All subsets compare identical targets. A clock-origin window can contain both
corrected and fallback cameras, so subset window counts need not sum to the
all-case count. Friday improves in six of sixteen windows, Saturday in four
of fifteen. Leave-one-window-out gain (T05 MAE minus corrected MAE) ranges
from -0.626 to -0.178 Friday and -0.813 to +0.028 Saturday. Thus the losses
are not rescued by a stable gain across windows.

| Date | Horizon | Targets / origins | T05 MAE | Corrected MAE |
|---|---|---:|---:|---:|
| Sep 18 | 1-5 | 242 / 11 | 20.076 | 20.310 |
| Sep 18 | 6-10 | 262 / 10 | 20.053 | 20.945 |
| Sep 18 | 11-15 | 248 / 11 | 21.090 | 21.412 |
| Sep 19 | 1-5 | 282 / 11 | 11.726 | 12.786 |
| Sep 19 | 6-10 | 273 / 11 | 14.955 | 15.448 |
| Sep 19 | 11-15 | 259 / 10 | 16.585 | 16.522 |

Camera A regresses 43.284 to 44.775 Friday and 31.754 to 34.546 Saturday;
E regresses on both dates too. Small gains for Friday C/D and Saturday B/C/F
do not offset these losses. Detailed per-camera and independent-origin
tables, including corrected/fallback splits, are in `recent_by_camera.csv`
and `recent_by_window.csv`; every leave-one-window-out value is saved.
These nonoverlapping forecast target windows are not independent recording
replicates, and the correction itself intentionally carries past information
across windows. Only two recent recording dates are available, with no
released Friday B/G or Saturday D answers.

The older fixed benchmark's forecast targets remain permanently absent from
its masked feed, so they must NOT become calibration errors later. Instead,
1,499 additional past forecasts were reconstructed ONLY for actually released
feed cells, using existing sample-96 T05 parameters and the original template
settings; no fit was run. Results are reported separately: Sep 11, 936 targets
(816 corrected), MAE 37.231374 to 39.124251; Sep 12, 1,100 (942 corrected),
17.281041 to 17.735700. Historical template coverage and the benchmark's
30-minute fallback limit interpretation; older scores are not pooled with
recent-date scores.

Decision: reject this rule. Both recent dates and both older stress dates
regress. Of the 2,105 real A-H forecast requests, 1,853 have sufficient earlier
released errors and would change; 252 fall back unchanged. This coverage uses
the existing sparse diagnostic prediction schedule, not an invented stream
of every-minute forecasts. A denser or shorter-memory history would be a
different experiment and was not tuned here. No candidate submission was
created. The run took 70.8 seconds, fitted zero T05 models, passed 24,097
cutoff/history checks plus template perturbation and cached-prediction
agreement checks, and verified SHA-256 preservation of fixed data, predictions,
T05 source and existing submissions. `design.json`, prediction/history traces,
all grouped errors, and `verification.json` preserve the audit.

### Prompt 25: independent nonlinear A-H forecast (rejected)

Run `powershell -ExecutionPolicy Bypass -File .\scripts\run_independent_forecast_trial.ps1`.
`--prepare` performs only the feasibility/feature check. A saved plan is reused
only if its protected inputs/source and prepared feature hashes still match;
otherwise prepare again before evaluating. Reports are saved in
`outputs/independent_forecast_trial/`. The protected production predictor and
`submission_t05_hidden_hidden_future_im_fallback_v1.csv` remain unchanged.
Its 0.806 public Kaggle score is user-reported and historical; no expected
Kaggle score is inferred here.

Feasibility passed before training: six training recordings on Sep 4, 5, 11
and 12 supply 6,523 eligible examples at 63 nonoverlapping forecast origins;
both evaluated dates are later than all training recordings. Fixed cases
and cached fully isolated T05 predictions match exactly by metadata/IDs.
The new model uses no template, no T05 prediction, and no T05 error as an
input or target, so there is no indirect template completion path to leak
the evaluated recording. No T05 fit or inference was run in this experiment.

One installed sklearn HistGradientBoostingRegressor predicts the change in
log1p count from the latest own-camera reading. Its 55 features cover own
recent readings/means/slopes/age/count, A-H levels/slopes/availability/ages,
camera, horizon, clock/weekday and supplied scheduled prayer-relative times
at now and target. Prayer data is the existing local timetable; it is not
a sermon/iqama or crowd-motion timestamp. All recording-time reads are
explicitly sliced through `now` before aggregates. Missing features use
native NaN splits plus availability, age and point-count indicators. Only
an own reading no older than five minutes is required for eligibility.

Hyperparameters were fixed before evaluation: absolute-error loss, 100
iterations, learning rate 0.05, at most 15 leaves and depth 4, minimum leaf
size 40, L2=10, early stopping disabled, seed 25. Training weights give each
camera/origin equal total weight across its horizons. Predicted log change
is bounded by the 99.5th training percentile of absolute change. There was
one fit, one evaluation pass, no tuning and no blend selection. Preprocessing,
subgroup thresholds and output bounds use training observations/labels only.
Recent and actual-request predictions were saved before evaluation labels
were joined for scoring.

| Date | Eligible / fixed cases | Origins | Isolated T05 MAE | Independent model MAE |
|---|---:|---:|---:|---:|
| Sep 18 | 403 / 752 | 9 | 16.402953 | 25.958044 |
| Sep 19 | 531 / 814 | 9 | 10.590168 | 18.447945 |

The independent model wins five Friday origins and one Saturday origin, but
large losses outweigh those wins. Leaving out every origin in turn still
leaves it worse on both dates: T05-minus-independent MAE gain ranges from
-11.381 to -6.281 Friday and -9.071 to -3.809 Saturday. A/E/F regress
substantially on both dates; Friday D improves from 4.795 to 3.887 over 58
targets/seven origins, which does not justify selecting a camera-specific
replacement after looking at evaluation errors.

| Date | Horizon | Targets / origins | T05 MAE | Independent MAE |
|---|---|---:|---:|---:|
| Sep 18 | 1-5 | 194 / 9 | 16.547 | 22.375 |
| Sep 18 | 6-10 | 149 / 7 | 18.409 | 27.558 |
| Sep 18 | 11-15 | 60 / 4 | 10.956 | 33.570 |
| Sep 19 | 1-5 | 237 / 9 | 9.751 | 10.053 |
| Sep 19 | 6-10 | 185 / 7 | 10.962 | 19.137 |
| Sep 19 | 11-15 | 109 / 4 | 11.784 | 35.532 |

Long-horizon groups have only four origins and are flagged sparse. Known-at-now
movement groups use the training 75th percentile absolute 10-minute log slope
(0.040515/minute), requiring three points spanning five minutes. Saturday
rapid rises regress 22.187 to 50.526 (78 targets, only two origins). Saturday
rapid falls improve slightly, 4.024 to 3.938 (30 targets/four origins); Friday
rapid falls are essentially unchanged, 3.749 to 3.844 (26/three). No eligible
Friday rapid-rise group exists at this prespecified threshold. Quieter groups
also worsen. A separate scorer-only abrupt-future-change diagnostic, using
the training 90th percentile absolute log change (0.668230), shows T05 versus
independent MAE 64.723 versus 113.713 Friday (30/four origins), and 28.347
versus 75.402 Saturday (66/five). Future change is NEVER used as a feature,
eligibility criterion, model-selection input or application gate.

Limited recent context (fewer than three observations or a span under five
minutes in the last ten) is a distribution mismatch: only 15 training cases,
but 206 of 934 eligible evaluation cases. Their MAEs regress 14.769 to
35.731 Friday and 10.759 to 19.906 Saturday. Even on the richer context that
resembles actual requests, the model loses: 16.886 to 23.067 on 311 Friday
cases/eight origins and 10.544 to 18.049 on 417 Saturday/eight origins.
All 2,105 real A-H forecast requests have eligible rich context and could
receive independent predictions; all would differ from the protected file.
This is applicability, not an expected accuracy improvement. Pseudo cases
contain no released Friday B/G or Saturday D target answers; gains for those
camera/day combinations cannot be validated. Only two recent recording
dates are available, and multiple camera targets share each crowd event.

Decision: reject replacement and do not recommend an untested blend. No
candidate submission was created. Both dates fail the prespecified >=3%
gain, leave-one-origin-out robustness and camera-regression criteria.
The two commands (feasibility plus fit/evaluation) took about 8.3 seconds
including startup: 2.06 seconds feature preparation plus 1.79 seconds
training/evaluation inside Python. There was one HGB fit and zero T05 fits.
All 10,210 feature-cutoff perturbation checks passed; SHA-256 checks confirm
that raw data, fixed cases/labels, cached predictions, T05/production source
and saved submissions are unchanged. `plan.json`, `verification.json`,
`by_{date,camera,horizon,window,movement,context,actual_change}.csv`, every
leave-one-window-out result and label-free predictions preserve the trial.

### Prompt 26: nonlinear simultaneous hidden reconstruction (rejected)

Run `powershell -ExecutionPolicy Bypass -File .\scripts\run_hidden_reconstruction_trial.ps1`.
Use `--inspect` for the baseline error analysis alone. All results are in
`outputs/hidden_reconstruction_trial/`. The protected user-reported public
0.806 submission, production predictors and existing benchmark outputs are
unchanged; these are local errors, not expected Kaggle scores.

Inspection of the unchanged current predictor confirmed the updated v2
weights: B uses dropout Ridge at 0.75, G dropout Ridge at 0.50, and D Ridge
at 0.50, blended with the single-donor estimate. Donor ratios are learned
from historical overlapping means, with preferred D/E/B donors for B/G/D;
unavailable preferred donors trigger the next eligible donor, then the
training target median if none remain. Ridge uses donor values/seen flags,
training medians for missing values, clock sin/cos and alpha=3000. Its dropout
variant uses three 20% dropout copies. Below two available donors, it falls
back to the single donor. It may interpolate visible donors over at most
two minutes on either side, using only the released feed.

Before model choice, the largest errors showed a specific limitation beyond
donor absence: global linear/ratio relationships compress different crowd
regimes. In D, counts above 40 have bias -6.781 over 37 cases/six windows,
while counts at most 20 have bias +4.413 over 112/eight. B above 40 has bias
-5.562 over 62/ten. The largest D errors at 13:04 and 13:14 on Sep 12 are
25.530 and 24.028 people even with ALL seven visible donors present. The
13:00-13:30 D window has 18 cases, MAE 13.863 and bias -13.684; 18:00-18:30
has 17, MAE 11.979 and bias +11.979. Baseline analysis by recording, camera,
30-minute window, availability and crowd group is saved with 25 largest
errors. Crowd groups use scorer truth for diagnosis only, not as inputs.

One different method was prespecified: a small nonlinear
HistGradientBoostingRegressor per target camera, with identical settings
across all folds (75 iterations, learning rate 0.07, seven leaves, depth
three, minimum leaf size 25, L2=15, absolute-error loss, no early stopping,
seed 26). It predicts the count directly from exact same-minute visible
A-H log levels, proportions of visible total, availability flags/total,
clock, Friday and scheduled prayer-relative time. No target history,
interpolation or template enters the new model. Native NaN handling and one
fixed 20% donor-dropout training copy address missing visible donors.
With fewer than two exact visible donors, retain the unchanged baseline.
No hyperparameter/model search or blend-weight selection was performed.

Each entire evaluated DATE is excluded before fitting, preprocessing or
donor selection. B and G are jointly removed for either Friday target;
Saturday D is entirely removed. Fixed primary cases/feed/reference remain
unchanged. Auxiliary Sep 4/5 cases use a fixed five-minute released-target
grid, mask the corresponding hidden cameras for the ENTIRE recording and
are scored separately. These older dates, like Sep 11/12, have previously
been inspected and are not fresh independent confirmation. This is
leave-date-out validation, not a chronological backtest: an eligible other
training date may be later than the held-out historical date. Production
test dates are later than all released training dates.

| Date / camera | Targets / 30-min windows | Current MAE | New MAE | Current normalized | New normalized |
|---|---:|---:|---:|---:|---:|
| Sep 11 B | 284 / 21 | 5.794805 | 5.726689 | 0.830938 | 0.821171 |
| Sep 11 G | 276 / 21 | 2.855762 | 2.960481 | 0.916672 | 0.950286 |
| Sep 12 D | 283 / 20 | 7.227049 | 4.804217 | 0.934333 | 0.621103 |

Normalized error is MAE divided by the SAME-case reference MAE, separately
for every camera/recording/window. The new model is used on 842/843 primary
cases; one D case retains baseline. The current predictor and reference
were freshly reproduced to absolute tolerance 1e-9 against the saved local
benchmark. Primary date-wide Friday MAE regresses 4.346277 to 4.363344;
Saturday improves 7.227049 to 4.804217. B wins 12/21 camera-windows, G 8/21,
D 16/20. After leaving out every window, B's gain ranges -0.171 to +0.247;
G always loses (-0.218 to -0.034), while D always gains (+2.116 to +2.678).
Windows are nonoverlapping time blocks, not independent recording replicates;
B and G share the same underlying crowd events.

| Auxiliary recording / camera | Targets / windows | Current MAE | New MAE | Current normalized | New normalized |
|---|---:|---:|---:|---:|---:|
| Sep 4 11:05 B | 35 / 7 | 6.256661 | 5.229148 | 0.990998 | 0.828249 |
| Sep 4 11:05 G | 26 / 5 | 3.409422 | 3.995585 | 0.838266 | 0.982385 |
| Sep 5 04:26 D | 41 / 7 | 9.158452 | 9.719344 | 1.001437 | 1.062768 |
| Sep 5 11:24 D | 24 / 5 | 6.357324 | 5.562904 | 1.011673 | 0.885253 |
| Sep 5 19:00 D | 24 / 5 | 9.490766 | 6.829160 | 0.921416 | 0.663012 |

Auxiliary coverage is 140/150 new-model predictions, with ten morning D
fallbacks. D's main gain repeats at midday/evening but not in the short
morning recording; G worsens on both evaluated Fridays, including +17.2%
on Sep 4. Small auxiliary groups and already examined dates limit confidence.
The method therefore fails the prespecified requirement of >=1% gain on
both primary dates, robust leave-one-window-out gains and no hidden-camera
regression. Reject the joint method; do not choose a D-only production scope
after inspecting this trial's evaluation errors. D is a useful finding for a
separately prespecified future test, not an adopted change here.

For every one of the 993 primary/auxiliary cases, the test replaces ALL
hidden-camera values across the ENTIRE evaluation recording (adding absent
columns too) and verifies identical features and predictions for both methods.
It also mutates held-out hidden values in the original training dataframe
and confirms the filtered fitting data are identical. Every finite new-model
donor feature is checked against an exact-minute genuinely released feed
cell; hidden columns are removed before any aggregation. These same A-H
visible-input types and scheduled timetable are available in the real test;
unavailable donors remain NaN and insufficient context retains baseline.
No hidden-camera labels are used in prediction, fitting or preprocessing;
scoring labels are joined only after all fold predictions are saved.

The main run took 10.48 seconds (about three more seconds for the separate
baseline inspection). Twelve small unit/fold estimators share one model
family and fixed parameters; T05 fits=0. The method was rejected, so the
conditional full four-task benchmark was not run and no candidate submission
was generated. Forecast, hidden-future and missing code and saved predictions
remain unchanged, as do all existing submissions, verified with SHA-256.
`design.json`, `fit_manifest.json`, `verification.json`, prediction files,
`fixed_*` and `additional_*` reports preserve the experiment and every
recording/camera/window MAE, normalization, coverage and omitted-window result.

### Prompt 27: Saturday D-only hidden candidate (locally accepted)

Run `powershell -ExecutionPolicy Bypass -File .\scripts\run_hidden_d_candidate.ps1`.
The runner is separate from production and saves its benchmark under
`outputs/hidden_d_candidate/`. It reproduces the existing nonlinear trial
settings without tuning, keeps B/G and all other tasks on their current methods,
and writes `outputs/submission_t05_hidden_d_only_v1.csv` only after the checks
pass. An identical existing candidate is verified/reused without writing; a
differing existing candidate is never overwritten. `--resume` can finish from
verified completed predictions; `--report-only` regenerates per-recording
omitted-window diagnostics without fitting or predicting.

The D result was reproduced within 1e-9 using the unchanged Prompt 26 code,
75-tree settings, features, dropout and minimum-two-visible-donor fallback.
Every evaluated date is completely excluded from fitting and preprocessing;
Sep 5 excludes all three recordings together. Replacing every hidden D value
throughout each evaluation recording leaves features and predictions unchanged.
No new scope, time gate, feature, parameter or blend was selected from Sep 12
errors: Saturday D-only was specified by the user before this run.

| Held-out recording | Cases / 30-min windows | Current MAE | D-only MAE | Current normalized | D-only normalized |
|---|---:|---:|---:|---:|---:|
| Sep 12 11:00 | 283 / 20 | 7.227049 | 4.804217 | 0.934333 | 0.621103 |
| Sep 5 04:26 | 41 / 7 | 9.158452 | 9.719344 | 1.001437 | 1.062768 |
| Sep 5 11:24 | 24 / 5 | 6.357324 | 5.562904 | 1.011673 | 0.885253 |
| Sep 5 19:00 | 24 / 5 | 9.490766 | 6.829160 | 0.921416 | 0.663012 |

Normalized errors use the same-case reference MAE separately for every
recording/window. New-model coverage is 282/283, 31/41, 24/24 and 24/24,
respectively; the other cases retain current predictions. D improves 16/20
primary windows. Leaving out every window of EACH recording gives gain ranges
(current MAE minus candidate MAE): Sep 12 +2.116 to +2.678; Sep 5 midday
+0.240 to +1.730; evening +1.253 to +3.762; morning -1.176 to -0.006.
Thus the principal gain is not a single-window result and repeats at midday
and evening, while the morning regression remains real. These dates have
already been examined: this is NOT a fresh independent test, and small
auxiliary groups and shared crowd events limit confidence.

The actual Sep 19 hidden-D audit reads target metadata ONLY, never hidden
people values. All 285 targets are between 11:00 and 21:14. All fall in
evaluated Sep 12 time blocks: 216 map to the 16 improving blocks, 69 to the
four regressing blocks; none is temporally uncovered. Counts by clock hour
11 through 21 are 30,30,30,30,25,23,25,27,29,21,15. No actual target falls
in the regressing Sep 5 04:26-08:00 period; 54 overlap its improving midday
period and 65 its improving evening period. These are temporal coverage
comparisons, not claims about Sep 19 target accuracy. There is no new
morning exclusion rule. Exact visible-input coverage is 284/285; one target
keeps the protected baseline.

The full benchmark generated fresh current predictions for all 3,719 fixed
cases once, with T05 fit-cache access strictly load-only. Its baseline matches
the existing saved predictions within 1e-9 and reproduces 0.7818975063504505.
The candidate replaces only the 283 fixed Saturday hidden-D predictions with
the reproduced nonlinear outputs; other predictions/references are identical.
Both full runs use the unchanged `local_score.score` function, byte-identical
scoring labels and the same date/task weighting. Predictions and reports are
in separate `full_baseline/` and `full_candidate/` directories; original
benchmark files remain untouched.

| Local metric (lower is better) | Current | D-only candidate |
|---|---:|---:|
| Forecast normalized task score | 0.856771 | 0.856771 |
| Hidden normalized task score | 0.895612 | 0.738997 |
| Hidden-future normalized task score | 0.618158 | 0.618158 |
| Missing normalized task score | 0.757049 | 0.757049 |
| Overall local score | 0.781897506 | 0.742743705 |
| Pooled hidden MAE, 843 cases | 5.313369 | 4.500010 |

The overall local change is -0.039153801. Task scores above are equal-date
means of each task's normalized error, rather than pooled ratios. Friday
hidden B/G is unchanged. All three other task scores are exactly unchanged.
The user-reported 0.806 public Kaggle score belongs to the protected prior
submission and is not comparable numerically to this local score; the new
candidate has no public result and no expected Kaggle score is claimed.

Submission verification uses the ACTUAL existing production formula: the
saved v1 reference blended with fitted D Ridge at the current weight. Local
held-out validation instead rebuilds its reference inside each training fold.
The proxy reference is not substituted into the production comparison. All
285 protected D predictions were reproduced with the production formula to
1e-9 before replacement; the one insufficient-input case retains its original
protected text value. This distinction required a verification fix, not a
model change or retuning. Completed full predictions were reused to finish
verification; T05 was not rerun or retrained during that finish step.

The new CSV has exactly `id,people`, all 4,457 unique IDs in sample order and
finite nonnegative predictions. Exactly 284 rows change, all hidden Saturday
D; the other 4,172 rows, including every B/G, forecast, hidden-future and
missing value, are EXACTLY equal as text to the protected 0.806 file. The
remaining D fallback is also text-identical. Every previous submission and
production source is preserved; no upload, commit or push was performed.

Recommendation: accept as a separate locally validated D-only candidate,
while retaining the production baseline. The morning regression, 69 real
targets in previously regressing blocks, repeated use of known validation
dates and small auxiliary recordings remain limitations. Computation was
about half a minute: approximately 18.2 seconds for reproduction/full scoring,
6.2 seconds for the finish/verification stage and about two seconds for the
additional recording sensitivity report. No T05 model was fitted. The run
checks 372 historical D masks, 285 actual D masks and 2,518 full-benchmark
forecast cutoffs; fixed data, source and prior artifacts are SHA-256 protected.
`D_by_*`, `D_recording_leave_one_window_out.csv`, `actual_D_time_audit.csv`,
`full_task_comparison.csv`, `full_score_comparison.csv` and `verification.json`
contain the full audit. `files_changed.txt` lists the exact new/modified files.

### Prompt 31: training-only camera relationships (diagnostic; no forecast change)

Run `scripts/run_camera_relationship_diagnostic.ps1`. The Arabic report is
`reports/prompt31_camera_relationships_ar.md`; outputs and three SVG plots are
in `outputs/camera_relationship_diagnostic/`. Only the six older training
recordings on four dates enter discovery. No September 18–19 pseudo labels
or test observations are loaded. Exact-minute observations retain gaps;
1/5-minute changes require complete observed intervals. The screen covers
synchronous associations and directional lags 1–15, plus residual changes
after the existing lam=0.8 prayer-aligned template with the entire target
date excluded from template construction/completion.

A/F and B/G synchronous residual five-minute changes remain associated:
median recording correlations 0.612/0.516 on 651/618 paired cells in five
recordings, with positive signs in 23/24 and 24/25 qualifying 30-minute
blocks. These overlapping pairs and temporal blocks are not independent
recording replicates. E exists on only two training dates and has no usable
independent residual-template coverage here. Morning September 5 also has
no residual-template coverage. B/G are hidden together on the real Friday.

No incremental lag passes the training-only gates. The strongest pooled
partial correlation is A leading B by five minutes (0.252, 641 pairs), but
only 14/23 blocks agree in sign, including 3/8 on September 12; September
11's recording-level partial correlation is only 0.050. T05 already uses
all-camera deviation histories, camera covariance and learned camera delays.
No new predictor, T05 inference/refit, submission or recent-date MAE comparison
was performed. Recommend moving to `missing`. Repeatedly examined September
18–19 dates would not constitute fresh independent validation.

The run took 189.34 seconds, plus 1.96 seconds to regenerate plot titles.
SHA-256 checks preserve 213 existing protected files, including the 0.806
baseline, separate D-only candidate, current notebook and production code.
The user's updated result is that the D-only candidate had the same displayed
public score as the baseline; its hidden-test benefit remains unknown.

### Prompt 32: learned endpoint-preserving missing shape (rejected)

Run `scripts/run_missing_shape_trial.ps1`; the Arabic report is
`reports/prompt32_missing_shape_ar.md`. Results are isolated under
`outputs/missing_shape_trial/`. One pooled regularized Ridge predicts the
normalized departure from endpoint interpolation using gap geometry, exterior
target slopes and exact released donor changes. A zero-at-endpoints envelope,
fixed half-strength shrinkage and level-dependent bound preserve endpoints.
This learns a residual response; it does not smooth existing donor corrections.

Synthetic contiguous gaps transplant the fixed metadata availability patterns
by camera/weekday, retaining donor masks and natural missingness. Each entire
evaluation date is excluded from fitting, preprocessing and baseline donor
calibration. The two folds use 175/141 synthetic gaps (1,557/1,338 targets),
43 features and identical settings. Missing permits released after-gap context;
no masked target or hidden/missing donor count is an input. All 76 endpoint
checks and 358 unavailable/target-interior perturbation checks pass.

The fixed benchmark remains 358 targets in 38 intervals. An explicit check
found 32 fully labeled gaps and six partial gaps caused by natural missingness;
all original cases are retained without inventing the absent labels.

| Date | Targets / intervals | Current MAE | Shape MAE | Linear MAE | Current / shape normalized |
|---|---:|---:|---:|---:|---:|
| Sep 11 | 163 / 17 | 6.549358 | 8.028013 | 8.600943 | 0.761470 / 0.933388 |
| Sep 12 | 195 / 21 | 4.191951 | 5.300601 | 5.569756 | 0.752627 / 0.951676 |

The existing equal-date missing metric worsens 0.757049 to 0.942532.
Pooled MAE worsens 5.265296 to 6.542412; equal-gap MAE 5.102759 to
6.467337. On 300 changed targets in 33 intervals, current/candidate MAE
is 4.946053/6.470078; 58 targets in five intervals retain current predictions.
Only five gaps improve, 28 regress and five remain unchanged. Removing the
two greatest total-error reductions still leaves a 1.488-person pooled MAE
regression. A/E/F regress substantially. The 32 fully labeled gaps alone
also regress on both dates. No scope or setting was revised after scoring.

Actual input coverage is 326/375 A-H missing targets; all 89 I-M remain
unsupported and unchanged. No camera-held-out transfer claim is made.
These are already examined validation dates, not fresh independent evidence.
Reject without tuning; no conditional full four-task run or submission was
created. The successful process took 6.53 seconds (3.25 inside Python), plus
0.32 seconds of saved-output/input checks. Two fold fits, zero T05 fits or
inference. SHA-256 verification preserves 107 protected files including the
0.806 baseline, D-only candidate, current notebook and fixed benchmark.

### Prompt 34: editable teammate notebook, exact baseline preserved

`notebooks/current_baseline.ipynb` now contains eight numbered Arabic sections,
with task targets, visible-input/output examples, editable settings, separate
model loading/training and prediction cells, source pointers and contribution
instructions. See `docs/teammate_baseline.md` for all commands and limitations.
`src/baseline_tasks.py` is the shared task implementation; the notebook and new
`src/evaluate_baseline_task.py` call the same functions. Historical
`local_score.fit_fold/predict_case` also delegates to these task functions.
`baseline_workflow.py` handles file orchestration and the legacy decimal CSV
round trips needed for exact compatibility. Historical builders remain for audit.

Clean-kernel Run All took 37.58 seconds: all 4,457 IDs, finite nonnegative
counts, zero unequal values, and the exact accepted SHA-256
`c7e0a0785fee013449b30a1041abdf939a58461bb5f359c5294e69b2c8ec6cd5`.
Outputs are under `outputs/current_baseline_manual/`, preserving previous files.
D-only remains optional and off. No T05 model was refitted, no upload or commit
was performed. An initial forecast last-bit mismatch was traced to GP target
batch sizes and resolved by retaining the original batches; it was not waived.

The four default per-task evaluations are in `outputs/prompt34_evaluation/`:
hidden, hidden-future and missing match saved predictions exactly; forecast's
maximum difference is 5.68e-14 from the shared blend arithmetic. A separate
`--task forecast-im` proxy uses the same I-M function/settings without fitting.
All 3,719 fixed cases, labels and feeds were also reconstructed exactly from
raw inputs to verify the fresh-clone evaluation path. The forecast evaluation
still requires the historical held-out sample96 covariance cache, which is not
distributed with Git; it refuses to fit or use production assets in its place.
Already examined dates and overlapping I-M proxy targets are not fresh
independent validation. The new task evaluator reads the notebook-exported
`task_config.json`; the legacy full-score command uses defaults and may refit
missing historical caches, so it is not the safe task-specific entry point.

### Prompt 35: improved D-now anchor for Saturday hidden-future (rejected)

Run `scripts/run_hidden_future_d_anchor_trial.ps1` (`--diagnose-only` skips
all fitting). Arabic report: `reports/prompt35_hidden_future_d_anchor_ar.md`;
outputs: `outputs/hidden_future_d_anchor_trial/`. The accepted prayer direct
model excludes D entirely and uses no reconstructed D history. Its only D-now
estimate is the 25% fixed single-donor reference in the final blend. This trial
replaces that component with the unchanged nonlinear D reconstructor, keeping
the 75% prayer model and all settings fixed. No reconstructed features enter
downstream training, so no in-sample stacking path is introduced.

All models and preprocessing exclude the entire evaluation date. D is masked
throughout each recording; reconstruction uses exact visible donors at now,
never future target-time donors, with the previous >=2-donor fallback rule.
Friday B/G remain unchanged (322 targets, 11 shared origins, MAE 3.872144).

| Saturday recording | Targets / origins | Used / fallback | Current MAE | Anchor MAE | Current / anchor normalized |
|---|---:|---:|---:|---:|---:|
| Sep 12 11:00 | 160 / 11 | 160 / 0 | 5.495854 | 5.242204 | 0.552231 / 0.526744 |
| Sep 5 04:26, separate relative-mask audit | 56 / 4 | 42 / 14 | 6.612951 | 6.894927 | 0.575821 / 0.600374 |
| Sep 5 11:24 | 37 / 3 | 26 / 11 | 5.209067 | 6.789208 | 1.166238 / 1.520010 |
| Sep 5 19:00 | 29 / 2 | 29 / 0 | 7.586332 | 6.647476 | 1.158048 / 1.014732 |

The primary gain is 4.62%, but only 6/11 origins win and leave-one-origin-out
gain ranges -0.083209 to +0.481148. Removing the 17:42 origin reverses the gain.
Midday auxiliary origins never improve; morning also regresses. Morning has no
clock overlap with the actual mask and is separately labeled as a start-relative
transplant; midday/evening retain the previous clock-aligned protocol. The
original fixed reference remains the normalization denominator. No time or
horizon rule was selected after scoring. These are repeatedly examined dates,
not fresh independent validation.

Reject without tuning; no conditional full-score run or submission was created.
All 282 combined D/future prediction checks and 564 separate prefix checks pass.
Real input applicability is 162/162 Saturday D hidden-future targets, not an
accuracy claim. Runtime was 17.66 seconds plus 0.61 seconds of additional checks.
Zero T05 fit/inference; SHA-256 preserves 124 protected files, including the
notebook, production code, baseline and independent D-only hidden candidate.

### Prompt 36: Friday B/G hidden-future review (diagnostic; close direction)

Run `scripts/run_friday_hidden_future_diagnostic.ps1`. Arabic report:
`reports/prompt36_friday_hidden_future_ar.md`; outputs:
`outputs/friday_hidden_future_diagnostic/`. This review reuses the accepted
322 September 11 predictions and 84 date-excluded September 4 predictions
from the earlier Friday-arrival trial. No model fitting, tuning or new
prediction is performed. The actual method is 75% direct HGB plus 25% fixed
single-donor estimate at now, without Friday prayer features or B/G histories.

| Recording / camera | Known / mask targets | Origins | MAE | Normalized | Signed bias |
|---|---:|---:|---:|---:|---:|
| Sep 4 B | 56/60 | 4 | 7.334129 | 0.585430 | -1.680144 |
| Sep 4 G | 28/57 | 2 | 4.728633 | 1.172519 | -0.352997 |
| Sep 11 B | 164/164 | 11 | 5.161389 | 0.639326 | -1.513941 |
| Sep 11 G | 158/158 | 11 | 2.533940 | 0.802937 | -0.230241 |

The single observable hypothesis reviewed was absence of the preferred donor
at now. It does not explain B's main errors: D is always present on the long
recording, and B's one old missing-D origin strongly favors the current model.
G's three missing-E origins favor the reference, but two are small differences,
the old recording has no E anywhere, and the long recording contributes only
one such origin. Large G errors also occur with E present. Choosing a missing-E
gate now would select a rule on repeatedly inspected evaluation errors, without
independent training-only calibration. Previous timing/prayer and nonlinear
hidden trials do not establish a different transferable Friday improvement.

Per-camera/recording/horizon/window tables include MAE, normalized error, bias,
coverage and leave-one-window-out results. There are only 15 nonoverlapping
target intervals, 28 camera/origins and two dates, not 406 independent events.
Sep 4 covers 11:05-14:30, lacks E, and has known G targets in only two origins;
it cannot validate afternoon/evening behavior. Recommend closing this direction
with the available evidence. No candidate, full-score run or submission was
created. Runtime 1.74 seconds; all 812 hidden/future feature checks pass and
SHA-256 preserves 148 protected files. Saturday and every other task stay unchanged.
