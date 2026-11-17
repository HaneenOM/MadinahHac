# Madinah Ethical AI Competition forecasting project

This project uses the supplied 2026 release. Its four CSV files match the
`MDA-Ethical-AI-KAUST-Academy.zip` in Downloads byte for byte. The originals
remain in the project root; `data/raw/` contains working copies. The official
starter is preserved as `notebooks/starter-official.ipynb`.

## Collaborator setup

The repository tracks source code, PowerShell benchmark commands, dependencies,
documentation, and the official starter notebook. Competition CSVs, the local
prayer timetable, generated notebooks, benchmark results, and submissions are
ignored by Git. After cloning, place `train.csv`, `test.csv`,
`recordings.csv`, `sample_submission.csv`, and `prayer_times_madinah.csv` in
`data/raw/` from the authorized local copies. Keep any Kaggle submission files
in `outputs/` locally.

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
