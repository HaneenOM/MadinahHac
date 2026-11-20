# P61 research ledger

Session starts 2026-11-19 18:09:25 UTC (21:09:25 Riyadh). Maximum finish 19:39:25 UTC; experiments stop by 19:29:25 UTC. Branch: `development/version7-improvements`. Existing staged P58/P59/P60 files belong to the user and are preserved.

V7 and its user-confirmed public **0.724** score remain accepted. The reference SHA-256 is `2eced7c2ea9765d93d798fc0130126a523e51e69b1fb4ac5bf6d5783d81cc88c`. Small documented reproduction differences are numerical differences, never improvement. No upload, default change, commit or push is authorized or performed.

## Evidence before experiments

- P54 donor-D timing and P58 reconstructed slope failed. P59 matching lacked fixed/mirror support. Do not repeat them.
- P56 convex missing weights failed; G/D hidden smoothing remains exploratory, not automatically included. I-M Ridge forecast failed.
- P60 is a completed missing-gap diagnosis only, not a trained candidate.
- Pre-V7 P32 Ridge gap shape, P46 Hermite, P49/P50 forecast trees and directional lag analyses were reviewed. Their baselines differ; their findings are not treated as V7 results.
- MISSING is promising because V7's A/E partner response is strong, while own-only D/H corrections have mixed support. Existing input masks and pseudo-gap generators can be reused without retraining T05 or pretrained models.

## Evaluation design

Development: 133,581 release-pattern pseudo rows on six source recordings, four dates (Sep 4/5/18/19). Entire source recordings held out. Many rows share physical readings; these are **not** 133,581 independent observations. No Sep 11/12 answers enter MISSING model fitting or applicability selection.

Assessment: fixed and mirror target IDs and release masks, with **both** Sep 11/12 dates excluded from every fit. Unmodified V7 is recomputed with exactly that information restriction. Therefore strict V7 ratios 0.743340 / 0.770864 differ from the historical local V7 0.705973 / 0.752909 MISSING scores. Fixed and mirror share answers and have been examined many times; neither is a fresh independent test.

Training-only applicability gate: >=2 source recordings; >=2/3 positive source gains; positive equal-source gain; positive gain after removing the best two physical gap variants. Added after the first direct model's aggregate fold, before camera breakdowns or any new assessment results. No camera scope is changed using assessment answers.

## Iterations

1. **Direct nonlinear normalized gap model** (`p61_missing`, CatBoost MAE, 350 depth-5 rounds, fixed half blend): features include gap geometry and V7's observable own/partner candidates, excluding truth, dates and generated V7 predictions. Development MAE 7.244045 -> 7.381702; selected D/H scope. Strict fixed 5.155221 -> 5.171258; mirror 6.065134 -> 6.081896. **Reject.** No post-hoc removal of H or other scope rescue.
2. **Nested cross-fitted residual learner** (`p61_missing_residual`, 300 depth-4 rounds): retains V7 and learns its remaining normalized error. Generated V7 training features and residual targets are inner-recording OOF inside each outer fold. Development MAE 7.244045 -> 7.204913; five of six source recordings improve. Scope frozen A/B/C/F/H. Fixed normalized ratio marginally improves 0.743340 -> 0.743173 but pooled MAE worsens 5.155221 -> 5.161054; mirror worsens to 6.123213 / 0.779732. **Reject.** Development success did not transfer; do not claim the tiny fixed ratio change as supported success.
3. **HIDDEN-FUTURE relative donor-level loadings** (`p61_hf_level`): regularized camera-specific correction to V7's uniform level scaling, preserving timing/template choices and plateau fallback. Four-date source-and-analog exclusion CV; 1,653 per-analog diagnostic rows on five recordings. Count MAE 6.020119 -> 6.023552; only three of five sources improve and Sep 11 regresses. **Reject before testday assessment.** This is an analog-level diagnostic, not a full V7 mixture score.
4. **Duplicate-aware equal-date V7 stack fitting** (`p61_missing_weighted`): same candidate columns/subsets and ridge strength; inverse multiplicity per original source/unit/minute, equal total date weight, fixed half blend. Motivation: overlapping synthetic variants and long recordings should not dominate merely by row count. Development MAE 7.244045 -> 7.250145. No A-H camera passes the gate; only L has a tiny two-recording gain (0.002574 people/row). Fixed/mirror have no L truth. **Reject general replacement; retain L as insufficiently validated exploration.**
5. **Unsupervised input-distribution adaptation** (`p61_missing_adapted`): bounded logistic density ratios, using only gap features, reweight V7's linear stack; no query labels. Development MAE 7.244045 -> 7.232549; scope frozen A/C/D/F/G. Strict fixed MAE 5.155221 -> 5.154931, but gain disappears after best-gap removal; mirror 6.065134 -> 6.076290. **Reject.** No post-hoc camera rescue or blend search.

## Implementation notes

- Initial direct assessment stopped on an absent partner column in the independent vectorized V7 copy before saving any predictions or scoring. Fixed the copy to supply NaN, matching original V7's subset fallback. Model settings and features were unchanged. Both later assessment runs assert agreement of the vectorized baseline with the original V7 predict path on every target.
- Independent poisoning of all unavailable assessment cells leaves release feeds and fitting frames unchanged. No future-context method was added to FORECAST/HIDDEN-FUTURE.
- Early residual recipe metadata inherited the direct-model hash label from the shared runner. Preserve it as history and record corrected source/config fingerprints in final verification. The later `CONFIG_PATH` declaration changes metadata only; numerical replay will verify the final source against saved development predictions.
- The first two assessment outcomes motivate checking data representation; they are now known development evidence for later hypotheses. No claim of an untouched holdout will be made.

Artifacts are under `outputs/prompt61/`; reproduction and final decisions will be recorded in the Arabic report.

## Queue closure

Five distinct hypotheses were implemented and fitted, not just diagnosed. Two nonlinear approaches, an HF level model, a training-weighting change and input-distribution adaptation supplied no defensible overall candidate. Continuing to adjust scopes or settings against these same answers would be assessment tuning rather than stronger evidence. Close the current queue, preserve the reusable tools, and reserve the remaining work for verification and reporting; do not fill the 90-minute maximum with more variations or waiting.

Label-free request comparison confirms differences beyond camera IDs: median A gap line level is 213.92 in development versus 282.04 in actual requests; F is 139.84 versus 81.62; H is 14.34 versus 30.28. Context geometry also differs (H thin-left fraction 22.1% in development, 0% actual; E 41.3% versus 12.7%). These are distribution observations, not a proven cause of failures. Input adaptation did not reliably fix them.
