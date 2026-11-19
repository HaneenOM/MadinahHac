"""Settings of the missing track and how each was chosen.

Every choice below was made on pseudo-gaps only, scored out of day (select_spec.py, cv_variants.py, cv_im.py):
test windows laid over the training recordings at every 3rd shift (shifts within 25 minutes of the real placement
excluded) and pseudo-gaps inside the released test windows. No fixed-set or mirror-set case entered a choice.

Per camera: value = line + sum of weight_k * (candidate_k - line), weights fitted on the pseudo-gaps.
  own      own-level smoother column (gaplib): own_<lam> random walk, spl_<lam> smoothing spline
  ratios   partner ratio fills r<partner>[shift]_<lam>; the model of the largest usable subset is applied
  extra    'v2' adds the accepted value as a candidate (the blend with the accepted method)
  loss     l1 = absolute error (iteratively reweighted), l2 = squared error
  pool     fit the weights on these cameras' rows instead (camera M has too few rows of its own)

Why these:
  A  E is its strong partner (ratio fill alone 0.41 of the line error on 11 and 12 Sep); reading E one minute earlier
     and later as well lowers the pooled score 0.538 -> 0.530; an own-level term or the accepted value adds nothing.
  E  same with A; a smoothing spline of E's own readings helps slightly when A is out (0.514 -> 0.508 -> 0.500).
  F  A and E as partners plus its own level (0.882 accepted -> 0.789).
  G  partner B with a long ratio memory (0.890 -> 0.779).
  B  partner G helps only together with the accepted value (0.902 -> 0.852).
  C  no partner; own-level smoother blended with the accepted value (0.994 -> 0.960).
  D, H  no partner and the accepted value is the line; own-level smoother (1.000 -> 0.978 and 0.925).
  I, J, K  test-only cameras, fitted on the released test rows; own level + the strongest same-minute partner
     (leave one test recording out: 0.895, 0.915, 0.839).
  L  its own level does not help (1.01 to 1.05); partner I only (0.987).
  M  260 pseudo-gap rows on one recording: weights of the own-level smoother borrowed from I, J, K, L (0.839 on those rows).
Smoothing strengths: lam = 4 for own levels and close partners was within 0.003 of the best value for every camera,
so one value is used; B/G use a longer ratio memory (16 and 64) because their ratio is noisier per minute.
"""
from __future__ import annotations

SPEC = {
    "cam_A": dict(own=None, ratios=("rE_4", "rE+1_4", "rE-1_4", "rF_4"), loss="l1"),
    "cam_E": dict(own="spl_30", ratios=("rA_4", "rA+1_4", "rA-1_4", "rF_4"), loss="l1"),
    "cam_F": dict(own="own_4", ratios=("rA_4", "rE_4"), loss="l1"),
    "cam_G": dict(own="own_4", ratios=("rB_16",), loss="l1"),
    "cam_B": dict(own="own_2", ratios=("rG_64",), extra=("v2",), loss="l1"),
    "cam_C": dict(own="own_4", extra=("v2",), loss="l1"),
    "cam_D": dict(own="own_4", loss="l1"),
    "cam_H": dict(own="own_4", loss="l1"),
    "cam_I": dict(own="own_4", ratios=("rL_4",), loss="l2"),
    "cam_J": dict(own="own_4", ratios=("rK_4",), loss="l2"),
    "cam_K": dict(own="own_4", ratios=("rJ_4",), loss="l2"),
    "cam_L": dict(own=None, ratios=("rI_16",), loss="l2"),
    "cam_M": dict(own="own_4", loss="l2", relative=True, pool=("cam_I", "cam_J", "cam_K", "cam_L")),
}
# candidate grids that the settings above need (gaplib.gap_candidates)
GRIDS = dict(lam_own=(2.0, 4.0), lam_spl=(30.0,), lam_ratio=(4.0, 16.0, 64.0), shifts=(-1, 1), lam_shift=4.0)
MAX_REL = 0.5          # the value stays within 50% of max(line, 5), the same bound as both existing methods
# Blend with the accepted value (cv_blend.py): final = accepted + SHRINK * (stack - accepted).
# On the pseudo-gaps 0.75 scores the same as 1.0 for A, B, C, D, F (within 0.003), better for E (0.489 against 0.498),
# and it is better than the accepted value on every held-out day for every camera, which 1.0 is not (D on 18 Sep).
# G, H and I to M improve steadily up to 1.0 (the accepted value is the plain line for H and I to M), so they stay at 1.0.
SHRINK = {"cam_A": 0.75, "cam_B": 0.75, "cam_C": 0.75, "cam_D": 0.75, "cam_E": 0.75, "cam_F": 0.75}
RIDGE = 1e-3
