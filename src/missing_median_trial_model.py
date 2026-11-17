"""Rejected experiment: three-minute median of v2's donor correction."""
from __future__ import annotations

import numpy as np

from v2_candidate import MISSING_WEIGHT
import v2_missing


def baseline(t, unit, minute, calibration):
    reference, _, gap = v2_missing.predict(t, unit, minute, calibration, 'linear')
    prediction = reference
    weight = MISSING_WEIGHT.get(unit, 0.)
    donor_count = 0
    if weight:
        method = 'blend_full' if weight == 1. else 'blend'
        fitted, donor_count, gap = v2_missing.predict(t, unit, minute, calibration, method, weight)
        if np.isfinite(fitted) and gap >= 6 and donor_count > 0:
            prediction = fitted
    return reference, prediction, donor_count


def predict(t, unit, minute, calibration):
    # Returning an observed endpoint verbatim is part of the interpolation contract.
    observed = t[unit].get(minute, np.nan) if unit in t else np.nan
    if np.isfinite(observed):
        return float(observed)
    ctx = v2_missing.gap_context(t, unit, minute)
    if ctx is None:
        return np.nan
    start, end, _, reference = ctx
    corrections = []
    for neighbor in (minute-1, minute, minute+1):
        if neighbor in (start, end):
            corrections.append(0.)
        else:
            linear, current, _ = baseline(t, unit, neighbor, calibration)
            corrections.append(current-linear)
    correction = float(np.median(corrections))
    limit = .5*max(reference, 5)*MISSING_WEIGHT.get(unit, 0.)
    correction = float(np.clip(correction, -limit, limit))
    return max(0., float(reference+correction))
