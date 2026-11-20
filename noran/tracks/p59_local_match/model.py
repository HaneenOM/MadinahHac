"""P59: context-only local segment ranking, separate from accepted V7.

Two distinct recording/prayer events are required; overlapping origins in one
recording do not count as independent neighbors. No fitting or parameter search.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / 'hidden_future/model.py'
sys.path.insert(0, str(BASE.parent))
spec = importlib.util.spec_from_file_location('p59_v7_analog', BASE)
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
C = v7.C
SPEC = dict(history=20, offsets=[-15,-10,-5,0,5,10,15], min_pairs=10,
            min_cameras=3, neighbors=2, level_shrink=0.8, max_shape_mae=0.35)


def library(frame, schedule):
    ref = C.reference_prayers(schedule)
    days = frame.groupby('recording').day.first().to_dict()
    tables = v7.full_tables(frame)
    analogs = {rec:C.Analog(t, str(days[rec]), schedule, ref, smooth=9,
                            strength=v7.PARAMS['STRENGTH']) for rec,t in tables.items()}
    return dict(analogs=analogs, tables=tables, ref_pr=ref, schedule=schedule)


def phase(minute, day, schedule):
    return int(np.searchsorted(schedule[str(day)], minute, side='right')-1)


def select(case, feed, model):
    """Selection reads only visible context; no target history/continuation."""
    now = int(case.now)
    cams = [c for c in C.CAMS if c not in C.HIDDEN_WITH[case.unit]]
    grid = np.arange(now-19, now+1)
    query = feed.loc[:now].reindex(index=grid, columns=cams).to_numpy(float)
    day = str(case.day)
    qphase = phase(now, day, model['schedule'])
    aligned = float(C.align([now], day, model['schedule'], model['ref_pr'], v7.PARAMS['STRENGTH'])[0])
    candidates = []
    for rec, an in model['analogs'].items():
        if an.weekday != pd.Timestamp(day).day_of_week or an.day == day:
            continue
        # Invert the prayer-aligned clock to constrain the candidate endpoint.
        native = np.arange(int(model['schedule'][an.day][0])-180, 1440)
        canon = C.align(native, an.day, model['schedule'], model['ref_pr'], v7.PARAMS['STRENGTH'])
        center = int(round(np.interp(aligned, canon, native)))
        for offset in SPEC['offsets']:
            origin = center + offset
            if phase(origin, an.day, model['schedule']) != qphase:
                continue
            hgrid = np.arange(origin-19, origin+1)
            # Read native minute rows directly. A round trip via an aligned
            # interpolated grid could borrow a donor reading after origin.
            historical_context = model['tables'][rec].reindex(index=hgrid, columns=cams).to_numpy(float)
            costs, levels, pairs = [], [], []
            for j, cam in enumerate(cams):
                historic = historical_context[:,j]
                ok = np.isfinite(query[:,j]) & np.isfinite(historic)
                if ok.sum() < SPEC['min_pairs']:
                    continue
                diff = np.log1p(query[ok,j]) - np.log1p(historic[ok])
                level = float(np.median(diff))
                costs.append(float(np.mean(np.abs(diff-level))))
                levels.append(level)
                pairs.append(int(ok.sum()))
            if len(costs) < SPEC['min_cameras']:
                continue
            distance = float(np.mean(costs))
            candidates.append(dict(recording=rec, day=an.day, event=f'{rec}|prayer_phase_{qphase}',
                 origin=origin, offset=offset, distance=distance, log_level=float(np.median(levels)),
                 cameras=len(costs), min_pairs=min(pairs), mean_pairs=float(np.mean(pairs))))
    ranked = sorted(candidates, key=lambda x:(x['distance'], x['recording'], x['origin']))
    selected, events = [], set()
    for item in ranked:
        if item['distance'] > SPEC['max_shape_mae'] or item['event'] in events:
            continue
        selected.append(item)
        events.add(item['event'])
        if len(selected) == SPEC['neighbors']:
            break
    return selected, candidates


def continuation(case, selected, model):
    """Read target continuation only AFTER neighbors have been selected."""
    if len(selected) < SPEC['neighbors']:
        return np.nan
    horizon = int(case.minute)-int(case.now)
    values = []
    for item in selected:
        an = model['analogs'][item['recording']]
        if case.unit not in an.smooth:
            return np.nan
        tau = C.align([item['origin']+horizon], item['day'], model['schedule'], model['ref_pr'], v7.PARAMS['STRENGTH'])
        value = float(an.at(an.smooth[case.unit], tau)[0])
        if not np.isfinite(value):
            return np.nan
        values.append(max(0., float(np.expm1(np.log1p(max(value,0.))+SPEC['level_shrink']*item['log_level']))))
    return float(np.mean(values))
