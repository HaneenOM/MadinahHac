"""Focused, leakage-aware P58 comparison against V7 HIDDEN-FUTURE.

Run from repository root with the existing Python 3.12 lab environment.
All experiment artifacts are separate from the accepted baseline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'src', ROOT / 'noran'):
    sys.path.insert(0, str(path))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import baseline_tasks as tasks
from pipeline import load, locate_data, table, CAMERAS
from prayer_experiments import timetable
from noran.tracks.p58_reconstructed_history import model as method

TEST_MASK = {'20260918_1100': ['cam_B', 'cam_G'], '20260919_1100': ['cam_D']}
PSEUDO_MASK = {'20260918_1100': ['cam_D'], '20260919_1100': ['cam_B', 'cam_G']}


def build(which, train, test):
    cases, labels, feeds = [], [], {}
    if which in ('fixed', 'mirror'):
        vals = ['20260911_1100', '20260912_1100']
        if which == 'mirror':
            vals.reverse()
        mapping = dict(zip(TEST_MASK, vals))
        masks = {val: TEST_MASK[rec] for rec, val in mapping.items()}
        for rec, val in mapping.items():
            truth = train[train.recording.eq(val)]
            day = str(truth.day.iloc[0])
            mask = test[test.recording.eq(rec) & test.unit.isin(CAMERAS)].copy()
            feed = mask[mask.id.isna()][['unit', 'minute_of_day']].merge(
                truth[['unit', 'minute_of_day', 'people']], on=['unit', 'minute_of_day'])
            feeds[val] = feed.pivot(index='minute_of_day', columns='unit', values='people').sort_index()
            target = mask[mask.kind.eq('hidden-future') & mask.id.notna()]
            for r in target.itertuples():
                value = truth[truth.unit.eq(r.unit) & truth.minute_of_day.eq(r.minute_of_day)].people
                if value.empty:
                    continue
                cid = f'{val}_{r.unit}_{int(r.minute_of_day)}'
                cases.append(dict(case_id=cid, recording=val, day=day, unit=r.unit,
                                  minute=int(r.minute_of_day), now=int(r.now_minute_of_day)))
                labels.append(dict(case_id=cid, truth=float(value.iloc[0])))
    else:
        masks = PSEUDO_MASK
        released = test[test.id.isna() & test.people.notna()]
        for rec, hidden in masks.items():
            g = released[released.recording.eq(rec)]
            day = str(g.day.iloc[0])
            feeds[rec] = g[~g.unit.isin(hidden)].pivot(index='minute_of_day', columns='unit', values='people').sort_index()
            minutes = sorted(g[g.unit.isin(CAMERAS)].minute_of_day.unique())
            stretches = []
            first = prev = minutes[0]
            for minute in minutes[1:]:
                if minute != prev + 1:
                    stretches.append((first, prev))
                    first = minute
                prev = minute
            stretches.append((first, prev))
            for unit in hidden:
                own = g[g.unit.eq(unit)].set_index('minute_of_day').people
                for first, last in stretches:
                    for offset in (7, 13):
                        now = int(first + offset)
                        for minute in range(now + 1, min(now + 15, int(last)) + 1):
                            if minute not in own.index:
                                continue
                            cid = f'{rec}_{unit}_{minute}_n{now}'
                            cases.append(dict(case_id=cid, recording=rec, day=day, unit=unit, minute=minute, now=now))
                            labels.append(dict(case_id=cid, truth=float(own[minute])))
    return pd.DataFrame(cases), pd.DataFrame(labels), feeds, masks


def score(out, detail, labels, runtime, checks):
    d = detail.merge(labels, on='case_id', validate='one_to_one')
    for col in ('baseline', 'prediction', 'reference'):
        d['err_' + col] = (d[col] - d.truth).abs()
    d['band'] = pd.cut(d.horizon, [0, 5, 10, 15], labels=['1-5', '6-10', '11-15'])
    d['window'] = d.recording + '|' + d.unit + '|' + d.now.astype(str)
    d.to_csv(out / 'scored.csv', index=False)
    for name, keys in [('recording_camera', ['recording', 'unit']), ('horizon', ['recording', 'unit', 'band']),
                       ('windows', ['recording', 'unit', 'now'])]:
        g = d.groupby(keys, observed=True).agg(n=('truth','size'), baseline_mae=('err_baseline','mean'),
                new_mae=('err_prediction','mean'), reference_mae=('err_reference','mean'),
                corrected=('corrected','sum'), windows=('window','nunique')).reset_index()
        g['baseline_norm'] = g.baseline_mae / g.reference_mae
        g['new_norm'] = g.new_mae / g.reference_mae
        g.to_csv(out / (name + '.csv'), index=False)
    days = d.groupby('day').agg(base=('err_baseline','mean'), new=('err_prediction','mean'), ref=('err_reference','mean'))
    windows = d.groupby('window').agg(base=('err_baseline','sum'), new=('err_prediction','sum'), n=('truth','size'))
    windows['gain'] = windows.base - windows.new
    sensitivity = {}
    for count in (0,1,2):
        keep = windows.sort_values('gain', ascending=False).iloc[count:]
        sensitivity[str(count)] = float((keep.base.sum()-keep.new.sum())/keep.n.sum())
    result = dict(status='complete', rows=len(d), windows=int(d.window.nunique()),
        baseline_mae=float(d.err_baseline.mean()), new_mae=float(d.err_prediction.mean()),
        baseline_norm=float((days.base/days.ref).mean()), new_norm=float((days.new/days.ref).mean()),
        corrected_rows=int(d.corrected.sum()), fallback_rows=int((~d.corrected).sum()),
        coverage=dict(reconstructed_points=d.reconstructed_points.describe().to_dict(),
                      observed_donor_minutes=d.observed_minutes.describe().to_dict(),
                      donors_now=d.donors_now.describe().to_dict()),
        gain_after_removing_best_windows=sensitivity, invariance_checks=checks,
        runtime_seconds=runtime, protocol='date holdout plus globally removed paired assessment camera labels; same fit for both methods',
        independent_validation=False, candidate_created=False)
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', choices=['fixed','mirror','testday'], required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/p58_verified')
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.output_dir.resolve() / args.set
    if (out / 'predictions.csv').exists():
        raise FileExistsError(f'Preserve existing predictions; choose a new --output-dir: {out}')
    out.mkdir(parents=True, exist_ok=True)
    train, test, _, _ = load(locate_data(None))
    schedule = timetable(ROOT / 'models/current_baseline/prayer_times_madinah.csv', train, test)
    cases, labels, feeds, masks = build(args.set, train, test)
    assert cases.case_id.is_unique
    cases.to_csv(out / 'cases.csv', index=False)
    labels.to_csv(out / 'labels.csv', index=False)
    # Remove assessment target-camera labels jointly across BOTH recordings.
    fitting = train.copy()
    released = test[test.id.isna() & test.people.notna()].copy()
    for rec, hidden in masks.items():
        fitting = fitting[~(fitting.recording.eq(rec) & fitting.unit.isin(hidden))]
        released = released[~(released.recording.eq(rec) & released.unit.isin(hidden))]
        assert not (fitting.recording.eq(rec) & fitting.unit.isin(hidden)).any()
        assert not (released.recording.eq(rec) & released.unit.isin(hidden)).any()
    context = dict(schedule=schedule, test_released=released, config=tasks.default_config()['hidden-future'], mission='hidden-future')
    rows, checks = [], 0
    os.environ.pop('P58_CASE_SET', None)  # global masking is done explicitly above
    with threadpool_limits(limits=2):
        for day, group in cases.groupby('day'):
            training = fitting[fitting.day.ne(day)].copy()
            assert not training.day.eq(day).any()
            model = method.fit(training, day, context)
            print(f'{args.set} {day}: fitted; {len(group)} targets', flush=True)
            checked = set()
            for r in group.itertuples():
                t = feeds[r.recording].drop(columns=masks[r.recording], errors='ignore')
                causal = t.loc[:int(r.now)]
                ref, fallback, _ = tasks.hidden_future_value(causal, r.unit, r.now, r.minute-r.now, r.day,
                                                  schedule, model['hf']['accepted'], context['config'])
                base = method.hf.predict(r, t, model['hf'], context)
                base = float(base) if np.isfinite(base) else fallback
                pred = method.predict(r, t, model, context)
                pred = float(pred) if np.isfinite(pred) else fallback
                window = (r.recording, r.unit, r.now)
                if window not in checked:
                    altered = t.copy()
                    altered.loc[altered.index > r.now] = 1e8
                    for unit in masks[r.recording]:
                        altered[unit] = 1e8
                    model['history_cache'].clear()
                    again = method.predict(r, altered, model, context)
                    again = float(again) if np.isfinite(again) else fallback
                    assert np.isclose(pred, again, rtol=0, atol=1e-8), (r.case_id, pred, again)
                    checks += 1
                    checked.add(window)
                hist = [value for key, value in model['history_cache'].items()
                        if key[:4] == (r.recording, r.unit, str(r.day), r.now)]
                npoints = int(np.isfinite(hist[-1][1]).sum()) if hist else 0
                recent = causal.reindex(range(r.now-14, r.now+1)).reindex(columns=CAMERAS)
                rows.append(dict(case_id=r.case_id, recording=r.recording, day=r.day, unit=r.unit,
                    minute=r.minute, now=r.now, horizon=r.minute-r.now, reference=ref,
                    baseline=base, prediction=pred, corrected=abs(pred-base)>1e-12,
                    reconstructed_points=npoints, observed_minutes=int(recent.notna().any(axis=1).sum()),
                    donors_now=int(recent.iloc[-1].notna().sum())))
    detail = pd.DataFrame(rows)
    assert len(detail)==len(cases) and np.isfinite(detail[['baseline','prediction','reference']]).all().all()
    assert (detail.prediction>=0).all()
    detail.to_csv(out / 'predictions.csv', index=False)
    # Assessment labels enter scoring only after predictions are persisted.
    score(out, detail, pd.read_csv(out / 'labels.csv'), round(time.perf_counter()-started, 2), checks)


if __name__ == '__main__':
    main()
