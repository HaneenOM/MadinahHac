"""Diagnostic only: T05, its weekly template, and recent persistence on A-H."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from pipeline import ROOT
from t05_forecast import gp
from t05_forecast.core import ABSENT, FORECAST, RELEASED, load_recordings
from t05_forecast.pipeline import FRI, SAT, P11, P12, make_template, sources_for, fit_cached, load_params
from t05_forecast.submit import member_settings, settings, fit_kwargs
from t05_forecast.template import build_template, complete_template, slope
from t05_forecast.reference import last_value

OUT = ROOT / 'outputs/weekly_forecast_diagnostic'
FIT_CACHE = OUT / 'fit_cache'
UNITS = [f'cam_{c}' for c in 'ABCDEFGH']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(recs):
    """Fixed 30-minute origins, every available target 1-15 minutes later."""
    OUT.mkdir(exist_ok=True)
    rows, labels = [], []
    for name in (FRI, SAT):
        rec = recs[name]
        for now in range(rec.first+15, int(rec.minutes[-1]), 30):
            for unit in UNITS:
                c = rec.col(unit)
                for h in range(1, 16):
                    minute = now+h; r = minute-rec.first
                    if r >= rec.n_minutes or rec.status[r, c] != RELEASED or not np.isfinite(rec.y[r, c]):
                        continue
                    case_id = f'{name}:{unit}:{now}:{h}'
                    rows.append(dict(case_id=case_id, day=rec.day, recording=name,
                                     unit=unit, now=now, minute=minute, horizon=h,
                                     window_id=f'{name}:{now}', camera_window=f'{name}:{unit}:{now}'))
                    labels.append(dict(case_id=case_id, truth=float(rec.y[r, c])))
    cases = pd.DataFrame(rows)
    labels = pd.DataFrame(labels)
    assert cases.case_id.is_unique and not cases.duplicated(['recording', 'unit', 'minute']).any()
    for name, df in [('cases.csv', cases), ('labels.csv', labels)]:
        path = OUT / name
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), df, check_dtype=False)
        else:
            df.to_csv(path, index=False)
    (OUT / 'case_hashes.json').write_text(json.dumps(
        {n: digest(OUT / n) for n in ('cases.csv', 'labels.csv')}, indent=2), encoding='utf-8')
    print('Fixed pseudo cases:', cases.groupby('day').agg(cases=('case_id', 'size'),
          windows=('window_id', 'nunique')).to_dict('index'), flush=True)
    return cases


def fit_template(recs, name, lam, excluded):
    """T05 template construction with no source from the held-out recording."""
    rec = recs[name]
    sources = [r for r in sources_for(recs, name) if r.name != excluded]
    template = build_template(rec, sources, lam=lam)
    historical = [r for r in recs.values() if r.split == 'train' and r.n_minutes >= 600
                  and r.name not in (name, excluded)]
    template, _ = complete_template(rec, template, historical, lam=lam)
    # Filling I-M from the excluded test recording would indirectly leak its
    # target counts into covariance fitting. No eligible I-M source remains.
    return template


def audit(recs, member):
    rows = []
    for heldout in (FRI, SAT):
        other = SAT if heldout == FRI else FRI
        for name in (other, P11, P12):
            original = make_template(recs, name, member['lam'])
            clean = fit_template(recs, name, member['lam'], heldout)
            original_field = gp.observed(recs[name], original, slope(original), clip=1.5)
            clean_field = gp.observed(recs[name], clean, slope(clean), clip=1.5)
            rows.append(dict(heldout=heldout, fit_recording=name,
                             original_cells=len(original_field['t']), clean_cells=len(clean_field['t']),
                             removed_cells=len(original_field['t'])-len(clean_field['t'])))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'fit_exclusion_audit.csv', index=False)
    print(result.to_string(index=False), flush=True)


def finite_mean(values):
    a = np.stack(values)
    count = np.isfinite(a).sum(axis=0)
    return np.divide(np.nansum(a, axis=0), count, out=np.full(count.shape, np.nan), where=count > 0)


def fitted_params(recs, heldout, member, fields):
    other = SAT if heldout == FRI else FRI
    names = (other, P11, P12)
    assert heldout not in names
    # A diagnostic-specific cache never overwrites the submitted model's cache.
    fingerprint = hashlib.sha256()
    datasets = []
    for name in names:
        key = (heldout, name, member['lam'])
        if key not in fields:
            tmpl = fit_template(recs, name, member['lam'], heldout)
            fields[key] = gp.observed(recs[name], tmpl, slope(tmpl), clip=1.5)
        ds = fields[key]
        for k in ('t', 'c', 'd', 'r'):
            fingerprint.update(np.ascontiguousarray(ds[k]).tobytes())
        datasets.append(ds)
    key = f'fully_excluded_{heldout}_{fingerprint.hexdigest()[:16]}_{member["key"]}'
    print(f'Fitting/loading {heldout}: {member["key"]}; cells {[len(d["t"]) for d in datasets]}', flush=True)
    kwargs = fit_kwargs(member)
    kwargs.update(threads=4, verbose=True)
    return fit_cached(FIT_CACHE, key, datasets, **kwargs)


def masked_at_now(source, targets, now):
    rec = source.copy()
    rec.truth = None; rec.ids = {}
    keep = (rec.status == RELEASED) & (rec.minutes[:, None] <= now)
    rec.y[~keep] = np.nan
    rec.status[:] = ABSENT; rec.status[keep] = RELEASED
    rec.now[:] = -1
    for r in targets.itertuples():
        row, col = int(r.minute)-rec.first, rec.col(r.unit)
        assert not np.isfinite(rec.y[row, col])
        rec.status[row, col] = FORECAST; rec.now[row, col] = now
    assert not np.isfinite(rec.y[rec.minutes > now]).any()
    return rec


def level_and_trend(rec, unit, now, template):
    col = rec.col(unit)
    valid = (rec.status[:, col] == RELEASED) & np.isfinite(rec.y[:, col]) & (rec.minutes <= now)
    rows = np.flatnonzero(valid)
    latest = np.nan; age = np.nan; level_gap = np.nan
    if len(rows):
        last = rows[-1]; age = int(now-rec.minutes[last])
        if age <= 5:
            latest = float(rec.y[last, col])
            if np.isfinite(template[last, col]):
                level_gap = float(np.log1p(latest)-np.log1p(template[last, col]))
    if not np.isfinite(level_gap):
        agreement = 'unavailable'
    elif abs(level_gap) <= np.log(1.25):
        agreement = 'within_25pct'
    elif abs(level_gap) <= np.log(2):
        agreement = '25pct_to_2x'
    else:
        agreement = 'over_2x'
    obs_slope = np.nan; tmpl_slope = np.nan; trend = 'insufficient'
    recent = rows[rec.minutes[rows] >= now-10]
    recent = recent[np.isfinite(template[recent, col])]
    if len(recent) >= 3 and np.ptp(rec.minutes[recent]) >= 5:
        x = rec.minutes[recent].astype(float); x -= x.mean()
        obs_slope = float(np.dot(x, np.log1p(rec.y[recent, col]))/np.dot(x, x))
        tmpl_slope = float(np.dot(x, np.log1p(template[recent, col]))/np.dot(x, x))
        if min(abs(obs_slope), abs(tmpl_slope)) < .005:
            trend = 'flat_or_weak'
        else:
            trend = 'same' if obs_slope*tmpl_slope > 0 else 'opposite'
    return dict(latest=latest, latest_age=age, log_level_gap=level_gap, level_group=agreement,
                observed_log_slope=obs_slope, template_log_slope=tmpl_slope, trend_group=trend)


def prediction_rows(recs, cases, members, frozen_only=False):
    fields = {}; rows = []; check_count = 0
    for name in (FRI, SAT):
        source = recs[name]
        today = cases[cases.recording.eq(name)]
        if today.empty:
            continue
        member_predictions = []; frozen_predictions = []; templates = []
        for i, member in enumerate(members, 1):
            print(f'{name} member {i}/{len(members)}', flush=True)
            tmpl = make_template(recs, name, member['lam'])
            dslope = slope(tmpl)
            # Template must be wholly independent of this recording's counts.
            altered_source = source.copy(); altered_source.y[:] = 1e6
            altered_recs = dict(recs); altered_recs[name] = altered_source
            np.testing.assert_allclose(tmpl, make_template(altered_recs, name, member['lam']),
                                       rtol=0, atol=1e-12, equal_nan=True)
            prefix = 'without_fri' if name == FRI else 'without_sat'
            frozen = load_params(ROOT / 'outputs/t05_cache' / f'params_{prefix}_{member["key"]}.npz')
            params = frozen if frozen_only else fitted_params(recs, name, member, fields)
            member_values = {}; frozen_values = {}
            for now, targets in today.groupby('now'):
                now = int(now)
                masked = masked_at_now(source, targets, now)
                model = gp.predict_recording(masked, tmpl, dslope, params, strict=True)
                legacy = model if frozen_only else gp.predict_recording(masked, tmpl, dslope, frozen, strict=True)
                # This stronger structural cutoff removes all later values before
                # the GP path. Perturbing them cannot change the masked recording.
                changed = source.copy(); changed.y[changed.minutes > now] = 1e6
                changed_mask = masked_at_now(changed, targets, now)
                np.testing.assert_allclose(masked.y, changed_mask.y, rtol=0, atol=0, equal_nan=True)
                assert np.array_equal(masked.status, changed_mask.status)
                check_count += len(targets)
                for r in targets.itertuples():
                    row, col = int(r.minute)-source.first, source.col(r.unit)
                    member_values[r.case_id] = model[row, col]
                    frozen_values[r.case_id] = legacy[row, col]
            member_predictions.append(np.array([member_values[c] for c in today.case_id]))
            frozen_predictions.append(np.array([frozen_values[c] for c in today.case_id]))
            templates.append(np.maximum(0., np.expm1(tmpl)))
            print(f'{name} member {i} predicted {len(today)} cases', flush=True)
        clean = finite_mean(member_predictions); frozen = finite_mean(frozen_predictions)
        mean_template = finite_mean(templates)
        latest_by_minute = last_value(source.y, 5)
        for idx, r in enumerate(today.itertuples()):
            minute = int(r.minute); now = int(r.now)
            row, col = minute-source.first, source.col(r.unit)
            diagnostics = level_and_trend(source, r.unit, now, mean_template)
            recent = diagnostics['latest']
            np.testing.assert_allclose(recent, latest_by_minute[now-source.first, col],
                                       rtol=0, atol=1e-12, equal_nan=True)
            value = clean[idx] if np.isfinite(clean[idx]) else recent
            legacy = frozen[idx] if np.isfinite(frozen[idx]) else recent
            rows.append(dict(**r._asdict(), t05=value, frozen_t05=legacy,
                             t05_model=bool(np.isfinite(clean[idx])),
                             template=float(mean_template[row, col]), **diagnostics))
        print(f'{name}: predictions complete', flush=True)
    detail = pd.DataFrame(rows).drop(columns=['Index'])
    detail.to_csv(OUT / 'predictions.csv', index=False)
    return detail, check_count


def summarize(scored, prefix):
    scored = scored.copy()
    scored['observed_direction'] = np.where(scored.observed_log_slope.gt(.005), 'rising',
        np.where(scored.observed_log_slope.lt(-.005), 'falling', 'weak_or_unavailable'))
    scored['template_direction'] = np.where(scored.template_log_slope.gt(.005), 'rising',
        np.where(scored.template_log_slope.lt(-.005), 'falling', 'weak_or_unavailable'))
    scored['level_side'] = np.where(scored.log_level_gap.gt(np.log(1.25)), 'above',
        np.where(scored.log_level_gap.lt(-np.log(1.25)), 'below', 'close_or_unavailable'))
    scored['horizon_band'] = pd.cut(scored.horizon, [0, 5, 10, 15], labels=['1-5', '6-10', '11-15']).astype(str)
    scored['common'] = scored[['t05', 'template', 'latest', 'frozen_t05']].notna().all(axis=1)
    common = scored[scored.common].copy()
    common['target_timestamp'] = common.day+':'+common.minute.astype(str)
    for method in ('t05', 'frozen_t05', 'template', 'latest'):
        common[f'{method}_error'] = (common[method]-common.truth).abs()
    matched = common[common.trend_group.ne('insufficient')]
    for groups, name in [(['day'], 'matched_input_by_date'),
                         (['day', 'horizon_band'], 'matched_input_by_horizon')]:
        result = matched.groupby(groups).agg(targets=('case_id', 'size'),
            target_timestamps=('target_timestamp', 'nunique'), origins=('window_id', 'nunique'),
            t05_mae=('t05_error', 'mean'), frozen_t05_mae=('frozen_t05_error', 'mean'),
            template_mae=('template_error', 'mean'), latest_mae=('latest_error', 'mean')).reset_index()
        result['few_examples'] = result.targets.lt(30) | result.target_timestamps.lt(30) | result.origins.lt(5)
        result.to_csv(OUT / f'{prefix}_{name}.csv', index=False)
    summaries = []
    for groups, name in [(['day'], 'by_date'), (['day', 'horizon_band'], 'by_horizon'),
                         (['day', 'unit'], 'by_camera'),
                         (['day', 'unit', 'horizon_band'], 'by_camera_horizon'),
                         (['day', 'level_group'], 'by_level'),
                         (['day', 'trend_group'], 'by_trend'),
                         (['day', 'observed_direction', 'template_direction'], 'by_direction'),
                         (['day', 'horizon_band', 'level_side', 'observed_direction', 'trend_group'], 'by_signed_level_trend'),
                         (['day', 'horizon_band', 'level_group', 'trend_group'], 'by_level_trend_horizon')]:
        result = common.groupby(groups).agg(
            targets=('case_id', 'size'), target_timestamps=('target_timestamp', 'nunique'),
            camera_windows=('camera_window', 'nunique'), origins=('window_id', 'nunique'),
            t05_mae=('t05_error', 'mean'), frozen_t05_mae=('frozen_t05_error', 'mean'),
            template_mae=('template_error', 'mean'), latest_mae=('latest_error', 'mean')).reset_index()
        result['few_examples'] = (result.targets.lt(30) | result.target_timestamps.lt(30)
                                  | result.camera_windows.lt(5) | result.origins.lt(5))
        result.to_csv(OUT / f'{prefix}_{name}.csv', index=False)
        if name in ('by_date', 'by_horizon', 'by_level', 'by_trend'):
            print(f'{prefix} {name}:'); print(result.to_string(index=False), flush=True)
    all_rows = []
    for day, g in scored.groupby('day'):
        available = g[g.t05.notna()]
        all_rows.append(dict(day=day, all_cases=len(g), predicted=len(available),
                             model=int(available.t05_model.sum()), fallback=int((~available.t05_model).sum()),
                             unavailable=int(g.t05.isna().sum()),
                             t05_mae=(available.t05-available.truth).abs().mean(),
                             frozen_t05_mae=(available.frozen_t05-available.truth).abs().mean(),
                             common_cases=int(g.common.sum())))
    pd.DataFrame(all_rows).to_csv(OUT / f'{prefix}_all_t05.csv', index=False)
    print(f'{prefix} T05 all:', pd.DataFrame(all_rows).to_string(index=False), flush=True)
    scored.to_csv(OUT / f'{prefix}_scored.csv', index=False)
    sensitivity = []
    for grouping in ('all', 'level_group', 'trend_group'):
        groupers = ['day'] if grouping == 'all' else ['day', grouping]
        for key, group in common.groupby(groupers):
            key = key if isinstance(key, tuple) else (key,)
            for alternative in ('latest', 'template'):
                for model in ('t05', 'frozen_t05'):
                    diff = group[f'{model}_error']-group[f'{alternative}_error']
                    total = diff.sum()
                    windows = group.assign(gain=diff).groupby('window_id').gain.agg(['sum', 'size'])
                    leave_one = (total-windows['sum'])/(len(group)-windows['size'])
                    sensitivity.append(dict(day=key[0], grouping=grouping,
                                            group='all' if len(key) == 1 else key[1],
                                            model=model, alternative=alternative, targets=len(group),
                                            origins=len(windows), alternative_gain_mae=diff.mean(),
                                            alternative_wins_origins=int(windows['sum'].gt(0).sum()),
                                            min_leave_one_origin_gain=leave_one.min(),
                                            max_leave_one_origin_gain=leave_one.max()))
    pd.DataFrame(sensitivity).to_csv(OUT / f'{prefix}_window_sensitivity.csv', index=False)
    camera_origins = common.groupby(['day', 'unit', 'window_id', 'horizon_band']).agg(
        targets=('case_id', 'size'), t05_mae=('t05_error', 'mean'),
        frozen_t05_mae=('frozen_t05_error', 'mean'), template_mae=('template_error', 'mean'),
        latest_mae=('latest_error', 'mean')).reset_index()
    camera_origins.to_csv(OUT / f'{prefix}_camera_origin_scores.csv', index=False)
    trend_cases = common[common.trend_group.isin(['same', 'opposite'])].copy()
    trend_cases['signed_lag'] = (trend_cases.truth-trend_cases.t05)*np.sign(trend_cases.observed_log_slope)
    trend_cases['frozen_signed_lag'] = (trend_cases.truth-trend_cases.frozen_t05)*np.sign(trend_cases.observed_log_slope)
    trend_cases['lags_observed_trend'] = trend_cases.signed_lag.gt(0)
    trend_cases['frozen_lags_observed_trend'] = trend_cases.frozen_signed_lag.gt(0)
    trend_cases.to_csv(OUT / f'{prefix}_trend_lag_cases.csv', index=False)
    for grouping, groups in [('camera', ['day', 'unit', 'horizon_band', 'trend_group']),
                             ('origin', ['day', 'window_id', 'horizon_band', 'trend_group'])]:
        lag = trend_cases.groupby(groups).agg(targets=('case_id', 'size'),
                origins=('window_id', 'nunique'), signed_lag=('signed_lag', 'mean'),
                frozen_signed_lag=('frozen_signed_lag', 'mean'),
                lag_fraction=('lags_observed_trend', 'mean'),
                frozen_lag_fraction=('frozen_lags_observed_trend', 'mean')).reset_index()
        lag['few_examples'] = lag.targets.lt(30) | lag.origins.lt(5)
        lag.to_csv(OUT / f'{prefix}_trend_lag_by_{grouping}.csv', index=False)
    return common


def benchmark_check(recs, members):
    # Reproduce the existing Sept 11-12 benchmark path without rewriting it.
    from local_score import load_cases
    from t05_local_forecast import masked_recording, predict as benchmark_predict
    cases, feeds = load_cases()
    cases = cases[cases.kind.eq('forecast')].copy()
    values, checks = benchmark_predict(cases, feeds)
    templates = {}
    for name in (P11, P12):
        rec = recs[name]
        fitting = [r for r in recs.values() if r.split == 'train' and r.name != name]
        prior = [r for r in fitting if r.weekday == rec.weekday]
        others = [r for r in fitting if r.weekday != rec.weekday]
        curves = []
        for member in members:
            tmpl = build_template(rec, prior, lam=member['lam'])
            tmpl, _ = complete_template(rec, tmpl, others, lam=member['lam'])
            curves.append(np.maximum(0., np.expm1(tmpl)))
        templates[name] = finite_mean(curves)
    current = pd.read_csv(ROOT / 'outputs/local_score/predictions.csv').set_index('case_id')
    masked_bench = {name: masked_recording(recs[name], feeds[name], cases[cases.recording.eq(name)])
                    for name in (P11, P12)}
    rows = []
    for r in cases.itertuples():
        rec = recs[r.recording]
        masked = masked_bench[r.recording]
        tmpl = templates[r.recording]
        info = level_and_trend(masked, r.unit, int(r.now), tmpl)
        # Existing benchmark uses a 30-minute persistence fallback. Preserve it
        # here and compare it with the requested 5-minute latest on common cases.
        value = values[r.case_id]
        predicted = value if np.isfinite(value) else float(current.loc[r.case_id, 'reference'])
        np.testing.assert_allclose(predicted, current.loc[r.case_id, 'prediction'], rtol=0, atol=1e-9)
        rows.append(dict(case_id=r.case_id, day=r.day, recording=r.recording, unit=r.unit,
                         now=int(r.now), minute=int(r.minute), horizon=int(r.horizon),
                         window_id=f'{r.recording}:{int(r.now)}',
                         camera_window=f'{r.recording}:{r.unit}:{int(r.now)}',
                         t05=predicted, frozen_t05=predicted, t05_model=bool(np.isfinite(value)),
                         template=float(tmpl[int(r.minute)-rec.first, rec.col(r.unit)]), **info))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'benchmark_predictions.csv', index=False)
    labels = pd.read_csv(ROOT / 'outputs/local_score/labels.csv')
    scored = detail.merge(labels, on='case_id', validate='one_to_one')
    summarize(scored, 'benchmark')
    return checks


def save_examples(common):
    # One example per recording/origin prevents a shared crowd event from
    # being presented as several independent failures across cameras.
    examples = common.sort_values('frozen_t05_error', ascending=False).drop_duplicates('window_id').head(5).copy()
    examples['now_clock'] = examples['now'].map(lambda x: f'{int(x)//60:02d}:{int(x)%60:02d}')
    examples['target_clock'] = examples.minute.map(lambda x: f'{int(x)//60:02d}:{int(x)%60:02d}')
    examples = examples.rename(columns={'frozen_t05': 'current_t05', 't05': 'clean_fit_t05',
                                        'template': 'weekly_template'})
    examples['matches_actual_context'] = examples.trend_group.ne('insufficient') & examples.latest.notna()
    examples[['day', 'unit', 'now_clock', 'target_clock', 'horizon', 'truth', 'current_t05',
              'clean_fit_t05', 'weekly_template', 'latest', 'level_group', 'trend_group',
              'matches_actual_context', 'window_id']].to_csv(OUT / 'five_examples.csv', index=False)


def save_input_coverage(recs, members):
    rows = []
    cases = pd.read_csv(OUT / 'cases.csv')
    for name in (FRI, SAT):
        rec = recs[name]
        tmpl = finite_mean([np.maximum(0., np.expm1(make_template(recs, name, m['lam']))) for m in members])
        targets = cases[cases.recording.eq(name)].to_dict('records')
        real = []
        for row, col in zip(*np.nonzero(rec.status == FORECAST)):
            if rec.units[col] in UNITS:
                now = int(rec.now[row, col])
                real.append(dict(case_id=rec.ids[(int(row), int(col))], day=rec.day,
                                 unit=rec.units[col], minute=int(rec.minutes[row]), now=now,
                                 horizon=int(rec.minutes[row])-now))
        for source, inputs in [('pseudo', targets), ('real_requests', real)]:
            for r in inputs:
                info = level_and_trend(rec, r['unit'], int(r['now']), tmpl)
                rows.append(dict(source=source, case_id=r['case_id'], day=rec.day,
                                 unit=r['unit'], now=r['now'], horizon=r['horizon'],
                                 latest_available=bool(np.isfinite(info['latest'])),
                                 trend_context=info['trend_group'] != 'insufficient',
                                 level_group=info['level_group'], trend_group=info['trend_group']))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'input_coverage.csv', index=False)
    frame.groupby(['source', 'day']).agg(cases=('case_id', 'size'),
        latest_available=('latest_available', 'sum'), trend_context=('trend_context', 'sum')).reset_index().to_csv(
            OUT / 'input_coverage_summary.csv', index=False)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--frozen-only', action='store_true', help='Audit frozen current T05 separately; not a clean held-out fit')
    parser.add_argument('--summarize', action='store_true', help='Rebuild reports from saved diagnostic predictions')
    args = parser.parse_args()
    if args.frozen_only:
        OUT = OUT / 'frozen_audit'
        OUT.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        common = summarize(pd.read_csv(OUT / 'released_scored.csv'), 'released')
        save_examples(common)
        summarize(pd.read_csv(OUT / 'benchmark_scored.csv'), 'benchmark')
        recs = load_recordings(ROOT / 'data/raw')
        cfg = json.loads((ROOT / 't05_forecast_config.json').read_text(encoding='utf-8'))
        save_input_coverage(recs, member_settings(settings(cfg)))
        verification = OUT / 'verification.json'
        if verification.exists():
            report = json.loads(verification.read_text(encoding='utf-8'))
            report['sparse_flag'] = 'Fewer than 30 targets or target timestamps, or fewer than 5 camera-windows/origins'
            report['example_selection'] = 'Five original-current-T05 errors from distinct recording/origin windows'
            verification.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return
    protected = {p.name: digest(p) for p in (ROOT / 'outputs').glob('submission*.csv')}
    recs = load_recordings(ROOT / 'data/raw')
    cfg = json.loads((ROOT / 't05_forecast_config.json').read_text(encoding='utf-8'))
    members = member_settings(settings(cfg))
    cases = prepare(recs)
    audit(recs, members[0])
    if args.prepare:
        return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        detail, checks = prediction_rows(recs, cases, members, frozen_only=args.frozen_only)
        # Labels become available only after every primary prediction is saved.
        labels = pd.read_csv(OUT / 'labels.csv')
        scored = detail.merge(labels, on='case_id', validate='one_to_one')
        assert len(scored) == len(cases)
        common = summarize(scored, 'released')
        save_examples(common)
        benchmark_checks = benchmark_check(recs, members)
        save_input_coverage(recs, members)
    assert protected == {p.name: digest(p) for p in (ROOT / 'outputs').glob('submission*.csv')}
    report = dict(primary_cases=len(cases), primary_member_cutoff_checks=checks,
                  benchmark_cutoff_checks=benchmark_checks, saved_submissions_unchanged=True,
                  fitting=('Frozen original T05 audit only; indirect fit dependency remains' if args.frozen_only else
                           'Clean T05 refit excludes heldout recording directly and indirectly; frozen current model also reported'),
                  agreement_thresholds='abs log1p ratio <=log(1.25), <=log(2), >log(2)',
                  trend_rule='Last 10 min, >=3 observed points spanning >=5 min, log slope threshold 0.005/min',
                  sparse_flag='Fewer than 30 targets or target timestamps, or fewer than 5 camera-windows/origins',
                  limitations='Only two days; released target availability differs from true forecast holes; short historical template coverage in Sept11-12 check')
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
