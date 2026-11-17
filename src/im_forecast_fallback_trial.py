"""Fixed rolling-origin I-M forecast fallback trial; inputs and labels separate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json

import numpy as np
import pandas as pd

from pipeline import ROOT, load, locate_data
from t05_forecast.core import load_recordings
from t05_forecast.reference import last_value, ref_forecast
from hidden_candidate import read_submission

OUT = ROOT / 'outputs/im_forecast_fallback_trial'
UNITS = [f'cam_{c}' for c in 'IJKLM']
# Selected from input coverage before scoring: all 555 real requests qualify.
# Matches the existing reference's five-minute freshness bound for all 3 values.
MAX_THREE_AGE = 5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context(series, now):
    past = series.loc[:now].dropna()
    row = dict(available=len(past))
    for k in range(1, 4):
        row[f'time{k}'] = int(past.index[-k]) if len(past) >= k else np.nan
        row[f'value{k}'] = float(past.iloc[-k]) if len(past) >= k else np.nan
    return row


def coverage(rows, source):
    result = rows[['day', 'unit', 'horizon', 'now', 'available', 'time1', 'time3']].copy()
    result['source'] = source
    result['reading_bucket'] = result.available.map(lambda n: '0' if n == 0 else str(n) if n < 3 else '3+')
    result['newest_age'] = result['now'] - result.time1
    result['oldest3_age'] = result['now'] - result.time3
    result['fallback_available'] = result.newest_age.le(5)
    for limit in (2, 5, 10, 15, 30):
        result[f'three_within_{limit}'] = result.oldest3_age.le(limit)
    result['horizon_band'] = pd.cut(result.horizon, [0, 5, 10, 15], labels=['1-5', '6-10', '11-15']).astype(str)
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    _, test, _, _ = load(locate_data(None))
    released = test[test.id.isna() & test.people.notna() & test.unit.isin(UNITS)]
    observed = {(rec, unit): g.set_index('minute_of_day').people.sort_index()
                for (rec, unit), g in released.groupby(['recording', 'unit'])}
    real = []
    for r in test[test.id.notna() & test.kind.eq('forecast') & test.unit.isin(UNITS)].itertuples():
        now = int(r.now_minute_of_day)
        real.append(dict(case_id=r.id, day=r.day, recording=r.recording, unit=r.unit,
                         minute=int(r.minute_of_day), now=now, horizon=int(r.minute_of_day)-now,
                         **context(observed[(r.recording, r.unit)], now)))
    # Deterministic exhaustive released-target/horizon pairs, no selection on errors.
    cases, labels = [], []
    for (rec, unit), s in observed.items():
        day = test.loc[test.recording.eq(rec), 'day'].iloc[0]
        first = int(test.loc[test.recording.eq(rec), 'minute_of_day'].min())
        for minute, value in s.items():
            for horizon in range(1, 16):
                now = int(minute)-horizon
                if now < first:
                    continue
                case_id = f'{rec}:{unit}:{int(minute)}:{horizon}'
                cases.append(dict(case_id=case_id, day=day, recording=rec, unit=unit,
                                  minute=int(minute), now=now, horizon=horizon,
                                  **context(s, now)))
                labels.append(dict(case_id=case_id, truth=float(value)))
    real = pd.DataFrame(real)
    cases = pd.DataFrame(cases)
    labels = pd.DataFrame(labels)
    assert real.case_id.is_unique and cases.case_id.is_unique
    assert len(real) == 555
    paths = ['cases.csv', 'labels.csv', 'real_inputs.csv']
    for name, df in zip(paths, (cases, labels, real)):
        path = OUT / name
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), df, check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
        else:
            df.to_csv(path, index=False)
    counts = pd.concat([coverage(real, 'real'), coverage(cases, 'validation')], ignore_index=True)
    counts.to_csv(OUT / 'coverage.csv', index=False)
    summary = counts.groupby(['source', 'day', 'unit']).agg(
        requests=('available', 'size'), zero=('available', lambda s: int(s.eq(0).sum())),
        one=('available', lambda s: int(s.eq(1).sum())),
        two=('available', lambda s: int(s.eq(2).sum())),
        three_plus=('available', lambda s: int(s.ge(3).sum())),
        fresh_fallback=('fallback_available', 'sum'),
        three_within_2=('three_within_2', 'sum'), three_within_5=('three_within_5', 'sum'),
        three_within_10=('three_within_10', 'sum'), three_within_15=('three_within_15', 'sum')).reset_index()
    summary.to_csv(OUT / 'coverage_by_date_camera.csv', index=False)
    (OUT / 'case_hashes.json').write_text(json.dumps({n: digest(OUT / n) for n in paths}, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print('Unique validation target readings:', cases[['recording', 'unit', 'minute']].drop_duplicates().shape[0])


def predict(row):
    """Call T05's actual last-value routine, then apply one fixed alternative."""
    recent = np.full((6, 1), np.nan)
    for k in range(1, 4):
        minute = row[f'time{k}']; value = row[f'value{k}']
        if np.isfinite(minute) and 0 <= row['now']-minute <= 5:
            recent[int(minute-row['now']+5), 0] = value
    base = float(last_value(recent, 5)[-1, 0])
    eligible = bool(row['available'] >= 3 and np.isfinite(base)
                    and row['now']-row['time3'] <= MAX_THREE_AGE)
    new = .5*(base+float(np.median([row[f'value{k}'] for k in range(1, 4)]))) if eligible else base
    return base, new, eligible


def run():
    before = {p.name: digest(p) for p in (ROOT / 'outputs').glob('submission*.csv')}
    hashes = json.loads((OUT / 'case_hashes.json').read_text(encoding='utf-8'))
    assert all(digest(OUT / name) == value for name, value in hashes.items())
    cases = pd.read_csv(OUT / 'cases.csv')
    real = pd.read_csv(OUT / 'real_inputs.csv')
    # Explicitly verify the five-minute boundary and missing-context behavior.
    for age in (0, 5, 6, None):
        row = dict(now=100, available=int(age is not None), time1=100-age if age is not None else np.nan,
                   value1=7., time2=np.nan, value2=np.nan, time3=np.nan, value3=np.nan)
        base, new, eligible = predict(row)
        assert not eligible
        assert (base == new == 7.) if age in (0, 5) else (np.isnan(base) and np.isnan(new))
    recs = load_recordings(ROOT / 'data/raw')
    refs = {name: ref_forecast(rec) for name, rec in recs.items() if rec.split == 'test'}
    real_rows, predictions = [], []
    checks = 0
    for source, inputs in [('real', real), ('validation', cases)]:
        for row in inputs.to_dict('records'):
            base, new, eligible = predict(row)
            assert row['minute'] > row['now']
            for k in range(1, 4):
                assert not np.isfinite(row[f'time{k}']) or row[f'time{k}'] <= row['now']
            # Perturb every later reading, including the held-out target. Rebuild
            # only the permitted context and check the actual prediction path.
            rec = recs[row['recording']]
            col = rec.col(row['unit'])
            times = rec.minutes
            y = rec.y[:, col]
            altered = y.copy(); altered[times > row['now']] = 1e6
            causal = pd.Series(altered, index=times)
            check_row = dict(row, **context(causal, int(row['now'])))
            check = predict(check_row)
            np.testing.assert_allclose([base, new], check[:2], rtol=0, atol=1e-10, equal_nan=True)
            assert eligible == check[2]
            checks += 1
            result = dict(row, baseline=base, candidate=new, eligible=eligible,
                          changed=bool(np.isfinite(new) and abs(new-base) > 1e-9))
            if source == 'real':
                ref = refs[row['recording']][int(row['minute'])-rec.first, col]
                np.testing.assert_allclose(base, ref, rtol=0, atol=1e-10, equal_nan=True)
                real_rows.append(result)
            else:
                predictions.append(result)
    predictions = pd.DataFrame(predictions)
    real_predictions = pd.DataFrame(real_rows)
    predictions.to_csv(OUT / 'predictions.csv', index=False)
    real_predictions.to_csv(OUT / 'real_predictions.csv', index=False)
    base_file = ROOT / 'outputs/submission_t05_hidden_hidden_future_v1.csv'
    saved = pd.read_csv(base_file)
    compare = real_predictions.merge(saved, left_on='case_id', right_on='id', validate='one_to_one')
    assert len(compare) == 555
    np.testing.assert_allclose(compare.baseline, compare.people, rtol=0, atol=1e-9)
    # Only the scorer reads masked target counts, after predictions are on disk.
    labels = pd.read_csv(OUT / 'labels.csv')
    scored = predictions.merge(labels, on='case_id', validate='one_to_one')
    assert len(scored) == len(cases)
    scored['horizon_band'] = pd.cut(scored.horizon, [0, 5, 10, 15], labels=['1-5', '6-10', '11-15']).astype(str)
    scored['baseline_error'] = (scored.truth-scored.baseline).abs()
    scored['candidate_error'] = (scored.truth-scored.candidate).abs()
    scored.to_csv(OUT / 'scored.csv', index=False)
    usable = scored[scored.baseline.notna()]
    summaries = []
    real_context = usable[usable.eligible & (usable['now']-usable.time1).le(1)]
    for scope, data in [('all_scoreable', usable), ('changed', usable[usable.changed]),
                        ('real_context', real_context)]:
        for keys, group in [(('all', 'all', 'all'), data),
                            *[((day, 'all', 'all'), g) for day, g in data.groupby('day')],
                            *[((day, unit, 'all'), g) for (day, unit), g in data.groupby(['day', 'unit'])],
                            *[(('all', 'all', band), g) for band, g in data.groupby('horizon_band')],
                            *[((day, unit, band), g) for (day, unit, band), g in data.groupby(['day', 'unit', 'horizon_band'])]]:
            old = group.baseline_error.mean(); new = group.candidate_error.mean()
            summaries.append(dict(scope=scope, day=keys[0], unit=keys[1], horizon_band=keys[2],
                                  cases=len(group), unique_targets=len(group[['recording', 'unit', 'minute']].drop_duplicates()),
                                  changed=int(group.changed.sum()), baseline_mae=old, candidate_mae=new,
                                  ratio=new/old if old else np.nan))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'scores.csv', index=False)
    print(summary[(summary.horizon_band.eq('all') | summary.day.eq('all'))
                  & ~summary.scope.eq('real_context')].to_string(index=False))
    print('Context matched to real requests (newest <=1 minute; all three <=5):')
    print(summary[summary.scope.eq('real_context') & summary.horizon_band.eq('all')].to_string(index=False))
    coverage_rows = []
    for name, frame in [('real', real_predictions), ('validation_all', scored),
                        ('validation_scoreable', usable), ('validation_real_context', real_context)]:
        coverage_rows.append(dict(source=name, cases=len(frame),
                                  fallback_available=int(frame.baseline.notna().sum()),
                                  eligible=int(frame.eligible.sum()), changed=int(frame.changed.sum()),
                                  max_newest_age=float((frame['now']-frame.time1).max()),
                                  median_newest_age=float((frame['now']-frame.time1).median())))
    pd.DataFrame(coverage_rows).to_csv(OUT / 'coverage_comparison.csv', index=False)
    comparison = pd.concat([real_predictions.assign(source='real'),
                            usable.assign(source='validation_scoreable'),
                            real_context.assign(source='validation_real_context')], ignore_index=True)
    comparison['horizon_band'] = pd.cut(comparison.horizon, [0, 5, 10, 15], labels=['1-5', '6-10', '11-15']).astype(str)
    comparison['target_hour'] = (comparison.minute // 60).astype(int)
    comparison['newest_age'] = comparison['now']-comparison.time1
    comparison['oldest3_age'] = comparison['now']-comparison.time3
    for groups, name in [(['source', 'day', 'unit', 'horizon_band'], 'coverage_by_horizon_band.csv'),
                         (['source', 'day', 'unit', 'target_hour'], 'coverage_by_hour.csv'),
                         (['source', 'newest_age', 'oldest3_age'], 'coverage_by_context_age.csv')]:
        comparison.groupby(groups).agg(cases=('case_id', 'size'), eligible=('eligible', 'sum'),
                                       changed=('changed', 'sum')).reset_index().to_csv(OUT / name, index=False)
    assert before == {p.name: digest(p) for p in (ROOT / 'outputs').glob('submission*.csv')}
    report = dict(recency_rule='Three latest readings all <=5 minutes old at now',
                  rule_selected_from_input_coverage_before_errors=True,
                  baseline_verified_real_requests=555, post_cutoff_checks=checks,
                  total_pseudo_cases=len(cases), scoreable_cases=len(usable),
                  unavailable_baseline_cases=int(scored.baseline.isna().sum()),
                  coverage=coverage_rows, saved_submissions_unchanged=True,
                  limitation='Overlapping rolling-origin cases, same two recording dates; no independent target labels')
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    print('Post-cutoff checks:', checks)


def build_candidate(*, data_dir=None, baseline_path=None, output_path=None, report_dir=None):
    """Fresh I-M predictions, preserving every other base value verbatim."""
    from pathlib import Path
    _, test, sample, _ = load(Path(data_dir) if data_dir else locate_data(None))
    protected = {p.name: digest(p) for p in (ROOT / 'outputs').glob('submission*.csv')}
    base_path = Path(baseline_path) if baseline_path else ROOT / 'outputs/submission_t05_hidden_hidden_future_v1.csv'
    base = read_submission(base_path, sample.id.tolist())
    base_values = {r['id']: float(r['people']) for r in base}
    result = [r.copy() for r in base]
    output_by_id = {r['id']: r for r in result}
    released = test[test.id.isna() & test.people.notna() & test.unit.isin(UNITS)]
    series = {(rec, unit): g.set_index('minute_of_day').people.sort_index()
              for (rec, unit), g in released.groupby(['recording', 'unit'])}
    target = test[test.id.notna()]
    changed_rows = []
    for r in target[target.kind.eq('forecast') & target.unit.isin(UNITS)].itertuples():
        now = int(r.now_minute_of_day)
        inputs = dict(now=now, **context(series[(r.recording, r.unit)], now))
        baseline, candidate, eligible = predict(inputs)
        assert np.isfinite(baseline) and abs(baseline-base_values[r.id]) <= 1e-9
        assert np.isfinite(candidate) and candidate >= 0
        if abs(candidate-baseline) > 1e-9:
            output_by_id[r.id]['people'] = str(float(candidate))
            changed_rows.append(dict(id=r.id, day=r.day, unit=r.unit,
                                     absolute_difference=abs(candidate-base_values[r.id])))
    path = Path(output_path) if output_path else ROOT / 'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
    if path.exists():
        assert read_submission(path, sample.id.tolist()) == result, 'Existing candidate differs; preserve it'
    else:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'people'])
            writer.writeheader(); writer.writerows(result)
    saved = read_submission(path, sample.id.tolist())
    metadata = target.set_index('id')
    changes = [r for r, old in zip(saved, base) if r['people'] != old['people']]
    assert len(changes) == len(changed_rows)
    assert all(metadata.loc[r['id'], 'kind'] == 'forecast' and
               metadata.loc[r['id'], 'unit'] in UNITS for r in changes)
    assert all(digest(ROOT / 'outputs' / name) == value for name, value in protected.items())
    changed = pd.DataFrame(changed_rows)
    reports = Path(report_dir) if report_dir else OUT
    reports.mkdir(parents=True, exist_ok=True)
    changed.to_csv(reports / 'candidate_changes.csv', index=False)
    counts = changed.groupby(['day', 'unit']).size().reset_index(name='changed')
    report = dict(file=str(path), ids=len(saved), changed=len(changes),
                  changed_task='forecast', changed_units=sorted(changed.unit.unique()),
                  by_date_camera=counts.to_dict('records'),
                  all_preexisting_submissions_preserved=True)
    (reports / 'candidate_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Candidate: {path}; {len(saved)} IDs; only {len(changes)} I-M forecast values changed')
    print(counts.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--build-candidate', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare:
        return
    run()
    if args.build_candidate:
        build_candidate()


if __name__ == '__main__':
    main()
