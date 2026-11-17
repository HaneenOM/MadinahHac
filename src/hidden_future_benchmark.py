"""Reproducible v2 hidden-future baseline on the two long held-out dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import ROOT, load, locate_data, table
from v2_candidate import HF_WEIGHT
from v2_experiments import cases, fit_direct, predict_hf
import v2_hidden


DATES = ('2026-09-11', '2026-09-12')
OUT = ROOT / 'outputs' / 'hidden_future_benchmark'


def build_cases(train: pd.DataFrame, test: pd.DataFrame):
    manifests = []
    feeds = []
    folds = []
    for day, rec, feed, targets in cases(train, test):
        if day not in DATES:
            continue
        selected = targets[targets.kind.eq('hidden-future')].copy()
        assert len(selected) and not feed.empty
        expected_units = {'cam_B', 'cam_G'} if day == DATES[0] else {'cam_D'}
        assert set(selected.unit) == expected_units
        selected['day'] = day
        selected['now'] = selected.now_minute_of_day.astype(int)
        selected['minute'] = selected.minute_of_day.astype(int)
        selected['horizon'] = selected['minute'] - selected['now']
        assert selected.horizon.between(1, 15).all()
        assert selected.people.notna().all()
        selected['case_id'] = (rec + '_' + selected.unit + '_' + selected.minute.astype(str))
        assert selected.case_id.is_unique
        manifests.append(selected[['case_id', 'id', 'day', 'recording', 'unit',
                                   'now', 'minute', 'horizon', 'people']].rename(
                                       columns={'id': 'template_target_id', 'people': 'truth'}))
        available = feed.reset_index().melt(id_vars='minute_of_day',
                                             var_name='unit', value_name='people').dropna()
        available.insert(0, 'recording', rec)
        available.insert(0, 'day', day)
        feeds.append(available)
        folds.append((day, rec, feed, selected))
    manifest = pd.concat(manifests, ignore_index=True).sort_values(
        ['day', 'recording', 'unit', 'minute']).reset_index(drop=True)
    available = pd.concat(feeds, ignore_index=True).sort_values(
        ['day', 'recording', 'unit', 'minute_of_day']).reset_index(drop=True)
    assert manifest.case_id.is_unique
    assert manifest.groupby('day').size().to_dict() == {DATES[0]: 322, DATES[1]: 160}
    assert set(available.day) == set(DATES)
    assert not available.duplicated(['recording', 'unit', 'minute_of_day']).any()
    return manifest, available, folds


def main():
    train, test, _, _ = load(locate_data(None))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, available, folds = build_cases(train, test)
    manifest.to_csv(OUT / 'cases.csv', index=False)
    available.to_csv(OUT / 'released_feed.csv', index=False)

    rows = []
    checks = 0
    for day, rec, feed, targets in folds:
        fit = train[train.day.ne(day)]
        assert day not in set(fit.day)
        fit_tables = table(fit)
        hidden = v2_hidden.fit(fit_tables)
        direct = fit_direct(fit_tables)
        assert direct is not None
        for r in targets.itertuples():
            now = int(r.now); minute = int(r.minute); horizon = int(r.horizon)
            reference = predict_hf(feed, r.unit, now, horizon,
                                   hidden, direct, 'fixed')
            predicted_direct = predict_hf(feed, r.unit, now, horizon,
                                          hidden, direct, 'direct')
            assert np.isfinite(reference) and np.isfinite(predicted_direct)
            prediction = reference + HF_WEIGHT * (predicted_direct - reference)
            # Every camera observation after now is changed for both paths.
            altered = feed.copy()
            altered.loc[altered.index > now] = 1e6
            for method, original in [('fixed', reference), ('direct', predicted_direct)]:
                changed = predict_hf(altered, r.unit, now, horizon,
                                     hidden, direct, method)
                assert np.isclose(original, changed, rtol=0, atol=1e-10)
                checks += 1
            rows.append({'case_id': r.case_id, 'day': day, 'recording': rec,
                         'unit': r.unit, 'now': now, 'minute': minute,
                         'horizon': horizon, 'truth': float(r.people),
                         'reference': reference, 'direct': predicted_direct,
                         'prediction': prediction})
        print(f'Validated {day}: {len(targets)} targets', flush=True)

    detail = pd.DataFrame(rows).sort_values(['day', 'recording', 'unit', 'minute'])
    assert len(detail) == len(manifest)
    assert detail.case_id.tolist() == manifest.case_id.tolist()
    # This is a fresh fit and prediction. The archived values serve only as
    # a regression check that the v2 baseline has been reproduced.
    archived = pd.read_csv(ROOT / 'outputs' / 'v2_selected_validation_predictions.csv')
    archived = archived[(archived.kind == 'hidden-future') &
                        archived.fold.isin(DATES)].copy()
    archived['case_id'] = (archived.recording + '_' + archived.unit + '_' +
                           archived.minute.astype(int).astype(str))
    comparison = detail.merge(archived[['case_id', 'truth', 'reference', 'direct',
                                        'candidate']], on='case_id', how='inner',
                              validate='one_to_one', suffixes=('', '_archived'))
    assert len(comparison) == len(detail)
    for current, old in [('truth', 'truth_archived'),
                         ('reference', 'reference_archived'),
                         ('direct', 'direct_archived'),
                         ('prediction', 'candidate')]:
        assert np.allclose(comparison[current], comparison[old], rtol=0, atol=1e-8), \
            f'v2 mismatch: {current}'

    detail['absolute_error'] = (detail.truth - detail.prediction).abs()
    detail['reference_absolute_error'] = (detail.truth - detail.reference).abs()
    detail.to_csv(OUT / 'predictions.csv', index=False)
    scores = detail.groupby('day').agg(
        predictions=('absolute_error', 'size'),
        mae_people=('absolute_error', 'mean'),
        reference_mae_people=('reference_absolute_error', 'mean')).reset_index()
    scores['normalized_error'] = (scores.mae_people /
                                  scores.reference_mae_people.replace(0, np.nan))
    scores.to_csv(OUT / 'scores_by_date.csv', index=False)
    print(scores.to_string(index=False))
    print(f'Post-cutoff invariance checks: {checks}; archived v2 values matched: {len(detail)}')
    print(f'Saved cases, released feed, predictions, scores: {OUT}')


if __name__ == '__main__':
    main()
