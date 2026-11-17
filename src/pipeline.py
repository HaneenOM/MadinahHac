"""Offline, date-held-out crowd-count experiments and submission builder."""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('forecast', 'hidden', 'hidden-future', 'missing')
CAMERAS = [f'cam_{c}' for c in 'ABCDEFGH']
FEATURES = ['last', 'age', 'mean5', 'mean15', 'slope5', 'slope15',
            'donor_last', 'donor_mean5', 'donor_slope', 'available',
            'minute', 'horizon']
_TRAINING_CACHE = {}


def locate_data(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not all((p / f).exists() for f in ('train.csv', 'test.csv', 'sample_submission.csv')):
            raise FileNotFoundError(f'CSV triplet missing from {p}')
        return p
    for base in (ROOT / 'data/raw', Path('/kaggle/input')):
        if base.exists():
            for p in [base, *sorted(x for x in base.rglob('*') if x.is_dir())]:
                if all((p / f).exists() for f in ('train.csv', 'test.csv', 'sample_submission.csv')):
                    return p
    raise FileNotFoundError('No data directory found; pass --data-dir')


def load(data: Path):
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    recordings = pd.read_csv(data / 'recordings.csv')
    assert list(sample.columns) == ['id', 'people']
    assert {'recording', 'day', 'unit', 'minute_of_day', 'people'} <= set(train)
    assert {'id', 'kind', 'now_minute_of_day', 'horizon'} <= set(test)
    assert {'recording', 'day', 'split'} <= set(recordings)
    holes = test[test.id.notna()].copy()
    assert holes.id.is_unique and sample.id.is_unique
    assert set(holes.id) == set(sample.id)
    assert holes.people.isna().all()
    assert set(holes.kind) == set(KINDS)
    assert train.people.notna().all() and (train.people >= 0).all()
    assert (test.loc[test.id.isna(), 'people'].dropna() >= 0).all()
    for df in (train, test):
        assert not df.duplicated(['recording', 'unit', 'minute_of_day']).any()
        assert pd.to_numeric(df.minute_of_day, errors='coerce').notna().all()
        assert df.day.notna().all() and df.unit.notna().all()
    assert (holes.loc[holes.kind.isin(['forecast', 'hidden-future']), 'now_minute_of_day']
            < holes.loc[holes.kind.isin(['forecast', 'hidden-future']), 'minute_of_day']).all()
    print(f'Checked {len(train)} train, {len(test)} test, {len(test)-len(holes)} released, {len(holes)} targets')
    print('Target kinds:', holes.kind.value_counts().to_dict())
    print('Train units:', sorted(train.unit.unique()), 'test units:', sorted(test.unit.unique()))
    return train, test, sample, recordings


def table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {rec: g.pivot(index='minute_of_day', columns='unit', values='people').sort_index()
            for rec, g in df[df.people.notna()].groupby('recording')}


def last_pair(series: pd.Series, now: int, window: int = 30):
    x = series.loc[:now].dropna()
    x = x[x.index >= now-window]
    if x.empty:
        return np.nan, np.nan, np.nan
    a = x.iloc[-1]
    age = now - x.index[-1]
    old = x[x.index <= now-5]
    slope = (a-old.iloc[-1]) / (x.index[-1]-old.index[-1]) if len(old) and x.index[-1] > old.index[-1] else 0.
    return float(a), float(age), float(slope)


def feature(t: pd.DataFrame, unit: str, now: int, horizon: int) -> list[float]:
    # Every read is explicitly cut off before any aggregate or interpolation.
    past = t.loc[:now]
    s = past[unit] if unit in past else pd.Series(dtype=float)
    last, age, slope = last_pair(s, now)
    v5 = s.loc[now-4:now].mean()
    v15 = s.loc[now-14:now].mean()
    v_old = s.loc[now-15:now-10].mean()
    slope15 = (v5-v_old)/10 if np.isfinite(v5) and np.isfinite(v_old) else np.nan
    others = past.drop(columns=unit, errors='ignore')
    recent = others.loc[now-4:now]
    donor_last = others.loc[:now].ffill().iloc[-1].mean() if len(others) else np.nan
    donor_mean = recent.mean().mean()
    early = others.loc[now-14:now-10].mean().mean()
    donor_slope = (donor_mean-early)/10 if np.isfinite(donor_mean) and np.isfinite(early) else np.nan
    available = int(recent.notna().any().sum()) if len(recent) else 0
    return [last, age, v5, v15, slope, slope15, donor_last, donor_mean,
            donor_slope, available, now, horizon]


def fit_forecast(tables: dict[str, pd.DataFrame], mode: str):
    key = tuple(sorted(tables))
    if key not in _TRAINING_CACHE:
        rows, labels = [], []
        for t in tables.values():
            for unit in t:
                s = t[unit].dropna()
                if len(s) < 30:
                    continue
                for now in range(int(s.index.min())+20, int(s.index.max())-15, 8):
                    for h in range(1, 16):
                        y = s.get(now+h, np.nan)
                        if np.isfinite(y):
                            f = feature(t, unit, now, h)
                            if np.isfinite(f[0]):
                                rows.append(f)
                                labels.append(float(y))
        _TRAINING_CACHE[key] = (pd.DataFrame(rows, columns=FEATURES), np.asarray(labels))
    X, y = _TRAINING_CACHE[key]
    if mode == 'log':
        target = np.log1p(y)-np.log1p(X['last'].to_numpy())
        weight = np.maximum(X['last'].to_numpy(), 1)
    else:
        target = y-X['last'].to_numpy()
        weight = None
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=120,
            learning_rate=.06, max_leaf_nodes=15, min_samples_leaf=40, random_state=12)
    model.fit(X, target, sample_weight=weight)
    return model, len(y)


def fit_donors(tables: dict[str, pd.DataFrame]):
    models = {}
    full = pd.concat(tables.values(), keys=tables.keys(), names=['recording', 'minute_of_day'])
    for unit in CAMERAS:
        if unit not in full:
            continue
        others = [c for c in CAMERAS if c != unit and c in full]
        valid = full[unit].notna() & full[others].notna().sum(axis=1).ge(2)
        if valid.sum() < 50:
            continue
        X = full.loc[valid, others].copy()
        med = X.median().fillna(0)
        model = Ridge(alpha=100.0).fit(X.fillna(med), full.loc[valid, unit])
        models[unit] = (others, med, model)
    return models


def donor(t: pd.DataFrame, unit: str, minute: int, models, cutoff=None, future=False):
    if unit not in models:
        return np.nan
    others, med, model = models[unit]
    vals = {}
    for c in others:
        if c not in t:
            vals[c] = np.nan
            continue
        if future:
            last, age, slope = last_pair(t[c], cutoff, 30)
            vals[c] = max(0, last + min(minute-cutoff, 15)*np.clip(slope, -2, 2)) if np.isfinite(last) else np.nan
        else:
            vals[c] = t[c].get(minute, np.nan)
    if sum(np.isfinite(v) for v in vals.values()) < 2:
        return np.nan
    x = pd.DataFrame([vals])[others].fillna(med)
    return max(0., float(model.predict(x)[0]))


def predict_one(row, t, fc_model, donor_models, mode, variant):
    unit = row.unit
    minute = int(row.minute_of_day)
    kind = row.kind
    now = int(row.now_minute_of_day) if pd.notna(row.now_minute_of_day) else None
    if kind in ('forecast', 'hidden-future'):
        # Slice all cameras before any feature or donor computation.
        causal = t.loc[:now]
        f = feature(causal, unit, now, minute-now)
        last, age, slope = f[0], f[1], f[4]
        if kind == 'forecast':
            if not np.isfinite(last):
                return np.nan
            if variant == 'persistence':
                return max(0., last)
            if variant == 'trend':
                return max(0., last + np.clip(slope, -2, 2)*(minute-now)*.35)
            x = pd.DataFrame([f], columns=FEATURES)
            residual = float(fc_model.predict(x)[0])
            pred = np.expm1(np.log1p(last)+residual) if mode == 'log' else last+residual
            return max(0., pred)
        fixed = donor(causal, unit, now, donor_models)
        if variant == 'future_donor':
            future = donor(causal, unit, minute, donor_models, cutoff=now, future=True)
            return future if np.isfinite(future) else fixed
        return fixed
    if kind == 'hidden':
        return donor(t, unit, minute, donor_models)
    s = t[unit] if unit in t else pd.Series(dtype=float)
    before = s.loc[:minute-1].dropna()
    after = s.loc[minute+1:].dropna()
    left = before.iloc[-1] if len(before) else np.nan
    right = after.iloc[0] if len(after) else np.nan
    if np.isfinite(left) and np.isfinite(right):
        linear = left+(right-left)*(minute-before.index[-1])/(after.index[0]-before.index[-1])
    elif np.isfinite(left):
        linear = left
    else:
        linear = right
    if variant == 'donor_blend' and np.isfinite(linear):
        d = donor(t, unit, minute, donor_models)
        if np.isfinite(d):
            dl = donor(t, unit, int(before.index[-1]), donor_models) if len(before) else np.nan
            dr = donor(t, unit, int(after.index[0]), donor_models) if len(after) else np.nan
            if np.isfinite(dl) and np.isfinite(dr):
                frac = (minute-before.index[-1])/(after.index[0]-before.index[-1])
                movement = d-(dl+(dr-dl)*frac)
                return max(0., linear + .5*movement)
    return max(0., linear) if np.isfinite(linear) else np.nan


def pseudo(train, test, recordings):
    # The two 615-minute test masks are transplanted to the matching prior weekday.
    mapping = {'20260918_1100': '20260911_1100', '20260919_1100': '20260912_1100'}
    result = []
    for test_rec, val_rec in mapping.items():
        truth = train[train.recording.eq(val_rec)]
        mask = test[test.recording.eq(test_rec) & test.unit.isin(CAMERAS)].copy()
        mask['recording'] = val_rec
        released = mask[mask.id.isna()][['recording', 'unit', 'minute_of_day']]
        feed = released.merge(truth, on=['recording', 'unit', 'minute_of_day'], how='left')
        feed = feed[feed.people.notna()].copy()
        targets = mask[mask.id.notna()].drop(columns='people').merge(
            truth[['recording', 'unit', 'minute_of_day', 'people']],
            on=['recording', 'unit', 'minute_of_day'], how='inner')
        result.append((val_rec, feed, targets))
    return result


def score_rows(rows):
    out = []
    for keys, g in rows.groupby(['fold', 'recording', 'kind'], dropna=False):
        ref = np.abs(g.truth-g.reference).mean()
        for variant, v in g.groupby('variant'):
            mae = np.abs(v.truth-v.prediction).mean()
            out.append(dict(fold=keys[0], recording=keys[1], kind=keys[2], variant=variant,
                            n=len(v), mae=mae, reference_mae=ref,
                            normalized_mae=mae/ref if ref > 0 else np.nan))
    return pd.DataFrame(out)


def validate(train, test, out: Path):
    cases = pseudo(train, test, None)
    rows = []
    for val_rec, feed, targets in cases:
        day = train.loc[train.recording.eq(val_rec), 'day'].iloc[0]
        fit = train[train.day.ne(day)]
        fit_t = table(fit)
        models = {m: fit_forecast(fit_t, m)[0] for m in ('count', 'log')}
        donors = fit_donors(fit_t)
        t = table(feed).get(val_rec, pd.DataFrame())
        for r in targets.itertuples():
            if r.kind == 'forecast':
                variants = ['persistence', 'trend', 'count', 'log']
                reference_variant = 'persistence'
            elif r.kind == 'hidden':
                variants = ['ridge']
                reference_variant = 'ridge'
            elif r.kind == 'hidden-future':
                variants = ['fixed', 'future_donor']
                reference_variant = 'fixed'
            else:
                variants = ['linear', 'donor_blend']
                reference_variant = 'linear'
            preds = {}
            for v in variants:
                mode = 'log' if v == 'log' else 'count'
                preds[v] = predict_one(r, t, models[mode], donors, mode, v)
            ref = preds[reference_variant]
            if not np.isfinite(ref):
                continue
            for v, pred in preds.items():
                if np.isfinite(pred):
                    rows.append(dict(fold=day, recording=val_rec, kind=r.kind, unit=r.unit,
                                     horizon=r.horizon, truth=r.people, reference=ref,
                                     variant=v, prediction=pred))
        print('Validated', val_rec, day, 'targets:', len(targets))
    detail = pd.DataFrame(rows)
    detail.to_csv(out/'validation_predictions.csv', index=False)
    scores = score_rows(detail)
    scores.to_csv(out/'validation_scores.csv', index=False)
    print(scores.to_string(index=False))
    return scores


def select(scores):
    selected = {}
    for kind, g in scores.groupby('kind'):
        summary = g.groupby('variant').apply(lambda x: np.average(x.normalized_mae, weights=x.n),
                                               include_groups=False).sort_values()
        winner = summary.index[0]
        per_fold = g[g.variant.eq(winner)].normalized_mae
        selected[kind] = winner if kind == 'forecast' and per_fold.lt(1).all() else 'reference'
    return selected


def submission(train, test, sample, choices, out):
    fit_t = table(train)
    models = {m: fit_forecast(fit_t, m)[0] for m in ('count', 'log')}
    donors = fit_donors(fit_t)
    feed = test[test.id.isna() & test.people.notna()]
    t = table(feed)
    ref = sample.set_index('id').people.to_dict()
    prediction = {}
    used_reference = defaultdict(int)
    for rec, g in test[test.id.notna()].groupby('recording'):
        rec_t = t[rec]
        for r in g.itertuples():
            v = choices[r.kind]
            if v == 'reference':
                prediction[r.id] = float(ref[r.id])
                used_reference[(r.kind, r.unit)] += 1
                continue
            mode = 'log' if v == 'log' else 'count'
            p = predict_one(r, rec_t, models[mode], donors, mode, v)
            if not np.isfinite(p):
                p = ref[r.id]
                used_reference[(r.kind, r.unit)] += 1
            prediction[r.id] = max(0., float(p))
    saved = pd.DataFrame({'id': sample.id, 'people': [prediction[i] for i in sample.id]})
    saved.to_csv(out/'submission.csv', index=False)
    check = pd.read_csv(out/'submission.csv')
    assert list(check.columns) == ['id', 'people'] and check.id.equals(sample.id)
    assert check.id.is_unique and len(check) == len(sample)
    assert np.isfinite(check.people).all() and (check.people >= 0).all()
    print('Submission validated:', len(check), 'reference fallbacks:', dict(used_reference))
    return used_reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir')
    p.add_argument('--output-dir')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    data = locate_data(args.data_dir)
    out = Path(args.output_dir) if args.output_dir else (Path('/kaggle/working') if Path('/kaggle/working').exists() else ROOT/'outputs')
    out.mkdir(parents=True, exist_ok=True)
    train, test, sample, recordings = load(data)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        scores = validate(train, test, out)
        choices = select(scores)
        print('Selected:', choices)
        (out/'selection.json').write_text(json.dumps(choices, indent=2))
        if not args.validate_only:
            submission(train, test, sample, choices, out)


if __name__ == '__main__':
    main()
