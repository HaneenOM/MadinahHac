"""Leakage invariance and sparse unseen-camera proxy checks."""
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline import (ROOT, load, locate_data, table, pseudo, fit_forecast,
                      fit_donors, predict_one)


class ConstantModel:
    def predict(self, x):
        return np.zeros(len(x))


def main():
    train, test, _, _ = load(locate_data(None))
    cases = pseudo(train, test, None)
    donor_models = fit_donors(table(train))
    leak_count = 0
    for _, feed, targets in cases:
        t = next(iter(table(feed).values()))
        for r in targets[targets.kind.isin(['forecast', 'hidden-future'])].iloc[::30].itertuples():
            now = int(r.now_minute_of_day)
            modified = t.copy()
            modified.loc[modified.index > now] = 100000.
            for variant in (['persistence', 'trend', 'count'] if r.kind == 'forecast'
                            else ['fixed', 'future_donor']):
                a = predict_one(r, t, ConstantModel(), donor_models, 'count', variant)
                b = predict_one(r, modified, ConstantModel(), donor_models, 'count', variant)
                assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b), (r.id, variant, a, b)
                leak_count += 1
    print('Leakage invariance checks passed:', leak_count)

    # The pooled forecast model never receives a unit ID. Withhold H from fitting
    # and expose only every fifth released H observation on the held-out day.
    val_rec, feed, targets = cases[1]
    day = train.loc[train.recording.eq(val_rec), 'day'].iloc[0]
    fit = train[train.day.ne(day) & train.unit.ne('cam_H')]
    model, n = fit_forecast(table(fit), 'count')
    sparse = feed[feed.unit.ne('cam_H') | feed.minute_of_day.mod(5).eq(0)]
    t = next(iter(table(sparse).values()))
    vals = []
    for r in targets[(targets.kind == 'forecast') & (targets.unit == 'cam_H')].itertuples():
        for variant in ('persistence', 'trend', 'count'):
            pred = predict_one(r, t, model, {}, 'count', variant)
            if np.isfinite(pred):
                vals.append((variant, abs(pred-r.people)))
    results = pd.DataFrame(vals, columns=['variant', 'error']).groupby('variant').error.agg(['size', 'mean'])
    results.to_csv(ROOT/'outputs/cold_start_proxy.csv')
    print('Cold-start proxy; H withheld from fit, sparse H context:', results.to_string())


if __name__ == '__main__':
    main()
