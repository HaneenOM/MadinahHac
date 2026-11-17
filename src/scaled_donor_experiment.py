"""Compare strongest single calibrated donor with the Ridge hidden baseline."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pipeline import load, locate_data, pseudo, table, fit_donors, donor

root = Path(__file__).resolve().parents[1]
train, test, _, _ = load(locate_data(None))
results = []
for rec, feed, targets in pseudo(train, test, None):
    day = train.loc[train.recording.eq(rec), 'day'].iloc[0]
    fit = train[train.day.ne(day)]
    ft = pd.concat(table(fit).values(), ignore_index=True)
    t = table(feed)[rec]
    ridge_models = fit_donors(table(fit))
    best = {}
    for unit in ft:
        candidates = []
        for other in ft:
            if other == unit:
                continue
            pair = ft[[unit, other]].dropna()
            if len(pair) < 50:
                continue
            m = LinearRegression().fit(pair[[other]], pair[unit])
            error = np.abs(m.predict(pair[[other]])-pair[unit]).mean()
            candidates.append((error, other, m))
        if candidates:
            best[unit] = min(candidates, key=lambda x: x[0])
    for r in targets[targets.kind.eq('hidden')].itertuples():
        if r.unit not in best:
            continue
        _, other, m = best[r.unit]
        v = t[other].get(r.minute_of_day, np.nan) if other in t else np.nan
        if np.isfinite(v):
            pred = max(0., float(m.predict(pd.DataFrame({other:[v]}))[0]))
            ridge = donor(t, r.unit, r.minute_of_day, ridge_models)
            if np.isfinite(ridge):
                results.append((day,rec,r.unit,r.people,pred,ridge))
scores = pd.DataFrame(results,columns=['fold','recording','unit','truth','scaled','ridge'])
summary = scores.groupby(['fold','recording']).apply(lambda x: pd.Series({
    'n':len(x), 'scaled_mae':np.abs(x.truth-x.scaled).mean(),
    'ridge_mae':np.abs(x.truth-x.ridge).mean()}), include_groups=False).reset_index()
summary.to_csv(root/'outputs/scaled_donor_scores.csv',index=False)
print(summary.to_string(index=False))
