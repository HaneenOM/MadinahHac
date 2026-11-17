"""Direct horizon-aware model for a fully masked target camera."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from pipeline import ROOT, load, locate_data, pseudo, table, feature, FEATURES

train, test, _, _ = load(locate_data(None))
baseline = pd.read_csv(ROOT/'outputs/validation_predictions.csv')
all_results = []
for rec, feed, targets in pseudo(train, test, None):
    day = train.loc[train.recording.eq(rec), 'day'].iloc[0]
    fit = train[train.day.ne(day)]
    rows, labels = [], []
    for t in table(fit).values():
        for unit in ('cam_B', 'cam_D', 'cam_G'):
            if unit not in t:
                continue
            target = t[unit]
            masked = t.drop(columns=unit)
            for now in range(int(t.index.min())+20, int(t.index.max())-15, 8):
                base = feature(masked, unit, now, 0)
                for h in range(1,16):
                    y = target.get(now+h, np.nan)
                    if np.isfinite(y):
                        f = base.copy()
                        f[-1] = h
                        rows.append(f+[ord(unit[-1])-65])
                        labels.append(y)
    columns = FEATURES+['camera']
    model = HistGradientBoostingRegressor(loss='absolute_error',max_iter=120,
            learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=40,random_state=19)
    training_x = pd.DataFrame(rows,columns=columns)
    used = training_x.columns[training_x.notna().any()].tolist()
    model.fit(training_x[used], labels)
    t = table(feed)[rec]
    evaluation = targets[targets.kind.eq('hidden-future')].copy()
    feat = []
    for r in evaluation.itertuples():
        now = int(r.now_minute_of_day)
        f = feature(t.loc[:now].drop(columns=r.unit,errors='ignore'),r.unit,now,int(r.horizon))
        feat.append(f+[ord(r.unit[-1])-65])
    evaluation['direct'] = np.maximum(0,model.predict(pd.DataFrame(feat,columns=columns)[used]))
    fixed = baseline[(baseline.recording.eq(rec)) & (baseline.kind.eq('hidden-future')) & (baseline.variant.eq('fixed'))]
    all_results.append(dict(fold=day,recording=rec,n=len(evaluation),
                            direct_mae=np.abs(evaluation.people-evaluation.direct).mean(),
                            fixed_mae=np.abs(fixed.truth-fixed.prediction).mean()))
results = pd.DataFrame(all_results)
results.to_csv(ROOT/'outputs/direct_hidden_future_scores.csv',index=False)
print(results.to_string(index=False))
