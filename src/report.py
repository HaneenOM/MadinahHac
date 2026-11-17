"""Summarize selected date-held-out errors by task, camera, and horizon."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root/'outputs'
detail = pd.read_csv(out/'validation_predictions.csv')
choice = json.loads((out/'selection.json').read_text())
proxy = {'hidden':'ridge', 'hidden-future':'fixed', 'missing':'linear'}
sel = pd.concat([detail[(detail.kind == k) & (detail.variant == (proxy[k] if v == 'reference' else v))]
                 for k, v in choice.items()], ignore_index=True)
sel['error'] = (sel.truth-sel.prediction).abs()
sel['reference_error'] = (sel.truth-sel.reference).abs()
for group, name in [(['kind'], 'validation_experiment_by_kind.csv'),
                    (['kind', 'unit'], 'validation_experiment_by_camera.csv'),
                    (['kind', 'horizon'], 'validation_experiment_by_horizon.csv')]:
    x = sel.groupby(group, dropna=False).agg(n=('error','size'), mae=('error','mean'),
            reference_mae=('reference_error','mean')).reset_index()
    x['normalized_mae'] = x.mae/x.reference_mae.replace(0, np.nan)
    x.to_csv(out/name, index=False)
    if group == ['kind']:
        print(x.to_string(index=False))

sample = pd.read_csv(out/'reference_submission.csv').set_index('id')
starter = pd.read_csv(out/'starter_submission.csv').set_index('id')
selected = pd.read_csv(out/'submission.csv').set_index('id')
test = pd.read_csv(root/'data/raw/test.csv').dropna(subset='id').set_index('id')
summary = []
for kind, g in test.groupby('kind'):
    ids = g.index
    summary.append(dict(kind=kind, n=len(ids), starter_changed=int((starter.loc[ids,'people'] != sample.loc[ids,'people']).sum()),
                        selected_changed=int((selected.loc[ids,'people'] != sample.loc[ids,'people']).sum())))
pd.DataFrame(summary).to_csv(out/'submission_comparison.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))

camera = test[['kind','unit']].copy()
camera['prediction'] = selected.loc[camera.index,'people']
camera['reference'] = sample.loc[camera.index,'people']
camera['unseen'] = camera.unit.str[-1].isin(list('IJKLM'))
camera.groupby(['kind','unit','unseen']).agg(n=('prediction','size'),
        prediction_mean=('prediction','mean'),reference_mean=('reference','mean'))\
        .reset_index().to_csv(out/'submission_by_camera.csv',index=False)
