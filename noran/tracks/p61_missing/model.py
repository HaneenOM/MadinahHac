"""Experimental normalized nonlinear gap completion. Does not modify V7.

All features come from gap candidates formed after release/outage masks.
No truth, date, recording, artificial shift, or learned V7 prediction is a feature.
The fixed 50/50 blend needs no learned second-stage targets.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT/'src', ROOT/'noran/tracks/missing'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
spec = importlib.util.spec_from_file_location('p61_v7_missing', ROOT/'noran/tracks/missing/model.py')
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
SETTINGS = json.loads((ROOT/'noran/configs/prompt61.json').read_text())


def features(rows):
    """A fixed schema; generic partner slots share information across cameras."""
    d = rows.reset_index(drop=True)
    scale = np.maximum(d.base.to_numpy(float), 5.)
    f = pd.DataFrame(index=d.index)
    for c in 'span pos n_left n_right'.split(): f[c] = d[c].astype(float)
    f['fraction'] = (d.m-d.a)/(d.b-d.a)
    f['log_level'] = np.log1p(d.base)
    for unit in 'ABCDEFGHIJKLM': f['unit_'+unit] = d.unit.eq('cam_'+unit).astype(float)
    for col in ('own_2', 'own_4', 'spl_30'):
        f[col] = (d[col]-d.base)/scale
    for slot in (0, 1):
        for suffix in ('4','16','64','-1_4','+1_4'):
            vals = np.full(len(d),np.nan)
            for unit, idx in d.groupby('unit').groups.items():
                ps = v7.gaplib.PARTNERS.get(unit,())
                if slot >= len(ps): continue
                letter = ps[slot][4:]
                key = 'r'+letter+suffix if suffix[0] in '-+' else 'r'+letter+'_'+suffix
                if key in d: vals[np.asarray(idx)] = d.loc[idx,key]
            f[f'partner{slot}_{suffix}'] = (np.asarray(vals)-d.base)/scale
        for side in ('cl','cr'):
            vals = np.zeros(len(d))
            for unit, idx in d.groupby('unit').groups.items():
                ps = v7.gaplib.PARTNERS.get(unit,())
                key = side+'_'+ps[slot][4:] if slot < len(ps) else ''
                if key in d: vals[np.asarray(idx)] = d.loc[idx,key]
            f[f'partner{slot}_{side}'] = vals
    ratio = f[['partner0_4','partner1_4']]
    f['partner_disagreement'] = ratio.min(axis=1).lt(0)&ratio.max(axis=1).gt(0)
    return f.replace([np.inf,-np.inf], np.nan).astype(float).fillna(-999.)


def train_nonlinear(rows):
    x = features(rows)
    scale = np.maximum(rows.base.to_numpy(float), 5.)
    y = (rows.truth.to_numpy(float)-rows.base.to_numpy(float))/scale
    # Exact count-MAE objective, with inverse pseudo-row multiplicity per source
    # balancing the four recordings rather than treating overlaps as independence.
    counts = rows.groupby('source').source.transform('size').to_numpy(float)
    weight = scale * len(rows)/counts/rows.source.nunique()
    model = CatBoostRegressor(**SETTINGS['model'])
    model.fit(x, y, sample_weight=weight, verbose=False)
    return model


def baseline_from_rows(rows, training, context):
    return dict(weights=v7.stack.fit_spec(rows,v7.SPEC,ridge=v7.RIDGE),
                calibration=v7.tasks.fit_missing(training,context['config']),
                config=context['config'],fit_rows=len(rows))


def baseline_values(rows, baseline):
    d = rows.copy()
    d['v2'] = v7.stack.v2_values(d,baseline['calibration'],baseline['config'])
    # The reference predict() fills absent partner columns with NaN before
    # applying its largest-available-subset model. Match that behavior here.
    for unit in d.unit.unique():
        for col in v7.SPEC.get(unit,{}).get('ratios',()):
            if col not in d: d[col]=np.nan
    value = v7.stack.apply_spec(d,baseline['weights'],v7.SPEC,max_rel=v7.MAX_REL)
    value = v7.stack.blend_with_accepted(d.unit,value,d.v2,v7.SHRINK)
    return np.where(np.isfinite(value), value, d.v2)


def nonlinear_values(rows, model):
    delta = model.predict(features(rows))
    limit = SETTINGS['max_correction_relative_to_line']
    return np.maximum(0., rows.base.to_numpy(float)+np.maximum(rows.base,5)*np.clip(delta,-limit,limit))


def from_rows(rows, training, context):
    return dict(baseline=baseline_from_rows(rows,training,context), nonlinear=train_nonlinear(rows))


def fit(training,day,context):
    rows = v7.pseudo_rows(training,context['test_released'],context['config'])
    return from_rows(rows,training,context)


def inference_rows(case,t):
    unit,minute = case.unit,int(case.minute)
    bounds = v7.gap_context(t,unit,minute)
    if bounds is None: return pd.DataFrame()
    a,b = bounds[:2]
    lo = a-v7.MARGIN
    local = t.reindex(range(lo,b+v7.MARGIN+1))
    visible = {u:local[u].to_numpy(float) for u in local}
    return pd.DataFrame(v7.gaplib.gap_rows(visible,np.zeros(len(local)),unit,np.array([minute-lo]),
        v7.gaplib.PARTNERS.get(unit,()),{},legacy=False,donors=tuple(visible),**v7.GRIDS))


def predict(case,t,model,context):
    rows = inference_rows(case,t)
    if rows.empty: return np.nan
    base = v7.predict(case,t,model['baseline'],context)
    if not np.isfinite(base):
        base = v7.tasks.missing_value(t,case.unit,int(case.minute),model['baseline']['calibration'],context['config'])[1]
    extra = float(nonlinear_values(rows,model['nonlinear'])[0])
    return float((1-SETTINGS['blend'])*base+SETTINGS['blend']*extra)
