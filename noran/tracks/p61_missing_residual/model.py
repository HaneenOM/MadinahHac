"""Nested-cross-fitted residual learner on top of unmodified V7 missing."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('p61_direct_for_residual',ROOT/'noran/tracks/p61_missing/model.py')
direct=importlib.util.module_from_spec(spec);spec.loader.exec_module(direct)
v7=direct.v7
CONFIG_PATH=ROOT/'noran/configs/prompt61_residual.json'
SETTINGS=json.loads(CONFIG_PATH.read_text())
inference_rows=direct.inference_rows
baseline_values=direct.baseline_values


def calibrated_rows(rows,training,config):
    rows=rows.copy().reset_index(drop=True)
    for source,g in rows.groupby('source'):
        calibration=v7.tasks.fit_missing(training[training.recording.ne(source)],config)
        rows.loc[g.index,'v2']=v7.stack.v2_values(g,calibration,config)
    return rows


def features(rows,baseline):
    f=direct.features(rows)
    f['v7_correction']=(np.asarray(baseline)-rows.base.to_numpy(float))/np.maximum(rows.base,5)
    return f


def from_rows(rows,training,context):
    rows=rows.reset_index(drop=True)
    baseline=direct.baseline_from_rows(rows,training,context)
    oof=np.full(len(rows),np.nan)
    for source,g in rows.groupby('source'):
        # With an outer development holdout, rows and training already exclude
        # that outer source. This additional exclusion makes these inner OOF.
        inner_training=training[training.recording.ne(source)].copy()
        inner_rows=calibrated_rows(rows[rows.source.ne(source)],inner_training,context['config'])
        inner=direct.baseline_from_rows(inner_rows,inner_training,context)
        oof[g.index]=direct.baseline_values(g,inner)
    assert np.isfinite(oof).all()
    scale=np.maximum(rows.base.to_numpy(float),5.)
    y=(rows.truth.to_numpy(float)-oof)/scale
    weight=scale*len(rows)/rows.groupby('source').source.transform('size').to_numpy(float)/rows.source.nunique()
    model=CatBoostRegressor(**SETTINGS['model'])
    model.fit(features(rows,oof),y,sample_weight=weight,verbose=False)
    return dict(baseline=baseline,nonlinear=model,nested_sources=sorted(rows.source.unique()))


def nonlinear_values(rows,model,baseline=None):
    if baseline is None: raise ValueError('Residual inference requires actual V7 baseline')
    delta=model.predict(features(rows,baseline))
    limit=SETTINGS['max_residual_relative_to_line']
    return np.maximum(0.,np.asarray(baseline)+np.maximum(rows.base,5)*np.clip(delta,-limit,limit))


def fit(training,day,context):
    return from_rows(v7.pseudo_rows(training,context['test_released'],context['config']),training,context)


def predict(case,t,model,context):
    rows=inference_rows(case,t)
    if rows.empty:return np.nan
    baseline=v7.predict(case,t,model['baseline'],context)
    if not np.isfinite(baseline):
        baseline=v7.tasks.missing_value(t,case.unit,int(case.minute),model['baseline']['calibration'],context['config'])[1]
    extra=nonlinear_values(rows,model['nonlinear'],np.array([baseline]))[0]
    return float((1-SETTINGS['blend'])*baseline+SETTINGS['blend']*extra)
