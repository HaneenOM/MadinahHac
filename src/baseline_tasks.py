"""Shared task algorithms for the editable notebook and date-held-out evaluation.

No files are written here. Training always receives an explicitly supplied frame.
Production uses official sample references; evaluation rebuilds fold references.
Legacy decimal formatting lives in baseline_workflow, not in the algorithms.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from pipeline import table
import v2_hidden
import v2_missing
from v2_experiments import fit_direct, predict_hf
from prayer_experiments import fit_direct_variant, predict_direct_variant
from im_forecast_fallback_trial import context, predict as predict_im

DEFAULT_CONFIG = {
    'forecast': {'model_weight': 1., 'im_max_three_age': 5, 'im_median_weight': .5},
    'hidden': {'alpha': 3000., 'blends': {'cam_B': ['dropout', .75],
        'cam_G': ['dropout', .5], 'cam_D': ['ridge', .5]}, 'd_only': False},
    'hidden-future': {'weight': .75, 'saturday_prayer': True,
        'model_params': {'loss': 'absolute_error', 'max_iter': 100,
            'learning_rate': .06, 'max_leaf_nodes': 12,
            'min_samples_leaf': 40, 'random_state': 27}},
    'missing': {'min_gap': 6, 'weights': {'cam_A': 1., 'cam_B': 1.,
        'cam_C': 1., 'cam_D': 0., 'cam_E': 1., 'cam_F': .5,
        'cam_G': 1., 'cam_H': 0.}},
}


def default_config():
    return copy.deepcopy(DEFAULT_CONFIG)


def forecast_value(reference, model_value, config):
    value = model_value if np.isfinite(model_value) else reference
    return float(reference + config['model_weight']*(value-reference))


def im_forecast_value(series, now, config):
    return predict_im(dict(now=now, **context(series, now)),
        max_three_age=config['im_max_three_age'], median_weight=config['im_median_weight'])


def load_forecast(data_dir, assets, frozen_config):
    """Load all 12 fitted covariances; a missing asset can NEVER trigger fitting."""
    from t05_forecast.core import load_recordings
    from t05_forecast.pipeline import FRI, SAT, make_template, load_params
    from t05_forecast.submit import member_settings, settings
    from t05_forecast.template import slope
    recs = load_recordings(data_dir)
    models=[]
    for member in member_settings(settings(frozen_config)):
        tm={name: make_template(recs, name, member['lam']) for name in (FRI,SAT)}
        params={name: load_params(assets/'t05'/f'params_without_{day}_{member["key"]}.npz')
                for name,day in [(FRI,'fri'),(SAT,'sat')]}
        models.append(dict(settings=member, template=tm,
            slope={n:slope(a) for n,a in tm.items()}, params=params))
    return dict(recordings=recs, members=models)


def condition_forecast(rec, members):
    """Shared ensemble inference for production AND held-out benchmark recordings.

    Each member is (template, template_slope, fitted_covariance). Never fits.
    """
    from t05_forecast import gp
    predictions=[]
    for template,dslope,params in members:
        p=gp.predict_recording(rec,template,dslope,params,strict=True)
        for j,u in enumerate(rec.units):
            if u[-1] in 'IJKLM':p[:,j]=np.nan
        predictions.append(p)
    stack=np.stack(predictions);count=np.isfinite(stack).sum(axis=0)
    return np.divide(np.nansum(stack,axis=0),count,out=np.full(rec.y.shape,np.nan),where=count>0)


def predict_forecast(models, targets, config):
    """Strict all-camera T05 conditioning; A-H ensemble and causal I-M fallback."""
    from t05_forecast import gp
    from t05_forecast.core import FORECAST
    from t05_forecast.reference import ref_forecast
    result={}
    for name,group in targets.groupby('recording',sort=False):
        rec=models['recordings'][name].copy()
        # Retain the original GP target batches (including timed HF cells):
        # changing matrix batch sizes changes last-bit BLAS rounding. Only the
        # forecast rows are returned; other tasks use their own predictors.
        mean=condition_forecast(rec,[(m['template'][name],m['slope'][name],m['params'][name])
                                     for m in models['members']])
        ref=ref_forecast(rec)
        for r in group.itertuples():
            row,col=int(r.minute_of_day)-rec.first,rec.col(r.unit)
            result[r.id]=forecast_value(ref[row,col],mean[row,col],config)
    return result


def fit_hidden(training, config, schedule=None):
    models={'linear': v2_hidden.fit(table(training),alpha=config['alpha'])}
    if config['d_only']:
        import hidden_reconstruction_trial as nonlinear
        models['nonlinear'],_ = nonlinear.fit_new(training,schedule)
    return models


def hidden_value(t, unit, minute, models, config, reference=None, day=None, schedule=None, *, legacy_mix=False):
    linear=models['linear']
    if reference is None:reference=v2_hidden.predict(t,unit,minute,linear,'single')[0]
    method,weight=config['blends'][unit]
    fitted=v2_hidden.predict(t,unit,minute,linear,method)[0]
    value=max(0.,(1-weight)*reference+weight*fitted if legacy_mix else reference+weight*(fitted-reference))
    if config['d_only'] and unit=='cam_D':
        import hidden_reconstruction_trial as nonlinear
        raw=t.reindex(index=[minute])
        x,available=nonlinear.make_features(raw,unit,[day],schedule)
        if available.iloc[0]>=2 and unit in models['nonlinear']:
            value=max(0.,float(models['nonlinear'][unit].predict(x)[0]))
    return float(reference),float(value),float(fitted)


def fit_hidden_future(training, schedule, config):
    frames=table(training)
    rec_day=training.groupby('recording').day.first().to_dict()
    return dict(hidden=v2_hidden.fit(frames),
        direct=fit_direct(frames,model_params=config['model_params']),
        prayer_direct=fit_direct_variant(frames,rec_day,schedule,'prayer',
            model_params=config['model_params']) if config['saturday_prayer'] else None)


def hidden_future_value(t,unit,now,horizon,day,schedule,models,config,reference=None,*,legacy_mix=False):
    # Both branches use the same prefix, including any model's donor features.
    causal=t.loc[:now]
    if reference is None:
        reference=predict_hf(causal,unit,now,horizon,models['hidden'],models['direct'],'fixed')
    if config['saturday_prayer'] and pd.Timestamp(day).day_of_week==5:
        fitted=predict_direct_variant(causal,unit,now,horizon,day,schedule,models['prayer_direct'],'prayer')
    else:
        fitted=predict_hf(causal,unit,now,horizon,models['hidden'],models['direct'],'direct')
    w=config['weight']
    value=(1-w)*reference+w*fitted if legacy_mix and not (config['saturday_prayer'] and pd.Timestamp(day).day_of_week==5) else reference+w*(fitted-reference)
    return float(reference),max(0.,float(value)),float(fitted)


def fit_missing(training, config):
    return v2_missing.fit_calibration(table(training))


def missing_value(t,unit,minute,calibration,config,reference=None):
    if reference is None:reference=v2_missing.predict(t,unit,minute,calibration,'linear')[0]
    value=reference;available=0;gap=np.nan;used=False
    weight=config['weights'].get(unit,0.)
    if weight:
        fitted,available,gap=v2_missing.predict(t,unit,minute,calibration,
            'blend_full' if weight==1. else 'blend',weight)
        if np.isfinite(fitted) and gap>=config['min_gap'] and available>0:
            value=fitted;used=True
    return float(reference),float(value),dict(used=used,donors=available,gap=gap)


def predict_hidden(models,targets,feeds,references,config,schedule):
    rows=[]
    for r in targets.itertuples():
        ref,value,fitted=hidden_value(feeds[r.recording],r.unit,int(r.minute_of_day),
            models,config,references[r.id],r.day,schedule,
            legacy_mix=r.unit=='cam_G' and config['blends'][r.unit][1]==.5)
        rows.append(dict(id=r.id,people=value,fitted=fitted))
    return pd.DataFrame(rows)


def predict_hidden_future(models,targets,feeds,references,config,schedule):
    rows=[]
    for r in targets.itertuples():
        now=int(r.now_minute_of_day)
        ref,value,fitted=hidden_future_value(feeds[r.recording],r.unit,now,
            int(r.minute_of_day)-now,r.day,schedule,models,config,references[r.id],legacy_mix=True)
        rows.append(dict(id=r.id,people=value,fitted=fitted))
    return pd.DataFrame(rows)


def predict_missing(calibration,targets,feeds,references,config):
    rows=[]
    for r in targets.itertuples():
        _,value,meta=missing_value(feeds[r.recording],r.unit,int(r.minute_of_day),
            calibration,config,references[r.id])
        rows.append(dict(id=r.id,people=value,**meta))
    return pd.DataFrame(rows)
