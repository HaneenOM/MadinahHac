"""Friday B/G error and source-support audit; no candidate model is trained."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import v2_hidden
import hidden_reconstruction_trial as previous
from hidden_candidate import HIDDEN_CANDIDATE
from pipeline import ROOT,load,locate_data,table
from v2_missing import interp_at

OUT=ROOT/'outputs/friday_hidden_diagnostic'
OLD=ROOT/'outputs/hidden_reconstruction_trial'
FIXED=ROOT/'outputs/local_score'
CAMERAS=('cam_B','cam_G')


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/'outputs').glob('submission*.csv'))+list((ROOT/'src').glob('*.py'))
    paths+=list((ROOT/'t05_forecast').glob('*.py'))+list(FIXED.glob('*.*'))+list(OLD.glob('*.*'))
    protected={str(p):previous.sha(p) for p in paths if p.is_file()}
    primary,feeds=previous.fixed_cases()
    train,test,_,_=load(locate_data(None))
    extra,_,extra_feeds=previous.extra_cases(train);feeds.update(extra_feeds)
    metadata=pd.concat([primary,extra],ignore_index=True)
    metadata=metadata[metadata.unit.isin(CAMERAS)].copy()
    saved=pd.read_csv(OLD/'predictions.csv').query("unit in ['cam_B','cam_G']").set_index('case_id')
    assert len(metadata)==621 and set(metadata.case_id)==set(saved.index)
    rows=[];support=[];checks=0
    with threadpool_limits(limits=4):
        for day,cases in metadata.groupby('day'):
            fitting=train[train.day.ne(day)]
            assert not fitting.day.eq(day).any() and not set(fitting.recording).intersection(cases.recording)
            models=v2_hidden.fit(table(fitting))  # unchanged baseline only
            for unit in CAMERAS:
                info=models[unit];preferred=v2_hidden.REFERENCE_DONOR[unit]
                # Reproduce the same eligibility used in current fitting.
                wide=pd.concat(table(fitting).values(),keys=table(fitting).keys(),names=['recording','minute'])
                valid=wide[unit].notna()&wide[info['donors']].notna().sum(axis=1).ge(2)
                days=fitting.drop_duplicates('recording').set_index('recording').day.to_dict()
                for name,g in wide[valid].groupby(level=0):
                    pair=g.reindex(columns=[unit,preferred]).dropna()
                    support.append(dict(heldout_day=day,unit=unit,preferred_donor=preferred,
                        training_recording=name,training_day=days[name],weekday=pd.Timestamp(days[name]).day_name(),
                        target_rows=len(g),preferred_overlap=len(pair),
                        overlap_ratio=float(pair[unit].mean()/pair[preferred].mean()) if len(pair) and pair[preferred].mean()>0 else None))
                pair=fitting.pivot(index=['recording','minute_of_day'],columns='unit',values='people').reindex(columns=[preferred])
                levels=pair[preferred].dropna()
                q25,q75=levels.quantile([.25,.75]).to_numpy() if len(levels) else (np.nan,np.nan)
                selected=cases[cases.unit.eq(unit)]
                for r in selected.itertuples():
                    frame=feeds[r.recording].drop(columns=['cam_B','cam_G'],errors='ignore')
                    minute=int(r.minute)
                    values={c:interp_at(frame[c],minute) if c in frame else np.nan for c in info['donors']}
                    available=sum(np.isfinite(v) for v in values.values())
                    exact=frame.reindex(index=[minute],columns=info['donors']).iloc[0].notna().sum()
                    chosen=next((donor for _,donor,_,_ in info['pairs'] if np.isfinite(values[donor])),None)
                    ref,_=v2_hidden.predict(frame,unit,minute,models,'single')
                    method,weight=HIDDEN_CANDIDATE[unit]
                    fitted,_=v2_hidden.predict(frame,unit,minute,models,method)
                    prediction=ref+weight*(fitted-ref)
                    np.testing.assert_allclose([prediction,ref],[saved.loc[r.case_id,'baseline'],saved.loc[r.case_id,'reference']],rtol=0,atol=1e-9)
                    altered=frame.copy()
                    altered['cam_B']=1e9+np.arange(len(altered));altered['cam_G']=2e9+np.arange(len(altered))
                    ref2,_=v2_hidden.predict(altered,unit,minute,models,'single')
                    fitted2,_=v2_hidden.predict(altered,unit,minute,models,method)
                    np.testing.assert_allclose([ref,fitted],[ref2,fitted2],rtol=0,atol=1e-10)
                    preferred_value=values[preferred]
                    level='absent' if not np.isfinite(preferred_value) else 'low' if preferred_value<=q25 else 'high' if preferred_value>=q75 else 'middle'
                    rows.append(dict(case_id=r.case_id,day=day,recording=r.recording,unit=unit,minute=minute,
                        window_id=r.window_id,prediction=prediction,reference=ref,unblended=fitted,
                        exact_donors=int(exact),usable_donors=available,selected_donor=chosen or 'target_median',
                        preferred_donor=preferred,preferred_available=bool(np.isfinite(preferred_value)),
                        preferred_level=level,route='Ridge_blend' if available>=2 else 'donor_fallback'))
                    checks+=1
    pred=pd.DataFrame(rows);pred.to_csv(OUT/'baseline_predictions_and_routes.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'donor_training_support.csv',index=False)
    labels=pd.concat([pd.read_csv(OLD/'fixed_scored.csv'),pd.read_csv(OLD/'additional_scored.csv')])[['case_id','truth']]
    scored=pred.merge(labels,on='case_id',validate='one_to_one')
    assert len(scored)==621
    for col in ('prediction','reference','unblended'):scored[col+'_error']=(scored[col]-scored.truth).abs()
    scored['bias']=scored.prediction-scored.truth
    scored['target_level']=pd.cut(scored.truth,[-1,10,20,40,np.inf],labels=['0-10','10-20','20-40','over40']).astype(str)
    for groups,name in [(['recording','unit'],'recording_camera'),(['recording','unit','window_id'],'window'),
                         (['recording','unit','target_level'],'target_level'),
                         (['recording','unit','usable_donors','preferred_available'],'availability'),
                         (['recording','unit','route'],'route'),(['recording','unit','selected_donor'],'selected_donor'),
                         (['recording','unit','preferred_level'],'observable_donor_level')]:
        result=scored.groupby(groups).agg(targets=('case_id','size'),windows=('window_id','nunique'),
            timestamps=('minute','nunique'),mae=('prediction_error','mean'),reference_mae=('reference_error','mean'),
            unblended_mae=('unblended_error','mean'),bias=('bias','mean')).reset_index()
        result['normalized']=result.mae/result.reference_mae.replace(0,np.nan)
        result.to_csv(OUT/f'by_{name}.csv',index=False)
    loo=[]
    for (recording,unit),g in scored.groupby(['recording','unit']):
        for window in g.window_id.unique():
            kept=g[g.window_id.ne(window)]
            loo.append(dict(recording=recording,unit=unit,omitted_window=window,targets=len(kept),
                            baseline_mae=kept.prediction_error.mean(),baseline_normalized=kept.prediction_error.mean()/kept.reference_error.mean()))
    pd.DataFrame(loo).to_csv(OUT/'baseline_leave_one_window_out.csv',index=False)
    scored.nlargest(25,'prediction_error').to_csv(OUT/'largest_errors.csv',index=False)
    scored.to_csv(OUT/'scored_diagnostic.csv',index=False)
    # Actual Friday input availability only; do not use hidden test values.
    visible=table(test[test.id.isna()&test.people.notna()&~test.unit.isin(CAMERAS)])
    real=[]
    for r in test[test.kind.eq('hidden')&test.unit.isin(CAMERAS)].itertuples():
        frame=visible[r.recording];preferred=v2_hidden.REFERENCE_DONOR[r.unit]
        value=interp_at(frame[preferred],int(r.minute_of_day)) if preferred in frame else np.nan
        real.append(dict(unit=r.unit,minute=int(r.minute_of_day),preferred_available=bool(np.isfinite(value))))
    pd.DataFrame(real).groupby('unit').agg(targets=('unit','size'),preferred_available=('preferred_available','sum')).reset_index().to_csv(OUT/'actual_preferred_donor_coverage.csv',index=False)
    train.groupby(['recording','day','unit']).size().rename('released_rows').reset_index().to_csv(OUT/'training_camera_coverage.csv',index=False)
    assert all(previous.sha(Path(p))==value for p,value in protected.items())
    report=dict(candidate_models_trained=0,baseline_folds_reproduced=2,t05_fits=0,submission_created=False,
                diagnostic_targets=len(scored),hidden_invariance_checks=checks,
                donor_fallback_rows=int(scored.route.eq('donor_fallback').sum()),
                ridge_blend_rows=int(scored.route.eq('Ridge_blend').sum()),protected_files_unchanged=True,
                elapsed_seconds=time.perf_counter()-started,protected_sha256=protected)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'by_recording_camera.csv').to_string(index=False))
    print('Target levels:');print(pd.read_csv(OUT/'by_target_level.csv').to_string(index=False))
    print('Observable donor levels:');print(pd.read_csv(OUT/'by_observable_donor_level.csv').to_string(index=False))
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2))


if __name__=='__main__':main()
