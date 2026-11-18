"""P41: normalized TARGET Ridge, isolated from production and Sep11 selection.

Run select before evaluate. Fixed masks, P40 helpers, no forecast model fitting.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import friday_hidden_improvement as p40
import v2_hidden
from pipeline import table
from released_hidden_training_trial import training_rows,expand,release_training,protected_snapshot,sha,row_keys
from evaluate_baseline_task import fixed_benchmark
from reliable_evaluation import annotate_cases,case_fingerprints

ROOT=p40.ROOT
WEIGHTS=(.25,.5,1.)
METHODS=('original','p40','ratio')

def snapshot():
    result=protected_snapshot()
    for f in (ROOT/'outputs/prompt40_friday_hidden').rglob('*'):
        if f.is_file():result[str(f)]=sha(f)
    result['P40_source']=sha(ROOT/'src/friday_hidden_improvement.py')
    return result

def design(raw,med,floor):
    # Scale is the mean of ACTUALLY observed, camera-median-normalized counts.
    # Missing counts never enter numerator/denominator by imputation.
    x=raw.reindex(columns=p40.DONORS).astype(float)
    seen=x.notna();normalized=x.div(med,axis=1)
    scale=normalized.mean(axis=1)
    ok=seen.sum(axis=1).ge(2)&scale.ge(floor)
    safe=scale.clip(lower=floor).fillna(floor)
    features=normalized.div(safe,axis=0).fillna(0.)
    features.columns=[u+'_relative' for u in features.columns]
    for u in p40.DONORS:features[u+'_seen']=seen[u].astype(float)
    minute=np.asarray(raw.index.get_level_values(-1),float)
    features['sin']=np.sin(2*np.pi*minute/1440)
    features['cos']=np.cos(2*np.pi*minute/1440)
    return features,scale,ok

def fitting_data(training,unit):
    tables=table(training)
    full=pd.concat(tables.values(),keys=tables.keys(),names=['recording','minute_of_day'])
    raw=full.reindex(columns=p40.DONORS)
    valid=full[unit].notna()&raw.notna().sum(axis=1).ge(2)
    return raw.loc[valid],full.loc[valid,unit].to_numpy(float)

def fit(training,unit):
    raw,y=fitting_data(training,unit)
    med=raw.median().clip(lower=1.).fillna(1.)
    initial=raw.div(med,axis=1).mean(axis=1)
    positive=initial[initial.gt(0)]
    floor=max(.01,float(positive.quantile(.01)))
    days=training.groupby('recording').day.first().to_dict()
    friday=np.array([pd.Timestamp(days[r]).dayofweek==4 for r in raw.index.get_level_values(0)])
    frames=[];targets=[];weights=[]
    for seed in (None,0,1,2):
        x=raw if seed is None else raw.mask(np.random.default_rng(seed).random(raw.shape)<.2)
        features,scale,ok=design(x,med,floor)
        frames.append(features.loc[ok]);targets.append(y[ok]/scale[ok].to_numpy())
        # The scale-squared weight makes normalized-target squared loss equal
        # to count squared loss, before regularization; Friday weight fixed at3.
        weights.append(scale[ok].to_numpy()**2*np.where(friday[ok],3.,1.))
    weight=np.concatenate(weights);weight/=weight.mean()
    model=Ridge(alpha=10.).fit(pd.concat(frames,ignore_index=True),np.concatenate(targets),sample_weight=weight)
    return dict(model=model,med=med,floor=floor,rows=len(raw),records=sorted(raw.index.get_level_values(0).unique()))

def predict(model,raw,fallback):
    features,scale,ok=design(raw,model['med'],model['floor'])
    value=np.maximum(0.,model['model'].predict(features)*scale.fillna(0).to_numpy())
    return np.where(ok,value,fallback),ok.to_numpy()

def anchor_models(new,schedule):
    return p40.fit_options(new,'cam_G',schedule,['friday_weighted'])['friday_weighted']

def values(baseline,g_model,raw,unit,days,schedule):
    reference,original,available=p40.current_values(baseline[unit],raw,unit)
    if unit=='cam_G':
        fitted=p40.fitted_values(g_model,'friday_weighted',raw,unit,days,schedule)
        anchor=np.where(available>=2,np.maximum(0.,reference+.5*(fitted-reference)),original)
    else:anchor=original.copy()
    return reference,original,anchor,available

def score(pred,labels,out):
    out.mkdir(exist_ok=False);pred.to_csv(out/'predictions.csv',index=False)
    scored=pred.merge(labels,on='case_id',validate='one_to_one')
    for m in (*METHODS,'reference'):scored[m+'_error']=(scored[m]-scored.truth).abs()
    scored.to_csv(out/'scored.csv',index=False)
    for groups,name in [(['day','recording','unit'],'recording'),(['unit','window_id'],'window'),(['unit'],'camera')]:
        rows=[]
        for key,g in scored.groupby(groups):
            key=key if isinstance(key,tuple) else (key,)
            row=dict(zip(groups,key));row.update(targets=len(g),ratio_model=int(g.ratio_model.sum()),fallback=int((~g.ratio_model).sum()))
            ref=g.reference_error.mean()
            for m in METHODS:row.update({m+'_mae':g[m+'_error'].mean(),m+'_normalized':g[m+'_error'].mean()/ref})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/f'by_{name}.csv',index=False)
    sensitivity=[]
    for unit,g in scored.groupby('unit'):
        for against in ('original','p40'):
            gain=g[against+'_error']-g.ratio_error
            windows=pd.DataFrame(dict(window=g.window_id,gain=gain)).groupby('window').gain.agg(['sum','mean'])
            best=windows['sum'].nlargest(2).index
            sensitivity.append(dict(unit=unit,against=against,gain=gain.mean(),
                improved_windows=int(windows['mean'].gt(0).sum()),worsened_windows=int(windows['mean'].lt(0).sum()),
                gain_without_two_best=gain[~g.window_id.isin(best)].mean(),
                best_two=list(best),largest_window_regression=-windows['mean'].min(),
                worst_window=windows['mean'].idxmin(),
                leave_one_min=min(gain[g.window_id.ne(w)].mean() for w in windows.index)))
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    return scored

def stability(training,out):
    rows=[]
    for unit in p40.CAMERAS:
        raw,y=fitting_data(training,unit);model=fit(training,unit)
        _,scale,ok=design(raw,model['med'],model['floor'])
        q=scale.quantile([1/3,2/3]).to_numpy()
        for (rec,minute),target,s,valid,seen,e in zip(raw.index,y,scale,ok,raw.notna().sum(axis=1),raw.cam_E.notna()):
            rows.append(dict(unit=unit,recording=rec,minute=minute,window_id=f'{rec}:{int(minute)//30}',
                target=target,scale=s,ratio=target/s if valid else np.nan,eligible=valid,
                donors=seen,E_available=e,crowd_band=int(np.searchsorted(q,s))))
    detail=pd.DataFrame(rows);detail.to_csv(out/'training_relationships.csv',index=False)
    for groups,name in [(['unit','recording'],'recording'),(['unit','recording','crowd_band'],'crowd'),
                        (['unit','recording','donors','E_available'],'availability'),(['unit','window_id'],'window')]:
        detail.groupby(groups).agg(pairs=('ratio','count'),windows=('window_id','nunique'),
            target_median=('target','median'),ratio_median=('ratio','median'),
            ratio_q25=('ratio',lambda x:x.quantile(.25)),ratio_q75=('ratio',lambda x:x.quantile(.75))).reset_index().to_csv(out/f'stability_{name}.csv',index=False)

def select(out):
    started=time.perf_counter();out.mkdir(exist_ok=False,parents=True);protected=snapshot()
    train,test,schedule=p40.inputs()
    spec=dict(novelty='Normalized target y/visible scale, reconstructed as scale*predicted ratio. Existing HGB already had donor shares but raw-count target; production has constant mean donor ratios.',
        donors=p40.DONORS,scale='Mean observed donor/fit-only donor median; >=2 donors. Median floored at1.',
        floor='max(.01, 1st percentile positive fit-only scale); below floor fall back to P40.',
        features='Normalized donor/scale, seen flags, clock sin/cos. No raw counts, Friday flag, B or G inputs.',
        family='Ridge alpha10, original +3 deterministic 20% dropout copies, training Friday weight3, scale-squared loss weights normalized to mean1.',
        weights=list(WEIGHTS),anchor='Frozen P40: B original; G expanded Friday-weight3 Ridge with .5 reference blend.',
        selection='Entire Sep11 excluded. Date folds Sep4/5/12, five-minute P40 cases. Minimize .5 old-Friday relative MAE +.5 mean Saturday-record relative MAE vs P40. Require >=0.5% gain for development support.',
        evaluation='Best ratio blend frozen per camera regardless of development support; fixed Sep11 labels used only subsequently. No post-evaluation alternate selection.',
        candidate_gate='Development supported AND positive vs P40 in both protocols, both gains survive removing two best windows, >=half windows improve in each. Otherwise retain existing candidates.',
        metric='Project-specific MAE/reference MAE on identical cases; not a confirmed official formula.')
    (out/'design.json').write_text(json.dumps(spec,indent=2))
    dev=train[train.day.ne(p40.PRIMARY_DAY)].copy();expanded,_=expand(training_rows(dev),test)
    assert p40.PRIMARY_DAY not in set(expanded.day)
    stability(expanded,out)
    cases,labels,feeds=p40.tuning_cases(dev,test)
    cases.to_csv(out/'development_cases.csv',index=False);labels.to_csv(out/'development_labels.csv',index=False)
    rows=[];manifest=[]
    with threadpool_limits(limits=4):
        for day,today in cases.groupby('day'):
            old=training_rows(dev[dev.day.ne(day)]);new,_=expand(old,test)
            assert day not in set(new.day) and p40.PRIMARY_DAY not in set(new.day)
            baseline=v2_hidden.fit(table(old));gm=anchor_models(new,schedule)
            for unit in p40.CAMERAS:
                model=fit(new,unit)
                manifest.append(dict(heldout=day,unit=unit,rows=model['rows'],records=model['records'],floor=model['floor'],med=model['med'].to_dict()))
                for rec,g in today[today.unit.eq(unit)].groupby('recording'):
                    raw=p40.raw_inputs(feeds[rec],g);ref,original,anchor,available=values(baseline,gm,raw,unit,g.day.tolist(),schedule)
                    fitted,ok=predict(model,raw,anchor)
                    for weight in WEIGHTS:
                        pred=anchor+weight*(fitted-anchor)
                        for r,a,b,c,d,valid in zip(g.itertuples(),ref,original,anchor,pred,ok):
                            rows.append(dict(case_id=r.case_id,day=day,recording=rec,unit=unit,window_id=r.window_id,weight=weight,
                                reference=a,original=b,p40=c,ratio=d,ratio_model=bool(valid)))
    predictions=pd.DataFrame(rows);predictions.to_csv(out/'development_predictions.csv',index=False)
    scored=predictions.merge(labels,on='case_id',validate='many_to_one')
    for m in (*METHODS,'reference'):scored[m+'_error']=(scored[m]-scored.truth).abs()
    by=scored.groupby(['unit','weight','day','recording']).agg(targets=('case_id','size'),
        original_mae=('original_error','mean'),p40_mae=('p40_error','mean'),ratio_mae=('ratio_error','mean'),reference_mae=('reference_error','mean')).reset_index()
    by['relative']=by.ratio_mae/by.p40_mae;by.to_csv(out/'development_by_recording.csv',index=False)
    objectives=[]
    for (unit,weight),g in by.groupby(['unit','weight']):
        fri=pd.to_datetime(g.day).dt.dayofweek.eq(4)
        objectives.append(dict(unit=unit,weight=weight,objective=.5*g.loc[fri,'relative'].mean()+.5*g.loc[~fri,'relative'].mean()))
    objectives=pd.DataFrame(objectives);objectives.to_csv(out/'development_objectives.csv',index=False)
    selected={}
    for unit in p40.CAMERAS:
        best=objectives[objectives.unit.eq(unit)].sort_values(['objective','weight']).iloc[0]
        selected[unit]=dict(weight=float(best.weight),objective=float(best.objective),development_supported=bool(best.objective<=.995))
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    (out/'fitting_manifest.json').write_text(json.dumps(manifest,indent=2))
    assert snapshot()==protected
    (out/'selection_verification.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,protected_files=len(protected),source_sha256=sha(__file__),fixed_evaluation_labels_used=False),indent=2))
    print(json.dumps(selected,indent=2))

def evaluate(out):
    started=time.perf_counter();protected=snapshot();selected=json.loads((out/'selected.json').read_text());locked=sha(out/'selected.json')
    assert sha(__file__)==json.loads((out/'selection_verification.json').read_text())['source_sha256']
    train,test,schedule=p40.inputs()
    with contextlib.redirect_stdout(io.StringIO()):cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
    assert case_fingerprints(cases,labels)==json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())['historical']
    cases=annotate_cases(cases,feeds)
    friday=cases[cases.kind.eq('hidden')&cases.unit.isin(p40.CAMERAS)&cases.day.eq(p40.PRIMARY_DAY)]
    base,expanded,_,blocked=release_training(train,test,feeds)
    strict=training_rows(train[train.day.ne(p40.PRIMARY_DAY)]);se,_=expand(strict,test)
    all_results={};coverage=[];manifests=[]
    actual_feeds=table(test[test.id.isna()&test.people.notna()])
    actual=test[test.kind.eq('hidden')&test.unit.isin(p40.CAMERAS)].copy();actual['minute']=actual.minute_of_day
    with threadpool_limits(limits=4):
        for protocol,old,new in [('strict',strict,se),('release_matched',base,expanded)]:
            if protocol=='release_matched':assert not (row_keys(new)&blocked)
            else:assert p40.PRIMARY_DAY not in set(new.day)
            baseline=v2_hidden.fit(table(old));gm=anchor_models(new,schedule);rows=[]
            for unit in p40.CAMERAS:
                model=fit(new,unit)
                manifests.append(dict(protocol=protocol,unit=unit,rows=model['rows'],records=model['records'],med=model['med'].to_dict(),floor=model['floor']))
                for rec,g in friday[friday.unit.eq(unit)].groupby('recording'):
                    raw=p40.raw_inputs(feeds[rec],g)
                    poison=feeds[rec].copy();poison['cam_B']=1e12;poison['cam_G']=-1e12
                    pd.testing.assert_frame_equal(raw,p40.raw_inputs(poison,g))
                    ref,original,anchor,available=values(baseline,gm,raw,unit,g.day.tolist(),schedule)
                    fitted,ok=predict(model,raw,anchor);pred=anchor+selected[unit]['weight']*(fitted-anchor)
                    coverage.append(dict(protocol=protocol,unit=unit,setting='fixed',targets=len(g),ratio_model=int(ok.sum()),mean_donors=available.mean()))
                    for r,a,b,c,d,valid in zip(g.itertuples(),ref,original,anchor,pred,ok):
                        row=r._asdict();row.update(reference=a,original=b,p40=c,ratio=d,ratio_model=bool(valid));rows.append(row)
                ga=actual[actual.unit.eq(unit)];raw=p40.raw_inputs(actual_feeds[ga.recording.iloc[0]],ga)
                _,_,ok=design(raw,model['med'],model['floor'])
                coverage.append(dict(protocol=protocol,unit=unit,setting='actual',targets=len(ga),ratio_model=int(ok.sum()),mean_donors=raw.notna().sum(axis=1).mean()))
            pred=pd.DataFrame(rows)
            # Confirm the two reference implementations reproduce saved P40.
            previous=pd.read_csv(ROOT/f'outputs/prompt40_friday_hidden/{protocol}/predictions.csv')
            merged=pred.merge(previous[['case_id','current','candidate']],on='case_id',validate='one_to_one')
            assert np.allclose(merged.original,merged.current,atol=1e-12,rtol=0)
            assert np.allclose(merged.p40,merged.candidate,atol=1e-12,rtol=0)
            all_results[protocol]=score(pred,labels,out/protocol)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    (out/'evaluation_fitting_manifest.json').write_text(json.dumps(manifests,indent=2))
    supported=[]
    for unit in p40.CAMERAS:
        credible=selected[unit]['development_supported']
        for protocol in all_results:
            s=pd.read_csv(out/protocol/'sensitivity.csv').query('unit==@unit and against=="p40"').iloc[0]
            credible=credible and s.gain>0 and s.gain_without_two_best>0 and s.improved_windows>=11
        if credible:supported.append(unit)
    assert protected==snapshot() and locked==sha(out/'selected.json')
    result=dict(seconds=time.perf_counter()-started,protected_files=len(protected),supported=supported,
        candidate_created=False,source_sha256=sha(__file__),selected_sha256=locked,
        note='No candidate unless predeclared support gates pass; never deploy diagnostic-only ratio predictions.')
    (out/'evaluation_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    for protocol in all_results:
        print(protocol);print(pd.read_csv(out/protocol/'by_camera.csv').to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['select','evaluate'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='select':select(args.output_dir)
    else:evaluate(args.output_dir)
