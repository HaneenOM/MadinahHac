"""P40: recording/date-aware development for Friday HIDDEN B/G only.

select: fixed small comparison on Sep4/5/12, NEVER Sep11 answers.
evaluate: frozen choices on unchanged Sep11 masks, strict and release-matched.
Candidate assembly can change only supported Friday HIDDEN rows of 0.806.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import csv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from pipeline import load,table
import v2_hidden
from v2_missing import interp_at
import hidden_reconstruction_trial as nonlinear
from prayer_experiments import timetable
from released_hidden_training_trial import (training_rows,expand,release_training,
    protected_snapshot,sha,BASELINE)
from reliable_evaluation import annotate_cases,case_fingerprints
from evaluate_baseline_task import fixed_benchmark

CAMERAS=('cam_B','cam_G')
PRIMARY_DAY='2026-09-11'
SPECS=('expanded_dropout','expanded_plain','friday_weighted','expanded_hgb')
WEIGHTS=(.5,.75,1.)
OLD_WEIGHT={'cam_B':.75,'cam_G':.5}
DONORS=[f'cam_{u}' for u in 'ACDEFH']


def inputs():
    with contextlib.redirect_stdout(io.StringIO()):
        train,test,_,_=load(ROOT/'data/raw')
        schedule=timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)
    return train,test,schedule


def raw_inputs(feed,cases):
    # Both target cameras removed before interpolation, seen flags or totals.
    safe=feed.drop(columns=list(CAMERAS),errors='ignore')
    index=pd.MultiIndex.from_arrays([cases.recording,cases.minute.astype(int)],names=['recording','minute_of_day'])
    rows=[[interp_at(safe[u],int(r.minute)) if u in safe else np.nan for u in DONORS]
          for r in cases.itertuples()]
    return pd.DataFrame(rows,index=index,columns=DONORS)


def fit_options(training,unit,schedule,specs=SPECS):
    full=pd.concat(table(training).values(),keys=table(training).keys(),names=['recording','minute_of_day'])
    valid=full[unit].notna() & full.reindex(columns=DONORS).notna().sum(axis=1).ge(2)
    raw=full.loc[valid].reindex(columns=DONORS);y=full.loc[valid,unit].to_numpy()
    day_map=training.groupby('recording').day.first().to_dict()
    days=[day_map[n] for n in raw.index.get_level_values(0)]
    med=raw.median().fillna(0.);x=v2_hidden._design(raw,DONORS,med)
    augmentation=[x]+[v2_hidden._design(raw,DONORS,med,dropout=True,seed=seed) for seed in range(3)]
    augmented=pd.concat(augmentation,ignore_index=True);replicated=np.tile(y,4)
    result={}
    for spec in specs:
        if spec=='expanded_plain':
            model=Ridge(alpha=3000.).fit(x,y)
        elif spec in ('expanded_dropout','friday_weighted'):
            weights=np.array([3. if spec=='friday_weighted' and pd.Timestamp(d).day_of_week==4 else 1. for d in days])
            model=Ridge(alpha=3000.).fit(augmented,replicated,sample_weight=np.tile(weights,4))
        else:
            design,_=nonlinear.make_features(raw,unit,days,schedule)
            dropped=raw.mask(np.random.default_rng(26).random(raw.shape)<.2)
            aug,seen=nonlinear.make_features(dropped,unit,days,schedule);keep=seen.ge(2)
            model=HistGradientBoostingRegressor(**nonlinear.PARAMS).fit(
                pd.concat([design,aug[keep]],ignore_index=True),np.concatenate([y,y[keep.to_numpy()]]))
        result[spec]=dict(model=model,med=med,training_rows=len(raw),training_recordings=sorted(raw.index.get_level_values(0).unique()))
    return result


def current_values(info,raw,unit,reference=None):
    arr=raw.reindex(columns=info['donors']).to_numpy(float)
    available=np.isfinite(arr).sum(axis=1)
    if reference is None:
        reference=np.full(len(raw),info['target_median'])
        pending=np.ones(len(raw),dtype=bool)
        for _,donor,coefficient,_ in info['pairs']:
            values=raw[donor].to_numpy() if donor in raw else np.full(len(raw),np.nan)
            use=pending & np.isfinite(values);reference[use]=np.maximum(0.,coefficient*values[use]);pending[use]=False
    reference=np.asarray(reference,float)
    fit=np.maximum(0.,info['dropout'].predict(v2_hidden._design(raw,info['donors'],info['med'])))
    fit=np.where(available>=2,fit,reference)
    value=np.maximum(0.,reference+OLD_WEIGHT[unit]*(fit-reference))
    return reference,value,available


def fitted_values(option,spec,raw,unit,days,schedule):
    if spec=='expanded_hgb':x,_=nonlinear.make_features(raw,unit,days,schedule)
    else:x=v2_hidden._design(raw,DONORS,option['med'])
    return np.maximum(0.,option['model'].predict(x))


def tuning_cases(train,test):
    # Fixed five-minute grid, sparse donor pattern transferred by recording
    # offset using release metadata only, including morning recordings.
    rows=[];labels=[];feeds={}
    mask=test[test.recording.eq('20260918_1100') & test.id.isna() & test.unit.isin(DONORS)]
    allowed={(r.unit,int(r.minute_of_day)-660) for r in mask.itertuples()}
    for name,g in train[train.day.ne(PRIMARY_DAY)].groupby('recording'):
        first=int(g.minute_of_day.min());day=g.day.iloc[0]
        visible=g[[((r.unit,int(r.minute_of_day)-first) in allowed) for r in g.itertuples()]]
        feeds.update(table(visible))
        targets=g[g.unit.isin(CAMERAS)&((g.minute_of_day-first)%5==0)]
        for r in targets.itertuples():
            key=f'development:{name}:{r.unit}:{int(r.minute_of_day)}'
            rows.append(dict(case_id=key,day=day,recording=name,unit=r.unit,minute=int(r.minute_of_day),
                window_id=f'{name}:{int(r.minute_of_day)//30*30}'))
            labels.append(dict(case_id=key,truth=float(r.people)))
    return pd.DataFrame(rows),pd.DataFrame(labels),feeds


def select(out):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    protected=protected_snapshot();train,test,schedule=inputs()
    design=dict(primary_holdout='Entire Sep11 excluded from ALL development fitting, preprocessing and selection',
        development_dates=['2026-09-04','2026-09-05','2026-09-12'],
        folds='Exclude entire development date; report each recording separately. B/G masked jointly in validation feed.',
        donor_mask='Fixed Friday released donor pattern transferred by recording offset, five-minute target grid; no label-based sampling.',
        options=list(SPECS),weights=list(WEIGHTS),alpha=3000.,friday_training_weight=3.,
        nonlinear=dict(nonlinear.PARAMS),
        new_reason_for_hgb='Previously rejected for G; new permitted released fitting rows, interpolation aligned to Ridge and held-recording selection. Not enabled merely because D improved.',
        objective='Per camera: 0.5 Friday record relative MAE + 0.5 equal-record Saturday relative MAE, relative to unchanged baseline. Only Sep4 Friday is available; short, E absent.',
        selection='Minimum development objective; improvement <0.5% retains unchanged baseline. No fixed Sep11 labels used.',
        support_gate='Frozen choice only: release-matched camera gain >0 and strict relative regression <=2%; if strict gain nonpositive require release gain >=3%; release gain survives removal of its two best camera-windows. No alternative model chosen on fixed labels.',
        Saturday='Never changed, including D-only candidate',rule='User-supplied P39: HIDDEN may use whole released table; any genuinely released row may train models.')
    (out/'design.json').write_text(json.dumps(design,indent=2))
    development=train[train.day.ne(PRIMARY_DAY)].copy()
    assert PRIMARY_DAY not in set(development.day)
    cases,labels,feeds=tuning_cases(development,test)
    cases.to_csv(out/'development_cases.csv',index=False);labels.to_csv(out/'development_labels.csv',index=False)
    rows=[];manifest=[]
    with threadpool_limits(limits=4):
        for day,today in cases.groupby('day'):
            old=training_rows(development[development.day.ne(day)])
            new,_=expand(old,test)
            assert day not in set(new.day) and PRIMARY_DAY not in set(new.day)
            baseline=v2_hidden.fit(table(old))
            for unit in CAMERAS:
                models=fit_options(new,unit,schedule)
                manifest.append(dict(heldout_date=day,unit=unit,current_records=sorted(old.recording.unique()),
                    expanded_records=sorted(new.recording.unique()),rows={k:v['training_rows'] for k,v in models.items()}))
                for name,g in today[today.unit.eq(unit)].groupby('recording'):
                    raw=raw_inputs(feeds[name],g);reference,current,available=current_values(baseline[unit],raw,unit)
                    for spec,model in models.items():
                        fitted=fitted_values(model,spec,raw,unit,g.day.tolist(),schedule)
                        for weight in WEIGHTS:
                            value=np.maximum(0.,reference+weight*(fitted-reference));value=np.where(available>=2,value,current)
                            for r,ref,cur,pred,av in zip(g.itertuples(),reference,current,value,available):
                                rows.append(dict(case_id=r.case_id,day=day,recording=name,unit=unit,spec=spec,weight=weight,
                                    reference=ref,current=cur,candidate=pred,donors=int(av)))
    pred=pd.DataFrame(rows);pred.to_csv(out/'development_predictions.csv',index=False)
    scored=pred.merge(labels,on='case_id',validate='many_to_one')
    for col in ('current','candidate','reference'):scored[col+'_error']=(scored[col]-scored.truth).abs()
    by=scored.groupby(['unit','spec','weight','day','recording']).agg(targets=('case_id','size'),
        current_mae=('current_error','mean'),candidate_mae=('candidate_error','mean'),reference_mae=('reference_error','mean')).reset_index()
    by['relative_to_current']=by.candidate_mae/by.current_mae
    by['normalized']=by.candidate_mae/by.reference_mae
    by.to_csv(out/'development_by_recording.csv',index=False)
    objectives=[];selected={}
    for (unit,spec,weight),g in by.groupby(['unit','spec','weight']):
        fri=g[pd.to_datetime(g.day).dt.dayofweek.eq(4)];sat=g[pd.to_datetime(g.day).dt.dayofweek.eq(5)]
        objectives.append(dict(unit=unit,spec=spec,weight=weight,
            objective=float(.5*fri.relative_to_current.mean()+.5*sat.relative_to_current.mean())))
    objective=pd.DataFrame(objectives);objective.to_csv(out/'development_objectives.csv',index=False)
    for unit in CAMERAS:
        best=objective[objective.unit.eq(unit)].sort_values(['objective','spec','weight']).iloc[0]
        selected[unit]=dict(spec=str(best.spec) if best.objective<.995 else 'unchanged',
            weight=float(best.weight),development_objective=float(best.objective))
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    (out/'development_fitting_manifest.json').write_text(json.dumps(manifest,indent=2))
    assert protected==protected_snapshot()
    (out/'selection_verification.json').write_text(json.dumps(dict(runtime_seconds=time.perf_counter()-started,
        protected_files_unchanged=len(protected),source_sha256=sha(__file__),fixed_labels_used=False),indent=2))
    print(json.dumps(selected,indent=2));print('Selection seconds:',time.perf_counter()-started)


def primary_predictions(cases,feeds,old,new,selected,schedule):
    baseline=v2_hidden.fit(table(old));rows=[]
    for unit in CAMERAS:
        choice=selected[unit];spec=choice['spec']
        model=fit_options(new,unit,schedule,[spec])[spec] if spec!='unchanged' else None
        for name,g in cases[cases.unit.eq(unit)].groupby('recording'):
            raw=raw_inputs(feeds[name],g);reference,current,available=current_values(baseline[unit],raw,unit)
            if model:
                fitted=fitted_values(model,spec,raw,unit,g.day.tolist(),schedule)
                value=np.maximum(0.,reference+choice['weight']*(fitted-reference));value=np.where(available>=2,value,current)
            else:value=current
            poison=feeds[name].copy();poison['cam_B']=1e8;poison['cam_G']=1e8
            pd.testing.assert_frame_equal(raw,raw_inputs(poison,g))
            for r,ref,cur,pred,av in zip(g.itertuples(),reference,current,value,available):
                rows.append(dict(case_id=r.case_id,reference=ref,current=cur,candidate=pred,
                    model=int(av>=2),donors=int(av)))
    return cases.merge(pd.DataFrame(rows),on='case_id',validate='one_to_one')


def reports(predictions,labels,out):
    out.mkdir();predictions.to_csv(out/'predictions.csv',index=False)
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    for col in ('current','candidate','reference'):scored[col+'_error']=(scored[col]-scored.truth).abs()
    scored['gain']=scored.current_error-scored.candidate_error
    scored.to_csv(out/'scored.csv',index=False)
    for groups,name in [(['unit'],'camera'),(['window_id'],'window'),(['unit','window_id'],'camera_window')]:
        result=scored.groupby(groups).agg(targets=('case_id','size'),current_mae=('current_error','mean'),
            candidate_mae=('candidate_error','mean'),reference_mae=('reference_error','mean'),
            model=('model','sum'),gain=('gain','mean')).reset_index()
        result['current_normalized']=result.current_mae/result.reference_mae
        result['candidate_normalized']=result.candidate_mae/result.reference_mae
        result.to_csv(out/f'by_{name}.csv',index=False)
    summary=[]
    for unit,g in [('B+G',scored),*list(scored.groupby('unit'))]:
        windows=g.groupby('window_id').gain.agg(['mean','sum'])
        omitted=windows['sum'].nlargest(2).index
        kept=g[~g.window_id.isin(omitted)]
        loo=[float(g[g.window_id.ne(w)].gain.mean()) for w in windows.index]
        summary.append(dict(unit=unit,targets=len(g),windows=len(windows),
            current_mae=float(g.current_error.mean()),candidate_mae=float(g.candidate_error.mean()),
            reference_mae=float(g.reference_error.mean()),current_normalized=float(g.current_error.mean()/g.reference_error.mean()),
            candidate_normalized=float(g.candidate_error.mean()/g.reference_error.mean()),
            gain=float(g.gain.mean()),relative_gain=float(g.gain.mean()/g.current_error.mean()),
            winning_windows=int(windows['mean'].gt(0).sum()),losing_windows=int(windows['mean'].lt(0).sum()),
            gain_without_two_best=float(kept.gain.mean()),min_leave_one_window_gain=min(loo),max_leave_one_window_gain=max(loo),
            largest_window_regression=float(-windows['mean'].min())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    return {r['unit']:r for r in summary}


def evaluate(out,build=False):
    started=time.perf_counter();out=Path(out)
    if (out/'evaluation_verification.json').exists():raise FileExistsError('Existing evaluation is protected')
    selected=json.loads((out/'selected.json').read_text());selection_hash=sha(out/'selected.json')
    protected=protected_snapshot();train,test,schedule=inputs()
    with contextlib.redirect_stdout(io.StringIO()):cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
    assert case_fingerprints(cases,labels)==json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())['historical']
    cases=annotate_cases(cases,feeds);friday=cases[cases.kind.eq('hidden')&cases.unit.isin(CAMERAS)&cases.day.eq(PRIMARY_DAY)]
    base,expanded,_,blocked=release_training(train,test,feeds)
    strict=training_rows(train[train.day.ne(PRIMARY_DAY)]);strict_expanded,_=expand(strict,test)
    pending={}
    with threadpool_limits(limits=4):
        for protocol,old,new in [('strict',strict,strict_expanded),('release_matched',base,expanded)]:
            if protocol=='strict':assert PRIMARY_DAY not in set(old.day) and PRIMARY_DAY not in set(new.day)
            pending[protocol]=primary_predictions(friday,feeds,old,new,selected,schedule)
            pending[protocol].to_csv(out/f'{protocol}_before_scoring.csv',index=False)
    results={p:reports(pred,labels,out/p) for p,pred in pending.items()}
    supported=[]
    for unit in CAMERAS:
        a=results['strict'][unit];b=results['release_matched'][unit]
        if selected[unit]['spec']!='unchanged' and b['gain']>0 and a['relative_gain']>=-.02 and \
            (a['gain']>0 or b['relative_gain']>=.03) and b['gain_without_two_best']>0:
            supported.append(unit)
    actual=[];actual_feeds=table(test[test.id.isna()&test.people.notna()])
    targets=test[test.kind.eq('hidden')&test.unit.isin(CAMERAS)].copy()
    targets['minute']=targets.minute_of_day;targets['window_id']=targets.recording+':'+(targets.minute//30*30).astype(str)
    for unit,g in targets.groupby('unit'):
        raw=raw_inputs(actual_feeds[g.recording.iloc[0]],g)
        actual.append(dict(unit=unit,targets=len(g),min_donors=int(raw.notna().sum(axis=1).min()),
            mean_donors=float(raw.notna().sum(axis=1).mean()),model=int(raw.notna().sum(axis=1).ge(2).sum()),
            primary_E_available=int(raw.cam_E.notna().sum()),primary_D_available=int(raw.cam_D.notna().sum())))
    pd.DataFrame(actual).to_csv(out/'actual_coverage.csv',index=False)
    candidate=None
    if supported and build:
        cfg=training_rows(train);expanded_train,_=expand(cfg,test)
        with BASELINE.open(newline='') as handle:original=list(csv.DictReader(handle))
        official=pd.read_csv(ROOT/'data/raw/sample_submission.csv').set_index('id').people
        new_values={}
        with threadpool_limits(limits=4):
            for unit in supported:
                choice=selected[unit];spec=choice['spec'];model=fit_options(expanded_train,unit,schedule,[spec])[spec]
                g=targets[targets.unit.eq(unit)];raw=raw_inputs(actual_feeds[g.recording.iloc[0]],g)
                fitted=fitted_values(model,spec,raw,unit,g.day.tolist(),schedule)
                refs=official.loc[g.id].to_numpy();value=np.maximum(0.,refs+choice['weight']*(fitted-refs))
                valid=raw.notna().sum(axis=1).ge(2).to_numpy()
                new_values.update({key:str(float(v)) for key,v,ok in zip(g.id,value,valid) if ok})
        output=[dict(id=r['id'],people=new_values.get(r['id'],r['people'])) for r in original]
        assert len(output)==len({r['id'] for r in output})==4457
        assert [r['id'] for r in output]==pd.read_csv(ROOT/'data/raw/sample_submission.csv').id.tolist()
        assert all(np.isfinite(float(r['people'])) and float(r['people'])>=0 for r in output)
        changed=[r['id'] for r,o in zip(output,original) if r['people']!=o['people']]
        assert set(changed)<=set(targets[targets.unit.isin(supported)].id)
        assert all(r['people']==o['people'] for r,o in zip(output,original) if r['id'] not in new_values)
        path=out/'submission_friday_hidden_v1.csv'
        with path.open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(output)
        candidate=dict(path=str(path),sha256=sha(path),supported_cameras=supported,changed_rows=len(changed),changed_ids=changed,
            ids=4457,unchanged_outside_supported_friday_hidden=True,saturday_text_unchanged=True)
        (out/'candidate_verification.json').write_text(json.dumps(candidate,indent=2))
    assert protected==protected_snapshot();assert selection_hash==sha(out/'selected.json')
    info=dict(selected=selected,supported=supported,results=results,candidate=candidate,
        source_sha256=sha(__file__),selection_sha256=selection_hash,protected_files_unchanged=len(protected),
        runtime_seconds=time.perf_counter()-started,t05_fits=0,
        limitations='Only one short Friday development recording, no E; full Sep11 fixed date repeatedly examined; Saturday proxy validation is not independent Friday evidence.')
    (out/'evaluation_verification.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k!='candidate'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['select','evaluate'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--build-candidate',action='store_true')
    args=parser.parse_args()
    if args.stage=='select':select(args.output_dir)
    else:evaluate(args.output_dir,args.build_candidate)
