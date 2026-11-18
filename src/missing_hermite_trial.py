"""P46: select exterior Hermite settings on Sep4/5 only, then frozen benchmark.

No production edits, no future forecasts, no T05 fitting. Select/evaluate are
separate commands. Benchmark labels are not used by select.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import baseline_tasks as tasks
import missing_hermite as hermite
import v2_missing
from pipeline import load,table
from evaluate_baseline_task import fixed_benchmark
from reliable_evaluation import case_fingerprints
from released_hidden_training_trial import protected_snapshot,sha

ROOT=Path(__file__).resolve().parents[1]
UNITS=[f'cam_{u}' for u in 'ABCDEFGH']
RESERVED={'2026-09-11','2026-09-12'}
SETTINGS=[dict(context=c,weight=w) for c in (5,10) for w in (.5,1.)]
BASE=ROOT/'outputs/submission_hidden_bgd_v1.csv'
METHODS=('current','linear','proposed','candidate')

def snapshot():
    result=protected_snapshot()
    for folder in ('missing_targeted_trial','missing_shape_trial'):
        for p in (ROOT/'outputs'/folder).rglob('*'):
            if p.is_file():result[str(p)]=sha(p)
    for p in (ROOT/'outputs').rglob('submission*.csv'):result[str(p)]=sha(p)
    return result

def inputs():
    with contextlib.redirect_stdout(io.StringIO()):train,test,_,_=load(ROOT/'data/raw')
    return train,test

def fixed(train,test):
    with contextlib.redirect_stdout(io.StringIO()):cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
    assert case_fingerprints(cases,labels)==json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())['historical']
    return cases[cases.kind.eq('missing')].copy(),labels,feeds

def gaps(cases,feeds):
    found={}
    for r in cases.itertuples():
        ctx=v2_missing.gap_context(feeds[r.recording],r.unit,int(r.minute))
        assert ctx is not None
        start,end,_,_=ctx;key=f'{r.recording}:{r.unit}:{start}-{end}'
        if key not in found:found[key]=dict(gap_id=key,recording=r.recording,day=r.day,unit=r.unit,start=start,end=end,length=end-start-1,rows=[])
        found[key]['rows'].append(r)
    for g in found.values():g['complete_labels']=len(g['rows'])==g['length']
    return list(found.values())

def templates(cases,feeds):
    # Convert ALL counts to availability only before deriving any selection input.
    metadata={r:t.notna().astype(float).where(t.notna()) for r,t in feeds.items()}
    result=gaps(cases,metadata)
    for g in result:
        g['mask']=metadata[g['recording']].reindex(range(g['start']-10,g['end']+11),columns=UNITS).notna().to_numpy()
    return result

def synthetic(older,specs):
    # Nonoverlapping synthetic interiors on a fixed 30min grid, no label-error sampling.
    records=[]
    for rec,group in older.groupby('recording'):
        original=table(group)[rec];day=group.day.iloc[0];weekday=pd.Timestamp(day).day_of_week
        for unit in UNITS:
            matching=[g for g in specs if g['unit']==unit and pd.Timestamp(g['day']).day_of_week==weekday]
            if not matching or unit not in original:continue
            for number,start in enumerate(range(int(original.index.min())+10,int(original.index.max())-35,30)):
                template=matching[number%len(matching)];length=template['length'];end=start+length+1
                truth=original.reindex(range(start,end+1))[unit]
                if not np.isfinite(truth.to_numpy()).all():continue
                frame=original.reindex(range(start-10,end+11),columns=UNITS).copy()
                frame=frame.mask(~template['mask']);frame.loc[start+1:end-1,unit]=np.nan
                hidden=['cam_B','cam_G'] if weekday==4 else ['cam_D']
                frame=frame.drop(columns=hidden,errors='ignore')
                # Only metadata templates for actually visible MISSING cameras exist.
                if unit not in frame:continue
                records.append(dict(recording=rec,day=day,unit=unit,start=start,end=end,length=length,
                    gap_id=f'{rec}:{unit}:{start}-{end}',frame=frame,truth=truth,template=template['gap_id']))
    return records

def select(out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False);protected=snapshot()
    train,test=inputs();cases,unused_labels,feeds=fixed(train,test)
    del unused_labels  # Never inspected or scored during selection.
    design=dict(settings=SETTINGS,geometry='OLS endpoint slopes from same-camera released exterior +/-5 or10 actual minutes, endpoints included, >=2 observations per side; monotone tangent limiter norm<=3; no interpolation of context.',
        blend='Convex blend with current endpoint-preserving method, NOT a blend toward right endpoint. Explicit observed values at endpoints.',
        selection='Both Sep11/12 excluded entirely from calibration and parameter selection. Leave-DATE-out Sep4/5 synthetic complete contiguous gaps, grid30min; actual fixed metadata lengths/release masks only.',
        camera_support='>=6 synthetic gaps across >=2 recordings; target-weighted gain>0, equal-gap gain>=1%, positive gain after removing two best training gaps. No camera rules from fixed labels.',
        fixed_support='Only development-supported cameras can be adopted. Positive fixed gain and gain after removing two best camera-gaps; >=2 improved complete-label gaps; no date regression >max(.25 person,5% current date/camera MAE). No gap-specific exceptions.',
        transfer='E absent from older selection data: no adoption. I-M never adopted without separate camera-transfer validation.',
        diagnostics='Frozen proposed curve evaluated even for development-unsupported A-H cameras, labeled diagnostic-only. Candidate column keeps those unchanged.',
        fallback='Current method when no complete bounded context or fewer than2 genuine observations in either exterior context. No extrapolation beyond recording.',
        normalization='Project-specific MAE / same-case linear reference MAE, not official Kaggle formula.')
    (out/'design.json').write_text(json.dumps(design,indent=2))
    older=train[~train.day.isin(RESERVED)].copy();assert not set(older.day)&RESERVED
    samples=synthetic(older,templates(cases,feeds));rows=[]
    pd.DataFrame([{k:v for k,v in s.items() if k not in ('frame','truth')} for s in samples]).to_csv(out/'synthetic_manifest.csv',index=False)
    with threadpool_limits(limits=4):
        for day in sorted({s['day'] for s in samples}):
            fitting=older[older.day.ne(day)];assert day not in set(fitting.day)
            calibration=tasks.fit_missing(fitting,tasks.default_config()['missing'])
            for sample in [s for s in samples if s['day']==day]:
                t=sample['frame'];unit=sample['unit']
                for minute in range(sample['start']+1,sample['end']):
                    linear,current,_=tasks.missing_value(t,unit,minute,calibration,tasks.default_config()['missing'])
                    assert np.isfinite(current+linear)
                    for setting in SETTINGS:
                        value,meta=hermite.predict(t,unit,minute,current,setting)
                        rows.append(dict(gap_id=sample['gap_id'],recording=sample['recording'],day=day,unit=unit,
                            context=setting['context'],weight=setting['weight'],minute=minute,truth=float(sample['truth'].loc[minute]),
                            current=current,proposed=value,linear=linear,used=meta['used']))
    scored=pd.DataFrame(rows)
    for m in ('current','proposed','linear'):scored[m+'_error']=abs(scored[m]-scored.truth)
    scored.to_csv(out/'selection_scored.csv',index=False)
    by=scored.groupby(['unit','context','weight','recording','day','gap_id']).agg(targets=('truth','size'),
        current_mae=('current_error','mean'),proposed_mae=('proposed_error','mean'),linear_mae=('linear_error','mean'),used=('used','sum')).reset_index()
    by.to_csv(out/'selection_by_gap.csv',index=False)
    objectives=[]
    for (unit,context,weight),g in by.groupby(['unit','context','weight']):
        gain=(g.current_mae-g.proposed_mae);removed=g.assign(total_gain=gain*g.targets).nlargest(2,'total_gain').gap_id
        remain=g[~g.gap_id.isin(removed)]
        target_gain=float(np.average(gain,weights=g.targets));relative=float(g.proposed_mae.mean()/g.current_mae.mean())
        supported=len(g)>=6 and g.recording.nunique()>=2 and target_gain>0 and relative<=.99 and len(remain)>0 and np.average(remain.current_mae-remain.proposed_mae,weights=remain.targets)>0
        objectives.append(dict(unit=unit,context=int(context),weight=float(weight),gaps=len(g),recordings=g.recording.nunique(),
            objective=relative,target_gain=target_gain,equal_gap_gain=float(gain.mean()),supported=bool(supported)))
    objectives=pd.DataFrame(objectives);objectives.to_csv(out/'selection_objectives.csv',index=False)
    global_best=by.groupby(['context','weight'])[['current_mae','proposed_mae']].mean()
    global_best['ratio']=global_best.proposed_mae/global_best.current_mae
    c,w=global_best.ratio.idxmin();default=dict(context=int(c),weight=float(w))
    selected={}
    for unit in UNITS:
        group=objectives[objectives.unit.eq(unit)]
        if len(group):
            # Choose only once on training. Support is a requirement, not a second fixed-label search.
            best=group.sort_values(['objective','context','weight']).iloc[0]
            selected[unit]=dict(context=int(best.context),weight=float(best.weight),supported=bool(best.supported),
                objective=float(best.objective),gaps=int(best.gaps),recordings=int(best.recordings))
        else:selected[unit]=dict(**default,supported=False,gaps=0,recordings=0)
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    (out/'selection_verification.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,source_sha256=sha(__file__),
        algorithm_sha256=sha(ROOT/'src/missing_hermite.py'),dates=sorted(older.day.unique()),fixed_labels_used=False,
        synthetic_gaps=len(samples),training_calibration='Each whole validation date excluded; both fixed dates always excluded.'),indent=2))
    assert snapshot()==protected;print(json.dumps(selected,indent=2))

def aggregate(g):
    result=dict(targets=len(g),gaps=g.gap_id.nunique(),geometry_eligible=int(g.used.sum()),changed=int(g.changed.sum()))
    per=g.groupby('gap_id')[[m+'_error' for m in METHODS]].mean()
    for m in METHODS:
        result[m+'_mae']=g[m+'_error'].mean();result[m+'_normalized']=g[m+'_error'].mean()/g.linear_error.mean()
        result['equal_gap_'+m+'_mae']=per[m+'_error'].mean()
        result['equal_gap_'+m+'_normalized']=(per[m+'_error']/per.linear_error.replace(0,np.nan)).mean()
    return result

def report(pred,labels,out):
    pred.to_csv(out/'predictions.csv',index=False)
    scored=pred.merge(labels,on='case_id',validate='one_to_one')
    for m in METHODS:scored[m+'_error']=abs(scored[m]-scored.truth)
    scored['changed']=abs(scored.candidate-scored.current)>1e-9;scored.to_csv(out/'scored.csv',index=False)
    pooled=[]
    for name,g in [('all',scored),('complete',scored[scored.complete_labels]),('partial',scored[~scored.complete_labels])]:
        pooled.append(dict(subset=name,**aggregate(g)))
        for keys,suffix in [(['day'],'date'),(['unit'],'camera'),(['day','unit'],'date_camera'),(['length'],'length'),(['gap_id'],'gap')]:
            result=[]
            for key,q in g.groupby(keys):
                key=key if isinstance(key,tuple) else (key,);result.append(dict(zip(keys,key),**aggregate(q)))
            pd.DataFrame(result).to_csv(out/f'{name}_by_{suffix}.csv',index=False)
    pd.DataFrame(pooled).to_csv(out/'pooled.csv',index=False)
    rows=[]
    for method in ('proposed','candidate'):
        for scope,g in [('all',scored)]+[(u,g) for u,g in scored.groupby('unit')]+[(d,g) for d,g in scored.groupby('day')]:
            gain=g.current_error-g[method+'_error'];by=pd.DataFrame(dict(gap=g.gap_id,gain=gain)).groupby('gap').gain.agg(['sum','mean'])
            best=by['sum'].nlargest(2).index
            rows.append(dict(method=method,scope=scope,targets=len(g),gaps=len(by),gain=gain.mean(),
                improved_gaps=int(by['sum'].gt(1e-9).sum()),worsened_gaps=int(by['sum'].lt(-1e-9).sum()),
                unchanged_gaps=int(by['sum'].abs().le(1e-9).sum()),
                gain_without_two_best=float(gain[~g.gap_id.isin(best)].mean()),best_two=list(best),
                largest_regression_mae=float(-by['mean'].min()),worst_mae_gap=by['mean'].idxmin(),
                largest_total_regression=float(-by['sum'].min()),worst_total_gap=by['sum'].idxmin()))
    pd.DataFrame(rows).to_csv(out/'sensitivity.csv',index=False)
    return scored

def actual_predictions(train,test,selected,supported):
    calibration=tasks.fit_missing(train,tasks.default_config()['missing']);feeds=table(test[test.id.isna()&test.people.notna()])
    rows=[]
    for r in test[test.kind.eq('missing')].itertuples():
        t=feeds[r.recording];ctx=v2_missing.gap_context(t,r.unit,int(r.minute_of_day))
        linear,current,_=tasks.missing_value(t,r.unit,int(r.minute_of_day),calibration,tasks.default_config()['missing'])
        setting=selected.get(r.unit,dict(context=5,weight=.5))
        shape=hermite.geometry(t,r.unit,int(r.minute_of_day),setting['context'])
        geometric=shape is not None and 'observed' not in shape
        eligible=geometric and r.unit in supported
        value=hermite.predict(t,r.unit,int(r.minute_of_day),current,setting)[0] if eligible else current
        rows.append(dict(id=r.id,day=r.day,recording=r.recording,unit=r.unit,group='A-H' if r.unit in UNITS else 'I-M',
            length=ctx[1]-ctx[0]-1 if ctx else np.nan,geometry=geometric,adopted=eligible,prediction=value))
    return pd.DataFrame(rows)

def evaluate(out,build):
    started=time.perf_counter();protected=snapshot();locked=sha(out/'selected.json')
    assert not (out/'evaluation_verification.json').exists()
    verification=json.loads((out/'selection_verification.json').read_text())
    assert verification['source_sha256']==sha(__file__) and verification['algorithm_sha256']==sha(ROOT/'src/missing_hermite.py')
    selected=json.loads((out/'selected.json').read_text());train,test=inputs();cases,labels,feeds=fixed(train,test)
    specifications=gaps(cases,feeds);assert len(cases)==358 and len(specifications)==38
    assert sum(g['complete_labels'] for g in specifications)==32
    rows=[];endpoint_checks=0
    with threadpool_limits(limits=4):
        for day,today in cases.groupby('day'):
            fitting=train[train.day.ne(day)];assert day not in set(fitting.day)
            calibration=tasks.fit_missing(fitting,tasks.default_config()['missing'])
            for gap in [g for g in specifications if g['day']==day]:
                frame=feeds[gap['recording']];setting=selected[gap['unit']]
                for endpoint in (gap['start'],gap['end']):
                    observed=float(frame.loc[endpoint,gap['unit']])
                    value,_=hermite.predict(frame,gap['unit'],endpoint,0.,setting)
                    assert value==observed;endpoint_checks+=1
                for r in gap['rows']:
                    linear,current,_=tasks.missing_value(frame,r.unit,int(r.minute),calibration,tasks.default_config()['missing'])
                    proposed,meta=hermite.predict(frame,r.unit,int(r.minute),current,setting)
                    candidate=proposed if setting['supported'] else current
                    poison=frame.copy();poison.loc[gap['start']+1:gap['end']-1,r.unit]=1e9
                    # Availability is an external mask: restore ALL unavailable cells before prediction.
                    poison=poison.where(frame.notna())
                    check,_=hermite.predict(poison,r.unit,int(r.minute),current,setting)
                    assert proposed==check
                    rows.append(dict(case_id=r.case_id,day=day,recording=r.recording,unit=r.unit,minute=r.minute,
                        gap_id=gap['gap_id'],length=gap['length'],complete_labels=gap['complete_labels'],
                        current=current,linear=linear,proposed=proposed,candidate=candidate,used=meta['used'],
                        development_supported=setting['supported'],limited=meta.get('limited',False)))
    scored=report(pd.DataFrame(rows),labels,out)
    # Current implementation must still reproduce the accepted historical missing benchmark.
    previous=pd.read_csv(ROOT/'outputs/missing_shape_trial/scored.csv')[['case_id','current','linear']]
    check=scored.merge(previous,on='case_id',suffixes=('','_old'),validate='one_to_one')
    np.testing.assert_allclose(check.current,check.current_old,atol=1e-9,rtol=0)
    np.testing.assert_allclose(check.linear,check.linear_old,atol=1e-9,rtol=0)
    supported=[]
    for unit,g in scored.groupby('unit'):
        by=g.groupby('gap_id').apply(lambda q:float((q.current_error-q.candidate_error).sum()),include_groups=False)
        keep=g[~g.gap_id.isin(by.nlargest(2).index)]
        improved_complete=g[g.complete_labels].groupby('gap_id').apply(lambda q:(q.current_error-q.candidate_error).sum(),include_groups=False).gt(1e-9).sum()
        date_ok=all(q.candidate_error.mean()-q.current_error.mean()<=max(.25,.05*q.current_error.mean()) for _,q in g.groupby('day'))
        if selected[unit]['supported'] and (g.current_error-g.candidate_error).mean()>0 and len(keep)>0 and \
                (keep.current_error-keep.candidate_error).mean()>0 and improved_complete>=2 and date_ok:supported.append(unit)
    actual=actual_predictions(train,test,selected,supported);actual.to_csv(out/'actual_coverage.csv',index=False)
    actual.groupby(['day','group','unit']).agg(targets=('id','size'),geometry=('geometry','sum'),adopted=('adopted','sum')).reset_index().to_csv(out/'actual_coverage_summary.csv',index=False)
    candidate_info=None
    if supported and build:
        assert sha(BASE)=='342af77dd40e4120b589485ecf02bf25bf3434650e81473692328903bcf67a4a'
        with BASE.open(newline='') as f:original=list(csv.DictReader(f))
        updates={r.id:str(float(r.prediction)) for r in actual[actual.adopted].itertuples()}
        output=[dict(id=r['id'],people=updates.get(r['id'],r['people'])) for r in original]
        sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
        assert len(output)==len({r['id'] for r in output})==4457 and [r['id'] for r in output]==sample.id.tolist()
        assert all(np.isfinite(float(r['people'])) and float(r['people'])>=0 for r in output)
        assert all(r['people']==o['people'] for r,o in zip(output,original) if r['id'] not in updates)
        path=out/'submission_missing_hermite_v1.csv'
        with path.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','people']);writer.writeheader();writer.writerows(output)
        candidate_info=dict(path=str(path),sha256=sha(path),changed_ids=[r['id'] for r,o in zip(output,original) if r['people']!=o['people']],
            updated_rows=len(updates),supported_cameras=supported,unchanged_outside_scope=True)
        (out/'candidate_verification.json').write_text(json.dumps(candidate_info,indent=2))
    after=snapshot();assert all(after[k]==v for k,v in protected.items()) and locked==sha(out/'selected.json')
    info=dict(seconds=time.perf_counter()-started,targets=len(scored),gaps=len(specifications),complete_gaps=32,partial_gaps=6,
        endpoint_checks=endpoint_checks,unavailable_poison_checks=len(scored),protected_files=len(protected),supported_cameras=supported,
        candidate=candidate_info,source_sha256=sha(__file__),selected_sha256=locked,t05_fits=0,
        limitation='Repeatedly examined Sep11/12, no fresh validation. I-M unchanged; old selection lacks E.')
    (out/'evaluation_verification.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k!='candidate'},indent=2))
    print(pd.read_csv(out/'all_by_date.csv').to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['select','evaluate'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--build-candidate',action='store_true')
    args=parser.parse_args()
    if args.stage=='select':select(args.output_dir)
    else:evaluate(args.output_dir,args.build_candidate)
