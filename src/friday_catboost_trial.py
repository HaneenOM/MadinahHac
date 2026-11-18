"""P42: actual CPU CatBoost training; isolated Friday HIDDEN experiment.

Select on date-held-out development, then evaluate frozen choices on Sep11.
No T05 fitting. HIDDEN permits surrounding released donor observations.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import importlib.metadata
import io
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from threadpoolctl import threadpool_limits
import friday_hidden_improvement as p40
import friday_ratio_trial as p41
from pipeline import table
import v2_hidden
from released_hidden_training_trial import training_rows,expand,release_training,sha,row_keys,BASELINE
from evaluate_baseline_task import fixed_benchmark
from reliable_evaluation import annotate_cases,case_fingerprints

ROOT=p40.ROOT
DONORS=p40.DONORS
SPECS={'counts_d4':dict(temporal=False,depth=4,l2_leaf_reg=10.),
       'context_d4':dict(temporal=True,depth=4,l2_leaf_reg=10.),
       'context_d5':dict(temporal=True,depth=5,l2_leaf_reg=20.)}
PARAMS=dict(loss_function='MAE',eval_metric='MAE',iterations=500,learning_rate=.05,
    random_seed=42,thread_count=4,border_count=64,random_strength=1.,
    bootstrap_type='Bernoulli',subsample=.8,nan_mode='Min',allow_writing_files=False,verbose=False)
WEIGHTS=(.5,1.)
INNER='2026-09-19'
P40_CANDIDATE=ROOT/'outputs/prompt40_friday_hidden/submission_friday_hidden_v1.csv'

def protected():
    result=p41.snapshot()
    for f in (ROOT/'outputs').rglob('submission*.csv'):result[str(f)]=sha(f)
    for f in ('src/friday_ratio_trial.py','reports/prompt41_friday_ratios_ar.md'):
        result[f]=sha(ROOT/f)
    return result

def provenance():
    paths=[ROOT/'data/raw'/n for n in ('train.csv','test.csv','recordings.csv','sample_submission.csv')]
    paths += [ROOT/'src'/n for n in ('friday_catboost_trial.py','friday_hidden_improvement.py',
        'friday_ratio_trial.py','released_hidden_training_trial.py','v2_hidden.py','v2_missing.py',
        'pipeline.py','evaluate_baseline_task.py','reliable_evaluation.py')]
    paths += [ROOT/'models/current_baseline/prayer_times_madinah.csv',ROOT/'models/local_evaluation/cases_manifest.json']
    return dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ('catboost','numpy','pandas','scipy','scikit-learn',
            'six','graphviz','plotly','matplotlib')})

def features(feeds,cases,schedule,temporal):
    """Exact central values and actual observed neighbours, independently per recording.

    Reindexing creates NaNs only. No interpolation/forward-fill of COUNT values.
    Before/after windows exclude the central minute. Ages are metadata only.
    """
    pieces=[]
    for rec,g in cases.groupby('recording',sort=False):
        feed=feeds[rec].reindex(columns=DONORS).copy()
        assert not set(feed.columns)&set(p40.CAMERAS)
        start=min(int(feed.index.min()),int(g.minute.min()))
        end=max(int(feed.index.max()),int(g.minute.max()))
        grid=feed.reindex(np.arange(start,end+1))
        raw=grid.reindex(g.minute.to_numpy()).reset_index(drop=True)
        result=pd.DataFrame(index=np.arange(len(g)))
        total=raw.sum(axis=1,min_count=1)
        for donor in DONORS:
            result[donor+'_count']=raw[donor]
            result[donor+'_log']=np.log1p(raw[donor].clip(lower=0))
            result[donor+'_share']=raw[donor]/total.where(total.ge(1.))
            result[donor+'_seen']=raw[donor].notna().astype(float)
            if temporal:
                series=grid[donor]
                past=series.shift(1).rolling(5,min_periods=1)
                future=series.iloc[::-1].shift(1).rolling(5,min_periods=1)
                columns={'before_mean':past.mean(),'after_mean':future.mean().iloc[::-1],
                    'before_n':past.count(),'after_n':future.count().iloc[::-1]}
                observed=pd.Series(np.where(series.notna(),grid.index,np.nan),index=grid.index)
                # Only timestamps are propagated, never people readings.
                columns['before_age']=pd.Series(grid.index,index=grid.index)-observed.shift(1).ffill()
                columns['after_age']=observed.shift(-1).bfill()-pd.Series(grid.index,index=grid.index)
                for suffix,values in columns.items():
                    if suffix.endswith('age'):values=values.where(values.le(5))
                    result[donor+'_'+suffix]=values.reindex(g.minute.to_numpy()).to_numpy()
        result['visible_total']=total
        result['available']=raw.notna().sum(axis=1)
        minute=g.minute.to_numpy(float)
        result['minute']=minute
        result['clock_sin']=np.sin(2*np.pi*minute/1440)
        result['clock_cos']=np.cos(2*np.pi*minute/1440)
        result['friday']=[float(pd.Timestamp(d).day_of_week==4) for d in g.day]
        result['dhuhr_offset']=[m-schedule[d][1] for m,d in zip(minute,g.day)]
        result['nearest_prayer']=[m-schedule[d][np.argmin(abs(schedule[d]-m))] for m,d in zip(minute,g.day)]
        result.index=g.case_id.to_numpy();pieces.append(result)
    return pd.concat(pieces).loc[cases.case_id].astype(float)

def dataset(training,schedule):
    """Fit rows and features derive only from explicitly supplied released table."""
    feeds=table(training);rows=[]
    for rec,g in training.groupby('recording'):
        available=feeds[rec].reindex(columns=DONORS).notna().sum(axis=1)
        for r in g[g.unit.isin(p40.CAMERAS)].itertuples():
            if available.get(r.minute_of_day,0)>=2:
                rows.append(dict(case_id=f'fit:{rec}:{r.unit}:{int(r.minute_of_day)}',recording=rec,day=r.day,
                    unit=r.unit,minute=int(r.minute_of_day),truth=float(r.people)))
    cases=pd.DataFrame(rows)
    # Targets are NEVER present in feed given to feature construction.
    safe={rec:t.reindex(columns=DONORS) for rec,t in feeds.items()}
    x=features(safe,cases,schedule,True)
    return cases,x

def columns(x,spec):
    if SPECS[spec]['temporal']:return list(x.columns)
    return [c for c in x if not any(s in c for s in ('_before_','_after_'))]

def augmentation(x,meta):
    dropped=x.copy();rng=np.random.default_rng(42)
    mask=rng.random((len(x),len(DONORS)))<.2
    for j,u in enumerate(DONORS):
        lost=mask[:,j]
        for c in x:
            if c.startswith(u+'_'):dropped.loc[lost,c]=0. if c.endswith('_seen') or c.endswith('_n') else np.nan
    total=dropped[[u+'_count' for u in DONORS]].sum(axis=1,min_count=1)
    dropped['visible_total']=total
    dropped['available']=dropped[[u+'_seen' for u in DONORS]].sum(axis=1)
    for u in DONORS:dropped[u+'_share']=dropped[u+'_count']/total.where(total.ge(1.))
    keep=dropped.available.ge(2).to_numpy()
    xx=pd.concat([x,dropped.iloc[np.flatnonzero(keep)]],ignore_index=True)
    y=meta.truth.to_numpy(float);w=np.where(pd.to_datetime(meta.day).dt.day_of_week.eq(4),3.,1.)
    return xx,np.concatenate([y,y[keep]]),np.concatenate([w,w[keep]])

def train_model(meta,x,unit,spec,iterations=500,validation=None):
    keep=meta.unit.eq(unit).to_numpy();m=meta.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    z=x.iloc[np.flatnonzero(keep)][columns(x,spec)].reset_index(drop=True)
    z,y,w=augmentation(z,m)
    params=dict(PARAMS,iterations=int(iterations),**{k:v for k,v in SPECS[spec].items() if k!='temporal'})
    model=CatBoostRegressor(**params)
    extra={}
    if validation is not None:
        vm,vx=validation;valid=vm.unit.eq(unit).to_numpy()
        extra=dict(eval_set=Pool(vx.iloc[np.flatnonzero(valid)][list(z.columns)],vm.loc[valid,'truth']),
            early_stopping_rounds=50,use_best_model=True)
    model.fit(Pool(z,y,weight=w),**extra)
    return model,dict(unit=unit,spec=spec,original_rows=len(m),augmented_rows=len(y),
        records=sorted(m.recording.unique()),dates=sorted(m.day.unique()),iterations=model.tree_count_,features=list(z.columns))

def baselines(old,new,schedule,cases,feeds):
    baseline=v2_hidden.fit(table(old));gm=p41.anchor_models(new,schedule);rows=[]
    for unit in p40.CAMERAS:
        for rec,g in cases[cases.unit.eq(unit)].groupby('recording'):
            raw=p40.raw_inputs(feeds[rec],g)
            ref,original,anchor,available=p41.values(baseline,gm,raw,unit,g.day.tolist(),schedule)
            for r,a,b,c in zip(g.itertuples(),ref,original,anchor):
                rows.append(dict(case_id=r.case_id,reference=a,original=b,p40=c))
    return cases.merge(pd.DataFrame(rows),on='case_id',validate='one_to_one')

def reports(predictions,labels,out):
    out.mkdir(exist_ok=False);predictions.to_csv(out/'predictions.csv',index=False)
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    for m in ('reference','original','p40','catboost'):scored[m+'_error']=abs(scored[m]-scored.truth)
    scored.to_csv(out/'scored.csv',index=False)
    for groups,name in [(['unit'],'camera'),(['day','recording','unit'],'recording'),(['unit','window_id'],'window')]:
        rows=[]
        for key,g in scored.groupby(groups):
            key=key if isinstance(key,tuple) else (key,);row=dict(zip(groups,key))
            row.update(targets=len(g),windows=g.window_id.nunique(),model=int(g.model.sum()),fallback=int((~g.model).sum()))
            for m in ('original','p40','catboost'):
                row[m+'_mae']=g[m+'_error'].mean();row[m+'_normalized']=g[m+'_error'].mean()/g.reference_error.mean()
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/f'by_{name}.csv',index=False)
    rows=[]
    for unit,g in scored.groupby('unit'):
        for against in ('original','p40'):
            gain=g[against+'_error']-g.catboost_error
            windows=pd.DataFrame(dict(window=g.window_id,gain=gain)).groupby('window').gain.agg(['sum','mean'])
            best=windows['sum'].nlargest(2).index
            rows.append(dict(unit=unit,against=against,gain=gain.mean(),relative_gain=gain.mean()/g[against+'_error'].mean(),
                improved_windows=int(windows['mean'].gt(0).sum()),worsened_windows=int(windows['mean'].lt(0).sum()),
                gain_without_two_best=gain[~g.window_id.isin(best)].mean(),best_two=list(best),
                largest_regression=-windows['mean'].min(),worst_window=windows['mean'].idxmin(),
                min_leave_one_gain=min(gain[g.window_id.ne(w)].mean() for w in windows.index)))
    pd.DataFrame(rows).to_csv(out/'sensitivity.csv',index=False)
    return scored

def select(out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False);before=protected()
    train,test,schedule=p40.inputs()
    design=dict(params=PARAMS,specs=SPECS,blends=WEIGHTS,friday_weight=3.,augmentation='One deterministic 20% entire-donor-channel dropout copy, >=2 central donors.',
        discovery='No prior CatBoost found. P26/P40 tested sklearn HGB, P41 normalized-target Ridge.',
        feature_protocol='Six donors ACDEFH only; exact central counts, log/count shares (denom>=1), seen flags, total, clock/Friday/prayer. Optional +/-5min observed-only means, sample counts and ages. No interpolation, cross-recording values, or B/G predictors.',
        folds='Sep11 excluded from ALL new development. Outer held dates Sep4/5/12 with P40 fixed 5min cases and sparse donor mask. Entire outer date excluded from all model fitting.',
        early_stop='Inner validation uses ONLY genuinely released Sep19 B/G labels. Entire Sep19 excluded from inner fitting. MAE early-stop50, cap500. Refit including released Sep19 with locked inner tree count before outer predictions.',
        final_iterations='Median inner tree count across three development folds of selected spec; fixed before primary evaluation.',
        objective='.5 relative MAE on Sep4 Friday + .5 mean relative MAE across 4 Saturday recordings, vs P40 (B unchanged). Min across 3 specs x2 blends per camera. >=.5% improvement supports development.',
        decision='Per-camera: development support; release gain vs P40 positive and survives removal of two best windows; strict relative regression <=2%; if strict gain nonpositive require release gain>=5%. No window/crowd exceptions or post-primary tuning.',
        strict='Entire Sep11 excluded. Other training dates including Sep12 are unmasked ordinary training, as historical benchmark. Add actual test released A-H.',
        release_matched='Replace Sep11/12 with fixed released feeds after ALL task masks. Add only genuinely released actual test A-H. No masked key in any fitting table.',
        metric='MAE / same-row reference MAE, project-specific, not confirmed Kaggle formula.')
    (out/'design.json').write_text(json.dumps(design,indent=2))
    (out/'provenance.json').write_text(json.dumps(provenance(),indent=2))
    dev=train[train.day.ne(p40.PRIMARY_DAY)].copy();cases,labels,feeds=p40.tuning_cases(dev,test)
    cases.to_csv(out/'development_cases.csv',index=False);labels.to_csv(out/'development_labels.csv',index=False)
    allx=features(feeds,cases,schedule,True);rows=[];manifests=[]
    with threadpool_limits(limits=4):
        for day,g in cases.groupby('day'):
            old=training_rows(dev[dev.day.ne(day)]);new,_=expand(old,test)
            assert not {day,p40.PRIMARY_DAY}&set(new.day)
            innertrain=new[new.day.ne(INNER)];inner=new[new.day.eq(INNER)]
            im,ix=dataset(innertrain,schedule);vm,vx=dataset(inner,schedule);fm,fx=dataset(new,schedule)
            baseline=baselines(old,new,schedule,g,feeds).set_index('case_id')
            for unit in p40.CAMERAS:
                target=g[g.unit.eq(unit)]
                for spec in SPECS:
                    probe,manifest=train_model(im,ix,unit,spec,validation=(vm,vx))
                    model,final=train_model(fm,fx,unit,spec,iterations=probe.tree_count_)
                    manifests.append(dict(outer_date=day,inner=manifest,refit=final,
                        inner_validation_records=sorted(vm.recording.unique())))
                    xx=allx.loc[target.case_id,final['features']]
                    fit=np.maximum(0.,model.predict(xx));available=xx.available.ge(2).to_numpy()
                    anchor=baseline.loc[target.case_id,'p40'].to_numpy()
                    for w in WEIGHTS:
                        pred=np.where(available,anchor+w*(fit-anchor),anchor)
                        for r,value,ok in zip(target.itertuples(),pred,available):
                            row=r._asdict();row.update(baseline.loc[r.case_id].to_dict())
                            row.update(spec=spec,weight=w,catboost=value,model=bool(ok),iterations=model.tree_count_);rows.append(row)
    pred=pd.DataFrame(rows);pred.to_csv(out/'development_predictions.csv',index=False)
    scored=pred.merge(labels,on='case_id',validate='many_to_one')
    for m in ('reference','original','p40','catboost'):scored[m+'_error']=abs(scored[m]-scored.truth)
    by=scored.groupby(['unit','spec','weight','day','recording']).agg(targets=('case_id','size'),model=('model','sum'),
        original_mae=('original_error','mean'),p40_mae=('p40_error','mean'),catboost_mae=('catboost_error','mean'),reference_mae=('reference_error','mean')).reset_index()
    for m in ('original','p40','catboost'):by[m+'_normalized']=by[m+'_mae']/by.reference_mae
    by.to_csv(out/'development_by_recording.csv',index=False)
    objectives=[]
    for (unit,spec,w),g in by.groupby(['unit','spec','weight']):
        fri=pd.to_datetime(g.day).dt.day_of_week.eq(4);relative=g.catboost_mae/g.p40_mae
        objectives.append(dict(unit=unit,spec=spec,weight=w,objective=.5*relative[fri].mean()+.5*relative[~fri].mean()))
    objectives=pd.DataFrame(objectives);objectives.to_csv(out/'development_objectives.csv',index=False)
    selected={}
    for unit in p40.CAMERAS:
        best=objectives[objectives.unit.eq(unit)].sort_values(['objective','spec','weight']).iloc[0]
        trees=[m['inner']['iterations'] for m in manifests if m['inner']['unit']==unit and m['inner']['spec']==best.spec]
        selected[unit]=dict(spec=best.spec,weight=float(best.weight),objective=float(best.objective),
            iterations=int(np.median(trees)),development_supported=bool(best.objective<=.995))
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    (out/'development_fitting_manifest.json').write_text(json.dumps(manifests,indent=2))
    assert before==protected()
    (out/'selection_verification.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,
        protected_files=len(before),source_sha256=sha(__file__),fixed_labels_used=False),indent=2))
    print(json.dumps(selected,indent=2));print('Selection seconds',time.perf_counter()-started)

def coverage(x,actual,fitx,unit,protocol):
    rows=[]
    for setting,z in [('fixed',x),('actual',actual)]:
        cols=[u+'_count' for u in DONORS]
        outside=pd.DataFrame({c:z[c].notna()&((z[c]<fitx[c].min())|(z[c]>fitx[c].max())) for c in cols})
        rows.append(dict(protocol=protocol,unit=unit,setting=setting,targets=len(z),model=int(z.available.ge(2).sum()),
            mean_donors=float(z.available.mean()),any_count_outside_fit_range=int(outside.any(axis=1).sum())))
        for col in z:
            rows.append(dict(protocol=protocol,unit=unit,setting=setting,feature=col,finite=int(z[col].notna().sum()),
                min=float(z[col].min()),max=float(z[col].max()),fit_min=float(fitx[col].min()),fit_max=float(fitx[col].max())))
    return rows

def build_candidate(out,supported,selected,train,test,schedule):
    full,_=expand(training_rows(train),test);meta,x=dataset(full,schedule)
    actual=test[test.kind.eq('hidden')&test.unit.isin(p40.CAMERAS)].copy()
    actual['minute']=actual.minute_of_day;actual['case_id']=actual.id
    feeds=table(test[test.id.isna()&test.people.notna()]);xx=features(feeds,actual,schedule,True)
    with BASELINE.open(newline='') as f:original=list(csv.DictReader(f))
    with P40_CANDIDATE.open(newline='') as f:parent=list(csv.DictReader(f))
    assert [r['id'] for r in original]==[r['id'] for r in parent]
    anchor={r['id']:float(r['people']) for r in parent};new_values={};manifest=[]
    for unit in supported:
        choice=selected[unit];model,info=train_model(meta,x,unit,choice['spec'],choice['iterations'])
        g=actual[actual.unit.eq(unit)];z=xx.loc[g.id,info['features']]
        fitted=np.maximum(0.,model.predict(z));a=np.array([anchor[key] for key in g.id])
        pred=np.where(z.available.ge(2),a+choice['weight']*(fitted-a),a)
        new_values.update({key:str(float(value)) for key,value in zip(g.id,pred)})
        model.save_model(str(out/f'production_candidate_{unit}.cbm'));manifest.append(info)
    output=[dict(id=r['id'],people=new_values.get(r['id'],r['people'])) for r in parent]
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    assert len(output)==len({r['id'] for r in output})==4457
    assert [r['id'] for r in output]==sample.id.tolist()
    assert all(np.isfinite(float(r['people'])) and float(r['people'])>=0 for r in output)
    allowed=set(actual[actual.unit.isin(set(supported)|{'cam_G'})].id)
    changed=[r['id'] for r,o in zip(output,original) if r['people']!=o['people']]
    assert set(changed)<=allowed
    assert all(r['people']==o['people'] for r,o in zip(output,original) if r['id'] not in allowed)
    assert all(r['people']==o['people'] for r,o in zip(output,parent) if r['id'] not in new_values)
    path=out/'submission_friday_catboost_v1.csv'
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','people']);writer.writeheader();writer.writerows(output)
    info=dict(path=str(path),sha256=sha(path),catboost_cameras=supported,
        retained_p40_G='cam_G' not in supported,changed_from_original=len(changed),changed_ids=changed,
        changed_from_p40=sum(r['people']!=o['people'] for r,o in zip(output,parent)),ids=4457,
        finite_nonnegative=True,exact_text_unchanged_outside_scope=True,production_fit=manifest)
    (out/'candidate_verification.json').write_text(json.dumps(info,indent=2))
    return info

def evaluate(out,build):
    started=time.perf_counter();before=protected();locked=sha(out/'selected.json')
    assert not (out/'evaluation_verification.json').exists()
    assert provenance()==json.loads((out/'provenance.json').read_text())
    selected=json.loads((out/'selected.json').read_text());train,test,schedule=p40.inputs()
    with contextlib.redirect_stdout(io.StringIO()):cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
    assert case_fingerprints(cases,labels)==json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())['historical']
    cases=annotate_cases(cases,feeds)
    friday=cases[cases.kind.eq('hidden')&cases.unit.isin(p40.CAMERAS)&cases.day.eq(p40.PRIMARY_DAY)]
    allx=features(feeds,friday,schedule,True)
    poisoned={r:t.assign(cam_B=1e10,cam_G=-1e10) for r,t in feeds.items()}
    pd.testing.assert_frame_equal(allx,features(poisoned,friday,schedule,True))
    base,new,_,blocked=release_training(train,test,feeds)
    strict=training_rows(train[train.day.ne(p40.PRIMARY_DAY)]);expanded,_=expand(strict,test)
    actual=test[test.kind.eq('hidden')&test.unit.isin(p40.CAMERAS)].copy()
    actual['minute']=actual.minute_of_day;actual['case_id']=actual.id
    actual_feeds=table(test[test.id.isna()&test.people.notna()]);actualx=features(actual_feeds,actual,schedule,True)
    manifests=[];cover=[]
    with threadpool_limits(limits=4):
        for protocol,old,newtraining in [('strict',strict,expanded),('release_matched',base,new)]:
            if protocol=='strict':assert p40.PRIMARY_DAY not in set(newtraining.day)
            else:assert not row_keys(newtraining)&blocked
            meta,x=dataset(newtraining,schedule);pred=baselines(old,newtraining,schedule,friday,feeds)
            pred=pred.set_index('case_id',drop=False)
            for unit in p40.CAMERAS:
                choice=selected[unit];model,info=train_model(meta,x,unit,choice['spec'],choice['iterations'])
                manifests.append(dict(protocol=protocol,**info));model.save_model(str(out/f'{protocol}_{unit}.cbm'))
                g=friday[friday.unit.eq(unit)];z=allx.loc[g.case_id,info['features']]
                fitted=np.maximum(0.,model.predict(z));anchor=pred.loc[g.case_id,'p40'].to_numpy()
                ok=z.available.ge(2).to_numpy();value=np.where(ok,anchor+choice['weight']*(fitted-anchor),anchor)
                pred.loc[g.case_id,'catboost']=value;pred.loc[g.case_id,'model']=ok
                cover+=coverage(z,actualx.loc[actual[actual.unit.eq(unit)].case_id,info['features']],
                    x.iloc[np.flatnonzero(meta.unit.eq(unit).to_numpy())][info['features']],unit,protocol)
            pred=pred.reset_index(drop=True);pred['model']=pred.model.astype(bool)
            previous=pd.read_csv(ROOT/f'outputs/prompt40_friday_hidden/{protocol}/predictions.csv')
            merged=pred.merge(previous[['case_id','current','candidate']],on='case_id',validate='one_to_one')
            np.testing.assert_allclose(merged.original,merged.current,rtol=0,atol=1e-12)
            np.testing.assert_allclose(merged.p40,merged.candidate,rtol=0,atol=1e-12)
            reports(pred,labels,out/protocol)
        supported=[]
        for unit in p40.CAMERAS:
            a=pd.read_csv(out/'strict/sensitivity.csv').query('unit==@unit and against=="p40"').iloc[0]
            b=pd.read_csv(out/'release_matched/sensitivity.csv').query('unit==@unit and against=="p40"').iloc[0]
            if selected[unit]['development_supported'] and b.gain>0 and b.gain_without_two_best>0 and a.relative_gain>=-.02 and (a.gain>0 or b.relative_gain>=.05):
                supported.append(unit)
        # Check all pre-existing artifacts BEFORE creating any new candidate.
        assert before==protected()
        candidate=build_candidate(out,supported,selected,train,test,schedule) if supported and build else None
    pd.DataFrame(cover).to_csv(out/'coverage_and_ranges.csv',index=False)
    (out/'evaluation_fitting_manifest.json').write_text(json.dumps(manifests,indent=2))
    after=protected();assert all(after[k]==v for k,v in before.items())
    assert locked==sha(out/'selected.json')
    info=dict(seconds=time.perf_counter()-started,protected_files=len(before),supported=supported,
        candidate=candidate,selected_sha256=locked,source_sha256=sha(__file__),
        hidden_poison_invariance=True,baseline_reproduced_atol=1e-12,t05_fits=0)
    (out/'evaluation_verification.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k!='candidate'},indent=2))
    for protocol in ('strict','release_matched'):
        print(protocol);print(pd.read_csv(out/protocol/'by_camera.csv').to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['select','evaluate'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--build-candidate',action='store_true')
    args=parser.parse_args()
    if args.stage=='select':select(args.output_dir)
    else:evaluate(args.output_dir,args.build_candidate)
