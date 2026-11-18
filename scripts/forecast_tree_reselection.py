"""P50: audit P49 stopping, select on template-covered released historical rows.

audit/train use P49's isolated environment; prepare/score use the project env.
No T05 fits, no changed features/targets, no production/submission writes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/prompt49_forecast_trees'
OUT=ROOT/'outputs/prompt50_forecast_reselection'
sys.path.insert(0,str(ROOT/'scripts'))
import forecast_tree_trial as prior
FAMILIES=prior.FAMILIES
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def tracked_fit(family,data):
    """Same P49 configuration; record both losses and distinguish stored/used trees."""
    import numpy as np
    model=prior.estimator(family,400,True)
    tx,ty,vx,vy=(data[k] for k in ('train_x','train_y','valid_x','valid_y'))
    history={};sets=[(tx,ty),(vx,vy)]
    if family=='catboost':
        model.fit(tx,ty,eval_set=sets[1],early_stopping_rounds=40,use_best_model=True)
        h=model.get_evals_result();learn=h['learn']['MAE'];valid=h['validation']['MAE']
        best_index=int(model.get_best_iteration());used=int(model.tree_count_);stored=used
    elif family=='lightgbm':
        import lightgbm as lgb
        model.fit(tx,ty,eval_set=sets,
            callbacks=[lgb.record_evaluation(history),lgb.early_stopping(40,verbose=False)])
        learn=history['training']['l1'];valid=history['valid_1']['l1']
        used=int(model.best_iteration_);best_index=used-1;stored=model.booster_.num_trees()
    else:
        model.fit(tx,ty,eval_set=sets,verbose=False)
        h=model.evals_result();learn=h['validation_0']['mae'];valid=h['validation_1']['mae']
        best_index=int(model.best_iteration);used=best_index+1;stored=model.get_booster().num_boosted_rounds()
    assert used==best_index+1 and len(learn)==len(valid)
    # XGB keeps patience trees but its sklearn predictor automatically uses best_iteration.
    preds=model.predict(data['valid_x'])
    if family=='xgboost':
        explicit=model.predict(data['valid_x'],iteration_range=(0,used))
        np.testing.assert_allclose(preds,explicit,rtol=0,atol=0)
    rows=[dict(round=i+1,training_mae=float(a),validation_mae=float(b)) for i,(a,b) in enumerate(zip(learn,valid))]
    return model,rows,dict(configured_maximum=400,evaluated_rounds=len(valid),best_index_zero_based=best_index,
        prediction_rounds=used,stored_trees=stored,first_validation_mae=float(valid[0]),
        best_validation_mae=float(valid[best_index]),last_validation_mae=float(valid[-1]))

def audit():
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor
    import lightgbm as lgb
    from xgboost import XGBRegressor
    if (OUT/'audit.json').exists():raise FileExistsError('Audit already completed')
    start=time.perf_counter();p=json.loads((OLD/'prepared.json').read_text());records=[];counts=[]
    assert sha(ROOT/'configs/prompt49_forecast_trees.json')==p['config_sha256']
    old_scores=pd.read_csv(OLD/'development_scores.csv').set_index(['family','fold'])
    for family in FAMILIES:
        for fold in [f for f in p['folds'] if f['role']=='development']:
            assert sha(OLD/(fold['fold']+'.npz'))==fold['npz_sha256']
            data=np.load(OLD/(fold['fold']+'.npz'))
            model,curve,info=tracked_fit(family,data)
            assert info['prediction_rounds']==old_scores.loc[(family,fold['fold']),'best_round']
            # Same unclipped residual objective and same clipped count evaluation as P49.
            values=np.maximum(data['valid_anchor']+model.predict(data['valid_x']),0)
            actual_mae=float(np.abs(values-data['valid_anchor']-data['valid_y']).mean())
            np.testing.assert_allclose(actual_mae,old_scores.loc[(family,fold['fold']),'mae'],rtol=1e-12,atol=1e-10)
            pd.DataFrame(curve).to_csv(OUT/f'audit_{fold["fold"]}_{family}_curve.csv',index=False)
            records.append(dict(family=family,fold=fold['fold'],reproduced_mae=actual_mae,**info))
        for f in p['folds']:
            if f['role']!='assessment':continue
            path=OLD/'models'/f"{f['fold']}_{family}"
            if family=='catboost':model=CatBoostRegressor().load_model(str(path.with_suffix('.cbm')));n=model.tree_count_
            elif family=='lightgbm':model=lgb.Booster(model_file=str(path.with_suffix('.txt')));n=model.num_trees()
            else:model=XGBRegressor();model.load_model(path.with_suffix('.json'));n=model.get_booster().num_boosted_rounds()
            counts.append(dict(family=family,fold=f['fold'],stored_trees=int(n)))
    dump(OUT/'audit.json',dict(development=records,assessment_models=counts,runtime_seconds=time.perf_counter()-start,
        original_curves='Not saved by P49. These curves are deterministic replays from hash-verified original matrices.',
        classification='A+B: legitimate optima on mismatched validation, no detected iteration-index/configuration error',
        original_config_sha256=sha(ROOT/'configs/prompt49_forecast_trees.json')))
    print(pd.DataFrame(records).to_string(index=False))

def prepare():
    import numpy as np
    import pandas as pd
    sys.path[:0]=[str(ROOT),str(ROOT/'src')]
    import forecast_tree_features as f
    from t05_forecast.core import load_recordings,RELEASED,ABSENT
    from pipeline import load
    from evaluate_baseline_task import fixed_benchmark
    import reliable_evaluation as ev
    if (OUT/'prepared.json').exists():raise FileExistsError('Selection already frozen')
    start=time.perf_counter();p=json.loads((OLD/'prepared.json').read_text())
    assert sha(ROOT/'src/forecast_tree_features.py')==p['feature_source_sha256']
    assert sha(ROOT/'configs/prompt49_forecast_trees.json')==p['config_sha256']
    for name,h in p['input_hashes'].items():assert sha(ROOT/'data/raw'/name)==h
    for fold in p['folds']:assert sha(OLD/(fold['fold']+'.npz'))==fold['npz_sha256']
    protected={str(q.relative_to(ROOT)):sha(q) for q in list((ROOT/'outputs').rglob('submission*.csv'))+
        list((ROOT/'models').rglob('*.*'))+list((ROOT/'src').glob('*.py'))+list(OLD.rglob('*.*'))
        if q.is_file() and 'env' not in q.relative_to(ROOT).parts}
    protected['notebooks/current_baseline.ipynb']=sha(ROOT/'notebooks/current_baseline.ipynb')
    for name in ['scripts/forecast_tree_trial.py','configs/prompt49_forecast_trees.json']:
        protected[name]=sha(ROOT/name)
    dump(OUT/'protected.json',protected)
    train,test,_,_=load(ROOT/'data/raw');recs=load_recordings(ROOT/'data/raw')
    cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
    known=json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())
    assert ev.case_fingerprints(cases,labels)==known['historical']
    # Selection sees only historical RELEASED observations; every fixed answer is erased.
    selection={n:r.copy() for n,r in recs.items() if r.split=='train'}
    for n,feed in feeds.items():
        r=selection[n];r.y[:]=np.nan;r.status[:]=ABSENT;r.truth=None
        for u in feed:
            if u not in r.units:continue
            s=feed[u].dropna();ix=s.index.to_numpy(int)-r.first
            r.y[ix,r.col(u)]=s.to_numpy();r.status[ix,r.col(u)]=RELEASED
    assessment_keys={(r.recording,r.unit,int(r.minute)) for r in cases.itertuples()}
    for name,r in selection.items():
        for n,u,t in assessment_keys:
            if n==name and u in r.units:assert not np.isfinite(r.y[t-r.first,r.col(u)])
    design=dict(base_config=prior.cfg(),features='Unchanged P49 97 features',target=prior.cfg()['target'],
        selection_dates=['2026-09-11','2026-09-12'],source_dates=['2026-09-04','2026-09-05','2026-09-11','2026-09-12'],
        mask='All fixed historical task answers removed from all selection sources. Also mask every selection-validation target from its context for the full recording.',
        split='Two leave-date-out selection folds; training may use other released historical date. Validate only fully available template at last/now/target.',
        aggregation='Median best fitted rounds, floor as P49; max400/patience40; no configuration search',
        assessment='18/19 excluded from ALL selection. Historical fixed target labels disjoint, but their dates reused for selection; historical corrected scores are NOT fully date-independent.',
        fallback='Unchanged T05; no I-M transfer',
        independence='No fully matched independent weekly-covered recording split exists; selection covers only prior-week-covered hours, not full afternoon/evening.')
    dump(OUT/'design.json',design)
    folds=[];audits=[];checks=0;shifts=[]
    for day in ('2026-09-11','2026-09-12'):
        fold='selection_'+day
        pool={n:r for n,r in selection.items() if r.day!=day}
        val={n:r for n,r in selection.items() if r.day==day}
        train_cases=f.training_cases(pool);val_cases=f.training_cases(val)
        assert not any((r.recording,r.unit,int(r.minute)) in assessment_keys for r in val_cases.itertuples())
        # Erase validation target union from all context, even if earlier than another now.
        val_inputs={n:r.copy() for n,r in val.items()}
        for r in val_cases.itertuples():
            s=val_inputs[r.recording];i=int(r.minute)-s.first;j=s.col(r.unit)
            s.y[i,j]=np.nan;s.status[i,j]=ABSENT
        train_inputs={n:f.masked_training_record(r,test) for n,r in pool.items()}
        tm={}
        for n,r in pool.items():
            tm[n],audit=f.weekly(r,pool,[day]);audit['fold']=fold;audits.append(audit)
        x,c=f.feature_rows(train_cases,train_inputs,tm,check=True);checks+=c
        vm={}
        for n,r in val.items():
            vm[n],audit=f.weekly(r,pool,[day]);audit['fold']=fold;audits.append(audit)
            poison=r.copy();poison.y[:]=1e9;different,_=f.weekly(poison,pool,[day])
            np.testing.assert_allclose(vm[n],different,rtol=0,atol=0,equal_nan=True)
        v,c=f.feature_rows(val_cases,val_inputs,vm,check=True);checks+=c
        before=len(v)
        v=v[v.template_available.eq(1)&v.template_now_log.notna()&v.template_last_log.notna()].copy()
        assert len(v)>30 and x.template_available.sum()>30
        assert f.feature_columns(x)==p['features']==f.feature_columns(v)
        y=x[['case_id','anchor']].merge(train_cases[['case_id','truth']],on='case_id',validate='one_to_one')
        vy=v[['case_id','anchor']].merge(val_cases[['case_id','truth']],on='case_id',validate='one_to_one')
        assert y.case_id.tolist()==x.case_id.tolist() and vy.case_id.tolist()==v.case_id.tolist()
        assert not set(x.day).intersection(set(v.day)) and not set(x.case_id)&set(v.case_id)
        np.savez_compressed(OUT/f'{fold}.npz',train_x=x[p['features']].to_numpy(),train_y=(y.truth-y.anchor).to_numpy(),
            valid_x=v[p['features']].to_numpy(),valid_y=(vy.truth-vy.anchor).to_numpy(),valid_anchor=v.anchor.to_numpy())
        x.to_csv(OUT/f'{fold}_training_features.csv',index=False);v.to_csv(OUT/f'{fold}_validation_features.csv',index=False)
        folds.append(dict(fold=fold,training_rows=len(x),training_template_rows=int(x.template_available.sum()),
            validation_eligible_before_template_filter=before,validation_rows=len(v),validation_template_rows=len(v),
            training_recordings=sorted(pool),validation_recordings=sorted(val),
            validation_windows=v[['recording','now']].drop_duplicates().shape[0],npz_sha256=sha(OUT/f'{fold}.npz')))
        for role,g in [('training',x),('selection',v)]:
            for unit,h in g.groupby('unit'):
                shifts.append(dict(dataset=fold,role=role,unit=unit,rows=len(h),template_fraction=h.template_available.mean(),
                    median_recent=h.own_n10.median(),median_donors=h.available_other_cameras.median(),
                    count_min=float(np.expm1(h.own_level.min())),count_median=float(np.expm1(h.own_level.median())),count_max=float(np.expm1(h.own_level.max()))))
    for name in ['dev_2026-09-04','dev_2026-09-05','representative_2026-09-18','representative_2026-09-19','actual']:
        g=pd.read_csv(OLD/('actual_features.csv' if name=='actual' else name+'_features.csv'))
        for unit,h in g.groupby('unit'):
            shifts.append(dict(dataset=name,role='P49_or_actual',unit=unit,rows=len(h),template_fraction=h.template_available.mean(),
                median_recent=h.own_n10.median(),median_donors=h.available_other_cameras.median(),count_min=float(np.expm1(h.own_level.min())),
                count_median=float(np.expm1(h.own_level.median())),count_max=float(np.expm1(h.own_level.max()))))
    pd.DataFrame(shifts).to_csv(OUT/'availability_shifts.csv',index=False)
    dump(OUT/'template_audit.json',audits)
    dump(OUT/'prepared.json',dict(folds=folds,checks=checks,features=p['features'],
        feature_source_sha256=sha(ROOT/'src/forecast_tree_features.py'),base_config_sha256=sha(ROOT/'configs/prompt49_forecast_trees.json'),
        design_sha256=sha(OUT/'design.json'),preparation_seconds=time.perf_counter()-start,
        historical_assessment_labels_used=False,representative_dates_used=False))
    print(json.dumps(folds,indent=2))

def train():
    import numpy as np
    import pandas as pd
    if (OUT/'training_manifest.json').exists():raise FileExistsError('Finished comparison protected')
    start=time.perf_counter();p=json.loads((OUT/'prepared.json').read_text());old=json.loads((OLD/'prepared.json').read_text())
    assert sha(ROOT/'src/forecast_tree_features.py')==p['feature_source_sha256']
    assert sha(ROOT/'configs/prompt49_forecast_trees.json')==p['base_config_sha256']
    assert sha(OUT/'design.json')==p['design_sha256']
    protected=json.loads((OUT/'protected.json').read_text())
    assert all(sha(ROOT/path)==h for path,h in protected.items())
    records=[];rounds={}
    for family in FAMILIES:
        best=[]
        for fold in p['folds']:
            assert sha(OUT/(fold['fold']+'.npz'))==fold['npz_sha256']
            data=np.load(OUT/(fold['fold']+'.npz'));t=time.perf_counter()
            model,curve,info=tracked_fit(family,data);elapsed=time.perf_counter()-t
            pd.DataFrame(curve).to_csv(OUT/f"{fold['fold']}_{family}_curve.csv",index=False)
            best.append(info['prediction_rounds']);records.append(dict(family=family,fold=fold['fold'],training_seconds=elapsed,**info))
        rounds[family]=max(1,int(np.median(best)))
    dump(OUT/'selected_rounds.json',rounds);dump(OUT/'selection_results.json',records)
    # Nothing below reads assessment labels: exactly reuse P49 fit/feature arrays.
    (OUT/'models').mkdir(exist_ok=True);timings=[]
    for fold in old['folds']:
        if fold['role']!='assessment':continue
        assert sha(OLD/(fold['fold']+'.npz'))==fold['npz_sha256']
        data=np.load(OLD/(fold['fold']+'.npz'));features=pd.read_csv(OLD/(fold['fold']+'_features.csv'))
        predictions=features[['case_id']].copy()
        for family in FAMILIES:
            model=prior.estimator(family,rounds[family]);t=time.perf_counter()
            prior.fit(model,family,data,False);fit_time=time.perf_counter()-t;t=time.perf_counter()
            value=np.maximum(data['valid_anchor']+model.predict(data['valid_x']),0);infer_time=time.perf_counter()-t
            assert np.isfinite(value).all() and (value>=0).all()
            if family=='catboost':stored=int(model.tree_count_)
            elif family=='lightgbm':stored=int(model.booster_.num_trees())
            else:stored=int(model.get_booster().num_boosted_rounds())
            assert stored==rounds[family]
            for i in sorted({0,len(value)//2,len(value)-1}):
                one=np.maximum(data['valid_anchor'][i]+model.predict(data['valid_x'][i:i+1])[0],0)
                np.testing.assert_allclose(one,value[i],rtol=1e-12,atol=1e-9)
            predictions[family]=value;prior.save_model(model,family,OUT/'models'/f"{fold['fold']}_{family}")
            timings.append(dict(fold=fold['fold'],family=family,fitted_trees=stored,training_seconds=fit_time,inference_seconds=infer_time))
        predictions.to_csv(OUT/(fold['fold']+'_predictions.csv'),index=False)
    pd.DataFrame(timings).to_csv(OUT/'runtimes.csv',index=False)
    dump(OUT/'training_manifest.json',dict(rounds=rounds,runtime_seconds=time.perf_counter()-start,assessment_labels_read=False,
        config_changed=False,features_changed=False,outer_examples_changed=False,t05_fits=0))
    print(json.dumps(dict(rounds=rounds,runtime_seconds=time.perf_counter()-start),indent=2))

def score():
    import numpy as np
    import pandas as pd
    summaries=[];sensitivity=[]
    for setting in ('representative','historical'):
        c=pd.read_csv(OLD/f'{setting}_scored.csv')
        for family in FAMILIES:c['p49_'+family]=c[family]
        for day,g in c.groupby('day'):
            p=pd.read_csv(OUT/f'{setting}_{day}_predictions.csv').set_index('case_id')
            for family in FAMILIES:
                values=g.case_id.map(p[family]);c.loc[g.index,family]=values.fillna(g.t05).to_numpy()
        methods=['t05',*[prefix+family for prefix in ('p49_','') for family in FAMILIES]]
        c['paired']=np.isfinite(c[['truth','reference',*methods]]).all(axis=1)
        for m in methods:c[m+'_error']=(c[m]-c.truth).abs()
        def metric(g,m):
            q=g[g.paired];ref=q.reference_error.mean();mae=q[m+'_error'].mean();baseline=q.t05_error.mean()
            basefamily=m.removeprefix('p49_')
            model=g['t05_model'] if m=='t05' else g[basefamily+'_model']
            return dict(rows=len(g),paired=len(q),windows=q.window_id.nunique(),mae=mae,ratio=mae/ref if ref else None,
                reference_mae=ref,t05_mae=baseline,gain_percent=100*(baseline-mae)/baseline if baseline else None,
                model_rows=int(model.sum()),fallback_rows=int(g.t05_fallback.sum()) if m=='t05' else int((~model & g[m].notna()).sum()),
                unavailable_rows=int(g[m].isna().sum()))
        for method in methods:
            for dims in ([],['day'],['day','unit'],['day','horizon_band'],['day','window_id']):
                groups=[((),c)] if not dims else c.groupby(dims)
                for keys,g in groups:
                    keys=keys if isinstance(keys,tuple) else (keys,)
                    summaries.append(dict(setting=setting,method=method,group='/'.join(dims) or 'all',**dict(zip(dims,keys)),**metric(g,method)))
            if method=='t05' or method.startswith('p49_'):continue
            groups=[(('all','all'),c),*(( (d,'all'),g) for d,g in c.groupby('day')),
                    *(( (d,u),g) for (d,u),g in c.groupby(['day','unit']))]
            for (day,unit),g in groups:
                gains=g[g.paired].groupby('window_id').apply(lambda z:float((z.t05_error-z[method+'_error']).sum()))
                best=gains.sort_values(ascending=False).index.tolist()
                for k in (0,1,2):sensitivity.append(dict(setting=setting,day=day,unit=unit,method=method,remove_best=k,
                    wins=int((gains>0).sum()),removed=best[:k],**metric(g[~g.window_id.isin(best[:k])],method)))
        c.to_csv(OUT/f'{setting}_scored.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'scores.csv',index=False);dump(OUT/'window_sensitivity.json',sensitivity)
    protected=json.loads((OUT/'protected.json').read_text());assert all(sha(ROOT/q)==h for q,h in protected.items())
    dump(OUT/'verification.json',dict(protected_unchanged=len(protected),t05_fits=0,
        baseline_sha256=sha(ROOT/'outputs/submission_hidden_bgd_v1.csv'),candidate_created=False))
    print(pd.DataFrame(summaries).query("group == 'day'")[['setting','day','method','paired','mae','ratio']].to_string(index=False))

def plots():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for row,family in enumerate(FAMILIES):
        for col,(prefix,days,title) in enumerate([
            ('audit_dev_',['2026-09-04','2026-09-05'],'P49 deterministic replay'),
            ('selection_',['2026-09-11','2026-09-12'],'P50 template-covered selection')]):
            ax=axes[row,col]
            for day,color in zip(days,['tab:blue','tab:orange']):
                df=pd.read_csv(OUT/f'{prefix}{day}_{family}_curve.csv')
                ax.plot(df['round'],df.training_mae,color=color,ls='--',alpha=.7,label=f'{day[5:]} training')
                ax.plot(df['round'],df.validation_mae,color=color,label=f'{day[5:]} validation')
                i=df.validation_mae.idxmin();ax.scatter(df.loc[i,'round'],df.loc[i,'validation_mae'],color=color,s=20,zorder=5)
            ax.set_title(f'{family}: {title}',fontsize=10)
            ax.set_xlabel('Evaluated boosting round');ax.set_ylabel('Residual MAE (before clipping)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'stopping_curves.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','prepare','train','score','plots']);mode=parser.parse_args().mode
    OUT.mkdir(parents=True,exist_ok=True)
    {'audit':audit,'prepare':prepare,'train':train,'score':score,'plots':plots}[mode]()
