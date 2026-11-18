"""P49: one common dataset and configuration per trained tree family.

prepare/score: project environment. train: isolated requirements-forecast-trees.
Assessment labels are never loaded by train. No GP fitting or production edits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/prompt49_forecast_trees'
CONFIG=ROOT/'configs/prompt49_forecast_trees.json'
FAMILIES=('catboost','lightgbm','xgboost')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def cfg():return json.loads(CONFIG.read_text())

def prepare():
    import numpy as np
    import pandas as pd
    sys.path[:0]=[str(ROOT),str(ROOT/'src')]
    import evaluation_artifacts as assets
    import reliable_evaluation as ev
    from evaluate_baseline_task import fixed_benchmark
    from pipeline import load
    from t05_forecast.core import load_recordings,RELEASED,ABSENT
    import forecast_tree_features as f
    started=time.perf_counter()
    if (OUT/'prepared.json').exists():raise FileExistsError('Prepared data already frozen; choose another output directory for a new experiment')
    train,test,_,_=load(ROOT/'data/raw');recs=load_recordings(ROOT/'data/raw')
    accepted=json.loads((ROOT/'models/current_baseline/manifest.json').read_text())
    assert assets.context()['inputs']==accepted['inputs']
    assert sha(ROOT/'data/raw/prayer_times_madinah.csv')==sha(ROOT/'models/current_baseline/prayer_times_madinah.csv')
    assert sha(ROOT/'outputs/submission_hidden_bgd_v1.csv')==accepted['accepted_submission_sha256']
    # Confirm prior experiment's protected sources/assets still match. New trial files are excluded.
    prior=ROOT/'outputs/prompt48_chronos/protected.json'
    if prior.exists():
        old=json.loads(prior.read_text())
        assert all(sha(ROOT/p)==h for p,h in old.items())
    else:
        paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'t05_forecast').glob('*.py'))
        paths+=list((ROOT/'models').rglob('*.*'))+list((ROOT/'outputs').rglob('submission*.csv'))
        old={str(p.relative_to(ROOT)):sha(p) for p in paths}
    protected={**{Path(path).as_posix():h for path,h in old.items()},
               'notebooks/current_baseline.ipynb':sha(ROOT/'notebooks/current_baseline.ipynb')}
    dump(OUT/'protected.json',protected);dump(OUT/'design.json',cfg())
    c,l,feeds=fixed_benchmark(train,test,prefer_saved=False)
    known=json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())
    assert ev.case_fingerprints(c,l)==known['historical']
    fixed={'historical':(c[c.kind.eq('forecast')].copy(),l,feeds)}
    c,l=ev.released_cases(test)
    assert ev.case_fingerprints(c,l)==known['representative']
    from pipeline import table
    fixed['representative']=(c,l,table(test[test.id.isna() & test.people.notna()]))
    development={n:r for n,r in recs.items() if r.day in ('2026-09-04','2026-09-05')}
    fold_specs=[];features=None;audits=[];checks=0;metadata=[]
    for day in ('2026-09-04','2026-09-05'):
        fold_specs.append((f'dev_{day}',{n:r for n,r in development.items() if r.day!=day},
            {n:r for n,r in development.items() if r.day==day},None,None,'development'))
    for setting,(cases,labels,ff) in fixed.items():
        entries=assets.validated_entries(setting,allow_legacy=True)
        # Reuse P48 inference only after exact case and independent saved-result agreement.
        cache_path=ROOT/f'outputs/prompt48_chronos/{setting}_fresh_t05.csv'
        earlier_path=ROOT/f'outputs/prompt38_{setting}/predictions.csv'
        if cache_path.exists() and earlier_path.exists() and prior.exists():
            cached=pd.read_csv(cache_path);earlier=pd.read_csv(earlier_path)
            compare=cached.merge(earlier[['case_id','prediction','reference']],on='case_id',validate='one_to_one',suffixes=('_new','_old'))
            for field in ('prediction','reference'):
                np.testing.assert_allclose(compare[field+'_new'],compare[field+'_old'],rtol=0,atol=0,equal_nan=True)
        else:
            # Cold setup: load validated covariances and infer once; no optimizer.
            import baseline_tasks as tasks
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                cached,_=ev.forecast_predictions(cases,ff,recs,setting,entries,tasks.default_config()['forecast'])
        cached.to_csv(OUT/f'{setting}_t05.csv',index=False)
        assert set(cached.case_id)==set(cases.case_id)
        for day,g in cases.groupby('day'):
            name=g.recording.iloc[0];target=recs[name].copy();target.y[:]=np.nan;target.status[:]=ABSENT;target.truth=None
            for u in ff[name]:
                if u not in target.units:continue
                s=ff[name][u].dropna();ix=s.index.to_numpy(int)-target.first
                target.y[ix,target.col(u)]=s.to_numpy();target.status[ix,target.col(u)]=RELEASED
            g=ev.annotate_cases(g,ff)
            g['horizon_band']=pd.cut(g.horizon,[0,5,10,15],labels=['1-5','6-10','11-15']).astype(str)
            b=cached[['case_id','prediction','reference']].rename(columns={'prediction':'t05'})
            g=g.merge(b,on='case_id',validate='one_to_one')
            g.to_csv(OUT/f'{setting}_{day}_cases.csv',index=False)
            labels[labels.case_id.isin(g.case_id)].to_csv(OUT/f'{setting}_{day}_labels.csv',index=False)
            fold_specs.append((f'{setting}_{day}',{n:r for n,r in recs.items() if r.day!=day},
                {name:target},g,None,'assessment'))
    for fold,pool,validation,assessment,_,role in fold_specs:
        excluded=sorted({r.day for r in validation.values()})
        assert not (set(excluded)&{r.day for r in pool.values()})
        raw_train=f.training_cases(pool)
        inputs={n:f.masked_training_record(r,test) if r.split=='train' else r.copy() for n,r in pool.items()}
        tm={}
        for n,r in pool.items():
            tm[n],audit=f.weekly(r,pool,excluded);audit['fold']=fold;audits.append(audit)
        x,ncheck=f.feature_rows(raw_train,inputs,tm,check=True);checks+=ncheck
        if features is None:features=f.feature_columns(x)
        assert features==f.feature_columns(x)
        y=x[['case_id','anchor']].merge(raw_train[['case_id','truth']],on='case_id',validate='one_to_one')
        assert len(y)==len(x) and y.case_id.tolist()==x.case_id.tolist()
        x[f.META+['anchor','template_available','own_age','own_n10','available_other_cameras']].to_csv(OUT/f'{fold}_training_metadata.csv',index=False)
        tcases=assessment if role=='assessment' else f.training_cases(validation)
        vinputs=validation if role=='assessment' else {n:f.masked_training_record(r,test) for n,r in validation.items()}
        vt={}
        for n,r in validation.items():
            vt[n],audit=f.weekly(r,pool,excluded);audit['fold']=fold;audits.append(audit)
            # Target truth cannot influence either the curve or coefficient selection.
            poisoned=r.copy();poisoned.y[:]=1e9
            check_t,_=f.weekly(poisoned,pool,excluded)
            np.testing.assert_allclose(vt[n],check_t,rtol=0,atol=0,equal_nan=True)
        v,ncheck=f.feature_rows(tcases,vinputs,vt,check=True);checks+=ncheck
        assert features==f.feature_columns(v)
        data=dict(train_x=x[features].to_numpy(),train_y=(y.truth-y.anchor).to_numpy(),
                  valid_x=v[features].to_numpy(),valid_anchor=v.anchor.to_numpy())
        if role=='development':
            vy=v[['case_id','anchor']].merge(tcases[['case_id','truth']],on='case_id',validate='one_to_one')
            data['valid_y']=(vy.truth-vy.anchor).to_numpy()
        np.savez_compressed(OUT/f'{fold}.npz',**data)
        v.to_csv(OUT/f'{fold}_features.csv',index=False)
        metadata.append(dict(fold=fold,role=role,excluded_dates=excluded,training_recordings=sorted(pool),
            training_rows=len(x),training_windows=x[['recording','now']].drop_duplicates().shape[0],
            assessment_rows=len(tcases),eligible_rows=len(v),training_template_rows=int(x.template_available.sum()),
            validation_template_rows=int(v.template_available.sum()),npz_sha256=sha(OUT/f'{fold}.npz')))
    # Production request coverage uses exactly the same feature implementation, no answers.
    rows=test.loc[test.kind.eq('forecast') & test.unit.isin(f.UNITS),
                  ['id','recording','day','unit','minute_of_day','now_minute_of_day','horizon']].copy()
    rows=rows.rename(columns={'id':'case_id','minute_of_day':'minute','now_minute_of_day':'now'})
    tmap={n:f.weekly(r,recs,[])[0] for n,r in recs.items() if r.split=='test'}
    actual,_=f.feature_rows(rows,recs,tmap,check=True)
    actual.to_csv(OUT/'actual_features.csv',index=False)
    pd.DataFrame(metadata).to_csv(OUT/'fold_inventory.csv',index=False)
    dump(OUT/'template_audit.json',audits)
    manifest=dict(config_sha256=sha(CONFIG),feature_source_sha256=sha(ROOT/'src/forecast_tree_features.py'),
        runner_sha256=sha(__file__),features=features,folds=metadata,causal_context_checks=checks,
        actual_forecast_rows=len(rows),actual_eligible_rows=len(actual),actual_im_rows=int(test.kind.eq('forecast').mul(test.unit.str[-1].isin(list('IJKLM'))).sum()),
        input_hashes=accepted['inputs'],preparation_seconds=time.perf_counter()-started,
        metric='Project local MAE/reference MAE, not authenticated official score',
        protocol='Strict date-held-out only. In released-observation cases, previously released earlier values can be context for later requests; never future labels.',
        t05='Reuse isolated diagnostic predictions, not production output; legacy covariance provenance remains incomplete',
        selection='Development September 4/5 only; all assessment dates excluded from early stopping and settings selection')
    dump(OUT/'prepared.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('features','input_hashes')},indent=2))

def estimator(family,rounds,early=False):
    c=cfg();common=dict(learning_rate=c['learning_rate'])
    if family=='catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**c[family],**common,iterations=rounds,random_seed=c['seed'],
            thread_count=c['threads'],allow_writing_files=False,verbose=False)
    if family=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**c[family],**common,n_estimators=rounds,random_state=c['seed'],
            n_jobs=c['threads'],verbosity=-1,deterministic=True,force_col_wise=True)
    from xgboost import XGBRegressor
    return XGBRegressor(**c[family],**common,n_estimators=rounds,random_state=c['seed'],n_jobs=c['threads'],
        eval_metric='mae',early_stopping_rounds=c['early_stopping_rounds'] if early else None)

def fit(model,family,data,early):
    kwargs={}
    if early:
        kwargs['eval_set']=[(data['valid_x'],data['valid_y'])]
        if family=='catboost':kwargs.update(early_stopping_rounds=cfg()['early_stopping_rounds'],use_best_model=True)
        elif family=='lightgbm':
            import lightgbm as lgb
            kwargs.update(callbacks=[lgb.early_stopping(cfg()['early_stopping_rounds'],verbose=False)])
        else:kwargs['verbose']=False
    model.fit(data['train_x'],data['train_y'],**kwargs)
    if family=='catboost':return model.tree_count_
    if family=='lightgbm':return model.best_iteration_ if early else cfg()['maximum_rounds']
    return model.best_iteration+1 if early else cfg()['maximum_rounds']

def save_model(model,family,path):
    if family=='catboost':model.save_model(str(path.with_suffix('.cbm')))
    elif family=='lightgbm':model.booster_.save_model(str(path.with_suffix('.txt')))
    else:model.save_model(str(path.with_suffix('.json')))

def train_models():
    import importlib.metadata as md
    import numpy as np
    import pandas as pd
    p=json.loads((OUT/'prepared.json').read_text());c=cfg();started=time.perf_counter()
    assert sha(CONFIG)==p['config_sha256']
    assert sha(ROOT/'src/forecast_tree_features.py')==p['feature_source_sha256']
    for name,digest in p['input_hashes'].items():assert sha(ROOT/'data/raw'/name)==digest
    protected=json.loads((OUT/'protected.json').read_text())
    assert all(sha(ROOT/path)==digest for path,digest in protected.items())
    if (OUT/'training_manifest.json').exists():raise FileExistsError('Do not silently retrain a finished experiment')
    for f in p['folds']:assert sha(OUT/(f['fold']+'.npz'))==f['npz_sha256']
    rounds={};records=[];devscores=[]
    for family in FAMILIES:
        selected=[]
        for f in p['folds']:
            if f['role']!='development':continue
            data=np.load(OUT/(f['fold']+'.npz'));model=estimator(family,c['maximum_rounds'],True)
            start=time.perf_counter();best=fit(model,family,data,True);elapsed=time.perf_counter()-start
            residual=model.predict(data['valid_x']);prediction=np.maximum(data['valid_anchor']+residual,0)
            truth=data['valid_anchor']+data['valid_y'];reference=data['valid_x'][:,p['features'].index('own_level')]
            reference=np.expm1(reference)
            devscores.append(dict(family=family,fold=f['fold'],rows=len(truth),best_round=best,
                mae=float(np.abs(prediction-truth).mean()),reference_mae=float(np.abs(reference-truth).mean()),
                training_seconds=elapsed))
            selected.append(best)
        rounds[family]=max(1,int(np.median(selected)))
    # Freeze settings before ANY assessment labels are loaded (not available to this mode).
    dump(OUT/'selected_rounds.json',rounds);pd.DataFrame(devscores).to_csv(OUT/'development_scores.csv',index=False)
    (OUT/'models').mkdir(exist_ok=True)
    for f in p['folds']:
        if f['role']!='assessment':continue
        data=np.load(OUT/(f['fold']+'.npz'))
        features=pd.read_csv(OUT/(f['fold']+'_features.csv'))
        predictions=features[['case_id']].copy()
        for family in FAMILIES:
            model=estimator(family,rounds[family]);start=time.perf_counter();fit(model,family,data,False)
            training=time.perf_counter()-start;start=time.perf_counter()
            values=np.maximum(data['valid_anchor']+model.predict(data['valid_x']),0)
            inference=time.perf_counter()-start
            assert np.isfinite(values).all() and (values>=0).all()
            # Native missing-value and single-vs-batch path consistency.
            for i in sorted(set([0,len(features)//2,len(features)-1])):
                single=np.maximum(data['valid_anchor'][i]+model.predict(data['valid_x'][i:i+1])[0],0)
                np.testing.assert_allclose(single,values[i],rtol=1e-12,atol=1e-9)
            predictions[family]=values
            save_model(model,family,OUT/'models'/f"{f['fold']}_{family}")
            records.append(dict(family=family,fold=f['fold'],rounds=rounds[family],training_seconds=training,
                inference_seconds=inference,training_rows=len(data['train_y']),predictions=len(values)))
        predictions.to_csv(OUT/(f['fold']+'_predictions.csv'),index=False)
    pd.DataFrame(records).to_csv(OUT/'runtimes.csv',index=False)
    dump(OUT/'training_manifest.json',dict(rounds=rounds,total_seconds=time.perf_counter()-started,
        common_features=p['features'],fold_hashes={f['fold']:f['npz_sha256'] for f in p['folds']},
        versions={n:md.version(n) for n in ['catboost','lightgbm','xgboost','numpy','pandas','scipy','scikit-learn']},
        assessment_labels_read=False,t05_fits=0))
    print(json.dumps(dict(rounds=rounds,runtime_seconds=time.perf_counter()-started),indent=2))

def score():
    import numpy as np
    import pandas as pd
    summaries=[];sens=[]
    for setting in ('representative','historical'):
        allrows=[]
        for day in (('2026-09-18','2026-09-19') if setting=='representative' else ('2026-09-11','2026-09-12')):
            fold=f'{setting}_{day}'
            c=pd.read_csv(OUT/f'{fold}_cases.csv');labels=pd.read_csv(OUT/f'{fold}_labels.csv')
            c=c.merge(labels,on='case_id',validate='one_to_one')
            local=OUT/f'{setting}_t05.csv'
            flags=pd.read_csv(local if local.exists() else ROOT/f'outputs/prompt48_chronos/{setting}_fresh_t05.csv')
            flags=flags[['case_id','model','fallback']].rename(columns={'model':'t05_model','fallback':'t05_fallback'})
            c=c.merge(flags,on='case_id',validate='one_to_one')
            predictions=pd.read_csv(OUT/f'{fold}_predictions.csv')
            c=c.merge(predictions,on='case_id',how='left',validate='one_to_one')
            for family in FAMILIES:
                c[family+'_model']=c[family].notna()
                c[family]=c[family].fillna(c.t05)
                c[family+'_changed']=c[family].ne(c.t05)&c[family+'_model']
            c['paired']=np.isfinite(c[['truth','t05','reference',*FAMILIES]]).all(axis=1)
            for method in ('t05',*FAMILIES,'reference'):c[method+'_error']=(c[method]-c.truth).abs()
            allrows.append(c)
        c=pd.concat(allrows,ignore_index=True)
        c.to_csv(OUT/f'{setting}_scored.csv',index=False)
        def metrics(g,family):
            q=g[g.paired];ref=q.reference_error.mean();mae=q[family+'_error'].mean();base=q.t05_error.mean()
            return dict(rows=len(g),paired=len(q),unique_targets=q.target_id.nunique(),windows=q.window_id.nunique(),
                model_rows=int(g[family+'_model'].sum()),
                fallback_rows=int(g.t05_fallback.sum()) if family=='t05' else int((~g[family+'_model'] & g[family].notna()).sum()),
                unavailable_rows=int(g[family].isna().sum()),
                mae=mae,reference_mae=ref,ratio=mae/ref if ref else None,t05_mae=base,gain_percent=100*(base-mae)/base if base else None)
        for family in ('t05',*FAMILIES):
            for dims in ([],['day'],['day','unit'],['day','horizon_band'],['day','unit','horizon_band'],['day','window_id']):
                groups=[((),c)] if not dims else c.groupby(dims,dropna=False)
                for keys,g in groups:
                    keys=keys if isinstance(keys,tuple) else (keys,)
                    summaries.append(dict(setting=setting,method=family,group='/'.join(dims) or 'all',**dict(zip(dims,keys)),**metrics(g,family)))
            if family=='t05':continue
            summaries.append(dict(setting=setting,method=family,group='changed_only',**metrics(c[c[family+'_changed']],family)))
            groups=[(('all','all'),c),*(( (day,'all'),g) for day,g in c.groupby('day')),
                    *(( (day,unit),g) for (day,unit),g in c.groupby(['day','unit']))]
            for (day,unit),g in groups:
                gains=g[g.paired].groupby('window_id').apply(lambda z:float((z.t05_error-z[family+'_error']).sum()))
                best=gains.sort_values(ascending=False).index.tolist()
                for k in (0,1,2):
                    sens.append(dict(setting=setting,method=family,day=day,unit=unit,remove_best=k,
                        winning_windows=int((gains>0).sum()),removed=best[:k],**metrics(g[~g.window_id.isin(best[:k])],family)))
                if unit=='all':
                    for w in gains.index:
                        sens.append(dict(setting=setting,method=family,day=day,unit=unit,remove_best='leave_one',removed=[w],**metrics(g[g.window_id.ne(w)],family)))
    pd.DataFrame(summaries).to_csv(OUT/'scores.csv',index=False)
    dump(OUT/'window_sensitivity.json',sens)
    protected=json.loads((OUT/'protected.json').read_text());assert all(sha(ROOT/p)==v for p,v in protected.items())
    dump(OUT/'verification.json',dict(protected_unchanged=len(protected),baseline_sha256=sha(ROOT/'outputs/submission_hidden_bgd_v1.csv'),
        t05_fits=0,shared_pipeline=True,all_model_predictions_finite_nonnegative=True,submission_created=False))
    print(pd.DataFrame(summaries).query("group == 'day'")[['setting','day','method','paired','mae','ratio','gain_percent']].to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','train','score']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.mode=='prepare':prepare()
    elif args.mode=='train':train_models()
    else:score()
