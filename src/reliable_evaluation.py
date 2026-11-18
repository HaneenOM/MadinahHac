"""Explicit evaluation settings, live task implementations, no GP optimization.

The project summary is NOT an authenticated official Kaggle score.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import baseline_tasks as tasks
import evaluation_artifacts as artifacts
from pipeline import load,table
from evaluate_baseline_task import fixed_benchmark
from prayer_experiments import timetable
from t05_forecast.core import ABSENT,RELEASED,FORECAST,load_recordings
from t05_forecast.pipeline import load_params
from t05_forecast.template import slope,build_template,complete_template
from t05_forecast.reference import last_value


def released_cases(test):
    """Exact P21 fixed 18/19 A-H cases, derived from metadata, labels separate."""
    cases=[];labels=[]
    for name,day in [(n,d) for d,n in artifacts.RECENT.items()]:
        g=test[test.recording.eq(name)];first=int(g.minute_of_day.min());last=int(g.minute_of_day.max())
        available=g[g.id.isna() & g.people.notna() & g.unit.str[-1].isin(list('ABCDEFGH'))]
        keys={(r.unit,int(r.minute_of_day)):r.people for r in available.itertuples()}
        for now in range(first+15,last,30):
            for unit in [f'cam_{u}' for u in 'ABCDEFGH']:
                for h in range(1,16):
                    minute=now+h
                    if (unit,minute) not in keys:continue
                    key=f'{name}:{unit}:{now}:{h}'
                    cases.append(dict(case_id=key,day=day,recording=name,unit=unit,now=now,
                        minute=minute,horizon=h,kind='forecast'))
                    labels.append(dict(case_id=key,truth=keys[unit,minute]))
    assert len(cases)==1566
    return pd.DataFrame(cases),pd.DataFrame(labels)


def im_cases(test):
    rows=[];labels=[]
    released=test[test.id.isna() & test.people.notna() & test.unit.str[-1].isin(list('IJKLM'))]
    first=test.groupby('recording').minute_of_day.min().to_dict()
    for r in released.itertuples():
        for h in range(1,16):
            minute=int(r.minute_of_day);now=minute-h
            if now<first[r.recording]:continue
            key=f'{r.recording}:{r.unit}:{minute}:{h}'
            rows.append(dict(case_id=key,recording=r.recording,day=r.day,unit=r.unit,
                minute=minute,now=now,horizon=h,kind='forecast'))
            labels.append(dict(case_id=key,truth=float(r.people)))
    return pd.DataFrame(rows),pd.DataFrame(labels)


def im_predictions(cases,feeds,config):
    rows=[]
    for r in cases.itertuples():
        s=feeds[r.recording][r.unit].loc[:int(r.now)]
        base,value,eligible=tasks.im_forecast_value(s,int(r.now),config)
        poison=feeds[r.recording][r.unit].copy();poison.loc[poison.index>int(r.now)]=1e8
        check=tasks.im_forecast_value(poison,int(r.now),config)
        np.testing.assert_allclose([base,value],[check[0],check[1]],equal_nan=True,rtol=0,atol=0)
        rows.append(dict(case_id=r.case_id,prediction=value,reference=base,
            model=False,fallback=bool(np.isfinite(value)),correction_used=eligible,
            latest_available=bool(np.isfinite(base)),prior_week_template=False,completed_template=False))
    return pd.DataFrame(rows),len(rows)


def case_fingerprints(cases,labels):
    import hashlib
    columns=['case_id','day','recording','kind','unit','minute','now','horizon']
    def digest(frame):
        return hashlib.sha256(frame.sort_values('case_id').to_json(orient='records',double_precision=15).encode()).hexdigest()
    return dict(cases=digest(cases[columns]),labels=digest(labels[['case_id','truth']]))


def actual_forecast_coverage(test):
    """Metadata/input coverage only. Never predicts or scores a Kaggle hole."""
    recs=load_recordings(ROOT/'data/raw');feeds=table(test[test.id.isna() & test.people.notna()])
    rows=[]
    for name,g in test[test.kind.eq('forecast')].groupby('recording'):
        target=artifacts.empty_target(recs[name]);eligible=artifacts.excluded_sources(recs,name)
        same=[r for r in eligible.values() if r.split=='train' and r.weekday==target.weekday]
        long=[r for r in same if r.n_minutes>=600]
        base=build_template(target,long or same,lam=.8)
        for r in g.itertuples():
            now=int(r.now_minute_of_day);s=feeds[name][r.unit].loc[:now].dropna();recent=s.loc[now-10:now]
            rows.append(dict(day=r.day,unit=r.unit,camera_group='A-H' if r.unit[-1] in 'ABCDEFGH' else 'I-M',
                recent=bool(len(s) and now-s.index[-1]<=5),
                trend_context=bool(len(recent)>=3 and np.ptp(recent.index)>=5),
                prior_week_template=bool(np.isfinite(base[int(r.minute_of_day)-target.first,target.col(r.unit)]))))
    return pd.DataFrame(rows).groupby(['day','camera_group']).agg(targets=('unit','size'),
        cameras=('unit',lambda s:','.join(sorted(s.unique()))),recent=('recent','sum'),
        trend_context=('trend_context','sum'),prior_week_template=('prior_week_template','sum')).reset_index()


def write_inventory(out):
    artifacts.inventory(out)
    path=Path(out)/'inventory.json';inventory=json.loads(path.read_text())
    known=json.loads((artifacts.ASSETS/'manifest.json').read_text())
    by_hash={e['sha256']:e for e in known['entries']}
    production=json.loads((ROOT/'models/current_baseline/manifest.json').read_text())
    for record in inventory['artifacts']:
        if record['sha256'] in by_hash:
            record['available_recipe_evidence']=by_hash[record['sha256']]
        elif record['sha256'] in production['assets'].values():
            record['available_manifest_evidence']=dict(path='models/current_baseline/manifest.json',
                input_hashes=production['inputs'],config=production['config'],
                note='Accepted-production reproduction manifest; not a training-execution record')
        if record['kind']=='saved_predictions':
            parent=(ROOT/record['path']).parent
            record['adjacent_run_evidence']=[dict(path=str(p.relative_to(ROOT)),sha256=artifacts.sha(p))
                for name in ('run_manifest.json','verification.json','case_hashes.json') if (p:=parent/name).exists()]
    path.write_text(json.dumps(inventory,indent=2),encoding='utf-8')


def annotate_cases(cases,feeds):
    cases=cases.copy()
    cases['target_id']=cases.recording+':'+cases.unit+':'+cases.minute.astype(int).astype(str)
    cases['camera_group']=np.where(cases.unit.str[-1].isin(list('ABCDEFGH')),'A-H','I-M')
    # Greedy disjoint future intervals. Report origin count separately; arbitrary
    # origins and adjacent rows are not automatically independent observations.
    mapping={}
    future=cases[cases['now'].notna()]
    for name,g in future.groupby('recording'):
        end=-1;number=0
        for now,a in g.groupby('now',sort=True):
            if int(now)+1>end:number+=1
            end=max(end,int(a.minute.max()))
            mapping[name,int(now)]=f'{name}:forecast_block:{number}'
    windows=[]
    lengths=[]
    for r in cases.itertuples():
        if pd.notna(r.now):
            w=mapping[r.recording,int(r.now)]
        elif r.kind=='hidden':
            w=f'{r.recording}:hidden_30min:{(int(r.minute)-660)//30}'
        else:
            import v2_missing
            ctx=v2_missing.gap_context(feeds[r.recording],r.unit,int(r.minute))
            w=f'{r.recording}:{r.unit}:gap:{ctx[0]}-{ctx[1]}' if ctx else f'{r.recording}:{r.unit}:unavailable'
        windows.append(w)
        lengths.append(ctx[1]-ctx[0]-1 if r.kind=='missing' and ctx else np.nan)
    cases['window_id']=windows
    cases['context_gap_length']=lengths
    counts=cases.groupby('window_id').case_id.transform('size')
    cases['complete_gap_labels']=cases.context_gap_length.eq(counts)
    return cases


def prefix_record(source,feed,targets,now):
    rec=artifacts.empty_target(source)
    rec.status[:]=ABSENT;rec.now[:]=-1
    for unit in feed:
        if unit not in rec.units:continue
        s=feed[unit].loc[:now].dropna()
        rows=s.index.to_numpy(int)-rec.first;c=rec.col(unit)
        rec.y[rows,c]=s.to_numpy();rec.status[rows,c]=RELEASED
    for r in targets.itertuples():
        i,j=int(r.minute)-rec.first,rec.col(r.unit)
        assert not np.isfinite(rec.y[i,j])
        rec.status[i,j]=FORECAST;rec.now[i,j]=now
    assert not np.isfinite(rec.y[rec.minutes>now]).any()
    return rec


def forecast_predictions(cases,feeds,recs,setting,entries,config):
    rows=[];checks=0
    for name,today in cases.groupby('recording',sort=False):
        source=recs[name];target=artifacts.empty_target(source)
        eligible=artifacts.excluded_sources(recs,name)
        model=[];base_templates=[]
        for member in artifacts.members():
            if setting=='historical':
                same=[r for r in eligible.values() if r.split=='train' and r.weekday==source.weekday]
                other=[r for r in eligible.values() if r.split=='train' and r.weekday!=source.weekday]
                base=build_template(target,same,lam=member['lam'])
                tmpl,_=complete_template(target,base,other,lam=member['lam'])
            else:
                same=[r for r in eligible.values() if r.split=='train' and r.weekday==source.weekday]
                long=[r for r in same if r.n_minutes>=600]
                base=build_template(target,long or same,lam=member['lam'])
                tmpl=artifacts.independent_template(recs,name,member['lam'],complete_new_cameras=True)
            entry=entries[name,member['key']]
            params=load_params(artifacts.ASSETS/setting/entry['filename'])
            model.append((tmpl,slope(tmpl),params));base_templates.append(base)
        for now,group in today.groupby('now'):
            now=int(now);feed=feeds[name]
            rec=prefix_record(source,feed,group,now)
            # Structural poison checks: every camera, all future observations;
            # hidden target values never enter rec, template sources or fitting.
            altered=feed.copy();altered.loc[altered.index>now]=1e8
            poisoned=prefix_record(source,altered,group,now)
            np.testing.assert_array_equal(rec.y,poisoned.y)
            altered_source=source.copy();altered_source.y[:]=1e8
            np.testing.assert_array_equal(rec.y,prefix_record(altered_source,feed,group,now).y)
            mean=tasks.condition_forecast(rec,model)
            last=last_value(rec.y,5)
            for r in group.itertuples():
                i,j=int(r.minute)-rec.first,rec.col(r.unit)
                raw=float(mean[i,j]);reference=float(last[now-rec.first,j])
                value=tasks.forecast_value(reference,raw,config)
                recent=feed[r.unit].loc[max(source.first,now-10):now].dropna()
                age=now-int(recent.index[-1]) if len(recent) else np.nan
                rows.append(dict(case_id=r.case_id,prediction=value,reference=reference,
                    raw_model_available=bool(np.isfinite(raw)),raw_model_prediction=raw,
                    model=bool(np.isfinite(raw) and np.isfinite(value)),fallback=bool(not np.isfinite(raw) and np.isfinite(value)),
                    latest_available=bool(np.isfinite(reference)),latest_age=age,
                    trend_context=bool(len(recent)>=3 and np.ptp(recent.index)>=5),
                    prior_week_template=bool(all(np.isfinite(a[i,j]) for a in base_templates)),
                    completed_template=bool(all(np.isfinite(a[0][i,j]) for a in model)),
                    donors_at_now=int(np.isfinite(rec.y[max(0,now-rec.first-5):now-rec.first+1]).any(axis=0).sum())))
            checks+=len(group)
    return pd.DataFrame(rows),checks


def task_predictions(cases,feeds,train,schedule,config):
    rows=[];checks=0
    import v2_hidden
    for (day,kind),today in cases.groupby(['day','kind'],sort=False):
        training=train[train.day.ne(day)].copy()
        assert day not in set(training.day)
        if kind=='hidden':models=tasks.fit_hidden(training,config[kind],schedule)
        elif kind=='hidden-future':models=tasks.fit_hidden_future(training,schedule,config[kind])
        else:models=tasks.fit_missing(training,config[kind])
        for r in today.itertuples():
            t=feeds[r.recording].copy();unit=r.unit;minute=int(r.minute)
            meta={};available=None
            if kind in ('hidden','hidden-future'):
                for u in v2_hidden.PATTERNS[unit]:t[u]=np.nan
            if kind=='hidden':
                from v2_missing import interp_at
                available=sum(np.isfinite(interp_at(t[u],minute)) for u in models['linear'][unit]['donors'] if u in t)
                reference,value,_=tasks.hidden_value(t,unit,minute,models,config[kind],day=day,schedule=schedule)
                poison=t.copy()
                for u in v2_hidden.PATTERNS[unit]:poison[u]=1e8
                _,check,_=tasks.hidden_value(poison,unit,minute,models,config[kind],day=day,schedule=schedule)
            elif kind=='hidden-future':
                now=int(r.now);t=t.loc[:now]
                reference,value,_=tasks.hidden_future_value(t,unit,now,int(r.horizon),day,schedule,models,config[kind])
                poison=feeds[r.recording].copy();poison.loc[poison.index>now]=1e8
                for u in v2_hidden.PATTERNS[unit]:poison[u]=1e8
                _,check,_=tasks.hidden_future_value(poison,unit,now,int(r.horizon),day,schedule,models,config[kind])
            else:
                reference,value,meta=tasks.missing_value(t,unit,minute,models,config[kind])
                poison=t.copy();poison.loc[minute,unit]=np.nan
                _,check,_=tasks.missing_value(poison,unit,minute,models,config[kind])
            np.testing.assert_allclose(value,check,rtol=0,atol=1e-10);checks+=1
            rows.append(dict(case_id=r.case_id,prediction=value,reference=reference,
                model=bool(meta.get('used',True) and (available is None or available>=2)),
                fallback=bool(not meta.get('used',True) or (available is not None and available<2)),
                correction_used=meta.get('used'),donors=meta.get('donors'),gap_length=meta.get('gap')))
    return pd.DataFrame(rows),checks


def summarize(predictions,labels,out,setting):
    # Predictor outputs are written before the scorer is handed the label frame.
    predictions.to_csv(out/'predictions.csv',index=False)
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    scored['scoreable']=np.isfinite(scored.prediction)&np.isfinite(scored.reference)&np.isfinite(scored.truth)
    scored['error']=(scored.prediction-scored.truth).abs()
    scored['reference_error']=(scored.reference-scored.truth).abs()
    scored.to_csv(out/'scored.csv',index=False)
    def report(frame,group):
        rows=[]
        for keys,g in frame.groupby(group,dropna=False):
            keys=keys if isinstance(keys,tuple) else (keys,)
            paired=g[g.scoreable];den=float(paired.reference_error.mean())
            row=dict(zip(group,keys));row.update(cases=len(g),unique_targets=g.target_id.nunique(),
                windows=g.window_id.nunique(),origins=g[['recording','now']].dropna().drop_duplicates().shape[0],
                cameras=','.join(sorted(g.unit.unique())),paired=len(paired),
                model=int(g.model.fillna(False).sum()),fallback=int(g.fallback.fillna(False).sum()),
                unavailable=int((~g.scoreable).sum()),mae=float(paired.error.mean()),reference_mae=den,
                project_normalized_mae=float(paired.error.mean()/den) if den>0 else None)
            if 'complete_gap_labels' in g and g.kind.eq('missing').all():
                row['complete_gaps']=g.loc[g.complete_gap_labels,'window_id'].nunique()
            for c in ('raw_model_available','latest_available','trend_context','prior_week_template','completed_template'):
                if c in g:row[c]=int(g[c].fillna(False).sum())
            rows.append(row)
        return pd.DataFrame(rows)
    for group,name in [(['day','kind','camera_group'],'date_task'),(['day','kind','unit'],'camera'),
        (['day','kind','horizon_band'],'horizon'),(['day','kind','window_id'],'window')]:
        report(scored,group).to_csv(out/f'by_{name}.csv',index=False)
    if setting=='representative':
        matched=scored[scored.latest_available & scored.trend_context]
        report(matched,['day','kind','camera_group']).to_csv(out/'matched_context.csv',index=False)
    # No cross-setting aggregate. Historical project summary only if all tasks
    # and dates are present and every row has a paired reference.
    summary=report(scored,['day','kind','camera_group'])
    aggregate=None
    if setting=='historical' and len(summary)==8 and scored.scoreable.all():
        aggregate=float(summary.groupby('day').project_normalized_mae.mean().mean())
    return summary,aggregate


def run(setting,out,allow_legacy=False,config=None,task='all'):
    started=time.perf_counter();out=Path(out)
    if out.exists():raise FileExistsError(f'Choose a new output directory: {out}')
    cfg=config or tasks.default_config()
    if cfg['hidden']['d_only']:raise ValueError('Baseline evaluation only: D-only must remain disabled')
    need_gp=(setting in ('historical','representative') and task in ('all','forecast'))
    entries=artifacts.validated_entries(setting,allow_legacy) if need_gp else None
    out.mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()):
        train,test,_,_=load(ROOT/'data/raw')
        schedule=timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)
        if setting=='historical':
            cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
        elif setting=='representative':
            cases,labels=released_cases(test);feeds=table(test[test.id.isna() & test.people.notna()])
        else:
            cases,labels=im_cases(test);feeds=table(test[test.id.isna() & test.people.notna()])
    fingerprints=case_fingerprints(cases,labels)
    expected=json.loads((artifacts.ASSETS/'cases_manifest.json').read_text())[setting]
    if fingerprints!=expected:raise ValueError('Fixed evaluation cases/labels changed; never silently replace masks')
    if setting=='representative' and task not in ('all','forecast'):raise ValueError('Representative setting is FORECAST only')
    if task!='all':cases=cases[cases.kind.eq(task)].copy();labels=labels[labels.case_id.isin(cases.case_id)]
    cases=annotate_cases(cases,feeds)
    cases['horizon_band']=pd.cut(cases.horizon,[0,5,10,15],labels=['1-5','6-10','11-15']).astype(str)
    cases.to_csv(out/'cases.csv',index=False)
    # Save scorer labels separately; no predictor receives this frame.
    labels.to_csv(out/'labels.csv',index=False)
    actual_forecast_coverage(test).to_csv(out/'actual_target_coverage.csv',index=False)
    results=[];checks=0
    with threadpool_limits(limits=4):
        if setting=='im-proxy':
            values,n=im_predictions(cases,feeds,cfg['forecast']);results.append(values);checks+=n
        if need_gp:
            values,n=forecast_predictions(cases[cases.kind.eq('forecast')],feeds,
                load_recordings(ROOT/'data/raw'),setting,entries,cfg['forecast'])
            results.append(values);checks+=n
        rest=cases[~cases.kind.eq('forecast')]
        if len(rest):
            values,n=task_predictions(rest,feeds,train,schedule,cfg)
            results.append(values);checks+=n
    predictions=cases.merge(pd.concat(results,ignore_index=True),on='case_id',validate='one_to_one')
    summary,aggregate=summarize(predictions,labels,out,setting)
    info=dict(setting=setting,task=task,official_metric_confirmed=False,
        metric='project MAE / paired reference MAE; equal-date/equal-task summary only within historical setting',
        project_historical_summary=aggregate,cases=len(cases),checks=checks,t05_fits=0,
        covariance_provenance='legacy_not_fully_verified' if need_gp else 'not_used',
        labels='Separate scorer frame; never supplied to task prediction or GP recipes',
        window_definition='Greedy disjoint future intervals per recording; hidden 30-min blocks; missing complete/partial gap. Disjoint does not imply statistical independence.',
        date_warning='Previously examined dates, not fresh independent validation',
        config=cfg,inputs=artifacts.context()['inputs'],versions=artifacts.environment(),
        task_training_folds={str(day):dict(excluded_date=str(day),
            recordings=sorted(train.loc[train.day.ne(day),'recording'].unique().tolist()),
            preprocessing='Fit only on these recordings, no evaluation-date values')
            for day in cases.day.unique()} if setting=='historical' else {},
        case_fingerprints=fingerprints,
        output_case_files={n:artifacts.sha(out/n) for n in ('cases.csv','labels.csv')},
        code={str(p.relative_to(ROOT)):artifacts.sha(p) for base in ('src','t05_forecast')
              for p in sorted((ROOT/base).glob('*.py'))},
        runtime_seconds=time.perf_counter()-started)
    (out/'run_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps({k:v for k,v in info.items() if k in
        ('setting','cases','project_historical_summary','runtime_seconds','covariance_provenance')},indent=2))
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting',choices=['historical','representative','im-proxy','inventory','import-legacy'],required=True)
    parser.add_argument('--task',choices=['all',*tasks.DEFAULT_CONFIG],default='all')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--allow-legacy-provenance',action='store_true')
    args=parser.parse_args()
    if args.setting=='inventory':write_inventory(args.output_dir)
    elif args.setting=='import-legacy':artifacts.import_legacy()
    else:
        if args.output_dir is None:parser.error('--output-dir required')
        cfg=json.loads(args.config.read_text()) if args.config else None
        run(args.setting,args.output_dir,args.allow_legacy_provenance,cfg,args.task)
