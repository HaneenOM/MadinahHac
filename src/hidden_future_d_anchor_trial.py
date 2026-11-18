"""Prompt 35: replace only the 25% fixed Saturday D reference anchor.

No reconstructed feature is fed to a downstream learner. The accepted prayer
model and the previous nonlinear reconstruction settings stay fixed. All fits
exclude the whole evaluated date; no T05 fit or inference occurs.
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

from pipeline import ROOT,table
import baseline_tasks as tasks
import hidden_reconstruction_trial as reconstruction
from prayer_experiments import timetable

OUT=ROOT/'outputs/hidden_future_d_anchor_trial'
FIXED=ROOT/'outputs/local_score'
BASE=ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
SPEC=dict(change='Saturday D only: current + .25*(nonlinear_D_at_now - fixed_single_donor_reference_at_now)',
    weight=.25,reconstruction_params=reconstruction.PARAMS,
    reconstruction='unchanged Prompt 26/27 model, exact-minute visible A-H, >=2 donors; D removed across entire recording; cutoff before any feature extraction',
    downstream='unchanged 75% accepted prayer direct model; no new downstream training on reconstructed features, hence no OOF stacking required',
    exclusion='entire evaluation DATE excluded from all fits, medians, donor selection and preprocessing',
    auxiliary='Sep5 midday/evening use existing clock-aligned real Saturday masks; morning has no overlapping clock mask, so separate relative-time transplant anchored to recording start',
    decision='Primary gain >=1%, >=3 winning origins, every leave-one-origin-out gain >0; >=50% primary coverage. Each auxiliary with >=20 used targets / >=2 used origins must gain and every leave-one-origin-out gain >0, >=2 winning origins. At least two supported auxiliary recordings. No primary horizon group >=10 targets / >=2 origins regresses >max(.5 person,10% current MAE).',
    validation='already examined historical dates; no independent fresh validation; origins are disjoint target intervals, not independent date replicates',
    fallback='retain accepted prediction with fewer than 2 exact donors or missing nonlinear model')


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def add_windows(frame):
    frame=frame.copy()
    frame['window_id']=frame.recording+':'+frame['now'].astype(int).astype(str)
    frame['horizon_band']=pd.cut(frame.horizon,[0,5,10,15],labels=['1-5','6-10','11-15']).astype(str)
    for name,g in frame.groupby('recording'):
        intervals=g.groupby('now').minute.agg(['min','max']).sort_index()
        assert (intervals['min'].iloc[1:].to_numpy()>intervals['max'].iloc[:-1].to_numpy()).all()
    return frame


def auxiliary(train,test):
    selected=[];labels=[];feeds={}
    template=test[test.recording.eq('20260919_1100') & test.unit.isin(reconstruction.UNITS)].copy()
    for name,g in train[train.weekday.eq('Sat') & train.day.eq('2026-09-05')].groupby('recording'):
        mask=template.copy();protocol='clock_aligned'
        if int(g.minute_of_day.max())<int(template.minute_of_day.min()):
            offset=int(g.minute_of_day.min())-int(template.minute_of_day.min())
            mask['minute_of_day']+=offset
            mask['now_minute_of_day']=mask.now_minute_of_day+offset
            protocol='relative_start_morning'
        visible=mask[mask.id.isna() & mask.unit.ne('cam_D')][['unit','minute_of_day']]
        feed=visible.merge(g[['recording','unit','minute_of_day','people']],on=['unit','minute_of_day'],how='inner')
        feeds.update(table(feed))
        targets=mask[mask.kind.eq('hidden-future') & mask.unit.eq('cam_D')]
        targets=targets.merge(g[['unit','minute_of_day','people']],on=['unit','minute_of_day'],how='inner',suffixes=('','_truth'))
        for r in targets.itertuples():
            key=f'{name}:cam_D:{int(r.minute_of_day)}'
            selected.append(dict(case_id=key,recording=name,day=g.day.iloc[0],unit='cam_D',
                minute=int(r.minute_of_day),now=int(r.now_minute_of_day),
                horizon=int(r.minute_of_day-r.now_minute_of_day),scope='auxiliary',protocol=protocol))
            labels.append(dict(case_id=key,truth=float(r.people_truth)))
    return add_windows(pd.DataFrame(selected)),pd.DataFrame(labels),feeds


def summarize(frame,prefix):
    frame=frame.copy()
    for name in ['current','candidate','reference']:frame[name+'_error']=(frame[name]-frame.truth).abs()
    def row(q):
        ref=q.reference_error.mean()
        result=dict(targets=len(q),origins=q.window_id.nunique(),used=int(q.used.sum()),
            fallback=int((~q.used).sum()),used_origins=q[q.used].window_id.nunique(),
            changed=int((q.candidate-q.current).abs().gt(1e-9).sum()))
        for name in ['current','candidate','reference']:
            result[name+'_mae']=q[name+'_error'].mean()
            result[name+'_normalized']=result[name+'_mae']/ref if ref>0 else np.nan
        result['gain']=result['current_mae']-result['candidate_mae']
        result['bias_current']=(q.current-q.truth).mean()
        return result
    for groups,name in [(['day','recording'],'recording'),(['day','unit'],'camera'),
        (['day','recording','horizon_band'],'horizon'),(['day','recording','window_id'],'window')]:
        rows=[]
        for key,q in frame.groupby(groups):
            key=key if isinstance(key,tuple) else (key,)
            rows.append(dict(zip(groups,key))|row(q))
        pd.DataFrame(rows).to_csv(OUT/f'{prefix}_by_{name}.csv',index=False)
    loo=[]
    for name,g in frame.groupby('recording'):
        for w in g.window_id.unique():
            q=g[g.window_id.ne(w)]
            if len(q):loo.append(dict(recording=name,omitted_window=w,**row(q)))
    pd.DataFrame(loo).to_csv(OUT/f'{prefix}_leave_one_window_out.csv',index=False)
    frame.to_csv(OUT/f'{prefix}_scored.csv',index=False)
    return frame


def permitted_feed(feed,now):
    return feed.drop(columns=['cam_D'],errors='ignore').loc[:now]


def reconstruct_at_now(feed,now,day,model,schedule):
    # Explicitly remove D from the ENTIRE feed before slicing the prefix.
    causal=permitted_feed(feed,now)
    raw=causal.reindex(index=[now])
    x,available=reconstruction.make_features(raw,'cam_D',[day],schedule)
    count=int(available.iloc[0])
    if count<2 or 'cam_D' not in model:return np.nan,count
    return max(0.,float(model['cam_D'].predict(x)[0])),count


def values(feed,now,h,day,current_models,new_models,schedule):
    safe=permitted_feed(feed,now)
    reference,current,_=tasks.hidden_future_value(safe,'cam_D',now,h,day,schedule,
        current_models,tasks.default_config()['hidden-future'])
    anchor,count=reconstruct_at_now(safe,now,day,new_models,schedule)
    used=np.isfinite(anchor)
    candidate=max(0.,current+SPEC['weight']*(anchor-reference)) if used else current
    return reference,current,candidate,anchor,count,used


def actual_input_coverage(test):
    feeds=table(test[test.id.isna() & test.people.notna()]);rows=[]
    for r in test[test.kind.eq('hidden-future') & test.unit.eq('cam_D') & test.weekday.eq('Sat')].itertuples():
        now=int(r.now_minute_of_day)
        prefix=permitted_feed(feeds[r.recording],now)
        donors=[u for u in reconstruction.UNITS if u!='cam_D']
        count=int(prefix.reindex(index=[now],columns=donors).notna().sum(axis=1).iloc[0])
        rows.append(dict(id=r.id,now=now,exact_donors=count,supported=count>=2))
    result=pd.DataFrame(rows);result.to_csv(OUT/'actual_coverage.csv',index=False)
    return result


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'design.json').write_text(json.dumps(SPEC,indent=2),encoding='utf-8')
    for name,value in json.loads((FIXED/'case_hashes.json').read_text()).items():assert sha(FIXED/name)==value
    cases=pd.read_csv(FIXED/'cases.csv');cases=cases[cases.kind.eq('hidden-future')].copy()
    assert len(cases)==482
    cases['scope']='fixed';cases['protocol']='fixed_clock_aligned';cases=add_windows(cases)
    baseline=pd.read_csv(FIXED/'predictions.csv')[['case_id','prediction','reference']]
    detail=cases.merge(baseline,on='case_id',validate='one_to_one').rename(columns={'prediction':'current'})
    detail['candidate']=detail.current;detail['used']=False
    scored=detail.merge(pd.read_csv(FIXED/'labels.csv'),on='case_id',validate='one_to_one')
    summarize(scored,'diagnostic')
    return cases


def run():
    started=time.perf_counter();cases=prepare()
    paths=list((ROOT/'outputs').rglob('submission*.csv'))+list(FIXED.rglob('*.*'))
    paths+=list((ROOT/'src').glob('*.py'))+list((ROOT/'t05_forecast').glob('*.py'))
    paths+=list((ROOT/'models/current_baseline').rglob('*.*'))+list((ROOT/'data/raw').glob('*.csv'))
    paths+=[ROOT/'notebooks/current_baseline.ipynb']
    protected={str(p):sha(p) for p in paths if p.is_file() and p.resolve()!=Path(__file__).resolve()}
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    schedule=timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)
    aux,aux_labels,aux_feeds=auxiliary(train,test)
    aux.to_csv(OUT/'auxiliary_cases.csv',index=False);aux_labels.to_csv(OUT/'auxiliary_labels.csv',index=False)
    fixed_feeds=table(pd.read_csv(FIXED/'released_feed.csv'));feeds={**fixed_feeds,**aux_feeds}
    saturday=cases[cases.unit.eq('cam_D')].copy()
    selected=pd.concat([saturday,aux],ignore_index=True)
    outputs=[];manifest=[];checks=0
    with threadpool_limits(limits=4):
        for day,today in selected.groupby('day'):
            fitting=train[train.day.ne(day)].copy();assert not fitting.day.eq(day).any()
            poisoned=train.copy();poisoned.loc[poisoned.day.eq(day),'people']=1e8
            pd.testing.assert_frame_equal(fitting,poisoned[poisoned.day.ne(day)])
            old=tasks.fit_hidden_future(fitting,schedule,tasks.default_config()['hidden-future'])
            new,fit_manifest=reconstruction.fit_new(fitting,schedule)
            manifest.append(dict(excluded_day=day,training_dates=sorted(fitting.day.unique()),
                training_recordings=sorted(fitting.recording.unique()),reconstruction=fit_manifest))
            for (name,now),group in today.groupby(['recording','now']):
                feed=feeds[name].drop(columns=['cam_D'],errors='ignore')
                assert 'cam_D' not in feed
                for r in group.itertuples():
                    now=int(now);h=int(r.horizon)
                    original=values(feed,now,h,day,old,new,schedule)
                    only_d=feed.copy();only_d['cam_D']=1e8+np.arange(len(only_d))
                    only_future=feed.copy();only_future.loc[only_future.index>now]=1e9
                    pd.testing.assert_frame_equal(permitted_feed(feed,now),permitted_feed(only_d,now))
                    pd.testing.assert_frame_equal(permitted_feed(feed,now),permitted_feed(only_future,now))
                    changed=feed.copy();changed['cam_D']=1e8+np.arange(len(changed))
                    changed.loc[changed.index>now]=1e9
                    altered=values(changed,now,h,day,old,new,schedule)
                    np.testing.assert_allclose(original[:4],altered[:4],rtol=0,atol=1e-10,equal_nan=True)
                    assert original[4:]==altered[4:];checks+=1
                    outputs.append(dict(**r._asdict(),reference=original[0],current=original[1],
                        candidate=original[2],new_anchor=original[3],exact_donors=original[4],used=original[5]))
    predictions=pd.DataFrame(outputs)
    # Original fixed references must remain the scoring denominator, not new D.
    saved=pd.read_csv(FIXED/'predictions.csv').set_index('case_id')
    fixed=predictions[predictions.scope.eq('fixed')]
    for col,oldcol in [('reference','reference'),('current','prediction')]:
        np.testing.assert_allclose(fixed[col],saved.loc[fixed.case_id,oldcol],rtol=0,atol=1e-9)
    predictions.to_csv(OUT/'predictions.csv',index=False)
    actual=actual_input_coverage(test)
    (OUT/'fit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    labels=pd.concat([pd.read_csv(FIXED/'labels.csv'),aux_labels],ignore_index=True)
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    summarize(scored[scored.scope.eq('fixed')],'saturday')
    summarize(scored[scored.scope.eq('auxiliary')],'auxiliary')
    # Friday is an unchanged control, never pooled with Saturday D.
    friday=pd.read_csv(OUT/'diagnostic_scored.csv').query("unit!='cam_D'")
    summarize(friday,'friday_unchanged')
    decision=decide()
    extra=full_and_actual(fixed,train,test,schedule) if decision['accepted'] else {}
    assert all(sha(p)==v for p,v in protected.items())
    result=dict(**decision,**extra,runtime_seconds=time.perf_counter()-started,
        prediction_invariance_checks=checks,t05_fits=0,t05_inference=0,
        separate_D_and_future_prefix_checks=2*checks,
        actual_targets=len(actual),actual_input_supported=int(actual.supported.sum()),
        downstream_reconstruction_feature_fits=0,date_exclusion_checks=len(manifest),
        model_fits=dict(prayer=2,ordinary_direct=2,linear_hidden_sets=2,nonlinear_camera_models=sum(len(m['reconstruction']) for m in manifest)),
        protected_files=len(protected),protected_files_unchanged=True,protected_sha256=protected)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'saturday_by_recording.csv').to_string(index=False))
    print(pd.read_csv(OUT/'auxiliary_by_recording.csv').to_string(index=False))
    print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))


def decide():
    primary=pd.read_csv(OUT/'saturday_by_recording.csv').iloc[0]
    windows=pd.read_csv(OUT/'saturday_by_window.csv');loo=pd.read_csv(OUT/'saturday_leave_one_window_out.csv')
    aux=pd.read_csv(OUT/'auxiliary_by_recording.csv');aw=pd.read_csv(OUT/'auxiliary_by_window.csv');al=pd.read_csv(OUT/'auxiliary_leave_one_window_out.csv')
    supported=aux[aux.used.ge(20)&aux.used_origins.ge(2)]
    horizon=pd.read_csv(OUT/'saturday_by_horizon.csv')
    major=horizon[horizon.targets.ge(10)&horizon.origins.ge(2)]
    accepted=bool(primary.gain>=.01*primary.current_mae and primary.used>=.5*primary.targets
        and windows.gain.gt(0).sum()>=3 and loo.gain.gt(0).all() and len(supported)>=2
        and supported.gain.gt(0).all() and al[al.recording.isin(supported.recording)].gain.gt(0).all()
        and all(aw[aw.recording.eq(n)].gain.gt(0).sum()>=2 for n in supported.recording)
        and (major.candidate_mae-major.current_mae<=np.maximum(.5,.1*major.current_mae)).all())
    return dict(accepted=accepted,primary_winning_origins=int(windows.gain.gt(0).sum()),
        primary_origins=len(windows),primary_loo_gain_min=float(loo.gain.min()),primary_loo_gain_max=float(loo.gain.max()),
        supported_auxiliary_recordings=supported.recording.tolist(),
        auxiliary_winning_origins={n:int(g.gain.gt(0).sum()) for n,g in aw.groupby('recording')},
        decision='Accept separate candidate' if accepted else 'Reject without tuning')


def full_and_actual(updates,train,test,schedule):
    """Conditional full score from protected baseline predictions, never T05 refit."""
    import local_score
    baseline=pd.read_csv(FIXED/'predictions.csv');candidate=baseline.copy()
    replace=updates.set_index('case_id').candidate;mask=candidate.case_id.isin(replace.index)
    candidate.loc[mask,'prediction']=candidate.loc[mask,'case_id'].map(replace)
    previous_out=local_score.OUT;scores={}
    for name,frame in [('full_baseline',baseline),('full_candidate',candidate)]:
        folder=OUT/name;folder.mkdir(exist_ok=True)
        (folder/'labels.csv').write_bytes((FIXED/'labels.csv').read_bytes())
        frame.to_csv(folder/'predictions.csv',index=False);local_score.OUT=folder
        with contextlib.redirect_stdout(io.StringIO()):scores[name]=local_score.score(frame,0)
    local_score.OUT=previous_out
    new,_=reconstruction.fit_new(train,schedule)
    old=tasks.fit_hidden_future(train,schedule,tasks.default_config()['hidden-future'])
    feed=table(test[test.id.isna()&test.people.notna()]);references=pd.read_csv(ROOT/'data/raw/sample_submission.csv').set_index('id').people
    with BASE.open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
    original=[r.copy() for r in rows];by_id={r['id']:r for r in rows};changed=[]
    for r in test[test.kind.eq('hidden-future')&test.unit.eq('cam_D')&test.weekday.eq('Sat')].itertuples():
        now=int(r.now_minute_of_day);h=int(r.minute_of_day)-now
        _,baseline_value,_=tasks.hidden_future_value(feed[r.recording],r.unit,now,h,r.day,schedule,old,
            tasks.default_config()['hidden-future'],reference=float(references[r.id]))
        assert abs(baseline_value-float(by_id[r.id]['people']))<=1e-9
        anchor,count=reconstruct_at_now(feed[r.recording],now,r.day,new,schedule)
        if np.isfinite(anchor):
            value=max(0.,baseline_value+.25*(anchor-float(references[r.id])))
            if abs(value-float(by_id[r.id]['people']))>1e-9:
                by_id[r.id]['people']=str(float(value));changed.append(r.id)
    path=ROOT/'outputs/submission_t05_hidden_future_d_anchor_v1.csv'
    if path.exists():raise FileExistsError(f'Preserving {path}')
    assert len(rows)==4457 and [r['id'] for r in rows]==pd.read_csv(ROOT/'data/raw/sample_submission.csv').id.tolist()
    assert len({r['id'] for r in rows})==4457
    assert all(np.isfinite(float(r['people'])) and float(r['people'])>=0 for r in rows)
    assert all(a['people']==b['people'] for a,b in zip(original,rows) if a['id'] not in set(changed))
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(rows)
    return dict(full_scores=scores,submission=str(path),changed_rows=len(changed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnose-only',action='store_true');args=parser.parse_args()
    if args.diagnose_only:prepare()
    else:run()
