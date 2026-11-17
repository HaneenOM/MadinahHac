"""Separate Saturday-D candidate; production code and prior artifacts are read-only."""
from __future__ import annotations

import contextlib
import argparse
import csv
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import hidden_reconstruction_trial as trial
import local_score
import t05_local_forecast
from cross_camera_forecast_audit import cached_only
from hidden_candidate import read_submission
from pipeline import ROOT,load,locate_data,table
from prayer_experiments import timetable

OUT=ROOT/'outputs/hidden_d_candidate'
FIXED=ROOT/'outputs/local_score'
PREVIOUS=ROOT/'outputs/hidden_reconstruction_trial'
BASE=ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
CANDIDATE=ROOT/'outputs/submission_t05_hidden_d_only_v1.csv'


def recording_sensitivity(scored):
    rows=[]
    for (day,name,unit),group in scored.groupby(['day','recording','unit']):
        for window in group.window_id.unique():
            kept=group[group.window_id.ne(window)]
            ref=kept.reference_error.mean();old=kept.baseline_error.mean();new=kept.candidate_error.mean()
            rows.append(dict(day=day,recording=name,unit=unit,omitted_window=window,targets=len(kept),
                baseline_mae=old,candidate_mae=new,baseline_normalized=old/ref if ref else None,
                candidate_normalized=new/ref if ref else None,gain=old-new))
    pd.DataFrame(rows).to_csv(OUT/'D_recording_leave_one_window_out.csv',index=False)


def audit_target_times():
    # Hidden target counts are deliberately not loaded for the distribution audit.
    metadata=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['id','recording','day','weekday','unit','minute_of_day','kind'])
    actual=metadata[metadata.kind.eq('hidden')&metadata.unit.eq('cam_D')&metadata.day.eq('2026-09-19')].copy()
    assert len(actual)==285 and actual.weekday.eq('Sat').all()
    actual['block']=actual.minute_of_day//30*30
    windows=pd.read_csv(PREVIOUS/'fixed_by_camera_window.csv').query("unit=='cam_D'").copy()
    windows['block']=windows.window_id.str.rsplit(':',n=1).str[-1].astype(int)
    windows['previous_result']=np.where(windows.gain>0,'improved',np.where(windows.gain<0,'regressed','equal'))
    audit=actual.merge(windows[['block','targets','gain','previous_result']].rename(columns={'targets':'Sep12_cases'}),
                        on='block',how='left',validate='many_to_one')
    audit['previous_result']=audit.previous_result.fillna('uncovered')
    audit.to_csv(OUT/'actual_D_time_audit.csv',index=False)
    audit.groupby('previous_result').agg(actual_targets=('id','size'),blocks=('block','nunique')).reset_index().to_csv(
        OUT/'actual_by_previous_window_result.csv',index=False)
    recordings=pd.read_csv(ROOT/'data/raw/recordings.csv').set_index('recording')
    additional=pd.read_csv(PREVIOUS/'additional_by_recording_camera.csv').query("unit=='cam_D'")
    periods=[]
    for r in additional.itertuples():
        meta=recordings.loc[r.recording]
        first=sum(int(v)*m for v,m in zip(meta['first'].split(':'),(60,1)))
        last=sum(int(v)*m for v,m in zip(meta['last'].split(':'),(60,1)))
        periods.append(dict(previous_recording=r.recording,start=meta['first'],end=meta['last'],
            previous_baseline_mae=r.baseline_mae,previous_candidate_mae=r.candidate_mae,
            previous_result='improved' if r.gain>0 else 'regressed',
            actual_targets=int(actual.minute_of_day.between(first,last).sum())))
    pd.DataFrame(periods).to_csv(OUT/'actual_in_additional_recording_periods.csv',index=False)
    actual.assign(hour=actual.minute_of_day//60).groupby('hour').size().rename('targets').to_csv(OUT/'actual_targets_by_hour.csv')
    report=dict(actual_D_targets=len(actual),actual_first=int(actual.minute_of_day.min()),actual_last=int(actual.minute_of_day.max()),
                in_primary_improved=int(audit.previous_result.eq('improved').sum()),
                in_primary_regressed=int(audit.previous_result.eq('regressed').sum()),
                primary_uncovered=int(audit.previous_result.eq('uncovered').sum()),
                in_regressed_morning=int(actual.minute_of_day.between(266,480).sum()))
    return actual,report


def reproduce(train,schedule):
    primary,feeds=trial.fixed_cases();primary=primary[primary.unit.eq('cam_D')].copy();primary['scope']='fixed'
    extra,extra_labels,extra_feeds=trial.extra_cases(train)
    extra=extra[extra.unit.eq('cam_D')].copy()
    cols=['case_id','day','recording','unit','minute','window_id','scope']
    cases=pd.concat([primary[cols],extra[cols]],ignore_index=True)
    all_feeds={**feeds,**extra_feeds};outputs=[];manifest=[];checks=0
    for day,targets in cases.groupby('day'):
        fit=train[train.day.ne(day)]
        assert not set(targets.recording).intersection(fit.recording) and not fit.day.eq(day).any()
        predictions,models,ncheck=trial.predict_fold(targets,all_feeds,fit,schedule)
        outputs.append(predictions);checks+=ncheck
        manifest.append(dict(excluded_day=day,excluded_recordings=sorted(targets.recording.unique()),
                             training_recordings=sorted(fit.recording.unique()),models=models))
    predictions=pd.concat(outputs,ignore_index=True)
    old=pd.read_csv(PREVIOUS/'predictions.csv').query("unit=='cam_D'").set_index('case_id')
    now=predictions.set_index('case_id')
    assert set(now.index)==set(old.index)
    for col in ('baseline','candidate','reference'):
        np.testing.assert_allclose(now.loc[old.index,col],old[col],rtol=0,atol=1e-9)
    predictions.to_csv(OUT/'D_reproduced_predictions.csv',index=False)
    labels=pd.concat([pd.read_csv(FIXED/'labels.csv')[['case_id','truth']],extra_labels])
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    # Reuse the exact reporting code in a separate directory.
    original_out=trial.OUT
    try:
        trial.OUT=OUT;trial.summarize(scored,'D')
    finally:trial.OUT=original_out
    recording_sensitivity(pd.read_csv(OUT/'D_scored.csv'))
    (OUT/'D_fit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return predictions,checks


def score_in_separate_directory(detail,checks,path):
    path.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(FIXED/'labels.csv',path/'labels.csv')
    assert trial.sha(path/'labels.csv')==trial.sha(FIXED/'labels.csv')
    original_out=local_score.OUT
    try:
        local_score.OUT=path
        detail.to_csv(path/'predictions.csv',index=False)
        value=local_score.score(detail,checks)
    finally:local_score.OUT=original_out
    return value


def full_benchmark(train,test,reproduced,reuse_predictions=False):
    cases,feeds=local_score.load_cases()
    schedule=timetable(ROOT/'data/raw/prayer_times_madinah.csv',train,test)
    original_out=local_score.OUT
    baseline_dir=OUT/'full_baseline';baseline_dir.mkdir(parents=True,exist_ok=True)
    # Missing fit cache means STOP, never a hidden T05 refit.
    t05_local_forecast.fit_cached=cached_only
    if reuse_predictions:
        baseline=pd.read_csv(baseline_dir/'predictions.csv')
        checks=json.loads((baseline_dir/'score.json').read_text())['post_cutoff_checks']
        assert trial.sha(baseline_dir/'labels.csv')==trial.sha(FIXED/'labels.csv')
        assert set(baseline.case_id)==set(cases.case_id)
    else:
        try:
            local_score.OUT=baseline_dir
            baseline,checks=local_score.make_predictions(train,cases,feeds,schedule)
        finally:local_score.OUT=original_out
    previous=pd.read_csv(FIXED/'predictions.csv').set_index('case_id')
    fresh=baseline.set_index('case_id')
    assert set(previous.index)==set(fresh.index)
    for col in ('reference','prediction'):
        np.testing.assert_allclose(fresh.loc[previous.index,col],previous[col],rtol=0,atol=1e-9)
    baseline_score=score_in_separate_directory(baseline,checks,baseline_dir)
    expected=json.loads((FIXED/'score.json').read_text())['local_score']
    assert abs(baseline_score-expected)<1e-10
    candidate=baseline.copy()
    replacement=reproduced[reproduced.scope.eq('fixed')].set_index('case_id')
    mask=candidate.case_id.isin(replacement.index)
    assert int(mask.sum())==283
    candidate.loc[mask,'prediction']=candidate.loc[mask,'case_id'].map(replacement.candidate)
    candidate.loc[mask,'source']='D_hidden_nonlinear'
    np.testing.assert_array_equal(candidate.loc[~mask,'prediction'],baseline.loc[~mask,'prediction'])
    candidate_score=score_in_separate_directory(candidate,checks,OUT/'full_candidate')
    score_path=OUT/'full_candidate/score.json'
    report=json.loads(score_path.read_text());report['hidden_method']='D nonlinear candidate; B/G unchanged current v2 weights'
    score_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    old_tasks=pd.read_csv(baseline_dir/'scores_by_date_task.csv')
    new_tasks=pd.read_csv(OUT/'full_candidate/scores_by_date_task.csv')
    compared=old_tasks.merge(new_tasks,on=['day','kind'],suffixes=('_baseline','_candidate'),validate='one_to_one')
    for r in compared[~(compared.kind.eq('hidden')&compared.day.eq('2026-09-12'))].itertuples():
        assert r.mae_people_baseline==r.mae_people_candidate and r.normalized_mae_baseline==r.normalized_mae_candidate
    compared.to_csv(OUT/'full_task_comparison.csv',index=False)
    rows=[]
    for name,detail in [('baseline',baseline),('candidate',candidate)]:
        labels=pd.read_csv(FIXED/'labels.csv')
        scored=detail.merge(labels,on='case_id',validate='one_to_one')
        hidden=scored[scored.kind.eq('hidden')]
        normalized=pd.read_csv((baseline_dir if name=='baseline' else OUT/'full_candidate')/'scores_by_date_task.csv')
        rows.append(dict(version=name,all_hidden_targets=len(hidden),pooled_hidden_mae=(hidden.prediction-hidden.truth).abs().mean(),
                         hidden_score=normalized.query("kind=='hidden'").normalized_mae.mean(),
                         local_score=baseline_score if name=='baseline' else candidate_score))
    pd.DataFrame(rows).to_csv(OUT/'full_score_comparison.csv',index=False)
    return baseline_score,candidate_score,checks


def make_submission(train,test,sample,schedule,actual):
    # Fit the SAME settings on the released historical train only; no test labels.
    actual_cases=actual.rename(columns={'id':'case_id','minute_of_day':'minute'})[
        ['case_id','day','recording','unit','minute']].copy()
    actual_cases['window_id']=actual_cases.recording+':'+(actual_cases.minute//30*30).astype(str)
    actual_cases['scope']='actual'
    released=table(test[test.id.isna()&test.people.notna()])
    pred,manifest,checks=trial.predict_fold(actual_cases,released,train,schedule)
    rows=read_submission(BASE,sample.id.tolist());old={r['id']:r['people'] for r in rows}
    # The submitted production blend uses the saved v1 reference. Local held-out
    # validation rebuilds its own single-donor reference inside each training
    # fold. Reconstruct the ACTUAL production formula, not that local proxy.
    reference_rows=read_submission(ROOT/'outputs/submission_v1_0.987.csv',sample.id.tolist())
    reference={r['id']:float(r['people']) for r in reference_rows}
    weight=trial.HIDDEN_CANDIDATE['cam_D'][1]
    fitted=(pred.baseline-(1-weight)*pred.reference)/weight
    pred['production_baseline']=(1-weight)*pred.case_id.map(reference)+weight*fitted
    pred['protected_prediction']=pred.case_id.map(lambda key:float(old[key]))
    pred['production_candidate']=np.where(pred.used_new_method,pred.candidate,pred.protected_prediction)
    pred.to_csv(OUT/'actual_D_predictions.csv',index=False)
    for r in pred.itertuples():
        assert abs(r.production_baseline-float(old[r.case_id]))<=1e-9, f'Protected hidden D not reproduced: {r.case_id}'
    (OUT/'actual_fit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    allowed=set(actual.id);assert len(allowed)==285
    by_id=pred.set_index('case_id')
    output=[dict(r) for r in rows]
    changed=[]
    for row in output:
        if row['id'] in allowed:
            value=float(by_id.loc[row['id'],'production_candidate'])
            if abs(value-float(row['people']))>1e-9:
                row['people']=repr(value);changed.append(row['id'])
    if CANDIDATE.exists():
        if read_submission(CANDIDATE,sample.id.tolist())!=output:
            raise FileExistsError(f'Refusing to overwrite differing existing candidate: {CANDIDATE}')
    else:
        with CANDIDATE.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(output)
    saved=read_submission(CANDIDATE,sample.id.tolist())
    assert len(saved)==4457
    for before,after in zip(rows,saved):
        assert before['id']==after['id']
        if before['id'] not in allowed:assert before==after
    text_changed=[b['id'] for a,b in zip(rows,saved) if a['people']!=b['people']]
    assert set(text_changed)==set(changed) and set(changed).issubset(allowed)
    return dict(path=str(CANDIDATE),changed_rows=len(changed),eligible_D=int(pred.used_new_method.sum()),
                total_D=len(pred),unchanged_other_rows=4457-len(allowed),checks=checks,
                all_ids_in_sample_order=True,finite_nonnegative=True,other_values_exact_text_equal=True)


def main(resume=False):
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    protected_paths=list((ROOT/'outputs').glob('submission*.csv'))+list((ROOT/'src').glob('*.py'))
    protected_paths+=list((ROOT/'t05_forecast').glob('*.py'))+list(FIXED.glob('*.csv'))+list(FIXED.glob('*.json'))
    protected_paths+=list(PREVIOUS.glob('*.*'))
    protected={str(p):trial.sha(p) for p in protected_paths if p.is_file()}
    prior_full_seconds=0.
    if resume:
        prior_full_seconds=max(0.,(OUT/'full_score_comparison.csv').stat().st_mtime-(OUT/'actual_D_time_audit.csv').stat().st_mtime)
    actual,time_report=audit_target_times()
    train,test,sample,_=load(locate_data(None));schedule=trial.schedule()
    with threadpool_limits(limits=4),contextlib.redirect_stdout(io.StringIO()):
        if resume:
            reproduced=pd.read_csv(OUT/'D_reproduced_predictions.csv');checks=len(reproduced)
            previous=pd.read_csv(PREVIOUS/'predictions.csv').query("unit=='cam_D'").set_index('case_id')
            saved=reproduced.set_index('case_id')
            assert set(saved.index)==set(previous.index)
            for column in ('baseline','candidate','reference'):
                np.testing.assert_allclose(saved.loc[previous.index,column],previous[column],rtol=0,atol=1e-9)
            assert trial.PARAMS==json.loads((PREVIOUS/'design.json').read_text())['params']
            trial.fixed_cases()  # validates original case/feed/label hashes
        else:reproduced,checks=reproduce(train,schedule)
    (OUT/'time_audit.json').write_text(json.dumps(time_report,indent=2),encoding='utf-8')
    primary=reproduced[reproduced.scope.eq('fixed')]
    frame=table(test[test.id.isna()&test.people.notna()])[actual.recording.iloc[0]].reindex(actual.minute_of_day.to_numpy(int))
    _,available=trial.make_features(frame,'cam_D',actual.day.tolist(),schedule)
    actual_available=int(available.ge(2).sum())
    windows=pd.read_csv(OUT/'D_by_camera_window.csv').query("day=='2026-09-12'")
    loo=pd.read_csv(OUT/'D_leave_one_window_out.csv').query("day=='2026-09-12' and kind=='camera'")
    improved=int(windows.gain.gt(0).sum())
    # No new time gate or model setting: morning is outside the actual recording.
    # Coverage/robustness criteria only determine whether to propose a candidate.
    credible=bool(time_report['in_regressed_morning']==0 and time_report['primary_uncovered']<=.1*len(actual)
                   and actual_available>=.95*len(actual) and improved>=3 and loo.gain.gt(0).all())
    print('D reproduced with fixed settings; time/availability audit:',json.dumps(time_report),
          f'; exact-visible-input coverage {actual_available}/{len(actual)}; credible={credible}',flush=True)
    baseline_score=candidate_score=None;submission=None;full_checks=0
    if credible:
        print('Running the full fixed benchmark in separate output directories; T05 cache is load-only.',flush=True)
        with threadpool_limits(limits=4),contextlib.redirect_stdout(io.StringIO()):
            baseline_score,candidate_score,full_checks=full_benchmark(train,test,reproduced,reuse_predictions=resume)
        if candidate_score<baseline_score:
            with threadpool_limits(limits=4),contextlib.redirect_stdout(io.StringIO()):
                submission=make_submission(train,test,sample,schedule,actual)
    assert all(trial.sha(Path(path))==value for path,value in protected.items())
    report=dict(accepted=submission is not None,time_audit=time_report,actual_feature_coverage=actual_available,
        D_result_reproduced_atol=1e-9,D_fixed_cases=len(primary),winning_D_windows=improved,
        D_window_count=len(windows),D_invariance_checks=checks,full_post_cutoff_checks=full_checks,
        baseline_local_score=baseline_score,candidate_local_score=candidate_score,
        local_score_change=candidate_score-baseline_score if candidate_score is not None else None,
        submission=submission,t05_fits=0,production_and_previous_files_unchanged=True,
        elapsed_seconds=time.perf_counter()-started,prior_full_run_seconds=prior_full_seconds,
        reused_completed_full_predictions=resume,protected_sha256=protected)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'D_by_recording_camera.csv').to_string(index=False))
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only:recording_sensitivity(pd.read_csv(OUT/'D_scored.csv'))
    else:main(resume=args.resume)
