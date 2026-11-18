"""P39: ONE unchanged Ridge family, expanded fitting rows, no T05 optimization.

Historical stress: each evaluated DATE excluded from fitting as before.
Release-matched: all 11/12 recording rows replaced by their fixed released feed;
all task masks applied globally before fitting. Extra rows are genuine released
18/19 A-H observations. Reference/blend/features are held fixed between methods.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import baseline_tasks as tasks
from pipeline import load,table
from evaluate_baseline_task import fixed_benchmark
from reliable_evaluation import annotate_cases,case_fingerprints
import v2_hidden
from v2_missing import interp_at

UNITS=[f'cam_{u}' for u in 'ABCDEFGH']
EVAL_NAMES=['20260911_1100','20260912_1100']
BASELINE=ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
RULE='For a forecast or hidden-future gap, use no row from after its `now`: not the rest of that recording, nor later recordings of the same day. Models may be trained on any released row, including rows after a gap’s now; only the inputs used for that gap are limited. Hidden and missing gaps may use the whole released table.'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_snapshot():
    paths=[ROOT/'notebooks/current_baseline.ipynb']
    paths += [ROOT/'src'/n for n in ('baseline_tasks.py','baseline_workflow.py','current_baseline.py',
        'v2_hidden.py','v2_missing.py','v2_experiments.py','prayer_experiments.py')]
    paths += list((ROOT/'models').rglob('*.*'))
    paths += list((ROOT/'outputs').glob('submission*.csv'))
    for folder,names in [('local_score',['cases.csv','labels.csv','released_feed.csv','case_hashes.json']),
        ('weekly_forecast_diagnostic',['cases.csv','labels.csv','case_hashes.json'])]:
        paths += [ROOT/'outputs'/folder/n for n in names]
    return {str(p):sha(p) for p in paths if p.is_file()}


def training_rows(frame):
    cols=['recording','day','unit','minute_of_day','people']
    return frame.loc[frame.unit.isin(UNITS)&frame.people.notna(),cols].copy()


def row_keys(frame):
    return set(zip(frame.recording,frame.unit,frame.minute_of_day.astype(int)))


def expand(train,test):
    extra=training_rows(test[test.id.isna() & test.people.notna()])
    assert not row_keys(train)&row_keys(extra)
    return pd.concat([training_rows(train),extra],ignore_index=True),extra


def release_training(train,test,feeds):
    # Replace whole pseudo-test recordings, not just the scored target rows:
    # even naturally unavailable/unscored target cells must stay unavailable.
    pieces=[training_rows(train[~train.recording.isin(EVAL_NAMES)])]
    day_map=train.groupby('recording').day.first().to_dict()
    for name in EVAL_NAMES:
        f=feeds[name].rename_axis('minute_of_day').reset_index().melt(
            id_vars='minute_of_day',var_name='unit',value_name='people').dropna(subset=['people'])
        f['recording']=name;f['day']=day_map[name]
        pieces.append(training_rows(f))
    base=pd.concat(pieces,ignore_index=True)
    blocked=set()
    for test_name,eval_name in [('20260918_1100',EVAL_NAMES[0]),('20260919_1100',EVAL_NAMES[1])]:
        masks=test[test.recording.eq(test_name)&test.id.notna()&test.unit.isin(UNITS)]
        blocked|={(eval_name,r.unit,int(r.minute_of_day)) for r in masks.itertuples()}
    assert not row_keys(base)&blocked
    expanded,extra=expand(base,test)
    assert not row_keys(expanded)&blocked
    for name in EVAL_NAMES:
        expected={(name,u,int(m)) for u in feeds[name] for m in feeds[name][u].dropna().index}
        assert row_keys(base[base.recording.eq(name)])==expected
    return base,expanded,extra,blocked


def example_counts(frame):
    full=pd.concat(table(frame).values())
    return {u:int((full[u].notna() & full[[c for c in full if c not in masked]].notna().sum(axis=1).ge(2)).sum())
            for u,masked in v2_hidden.PATTERNS.items()}


def predict_pair(cases,feeds,base_model,candidate_model,cfg):
    rows=[]
    for r in cases.itertuples():
        t=feeds[r.recording].copy();minute=int(r.minute)
        for u in v2_hidden.PATTERNS[r.unit]:t[u]=np.nan
        ref,current,_=tasks.hidden_value(t,r.unit,minute,base_model,cfg)
        _,candidate,_=tasks.hidden_value(t,r.unit,minute,candidate_model,cfg,reference=ref)
        poison=t.copy()
        for u in v2_hidden.PATTERNS[r.unit]:poison[u]=1e8
        _,check,_=tasks.hidden_value(poison,r.unit,minute,candidate_model,cfg,reference=ref)
        np.testing.assert_allclose(candidate,check,rtol=0,atol=1e-10)
        donors=base_model['linear'][r.unit]['donors']
        available=sum(np.isfinite(interp_at(t[u],minute)) for u in donors if u in t)
        rows.append(dict(case_id=r.case_id,current=current,candidate=candidate,reference=ref,
            donors=available,model=available>=2,fallback=available<2))
    return cases.merge(pd.DataFrame(rows),on='case_id',validate='one_to_one')


def score(predictions,labels,out):
    out.mkdir(parents=True,exist_ok=False)
    predictions.to_csv(out/'predictions.csv',index=False)
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    assert len(scored)==len(predictions)
    for method in ('current','candidate','reference'):
        assert np.isfinite(scored[method]).all()
        scored[method+'_error']=(scored[method]-scored.truth).abs()
    scored['delta']=scored.candidate_error-scored.current_error
    scored.to_csv(out/'scored.csv',index=False)
    def aggregate(groups):
        summary=scored.groupby(groups).agg(targets=('case_id','size'),
            windows=('window_id','nunique'),current_mae=('current_error','mean'),
            candidate_mae=('candidate_error','mean'),reference_mae=('reference_error','mean'),
            signed_bias_current=('current',lambda s:float((s-scored.loc[s.index,'truth']).mean())),
            signed_bias_candidate=('candidate',lambda s:float((s-scored.loc[s.index,'truth']).mean())),
            model=('model','sum'),fallback=('fallback','sum')).reset_index()
        summary['current_normalized']=summary.current_mae/summary.reference_mae
        summary['candidate_normalized']=summary.candidate_mae/summary.reference_mae
        summary['delta']=summary.candidate_mae-summary.current_mae
        return summary
    reports={}
    for groups,name in [(['day'],'date'),(['day','unit'],'date_camera'),
        (['day','window_id'],'window'),(['day','unit','window_id'],'camera_window')]:
        reports[name]=aggregate(groups);reports[name].to_csv(out/f'by_{name}.csv',index=False)
    loo=[]
    for day,g in scored.groupby('day'):
        for window in g.window_id.unique():
            kept=g[g.window_id.ne(window)]
            loo.append(dict(day=day,omitted_window=window,targets=len(kept),delta=float(kept.delta.mean())))
    pd.DataFrame(loo).to_csv(out/'leave_one_window_out.csv',index=False)
    best=scored.groupby('window_id').delta.sum().nsmallest(2).index
    without=scored[~scored.window_id.isin(best)]
    wins=reports['window'].groupby('day').delta.agg(wins=lambda s:int((s<0).sum()),
        losses=lambda s:int((s>0).sum()),total='size').reset_index()
    wins.to_csv(out/'window_counts.csv',index=False)
    cam=reports['date_camera'];dates=reports['date'];sensitivity=pd.DataFrame(loo)
    # Fixed BEFORE evaluation: both dates, majority windows each, all leave-one
    # windows, two-best removal; no material camera regression in either protocol.
    passed=bool((dates.delta<0).all() and (wins.wins>wins.total/2).all()
        and (sensitivity.delta<0).all() and without.delta.mean()<0
        and (cam.delta<=np.maximum(.15,.05*cam.current_mae)).all())
    info=dict(targets=len(scored),windows=int(scored.window_id.nunique()),
        current_mae=float(scored.current_error.mean()),candidate_mae=float(scored.candidate_error.mean()),
        without_two_best_windows_delta=float(without.delta.mean()),omitted_best_windows=list(best),
        max_leave_one_window_delta=float(sensitivity.delta.max()),passed_fixed_gate=passed)
    (out/'summary.json').write_text(json.dumps(info,indent=2))
    return info


def audit(train,test):
    expanded,extra=expand(train,test)
    return dict(training_rows=len(train),released_test_rows=int((test.id.isna()&test.people.notna()).sum()),
        extra_existing_AH_rows=len(extra),current_hidden_examples=example_counts(training_rows(train)),
        additional_hidden_examples=example_counts(extra),expanded_hidden_examples=example_counts(expanded),
        extra_by_date_camera=extra.groupby(['day','unit']).size().reset_index(name='rows').to_dict('records'),
        target_counts=test[test.id.notna()].groupby(['kind','unit']).size().reset_index(name='targets').to_dict('records'),
        t05='Production loads 12 frozen covariances; recipe uses opposite test day + Sep11/12, not a verified creation log. Template completion fits coefficients from other recordings, then applies them to historical template donors.',
        hidden='Production fits on all train dates, not local date-held-out folds; test only inference. Verified expansion opportunity.',
        hidden_future='Production trains raw-count absolute-error HGB + prayer variant on all train dates, donor features prefix-cut at each own example origin; test only inference.',
        missing='Production fits donor deviations from train only; released test supplies observed endpoints/donor inputs, not calibration fitting.',
        im='I-M forecast is nonparametric; no training. I-M missing retains reference. No transfer attempted.')


def build_candidate(train,test,cfg,out,expected_protected):
    expanded,_=expand(train,test);models=tasks.fit_hidden(expanded,cfg)
    feeds=table(test[test.id.isna()&test.people.notna()])
    with BASELINE.open(newline='') as handle:original=list(csv.DictReader(handle))
    with (ROOT/'data/raw/sample_submission.csv').open(newline='') as handle:
        refs={r['id']:float(r['people']) for r in csv.DictReader(handle)}
    targets=test[test.id.notna()&test.kind.eq('hidden')]
    predicted=tasks.predict_hidden(models,targets,feeds,refs,cfg,None).set_index('id').people.to_dict()
    allowed=set(targets.id);rows=[];changed=[]
    for row in original:
        value=str(float(predicted[row['id']])) if row['id'] in allowed else row['people']
        if value!=row['people']:changed.append(row['id'])
        rows.append(dict(id=row['id'],people=value))
    assert len(rows)==len(set(r['id'] for r in rows))==4457
    assert [r['id'] for r in rows]==pd.read_csv(ROOT/'data/raw/sample_submission.csv').id.tolist()
    assert all(np.isfinite(float(r['people'])) and float(r['people'])>=0 for r in rows)
    assert set(changed)<=allowed
    assert all(a['people']==b['people'] for a,b in zip(rows,original) if a['id'] not in allowed)
    path=out/'submission_hidden_released_training_v1.csv'
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(rows)
    assert expected_protected==protected_snapshot()
    (out/'candidate_verification.json').write_text(json.dumps(dict(path=str(path),ids=len(rows),
        allowed_rows=len(allowed),changed_text_rows=len(changed),changed_ids=changed,
        outside_hidden_text_unchanged=True,sha256=sha(path),d_only=False),indent=2))


def run(out,allow_candidate=False):
    started=time.perf_counter();out=Path(out)
    out.mkdir(parents=True,exist_ok=False);protected=protected_snapshot()
    cfg=tasks.default_config()['hidden'];assert not cfg['d_only']
    design=dict(rule=RULE,rule_source='User verbatim P39 and Downloads/Pasted markdown(1).md, competition-specific bullet 3',
        method='Existing Ridge/dropout Ridge, identical features, alpha=3000, B=.75/G=.50/D=.50; add only genuinely released test A-H fitting rows',
        config=cfg,extra_recordings=['20260918_1100','20260919_1100'],
        release_protocol='Fixed Sep11/12 masks for ALL FOUR tasks applied globally. Replace those whole training recordings with their released feeds. Baseline uses this sanitized train, candidate additionally uses genuine Sep18/19 released A-H. Same baseline-trained reference for both.',
        strict_protocol='Unchanged fixed benchmark; exclude evaluated DATE entirely in each fold; candidate additionally uses released Sep18/19 A-H. Other fold labels remain standard historical training data, not joint release-protocol fitting.',
        acceptance='In BOTH protocols: both dates improve; majority of disjoint clock windows improve each date; every leave-one-window delta negative; gain survives removing two best windows; per-camera regression <= max(.15 MAE,5% of current). No tuning.',
        score='Project-specific MAE/reference MAE on identical rows; no cross-protocol aggregate.',
        candidate_if_pass=allow_candidate)
    # Split/settings/gate fixed and written BEFORE loading new evaluation errors.
    (out/'design.json').write_text(json.dumps(design,indent=2))
    with contextlib.redirect_stdout(io.StringIO()):
        train,test,_,_=load(ROOT/'data/raw')
        cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
    assert case_fingerprints(cases,labels)==json.loads((ROOT/'models/local_evaluation/cases_manifest.json').read_text())['historical']
    cases=annotate_cases(cases,feeds);hidden=cases[cases.kind.eq('hidden')].copy()
    (out/'audit.json').write_text(json.dumps(audit(train,test),indent=2))
    base,expanded,extra,blocked=release_training(train,test,feeds)
    # Replacing every masked answer in original train cannot alter either fitting
    # table. Inference similarly receives the released feed, never raw train.
    poisoned=train.copy();hit=[key in blocked for key in zip(poisoned.recording,poisoned.unit,poisoned.minute_of_day)]
    poisoned.loc[hit,'people']=1e8
    safe_base,safe_expanded,_,_=release_training(poisoned,test,feeds)
    pd.testing.assert_frame_equal(base,safe_base);pd.testing.assert_frame_equal(expanded,safe_expanded)
    fitting=[];predictions={}
    with threadpool_limits(limits=4):
        for protocol in ('release_matched','strict_date_held_out'):
            records=[]
            folds=[('joint',hidden,base,expanded)] if protocol=='release_matched' else [
                (day,g,training_rows(train[train.day.ne(day)]),expand(train[train.day.ne(day)],test)[0])
                for day,g in hidden.groupby('day')]
            for fold,targets,old,new in folds:
                if protocol=='strict_date_held_out':assert not old.day.eq(fold).any() and not new.day.eq(fold).any()
                old_model=tasks.fit_hidden(old,cfg);new_model=tasks.fit_hidden(new,cfg)
                records.append(predict_pair(targets,feeds,old_model,new_model,cfg))
                fitting.append(dict(protocol=protocol,fold=fold,current_rows=len(old),expanded_rows=len(new),
                    current_examples=example_counts(old),expanded_examples=example_counts(new),
                    training_recordings=sorted(new.recording.unique().tolist()),
                    excluded_date=fold if protocol=='strict_date_held_out' else None))
            predictions[protocol]=pd.concat(records,ignore_index=True)
            predictions[protocol].to_csv(out/f'{protocol}_predictions_before_scoring.csv',index=False)
    (out/'fitting_manifest.json').write_text(json.dumps(fitting,indent=2))
    # All comparisons are now fixed before joining scorer labels.
    results={p:score(pred,labels,out/p) for p,pred in predictions.items()}
    # Confirm unchanged strict baseline numerically against saved evidence, never
    # use that file as an alternative predictor.
    historical=ROOT/'outputs/local_score/predictions.csv'
    if historical.exists():
        paired=predictions['strict_date_held_out'].merge(pd.read_csv(historical)[['case_id','prediction']],on='case_id')
        np.testing.assert_allclose(paired.current,paired.prediction,rtol=0,atol=1e-9)
    real=[];real_feeds=table(test[test.id.isna()&test.people.notna()])
    for r in test[test.kind.eq('hidden')].itertuples():
        donors=[u for u in UNITS if u not in v2_hidden.PATTERNS[r.unit]]
        available=sum(np.isfinite(interp_at(real_feeds[r.recording][u],int(r.minute_of_day))) for u in donors if u in real_feeds[r.recording])
        real.append(dict(day=r.day,unit=r.unit,donors=available,model=available>=2))
    pd.DataFrame(real).groupby(['day','unit']).agg(targets=('model','size'),model=('model','sum'),
        min_donors=('donors','min'),mean_donors=('donors','mean')).to_csv(out/'actual_target_coverage.csv')
    passed=all(r['passed_fixed_gate'] for r in results.values())
    if passed and allow_candidate:build_candidate(train,test,cfg,out,protected)
    assert protected==protected_snapshot()
    report=dict(results=results,accepted=passed,candidate_generated=bool(passed and allow_candidate),
        protected_files_unchanged=len(protected),t05_fits=0,one_candidate_family='Existing regularized linear reconstruction; training rows only',
        code_sha256=sha(__file__),input_hashes={n:sha(ROOT/'data/raw'/n) for n in ('train.csv','test.csv','sample_submission.csv','recordings.csv')},
        repeated_dates='All dates have been examined previously; no fresh independent validation',
        runtime_seconds=time.perf_counter()-started)
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--build-if-credible',action='store_true');args=parser.parse_args()
    run(args.output_dir,args.build_if_credible)
