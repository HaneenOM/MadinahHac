"""Saved-output stress checks only: no fitting, tuning or prediction calls."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import hidden_reconstruction_trial as trial
from pipeline import ROOT,table

OUT=ROOT/'outputs/hidden_d_stress_check'
SAVED=ROOT/'outputs/hidden_d_candidate'
FIXED=ROOT/'outputs/local_score'
META=['case_id','day','recording','unit','minute','window_id','used_new_method']


def clock(minute):return f'{int(minute)//60:02d}:{int(minute)%60:02d}'


def visible_inputs():
    """Never parse a hidden D count: read people ONLY in visible-row branches."""
    rows=[];targets=[]
    with (ROOT/'data/raw/test.csv').open(newline='',encoding='utf-8-sig') as handle:
        for r in csv.DictReader(handle):
            if r['kind']=='hidden' and r['unit']=='cam_D' and r['day']=='2026-09-19':
                targets.append(dict(case_id=r['id'],day=r['day'],recording=r['recording'],unit=r['unit'],
                                    minute=int(r['minute_of_day']),window_id=f"{r['recording']}:{int(r['minute_of_day'])//30*30}"))
            elif not r['id'] and not r['kind'] and r['unit']!='cam_D' and r['people']:
                rows.append(dict(recording=r['recording'],unit=r['unit'],minute_of_day=int(r['minute_of_day']),people=float(r['people'])))
    real=table(pd.DataFrame(rows))
    feed=pd.read_csv(FIXED/'released_feed.csv')
    historical=table(feed[feed.unit.ne('cam_D')])
    rows=[]
    with (ROOT/'data/raw/train.csv').open(newline='',encoding='utf-8-sig') as handle:
        for r in csv.DictReader(handle):
            if r['day']=='2026-09-05' and r['unit']!='cam_D' and r['people']:
                rows.append(dict(recording=r['recording'],unit=r['unit'],minute_of_day=int(r['minute_of_day']),people=float(r['people'])))
    historical.update(table(pd.DataFrame(rows)))
    return historical,real,pd.DataFrame(targets)


def features(cases,feeds,schedule):
    parts=[];raws=[]
    for name,g in cases.groupby('recording',sort=False):
        frame=feeds[name].reindex(index=g.minute.to_numpy(int))
        x,available=trial.make_features(frame,'cam_D',g.day.tolist(),schedule)
        assert 'cam_D' not in frame
        x=x.reset_index(drop=True);x.insert(0,'case_id',g.case_id.to_numpy())
        parts.append(x)
        raw=frame.reindex(columns=[u for u in trial.UNITS if u!='cam_D']).reset_index(drop=True)
        raw.insert(0,'case_id',g.case_id.to_numpy());raws.append(raw)
    return pd.concat(parts,ignore_index=True).set_index('case_id'),pd.concat(raws,ignore_index=True).set_index('case_id')


def range_comparison(reference,actual,name,actual_cases):
    numeric=[];flags=pd.DataFrame(False,index=actual.index,columns=actual.columns)
    missing_flags=pd.DataFrame(False,index=actual.index,columns=actual.columns)
    for column in actual:
        observed=reference[column].dropna();values=actual[column]
        low=float(observed.min()) if len(observed) else None
        high=float(observed.max()) if len(observed) else None
        outside=values.notna() & ((values < low-1e-10)|(values > high+1e-10)) if len(observed) else values.notna()
        unseen_missing=values.isna() & (not reference[column].isna().any())
        flags[column]=outside;missing_flags[column]=unseen_missing
        numeric.append(dict(reference=name,feature=column,reference_n=len(reference),reference_finite=len(observed),
            evaluated_min=low,evaluated_max=high,actual_min=values.min(),actual_max=values.max(),
            outside_range=int(outside.sum()),unseen_missing=int(unseen_missing.sum())))
    pd.DataFrame(numeric).to_csv(OUT/f'{name}_feature_ranges.csv',index=False)
    level=[c for c in actual if c.endswith('_log')]
    share=[c for c in actual if c.endswith('_share')]
    avail=[c for c in actual if c.endswith('_seen') or c=='available']
    temporal=[c for c in actual if c in ('clock_sin','clock_cos','friday','dhuhr_offset','nearest_prayer')]
    row=pd.DataFrame(index=actual.index)
    row['outside_any']=flags.any(axis=1)
    row['outside_level']=flags[level].any(axis=1)
    row['outside_composition']=flags[share].any(axis=1)
    row['outside_availability']=flags[avail].any(axis=1)
    row['outside_time_prayer']=flags[temporal].any(axis=1)
    row['unseen_missing']=missing_flags.any(axis=1)
    seen=[c for c in actual if c.endswith('_seen')]
    patterns=set(map(tuple,reference[seen].to_numpy()))
    row['unseen_availability_pattern']=[tuple(r) not in patterns for r in actual[seen].to_numpy()]
    row=row.join(actual_cases.set_index('case_id')[['minute','used_new_method']])
    row.reset_index().to_csv(OUT/f'{name}_actual_range_flags.csv',index=False)
    return dict(reference=name,reference_targets=len(reference),actual_targets=len(actual),
                **{c:int(row[c].sum()) for c in row.columns if c not in ('minute','used_new_method')},
                corrected_outside_any=int(row.loc[row.used_new_method,'outside_any'].sum()))


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    protected_paths=list((ROOT/'outputs').glob('submission*.csv'))+list((ROOT/'src').glob('*.py'))
    protected_paths+=list((ROOT/'t05_forecast').glob('*.py'))+list(SAVED.rglob('*'))
    protected={str(p):trial.sha(p) for p in protected_paths if p.is_file()}
    scored=pd.read_csv(SAVED/'D_scored.csv')
    assert scored.unit.eq('cam_D').all() and len(scored)==372
    primary=scored[scored.day.eq('2026-09-12')].copy()
    windows=primary.groupby('window_id').agg(targets=('case_id','size'),first_minute=('minute','min'),
        last_minute=('minute','max'),baseline_sum=('baseline_error','sum'),candidate_sum=('candidate_error','sum')).reset_index()
    windows['absolute_improvement']=windows.baseline_sum-windows.candidate_sum
    windows['baseline_mae']=windows.baseline_sum/windows.targets
    windows['candidate_mae']=windows.candidate_sum/windows.targets
    windows['mae_gain']=windows.baseline_mae-windows.candidate_mae
    windows.sort_values('absolute_improvement',ascending=False).to_csv(OUT/'Sep12_window_contributions.csv',index=False)
    top=windows.nlargest(2,'absolute_improvement')
    kept=primary[~primary.window_id.isin(top.window_id)]
    removed=dict(selection='Two largest SUM reductions in absolute error, not largest small-window mean gains',
        removed_windows=top.to_dict('records'),remaining_targets=len(kept),remaining_windows=kept.window_id.nunique(),
        baseline_mae=kept.baseline_error.mean(),candidate_mae=kept.candidate_error.mean(),
        gain=kept.baseline_error.mean()-kept.candidate_error.mean(),
        top_two_fraction_total_gain=float(top.absolute_improvement.sum()/windows.absolute_improvement.sum()),
        improved_windows=int(windows.mae_gain.gt(1e-10).sum()),worsened_windows=int(windows.mae_gain.lt(-1e-10).sum()),
        equal_windows=int(windows.mae_gain.abs().le(1e-10).sum()),
        largest_mae_regression=windows.nsmallest(1,'mae_gain').to_dict('records'),
        largest_sum_regression=windows.nsmallest(1,'absolute_improvement').to_dict('records'))
    recording_meta=pd.read_csv(ROOT/'data/raw/recordings.csv').set_index('recording')
    records=[]
    for name,g in scored[scored.day.eq('2026-09-05')].groupby('recording'):
        meta=recording_meta.loc[name]
        records.append(dict(recording=name,recording_start=meta['first'],recording_end=meta['last'],
            evaluated_start=clock(g.minute.min()),evaluated_end=clock(g.minute.max()),targets=len(g),windows=g.window_id.nunique(),
            baseline_mae=g.baseline_error.mean(),candidate_mae=g.candidate_error.mean(),new_method=int(g.used_new_method.sum())))
    pd.DataFrame(records).to_csv(OUT/'Sep5_separate_recordings.csv',index=False)
    historical,real,metadata=visible_inputs();schedule=trial.schedule()
    saved_actual=pd.read_csv(SAVED/'actual_D_predictions.csv')[['case_id','used_new_method']]
    metadata=metadata.merge(saved_actual,on='case_id',validate='one_to_one')
    assert len(metadata)==285 and metadata.case_id.is_unique
    evaluated_features,evaluated_raw=features(scored[META],historical,schedule)
    actual_features,actual_raw=features(metadata,real,schedule)
    actual_features.reset_index().to_csv(OUT/'actual_visible_features.csv',index=False)
    evaluated_features.reset_index().to_csv(OUT/'evaluated_visible_features.csv',index=False)
    reference_primary=evaluated_features.loc[primary.case_id]
    summaries=[]
    for name,ref in [('Sep12',reference_primary),('all_evaluated_D',evaluated_features)]:
        summaries.append(range_comparison(ref,actual_features,name,metadata))
    pd.DataFrame(summaries).to_csv(OUT/'feature_range_summary.csv',index=False)
    # Raw count ranges make log-feature shifts understandable in people units.
    raw_ranges=[]
    for name,ref in [('Sep12',evaluated_raw.loc[primary.case_id]),('all_evaluated_D',evaluated_raw)]:
        for camera in ref:
            a=actual_raw[camera];r=ref[camera]
            raw_ranges.append(dict(reference=name,camera=camera,evaluated_min=r.min(),evaluated_max=r.max(),
                actual_min=a.min(),actual_max=a.max(),outside_range=int((a.notna()&((a<r.min()-1e-10)|(a>r.max()+1e-10))).sum())))
    pd.DataFrame(raw_ranges).to_csv(OUT/'visible_count_ranges.csv',index=False)
    for name,case_frame in [('Sep12',primary),('Sep5',scored[scored.day.eq('2026-09-05')]),('actual',metadata)]:
        case_frame.assign(hour=case_frame.minute//60).groupby('hour').size().rename('targets').to_csv(OUT/f'{name}_by_hour.csv')
    assert all(trial.sha(Path(p))==expected for p,expected in protected.items())
    result=dict(training_calls=0,prediction_calls=0,submission_created=False,
        protected_files_unchanged=True,removed_top_two=removed,feature_ranges=summaries,
        elapsed_seconds=time.perf_counter()-started,protected_sha256=protected)
    (OUT/'stress_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':main()
