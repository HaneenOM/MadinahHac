"""Independent P59 artifact checks and temporal-window sensitivity; no fitting."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'src',ROOT/'noran/experiments'):sys.path.insert(0,str(path))
import numpy as np
import pandas as pd
from pipeline import load,locate_data
from prayer_experiments import timetable
from noran.tracks.p59_local_match import model as M
import prompt54_hf as P

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/p59_verified')
    args=parser.parse_args();out=args.output_dir.resolve()
    results=json.loads((out/'result.json').read_text())
    train,test,_,_=load(locate_data(None))
    schedule=timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)
    previous=json.loads((ROOT/'outputs/p58_verified/verification.json').read_text())
    for path,digest in previous['hashes'].items():assert sha(ROOT/path)==digest,path
    verified=[]
    for which in ('fixed','mirror','testday'):
        p=pd.read_csv(out/which/'predictions.csv')
        b=pd.read_csv(ROOT/'outputs/p58_verified'/which/'predictions.csv').set_index('case_id')
        cases,feeds,labels=P.protocol(which,train,test)
        assert p.case_id.is_unique and set(p.case_id)==set(cases.case_id)==set(b.index)
        aligned=b.reindex(p.case_id)
        assert np.array_equal(p.baseline.to_numpy(),aligned.baseline.to_numpy())
        assert np.isfinite(p[['baseline','prediction','reference']]).all().all() and (p.prediction>=0).all()
        assert (p.horizon==p.minute-p.now).all()
        changed=p.prediction.ne(p.baseline)
        assert np.array_equal(changed.to_numpy(),p.changed.to_numpy())
        d=p.merge(labels,on='case_id',validate='one_to_one')
        d['base_error']=(d.baseline-d.truth).abs();d['new_error']=(d.prediction-d.truth).abs()
        temporal=d.groupby(['recording','now']).agg(base=('base_error','sum'),new=('new_error','sum'),n=('truth','size'),changed=('changed','sum'))
        temporal['gain']=temporal.base-temporal.new
        temporal.to_csv(out/which/'temporal_windows.csv')
        sensitivity={}
        for count in (0,1,2):
            keep=temporal.sort_values('gain',ascending=False).iloc[count:]
            sensitivity[str(count)]=float((keep.base.sum()-keep.new.sum())/keep.n.sum())
        matched=d[d.used]
        selected=pd.read_csv(out/'selected_matches.csv').query('protocol==@which')
        used_windows=p[p.used][['recording','unit','now']].drop_duplicates().rename(columns={'recording':'query_recording'})
        used_sources=selected.merge(used_windows,on=['query_recording','unit','now'],how='inner')
        verification=dict(protocol=which,rows=len(p),changed=int(changed.sum()),
            temporal_changed_windows=int((temporal.changed>0).sum()),camera_changed_windows=len(p[p.changed].drop_duplicates(['recording','unit','now'])),
            gain_remove_best_temporal_windows=sensitivity,
            matched_subset_baseline_mae=float(matched.base_error.mean()) if len(matched) else None,
            matched_subset_new_mae=float(matched.new_error.mean()) if len(matched) else None,
            individual_improved=int((d.new_error<d.base_error).sum()),individual_worsened=int((d.new_error>d.base_error).sum()),
            matched_source_events=sorted(used_sources.event.unique().tolist()),
            matched_source_recordings=sorted(used_sources.recording.unique().tolist()),
            matched_source_dates=sorted(used_sources.day.unique().tolist()))
        verified.append(verification)
    # Inference-only coverage on competition requests: no assessment answers.
    cases,feeds,_=P.protocol('actual',train,test)
    lib=M.library(train,schedule)
    actual_rows=[];extra_checks=0
    for _,group in cases.groupby(['recording','unit','now']):
        r=next(group.itertuples())
        selected,candidates=M.select(r,feeds[r.recording],lib)
        # Poison ALL historical target-camera context/continuation arrays. The
        # selected context-only matches and their level estimates must not move.
        poisoned=dict(lib,tables={k:v.copy() for k,v in lib['tables'].items()},analogs={})
        for rec,t in poisoned['tables'].items():
            for unit in M.C.HIDDEN_WITH[r.unit]:t[unit]=1e8
        for rec,an in lib['analogs'].items():
            new=copy.copy(an);new.raw=dict(an.raw);new.smooth=dict(an.smooth)
            for unit in M.C.HIDDEN_WITH[r.unit]:
                if unit in an.raw:new.raw[unit]=np.full_like(an.raw[unit],1e8)
                if unit in an.smooth:new.smooth[unit]=np.full_like(an.smooth[unit],1e8)
            poisoned['analogs'][rec]=new
        again,all_again=M.select(r,feeds[r.recording],poisoned)
        assert selected==again and candidates==all_again
        extra_checks+=1
        for case in group.itertuples():
            reason='insufficient_context_events'
            eligible=False
            if int(schedule[case.day][2])+55<=case.now<=int(schedule[case.day][3])-35:
                reason='v7_afternoon_fallback'
            elif len(selected)>=2:
                value=M.continuation(case,selected,lib)
                original=M.v7.analog_value(feeds[case.recording].loc[:case.now],case.unit,case.now,case.horizon,
                              case.day,dict(lib,params=dict(M.v7.PARAMS)),schedule)
                eligible=bool(np.isfinite(value) and np.isfinite(original))
                reason='eligible' if eligible else 'continuation_or_v7_analog_unavailable'
            actual_rows.append(dict(case_id=case.case_id,recording=case.recording,unit=case.unit,
                                    now=case.now,horizon=case.horizon,eligible=eligible,reason=reason))
    actual=pd.DataFrame(actual_rows)
    actual.to_csv(out/'actual_request_coverage.csv',index=False)
    coverage=actual.groupby('unit').agg(requests=('case_id','size'),eligible=('eligible','sum'),windows=('now','nunique')).reset_index()
    coverage.to_csv(out/'actual_coverage_summary.csv',index=False)
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip()
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='verification.json']
    files+=[ROOT/'noran/tracks/p59_local_match/model.py',ROOT/'noran/experiments/prompt59_local_match.py',
            ROOT/'noran/experiments/prompt59_verify.py',ROOT/'noran/configs/prompt59_local_match.json']
    report_path=ROOT/'reports/prompt59_local_match_ar.md'
    if report_path.exists():files.append(report_path)
    report=dict(status='passed',results=verified,actual_coverage=coverage.to_dict('records'),
                historical_target_selection_invariance_checks=extra_checks,
                selection_mask_postnow_checks=results['selection_invariance_checks'],
                all_p58_hashes_preserved=True,tracked_changes=[],candidate_created=False,
                hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='hashes'},indent=2))

if __name__=='__main__':main()
