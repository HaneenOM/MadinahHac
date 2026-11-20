"""P61 level-only HF hypothesis: historical development before any assessment."""
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'noran/experiments'):sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import prompt61 as session
import prompt54_hf as P
M=P.module('noran/tracks/p61_hf_level/model.py','p61_hf_level')
OUT=ROOT/'outputs/prompt61/hf_level'


def develop():
    session.initialize();session.check_time();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'development_result.json').exists():raise FileExistsError('Already completed')
    tick=time.perf_counter();train,test,_,schedule=P.setup()
    rows,x=M.examples(train,schedule)
    # All examples and feature scope are frozen before opening the CV labels.
    rows.to_csv(OUT/'development_cases.csv',index=False);np.save(OUT/'development_features.npy',x)
    session.save(OUT/'recipe_lock.json',dict(model=session.sha(M.__file__),config=session.sha(ROOT/'noran/configs/prompt61_hf_level.json'),
        rows=len(rows),feature_sha=session.sha(OUT/'development_features.npy'),gate=M.SETTINGS['assessment_gate']))
    pred=np.full(len(rows),np.nan);folds=[]
    for day in sorted(rows.day.unique()):
        # The held-out source date cannot appear as a training source OR analog,
        # so neither target labels nor generated features inherit its answers.
        fitting=~(rows.day.eq(day)|rows.analog_day.eq(day))
        hold=rows.day.eq(day)
        weights=M.fit_rows(rows[fitting].reset_index(drop=True),x[fitting])
        pred[hold]=M.analog_pair(rows[hold].reset_index(drop=True),x[hold],weights)
        folds.append(dict(day=day,train_rows=int(fitting.sum()),heldout_rows=int(hold.sum()),
            weights={k:v.tolist() for k,v in weights.items()}))
    rows['trial']=pred;rows['e_v7']=abs(rows.base-rows.truth);rows['e_trial']=abs(rows.trial-rows.truth);rows['gain']=rows.e_v7-rows.e_trial
    assert np.isfinite(pred).all()
    rows.to_csv(OUT/'development_scored.csv',index=False)
    by=rows.groupby('source').agg(n=('truth','size'),v7=('e_v7','mean'),trial=('e_trial','mean'),gain=('gain','mean'))
    by.to_csv(OUT/'development_source.csv')
    rows.groupby(['source','unit']).agg(n=('truth','size'),v7=('e_v7','mean'),trial=('e_trial','mean'),gain=('gain','mean')).to_csv(OUT/'development_camera.csv')
    full=by.reindex(['20260911_1100','20260912_1100'])
    passed=(by.gain>0).sum()>=np.ceil(2*len(by)/3) and by.gain.mean()>0 and (full.gain>0).all()
    result=dict(rows=len(rows),source_recordings=len(by),source_dates=rows.day.nunique(),
        uniform_analog_mae=rows.e_v7.mean(),trial_analog_mae=rows.e_trial.mean(),equal_source_gain=by.gain.mean(),
        positive_sources=int((by.gain>0).sum()),assessment_gate_passed=bool(passed),folds=folds,
        runtime_seconds=time.perf_counter()-tick,
        caveat='Per-analog level diagnostic, not full V7 mixture score; heavily overlapping rows. Sep 11/12 deliberately development, never fresh assessment.')
    session.save(OUT/'development_result.json',result)
    print(by.to_string());print(json.dumps(result,indent=2,default=str),flush=True)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['develop'])
    parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args()
    OUT=args.output_dir.resolve()
    session_path=next((p/'session.json' for p in (OUT,*OUT.parents) if (p/'session.json').exists()),None)
    if session_path is not None:
        import datetime
        session.START=datetime.datetime.fromisoformat(json.loads(session_path.read_text())['start_utc']).timestamp()
    else:session.START=time.time()
    session.OUT=OUT
    with threadpool_limits(limits=2):develop()
if __name__=='__main__':main()
