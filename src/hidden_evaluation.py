"""Live HIDDEN benchmark; never load production-fitted estimators or predictions."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import baseline_tasks as tasks
from pipeline import load,table
from prayer_experiments import timetable
from evaluate_baseline_task import fixed_benchmark
from reliable_evaluation import annotate_cases
from released_hidden_training_trial import release_training,row_keys
import v2_hidden

ROOT=Path(__file__).resolve().parents[1]
def run(config,out,data_dir=None,protocol='strict'):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    train,test,_,_=load(data_dir or ROOT/'data/raw')
    schedule=timetable(ROOT/'models/current_baseline/prayer_times_madinah.csv',train,test)
    cases,labels,feeds=fixed_benchmark(train,test,prefer_saved=False)
    cases=annotate_cases(cases[cases.kind.eq('hidden')],feeds)
    extra=test[test.id.isna()&test.people.notna()].copy()
    if protocol=='release_matched':base,_,_,blocked=release_training(train,test,feeds)
    rows=[];manifests=[];checks=0
    with threadpool_limits(limits=4):
        for day,today in cases.groupby('day'):
            training=train[train.day.ne(day)].copy() if protocol=='strict' else base
            released=extra[extra.day.ne(day)] if protocol=='strict' else extra
            if protocol=='strict':assert day not in set(training.day)|set(released.day)
            else:assert not row_keys(training)&blocked and not row_keys(released)&blocked
            models=tasks.fit_hidden(training,config['hidden'],schedule,released=released)
            manifests.append(dict(day=day,protocol=protocol,fit=models.get('bgd',{}).get('manifest',{}),
                training_records=sorted(training.recording.unique()),extra_records=sorted(released.recording.unique())))
            targets=today.rename(columns={'case_id':'id','minute':'minute_of_day'})
            refs={r.id:v2_hidden.predict(feeds[r.recording],r.unit,int(r.minute_of_day),models['linear'],'single')[0] for r in targets.itertuples()}
            pred=tasks.predict_hidden(models,targets,feeds,refs,config['hidden'],schedule).set_index('id')
            poisoned={rec:feed.copy() for rec,feed in feeds.items()}
            for rec,g in targets.groupby('recording'):
                for unit in g.unit:
                    for masked in v2_hidden.PATTERNS[unit]:poisoned[rec][masked]=1e10
            check=tasks.predict_hidden(models,targets,poisoned,refs,config['hidden'],schedule).set_index('id')
            np.testing.assert_allclose(pred.people,check.people,atol=0,rtol=0);checks+=len(pred)
            for r in today.itertuples():
                rows.append(dict(case_id=r.case_id,day=day,unit=r.unit,recording=r.recording,window_id=r.window_id,
                    prediction=float(pred.at[r.case_id,'people']),reference=refs[r.case_id]))
    predictions=pd.DataFrame(rows);predictions.to_csv(out/'predictions.csv',index=False)
    scored=predictions.merge(labels,on='case_id',validate='one_to_one')
    scored['error']=abs(scored.prediction-scored.truth);scored['reference_error']=abs(scored.reference-scored.truth)
    scored.to_csv(out/'scored.csv',index=False)
    for keys,name in [(['day'],'date'),(['day','unit'],'date_camera'),(['day','unit','window_id'],'window')]:
        by=scored.groupby(keys).agg(targets=('case_id','size'),mae=('error','mean'),reference_mae=('reference_error','mean')).reset_index()
        by['normalized_mae']=by.mae/by.reference_mae;by.to_csv(out/f'by_{name}.csv',index=False)
    info=dict(protocol=protocol,targets=len(scored),config=config,folds=manifests,poison_checks=checks,
        production_models_loaded=False,metric='Project-specific MAE / same-case reference MAE; not verified official scoring',
        limitations='Previously examined dates, not independent fresh validation; strict and release-matched are separate settings.',
        runtime_seconds=time.perf_counter()-started)
    (out/'verification.json').write_text(json.dumps(info,indent=2))
    print(pd.read_csv(out/'by_date_camera.csv').to_string(index=False));return info
