"""Notebook orchestration, compatibility serialization and verification only.

Change algorithms in baseline_tasks or its named source modules, not here.
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from current_baseline import preflight, sha256, ASSETS, REFERENCE, ROOT
from pipeline import load, table
from prayer_experiments import timetable
import baseline_tasks as tasks


def load_project(data_dir):
    manifest,frozen_config=preflight(data_dir)
    train,test,sample,recordings=load(data_dir)
    refs=sample.set_index('id').people.to_dict()
    with (Path(data_dir)/'sample_submission.csv').open(newline='',encoding='utf-8') as handle:
        ref_text={r['id']:r['people'] for r in csv.DictReader(handle)}
    return dict(train=train,test=test,sample=sample,recordings=recordings,
        targets=test[test.id.notna()].copy(),feeds=table(test[test.id.isna()&test.people.notna()]),
        references=refs,reference_text=ref_text,manifest=manifest,frozen_config=frozen_config)


def load_schedule(project):
    return timetable(ASSETS/'prayer_times_madinah.csv',project['train'],project['test'])


def task_targets(project,kind):
    return project['targets'][project['targets'].kind.eq(kind)].copy()


def legacy_text(values, *, parse_first):
    """Retain the two historical pandas CSV rounds, without intermediate files.

    v2 started from decimal strings before the first read; T05 started from
    computed floats. Hidden/HF later replaced selected rows with raw str(float).
    This is serialization compatibility, not a second prediction algorithm.
    """
    frame=pd.DataFrame({'id':list(values),'people':list(values.values())})
    if parse_first:frame=pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    for i in range(2):
        text=frame.to_csv(index=False)
        if i==0:frame=pd.read_csv(io.StringIO(text))
    return {r['id']:r['people'] for r in csv.DictReader(io.StringIO(text))}


def as_frame(values):
    return pd.DataFrame([dict(id=k,people=float(v),people_text=v) for k,v in values.items()])


def forecast_output(project,models,config):
    targets=task_targets(project,'forecast')
    raw=tasks.predict_forecast(models,targets,config)
    values=legacy_text(raw,parse_first=False)
    for r in targets[targets.unit.str[-1].isin(list('IJKLM'))].itertuples():
        s=project['feeds'][r.recording][r.unit]
        base,new,_=tasks.im_forecast_value(s,int(r.now_minute_of_day),config)
        assert np.isfinite(base) and abs(base-float(values[r.id]))<=1e-9
        if abs(new-base)>1e-9:values[r.id]=str(float(new))
    return as_frame(values)


def hidden_output(project,predictions,config):
    # The old v2 stage used these original weights before the accepted B/D update.
    from v2_candidate import HIDDEN
    fitted=predictions.set_index('id');initial={}
    targets=task_targets(project,'hidden')
    for r in targets.itertuples():
        old_weight=HIDDEN[r.unit][1];ref=project['references'][r.id]
        initial[r.id]=str(max(0.,float((1-old_weight)*ref+old_weight*fitted.at[r.id,'fitted'])))
        if r.unit=='cam_G' and config['blends'][r.unit][1]==old_weight:
            initial[r.id]=str(float(fitted.at[r.id,'people']))
    values=legacy_text(initial,parse_first=True)
    for r in targets.itertuples():
        changed_weight=config['blends'][r.unit][1]!=HIDDEN[r.unit][1]
        optional=config['d_only'] and r.unit=='cam_D'
        new=float(fitted.at[r.id,'people'])
        if (changed_weight or optional) and abs(new-float(values[r.id]))>1e-9:
            values[r.id]=str(new)
    return as_frame(values)


def hidden_future_output(project,predictions,models,config):
    from v2_experiments import predict_hf
    fitted=predictions.set_index('id');initial={}
    targets=task_targets(project,'hidden-future')
    for r in targets.itertuples():
        now=int(r.now_minute_of_day);h=int(r.minute_of_day)-now
        direct=predict_hf(project['feeds'][r.recording],r.unit,now,h,models['hidden'],models['direct'],'direct')
        w=config['weight'];ref=project['references'][r.id]
        initial[r.id]=str(max(0.,float((1-w)*ref+w*direct)))
        if r.weekday!='Sat' or not config['saturday_prayer']:
            initial[r.id]=str(float(fitted.at[r.id,'people']))
    values=legacy_text(initial,parse_first=True)
    for r in targets.itertuples():
        new=float(fitted.at[r.id,'people'])
        if config['saturday_prayer'] and r.weekday=='Sat' and abs(new-float(values[r.id]))>1e-9:
            values[r.id]=str(new)
    return as_frame(values)


def missing_output(project,predictions):
    values={r.id:str(float(r.people)) if r.used else project['reference_text'][r.id]
            for r in predictions.itertuples()}
    return as_frame(legacy_text(values,parse_first=True))


def input_example(project,targets,kind):
    """A small visible-data excerpt, NOT a claim to show every model feature."""
    r=targets.iloc[0];t=project['feeds'][r.recording]
    if kind in ('forecast','hidden-future'):
        now=int(r.now_minute_of_day);return t.loc[now-5:now]
    if kind=='missing':
        from v2_missing import gap_context
        ctx=gap_context(t,r.unit,int(r.minute_of_day))
        if ctx:return t.loc[ctx[0]-2:ctx[1]+2]
    minute=int(r.minute_of_day);return t.loc[minute-2:minute+2]


def snapshot_protected():
    paths=list((ROOT/'outputs').rglob('submission*.csv'))
    paths+=list((ROOT/'notebooks').glob('starter*.ipynb'))
    paths+=list(ASSETS.rglob('*.*'))
    return {str(p):sha256(p) for p in paths if p.is_file()}


def save_combined(project,task_frames,config,output,*,verify_accepted=True,
                  filename='submission_current_baseline.csv',protected=None,started=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    collected={}
    for kind,frame in task_frames.items():
        ids=task_targets(project,kind).id.tolist()
        assert frame.id.is_unique and set(frame.id)==set(ids),f'{kind}: target IDs differ'
        assert np.isfinite(frame.people).all() and frame.people.ge(0).all()
        collected.update(frame.set_index('id').people_text.to_dict())
    ids=project['sample'].id.tolist()
    assert len(collected)==len(ids)==4457 and set(collected)==set(ids)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['id','people'])
    writer.writeheader();writer.writerows(dict(id=k,people=collected[k]) for k in ids)
    payload=stream.getvalue().encode('utf-8')
    import hashlib
    digest=hashlib.sha256(payload).hexdigest()
    expected=project['manifest']['accepted_submission_sha256']
    if config['hidden']['d_only'] and verify_accepted:
        raise ValueError('D-only is a separate candidate: disable verify_accepted and choose a new filename.')
    report=dict(ids=len(ids),valid_nonnegative=True,t05_parameter_sets=12,t05_models_fitted=0,
        d_only_enabled=config['hidden']['d_only'],output_sha256=digest,expected_sha256=expected,
        sha256_matches_accepted=digest==expected,reference_available=REFERENCE.exists())
    if REFERENCE.exists():
        with REFERENCE.open(newline='',encoding='utf-8') as handle:old=list(csv.DictReader(handle))
        assert [r['id'] for r in old]==ids
        differences=np.array([abs(float(collected[r['id']])-float(r['people'])) for r in old])
        metadata=project['targets'].set_index('id').kind
        report.update(reference_sha256=sha256(REFERENCE),max_absolute_difference=float(differences.max()),
            unequal_values=int((differences!=0).sum()),mismatches_over_1e_9=int((differences>1e-9).sum()),
            by_task={kind:dict(targets=int(metadata.eq(kind).sum()),
                unequal_values=sum(collected[r['id']] and float(collected[r['id']])!=float(r['people'])
                    for r in old if metadata[r['id']]==kind)) for kind in task_frames})
    report['protected_files_unchanged']=all(sha256(p)==value for p,value in (protected or {}).items())
    report['runtime_seconds']=time.perf_counter()-started if started else None
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['protected_files_unchanged']:raise AssertionError('A protected file changed')
    if verify_accepted and digest!=expected:
        raise AssertionError(f'Baseline differs: inspect {output / "verification.json"}')
    if not verify_accepted and filename=='submission_current_baseline.csv':
        raise ValueError('Choose a separately named candidate filename before saving changes.')
    destination=output/filename
    if destination.exists() and destination.read_bytes()!=payload:
        raise FileExistsError(f'Preserving different existing file: {destination}')
    if not destination.exists():destination.write_bytes(payload)
    saved=pd.read_csv(destination)
    assert saved.id.tolist()==ids and saved.id.is_unique
    assert np.isfinite(saved.people).all() and saved.people.ge(0).all()
    report['output_path']=str(destination)
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def run(data_dir=None,output_dir=None):
    """CLI compatibility; the notebook calls these steps in separate task cells."""
    started=time.perf_counter();protected=snapshot_protected()
    project=load_project(Path(data_dir) if data_dir else ROOT/'data/raw')
    config=tasks.default_config();schedule=load_schedule(project)
    models=tasks.load_forecast(Path(data_dir) if data_dir else ROOT/'data/raw',ASSETS,project['frozen_config'])
    forecast=forecast_output(project,models,config['forecast'])
    hidden=tasks.fit_hidden(project['train'],config['hidden'],schedule)
    hp=tasks.predict_hidden(hidden,task_targets(project,'hidden'),project['feeds'],project['references'],config['hidden'],schedule)
    hf=tasks.fit_hidden_future(project['train'],schedule,config['hidden-future'])
    hfp=tasks.predict_hidden_future(hf,task_targets(project,'hidden-future'),project['feeds'],project['references'],config['hidden-future'],schedule)
    missing=tasks.fit_missing(project['train'],config['missing'])
    mp=tasks.predict_missing(missing,task_targets(project,'missing'),project['feeds'],project['references'],config['missing'])
    return save_combined(project,{'forecast':forecast,'hidden':hidden_output(project,hp,config['hidden']),
        'hidden-future':hidden_future_output(project,hfp,hf,config['hidden-future']),
        'missing':missing_output(project,mp)},config,output_dir or ROOT/'outputs/current_baseline_manual',
        protected=protected,started=started)
