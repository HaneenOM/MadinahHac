"""Build the separate, exploratory G/D hidden candidate from accepted V7 text.
Reuses verified local V7 mission predictions; fits G's unchanged V3 only to fill
five neighbor minutes missing from the saved target-only cache. No T05, no upload.
"""
from pathlib import Path
from types import SimpleNamespace
import csv,json,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'noran/experiments'))
import prompt56 as S
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
OUT=S.OUT/'candidate'

def smooth(raw,t,unit):
    masked=('cam_B','cam_G') if unit=='cam_G' else ('cam_D',)
    donors=[c for c in t if c.startswith('cam_') and c[-1] in 'ABCDEFGH' and c not in masked]
    good=t[donors].notna().sum(axis=1).ge(2)
    eligible=raw.where(good.reindex(raw.index).fillna(False))
    center=np.log1p(raw)
    left=np.log1p(eligible.reindex(raw.index-1).set_axis(raw.index)).fillna(center)
    right=np.log1p(eligible.reindex(raw.index+1).set_axis(raw.index)).fillna(center)
    return np.expm1((2*center+left+right)/4)

def main():
    # The agent enforces the original session deadline externally. This saved
    # generation command must also remain usable after that session has ended.
    protected=S.initialize();OUT.mkdir(parents=True,exist_ok=True)
    final=OUT/'submission_prompt56_hidden_gd.csv'
    if final.exists():raise FileExistsError('Candidate already exists; preserve it and inspect verification.json.')
    started=time.perf_counter();train,test,sample,schedule=S.setup()
    targets=test[test.id.notna()&test.kind.eq('hidden')&test.unit.isin(['cam_G','cam_D'])].copy()
    released=test[test.id.isna()&test.people.notna()];feeds=S.table(released)
    cached_path=ROOT/'outputs/noran/tracks/hidden_v4/test_predictions.csv'
    cache=pd.read_csv(cached_path)
    local_path=ROOT/'outputs/version7/submission_version7.csv';local=pd.read_csv(local_path).set_index('id').people
    assert cache.id.is_unique and np.isfinite(cache.people).all()
    assert np.allclose(cache.people,local.reindex(cache.id),atol=1e-9,rtol=0)
    cache=cache.merge(test[['id','recording','day','unit','minute_of_day']].dropna(subset=['id']),on='id',validate='one_to_one')
    baseline=S.mod('noran/tracks/p56_hidden_gd/model.py','p56_candidate').member.v7
    context=dict(schedule=schedule,config=S.tasks.default_config()['hidden'],mission='hidden',test_released=released)
    fitted_g=None;fit_seconds=0.;extra_checks=0;extra_minutes=[];rows=[]
    with threadpool_limits(limits=2):
        for (rec,unit),group in targets.groupby(['recording','unit']):
            t=feeds[rec];day=group.day.iloc[0]
            raw=cache[cache.recording.eq(rec)&cache.unit.eq(unit)].set_index('minute_of_day').people.copy()
            masked=('cam_B','cam_G') if unit=='cam_G' else ('cam_D',)
            donors=[c for c in t if c.startswith('cam_') and c[-1] in 'ABCDEFGH' and c not in masked]
            good=t[donors].notna().sum(axis=1).ge(2)
            needed={int(m)+offset for m in group.minute_of_day for offset in (-1,0,1)
                    if (int(m)+offset in good.index and good.loc[int(m)+offset]) or offset==0}
            missing=sorted(needed-set(raw.index))
            if missing:
                assert unit=='cam_G',f'Unexpected missing neighbor scope: {rec}/{unit}/{missing}'
                tick=time.perf_counter()
                fitted_g=dict(v3=baseline.v3.fit(train.copy(),None,context),tabicl=None)
                fit_seconds+=time.perf_counter()-tick
                # G has SHARE=0, so the unchanged V3 is exactly its V7 computation.
                for m in list(raw.index[::100])+missing:
                    case=SimpleNamespace(day=day,recording=rec,unit=unit,kind='hidden',minute=int(m))
                    value=baseline.predict(case,t,fitted_g,context)
                    if m in raw:np.testing.assert_allclose(value,raw[m],atol=1e-8,rtol=0)
                    else:raw.loc[m]=value;extra_minutes.append(dict(recording=rec,unit=unit,minute=int(m),prediction=float(value)))
                    altered=t.copy()
                    for cam in masked:altered[cam]=1e8
                    np.testing.assert_allclose(baseline.predict(case,altered,fitted_g,context),value,atol=1e-8,rtol=0)
                    extra_checks+=1
            raw=raw.sort_index();smoothed=smooth(raw,t,unit)
            altered=t.copy()
            for cam in masked:altered[cam]=1e8
            np.testing.assert_allclose(smooth(raw,altered,unit),smoothed,atol=0,rtol=0)
            for r in group.itertuples():
                original=float(raw[r.minute_of_day]);new=float(smoothed[r.minute_of_day])
                rows.append(dict(id=r.id,recording=rec,unit=unit,minute=int(r.minute_of_day),
                    local_v7=original,smoothed_local=new,delta=new-original))
    delta=pd.DataFrame(rows).set_index('id');assert len(delta)==len(targets)
    reference=ROOT/'outputs/version7/reference/submission_v7_board_0.724.csv'
    expected='2eced7c2ea9765d93d798fc0130126a523e51e69b1fb4ac5bf6d5783d81cc88c'
    assert S.sha(reference)==expected
    with reference.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;accepted=list(reader)
    assert fields==['id','people'];changed=[];result=[]
    for row in accepted:
        new=dict(row)
        if row['id'] in delta.index:
            change=float(delta.at[row['id'],'delta'])
            if abs(change)>1e-8:
                value=float(row['people'])+change
                assert np.isfinite(value) and value>=0
                new['people']=str(value);changed.append(row['id'])
        result.append(new)
    ids=[r['id'] for r in result]
    assert len(ids)==len(set(ids))==4457
    assert ids==sample.id.tolist() and set(ids)==set(sample.id)
    assert all(np.isfinite(float(r['people'])) and float(r['people'])>=0 for r in result)
    untouched=0
    for old,new in zip(accepted,result):
        assert old['id']==new['id']
        if old['id'] not in changed:
            assert old['people']==new['people'];untouched+=1
    assert set(changed)<=set(targets.id)
    ids=[r['id'] for r in result]
    assert len(ids)==len(set(ids))==4457
    assert ids==sample.id.tolist()
    assert set(ids)==set(sample.id)
    values=np.array([float(r['people']) for r in result])
    assert np.isfinite(values).all() and (values>=0).all()
    with final.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(result)
    delta.to_csv(OUT/'applied_deltas.csv')
    (OUT/'extra_neighbor_predictions.json').write_text(json.dumps(extra_minutes,indent=2))
    report=dict(status='format and scope verified; exploratory small local improvement, no promotion',
        path=str(final),sha256=S.sha(final),accepted_sha256=S.sha(reference),local_baseline_sha256=S.sha(local_path),
        cached_v7_prediction_sha256=S.sha(cached_path),candidate_config_sha256=S.sha(ROOT/'noran/configs/prompt56_candidate.json'),
        rows=4457,unique_ids=4457,sample_order=True,exact_id_set=True,finite_nonnegative=True,
        supported_scope_rows=len(targets),changed_rows=len(changed),exact_unchanged_prediction_strings=untouched,
        outside_supported_scope_unchanged=True,masked_transform_checks=len(targets),extra_full_model_mask_checks=extra_checks,
        extra_neighbor_minutes=len(extra_minutes),g_only_v7_fit_seconds=fit_seconds,
        max_abs_delta=float(delta.delta.abs().max()),mean_abs_delta=float(delta.delta.abs().mean()),
        max_local_vs_accepted_reproduction_noise=float(np.max(np.abs(local.reindex([r['id'] for r in accepted])-np.array([float(r['people']) for r in accepted])))),
        protected_files_unchanged=protected,runtime_seconds=time.perf_counter()-started,
        elapsed_session_seconds=time.time()-S.START)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    S.initialize()
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='outputs/prompt56/candidate')
    args=parser.parse_args()
    OUT=(ROOT/args.output_dir).resolve()
    if not OUT.is_relative_to(ROOT):raise ValueError('Output directory must stay inside this project.')
    main()
