"""P43: user-requested B-only candidate, frozen P42 settings, no new tuning."""
from __future__ import annotations
import csv
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import friday_catboost_trial as p42
from released_hidden_training_trial import BASELINE,expand,training_rows,row_keys,sha
from pipeline import table

ROOT=p42.ROOT
OUTPUT=ROOT/'outputs/submission_catboost_b_only_v1.csv'
ARTIFACTS=ROOT/'outputs/prompt43_catboost_b_only'
P42=ROOT/'outputs/prompt42_friday_catboost'
BASELINE_SHA='c7e0a0785fee013449b30a1041abdf939a58461bb5f359c5294e69b2c8ec6cd5'

def main(verify_existing=False):
    started=time.perf_counter()
    if verify_existing:
        assert OUTPUT.exists() and (ARTIFACTS/'model_cam_B.cbm').exists()
        assert not (ARTIFACTS/'verification.json').exists(), 'Existing verification is protected'
    elif OUTPUT.exists() or ARTIFACTS.exists():raise FileExistsError('Preserve existing candidate/artifacts; no overwrite.')
    protected=p42.protected()
    for f in P42.rglob('*'):
        if f.is_file():protected[str(f)]=sha(f)
    assert sha(BASELINE)==BASELINE_SHA
    selected=json.loads((P42/'selected.json').read_text())['cam_B']
    assert selected['spec']=='counts_d4' and selected['weight']==.5 and selected['iterations']==269
    saved=json.loads((P42/'provenance.json').read_text())
    assert p42.provenance()==saved, 'P42 code, inputs or dependency versions changed'
    train,test,schedule=p42.p40.inputs()
    training,extra=expand(training_rows(train),test)
    targets=test[test.kind.eq('hidden')&test.unit.eq('cam_B')&pd.to_datetime(test.day).dt.day_of_week.eq(4)].copy()
    assert len(targets)==284 and set(targets.recording)=={'20260918_1100'}
    assert not row_keys(training)&row_keys(test[test.id.notna()])
    targets['case_id']=targets.id;targets['minute']=targets.minute_of_day
    with BASELINE.open(newline='') as handle:original=list(csv.DictReader(handle))
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv',dtype=str)
    assert len(original)==len({r['id'] for r in original})==4457
    assert [r['id'] for r in original]==sample.id.tolist()
    anchors={r['id']:float(r['people']) for r in original}
    # Full released TRAIN is allowed in final fitting, including Sep11 labels.
    # No benchmark fitting or model/feature/iteration selection is performed.
    with threadpool_limits(limits=4):
        meta,x=p42.dataset(training,schedule)
        keep=meta.unit.eq('cam_B').to_numpy()
        meta=meta.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        x=x.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        assert set(meta.unit)=={'cam_B'}
        if verify_existing:
            model=p42.CatBoostRegressor();model.load_model(str(ARTIFACTS/'model_cam_B.cbm'))
            cols=p42.columns(x,selected['spec']);_,yy,_=p42.augmentation(x[cols],meta)
            info=dict(unit='cam_B',spec=selected['spec'],original_rows=len(meta),augmented_rows=len(yy),
                records=sorted(meta.recording.unique()),dates=sorted(meta.day.unique()),iterations=model.tree_count_,features=cols)
            params=model.get_all_params()
            assert params['depth']==4 and params['loss_function']=='MAE' and params['l2_leaf_reg']==10
            assert model.tree_count_==selected['iterations'] and model.feature_names_==cols
        else:model,info=p42.train_model(meta,x,'cam_B',selected['spec'],selected['iterations'])
        feeds=table(test[test.id.isna()&test.people.notna()])
        xx=p42.features(feeds,targets,schedule,False)
        assert list(xx.columns)==info['features']
        poison={name:feed.assign(cam_B=1e12,cam_G=-1e12) for name,feed in feeds.items()}
        pd.testing.assert_frame_equal(xx,p42.features(poison,targets,schedule,False))
        fitted=np.maximum(0.,model.predict(xx))
    anchor=np.array([anchors[key] for key in targets.id])
    available=xx.available.ge(2).to_numpy()
    candidate=np.maximum(0.,anchor+selected['weight']*(fitted-anchor))
    assert np.isfinite(candidate).all()
    # If fewer than two exact visible donors, retain ORIGINAL BASELINE TEXT.
    replacements={key:str(float(value)) for key,value,ok in zip(targets.id,candidate,available) if ok}
    output=[dict(id=r['id'],people=replacements.get(r['id'],r['people'])) for r in original]
    changed=[r['id'] for r,o in zip(output,original) if r['people']!=o['people']]
    assert len(output)==len({r['id'] for r in output})==4457
    assert [r['id'] for r in output]==sample.id.tolist()
    values=np.array([float(r['people']) for r in output]);assert np.isfinite(values).all() and (values>=0).all()
    allowed=set(targets.id);assert set(changed)<=allowed
    assert all(r['people']==o['people'] for r,o in zip(output,original) if r['id'] not in allowed)
    assert all(r['people']==o['people'] for r,o in zip(output,original) if r['id'] not in replacements)
    if not verify_existing:
        ARTIFACTS.mkdir(parents=True,exist_ok=False)
        model.save_model(str(ARTIFACTS/'model_cam_B.cbm'))
    # Verify saved artifact reproduces final fitted B counts without refitting.
    check=p42.CatBoostRegressor();check.load_model(str(ARTIFACTS/'model_cam_B.cbm'))
    np.testing.assert_allclose(np.maximum(0.,check.predict(xx)),fitted,atol=1e-12,rtol=0)
    records=meta.groupby(['day','recording']).agg(examples=('truth','size'),
        first_minute=('minute','min'),last_minute=('minute','max')).reset_index()
    records['source']=np.where(records.recording.isin(set(train.recording)),'train.csv released','test.csv genuinely released (id missing)')
    if not verify_existing:records.to_csv(ARTIFACTS/'B_training_recordings.csv',index=False)
    else:pd.testing.assert_frame_equal(records,pd.read_csv(ARTIFACTS/'B_training_recordings.csv'))
    detail=targets[['id','day','recording','unit','minute']].copy()
    detail['baseline']=anchor;detail['catboost']=fitted;detail['used_catboost']=available
    detail['visible_donors']=xx.available.to_numpy();detail['candidate']=np.where(available,candidate,anchor)
    if not verify_existing:
        detail.to_csv(ARTIFACTS/'B_predictions.csv',index=False)
        with OUTPUT.open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(output)
    # Re-read actual serialized strings, not only the in-memory assembly.
    with OUTPUT.open(newline='') as handle:written=list(csv.DictReader(handle))
    assert written==output
    assert all(w['people']==b['people'] for w,b in zip(written,original) if w['id'] not in allowed)
    after=p42.protected()
    for f in P42.rglob('*'):
        if f.is_file():after[str(f)]=sha(f)
    assert all(after[key]==digest for key,digest in protected.items())
    result=dict(output=str(OUTPUT.relative_to(ROOT)),sha256=sha(OUTPUT),baseline=str(BASELINE.relative_to(ROOT)),
        baseline_sha256=BASELINE_SHA,selected=selected,selected_file_sha256=sha(P42/'selected.json'),
        production_status='Separate user-requested exploratory candidate; P42 accept/reject conclusions unchanged.',
        method='0.5 protected baseline B +0.5 nonnegative CatBoost; counts_d4, 269 trees; no new tuning or early-stop on actual targets.',
        fallback='Keep original baseline prediction AND original text when fewer than two exact visible ACDEFH donors.',
        donors=p42.DONORS,training=info,training_B_recordings=records.to_dict('records'),
        extra_released_AH_rows=len(extra),extra_B_examples=int(meta.recording.isin(set(extra.recording)).sum()),
        ids=4457,sample_order_exact=True,finite_nonnegative=True,intended_targets=len(targets),changed_rows=len(changed),
        unchanged_prediction_strings=4457-len(changed),unchanged_outside_scope=4457-len(targets),
        G_Saturday_D_and_other_tasks_exactly_unchanged=True,fallback_count=int((~available).sum()),
        changed_ids=changed,protected_files_unchanged=len(protected),hidden_poison_invariance=True,
        model_sha256=sha(ARTIFACTS/'model_cam_B.cbm'),source_sha256=sha(__file__),
        p42_provenance=saved,seconds=time.perf_counter()-started,verification_only=verify_existing)
    (ARTIFACTS/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('changed_ids','p42_provenance','training_B_recordings','training')},indent=2))
    print(records.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing',action='store_true',help='Finalize checks using existing fitted artifact; never refit or overwrite CSV/model.')
    main(parser.parse_args().verify_existing)
