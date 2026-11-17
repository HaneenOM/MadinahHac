"""Select conservative v2 blends and write a checked submission."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline import ROOT,load,locate_data,table
from v2_experiments import fit_direct,predict_hf
import v2_hidden
import v2_missing

OUT=ROOT/'outputs'
HIDDEN={'cam_B':('dropout',.5),'cam_G':('dropout',.5),'cam_D':('ridge',.25)}
HF_WEIGHT=.75
MISSING_WEIGHT={'cam_A':1.,'cam_B':1.,'cam_C':1.,'cam_D':0.,
                'cam_E':1.,'cam_F':.5,'cam_G':1.,'cam_H':0.}


def selected_validation():
    raw=pd.read_csv(OUT/'v2_validation_predictions.csv')
    keys=['fold','recording','kind','unit','minute']
    meta=raw.drop_duplicates(keys).set_index(keys)
    methods=raw.pivot(index=keys,columns='method',values='prediction')
    x=meta[['truth','reference','horizon','gap_length','donor_available']].join(methods)
    weights=[]; candidates=[]
    for r in x.reset_index().itertuples():
        if r.kind=='hidden':
            method,w=HIDDEN[r.unit]
        elif r.kind=='hidden-future':
            method,w='direct',HF_WEIGHT
        else:
            method='blend_full' if MISSING_WEIGHT.get(r.unit,0)==1 else 'blend'
            w=1. if MISSING_WEIGHT.get(r.unit,0)>0 and r.gap_length>=6 and r.donor_available>0 else 0.
        p=getattr(r,method)
        if not np.isfinite(p):
            w=0.
        weights.append(w)
        candidates.append(r.reference+w*(p-r.reference) if w else r.reference)
    x=x.reset_index()
    x['candidate']=candidates
    x['blend_weight']=weights
    x['reference_error']=(x.truth-x.reference).abs()
    x['candidate_error']=(x.truth-x.candidate).abs()
    x.to_csv(OUT/'v2_selected_validation_predictions.csv',index=False)
    for groups,name in [(['fold','recording','kind'],'v2_selected_scores_by_recording.csv'),
                        (['kind','unit'],'v2_selected_scores_by_camera.csv')]:
        s=x.groupby(groups).agg(n=('truth','size'),mae=('candidate_error','mean'),
                reference_mae=('reference_error','mean')).reset_index()
        s['normalized_mae']=s.mae/s.reference_mae.replace(0,np.nan)
        s.to_csv(OUT/name,index=False)
    print(pd.read_csv(OUT/'v2_selected_scores_by_recording.csv').to_string(index=False))


def build():
    train,test,sample,_=load(locate_data(None))
    original=OUT/'submission_v1_0.987.csv'
    if not original.exists():
        raise FileNotFoundError('The exact first submission is required for the combined candidate')
    with original.open(newline='',encoding='utf-8') as f:
        old=list(csv.DictReader(f))
    assert len(old)==len(sample) and [x['id'] for x in old]==sample.id.tolist()
    old_values={x['id']:x['people'] for x in old}
    hidden=v2_hidden.fit(table(train))
    direct=fit_direct(table(train))
    calibration=v2_missing.fit_calibration(table(train))
    released=table(test[test.id.isna() & test.people.notna()])
    target=test[test.id.notna()]
    new_values=old_values.copy()
    methods=Counter()
    for r in target.itertuples():
        if r.kind=='forecast':
            continue
        t=released[r.recording]
        previous=float(old_values[r.id])
        prediction=previous
        method='reference'
        if r.kind=='hidden':
            variant,weight=HIDDEN[r.unit]
            fitted,_=v2_hidden.predict(t,r.unit,int(r.minute_of_day),hidden,variant)
            if np.isfinite(fitted):
                prediction=(1-weight)*previous+weight*fitted
                method=f'{variant}_{weight}'
        elif r.kind=='hidden-future':
            now=int(r.now_minute_of_day)
            h=int(r.minute_of_day)-now
            fitted=predict_hf(t,r.unit,now,h,hidden,direct,'direct')
            if np.isfinite(fitted):
                prediction=(1-HF_WEIGHT)*previous+HF_WEIGHT*fitted
                method=f'direct_{HF_WEIGHT}'
        elif r.kind=='missing':
            weight=MISSING_WEIGHT.get(r.unit,0.)
            if weight:
                variant='blend_full' if weight==1. else 'blend'
                fitted,available,gap=v2_missing.predict(t,r.unit,int(r.minute_of_day),
                                                         calibration,variant,weight)
                if np.isfinite(fitted) and gap>=6 and available>0:
                    prediction=fitted
                    method=f'{variant}_{weight}'
        if not np.isfinite(prediction):
            prediction=previous;method='reference'
        new_values[r.id]=str(max(0.,float(prediction))) if method!='reference' else old_values[r.id]
        methods[(r.kind,method)]+=1
    path=OUT/'submission_v2.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['id','people'])
        writer.writeheader()
        writer.writerows({'id':x['id'],'people':new_values[x['id']]} for x in old)
    with path.open(newline='',encoding='utf-8') as f:
        saved=list(csv.DictReader(f))
    assert len(saved)==4457 and [x['id'] for x in saved]==sample.id.tolist()
    assert len({x['id'] for x in saved})==4457
    assert all(np.isfinite(float(x['people'])) and float(x['people'])>=0 for x in saved)
    kinds=target.set_index('id').kind.to_dict()
    assert all(x['people']==old_values[x['id']] for x in saved if kinds[x['id']]=='forecast')
    changes=Counter(kinds[x['id']] for x in saved if x['people']!=old_values[x['id']])
    print('Submitted rows:',len(saved),'changed by kind:',dict(changes))
    print('Methods:',dict(methods))
    (OUT/'v2_selection.json').write_text(json.dumps({
        'hidden':HIDDEN,'hidden_future_direct_weight':HF_WEIGHT,
        'missing_weights':MISSING_WEIGHT,'missing_min_gap':6,
        'changed_by_kind':dict(changes),'methods':{str(k):v for k,v in methods.items()},
        'first_submission':'submission_v1_0.987.csv'},indent=2))


if __name__=='__main__':
    selected_validation()
    build()
