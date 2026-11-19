"""Read-only evidence consolidation and protected-baseline verification for PROMPT 56."""
from pathlib import Path
import json,sys,time
import csv
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'noran/experiments'))
import prompt56 as S
OUT=S.OUT

def metric(d,col):
    a=d.assign(err=(d[col]-d.truth).abs(),referr=(d.reference-d.truth).abs()).groupby('day')[['err','referr']].mean()
    return float((a.err/a.referr).mean())

def scopes():
    reports=[]
    for which in ('fixed','mirror','testday','testday_strict'):
        path=OUT/f'hidden_{which}_scored.csv'
        if not path.exists():continue
        d=pd.read_csv(path);d['gd_trial']=np.where(d.unit.isin(['cam_G','cam_D']),d.trial,d.v7)
        d['gd_gain']=(d.v7-d.truth).abs()-(d.gd_trial-d.truth).abs()
        events=d.groupby('event').gd_gain.sum().sort_values(ascending=False)
        windows=d.groupby('window').gd_gain.sum()
        sensitivity=[]
        for n in (0,1,2):
            rem=d[~d.event.isin(events.head(n).index)]
            sensitivity.append(dict(best_events_removed=n,mae_gain=float(rem.gd_gain.mean()),ratio_gain=metric(rem,'v7')-metric(rem,'gd_trial')))
        report=dict(protocol=which,rows=len(d),v7_mae=float((d.v7-d.truth).abs().mean()),
            gd_mae=float((d.gd_trial-d.truth).abs().mean()),v7_ratio=metric(d,'v7'),gd_ratio=metric(d,'gd_trial'),
            camera_windows=len(windows),event_windows=len(events),improved=int((windows>1e-8).sum()),
            worsened=int((windows< -1e-8).sum()),changed=int(((d.gd_trial-d.v7).abs()>1e-8).sum()),sensitivity=sensitivity)
        for unit,g in d.groupby('unit'):
            w=g.groupby('window').gd_gain.sum()
            report.setdefault('by_camera',{})[unit]=dict(rows=len(g),windows=len(w),v7_mae=float((g.v7-g.truth).abs().mean()),
                gd_mae=float((g.gd_trial-g.truth).abs().mean()),wins=int((w>1e-8).sum()),losses=int((w< -1e-8).sum()))
        d.to_csv(OUT/f'hidden_{which}_gd_scored.csv',index=False)
        (OUT/f'hidden_{which}_gd_result.json').write_text(json.dumps(report,indent=2));reports.append(report)
    return reports

def preprocessing_checks():
    train,test,_,_=S.setup();counts=[]
    for mission,sets in [('hidden',('fixed','mirror','testday','testday_strict')),('missing',('fixed','mirror'))]:
        for which in sets:
            cases,feeds,_=S.protocols(which,mission,train,test)
            is_test=which.startswith('testday')
            poison=(test if is_test else train).copy()
            for rec,g in cases.groupby('recording'):
                if mission=='hidden':mask=poison.recording.eq(rec)&poison.unit.isin(g.unit.unique())
                else:
                    keys=set(zip(g.unit,g.minute));mask=poison.recording.eq(rec)&pd.Series(list(zip(poison.unit,poison.minute_of_day)),index=poison.index).isin(keys)
                poison.loc[mask,'people']=1e8
            _,other,_=S.protocols(which,mission,train if is_test else poison,poison if is_test else test)
            for rec,t in feeds.items():pd.testing.assert_frame_equal(t,other[rec])
            for day,g in cases.groupby('day'):
                training,ctx=S.ctx_for(day,which,mission,train,test,{},g)
                assert not training.day.eq(day).any()
                if is_test:assert not (ctx['test_released'].recording.eq(g.recording.iloc[0])&ctx['test_released'].unit.isin(g.unit.unique())).any()
                if which=='testday_strict':
                    for record,units in {'20260918_1100':['cam_D'],'20260919_1100':['cam_B','cam_G']}.items():
                        assert not (ctx['test_released'].recording.eq(record)&ctx['test_released'].unit.isin(units)).any()
            counts.append(dict(mission=mission,protocol=which,targets=len(cases),released_feed_invariant=True,assessment_excluded=True))
    return counts

def main():
    protected=S.initialize();r=scopes();checks=preprocessing_checks()
    result=dict(elapsed_seconds=time.time()-S.START,deadline_seconds_remaining=S.DEADLINE-time.time(),
        protected_files_unchanged=protected,hidden_gd=r,preprocessing_checks=checks,
        candidate_config_sha256=S.sha(ROOT/'noran/configs/prompt56_candidate.json'),
        note='Fixed development; mirror/testday examined only after scope lock. All dates previously examined. No automatic promotion.')
    candidate=OUT/'candidate/submission_prompt56_hidden_gd.csv'
    if candidate.exists():
        reference=ROOT/'outputs/version7/reference/submission_v7_board_0.724.csv'
        with reference.open(newline='',encoding='utf-8') as stream:old=list(csv.DictReader(stream))
        with candidate.open(newline='',encoding='utf-8') as stream:new=list(csv.DictReader(stream))
        sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
        ids=[r['id'] for r in new]
        assert len(ids)==len(set(ids))==4457 and ids==sample.id.tolist()
        assert ids==[r['id'] for r in old]
        values=np.array([float(r['people']) for r in new]);assert np.isfinite(values).all() and (values>=0).all()
        raw=pd.read_csv(ROOT/'data/raw/test.csv')
        scope=set(raw.loc[raw.kind.eq('hidden')&raw.unit.isin(['cam_G','cam_D']),'id'].dropna())
        changes=[];unchanged_outside=0
        for before,after in zip(old,new):
            if before['id'] not in scope:
                assert before['people']==after['people'];unchanged_outside+=1
            if before['people']!=after['people']:changes.append(before['id'])
        assert set(changes)<=scope
        recipe=pd.read_csv(OUT/'candidate/applied_deltas.csv').set_index('id').delta
        for before,after in zip(old,new):
            if before['id'] in changes:
                expected=float(before['people'])+recipe[before['id']]
                np.testing.assert_allclose(expected,float(after['people']),atol=1e-10,rtol=0)
        report=json.loads((OUT/'candidate/verification.json').read_text())
        assert S.sha(candidate)==report['sha256']
        assert S.sha(reference)=='2eced7c2ea9765d93d798fc0130126a523e51e69b1fb4ac5bf6d5783d81cc88c'
        result['candidate_independent_checks']=dict(rows=4457,unique_ids=4457,sample_order=True,finite_nonnegative=True,
            changed_rows=len(changes),outside_scope_exact_strings_unchanged=unchanged_outside,
            delta_transfer_verified=True,sha256=S.sha(candidate))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
