"""Independent P61 artifact, scope, clustering, replay and invariance checks."""
from pathlib import Path
import argparse,json,time,sys
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'noran/experiments'):sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import prompt61 as S
import prompt54_hf as HF


def clustered(d,key):
    groups=d.groupby(key).gain.sum().sort_values(ascending=False)
    result=[]
    for n in (0,1,2):
        rem=d[~d[key].isin(groups.head(n).index)]
        result.append(dict(best_clusters_removed=n,mae_gain=float(rem.gain.mean()),rows=len(rem)))
    return dict(clusters=len(groups),positive=int((groups>1e-8).sum()),negative=int((groups< -1e-8).sum()),sensitivity=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    tick=time.perf_counter();protected=S.initialize();train,test,old,rel,ctx=S.inputs()
    out=S.OUT/'verification';out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods={'direct':S.OUT,'residual':S.OUT/'residual','weighted':S.OUT/'weighted','adapted':S.OUT/'adapted'}
    result=dict(protected_unchanged=protected,reference_sha256=S.sha(ROOT/'outputs/version7/reference/submission_v7_board_0.724.csv'),
        excluded_missing_assessment_dates=list(S.ASSESSMENT_DAYS),development={},assessment={},replay=[])
    fig,ax=plt.subplots(figsize=(10,4.5));offsets=np.linspace(-.27,.27,len(methods))
    for j,(method,folder) in enumerate(methods.items()):
        path=folder/'development_scored.csv'
        if not path.exists():raise FileNotFoundError(f'Incomplete method: {method}')
        d=pd.read_csv(path)
        assert len(d)==133581 and np.isfinite(d[['v7','trial','truth','reference']]).all().all()
        assert not d.source.str.startswith(('20260911','20260912')).any()
        d['gain']=abs(d.v7-d.truth)-abs(d.trial-d.truth)
        d['physical_minute']=d.minute0+d.m
        d['time_block']=d.source+'|'+(d.physical_minute//30).astype(str)
        unique=len(d[['source','unit','physical_minute']].drop_duplicates())
        selected=json.loads((folder/'selection.json').read_text())['selected_units'] if (folder/'selection.json').exists() else []
        d['selected_gain']=np.where(d.unit.isin(selected),d.gain,0.)
        by=d.groupby('source').gain.mean()
        ax.bar(np.arange(len(by))+offsets[j],by.values,width=.16,label=method)
        selected_d=d.copy();selected_d['gain']=selected_d.selected_gain
        result['development'][method]=dict(rows=len(d),source_recordings=d.source.nunique(),dates=d.day.nunique(),
            distinct_original_readings=unique,selected_units=selected,
            full_scope_blocks=clustered(d,'time_block'),selected_scope_blocks=clustered(selected_d,'time_block'),
            overlap_warning='30-minute blocks are sensitivity units, not independent recordings')
        for which in ('fixed','mirror'):
            pred=folder/f'{which}_scored.csv'
            if not pred.exists():continue
            a=pd.read_csv(pred);cases,feeds,labels=S.P.protocols(which,'missing',train,test)
            assert a.case_id.is_unique and set(a.case_id)==set(cases.case_id)
            truth=a[['case_id','truth']].merge(labels,on='case_id',suffixes=('_saved','_raw'),validate='one_to_one')
            np.testing.assert_array_equal(truth.truth_saved,truth.truth_raw)
            assert np.isfinite(a[['v7','trial','reference']]).all().all() and a.trial.ge(0).all()
            # All methods must have identical strict-V7 baselines and references.
            original=pd.read_csv(S.OUT/f'{which}_scored.csv').set_index('case_id')
            np.testing.assert_allclose(a.v7,original.loc[a.case_id].v7,rtol=0,atol=1e-10)
            assert np.array_equal(a.loc[~a.unit.isin(selected),'v7'],a.loc[~a.unit.isin(selected),'trial'])
            blocks={}
            for rec,t in feeds.items():
                ends=np.r_[0,np.cumsum(np.diff(t.index)>1)]
                blocks.update({(rec,int(m)):int(b) for m,b in zip(t.index,ends)})
            a['physical_block']=[f'{r.recording}|{blocks.get((r.recording,int(r.minute)), -1)}' for r in a.itertuples()]
            a['gain']=abs(a.v7-a.truth)-abs(a.trial-a.truth)
            result['assessment'][method+'_'+which]=dict(changed=int((abs(a.trial-a.v7)>1e-8).sum()),
                camera_gaps=len(a[['recording','unit','gap_start','gap_end']].drop_duplicates()),
                physical_blocks=clustered(a,'physical_block'))
            a.groupby('physical_block').agg(n=('truth','size'),gain=('gain','sum')).to_csv(out/f'{method}_{which}_blocks.csv')
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(np.arange(len(by)),[r.replace('2026','').replace('_','\n') for r in by.index]);ax.set_ylabel('V7 MAE minus trial MAE');ax.set_title('Recording-held-out development: positive is better');ax.legend(ncol=4);fig.tight_layout();fig.savefig(out/'development_recording_gains.png',dpi=160);plt.close(fig)
    rows=S.pseudo(old,ctx)
    sample=rows.iloc[::100].head(500).copy()
    before=S.M.features(sample)
    ignored=['truth','source','window','shift','minute0','day','v2','n_targets']
    after=S.M.features(sample.drop(columns=ignored,errors='ignore'))
    pd.testing.assert_frame_equal(before,after)
    result['label_and_metadata_feature_exclusion_rows']=len(sample)
    if args.replay:
        source='20260904_1105'
        fitting=rows[rows.source.ne(source)].copy().reset_index(drop=True)
        training=old[old.recording.ne(source)].copy()
        fitting=S.crossfit_v2(fitting,training,ctx['config'])
        hold=rows[rows.source.eq(source)].reset_index(drop=True)
        for method in ('direct','residual'):
            module=S.P.mod(f'noran/tracks/p61_missing{ "_residual" if method=="residual" else ""}/model.py','p61_verify_'+method)
            model=module.from_rows(fitting,training,ctx)
            baseline=module.baseline_values(hold,model['baseline'])
            extra=module.nonlinear_values(hold,model['nonlinear'],baseline) if method=='residual' else module.nonlinear_values(hold,model['nonlinear'])
            trial=.5*(baseline+extra)
            saved=pd.read_csv(methods[method]/'development_scored.csv');saved=saved[saved.source.eq(source)]
            error=float(np.max(abs(trial-saved.trial)))
            np.testing.assert_allclose(trial,saved.trial,rtol=0,atol=1e-8)
            result['replay'].append(dict(method=method,source=source,rows=len(trial),max_difference=error,
                current_model_sha256=S.sha(module.__file__),reason='Verify metadata/schema-copy repairs do not change saved numerical development results'))
            print('REPLAY',method,error,flush=True)
    # HF rule remains a rejected experiment, but audit its causal/masking path.
    hm=HF.module('noran/tracks/p61_hf_level/model.py','p61_hf_verify')
    _,_,_,schedule=HF.setup();context=HF.context(schedule)
    model=hm.fit(train,None,context)
    cases,feeds,_=HF.protocol('actual',train,test)
    checks=finite=0
    for _,g in cases.groupby(['recording','unit']):
        for r in g.iloc[::max(1,len(g)//3)].head(3).itertuples():
            t=feeds[r.recording];original=hm.predict(r,t,model,context)
            finite+=int(np.isfinite(original))
            for kind in ('future','joint_hidden'):
                altered=t.copy()
                if kind=='future':altered.loc[altered.index>r.now,:]=1e8
                else:
                    for unit in hm.C.HIDDEN_WITH[r.unit]:altered[unit]=1e8
                np.testing.assert_allclose(original,hm.predict(r,altered,model,context),rtol=0,atol=1e-8,equal_nan=True)
                checks+=1
    result['hf_invariance_checks']=checks;result['hf_finite_representatives']=finite
    assert finite>=3
    # Exact accepted reference validity, without writing a derivative submission.
    reference=pd.read_csv(ROOT/'outputs/version7/reference/submission_v7_board_0.724.csv');sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    assert len(reference)==4457 and reference.id.is_unique and reference.id.tolist()==sample.id.tolist()
    assert np.isfinite(reference.people).all() and reference.people.ge(0).all()
    source_paths=[*ROOT.glob('noran/configs/prompt61*.json'),*ROOT.glob('noran/experiments/prompt61*.py'),*ROOT.glob('noran/tracks/p61*/model.py')]
    result['current_source_hashes']={str(p.relative_to(ROOT)):S.sha(p) for p in source_paths}
    result['protected_unchanged_after']=S.initialize();result['runtime_seconds']=time.perf_counter()-tick
    S.save(out/'verification.json',result)
    print(json.dumps(result,indent=2,default=str),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
