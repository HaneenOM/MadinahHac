"""P60 input-based gap diagnosis; optional one frozen postprocessing trial."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'src',ROOT/'noran/experiments',ROOT/'noran/tracks/missing'):sys.path.insert(0,str(path))
import numpy as np
import pandas as pd
from pipeline import load,locate_data
from v2_missing import gap_context,interp_at
import prompt56 as P
import miss_gaplib as G

OLD=ROOT.parent/'MadinahHac'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/p60_diagnosis')
    args=parser.parse_args();out=args.output_dir.resolve()
    if (out/'result.json').exists():raise FileExistsError('Choose a new --output-dir')
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    # Baseline source and official input compatibility with the preserved P56 run.
    source=[*Path('noran/tracks/missing').glob('*.py')]
    source=[p.relative_to(ROOT) for p in (ROOT/'noran/tracks/missing').glob('*.py')]
    source += [Path('src/baseline_tasks.py'),Path('src/v2_missing.py')]
    for p in source:assert sha(ROOT/p)==sha(OLD/p),str(p)
    manifest=json.loads((ROOT/'models/version7/manifest.json').read_text())
    for p,h in manifest['inputs'].items():assert sha(ROOT/'data/raw'/p)==h==sha(OLD/'data/raw'/p)
    train,test,_,_=load(locate_data(None))
    gaprows=[];detail=[];cache_hashes={}
    for which in ('fixed','mirror'):
        cases,feeds,labels=P.protocols(which,'missing',train,test)
        path=OLD/'outputs/prompt56'/f'missing_{which}_predictions.csv'
        cache_hashes[str(path)]=sha(path)
        baseline=pd.read_csv(path)
        assert baseline.case_id.is_unique and set(baseline.case_id)==set(cases.case_id)
        d=baseline.drop(columns=['trial','fallback']).merge(labels,on='case_id',validate='one_to_one')
        for rec,g in d.groupby('recording'):
            t=feeds[rec]
            for (unit,a,b),gap in g.groupby(['unit','gap_start','gap_end']):
                a,b=int(a),int(b);s=t[unit]
                left=s.reindex(range(a-12,a+1)).dropna();right=s.reindex(range(b,b+13)).dropna()
                assert len(left) and len(right)
                donor_counts=[];disagreements=[];eligible_ratios=[]
                for r in gap.itertuples():
                    context=gap_context(t,unit,r.minute)
                    assert context[:2]==(a,b) and np.isclose(context[3],r.reference,rtol=0,atol=1e-8)
                    frac=(r.minute-a)/(b-a)
                    donors=[]
                    for donor in G.PARTNERS.get(unit,()):
                        if donor not in t:continue
                        x0,x1,xm=(interp_at(t[donor],m) for m in (a,b,int(r.minute)))
                        if np.isfinite([x0,x1,xm]).all():donors.append(xm-(1-frac)*x0-frac*x1)
                    donor_counts.append(len(donors))
                    disagreements.append(bool(len(donors)>1 and min(donors)<0<max(donors)))
                    row=r._asdict();row.pop('Index',None)
                    row.update(protocol=which,fraction=frac,edge=frac<=.2 or frac>=.8,
                         correction=r.v7-r.reference,error_v7=abs(r.v7-r.truth),error_line=abs(r.reference-r.truth),
                         n_left=len(left),n_right=len(right),partners=len(donors),donor_disagreement=disagreements[-1],
                         complete=len(gap)==b-a-1,gap_length=b-a-1)
                    detail.append(row)
                ev=float((gap.v7-gap.truth).abs().mean());el=float((gap.reference-gap.truth).abs().mean())
                vals=gap.v7-gap.reference
                gaprows.append(dict(protocol=which,day=gap.day.iloc[0],recording=rec,unit=unit,
                    gap_start=a,gap_end=b,gap_length=b-a-1,n=len(gap),complete=len(gap)==b-a-1,
                    n_left=len(left),n_right=len(right),thin_context=min(len(left),len(right))<=2,
                    endpoint_jump=abs(float(s.loc[a])-float(left.median()))+abs(float(s.loc[b])-float(right.median())),
                    mean_partners=float(np.mean(donor_counts)),no_partner=max(donor_counts)==0,
                    donor_disagreement=float(np.mean(disagreements)),correction_abs=float(vals.abs().mean()),
                    v7_mae=ev,line_mae=el,line_gain=el-ev,correction_hurts=ev>el+1e-8))
    gaps=pd.DataFrame(gaprows);d=pd.DataFrame(detail)
    gaps.to_csv(out/'gap_diagnosis.csv',index=False);d.to_csv(out/'row_diagnosis.csv',index=False)
    summaries=[]
    for keys,name in [(['protocol'],'overall'),(['protocol','unit'],'camera'),(['protocol','complete'],'complete'),
                       (['protocol','thin_context'],'thin_context'),(['protocol','no_partner'],'no_partner')]:
        g=gaps.groupby(keys).agg(gaps=('n','size'),targets=('n','sum'),hurts=('correction_hurts','sum'),
               gap_mean_v7=('v7_mae','mean'),gap_mean_line=('line_mae','mean')).reset_index()
        g.to_csv(out/(name+'.csv'),index=False)
        print(name);print(g.to_string(index=False))
    # Equal-gap edge/interior comparison, to avoid a few long gaps dominating.
    edge=d.groupby(['protocol','recording','unit','gap_start','gap_end','edge']).agg(
        v7_mae=('error_v7','mean'),line_mae=('error_line','mean'),n=('truth','size')).reset_index()
    edge['hurts']=edge.v7_mae>edge.line_mae+1e-8
    edge.to_csv(out/'edge_by_gap.csv',index=False)
    print('edge');print(edge.groupby(['protocol','edge']).agg(gaps=('n','size'),hurts=('hurts','sum'),v7=('v7_mae','mean'),line=('line_mae','mean')).to_string())
    result=dict(status='diagnosis_complete',runtime_seconds=time.perf_counter()-started,cache_hashes=cache_hashes,
        baseline='V7 v7 column of preserved P56 predictions, never P56 trial',source_compatibility=True)
    (out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
