"""Label-free request similarity audit and compact P61 gap examples."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'noran/experiments'):sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
import prompt61 as S
from pipeline import table


def main():
    S.initialize();train,test,_,_,ctx=S.inputs();out=S.OUT/'diagnosis';out.mkdir(exist_ok=True)
    panels=[]
    development=pd.read_csv(S.OUT/'development_scored.csv')
    panels.append(development.assign(protocol='development')[['protocol','source','unit','base','span','n_left','n_right']])
    for which in ('fixed','mirror','actual'):
        if which=='actual':
            feeds=table(test[test.id.isna()&test.people.notna()]);parts=[]
            for r in test[test.id.notna()&test.kind.eq('missing')].itertuples():
                case=type('Case',(),dict(unit=r.unit,minute=int(r.minute_of_day)))()
                raw=S.M.inference_rows(case,feeds[r.recording]);raw['source']=r.recording;parts.append(raw)
        else:
            cases,feeds,_=S.P.protocols(which,'missing',train,test);parts=[]
            for r in cases.itertuples():
                raw=S.M.inference_rows(r,feeds[r.recording]);raw['source']=r.recording;parts.append(raw)
        d=pd.concat(parts,ignore_index=True);d['protocol']=which
        panels.append(d[['protocol','source','unit','base','span','n_left','n_right']])
    rows=pd.concat(panels,ignore_index=True)
    summary=[]
    for (protocol,unit),g in rows.groupby(['protocol','unit']):
        summary.append(dict(protocol=protocol,unit=unit,rows=len(g),recordings=g.source.nunique(),
            level_p10=g.base.quantile(.1),level_p50=g.base.median(),level_p90=g.base.quantile(.9),
            gap_span_median=g.span.median(),left_median=g.n_left.median(),right_median=g.n_right.median(),
            thin_left_fraction=g.n_left.le(2).mean()))
    pd.DataFrame(summary).to_csv(out/'input_similarity.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(12,6))
    for ax,letter in zip(axs.flat,'ACDFGH'):
        for which in ('development','fixed','mirror','actual'):
            values=np.sort(rows.loc[rows.protocol.eq(which)&rows.unit.eq('cam_'+letter),'base'].to_numpy())
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=which)
        ax.set_title('Camera '+letter);ax.set_xlabel('Gap line level (observable)');ax.set_ylabel('CDF');ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'input_level_distributions.png',dpi=160);plt.close(fig)
    # Honest best/worst illustrations: picked by saved residual assessment gains,
    # never used to change its recipe or select an exception.
    d=pd.read_csv(S.OUT/'residual/fixed_scored.csv')
    gap=d.groupby(['recording','unit','gap_start','gap_end']).gain.sum().sort_values()
    chosen=[gap.index[0],gap.index[-1]]
    _,feeds,_=S.P.protocols('fixed','missing',train,test)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    examples=[]
    for ax,key in zip(axs,chosen):
        rec,unit,a,b=key;g=d[d.recording.eq(rec)&d.unit.eq(unit)&d.gap_start.eq(a)&d.gap_end.eq(b)]
        t=feeds[rec];obs=t[unit].loc[a-8:b+8]
        ax.plot(obs.index,obs,'ko-',markersize=3,label='released context')
        ax.plot(g.minute,g.truth,'k--',label='assessment truth')
        ax.plot(g.minute,g.v7,'b.-',label='V7')
        ax.plot(g.minute,g.trial,'r.-',label='residual trial')
        ax.set_title(f'{rec} {unit}: {g.gain.sum():+.3f} total gain');ax.set_xlabel('Real minute of day');ax.set_ylabel('People');ax.legend(fontsize=8);ax.grid(alpha=.2)
        examples.append(dict(recording=rec,unit=unit,gap_start=int(a),gap_end=int(b),gain=g.gain.sum(),rows=len(g)))
    fig.tight_layout();fig.savefig(out/'residual_best_worst_gaps.png',dpi=160);plt.close(fig)
    S.save(out/'result.json',dict(input_audit='No target labels used in input distribution statistics',
        actual_rows=int(rows.protocol.eq('actual').sum()),examples=examples,
        example_selection='Worst and best saved fixed residual gaps; illustrative, no post-hoc rule changes'))
    print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(examples,default=str))


if __name__=='__main__':main()
