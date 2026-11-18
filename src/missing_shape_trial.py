"""Prompt 32: one fixed endpoint-preserving Ridge residual model; no tuning."""
from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pipeline import ROOT, table
from missing_median_trial_model import baseline
import v2_missing

OUT = ROOT/'outputs/missing_shape_trial'
LOCAL = ROOT/'outputs/local_score'
BASE = ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
UNITS = [f'cam_{c}' for c in 'ABCDEFGH']
SPEC = dict(family='pooled Ridge, no intercept', mean_loss_l2=.1, shrink=.5,
    origin_step=30, context_minutes=5, minimum_donors=2,
    training='contiguous complete original target intervals; exact transplanted metadata release masks including donor outages; templates cycled deterministically by camera/weekday; whole evaluated DATE excluded',
    target='(truth - endpoint interpolation) / max(mean endpoints,5)',
    features='position/length/endpoint scale and jump, observed exterior slopes and counts, camera indicators, exact-minute donor deviations/endpoint changes/availability',
    shape='4*p*(1-p) multiplies standardized basis; no intercept; half-strength prediction, bounded by envelope times half interpolated level (floor 5)',
    fallback='unchanged current missing predictor when exterior slopes or >=2 exact donors unavailable; I-M always unchanged',
    gate='both dates improve >=1%; >=3 improved gaps each; both dates and pooled gain survive removing two best gaps by total error reduction; no camera with >=10 targets regresses >max(.5 person,10% current MAE)',
    validation='fixed 358 targets / 38 intervals (32 fully labeled, 6 partly labeled due to natural absence); already examined dates, not fresh independent validation',
    rules='missing has no now restriction; only genuinely released cells, including after-gap context, may be read; never hidden/missing/forecast targets')


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def window(t, start, end):
    return t.reindex(index=range(start-5,end+6),columns=UNITS).to_numpy(float,copy=True)


def features(values, unit, length, position):
    """The window is released-only; erase the WHOLE target interval again here."""
    a = values.copy(); j=UNITS.index(unit); end=length+6
    a[6:end,j]=np.nan
    left,right=a[5,j],a[end,j]
    if not np.isfinite(left+right):return None
    p=position/(length+1); scale=max((left+right)/2,5.)
    linear=(1-p)*left+p*right; envelope=4*p*(1-p)
    slopes=[];counts=[]
    for idx in [np.arange(0,6),np.arange(end,end+6)]:
        good=idx[np.isfinite(a[idx,j])]
        if len(good)<2:return None
        x=good-good.mean();slopes.append(float(x@a[good,j]/(x@x)))
        counts.append(len(good))
    x=[1.,p,p*p,np.log1p(length),np.log1p(scale),left/scale,right/scale,
       (right-left)/scale,slopes[0]*(length+1)/scale,slopes[1]*(length+1)/scale,
       counts[0]/6,counts[1]/6]
    donors=0
    for k in range(8):
        av,bv,cv=a[5,k],a[end,k],a[5+position,k]
        valid=k!=j and np.isfinite(av+bv+cv)
        if valid:
            ds=max((av+bv)/2,5.)
            x.extend([(cv-((1-p)*av+p*bv))/ds,(bv-av)/ds,1.]);donors+=1
        else:x.extend([0.,0.,0.])
    if donors<SPEC['minimum_donors']:return None
    x.extend([float(unit==u) for u in UNITS[1:]])
    return dict(x=np.array(x),scale=scale,linear=linear,envelope=envelope,donors=donors)


def gap_specs(cases, feeds):
    gaps={}
    for r in cases.itertuples():
        ctx=v2_missing.gap_context(feeds[r.recording],r.unit,int(r.minute))
        assert ctx is not None
        start,end,_,_=ctx;key=f'{r.recording}:{r.unit}:{start}-{end}'
        if key not in gaps:
            a=window(feeds[r.recording],start,end)
            gaps[key]=dict(gap_id=key,recording=r.recording,day=r.day,unit=r.unit,
                start=start,end=end,length=end-start-1,mask=np.isfinite(a),cases=[])
        gaps[key]['cases'].append(r)
    for g in gaps.values():
        # Natural absent training counts mean six fixed gaps have partial labels.
        # Preserve every fixed target; never manufacture the unavailable answers.
        assert set(int(r.minute) for r in g['cases']).issubset(range(g['start']+1,g['end']))
        g['complete_labels']=len(g['cases'])==g['length']
    return list(gaps.values())


def train_rows(train, specs, excluded):
    fitting=train[train.day.ne(excluded)]
    assert excluded not in set(fitting.day)
    frames=table(fitting);features_out=[];labels=[];weights=[];manifest=[]
    dates=fitting.groupby('recording').day.first().to_dict()
    for name,t in frames.items():
        weekday=pd.Timestamp(dates[name]).day_of_week
        for unit in UNITS:
            templates=[g for g in specs if g['unit']==unit and pd.Timestamp(g['day']).day_of_week==weekday]
            if not templates or unit not in t:continue
            for i,start in enumerate(range(int(t.index.min())+5,int(t.index.max())-25,30)):
                g=templates[i%len(templates)];length=g['length'];end=start+length+1
                original=t.reindex(range(start,end+1))[unit].to_numpy(float)
                if not np.isfinite(original).all():continue
                a=window(t,start,end);a[~g['mask']]=np.nan
                a[6:length+6,UNITS.index(unit)]=np.nan
                rows=[];answers=[]
                for position in range(1,length+1):
                    f=features(a,unit,length,position)
                    if f is not None:
                        rows.append(f);answers.append((original[position]-f['linear'])/f['scale'])
                if not rows:continue
                # Equal total weight per synthetic gap, irrespective of its length.
                features_out.extend(rows);labels.extend(answers);weights.extend([1/len(rows)]*len(rows))
                manifest.append(dict(recording=name,day=dates[name],unit=unit,start=start,end=end,
                                     template=g['gap_id'],rows=len(rows)))
    assert features_out
    return features_out,np.array(labels),np.array(weights),pd.DataFrame(manifest)


def fit(rows,y,weights):
    x=np.stack([r['x'] for r in rows]);e=np.array([r['envelope'] for r in rows])
    weights=weights*len(weights)/weights.sum()
    scaler=StandardScaler().fit(x,sample_weight=weights)
    design=scaler.transform(x)*e[:,None]
    assert np.isfinite(design).all() and np.isfinite(y).all()
    model=Ridge(alpha=SPEC['mean_loss_l2']*weights.sum(),fit_intercept=False)
    model.fit(design,y,sample_weight=weights)
    return scaler,model


def predict(f,model):
    scaler,reg=model
    residual=float(reg.predict(scaler.transform(f['x'][None,:])*f['envelope'])[0])*f['scale']
    cap=.5*max(f['linear'],5)*f['envelope']
    value=max(0.,f['linear']+SPEC['shrink']*np.clip(residual,-cap,cap))
    assert np.isfinite(value)
    return value


def endpoint_predict(a,unit,length,position,model,fallback):
    j=UNITS.index(unit)
    if position==0:return float(a[5,j])
    if position==length+1:return float(a[length+6,j])
    f=features(a,unit,length,position)
    return predict(f,model) if f is not None else fallback


def summaries(scored):
    for name in ['current','candidate','linear']:
        scored[f'{name}_error']=(scored[name]-scored.truth).abs()
    scored['changed']=(scored.candidate-scored.current).abs().gt(1e-9)
    scored['gap_band']=pd.cut(scored.length,[0,5,10,20,30],labels=['1-5','6-10','11-20','21-30']).astype(str)
    def aggregate(g):
        result=dict(targets=len(g),gaps=g.gap_id.nunique(),eligible=int(g.used.sum()),
            fallback=int((~g.used).sum()),changed=int(g.changed.sum()))
        reference=g.linear_error.mean()
        for n in ['current','candidate','linear']:
            result[n+'_mae']=g[n+'_error'].mean()
            result[n+'_normalized']=result[n+'_mae']/reference if reference>0 else np.nan
        per_gap=g.groupby('gap_id')[[n+'_error' for n in ['current','candidate','linear']]].mean()
        for n in ['current','candidate','linear']:
            result['equal_gap_'+n+'_mae']=per_gap[n+'_error'].mean()
            result['equal_gap_'+n+'_normalized']=(per_gap[n+'_error']/per_gap.linear_error).mean()
        result['gain']=result['current_mae']-result['candidate_mae']
        return result
    for subset,g in [('all',scored),('changed',scored[scored.changed]),('fully_labeled',scored[scored.complete_labels])]:
        for groups,suffix in [(['day'],'date'),(['unit'],'camera'),(['day','unit'],'date_camera'),
                              (['length'],'length'),(['gap_band'],'length_band'),(['day','gap_id'],'gap')]:
            rows=[]
            for key,q in g.groupby(groups):
                key=key if isinstance(key,tuple) else (key,)
                rows.append(dict(zip(groups,key))|aggregate(q))
            pd.DataFrame(rows).to_csv(OUT/f'{subset}_by_{suffix}.csv',index=False)
        pooled=aggregate(g)
        gaps=g.groupby('gap_id')[[f'{n}_error' for n in ['current','candidate','linear']]].mean()
        for n in ['current','candidate','linear']:
            pooled['equal_gap_'+n+'_mae']=gaps[n+'_error'].mean()
            pooled['equal_gap_'+n+'_normalized']=float((gaps[n+'_error']/gaps.linear_error).mean())
        pd.DataFrame([pooled]).to_csv(OUT/f'{subset}_pooled.csv',index=False)
    gaps=scored.groupby(['day','gap_id']).agg(unit=('unit','first'),length=('length','first'),
        start=('start','first'),end=('end','first'),targets=('case_id','size'),
        used=('used','sum'),current_mae=('current_error','mean'),candidate_mae=('candidate_error','mean'),
        linear_mae=('linear_error','mean'),current_total=('current_error','sum'),candidate_total=('candidate_error','sum')).reset_index()
    gaps['reduction']=gaps.current_total-gaps.candidate_total
    gaps['complete_labels']=gaps.targets.eq(gaps.length)
    gaps.to_csv(OUT/'complete_gaps.csv',index=False)
    rows=[]
    for day,q in [('pooled',scored),*list(scored.groupby('day'))]:
        gg=gaps if day=='pooled' else gaps[gaps.day==day]
        best=gg.nlargest(2,'reduction').gap_id.tolist()
        kept=q[~q.gap_id.isin(best)]
        rows.append(dict(day=day,removed=';'.join(best),**aggregate(kept)))
    removed=pd.DataFrame(rows);removed.to_csv(OUT/'without_two_best_gaps.csv',index=False)
    days=pd.read_csv(OUT/'all_by_date.csv');cams=pd.read_csv(OUT/'all_by_date_camera.csv')
    bad=cams[(cams.targets>=10)&((cams.candidate_mae-cams.current_mae)>np.maximum(.5,.1*cams.current_mae))]
    wins=gaps.assign(win=gaps.reduction.gt(1e-9),loss=gaps.reduction.lt(-1e-9)).groupby('day').agg(
        gaps=('gap_id','size'),improved=('win','sum'),worsened=('loss','sum'))
    wins['unchanged']=wins.gaps-wins.improved-wins.worsened
    wins.to_csv(OUT/'gap_counts.csv')
    accepted=bool((days.gain>=.01*days.current_mae).all() and wins.improved.ge(3).all()
        and removed.gain.gt(0).all() and bad.empty)
    scored.to_csv(OUT/'scored.csv',index=False)
    return accepted,dict(gap_counts=wins.to_dict('index'),largest_regression=gaps.loc[gaps.reduction.idxmin()].to_dict(),
                         camera_regressions=bad.to_dict('records'),without_two_best=removed.to_dict('records'))


def full_benchmark_and_submission(updates,train,test,specs):
    """Only called after acceptance: same fixed full score, no T05 inference or fit."""
    import csv
    import local_score
    original=pd.read_csv(LOCAL/'predictions.csv')
    assert len(original)==3719
    candidate=original.copy();indexed=updates.set_index('case_id').candidate
    mask=candidate.case_id.isin(indexed.index)
    candidate.loc[mask,'prediction']=candidate.loc[mask,'case_id'].map(indexed)
    original_out=local_score.OUT
    for name,frame in [('full_baseline',original),('full_candidate',candidate)]:
        path=OUT/name;path.mkdir(exist_ok=True)
        # score reads labels from its own output dir; copy exact fixed bytes.
        (path/'labels.csv').write_bytes((LOCAL/'labels.csv').read_bytes())
        frame.to_csv(path/'predictions.csv',index=False)
        local_score.OUT=path
        with contextlib.redirect_stdout(io.StringIO()):local_score.score(frame,0)
    local_score.OUT=original_out
    rows,y,w,manifest=train_rows(train,specs,None);model=fit(rows,y,w)
    manifest.to_csv(OUT/'production_fit_manifest.csv',index=False)
    feeds=table(test[test.id.isna() & test.people.notna()])
    with BASE.open(encoding='utf-8-sig',newline='') as handle:source=list(csv.DictReader(handle))
    by_id={r['id']:r for r in source};changes={}
    for r in test[test.kind.eq('missing')&test.unit.isin(UNITS)].itertuples():
        ctx=v2_missing.gap_context(feeds[r.recording],r.unit,int(r.minute_of_day))
        if ctx is None:continue
        start,end,_,_=ctx;f=features(window(feeds[r.recording],start,end),r.unit,end-start-1,int(r.minute_of_day)-start)
        if f is not None:changes[r.id]=predict(f,model)
    for k,v in changes.items():by_id[k]['people']=repr(float(v))
    path=ROOT/'outputs/submission_t05_missing_shape_v1.csv'
    if path.exists():raise FileExistsError(f'Refusing overwrite: {path}')
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(source)
    actual=pd.read_csv(path);sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv')
    assert len(actual)==4457 and actual.id.tolist()==sample.id.tolist() and actual.id.is_unique
    assert np.isfinite(actual.people).all() and actual.people.ge(0).all()
    with BASE.open(encoding='utf-8-sig',newline='') as handle:old=list(csv.DictReader(handle))
    assert all(a['people']==b['people'] for a,b in zip(old,source) if a['id'] not in changes)
    return dict(submission=str(path),supported_rows=len(changes),production_model_fits=1)


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'t05_forecast').glob('*.py'))
    paths+=list((ROOT/'outputs').rglob('submission*.csv'))+list(LOCAL.glob('*.*'))
    paths+=list((ROOT/'models').rglob('*.*'))+list((ROOT/'data/raw').glob('*.csv'))
    paths+=[ROOT/'notebooks/current_baseline.ipynb']
    protected={str(p):sha(p) for p in paths if p.is_file() and p.resolve()!=Path(__file__).resolve()}
    for name,expected in json.loads((LOCAL/'case_hashes.json').read_text()).items():assert sha(LOCAL/name)==expected
    (OUT/'design.json').write_text(json.dumps(SPEC,indent=2))
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    cases=pd.read_csv(LOCAL/'cases.csv');cases=cases[cases.kind.eq('missing')].copy()
    feeds=table(pd.read_csv(LOCAL/'released_feed.csv'));specs=gap_specs(cases,feeds)
    assert len(cases)==358 and len(specs)==38
    metadata=[{k:v for k,v in g.items() if k not in ['mask','cases']}|dict(released_cells=int(g['mask'].sum())) for g in specs]
    pd.DataFrame(metadata).to_csv(OUT/'fixed_gap_metadata.csv',index=False)
    rows_out=[];fit_info=[];endpoints=0;leakchecks=0;models={}
    with threadpool_limits(limits=4):
        for day,today in cases.groupby('day'):
            rows,y,weights,manifest=train_rows(train,specs,day)
            # Whole-date exclusion must hold even if all held-out answers change.
            altered=train.copy();altered.loc[altered.day.eq(day),'people']=1e8
            pd.testing.assert_frame_equal(train[train.day.ne(day)],altered[altered.day.ne(day)])
            manifest.to_csv(OUT/f'training_manifest_{day}.csv',index=False)
            model=fit(rows,y,weights);models[day]=model
            fit_info.append(dict(day=day,training_targets=len(rows),synthetic_gaps=len(manifest),
                                 training_dates=sorted(manifest.day.unique()),features=len(rows[0]['x'])))
            calibration=v2_missing.fit_calibration(table(train[train.day.ne(day)]))
            for g in [s for s in specs if s['day']==day]:
                t=feeds[g['recording']];a=window(t,g['start'],g['end']);unit=g['unit'];length=g['length']
                # Fill every unavailable cell, including all target and hidden values,
                # then enforce the original release mask again before extraction.
                poison=a.copy();poison[~np.isfinite(a)]=1e8
                poison[~g['mask']]=np.nan
                # features itself must erase the complete target interior.
                poison[6:length+6,UNITS.index(unit)]=1e8
                for position in [0,length+1]:
                    assert endpoint_predict(a,unit,length,position,model,np.nan)==a[5+position,UNITS.index(unit)]
                    endpoints+=1
                for r in g['cases']:
                    position=int(r.minute)-g['start'];f=features(a,unit,length,position)
                    f2=features(poison,unit,length,position)
                    assert (f is None)==(f2 is None)
                    if f is not None:np.testing.assert_array_equal(f['x'],f2['x'])
                    linear,current,_=baseline(t,unit,int(r.minute),calibration)
                    prediction=predict(f,model) if f is not None else current
                    leakchecks+=1
                    rows_out.append(dict(case_id=r.case_id,day=day,recording=r.recording,unit=unit,
                        minute=r.minute,gap_id=g['gap_id'],start=g['start'],end=g['end'],length=length,
                        complete_labels=g['complete_labels'],used=f is not None,current=current,linear=linear,candidate=prediction))
    detail=pd.DataFrame(rows_out)
    assert np.isfinite(detail[['candidate','current','linear']]).all().all()
    cached=pd.read_csv(LOCAL/'predictions.csv').set_index('case_id')
    np.testing.assert_allclose(detail.current,cached.loc[detail.case_id,'prediction'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(detail.linear,cached.loc[detail.case_id,'reference'],rtol=0,atol=1e-9)
    detail.to_csv(OUT/'predictions.csv',index=False)
    # Real coverage is assessed without target labels or fitting on test observations.
    actual_feeds=table(test[test.id.isna()&test.people.notna()]);coverage=[]
    for r in test[test.kind.eq('missing')].itertuples():
        supported=False
        if r.unit in UNITS:
            ctx=v2_missing.gap_context(actual_feeds[r.recording],r.unit,int(r.minute_of_day))
            if ctx is not None:
                start,end,_,_=ctx
                supported=features(window(actual_feeds[r.recording],start,end),r.unit,end-start-1,int(r.minute_of_day)-start) is not None
        coverage.append(dict(id=r.id,day=r.day,unit=r.unit,group='A-H' if r.unit in UNITS else 'I-M',supported=supported))
    coverage=pd.DataFrame(coverage);coverage.to_csv(OUT/'real_coverage.csv',index=False)
    coverage.groupby(['group','day']).agg(targets=('id','size'),supported=('supported','sum')).to_csv(OUT/'real_coverage_summary.csv')
    # All predictions and metadata coverage are saved BEFORE evaluation labels enter.
    scored=detail.merge(pd.read_csv(LOCAL/'labels.csv'),on='case_id',validate='one_to_one')
    assert len(scored)==358
    accepted,result=summaries(scored)
    extra=full_benchmark_and_submission(detail,train,test,specs) if accepted else {}
    assert all(sha(p)==h for p,h in protected.items())
    result.update(accepted=accepted,model_fits=2,t05_fits=0,t05_inference=0,
        runtime_seconds=time.perf_counter()-started,fit_info=fit_info,endpoints_checked=endpoints,
        unavailable_input_checks=leakchecks,protected_files=len(protected),protected_files_unchanged=True,
        protected_sha256=protected,**extra)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'all_by_date.csv').to_string(index=False))
    print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))


if __name__=='__main__':main()
