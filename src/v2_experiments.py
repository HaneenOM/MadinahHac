"""Date-held-out experiments for hidden, hidden-future, and missing gaps."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pipeline import ROOT, load, locate_data, table, last_pair
import v2_hidden
import v2_missing

DONORS=[f'cam_{c}' for c in 'ABCDEFGH']
DIRECT_COLS=['minute','horizon','target_id']+[f'{c}_{s}' for c in DONORS for s in ('last','age','slope')]


def cases(train,test):
    """Transplant the released Friday/Saturday mask onto each matching training recording."""
    templates={
        'Fri':test[test.recording.eq('20260918_1100')],
        'Sat':test[test.recording.eq('20260919_1100')],
    }
    out=[]
    for rec,g in train.groupby('recording'):
        day=g.day.iloc[0]; weekday=g.weekday.iloc[0]
        mask=templates[weekday]
        mask=mask[mask.unit.isin(DONORS)].copy()
        mask['recording']=rec
        released=mask[mask.id.isna()][['recording','unit','minute_of_day']]
        feed=released.merge(g[['recording','unit','minute_of_day','people']],
                            on=['recording','unit','minute_of_day'],how='inner')
        targets=mask[mask.id.notna() & mask.kind.isin(['hidden','hidden-future','missing'])]\
            .drop(columns='people').merge(g[['recording','unit','minute_of_day','people']],
                         on=['recording','unit','minute_of_day'],how='inner')
        if len(targets):
            out.append((day,rec,table(feed).get(rec,pd.DataFrame()),targets))
    return out


def direct_feature(t,unit,now,h):
    # All paths, including per-donor lags, are cut off before feature construction.
    causal=t.loc[:now].drop(columns=list(v2_hidden.PATTERNS[unit]),errors='ignore')
    values=[now,h,ord(unit[-1])-65]
    for donor in DONORS:
        if donor in causal:
            values.extend(last_pair(causal[donor],now,30))
        else:
            values.extend([np.nan,np.nan,np.nan])
    return values


def fit_direct(tables, *, model_params=None):
    rows=[]; labels=[]
    for t in tables.values():
        for unit in v2_hidden.PATTERNS:
            if unit not in t:
                continue
            y=t[unit]
            for now in range(int(t.index.min())+20,int(t.index.max())-15,8):
                base=direct_feature(t,unit,now,0)
                for h in range(1,16):
                    truth=y.get(now+h,np.nan)
                    if np.isfinite(truth):
                        x=base.copy(); x[1]=h
                        rows.append(x); labels.append(truth)
    if len(rows)<100:
        return None
    X=pd.DataFrame(rows,columns=DIRECT_COLS)
    cols=X.columns[X.notna().any()].tolist()
    params=dict(loss='absolute_error',max_iter=100,learning_rate=.06,
                max_leaf_nodes=12,min_samples_leaf=40,random_state=27)
    params.update(model_params or {})
    model=HistGradientBoostingRegressor(**params)
    model.fit(X[cols],labels)
    return model,cols


def forecast_donors(t,unit,now,h,models):
    causal=t.loc[:now]
    future={}
    for donor in models[unit]['donors']:
        if donor not in causal:
            continue
        last,age,slope=last_pair(causal[donor],now,30)
        if np.isfinite(last):
            # Damped, clipped linear donor forecast; no downstream fit on its outputs.
            future[donor]=max(0.,last+.35*h*np.clip(slope,-2.,2.))
    return future


def predict_hf(t,unit,now,h,models,direct,method):
    causal=t.loc[:now]
    if method=='fixed':
        return v2_hidden.predict(causal,unit,now,models,'single')[0]
    if method=='donor_forecast':
        future=forecast_donors(causal,unit,now,h,models)
        return v2_hidden.predict(causal,unit,now+h,models,'ridge',future)[0]
    x=direct_feature(causal,unit,now,h)
    model,cols=direct
    return max(0.,float(model.predict(pd.DataFrame([x],columns=DIRECT_COLS)[cols])[0]))


def evaluate(train,test,out):
    all_rows=[]; coverage=[]; invariance=0
    for day in sorted(train.day.unique()):
        fold_cases=[x for x in cases(train,test) if x[0]==day]
        if not fold_cases:
            continue
        fit=train[train.day.ne(day)]
        fit_t=table(fit)
        hidden=v2_hidden.fit(fit_t)
        calibration=v2_missing.fit_calibration(fit_t)
        direct=fit_direct(fit_t)
        for _,rec,t,targets in fold_cases:
            coverage.append({'fold':day,'recording':rec,'targets_with_truth':len(targets),
                             **{f'known_{k}':int(targets.kind.eq(k).sum())
                                for k in ('hidden','hidden-future','missing')}})
            for r in targets.itertuples():
                unit=r.unit; minute=int(r.minute_of_day); truth=float(r.people)
                if r.kind=='hidden':
                    preds={m:v2_hidden.predict(t,unit,minute,hidden,m) for m in ('single','ridge','dropout')}
                    reference=preds['single'][0]
                    donor_count=preds['single'][1]
                    gap=np.nan; h=np.nan
                elif r.kind=='hidden-future':
                    now=int(r.now_minute_of_day); h=minute-now
                    preds={m:(predict_hf(t,unit,now,h,hidden,direct,m),0)
                           for m in ('fixed','donor_forecast','direct')}
                    reference=preds['fixed'][0]
                    donor_count=v2_hidden.predict(t.loc[:now],unit,now,hidden,'single')[1]
                    gap=np.nan
                    if len(all_rows)%40==0:
                        changed=t.copy()
                        changed.loc[changed.index>now]=1e6
                        for m in preds:
                            a=preds[m][0]; b=predict_hf(changed,unit,now,h,hidden,direct,m)
                            assert np.isclose(a,b,equal_nan=True),(r.id,m,a,b)
                            invariance+=1
                else:
                    preds={m:v2_missing.predict(t,unit,minute,calibration,m,w)
                           for m,w in [('linear',0),('single',.5),('blend',.5),('blend_full',1.)]}
                    reference=preds['linear'][0]
                    donor_count=preds['blend'][1]
                    gap=preds['linear'][2]; h=np.nan
                if not np.isfinite(reference):
                    continue
                for method,item in preds.items():
                    pred=item[0]
                    if not np.isfinite(pred):
                        continue
                    all_rows.append(dict(fold=day,recording=rec,kind=r.kind,unit=unit,
                                         minute=minute,horizon=h,gap_length=gap,
                                         donor_available=donor_count,truth=truth,
                                         reference=reference,method=method,prediction=pred))
        print('Completed date',day,'cases',len(fold_cases),flush=True)
    detail=pd.DataFrame(all_rows)
    detail.to_csv(out/'v2_validation_predictions.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'v2_coverage.csv',index=False)
    scores=score(detail,['fold','recording','kind','method'])
    scores.to_csv(out/'v2_scores_by_recording.csv',index=False)
    score(detail,['kind','unit','method']).to_csv(out/'v2_scores_by_camera.csv',index=False)
    hf=detail[detail.kind.eq('hidden-future')].copy()
    hf['horizon_band']=pd.cut(hf.horizon,[0,5,10,15],labels=['1-5','6-10','11-15'])
    score(hf,['fold','horizon_band','method']).to_csv(out/'v2_scores_horizon.csv',index=False)
    miss=detail[detail.kind.eq('missing')].copy()
    miss['gap_band']=pd.cut(miss.gap_length,[0,5,10,20,30],labels=['1-5','6-10','11-20','21-30'])
    miss['donor_band']=pd.cut(miss.donor_available,[-1,0,2,99],labels=['0','1-2','3+'])
    score(miss,['fold','gap_band','donor_band','method']).to_csv(out/'v2_scores_missing_gap.csv',index=False)
    print('Post-cutoff invariance comparisons:',invariance)
    return detail,scores


def score(detail,groups):
    if detail.empty:
        return pd.DataFrame()
    x=detail.copy()
    x['error']=(x.truth-x.prediction).abs()
    x['reference_error']=(x.truth-x.reference).abs()
    result=x.groupby(groups,dropna=False,observed=True).agg(
        n=('error','size'),mae=('error','mean'),reference_mae=('reference_error','mean')).reset_index()
    result['normalized_mae']=result.mae/result.reference_mae.replace(0,np.nan)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    args=parser.parse_args()
    train,test,_,_=load(locate_data(args.data_dir))
    detail,scores=evaluate(train,test,ROOT/'outputs')
    long=scores[scores.fold.isin(['2026-09-11','2026-09-12'])]
    print(long.to_string(index=False))


if __name__=='__main__':
    main()
