"""Historical camera relationships for simultaneous hidden-camera masks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from v2_missing import interp_at

PATTERNS={'cam_B':{'cam_B','cam_G'},'cam_G':{'cam_B','cam_G'},'cam_D':{'cam_D'}}
REFERENCE_DONOR={'cam_B':'cam_D','cam_G':'cam_E','cam_D':'cam_B'}


def _design(frame: pd.DataFrame, donors: list[str], med: pd.Series, dropout=False, seed=0):
    x=frame.reindex(columns=donors).copy()
    if dropout:
        rng=np.random.default_rng(seed)
        mask=rng.random(x.shape)<.2
        x=x.mask(mask)
    seen=x.notna().astype(float)
    z=pd.DataFrame(index=x.index)
    for c in donors:
        z[f'{c}_value']=x[c].fillna(med[c]).to_numpy()
        z[f'{c}_seen']=seen[c].to_numpy()
    minute=np.asarray(x.index.get_level_values(-1),float)
    z['minute_sin']=np.sin(2*np.pi*minute/1440)
    z['minute_cos']=np.cos(2*np.pi*minute/1440)
    return z


def fit(tables: dict[str,pd.DataFrame]):
    full=pd.concat(tables.values(),keys=tables.keys(),names=['recording','minute_of_day'])
    out={}
    for unit,masked in PATTERNS.items():
        if unit not in full:
            continue
        donors=[c for c in full if c not in masked and c.startswith('cam_')]
        y=full[unit]
        valid=y.notna() & full[donors].notna().sum(axis=1).ge(2)
        if valid.sum()<100:
            continue
        frame=full.loc[valid,donors]
        target=y.loc[valid]
        med=frame.median().fillna(0.)
        X=_design(frame,donors,med)
        ridge=Ridge(alpha=3000.).fit(X,target)
        augmented=[X]
        for seed in range(3):
            augmented.append(_design(frame,donors,med,dropout=True,seed=seed))
        drop=Ridge(alpha=3000.).fit(pd.concat(augmented,ignore_index=True),
                                    np.tile(target.to_numpy(),4))
        pairs=[]
        for c in donors:
            pair=full.loc[valid,[unit,c]].dropna()
            if len(pair)<80:
                continue
            x=pair[c].to_numpy(); yy=pair[unit].to_numpy()
            # The released sample uses the ratio of overlapping mean counts.
            slope=float(np.mean(yy)/np.mean(x))
            err=float(np.abs(yy-slope*x).mean())
            pairs.append((err,c,slope,len(pair)))
        out[unit]={'donors':donors,'med':med,'target_median':float(target.median()),
                   'ridge':ridge,'dropout':drop,
                   'pairs':sorted(pairs,key=lambda z:(z[1]!=REFERENCE_DONOR[unit],z[0]))}
    return out


def predict(t: pd.DataFrame, unit: str, minute: int, models, method='single',
            override: dict[str,float]|None=None):
    if unit not in models:
        return np.nan,0
    info=models[unit]
    values={c:(override[c] if override is not None and c in override else
               interp_at(t[c],minute) if c in t else np.nan) for c in info['donors']}
    available=sum(np.isfinite(v) for v in values.values())
    if method=='single':
        for _,c,slope,_ in info['pairs']:
            if np.isfinite(values[c]):
                return max(0.,slope*values[c]),available
        return info['target_median'],available
    if available<2:
        return predict(t,unit,minute,models,'single',override)
    ix=pd.MultiIndex.from_tuples([('inference',minute)],names=['recording','minute_of_day'])
    frame=pd.DataFrame([values],index=ix)
    X=_design(frame,info['donors'],info['med'])
    p=info['dropout' if method=='dropout' else 'ridge'].predict(X)[0]
    return max(0.,float(p)),available
