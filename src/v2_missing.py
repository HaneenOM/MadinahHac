"""Endpoint-preserving donor-deviation experiments for short camera outages."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd


def interp_at(s: pd.Series, minute: int, tolerance: int = 2) -> float:
    """Read a released donor value, with short local interpolation if needed."""
    x = s.dropna()
    v = x.get(minute, np.nan)
    if np.isfinite(v):
        return float(v)
    left = x.loc[:minute-1]
    right = x.loc[minute+1:]
    if len(left) and len(right) and minute-left.index[-1] <= tolerance and right.index[0]-minute <= tolerance:
        f = (minute-left.index[-1])/(right.index[0]-left.index[-1])
        return float(left.iloc[-1]*(1-f)+right.iloc[0]*f)
    return np.nan


def gap_context(t: pd.DataFrame, unit: str, minute: int):
    if unit not in t:
        return None
    s = t[unit].dropna()
    left = s.loc[:minute-1]
    right = s.loc[minute+1:]
    if not len(left) or not len(right):
        return None
    start, end = int(left.index[-1]), int(right.index[0])
    if end-start > 30:
        return None
    frac = (minute-start)/(end-start)
    base = float(left.iloc[-1]*(1-frac)+right.iloc[0]*frac)
    return start, end, frac, base


def deviations(t: pd.DataFrame, unit: str, minute: int, context):
    start, end, frac, _ = context
    out = {}
    for donor in t.columns:
        if donor == unit:
            continue
        s = t[donor]
        a = interp_at(s,start)
        b = interp_at(s,end)
        c = interp_at(s,minute)
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
            out[donor] = c-(a*(1-frac)+b*frac)
    return out


def fit_calibration(tables: dict[str,pd.DataFrame]):
    """Pair coefficients learned from complete historical windows only."""
    pairs = defaultdict(list)
    for t in tables.values():
        for start in range(int(t.index.min())+5,int(t.index.max())-20,5):
            for length in (6,10,15,20):
                end=start+length
                for frac in (.25,.5,.75):
                    minute=start+round(length*frac)
                    actual_frac=(minute-start)/length
                    vals={}
                    for unit in t:
                        s=t[unit]
                        a,b,c=(s.get(x,np.nan) for x in (start,end,minute))
                        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
                            vals[unit]=c-(a*(1-actual_frac)+b*actual_frac)
                    for unit,y in vals.items():
                        for donor,x in vals.items():
                            if donor != unit:
                                pairs[(unit,donor)].append((x,y))
    calibrated={}
    for key, samples in pairs.items():
        if len(samples)<40:
            continue
        a=np.asarray(samples)
        x,y=a[:,0],a[:,1]
        slope=float(np.dot(x,y)/(np.dot(x,x)+5000.))
        slope=float(np.clip(slope,-1.5,1.5))
        baseline=float(np.abs(y).mean())
        residual=float(np.abs(y-slope*x).mean())
        calibrated[key]=(slope,baseline,residual,len(a))
    return calibrated


def predict(t: pd.DataFrame, unit: str, minute: int, calibration, method: str, weight: float=1.):
    context=gap_context(t,unit,minute)
    if context is None:
        return np.nan,0,np.nan
    start,end,frac,base=context
    if method=='linear':
        return max(0.,base),0,end-start-1
    dev=deviations(t,unit,minute,context)
    candidates=[]
    for donor,x in dev.items():
        if (unit,donor) not in calibration:
            continue
        slope,baseline,residual,n=calibration[(unit,donor)]
        improvement=baseline-residual
        if improvement<=0:
            continue
        candidates.append((improvement,donor,slope*x))
    if not candidates:
        return max(0.,base),0,end-start-1
    candidates.sort(reverse=True)
    if method=='single':
        correction=candidates[0][2]
    else:
        top=candidates[:4]
        strengths=np.asarray([x[0] for x in top])
        strengths=strengths/(strengths.sum()+1e-9)
        correction=float(np.dot(strengths,[x[2] for x in top]))
    correction=np.clip(correction,-.5*max(base,5),.5*max(base,5))
    return max(0.,float(base+weight*correction)),len(candidates),end-start-1
