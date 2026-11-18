"""P46: local Hermite geometry only; no learned target or donor correction."""
from __future__ import annotations
import numpy as np
import pandas as pd
import v2_missing

def geometry(frame,unit,minute,context_minutes):
    """Nearest true endpoints and exterior OLS tangents, using minute coordinates.

    At least one EXTRA released observation per side is required. No target
    interior or other camera is read. No interpolation of context readings.
    """
    if unit not in frame:return None
    series=frame[unit].dropna().sort_index()
    observed=series.get(minute,np.nan)
    if np.isfinite(observed):return dict(observed=float(observed))
    ctx=v2_missing.gap_context(frame,unit,minute)
    if ctx is None:return None
    start,end,_,_=ctx
    left=series.loc[start-context_minutes:start]
    right=series.loc[end:end+context_minutes]
    if len(left)<2 or len(right)<2:return None
    def slope(values):
        x=values.index.to_numpy(float);x=x-x.mean()
        return float(x@values.to_numpy(float)/(x@x))
    a,b=float(series.loc[start]),float(series.loc[end])
    m0,m1=slope(left),slope(right);secant=(b-a)/(end-start)
    # Fritsch-Carlson sufficient monotonicity limiter for this interval:
    # slopes agree with secant; normalized tangent vector has norm <=3.
    # A flat secant receives zero tangents. This is not scipy's PCHIP rule.
    original=(m0,m1)
    if secant==0:m0=m1=0.
    else:
        if m0*secant<=0:m0=0.
        if m1*secant<=0:m1=0.
        norm=np.hypot(m0/secant,m1/secant)
        if norm>3:m0*=3/norm;m1*=3/norm
    return dict(start=start,end=end,a=a,b=b,m0=m0,m1=m1,left_n=len(left),right_n=len(right),
        limited=(m0,m1)!=original)

def value(shape,minute):
    if 'observed' in shape:return shape['observed']
    a,b=shape['a'],shape['b'];start,end=shape['start'],shape['end']
    if minute==start:return a
    if minute==end:return b
    assert start<minute<end
    p=(minute-start)/(end-start);h=end-start
    result=(2*p**3-3*p*p+1)*a+(p**3-2*p*p+p)*h*shape['m0']+(-2*p**3+3*p*p)*b+(p**3-p*p)*h*shape['m1']
    # The limiter gives monotonicity. Clip only floating-point boundary noise.
    return float(np.clip(result,max(0.,min(a,b)),max(a,b)))

def predict(frame,unit,minute,current,setting):
    shape=geometry(frame,unit,minute,int(setting['context']))
    if shape is None:return float(current),dict(used=False)
    h=value(shape,minute)
    if 'observed' in shape:return h,dict(used=False,endpoint=True)
    w=float(setting['weight'])
    result=max(0.,float(current+w*(h-current)))
    assert np.isfinite(result)
    return result,dict(used=True,hermite=h,**shape)
