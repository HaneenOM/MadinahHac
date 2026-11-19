"""V7 missing stack with convex candidate weights; A-H only. No assessment fitting."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[3]
for path in (ROOT,ROOT/'src',ROOT/'noran/tracks/missing'):
    if str(path) not in sys.path: sys.path.insert(0,str(path))
spec=importlib.util.spec_from_file_location('p56_missing_original',ROOT/'noran/tracks/missing/model.py')
v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)

def constrained(x,y,ridge,loss):
    n,k=x.shape
    penalty=ridge*(x*x).mean()+1e-9
    w=np.zeros(k)
    for iteration in range(26 if loss=='l1' else 1):
        rw=np.ones(n) if iteration==0 else 1/np.maximum(np.abs(y-x@w),.05*np.abs(y).mean()+1e-9)
        xx=x.T@(x*rw[:,None])/n
        xy=x.T@(y*rw)/n
        reg=penalty*rw.mean()
        h=xx+reg*np.eye(k)
        fit=minimize(lambda z:float(z@h@z-2*z@xy),w,jac=lambda z:2*(h@z-xy),
                     bounds=[(0.,1.)]*k,constraints=[{'type':'ineq','fun':lambda z:1-z.sum(),'jac':lambda z:-np.ones(k)}],
                     method='SLSQP',options={'maxiter':150,'ftol':1e-9})
        if not fit.success: raise RuntimeError(f'Convex fit failed: {fit.message}')
        w=fit.x
    assert w.min()>-1e-7 and w.sum()<1+1e-7
    return w

def from_rows(training,context,rows):
    weights=v7.stack.fit_spec(rows,v7.SPEC,ridge=v7.RIDGE)
    baseline=dict(weights=weights,calibration=v7.tasks.fit_missing(training,context['config']),
                  config=context['config'],fit_rows=len(rows))
    alternate={u:{s:dict(m,w=m['w'].copy()) for s,m in subsets.items()} for u,subsets in weights.items()}
    stats=[]
    for unit,subsets in alternate.items():
        if unit[-1] not in 'ABCDEFGH':continue
        setting=v7.SPEC[unit]
        frame=rows[rows.unit.isin(setting['pool'])] if setting.get('pool') else rows[rows.unit.eq(unit)]
        base=frame.base.to_numpy(float);y=frame.truth.to_numpy(float)-base
        scale=np.maximum(base,5) if setting.get('relative') else np.ones(len(base))
        for subset,m in subsets.items():
            if not m['columns']:continue
            x=frame[m['columns']].to_numpy(float)-base[:,None]
            ok=np.isfinite(x).all(axis=1)
            old=m['w'].copy()
            m['w']=constrained(x[ok]/scale[ok,None],y[ok]/scale[ok],v7.RIDGE,setting.get('loss','l2'))
            stats.append(dict(unit=unit,subset=list(subset),rows=int(ok.sum()),old=old.tolist(),new=m['w'].tolist(),
                              negative_old=int((old<0).sum()),old_sum=float(old.sum())))
    return dict(baseline=baseline,trial=dict(baseline,weights=alternate),weight_audit=stats)

def fit(training,day,context):
    rows=v7.pseudo_rows(training,context['test_released'],context['config'])
    return from_rows(training,context,rows)

def predict(case,t,model,context):
    return v7.predict(case,t,model['trial'],context)
