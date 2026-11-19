"""Three-minute triangular smoothing of V7 HIDDEN predictions on the real minute grid.

Only moments with >=2 visible A-H donors are eligible. No gap is bridged;
missing neighbors simply keep the current-minute contribution. No HF/forecast use.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
for path in (ROOT,ROOT/'src'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
spec=importlib.util.spec_from_file_location('p56_hidden_original',ROOT/'noran/tracks/hidden_v4/model.py')
v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
MASK={'cam_B':('cam_B','cam_G'),'cam_G':('cam_B','cam_G'),'cam_D':('cam_D',)}

def fit(training,day,context):return v7.fit(training,day,context)

def smooth_series(values):
    """No compression of missing minutes, no traversal of recording boundaries."""
    center=np.log1p(values)
    left=np.log1p(values.reindex(values.index-1).set_axis(values.index))
    right=np.log1p(values.reindex(values.index+1).set_axis(values.index))
    return np.expm1((2*center+left.fillna(center)+right.fillna(center))/4)

def predict(case,t,model,context):
    own=v7.predict(case,t,model,context)
    if not np.isfinite(own):return own
    minute=int(case.minute)
    donors=[c for c in t.columns if c.startswith('cam_') and c[-1] in 'ABCDEFGH' and c not in MASK[case.unit]]
    total=2*np.log1p(max(0.,own))
    for offset in (-1,1):
        m=minute+offset
        value=own
        if m in t.index and t.loc[m,donors].notna().sum()>=2:
            neighbor=SimpleNamespace(**{**vars(case),'minute':m}) if hasattr(case,'__dict__') else SimpleNamespace(
                day=case.day,recording=case.recording,unit=case.unit,kind='hidden',minute=m,case_id=f'neighbor_{m}')
            extra=v7.predict(neighbor,t,model,context)
            if np.isfinite(extra):value=extra
        total+=np.log1p(max(0.,value))
    return float(max(0.,np.expm1(total/4)))
