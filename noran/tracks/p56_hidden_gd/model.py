"""Fixed-development-selected G/D scope of the unchanged three-minute HIDDEN smoother."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('p56_smooth_member',ROOT/'noran/tracks/p56_hidden_smooth/model.py')
member=importlib.util.module_from_spec(spec);spec.loader.exec_module(member)
def fit(training,day,context):return member.fit(training,day,context)
def predict(case,t,model,context):
    if case.unit not in ('cam_G','cam_D'):return member.v7.predict(case,t,model,context)
    normalized=SimpleNamespace(day=case.day,recording=case.recording,kind='hidden',unit=case.unit,
                               minute=int(case.minute),case_id=getattr(case,'case_id',''))
    return member.predict(normalized,t,model,context)
