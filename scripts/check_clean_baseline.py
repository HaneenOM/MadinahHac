"""Source/assets/raw-only copy, plus isolated copies of documented dependencies.

Never copies submissions, caches, .git or model estimators. No network install
is performed: dependency files are copied from the active documented environment.
"""
from pathlib import Path
import argparse
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import time
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter();dest=args.output_dir.resolve()
    dest.mkdir(parents=True,exist_ok=False);project=dest/'project';project.mkdir()
    for directory in ('src','t05_forecast','scripts','notebooks','configs','models/current_baseline','models/local_evaluation'):
        for source in (ROOT/directory).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix in ('.py','.json','.csv','.npz','.ipynb','.md'):
                target=project/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for source in list(ROOT.glob('requirements*.txt'))+[ROOT/'t05_forecast_config.json']+[ROOT/'data/raw'/n for n in ('train.csv','test.csv','recordings.csv','sample_submission.csv')]:
        target=project/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    assert not (project/'outputs').exists()
    requested=[]
    def requirements(path):
        for line in path.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            if line.startswith('-r '):requirements(path.parent/line[3:])
            else:requested.append(Requirement(line))
    requirements(ROOT/'requirements-baseline.txt')
    queue=list(requested);seen={};deps=dest/'dependencies';deps.mkdir()
    while queue:
        req=queue.pop(0)
        if req.marker and not req.marker.evaluate({'extra':''}):continue
        name=canonicalize_name(req.name)
        if name in seen:continue
        dist=metadata.distribution(req.name)
        assert not req.specifier or dist.version in req.specifier,(req,dist.version)
        seen[name]=dist.version;base=Path(dist.locate_file('')).resolve()
        for entry in dist.files or []:
            source=Path(dist.locate_file(entry)).resolve()
            if not source.is_relative_to(base) or not source.is_file():continue
            relative=source.relative_to(base)
            target=deps/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
        queue.extend(Requirement(r) for r in dist.requires or [])
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(project),str(project/'src'),str(deps)])
    env['PYTHONNOUSERSITE']='1';env['PYTHONIOENCODING']='utf-8'
    commands=[['scripts/run_current_baseline_notebook.py'],['src/evaluate_baseline_task.py','--task','hidden','--output-dir','outputs/clean_hidden']]
    runs=[]
    for i,command in enumerate(commands):
        with (dest/f'run_{i}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-S',*command],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
        runs.append(dict(command=command,exit_code=result.returncode))
        if result.returncode:raise RuntimeError(f'Clean run failed: {dest/f"run_{i}.log"}')
    verified=json.loads((project/'outputs/current_baseline_0800/verification.json').read_text())
    assert verified['sha256_matches_accepted'] and not verified['reference_available']
    report=dict(commands=runs,notebook=verified,dependencies=seen,copied_submission_files=0,copied_fitted_hidden_models=0,
        dependency_install='Isolated copy of declared installed distributions and their active dependency closure; no fresh network pip install.',
        runtime_seconds=time.perf_counter()-started)
    (dest/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='dependencies'},indent=2))

if __name__=='__main__':main()
