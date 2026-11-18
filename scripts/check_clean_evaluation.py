"""Run the documented evaluation in a new source/assets/raw-only project copy.

No old predictions, outputs or .git are copied. --dependency-dir is an optional
installed third-party package directory, e.g. this machine's .vendor; a normal
venv needs no such argument. Never deletes or overwrites another directory.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--dependency-dir',type=Path)
    args=parser.parse_args();started=time.perf_counter()
    dest=args.output_dir.resolve()
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);project=dest/'project';project.mkdir()
    protected=[ROOT/'notebooks/current_baseline.ipynb',ROOT/'src/baseline_tasks.py',
        ROOT/'src/baseline_workflow.py',ROOT/'src/current_baseline.py']
    protected+=list((ROOT/'models/current_baseline').rglob('*.*'))
    protected+=list((ROOT/'outputs').glob('submission*.csv'))
    before={str(p):digest(p) for p in protected}
    sources=[]
    for base in ('src','t05_forecast','tests'):
        sources+=list((ROOT/base).glob('*.py'))
    sources+=list((ROOT/'models').rglob('*.npz'))+list((ROOT/'models').rglob('*.json'))
    sources+=list((ROOT/'models').rglob('*.csv'))
    sources += [ROOT/'t05_forecast_config.json',ROOT/'requirements-evaluation.txt']
    sources += [ROOT/'data/raw'/n for n in ('train.csv','test.csv','sample_submission.csv','recordings.csv')]
    for source in sources:
        target=project/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    (project/'outputs').mkdir()
    env=os.environ.copy();paths=[str(project),str(project/'src')]
    if args.dependency_dir:paths.append(str(args.dependency_dir.resolve()))
    env['PYTHONPATH']=os.pathsep.join(paths);env['PYTHONIOENCODING']='utf-8'
    commands=[['-m','unittest','discover','-s','tests','-p','test_evaluation_protocol.py']]
    for setting in ('historical','representative','im-proxy'):
        command=['src/reliable_evaluation.py','--setting',setting,'--output-dir',f'outputs/{setting}']
        if setting!='im-proxy':command+=['--allow-legacy-provenance']
        commands.append(command)
    runs=[]
    for i,command in enumerate(commands):
        with (dest/f'run_{i}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,*command],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
        runs.append(dict(command=command,exit_code=result.returncode))
        if result.returncode:raise RuntimeError(f'Clean run failed; see {dest/f"run_{i}.log"}')
    assert before=={str(p):digest(p) for p in protected}
    manifests={s:json.loads((project/'outputs'/s/'run_manifest.json').read_text())
               for s in ('historical','representative','im-proxy')}
    report=dict(source_files=len(sources),copied_old_prediction_files=0,
        dependencies='Existing installed packages, requirements-evaluation.txt; fresh network install not tested',
        commands=runs,protected_files_unchanged=len(before),
        t05_fits=sum(m['t05_fits'] for m in manifests.values()),
        settings={s:{k:m[k] for k in ('cases','checks','project_historical_summary','runtime_seconds')} for s,m in manifests.items()},
        runtime_seconds=time.perf_counter()-started)
    (dest/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
