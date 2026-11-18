"""Execute the teammate notebook with this interpreter; never touch starter notebooks."""
from pathlib import Path
import json
import os
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
kernel_root = ROOT / 'outputs/current_baseline_0800/jupyter'
kernel = kernel_root / 'kernels/madinah-baseline'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Madinah baseline', 'language': 'python'}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(kernel_root) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
notebook = nbformat.read(ROOT / 'notebooks/current_baseline.ipynb', as_version=4)
executed = NotebookClient(notebook, timeout=1800, kernel_name='madinah-baseline',
                          resources={'metadata': {'path': str(ROOT / 'notebooks')}}).execute()
path = ROOT / 'outputs/current_baseline_0800/current_baseline_executed.ipynb'
nbformat.write(executed, path)
print(path.relative_to(ROOT))
print((ROOT / 'outputs/current_baseline_0800/verification.json').read_text(encoding='utf-8'))
