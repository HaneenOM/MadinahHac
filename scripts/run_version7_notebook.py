r"""Execute notebooks/version7.ipynb with this interpreter (the version 7 environment); never touch other notebooks.

Usage (repository root, PowerShell):
    .\.venv_lab\Scripts\python.exe scripts\run_version7_notebook.py
Writes outputs/version7/version7_executed.ipynb and prints outputs/version7/verification.json.
"""
from pathlib import Path
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
kernel_root = ROOT / 'outputs/version7/jupyter'
kernel = kernel_root / 'kernels/madinah-version7'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Madinah version 7', 'language': 'python'}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(kernel_root) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
os.environ.setdefault('MADINAH_PYTHON', sys.executable)
notebook = nbformat.read(ROOT / 'notebooks/version7.ipynb', as_version=4)
started = time.perf_counter()
executed = NotebookClient(notebook, timeout=3600, kernel_name='madinah-version7',
                          resources={'metadata': {'path': str(ROOT / 'notebooks')}}).execute()
path = ROOT / 'outputs/version7/version7_executed.ipynb'
nbformat.write(executed, path)
print(path.relative_to(ROOT))
print(f'notebook wall time: {time.perf_counter() - started:.0f} s')
print((ROOT / 'outputs/version7/verification.json').read_text(encoding='utf-8'))
