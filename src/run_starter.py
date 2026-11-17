"""Execute the locally adapted official notebook from the project root."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
kernel_root = root/'outputs/.kernels'
kernel_dir = kernel_root/'kernels/madinah-local'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Madinah local Python', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(kernel_root) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
nb = nbformat.read(root/'notebooks/starter-local.ipynb', as_version=4)
executed = NotebookClient(nb, timeout=900, kernel_name='madinah-local',
                          resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(executed, root/'notebooks/starter-executed.ipynb')
print(root/'outputs/starter_submission.csv')
