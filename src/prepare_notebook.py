"""Make a locally executable copy without changing the supplied notebook."""
import json
from pathlib import Path

source = Path('notebooks/starter-official.ipynb')
nb = json.loads(source.read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source'])
    code = code.replace('INPUT = "/kaggle/input"', 'INPUT = "data/raw"')
    code = code.replace('OUT = "/kaggle/working"', 'OUT = "outputs"')
    code = code.replace('os.path.join(OUT, "submission.csv")',
                        'os.path.join(OUT, "starter_submission.csv")')
    cell['source'] = code.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None
Path('notebooks/starter-local.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
