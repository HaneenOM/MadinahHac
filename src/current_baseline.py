"""Frozen-asset preflight and backwards-compatible baseline entry point.

run delegates to the editable task workflow; run_legacy retains the archived
builder chain for historical audit. Neither path refits the T05 ensemble.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hidden_candidate import build as build_hidden, read_submission
from hidden_future_prayer_candidate import build as build_prayer
from im_forecast_fallback_trial import build_candidate as build_im
from t05_forecast.submit import build as build_t05, member_settings, settings
from v2_candidate import build as build_v2

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'models/current_baseline'
REFERENCE = ROOT / 'outputs/submission_hidden_bgd_v1.csv'
PREVIOUS_REFERENCE = ROOT / 'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(data_dir):
    manifest = json.loads((ASSETS / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['inputs'].items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'Place the authorized competition input at {path}')
        if sha256(path) != expected:
            raise ValueError(f'{name}: input differs from the accepted competition release')
    for name, expected in manifest['assets'].items():
        if sha256(ASSETS / name) != expected:
            raise ValueError(f'Frozen baseline asset differs: {name}')
    for name, expected in manifest['versions'].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'{name}=={expected} required; found {actual}. Install requirements.txt.')
    cfg = json.loads((ROOT / 't05_forecast_config.json').read_text(encoding='utf-8'))
    if cfg != manifest['config']:
        raise ValueError('T05 configuration differs from the accepted baseline')
    required = {f't05/params_without_{day}_{member["key"]}.npz'
                for member in member_settings(settings(cfg)) for day in ('fri', 'sat')}
    if len(required) != 12 or not required <= manifest['assets'].keys():
        raise ValueError('All twelve accepted T05 parameter sets must be supplied; refusing to retrain')
    return manifest, cfg


def run_legacy(data_dir=None, output_dir=None):
    """Generate fresh predictions in a dedicated directory and verify the saved file."""
    started = time.perf_counter()
    data_dir = Path(data_dir) if data_dir else ROOT / 'data/raw'
    out = Path(output_dir) if output_dir else ROOT / 'outputs/current_baseline'
    manifest, cfg = preflight(data_dir)
    manifest=json.loads((ASSETS/'manifest_0806.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    protected = {p: sha256(p) for p in (ROOT / 'outputs').glob('submission*.csv')}
    protected.update({p: sha256(p) for p in (ROOT / 'notebooks').glob('starter*.ipynb')})
    reference = data_dir / 'sample_submission.csv'
    ids = pd.read_csv(reference).id.tolist()
    # v1 retained the supplied sample verbatim for hidden/hidden-future/missing.
    # Its forecast values are immaterial: T05 regenerates every forecast row.
    with (out / 'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        print('1. v2 from supplied reference (no historical submission dependency)', flush=True)
        build_v2(out / 'v2.csv', data_dir=data_dir, reference_path=reference,
                 selection_path=out / 'v2_selection.json')
        print('2. T05 inference using twelve frozen parameter sets', flush=True)
        cfg['base_submission'] = str(out / 'v2.csv')
        build_t05(data_dir, ASSETS / 't05', out / 't05', cfg, plots=False)
        # Preserve the original integrated builder CSV round trip and formatting.
        pd.read_csv(out / 't05/submission.csv').to_csv(out / 'integrated.csv', index=False)
        print('3. Accepted hidden donor/Ridge weights; D-only disabled', flush=True)
        build_hidden(out / 'hidden.csv', data_dir=data_dir,
                     baseline_path=out / 'integrated.csv', reference_path=reference)
        print('4. Accepted Saturday-only hidden-future prayer features', flush=True)
        build_prayer(out / 'prayer.csv', data_dir=data_dir, baseline_path=out / 'hidden.csv',
                     reference_path=reference, timetable_path=ASSETS / 'prayer_times_madinah.csv')
        print('5. Accepted I-M forecast fallback (no diagnostic or trial fitting)', flush=True)
        # This builder preserves old files; regenerate a fresh intermediate on every run.
        final_stage = out / 'fresh_candidate.csv'
        if final_stage.exists():
            final_stage.unlink()
        build_im(data_dir=data_dir, baseline_path=out / 'prayer.csv',
                 output_path=final_stage, report_dir=out)

    actual = read_submission(final_stage, ids)
    actual_sha = sha256(final_stage)
    verification = dict(ids=len(actual), valid_nonnegative=True, t05_parameter_sets=12,
                        t05_models_fitted=0, d_only_enabled=False,
                        output_sha256=actual_sha,
                        expected_sha256=manifest['accepted_submission_sha256'],
                        sha256_matches_accepted=actual_sha == manifest['accepted_submission_sha256'],
                        reference_available=PREVIOUS_REFERENCE.exists())
    if PREVIOUS_REFERENCE.exists():
        expected = {r['id']: float(r['people']) for r in read_submission(PREVIOUS_REFERENCE, ids)}
        difference = np.array([abs(float(r['people']) - expected[r['id']]) for r in actual])
        kinds = pd.read_csv(data_dir / 'test.csv').dropna(subset=['id']).set_index('id').kind
        verification.update(reference_sha256=sha256(PREVIOUS_REFERENCE),
                            max_absolute_difference=float(difference.max()),
                            unequal_values=int((difference != 0).sum()),
                            mismatches_over_1e_9=int((difference > 1e-9).sum()),
                            by_task={kind: dict(targets=int((kinds.loc[ids] == kind).sum()),
                                               unequal_values=int(((difference != 0) & (kinds.loc[ids] == kind)).sum()))
                                     for kind in kinds.unique()})
    verification['protected_files_unchanged'] = all(sha256(p) == value for p, value in protected.items())
    verification['runtime_seconds'] = time.perf_counter() - started
    (out / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    if not verification['protected_files_unchanged']:
        raise AssertionError('A protected file changed')
    if not verification['sha256_matches_accepted']:
        raise AssertionError(f'Exact baseline not reproduced: inspect {out / "verification.json"}')
    destination = out / 'submission_current_baseline.csv'
    if destination.exists() and sha256(destination) != actual_sha:
        raise FileExistsError(f'Preserving different existing output: {destination}')
    destination.write_bytes(final_stage.read_bytes())
    read_submission(destination, ids)
    print(json.dumps(verification, indent=2))
    print(f'Output: {destination.relative_to(ROOT)}')
    return verification


def run(data_dir=None, output_dir=None):
    """Task-level baseline; kept as an entry point for existing Python callers."""
    from baseline_workflow import run as run_tasks
    return run_tasks(data_dir=data_dir, output_dir=output_dir)
