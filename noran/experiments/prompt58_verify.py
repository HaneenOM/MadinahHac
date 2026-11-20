"""Independent artifact, mask and context checks for the focused P58 run."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src', ROOT / 'noran', ROOT / 'noran/experiments'):
    sys.path.insert(0, str(p))
import numpy as np
import pandas as pd
from pipeline import load, locate_data
from prompt58_evaluate import build, TEST_MASK


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/p58_verified')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    manifest = json.loads((ROOT / 'models/version7/manifest.json').read_text(encoding='utf-8'))
    hashes = {}
    for name, expected in manifest['inputs'].items():
        hashes['data/raw/' + name] = sha(ROOT / 'data/raw' / name)
        assert hashes['data/raw/' + name] == expected
    for name, expected in manifest['assets'].items():
        hashes[name] = sha(ROOT / name)
        assert hashes[name] == expected
    ref = ROOT / 'outputs/version7/reference/submission_v7_board_0.724.csv'
    assert sha(ref) == manifest['accepted_submission_sha256']
    hashes[str(ref.relative_to(ROOT))] = sha(ref)
    changed = subprocess.check_output(['git','diff','HEAD','--name-only'], cwd=ROOT, text=True).splitlines()
    assert not changed, f'Tracked files were changed: {changed}'
    train, test, _, _ = load(locate_data(None))
    spec = importlib.util.spec_from_file_location('p58_prior_protocol', ROOT / 'noran/experiments/prompt54_hf.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    context_rows, summaries = [], []
    for which in ('fixed', 'mirror', 'testday'):
        cases, _, feeds, masks = build(which, train, test)
        old_cases, _, _ = prior.protocol(which, train, test)
        assert set(cases.case_id) == set(old_cases.case_id), which
        p = pd.read_csv(out / which / 'predictions.csv')
        d = pd.read_csv(out / which / 'scored.csv')
        assert p.case_id.is_unique and set(p.case_id) == set(cases.case_id)
        assert (p.horizon == p.minute-p.now).all() and p.horizon.between(1,15).all()
        assert np.isfinite(p[['prediction','baseline','reference']]).all().all() and (p.prediction>=0).all()
        assert np.array_equal(p.prediction.to_numpy(), d.prediction.to_numpy())
        actual_changed = int((p.prediction != p.baseline).sum())
        assert actual_changed == int(p.corrected.sum())
        for r in cases.drop_duplicates(['recording','unit','now']).itertuples():
            t = feeds[r.recording].drop(columns=masks[r.recording], errors='ignore').loc[:r.now]
            recent = t.reindex(range(r.now-14,r.now+1)).reindex(columns=[f'cam_{c}' for c in 'ABCDEFGH'])
            counts = recent.notna().sum(axis=1)
            context_rows.append(dict(protocol=which, recording=r.recording, unit=r.unit, now=r.now,
                observed_minutes=int((counts>=2).sum()), donors_now=int(counts.iloc[-1]),
                donor_mean=float(counts.mean()), span=int(np.ptp(counts.index[counts>=2])) if (counts>=2).any() else 0))
        result = json.loads((out / which / 'result.json').read_text())
        summaries.append(dict(protocol=which, rows=len(p), actual_changed_rows=actual_changed,
                              camera_windows=len(cases.drop_duplicates(['recording','unit','now'])),
                              temporal_windows=len(cases.drop_duplicates(['recording','now'])), **{k:result[k] for k in
                               ['baseline_mae','new_mae','baseline_norm','new_norm','corrected_rows','fallback_rows','invariance_checks','runtime_seconds']}))
        for name in ['cases.csv','predictions.csv','scored.csv','result.json']:
            hashes[str((out / which / name).relative_to(ROOT))] = sha(out / which / name)
    for rec, hidden in TEST_MASK.items():
        g = test[test.recording.eq(rec)]
        feed = g[g.id.isna() & g.people.notna() & ~g.unit.isin(hidden)].pivot(index='minute_of_day', columns='unit', values='people').sort_index()
        requests = g[g.kind.eq('hidden-future') & g.id.notna()]
        for (unit, now), _ in requests.groupby(['unit','now_minute_of_day']):
            now = int(now)
            recent = feed.loc[:now].reindex(range(now-14,now+1)).reindex(columns=[f'cam_{c}' for c in 'ABCDEFGH'])
            counts = recent.notna().sum(axis=1)
            context_rows.append(dict(protocol='competition', recording=rec, unit=unit, now=now,
                observed_minutes=int((counts>=2).sum()), donors_now=int(counts.iloc[-1]),
                donor_mean=float(counts.mean()), span=int(np.ptp(counts.index[counts>=2])) if (counts>=2).any() else 0))
    context = pd.DataFrame(context_rows)
    context.to_csv(out / 'context_by_window.csv', index=False)
    context.groupby(['protocol','unit']).agg(windows=('now','size'), minutes_min=('observed_minutes','min'),
          minutes_mean=('observed_minutes','mean'),donors_min=('donors_now','min'),donors_mean=('donors_now','mean')).to_csv(out / 'context_summary.csv')
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in [ROOT / 'noran/tracks/p58_reconstructed_history/model.py',
                   ROOT / 'noran/experiments/prompt58_evaluate.py', ROOT / 'noran/configs/prompt58_reconstructed_history.json',
                   ROOT / 'reports/prompt58_reconstructed_history_ar.md']})
    report = dict(status='passed', hashes=hashes, tracked_files_changed=changed,
                  case_sets_match_existing_protocol=True, candidate_created=False,
                  combined_mask_and_postnow_checks=sum(s['invariance_checks'] for s in summaries))
    (out / 'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(context.groupby(['protocol','unit']).agg(windows=('now','size'),minutes=('observed_minutes','mean'),donors=('donors_now','mean')).to_string())


if __name__ == '__main__':
    main()
