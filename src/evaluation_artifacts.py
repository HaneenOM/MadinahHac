"""Evaluation-only covariance recipes, provenance inventory and fail-closed loading.

Importing an old cache records its present compatibility, NEVER its creation
provenance. No function in this module trains a GP or writes a submission.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
from pipeline import ROOT
from t05_forecast import gp
from t05_forecast.core import load_recordings
from t05_forecast.pipeline import load_params, sources_for
from t05_forecast.submit import member_settings, settings, fit_kwargs
from t05_forecast.template import build_template, complete_template, slope

ASSETS = ROOT / 'models/local_evaluation'
HISTORICAL = {'2026-09-11': '20260911_1100', '2026-09-12': '20260912_1100'}
RECENT = {'2026-09-18': '20260918_1100', '2026-09-19': '20260919_1100'}
FIT_SOURCE = {'20260911_1100': ('20260919_1100', '20260912_1100'),
              '20260912_1100': ('20260918_1100', '20260911_1100')}
GP_CODE = ['src/evaluation_artifacts.py', 't05_forecast/core.py',
           't05_forecast/gp.py', 't05_forecast/template.py',
           't05_forecast/prayer.py', 't05_forecast/pipeline.py', 't05_forecast/submit.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def json_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def environment():
    result={}
    for n in ('numpy','pandas','scipy','scikit-learn'):
        result[n]=importlib.metadata.version(n)
    # GP inference is NumPy/SciPy; torch is only needed by the disabled optimizer.
    return result


def context():
    return dict(inputs={p.name: sha(p) for p in sorted((ROOT/'data/raw').glob('*.csv'))
                        if p.name in ('train.csv','test.csv','sample_submission.csv','recordings.csv')},
                gp_code={n: sha(ROOT/n) for n in GP_CODE}, versions=environment(),
                config=json.loads((ROOT/'t05_forecast_config.json').read_text()))


def members():
    return member_settings(settings(json.loads((ROOT/'t05_forecast_config.json').read_text())))


def empty_target(rec):
    target = rec.copy()
    target.y[:] = np.nan
    target.truth = None
    target.ids = {}
    return target


def excluded_sources(recs, heldout):
    day = recs[heldout].day
    eligible = {n:r for n,r in recs.items() if r.day != day}
    assert all(r.day != day for r in eligible.values())
    return eligible


def independent_template(recs, name, lam, *, complete_new_cameras=False):
    """Target values are erased; all fitting sources exclude its ENTIRE date."""
    target = empty_target(recs[name])
    eligible = excluded_sources(recs, name)
    tmpl = build_template(target, sources_for({**eligible, name:target}, name), lam=lam)
    historical = [r for r in eligible.values() if r.split=='train' and r.n_minutes>=600]
    tmpl, _ = complete_template(target, tmpl, historical, lam=lam)
    if complete_new_cameras:
        other = '20260919_1100' if name=='20260918_1100' else '20260918_1100'
        tmpl, _ = complete_template(target, tmpl, [eligible[other]], lam=lam)
    return tmpl


def field_hash(datasets):
    # This exact legacy digest is also reconstructed to corroborate recent keys.
    legacy = hashlib.sha256()
    full = hashlib.sha256()
    for ds in datasets:
        for k in ('t','c','d','r'):
            a = np.ascontiguousarray(ds[k])
            legacy.update(a.tobytes())
            full.update(f'{k}:{a.dtype}:{a.shape}'.encode()); full.update(a.tobytes())
    return full.hexdigest(), legacy.hexdigest()


def recipes(setting, recs=None):
    recs = recs or load_recordings(ROOT/'data/raw')
    result=[]; fields={}
    names = HISTORICAL if setting=='historical' else RECENT
    for day, heldout in names.items():
        eligible = excluded_sources(recs, heldout)
        for member in members():
            lam = member['lam']
            if setting=='historical':
                other, prior = FIT_SOURCE[heldout]
                source = eligible[other]
                tmpl = build_template(source, [eligible[prior]], lam=lam)
                field = gp.observed(source, tmpl, slope(tmpl), clip=1.5)
                pick = np.linspace(0,len(field['t'])-1,96,dtype=int)
                datasets=[{k:field[k][pick] for k in ('t','c','d','r')}]
                training=[other]; template_sources=[prior]
                key=f'heldout_{heldout}_sample96_{member["key"]}'
                folder='outputs/local_score/t05_fit_cache'
            else:
                other = '20260919_1100' if heldout=='20260918_1100' else '20260918_1100'
                training=[other,'20260911_1100','20260912_1100']
                datasets=[]
                for name in training:
                    cache_key=(heldout,name,lam)
                    if cache_key not in fields:
                        target=eligible[name]
                        # Reproduce the clean P21 fit recipe, no heldout source
                        # may enter template completion, even indirectly.
                        same=sources_for(eligible,name)
                        tmpl=build_template(target,same,lam=lam)
                        hist=[r for n,r in eligible.items() if r.split=='train'
                              and r.n_minutes>=600 and n!=name]
                        tmpl,_=complete_template(target,tmpl,hist,lam=lam)
                        fields[cache_key]=gp.observed(target,tmpl,slope(tmpl),clip=1.5)
                    datasets.append(fields[cache_key])
                template_sources=sorted(n for n,r in eligible.items() if r.split=='train')
                _,old_hash=field_hash(datasets)
                key=f'fully_excluded_{heldout}_{old_hash[:16]}_{member["key"]}'
                folder='outputs/weekly_forecast_diagnostic/fit_cache'
            fingerprint,_=field_hash(datasets)
            result.append(dict(setting=setting,heldout=heldout,excluded_date=day,
                training_recordings=training,template_source_recordings=template_sources,
                field_rows=[len(d['t']) for d in datasets],field_sha256=fingerprint,
                member_key=member['key'],fit_settings=canonical(fit_kwargs(member)),
                filename=f'params_{key}.npz',legacy_path=f'{folder}/params_{key}.npz'))
    return result


def inventory(out):
    """All local NPZ and cached prediction CSVs; no prediction is reused."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records=[]
    paths=set((ROOT/'outputs').rglob('*.npz')) | set((ROOT/'models').rglob('*.npz'))
    paths |= set((ROOT/'outputs').rglob('*predictions*.csv'))
    for p in sorted(paths):
        records.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
            sha256=sha(p),kind='covariance' if p.suffix=='.npz' else 'saved_predictions',
            creation_code_hash='unknown',creation_input_hashes='unknown',
            training_provenance='unknown; see linked legacy evidence, never infer from filename'))
    (out/'inventory.json').write_text(json.dumps(dict(artifacts=records,
        current_context=context(),note='Current hashes are inventory evidence, not proof of creation provenance.'),indent=2))


def import_legacy():
    """One explicit migration, not an automatic blessing or refit."""
    if (ASSETS/'manifest.json').exists():raise FileExistsError('Do not overwrite an established manifest')
    entries=[]
    recs=load_recordings(ROOT/'data/raw')
    for setting in ('historical','representative'):
        for recipe in recipes(setting,recs):
            source=ROOT/recipe['legacy_path']
            if not source.exists():raise FileNotFoundError(source)
            # A successful parse is only a structural check.
            params=load_params(source)
            assert all(np.isfinite(a).all() for a in (params.ells,params.B,params.B0,params.sig2))
            target=ASSETS/setting/recipe['filename'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=sha(source):raise FileExistsError(target)
            else:shutil.copyfile(source,target)
            evidence=[]
            if setting=='representative':
                for log in ('run.log','run_final.log','verification.json','fit_exclusion_audit.csv'):
                    p=ROOT/'outputs/weekly_forecast_diagnostic'/log
                    if p.exists():
                        evidence.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
                raw=(ROOT/'outputs/weekly_forecast_diagnostic/run.log').read_bytes()
                log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8',errors='replace')
                prefix=f'Fitting/loading {recipe["heldout"]}: {recipe["member_key"]}; cells {recipe["field_rows"]}'
                assert prefix in log, f'Missing corroborating fit log: {prefix}'
            entries.append(dict(**recipe,sha256=sha(target),
                creation_code_version='unknown',creation_input_hashes='unknown',
                provenance='legacy_not_fully_verified',
                reconstructed_recipe_evidence=('field digest matches legacy key and fitting log'
                    if setting=='representative' else 'current recipe reconstructed; no original field digest stored'),
                evidence=evidence))
    manifest=dict(schema=1,context_at_import=context(),entries=entries,
        warning='Import fingerprints establish future compatibility only. Original fit execution/code/input provenance remains incomplete.',
        refit_policy='No automatic GP fitting. Missing or mismatching assets stop evaluation.')
    (ASSETS/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def validated_entries(setting,allow_legacy=False):
    manifest=json.loads((ASSETS/'manifest.json').read_text())
    if canonical(manifest['context_at_import'])!=canonical(context()):
        raise ValueError('Evaluation cache context changed (inputs, GP code, packages or settings). No automatic reuse/refit.')
    actual=recipes(setting)
    stored={(e['heldout'],e['member_key']):e for e in manifest['entries'] if e['setting']==setting}
    for r in actual:
        e=stored[(r['heldout'],r['member_key'])]
        for k,v in r.items():
            if canonical(e[k])!=canonical(v):raise ValueError(f'Cache recipe mismatch: {k}')
        p=ASSETS/setting/r['filename']
        if sha(p)!=e['sha256']:raise ValueError(f'Covariance bytes changed: {p.name}')
        if e['provenance']!='verified_at_creation' and not allow_legacy:
            raise ValueError('Legacy training provenance incomplete. Explicit --allow-legacy-provenance required; result remains conditional.')
    return stored
