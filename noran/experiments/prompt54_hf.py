"""HIDDEN-FUTURE-only diagnosis/evaluation. Never runs T05 or other missions.

Development labels: fixed. Mirror and testday are opened only after recipe lock.
All dates have been examined before; these are reused, correlated local tests.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / 'src', ROOT / 'noran'):
    sys.path.insert(0, str(folder))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import baseline_tasks as tasks
from pipeline import load, locate_data, table, CAMERAS
from prayer_experiments import timetable

OUT = ROOT / 'outputs/prompt54_hf'

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

V7 = module('noran/tracks/hf_final/model.py', 'prompt54_v7')
BASE, C = V7.base, V7.base.C

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protect():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / 'protected_hashes.json'
    if not snapshot.exists():
        paths = []
        for folder in ('data/raw', 'models', 'notebooks', 'configs', 'src', 'noran/tracks'):
            paths += [p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        for folder in ('outputs/version7', 'outputs/noran'):
            paths += [p for p in (ROOT / folder).rglob('*') if p.is_file()
                      and 'huggingface' not in p.parts and p.suffix in ('.csv', '.json', '.ipynb')]
        snapshot.write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in paths}, indent=2))
    saved = json.loads(snapshot.read_text())
    wrong = [p for p, digest in saved.items() if not (ROOT / p).exists() or sha(ROOT / p) != digest]
    assert not wrong, f'Protected files changed: {wrong}'
    return len(saved)

def setup():
    train, test, sample, _ = load(locate_data(None))
    schedule = timetable(ROOT / 'models/current_baseline/prayer_times_madinah.csv', train, test)
    return train, test, sample, schedule

def context(schedule):
    # V7 does not use released test rows in fitting. Supply none in evaluation,
    # so assessment hidden answers cannot leak through the optional context.
    return dict(schedule=schedule, config=tasks.default_config()['hidden-future'], mission='hidden-future')

def protocol(which, train, test):
    """Same case/feed construction as track_eval, restricted to HF; labels kept separate."""
    case_rows, labels, feeds = [], [], {}
    if which in ('fixed', 'mirror', 'auxiliary'):
        mapping = ({'20260918_1100': '20260911_1100', '20260919_1100': '20260912_1100'} if which == 'fixed'
                   else {'20260918_1100': '20260912_1100', '20260919_1100': '20260911_1100'})
        pairs=list(mapping.items())
        if which == 'auxiliary':
            # Original clock-aligned masks on old short recordings. No shifted morning protocol.
            pairs=[('20260918_1100' if g.weekday.iloc[0]=='Fri' else '20260919_1100',rec)
                   for rec,g in train[train.day.isin(['2026-09-04','2026-09-05'])].groupby('recording')]
        for source, rec in pairs:
            truth = train[train.recording.eq(rec)]
            mask = test[test.recording.eq(source) & test.unit.isin(CAMERAS)].copy()
            mask['recording'] = rec
            feed = mask[mask.id.isna()][['recording', 'unit', 'minute_of_day']].merge(
                truth, on=['recording', 'unit', 'minute_of_day'], how='left')
            if not feed.people.notna().any():
                continue
            feeds[rec] = table(feed[feed.people.notna()])[rec]
            target = mask[mask.id.notna() & mask.kind.eq('hidden-future')].drop(columns='people').merge(
                truth[['recording', 'unit', 'minute_of_day', 'people']],
                on=['recording', 'unit', 'minute_of_day'], how='inner')
            for r in target.itertuples():
                minute, now = int(r.minute_of_day), int(r.now_minute_of_day)
                key = f'{rec}_{r.unit}_{minute}'
                case_rows.append(dict(case_id=key, day=truth.day.iloc[0], recording=rec, kind='hidden-future',
                                      unit=r.unit, minute=minute, now=now, horizon=minute-now))
                labels.append(dict(case_id=key, truth=float(r.people)))
    elif which == 'testday':
        # Exact track_eval TESTDAY_HIDDEN / origins, including natural unavailable cameras.
        for rec, hidden in {'20260918_1100': ['cam_D'], '20260919_1100': ['cam_B', 'cam_G']}.items():
            released = test[test.recording.eq(rec) & test.id.isna() & test.people.notna()]
            feeds[rec] = table(released[~released.unit.isin(hidden)])[rec]
            minutes = sorted(released[released.unit.isin(CAMERAS)].minute_of_day.unique())
            stretches, first, prev = [], minutes[0], minutes[0]
            for m in minutes[1:]:
                if m != prev + 1:
                    stretches.append((first, prev))
                    first = m
                prev = m
            stretches.append((first, prev))
            for unit in hidden:
                own = released[released.unit.eq(unit)].set_index('minute_of_day').people
                for first, last in stretches:
                    for offset in (7, 13):
                        now = first + offset
                        for minute in range(now+1, min(now+15, last)+1):
                            if minute not in own.index:
                                continue
                            key = f'{rec}_{unit}_{minute}_n{now}'
                            case_rows.append(dict(case_id=key, day=released.day.iloc[0], recording=rec,
                                                  kind='hidden-future', unit=unit, minute=minute, now=now, horizon=minute-now))
                            labels.append(dict(case_id=key, truth=float(own[minute])))
    elif which == 'actual':
        feeds = table(test[test.id.isna() & test.people.notna()])
        for r in test[test.id.notna() & test.kind.eq('hidden-future')].itertuples():
            now, minute = int(r.now_minute_of_day), int(r.minute_of_day)
            case_rows.append(dict(case_id=r.id, day=r.day, recording=r.recording, kind='hidden-future',
                                  unit=r.unit, minute=minute, now=now, horizon=minute-now))
    else:
        raise ValueError(which)
    cases = pd.DataFrame(case_rows)
    assert cases.case_id.is_unique and cases.horizon.between(1, 15).all()
    for r in cases.itertuples():
        assert feeds[r.recording].reindex(columns=C.HIDDEN_WITH[r.unit]).isna().all().all()
    return cases, feeds, pd.DataFrame(labels)

def components(t, unit, now, horizons, day, model, schedule):
    p, ref = model['params'], model['ref_pr']
    causal = t.loc[:now].drop(columns=list(C.HIDDEN_WITH[unit]), errors='ignore')
    cams = [c for c in C.CAMS if c in causal and c not in C.HIDDEN_WITH[unit]]
    wins = {c: C.window(causal, c, now, p['K']) for c in cams}
    shift_wins = {c: C.window(causal, c, now, p['SHIFT_WINDOW']) for c in p['SHIFT_CAMS'] if c in cams}
    hs = np.array([0, *horizons], dtype=float)
    tau = C.align(now + hs, day, schedule, ref, p['STRENGTH'])
    num, den = np.zeros(len(hs)), np.zeros(len(hs))
    entries = []
    weekday = pd.Timestamp(day).day_of_week
    for rec, an in model['analogs'].items():
        if unit not in an.smooth:
            continue
        delta = C.best_shift(shift_wins, an, list(shift_wins), day, schedule, ref, p['STRENGTH'], p['SHIFT_MAX'], p['SHIFT_PENALTY'])
        target = an.at(an.smooth[unit], tau + delta)
        logs, donors = [], []
        for cam in cams:
            ratio, pairs = C.analog_ratio(*wins[cam], an, cam, day, schedule, ref,
                                         min_pairs=p['MIN_PAIRS'], strength=p['STRENGTH'], delta=delta)
            if np.isfinite(ratio) and ratio > 0:
                logs.append(np.log(ratio)); donors.append(cam)
        if len(logs) < p['MIN_CAMS']:
            continue
        same = an.weekday == weekday
        weight = p['SAME_WEIGHT'] if same else 1.0
        level = np.exp(p['BETA'] * np.mean(logs))
        ok = np.isfinite(target)
        num[ok] += weight * level * target[ok]; den[ok] += weight
        entries.append(dict(recording=rec, same=same, shift=delta, donors=len(logs), level=level,
                            log_std=float(np.std(logs)), log_min=float(min(logs)), log_max=float(max(logs)),
                            target0=float(target[0]), covers=int(ok[1:].sum()),
                            ratio_by_camera=dict(zip(donors, np.exp(logs).tolist()))))
    analog = np.divide(num, den, out=np.full(len(hs), np.nan), where=den>0)
    usable = [x for x in entries if x['covers']]
    same = [x for x in usable if x['same']]
    diagnostics = dict(estimated_now=float(analog[0]), analog_count=len(usable), same_weekday_count=len(same),
                       donor_count=sum(len(v[0]) >= p['MIN_PAIRS'] for v in wins.values()),
                       observed_at_now=sum(now in causal[c].dropna().index for c in cams),
                       history_points=sum(len(v[0]) for v in wins.values()),
                       donor_disagreement=float(np.mean([x['log_std'] for x in same or usable])) if usable else np.nan,
                       mean_shift=float(np.mean([x['shift'] for x in same or usable])) if usable else np.nan,
                       level=float(np.mean([x['level'] for x in same or usable])) if usable else np.nan,
                       details=json.dumps(entries, separators=(',', ':')))
    return analog[1:], diagnostics

def collect(which):
    protected = protect()
    path = OUT / f'{which}_predictions.csv'
    if path.exists():
        print(f'Reusing {path}', flush=True)
        return
    started = time.perf_counter()
    train, test, sample, schedule = setup()
    cases, feeds, labels = protocol(which, train, test)
    ctx = context(schedule)
    if which == 'actual':
        cached_old = pd.read_csv(ROOT / 'outputs/noran/baseline_local/submission_baseline_local.csv').set_index('id').people
        official_reference = sample.set_index('id').people
    rows, runtimes, consistency = [], [], []
    checks = dict(post_now=0, hidden_mask=0, failures=0)
    with threadpool_limits(limits=2):
        for day, today in cases.groupby('day'):
            training = train if which == 'actual' else train[train.day.ne(day)].copy()
            assert which == 'actual' or not training.day.eq(day).any()
            tick = time.perf_counter(); model = V7.fit(training, None if which == 'actual' else day, ctx)
            fit_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            for (rec, unit, now), group in today.groupby(['recording', 'unit', 'now']):
                now = int(now); t = feeds[rec]
                hs = group.horizon.astype(int).tolist()
                analog, diag = components(t, unit, now, hs, day, model, schedule)
                late = int(schedule[day][2])+55 <= now <= int(schedule[day][3])-35
                # Public V7 path is authoritative; diagnostics never replace its prediction.
                for i, r in enumerate(group.itertuples()):
                    ref, old, _ = tasks.hidden_future_value(t, unit, now, int(r.horizon), day, schedule, model['accepted'], ctx['config'])
                    raw = V7.predict(r, t, model, ctx)
                    if which == 'actual':
                        ref = float(official_reference[r.case_id])
                        if not np.isfinite(raw):
                            old = float(cached_old[r.case_id])
                    pred = float(raw) if np.isfinite(raw) else old
                    reconstructed = .8*analog[i]+.2*old if not late and np.isfinite(analog[i]) else old
                    consistency.append(abs(pred-reconstructed))
                    rows.append(dict(case_set=which, **r._asdict(), v7=pred, old=old, reference=ref,
                                     analog=float(analog[i]), fallback=not np.isfinite(raw), late=late, **diag))
                # One call per camera/origin, three separate variants; no prediction caches.
                r = next(group.itertuples()); original = float(V7.predict(r, t, model, ctx))
                for kind in ('post_now', 'hidden_mask'):
                    altered = t.copy()
                    if kind == 'post_now': altered.loc[altered.index > now, :] = 1e8
                    else:
                        for cam in C.HIDDEN_WITH[unit]: altered[cam] = 1e8
                    again = float(V7.predict(r, altered, model, ctx))
                    checks[kind] += 1
                    checks['failures'] += int(not np.isclose(original, again, atol=1e-9, rtol=0, equal_nan=True))
            runtimes.append(dict(day=day, training_recordings=sorted(training.recording.unique()),
                                 training_days=sorted(training.day.unique()), assessment_day_excluded=which != 'actual',
                                 fit_seconds=fit_seconds, inference_with_diagnostics_checks_seconds=time.perf_counter()-tick))
            print(f'{which}: {day} complete, {len(rows)} cases, {time.perf_counter()-started:.1f}s', flush=True)
    detail = pd.DataFrame(rows).drop(columns='Index', errors='ignore')
    assert np.isfinite(detail[['v7','old','reference']]).all().all()
    assert max(consistency) < 1e-9
    assert checks['failures'] == 0
    detail.to_csv(path, index=False)
    if which != 'actual':
        # Predictions are saved before joining the separate answers for scoring.
        detail = detail.merge(labels, on='case_id', validate='one_to_one')
        detail.to_csv(OUT / f'{which}_scored.csv', index=False)
    else:
        accepted = pd.read_csv(ROOT / 'outputs/version7/reference/submission_v7_board_0.724.csv').set_index('id').people
        assert np.allclose(detail.v7, accepted.reindex(detail.case_id), atol=1e-9, rtol=0)
    (OUT / f'{which}_collection.json').write_text(json.dumps(dict(
        protected_files=protected, checks=checks, vector_public_max_diff=max(consistency),
        folds=runtimes, runtime_seconds=time.perf_counter()-started), indent=2))
    summarize(detail, which)
    protect()

def summarize(d, which):
    d = d.copy()
    d['weekday'] = pd.to_datetime(d.day).dt.day_name()
    d['band'] = pd.cut(d.horizon, [0,5,10,15], labels=['1-5','6-10','11-15'])
    d['window'] = d.recording + '|' + d.unit + '|' + d.now.astype(str)
    d['shape_change'] = d.analog - d.estimated_now
    d['fallback_reason'] = np.where(d.late, 'late_plateau', np.where(d.analog.isna(), 'no_analog', 'analog'))
    if 'truth' not in d:
        d.to_csv(OUT / f'{which}_inputs.csv', index=False)
        return
    for col in ('v7','old','reference'):
        d['err_'+col] = (d[col]-d.truth).abs()
    d['bias'] = d.v7-d.truth
    for name, keys in dict(recording=['recording','weekday','unit'], horizon=['recording','unit','band'],
                           window=['recording','unit','now'], behavior=['unit','fallback_reason']).items():
        g=d.groupby(keys, observed=True).agg(n=('truth','size'), windows=('window','nunique'),
            v7_mae=('err_v7','mean'), old_mae=('err_old','mean'), ref_mae=('err_reference','mean'),
            bias=('bias','mean'), fallback=('fallback','sum'), donor_count=('donor_count','mean'),
            donor_disagreement=('donor_disagreement','mean'), mean_shape_change=('shape_change','mean'))
        g['v7_ratio']=g.v7_mae/g.ref_mae
        g.to_csv(OUT / f'{which}_diagnosis_by_{name}.csv')
        if name=='recording': print(g.to_string(), flush=True)
    d.to_csv(OUT / f'{which}_diagnostics.csv', index=False)

def metric(d, column):
    q=d.assign(error=(d[column]-d.truth).abs(), ref_error=(d.reference-d.truth).abs())
    daily=q.groupby('day')[['error','ref_error']].mean()
    return float((daily.error/daily.ref_error).mean())

def comparison(d, which):
    d=d.copy()
    d['band']=pd.cut(d.horizon,[0,5,10,15],labels=['1-5','6-10','11-15'])
    d['weekday']=pd.to_datetime(d.day).dt.day_name()
    d['window']=d.recording+'|'+d.unit+'|'+d.now.astype(str)
    # Event windows cluster B/G together, including overlapping testday horizons.
    d['event']=d.recording+'|'+d.now.astype(str)
    for col in ('v7','candidate','reference'):
        d['error_'+col]=(d[col]-d.truth).abs()
    d['changed']=(d.candidate-d.v7).abs()>1e-9
    d['gain']=d.error_v7-d.error_candidate
    for name,keys in dict(recording=['recording','weekday','unit'], horizon=['recording','unit','band'],
                          window=['recording','unit','now'], scope=['unit','supported']).items():
        g=d.groupby(keys,observed=True).agg(n=('truth','size'),windows=('window','nunique'),
            event_windows=('event','nunique'),v7_mae=('error_v7','mean'),candidate_mae=('error_candidate','mean'),
            reference_mae=('error_reference','mean'),supported=('supported','sum'),
            v7_fallback=('fallback','sum'),candidate_fallback=('candidate_fallback','sum'),changed=('changed','sum'),
            gain=('gain','mean'))
        g['v7_ratio']=g.v7_mae/g.reference_mae
        g['candidate_ratio']=g.candidate_mae/g.reference_mae
        g.to_csv(OUT / f'{which}_comparison_by_{name}.csv')
        if name=='recording': print(g.to_string(),flush=True)
    w=d.groupby('window').gain.sum()
    events=d.groupby('event').gain.sum().sort_values(ascending=False)
    sensitivity=[]
    for number in (0,1,2):
        removed=events.head(number).index.tolist()
        remaining=d[~d.event.isin(removed)]
        sensitivity.append(dict(best_event_windows_removed=number,removed=removed,
            rows=len(remaining),mae_gain=float(remaining.gain.mean()),
            ratio_v7=metric(remaining,'v7'),ratio_candidate=metric(remaining,'candidate')))
    result=dict(case_set=which,rows=len(d),camera_windows=int(len(w)),event_windows=int(len(events)),
        v7_mae=float(d.error_v7.mean()),candidate_mae=float(d.error_candidate.mean()),
        reference_mae=float(d.error_reference.mean()),v7_ratio=metric(d,'v7'),candidate_ratio=metric(d,'candidate'),
        supported=int(d.supported.sum()),changed=int(d.changed.sum()),
        v7_fallback=int(d.fallback.sum()),candidate_fallback=int(d.candidate_fallback.sum()),
        improved_camera_windows=int((w>1e-9).sum()),worsened_camera_windows=int((w< -1e-9).sum()),
        tied_camera_windows=int((w.abs()<=1e-9).sum()),
        improved_event_windows=int((events>1e-9).sum()),worsened_event_windows=int((events< -1e-9).sum()),
        sensitivity=sensitivity)
    d.to_csv(OUT/f'{which}_experiment_scored.csv',index=False)
    (OUT/f'{which}_experiment_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def lock_recipe():
    paths=['noran/configs/prompt54_hf_b_d_timing.json','noran/tracks/hf_b_d_timing/model.py']
    fingerprint={p:sha(ROOT/p) for p in paths}
    path=OUT/'recipe_lock.json'
    if path.exists():
        assert json.loads(path.read_text())['sha256']==fingerprint, 'Locked recipe changed'
    else:
        path.write_text(json.dumps(dict(sha256=fingerprint,locked_unix_time=time.time(),
            development='fixed',assessment=['mirror','testday'],assessment_labels_read=False),indent=2))
    return fingerprint

def evaluate(which):
    protect(); lock_recipe()
    prediction_path=OUT/f'{which}_experiment_predictions.csv'
    if prediction_path.exists():
        print(f'Reusing {prediction_path}',flush=True)
        comparison(pd.read_csv(OUT/f'{which}_experiment_scored.csv'),which)
        return
    # collect() saves V7 predictions first; evaluation does not inspect their labels yet.
    collect(which)
    baseline=pd.read_csv(OUT/f'{which}_predictions.csv').set_index('case_id')
    train,test,sample,schedule=setup()
    cases,feeds,labels=protocol(which,train,test)
    ctx=context(schedule)
    trial=module('noran/tracks/hf_b_d_timing/model.py','prompt54_trial')
    checks=dict(post_now=0,hidden_mask=0,cut_prefix=0,failures=0,unsupported_changes=0)
    rows,folds=[],[]
    with threadpool_limits(limits=2):
        for day,today in cases.groupby('day'):
            training=train[train.day.ne(day)].copy()
            assert not training.day.eq(day).any()
            # Tainting all held-out labels must leave the supplied fitting frame unchanged.
            poisoned=train.copy(); poisoned.loc[poisoned.day.eq(day),'people']=1e8
            pd.testing.assert_frame_equal(training,poisoned[poisoned.day.ne(day)])
            tick=time.perf_counter(); fitted=trial.fit(training,day,ctx); fit_seconds=time.perf_counter()-tick
            inference=check_time=0.
            for r in today.itertuples():
                t=feeds[r.recording]; base=baseline.loc[r.case_id]
                tick=time.perf_counter()
                raw=trial.predict(r,t,fitted,ctx)
                support=trial.supported(r,t,fitted,ctx)
                value=float(raw) if np.isfinite(raw) else float(base.v7)
                inference+=time.perf_counter()-tick
                checks['unsupported_changes']+=int(not support and not np.isclose(value,base.v7,rtol=0,atol=1e-9))
                # All cases and all masks, separately, without a prediction-result cache.
                tick=time.perf_counter()
                for kind in ('post_now','hidden_mask','cut_prefix'):
                    altered=t.copy()
                    if kind=='post_now': altered.loc[altered.index>int(r.now),:]=1e8
                    elif kind=='hidden_mask':
                        for cam in C.HIDDEN_WITH[r.unit]: altered[cam]=1e8
                    else: altered=altered.loc[:int(r.now)]
                    again=trial.predict(r,altered,fitted,ctx)
                    checks[kind]+=1
                    checks['failures']+=int(not np.isclose(raw,again,rtol=0,atol=1e-9,equal_nan=True))
                check_time+=time.perf_counter()-tick
                rows.append(dict(case_id=r.case_id,candidate=max(0.,value),supported=support,
                                 candidate_fallback=not np.isfinite(raw)))
            folds.append(dict(day=day,training_days=sorted(training.day.unique()),
                training_recordings=sorted(training.recording.unique()),heldout_taint_exclusion=True,
                fit_seconds=fit_seconds,inference_seconds=inference,invariance_seconds=check_time))
            print(f'{which}: candidate {day} predicted',flush=True)
    assert checks['failures']==checks['unsupported_changes']==0, checks
    detail=pd.DataFrame(rows).merge(baseline.reset_index(),on='case_id',validate='one_to_one')
    detail.to_csv(prediction_path,index=False)
    # Answers enter the comparison only after all predictions are saved.
    detail=detail.merge(labels,on='case_id',validate='one_to_one')
    comparison(detail,which)
    (OUT/f'{which}_experiment_runtime.json').write_text(json.dumps(dict(folds=folds,checks=checks),indent=2))
    protect()

def finalize():
    protected=protect(); lock_recipe()
    train,test,_,_=setup()
    input_rows,window_counts,sensitivity,summary_rows=[],[],[],[]
    timings={}; checks={}
    for which in ('fixed','mirror','testday','auxiliary'):
        d=pd.read_csv(OUT/f'{which}_experiment_scored.csv')
        result=json.loads((OUT/f'{which}_experiment_result.json').read_text())
        runtime=json.loads((OUT/f'{which}_experiment_runtime.json').read_text())
        collection=json.loads((OUT/f'{which}_collection.json').read_text())
        timings[which]=dict(v7_collection_seconds=collection['runtime_seconds'],
            candidate_fit_seconds=sum(f['fit_seconds'] for f in runtime['folds']),
            candidate_inference_seconds=sum(f['inference_seconds'] for f in runtime['folds']),
            candidate_invariance_seconds=sum(f['invariance_seconds'] for f in runtime['folds']))
        checks[which]=runtime['checks']
        summary_rows.append(result)
        w=d.groupby(['recording','unit','now']).agg(gain=('gain','sum'),n=('case_id','size')).reset_index()
        for (rec,unit),g in w.groupby(['recording','unit']):
            window_counts.append(dict(case_set=which,recording=rec,unit=unit,windows=len(g),
                improved=int(g.gain.gt(1e-9).sum()),worsened=int(g.gain.lt(-1e-9).sum()),
                tied=int(g.gain.abs().le(1e-9).sum())))
            q=d[d.recording.eq(rec)&d.unit.eq(unit)]
            ordered=g.sort_values('gain',ascending=False)
            for n in (0,1,2):
                rem=q[~q.now.isin(ordered.head(n).now)]
                if len(rem):
                    sensitivity.append(dict(case_set=which,recording=rec,unit=unit,best_windows_removed=n,
                        rows=len(rem),mae_gain=float(rem.gain.mean()),
                        v7_ratio=metric(rem,'v7'),candidate_ratio=metric(rem,'candidate')))
        # Truth-at-now is an explanatory diagnostic only, never an input or a gate.
        raw=train if which!='testday' else test[test.id.isna()&test.people.notna()]
        tabs=table(raw)
        d['oracle_past5']= [float(tabs[r.recording][r.unit].reindex(range(int(r.now)-4,int(r.now)+1)).mean())
                            for r in d.itertuples()]
        d['oracle_level_error']=d.estimated_now-d.oracle_past5
        d['oracle_change_error']=(d.analog-d.estimated_now)-(d.truth-d.oracle_past5)
        d.to_csv(OUT/f'{which}_error_decomposition.csv',index=False)
    for which in ('fixed','mirror','testday','auxiliary','actual'):
        d=pd.read_csv(OUT/f'{which}_predictions.csv')
        d['window']=d.recording+'|'+d.unit+'|'+d.now.astype(str)
        for (rec,unit),g in d.groupby(['recording','unit']):
            windows=g.drop_duplicates('window')
            input_rows.append(dict(case_set=which,recording=rec,unit=unit,targets=len(g),
                camera_windows=len(windows),windows_with_some_same_weekday_coverage=int(windows.same_weekday_count.gt(0).sum()),
                donor_count_min=int(g.donor_count.min()),donor_count_max=int(g.donor_count.max()),
                donors_mean=float(g.donor_count.mean()),v7_fallback=int(g.fallback.sum()),
                late_plateau=int(g.late.sum()),other_fallback=int((g.fallback&~g.late).sum()),
                level_mean=float(g.level.mean()),disagreement_mean=float(g.donor_disagreement.mean())))
    pd.DataFrame(input_rows).to_csv(OUT/'input_similarity.csv',index=False)
    pd.DataFrame(window_counts).to_csv(OUT/'window_gain_counts.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity_by_recording_camera.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='sensitivity'} for r in summary_rows]).to_csv(OUT/'summary.csv',index=False)
    check_total=sum(v[k] for v in checks.values() for k in ('post_now','hidden_mask','cut_prefix'))
    report=dict(decision='reject; keep Version 7; no candidate submission created',
        trials=1,additional_revisions=0,protected_files_unchanged=protected,
        experiment_checks=checks,total_prediction_invariance_checks=check_total,
        timing_seconds=timings,recipe_lock=json.loads((OUT/'recipe_lock.json').read_text()),
        limitation='All dates previously examined. Fixed development; mirror/testday assessment after recipe lock. '
        'Auxiliary short historical recordings use original clock-aligned masks. No independent new dates. '
        'Testday has naturally absent D for B/G, so the trial has zero coverage there.')
    (OUT/'final_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    protect()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=['collect','evaluate','finalize','check'])
    parser.add_argument('--set', default='fixed', choices=['fixed','mirror','testday','auxiliary','actual'])
    args=parser.parse_args()
    if args.action=='collect': collect(args.set)
    elif args.action=='evaluate': evaluate(args.set)
    elif args.action=='finalize': finalize()
    else: print('Protected hashes OK:', protect())

if __name__=='__main__': main()
