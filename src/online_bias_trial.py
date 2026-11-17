"""A single prespecified causal bias rule; cached T05 parameters are load-only."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import weekly_forecast_diagnostic as weekly
import t05_local_forecast as older_t05
from cross_camera_forecast_audit import cached_only
from pipeline import ROOT
from short_term_change_trial import actual_cases
from t05_forecast import gp
from t05_forecast.core import ABSENT, RELEASED, load_recordings
from t05_forecast.pipeline import FRI, SAT, load_params, make_template
from t05_forecast.reference import last_value
from t05_forecast.submit import member_settings, settings
from t05_forecast.template import build_template, complete_template, slope

OUT = ROOT/'outputs/online_bias_trial'
BASE = ROOT/'outputs/submission_t05_hidden_hidden_future_im_fallback_v1.csv'
RULE = dict(name='half_median_90m', shrink=.5, lookback_minutes=90,
            min_distinct_targets=10, min_origins=2, cap_training_median_fraction=.10,
            cap_floor_people=1., timestamp_rule='past_origin < past_target < current_now',
            design='One rule, no search. Three 30-minute training-grid intervals; two origins and ten distinct targets required; 50% shrinkage; caps from training counts only.',
            acceptance='>=1% gain on both recent dates, positive leave-one-window-out gains, >=3 winning origins/day, no camera/horizon group with >=30 cases and >=5 origins regressing more than max(1 person,10% baseline).')


def caps(recs, excluded_day=None):
    result = {}
    for unit in weekly.UNITS:
        values = [r.y[:,r.col(unit)][r.status[:,r.col(unit)] == RELEASED]
                  for r in recs.values() if r.split == 'train' and r.day != excluded_day and unit in r.units]
        level = float(np.nanmedian(np.concatenate(values)))
        result[unit] = max(RULE['cap_floor_people'], RULE['cap_training_median_fraction']*level)
    return result


def reconstruct(recs, cases, members, older=False):
    """Exact forecasts at original cutoffs; no training, no post-cutoff data."""
    fields, rows, checks = {}, [], 0
    for name, today in cases.groupby('recording', sort=False):
        source = recs[name]
        full, frozen = [], []
        for member in members:
            if older:
                fitting = [r for r in recs.values() if r.split=='train' and r.day != source.day]
                prior = [r for r in fitting if r.weekday == source.weekday]
                others = [r for r in fitting if r.weekday != source.weekday]
                tmpl = build_template(source, prior, lam=member['lam'])
                tmpl,_ = complete_template(source, tmpl, others, lam=member['lam'])
                params = older_t05.heldout_params(recs, name, member)
                legacy = params
            else:
                tmpl = make_template(recs, name, member['lam'])
                with contextlib.redirect_stdout(io.StringIO()):
                    params = weekly.fitted_params(recs, name, member, fields)
                prefix = 'without_fri' if name==FRI else 'without_sat'
                legacy = load_params(ROOT/'outputs/t05_cache'/f'params_{prefix}_{member["key"]}.npz')
            # Neither target templates nor fitting templates may use this recording's counts.
            changed = source.copy(); changed.y[:]=1e6
            altered = dict(recs); altered[name]=changed
            if not older:
                np.testing.assert_allclose(tmpl, make_template(altered,name,member['lam']),
                                           rtol=0,atol=1e-12,equal_nan=True)
            dslope = slope(tmpl)
            clean_values, frozen_values = {}, {}
            for now, targets in today.groupby('now'):
                masked = weekly.masked_at_now(source, targets, int(now))
                perturb = source.copy(); perturb.y[perturb.minutes > now]=1e9
                alternate = weekly.masked_at_now(perturb,targets,int(now))
                np.testing.assert_allclose(masked.y,alternate.y,rtol=0,atol=0,equal_nan=True)
                assert np.array_equal(masked.status,alternate.status)
                prediction = gp.predict_recording(masked,tmpl,dslope,params,strict=True)
                legacy_prediction = prediction if older else gp.predict_recording(masked,tmpl,dslope,legacy,strict=True)
                for r in targets.itertuples():
                    row,col=int(r.minute)-source.first,source.col(r.unit)
                    clean_values[r.case_id]=prediction[row,col]
                    frozen_values[r.case_id]=legacy_prediction[row,col]
                checks += len(targets)
            full.append(np.array([clean_values[c] for c in today.case_id]))
            frozen.append(np.array([frozen_values[c] for c in today.case_id]))
        clean, legacy = weekly.finite_mean(full),weekly.finite_mean(frozen)
        fallback = last_value(source.y,30 if older else 5)
        for r,value,old in zip(today.itertuples(index=False),clean,legacy):
            ref=fallback[int(r.now)-source.first,source.col(r.unit)]
            rows.append(dict(**r._asdict(),t05=float(value if np.isfinite(value) else ref),
                             frozen_t05=float(old if np.isfinite(old) else ref)))
    return pd.DataFrame(rows),checks


def calibration_grid(recs,names):
    rows=[]
    for name in names:
        rec=recs[name]
        for now in range(rec.first+15,int(rec.minutes[-1]),30):
            for unit in weekly.UNITS:
                if unit not in rec.units: continue
                col=rec.col(unit)
                for h in range(1,16):
                    minute=now+h; row=minute-rec.first
                    if row>=rec.n_minutes or rec.status[row,col]!=RELEASED or not np.isfinite(rec.y[row,col]): continue
                    rows.append(dict(case_id=f'{name}:{unit}:{now}:{h}',day=rec.day,recording=name,
                                     unit=unit,now=now,minute=minute,horizon=h,window_id=f'{name}:{now}'))
    return pd.DataFrame(rows)


def correct(cases,pool,recs,limits,prediction_column='t05'):
    rows=[]; checks=0
    for (name,unit,now),targets in cases.groupby(['recording','unit','now'],sort=False):
        rec=recs[name]; col=rec.col(unit); now=int(now)
        past=pool[pool.recording.eq(name)&pool.unit.eq(unit)&pool.minute.lt(now)
                  &pool.minute.ge(now-RULE['lookback_minutes'])&pool['now'].lt(pool.minute)
                  &pool[prediction_column].notna()].copy()
        # Distinct released target times, not repeated predictions of one truth.
        assert not past.minute.duplicated().any()
        rr=past.minute.to_numpy(int)-rec.first
        assert (rec.status[rr,col]==RELEASED).all()
        # Strict prefix: even the reading AT current now is excluded from errors.
        end=max(0,min(rec.n_minutes,now-rec.first))
        prefix=rec.y[:end,col].copy()
        assert ((rr>=0)&(rr<len(prefix))).all()
        truth=prefix[rr]
        assert np.isfinite(truth).all()
        # Actual future perturbation check for the history-read path.
        changed=rec.y[:,col].copy(); changed[rec.minutes>=now]=1e9
        np.testing.assert_array_equal(truth,changed[:end][rr])
        sufficient=len(past)>=RULE['min_distinct_targets'] and past['now'].nunique()>=RULE['min_origins']
        bias=float(np.clip(RULE['shrink']*np.median(truth-past[prediction_column].to_numpy()),
                           -limits[unit],limits[unit])) if sufficient else 0.
        used='|'.join(past.case_id.astype(str)) if sufficient else ''
        for r in targets.itertuples(index=False):
            rows.append(dict(**r._asdict(),corrected=max(0.,r.t05+bias),bias=bias,
                             correction_eligible=bool(sufficient),past_errors=len(past),
                             past_origins=past['now'].nunique(),
                             latest_error_target=int(past.minute.max()) if len(past) else -1,
                             used_past_case_ids=used))
        checks+=len(targets)
    return pd.DataFrame(rows),checks


def summarize(scored,prefix):
    scored=scored.copy()
    scored['horizon_band']=pd.cut(scored.horizon,[0,5,10,15],labels=['1-5','6-10','11-15']).astype(str)
    scored['baseline_error']=(scored.t05-scored.truth).abs()
    scored['corrected_error']=(scored.corrected-scored.truth).abs()
    for groups,suffix in [(['day'],'date'),(['day','unit'],'camera'),(['day','horizon_band'],'horizon'),(['day','window_id'],'window')]:
        rows=[]
        for scope,frame in [('all',scored),('corrected',scored[scored.correction_eligible]),('fallback',scored[~scored.correction_eligible])]:
            if frame.empty: continue
            result=frame.groupby(groups).agg(targets=('case_id','size'),windows=('window_id','nunique'),
                timestamps=('minute','nunique'),eligible=('correction_eligible','sum'),
                baseline_mae=('baseline_error','mean'),corrected_mae=('corrected_error','mean')).reset_index()
            result['scope']=scope; result['gain']=result.baseline_mae-result.corrected_mae
            result['sparse']=result.targets.lt(30)|result.windows.lt(5)|result.timestamps.lt(30)
            rows.append(result)
        pd.concat(rows).to_csv(OUT/f'{prefix}_by_{suffix}.csv',index=False)
    rows=[]
    for day,g in scored.groupby('day'):
        for window in g.window_id.unique():
            kept=g[g.window_id.ne(window)]
            rows.append(dict(day=day,omitted_window=window,targets=len(kept),windows=kept.window_id.nunique(),
                             baseline_mae=kept.baseline_error.mean(),corrected_mae=kept.corrected_error.mean(),
                             gain=(kept.baseline_error-kept.corrected_error).mean()))
    pd.DataFrame(rows).to_csv(OUT/f'{prefix}_leave_one_window_out.csv',index=False)
    scored.to_csv(OUT/f'{prefix}_scored.csv',index=False)


def main():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/'outputs').glob('submission*.csv'))
    paths+=list((ROOT/'t05_forecast').glob('*.py'))+list((ROOT/'src').glob('t05*.py'))
    for directory,names in [(weekly.OUT,['cases.csv','labels.csv','predictions.csv','benchmark_predictions.csv']),
                            (ROOT/'outputs/local_score',['cases.csv','labels.csv','released_feed.csv'])]:
        paths += [directory/n for n in names]
        for name,expected in json.loads((directory/'case_hashes.json').read_text()).items():
            assert weekly.digest(directory/name)==expected
    protected={str(p):weekly.digest(p) for p in paths}
    recs=load_recordings(ROOT/'data/raw'); limits=caps(recs)
    # Freeze design and training-only caps before opening any evaluation labels.
    (OUT/'design.json').write_text(json.dumps(dict(rule=RULE,caps=limits,
        training_recordings=sorted(n for n,r in recs.items() if r.split=='train')),indent=2),encoding='utf-8')
    weekly.fit_cached=cached_only; older_t05.fit_cached=cached_only
    members=member_settings(settings(json.loads((ROOT/'t05_forecast_config.json').read_text())))
    cached=pd.read_csv(weekly.OUT/'predictions.csv')
    columns=['case_id','day','recording','unit','now','minute','horizon','window_id']
    fixed=pd.read_csv(weekly.OUT/'cases.csv')
    pd.testing.assert_frame_equal(cached[columns],fixed[columns],check_dtype=False)
    with threadpool_limits(limits=4):
        reconstructed,checks=reconstruct(recs,cached[columns],members)
        left=reconstructed.set_index('case_id'); right=cached.set_index('case_id')
        for method in ('t05','frozen_t05'):
            np.testing.assert_allclose(left.loc[right.index,method],right[method],rtol=0,atol=1e-9)
        reconstructed.to_csv(OUT/'reconstructed_past_predictions.csv',index=False)
        recent,ncheck=correct(reconstructed[columns+['t05']],reconstructed,recs,limits); checks+=ncheck
        recent.to_csv(OUT/'recent_predictions.csv',index=False)
        requested=actual_cases(recs)
        base=pd.read_csv(BASE).rename(columns={'id':'case_id','people':'t05'})
        actual,ncheck=correct(requested.merge(base,on='case_id',validate='one_to_one'),
                              reconstructed,recs,limits,prediction_column='frozen_t05'); checks+=ncheck
        actual.to_csv(OUT/'actual_prediction_diagnostic.csv',index=False)
        # Older fixed targets remain permanently masked. Build NEW past
        # predictions only for cells genuinely present in their released feed.
        feed=pd.read_csv(ROOT/'outputs/local_score/released_feed.csv')
        older_cached=pd.read_csv(weekly.OUT/'benchmark_predictions.csv')
        old_recs=dict(recs)
        for name in older_cached.recording.unique():
            source=recs[name].copy(); source.y[:]=np.nan; source.status[:]=ABSENT; source.truth=None
            for r in feed[feed.recording.eq(name)].itertuples():
                row,col=int(r.minute_of_day)-source.first,source.col(r.unit)
                source.y[row,col]=r.people; source.status[row,col]=RELEASED
            old_recs[name]=source
        grid=calibration_grid(old_recs,older_cached.recording.unique())
        # Fit/template sources must retain their released original data, while
        # only the evaluated recording receives the fixed masked feed.
        old_pools=[]
        for name,targets in grid.groupby('recording'):
            fold=dict(recs); fold[name]=old_recs[name]
            pool,ncheck=reconstruct(fold,targets,members,older=True); checks+=ncheck
            old_pools.append(pool)
        old_pool=pd.concat(old_pools,ignore_index=True)
        old_pool.to_csv(OUT/'older_reconstructed_past_predictions.csv',index=False)
        old_outputs=[]
        for day,targets in older_cached.groupby('day'):
            output,ncheck=correct(targets[columns+['t05']],old_pool,old_recs,caps(recs,excluded_day=day))
            checks+=ncheck; old_outputs.append(output)
        older=pd.concat(old_outputs,ignore_index=True)
        older.to_csv(OUT/'older_predictions.csv',index=False)
    # Evaluation labels are never used by correction; only now join for scoring.
    summarize(recent.merge(pd.read_csv(weekly.OUT/'labels.csv'),on='case_id',validate='one_to_one'),'recent')
    summarize(older.merge(pd.read_csv(ROOT/'outputs/local_score/labels.csv'),on='case_id',validate='one_to_one'),'older')
    days=pd.read_csv(OUT/'recent_by_date.csv').query("scope=='all'")
    windows=pd.read_csv(OUT/'recent_by_window.csv').query("scope=='all'")
    loo=pd.read_csv(OUT/'recent_leave_one_window_out.csv')
    major=pd.concat([pd.read_csv(OUT/'recent_by_camera.csv'),pd.read_csv(OUT/'recent_by_horizon.csv')])
    major=major[(major.scope=='all')&major.targets.ge(30)&major.windows.ge(5)]
    bad=major[major.corrected_mae-major.baseline_mae>np.maximum(1.,major.baseline_mae*.1)]
    winning=windows.assign(win=windows.gain.gt(0)).groupby('day').win.sum()
    accepted=bool((days.gain>=days.baseline_mae*.01).all() and loo.gain.gt(0).all()
                  and winning.ge(3).all() and bad.empty)
    candidate=None
    if accepted:
        # This branch only changes requested A-H forecast IDs in a NEW file.
        candidate=ROOT/'outputs/submission_t05_online_bias_v1.csv'
        if candidate.exists(): raise FileExistsError(f'Refusing overwrite: {candidate}')
        with BASE.open(newline='',encoding='utf-8-sig') as handle: original=list(csv.DictReader(handle))
        changes=actual[actual.correction_eligible].set_index('case_id').corrected.to_dict()
        sample_ids=pd.read_csv(ROOT/'data/raw/sample_submission.csv').id.tolist()
        assert len(original)==4457 and [r['id'] for r in original]==sample_ids
        rows=[dict(r) for r in original]
        for row in rows:
            if row['id'] in changes: row['people']=repr(float(changes[row['id']]))
        with candidate.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['id','people']);writer.writeheader();writer.writerows(rows)
        reread=pd.read_csv(candidate)
        assert reread.id.tolist()==sample_ids and reread.id.is_unique
        assert np.isfinite(reread.people).all() and reread.people.ge(0).all()
        assert all(a['people']==b['people'] for a,b in zip(original,rows) if a['id'] not in changes)
    assert all(weekly.digest(Path(p))==value for p,value in protected.items())
    report=dict(rule=RULE['name'],accepted=accepted,t05_fits=0,checks=checks,
                recent_cases=len(recent),recent_corrected=int(recent.correction_eligible.sum()),
                older_calibration_cases=len(old_pool),actual_requests=len(actual),
                actual_eligible=int(actual.correction_eligible.sum()),
                actual_would_change=int((actual.corrected-actual.t05).abs().gt(1e-9).sum()),
                winning_windows=winning.to_dict(),major_regressions=bad.fillna('').to_dict('records'),
                candidate=str(candidate) if candidate else None,protected_files_unchanged=True,
                elapsed_seconds=time.perf_counter()-started,protected_sha256=protected)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'recent_by_date.csv').to_string(index=False))
    print('Older stress:');print(pd.read_csv(OUT/'older_by_date.csv').to_string(index=False))
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2))


if __name__=='__main__':main()
