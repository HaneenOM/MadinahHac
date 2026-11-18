"""Prompt 31: bounded training-only camera/lag discovery; no forecast fitting.

Only train.csv and training metadata are loaded. Evaluation labels, test.csv,
and cached predictions are deliberately unnecessary for this diagnostic.
"""
from __future__ import annotations

import hashlib
import json
import time
from html import escape
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from t05_forecast.core import RELEASED, Recording
from t05_forecast.pipeline import make_template

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/camera_relationship_diagnostic'
UNITS = [f'cam_{c}' for c in 'ABCDEFGH']
SPEC = dict(
    discovery='train.csv only; all six recordings on four dates; no test/evaluation counts loaded',
    alignment='exact recording/minute; missing cells remain NaN; no observation interpolation',
    levels='log1p counts standardized separately within recording/camera',
    changes='1 and 5 minute log changes; every minute in each change interval must be observed',
    lags='source(t) versus target(t+lag), all integer lags 0..15; no boundary crossing',
    residuals='subtract T05 lam=.8 weekly/prayer template, excluding entire target date from sources and completion; template interpolation is existing nuisance model only, never fills observations',
    windows='nonoverlapping 30 minute blocks; source change interval and target endpoint must lie in same block; blocks are sensitivity units, not independent recording replicates',
    selection='at most 3 fixed pair/lag relationships; residual 5-minute change partial correlation controlling target change at source time',
    gates=dict(min_paired=200, min_record_paired=30, min_records=3, min_dates=3,
               min_abs_partial=.25, min_record_signed_median=.20,
               min_record_sign_agreement=.75, min_weekday_signed_partial=.15,
               min_windows=8, min_window_paired=10, min_window_sign_agreement=.65,
               min_lag_advantage_over_sync=.05, min_leave_date_out_signed_partial=.15),
    interpretation='exploratory many-pair/lag screen, no significance or movement/location inference; no new method unless stable evidence exceeds existing T05 representation',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training():
    table = pd.read_csv(ROOT / 'data/raw/train.csv')
    meta = pd.read_csv(ROOT / 'data/raw/recordings.csv').set_index('recording')
    assert not table.duplicated(['recording', 'unit', 'minute_of_day']).any()
    recs = {}
    for name, g in table.groupby('recording'):
        m = meta.loc[name]
        assert m['split'] == 'train' and m.day < '2026-09-18'
        first = int(g.minute_of_day.min())
        last = int(g.minute_of_day.max())
        y = np.full((last-first+1, 8), np.nan)
        for unit, camera in g.groupby('unit'):
            y[camera.minute_of_day.to_numpy(int)-first, UNITS.index(unit)] = camera.people
        status = np.where(np.isfinite(y), RELEASED, 0).astype('int8')
        recs[name] = Recording(name, m.day, m.weekday, first, UNITS.copy(), y,
                               status, np.full(y.shape, -1), split='train')
    return recs


def delta(values, step):
    result = np.full_like(values, np.nan)
    good = np.ones((len(values)-step, 8), dtype=bool)
    for offset in range(step+1):
        good &= np.isfinite(values[offset:len(values)-step+offset])
    result[step:] = np.where(good, values[step:]-values[:-step], np.nan)
    return result


def corr(x, y):
    if len(x) < 4:
        return np.nan
    a, b = x-x.mean(), y-y.mean()
    den = np.sqrt(a@a * (b@b))
    return float(a@b/den) if den > 1e-12 else np.nan


def partial(x, y, z):
    if len(x) < 5:
        return np.nan
    design = np.column_stack([np.ones(len(z)), z])
    return corr(x-design@np.linalg.lstsq(design, x, rcond=None)[0],
                y-design@np.linalg.lstsq(design, y, rcond=None)[0])


def signed_min(values, sign):
    a = np.asarray(values, float)*sign
    return float(np.min(a)) if len(a) and np.isfinite(a).all() else np.nan


def svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#203044;font-size:13px}.title{font-size:20px;font-weight:bold}</style>',
            f'<text x="30" y="30" class="title">{escape(title)}</text>']


def save_svg(name, parts):
    path = OUT / name
    path.write_text('\n'.join(parts+['</svg>']), encoding='utf-8')
    ElementTree.parse(path)


COLORS = ['#1368a4', '#d34d43', '#258044', '#8b52a3', '#dc8d16', '#087f8c', '#b64e85', '#60676d']


def trace_panel(parts, values, minutes, top, title, indices=range(8), ylim=(-3, 3)):
    left, width, height = 80, 990, 155
    low, high = ylim
    parts.append(f'<text x="30" y="{top-12}">{escape(title)}</text>')
    def xx(t): return left+(t-minutes[0])/max(1, minutes[-1]-minutes[0])*width
    def yy(v): return top+height-(v-low)/(high-low)*height
    for v in np.linspace(low, high, 5):
        parts.append(f'<path d="M{left} {yy(v):.1f}h{width}" stroke="#e2e7ed"/><text x="40" y="{yy(v)+4:.1f}">{v:.1f}</text>')
    for t in np.linspace(minutes[0], minutes[-1], 6).astype(int):
        parts.append(f'<text x="{xx(t)-15:.1f}" y="{top+height+19}">{t//60:02d}:{t%60:02d}</text>')
    for j in indices:
        commands, active = [], False
        for t, v in zip(minutes, values[:, j]):
            if not np.isfinite(v):
                active = False
                continue
            commands.append(f'{"L" if active else "M"}{xx(t):.1f},{yy(float(np.clip(v, low, high))):.1f}')
            active = True
        parts.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{COLORS[j]}" stroke-width="1.5"/>')


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = list((ROOT/'outputs').rglob('submission*.csv'))
    protected_paths += list((ROOT/'src').glob('*.py')) + list((ROOT/'t05_forecast').glob('*.py'))
    protected_paths += list((ROOT/'models').rglob('*.*')) + [ROOT/'notebooks/current_baseline.ipynb']
    protected_paths += list((ROOT/'data/raw').glob('*.csv'))
    protected_paths += list((ROOT/'outputs/weekly_forecast_diagnostic').rglob('*.*'))
    protected = {str(p): sha(p) for p in protected_paths if p.is_file() and p.resolve() != Path(__file__).resolve()}
    (OUT/'design.json').write_text(json.dumps(SPEC, indent=2), encoding='utf-8')
    recs = load_training()
    fields, coverage, templates = {}, [], []
    for name, rec in recs.items():
        eligible = {n: r for n, r in recs.items() if r.day != rec.day}
        eligible[name] = rec
        tmpl = make_template(eligible, name, lam=.8)
        # Verify that the entire target date is absent from every template input.
        changed = rec.copy(); changed.y[:] = 1e9
        mutated = dict(eligible); mutated[name] = changed
        np.testing.assert_allclose(tmpl, make_template(mutated, name, .8), equal_nan=True)
        z = np.log1p(rec.y)
        residual = z-tmpl
        # Avoid warnings for E, which is absent from all short recordings.
        norm = np.full_like(z, np.nan)
        for j, unit in enumerate(UNITS):
            good = np.isfinite(z[:, j])
            if good.any():
                norm[good, j] = (z[good, j]-z[good, j].mean())/max(z[good, j].std(), 1e-9)
            coverage.append(dict(recording=name, day=rec.day, weekday=rec.weekday, unit=unit,
                                 observed=int(good.sum()), minutes=rec.n_minutes,
                                 residual_observed=int(np.isfinite(residual[:, j]).sum())))
        fields[name] = dict(level=norm, change1=delta(z, 1), change5=delta(z, 5),
                            residual=residual, residual_change1=delta(residual, 1),
                            residual_change5=delta(residual, 5))
        templates.append(dict(recording=name, excluded_date=rec.day,
                              allowed_source_dates=sorted({r.day for n,r in eligible.items() if n != name}),
                              finite_template_cells=int(np.isfinite(tmpl).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    (OUT/'template_sources.json').write_text(json.dumps(templates, indent=2))
    record_rows, window_rows, samples = [], [], {}
    for name, modes in fields.items():
        rec = recs[name]
        for mode, values in modes.items():
            step = 5 if mode.endswith('5') else 1 if mode.endswith('1') else 0
            for source in range(8):
                for target in range(8):
                    if source == target:
                        continue
                    for lag in range(16):
                        idx = np.arange(step, len(values)-lag)
                        x, y, z = values[idx, source], values[idx+lag, target], values[idx, target]
                        good = np.isfinite(x) & np.isfinite(y)
                        base = dict(mode=mode, source=UNITS[source], target=UNITS[target], lag=lag,
                                    recording=name, day=rec.day, weekday=rec.weekday)
                        both = good & np.isfinite(z)
                        record_rows.append(dict(**base, paired=int(good.sum()), r=corr(x[good],y[good]),
                                                partial_paired=int(both.sum()), partial_r=partial(x[both],y[both],z[both]) if lag else np.nan))
                        # Require the complete paired intervals inside one sensitivity block.
                        block = (idx-step)//30
                        inside = block == (idx+lag)//30
                        for w in np.unique(block[good & inside]):
                            keep = good & inside & (block == w)
                            pkeep = keep & both
                            if keep.sum() >= 4:
                                window_rows.append(dict(**base, window=int(w), paired=int(keep.sum()),
                                                        r=corr(x[keep], y[keep]), partial_paired=int(pkeep.sum()),
                                                        partial_r=partial(x[pkeep], y[pkeep], z[pkeep]) if lag else np.nan))
                        if mode == 'residual_change5':
                            samples[(source, target, lag, name)] = (x[both], y[both], z[both])
    recordings = pd.DataFrame(record_rows)
    windows = pd.DataFrame(window_rows)
    recordings.to_csv(OUT/'by_recording.csv', index=False)
    windows.to_csv(OUT/'by_window.csv', index=False)
    summary = recordings.groupby(['mode','source','target','lag']).agg(
        paired=('paired','sum'), record_r_median=('r','median'),
        record_r_min=('r','min'), record_r_max=('r','max')).reset_index()
    summary.to_csv(OUT/'all_relationships.csv', index=False)
    candidates = []
    gate = SPEC['gates']
    for (source, target, lag), g in recordings[recordings['mode'].eq('residual_change5') & recordings.lag.gt(0)].groupby(['source','target','lag']):
        s, t = UNITS.index(source), UNITS.index(target)
        parts = [samples[(s,t,lag,n)] for n in recs]
        x,y,z = [np.concatenate([a[k] for a in parts]) for k in range(3)]
        p = partial(x,y,z)
        sign = 1 if p >= 0 else -1
        enough = g[g.partial_paired.ge(gate['min_record_paired']) & g.partial_r.notna()]
        w = windows[(windows['mode']=='residual_change5') & (windows.source==source) & (windows.target==target) & (windows.lag==lag)]
        w = w[w.partial_paired.ge(gate['min_window_paired']) & w.partial_r.notna()]
        weekday_r, loo = {}, {}
        for weekday in ['Fri','Sat']:
            pp = [samples[(s,t,lag,n)] for n,r in recs.items() if r.weekday==weekday]
            weekday_r[weekday] = partial(*[np.concatenate([a[k] for a in pp]) for k in range(3)])
        for day in sorted({r.day for r in recs.values()}):
            pp = [samples[(s,t,lag,n)] for n,r in recs.items() if r.day != day]
            loo[day] = partial(*[np.concatenate([a[k] for a in pp]) for k in range(3)])
        sync = recordings[(recordings['mode']=='residual_change5') & (recordings.source==source) & (recordings.target==target) & recordings.lag.eq(0)]
        # Same pair, same residual change definition; robust record median at zero lag.
        sync_r = sync.r.median()
        agreement = float((enough.partial_r*sign > 0).mean()) if len(enough) else 0.
        w_agree = float((w.partial_r*sign > 0).mean()) if len(w) else 0.
        criteria = dict(paired=len(x)>=gate['min_paired'], records=len(enough)>=gate['min_records'],
            dates=enough.day.nunique()>=gate['min_dates'], partial=abs(p)>=gate['min_abs_partial'],
            record_median=(enough.partial_r*sign).median()>=gate['min_record_signed_median'],
            record_sign=agreement>=gate['min_record_sign_agreement'],
            weekdays=signed_min(list(weekday_r.values()),sign)>=gate['min_weekday_signed_partial'],
            windows=len(w)>=gate['min_windows'], window_sign=w_agree>=gate['min_window_sign_agreement'],
            lag_advantage=abs(p)-abs(sync_r)>=gate['min_lag_advantage_over_sync'],
            leave_date_out=signed_min(list(loo.values()),sign)>=gate['min_leave_date_out_signed_partial'])
        candidates.append(dict(source=source,target=target,lag=lag,paired=len(x),partial_r=p,
            records=len(enough),dates=enough.day.nunique(),record_partial_median=enough.partial_r.median(),
            record_sign_agreement=agreement, windows=len(w),window_sign_agreement=w_agree,
            friday_partial=weekday_r['Fri'],saturday_partial=weekday_r['Sat'], synchronous_r=sync_r,
            min_signed_leave_date_out=signed_min(list(loo.values()),sign),
            failures=','.join(k for k,v in criteria.items() if not v), stable=all(criteria.values())))
    ranking = pd.DataFrame(candidates)
    ranking['abs_partial'] = ranking.partial_r.abs()
    ranking = ranking.sort_values('abs_partial',ascending=False)
    ranking.to_csv(OUT/'lag_screen.csv', index=False)
    stable = ranking[ranking.stable].head(3)
    stable.to_csv(OUT/'selected_relationships.csv', index=False)
    # At most three illustrative pairs; these are explicitly NOT selected methods.
    examples = ranking.drop_duplicates(['source','target']).head(3)
    examples.to_csv(OUT/'exploratory_examples.csv', index=False)
    draw_plots(recs, fields, recordings, examples)
    assert all(sha(Path(p))==expected for p,expected in protected.items())
    result = dict(runtime_seconds=time.perf_counter()-started,training_recordings=len(recs),
        training_dates=len({r.day for r in recs.values()}), stable_relationships=len(stable),
        selected=stable.fillna('').to_dict('records'), new_forecast_method_tested=False,
        t05_fits=0,t05_inference=0,evaluation_labels_loaded=False,
        reason='No stable incremental lag passes training-only gates.' if stable.empty else 'Stable relationships need representation review; T05 already models linear residual camera lags.',
        protected_files=len(protected),protected_files_unchanged=True, protected_sha256=protected,
        plots=['normalized_traces.svg','relationship_heatmap.svg','lag_examples.svg'])
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='protected_sha256'},indent=2))


def draw_plots(recs, fields, recordings, examples):
    parts = svg_start(1130, 1400, 'Training camera traces: standardized log1p counts')
    parts.append('<text x="30" y="55">Gaps remain blank. Each camera standardized within recording; y range clipped to +/-3.</text>')
    for j,u in enumerate(UNITS):
        parts.append(f'<text x="{40+j*130}" y="80" style="fill:{COLORS[j]}">{u}</text>')
    for i,(name,rec) in enumerate(recs.items()):
        trace_panel(parts,fields[name]['level'],rec.minutes,115+i*210,f'{name} | {rec.weekday} | E absent in first four recordings')
    save_svg('normalized_traces.svg',parts)
    parts = svg_start(1120,640,'Camera relationships: median recording correlation, synchronous')
    parts.append('<text x="30" y="58">Cell = median Pearson r across recordings with >=30 exact pairs. Number below = supported recordings.</text>')
    for panel,mode in enumerate(['level','change5','residual_change5']):
        left = 70+panel*355
        parts.append(f'<text x="{left}" y="95">{mode}</text>')
        for j,u in enumerate(UNITS):
            parts.append(f'<text x="{left+j*35+10}" y="120">{u[-1]}</text><text x="{left-22}" y="{145+j*45}">{u[-1]}</text>')
            for k,v in enumerate(UNITS):
                g = recordings[(recordings['mode']==mode)&(recordings.source==u)&(recordings.target==v)&recordings.lag.eq(0)&recordings.paired.ge(30)]
                r = g.r.median() if len(g) else np.nan
                color = '#eeeeee' if not np.isfinite(r) else f'rgb({int(245-130*max(-r,0))},{int(245-155*abs(r))},{int(245-130*max(r,0))})'
                x,y=left+k*35,130+j*45
                parts.append(f'<rect x="{x}" y="{y}" width="34" height="44" fill="{color}"/>')
                if np.isfinite(r):
                    parts.append(f'<text x="{x+2}" y="{y+17}" style="font-size:10px">{r:.2f}</text><text x="{x+12}" y="{y+34}" style="font-size:9px">{len(g)}</text>')
    parts.append('<text x="30" y="545">Red = positive; blue = negative. E has only two recordings; levels alone can reflect the shared daily template.</text>')
    parts.append('<text x="30" y="574">Residuals subtract an entire-date-excluded weekly/prayer template. Blank diagonal. No location or movement inference.</text>')
    save_svg('relationship_heatmap.svg',parts)
    parts = svg_start(1130,800,'Exploratory lag alignments: highest pooled partial correlations')
    parts.append('<text x="30" y="55">Illustrations only, not validated relationships. Residual 5-minute changes; gaps retained; source shifted forward by lag.</text>')
    for i,row in enumerate(examples.itertuples()):
        g=recordings[(recordings['mode']=='residual_change5')&(recordings.source==row.source)&(recordings.target==row.target)&recordings.lag.eq(row.lag)&recordings.partial_paired.ge(30)]
        # Display the weakest recording, avoiding an impressive-window-only plot.
        sign=1 if row.partial_r>=0 else -1
        name=g.loc[(g.partial_r*sign).idxmin(),'recording'] if len(g) else next(iter(recs))
        rec=recs[name]; values=fields[name]['residual_change5']; s,t=UNITS.index(row.source),UNITS.index(row.target)
        shown=np.full_like(values,np.nan)
        shown[row.lag:,s]=values[:-row.lag,s]; shown[:,t]=values[:,t]
        title=f'{row.source} -> {row.target}, lag {row.lag}m | pooled partial r={row.partial_r:.2f} | weakest recording {name}'
        trace_panel(parts,shown,rec.minutes,105+i*225,title,[s,t],(-.6,.6))
        parts.append(f'<text x="80" y="{285+i*225}" style="fill:{COLORS[s]}">{row.source} shifted +{row.lag}m</text><text x="350" y="{285+i*225}" style="fill:{COLORS[t]}">{row.target} unshifted</text>')
    save_svg('lag_examples.svg',parts)


if __name__ == '__main__':
    main()
