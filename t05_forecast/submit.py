"""Predict every hole with the template + deviation model, build submission.csv, draw the result."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import gp
from .core import FORECAST, HIDDEN, HIDDEN_FUTURE, KIND_CODE, KIND_NAME, KINDS, MISSING, load_recordings, submission_frame
from .missing import fill_missing_direct
from .pipeline import FRI, OTHER_TEST_DAY, P11, P12, SAT, deviation_fields, fit_cached
from .prayer import prayer_minutes
from .reference import donor_table, reference

COLORS = {FORECAST: "#d62728", HIDDEN: "#9467bd", HIDDEN_FUTURE: "#e377c2", MISSING: "#ff7f0e"}

DEFAULTS = {
    "mode": "strict",       # strict: parameters for a test recording are fitted without that recording; all: on every released row
    "steps": 160, "lam": 0.8, "rank": 2, "ells": "4,45", "kinds": "", "delay": "1", "delta": "1", "same": "1", "threads": 0,
    "missing_direct": 0.5,  # share of the direct drop-out filler where both methods apply (0 = model only, 1 = direct only)
    "missing_model_units": "A,E,F",            # drop-outs of these cameras: deviation model averaged with the direct filler
    "missing_direct_units": "B,C,D,G,H",       # drop-outs of these cameras: direct filler only; all other cameras: reference
    "forecast_reference_units": "I,J,K,L,M",   # forecast holes of these cameras stay at the reference (no earlier week exists)
    "w_forecast": 1.0, "w_hidden": 1.0, "w_hidden-future": 1.0, "w_missing": 1.0,   # final = reference + w * (ours - reference)
}


def _unit_set(text: str) -> set:
    return {f"cam_{x.strip()}" for x in str(text).split(",") if x.strip()}


def settings(cfg: dict | None) -> dict:
    s = dict(DEFAULTS)
    s.update(cfg or {})
    s["steps"], s["rank"], s["lam"] = int(s["steps"]), int(s["rank"]), float(s["lam"])
    s["ells"] = tuple(float(x) for x in str(s["ells"]).split(","))
    s["kinds"] = tuple(str(s["kinds"]).split(",")) if s["kinds"] else tuple("ou" for _ in s["ells"])
    s["use_delay"], s["use_delta"], s["use_same"] = str(s["delay"]) == "1", str(s["delta"]) == "1", str(s["same"]) == "1"
    s["threads"] = int(s["threads"]) or None
    s["missing_direct"] = float(s["missing_direct"])
    s["weights"] = {KIND_CODE[k]: float(s[f"w_{k}"]) for k in KINDS}
    s["key"] = (f"lam{s['lam']}_r{s['rank']}_e{'-'.join(str(e) for e in s['ells'])}_{'-'.join(s['kinds'])}"
                f"_dl{int(s['use_delay'])}_dt{int(s['use_delta'])}_sm{int(s['use_same'])}_s{s['steps']}")
    members = s.get("members")
    if isinstance(members, str):          # command-line form: "ells=3,12,60;kinds=ou,m32,ou|ells=4,45"
        members = [dict(kv.split("=", 1) for kv in part.split(";") if kv) for part in members.split("|") if part]
    s["members"] = members or None
    s["cfg"] = {k: v for k, v in (cfg or {}).items() if k != "members"}
    return s


def member_settings(s: dict) -> list:
    """Settings of every model in the ensemble (a single model when no members are listed)."""
    if not s["members"]:
        return [s]
    return [settings({**s["cfg"], **m}) for m in s["members"]]


def load_models(data_dir, cache_dir, s: dict, verbose: bool = True) -> list:
    return [(load_model(data_dir, cache_dir, ms, verbose=verbose and i == 0), ms) for i, ms in enumerate(member_settings(s))]


def predict_ensemble(models: list, rec, tmpl_day: str, param_day: str, direct_main: list, direct_new: list) -> np.ndarray:
    """Average of the members' values for every hole (NaN where no member predicts)."""
    preds = []
    for m, ms in models:
        p = predict_holes(rec, m["tm"][tmpl_day], m["sl"][tmpl_day], m["params"][param_day], ms, direct_main, direct_new)
        only = {k.strip() for k in str(ms.get("only_kinds", "")).split(",") if k.strip()}
        if only:                                  # this member is used for some kinds of hole only
            for kind in KINDS:
                if kind not in only:
                    p[rec.status == KIND_CODE[kind]] = np.nan
        preds.append(p)
    stack = np.stack(preds)
    cnt = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)


def fit_kwargs(s: dict) -> dict:
    return dict(steps=s["steps"], rank=s["rank"], ells=s["ells"], kinds=s["kinds"], use_delay=s["use_delay"],
                use_delta=s["use_delta"], use_same_minute=s["use_same"], verbose=False, threads=s["threads"])


def predict_holes(rec, tmpl, slope, params, s: dict, direct_main: list, direct_new: list) -> np.ndarray:
    """[T, U] our value for every hole we can predict (NaN where we cannot).

    Deviation model for all four kinds; for drop-outs it is averaged with the direct filler
    (share `missing_direct`). `direct_main` are the recordings the direct filler learns from
    for cameras A-H, `direct_new` those for cameras that exist only on the test days.
    """
    model = gp.predict_recording(rec, tmpl, slope, params, strict=True)
    share = s["missing_direct"]
    model_units, direct_units = _unit_set(s["missing_model_units"]), _unit_set(s["missing_direct_units"])
    direct = fill_missing_direct(rec, direct_main) if direct_main else np.full(rec.y.shape, np.nan)
    for c, unit in enumerate(rec.units):
        m = rec.status[:, c] == MISSING
        if m.any():
            if unit in model_units:
                both = m & np.isfinite(model[:, c]) & np.isfinite(direct[:, c])
                model[both, c] = (1 - share) * model[both, c] + share * direct[both, c]
            elif unit in direct_units:
                model[m, c] = direct[m, c]
            else:
                model[m, c] = np.nan
        if unit in _unit_set(s["forecast_reference_units"]):
            model[rec.status[:, c] == FORECAST, c] = np.nan
    return model


def finalize(rec, model: np.ndarray, ref: np.ndarray, weights: dict) -> np.ndarray:
    """Holes without a model value take the reference; then final = reference + w * (ours - reference)."""
    final = np.full(rec.y.shape, np.nan)
    mv = np.where(np.isfinite(model), model, ref)
    for code, w in weights.items():
        m = rec.status == code
        final[m] = (ref + w * (mv - ref))[m]
    return final


def draw(rec, tmpl, pred, ref, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = rec.units
    fig, axes = plt.subplots(len(units), 1, figsize=(17, 2.0 * len(units)), sharex=True)
    hours = rec.minutes / 60.0
    prayers = prayer_minutes(rec.day)
    for ax, (c, unit) in zip(axes, enumerate(units)):
        ax.plot(hours, np.expm1(tmpl[:, c]), color="#9ecae1", lw=1.2, label="last week's curve (template)")
        ax.plot(hours, rec.y[:, c], color="k", lw=0.9, label="released today")
        for code, color in COLORS.items():
            m = rec.status[:, c] == code
            if m.any():
                ax.scatter(hours[m], ref[m, c], s=3, color="#7f7f7f", alpha=0.6, label="reference", zorder=2)
                ax.scatter(hours[m], pred[m, c], s=5, color=color, label=f"ours: {KIND_NAME[code]}", zorder=3)
        for name in ("dhuhr", "asr", "maghrib", "isha"):
            ax.axvline(prayers[name] / 60.0, color="g", ls="--", lw=0.7, alpha=0.6)
        ax.set_ylabel(unit.replace("cam_", "cam "))
        ax.grid(alpha=0.25)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    axes[0].legend(handles, labels, loc="upper right", ncol=4, fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xticks(np.arange(11, 22))
    axes[-1].set_xlabel("clock time (hour, Madinah); dashed green: Dhuhr, Asr, Maghrib, Isha")
    fig.tight_layout()
    fig.savefig(out_png, dpi=75)
    plt.close(fig)


def load_model(data_dir: str | Path, cache_dir: str | Path, s: dict, verbose: bool = True) -> dict:
    """Recordings, templates and fitted parameters for both test recordings."""
    recs = load_recordings(data_dir)
    train = [r for r in recs.values() if r.split == "train"]
    units = sorted({u for r in train for u in r.units})
    tm, sl, field = deviation_fields(recs, (FRI, SAT, P11, P12), s["lam"], verbose=verbose)
    if s["mode"] == "strict":
        params = {FRI: fit_cached(cache_dir, f"without_fri_{s['key']}", [field[n] for n in (SAT, P11, P12)], **fit_kwargs(s)),
                  SAT: fit_cached(cache_dir, f"without_sat_{s['key']}", [field[n] for n in (FRI, P11, P12)], **fit_kwargs(s))}
    else:
        p_all = fit_cached(cache_dir, f"all_{s['key']}", [field[n] for n in (FRI, SAT, P11, P12)], **fit_kwargs(s))
        params = {FRI: p_all, SAT: p_all}
    return {"recs": recs, "train": train, "units": units, "table": donor_table(train, units), "tm": tm, "sl": sl,
            "field": field, "params": params, "long_train": [r for r in train if r.n_minutes >= 600]}


def build(data_dir: str | Path, cache_dir: str | Path, out_dir: str | Path, cfg: dict | None = None,
          compare: str | Path | None = None, plots: bool = True) -> pd.DataFrame:
    """Fit (or load) the model, predict every hole of both test recordings and write submission.csv."""
    t0 = time.time()
    s = settings(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"mode={s['mode']} settings={s['key']} missing_direct={s['missing_direct']} "
          f"weights={ {KIND_NAME[k]: v for k, v in s['weights'].items()} }", flush=True)
    models = load_models(data_dir, cache_dir, s)
    m = models[0][0]
    recs = m["recs"]
    print(f"parameters ready after {time.time() - t0:.0f}s; models averaged: {len(models)}", flush=True)
    for mm, ms in models:
        for day in (FRI, SAT):
            print(f"--- model {ms['key']} used for {day} ---\n" + gp.describe(mm["params"][day]), flush=True)

    preds, rows = {}, []
    for day in (FRI, SAT):
        rec = recs[day]
        ref = reference(rec, m["table"])
        model = predict_ensemble(models, rec, day, day, m["long_train"], [recs[OTHER_TEST_DAY[day]]])
        final = finalize(rec, model, ref, s["weights"])
        if s.get("base_submission"):
            # hybrid trial: holes of the listed kinds keep the values of an earlier submission
            base = pd.read_csv(s["base_submission"])
            base_value = dict(zip(base["id"], base["people"]))
            keep = {k.strip() for k in str(s.get("keep_base_kinds", "")).split(",") if k.strip()}
            for (r, c), gid in rec.ids.items():
                if KIND_NAME[int(rec.status[r, c])] in keep:
                    final[r, c] = base_value[gid]
                    model[r, c] = base_value[gid]
        preds[day] = final
        for code in (FORECAST, HIDDEN, HIDDEN_FUTURE, MISSING):
            k = rec.status == code
            rows.append((day, KIND_NAME[code], int(k.sum()), int((k & ~np.isfinite(model)).sum()),
                         float(np.nanmean(ref[k])), float(np.nanmean(final[k])), float(np.nanmean(np.abs(final[k] - ref[k])))))
        if plots:
            draw(rec, m["tm"][day], final, ref, out / f"predictions_{day}.png",
                 f"{day} ({rec.weekday}): released rows, last week's curve and our predictions")
    summary = pd.DataFrame(rows, columns=["recording", "kind", "holes", "holes_left_at_reference", "mean_reference", "mean_ours", "mean_abs_move"])

    sub = submission_frame(recs, preds, Path(data_dir) / "sample_submission.csv")
    sub.to_csv(out / "submission.csv", index=False)
    back = pd.read_csv(out / "submission.csv")
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv")
    assert list(back["id"]) == list(sample["id"]) and len(back) == len(sample) == back["id"].nunique()
    assert np.isfinite(back["people"]).all() and (back["people"] >= 0).all()
    summary.to_csv(out / "submission_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))
    print(f"wrote {out / 'submission.csv'}: {len(back)} rows, ids and order equal to the sample, all values finite and >= 0")
    if compare:
        other = pd.read_csv(compare)
        test = pd.read_csv(Path(data_dir) / "test.csv")
        g = test[test["id"].notna()].set_index("id").loc[sample["id"]].reset_index()
        g["d"] = np.abs(back["people"].to_numpy() - other["people"].to_numpy())
        print(f"mean absolute difference to {Path(compare).name}:")
        print(g.pivot_table(index="recording", columns="kind", values="d", aggfunc="mean").round(2).to_string())
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return summary
