"""Forecast values for all 2,660 forecast targets of data/raw/test.csv.

Cameras A to H: model.forecast_at_now with the track's settings, the production templates and the frozen production
covariances in models/current_baseline/t05/ (nothing is fitted). Cameras I to M: copied from the accepted baseline
file outputs/noran/baseline_local/submission_baseline_local.csv.

Checks made here: with delay 0 and no timing edit the same code reproduces the accepted file for cameras A to H;
overwriting every reading after `now` leaves the new values unchanged.

Run from the repository root:  .venv/Scripts/python.exe -X utf8 noran/tracks/forecast/make_test_predictions.py
Output: outputs/noran/tracks/forecast/test_predictions.csv (id, people) and test_delays.csv (delay per day and now).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for folder in (ROOT, ROOT / "src", HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import model  # noqa: E402
from t05_forecast.core import FORECAST, load_recordings  # noqa: E402
from t05_forecast.pipeline import FRI, SAT, load_params  # noqa: E402
from t05_forecast.reference import ref_forecast  # noqa: E402
from t05_forecast.submit import member_settings, settings  # noqa: E402

OUT = ROOT / "outputs/noran/tracks/forecast"
ASSETS = ROOT / "models/current_baseline"
BASELINE = ROOT / "outputs/noran/baseline_local/submission_baseline_local.csv"


def production_members(recs, name, members_cfg, keep):
    day = "fri" if name == FRI else "sat"
    return [(model.production_template(recs, name, m["lam"], keep),
             load_params(ASSETS / "t05" / f"params_without_{day}_{m['key']}.npz")) for m in members_cfg]


def main() -> None:
    torch.set_num_threads(2)
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(model.DEFAULTS)
    accepted_cfg = dict(model.DEFAULTS, method="none", sd_delta=None, ell_delta=None)
    keep = tuple(model.TEMPLATE_KEEP)
    members_cfg = member_settings(settings(json.loads((ROOT / "t05_forecast_config.json").read_text(encoding="utf-8"))))
    recs = load_recordings(ROOT / "data/raw")
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    targets = test[test.kind.eq("forecast")].copy()
    base = pd.read_csv(BASELINE).set_index("id").people
    assert len(targets) == 2660 and targets.id.isin(base.index).all()

    values, rows, delays = {}, [], []
    checks = moved = 0
    worst_repro = 0.0
    with threadpool_limits(limits=2):
        for name in (FRI, SAT):
            rec = recs[name]
            base_members = production_members(recs, name, members_cfg, model.PRAYERS)
            new_members = base_members if keep == tuple(model.PRAYERS) else production_members(recs, name, members_cfg, keep)
            ref = ref_forecast(rec)
            fc = rec.status == FORECAST
            for now in np.unique(rec.now[fc]):
                now = int(now)
                mask = fc & (rec.now == now)
                a, _ = model.forecast_at_now(rec, base_members, now, accepted_cfg)
                b, info = model.forecast_at_now(rec, new_members, now, cfg)
                altered = rec.copy()
                altered.y[(altered.minutes > now)[:, None] & np.isfinite(altered.y)] = 1e8
                again, _ = model.forecast_at_now(altered, new_members, now, cfg)
                delays.append(dict(recording=name, day=rec.day, now=now, clock=f"{now // 60:02d}:{now % 60:02d}", delay=info["shift"]))
                for r, c in zip(*np.nonzero(mask)):
                    unit, gid = rec.units[c], rec.ids[(int(r), int(c))]
                    if unit[-1] in model.NEW_CAMERAS:
                        values[gid] = float(base[gid])                           # accepted rule, copied
                        continue
                    checks += 1
                    moved += int(not np.isclose(np.nan_to_num(b[r, c], nan=-1.0), np.nan_to_num(again[r, c], nan=-1.0), rtol=0, atol=1e-8))
                    acc = a[r, c] if np.isfinite(a[r, c]) else ref[r, c]
                    new = b[r, c] if np.isfinite(b[r, c]) else ref[r, c]
                    worst_repro = max(worst_repro, abs(acc - float(base[gid])))
                    values[gid] = float(max(0.0, new))
                    rows.append(dict(id=gid, recording=name, day=rec.day, unit=unit, now=now, minute=int(rec.minutes[r]),
                                     reference=ref[r, c], baseline=float(base[gid]), new=values[gid]))
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    order = sample[sample.id.isin(values)].id
    out = pd.DataFrame(dict(id=order.to_numpy(), people=[values[i] for i in order]))
    assert len(out) == 2660 and set(out.id) == set(targets.id) and np.isfinite(out.people).all() and (out.people >= 0).all()
    out.to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(delays).to_csv(OUT / "test_delays.csv", index=False)

    d = pd.DataFrame(rows)
    d["move"] = (d.new - d.baseline).abs()
    d["base_move"] = (d.baseline - d.reference).abs()
    pd.set_option("display.width", 220)
    print(f"settings: method={cfg['method']}, timing sd/memory={cfg['sd_delta']}/{cfg['ell_delta']}")
    print(f"accepted file reproduced for cameras A to H with delay 0: max abs difference {worst_repro:.2e}")
    print(f"leak checks {checks}, failures {moved}")
    im = targets[targets.unit.str[-1].isin(list(model.NEW_CAMERAS))]
    print(f"cameras I to M: {len(im)} targets copied from the baseline file; cameras A to H: {len(d)} targets from this method")
    big = d[d.unit.isin(["cam_A", "cam_E", "cam_F"])]
    for name, g in d.groupby("recording"):
        gb = big[big.recording.eq(name)]
        t = gb.groupby("now").agg(aef_abs_diff=("move", "sum"), aef_baseline_move_from_reference=("base_move", "sum")).reset_index()
        allc = g.groupby("now").move.sum().reset_index(name="all_abs_diff")
        t = t.merge(allc, on="now")
        t["delay"] = [next(x["delay"] for x in delays if x["recording"] == name and x["now"] == n) for n in t.now]
        t["clock"] = [f"{n // 60:02d}:{n % 60:02d}" for n in t.now]
        print(f"{name} ({g.day.iloc[0]}): summed absolute difference from the baseline file per now")
        print(t[["clock", "delay", "aef_abs_diff", "all_abs_diff", "aef_baseline_move_from_reference"]].round(2).to_string(index=False))
        print(f"  total A,E,F {t.aef_abs_diff.sum():.1f}; all A-H {t.all_abs_diff.sum():.1f}; "
              f"mean abs difference per A-H target {g.move.mean():.3f}")
    print(f"wrote {OUT / 'test_predictions.csv'} ({len(out)} rows) in {time.perf_counter() - started:.0f} s")
    if moved:
        print("LEAK CHECK FAILED")
        sys.exit(2)


if __name__ == "__main__":
    main()
