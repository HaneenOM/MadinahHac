"""Shared test harness for one mission (hidden, hidden-future or missing).

Every track is scored here on the same cases, against the same reference and the same accepted baseline,
so numbers from different tracks can be compared and added up.

Usage (repository root):
    .venv/Scripts/python.exe noran/track_eval.py <mission> <path/to/module.py> [--set fixed|mirror] [--label NAME] [--no-checks]

Two case sets
  fixed   the team's own local test: 11 Sep under the real Friday mask (B and G hidden) and 12 Sep under the
          real Saturday mask (D hidden). This is the number the team records.
  mirror  the same two dates with the masks swapped: 11 Sep under the Saturday mask (D hidden) and 12 Sep under
          the Friday mask (B and G hidden). Nobody has tuned on it. Use it to choose between variants, and the
          fixed set to confirm.
  testday (hidden and hidden-future only) the two REAL test days, where a full same-weekday day exists one week
          earlier, as it does for every real target. A camera that is released there is treated as hidden:
          D on Friday 18 Sep, B and G together on Saturday 19 Sep. Hidden cases = its released minutes.
          Hidden-future cases = two pseudo-origins inside every released stretch (7 and 13 minutes after the
          stretch starts), targets up to 15 minutes later inside the same stretch. Models are fitted on all four
          training dates. The reference is weaker than the organizers' (its preferred donor is the other hidden
          camera, which is absent on that day), so read these numbers as comparisons between methods.

The module must define two functions:

    fit(training, day, context) -> model
        training : pandas frame of the training rows of every date EXCEPT `day`
                   (columns recording, day, weekday, unit, minute, minute_of_day, people).
                   day is None when the module is run on the real test recordings (noran/track_predict.py):
                   training then holds every training date.
        context  : {'schedule': prayer timetable frame,
                    'test_released': released rows of the two test recordings (same columns as training);
                                     the host allows fitting on them,
                    'config': the accepted baseline settings of this mission,
                    'mission': mission name}

    predict(case, t, model, context) -> float
        case : case_id, day, recording, kind, unit, minute, now, horizon
               (now and horizon are empty for hidden and missing)
        t    : the released feed of that recording under the mask, a minute x camera table.
               It never contains the answer of the case.
        Return NaN to keep the accepted baseline value for this case.

What the harness does
  1. fits once per held-out date, predicts every case of the mission, saves predictions,
  2. only then reads the answers and scores: per date, per date and camera, and the team's metric
     (mean over the two dates of MAE / reference MAE),
  3. leak checks: hidden -> the columns of the hidden cameras are overwritten with 1e8, the value must not move;
     hidden-future -> every reading after `now` is overwritten with 1e8, the value must not move,
  4. robustness: wins and losses per window against the accepted baseline, and the gain with each window left out,
  5. (fixed set) the total local score if this mission's values replaced the baseline's.

Outputs: outputs/noran/tracks/<label>[__mirror]/{predictions.csv, scored.csv, by_date_camera.csv, result.json}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(folder))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import baseline_tasks as tasks  # noqa: E402
import local_score  # noqa: E402
import v2_hidden  # noqa: E402
from pipeline import CAMERAS, load, locate_data, table  # noqa: E402
from prayer_experiments import timetable  # noqa: E402

MISSIONS = ("hidden", "hidden-future", "missing")
LOCAL = ROOT / "outputs/local_score"
MIRROR = ROOT / "outputs/noran/mirror"
MIRROR_MAP = {"20260918_1100": "20260912_1100", "20260919_1100": "20260911_1100"}   # test mask -> training day
TESTDAY = ROOT / "outputs/noran/testday"
TESTDAY_HIDDEN = {"20260918_1100": ["cam_D"], "20260919_1100": ["cam_B", "cam_G"]}   # released cameras treated as hidden
TESTDAY_ORIGINS = (7, 13)                                                            # minutes after a stretch starts


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"track_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    for name in ("fit", "predict"):
        if not hasattr(module, name):
            raise AttributeError(f"{path} must define {name}()")
    return module


def build_mirror(train, test, schedule) -> None:
    """Swapped-mask cases with the accepted baseline's values. Written once; later runs only read."""
    MIRROR.mkdir(parents=True, exist_ok=True)
    case_rows, labels, feed_parts = [], [], []
    for test_rec, val_rec in MIRROR_MAP.items():
        truth = train[train.recording.eq(val_rec)]
        day = truth.day.iloc[0]
        mask = test[test.recording.eq(test_rec) & test.unit.isin(CAMERAS)].copy()
        mask["recording"] = val_rec
        released = mask[mask.id.isna()][["recording", "unit", "minute_of_day"]]
        feed = released.merge(truth, on=["recording", "unit", "minute_of_day"], how="left")
        feed = feed[feed.people.notna()][["recording", "unit", "minute_of_day", "people"]].copy()
        feed.insert(0, "day", day)
        feed_parts.append(feed)
        targets = mask[mask.id.notna() & mask.kind.isin(MISSIONS)].drop(columns="people").merge(
            truth[["recording", "unit", "minute_of_day", "people"]], on=["recording", "unit", "minute_of_day"], how="inner")
        for r in targets.itertuples():
            minute = int(r.minute_of_day)
            now = int(r.now_minute_of_day) if pd.notna(r.now_minute_of_day) else None
            case_id = f"{val_rec}_{r.unit}_{minute}"
            case_rows.append(dict(case_id=case_id, day=day, recording=val_rec, kind=r.kind, unit=r.unit, minute=minute,
                                  now=now, horizon=(minute - now) if now is not None else None))
            labels.append(dict(case_id=case_id, truth=float(r.people)))
    cases = pd.DataFrame(case_rows).sort_values(["day", "kind", "unit", "minute"])
    feed = pd.concat(feed_parts, ignore_index=True)
    assert cases.case_id.is_unique and not feed.duplicated(["recording", "unit", "minute_of_day"]).any()
    feeds = table(feed)
    rows = []
    for day, today in cases.groupby("day"):
        models = local_score.fit_fold(train, day, schedule)
        for r in today.itertuples():
            reference, prediction, _ = local_score.predict_case(r, feeds[r.recording], models, {}, schedule)
            rows.append(dict(case_id=r.case_id, day=day, kind=r.kind, reference=reference, prediction=prediction))
    cases.to_csv(MIRROR / "cases.csv", index=False)
    feed.to_csv(MIRROR / "released_feed.csv", index=False)
    pd.DataFrame(labels).to_csv(MIRROR / "labels.csv", index=False)
    pd.DataFrame(rows).to_csv(MIRROR / "predictions.csv", index=False)


def build_testday(train, test, schedule) -> None:
    """Pseudo-hidden cases on the two real test days with the accepted baseline's values. Written once."""
    TESTDAY.mkdir(parents=True, exist_ok=True)
    released = test[test.id.isna() & test.people.notna()]
    case_rows, labels, feed_parts = [], [], []
    for rec, hidden_units in TESTDAY_HIDDEN.items():
        g = released[released.recording.eq(rec)]
        day = g.day.iloc[0]
        feed = g[~g.unit.isin(hidden_units)][["recording", "unit", "minute_of_day", "people"]].copy()
        feed.insert(0, "day", day)
        feed_parts.append(feed)
        minutes = sorted(g[g.unit.isin(CAMERAS)].minute_of_day.unique())
        stretches, start, prev = [], minutes[0], minutes[0]
        for m in minutes[1:]:
            if m != prev + 1:
                stretches.append((start, prev))
                start = m
            prev = m
        stretches.append((start, prev))
        for unit in hidden_units:
            own = g[g.unit.eq(unit)].set_index("minute_of_day").people
            for minute, value in own.items():
                case_id = f"{rec}_{unit}_{int(minute)}"
                case_rows.append(dict(case_id=case_id, day=day, recording=rec, kind="hidden", unit=unit,
                                      minute=int(minute), now=None, horizon=None))
                labels.append(dict(case_id=case_id, truth=float(value)))
            for first, last in stretches:
                for offset in TESTDAY_ORIGINS:
                    now = first + offset
                    for minute in range(now + 1, min(now + 15, last) + 1):
                        if minute in own.index:
                            case_id = f"{rec}_{unit}_{minute}_n{now}"
                            case_rows.append(dict(case_id=case_id, day=day, recording=rec, kind="hidden-future", unit=unit,
                                                  minute=minute, now=now, horizon=minute - now))
                            labels.append(dict(case_id=case_id, truth=float(own[minute])))
    cases = pd.DataFrame(case_rows)
    feed = pd.concat(feed_parts, ignore_index=True)
    assert cases.case_id.is_unique and not feed.duplicated(["recording", "unit", "minute_of_day"]).any()
    feeds = table(feed)
    config = tasks.default_config()
    hidden_models = tasks.fit_hidden(train, config["hidden"], schedule)
    hf_models = tasks.fit_hidden_future(train, schedule, config["hidden-future"])
    rows = []
    for r in cases.itertuples():
        t = feeds[r.recording]
        if r.kind == "hidden":
            reference, prediction, _ = tasks.hidden_value(t, r.unit, int(r.minute), hidden_models, config["hidden"],
                                                          day=r.day, schedule=schedule)
        else:
            reference, prediction, _ = tasks.hidden_future_value(t, r.unit, int(r.now), int(r.horizon), r.day, schedule,
                                                                 hf_models, config["hidden-future"])
        rows.append(dict(case_id=r.case_id, day=r.day, kind=r.kind, reference=reference, prediction=prediction))
    cases.to_csv(TESTDAY / "cases.csv", index=False)
    feed.to_csv(TESTDAY / "released_feed.csv", index=False)
    pd.DataFrame(labels).to_csv(TESTDAY / "labels.csv", index=False)
    pd.DataFrame(rows).to_csv(TESTDAY / "predictions.csv", index=False)


def benchmark(which: str, train, test, schedule):
    if which == "fixed":
        cases, feeds = local_score.load_cases()
        return cases, feeds, LOCAL / "labels.csv", pd.read_csv(LOCAL / "predictions.csv")
    if which == "testday":
        if not (TESTDAY / "predictions.csv").exists():
            build_testday(train, test, schedule)
        cases = pd.read_csv(TESTDAY / "cases.csv")
        feeds = table(pd.read_csv(TESTDAY / "released_feed.csv"))
        return cases, feeds, TESTDAY / "labels.csv", pd.read_csv(TESTDAY / "predictions.csv")
    if not (MIRROR / "predictions.csv").exists():
        build_mirror(train, test, schedule)
    cases = pd.read_csv(MIRROR / "cases.csv")
    feeds = table(pd.read_csv(MIRROR / "released_feed.csv"))
    return cases, feeds, MIRROR / "labels.csv", pd.read_csv(MIRROR / "predictions.csv")


def window_key(mission: str, row) -> str:
    if mission == "hidden":
        return f"{row.day}|{row.unit}|{int(row.minute) // 30}"
    if mission == "hidden-future":
        return f"{row.day}|{row.unit}|{int(row.now)}"
    return f"{row.day}|{row.unit}|{row.outage}"


def add_outages(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["day", "unit", "minute"]).copy()
    new = (frame.groupby(["day", "unit"]).minute.diff().fillna(99) != 1)
    frame["outage"] = new.groupby([frame.day, frame.unit]).cumsum()
    return frame


def make_context(train, test, schedule, mission: str) -> dict:
    released = test[test.id.isna() & test.people.notna()][["recording", "day", "weekday", "unit", "minute", "minute_of_day", "people"]].copy()
    return {"schedule": schedule, "test_released": released, "config": tasks.default_config()[mission], "mission": mission}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mission", choices=MISSIONS)
    parser.add_argument("module", type=Path)
    parser.add_argument("--set", dest="which", choices=("fixed", "mirror", "testday"), default="fixed")
    parser.add_argument("--label")
    parser.add_argument("--no-checks", action="store_true")
    args = parser.parse_args()
    mission, label = args.mission, args.label or args.module.stem
    out = ROOT / "outputs/noran/tracks" / (label + ("" if args.which == "fixed" else f"__{args.which}"))
    if args.which == "testday" and mission == "missing":
        parser.error("the testday set covers hidden and hidden-future only")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train, test, _, _ = load(locate_data(None))
    schedule = timetable(ROOT / "models/current_baseline/prayer_times_madinah.csv", train, test)
    cases, feeds, labels_path, saved = benchmark(args.which, train, test, schedule)
    selected = cases[cases.kind.eq(mission)].copy()
    base = saved[saved.kind.eq(mission)].set_index("case_id")
    context = make_context(train, test, schedule, mission)
    module = load_module(args.module.resolve())

    rows, checks, moved = [], 0, 0
    with threadpool_limits(limits=2):
        for day, today in selected.groupby("day"):
            training = train[train.day.ne(day)].copy()
            assert not training.day.eq(day).any()
            if args.which == "testday":
                rec = today.recording.iloc[0]
                ctx_released = context["test_released"]
                drop = ctx_released.recording.eq(rec) & ctx_released.unit.isin(TESTDAY_HIDDEN[rec])
                context = dict(context, test_released=ctx_released[~drop].copy())
            model = module.fit(training, day, context)
            for r in today.itertuples():
                t = feeds[r.recording]
                value = module.predict(r, t, model, context)
                value = float(value) if value is not None else np.nan
                used = bool(np.isfinite(value))
                if used and not args.no_checks and mission in ("hidden", "hidden-future"):
                    altered = t.copy()
                    if mission == "hidden":
                        for unit in v2_hidden.PATTERNS[r.unit]:
                            altered[unit] = 1e8
                    else:
                        altered.loc[altered.index > int(r.now)] = 1e8
                    again = float(module.predict(r, altered, model, context))
                    checks += 1
                    if not np.isclose(value, again, rtol=0, atol=1e-8):
                        moved += 1
                rows.append(dict(case_id=r.case_id, day=day, unit=r.unit, minute=int(r.minute),
                                 now=(int(r.now) if pd.notna(r.now) else -1), used=used,
                                 prediction=max(0.0, value) if used else float(base.at[r.case_id, "prediction"]),
                                 baseline=float(base.at[r.case_id, "prediction"]), reference=float(base.at[r.case_id, "reference"])))
    detail = pd.DataFrame(rows)
    assert len(detail) == len(selected) and np.isfinite(detail[["prediction", "baseline", "reference"]]).all().all()
    detail.to_csv(out / "predictions.csv", index=False)

    labels = pd.read_csv(labels_path)                          # answers enter only after the predictions are saved
    d = detail.merge(labels, on="case_id", validate="one_to_one")
    for name in ("prediction", "baseline", "reference"):
        d[f"err_{name}"] = (d[name] - d.truth).abs()
    if mission == "missing":
        d = add_outages(d)
    d["window"] = [window_key(mission, r) for r in d.itertuples()]
    d.to_csv(out / "scored.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"case set: {args.which}")
    by_cam = d.groupby(["day", "unit"]).agg(n=("truth", "size"), used=("used", "sum"), ref_mae=("err_reference", "mean"),
                                           baseline_mae=("err_baseline", "mean"), new_mae=("err_prediction", "mean")).reset_index()
    by_cam["baseline_norm"] = by_cam.baseline_mae / by_cam.ref_mae
    by_cam["new_norm"] = by_cam.new_mae / by_cam.ref_mae
    by_cam.to_csv(out / "by_date_camera.csv", index=False)
    print(by_cam.round(4).to_string(index=False))

    by_day = d.groupby("day").agg(n=("truth", "size"), ref=("err_reference", "mean"), baseline=("err_baseline", "mean"),
                                  new=("err_prediction", "mean"))
    by_day["baseline_norm"] = by_day.baseline / by_day.ref
    by_day["new_norm"] = by_day.new / by_day.ref
    print()
    print(by_day.round(4).to_string())
    metric_base, metric_new = float(by_day.baseline_norm.mean()), float(by_day.new_norm.mean())

    robust = {}
    for day, g in d.groupby("day"):
        w = g.groupby("window").agg(base=("err_baseline", "sum"), new=("err_prediction", "sum"), n=("truth", "size"))
        gain_all = (w.base.sum() - w.new.sum()) / len(g)
        loo = [((w.base.sum() - w.base[k]) - (w.new.sum() - w.new[k])) / (len(g) - w.n[k]) for k in w.index if len(g) > w.n[k]]
        robust[day] = dict(windows=int(len(w)), wins=int((w.new < w.base - 1e-9).sum()), losses=int((w.new > w.base + 1e-9).sum()),
                           mae_gain=float(gain_all), gain_leave_one_window_out=[float(min(loo)), float(max(loo))] if loo else None)
        print(f"{day}: windows {robust[day]['windows']}, wins {robust[day]['wins']}, losses {robust[day]['losses']}, "
              f"MAE gain {gain_all:+.4f} people; with each window left out {robust[day]['gain_leave_one_window_out']}")

    result = dict(label=label, case_set=args.which, mission=mission, metric_baseline=metric_base, metric_new=metric_new,
                  by_day={k: dict(baseline=float(v.baseline_norm), new=float(v.new_norm)) for k, v in by_day.iterrows()},
                  cases=int(len(d)), cases_changed=int(d.used.sum()), leak_checks=checks, leak_check_failures=moved,
                  robustness=robust, runtime_seconds=round(time.perf_counter() - started, 1))
    print(f"\n{mission} [{args.which}]: team metric {metric_base:.4f} (accepted) -> {metric_new:.4f} (this track); "
          f"cases changed {int(d.used.sum())}/{len(d)}; leak checks {checks}, failures {moved}")
    if args.which == "fixed":
        task_scores = pd.read_csv(LOCAL / "scores_by_date_task.csv")
        total = task_scores.pivot(index="kind", columns="day", values="normalized_mae")
        total_base = float(total.mean(axis=0).mean())
        total.loc[mission] = by_day.new_norm
        total_new = float(total.mean(axis=0).mean())
        result.update(local_total_baseline=total_base, local_total_with_this_mission=total_new)
        print(f"local total if only this mission changes: {total_base:.6f} -> {total_new:.6f}")
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if moved:
        print("LEAK CHECK FAILED: predictions moved when forbidden inputs were overwritten.")
        sys.exit(2)


if __name__ == "__main__":
    main()
