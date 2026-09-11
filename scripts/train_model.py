#!/usr/bin/env python3
"""Train and honestly evaluate a gradient boosted model for taxi-out time.

Validation is built to match the composition of the real ranking set rather than to
flatter the model:

  composition  test  = January 2025 at all 10 observed airports
                     + July 2025 at EDDF, EGLL and EHAM only
               train = every other month
               This mirrors the scored set exactly. It does use later months to predict
               January, which is stated openly rather than hidden.

  forward      test  = July 2025 at EDDF, EGLL and EHAM only
               train = January to June 2025
               Strictly past to future, no information from after the test month.

The metric optimised and reported is the official raw RMSE on the untouched target.
Normal-range figures are reported alongside as diagnostics only, never as the headline.

Usage:
    .venv/bin/python scripts/train_model.py
    .venv/bin/python scripts/train_model.py --make-submission --version 2
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import baselines as B  # noqa: E402
import features as F  # noqa: E402
import target_encoding as TE  # noqa: E402

TARGET = F.TARGET
# The feature list actually used by a run. It stays exactly F.FEATURES unless
# --target-encoding is passed, so the default path reproduces v2 unchanged.
FEATURE_SET: list[str] = list(F.FEATURES)
JULY_AIRPORTS = ["EDDF", "EGLL", "EHAM"]  # the only airports present in July 2026

PARAMS = {
    "objective": "regression",       # L2, the metric we are scored on
    "metric": "rmse",
    "learning_rate": 0.06,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "num_threads": 8,
    "verbosity": -1,
    "seed": 20260901,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100
BOOTSTRAP = 1000


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def as_matrix(df: pl.DataFrame) -> np.ndarray:
    return (
        df.select([pl.col(c).fill_null(-1).cast(pl.Float32) if c in F.CATEGORICAL
                   else pl.col(c).cast(pl.Float32) for c in FEATURE_SET])
        .to_numpy()
    )


def dataset(df: pl.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(
        as_matrix(df),
        label=df[TARGET].cast(pl.Float64).to_numpy(),
        feature_name=FEATURE_SET,
        categorical_feature=F.CATEGORICAL,
        reference=reference,
        free_raw_data=True,
    )


def composition_mask(df: pl.DataFrame) -> pl.Series:
    """Rows that mirror the scored set: all of January, plus July at three airports."""
    airport_code = df["AIRPORT"]
    jan = df["month"] == 1
    jul = (df["month"] == 7) & airport_code.is_in(july_codes(df))
    return jan | jul


_JULY_CODES: list[int] | None = None


def july_codes(df: pl.DataFrame) -> list[int]:
    """Integer codes of EDDF, EGLL and EHAM under the shared categorical encoding."""
    global _JULY_CODES
    if _JULY_CODES is None:
        raise RuntimeError("july_codes not initialised")
    return _JULY_CODES


def init_airport_codes(data_dir: Path) -> dict[int, str]:
    """Recover the airport code mapping from the ranking table, which holds all 10."""
    global _JULY_CODES
    raw = (
        pl.scan_parquet(str(data_dir / "ranking.parquet"))
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("ADEP_mvt").unique().collect()["ADEP_mvt"].drop_nulls().sort().to_list()
    )
    train_names = (
        pl.scan_parquet([str(p) for p in sorted(data_dir.glob("training_*.parquet"))])
        .filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()
    )["ADEP_mvt"].drop_nulls().to_list()
    names = sorted(set(raw) | set(train_names))
    mapping = {i: n for i, n in enumerate(names)}
    _JULY_CODES = [i for i, n in mapping.items() if n in JULY_AIRPORTS]
    return mapping


def report(name: str, df: pl.DataFrame, pred: np.ndarray, names: dict[int, str]) -> list[str]:
    truth = df[TARGET].cast(pl.Float64).to_numpy()
    err2 = (pred - truth) ** 2
    out = [f"### {name}", "", f"- Rows: {len(truth):,}", f"- **Raw RMSE: {rmse(pred, truth):.2f} s**",
           f"- MAE: {float(np.mean(np.abs(pred - truth))):.2f} s, "
           f"median absolute error: {float(np.median(np.abs(pred - truth))):.2f} s"]

    normal = truth <= 3600
    out += [f"- Diagnostic, target at or below 3600 s ({normal.sum():,} rows, "
            f"{100 * normal.mean():.2f} %): RMSE {rmse(pred[normal], truth[normal]):.2f} s"]
    if (~normal).any():
        out += [f"- Diagnostic, target above 3600 s ({(~normal).sum():,} rows): "
                f"RMSE {rmse(pred[~normal], truth[~normal]):.2f} s, "
                f"{100 * err2[~normal].sum() / err2.sum():.1f} % of all squared error"]
    else:
        out += ["- Diagnostic, target above 3600 s: no rows in this split"]

    miss = df["aobt_missing"].to_numpy().astype(bool)
    if miss.any():
        out += [f"- Rows without an NM off-block time ({miss.sum():,}, {100 * miss.mean():.2f} %): "
                f"RMSE {rmse(pred[miss], truth[miss]):.2f} s; with it: {rmse(pred[~miss], truth[~miss]):.2f} s"]

    frame = df.select(["month", "AIRPORT"]).with_columns(
        pl.Series("se", err2), pl.Series("pred", pred)
    )
    out += ["", "| month | rows | RMSE s |", "|---|---:|---:|"]
    for r in frame.group_by("month").agg(pl.len().alias("n"), pl.col("se").mean().alias("mse")).sort("month").iter_rows(named=True):
        out.append(f"| {int(r['month']):02d} | {r['n']:,} | {np.sqrt(r['mse']):.2f} |")

    out += ["", "| airport | month | rows | RMSE s |", "|---|---|---:|---:|"]
    agg = frame.group_by(["AIRPORT", "month"]).agg(pl.len().alias("n"), pl.col("se").mean().alias("mse"))
    for r in agg.sort([pl.col("mse").sqrt()], descending=True).iter_rows(named=True):
        out.append(f"| {names.get(int(r['AIRPORT']), r['AIRPORT'])} | {int(r['month']):02d} | {r['n']:,} | {np.sqrt(r['mse']):.2f} |")
    return out + [""]


def bootstrap_win_rate(control: np.ndarray, model: np.ndarray, truth: np.ndarray,
                       rng: np.random.Generator) -> tuple[float, float, float]:
    """How often does the model still win when the test rows are resampled?

    A handful of extreme targets carry a third of the squared error, so a single
    favourable outlier can flip a naive RMSE comparison. This resamples rows with
    replacement and reports the share of resamples the model wins, plus the 5th and
    95th percentile of the RMSE difference.
    """
    n = len(truth)
    se_c = (control - truth) ** 2
    se_m = (model - truth) ** 2
    diffs = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(se_c[idx].mean()) - np.sqrt(se_m[idx].mean())
    return float((diffs > 0).mean()), float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))


def run_split(label: str, fit: pl.DataFrame, inner: pl.DataFrame, test: pl.DataFrame,
              names: dict[int, str], save_as: Path | None = None,
              model_as: Path | None = None, rounds: int | None = None,
              target_encoding: bool = False) -> tuple[list[str], float, float, int]:
    print(f"[{label}] fit {fit.height:,} | inner validation {inner.height:,} | test {test.height:,}")
    control_source = fit
    if target_encoding:
        fit, (inner, test) = encode_split(fit, [inner, test])

    t_train = time.perf_counter()
    dtrain = dataset(fit)
    if rounds:
        # A known good tree count from an earlier run: skip early stopping entirely.
        booster = lgb.train(PARAMS, dtrain, num_boost_round=rounds)
        best = rounds
    else:
        dvalid = dataset(inner, reference=dtrain)
        booster = lgb.train(PARAMS, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        best = booster.best_iteration or MAX_ROUNDS
        del dvalid
    print(f"  training: {time.perf_counter() - t_train:.1f} s for {best} trees")
    if model_as is not None:
        model_as.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_as), num_iteration=best)

    pred = np.clip(booster.predict(as_matrix(test), num_iteration=best), 0, None)
    truth = test[TARGET].cast(pl.Float64).to_numpy()

    # The control is the unchanged v1 method, fitted on the same rows without encodings.
    gmed = float(control_source[TARGET].median())
    control = B.predict(B.fit_lookups(control_source), gmed, test).to_numpy().clip(60, 7200)

    if save_as is not None:
        # Kept so post-model experiments (for example the LIRF tail overlay) can be run
        # against the exact predictions this split produced, without retraining.
        test.select("MVT_ID_mvt").with_columns(
            pl.Series("pred", pred), pl.Series("control", control)
        ).write_parquet(save_as)

    win, lo, hi = bootstrap_win_rate(control, pred, truth, np.random.default_rng(20260901))

    selection = ("fixed from a preserved gate run" if rounds else
                 "early stopping on a held-out slice inside the training window")
    lines = [f"## Split: {label}", "", f"Trees used: {best} ({selection})", ""]
    lines += report("Control, hierarchical median (the v1 method)", test, control, names)
    lines += report("LightGBM", test, pred, names)
    lines += [f"### Is the difference real?", "",
              f"- Paired bootstrap over {BOOTSTRAP:,} resamples of the test rows: the model wins "
              f"{100 * win:.1f} % of them.",
              f"- RMSE improvement, 5th to 95th percentile: {lo:.1f} s to {hi:.1f} s.",
              "- A wide or straddling interval means the extreme targets, not the model, are "
              "driving the comparison.", ""]

    imp = sorted(zip(FEATURE_SET, booster.feature_importance("gain")), key=lambda x: -x[1])[:15]
    total = sum(booster.feature_importance("gain")) or 1.0
    lines += ["### Top features by gain", "", "| feature | share of gain |", "|---|---:|"]
    lines += [f"| `{n}` | {100 * g / total:.1f} % |" for n, g in imp] + [""]

    del dtrain
    return lines, rmse(control, truth), rmse(pred, truth), best


def encode_split(fit: pl.DataFrame, others: list[pl.DataFrame]) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Cross fit the training rows, then encode every other frame from those rows only.

    `fit` gets leave-one-month-out values, so no training row sees its own target.
    Everything else is encoded from maps built on the whole of `fit`, which is what the
    final model would have available at prediction time.
    """
    t0 = time.perf_counter()
    fit_enc = TE.cross_fit_train(fit)
    t1 = time.perf_counter()
    maps = TE.fit_maps(fit)
    out = [TE.transform(d, maps) for d in others]
    print(f"  target encoding: cross fit {t1 - t0:.1f} s, maps and transforms "
          f"{time.perf_counter() - t1:.1f} s, {len(TE.FEATURES)} features added")
    del maps
    return fit_enc, out


def check_gate(report: Path, preds: Path) -> tuple[float, float]:
    """Refuse a final fit unless an earlier run demonstrably beat the control.

    The final model is never validated on anything, so it may only be built on the
    authority of a preserved split report and the predictions that report was written from.
    """
    if not report.exists():
        raise SystemExit(f"ERROR: gate report {report} not found. Run the composition split first.")
    if not preds.exists():
        raise SystemExit(f"ERROR: gate predictions {preds} not found. They must come from the same run.")
    m = re.search(r"\|\s*composition\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|", report.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"ERROR: no composition summary row found in {report}.")
    control, model = float(m.group(1)), float(m.group(2))
    if model >= control:
        raise SystemExit(f"ERROR: gate failed. Model {model:.2f} s does not beat control {control:.2f} s.")
    return control, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features", type=Path)
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--out", default="reports", type=Path)
    ap.add_argument("--submissions", default="submissions", type=Path)
    ap.add_argument("--models-dir", default="models", type=Path,
                    help="where boosters are written. Point a test run somewhere else so it "
                         "cannot overwrite a real artefact")
    ap.add_argument("--team-name", default="jubilant-vase")
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--splits", default="composition,forward,seasonal",
                    help="comma separated subset of composition,forward,seasonal,december,"
                         "october_selection")
    ap.add_argument("--report-name", default="model_lgbm.md",
                    help="report filename, so a partial rerun never clobbers a full run")
    ap.add_argument("--learning-rate", type=float, default=0.06,
                    help="LightGBM learning rate. The default is what v2 was built with, so "
                         "the default path stays exactly reproducible")
    ap.add_argument("--target-encoding", action="store_true",
                    help="add leave-one-month-out cross fitted target encodings. Off by "
                         "default, so the default path reproduces v2 exactly")
    ap.add_argument("--weather-features", action="store_true",
                    help="add leakage-safe prior METAR features produced by attach_weather.py. "
                         "Off by default, so the default path reproduces v2 exactly")
    ap.add_argument("--arrival-taxi-features", action="store_true",
                    help="add the two strictly prior completed-arrival taxi features produced "
                         "by attach_arrival_taxi.py")
    ap.add_argument("--ground-queue-features", action="store_true",
                    help="add the active-departure queue feature produced by "
                         "attach_ground_queue.py")
    ap.add_argument("--schedule-features", action="store_true",
                    help="add the row-local scheduled minute-of-day feature")
    ap.add_argument("--drop-congestion", action="store_true",
                    help="remove the complete predeclared congestion feature block. Off by "
                         "default, so the default path reproduces v2 exactly")
    ap.add_argument("--final-only", action="store_true",
                    help="skip all validation splits and fit the final model once. Requires a "
                         "preserved gate report and predictions proving the model beat the control, "
                         "and an explicit --rounds")
    ap.add_argument("--gate-report", default="reports/model_lgbm_composition_rerun.md", type=Path,
                    help="preserved report whose composition row authorises the final fit")
    ap.add_argument("--gate-preds", default="reports/preds_composition.parquet", type=Path,
                    help="preserved composition predictions from that same run")
    ap.add_argument("--overlay", action="store_true",
                    help="apply the calibrated LIRF extreme-taxi overlay to the submission")
    ap.add_argument("--overlay-threshold", type=int, default=14400)
    ap.add_argument("--rounds", type=int, default=0,
                    help="fixed tree count from an earlier run; skips early stopping")
    ap.add_argument("--num-leaves", type=int, default=127,
                    help="LightGBM leaf count; default preserves v2/v3")
    ap.add_argument("--min-data-in-leaf", type=int, default=200,
                    help="LightGBM minimum rows per leaf; default preserves v2/v3")
    args = ap.parse_args()
    if args.num_leaves < 2:
        ap.error("--num-leaves must be at least 2")
    if args.min_data_in_leaf < 1:
        ap.error("--min-data-in-leaf must be positive")

    train_path = args.features / "train_departures.parquet"
    if not train_path.exists():
        print(f"ERROR: {train_path} missing. Run scripts/build_features.py first.", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    names = init_airport_codes(args.data_dir)
    PARAMS["learning_rate"] = args.learning_rate
    PARAMS["num_leaves"] = args.num_leaves
    PARAMS["min_data_in_leaf"] = args.min_data_in_leaf
    full = pl.read_parquet(train_path)
    print(f"Training departures: {full.height:,}")
    print(f"Learning rate: {args.learning_rate}  (v2 was built at 0.06)")
    print(f"Capacity: num_leaves={args.num_leaves}, "
          f"min_data_in_leaf={args.min_data_in_leaf}")

    global FEATURE_SET
    FEATURE_SET = list(F.FEATURES)
    if args.drop_congestion:
        FEATURE_SET = [name for name in FEATURE_SET if name not in F.CONGESTION_NUMERIC]
        print(f"Congestion ablation ON: removed {len(F.CONGESTION_NUMERIC)} features")
    if args.weather_features:
        FEATURE_SET += list(F.WEATHER_NUMERIC)
        print(f"Weather features ON: {len(F.WEATHER_NUMERIC)} public-domain METAR features")
    if args.arrival_taxi_features:
        FEATURE_SET += list(F.ARRIVAL_TAXI_NUMERIC)
        print(f"Arrival taxi features ON: {len(F.ARRIVAL_TAXI_NUMERIC)} prior-only features")
    if args.ground_queue_features:
        FEATURE_SET += list(F.GROUND_QUEUE_NUMERIC)
        print(f"Ground queue features ON: {len(F.GROUND_QUEUE_NUMERIC)} state feature")
    if args.schedule_features:
        FEATURE_SET += list(F.SCHEDULE_NUMERIC)
        print(f"Schedule features ON: {len(F.SCHEDULE_NUMERIC)} row-local clock feature")
    if args.target_encoding:
        FEATURE_SET += list(TE.FEATURES)
        print(f"Target encoding ON: {len(FEATURE_SET)} features ({len(TE.FEATURES)} encodings)")
    missing_features = sorted(set(FEATURE_SET) - set(full.columns))
    if missing_features:
        print(f"ERROR: feature table is missing: {', '.join(missing_features)}", file=sys.stderr)
        return 1

    lines = ["# Learned model: LightGBM with composition matched validation", "",
             f"**Learning rate {args.learning_rate}**, "
             f"{'fixed ' + str(args.rounds) + ' trees' if args.rounds else 'early stopped'}, "
             f"{len(FEATURE_SET)} features"
             f"{' including target encodings' if args.target_encoding else ''}"
             f"{' including prior METAR weather' if args.weather_features else ''}"
             f"{' including completed-arrival taxi state' if args.arrival_taxi_features else ''}"
             f"{' including active ground queue state' if args.ground_queue_features else ''}"
             f"{' including scheduled minute of day' if args.schedule_features else ''}"
             f"{' excluding congestion' if args.drop_congestion else ''}.", "",
             f"Capacity: num_leaves={args.num_leaves}, "
             f"min_data_in_leaf={args.min_data_in_leaf}.", "",
             "Raw RMSE on the untouched target is the headline everywhere. Normal-range",
             "figures are diagnostics. The July test airports are EDDF, EGLL and EHAM, the",
             "only airports present in July 2026.", ""]

    if args.final_only:
        if not args.rounds:
            print("ERROR: --final-only requires an explicit --rounds from the gate run.", file=sys.stderr)
            return 1
        if not args.make_submission:
            print("ERROR: --final-only is only for building a submission; pass --make-submission.", file=sys.stderr)
            return 1
        control, model = check_gate(args.gate_report, args.gate_preds)
        print(f"Gate passed: {args.gate_report.name} records control {control:.2f} s "
              f"against model {model:.2f} s. Fitting the final model once on all 12 months.")
        return build_submission(args, full, rounds=args.rounds)

    day = pl.col(F.TAKEOFF).dt.day()
    wanted = {x.strip() for x in args.splits.split(",") if x.strip()}
    rounds = args.rounds or None
    results: dict[str, tuple[list[str], float, float, int]] = {}

    if "composition" in wanted:
        # The exact airport and month mix of the scored set. Neither month is in training.
        comp_test = full.filter(composition_mask(full))
        comp_pool = full.filter((pl.col("month") != 1) & (pl.col("month") != 7))
        results["composition"] = run_split(
            "composition (January all airports + July EDDF/EGLL/EHAM, neither month in training)",
            comp_pool.filter(pl.col("month") != 12), comp_pool.filter(pl.col("month") == 12),
            comp_test, names, save_as=args.out / "preds_composition.parquet",
            model_as=args.models_dir / "lgbm_composition.txt", rounds=rounds,
            target_encoding=args.target_encoding)

    if "forward" in wanted:
        # Strictly past to future, with no information from after the test month.
        fwd = full.filter(pl.col("month") <= 6)
        fwd_test = full.filter((pl.col("month") == 7) & pl.col("AIRPORT").is_in(july_codes(full)))
        results["forward"] = run_split(
            "forward (train 01-06, test July at the three airports)",
            fwd.filter(pl.col("month") != 6), fwd.filter(pl.col("month") == 6), fwd_test, names,
            save_as=args.out / "preds_forward.parquet",
            model_as=args.models_dir / "lgbm_forward.txt", rounds=rounds,
            target_encoding=args.target_encoding)

    if "seasonal" in wanted:
        # Same composition, but the model sees January and July the way the real model will.
        # Split by day of month, with day 21 dropped as a buffer.
        seas_test = full.filter(composition_mask(full) & (day >= 22))
        seas_pool = full.filter(day <= 20)
        results["seasonal"] = run_split(
            "seasonal (train days 01-20 of every month, test days 22-31 in the scored composition)",
            seas_pool.filter(day <= 18), seas_pool.filter(day > 18), seas_test, names,
            save_as=args.out / "preds_seasonal.parquet",
            model_as=args.models_dir / "lgbm_seasonal.txt", rounds=rounds,
            target_encoding=args.target_encoding)

    if "december" in wanted:
        # Strict all-airport forward check. November is retained as a one-month buffer,
        # mirroring the June buffer between fit and July test in the forward split.
        dec_test = full.filter(pl.col("month") == 12)
        dec_fit = full.filter(pl.col("month") <= 10)
        dec_inner = full.filter(pl.col("month") == 11)
        results["december"] = run_split(
            "December forward (train 01-10, November buffer, test December all airports)",
            dec_fit, dec_inner, dec_test, names,
            save_as=args.out / "preds_december.parquet",
            model_as=args.models_dir / "lgbm_december.txt", rounds=rounds,
            target_encoding=args.target_encoding)

    if "october_selection" in wanted:
        # Disjoint membership-selection split. October selects an airport set only and is
        # never reused as a confirmation gate.
        october_test = full.filter(pl.col("month") == 10)
        october_fit = full.filter(pl.col("month") <= 8)
        october_inner = full.filter(pl.col("month") == 9)
        results["october_selection"] = run_split(
            "October membership selection (train 01-08, September inner, test October)",
            october_fit, october_inner, october_test, names,
            save_as=args.out / "preds_october_selection.parquet",
            model_as=args.models_dir / "lgbm_october_selection.txt", rounds=rounds,
            target_encoding=args.target_encoding)

    if not results:
        print(f"ERROR: no valid split selected from '{args.splits}'", file=sys.stderr)
        return 1

    summary = ["## Summary", "",
               "| split | control RMSE s | LightGBM RMSE s | improvement |", "|---|---:|---:|---:|"]
    for name, (_, c, m, _) in results.items():
        summary.append(f"| {name} | {c:.2f} | {m:.2f} | {c - m:.2f} s ({100 * (c - m) / c:.1f} %) |")
    summary.append("")

    body = [line for name in results for line in results[name][0]]
    (args.out / args.report_name).write_text("\n".join(lines + summary + body), encoding="utf-8")
    print("\n".join(summary))
    print(f"Wrote {args.out / args.report_name}")

    if args.make_submission:
        if "composition" not in results:
            print("\nThe composition split was not run, so v2 cannot be justified. No submission written.",
                  file=sys.stderr)
            return 1
        _, c_comp, m_comp, _ = results["composition"]
        if m_comp >= c_comp:
            print("\nModel does not beat the control on the composition split. No submission written.")
            return 0
        return build_submission(args, full, rounds=args.rounds or max(r[3] for r in results.values()))
    return 0


def build_submission(args, full: pl.DataFrame, rounds: int) -> int:
    """Fit once on every month, predict the ranking set, overlay, write and validate."""
    rank = pl.read_parquet(args.features / "ranking_departures.parquet")
    if args.target_encoding:
        full, (rank,) = encode_split(full, [rank])
    print(f"Fitting the final model on all 12 months ({full.height:,} rows) with {rounds} trees ...")
    t0 = time.perf_counter()
    booster = lgb.train(PARAMS, dataset(full), num_boost_round=rounds)
    print(f"  training: {time.perf_counter() - t0:.1f} s")
    pred = np.clip(booster.predict(as_matrix(rank)), 60, None)

    if args.overlay:
        # A handful of LIRF departures with no NM off-block record have taxi times in the
        # tens of thousands of seconds, and no L2 model with min_data_in_leaf 200 can fit
        # them. The schedule delta reconstructs them, so the correction sits on top of the
        # model rather than inside it. The calibration uses 2025 LIRF rows only.
        import experiment_tail_overlay as overlay
        scope = [overlay.airport_codes(args.data_dir)["LIRF"]]
        coef = overlay.fit_calibration(full, scope, args.overlay_threshold)
        if coef is None:
            print("WARNING: too few LIRF trigger rows to calibrate; overlay skipped", file=sys.stderr)
        else:
            pred, touched = overlay.apply_overlay(pred, rank, scope, args.overlay_threshold, coef)
            pred = np.clip(pred, 60, None)
            print(f"Overlay: LIRF only, calibrated a={coef[0]:.1f} b={coef[1]:.4f}, "
                  f"threshold {args.overlay_threshold:,} s, {touched} of {rank.height:,} rows replaced")

    template = pl.read_parquet(args.data_dir / "submitting.parquet")
    dtype = template.schema[TARGET]
    sub = (
        template.drop(TARGET)
        .join(rank.select("MVT_ID_mvt").with_columns(pl.Series("pred", pred)), on="MVT_ID_mvt", how="left")
        .with_columns(pl.col("pred").fill_null(float(full[TARGET].median())).round(0).cast(dtype).alias(TARGET))
        .select(template.columns)
    )
    args.submissions.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    out = args.submissions / f"{args.team_name}_v{args.version}.parquet"
    model_path = args.models_dir / f"lgbm_v{args.version}.txt"
    sub.write_parquet(out)
    booster.save_model(str(model_path))
    print(f"Wrote {out} ({sub.height:,} rows) and {model_path}")

    print("\nValidating ...")
    rc = subprocess.run([sys.executable, str(Path(__file__).parent / "validate_submission.py"),
                         str(out), "--data-dir", str(args.data_dir)]).returncode
    if rc != 0:
        print("Submission failed local validation. Do not upload it.", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
