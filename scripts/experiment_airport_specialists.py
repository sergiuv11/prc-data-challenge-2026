#!/usr/bin/env python3
"""Blend fixed airport-only LightGBM specialists with preserved baseline predictions.

The default experiment remains EDDF, EGLL and EHAM. An explicit airport list supports later
predeclared branches without duplicating the validated implementation. The blend stays 50/50.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import features as F
import train_model as T

DEFAULT_AIRPORTS = ("EDDF", "EGLL", "EHAM")
ID = "MVT_ID_mvt"
PRED = "pred"
BLEND_WEIGHT = 0.5


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def split_frames(full: pl.DataFrame, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    day = pl.col(F.TAKEOFF).dt.day()
    if split == "composition":
        fit = full.filter(
            (pl.col("month") != 1) & (pl.col("month") != 7) & (pl.col("month") != 12)
        )
        test = full.filter(T.composition_mask(full))
    elif split == "forward":
        fit = full.filter(pl.col("month") <= 5)
        test = full.filter(
            (pl.col("month") == 7) & pl.col("AIRPORT").is_in(T.july_codes(full))
        )
    elif split == "seasonal":
        fit = full.filter(day <= 18)
        test = full.filter(T.composition_mask(full) & (day >= 22))
    elif split == "december":
        # Strict all-airport forward check: January through October train, November is a
        # one-month buffer matching the June buffer in the July-forward definition.
        fit = full.filter(pl.col("month") <= 10)
        test = full.filter(pl.col("month") == 12)
    elif split == "october_selection":
        fit = full.filter(pl.col("month") <= 8)
        test = full.filter(pl.col("month") == 10)
    else:
        raise ValueError(f"unsupported split {split!r}")
    return fit, test


def inner_frame(full: pl.DataFrame, split: str, airport_code: int) -> pl.DataFrame:
    """Held-out rows used only to select one specialist's tree count."""
    airport = pl.col("AIRPORT") == airport_code
    if split == "composition":
        mask = (pl.col("month") == 12) & airport
    elif split == "forward":
        mask = (pl.col("month") == 6) & airport
    elif split == "seasonal":
        day = pl.col(F.TAKEOFF).dt.day()
        mask = day.is_between(19, 20) & airport
    elif split == "december":
        mask = (pl.col("month") == 11) & airport
    elif split == "october_selection":
        mask = (pl.col("month") == 9) & airport
    else:
        raise ValueError(f"unsupported split {split!r}")
    return full.filter(mask)


def load_baseline(path: Path, test: pl.DataFrame) -> np.ndarray:
    baseline = pl.read_parquet(path)
    required = {ID, PRED}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"baseline is missing columns: {', '.join(sorted(missing))}")
    baseline = baseline.select(ID, PRED)
    if baseline[ID].is_null().any() or baseline[ID].is_duplicated().any():
        raise ValueError("baseline has null or duplicate movement IDs")
    if baseline.height != test.height or not baseline[ID].equals(test[ID]):
        raise ValueError("baseline IDs or row order do not exactly match the selected split")
    prediction = baseline[PRED].cast(pl.Float64).to_numpy()
    if not np.isfinite(prediction).all():
        raise ValueError("baseline contains non-finite predictions")
    return prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features", type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--split", choices=["composition", "forward", "seasonal", "december",
                                                   "october_selection"],
                        default="composition")
    parser.add_argument("--airports", default=",".join(DEFAULT_AIRPORTS),
                        help="comma-separated predeclared specialist airports")
    parser.add_argument("--rounds", type=int, default=436)
    parser.add_argument("--early-stop", "--early-stop-on-november", dest="early_stop",
                        action="store_true",
                        help="select each specialist's tree count on the split's held-out "
                             "inner rows")
    parser.add_argument("--max-rounds", type=int, default=T.MAX_ROUNDS,
                        help="maximum trees when --early-stop is active")
    parser.add_argument("--drop-congestion", action="store_true",
                        help="remove the complete predeclared congestion feature block")
    parser.add_argument("--arrival-taxi-features", action="store_true",
                        help="add the two strictly prior completed-arrival taxi features")
    parser.add_argument("--ground-queue-features", action="store_true",
                        help="add the active-departure queue feature")
    parser.add_argument("--schedule-features", action="store_true",
                        help="add the row-local scheduled minute-of-day feature")
    parser.add_argument("--out", default="reports/airport_a", type=Path)
    parser.add_argument("--models-dir", default="models/airport_a", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.max_rounds < 1:
        parser.error("--max-rounds must be positive")
    airports = tuple(dict.fromkeys(
        value.strip().upper() for value in args.airports.split(",") if value.strip()
    ))
    if not airports:
        parser.error("--airports must contain at least one ICAO code")

    train_path = args.features / "train_departures.parquet"
    if not train_path.exists():
        parser.error(f"{train_path} does not exist")

    T.FEATURE_SET = list(F.FEATURES)
    if args.drop_congestion:
        T.FEATURE_SET = [name for name in T.FEATURE_SET
                         if name not in F.CONGESTION_NUMERIC]
    if args.arrival_taxi_features:
        T.FEATURE_SET += list(F.ARRIVAL_TAXI_NUMERIC)
    if args.ground_queue_features:
        T.FEATURE_SET += list(F.GROUND_QUEUE_NUMERIC)
    if args.schedule_features:
        T.FEATURE_SET += list(F.SCHEDULE_NUMERIC)
    T.PARAMS["learning_rate"] = 0.06
    names = T.init_airport_codes(args.data_dir)
    codes = {name: code for code, name in names.items() if name in airports}
    missing_airports = sorted(set(airports) - set(codes))
    if missing_airports:
        raise ValueError(f"airport mapping is missing: {', '.join(missing_airports)}")

    full = pl.read_parquet(train_path)
    missing_features = sorted(set(T.FEATURE_SET) - set(full.columns))
    if missing_features:
        raise ValueError(f"feature table is missing: {', '.join(missing_features)}")
    fit, test = split_frames(full, args.split)
    baseline = load_baseline(args.baseline, test)
    candidate = baseline.copy()
    truth = test[F.TARGET].cast(pl.Float64).to_numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for airport in airports:
        code = codes[airport]
        local_fit = fit.filter(pl.col("AIRPORT") == code)
        mask = test["AIRPORT"].to_numpy() == code
        local_test = test.filter(pl.col("AIRPORT") == code)
        if local_fit.is_empty() or local_test.is_empty():
            raise ValueError(f"{airport} has no fit or test rows for {args.split}")

        started = time.perf_counter()
        dataset = T.dataset(local_fit)
        if args.early_stop:
            local_inner = inner_frame(full, args.split, code)
            if local_inner.is_empty():
                raise ValueError(f"{airport} has no early-stopping rows for {args.split}")
            validation = T.dataset(local_inner, reference=dataset)
            booster = lgb.train(
                T.PARAMS,
                dataset,
                num_boost_round=args.max_rounds,
                valid_sets=[validation],
                callbacks=[lgb.early_stopping(T.EARLY_STOP, verbose=False)],
            )
            trees = booster.best_iteration or args.max_rounds
            inner_rows = local_inner.height
            del validation, local_inner
        else:
            booster = lgb.train(T.PARAMS, dataset, num_boost_round=args.rounds)
            trees = args.rounds
            inner_rows = 0
        elapsed = time.perf_counter() - started
        local_prediction = np.clip(
            booster.predict(T.as_matrix(local_test), num_iteration=trees), 0, None
        )
        blend = ((1.0 - BLEND_WEIGHT) * baseline[mask]
                 + BLEND_WEIGHT * local_prediction)
        candidate[mask] = blend
        local_truth = truth[mask]
        rows.append({
            "airport": airport,
            "fit_rows": local_fit.height,
            "inner_rows": inner_rows,
            "test_rows": local_test.height,
            "baseline_rmse": rmse(baseline[mask], local_truth),
            "specialist_rmse": rmse(local_prediction, local_truth),
            "blend_rmse": rmse(blend, local_truth),
            "seconds": elapsed,
        })
        booster.save_model(
            str(args.models_dir / f"{airport}_{args.split}.txt"), num_iteration=trees
        )
        print(f"{airport}: fit {local_fit.height:,}, test {local_test.height:,}, "
              f"trained {trees} trees in {elapsed:.1f} s", flush=True)
        rows[-1]["trees"] = trees
        del dataset, booster, local_fit, local_test

    if not np.isfinite(candidate).all():
        raise ValueError("candidate contains non-finite predictions")
    prediction_path = args.out / f"preds_{args.split}.parquet"
    test.select(ID).with_columns(pl.Series(PRED, candidate)).write_parquet(prediction_path)
    if not pl.read_parquet(prediction_path, columns=[ID])[ID].equals(test[ID]):
        raise ValueError("written prediction IDs or row order changed")

    overall_baseline = rmse(baseline, truth)
    overall_candidate = rmse(candidate, truth)
    report = [
        f"# Airport specialists: {args.split}", "",
        f"Fixed blend: {100 * (1 - BLEND_WEIGHT):.0f} percent preserved baseline and "
        f"{100 * BLEND_WEIGHT:.0f} percent airport specialist.", "",
        ("Trees selected independently per airport on the split's held-out inner rows."
         if args.early_stop else f"Trees: {args.rounds}."),
        f"Features: {len(T.FEATURE_SET)}. Learning rate: 0.06.", "",
        "| airport | fit rows | inner rows | test rows | trees | baseline RMSE s | specialist RMSE s | blend RMSE s | train s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['airport']} | {row['fit_rows']:,} | {row['inner_rows']:,} | "
        f"{row['test_rows']:,} | {row['trees']:,} | "
        f"{row['baseline_rmse']:.3f} | {row['specialist_rmse']:.3f} | "
        f"{row['blend_rmse']:.3f} | {row['seconds']:.1f} |"
        for row in rows
    )
    report += ["", "| model | overall raw RMSE s |", "|---|---:|",
               f"| preserved baseline | {overall_baseline:.3f} |",
               f"| specialist blend | {overall_candidate:.3f} |",
               f"| **change** | **{overall_candidate - overall_baseline:+.3f}** |", ""]
    report_path = args.out / f"airport_specialists_{args.split}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Wrote {prediction_path} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
