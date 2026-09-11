#!/usr/bin/env python3
"""Blend fixed medium and heavy wake-category specialists over preserved v3 predictions."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import features as F
import train_model as T

ID = "MVT_ID_mvt"
PRED = "pred"
WAKE_COLUMN = "WK_TBL_CAT_flt"
WAKE_CATEGORIES = ("M", "H")
BLEND_WEIGHT = 0.5


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def wake_codes(data_dir: Path) -> dict[str, int]:
    """Reproduce the shared Enum encoding used by build_features.py."""
    paths = [str(path) for path in sorted(data_dir.glob("training_*.parquet"))]
    paths.append(str(data_dir / "ranking.parquet"))
    categories = (
        pl.scan_parquet(paths)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select(WAKE_COLUMN)
        .unique()
        .collect()[WAKE_COLUMN]
        .drop_nulls()
        .sort()
        .to_list()
    )
    return {category: code for code, category in enumerate(categories)}


def split_frames(full: pl.DataFrame, split: str) -> tuple[pl.DataFrame, pl.DataFrame,
                                                           pl.DataFrame | None]:
    if split == "composition":
        fit = full.filter(
            (pl.col("month") != 1) & (pl.col("month") != 7) & (pl.col("month") != 12)
        )
        inner = full.filter(pl.col("month") == 12)
        test = full.filter(T.composition_mask(full))
    elif split == "forward":
        fit = full.filter(pl.col("month") <= 5)
        inner = full.filter(pl.col("month") == 6)
        test = full.filter(
            (pl.col("month") == 7) & pl.col("AIRPORT").is_in(T.july_codes(full))
        )
    elif split == "december":
        fit = full.filter(pl.col("month") <= 10)
        inner = full.filter(pl.col("month") == 11)
        test = full.filter(pl.col("month") == 12)
    else:
        raise ValueError(f"unsupported split {split!r}")
    return fit, test, inner


def load_baseline(path: Path, test: pl.DataFrame) -> np.ndarray:
    baseline = pl.read_parquet(path)
    missing = {ID, PRED} - set(baseline.columns)
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
    parser.add_argument("--split", choices=["composition", "forward", "december"],
                        default="composition")
    parser.add_argument("--rounds", type=int, default=436)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=T.MAX_ROUNDS)
    parser.add_argument("--out", default="reports/wake_specialists", type=Path)
    parser.add_argument("--models-dir", default="models/wake_specialists", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.max_rounds < 1:
        parser.error("--rounds and --max-rounds must be positive")
    if args.split in {"composition", "forward"} and args.early_stop:
        parser.error("--early-stop is forbidden for composition and forward; use 436 trees")
    if args.split in {"composition", "forward"} and args.rounds != 436:
        parser.error("composition and forward are frozen at exactly --rounds 436")
    if args.split == "december" and not args.early_stop:
        parser.error("December requires --early-stop on November")

    train_path = args.features / "train_departures.parquet"
    if not train_path.exists():
        parser.error(f"{train_path} does not exist")

    T.FEATURE_SET = list(F.FEATURES)
    T.PARAMS["learning_rate"] = 0.06
    T.init_airport_codes(args.data_dir)
    codes = wake_codes(args.data_dir)
    missing_categories = sorted(set(WAKE_CATEGORIES) - set(codes))
    if missing_categories:
        raise ValueError(f"wake mapping is missing: {', '.join(missing_categories)}")

    full = pl.read_parquet(train_path)
    fit, test, inner = split_frames(full, args.split)
    baseline = load_baseline(args.baseline, test)
    candidate = baseline.copy()
    truth = test[F.TARGET].cast(pl.Float64).to_numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for category in WAKE_CATEGORIES:
        code = codes[category]
        local_fit = fit.filter(pl.col(WAKE_COLUMN) == code)
        local_test = test.filter(pl.col(WAKE_COLUMN) == code)
        mask = test[WAKE_COLUMN].to_numpy() == code
        if local_fit.is_empty() or local_test.is_empty():
            raise ValueError(f"wake category {category} has no fit or test rows")

        started = time.perf_counter()
        dataset = T.dataset(local_fit)
        if args.early_stop:
            assert inner is not None
            local_inner = inner.filter(pl.col(WAKE_COLUMN) == code)
            if local_inner.is_empty():
                raise ValueError(f"wake category {category} has no inner rows")
            validation = T.dataset(local_inner, reference=dataset)
            booster = lgb.train(
                T.PARAMS, dataset, num_boost_round=args.max_rounds,
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

        local_prediction = np.clip(
            booster.predict(T.as_matrix(local_test), num_iteration=trees), 0, None
        )
        blend = ((1.0 - BLEND_WEIGHT) * baseline[mask]
                 + BLEND_WEIGHT * local_prediction)
        candidate[mask] = blend
        elapsed = time.perf_counter() - started
        local_truth = truth[mask]
        rows.append({
            "category": category,
            "fit_rows": local_fit.height,
            "inner_rows": inner_rows,
            "test_rows": local_test.height,
            "trees": trees,
            "baseline_rmse": rmse(baseline[mask], local_truth),
            "specialist_rmse": rmse(local_prediction, local_truth),
            "blend_rmse": rmse(blend, local_truth),
            "seconds": elapsed,
        })
        booster.save_model(
            str(args.models_dir / f"wake_{category}_{args.split}.txt"),
            num_iteration=trees,
        )
        print(f"Wake {category}: fit {local_fit.height:,}, test {local_test.height:,}, "
              f"trained {trees} trees in {elapsed:.1f} s", flush=True)
        del dataset, booster, local_fit, local_test

    if not np.isfinite(candidate).all():
        raise ValueError("candidate contains non-finite predictions")
    prediction_path = args.out / f"preds_{args.split}.parquet"
    test.select(ID).with_columns(pl.Series(PRED, candidate)).write_parquet(prediction_path)
    if not pl.read_parquet(prediction_path, columns=[ID])[ID].equals(test[ID]):
        raise ValueError("written prediction IDs or row order changed")

    report = [f"# Wake-category specialists: {args.split}", "",
              "Fixed indivisible categories: medium and heavy.",
              f"Fixed blend: {100 * (1 - BLEND_WEIGHT):.0f} percent preserved v3 and "
              f"{100 * BLEND_WEIGHT:.0f} percent wake specialist.", "",
              ("Trees selected on the split's inner rows."
               if args.early_stop else f"Trees: {args.rounds}."),
              f"Features: {len(T.FEATURE_SET)}. Learning rate: 0.06.", "",
              "| wake | fit rows | inner rows | test rows | trees | baseline RMSE s | specialist RMSE s | blend RMSE s | train s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    report.extend(
        f"| {row['category']} | {row['fit_rows']:,} | {row['inner_rows']:,} | "
        f"{row['test_rows']:,} | {row['trees']:,} | {row['baseline_rmse']:.3f} | "
        f"{row['specialist_rmse']:.3f} | {row['blend_rmse']:.3f} | "
        f"{row['seconds']:.1f} |"
        for row in rows
    )
    report += ["", "| model | overall raw RMSE s |", "|---|---:|",
               f"| preserved v3 | {rmse(baseline, truth):.3f} |",
               f"| wake blend | {rmse(candidate, truth):.3f} |",
               f"| **change** | **{rmse(candidate, truth) - rmse(baseline, truth):+.3f}** |", ""]
    report_path = args.out / f"wake_specialists_{args.split}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Wrote {prediction_path} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
