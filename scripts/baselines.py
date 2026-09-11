#!/usr/bin/env python3
"""Steps 4 and 5 of the plan: honest reference predictions with time-aware validation.

Baselines are hierarchical medians with backoff: use the most specific group that
has enough observations in the training window, otherwise fall back to a broader
group, and finally to the global median. If a learned model cannot beat these, the
model is not earning its keep.

Validation never mixes the future into the past. Each fold trains on earlier months
and scores a later month, except the explicitly labelled seasonal January fold,
which exists because the ranking set contains a January and no earlier January is
available in a single-year training set.

Usage:
    .venv/bin/python scripts/baselines.py                       # validate only
    .venv/bin/python scripts/baselines.py --make-submission     # + write a submission
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

TARGET = "TAXITIME_SEC_mvt"
TAKEOFF = "MVT_TIME_UTC_mvt"

# Most specific first. Each level needs MIN_COUNT observations to be trusted.
LEVELS: list[list[str]] = [
    ["AIRPORT", "STAND_mvt", "RUNWAY_mvt", "hour"],
    ["AIRPORT", "STAND_mvt", "RUNWAY_mvt"],
    ["AIRPORT", "RUNWAY_mvt", "hour", "weekday"],
    ["AIRPORT", "RUNWAY_mvt", "hour"],
    ["AIRPORT", "RUNWAY_mvt"],
    ["AIRPORT", "hour"],
    ["AIRPORT"],
]
MIN_COUNT = 30

FOLDS = [
    ("train 01-05 -> test 06", [1, 2, 3, 4, 5], [6]),
    ("train 01-06 -> test 07", [1, 2, 3, 4, 5, 6], [7]),
    ("train 01-10 -> test 11", list(range(1, 11)), [11]),
    ("train 01-11 -> test 12", list(range(1, 12)), [12]),
    ("seasonal: train 02-12 -> test 01", list(range(2, 13)), [1]),
]


def prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Departure rows with the calendar features every baseline needs."""
    return (
        lf.filter(pl.col("PHASE_mvt") == "DEP")
        .with_columns(
            pl.col("ADEP_mvt").alias("AIRPORT"),
            pl.col(TAKEOFF).dt.hour().alias("hour"),
            pl.col(TAKEOFF).dt.weekday().alias("weekday"),
            pl.col(TAKEOFF).dt.month().alias("month"),
        )
    )


def fit_lookups(train: pl.DataFrame) -> list[tuple[list[str], pl.DataFrame]]:
    tables = []
    for i, keys in enumerate(LEVELS):
        t = (
            train.group_by(keys)
            .agg(pl.col(TARGET).median().alias(f"med_{i}"), pl.len().alias(f"n_{i}"))
            .filter(pl.col(f"n_{i}") >= MIN_COUNT)
            .drop(f"n_{i}")
        )
        tables.append((keys, t))
    return tables


def predict(tables: list[tuple[list[str], pl.DataFrame]], global_median: float, df: pl.DataFrame) -> pl.Series:
    out = df.select(pl.lit(None, dtype=pl.Float64).alias("pred"))
    joined = df
    for i, (keys, t) in enumerate(tables):
        joined = joined.join(t, on=keys, how="left")
    cols = [f"med_{i}" for i in range(len(tables)) if f"med_{i}" in joined.columns]
    pred = joined.select(pl.coalesce([pl.col(c) for c in cols] + [pl.lit(global_median)]).alias("pred"))["pred"]
    return pred.cast(pl.Float64)


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--out", default="reports", type=Path)
    ap.add_argument("--submissions", default="submissions", type=Path)
    ap.add_argument("--team-name", default="jubilant-vase")
    ap.add_argument("--version", type=int, default=1, help="submission version number")
    ap.add_argument("--make-submission", action="store_true")
    args = ap.parse_args()

    training = sorted(args.data_dir.rglob("training_*.parquet"))
    if not training:
        print(f"ERROR: no training_*.parquet found under {args.data_dir}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    dep = prepare(pl.scan_parquet([str(p) for p in training])).filter(pl.col(TARGET).is_not_null()).collect()
    print(f"Training departures with a target: {dep.height:,}")

    lines = ["# Baselines (hierarchical medians, time-aware folds)", "",
             f"Backoff order: {' -> '.join('+'.join(k) for k in LEVELS)} -> global median",
             f"A group is used only with at least {MIN_COUNT} observations.", "",
             "| fold | test rows | global median RMSE | airport median RMSE | hierarchical RMSE |",
             "|---|---:|---:|---:|---:|"]

    for label, train_months, test_months in FOLDS:
        tr = dep.filter(pl.col("month").is_in(train_months))
        te = dep.filter(pl.col("month").is_in(test_months))
        if tr.is_empty() or te.is_empty():
            continue
        truth = te[TARGET].cast(pl.Float64).to_numpy()
        gmed = float(tr[TARGET].median())

        ap_med = tr.group_by("AIRPORT").agg(pl.col(TARGET).median().alias("med_ap"))
        ap_pred = (
            te.join(ap_med, on="AIRPORT", how="left")
            .select(pl.coalesce([pl.col("med_ap"), pl.lit(gmed)]).alias("p"))["p"]
            .cast(pl.Float64).to_numpy()
        )
        hier = predict(fit_lookups(tr), gmed, te).to_numpy()

        lines.append(
            f"| {label} | {te.height:,} | {rmse(np.full_like(truth, gmed), truth):.1f} | "
            f"{rmse(ap_pred, truth):.1f} | {rmse(hier, truth):.1f} |"
        )
        print(lines[-1])

    lines += ["", "RMSE is in seconds. The public leaderboard is the same metric on the hidden 2026 months.", ""]
    (args.out / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {args.out / 'baselines.md'}")

    if args.make_submission:
        ranking_path = next(iter(sorted(args.data_dir.rglob("ranking.parquet"))), None)
        template_path = next(iter(sorted(args.data_dir.rglob("submitting.parquet"))), None)
        if ranking_path is None or template_path is None:
            print("ERROR: ranking.parquet / submitting.parquet not found, cannot build a submission.", file=sys.stderr)
            return 1
        gmed = float(dep[TARGET].median())
        tables = fit_lookups(dep)
        rank_dep = prepare(pl.scan_parquet(str(ranking_path))).collect()
        preds = predict(tables, gmed, rank_dep)
        pred_df = rank_dep.select("MVT_ID_mvt").with_columns(preds.alias("pred"))

        template = pl.read_parquet(str(template_path))
        target_dtype = template.schema[TARGET]
        sub = (
            template.drop(TARGET)
            .join(pred_df, on="MVT_ID_mvt", how="left")
            .with_columns(pl.col("pred").fill_null(gmed).round(0).clip(60, 7200).cast(target_dtype).alias(TARGET))
            .select(template.columns)
        )
        args.submissions.mkdir(parents=True, exist_ok=True)
        out = args.submissions / f"{args.team_name}_v{args.version}.parquet"
        sub.write_parquet(out)
        print(f"Wrote {out} ({sub.height:,} rows). Validate it before uploading:")
        print(f"  .venv/bin/python scripts/validate_submission.py {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
