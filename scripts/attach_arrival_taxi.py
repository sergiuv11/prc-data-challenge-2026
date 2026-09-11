#!/usr/bin/env python3
"""Attach strictly prior completed-arrival taxi state to cached departure features.

For each departure, the two features summarize arrivals at the same airport whose in-block
time is within the preceding 60 minutes. An arrival is never visible before its in-block time,
when its taxi-in duration becomes known. Training months are processed independently so their
boundaries do not use preceding-month context absent from the supplied ranking periods.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import features as F

ID = "MVT_ID_mvt"
PHASE = "PHASE_mvt"
BLOCK = "BLOCK_TIME_UTC_mvt"
MEAN = "arr_taxi_mean_60"
COUNT = "arr_taxi_n_60"
WINDOW = "60m"
RAW_COLUMNS = [
    ID, PHASE, "ADEP_mvt", "ADES_mvt", F.TAKEOFF, BLOCK, F.TARGET,
]


def build_one(path: Path) -> pl.DataFrame:
    movements = pl.read_parquet(path, columns=RAW_COLUMNS)
    arrivals = (
        movements
        .filter(
            (pl.col(PHASE) == "ARR")
            & pl.col(BLOCK).is_not_null()
            & pl.col(F.TARGET).is_not_null()
        )
        .select(
            pl.col("ADES_mvt").alias("airport"),
            pl.col(BLOCK).alias("event_time"),
            pl.col(F.TARGET).cast(pl.Float64).alias("arrival_taxi"),
            pl.lit(None, dtype=pl.Float64).alias("query_id"),
            pl.lit(False).alias("is_query"),
        )
    )
    departures = (
        movements
        .filter(pl.col(PHASE) == "DEP")
        .select(
            pl.col("ADEP_mvt").alias("airport"),
            pl.col(F.TAKEOFF).alias("event_time"),
            pl.lit(None, dtype=pl.Float64).alias("arrival_taxi"),
            pl.col(ID).cast(pl.Float64).alias("query_id"),
            pl.lit(True).alias("is_query"),
        )
    )
    events = pl.concat([arrivals, departures]).sort(
        ["airport", "event_time", "is_query"]
    )
    rolled = events.rolling(
        index_column="event_time", period=WINDOW, group_by="airport", closed="both"
    ).agg(
        pl.col("arrival_taxi").mean().alias(MEAN),
        pl.col("arrival_taxi").count().alias(COUNT),
    )
    if not rolled.select("airport", "event_time").equals(
        events.select("airport", "event_time")
    ):
        raise RuntimeError(f"rolling output order changed for {path.name}")
    result = (
        rolled
        .with_columns(
            events["query_id"],
            events["is_query"],
        )
        .filter(pl.col("is_query"))
        .select(
            pl.col("query_id").alias(ID),
            pl.col(MEAN).cast(pl.Float32),
            pl.when(pl.col(COUNT) > 0)
            .then(pl.col(COUNT).cast(pl.Float32))
            .otherwise(None)
            .alias(COUNT),
        )
    )
    if result.height != departures.height:
        raise RuntimeError(
            f"departure count changed for {path.name}: {departures.height:,} to "
            f"{result.height:,}"
        )
    if result[ID].is_null().any() or result[ID].is_duplicated().any():
        raise RuntimeError(f"null or duplicate departure ID produced for {path.name}")
    return result


def attach(base_path: Path, additions: pl.DataFrame, out_path: Path) -> dict[str, float]:
    base = pl.read_parquet(base_path)
    if any(name in base.columns for name in F.ARRIVAL_TAXI_NUMERIC):
        raise ValueError(f"{base_path} already contains arrival taxi features")
    if additions[ID].is_null().any() or additions[ID].is_duplicated().any():
        raise RuntimeError(f"arrival feature additions have null or duplicate IDs for {base_path}")
    missing = base.select(ID).join(additions.select(ID), on=ID, how="anti").height
    extra = additions.select(ID).join(base.select(ID), on=ID, how="anti").height
    if additions.height != base.height or missing or extra:
        raise RuntimeError(
            f"arrival feature ID set differs for {base_path}: base {base.height:,}, "
            f"additions {additions.height:,}, missing {missing:,}, extra {extra:,}"
        )
    joined = base.join(additions, on=ID, how="left", maintain_order="left")
    if joined.height != base.height or not joined[ID].equals(base[ID]):
        raise RuntimeError(f"ID count or order changed while attaching to {base_path}")
    minimum_count = joined[COUNT].drop_nulls().min()
    if minimum_count is None:
        raise RuntimeError(f"every arrival support count is null in {base_path}")
    if minimum_count < 1:
        raise RuntimeError(f"invalid support count in {base_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(out_path, compression="zstd")
    check = pl.read_parquet(out_path, columns=[ID])
    if not check[ID].equals(base[ID]):
        raise RuntimeError(f"written ID order changed in {out_path}")
    return {
        "rows": float(joined.height),
        "mean_coverage": 100.0 * (1.0 - joined[MEAN].null_count() / joined.height),
        "five_coverage": 100.0 * joined[COUNT].fill_null(0).ge(5).mean(),
        "mean_support": float(joined[COUNT].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--features", default="data/features", type=Path)
    parser.add_argument("--out", default="data/features_arrival_taxi", type=Path)
    args = parser.parse_args()

    training_paths = sorted(args.data_dir.glob("training_*.parquet"))
    ranking_path = args.data_dir / "ranking.parquet"
    if not training_paths or not ranking_path.exists():
        parser.error("training files or ranking.parquet are missing")

    training_parts = []
    for path in training_paths:
        part = build_one(path)
        training_parts.append(part)
        print(f"{path.name}: {part.height:,} departures", flush=True)
    training_additions = pl.concat(training_parts)
    ranking_additions = build_one(ranking_path)

    train_stats = attach(
        args.features / "train_departures.parquet",
        training_additions,
        args.out / "train_departures.parquet",
    )
    rank_stats = attach(
        args.features / "ranking_departures.parquet",
        ranking_additions,
        args.out / "ranking_departures.parquet",
    )

    lines = [
        "# Prior completed-arrival taxi features", "",
        "Only arrivals already in-block by the departure takeoff are included.",
        "Training months are isolated so boundaries do not use unavailable prior context.", "",
        "| dataset | rows | any-arrival coverage | at-least-5 coverage | mean support |",
        "|---|---:|---:|---:|---:|",
        f"| training | {int(train_stats['rows']):,} | {train_stats['mean_coverage']:.2f}% | "
        f"{train_stats['five_coverage']:.2f}% | {train_stats['mean_support']:.2f} |",
        f"| ranking | {int(rank_stats['rows']):,} | {rank_stats['mean_coverage']:.2f}% | "
        f"{rank_stats['five_coverage']:.2f}% | {rank_stats['mean_support']:.2f} |", "",
    ]
    report = args.out / "arrival_taxi_features.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out} and {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
