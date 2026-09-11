#!/usr/bin/env python3
"""Attach the count of other departures actively taxiing at each flight's AOBT."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

import features as F

ID = "MVT_ID_mvt"
PHASE = "PHASE_mvt"
AOBT = "AOBT_3_flt"
FEATURE = "dep_taxiing_at_aobt"
RAW_COLUMNS = [ID, PHASE, "ADEP_mvt", F.TAKEOFF, AOBT]


def build_one(path: Path) -> pl.DataFrame:
    departures = (
        pl.read_parquet(path, columns=RAW_COLUMNS)
        .filter(pl.col(PHASE) == "DEP")
        .select(ID, pl.col("ADEP_mvt").alias("airport"), AOBT, F.TAKEOFF)
    )
    query = departures[AOBT].to_numpy()
    takeoff = departures[F.TAKEOFF].to_numpy()
    airport = departures["airport"].to_numpy()
    output = np.full(departures.height, np.nan, dtype=np.float32)

    for name in departures["airport"].drop_nulls().unique().to_list():
        group = airport == name
        group_query = query[group]
        group_takeoff = takeoff[group]
        present = ~np.isnat(group_query)
        valid_interval = present & ~np.isnat(group_takeoff) & (group_takeoff > group_query)
        starts = np.sort(group_query[valid_interval])
        ends = np.sort(group_takeoff[valid_interval])
        values = np.full(group_query.shape[0], np.nan, dtype=np.float32)
        if present.any():
            point = group_query[present]
            active = (
                np.searchsorted(starts, point, side="right")
                - np.searchsorted(ends, point, side="right")
            )
            active -= valid_interval[present].astype(active.dtype)
            if np.any(active < 0):
                raise RuntimeError(f"negative queue count produced for {name} in {path.name}")
            values[present] = active.astype(np.float32)
        output[group] = values

    result = departures.select(ID).with_columns(
        pl.Series(FEATURE, output).fill_nan(None)
    )
    if result.height != departures.height:
        raise RuntimeError(f"departure count changed for {path.name}")
    if result[ID].is_null().any() or result[ID].is_duplicated().any():
        raise RuntimeError(f"null or duplicate departure ID produced for {path.name}")
    expected_nulls = int((np.isnat(query) | pl.Series(airport).is_null().to_numpy()).sum())
    if result[FEATURE].null_count() != expected_nulls:
        raise RuntimeError(
            f"queue null count differs from missing AOBT count for {path.name}: "
            f"{result[FEATURE].null_count():,} versus {expected_nulls:,}"
        )
    return result


def attach(base_path: Path, additions: pl.DataFrame, out_path: Path) -> dict[str, float]:
    base = pl.read_parquet(base_path)
    if FEATURE in base.columns:
        raise ValueError(f"{base_path} already contains {FEATURE}")
    if additions[ID].is_null().any() or additions[ID].is_duplicated().any():
        raise RuntimeError(f"queue additions have null or duplicate IDs for {base_path}")
    missing = base.select(ID).join(additions.select(ID), on=ID, how="anti").height
    extra = additions.select(ID).join(base.select(ID), on=ID, how="anti").height
    if additions.height != base.height or missing or extra:
        raise RuntimeError(
            f"queue ID set differs for {base_path}: base {base.height:,}, additions "
            f"{additions.height:,}, missing {missing:,}, extra {extra:,}"
        )
    joined = base.join(additions, on=ID, how="left", maintain_order="left")
    if joined.height != base.height or not joined[ID].equals(base[ID]):
        raise RuntimeError(f"ID count or order changed while attaching to {base_path}")
    minimum = joined[FEATURE].drop_nulls().min()
    if minimum is None or minimum < 0:
        raise RuntimeError(f"invalid queue counts in {base_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined.write_parquet(out_path, compression="zstd")
    check = pl.read_parquet(out_path, columns=[ID])
    if not check[ID].equals(base[ID]):
        raise RuntimeError(f"written ID order changed in {out_path}")
    return {
        "rows": float(joined.height),
        "coverage": 100.0 * (1.0 - joined[FEATURE].null_count() / joined.height),
        "mean": float(joined[FEATURE].mean()),
        "median": float(joined[FEATURE].median()),
        "p95": float(joined[FEATURE].quantile(0.95)),
        "maximum": float(joined[FEATURE].max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--features", default="data/features", type=Path)
    parser.add_argument("--out", default="data/features_ground_queue", type=Path)
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
        "# Active departure queue feature", "",
        "Count of other valid departure taxi intervals active at each flight's AOBT.",
        "Intervals are start-inclusive, takeoff-exclusive and processed per supplied period.", "",
        "| dataset | rows | coverage | mean | median | p95 | maximum |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| training | {int(train_stats['rows']):,} | {train_stats['coverage']:.2f}% | "
        f"{train_stats['mean']:.2f} | {train_stats['median']:.0f} | "
        f"{train_stats['p95']:.0f} | {train_stats['maximum']:.0f} |",
        f"| ranking | {int(rank_stats['rows']):,} | {rank_stats['coverage']:.2f}% | "
        f"{rank_stats['mean']:.2f} | {rank_stats['median']:.0f} | "
        f"{rank_stats['p95']:.0f} | {rank_stats['maximum']:.0f} |", "",
    ]
    report = args.out / "ground_queue_features.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out} and {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
