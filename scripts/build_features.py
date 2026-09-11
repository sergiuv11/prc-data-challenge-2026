#!/usr/bin/env python3
"""Build and cache the modelling tables.

Congestion windows are computed within each dataset. The twelve training months are
processed together so month boundaries are correct, and the ranking file is processed
as one piece so January and July 2026 departures see their own real neighbours.

Categorical columns are encoded once over the union of both datasets, so a stand or
operator has the same code everywhere.

Usage:
    .venv/bin/python scripts/build_features.py [--data-dir data/clean] [--out data/features]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402


def encode(train: pl.DataFrame, rank: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """Map every categorical column to a stable integer code shared by both tables."""
    sizes: dict[str, int] = {}
    for col in F.CATEGORICAL:
        cats = (
            pl.concat([train.select(col), rank.select(col)])
            .drop_nulls().unique().sort(col)[col].to_list()
        )
        enum = pl.Enum(cats)
        sizes[col] = len(cats)
        train = train.with_columns(pl.col(col).cast(enum).to_physical().cast(pl.Int32).alias(col))
        rank = rank.with_columns(pl.col(col).cast(enum).to_physical().cast(pl.Int32).alias(col))
    return train, rank, sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--out", default="data/features", type=Path)
    args = ap.parse_args()

    training = sorted(args.data_dir.glob("training_*.parquet"))
    ranking = args.data_dir / "ranking.parquet"
    if not training or not ranking.exists():
        print(f"ERROR: expected training_*.parquet and ranking.parquet in {args.data_dir}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    print("Building training features ...")
    train = F.build(F.load_movements([str(p) for p in training]))
    print(f"  {train.height:,} departures")

    print("Building ranking features ...")
    rank = F.build(F.load_movements([str(ranking)]))
    print(f"  {rank.height:,} departures")

    train, rank, sizes = encode(train, rank)
    train.write_parquet(args.out / "train_departures.parquet", compression="zstd")
    rank.write_parquet(args.out / "ranking_departures.parquet", compression="zstd")
    (args.out / "category_sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out}/train_departures.parquet and ranking_departures.parquet")
    print("Category sizes: " + ", ".join(f"{k}={v}" for k, v in sizes.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
