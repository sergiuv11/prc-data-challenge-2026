#!/usr/bin/env python3
"""Build a fixed-weight blend after validating two prediction files exactly.

The default is the predeclared 50/50 LightGBM and CatBoost diversity test. This script does
not inspect targets or optimise a weight, so running it on the composition predictions cannot
quietly tune the blend against the held-out answers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


ID = "MVT_ID_mvt"
PRED = "pred"


def load(path: Path, label: str) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = {ID, PRED} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")
    frame = frame.select([ID, PRED])
    if frame[ID].is_null().any() or frame[ID].is_duplicated().any():
        raise ValueError(f"{label} contains null or duplicate movement IDs")
    pred = frame[PRED].cast(pl.Float64).to_numpy()
    if not np.isfinite(pred).all():
        raise ValueError(f"{label} contains non-finite predictions")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--left-weight", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.left_weight <= 1.0:
        parser.error("--left-weight must be between 0 and 1")

    try:
        left = load(args.left, "left predictions")
        right = load(args.right, "right predictions")
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if left.height != right.height or not left[ID].equals(right[ID]):
        print("ERROR: prediction files do not have identical movement IDs and order",
              file=sys.stderr)
        return 1

    weight = args.left_weight
    blended = weight * left[PRED].cast(pl.Float64).to_numpy() \
        + (1.0 - weight) * right[PRED].cast(pl.Float64).to_numpy()
    output = left.select(ID).with_columns(pl.Series(PRED, blended))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(args.out)
    print(f"Wrote {args.out}: {output.height:,} rows, left weight {weight:.3f}, "
          f"right weight {1.0 - weight:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
