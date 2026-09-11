#!/usr/bin/env python3
"""Step 12 of the plan: refuse to upload anything the ranking script would reject.

The organisers' ranking script errors if an MVT_ID_mvt does not match, if rows are
missing, or if extra rows are present. This checks all of that locally, plus the
things that silently ruin a score: nulls, infinities, negatives and absurd values.

Usage:
    .venv/bin/python scripts/validate_submission.py submissions/jubilant-vase_v1.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

TARGET = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"
EXPECTED_ROWS = 215_876  # observed 'usedPairs' on the public 2026 leaderboard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    args = ap.parse_args()

    template_path = next(iter(sorted(args.data_dir.rglob("submitting.parquet"))), None)
    if template_path is None:
        print(f"ERROR: submitting.parquet not found under {args.data_dir}", file=sys.stderr)
        return 2
    if not args.submission.exists():
        print(f"ERROR: {args.submission} does not exist", file=sys.stderr)
        return 2

    template = pl.read_parquet(str(template_path))
    sub = pl.read_parquet(str(args.submission))
    errors: list[str] = []
    warnings: list[str] = []

    name = args.submission.name
    if not name.endswith(".parquet"):
        errors.append(f"file name '{name}' must end in .parquet")
    stem = name[: -len(".parquet")]
    if "_v" not in stem or not stem.rsplit("_v", 1)[1].isdigit():
        errors.append(f"file name '{name}' must follow <team-name>_v<integer>.parquet")

    for c in (ID, TARGET):
        if c not in sub.columns:
            errors.append(f"missing required column '{c}'")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1

    if sub.height != template.height:
        errors.append(f"row count {sub.height:,} != template {template.height:,}")
    if template.height != EXPECTED_ROWS:
        warnings.append(f"template has {template.height:,} rows, leaderboard reported {EXPECTED_ROWS:,} scored pairs")

    dups = sub.height - sub[ID].n_unique()
    if dups:
        errors.append(f"{dups:,} duplicate {ID} values")

    sub_ids = set(sub[ID].to_list())
    tpl_ids = set(template[ID].to_list())
    missing = tpl_ids - sub_ids
    extra = sub_ids - tpl_ids
    if missing:
        errors.append(f"{len(missing):,} required {ID} values are missing (e.g. {list(missing)[:3]})")
    if extra:
        errors.append(f"{len(extra):,} unexpected {ID} values present (e.g. {list(extra)[:3]})")

    vals = sub[TARGET]
    if vals.dtype not in (pl.Float32, pl.Float64, pl.Int16, pl.Int32, pl.Int64, pl.UInt16, pl.UInt32, pl.UInt64):
        errors.append(f"{TARGET} has non-numeric dtype {vals.dtype}")
    else:
        f = vals.cast(pl.Float64)
        n_null = int(f.is_null().sum())
        n_nan = int(f.is_nan().sum())
        n_inf = int(f.is_infinite().sum())
        n_neg = int((f < 0).sum())
        n_big = int((f > args.max_seconds).sum())
        if n_null:
            errors.append(f"{n_null:,} null predictions")
        if n_nan:
            errors.append(f"{n_nan:,} NaN predictions")
        if n_inf:
            errors.append(f"{n_inf:,} infinite predictions")
        if n_neg:
            errors.append(f"{n_neg:,} negative predictions")
        if n_big:
            warnings.append(f"{n_big:,} predictions above {args.max_seconds:g} s")
        if f.n_unique() == 1:
            warnings.append("every prediction is identical (constant submission)")
        print(f"Predictions: n={f.len():,} min={f.min():.1f} p50={f.median():.1f} "
              f"mean={f.mean():.1f} p99={f.quantile(0.99):.1f} max={f.max():.1f} (seconds)")

    if sub[ID].dtype != template[ID].dtype:
        warnings.append(f"{ID} dtype {sub[ID].dtype} differs from template {template[ID].dtype}")

    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print(f"\n{args.submission} is NOT safe to upload.")
        return 1
    print(f"\nPASS: {args.submission} matches the template exactly and is safe to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
