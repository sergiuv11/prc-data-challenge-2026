#!/usr/bin/env python3
"""Rewrite the official Parquet files with a portable schema.

The organisers produced the files from R. `AIRCRAFT_OPERATOR_flt` carries the
Arrow extension type `arrow.r.vctrs` (an R hashed vector), which Polars refuses to
read at all. The underlying storage is an ordinary string, so this step strips the
extension annotation and writes byte-identical values to `data/clean/`.

Nothing else is changed: no rows dropped, no values altered, no columns added.
The originals in `data/raw/` stay untouched as the reference copy.

Usage:
    .venv/bin/python scripts/normalize_parquet.py [--src data/raw] [--dst data/clean]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def plain_schema(schema: pa.Schema) -> pa.Schema:
    """Same fields and types, without extension metadata."""
    return pa.schema([pa.field(f.name, f.type, nullable=f.nullable) for f in schema])


def normalize(src: Path, dst: Path) -> tuple[int, list[str]]:
    pf = pq.ParquetFile(src)
    stripped = [f.name for f in pf.schema_arrow
                if f.metadata and b"ARROW:extension:name" in f.metadata]
    table = pq.read_table(src)
    table = pa.Table.from_arrays(list(table.columns), schema=plain_schema(table.schema))
    pq.write_table(table, dst, compression="zstd")
    return table.num_rows, stripped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw", type=Path)
    ap.add_argument("--dst", default="data/clean", type=Path)
    args = ap.parse_args()

    files = sorted(args.src.glob("*.parquet"))
    if not files:
        print(f"ERROR: no parquet files in {args.src}", file=sys.stderr)
        return 1
    args.dst.mkdir(parents=True, exist_ok=True)

    total = 0
    for f in files:
        rows, stripped = normalize(f, args.dst / f.name)
        total += rows
        note = f"  stripped extension type on: {', '.join(stripped)}" if stripped else ""
        print(f"{f.name}: {rows:,} rows{note}")
    print(f"\n{len(files)} files, {total:,} rows written to {args.dst}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
