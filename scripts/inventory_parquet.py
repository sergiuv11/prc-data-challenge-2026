#!/usr/bin/env python3
"""Inventory competition Parquet files without loading their row data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ParquetInventory:
    path: str
    size_bytes: int
    rows: int
    row_groups: int
    columns: list[str]
    schema: str


def parquet_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, received: {root}")
        yield root
        return

    if not root.is_dir():
        raise FileNotFoundError(f"Path does not exist: {root}")

    files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .parquet files found under: {root}")
    yield from files


def inspect(path: Path) -> ParquetInventory:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    arrow_schema = parquet.schema_arrow
    return ParquetInventory(
        path=str(path.resolve()),
        size_bytes=path.stat().st_size,
        rows=metadata.num_rows,
        row_groups=metadata.num_row_groups,
        columns=list(arrow_schema.names),
        schema=str(arrow_schema),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Parquet metadata and print a dataset inventory as JSON."
    )
    parser.add_argument("path", type=Path, help="Parquet file or directory to inspect")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Parent directories are created safely.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        inventories = [inspect(path) for path in parquet_files(args.path)]
    except (OSError, ValueError) as exc:
        print(f"inventory error: {exc}", file=sys.stderr)
        return 1

    schemas = {entry.schema for entry in inventories}
    document = {
        "file_count": len(inventories),
        "total_size_bytes": sum(entry.size_bytes for entry in inventories),
        "total_rows": sum(entry.rows for entry in inventories),
        "schema_count": len(schemas),
        "files": [asdict(entry) for entry in inventories],
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
