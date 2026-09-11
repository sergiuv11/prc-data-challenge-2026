#!/usr/bin/env python3
"""Residual audit of the departures with no Network Manager off-block time.

Cheap and reproducible: it reads the preserved composition predictions and the cached
feature table, and trains nothing. The question it answers is where the error actually
lives inside the 1.25 % of rows that carry most of the squared error, and which of the
signals that might fix them are available on the 2026 ranking set.

Usage:
    .venv/bin/python scripts/audit_missing_aobt.py [--preds reports/preds_composition.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402

TARGET = F.TARGET
JULY = ["EDDF", "EGLL", "EHAM"]


def airport_names(data_dir: Path) -> dict[int, str]:
    names = (
        pl.scan_parquet([str(p) for p in sorted(data_dir.glob("training_*.parquet"))]
                        + [str(data_dir / "ranking.parquet")])
        .filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()
    )["ADEP_mvt"].drop_nulls().sort().to_list()
    return {i: n for i, n in enumerate(names)}


def band(col: str = TARGET) -> pl.Expr:
    return (
        pl.when(pl.col(col) <= 1800).then(pl.lit("1. <=1800"))
        .when(pl.col(col) <= 3600).then(pl.lit("2. 1800-3600"))
        .when(pl.col(col) <= 7200).then(pl.lit("3. 3600-7200"))
        .when(pl.col(col) <= 20000).then(pl.lit("4. 7200-20k"))
        .otherwise(pl.lit("5. >20k")).alias("band")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="reports/preds_composition.parquet", type=Path)
    ap.add_argument("--features", default="data/features/train_departures.parquet", type=Path)
    ap.add_argument("--ranking", default="data/features/ranking_departures.parquet", type=Path)
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    args = ap.parse_args()

    names = airport_names(args.data_dir)
    preds = pl.read_parquet(args.preds)
    cols = ["MVT_ID_mvt", TARGET, "AIRPORT", "month", "aobt_missing", "takeoff_minus_schedule",
            "RUNWAY_mvt", "STAND_mvt", "dep_prev_30", "arr_prev_30", "hour", "AIRCRAFT_TYPE_mvt"]
    df = pl.read_parquet(args.features, columns=cols).join(preds, on="MVT_ID_mvt", how="inner")
    df = df.with_columns(
        (pl.col("pred") - pl.col(TARGET)).alias("err"),
        ((pl.col("pred") - pl.col(TARGET)) ** 2).alias("se"),
        band(),
        pl.col("AIRPORT").replace_strict(names, default="?").alias("ap"),
    )
    total_se = float(df["se"].sum())
    n = df.height
    print(f"Composition test rows: {n:,}   overall RMSE {np.sqrt(total_se / n):.2f} s\n")

    # 1. How concentrated is the error?
    print("=== 1. Error concentration ===")
    for label, mask in [("no NM off-block time", pl.col("aobt_missing") == 1),
                        ("has NM off-block time", pl.col("aobt_missing") == 0)]:
        d = df.filter(mask)
        se = float(d["se"].sum())
        print(f"  {label:<24} {d.height:>7,} rows ({100*d.height/n:5.2f} %)   "
              f"RMSE {np.sqrt(se/d.height):>8.1f} s   {100*se/total_se:5.1f} % of squared error")

    # 2. Inside the missing population, which airports and which target bands?
    miss = df.filter(pl.col("aobt_missing") == 1)
    miss_se = float(miss["se"].sum())
    print(f"\n=== 2. Missing-AOBT rows by airport ({miss.height:,} rows, {100*miss_se/total_se:.1f} % of all error) ===")
    t = (miss.group_by("ap").agg(pl.len().alias("rows"), pl.col("se").sum().alias("se"),
                                 pl.col(TARGET).median().alias("median_target"))
         .with_columns((pl.col("se") / pl.col("rows")).sqrt().alias("rmse"),
                       (100 * pl.col("se") / total_se).alias("pct_all_err"))
         .sort("se", descending=True))
    print(f"  {'airport':<8}{'rows':>7}{'RMSE s':>10}{'median target':>15}{'% of ALL error':>16}")
    for r in t.iter_rows(named=True):
        print(f"  {r['ap']:<8}{r['rows']:>7,}{r['rmse']:>10.0f}{r['median_target']:>15.0f}{r['pct_all_err']:>15.1f}%")

    print(f"\n=== 3. Missing-AOBT rows by target band ===")
    t = (miss.group_by("band").agg(pl.len().alias("rows"), pl.col("se").sum().alias("se"))
         .with_columns((pl.col("se") / pl.col("rows")).sqrt().alias("rmse"),
                       (100 * pl.col("se") / total_se).alias("pct_all_err")).sort("band"))
    print(f"  {'band':<16}{'rows':>7}{'RMSE s':>10}{'% of ALL error':>16}")
    for r in t.iter_rows(named=True):
        print(f"  {r['band']:<16}{r['rows']:>7,}{r['rmse']:>10.0f}{r['pct_all_err']:>15.1f}%")

    # 4. How reconstructible is the target from the schedule delta, per airport?
    print(f"\n=== 4. Is the target reconstructible from takeoff_minus_schedule? (missing-AOBT rows) ===")
    rec = miss.with_columns((pl.col("takeoff_minus_schedule") - pl.col(TARGET)).abs().alias("ae"))
    t = (rec.group_by("ap").agg(pl.len().alias("rows"), pl.col("ae").median().alias("median_ae"),
                                (pl.col("ae") <= 60).mean().mul(100).alias("within_60s"),
                                (pl.col("ae") <= 300).mean().mul(100).alias("within_300s"))
         .sort("within_60s", descending=True))
    print(f"  {'airport':<8}{'rows':>7}{'median |delta-target|':>24}{'within 60s':>13}{'within 300s':>13}")
    for r in t.iter_rows(named=True):
        print(f"  {r['ap']:<8}{r['rows']:>7,}{r['median_ae']:>24.0f}{r['within_60s']:>12.1f}%{r['within_300s']:>12.1f}%")

    # 5. Would a tail classifier have anything to work with? Ranking-safe features only.
    print(f"\n=== 5. Separability of the >3600 s tail among missing-AOBT rows (ranking-safe features) ===")
    miss = miss.with_columns((pl.col(TARGET) > 3600).alias("is_tail"))
    print(f"  tail prevalence in this population: {100*miss['is_tail'].mean():.2f} % "
          f"({miss['is_tail'].sum()} of {miss.height:,})")
    for col in ["takeoff_minus_schedule", "dep_prev_30", "arr_prev_30", "hour"]:
        a = miss.filter(pl.col("is_tail"))[col].cast(pl.Float64).drop_nulls()
        b = miss.filter(~pl.col("is_tail"))[col].cast(pl.Float64).drop_nulls()
        if a.len() == 0 or b.len() == 0:
            continue
        print(f"    {col:<24} tail median {a.median():>9.0f}   normal median {b.median():>9.0f}")

    # 6. Exposure: how many missing-AOBT rows are in the real 2026 ranking set?
    rank = pl.read_parquet(args.ranking, columns=["AIRPORT", "month", "aobt_missing", "takeoff_minus_schedule"])
    rm = rank.filter(pl.col("aobt_missing") == 1).with_columns(
        pl.col("AIRPORT").replace_strict(names, default="?").alias("ap"))
    print(f"\n=== 6. 2026 exposure: {rm.height:,} of {rank.height:,} ranking rows have no NM off-block time "
          f"({100*rm.height/rank.height:.2f} %) ===")
    t = rm.group_by(["ap", "month"]).agg(pl.len().alias("rows")).sort("rows", descending=True)
    print("  " + ", ".join(f"{r['ap']}/{r['month']:02d}:{r['rows']}" for r in t.iter_rows(named=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
