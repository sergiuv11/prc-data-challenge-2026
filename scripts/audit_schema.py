#!/usr/bin/env python3
"""Step 2 of the plan: an honest, automatic report on what the organisers gave us.

Reads the training monthly parquet files, ranking.parquet and submitting.parquet,
and writes a plain-language markdown report plus machine-readable CSVs.

Usage:
    .venv/bin/python scripts/audit_schema.py [--data-dir data/raw] [--out reports]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

TARGET = "TAXITIME_SEC_mvt"
AIRPORTS = {
    "EDDF": "Frankfurt Main", "EDDM": "Munich", "EGLL": "London Heathrow",
    "EHAM": "Amsterdam Schiphol", "LEBL": "Barcelona-El Prat", "LEMD": "Madrid-Barajas",
    "LFPG": "Paris Charles de Gaulle", "LIRF": "Rome-Fiumicino", "LTAI": "Antalya",
    "LTFM": "Istanbul", "LSZH": "Zurich",
}


def find_files(data_dir: Path) -> tuple[list[Path], Path | None, Path | None]:
    training = sorted(data_dir.rglob("training_*.parquet"))
    ranking = next(iter(sorted(data_dir.rglob("ranking.parquet"))), None)
    submitting = next(iter(sorted(data_dir.rglob("submitting.parquet"))), None)
    return training, ranking, submitting


def reporting_airport(df: pl.LazyFrame) -> pl.LazyFrame:
    """The airport whose movement this row is: ADEP for departures, ADES for arrivals."""
    return df.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .alias("AIRPORT")
    )


def null_table(lf: pl.LazyFrame, schema: dict[str, pl.DataType], n_rows: int) -> pl.DataFrame:
    nulls = lf.select([pl.col(c).null_count().alias(c) for c in schema]).collect()
    return pl.DataFrame(
        {
            "column": list(schema),
            "dtype": [str(schema[c]) for c in schema],
            "nulls": [int(nulls[c][0]) for c in schema],
            "null_pct": [round(100.0 * int(nulls[c][0]) / max(n_rows, 1), 3) for c in schema],
        }
    ).sort("null_pct", descending=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--out", default="reports", type=Path)
    args = ap.parse_args()

    training, ranking_path, submitting_path = find_files(args.data_dir)
    if not training:
        print(f"ERROR: no training_*.parquet found under {args.data_dir}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet([str(p) for p in training])
    schema = dict(lf.collect_schema())
    n_rows = int(lf.select(pl.len()).collect().item())

    lines: list[str] = ["# Data audit: PRC Data Challenge 2026", ""]
    lines += [f"- Training files: {len(training)}", f"- Training rows: {n_rows:,}", ""]

    # --- 1. Schema and completeness -------------------------------------------------
    nt = null_table(lf, schema, n_rows)
    nt.write_csv(args.out / "training_nulls.csv")
    lines += ["## 1. Columns, types and missing values", "", "| column | dtype | nulls | null % |", "|---|---|---:|---:|"]
    lines += [f"| `{r['column']}` | {r['dtype']} | {r['nulls']:,} | {r['null_pct']} |" for r in nt.iter_rows(named=True)]
    lines += [""]

    lf = reporting_airport(lf)

    # --- 2. Volumes -----------------------------------------------------------------
    by_ap = (
        lf.group_by(["AIRPORT", "PHASE_mvt"]).agg(pl.len().alias("movements"))
        .collect()
        .pivot(on="PHASE_mvt", index="AIRPORT", values="movements")
        .fill_null(0)
        .sort("AIRPORT")
    )
    by_ap = by_ap.with_columns(
        pl.col("AIRPORT").replace_strict(AIRPORTS, default="(unexpected)").alias("name")
    )
    by_ap.write_csv(args.out / "movements_by_airport.csv")
    cols = [c for c in ("DEP", "ARR") if c in by_ap.columns]
    lines += ["## 2. How much data per airport", "", "| ICAO | name | " + " | ".join(cols) + " |",
              "|---|---|" + "---:|" * len(cols)]
    for r in by_ap.iter_rows(named=True):
        lines.append(f"| {r['AIRPORT']} | {r['name']} | " + " | ".join(f"{int(r[c]):,}" for c in cols) + " |")
    lines += [""]

    by_month = (
        lf.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("month"))
        .group_by(["month", "PHASE_mvt"]).agg(pl.len().alias("movements"))
        .collect().pivot(on="PHASE_mvt", index="month", values="movements").fill_null(0).sort("month")
    )
    by_month.write_csv(args.out / "movements_by_month.csv")
    lines += ["### Movements per month", "", "| month | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for r in by_month.iter_rows(named=True):
        lines.append(f"| {r['month']} | " + " | ".join(f"{int(r[c]):,}" for c in cols) + " |")
    lines += [""]

    # --- 3. The target ---------------------------------------------------------------
    dep = lf.filter(pl.col("PHASE_mvt") == "DEP")
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    tstats = dep.select(
        [pl.len().alias("departures"), pl.col(TARGET).null_count().alias("target_nulls"),
         pl.col(TARGET).min().alias("min"), pl.col(TARGET).mean().alias("mean"),
         pl.col(TARGET).std().alias("std"), pl.col(TARGET).max().alias("max"),
         (pl.col(TARGET) <= 0).sum().alias("le_zero"),
         (pl.col(TARGET) < 60).sum().alias("under_1min"),
         (pl.col(TARGET) > 3600).sum().alias("over_60min")]
        + [pl.col(TARGET).quantile(q).alias(f"p{q*100:g}") for q in qs]
    ).collect()
    tstats.write_csv(args.out / "target_stats.csv")
    s = tstats.row(0, named=True)
    lines += ["## 3. The target: taxi-out seconds (departures only)", ""]
    lines += [f"- Departures: {int(s['departures']):,}  (target missing on {int(s['target_nulls']):,})",
              f"- Mean {s['mean']:.1f} s ({s['mean']/60:.1f} min), std {s['std']:.1f} s, median {s['p50']:.0f} s",
              f"- Range {s['min']:.0f} s .. {s['max']:.0f} s",
              f"- Suspicious: {int(s['le_zero']):,} rows <= 0 s, {int(s['under_1min']):,} under 1 min, {int(s['over_60min']):,} over 60 min",
              "- Percentiles (s): " + ", ".join(f"p{q*100:g}={s[f'p{q*100:g}']:.0f}" for q in qs), ""]

    per_ap = (
        dep.group_by("AIRPORT").agg(
            pl.len().alias("departures"), pl.col(TARGET).median().alias("median_s"),
            pl.col(TARGET).mean().alias("mean_s"), pl.col(TARGET).std().alias("std_s"),
            pl.col(TARGET).quantile(0.9).alias("p90_s"),
            pl.col("RUNWAY_mvt").n_unique().alias("runways"),
            pl.col("STAND_mvt").n_unique().alias("stands"),
        ).sort("median_s", descending=True).collect()
    )
    per_ap.write_csv(args.out / "target_by_airport.csv")
    lines += ["### Taxi-out per airport", "", "| airport | departures | median | mean | std | p90 | runways | stands |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_ap.iter_rows(named=True):
        lines.append(
            f"| {r['AIRPORT']} | {r['departures']:,} | {r['median_s']:.0f} s ({r['median_s']/60:.1f} min) | "
            f"{r['mean_s']:.0f} s | {r['std_s']:.0f} s | {r['p90_s']:.0f} s | {r['runways']} | {r['stands']} |"
        )
    lines += [""]

    # --- 4. Integrity ----------------------------------------------------------------
    dup_mvt = int(lf.select(pl.len() - pl.col("MVT_ID_mvt").n_unique()).collect().item())
    consistency = dep.select([
        (pl.col("ADEP_mvt") != pl.col("ADEP_flt")).sum().alias("adep_mvt_vs_flt_mismatch"),
        (pl.col("AIRCRAFT_TYPE_mvt") != pl.col("AIRCRAFT_TYPE_flt")).sum().alias("actype_mismatch"),
        pl.col("FLIGHT_ID_mvt").is_null().sum().alias("unmatched_to_nm_flight"),
    ]).collect().row(0, named=True)
    lines += ["## 4. Integrity checks", "",
              f"- Duplicate `MVT_ID_mvt`: {dup_mvt:,}",
              f"- Departures not matched to an NM flight (`FLIGHT_ID_mvt` null): {consistency['unmatched_to_nm_flight']:,}",
              f"- Departures where movement ADEP != flight ADEP: {consistency['adep_mvt_vs_flt_mismatch']:,}",
              f"- Departures where movement aircraft type != flight aircraft type: {consistency['actype_mismatch']:,}", ""]

    # --- 5. Categorical cardinality ---------------------------------------------------
    cats = ["AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt",
            "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "RUNWAY_mvt", "STAND_mvt", "ADES_mvt"]
    cats = [c for c in cats if c in schema]
    card = lf.select([pl.col(c).n_unique().alias(c) for c in cats]).collect().row(0, named=True)
    lines += ["## 5. Category sizes (whole training set)", "", "| column | distinct values |", "|---|---:|"]
    lines += [f"| `{c}` | {card[c]:,} |" for c in cats]
    lines += [""]

    # --- 6. Training vs ranking vs submitting -----------------------------------------
    lines += ["## 6. What changes in the ranking data", ""]
    if ranking_path is None:
        lines += ["`ranking.parquet` not found locally, section skipped.", ""]
    else:
        rlf = reporting_airport(pl.scan_parquet(str(ranking_path)))
        rschema = dict(pl.scan_parquet(str(ranking_path)).collect_schema())
        r_rows = int(rlf.select(pl.len()).collect().item())
        r_dep = rlf.filter(pl.col("PHASE_mvt") == "DEP")
        r_dep_rows = int(r_dep.select(pl.len()).collect().item())
        missing_cols = [c for c in schema if c not in rschema]
        extra_cols = [c for c in rschema if c not in schema]
        lines += [f"- Ranking rows: {r_rows:,} (departures {r_dep_rows:,})",
                  f"- Columns missing vs training: {missing_cols or 'none'}",
                  f"- Columns added vs training: {extra_cols or 'none'}", ""]
        rn = null_table(r_dep, rschema, r_dep_rows).rename({"nulls": "dep_nulls", "null_pct": "dep_null_pct"})
        tn = null_table(dep, schema, int(s["departures"])).select(["column", "null_pct"]).rename({"null_pct": "train_dep_null_pct"})
        cmp = rn.join(tn, on="column", how="left")
        cmp.write_csv(args.out / "ranking_vs_training_nulls.csv")
        lines += ["### Availability on ranking departures (this decides what we may use as a feature)", "",
                  "| column | ranking DEP null % | training DEP null % |", "|---|---:|---:|"]
        for r in cmp.sort("dep_null_pct", descending=True).iter_rows(named=True):
            lines.append(f"| `{r['column']}` | {r['dep_null_pct']} | {r['train_dep_null_pct']} |")
        lines += ["", "Any column that is ~100% null on ranking departures is unusable as a feature.", ""]
        r_months = (
            rlf.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("month"))
            .group_by(["month", "PHASE_mvt"]).agg(pl.len().alias("movements")).collect().sort("month")
        )
        lines += ["### Ranking months", "", "| month | phase | movements |", "|---|---|---:|"]
        lines += [f"| {r['month']} | {r['PHASE_mvt']} | {r['movements']:,} |" for r in r_months.iter_rows(named=True)]
        lines += [""]

        if submitting_path is not None:
            sub = pl.read_parquet(str(submitting_path))
            r_ids = r_dep.select("MVT_ID_mvt").collect().to_series()
            same = set(sub["MVT_ID_mvt"].to_list()) == set(r_ids.to_list())
            lines += ["### Submission template", "",
                      f"- Rows: {sub.height:,}  columns: {sub.columns}",
                      f"- Dtypes: {[str(t) for t in sub.dtypes]}",
                      f"- IDs identical to ranking departures: {same}",
                      f"- Duplicate IDs in template: {sub.height - sub['MVT_ID_mvt'].n_unique():,}", ""]

    out_md = args.out / "data_audit.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md} and CSVs in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
