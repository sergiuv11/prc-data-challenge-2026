#!/usr/bin/env python3
"""Generate a synthetic dataset with the official 2026 schema.

No competition data is used or needed. This exists so the ingestion, audit,
leakage and submission code can be exercised and tested before access is granted,
and so the reusable system can ship with test data that is not PRC data.

Usage:
    .venv/bin/python scripts/make_synthetic_fixture.py --out data/synthetic
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTAI", "LTFM", "LSZH"]
RUNWAYS = {a: [f"{a[-2:]}{s}" for s in ("L", "R", "C")] for a in AIRPORTS}
TYPES = ["A320", "A21N", "B738", "A333", "B77W", "E195", "A359"]
WTC = {"A320": "M", "A21N": "M", "B738": "M", "A333": "H", "B77W": "H", "E195": "M", "A359": "H"}
SEGMENTS = ["Mainline", "Low-Cost", "Regional", "All-Cargo", "Charter", "Business Aviation"]


def month_frame(rng: np.random.Generator, start: datetime, end: datetime, n: int, offset: int) -> pl.DataFrame:
    span = int((end - start).total_seconds())
    secs = np.sort(rng.integers(0, span, n))
    takeoff = np.datetime64(start, "s") + secs.astype("timedelta64[s]")
    ap = rng.choice(AIRPORTS, n)
    rwy = np.array([rng.choice(RUNWAYS[a]) for a in ap])
    typ = rng.choice(TYPES, n)
    hour = takeoff.astype("datetime64[h]").astype(int) % 24

    base = {a: b for a, b in zip(AIRPORTS, rng.uniform(600, 1100, len(AIRPORTS)))}
    peak = 180 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2) + 150 * np.exp(-0.5 * ((hour - 18) / 2.5) ** 2)
    heavy = np.where(np.isin(typ, ["A333", "B77W", "A359"]), 90.0, 0.0)
    taxi = np.clip(np.array([base[a] for a in ap]) + peak + heavy + rng.gamma(2.0, 60.0, n), 60, 7200).round()

    offblock = takeoff - taxi.astype("int64").astype("timedelta64[s]")
    # NM actual off-block: the same event seen by another system, with jitter and gaps.
    aobt = offblock + rng.normal(0, 45, n).round().astype("int64").astype("timedelta64[s]")
    sched = offblock - rng.normal(300, 600, n).round().astype("int64").astype("timedelta64[s]")
    arvt = takeoff + np.timedelta64(7200, "s")

    takeoff, offblock = takeoff.astype("datetime64[us]"), offblock.astype("datetime64[us]")
    aobt, sched, arvt = (x.astype("datetime64[us]") for x in (aobt, sched, arvt))

    phase = np.where(rng.random(n) < 0.5, "DEP", "ARR")
    ids = np.arange(offset, offset + n, dtype=np.int64)

    df = pl.DataFrame({
        "MVT_ID_mvt": ids,
        "FLIGHT_ID_mvt": ids + 900_000_000,
        "FLIGHT_mvt": [f"XX{i % 9000 + 1000}" for i in ids],
        "FLIGHT_RULE_mvt": rng.choice(["I", "V"], n, p=[0.98, 0.02]),
        "ADEP_mvt": np.where(phase == "DEP", ap, "LOWW"),
        "ADES_mvt": np.where(phase == "DEP", "LOWW", ap),
        "PHASE_mvt": phase,
        "MVT_TIME_UTC_mvt": takeoff,
        "BLOCK_TIME_UTC_mvt": offblock,
        "SCHED_TIME_UTC_mvt": sched,
        "AIRCRAFT_TYPE_mvt": typ,
        "RUNWAY_mvt": rwy,
        "STAND_mvt": [f"{a[-1]}{rng.integers(1, 60)}" for a in ap],
        "TAXITIME_SEC_mvt": taxi.astype(np.int32),
        "LOBT_flt": offblock,
        "CALLSIGN_flt": [f"ABC{i % 900 + 100}" for i in ids],
        "ADEP_flt": np.where(phase == "DEP", ap, "LOWW"),
        "ADES_flt": np.where(phase == "DEP", "LOWW", ap),
        "ADES_FILED_flt": np.where(phase == "DEP", "LOWW", ap),
        "MARKET_SEGMENT_flt": rng.choice(SEGMENTS, n),
        "IOBT_flt": sched,
        "FLIGHT_RULE_flt": rng.choice(["I", "Y", "Z"], n, p=[0.97, 0.02, 0.01]),
        "FLIGHT_TYPE_flt": rng.choice(["S", "N", "G", "X"], n, p=[0.9, 0.06, 0.03, 0.01]),
        "AIRCRAFT_TYPE_flt": typ,
        "WK_TBL_CAT_flt": [WTC[t] for t in typ],
        "AIRCRAFT_OPERATOR_flt": rng.choice([f"OP{i:02d}" for i in range(40)], n),
        "EOBT_1_flt": sched,
        "ARVT_1_flt": arvt,
        "AOBT_3_flt": aobt,
        "ARVT_3_flt": arvt,
    })
    # Real data is incomplete: blank some NM matches and some actual off-block times.
    unmatched = pl.Series(rng.random(n) < 0.05)
    no_aobt = pl.Series(rng.random(n) < 0.12)
    return df.with_columns(
        pl.when(unmatched).then(None).otherwise(pl.col("FLIGHT_ID_mvt")).alias("FLIGHT_ID_mvt"),
        pl.when(no_aobt).then(None).otherwise(pl.col("AOBT_3_flt")).alias("AOBT_3_flt"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic", type=Path)
    ap.add_argument("--per-month", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    offset = 1
    for m in range(1, 13):
        start = datetime(2025, m, 1)
        end = datetime(2025 + (m == 12), (m % 12) + 1, 1)
        df = month_frame(rng, start, end, args.per_month, offset)
        offset += args.per_month
        df.write_parquet(args.out / f"training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet")

    ranking = pl.concat([
        month_frame(rng, datetime(2026, 1, 1), datetime(2026, 2, 1), args.per_month, offset),
        month_frame(rng, datetime(2026, 7, 1), datetime(2026, 8, 1), args.per_month, offset + args.per_month),
    ])
    ranking = ranking.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP").then(None).otherwise(pl.col("BLOCK_TIME_UTC_mvt")).alias("BLOCK_TIME_UTC_mvt"),
        pl.when(pl.col("PHASE_mvt") == "DEP").then(None).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"),
    )
    ranking.write_parquet(args.out / "ranking.parquet")
    ranking.filter(pl.col("PHASE_mvt") == "DEP").select(["MVT_ID_mvt", "TAXITIME_SEC_mvt"]).write_parquet(
        args.out / "submitting.parquet"
    )
    print(f"Synthetic fixture written to {args.out}/ (schema only, no PRC data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
