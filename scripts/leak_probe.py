#!/usr/bin/env python3
"""Step 3 of the plan: find out which columns quietly reveal the answer.

Taxi-out time is, by definition, takeoff time minus off-block time. The ranking
file blanks BLOCK_TIME_UTC_mvt (off-block) and the target, but it does NOT blank
MVT_TIME_UTC_mvt (takeoff). Several Network Manager timestamps are candidates for
the missing off-block moment, so each of them is a potential shortcut to the answer.

This script measures, honestly and numerically:
  1. how exactly `takeoff - <candidate off-block>` reproduces the target in 2025;
  2. how often that candidate is actually populated on the 2026 ranking departures.

A candidate is only exploitable if it is both accurate AND present in ranking data.

Usage:
    .venv/bin/python scripts/leak_probe.py [--data-dir data/raw] [--out reports]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

TARGET = "TAXITIME_SEC_mvt"
TAKEOFF = "MVT_TIME_UTC_mvt"

# (name, off-block candidate column, what it is)
CANDIDATES = [
    ("BLOCK_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "airport off-block (blanked in ranking; definitional check)"),
    ("AOBT_3_flt", "AOBT_3_flt", "NM actual off-block time of the flown (M3) trajectory"),
    ("LOBT_flt", "LOBT_flt", "NM last known off-block time"),
    ("IOBT_flt", "IOBT_flt", "NM initial off-block time (planned)"),
    ("EOBT_1_flt", "EOBT_1_flt", "NM estimated off-block time of the filed (M1) trajectory"),
    ("SCHED_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "airport scheduled departure time"),
]


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "median_abs": float("nan"),
                "within_1s": float("nan"), "within_30s": float("nan"), "within_60s": float("nan")}
    err = pred[ok] - truth[ok]
    a = np.abs(err)
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(a)),
        "median_abs": float(np.median(a)),
        "within_1s": float(100.0 * np.mean(a <= 1)),
        "within_30s": float(100.0 * np.mean(a <= 30)),
        "within_60s": float(100.0 * np.mean(a <= 60)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--out", default="reports", type=Path)
    args = ap.parse_args()

    training = sorted(args.data_dir.rglob("training_*.parquet"))
    ranking_path = next(iter(sorted(args.data_dir.rglob("ranking.parquet"))), None)
    if not training:
        print(f"ERROR: no training_*.parquet found under {args.data_dir}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    train_schema = dict(pl.scan_parquet([str(p) for p in training]).collect_schema())
    usable = [(n, c, d) for (n, c, d) in CANDIDATES if c in train_schema]

    dep = (
        pl.scan_parquet([str(p) for p in training])
        .filter((pl.col("PHASE_mvt") == "DEP") & pl.col(TARGET).is_not_null())
        .select([TARGET, TAKEOFF, "FLIGHT_ID_mvt", "ADEP_mvt"] + [c for _, c, _ in usable])
        .collect()
    )
    truth = dep[TARGET].cast(pl.Float64).to_numpy()

    rows: list[dict] = []
    for name, col, desc in usable:
        implied = (dep[TAKEOFF] - dep[col]).dt.total_seconds().cast(pl.Float64).to_numpy()
        m = metrics(implied, truth)
        m.update({
            "candidate": name,
            "description": desc,
            "coverage_train_pct": round(100.0 * float(dep[col].is_not_null().mean()), 3),
        })
        rows.append(m)

    res = pl.DataFrame(rows).select(
        ["candidate", "description", "coverage_train_pct", "n", "rmse", "mae",
         "median_abs", "within_1s", "within_30s", "within_60s"]
    ).sort("rmse")

    # Same test restricted to departures that matched an NM flight record.
    matched = dep.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
    truth_m = matched[TARGET].cast(pl.Float64).to_numpy()
    rows_m = []
    for name, col, desc in usable:
        implied = (matched[TAKEOFF] - matched[col]).dt.total_seconds().cast(pl.Float64).to_numpy()
        m = metrics(implied, truth_m)
        m["candidate"] = name
        rows_m.append(m)
    res_m = pl.DataFrame(rows_m).select(["candidate", "n", "rmse", "mae", "median_abs", "within_60s"])

    res.write_csv(args.out / "leak_candidates_training.csv")
    res_m.write_csv(args.out / "leak_candidates_training_nm_matched.csv")

    lines = ["# Leakage probe: does any provided timestamp reveal taxi-out time?", "",
             "`implied taxi-out = MVT_TIME_UTC_mvt (takeoff) - <candidate off-block>`", "",
             f"Training departures with a known target: {dep.height:,}", "",
             "## All training departures", "",
             "| candidate | populated % | n compared | RMSE s | MAE s | median abs s | <=1s % | <=30s % | <=60s % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res.iter_rows(named=True):
        lines.append(
            f"| `{r['candidate']}` | {r['coverage_train_pct']} | {r['n']:,} | {r['rmse']:.1f} | {r['mae']:.1f} | "
            f"{r['median_abs']:.1f} | {r['within_1s']:.1f} | {r['within_30s']:.1f} | {r['within_60s']:.1f} |"
        )
    lines += ["", "## Departures matched to an NM flight record only", "",
              "| candidate | n compared | RMSE s | MAE s | median abs s | <=60s % |", "|---|---:|---:|---:|---:|---:|"]
    for r in res_m.iter_rows(named=True):
        lines.append(f"| `{r['candidate']}` | {r['n']:,} | {r['rmse']:.1f} | {r['mae']:.1f} | {r['median_abs']:.1f} | {r['within_60s']:.1f} |")

    # Availability of the same candidates on the 2026 ranking departures.
    lines += ["", "## Availability on the 2026 ranking departures", ""]
    if ranking_path is None:
        lines += ["`ranking.parquet` not found locally, availability not checked.", ""]
    else:
        rlf = pl.scan_parquet(str(ranking_path))
        rschema = dict(rlf.collect_schema())
        rdep = rlf.filter(pl.col("PHASE_mvt") == "DEP")
        present = [c for _, c, _ in usable if c in rschema] + [TAKEOFF]
        cov = rdep.select(
            [pl.len().alias("dep_rows")] + [(100.0 * pl.col(c).is_not_null().mean()).alias(c) for c in present]
        ).collect().row(0, named=True)
        lines += [f"Ranking departures: {int(cov['dep_rows']):,}", "",
                  "| column | populated % on ranking departures |", "|---|---:|"]
        for c in present:
            lines.append(f"| `{c}` | {cov[c]:.3f} |")
        lines += ["", "### Verdict", ""]
        best = res.filter(pl.col("candidate") != "BLOCK_TIME_UTC_mvt").row(0, named=True)
        avail = cov.get(best["candidate"], 0.0)
        lines += [
            f"- Most accurate non-blanked candidate in training: `{best['candidate']}` "
            f"(RMSE {best['rmse']:.1f} s, {best['within_60s']:.1f} % within a minute).",
            f"- It is populated on {avail:.2f} % of the ranking departures.",
            "- Exploitable only if both numbers are strong. Otherwise treat it as one feature among many,",
            "  never as the whole solution: the organisers can correct an unintended leak mid-competition.",
            "",
        ]

    out_md = args.out / "leak_probe.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
