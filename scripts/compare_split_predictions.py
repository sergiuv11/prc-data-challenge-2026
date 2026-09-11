#!/usr/bin/env python3
"""Compare two prediction files for the same validation split with a paired bootstrap."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402

ID = "MVT_ID_mvt"
PRED = "pred"
TARGET = F.TARGET


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def load(path: Path, name: str) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = {ID, PRED} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")
    frame = frame.select([ID, PRED])
    pred = frame[PRED].cast(pl.Float64).to_numpy()
    if frame[ID].is_null().any() or frame[ID].is_duplicated().any():
        raise ValueError(f"{name} has null or duplicate movement IDs")
    if not np.isfinite(pred).all():
        raise ValueError(f"{name} has non-finite predictions")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--features", default="data/features/train_departures.parquet", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--minimum-win-rate", type=float, default=0.95)
    args = parser.parse_args()

    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if not 0.0 <= args.minimum_win_rate <= 1.0:
        parser.error("--minimum-win-rate must be between 0 and 1")

    try:
        baseline = load(args.baseline, "baseline")
        candidate = load(args.candidate, "candidate")
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if baseline.height != candidate.height or not baseline[ID].equals(candidate[ID]):
        print("ERROR: baseline and candidate IDs or row order differ", file=sys.stderr)
        return 1

    truth = pl.read_parquet(args.features, columns=[ID, TARGET])
    joined = baseline.join(truth, on=ID, how="left", validate="1:1")
    if joined.height != baseline.height or joined[TARGET].null_count() != 0:
        print("ERROR: predictions do not map one-to-one to feature targets", file=sys.stderr)
        return 1

    y = joined[TARGET].cast(pl.Float64).to_numpy()
    base = baseline[PRED].cast(pl.Float64).to_numpy()
    cand = candidate[PRED].cast(pl.Float64).to_numpy()
    base_rmse = rmse(base, y)
    cand_rmse = rmse(cand, y)

    base_se = (base - y) ** 2
    cand_se = (cand - y) ** 2
    advantage = base_se - cand_se
    net_advantage = float(advantage.sum())
    largest_advantage = float(advantage.max())
    largest_share = (100.0 * largest_advantage / net_advantage
                     if net_advantage > 0 else float("nan"))
    rng = np.random.default_rng(20260901)
    n = len(y)
    changes = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        index = rng.integers(0, n, n)
        changes[i] = np.sqrt(base_se[index].mean()) - np.sqrt(cand_se[index].mean())

    win_rate = float((changes > 0).mean())
    improves = cand_rmse < base_rmse
    confident = win_rate >= args.minimum_win_rate
    passed = improves and confident
    verdict = "PASS" if passed else "FAIL"
    low, high = np.percentile(changes, [5, 95])

    lines = [f"# Confirmation: {args.label}", "",
             f"Rows: {n:,}.", "",
             "| model | raw RMSE s |", "|---|---:|",
             f"| preserved v2 | {base_rmse:.3f} |",
             f"| candidate | {cand_rmse:.3f} |",
             f"| **change** | **{cand_rmse - base_rmse:+.3f}** |", "",
             f"Paired bootstrap, {args.bootstrap:,} resamples: candidate wins "
             f"{100 * win_rate:.1f} percent, improvement interval {low:+.2f} s to "
             f"{high:+.2f} s.", "",
             (f"Largest single-row contribution to the net squared-error advantage: "
              f"{largest_share:.1f} percent."
              if np.isfinite(largest_share) else
              "Largest single-row share is not defined because the net advantage is not positive."),
             "",
             f"**Gate: point estimate improves {improves}, bootstrap win rate at least "
             f"{100 * args.minimum_win_rate:.0f} percent {confident}. Verdict: {verdict}.**", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
