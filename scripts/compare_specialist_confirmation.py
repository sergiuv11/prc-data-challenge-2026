#!/usr/bin/env python3
"""Apply a locked forward, seasonal or December gate to partition specialists."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import experiment_tail_overlay as OV  # noqa: E402
import features as F  # noqa: E402
import train_model as T  # noqa: E402

ID = "MVT_ID_mvt"
PRED = "pred"
TARGET = F.TARGET
OVERLAY_THRESHOLD = 14400


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def load_predictions(path: Path, label: str) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = {ID, PRED} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")
    frame = frame.select(ID, PRED)
    values = frame[PRED].cast(pl.Float64).to_numpy()
    if frame[ID].is_null().any() or frame[ID].is_duplicated().any():
        raise ValueError(f"{label} has null or duplicate movement IDs")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has non-finite predictions")
    return frame


def bootstrap(base: np.ndarray, candidate: np.ndarray, truth: np.ndarray,
              count: int) -> tuple[float, float, float]:
    base_se = (base - truth) ** 2
    candidate_se = (candidate - truth) ** 2
    rng = np.random.default_rng(20260901)
    changes = np.empty(count)
    for iteration in range(count):
        index = rng.integers(0, len(truth), len(truth))
        changes[iteration] = (
            np.sqrt(base_se[index].mean()) - np.sqrt(candidate_se[index].mean())
        )
    low, high = np.percentile(changes, [5, 95])
    return float((changes > 0).mean()), float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--split", required=True,
                        choices=["forward", "seasonal", "december",
                                 "october_selection"])
    candidate_scope = parser.add_mutually_exclusive_group(required=True)
    candidate_scope.add_argument(
        "--candidate-airports", help="comma-separated airports allowed to change"
    )
    candidate_scope.add_argument(
        "--candidate-wake-categories",
        help="comma-separated raw wake categories allowed to change",
    )
    candidate_scope.add_argument(
        "--all-rows-may-change", action="store_true",
        help="declare that a global feature or model change may alter every prediction",
    )
    parser.add_argument("--features", default="data/features/train_departures.parquet",
                        type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--minimum-win-rate", type=float, default=0.95)
    parser.add_argument("--maximum-row-share", type=float, default=50.0)
    args = parser.parse_args()

    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if not 0.0 <= args.minimum_win_rate <= 1.0:
        parser.error("--minimum-win-rate must be between 0 and 1")
    if not 0.0 < args.maximum_row_share <= 100.0:
        parser.error("--maximum-row-share must be greater than 0 and at most 100")

    airports: tuple[str, ...] = ()
    wake_categories: tuple[str, ...] = ()
    if args.candidate_airports:
        airports = tuple(dict.fromkeys(
            value.strip().upper() for value in args.candidate_airports.split(",")
            if value.strip()
        ))
        if not airports:
            parser.error("--candidate-airports must contain at least one ICAO code")
    elif args.candidate_wake_categories:
        wake_categories = tuple(dict.fromkeys(
            value.strip().upper() for value in args.candidate_wake_categories.split(",")
            if value.strip()
        ))
        if not wake_categories:
            parser.error("--candidate-wake-categories must contain at least one category")

    try:
        baseline = load_predictions(args.baseline, "baseline")
        candidate = load_predictions(args.candidate, "candidate")
        if baseline.height != candidate.height or not baseline[ID].equals(candidate[ID]):
            raise ValueError("baseline and candidate IDs or row order differ")

        full = pl.read_parquet(args.features, columns=[
            ID, TARGET, F.TAKEOFF, "AIRPORT", "WK_TBL_CAT_flt", "month", "aobt_missing",
            "takeoff_minus_schedule",
        ])
        T.init_airport_codes(args.data_dir)
        day = pl.col(F.TAKEOFF).dt.day()
        if args.split == "forward":
            test = full.filter(
                (pl.col("month") == 7) & pl.col("AIRPORT").is_in(T.july_codes(full))
            )
            overlay_fit = full.filter(pl.col("month") <= 5)
        elif args.split == "seasonal":
            test = full.filter(T.composition_mask(full) & (day >= 22))
            overlay_fit = full.filter(day <= 20)
        elif args.split == "december":
            test = full.filter(pl.col("month") == 12)
            overlay_fit = full.filter(pl.col("month") <= 10)
        else:
            test = full.filter(pl.col("month") == 10)
            overlay_fit = full.filter(pl.col("month") <= 8)

        if baseline.height != test.height or not baseline[ID].equals(test[ID]):
            raise ValueError("prediction IDs do not exactly match the selected split")

        codes = OV.airport_codes(args.data_dir)
        unknown = sorted(set(airports) - set(codes))
        if unknown:
            raise ValueError(f"unknown candidate airports: {', '.join(unknown)}")
        wake_names = (
            pl.scan_parquet(
                [str(path) for path in sorted(args.data_dir.glob("training_*.parquet"))]
                + [str(args.data_dir / "ranking.parquet")]
            )
            .filter(pl.col("PHASE_mvt") == "DEP")
            .select("WK_TBL_CAT_flt").unique().collect()
        )["WK_TBL_CAT_flt"].drop_nulls().sort().to_list()
        wake_codes = {name: value for value, name in enumerate(wake_names)}
        unknown_wake = sorted(set(wake_categories) - set(wake_codes))
        if unknown_wake:
            raise ValueError(
                f"unknown candidate wake categories: {', '.join(unknown_wake)}"
            )
        lirf = codes["LIRF"]
        coefficient = OV.fit_calibration(overlay_fit, [lirf], OVERLAY_THRESHOLD)
        if coefficient is None:
            raise ValueError("not enough split-training LIRF rows to fit the overlay")
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    truth = test[TARGET].cast(pl.Float64).to_numpy()
    base_raw = baseline[PRED].cast(pl.Float64).to_numpy()
    candidate_raw = candidate[PRED].cast(pl.Float64).to_numpy()
    base_overlay, touched = OV.apply_overlay(
        base_raw, test, [lirf], OVERLAY_THRESHOLD, coefficient
    )
    candidate_overlay, candidate_touched = OV.apply_overlay(
        candidate_raw, test, [lirf], OVERLAY_THRESHOLD, coefficient
    )
    if touched != candidate_touched:
        raise RuntimeError("overlay touched different rows in the two arms")

    if args.all_rows_may_change:
        outside = np.zeros(test.height, dtype=bool)
        scope_description = "the global candidate scope"
    elif airports:
        candidate_codes = np.array([codes[airport] for airport in airports])
        outside = ~np.isin(test["AIRPORT"].to_numpy(), candidate_codes)
        scope_description = f"airports {', '.join(airports)}"
    else:
        candidate_codes = np.array([wake_codes[category] for category in wake_categories])
        outside = ~np.isin(test["WK_TBL_CAT_flt"].to_numpy(), candidate_codes)
        scope_description = f"wake categories {', '.join(wake_categories)}"
    outside_delta = np.abs(candidate_raw[outside] - base_raw[outside])
    changed_outside = int(np.count_nonzero(outside_delta))
    maximum_outside_delta = float(outside_delta.max(initial=0.0))
    unchanged = changed_outside == 0

    raw_change = rmse(candidate_raw, truth) - rmse(base_raw, truth)
    overlay_change = rmse(candidate_overlay, truth) - rmse(base_overlay, truth)
    win_rate, low, high = bootstrap(
        base_overlay, candidate_overlay, truth, args.bootstrap
    )
    row_advantage = ((base_overlay - truth) ** 2
                     - (candidate_overlay - truth) ** 2)
    net_advantage = float(row_advantage.sum())
    largest_share = (100.0 * float(row_advantage.max()) / net_advantage
                     if net_advantage > 0 else float("nan"))

    improves_raw = raw_change < 0
    improves_overlay = overlay_change < 0
    confident = win_rate >= args.minimum_win_rate
    broad = bool(np.isfinite(largest_share)
                 and largest_share <= args.maximum_row_share)
    passed = improves_raw and improves_overlay and confident and unchanged and broad
    verdict = "PASS" if passed else "FAIL"

    lines = [f"# {args.split.title()} confirmation: {args.label}", "",
             f"Rows: {test.height:,}. Overlay fitted only on this split's training rows "
             f"(a={coefficient[0]:.4f}, b={coefficient[1]:.6f}) and applied identically "
             f"to both arms, touching {touched} rows.", "",
             "| arm | raw RMSE s | overlay adjusted RMSE s |", "|---|---:|---:|",
             f"| baseline | {rmse(base_raw, truth):.3f} | {rmse(base_overlay, truth):.3f} |",
             f"| {args.label} | {rmse(candidate_raw, truth):.3f} | "
             f"{rmse(candidate_overlay, truth):.3f} |",
             f"| **change** | **{raw_change:+.3f}** | **{overlay_change:+.3f}** |", "",
             f"Paired overlay bootstrap, {args.bootstrap:,} resamples: candidate wins "
             f"{100 * win_rate:.1f} percent, improvement interval {low:+.2f} to "
             f"{high:+.2f} seconds.", "",
             f"Non-candidate invariance outside {scope_description}, across "
             f"{int(outside.sum()):,} rows: "
             f"{changed_outside:,} changed, maximum absolute change "
             f"{maximum_outside_delta:.12g} seconds.", "",
             (f"Largest single-row contribution to net overlay squared-error advantage: "
              f"{largest_share:.1f} percent."
              if np.isfinite(largest_share) else
              "Largest single-row share is undefined because net advantage is not positive."),
             "", f"**Gate: raw improves {improves_raw}, overlay improves "
             f"{improves_overlay}, overlay bootstrap at least "
             f"{100 * args.minimum_win_rate:.0f} percent {confident}, non-candidate rows "
             f"unchanged {unchanged}, largest-row share at most "
             f"{args.maximum_row_share:.0f} percent {broad}. Verdict: {verdict}.**", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
