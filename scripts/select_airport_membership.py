#!/usr/bin/env python3
"""Select airport specialists mechanically on the predeclared October split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import experiment_tail_overlay as OV  # noqa: E402
import features as F  # noqa: E402

ID = "MVT_ID_mvt"
PRED = "pred"
TARGET = F.TARGET
OVERLAY_THRESHOLD = 14_400


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


def bootstrap_win_rate(base: np.ndarray, candidate: np.ndarray, truth: np.ndarray,
                       count: int, seed: int) -> float:
    base_se = (base - truth) ** 2
    candidate_se = (candidate - truth) ** 2
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(count):
        index = rng.integers(0, len(truth), len(truth))
        change = np.sqrt(base_se[index].mean()) - np.sqrt(candidate_se[index].mean())
        wins += change > 0
    return float(wins / count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-predictions", required=True, type=Path)
    parser.add_argument("--all-specialists-predictions", required=True, type=Path)
    parser.add_argument("--features", default="data/features/train_departures.parquet",
                        type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--out", default="reports/airport_membership", type=Path)
    parser.add_argument("--minimum-improvement", type=float, default=1.0)
    parser.add_argument("--minimum-win-rate", type=float, default=0.80)
    parser.add_argument("--maximum-row-share", type=float, default=50.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    if args.minimum_improvement <= 0:
        parser.error("--minimum-improvement must be positive")
    if not 0.0 <= args.minimum_win_rate <= 1.0:
        parser.error("--minimum-win-rate must be between 0 and 1")
    if not 0.0 < args.maximum_row_share <= 100.0:
        parser.error("--maximum-row-share must be greater than 0 and at most 100")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")

    try:
        global_frame = load_predictions(args.global_predictions, "global predictions")
        specialists = load_predictions(
            args.all_specialists_predictions, "all-specialists predictions"
        )
        if (global_frame.height != specialists.height
                or not global_frame[ID].equals(specialists[ID])):
            raise ValueError("prediction files have different movement IDs or row order")

        full = pl.read_parquet(args.features, columns=[
            ID, TARGET, "AIRPORT", "month", "aobt_missing", "takeoff_minus_schedule",
        ])
        october = full.filter(pl.col("month") == 10)
        if global_frame.height != october.height or not global_frame[ID].equals(october[ID]):
            raise ValueError("prediction IDs do not exactly match October")

        codes = OV.airport_codes(args.data_dir)
        inverse_codes = {value: key for key, value in codes.items()}
        observed_codes = sorted(october["AIRPORT"].unique().drop_nulls().to_list())
        if set(observed_codes) != set(inverse_codes):
            raise ValueError("October does not contain exactly the ten mapped airports")

        lirf = codes["LIRF"]
        overlay_fit = full.filter(pl.col("month") <= 8)
        coefficient = OV.fit_calibration(overlay_fit, [lirf], OVERLAY_THRESHOLD)
        if coefficient is None:
            raise ValueError("not enough January-August LIRF rows to fit the overlay")
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    truth = october[TARGET].cast(pl.Float64).to_numpy()
    global_raw = global_frame[PRED].cast(pl.Float64).to_numpy()
    specialist_raw = specialists[PRED].cast(pl.Float64).to_numpy()
    global_overlay, touched = OV.apply_overlay(
        global_raw, october, [lirf], OVERLAY_THRESHOLD, coefficient
    )
    specialist_overlay, candidate_touched = OV.apply_overlay(
        specialist_raw, october, [lirf], OVERLAY_THRESHOLD, coefficient
    )
    if touched != candidate_touched:
        raise RuntimeError("overlay touched different rows in the two arms")

    selected: list[str] = []
    rows: list[dict[str, float | int | bool | str]] = []
    airport_values = october["AIRPORT"].to_numpy()
    for code in observed_codes:
        airport = inverse_codes[code]
        mask = airport_values == code
        base = global_overlay[mask]
        candidate = specialist_overlay[mask]
        local_truth = truth[mask]
        base_rmse = rmse(base, local_truth)
        candidate_rmse = rmse(candidate, local_truth)
        improvement = base_rmse - candidate_rmse
        base_se = (base - local_truth) ** 2
        candidate_se = (candidate - local_truth) ** 2
        advantage = base_se - candidate_se
        net_advantage = float(advantage.sum())
        largest_share = (100.0 * float(advantage.max()) / net_advantage
                         if net_advantage > 0 else float("nan"))
        win_rate = bootstrap_win_rate(
            base, candidate, local_truth, args.bootstrap, 20260901 + int(code)
        )
        passes = bool(
            improvement >= args.minimum_improvement
            and win_rate >= args.minimum_win_rate
            and np.isfinite(largest_share)
            and largest_share <= args.maximum_row_share
        )
        if passes:
            selected.append(airport)
        rows.append({
            "airport": airport,
            "rows": int(mask.sum()),
            "global_overlay_rmse": base_rmse,
            "blend_overlay_rmse": candidate_rmse,
            "improvement": improvement,
            "bootstrap_win_rate": win_rate,
            "largest_row_share": largest_share,
            "selected": passes,
        })

    frozen = global_raw.copy()
    selected_codes = [codes[airport] for airport in selected]
    selected_mask = np.isin(airport_values, selected_codes)
    frozen[selected_mask] = specialist_raw[selected_mask]
    output_predictions = october.select(ID).with_columns(pl.Series(PRED, frozen))

    args.out.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out / "preds_october_selected.parquet"
    membership_path = args.out / "membership.json"
    report_path = args.out / "selection.md"
    output_predictions.write_parquet(prediction_path)
    membership = {
        "split": {"fit_months": [1, 2, 3, 4, 5, 6, 7, 8],
                  "inner_month": 9, "selection_month": 10},
        "rule": {"arm": "calibrated LIRF overlay adjusted",
                 "minimum_rmse_improvement_seconds": args.minimum_improvement,
                 "minimum_bootstrap_win_rate": args.minimum_win_rate,
                 "maximum_largest_row_share_percent": args.maximum_row_share,
                 "bootstrap_resamples": args.bootstrap},
        "overlay": {"threshold_seconds": OVERLAY_THRESHOLD,
                    "intercept": coefficient[0], "slope": coefficient[1],
                    "october_rows_touched": touched},
        "selected_airports": selected,
        "airports": rows,
    }
    membership_path.write_text(
        json.dumps(membership, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = ["# October airport-specialist membership selection", "",
             "Selection only, not a confirmation gate.", "",
             f"Rule: overlay-adjusted RMSE improves by at least "
             f"{args.minimum_improvement:.1f} s, bootstrap wins at least "
             f"{100 * args.minimum_win_rate:.0f} percent, and largest-row share is at most "
             f"{args.maximum_row_share:.0f} percent.", "",
             f"Overlay fitted on January-August only, a={coefficient[0]:.4f}, "
             f"b={coefficient[1]:.6f}, touching {touched} October rows.", "",
             "| airport | rows | global overlay RMSE s | blend overlay RMSE s | improvement s | bootstrap wins | largest row share | selected |",
             "|---|---:|---:|---:|---:|---:|---:|---|" ]
    for row in rows:
        share = (f"{row['largest_row_share']:.1f} %"
                 if np.isfinite(float(row["largest_row_share"])) else "n/a")
        lines.append(
            f"| {row['airport']} | {row['rows']:,} | "
            f"{row['global_overlay_rmse']:.3f} | {row['blend_overlay_rmse']:.3f} | "
            f"{row['improvement']:+.3f} | "
            f"{100 * float(row['bootstrap_win_rate']):.1f} % | {share} | "
            f"{'yes' if row['selected'] else 'no'} |"
        )
    lines += ["", f"**Frozen membership: {', '.join(selected) if selected else 'none'}.**", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {membership_path}, {prediction_path} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
