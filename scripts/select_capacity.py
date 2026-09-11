#!/usr/bin/env python3
"""Select one predeclared global-capacity configuration on the October holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

import experiment_airport_specialists as E
import experiment_tail_overlay as OV
import features as F
import train_model as T

CONFIGS = {
    "incumbent": {"num_leaves": 127, "min_data_in_leaf": 200},
    "leaves_down": {"num_leaves": 63, "min_data_in_leaf": 200},
    "leaves_up": {"num_leaves": 255, "min_data_in_leaf": 200},
    "min_data_down": {"num_leaves": 127, "min_data_in_leaf": 50},
    "min_data_up": {"num_leaves": 127, "min_data_in_leaf": 500},
}
BOOTSTRAP = 2000
SELECTION_CONFIDENCE = 0.9875
THRESHOLD = 14400


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def win_rate(a: np.ndarray, b: np.ndarray, truth: np.ndarray, seed: int) -> float:
    """Share of paired resamples where b has lower RMSE than a."""
    se_a = (a - truth) ** 2
    se_b = (b - truth) ** 2
    rng = np.random.default_rng(seed)
    n = truth.size
    wins = 0
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        wins += int(se_b[idx].mean() < se_a[idx].mean())
    return wins / BOOTSTRAP


def load(path: Path, ids: pl.Series) -> np.ndarray:
    frame = pl.read_parquet(path).select("MVT_ID_mvt", "pred")
    if frame.height != len(ids) or not frame["MVT_ID_mvt"].equals(ids):
        raise ValueError(f"prediction IDs or order do not match October: {path}")
    values = frame["pred"].cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite prediction in {path}")
    return values


def lower_capacity_key(name: str) -> tuple[int, int, int]:
    cfg = CONFIGS[name]
    return cfg["num_leaves"], -cfg["min_data_in_leaf"], name != "incumbent"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/train_departures.parquet", type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--pred-dir", default="reports/capacity_selection/complete", type=Path)
    parser.add_argument("--out", default="reports/capacity_selection/selection.md", type=Path)
    parser.add_argument("--selected", default="reports/capacity_selection/selected.json", type=Path)
    args = parser.parse_args()

    names = T.init_airport_codes(args.data_dir)
    lirf = next(code for code, name in names.items() if name == "LIRF")
    full = pl.read_parquet(args.features)
    _, october = E.split_frames(full, "october_selection")
    truth = october[F.TARGET].cast(pl.Float64).to_numpy()
    coef = OV.fit_calibration(full.filter(pl.col("month") <= 8), [lirf], THRESHOLD)
    if coef is None:
        raise ValueError("October overlay calibration has insufficient fit rows")

    raw: dict[str, np.ndarray] = {}
    overlaid: dict[str, np.ndarray] = {}
    touched: int | None = None
    for name in CONFIGS:
        raw[name] = load(args.pred_dir / f"{name}.parquet", october["MVT_ID_mvt"])
        overlaid[name], count = OV.apply_overlay(raw[name], october, [lirf], THRESHOLD, coef)
        if touched is None:
            touched = count
        elif count != touched:
            raise ValueError("overlay scope changed between candidates")

    incumbent = "incumbent"
    eligible: list[str] = []
    rows = []
    for index, name in enumerate(CONFIGS):
        raw_rmse = rmse(raw[name], truth)
        overlay_rmse = rmse(overlaid[name], truth)
        if name == incumbent:
            win = float("nan")
            qualifies = False
        else:
            win = win_rate(overlaid[incumbent], overlaid[name], truth, 20260902 + index)
            qualifies = (
                raw_rmse < rmse(raw[incumbent], truth)
                and overlay_rmse < rmse(overlaid[incumbent], truth)
                and win >= SELECTION_CONFIDENCE
            )
            if qualifies:
                eligible.append(name)
        advantage = (overlaid[incumbent] - truth) ** 2 - (overlaid[name] - truth) ** 2
        net = float(advantage.sum())
        share = 100.0 * float(advantage.max()) / net if net > 0 else float("nan")
        rows.append((name, raw_rmse, overlay_rmse, win, share, qualifies))

    chosen: str | None = None
    if eligible:
        best = min(eligible, key=lambda name: rmse(overlaid[name], truth))
        contenders = [best]
        for name in eligible:
            if name == best:
                continue
            if win_rate(overlaid[name], overlaid[best], truth, 20261902 + len(contenders)) \
                    < SELECTION_CONFIDENCE:
                contenders.append(name)
        chosen = min(contenders, key=lower_capacity_key)

    lines = ["# V4i global-capacity selection", "",
             "Selector: January-August fit, September early stopping, October test.",
             f"LIRF calibration a={coef[0]:.4f}, b={coef[1]:.6f}; overlay touched "
             f"{touched} rows.", "",
             "| configuration | leaves | min data | raw RMSE s | overlay RMSE s | "
             "bootstrap vs incumbent | largest row share | eligible |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, raw_rmse, overlay_rmse, win, share, qualifies in rows:
        cfg = CONFIGS[name]
        win_text = "reference" if np.isnan(win) else f"{100 * win:.2f} %"
        share_text = "reference" if np.isnan(share) else f"{share:.1f} %"
        lines.append(
            f"| {name} | {cfg['num_leaves']} | {cfg['min_data_in_leaf']} | "
            f"{raw_rmse:.3f} | {overlay_rmse:.3f} | {win_text} | {share_text} | "
            f"{'yes' if qualifies else 'no'} |"
        )
    if chosen is None:
        lines += ["", "**No challenger cleared the 98.75 percent multiplicity-corrected "
                         "selection threshold. The branch closes without gate runs.**", ""]
        payload = {"selected": None, "reason": "no challenger eligible"}
    else:
        lines += ["", f"**Frozen selection: {chosen}. Proceed to the July gate.**", ""]
        payload = {"selected": chosen, **CONFIGS[chosen]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.selected.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out} and {args.selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
