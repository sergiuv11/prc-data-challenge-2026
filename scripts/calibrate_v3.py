#!/usr/bin/env python3
"""Fit per-airport affine calibration on a buffer month and apply it to V3 predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import experiment_airport_specialists as E
import features as F
import train_model as T

AIRPORTS = ("EDDF", "EGLL", "EHAM")
ID = "MVT_ID_mvt"
PRED = "pred"


def checked_booster(path: Path) -> lgb.Booster:
    booster = lgb.Booster(model_file=str(path))
    if booster.feature_name() != list(F.FEATURES):
        raise ValueError(f"{path} feature names or order do not match V3")
    return booster


def complete_prediction(
    frame: pl.DataFrame,
    global_model: lgb.Booster,
    specialist_dir: Path,
    suffix: str,
    codes: dict[str, int],
) -> np.ndarray:
    prediction = np.clip(global_model.predict(T.as_matrix(frame)), 0, None)
    for airport in AIRPORTS:
        mask = frame["AIRPORT"].to_numpy() == codes[airport]
        if not mask.any():
            continue
        specialist = checked_booster(specialist_dir / f"{airport}_{suffix}.txt")
        local = frame.filter(pl.col("AIRPORT") == codes[airport])
        local_prediction = np.clip(specialist.predict(T.as_matrix(local)), 0, None)
        prediction[mask] = 0.5 * prediction[mask] + 0.5 * local_prediction
    return prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features", type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--split", required=True,
                        choices=["october_selection", "forward", "december", "composition"])
    parser.add_argument("--calibration-month", required=True, type=int)
    parser.add_argument("--global-model", required=True, type=Path)
    parser.add_argument("--specialist-models", required=True, type=Path)
    parser.add_argument("--specialist-suffix", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.calibration_month <= 12:
        parser.error("--calibration-month must be between 1 and 12")

    T.FEATURE_SET = list(F.FEATURES)
    names = T.init_airport_codes(args.data_dir)
    codes = {name: code for code, name in names.items()}
    if not set(AIRPORTS).issubset(codes):
        raise ValueError("specialist airport mapping is incomplete")

    full = pl.read_parquet(args.features / "train_departures.parquet")
    calibration = full.filter(pl.col("month") == args.calibration_month)
    _, test = E.split_frames(full, args.split)
    if calibration.is_empty() or test.is_empty():
        raise ValueError("calibration or test frame is empty")

    baseline = pl.read_parquet(args.baseline).select(ID, PRED)
    if baseline.height != test.height or not baseline[ID].equals(test[ID]):
        raise ValueError("baseline IDs or row order do not match the selected split")
    base = baseline[PRED].cast(pl.Float64).to_numpy()
    if not np.isfinite(base).all():
        raise ValueError("baseline contains non-finite predictions")

    global_model = checked_booster(args.global_model)
    calibration_prediction = complete_prediction(
        calibration, global_model, args.specialist_models, args.specialist_suffix, codes
    )
    truth = calibration[F.TARGET].cast(pl.Float64).to_numpy()
    candidate = base.copy()
    coefficient_rows = []
    for airport, code in sorted(codes.items()):
        fit_mask = calibration["AIRPORT"].to_numpy() == code
        test_mask = test["AIRPORT"].to_numpy() == code
        if fit_mask.sum() < 1000 or not test_mask.any():
            raise ValueError(f"insufficient calibration or test rows for {airport}")
        slope, intercept = np.polyfit(calibration_prediction[fit_mask], truth[fit_mask], 1)
        if not np.isfinite([slope, intercept]).all():
            raise ValueError(f"non-finite calibration for {airport}")
        candidate[test_mask] = intercept + slope * base[test_mask]
        coefficient_rows.append((airport, int(fit_mask.sum()), float(intercept), float(slope)))

    identity = 0.0 + 1.0 * base
    if not np.array_equal(identity, base):
        raise ValueError("identity transform did not exactly reproduce preserved V3")
    before_clip = candidate.copy()
    candidate = np.clip(candidate, 0, None)
    clipped = int(np.count_nonzero(before_clip < 0))
    if not np.isfinite(candidate).all():
        raise ValueError("calibrated candidate contains non-finite predictions")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    test.select(ID).with_columns(pl.Series(PRED, candidate)).write_parquet(args.out)
    written = pl.read_parquet(args.out)
    if written.height != test.height or not written[ID].equals(test[ID]):
        raise ValueError("written IDs or row order changed")

    lines = [f"# V3 affine calibration: {args.split}", "",
             f"Calibration month: {args.calibration_month}. Rows: {calibration.height:,}.",
             f"Test rows: {test.height:,}. Predictions clipped at zero: {clipped:,}.", "",
             "| airport | calibration rows | intercept | slope |",
             "|---|---:|---:|---:|"]
    lines.extend(
        f"| {airport} | {rows:,} | {intercept:.6f} | {slope:.8f} |"
        for airport, rows, intercept, slope in coefficient_rows
    )
    lines += ["", "Identity coefficients reproduce the preserved V3 array exactly.", ""]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
