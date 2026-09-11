#!/usr/bin/env python3
"""Combine a candidate global prediction with the preserved V3 specialist boosters.

This changes only the global model. EDDF, EGLL and EHAM keep their original 43-feature
specialists and fixed 50/50 blend. The script first reproduces the preserved baseline
blend from those boosters, which makes model/file mismatches fail before output is written.
"""
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


def load_predictions(path: Path, expected_ids: pl.Series, label: str) -> np.ndarray:
    frame = pl.read_parquet(path).select(ID, PRED)
    if frame[ID].is_null().any() or frame[ID].is_duplicated().any():
        raise ValueError(f"{label} has null or duplicate movement IDs")
    if frame.height != len(expected_ids) or not frame[ID].equals(expected_ids):
        raise ValueError(f"{label} IDs or row order do not match the selected split")
    values = frame[PRED].cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite predictions")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features", type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--split", required=True,
                        choices=["composition", "forward", "december", "october_selection"])
    parser.add_argument("--candidate-global", required=True, type=Path)
    parser.add_argument("--baseline-global", required=True, type=Path)
    parser.add_argument("--baseline-complete", required=True, type=Path)
    parser.add_argument("--specialist-models", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-targeted-only", action="store_true",
                        help="assert the supplied complete baseline only on the three specialist "
                             "airports; used once to extract V3 from an all-airport blend")
    args = parser.parse_args()

    T.FEATURE_SET = list(F.FEATURES)
    names = T.init_airport_codes(args.data_dir)
    codes = {name: code for code, name in names.items() if name in AIRPORTS}
    if set(codes) != set(AIRPORTS):
        raise ValueError("airport code mapping is incomplete")

    full = pl.read_parquet(args.features / "train_departures.parquet")
    _, test = E.split_frames(full, args.split)
    ids = test[ID]
    base_global = load_predictions(args.baseline_global, ids, "baseline global")
    candidate_global = load_predictions(args.candidate_global, ids, "candidate global")
    preserved_complete = load_predictions(args.baseline_complete, ids, "baseline complete")

    reproduced = base_global.copy()
    candidate = candidate_global.copy()
    targeted = np.zeros(test.height, dtype=bool)
    for airport in AIRPORTS:
        code = codes[airport]
        mask = test["AIRPORT"].to_numpy() == code
        if not mask.any():
            raise ValueError(f"{airport} has no rows in {args.split}")
        targeted |= mask
        model_path = args.specialist_models / f"{airport}_{args.split}.txt"
        booster = lgb.Booster(model_file=str(model_path))
        if booster.feature_name() != list(F.FEATURES):
            raise ValueError(f"{model_path} feature names or order do not match V3")
        local = test.filter(pl.col("AIRPORT") == code)
        specialist = np.clip(booster.predict(T.as_matrix(local)), 0, None)
        reproduced[mask] = 0.5 * base_global[mask] + 0.5 * specialist
        candidate[mask] = 0.5 * candidate_global[mask] + 0.5 * specialist

    check_mask = targeted if args.baseline_targeted_only else np.ones(test.height, dtype=bool)
    maximum_delta = float(np.max(np.abs(reproduced[check_mask] - preserved_complete[check_mask])))
    if maximum_delta > 1e-9:
        raise ValueError(f"preserved V3 reproduction failed, maximum delta {maximum_delta:.12g}")
    if not np.isfinite(candidate).all():
        raise ValueError("combined candidate contains non-finite predictions")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    test.select(ID).with_columns(pl.Series(PRED, candidate)).write_parquet(args.out)
    written = pl.read_parquet(args.out)
    if written.height != test.height or not written[ID].equals(ids):
        raise ValueError("written IDs or row order changed")
    print(f"PASS: reproduced preserved V3 to {maximum_delta:.3g} seconds")
    print(f"Wrote {args.out} with {test.height:,} complete-system predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
