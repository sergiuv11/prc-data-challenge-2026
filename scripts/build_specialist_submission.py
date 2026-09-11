#!/usr/bin/env python3
"""Build the gated v3 submission from v2 plus three airport specialists.

This refuses to continue unless the preserved v2 model and unchanged LIRF overlay reproduce
the existing rounded v2 submission exactly. It never uploads the generated file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import experiment_tail_overlay as O
import features as F
import train_model as T

AIRPORTS = ("EDDF", "EGLL", "EHAM")
ID = "MVT_ID_mvt"
TARGET = F.TARGET
ROUNDS = 436
BLEND_WEIGHT = 0.5
OVERLAY_THRESHOLD = 14_400


def submission_frame(template: pl.DataFrame, ranking: pl.DataFrame,
                     prediction: np.ndarray, fallback: float) -> pl.DataFrame:
    if len(prediction) != ranking.height or not np.isfinite(prediction).all():
        raise ValueError("prediction length or finiteness check failed")
    dtype = template.schema[TARGET]
    output = (
        template.drop(TARGET)
        .join(ranking.select(ID).with_columns(pl.Series("pred", prediction)),
              on=ID, how="left", validate="1:1")
        .with_columns(
            pl.col("pred").fill_null(fallback).round(0).cast(dtype).alias(TARGET)
        )
        .drop("pred")
    )
    if output.height != template.height or not output[ID].equals(template[ID]):
        raise ValueError("submission row identity or order differs from the template")
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features", type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--global-model", default="models/lgbm_v2.txt", type=Path)
    parser.add_argument("--v2-submission", default="submissions/jubilant-vase_v2.parquet",
                        type=Path)
    parser.add_argument("--out", default="submissions/jubilant-vase_v3.parquet", type=Path)
    parser.add_argument("--models-dir", default="models/airport_v3", type=Path)
    parser.add_argument("--audit", default="reports/airport_v3/build_audit.json", type=Path)
    parser.add_argument("--verify-v2-only", action="store_true",
                        help="stop after proving the preserved model exactly rebuilds v2")
    args = parser.parse_args()

    required = [args.features / "train_departures.parquet",
                args.features / "ranking_departures.parquet",
                args.data_dir / "submitting.parquet", args.global_model, args.v2_submission]
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise FileNotFoundError(f"required files are missing: {', '.join(absent)}")

    T.FEATURE_SET = list(F.FEATURES)
    T.PARAMS["learning_rate"] = 0.06
    names = T.init_airport_codes(args.data_dir)
    codes = {name: code for code, name in names.items() if name in AIRPORTS}
    if set(codes) != set(AIRPORTS):
        raise ValueError("airport mapping does not contain all three specialists")

    full = pl.read_parquet(args.features / "train_departures.parquet")
    ranking = pl.read_parquet(args.features / "ranking_departures.parquet")
    template = pl.read_parquet(args.data_dir / "submitting.parquet")
    fallback = float(full[TARGET].median())

    global_model = lgb.Booster(model_file=str(args.global_model))
    if global_model.feature_name() != list(F.FEATURES) or global_model.num_trees() != ROUNDS:
        raise ValueError("preserved global model does not match the gated v2 configuration")
    global_raw = np.clip(global_model.predict(T.as_matrix(ranking)), 60, None)

    overlay_codes = O.airport_codes(args.data_dir)
    lirf_scope = [overlay_codes["LIRF"]]
    coefficient = O.fit_calibration(full, lirf_scope, OVERLAY_THRESHOLD)
    if coefficient is None:
        raise ValueError("the preserved LIRF overlay cannot be calibrated")
    global_overlay, v2_touched = O.apply_overlay(
        global_raw, ranking, lirf_scope, OVERLAY_THRESHOLD, coefficient
    )
    global_overlay = np.clip(global_overlay, 60, None)
    reproduced_v2 = submission_frame(template, ranking, global_overlay, fallback)
    preserved_v2 = pl.read_parquet(args.v2_submission)
    if reproduced_v2.columns != preserved_v2.columns:
        raise ValueError("reproduced v2 columns differ from the preserved submission")
    if (not reproduced_v2[ID].equals(preserved_v2[ID])
            or not reproduced_v2[TARGET].equals(preserved_v2[TARGET])):
        different = int((reproduced_v2[TARGET] != preserved_v2[TARGET]).sum())
        raise ValueError(f"v2 reproduction failed on {different:,} rounded predictions")
    print(f"Exact supplied V2 foundation reproduction PASS: {preserved_v2.height:,} rows, "
          f"{v2_touched} overlay replacements", flush=True)
    if args.verify_v2_only:
        return 0

    candidate = global_raw.copy()
    model_audit = []
    args.models_dir.mkdir(parents=True, exist_ok=True)
    for airport in AIRPORTS:
        code = codes[airport]
        local_train = full.filter(pl.col("AIRPORT") == code)
        mask = ranking["AIRPORT"].to_numpy() == code
        local_ranking = ranking.filter(pl.col("AIRPORT") == code)
        if local_train.is_empty() or not mask.any() or local_ranking.height != int(mask.sum()):
            raise ValueError(f"{airport} has inconsistent train or ranking rows")

        started = time.perf_counter()
        dataset = T.dataset(local_train)
        specialist = lgb.train(T.PARAMS, dataset, num_boost_round=ROUNDS)
        local_prediction = np.clip(specialist.predict(T.as_matrix(local_ranking)), 60, None)
        candidate[mask] = ((1.0 - BLEND_WEIGHT) * global_raw[mask]
                           + BLEND_WEIGHT * local_prediction)
        elapsed = time.perf_counter() - started
        model_path = args.models_dir / f"{airport}_final.txt"
        specialist.save_model(str(model_path))
        model_audit.append({"airport": airport, "train_rows": local_train.height,
                            "ranking_rows": local_ranking.height,
                            "training_seconds": elapsed, "model": str(model_path)})
        print(f"{airport}: trained {ROUNDS} trees on {local_train.height:,} rows, "
              f"predicted {local_ranking.height:,} rows in {elapsed:.1f} s", flush=True)
        del dataset, specialist, local_train, local_ranking

    candidate_overlay, v3_touched = O.apply_overlay(
        candidate, ranking, lirf_scope, OVERLAY_THRESHOLD, coefficient
    )
    candidate_overlay = np.clip(candidate_overlay, 60, None)
    submission = submission_frame(template, ranking, candidate_overlay, fallback)
    changed = int((submission[TARGET] != preserved_v2[TARGET]).sum())
    if changed == 0:
        raise ValueError("v3 is identical to v2 after rounding")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.write_parquet(args.out, compression="zstd")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "configuration": {"airports": list(AIRPORTS), "blend_weight": BLEND_WEIGHT,
                          "features": len(F.FEATURES), "learning_rate": 0.06,
                          "rounds": ROUNDS, "overlay_threshold": OVERLAY_THRESHOLD},
        "v2_exact_reproduction": True,
        "rows": submission.height,
        "changed_rounded_predictions_vs_v2": changed,
        "overlay_replacements": v3_touched,
        "overlay_coefficient": {"intercept": coefficient[0], "slope": coefficient[1]},
        "prediction_seconds": {"min": float(submission[TARGET].min()),
                               "median": float(submission[TARGET].median()),
                               "mean": float(submission[TARGET].mean()),
                               "p99": float(submission[TARGET].quantile(0.99)),
                               "max": float(submission[TARGET].max())},
        "models": model_audit,
        "submission": str(args.out),
        "sha256": sha256(args.out),
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} and {args.audit}")
    print(f"Changed rounded predictions versus v2: {changed:,}")
    print(f"SHA-256: {audit['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
