#!/usr/bin/env python3
"""Dedicated model for departures with no Network Manager off-block record.

Median separation between tail and normal rows is weak, but medians cannot see nonlinear
interactions, and a tree model can. This trains directly on the missing-AOBT subset, where
the shared model is handicapped: for these rows every NM derived feature is null, so the
features that carry 57 % of the main model's gain are simply absent, and the model spends
its capacity on rows that do have them.

The subset is small, roughly twenty thousand rows, so this costs seconds rather than the
quarter hour a full model takes.

Protocol, fixed before running:

- Train on missing-AOBT rows from the composition fit months only, the same rows the
  baseline model was fitted on.
- Early stop on the missing-AOBT rows of the inner month, never on the test set.
- Use only features that actually carry values for this population, chosen by measured null
  rate rather than by assumption.
- Score by substituting these predictions for the missing-AOBT rows of the preserved
  composition predictions, leaving all other rows exactly as the baseline produced them.
- Apply the LIRF overlay identically to both arms, so the comparison isolates the subset
  model.

Stopping rule, predeclared: keep only if the substituted result beats the pool-fitted
composition equivalent and wins at least 95 % of paired bootstrap resamples. Otherwise
close the branch and do not build the classifier hurdle.

Usage:
    .venv/bin/python scripts/experiment_missing_aobt_model.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import experiment_tail_overlay as OV  # noqa: E402
import features as F  # noqa: E402

TARGET = F.TARGET
JULY = ["EDDF", "EGLL", "EHAM"]
OVERLAY_T = 14400
# The overlay coefficients must be fitted on the split's own pool. v2 uses coefficients
# fitted on all twelve months, which include January and July, so reusing them here would
# leak the test months into the evaluation. Both arms get the same pool-fitted values, so
# the overlay cancels in the comparison.

# Tuned for a subset roughly seventy times smaller than the full training set.
PARAMS = {
    "objective": "regression", "metric": "rmse", "learning_rate": 0.05,
    "num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "num_threads": 8, "verbosity": -1, "seed": 20260901,
}


def rmse(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def apply_overlay(pred: np.ndarray, ap: np.ndarray, lirf: int, miss: np.ndarray,
                  tms: np.ndarray, coef: tuple[float, float]) -> np.ndarray:
    hit = (ap == lirf) & (miss == 1) & (np.nan_to_num(tms, nan=-1e9) > OVERLAY_T)
    out = pred.copy()
    out[hit] = np.clip(coef[0] + coef[1] * tms[hit], 60, None)
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--features", default="data/features/train_departures.parquet", type=Path)
    ap_.add_argument("--preds", default="reports/preds_composition.parquet", type=Path)
    ap_.add_argument("--data-dir", default="data/clean", type=Path)
    ap_.add_argument("--out", default="reports/miss_b", type=Path)
    ap_.add_argument("--models-dir", default="models/miss_b", type=Path)
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    names = (
        pl.scan_parquet([str(p) for p in sorted(args.data_dir.glob("training_*.parquet"))]
                        + [str(args.data_dir / "ranking.parquet")])
        .filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()
    )["ADEP_mvt"].drop_nulls().sort().to_list()
    code = {n: i for i, n in enumerate(names)}
    lirf, july = code["LIRF"], [code[a] for a in JULY]

    full = pl.read_parquet(args.features)
    miss = full.filter(pl.col("aobt_missing") == 1)

    # Which features actually carry values for this population? Measured, not assumed.
    usable = [c for c in F.FEATURES
              if c != "aobt_missing" and miss[c].null_count() / miss.height < 0.5]
    dropped = [c for c in F.FEATURES if c not in usable and c != "aobt_missing"]
    print(f"Missing-AOBT rows in the whole training set: {miss.height:,}")
    print(f"Usable features: {len(usable)}  (dropped {len(dropped)} that are null for this population)")
    print(f"  dropped: {', '.join(dropped)}\n")

    comp_mask = (pl.col("month") == 1) | ((pl.col("month") == 7) & pl.col("AIRPORT").is_in(july))
    # Overlay calibration from the composition fit pool only, never from the test months.
    overlay_coef = OV.fit_calibration(full.filter((pl.col("month") != 1) & (pl.col("month") != 7)),
                                      [lirf], OVERLAY_T)
    print(f"Overlay calibration fitted on the composition pool: a={overlay_coef[0]:.4f}, b={overlay_coef[1]:.6f}")
    print("  (v2 ships a={:.1f}, b={:.4f}, fitted on all 12 months, unusable here)\n".format(-5986.9, 1.1833))
    pool = miss.filter((pl.col("month") != 1) & (pl.col("month") != 7))
    fit = pool.filter(pl.col("month") != 12)
    inner = pool.filter(pl.col("month") == 12)
    test = miss.filter(comp_mask)
    print(f"fit {fit.height:,} | inner {inner.height:,} | test {test.height:,}")

    def mat(d: pl.DataFrame) -> np.ndarray:
        return d.select([pl.col(c).fill_null(-1).cast(pl.Float32) if c in F.CATEGORICAL
                         else pl.col(c).cast(pl.Float32) for c in usable]).to_numpy()

    t0 = time.perf_counter()
    dtrain = lgb.Dataset(mat(fit), label=fit[TARGET].cast(pl.Float64).to_numpy(),
                         feature_name=usable, categorical_feature=[c for c in usable if c in F.CATEGORICAL],
                         free_raw_data=False)
    dvalid = lgb.Dataset(mat(inner), label=inner[TARGET].cast(pl.Float64).to_numpy(),
                         reference=dtrain, free_raw_data=False)
    booster = lgb.train(PARAMS, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    best = booster.best_iteration or 3000
    elapsed = time.perf_counter() - t0
    print(f"trained {best} trees in {elapsed:.1f} s\n")
    booster.save_model(str(args.models_dir / "lgbm_missing_aobt.txt"), num_iteration=best)

    sub_pred = np.clip(booster.predict(mat(test), num_iteration=best), 60, None)

    # Substitute into the preserved composition predictions, changing nothing else.
    base = pl.read_parquet(args.preds)
    cols = ["MVT_ID_mvt", TARGET, "AIRPORT", "aobt_missing", "takeoff_minus_schedule"]
    joined = full.select(cols).join(base, on="MVT_ID_mvt", how="inner")
    y = joined[TARGET].cast(pl.Float64).to_numpy()
    p_base = joined["pred"].to_numpy()
    apc = joined["AIRPORT"].to_numpy()
    mflag = joined["aobt_missing"].to_numpy()
    tms = joined["takeoff_minus_schedule"].cast(pl.Float64).to_numpy()

    idx = {int(m): i for i, m in enumerate(joined["MVT_ID_mvt"].to_numpy())}
    p_new = p_base.copy()
    for m, v in zip(test["MVT_ID_mvt"].to_numpy(), sub_pred):
        p_new[idx[int(m)]] = v

    lines = ["# Experiment: dedicated model for missing-AOBT departures", "",
             f"Subset training rows {fit.height:,}, inner {inner.height:,}, test {test.height:,}.",
             f"Trained {best} trees in {elapsed:.1f} s on {len(usable)} usable features.", ""]
    lines += ["| arm | overall RMSE s | missing-population RMSE s |", "|---|---:|---:|"]
    mm = mflag == 1
    rows = []
    for label, p in [("baseline (v2 model, no overlay)", p_base),
                     ("baseline + pool-fitted overlay = composition equivalent", apply_overlay(p_base, apc, lirf, mflag, tms, overlay_coef)),
                     ("subset model substituted, no overlay", p_new),
                     ("subset substituted + pool-fitted overlay", apply_overlay(p_new, apc, lirf, mflag, tms, overlay_coef))]:
        rows.append((label, rmse(p, y), rmse(p[mm], y[mm])))
        lines.append(f"| {label} | {rmse(p, y):.3f} | {rmse(p[mm], y[mm]):.2f} |")
        print(f"{label:<44} overall {rmse(p, y):8.3f}   missing {rmse(p[mm], y[mm]):9.2f}")

    # Paired bootstrap, both arms carrying the overlay so it cancels.
    a = apply_overlay(p_base, apc, lirf, mflag, tms, overlay_coef)
    b = apply_overlay(p_new, apc, lirf, mflag, tms, overlay_coef)
    rng = np.random.default_rng(20260901)
    se_a, se_b, n = (a - y) ** 2, (b - y) ** 2, len(y)
    d = np.array([np.sqrt(se_a[i].mean()) - np.sqrt(se_b[i].mean())
                  for i in (rng.integers(0, n, n) for _ in range(1000))])
    win = float((d > 0).mean())
    lines += ["", f"- Paired bootstrap, 1,000 resamples: the subset model wins {100*win:.1f} % of them.",
              f"- RMSE difference, 5th to 95th percentile: {np.percentile(d,5):+.2f} s to {np.percentile(d,95):+.2f} s.", ""]
    print(f"\nbootstrap win rate {100*win:.1f} %   5th-95th {np.percentile(d,5):+.2f} to {np.percentile(d,95):+.2f} s")

    keep = rows[3][1] < rows[1][1] and win >= 0.95
    lines += [f"**Predeclared rule: keep only if it beats the composition equivalent and wins at least 95 % of "
              f"resamples. Verdict: {'KEEP' if keep else 'REJECT, close the branch'}.**", ""]
    print(f"\nVERDICT: {'KEEP' if keep else 'REJECT, close the branch'}")

    imp = sorted(zip(usable, booster.feature_importance("gain")), key=lambda x: -x[1])[:10]
    tot = sum(booster.feature_importance("gain")) or 1.0
    lines += ["## Top features on this population", "", "| feature | share of gain |", "|---|---:|"]
    lines += [f"| `{n}` | {100*g/tot:.1f} % |" for n, g in imp] + [""]
    (args.out / "experiment_missing_aobt.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {args.out / 'experiment_missing_aobt.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
