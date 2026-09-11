#!/usr/bin/env python3
"""CatBoost on the composition split, as a diversity partner for the LightGBM model.

The point is not that CatBoost should beat LightGBM. It is that the two handle categorical
features by genuinely different mechanisms, so their errors should decorrelate and an
ensemble should beat either. LightGBM splits on category sets directly. CatBoost builds
ordered target statistics, a leakage-safe target encoding with its own permutation scheme
baked into boosting, which is the thing the explicit encoding experiment failed to deliver.

Everything here is read-only with respect to the feature cache and to v2. Outputs go to the
isolated paths given on the command line, and CatBoost is told not to write files at all, so
no `catboost_info` directory is ever created.

Memory notes, which are the real risk:

- `max_ctr_complexity=1` forbids target statistics over *combinations* of features. The
  default of 4 would build them over pairs and triples of a 1,900 value stand and a 1,569
  value destination, which is the most likely way this run exhausts the machine.
- `border_count=128` halves the quantisation footprint against the default 254.
- `used_ram_limit` controls categorical target-statistics memory only. It is not an
  operating-system hard cap, so the process is monitored separately during the real fit.

Categorical handling differs from the LightGBM path in one way that matters: CatBoost
rejects NaN in categorical features outright, so codes are filled with -1 and cast to a
signed integer. Numeric NaN is kept, because CatBoost handles it natively.

Usage:
    # synthetic smoke test, seconds, touches no real data
    .venv/bin/python scripts/make_synthetic_fixture.py --out /tmp/syn
    .venv/bin/python scripts/build_features.py --data-dir /tmp/syn --out /tmp/synf
    .venv/bin/python scripts/train_catboost.py --features /tmp/synf --data-dir /tmp/syn \
        --out /tmp/cbsmoke --models-dir /tmp/cbsmoke --iterations 20

    # the real fit, only once authorised
    .venv/bin/python scripts/train_catboost.py --out reports/cb_a --models-dir models/cb_a
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).parent))
import baselines as B  # noqa: E402
import features as F  # noqa: E402

TARGET = F.TARGET
JULY_AIRPORTS = ["EDDF", "EGLL", "EHAM"]
MAX_ITERATIONS = 1500

PARAMS = {
    "loss_function": "RMSE",
    "boosting_type": "Plain",      # Ordered is far heavier at this row count
    "depth": 8,                    # about 256 leaves, against LightGBM's 127
    "learning_rate": 0.06,         # matches the v2 baseline
    "border_count": 128,           # default 254; halves quantisation memory
    "max_ctr_complexity": 1,       # no target statistics over feature combinations
    "one_hot_max_size": 2,
    "random_seed": 20260901,
    "od_type": "Iter",
    "od_wait": 100,
    "allow_writing_files": False,  # never create catboost_info anywhere
    "verbose": 0,
}


def rmse(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def airport_names(data_dir: Path) -> dict[int, str]:
    names = (
        pl.scan_parquet([str(p) for p in sorted(data_dir.glob("training_*.parquet"))]
                        + [str(data_dir / "ranking.parquet")])
        .filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()
    )["ADEP_mvt"].drop_nulls().sort().to_list()
    return {i: n for i, n in enumerate(names)}


def frame(df: pl.DataFrame):
    """Categoricals as integers with no nulls, numerics as float32 with NaN preserved."""
    return df.select(
        [pl.col(c).fill_null(-1).cast(pl.Int32) for c in F.CATEGORICAL]
        + [pl.col(c).cast(pl.Float32) for c in F.NUMERIC]
    ).to_pandas()


def pool(df: pl.DataFrame) -> Pool:
    return Pool(frame(df), df[TARGET].cast(pl.Float64).to_numpy(),
                cat_features=list(range(len(F.CATEGORICAL))))


def report(name: str, df: pl.DataFrame, pred: np.ndarray, names: dict[int, str]) -> list[str]:
    truth = df[TARGET].cast(pl.Float64).to_numpy()
    err2 = (pred - truth) ** 2
    out = [f"### {name}", "", f"- Rows: {len(truth):,}", f"- **Raw RMSE: {rmse(pred, truth):.3f} s**",
           f"- MAE: {float(np.mean(np.abs(pred - truth))):.2f} s, "
           f"median absolute error: {float(np.median(np.abs(pred - truth))):.2f} s"]
    normal = truth <= 3600
    out += [f"- Diagnostic, target at or below 3600 s ({normal.sum():,} rows): "
            f"RMSE {rmse(pred[normal], truth[normal]):.2f} s"]
    if (~normal).any():
        out += [f"- Diagnostic, target above 3600 s ({(~normal).sum():,} rows): "
                f"RMSE {rmse(pred[~normal], truth[~normal]):.2f} s, "
                f"{100 * err2[~normal].sum() / err2.sum():.1f} % of all squared error"]
    else:
        out += ["- Diagnostic, target above 3600 s: no rows"]
    miss = df["aobt_missing"].to_numpy().astype(bool)
    if miss.any():
        out += [f"- Rows without an NM off-block time ({miss.sum():,}): "
                f"RMSE {rmse(pred[miss], truth[miss]):.2f} s; with it: {rmse(pred[~miss], truth[~miss]):.2f} s"]
    frame_ = df.select(["month", "AIRPORT"]).with_columns(pl.Series("se", err2))
    out += ["", "| month | rows | RMSE s |", "|---|---:|---:|"]
    for r in frame_.group_by("month").agg(pl.len().alias("n"), pl.col("se").mean().alias("mse")).sort("month").iter_rows(named=True):
        out.append(f"| {int(r['month']):02d} | {r['n']:,} | {np.sqrt(r['mse']):.2f} |")
    out += ["", "| airport | month | rows | RMSE s |", "|---|---|---:|---:|"]
    agg = frame_.group_by(["AIRPORT", "month"]).agg(pl.len().alias("n"), pl.col("se").mean().alias("mse"))
    for r in agg.sort([pl.col("mse").sqrt()], descending=True).iter_rows(named=True):
        out.append(f"| {names.get(int(r['AIRPORT']), r['AIRPORT'])} | {int(r['month']):02d} | "
                   f"{r['n']:,} | {np.sqrt(r['mse']):.2f} |")
    return out + [""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features", type=Path)
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--out", default="reports/cb_a", type=Path)
    ap.add_argument("--models-dir", default="models/cb_a", type=Path)
    ap.add_argument("--report-name", default="model_catboost.md")
    ap.add_argument("--reference-predictions", type=Path,
                    help="Optional prediction file whose composition IDs must match before fitting")
    ap.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--used-ram-limit", default="5gb")
    args = ap.parse_args()

    if not 1 <= args.iterations <= MAX_ITERATIONS:
        ap.error(f"--iterations must be between 1 and {MAX_ITERATIONS}")

    train_path = args.features / "train_departures.parquet"
    if not train_path.exists():
        print(f"ERROR: {train_path} missing. Run scripts/build_features.py first.", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    params = dict(PARAMS)
    params.update(iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
                  used_ram_limit=args.used_ram_limit, thread_count=8,
                  # Belt and braces: files are disabled, but if that ever changes the
                  # directory must still be inside the isolated output, never the repo root.
                  train_dir=str(args.out / "catboost_info"))

    names = airport_names(args.data_dir)
    code = {v: k for k, v in names.items()}
    july = [code[a] for a in JULY_AIRPORTS if a in code]

    full = pl.read_parquet(train_path)
    comp = (pl.col("month") == 1) | ((pl.col("month") == 7) & pl.col("AIRPORT").is_in(july))
    pool_rows = full.filter((pl.col("month") != 1) & (pl.col("month") != 7))
    fit = pool_rows.filter(pl.col("month") != 12)
    inner = pool_rows.filter(pl.col("month") == 12)
    test = full.filter(comp)
    del full, pool_rows
    gc.collect()
    print(f"CatBoost | depth {args.depth}, lr {args.learning_rate}, max iterations {args.iterations}, "
          f"ram limit {args.used_ram_limit}")
    print(f"fit {fit.height:,} | inner {inner.height:,} | test {test.height:,}")
    if min(fit.height, inner.height, test.height) == 0:
        print("ERROR: a split is empty; check the feature table and airport codes.", file=sys.stderr)
        return 1
    if args.reference_predictions is not None:
        reference = pl.read_parquet(args.reference_predictions, columns=["MVT_ID_mvt"])
        test_ids = test.select("MVT_ID_mvt")
        if reference.height != test_ids.height or not reference.equals(test_ids):
            print("ERROR: composition IDs do not exactly match the reference predictions.",
                  file=sys.stderr)
            return 1
        print(f"  verified {test_ids.height:,} composition IDs against {args.reference_predictions}")

    gmed = float(fit[TARGET].median())
    lookups = B.fit_lookups(fit)

    t0 = time.perf_counter()
    dtrain, dvalid = pool(fit), pool(inner)
    del fit
    gc.collect()
    t_pool = time.perf_counter() - t0
    print(f"  pools built in {t_pool:.1f} s")

    t0 = time.perf_counter()
    model = CatBoostRegressor(**params)
    model.fit(dtrain, eval_set=dvalid, use_best_model=True)
    t_fit = time.perf_counter() - t0
    best_raw = model.get_best_iteration()
    best = args.iterations if best_raw is None or best_raw < 0 else int(best_raw)
    print(f"  trained {model.tree_count_} trees (best iteration {best}) in {t_fit:.1f} s")
    del dtrain, dvalid
    gc.collect()

    model.save_model(str(args.models_dir / "catboost.cbm"))
    pred = np.clip(model.predict(frame(test)), 60, None)
    truth = test[TARGET].cast(pl.Float64).to_numpy()

    control = B.predict(lookups, gmed, test).to_numpy().clip(60, 7200)

    # Same schema the LightGBM runs emit, so compare_predictions.py works unchanged.
    test.select("MVT_ID_mvt").with_columns(
        pl.Series("pred", pred), pl.Series("control", control)
    ).write_parquet(args.out / "preds_composition.parquet")

    # Inner-month predictions, kept so an ensemble weight can later be fitted on rows that
    # are inside the training pool and never part of the composition test.
    inner_pred = np.clip(model.predict(frame(inner)), 60, None)
    inner.select("MVT_ID_mvt").with_columns(pl.Series("pred", inner_pred)).write_parquet(
        args.out / "preds_inner_month12.parquet")

    lines = ["# CatBoost on the composition split", "",
             f"Parameters: depth {args.depth}, learning rate {args.learning_rate}, "
             f"border_count {params['border_count']}, max_ctr_complexity {params['max_ctr_complexity']}, "
             f"boosting {params['boosting_type']}, ram limit {args.used_ram_limit}.",
             f"Trees {model.tree_count_}, best iteration {best}. "
             f"Pools {t_pool:.1f} s, fit {t_fit:.1f} s.", "",
             f"Control (hierarchical median) RMSE: {rmse(control, truth):.3f} s", ""]
    lines += report("CatBoost", test, pred, names)
    lines += report("Control, hierarchical median", test, control, names)

    imp = sorted(zip(F.CATEGORICAL + F.NUMERIC, model.get_feature_importance()), key=lambda x: -x[1])[:15]
    tot = float(sum(model.get_feature_importance())) or 1.0
    lines += ["### Top features by importance", "", "| feature | share |", "|---|---:|"]
    lines += [f"| `{n}` | {100 * g / tot:.1f} % |" for n, g in imp] + [""]
    lines += ["Judge this against the LightGBM baseline with:", "",
              "```", f"scripts/compare_predictions.py --candidate {args.out}/preds_composition.parquet \\",
              "    --label 'CatBoost' --gate-policy shipping", "```", ""]
    (args.out / args.report_name).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCatBoost RMSE {rmse(pred, truth):.3f} s | control {rmse(control, truth):.3f} s")
    print(f"Wrote {args.out / args.report_name}, preds_composition.parquet, preds_inner_month12.parquet")
    print(f"      {args.models_dir / 'catboost.cbm'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
