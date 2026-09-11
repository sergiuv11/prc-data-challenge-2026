#!/usr/bin/env python3
"""Controlled experiment: a post-model overlay for the LIRF extreme-taxi tail.

Eight rows out of 218,319 carry about a third of the squared error on the composition
set, and at LIRF those rows are reconstructable: when the Network Manager record is
missing, `takeoff_minus_schedule` matches the target almost exactly. A gradient boosted
model with `min_data_in_leaf = 200` cannot fit eight rows, so the correction has to be
applied on top of the model rather than learned inside it.

The overlay is deliberately narrow:

    airport is in SCOPE  and  AOBT_3_flt is missing  and  takeoff_minus_schedule > T
        -> predict takeoff_minus_schedule instead of the model's value

This script measures the rule rather than assuming it. It sweeps the threshold, compares
scopes, checks whether the rule holds in every month of 2025 or only in January, and
counts how many rows it would actually touch in the hidden 2026 ranking set.

Usage:
    .venv/bin/python scripts/experiment_tail_overlay.py
    .venv/bin/python scripts/experiment_tail_overlay.py --predictions reports/preds_composition.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import baselines as B  # noqa: E402
import features as F  # noqa: E402

TARGET = F.TARGET
JULY_AIRPORTS = ["EDDF", "EGLL", "EHAM"]
THRESHOLDS = [3600, 5400, 7200, 10800, 14400, 21600, 28800]
NEEDED = ["MVT_ID_mvt", TARGET, F.TAKEOFF, "AIRPORT", "STAND_mvt", "RUNWAY_mvt", "hour",
          "month", "weekday", "aobt_missing", "takeoff_minus_schedule"]


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def airport_codes(data_dir: Path) -> dict[str, int]:
    """Reproduce the shared categorical encoding used by build_features.py."""
    names = (
        pl.scan_parquet([str(p) for p in sorted(data_dir.glob("training_*.parquet"))] + [str(data_dir / "ranking.parquet")])
        .filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()
    )["ADEP_mvt"].drop_nulls().sort().to_list()
    return {n: i for i, n in enumerate(names)}


def overlay_forms(fit_tms: np.ndarray, fit_y: np.ndarray, tms: np.ndarray,
                  fallback: float) -> dict[str, np.ndarray]:
    """Candidate replacement values for a triggered row.

    `identity` takes the schedule delta at face value. `calibrated` fits
    target ~ a + b * delta on other months, which hedges the rows where a large delta
    means a delayed pushback rather than a long taxi.
    """
    A = np.vstack([np.ones_like(fit_tms), fit_tms]).T
    coef, _, _, _ = np.linalg.lstsq(A, fit_y, rcond=None)
    return {
        "no overlay": np.full_like(tms, fallback),
        "identity": tms,
        "shrunk 0.95": 0.95 * tms + 0.05 * fallback,
        "calibrated": coef[0] + coef[1] * tms,
    }


def leave_one_month_out(lirf_rows: pl.DataFrame, thresholds: list[int], fallback: float) -> list[str]:
    """Score each overlay form on months it was not fitted on."""
    lines = ["## Leave-one-month-out validation of the overlay form", "",
             "Each month is scored by a rule fitted on the other eleven, so the calibration",
             "is never evaluated on its own data.", "",
             "| threshold s | form | rows | SSE removed | share of tail SSE removed | rows made worse |",
             "|---|---|---:|---:|---:|---:|"]
    for t in thresholds:
        sub = lirf_rows.filter(pl.col("takeoff_minus_schedule") > t)
        if sub.height < 24:
            continue
        totals: dict[str, float] = {}
        worse: dict[str, int] = {}
        rows = 0
        for m in range(1, 13):
            te = sub.filter(pl.col("month") == m)
            tr = sub.filter(pl.col("month") != m)
            if te.is_empty() or tr.height < 10:
                continue
            y = te[TARGET].cast(pl.Float64).to_numpy()
            tms = te["takeoff_minus_schedule"].cast(pl.Float64).to_numpy()
            rows += len(y)
            forms = overlay_forms(tr["takeoff_minus_schedule"].cast(pl.Float64).to_numpy(),
                                  tr[TARGET].cast(pl.Float64).to_numpy(), tms, fallback)
            for k, p in forms.items():
                totals[k] = totals.get(k, 0.0) + float(((p - y) ** 2).sum())
                worse[k] = worse.get(k, 0) + int((((p - y) ** 2) > ((fallback - y) ** 2)).sum())
        base = totals["no overlay"]
        for k, v in sorted(totals.items(), key=lambda x: x[1]):
            if k == "no overlay":
                continue
            lines.append(f"| {t:,} | {k} | {rows} | {base - v:.4g} | {100 * (1 - v / base):.2f} % | {worse[k]} |")
    return lines + [""]


def apply_overlay(pred: np.ndarray, df: pl.DataFrame, scope: list[int], threshold: int,
                  coef: tuple[float, float] | None = None) -> tuple[np.ndarray, int]:
    """Replace triggered rows. With `coef`, use the calibrated form instead of the raw delta.

    The calibration must be fitted on months that are not in `df`, otherwise the rule is
    being scored on its own data.
    """
    tms = df["takeoff_minus_schedule"].cast(pl.Float64).to_numpy()
    hit = (
        df["AIRPORT"].is_in(scope).to_numpy()
        & (df["aobt_missing"] == 1).to_numpy()
        & (np.nan_to_num(tms, nan=-1e9) > threshold)
    )
    out = pred.copy()
    out[hit] = tms[hit] if coef is None else coef[0] + coef[1] * tms[hit]
    return out, int(hit.sum())


def fit_calibration(rows: pl.DataFrame, scope: list[int], threshold: int) -> tuple[float, float] | None:
    """Fit target ~ a + b * schedule delta on trigger rows outside the test months."""
    sub = rows.filter(pl.col("AIRPORT").is_in(scope) & (pl.col("aobt_missing") == 1)
                      & (pl.col("takeoff_minus_schedule") > threshold))
    if sub.height < 20:
        return None
    x = sub["takeoff_minus_schedule"].cast(pl.Float64).to_numpy()
    y = sub[TARGET].cast(pl.Float64).to_numpy()
    a = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/train_departures.parquet", type=Path)
    ap.add_argument("--ranking", default="data/features/ranking_departures.parquet", type=Path)
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--predictions", type=Path,
                    help="optional parquet with MVT_ID_mvt and pred, e.g. the LightGBM composition predictions")
    ap.add_argument("--out", default="reports", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    codes = airport_codes(args.data_dir)
    lirf, july = [codes["LIRF"]], [codes[a] for a in JULY_AIRPORTS]
    all_codes = sorted(codes.values())

    full = pl.scan_parquet(str(args.features)).select(NEEDED).collect()
    comp = full.filter((pl.col("month") == 1) | ((pl.col("month") == 7) & pl.col("AIRPORT").is_in(july)))
    truth = comp[TARGET].cast(pl.Float64).to_numpy()
    lines = ["# Experiment: LIRF extreme-taxi overlay", "",
             f"Composition test rows: {comp.height:,}", ""]

    # Baseline predictor: the v1 hierarchical median control, fitted without January or July.
    pool = full.filter((pl.col("month") != 1) & (pl.col("month") != 7))
    gmed = float(pool[TARGET].median())
    base = B.predict(B.fit_lookups(pool), gmed, comp).to_numpy().clip(60, 7200)
    base_rmse = rmse(base, truth)
    lines += [f"Control (hierarchical median) RMSE: **{base_rmse:.3f} s**", ""]

    predictors: list[tuple[str, np.ndarray]] = [("control", base)]
    if args.predictions and args.predictions.exists():
        p = pl.read_parquet(str(args.predictions))
        joined = comp.select("MVT_ID_mvt").join(p, on="MVT_ID_mvt", how="left")
        predictors.append(("LightGBM", joined["pred"].cast(pl.Float64).to_numpy()))
        lines += [f"LightGBM RMSE: **{rmse(predictors[-1][1], truth):.3f} s**", ""]

    # --- threshold and scope sweep -------------------------------------------------
    lines += ["## Threshold, scope and form sweep", "",
              "The calibrated form is fitted on the ten months that are not in the test set,",
              "so it is never scored on its own data.", "",
              "| predictor | scope | form | threshold s | rows replaced | RMSE s | change s |",
              "|---|---|---|---:|---:|---:|---:|"]
    best: tuple[float, str, list[int], int, str] | None = None
    last = predictors[-1][0]
    for pname, pred in predictors:
        p0 = rmse(pred, truth)
        for sname, scope in [("LIRF only", lirf), ("all airports", all_codes)]:
            for t in THRESHOLDS:
                for form in ("identity", "calibrated"):
                    coef = fit_calibration(pool, scope, t) if form == "calibrated" else None
                    if form == "calibrated" and coef is None:
                        continue
                    adj, n = apply_overlay(pred, comp, scope, t, coef)
                    r = rmse(adj, truth)
                    lines.append(f"| {pname} | {sname} | {form} | {t:,} | {n} | {r:.3f} | {r - p0:+.3f} |")
                    if pname == last and (best is None or r < best[0]):
                        best = (r, sname, scope, t, form)
    lines += [""]

    # --- does the rule hold outside January? ---------------------------------------
    lines += ["## Stability: the same rule applied month by month across 2025", "",
              "Fitted control per month is not recomputed; the point is whether replacing these",
              "rows moves RMSE the same way everywhere, or whether January 2025 was lucky.", "",
              "| month | LIRF rows triggered | median abs error of the rule s | worst abs error s |",
              "|---|---:|---:|---:|"]
    for t in [10800]:
        for m in range(1, 13):
            sub = full.filter(
                (pl.col("month") == m) & pl.col("AIRPORT").is_in(lirf) & (pl.col("aobt_missing") == 1)
                & (pl.col("takeoff_minus_schedule") > t)
            )
            if sub.is_empty():
                lines.append(f"| {m:02d} | 0 | n/a | n/a |")
                continue
            err = (sub["takeoff_minus_schedule"].cast(pl.Float64) - sub[TARGET].cast(pl.Float64)).abs()
            lines.append(f"| {m:02d} | {sub.height} | {err.median():.0f} | {err.max():.0f} |")
    lines += [""]

    # --- what would it touch in the real 2026 ranking set? ---------------------------
    rank = pl.scan_parquet(str(args.ranking)).select(
        ["MVT_ID_mvt", "AIRPORT", "month", "weekday", "aobt_missing", "takeoff_minus_schedule"]).collect()
    lines += ["## Exposure on the hidden 2026 ranking set", "",
              "| threshold s | LIRF rows triggered | all-airport rows triggered |", "|---|---:|---:|"]
    for t in THRESHOLDS:
        nl = rank.filter(pl.col("AIRPORT").is_in(lirf) & (pl.col("aobt_missing") == 1)
                         & (pl.col("takeoff_minus_schedule") > t)).height
        na = rank.filter((pl.col("aobt_missing") == 1) & (pl.col("takeoff_minus_schedule") > t)).height
        lines.append(f"| {t:,} | {nl} | {na} |")
    lines += ["", "LIRF appears only in January 2026, so the overlay cannot affect the July rows.", ""]

    # --- which airports show the pattern at all? ------------------------------------
    lines += ["## Is the pattern LIRF specific?", "",
              "Trigger definition: missing NM off-block and a schedule delta above 10,800 s.", "",
              "| airport | trigger rows | median absolute error of the rule s | within 60 s |",
              "|---|---:|---:|---:|"]
    inv = {v: k for k, v in codes.items()}
    for code in sorted(full["AIRPORT"].unique().drop_nulls().to_list()):
        t = full.filter((pl.col("AIRPORT") == code) & (pl.col("aobt_missing") == 1)
                        & (pl.col("takeoff_minus_schedule") > 10800))
        if t.height < 3:
            continue
        e = (t["takeoff_minus_schedule"].cast(pl.Float64) - t[TARGET].cast(pl.Float64)).abs()
        lines.append(f"| {inv.get(code, code)} | {t.height} | {e.median():.0f} | {100 * (e <= 60).mean():.1f} % |")
    lines += [""]

    lirf_rows = full.filter((pl.col("AIRPORT").is_in(lirf)) & (pl.col("aobt_missing") == 1))
    lines += leave_one_month_out(lirf_rows, THRESHOLDS,
                                 float(full.filter(pl.col("AIRPORT").is_in(lirf))[TARGET].median()))

    if best:
        lines += ["## Verdict", "",
                  f"Best configuration on the composition set: {best[1]}, {best[4]} form, "
                  f"threshold {best[3]:,} s, RMSE {best[0]:.3f} s.",
                  "Judge this against the stability table above, not on the composition number alone:",
                  "it is decided by single figures of rows and will not survive if the rule is noisy",
                  "in other months.", ""]

    out = args.out / "experiment_tail_overlay.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
