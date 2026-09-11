#!/usr/bin/env python3
"""Compare a candidate's composition predictions against the preserved baseline.

Every candidate from here on is judged the same way, by the same code, so results are
comparable across experiments and no comparison can quietly use a different rule.

The LIRF overlay is fitted on the composition **fit pool** only, months other than January
and July, and applied identically to both arms. v2 ships coefficients fitted on all twelve
months; reusing those here would leak the test months into the evaluation.

Predeclared gate: a candidate must improve raw composition RMSE **and** overlay adjusted
composition RMSE, and win the paired bootstrap convincingly. Forward and seasonal
confirmation is required before any submission is built.

Usage:
    .venv/bin/python scripts/compare_predictions.py --candidate reports/lr_004/preds_composition.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import experiment_tail_overlay as OV  # noqa: E402
import features as F  # noqa: E402

TARGET = F.TARGET
JULY = ["EDDF", "EGLL", "EHAM"]
OVERLAY_T = 14400
BOOTSTRAP = 2000


def rmse(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="reports/preds_composition.parquet", type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--features", default="data/features/train_departures.parquet", type=Path)
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gate-policy", choices=["both", "shipping"], default="both",
                    help="both requires 95 percent bootstrap confidence in raw and overlay "
                         "arms; shipping requires raw and overlay point improvements but "
                         "gates confidence on the overlay architecture that is submitted")
    candidate_scope = ap.add_mutually_exclusive_group()
    candidate_scope.add_argument(
        "--candidate-airports",
        help="comma-separated airports allowed to change; all other predictions must "
             "remain exactly equal",
    )
    candidate_scope.add_argument(
        "--candidate-wake-categories",
        help="comma-separated raw wake categories allowed to change; all other predictions "
             "must remain exactly equal",
    )
    ap.add_argument("--maximum-row-share", type=float, default=50.0,
                    help="maximum percentage of net overlay SSE advantage from one row")
    args = ap.parse_args()

    if not 0.0 < args.maximum_row_share <= 100.0:
        ap.error("--maximum-row-share must be greater than 0 and at most 100")

    names = (
        pl.scan_parquet([str(p) for p in sorted(args.data_dir.glob("training_*.parquet"))]
                        + [str(args.data_dir / "ranking.parquet")])
        .filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()
    )["ADEP_mvt"].drop_nulls().sort().to_list()
    code = {n: i for i, n in enumerate(names)}
    lirf = code["LIRF"]

    wake_names = (
        pl.scan_parquet([str(p) for p in sorted(args.data_dir.glob("training_*.parquet"))]
                        + [str(args.data_dir / "ranking.parquet")])
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("WK_TBL_CAT_flt").unique().collect()
    )["WK_TBL_CAT_flt"].drop_nulls().sort().to_list()
    wake_code = {name: value for value, name in enumerate(wake_names)}

    full = pl.read_parquet(args.features, columns=[
        "MVT_ID_mvt", TARGET, "AIRPORT", "WK_TBL_CAT_flt", "month", "aobt_missing",
        "takeoff_minus_schedule"])
    coef = OV.fit_calibration(full.filter((pl.col("month") != 1) & (pl.col("month") != 7)),
                              [lirf], OVERLAY_T)
    print(f"Overlay calibration from the composition fit pool: a={coef[0]:.4f}, b={coef[1]:.6f}")

    base = pl.read_parquet(args.baseline).select(["MVT_ID_mvt", "pred"]).rename({"pred": "base"})
    cand = pl.read_parquet(args.candidate).select(["MVT_ID_mvt", "pred"]).rename({"pred": "cand"})
    j = full.join(base, on="MVT_ID_mvt", how="inner").join(cand, on="MVT_ID_mvt", how="inner")
    if j.height != base.height or j.height != cand.height:
        print(f"ERROR: row mismatch. baseline {base.height:,}, candidate {cand.height:,}, "
              f"joined {j.height:,}", file=sys.stderr)
        return 1

    y = j[TARGET].cast(pl.Float64).to_numpy()
    tms = j["takeoff_minus_schedule"].cast(pl.Float64).to_numpy()
    hit = (j["AIRPORT"].to_numpy() == lirf) & (j["aobt_missing"].to_numpy() == 1) \
        & (np.nan_to_num(tms, nan=-1e9) > OVERLAY_T)

    def overlaid(p: np.ndarray) -> np.ndarray:
        out = p.copy()
        out[hit] = np.clip(coef[0] + coef[1] * tms[hit], 60, None)
        return out

    b_raw, c_raw = j["base"].to_numpy(), j["cand"].to_numpy()
    b_ovl, c_ovl = overlaid(b_raw), overlaid(c_raw)
    print(f"Rows {j.height:,}, overlay touches {hit.sum()}\n")

    unchanged_passes = True
    unchanged_detail = "Not requested."
    if args.candidate_airports:
        requested = [value.strip().upper() for value in args.candidate_airports.split(",")
                     if value.strip()]
        unknown = sorted(set(requested) - set(code))
        if unknown:
            print(f"ERROR: unknown candidate airports: {', '.join(unknown)}", file=sys.stderr)
            return 1
        candidate_codes = np.array([code[value] for value in requested])
        outside = ~np.isin(j["AIRPORT"].to_numpy(), candidate_codes)
        outside_delta = np.abs(c_raw[outside] - b_raw[outside])
        changed_outside = int(np.count_nonzero(outside_delta))
        maximum_outside_delta = float(outside_delta.max(initial=0.0))
        unchanged_passes = changed_outside == 0
        unchanged_detail = (f"{int(outside.sum()):,} rows outside {', '.join(requested)}: "
                            f"{changed_outside:,} changed, maximum absolute change "
                            f"{maximum_outside_delta:.12g} seconds.")
    elif args.candidate_wake_categories:
        requested = [value.strip().upper()
                     for value in args.candidate_wake_categories.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(wake_code))
        if unknown:
            print(f"ERROR: unknown candidate wake categories: {', '.join(unknown)}",
                  file=sys.stderr)
            return 1
        candidate_codes = np.array([wake_code[value] for value in requested])
        outside = ~np.isin(j["WK_TBL_CAT_flt"].to_numpy(), candidate_codes)
        outside_delta = np.abs(c_raw[outside] - b_raw[outside])
        changed_outside = int(np.count_nonzero(outside_delta))
        maximum_outside_delta = float(outside_delta.max(initial=0.0))
        unchanged_passes = changed_outside == 0
        unchanged_detail = (f"{int(outside.sum()):,} rows outside wake categories "
                            f"{', '.join(requested)}: {changed_outside:,} changed, maximum "
                            f"absolute change {maximum_outside_delta:.12g} seconds.")

    base_overlay_se = (b_ovl - y) ** 2
    candidate_overlay_se = (c_ovl - y) ** 2
    row_advantage = base_overlay_se - candidate_overlay_se
    net_advantage = float(row_advantage.sum())
    largest_advantage = float(row_advantage.max())
    largest_share = (100.0 * largest_advantage / net_advantage
                     if net_advantage > 0 else float("nan"))
    row_share_passes = bool(np.isfinite(largest_share)
                            and largest_share <= args.maximum_row_share)

    lines = [f"# Comparison: {args.label} against the preserved baseline", "",
             f"Rows {j.height:,}. Overlay fitted on the composition pool "
             f"(a={coef[0]:.4f}, b={coef[1]:.6f}), applied identically to both arms, "
             f"touching {hit.sum()} rows.", "",
             "| arm | raw composition RMSE s | overlay adjusted RMSE s |", "|---|---:|---:|",
             f"| baseline | {rmse(b_raw, y):.3f} | {rmse(b_ovl, y):.3f} |",
             f"| {args.label} | {rmse(c_raw, y):.3f} | {rmse(c_ovl, y):.3f} |",
             f"| **change** | **{rmse(c_raw, y) - rmse(b_raw, y):+.3f}** | "
             f"**{rmse(c_ovl, y) - rmse(b_ovl, y):+.3f}** |", "",
             f"Non-candidate invariance: {unchanged_detail}", "",
             (f"Largest single-row contribution to the net overlay squared-error advantage: "
              f"{largest_share:.1f} percent."
              if np.isfinite(largest_share) else
              "Largest single-row share is undefined because net overlay advantage is not positive."),
             ""]
    for line in lines[4:]:
        print(line)

    rng = np.random.default_rng(20260901)
    n = j.height
    win_rates: dict[str, float] = {}
    for tag, a, b in [("raw", b_raw, c_raw), ("overlay adjusted", b_ovl, c_ovl)]:
        sa, sb = (a - y) ** 2, (b - y) ** 2
        d = np.array([np.sqrt(sa[i].mean()) - np.sqrt(sb[i].mean())
                      for i in (rng.integers(0, n, n) for _ in range(BOOTSTRAP))])
        win = float((d > 0).mean())
        win_rates[tag] = win
        msg = (f"- Paired bootstrap, {tag}, {BOOTSTRAP:,} resamples: candidate wins "
               f"{100*win:.1f} %, improvement {np.percentile(d,5):+.2f} s to {np.percentile(d,95):+.2f} s.")
        lines.append(msg)
        print("\n" + msg)

    improves_raw = rmse(c_raw, y) < rmse(b_raw, y)
    improves_ovl = rmse(c_ovl, y) < rmse(b_ovl, y)
    confident_raw = win_rates["raw"] >= 0.95
    confident_ovl = win_rates["overlay adjusted"] >= 0.95
    confidence_passes = (confident_raw and confident_ovl
                         if args.gate_policy == "both" else confident_ovl)
    verdict = ("PASS composition gate, now require forward and seasonal confirmation"
               if improves_raw and improves_ovl and confidence_passes and unchanged_passes
               and row_share_passes else "FAIL composition gate")
    lines += ["", f"**Gate: raw improves {improves_raw}, overlay adjusted improves "
                  f"{improves_ovl}, raw bootstrap at least 95 percent {confident_raw}, "
                  f"overlay bootstrap at least 95 percent {confident_ovl}. "
                  f"Non-candidate rows unchanged {unchanged_passes}, largest-row share at most "
                  f"{args.maximum_row_share:.0f} percent {row_share_passes}. "
                  f"Policy: {args.gate_policy}. Verdict: {verdict}.**", ""]
    print(f"\nVERDICT: {verdict}")

    # Where did the change come from?
    lines += ["## Where the change comes from (overlay adjusted)", "",
              "| slice | rows | baseline RMSE | candidate RMSE |", "|---|---:|---:|---:|"]
    for label, m in [("target <= 3600 s", y <= 3600), ("target > 3600 s", y > 3600),
                     ("no NM off-block", j["aobt_missing"].to_numpy() == 1),
                     ("has NM off-block", j["aobt_missing"].to_numpy() == 0),
                     ("January", j["month"].to_numpy() == 1), ("July", j["month"].to_numpy() == 7)]:
        lines.append(f"| {label} | {int(m.sum()):,} | {rmse(b_ovl[m], y[m]):.2f} | {rmse(c_ovl[m], y[m]):.2f} |")
    print("\n" + "\n".join(lines[-8:]))

    out = args.out or args.candidate.parent / "comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
