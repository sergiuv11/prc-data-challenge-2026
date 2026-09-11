#!/usr/bin/env python3
"""Leakage-safe target encodings for taxi-out prediction.

The classic failure is to compute a group mean over the whole training set and use it as
a feature. Every row then contributes to its own group's mean, the model reads a signal
that will not exist at prediction time, and the apparent gain evaporates on the hidden set.

The defence here rests on one structural idea: **a single function computes an encoding
from an explicit set of source rows, and nothing else ever computes an encoding.**

- Training rows are encoded by leave-one-month-out cross fitting. A row in month m is
  encoded from the other available months, so it never sees its own target, and never sees a
  neighbouring flight caught in the same disruption on the same day.
- A validation or test set is encoded from its split's training portion only.
- The ranking set is encoded from all twelve months of 2025, which is exactly what the
  final model is fitted on.

Leave-one-month-out is preferred over a past-only expanding window because it matches
deployment. The 2026 rows are encoded from all of 2025, so a training row encoded from
eleven months is the closest analogue. A past-only scheme would give January almost no
support and December eleven months of it, putting the feature on a different scale in
training than at prediction time.

Two further choices, both deliberate:

- The statistic is a **mean**, because RMSE is minimised by conditional means. It is taken
  over a **winsorised** target, capped at `CAP` seconds. A single 87,000 second record
  would otherwise dominate its stand's encoding for the whole year. The cap applies only to
  the feature; the training label itself is never altered. The extreme rows are handled by
  the LIRF overlay, not by encodings.
- Every encoding is **smoothed toward its parent**: a stand toward its airport, the airport
  toward the global mean. A stand seen three times must not produce a confident encoding.

Usage:
    from target_encoding import SPECS, cross_fit_train, fit_maps, transform

    train = cross_fit_train(train)             # leave-one-month-out, for model fitting
    maps  = fit_maps(train)                    # one map per spec, from these rows only
    test  = transform(test, maps)              # encode unseen rows from those maps
"""
from __future__ import annotations

import polars as pl

TARGET = "TAXITIME_SEC_mvt"
BLOCK = "month"

# Winsorisation cap for the encoded statistic only. p99.9 of the target is about 4,500 s.
CAP = 3600.0
# Empirical Bayes smoothing weight: a group needs about this many observations before its
# own mean outweighs its parent's.
SMOOTHING = 100.0

# (feature prefix, grouping keys). Parent for smoothing is always AIRPORT.
SPECS: list[tuple[str, list[str]]] = [
    ("te_stand", ["AIRPORT", "STAND_mvt"]),
    ("te_runway_hour", ["AIRPORT", "RUNWAY_mvt", "hour"]),
    ("te_operator", ["AIRPORT", "AIRCRAFT_OPERATOR_flt"]),
    ("te_dest", ["AIRPORT", "ADES_mvt"]),
]

FEATURES: list[str] = [f"{name}{suffix}" for name, _ in SPECS for suffix in ("", "_count")]


def _capped(col: str = TARGET) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).clip(None, CAP)


def _airport_prior(source: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Per airport mean, itself smoothed toward the global mean."""
    grand = float(source.select(_capped().mean()).item())
    per_airport = source.group_by("AIRPORT").agg(
        _capped().sum().alias("s"), pl.len().alias("n")
    ).with_columns(
        ((pl.col("s") + SMOOTHING * grand) / (pl.col("n") + SMOOTHING)).alias("prior")
    ).select(["AIRPORT", "prior"])
    return per_airport, grand


def fit_map(source: pl.DataFrame, name: str, keys: list[str]) -> pl.DataFrame:
    """Build one encoding map from an explicit set of source rows.

    This is the only place an encoding value is ever computed. Cross fitting calls it once
    per held-out block, and the final fit calls it once on everything, so training and
    inference cannot drift apart.
    """
    prior, grand = _airport_prior(source)
    return (
        source.group_by(keys)
        .agg(_capped().sum().alias("s"), pl.len().alias("n"))
        .join(prior, on="AIRPORT", how="left")
        .with_columns(pl.col("prior").fill_null(grand))
        .with_columns(
            ((pl.col("s") + SMOOTHING * pl.col("prior")) / (pl.col("n") + SMOOTHING)).alias(name),
            pl.col("n").cast(pl.Int32).alias(f"{name}_count"),
        )
        .select(keys + [name, f"{name}_count"])
    )


def fit_maps(source: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """One map per spec, all fitted from the same explicit source rows."""
    return {name: fit_map(source, name, keys) for name, keys in SPECS}


def transform(df: pl.DataFrame, maps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Attach encodings to rows that took no part in fitting the maps.

    Unseen groups stay null. LightGBM routes nulls itself, and the paired `_count` column
    tells the model how much evidence stands behind each value.
    """
    out = df
    for name, keys in SPECS:
        out = out.join(maps[name], on=keys, how="left")
    return out


def cross_fit_train(train: pl.DataFrame) -> pl.DataFrame:
    """Leave-one-month-out encodings for training rows.

    Each month is encoded from the other available months, so no row can see its own target and no
    row can be encoded by another flight from the same day and the same disruption.
    """
    blocks = sorted(train[BLOCK].unique().to_list())
    parts = []
    for b in blocks:
        held = train.filter(pl.col(BLOCK) == b)
        source = train.filter(pl.col(BLOCK) != b)
        if source.is_empty():
            raise ValueError(f"block {b} has no source rows to encode from")
        parts.append(transform(held, fit_maps(source)))
    return pl.concat(parts, how="vertical_relaxed")
