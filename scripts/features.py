#!/usr/bin/env python3
"""Feature engineering for taxi-out prediction (Step 6 of the plan).

Every feature is computed only from columns that are populated on the 2026 ranking
departures, as verified by `scripts/audit_schema.py`. The two blanked columns,
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, are never read.

Congestion is built from the complete movement stream within each dataset. The twelve
training months are processed together, while the ranking file is processed separately,
so every departure sees the real neighbouring arrivals and departures from its own period.
"""
from __future__ import annotations

import polars as pl

TARGET = "TAXITIME_SEC_mvt"
TAKEOFF = "MVT_TIME_UTC_mvt"
SCHED = "SCHED_TIME_UTC_mvt"

CATEGORICAL = [
    "AIRPORT", "RUNWAY_mvt", "STAND_mvt", "stand_area", "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
    "FLIGHT_RULE_mvt", "ADES_mvt",
]

CONGESTION_NUMERIC = [
    "dep_prev_15", "dep_prev_30", "dep_prev_60", "dep_next_15", "dep_next_30",
    "arr_prev_15", "arr_prev_30", "arr_prev_60", "arr_next_15", "arr_next_30",
    "runway_prev_15", "runway_prev_30", "sched_dep_prev_30", "sched_dep_next_30",
    "dep_share_runway_30",
]

NUMERIC = [
    # The Network Manager's own view of the same event. Strongest single signal,
    # deliberately kept as one feature among many rather than used as an answer.
    "implied_taxi_aobt", "aobt_missing",
    # Takeoff-relative deltas. `takeoff_minus_schedule` is the only one of these that
    # survives when the NM record is absent, and it is what carries the extreme targets.
    "takeoff_minus_schedule", "takeoff_minus_lobt", "takeoff_minus_eobt", "takeoff_minus_iobt",
    # Pushback punctuality, relative to the actual off-block.
    "sched_minus_aobt", "eobt_minus_aobt", "iobt_minus_aobt", "lobt_minus_aobt",
    "planned_flight_time",
    "hour", "minute_of_day", "weekday", "month", "day_of_year", "is_weekend",
    *CONGESTION_NUMERIC,
]

# Optional public-domain METAR features produced by attach_weather.py. They are deliberately
# excluded from FEATURES so the default pipeline remains an exact reproduction of v2.
WEATHER_NUMERIC = [
    "wx_age_minutes", "wx_missing", "wx_temp_c", "wx_dewpoint_c",
    "wx_dewpoint_depression_c", "wx_below_3c", "wx_freezing",
    "wx_wind_knots", "wx_gust_knots", "wx_wind_sin", "wx_wind_cos",
    "wx_visibility_km", "wx_ceiling_ft", "wx_sky_cover",
    "wx_rain", "wx_snow", "wx_freezing_precip", "wx_fog_mist", "wx_thunder",
]

# Optional strictly prior completed-arrival state produced by attach_arrival_taxi.py.
# Excluded from FEATURES so the default pipeline remains an exact v2/v3 reproduction.
ARRIVAL_TAXI_NUMERIC = ["arr_taxi_mean_60", "arr_taxi_n_60"]

# Optional instantaneous ground-state feature produced by attach_ground_queue.py.
GROUND_QUEUE_NUMERIC = ["dep_taxiing_at_aobt"]

# Optional row-local scheduled clock feature. Excluded from FEATURES so the default model
# remains an exact v2/v3 reproduction even when a freshly built cache contains the column.
SCHEDULE_NUMERIC = ["sched_minute_of_day"]

FEATURES = CATEGORICAL + NUMERIC

# Columns actually needed from the raw files. Reading only these keeps peak memory low.
READ_COLUMNS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", TAKEOFF, SCHED, TARGET,
    "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt",
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
    "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "ARVT_1_flt",
]


def with_airport(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Reporting airport: ADEP for a departure, ADES for an arrival."""
    return lf.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .alias("AIRPORT")
    )


def _window_count(src: pl.DataFrame, keys: list[str], index: str, period: str,
                  offset: str, name: str) -> pl.DataFrame:
    """One row per distinct (keys, index) with the movement count in that time window."""
    counted = (
        src.sort(index)
        .rolling(index_column=index, period=period, offset=offset, group_by=keys)
        .agg(pl.len().alias(name))
    )
    return counted.unique(subset=keys + [index], keep="first")


def congestion(movements: pl.DataFrame) -> pl.DataFrame:
    """Traffic pressure around each departure, from movement times only.

    Departure and arrival counts are computed over the *combined* movement stream. A
    per-phase frame contains only that phase's timestamps, so joining an arrival-only
    count table back onto a departure's takeoff time misses almost every row.
    """
    base = movements.select(["AIRPORT", TAKEOFF, "PHASE_mvt"]).with_columns(
        (pl.col("PHASE_mvt") == "DEP").cast(pl.Int32).alias("is_dep"),
        (pl.col("PHASE_mvt") == "ARR").cast(pl.Int32).alias("is_arr"),
    )
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP").select(["AIRPORT", "RUNWAY_mvt", TAKEOFF, SCHED])
    out = movements.filter(pl.col("PHASE_mvt") == "DEP").select(["MVT_ID_mvt", "AIRPORT", "RUNWAY_mvt", TAKEOFF, SCHED])

    windows = [("15m", "-15m", "prev_15"), ("30m", "-30m", "prev_30"), ("60m", "-60m", "prev_60"),
               ("15m", "0s", "next_15"), ("30m", "0s", "next_30")]
    for period, offset, suffix in windows:
        counts = (
            base.sort(TAKEOFF)
            .rolling(index_column=TAKEOFF, period=period, offset=offset, group_by=["AIRPORT"])
            .agg(pl.col("is_dep").sum().alias(f"dep_{suffix}"), pl.col("is_arr").sum().alias(f"arr_{suffix}"))
            .unique(subset=["AIRPORT", TAKEOFF], keep="first")
        )
        out = out.join(counts, on=["AIRPORT", TAKEOFF], how="left")

    for period, offset, name in [("15m", "-15m", "runway_prev_15"), ("30m", "-30m", "runway_prev_30")]:
        out = out.join(_window_count(dep, ["AIRPORT", "RUNWAY_mvt"], TAKEOFF, period, offset, name),
                       on=["AIRPORT", "RUNWAY_mvt", TAKEOFF], how="left")

    # Planned pressure: how many departures were *scheduled* around this one.
    sched_src = dep.select(["AIRPORT", SCHED])
    for period, offset, name in [("30m", "-30m", "sched_dep_prev_30"), ("30m", "0s", "sched_dep_next_30")]:
        out = out.join(_window_count(sched_src, ["AIRPORT"], SCHED, period, offset, name),
                       on=["AIRPORT", SCHED], how="left")

    return out.select(
        ["MVT_ID_mvt", "dep_prev_15", "dep_prev_30", "dep_prev_60", "dep_next_15", "dep_next_30",
         "arr_prev_15", "arr_prev_30", "arr_prev_60", "arr_next_15", "arr_next_30",
         "runway_prev_15", "runway_prev_30", "sched_dep_prev_30", "sched_dep_next_30"]
    ).with_columns(
        # How much of the airport's recent departure flow used this runway: a cheap proxy
        # for the runway configuration in force at the time.
        (pl.col("runway_prev_30") / pl.col("dep_prev_30").cast(pl.Float32)).alias("dep_share_runway_30")
    )


def build(movements: pl.DataFrame) -> pl.DataFrame:
    """Departure rows with every model feature attached."""
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP").join(congestion(movements), on="MVT_ID_mvt", how="left")
    secs = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()

    return dep.with_columns(
        # Null where the Network Manager record is missing (about 1.5 % of ranking rows).
        # Left as null on purpose: LightGBM learns its own route for missing values, and
        # `aobt_missing` lets the model treat those rows as a distinct population.
        # Clipped wide on purpose. Real taxi-out reaches 131,167 s in this dataset and the
        # extreme records dominate RMSE, so a tight clip would erase the signal that
        # predicts them. Only physically absurd values are cut.
        secs(TAKEOFF, "AOBT_3_flt").clip(-7200, 172800).alias("implied_taxi_aobt"),
        pl.col("AOBT_3_flt").is_null().cast(pl.Int8).alias("aobt_missing"),
        secs("AOBT_3_flt", SCHED).clip(-14400, 14400).alias("sched_minus_aobt"),
        secs("AOBT_3_flt", "EOBT_1_flt").clip(-14400, 14400).alias("eobt_minus_aobt"),
        secs("AOBT_3_flt", "IOBT_flt").clip(-14400, 14400).alias("iobt_minus_aobt"),
        secs("AOBT_3_flt", "LOBT_flt").clip(-14400, 14400).alias("lobt_minus_aobt"),
        # The only strong signal that survives a missing NM record, and it tracks the
        # extreme taxi times closely (most visibly at LIRF). Kept on the same wide scale.
        secs(TAKEOFF, SCHED).clip(-7200, 172800).alias("takeoff_minus_schedule"),
        secs(TAKEOFF, "LOBT_flt").clip(-7200, 172800).alias("takeoff_minus_lobt"),
        secs(TAKEOFF, "EOBT_1_flt").clip(-7200, 172800).alias("takeoff_minus_eobt"),
        secs(TAKEOFF, "IOBT_flt").clip(-7200, 172800).alias("takeoff_minus_iobt"),
        secs("ARVT_1_flt", "EOBT_1_flt").clip(0, 86400).alias("planned_flight_time"),
        pl.col(TAKEOFF).dt.hour().alias("hour"),
        (pl.col(TAKEOFF).dt.hour().cast(pl.Int16) * 60
         + pl.col(TAKEOFF).dt.minute().cast(pl.Int16)).alias("minute_of_day"),
        (pl.col(SCHED).dt.hour().cast(pl.Int16) * 60
         + pl.col(SCHED).dt.minute().cast(pl.Int16)).alias("sched_minute_of_day"),
        pl.col(TAKEOFF).dt.weekday().alias("weekday"),
        pl.col(TAKEOFF).dt.month().alias("month"),
        pl.col(TAKEOFF).dt.ordinal_day().alias("day_of_year"),
        (pl.col(TAKEOFF).dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        # Stands are named per terminal or pier; the leading characters carry that grouping.
        pl.col("STAND_mvt").str.replace_all(r"[0-9].*$", "").alias("stand_area"),
    ).select(["MVT_ID_mvt", TARGET, TAKEOFF] + FEATURES + SCHEDULE_NUMERIC)


def load_movements(paths: list[str]) -> pl.DataFrame:
    """Read only the columns the model needs, with the reporting airport attached."""
    return with_airport(pl.scan_parquet(paths).select(READ_COLUMNS)).collect()
