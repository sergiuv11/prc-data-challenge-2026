#!/usr/bin/env python3
"""Attach only the latest prior METAR observation to each cached flight feature row."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402

RAW_FIELDS = [
    "station", "valid", "tmpf", "dwpf", "drct", "sknt", "vsby", "gust", "wxcodes",
    "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4",
]


def numeric(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False)


def load_weather(directory: Path) -> pl.DataFrame:
    paths = sorted(directory.glob("metar_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no metar_*.csv files found under {directory}")
    frames = [
        pl.read_csv(
            path,
            comment_prefix="#",
            null_values=["null", "M", ""],
            schema_overrides={name: pl.String for name in RAW_FIELDS},
        ).select(RAW_FIELDS)
        for path in paths
    ]
    weather = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.col("station").str.to_uppercase(),
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC", strict=True),
        *[numeric(name).alias(name) for name in
          ["tmpf", "dwpf", "drct", "sknt", "vsby", "gust", "skyl1", "skyl2", "skyl3", "skyl4"]],
        pl.col("wxcodes").fill_null("").str.to_uppercase(),
        *[pl.col(name).fill_null("").str.to_uppercase().alias(name)
          for name in ["skyc1", "skyc2", "skyc3", "skyc4"]],
    )
    # If multiple feeds supplied the same station and valid time, keep one deterministic row.
    return weather.sort(["station", "valid"]).unique(
        subset=["station", "valid"], keep="last", maintain_order=True)


def airport_mapping(data_dir: Path) -> pl.DataFrame:
    paths = [str(path) for path in sorted(data_dir.glob("training_*.parquet"))]
    paths.append(str(data_dir / "ranking.parquet"))
    names = (
        pl.scan_parquet(paths)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("ADEP_mvt").unique().collect()["ADEP_mvt"]
        .drop_nulls().sort().to_list()
    )
    return pl.DataFrame({"AIRPORT": range(len(names)), "AIRPORT_ICAO": names},
                        schema={"AIRPORT": pl.Int32, "AIRPORT_ICAO": pl.String})


def weather_expressions() -> list[pl.Expr]:
    direction = numeric("drct") * math.pi / 180.0
    ceiling = pl.min_horizontal(*[
        pl.when(pl.col(f"skyc{i}").is_in(["BKN", "OVC", "VV"]))
        .then(numeric(f"skyl{i}"))
        .otherwise(None)
        for i in range(1, 5)
    ])
    cover_values = {"": None, "CLR": 0, "SKC": 0, "NSC": 0, "NCD": 0,
                    "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4}
    sky_cover = pl.max_horizontal(*[
        pl.col(f"skyc{i}").replace_strict(cover_values, default=None, return_dtype=pl.Int8)
        for i in range(1, 5)
    ])
    temp_c = (numeric("tmpf") - 32.0) * (5.0 / 9.0)
    dew_c = (numeric("dwpf") - 32.0) * (5.0 / 9.0)
    codes = pl.col("wxcodes")
    missing = pl.col("valid").is_null()
    return [
        ((pl.col(F.TAKEOFF) - pl.col("valid")).dt.total_seconds() / 60.0)
        .alias("wx_age_minutes"),
        missing.cast(pl.Int8).alias("wx_missing"),
        temp_c.alias("wx_temp_c"),
        dew_c.alias("wx_dewpoint_c"),
        (temp_c - dew_c).alias("wx_dewpoint_depression_c"),
        (temp_c <= 3.0).cast(pl.Int8).alias("wx_below_3c"),
        (temp_c <= 0.0).cast(pl.Int8).alias("wx_freezing"),
        numeric("sknt").alias("wx_wind_knots"),
        numeric("gust").alias("wx_gust_knots"),
        direction.sin().alias("wx_wind_sin"),
        direction.cos().alias("wx_wind_cos"),
        (numeric("vsby") * 1.609344).alias("wx_visibility_km"),
        ceiling.alias("wx_ceiling_ft"),
        sky_cover.alias("wx_sky_cover"),
        codes.str.contains(r"RA|DZ").cast(pl.Int8).alias("wx_rain"),
        codes.str.contains(r"SN|SG").cast(pl.Int8).alias("wx_snow"),
        codes.str.contains(r"FZRA|FZDZ|PL|IC").cast(pl.Int8).alias("wx_freezing_precip"),
        codes.str.contains(r"FG|BR").cast(pl.Int8).alias("wx_fog_mist"),
        codes.str.contains("TS").cast(pl.Int8).alias("wx_thunder"),
    ]


def attach(source: Path, destination: Path, weather: pl.DataFrame,
           mapping: pl.DataFrame) -> dict[str, object]:
    features = pl.read_parquet(source)
    original_ids = features["MVT_ID_mvt"]
    joined = (
        features.with_row_index("__row")
        .join(mapping, on="AIRPORT", how="left", validate="m:1")
        .sort(["AIRPORT_ICAO", F.TAKEOFF])
        .join_asof(
            weather.sort(["station", "valid"]),
            left_on=F.TAKEOFF,
            right_on="valid",
            by_left="AIRPORT_ICAO",
            by_right="station",
            strategy="backward",
            tolerance="3h",
            check_sortedness=False,
        )
        .with_columns(weather_expressions())
        .sort("__row")
    )
    if joined["AIRPORT_ICAO"].null_count():
        raise ValueError(f"{source} contains an unmapped airport code")
    age = joined["wx_age_minutes"].drop_nulls()
    if age.len() and (age.min() < 0 or age.max() > 180):
        raise ValueError(f"{source} joined a future or stale METAR: age range {age.min()} to {age.max()}")

    output = joined.select(features.columns + F.WEATHER_NUMERIC)
    if output.height != features.height or not output["MVT_ID_mvt"].equals(original_ids):
        raise ValueError(f"{source} row identity or order changed during the weather join")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(destination, compression="zstd")

    audit = (
        joined.group_by("AIRPORT_ICAO")
        .agg(
            pl.len().alias("rows"),
            pl.col("valid").is_not_null().sum().alias("matched"),
            pl.col("wx_age_minutes").median().alias("median_age_minutes"),
            pl.col("wx_age_minutes").max().alias("max_age_minutes"),
        )
        .sort("AIRPORT_ICAO")
    )
    print(f"{source.name}: {features.height:,} rows, "
          f"{100 * (1 - output['wx_missing'].mean()):.3f} percent matched")
    print(audit)
    return {
        "rows": features.height,
        "matched": int((output["wx_missing"] == 0).sum()),
        "missing": int(output["wx_missing"].sum()),
        "min_age_minutes": float(age.min()) if age.len() else None,
        "max_age_minutes": float(age.max()) if age.len() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features", type=Path)
    parser.add_argument("--weather", default="data/external/metar", type=Path)
    parser.add_argument("--data-dir", default="data/clean", type=Path)
    parser.add_argument("--out", default="data/features_weather", type=Path)
    args = parser.parse_args()

    weather = load_weather(args.weather)
    mapping = airport_mapping(args.data_dir)
    expected = set(mapping["AIRPORT_ICAO"].to_list())
    observed = set(weather["station"].to_list())
    absent = sorted(expected - observed)
    if absent:
        print(f"ERROR: weather contains no rows for {', '.join(absent)}", file=sys.stderr)
        return 1

    audit = {
        "source_observations": weather.height,
        "source_stations": sorted(observed),
        "train": attach(args.features / "train_departures.parquet",
                        args.out / "train_departures.parquet", weather, mapping),
        "ranking": attach(args.features / "ranking_departures.parquet",
                          args.out / "ranking_departures.parquet", weather, mapping),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weather_join_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote weather-enriched features and {args.out / 'weather_join_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
