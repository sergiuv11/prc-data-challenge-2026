# External data

## Iowa Environmental Mesonet global METAR archive

Source: Iowa Environmental Mesonet (IEM), Iowa State University.

- Dataset documentation: https://mesonet.agron.iastate.edu/info/datasets/metar.html
- Download API documentation: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
- Usage and disclaimer: https://mesonet.agron.iastate.edu/disclaimer.php

The IEM documents the archive as worldwide airport METAR observations with a temporal
domain from 1900 to the present. Its usage statement says materials on the website are in
the public domain and may be used freely for lawful purposes. Attribution is appreciated.
We attribute the IEM in the repository and eventual paper.

The archive aggregates observations from sources including Unidata IDD, NCEI ISD and MADIS.
IEM warns that the archive is provided as-is with limited quality control. Missingness and
observation age are therefore explicit model features rather than silently imputed facts.

### Scope used here

Only the ten airports present in the delivered PRC files are requested:

`EDDF`, `EDDM`, `EGLL`, `EHAM`, `LEBL`, `LEMD`, `LFPG`, `LIRF`, `LSZH`, `LTFM`.

The downloader requests:

- all of 2025 for model training;
- January 2026 for ranking predictions;
- July 2026 for ranking predictions.

The raw CSV files remain under ignored `data/external/metar/`. The public repository contains
the downloader, checksums produced after download and transformation code, but not copied data.

### Prediction-time safety

For each flight, `scripts/attach_weather.py` selects only the latest observation whose `valid`
timestamp is less than or equal to the supplied movement time. Future observations are never
joined. Observations older than three hours are treated as missing. The transformation asserts
that every matched age is between zero and 180 minutes.

This remains reproducible for the ranked period. On 2026-09-02, direct sample requests for
2025-01-15, 2026-01-15 and 2026-07-15 returned observations for all ten airports.

### Variables

The model receives physically motivated fields only: temperature, dew point, dew-point
depression, freezing and near-freezing flags, wind and gust, cyclic wind direction,
visibility, ceiling, sky-cover severity, present-weather flags, observation age and a missing
indicator. Non-US precipitation accumulation is unavailable from this archive and is not used.
