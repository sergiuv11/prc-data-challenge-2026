# PRC Data Challenge 2026: predicting taxi-out time

Open, reproducible solution submitted by team **jubilant-vase** to the
[EUROCONTROL PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/).

**Official score: 286.656 seconds RMSE**, scored on all 215,876 ranking pairs.

## The task

Predict taxi-out time, the interval between a flight's actual off-block time and its
takeoff, from movement and flight-plan data at ten major European airports.

## Architecture

The shipped solution, V3, has three parts:

1. **A single global LightGBM model** with L2 objective, 127 leaves, `min_data_in_leaf=200`
   and 436 trees, fitted once on all 2,085,047 departures of 2025 across 43 features.
2. **A calibrated LIRF overlay** applied at 14,400 seconds, which changes a small number of
   extreme-tail predictions.
3. **Three fixed airport specialists** for EDDF, EGLL and EHAM, blended 50/50 with the
   global model. Every other airport is served by the global model alone.

V3 uses **no external data**. A METAR weather attachment was implemented and evaluated, but
it is not part of the shipped model.

## Reproducing the result

One command, fail-closed, from the official source files:

```bash
bash scripts/reproduce_v3.sh
```

It records input hashes, uses isolated derived caches, repeats all three temporal validation
arms with their statistical gates, trains the global and specialist models, constructs the
submission and runs the structural validator. It contains no upload path.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for requirements and exact steps.

### An honest note on determinism

A clean eight-thread retrain passes every validation gate but is **not bit-identical** to the
submitted boosters. Fresh V3 differed from submitted V3 on 171,075 rounded rows, with a mean
absolute difference of 3.37 seconds. Retraining reproduces the method and the measured
performance, not the exact artifact.

## Validation

Three temporal splits, each with a predeclared acceptance gate and 1,000-resample paired
bootstrap testing:

| split | what it tests |
|---|---|
| composition | a held-out mixture that removes January and July from training |
| forward | strict future prediction |
| seasonal | a held-out season |

V3 won 100 percent of bootstrap resamples at all three gates.
Full numbers in [docs/RESULTS.md](docs/RESULTS.md) and method in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Data

Competition data is restricted and is **not** included here. Obtain it through the official
OpenSky S3 console with your own credentials and place it under `data/raw/`.
External data sources considered are documented in [docs/EXTERNAL_DATA.md](docs/EXTERNAL_DATA.md).

## Licence

GPL-3.0. See [LICENSE](LICENSE).

## Author

Sergiu Vincze, [SevinHub](https://sevinhub.com), an independent technology studio in Antwerp, Belgium.
