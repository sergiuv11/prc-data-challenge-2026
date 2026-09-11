# Reproducibility

## Requirements

- Linux with Bash
- Python 3.12
- About 8 GB RAM recommended for sequential execution
- About 5 GB free disk space for private inputs, normalised data, features, models and reports
- The 14 official competition Parquet files under `data/raw/`

Dependencies are pinned in `requirements.txt`. Generated data, models, reports and submissions are
ignored by Git because they either contain restricted row-level information or are derived from it.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Use the official OpenSky SSO console with your own credentials to place
the twelve training files, `ranking.parquet` and `submitting.parquet` in `data/raw/`. Credentials
are never passed to a project script.

## One-command V3 reproduction

```bash
./scripts/reproduce_v3.sh
```

The runner records SHA-256 hashes for every private input, then performs ten fail-closed stages:
normalisation, schema audit, leakage audit, feature
construction, three global validation splits, three specialist validation splits, all statistical
gates, final global training, final specialist training and structural submission validation.

It writes the final local artifact to:

```text
submissions/reproduction_v3/jubilant-vase_v3.parquet
```

It never accesses the submission bucket and never uploads the file. Upload remains a separate,
deliberate participant action after reviewing the generated audit.

Set `PYTHON` only when using a different compatible interpreter:

```bash
PYTHON=/path/to/venv/bin/python ./scripts/reproduce_v3.sh
```

## Expected invariants

- 12 training input files
- 2,085,047 training departures
- 215,876 ranked departures and submission rows
- exactly 43 final features in the same order for every booster
- exactly three specialist airports: EDDF, EGLL and EHAM
- 436 trees per final model
- exact self-consistency reconstruction of the freshly generated V2 foundation before V3 is written
- no null, duplicate, missing, extra, negative or non-finite submission values

All clean and feature caches created by the runner stay under `data/reproduction_v3/`, so the run
does not overwrite caches used by other experiments. The private input hashes are written to
`reports/reproduction_v3/input_SHA256SUMS` for diagnosing differences without publishing data.

## Numeric reproducibility boundary

The public runner reproduces the architecture, validation decisions and final construction, but it
does not promise a bit-identical model file. LightGBM uses the documented fixed count of eight CPU
threads, and floating-point histogram reductions can resolve near-tied splits differently across
runs. A clean rerun on the original VPS reproduced every published validation RMSE to rounding and
passed all gates. Its freshly trained V2 differed from the submitted V2 on 186,170 of 215,876
rounded predictions, with mean absolute difference 4.44 seconds and maximum 378 seconds. After the
specialist layer, fresh V3 differed from submitted V3 on 171,075 rows, with mean absolute difference
3.37 seconds and maximum 378 seconds. These differences are small relative to the 286.656-second
official RMSE, but they are systematic and not zero.

LightGBM also offers deterministic histogram construction with a forced row-wise or column-wise
algorithm. We did not enable it retrospectively because that would describe a different training
configuration from the submitted model. The public runner therefore preserves the actual
eight-thread method and states its measured numeric boundary.

The original submitted V3 SHA-256 is published in `docs/RESULTS.md` as provenance, not as an
expected output of retraining. Exact artifact preservation requires retaining the original trained
boosters and submission, which cannot be published because they are derived from the restricted
competition data.
