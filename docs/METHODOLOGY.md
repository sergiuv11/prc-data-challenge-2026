# Methodology

## Prediction task

The system predicts taxi-out time in seconds for each departure. Taxi-out is the interval from
leaving the stand to takeoff. The competition evaluates Root Mean Square Error (RMSE), so a few
large errors matter much more than many small ones.

## Data boundary

The model uses only fields present on ranking departures. The organiser blanks the movement
off-block timestamp and target, but retains movement, flight-plan and Network Manager fields.
The leakage audit measures every off-block-like timestamp rather than assuming it is either safe
or equivalent to the answer. `AOBT_3_flt` is useful but does not reconstruct the target: its direct
2025 RMSE is 384.9 seconds and only 21.0 percent of rows are within one minute.

Training and ranking Parquet files are processed separately when computing traffic windows. This
prevents a 2025 row from seeing a 2026 neighbour. All categorical values are then encoded with one
stable mapping over the union of the two feature tables.

## Features

The final system has 43 inputs:

- 11 categorical features: airport, runway, stand, stand area, aircraft type, operator, market
  segment, wake category, flight type, flight rule and destination.
- 17 row-level numeric features: Network Manager implied taxi time and missingness, time deltas
  among takeoff, scheduled and flight-plan timestamps, planned flight duration, UTC calendar
  fields and weekend status.
- 15 congestion features: prior and subsequent departure and arrival counts, runway flow,
  scheduled departure pressure and recent runway share.

The exact feature definitions are executable in `scripts/features.py`. Missing numeric values stay
missing so LightGBM can learn their routing. Categories use stable integer codes and are declared as
categorical features to LightGBM.

## Model architecture

V3 contains four LightGBM regressors with one fixed configuration:

1. One global model trained on all 2,085,047 departures.
2. One EDDF specialist trained only on EDDF departures.
3. One EGLL specialist trained only on EGLL departures.
4. One EHAM specialist trained only on EHAM departures.

Every model uses L2 regression, learning rate 0.06, 127 leaves, minimum 200 rows per leaf, feature
fraction 0.8, bagging fraction 0.8, L2 regularisation 1.0, seed 20260901, eight CPU threads and 436
trees. At EDDF, EGLL and EHAM, the prediction is an equal blend of the global and local model.
Other airports use the global prediction unchanged.

A narrow post-model rule handles LIRF departures for which the Network Manager off-block value is
missing and takeoff is more than 14,400 seconds after schedule. Ordinary L2 leaves cannot represent
these rare extreme values reliably. A linear relationship is fitted from 2025 LIRF rows satisfying
the same rule and replaces only matching predictions. It changed 15 of 215,876 ranked rows.

## Validation design

Random row splitting would mix nearby airport operations and overstate generalisation. The system
uses temporal simulations:

- Composition: fit February through June and August through November, reserve December as the
  inner month, then test January at all airports plus July at the three airports that occur in the
  delivered July ranking data.
- Forward: train January through May, keep June as a buffer, then test July at EDDF, EGLL and EHAM.
- Seasonal: train on days 1 to 18, keep days 19 to 21 as a buffer, then test days 22 onward in the
  competition-matched month and airport composition.

A candidate must improve both raw and submitted-system RMSE. The submitted-system comparison fits
the LIRF overlay only on the permitted training pool and applies it identically to both arms. It
must also win at least 95 percent of 2,000 paired bootstrap resamples and must not derive over half
of its net advantage from one row.

## Final construction

After the specialist architecture passed all three gates, the global and three local models were
trained on all 2025 departures. Predictions are clipped to a physical minimum of 60 seconds,
specialists are blended at their three airports, and the LIRF rule is applied last. The output is
joined one-to-one to the organiser template and cast to its target type. The validator rejects
missing IDs, extra IDs, duplicates, nulls, non-finite values, negative values or a mismatched row
count before any upload is considered.
