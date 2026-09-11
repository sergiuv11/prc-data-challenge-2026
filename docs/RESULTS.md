# Results

## Official submissions

| version | architecture | official RMSE |
|---|---|---:|
| V2 | Global LightGBM plus narrow LIRF overlay | 290.488 s |
| V3 | V2 plus EDDF, EGLL and EHAM specialists | **286.656 s** |

V3 improved the official score by 3.832 seconds across all 215,876 scored departures. The best
score across a team's valid submissions determines its standing, so V2 remains preserved but V3 is
the current official model.

## Local validation

The global model first established a large gain over a hierarchical historical-median control:

| split | median control RMSE | global LightGBM RMSE |
|---|---:|---:|
| Competition-matched composition | 501.51 s | 357.12 s |
| Strict July forward | 348.72 s | 250.25 s |
| Seasonal day split | 537.65 s | 405.44 s |

These global reference values use early stopping, which selected 381 trees for July. The specialist
gate below deliberately freezes the final architecture at 436 trees, giving its slightly different
250.341-second global July baseline.

The V3 specialist layer was then compared against that preserved global system:

| gate | global system RMSE | V3 RMSE | gain | bootstrap wins |
|---|---:|---:|---:|---:|
| Composition, with equal overlay | 325.931 s | 323.555 s | 2.375 s | 100.0% |
| July forward | 250.341 s | 241.598 s | 8.743 s | 100.0% |
| Seasonal | 403.286 s | 401.471 s | 1.815 s | 100.0% |

The final file contains 215,876 template-aligned rows. Relative to V2, 112,221 rounded predictions
changed, all at EDDF, EGLL or EHAM. Its original SHA-256 is
`1bd67c9fe503de7ecb060b582a747519f3f62d8cee03559672464cb335a1bfe4`.

The one-command clean-room run reproduced all local RMSE values to rounding, all three statistical
gate decisions, the feature schema and final construction. It does not claim bit-identical trained
boosters because LightGBM's eight-thread floating-point reductions can choose different near-tied
splits. The measured numeric boundary is documented in `docs/REPRODUCIBILITY.md`.

## Negative results

The experiment log includes every serious branch, including target encoding, public METAR,
CatBoost, learning-rate changes, more airport specialists, alternative early stopping, congestion
ablation, seed ensembles, wake specialists, completed-arrival state, active taxi queue, scheduled
clock time, model-capacity selection and post-hoc calibration. Each rejected branch kept V3 safe.
Every architectural decision was made against a predeclared acceptance gate. Branches that failed their gate were closed without being built or submitted.

## Interpretation

Local RMSE is deliberately pessimistic and varies sharply across time splits because a very small
number of extreme movements dominate squared error. It is useful for comparing two models on the
same rows, not for predicting the exact leaderboard score. The official V2 to V3 improvement agrees
with the local direction, but the hidden leaderboard is never used as training data.
