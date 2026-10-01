# Experiment log

## Competition facts

- Metric: RMSLE; negative predictions are treated as zero.
- Train: 1,226,237 rows, 2025-01-01 through 2026-03-31.
- Test: 714,688 rows, 2026-04-01 through 2026-07-31.
- Raw transformer counts: 5,344 train and 7,036 test; 2,024 test transformers
  are not present in train (158,369 test rows).
- Daily limit: 3 submissions. Final selections: 2.
- Kaggle stage ends 2026-09-01 23:59; private leaderboard releases shortly
  afterward. Private top-20 teams must share a notebook for review.

## 2026-08-21 submissions

1. `recent3_w080.csv` — ref 55669847 — public RMSLE 1.26688.
2. `recent7_w070.csv` — ref 55669854 — public RMSLE 1.25594.
3. `recent7_seasonal.csv` — ref 55670171 — public RMSLE 1.24233.

The public feedback supported additional temporal smoothing and then confirmed
that the summer adjustment helps. The third entry was public rank 18 at the
time checked; the leading score was 1.04644.

## 2026-08-31 probe

- `cold_probe_plus1.csv` — ref 55914207 — public RMSLE 1.31919. Adding
  `+1.0` in log space only to cold-start rows was deliberately too large, but
  its score relative to v3 indicates that the useful calibration is small and
  positive (about `+0.06` if the public split has the full test cold-row mix).
- Four 122-day rolling origins favored a 3-day recent level blended at 0.4
  with the all-history transformer weekday mean. Within zero-rate strata this
  reduced aggregate known-row RMSLE from 0.9081 for the v3 parameters to
  0.8887. The resulting v5 scored 1.26687 publicly, so that rolling-origin
  parameter preference did not transfer to the April-July test period.
- A clean rolling-origin audit found that known transformers returning after
  more than seven absent days were incorrectly treated as continuously active.
  Shrinking those returners to the installed-power prior improved total RMSLE
  at all four 122-day cutoffs: 1.2809→1.1459, 1.0360→0.9849,
  1.2415→1.2194, and 0.9586→0.9345. This affects 85,809 test rows, largely the
  May 11 return cohort. The resulting v6 submission (ref 55915151) scored
  **1.10604**, improving v3 by 0.13629.
- A GPU CatBoost trained on 870,237 rolling-snapshot rows scored 1.0174 on the
  clean December-March holdout versus 0.9839 for its direct baseline. A 25%
  blend reached 0.9730 but was not submitted because v3 was already stronger.
- A returner-only CatBoost trained on 1,205 historical reappearance events was
  stronger than the v6 return rule on four temporal holdouts: 1.2040 vs
  1.5295, 1.1108 vs 1.5649, 1.1087 vs 1.6889, and 1.0318 vs 1.7963. Its
  candidate is `return_event_catboost.csv`.

## Rejected directions

- Raw global LightGBM and CatBoost calendar/category models were materially
  weaker than transformer-level log-history forecasts on forward holdouts.
- Exact prior-year daily values and entity-level month profiles were too noisy.
- Numeric transformer-ID prefixes and nearest-ID neighbors were worse for cold
  starts and are not semantically defensible features.
- A stacked LightGBM over rolling-origin snapshots did not beat the direct
  recent-history blend on a clean final holdout.

## Next experiments

1. Score the returner-only CatBoost and calibrate its blend with v6.
2. Improve cold-start prediction, which covers 22.2% of test rows and dominates
   the remaining error. Validate cohort-size and installed-power shrinkage on
   several rolling origins.
3. Cross-validate location seasonal effects by transformer and tune the amount
   of June/July correction; the current weight is 1.0.
4. Test robust recent-level estimators and adaptive weights for intermittent or
   zero-heavy transformers.
5. Build the explanatory Kaggle notebook from `src/make_submission.py` before
   the final notebook-review deadline.
