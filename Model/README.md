# SSU Data Dash 2026 - valuation pipeline

Modules: common.py (config, cleaning, 22 remark flags) | trend.py (as-of segment index, seasonality, damped-trend
forecast) | comps.py (hierarchical as-of comparables) | features.py (frames) | models.py (stage 1 / stage 2 LightGBM,
tuning) | run.py (CLI) | diagnostics.py, test_part1.py, list_benchmark.py (checks).

Order of use
    python test_part1.py                              # unit tests
    python diagnostics.py                             # data / flag / index / comps checks
    python run.py tune --stage off_market --n-iter 30 # writes params/off_market.json
    python run.py tune --stage list_aware --n-iter 30 # writes params/list_aware.json (needs the off-market params)
    python run.py validate                            # F1, F2, S25 + list_price baseline -> reports/validation.csv
    python run.py predict                             # submission_off_market.csv, submission_list_aware.csv
    python run.py check                               # leakage / OOT / determinism / submission checks

Add --fast to any run.py command for a smoke test (tiny rounds; does not write params).
For list-aware text ablations, add `--list-text a`, `--list-text b`, or
`--list-text ab`; the default `none` preserves the baseline. Option A adds
row-local rule flags. Option B fits TF-IDF plus deterministic TruncatedSVD only
on eligible historical stage-2 rows for each fold/final fit. These options never
enter the off-market model.
Commit params/*.json so judges reproduce the tuned models.  Delete cache/ to force recomputation.
Constants to confirm: ID_COL / PRED_COL / submission file names (common.py), FIX_ORIGINAL_PATTERN, TUNE_WEIGHTS, BLOCK_SEEDS.
