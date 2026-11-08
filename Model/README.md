# SSU Data Dash 2026 - valuation pipeline

Modules: common.py (config, cleaning, 22 remark flags) | trend.py (as-of segment index, seasonality, damped-trend
forecast) | comps.py (hierarchical as-of comparables) | features.py (frames) | models.py (stage 1 / stage 2 LightGBM,
tuning) | run.py (CLI).

Order of use
    python run.py check                               # leakage / OOT
    python run.py tune --stage off_market --n-iter 30 # writes params/off_market.json
    python run.py tune --stage list_aware --n-iter 30 # writes params/list_aware.json (needs the off-market params)
    python run.py validate                            # F1, F2, S25 + list_price baseline -> reports/validation.csv
    python run.py predict                             # submission_off_market.csv, submission_list_aware.csv
    python run.py check                               # Submission Verification

Add --fast to any run.py command for a smoke test (tiny rounds; does not write params).
Commit params/*.json so judges reproduce the tuned models.  Delete cache/ to force recomputation.
Constants to confirm: ID_COL / PRED_COL / submission file names (common.py), FIX_ORIGINAL_PATTERN, TUNE_WEIGHTS, BLOCK_SEEDS.
