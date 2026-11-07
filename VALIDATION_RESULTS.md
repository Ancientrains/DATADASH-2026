# Validation results

Run commit: `76ec8f8` (the analysis-only commit on top of the current
pipeline). Run command:

```bash
DATADASH_DATA=/path/to/data python run.py validate
python run.py report
```

The validator uses monthly walk-forward blocks from December 2025 through May
2026. For each block, it trains on sales strictly before the first day of that
month and scores homes sold in that month. `n` is the number of distinct held-out
homes per track; the prediction file has twice as many rows because it contains
both tracks.

| Track | Config | n | MdAPE | MAPE | MdSE |
|---|---|---:|---:|---:|---:|
| Off-market | condition keywords + as-of comps + seasonal/damped trend + 3 seeds | 10,069 | 7.19% | 9.52% | -0.31% |
| List-aware | log(sold/list) + raw as-of comps + time-blocked off-market stack + 3 seeds | 10,069 | 3.82% | 4.82% | 0.07% |

These are validation results, not hidden-test scores.
