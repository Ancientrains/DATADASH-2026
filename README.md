# DATADASH-2026
Repo for DataDash 2026

# SSU Data Dash 2026: Toronto Home-Price Prediction

Our submission to the SSU Data Dash 2026 case challenge. The task is to build an automated valuation model (AVM) that predicts what a Toronto home will sell for, then score it on synthetic sales from June to September 2026 that we have never seen. All data is synthetic, calibrated to the Toronto market (property-type price levels, neighbourhood trends, the 2021 to 2026 boom and correction), so no real listing, address or price is involved.

**Team:** Michael

## The problem

A pricing model can answer two different questions, and this project treats them as two separate tracks.

The **off-market** track asks what a home is worth before it is listed. There is no asking price, so the model has to rely on property features, location, recent comparable sales, market trends and the listing text. This is the hard question and it carries double weight in scoring. The off-market model never uses the list price.

The **list-aware** track asks what a listed home will close at, given its list price. The list price already carries most of what the market knows, which makes this track much easier.

Two pitfalls shape the whole design. The first is mixing the tracks up: a list-aware result reported as an off-market one overstates what the model can do. The second is time leakage: a model that learns from sales after the one it predicts looks excellent in testing and fails in use, which is exactly what a random train/test split does. Everything here trains on the past and tests on the future.

## Data

The data comes from `huggingface.co/datasets/Baliyang/SSU_data_dash` and the starter code downloads it automatically.

| File | Rows | Description |
|---|---|---|
| `listings` | 150,000 | Sales from January 2021 to May 2026 with sold prices (training data) |
| `test_off_market` | 3,974 | Homes to value before listing, with no list price |
| `test_list_aware` | 4,064 | Listed homes with their list price |

The two test files contain different homes. Columns cover the target (`sold_price`), listing details (`list_price`, dates, `days_on_market`), property attributes (type, style, bedrooms, washrooms, parking, garage, area, lot size), location (municipality, community, latitude, longitude), the listing description (`remarks`), and a third-party estimate (`external_avm_estimate`) that exists only in the training listings. Location is synthetic, so neighbourhood names and coordinates will not line up with real maps or census data. The third-party estimate is not available at test time, so any use of it has to respect its date and the fact that the model will not have it at prediction time.

## Method

_Describe the final approach here: features (location, comparable recent sales, market trends, text), models and any ensembling, and how uncertainty is estimated._

Validation is walk-forward. Each month from December 2025 to May 2026 is predicted using only sales that happened before it, which mirrors how the hidden test is scored. Any feature built from neighbourhood or market aggregates is computed from earlier sales only.

## Results

Reported on the starter's walk-forward validation (December 2025 to May 2026). MdAPE is the median of |predicted − sold| / sold, and MAPE is the mean.

| Track | Model | MdAPE | MAPE |
|---|---|---|---|
| Both | Maximum error: median sale of same type and neighbourhood, previous 12 months | 17.7% | 22.0% |
| Off-market | Starter baseline: median of 10 nearest earlier sales | 20.8% | 27.4% |
| Off-market | **Ours** | _tbd_ | _tbd_ |
| List-aware | Starter baseline: the list price | 7.3% | 8.1% |
| List-aware | **Ours** | _tbd_ | _tbd_ |

To earn accuracy points on a track, hidden-test MdAPE must beat the maximum-error baseline, which the starter's off-market baseline does not.

## Error analysis

_Where the model fails by price band, property type and neighbourhood, and why. Include the prediction-interval coverage if uncertainty is estimated._

## Reproducing our results

```bash
pip install -r requirements.txt
python run.py validate   # walk-forward test, December 2025 to May 2026
python run.py report     # results table and figure for the report
python run.py predict    # writes both submission files
python run.py check      # validates the submission files before hand-in
```

Features live in `features.py` and models in `models.py`. The predict step writes two files with the columns `ml_num,predicted_price`:

```
submissions/submission_off_market.csv     # one row per home in test_off_market
submissions/submission_list_aware.csv     # one row per home in test_list_aware
```

## Repository layout

```
features.py        feature engineering
models.py          models and ensembling
run.py             validate / report / predict / check
report/report.tex  4-page report (Overleaf-compatible)
submissions/       final prediction files
```

## Rules we followed

The off-market model does not use the list price. Outside data, if any, covers May 2026 or earlier and is listed with its URL in the report. No real listing or sales records were used, and no attempt was made to recover the hidden test prices. Every number in the video comes from our own validation.

## Links

Report (PDF): _link_  
Video (5 min, YouTube unlisted): _link_

## Timeline

Case released October 8, submissions due November 8 at 11:59 pm EST, round 1 review November 9 to 15, finalists announced November 16, and the in-person finale on November 20.
