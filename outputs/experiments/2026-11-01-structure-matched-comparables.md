# Experiment record: structure-matched within-building comparables

**Status:** Rejected.  This record is retained to document model selection and
avoid presenting a selectively chosen validation result.

## Question

Would matching a home to physically similar earlier units at the same coordinate
improve the final off-market model beyond its existing building, repeat-sale,
community and KNN comparable features?

## Preregistered design

The test ran on branch `codex/structure-matched-comps-experiment` using the
E1-safe pipeline and the existing three LightGBM seeds.  No hyperparameters,
target, trend model, keyword features, or matching weights were tuned after
seeing fold outcomes.

For each query home, candidates were restricted to sales that were:

- at the same rounded coordinate and the same property type;
- sold strictly before the as-of reference date;
- at least three eligible candidates.

The fixed structural distance used log area, bedrooms, washrooms, parking and
garage.  Features were the median matched log price, matched price per square
foot, estimated matched price, median matching distance and candidate count.
The existing fallbacks remained active when fewer than three matches existed.

**Adoption rule:** all of F1, F2 and S25 must improve off-market MdAPE by at
least 0.15 percentage points, with no material condo/freehold regression.

## Result

| Horizon-matched fold | E1-safe baseline MdAPE | Matched-comparable MdAPE | Change |
|---|---:|---:|---:|
| F1 (2026-02 to 2026-05) | 6.95% | 6.99% | +0.04 points |
| F2 (2025-10 to 2026-01) | 6.88% | 6.83% | -0.05 points |
| S25 (2025-06 to 2025-09) | 7.17% | 7.03% | -0.14 points |

## Decision and explanation

The experiment is rejected.  Although it improved the seasonal S25 fold, it
was worse on the most recent F1 fold and did not meet the predeclared
0.15-point improvement threshold on any fold.  The observed differences are
also smaller than the handoff's estimated seed noise (about +/-0.15 MdAPE
points).  Keeping it would add feature complexity and a further selection
opportunity without reliable evidence of hidden-test benefit.

The experiment implementation remains only on its dedicated branch and is not
part of the E1-safe baseline design.
