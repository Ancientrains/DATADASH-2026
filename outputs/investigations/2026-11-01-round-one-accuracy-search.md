# Round-one accuracy investigation

**Objective:** identify a hidden-test-safe improvement to the E1-safe final
pipeline without repeating Section 10 recommendations or expanding the tuning
search irresponsibly.

## Process

Two independent read-only idea reviews were commissioned outside Section 10:

1. Row-local/categorical/text/coordinate feature ideas.
2. Model and comparable-feature ideas.

A separate adversarial reviewer compared these suggestions to the Section 10
evidence and rejected overlaps, weak behavioral speculation, extra ensembling,
and tuning expansions.  It ranked only two experiments worth running:

1. Structure-matched within-building comparables.
2. Row-local geometry/range features.

Both used the same predeclared adoption rule: improve off-market MdAPE by at
least **0.15 percentage points on each** of chronological F1, F2, and S25.
The baseline was the E1-safe three-seed off-market model.

## Results

| Candidate | F1 change | F2 change | S25 change | Decision |
|---|---:|---:|---:|---|
| Structure-matched same-building comparables | +0.04 | -0.05 | -0.14 | Reject |
| Geometry/range feature package | +0.08 | -0.01 | -0.10 | Reject |

Negative values improve MdAPE; positive values worsen it.

## Conclusion

Neither candidate satisfied the all-fold threshold.  Differences were also at
or below the approximately +/-0.15-point seed noise documented in the
handoff.  We therefore retain the E1-safe baseline and record both negative
results rather than select the favorable seasonal-fold result after the fact.

Detailed records:

- [Structure-matched comparables](../experiments/2026-11-01-structure-matched-comparables.md)
- [Geometry/range features](../experiments/2026-11-01-geometry-range-features.md)
