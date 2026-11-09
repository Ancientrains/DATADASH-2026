# Experiment record: row-local geometry and range features

**Status:** Rejected.

## Frozen change

Added only row-local geometry fields: area low/high bounds and width, log area,
lot area and its log, missing/zero-lot flag, area per bedroom, and area per
washroom.  The E1-safe comparable cache, model settings, three seeds, and
off-market target were unchanged.

**Adoption rule:** improve MdAPE by at least 0.15 points on each of F1, F2 and
S25.

| Fold | E1-safe baseline | Geometry package | Change |
|---|---:|---:|---:|
| F1 | 6.95% | 7.03% | +0.08 |
| F2 | 6.88% | 6.87% | -0.01 |
| S25 | 7.17% | 7.07% | -0.10 |

The package is rejected: the changes are below the estimated +/-0.15-point
seed noise, it worsened F1, and it missed the preregistered all-fold gate.
