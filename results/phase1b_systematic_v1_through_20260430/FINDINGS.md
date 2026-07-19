# Phase 1B systematic findings

## Verdict

The Phase 1B systematic two-factor run completed successfully and is ready for
research use. It did **not** access the sealed holdout beginning May 1, 2026.
The run generated useful hypotheses, but its new systematic interactions did
not clear the robustness standard for direct strategy promotion.

## Run scale

- Discovery data ended April 30, 2026.
- 15,000 dual features were compiled and 13,873 were built.
- 249,714 feature-target pairs were screened; 239,639 had valid inference.
- 105 candidates received exact diagnostics.
- The source Phase 1A run remained immutable and was not rebuilt.
- CUDA was enabled for the broad screening path.

## What the exact diagnostics found

Across the 105 exact candidates:

- 59 were statistically significant discovery relationships.
- 25 new systematic interactions were classified as not robust.
- 16 retained the earlier robust Phase 1 anomaly-candidate label.
- 4 required Phase 2 testing and 1 remained exploratory.
- 90 were broad across symbols; 15 were moderately concentrated.
- 38 weakened recently and 20 were historically strong but currently weak.
- 22 were persistent, 19 strengthened recently, and 6 were regime-unstable.
- 72 were retained for monitoring, 32 were rejected as concentrated or
  unstable, and 1 was advanced for conditional Phase 2 testing.

The largest raw systematic spreads were mostly intersections of market
residual return with relative volume, especially for beta-residual returns
through the close. Those interactions failed promotion because their recent
effects retained too little of the full-history effect. Strong historical
significance alone was therefore not treated as a tradable edge.

## Evidence boundary

These files describe anomaly discovery, not an execution-qualified strategy.
No quote-fill validation or sealed-holdout performance is included. The exact
candidate table is the canonical machine-readable findings surface; the
feature and target registries document what was tested.

The local 283 MB `master_results.csv`, generated feature Parquets, fingerprint
cache, progress checkpoint, and duplicate JSON feature registry are excluded
because they are reproducible working artifacts rather than review outputs.
