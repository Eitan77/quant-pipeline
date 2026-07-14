# Final Phase 1 discovery repair notes

This revision repairs Phase 1 without opening the sealed holdout beginning
May 1, 2026. The next run must use
`phase1_final_discovery_through_20260430`; old caches are incompatible.

## Data and timing corrections

- Intraday returns, rolling momentum/volatility, bar sequences, run lengths,
  VWAP slopes, and VWAP crosses reset at every session and bar-gap boundary.
- Continuous close-to-close returns are separately named.
- Research OHLC/VWAP is split adjusted from the pinned corporate-action
  ledger. Cross-session returns use the total-return series, including cash
  dividends; target entries and exits retain raw execution prices.
- Daily point-in-time membership controls the tradable cross-section. QQQ has
  an explicit non-tradable benchmark role.
- The XNYS calendar supplies holidays, DST-aware opens/closes, and shortened
  sessions. Missing bars split rolling sequences and incomplete sessions are
  excluded under the configured policy.
- Features are built before optional decision-time filtering. Time variables
  use the completed-bar availability timestamp.
- Targets locate the first bar starting at or after the decision and store raw
  entry price, raw exit price, exit timestamp, and actual horizon.

## Statistical corrections

- Broad screening uses pair-specific observations, sessions, symbols,
  decision timestamps, and years.
- Exact diagnostics add two-way date/symbol clustering, session-level
  quantile uncertainty, session bootstrap intervals, HAC daily spreads, and
  HAC cross-sectional IC.
- Categorical variables use a clustered dummy-variable omnibus scan instead
  of Pearson or an unclustered rank test.
- Normalization denominators are explicitly prior-only or inclusive.
- Primary promotion targets are 5, 15, 30, 60, 120 minutes, and EOD. The full
  five-minute horizon grid remains exploratory only.
- Raw, benchmark-adjusted, and prior-only rolling-beta residual target
  families are registered separately.
- Candidate clustering combines economic family, feature-value correlation,
  parameter redundancy, target-horizon family, and response direction.
- Final discovery and candidate ranking use every eligible observation through
  April 30, 2026. Annual folds are labeled historical subperiod stability,
  never independent confirmation.
- Recent 5-year, 3-year, 2-year, 12-month, January-April 2026, and optional
  session-weighted half-life diagnostics are reported without replacing the
  unweighted primary test.
- Symbol breadth, effect concentration, leave-top-symbol, time-of-day, and
  available sector, industry, liquidity, and market-regime diagnostics guide
  Phase 2 advancement recommendations.
- Candidate ranking exposes every score, penalty, and hard gate.

## Engineering and reports

- A fingerprint covers configuration, package source, Git revision, source
  provenance, registries, corporate actions, membership source, sector map,
  and exchange-calendar version. Mismatched caches are refused.
- Reports separate broad tests, coverage-qualified tests, primary and
  exploratory FDR families, exact candidates, redundancy clusters, and robust
  Phase 1 anomaly candidates.
- Coverage and feature-build ledgers include skipped and unavailable items.
- The manifest records the data snapshot, missing bars, excluded sessions,
  adjustment sources, and the sealed-holdout boundary.
- Cache metadata records sorted row keys, row counts, key hashes, schemas, and
  fingerprints; resume refuses reordered, duplicated, or mismatched caches.
- Every target batch reruns timing, uniqueness, eligibility, price, and
  holdout checks before screening.

## Deliberate fail-closed limitations

- The configured membership source is an effective-date reconstruction. Its
  quality label is preserved in every feature cache and manifest; it is not
  represented as an exchange-native announcement-time history.
- Sector-dependent features remain explicitly unavailable because no reviewed
  point-in-time sector map is configured; they are not approximated.
- Final one-day and one-month pre-holdout real-data smoke tests, including
  cache resume, recent diagnostics, concentration reports, and centralized
  holdout rejection, completed successfully. The full run has not started and
  no data dated May 1, 2026 or later was inspected or used.
