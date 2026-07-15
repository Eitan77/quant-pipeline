# Phase 1 final interpretation

## Verdict

Phase 1 completed successfully using discovery data from June 21, 2019 through
April 30, 2026. The sealed holdout beginning May 1, 2026 was not accessed.
These results are anomaly evidence, not executable strategy returns.

The corrected scan evaluated 186,732 unique feature-target pairs, promoted 80
pairs to exact diagnostics, identified 16 robust Phase 1 candidates, and marked
four additional candidates as requiring Phase 2 investigation. No duplicate
pairs or invalid p-values/FDR values were present in the final artifacts.

## Most credible research families

1. **Intraday price-location continuation.** High session range position,
   close location, and closely related range-position features predict positive
   benchmark-adjusted returns over 30-120 minutes. The strongest representative,
   `session_range_position -> fwd_return_120m_benchmark_adjusted`, produced a
   3.36 bps full-sample top-minus-bottom spread, 6.06 bps over the recent 12
   months, and 11.49 bps during January-April 2026.
2. **Opening breakout/breakdown continuation.** A 5-minute opening breakout
   predicts positive subsequent benchmark-adjusted returns, while an opening
   breakdown predicts negative subsequent returns. At the 120-minute horizon,
   the signed full-sample spreads were approximately 2.10 bps and 2.41 bps,
   respectively.
3. **VWAP-slope continuation.** Positive VWAP slope predicts positive
   120-minute beta-residual returns. The full-sample spread was 2.09 bps and the
   recent 12-month signed effect was 4.65 bps.
4. **Market-residual reversal.** Large market-residual moves show subsequent
   reversal, including toward the close, but this family weakened materially in
   recent periods and should be lower priority.
5. **Conditional higher-high relationship.** The automated diagnostic marked
   `higher_high -> fwd_return_120m_beta_residual` for fixed, predeclared
   trend-state testing. This is a conditional Phase 2 hypothesis, not an
   independent discovery.

Range-position and close-location variants are strongly redundant. They should
be treated as representatives of one economic family rather than separate
independent discoveries.

## Durability and limitations

- Most promoted effects were broad across symbols and persisted after removing
  the strongest exact decision time.
- The strongest timing commonly appeared near the open for price-location and
  opening-pattern families, but the exact-time diagnostics generally classified
  them as persistent through the session rather than single-timestamp effects.
- Point-in-time sector and industry metadata was unavailable. Sector and
  industry scope fields therefore explicitly report insufficient evidence.
- Most historical spreads are only 1-3 bps before trading costs. Phase 2 must
  test selectivity, turnover, execution delay, slippage, and realistic fills.
- January-April 2026 effects are encouraging but remain discovery-period
  evidence and must not be treated as holdout confirmation.

## Recommended Phase 2 order

1. Test one representative price-location continuation signal across fixed
   quantile thresholds, 30/60/120-minute holds, and exact decision times.
2. Test opening breakout and opening breakdown continuation as separate long and
   short hypotheses.
3. Test VWAP-slope continuation over the 120-minute beta-residual target.
4. Run the predeclared trend-state conditional test for the higher-high
   relationship.
5. Retain market-residual reversal for monitoring unless targeted conditioning
   restores a meaningful recent gross edge.

No holdout, quote-fill, bar-fill, or strategy optimization work is included in
this results package.

## Files

- `report.md`: generated Phase 1 report and candidate summaries.
- `ranked_candidates.html`: browsable ranked-candidate report.
- `detailed_candidates.csv`: complete exact diagnostics for all 80 candidates.
- `cluster_level_anomalies.csv`: redundant anomaly-family grouping.
- `candidate_regime_diagnostics.csv`: volatility, breadth, dispersion, trend,
  and gap breakdowns.
- `candidate_sector_diagnostics.csv`, `candidate_industry_diagnostics.csv`, and
  `candidate_scope_classification.csv`: scope results and explicit metadata
  availability statuses.
- `effect_by_exact_decision_time.csv`: exact America/New_York decision-time
  results.
- `feature_registry.csv`, `target_registry.csv`, `scan_coverage.csv`, and
  `feature_build_report.csv`: construction and coverage records.
- `manifest.json`: configuration, provenance, integrity, and holdout controls.
- `complete_machine_readable_tables.zip`: `master_results.csv`,
  `coverage_report.csv`, and `fingerprint.json` for the full corrected screen.

Raw bars, feature/target caches, diagnostic working frames, checkpoints,
journals, logs, and invalid archived artifacts are intentionally excluded.
