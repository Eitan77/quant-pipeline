# Phase 2 execution-first candidate report

## Verdict

**0 of 10 distinct mechanisms met the requested "holy shit" standard; 2 of 10 passed the minimum descriptive execution-qualification gate.** The two survivors are low-return research sleeves, not money printers. A pass here means discovery-period quote execution survived, not that the sealed holdout passed or that live trading is approved. Any failed row is explicitly rejected; rebate-only results are not promoted.

Gate: at least 50% exact-window quote-path coverage, at least 5% paired fill rate, 100 filled orders, and 40 active sessions; positive +1 bp CAGR in at least two of three chronological folds; +3 bp CAGR no worse than -0.20% (approximately break-even); maximum drawdown no worse than -5%; and peak-to-trough duration no longer than 31 calendar days.

## Selected distinct mechanisms and execution grid

Slippage/cost is per side and is added after the observed SIP bid/ask or paired passive fill. Delay is explicit in the `delay_seconds` column: 0 is the required quote-base model and 5 is the conservative timing variant. Passive fills require the quote to trade through the resting price within the ten-second path; unmatched long/short fills are canceled.

| candidate_id | mechanism_id | execution | delay_seconds | filled | fill_rate | coverage | positive_folds | cagr | stress_cagr_3bp | max_dd | pt_days | status | cagr_-1bp | cagr_-0.5bp | cagr_0bp | cagr_1bp | cagr_3bp | cagr_5bp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S09_gap_reaction_q95__passive__d5 | S09_gap_reaction | passive | 5 | 132 | 6.88% | 66.04% | 3 | 1.49% | 1.34% | -0.26% | 2 | execution_qualified_discovery | 1.64% | 1.60% | 1.57% | 1.49% | 1.34% | 1.19% |
| S08_breadth_reversal_q99__market__d0 | S08_breadth_reversal | market | 0 | 238 | 48.77% | 67.62% | 3 | 0.90% | 0.56% | -1.19% | 22 | execution_qualified_discovery | 1.23% | 1.15% | 1.06% | 0.90% | 0.56% | 0.22% |
| S04_extreme_momentum_q95__market__d0 | S04_extreme_momentum | market | 0 | 2516 | 57.21% | 73.74% | 2 | 4.62% | -0.68% | -2.43% | 72 | rejected | 10.21% | 8.79% | 7.38% | 4.62% | -0.68% | -5.72% |
| S03_opening_continuation_q95__passive__d5 | S03_opening_continuation | passive | 5 | 256 | 3.16% | 64.75% | 2 | 0.89% | -0.03% | -0.35% | 75 | rejected | 1.81% | 1.58% | 1.34% | 0.89% | -0.03% | -0.93% |
| S05_volume_confirmed_q95__market__d0 | S05_volume_confirmed | market | 0 | 2492 | 59.67% | 75.45% | 2 | 3.83% | -1.62% | -2.32% | 46 | rejected | 9.58% | 8.11% | 6.66% | 3.83% | -1.62% | -6.79% |
| S10_bar_structure_q99__market__d5 | S10_bar_structure | market | 5 | 198 | 34.26% | 57.61% | 1 | 1.46% | 0.38% | -1.13% | 201 | rejected | 2.56% | 2.28% | 2.01% | 1.46% | 0.38% | -0.69% |
| S07_market_residual_q99__market__d5 | S07_market_residual | market | 5 | 472 | 36.65% | 57.07% | 2 | 0.80% | -0.27% | -1.29% | 67 | rejected | 1.87% | 1.60% | 1.33% | 0.80% | -0.27% | -1.33% |
| S02_range_location__passive__d5 | S02_range_location | passive | 5 | 466 | 5.80% | 77.85% | 1 | 0.10% | -1.45% | -0.68% | 34 | rejected | 1.67% | 1.27% | 0.88% | 0.10% | -1.45% | -2.97% |
| S06_vwap_displacement_q99__passive__d5 | S06_vwap_displacement | passive | 5 | 108 | 7.64% | 54.95% | 1 | -1.27% | -2.59% | -4.51% | 196 | rejected | 0.06% | -0.27% | -0.61% | -1.27% | -2.59% | -3.88% |
| S01_session_momentum__passive__d0 | S01_session_momentum | passive | 0 | 1092 | 13.60% | 80.08% | 1 | -1.65% | -8.57% | -6.77% | 186 | rejected | 5.77% | 3.87% | 1.99% | -1.65% | -8.57% | -14.99% |

## Chronological quote-period results at +1 bp per side

| candidate_id | cagr_development | cagr_recent | cagr_train | max_dd_development | max_dd_recent | max_dd_train | pt_days_development | pt_days_recent | pt_days_train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S09_gap_reaction_q95__passive__d5 | 0.69% | 2.61% | 1.25% | -0.05% | -0.07% | -0.26% | 81.0 | 3.0 | 2.0 |
| S08_breadth_reversal_q99__market__d0 | 0.50% | 1.42% | 0.81% | -0.30% | -1.19% | -0.10% | 33.0 | 22.0 | 68.0 |
| S04_extreme_momentum_q95__market__d0 | 11.40% | 7.57% | -4.25% | -1.25% | -1.83% | -2.43% | 18.0 | 71.0 | 72.0 |
| S03_opening_continuation_q95__passive__d5 | 0.54% | -0.20% | 2.35% | -0.11% | -0.20% | -0.35% | 24.0 | 85.0 | 75.0 |
| S05_volume_confirmed_q95__market__d0 | 9.45% | 7.75% | -4.92% | -0.72% | -1.96% | -2.32% | 10.0 | 29.0 | 46.0 |
| S10_bar_structure_q99__market__d5 | -0.87% | 7.86% | -2.17% | -0.60% | -0.66% | -0.78% | 19.0 | 5.0 | 95.0 |
| S07_market_residual_q99__market__d5 | 0.97% | 3.93% | -2.34% | -0.84% | -0.90% | -1.15% | 35.0 | 15.0 | 105.0 |
| S02_range_location__passive__d5 | 1.59% | -0.94% | -0.35% | -0.46% | -0.65% | -0.68% | 22.0 | 92.0 | 34.0 |
| S06_vwap_displacement_q99__passive__d5 | -2.82% | 4.07% | -4.80% | -2.96% | -1.55% | -2.41% | 73.0 | 35.0 | 21.0 |
| S01_session_momentum__passive__d0 | -7.96% | 6.54% | -2.93% | -3.66% | -1.96% | -3.70% | 120.0 | 35.0 | 47.0 |

## Fixed-spec historical bar diagnostic at +1 bp per side

This backward diagnostic uses completed-bar prices, not quotes, and is therefore secondary evidence only. It is included to expose long-history instability rather than to claim executable performance.

| candidate_id | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S01_session_momentum | -33.93% | -51.13% | 24.49% | 1.30% | -34.71% | -18.74% | 4.95% | 50.27% |
| S02_range_location | -24.42% | -16.07% | -4.75% | -16.05% | -12.15% | -14.14% | 2.09% | 34.33% |
| S03_opening_continuation_q95 | 7.03% | -5.43% | -3.46% | -17.36% | -17.01% | -13.51% | -3.30% | -37.20% |
| S04_extreme_momentum_q95 | -7.40% | -7.09% | 2.54% | -1.91% | -4.91% | -4.48% | 2.03% | 16.62% |
| S05_volume_confirmed_q95 | -7.24% | -6.85% | 3.40% | -2.70% | -2.20% | -4.78% | 3.06% | 17.16% |
| S06_vwap_displacement_q99 | -7.67% | -19.57% | -12.55% | -12.30% | -19.22% | -4.52% | -25.97% | 29.71% |
| S07_market_residual_q99 | -1.52% | -4.57% | -0.29% | -2.76% | -4.83% | -4.42% | -4.07% | 13.11% |
| S08_breadth_reversal_q99 | -0.65% | -0.27% | -0.25% | -1.30% | -0.80% | -2.40% | 2.14% | 3.76% |
| S09_gap_reaction_q95 | 2.58% | 1.37% | 5.62% | 11.72% | 4.04% | 5.97% | -3.40% | 5.33% |
| S10_bar_structure_q99 | -2.59% | 3.81% | -5.70% | -7.04% | -1.86% | -1.39% | 3.23% | 5.82% |

## Mechanical leverage diagnostic at +1 bp per side

| candidate_id | leverage | cagr | sharpe | max_dd | pt_days |
| --- | --- | --- | --- | --- | --- |
| S01_session_momentum__passive__d0 | 1.0 | -1.65% | -0.34887498891560664 | -6.77% | 186 |
| S01_session_momentum__passive__d0 | 2.0 | -3.48% | -0.34887498891560664 | -13.16% | 186 |
| S01_session_momentum__passive__d0 | 4.0 | -7.58% | -0.34887498891560664 | -24.85% | 186 |
| S02_range_location__passive__d5 | 1.0 | 0.10% | 0.08561789337175511 | -0.68% | 34 |
| S02_range_location__passive__d5 | 2.0 | 0.18% | 0.08561789337175511 | -1.36% | 34 |
| S02_range_location__passive__d5 | 4.0 | 0.30% | 0.08561789337175511 | -2.70% | 34 |
| S03_opening_continuation_q95__passive__d5 | 1.0 | 0.89% | 0.7140974324419728 | -0.35% | 75 |
| S03_opening_continuation_q95__passive__d5 | 2.0 | 1.76% | 0.7140974324419728 | -0.70% | 75 |
| S03_opening_continuation_q95__passive__d5 | 4.0 | 3.50% | 0.7140974324419728 | -1.40% | 75 |
| S04_extreme_momentum_q95__market__d0 | 1.0 | 4.62% | 1.1333690344777492 | -2.43% | 72 |
| S04_extreme_momentum_q95__market__d0 | 2.0 | 9.28% | 1.1333690344777492 | -4.82% | 72 |
| S04_extreme_momentum_q95__market__d0 | 4.0 | 18.64% | 1.1333690344777492 | -9.49% | 72 |
| S05_volume_confirmed_q95__market__d0 | 1.0 | 3.83% | 0.925880302476687 | -2.32% | 46 |
| S05_volume_confirmed_q95__market__d0 | 2.0 | 7.62% | 0.925880302476687 | -4.61% | 46 |
| S05_volume_confirmed_q95__market__d0 | 4.0 | 15.02% | 0.925880302476687 | -9.07% | 46 |
| S06_vwap_displacement_q99__passive__d5 | 1.0 | -1.27% | -0.31875753087025716 | -4.51% | 196 |
| S06_vwap_displacement_q99__passive__d5 | 2.0 | -2.66% | -0.31875753087025716 | -8.88% | 196 |
| S06_vwap_displacement_q99__passive__d5 | 4.0 | -5.78% | -0.31875753087025716 | -17.25% | 196 |
| S07_market_residual_q99__market__d5 | 1.0 | 0.80% | 0.3784817673850532 | -1.29% | 67 |
| S07_market_residual_q99__market__d5 | 2.0 | 1.55% | 0.3784817673850532 | -2.57% | 67 |
| S07_market_residual_q99__market__d5 | 4.0 | 2.94% | 0.3784817673850532 | -5.10% | 67 |
| S08_breadth_reversal_q99__market__d0 | 1.0 | 0.90% | 0.6243205038243623 | -1.19% | 22 |
| S08_breadth_reversal_q99__market__d0 | 2.0 | 1.78% | 0.6243205038243623 | -2.37% | 22 |
| S08_breadth_reversal_q99__market__d0 | 4.0 | 3.50% | 0.6243205038243623 | -4.71% | 22 |
| S09_gap_reaction_q95__passive__d5 | 1.0 | 1.49% | 2.2356979959173784 | -0.26% | 2 |
| S09_gap_reaction_q95__passive__d5 | 2.0 | 3.00% | 2.2356979959173784 | -0.52% | 2 |
| S09_gap_reaction_q95__passive__d5 | 4.0 | 6.07% | 2.2356979959173784 | -1.05% | 2 |
| S10_bar_structure_q99__market__d5 | 1.0 | 1.46% | 0.5966943652143749 | -1.13% | 201 |
| S10_bar_structure_q99__market__d5 | 2.0 | 2.88% | 0.5966943652143749 | -2.26% | 201 |
| S10_bar_structure_q99__market__d5 | 4.0 | 5.59% | 0.5966943652143749 | -4.51% | 201 |

## Distinctness

The final comparison contains exactly one configuration per economic mechanism. Maximum absolute pairwise daily-return correlation was 0.739; maximum filled-trade Jaccard overlap was 0.452. Full matrices are in `daily_return_correlations.csv` and `strategy_overlap.csv`.

## Integrity and limitations

- Source universe: point-in-time `analysis_eligible` symbols, 2025-05-01 through 2026-04-30 for quote replay; full bar diagnostic begins 2019-06-21.
- Extreme thresholds were frozen from 2025-05-01 through 2025-08-31 on the complete bar universe, never from quote availability.
- The original strict 95%-coverage/all-fold gate produced zero passes. The displayed operational gate was documented after the event-style quote-coverage audit, so it is descriptive rather than prospective and cannot itself authorize a holdout test.
- Sealed holdout begins 2026-05-01 and was not accessed. This report cannot honestly call any candidate holdout-validated or live-approved.
- Quote replay rejects missing, locked/crossed, non-positive-size-invalid, and wider-than-10-bp snapshots. Missing/partial fills remain unfilled.
- Passive quote trade-through does not prove exchange queue priority. Borrow/locate, fees beyond the explicit grid, capacity, and live order-management remain external gates.
- Standalone full-capital statistics can imply concentration. A future multi-sleeve portfolio must net symbols and apply the common 5%/10%/20% symbol-cap grid before leverage.

## Reproducible artifacts

- `selected_candidate_statuses.csv`: objective selection and gate verdicts.
- `selected_quote_cost_stress.csv`: every selected strategy at -1, -0.5, 0, +1, +3, and +5 bp per side.
- `selected_quote_period_stats.csv`: every selected strategy in every chronological quote fold.
- `selected_historical_annual_stats.csv`: fixed-spec annual bar diagnostics.
- `fill_status_counts.csv` in the quote run: filled, unfilled, canceled-unpaired, and missing paths.
- `failed_configurations.csv`: every tested variant not selected plus selected mechanisms that failed the gate.
