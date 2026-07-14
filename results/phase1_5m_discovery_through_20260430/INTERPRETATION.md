# Corrected Phase 1 interpretation

The corrected scan evaluated 118,716 feature-target relationships using exact
one-way session-clustered inference and a global Benjamini-Hochberg promotion
gate. The broad screen and hybrid exact diagnostics agreed.

The primary credible family is early-session relative-strength and
price-location continuation into the remaining session. Representative exact
results produced correlations around 0.03 and top-minus-bottom decile spreads
around 18 to 29 basis points, with broad year, symbol, time, and outlier
stability. The 250 exact candidates are mostly redundant variants of this one
family and must not be interpreted as 250 independent anomalies.

Secondary findings include statistically significant but economically small
one-bar reversal over 5 to 25 minutes and short-horizon breadth continuation.
These effects are generally below one basis point and require realistic cost
testing before promotion.

## Excluded numerical artifacts

The following signed-feature rolling-mean ratios are invalid because their
denominators cross zero and can create extreme ratios:

- `return_1_mean_ratio_1560`
- `return_1_mean_ratio_4680`
- `distance_session_vwap_mean_ratio_1560`
- `distance_session_vwap_mean_ratio_4680`

Rows involving these four features remain in the immutable full result ledger
for auditability but are excluded from interpretation. The feature registry
implementation has been corrected so future runs do not generate them.

These are statistical relationships, not validated trading strategies. Stage
2 conditional, multivariable, strategy, bar-fill, quote-fill, and sealed
holdout work had not been run when these artifacts were produced.
