# Quant Pipeline

Clean-room quantitative feature discovery built around point-in-time
market data. The project does not consume prior strategy findings, backtests,
or trading results.

## Phase 1.3 final corrected scanner

Phase 1 builds registered features from completed, session-aligned bars and
tests them against forward returns that begin strictly after the information
cutoff. It includes:

- causal feature and target registries;
- timing, availability, uniqueness, and holdout validation;
- CUDA-batched discovery screening plus two-way clustered, session-bootstrap,
  HAC-spread, and HAC-IC exact inference;
- primary-target global, family, cluster, and pair-level FDR;
- hybrid GPU/CPU exact diagnostics for shortlisted candidates;
- year, symbol, time, outlier, and cross-sectional IC diagnostics;
- resumable caches, manifests, coverage ledgers, and reports.

This is anomaly research, not a trading engine. It does not report strategy
Sharpe ratios, portfolio returns, or claim that a relationship is tradable.

## Data boundary

Data is supplied externally through a DuckDB catalog and is never committed.
The included example configuration seals May 1, 2026 onward. Local `runs/`,
Parquet caches, databases, logs, charts, and result tables are gitignored.

## Installation and tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Running Phase 1

Edit the local catalog and output paths in `configs/discovery_5m.yaml`, then:

```powershell
python -m quant_pipeline_launcher configs/discovery_5m.yaml
```

CUDA is used when enabled and available; otherwise the correlation backend can
fall back to CPU. Generated caches remain under the configured local output
directory. Curated aggregate results can be copied into `results/` after
review.

See [PHASE1_3_FINAL_REPAIR_NOTES.md](PHASE1_3_FINAL_REPAIR_NOTES.md) for the
final repair contract, current data-source limitations, and rerun boundary.
