from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScanConfig:
    catalog_path: str = "D:/AlgoResearch/data/catalog.duckdb"
    source_table: str = "derived_bars_5m"
    feed: str = "sip"
    adjustment: str = "raw"
    start: str = "2019-06-21"
    selection_end: str | None = None
    confirmation_start: str | None = None
    discovery_end: str = "2026-04-30"  # sealed holdout begins 2026-05-01
    sealed_holdout_start: str = "2026-05-01"
    allow_holdout_access: bool = False
    use_separate_confirmation_period: bool = False
    run_historical_walk_forward_diagnostics: bool = True
    run_recent_period_diagnostics: bool = True
    run_recency_weighted_diagnostics: bool = True
    recency_half_lives_months: list[int] = field(default_factory=lambda: [6, 12, 24])
    universe: list[str] = field(default_factory=list)
    membership_table: str = "qqq_pit_membership_daily"
    membership_source_quality: str = "effective_date_reconstruction"
    require_membership: bool = True
    benchmark_symbols: list[str] = field(default_factory=lambda: ["QQQ"])
    corporate_actions_path: str = "D:/AlgoResearch/Quant Pipeline/reference/corporate_actions_through_20260430.parquet"
    require_corporate_actions: bool = True
    exchange_calendar: str = "XNYS"
    maximum_missing_bars_per_session: int = 0
    decision_times_et: list[str] = field(default_factory=list)
    lookbacks: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 48, 60, 78])
    target_horizons_minutes: list[int] = field(default_factory=lambda: list(range(5, 390, 5)))
    primary_target_horizons_minutes: list[int] = field(default_factory=lambda: [5, 15, 30, 60, 120])
    quantiles: int = 10
    min_observations: int = 500
    min_sessions: int = 100
    min_symbols: int = 20
    min_decision_timestamps: int = 100
    min_years: int = 3
    min_bin_observations: int = 30
    outlier_policy: str = "none"  # none|winsorize_1pct
    output_root: str = "D:/AlgoResearch/Quant Pipeline/runs"
    experiment_id: str = "phase1_final_discovery_through_20260430"
    benchmark_symbol: str = "QQQ"
    sector_map_path: str | None = None
    use_cuda: bool = True
    cuda_device: str = "cuda:0"
    cuda_target_batch_group_size: int = 3
    scan_batch_rows: int = 250_000
    feature_chunk_size: int = 24
    feature_build_batch_chunks: int = 2
    feature_workers: int = 16
    exact_workers: int = 6
    target_chunk_size: int = 4
    target_build_batch_chunks: int = 2
    resume: bool = True
    checkpoint_every_pairs: int = 25
    normalization_windows_sessions: list[int] = field(default_factory=lambda: [20, 60])
    opening_windows_minutes: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 15, 20, 30, 45, 60, 90])
    cross_sectional_min_symbols: int = 5
    multiple_testing_grouping: str = "feature_family_target_family"
    sensitivity_outlier_policy: str = "winsorize_1pct"
    exact_bootstrap_samples: int = 500
    primary_fdr_threshold: float = 0.05
    minimum_effect_bps: float = 1.0
    confirmation_min_sessions: int = 100
    confirmation_min_symbols: int = 20
    confirmation_min_effect_bps: float = 1.0
    confirmation_min_discovery_ratio: float = 0.25
    confirmation_max_discovery_ratio: float = 4.0
    confirmation_alpha: float = 0.10
    confirmation_min_positive_fold_fraction: float = 0.60
    beta_window_sessions: int = 60
    beta_min_observations: int = 40
    max_candidates_per_feature_family: int = 20
    max_candidates_per_cluster: int = 3
    max_candidates_per_target_family: int = 30
    regime_min_observations: int = 500
    regime_min_sessions: int = 100
    regime_min_symbols: int = 20
    exact_time_min_observations: int = 500
    exact_time_min_sessions: int = 100
    exact_time_min_symbols: int = 20
    scope_min_observations: int = 500
    scope_min_sessions: int = 100
    scope_min_symbols: int = 3
    trend_threshold_bps: float = 20.0
    gap_threshold_bps: float = 20.0
    market_wide_min_direction_pct: float = 0.60
    market_wide_max_group_share: float = 0.40
    specific_scope_min_group_share: float = 0.60
    cache_schema_version: str = "phase1_final"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScanConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        config=cls(**values); config.validate(); return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        import pandas as pd
        if pd.Timestamp(self.discovery_end)>=pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("discovery_end must precede sealed_holdout_start")
        if self.allow_holdout_access:raise ValueError("Phase 1 configuration may not allow holdout access")
        if self.use_separate_confirmation_period and (not self.selection_end or not self.confirmation_start):
            raise ValueError("Separate confirmation mode requires selection_end and confirmation_start")
        if any(value<=0 for value in self.recency_half_lives_months):raise ValueError("Recency half-lives must be positive")
        if self.feature_build_batch_chunks<=0:raise ValueError("feature_build_batch_chunks must be positive")
        if self.target_build_batch_chunks<=0:raise ValueError("target_build_batch_chunks must be positive")
        if self.cuda_target_batch_group_size<=0:raise ValueError("cuda_target_batch_group_size must be positive")
