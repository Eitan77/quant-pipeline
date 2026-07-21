
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 20)
DEFAULT_RETURN_WINDOWS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 40, 60)

TUPLE_FIELDS = {
    "target_horizons_sessions",
    "return_windows_sessions",
    "trend_windows_sessions",
    "location_windows_sessions",
    "volatility_windows_sessions",
    "volume_windows_sessions",
    "session_component_windows_sessions",
    "exact_cost_scenarios_bps_per_side",
}


@dataclass(frozen=True)
class InterdayConfig:
    experiment_id: str = "interday_2a_single_variable_through_20260430"

    catalog_path: str = "D:/AlgoResearch/data/catalog.duckdb"
    source_table: str = "derived_bars_5m"
    feed: str = "sip"
    source_adjustment: str = "raw"
    exchange_calendar: str = "XNYS"

    source_start: str = "2018-01-02"
    analysis_start: str = "2019-06-21"
    discovery_end: str = "2026-04-30"
    sealed_holdout_start: str = "2026-05-01"
    allow_holdout_access: bool = False

    output_root: str = "D:/AlgoResearch/Quant Pipeline/runs"

    security_id_column: str = "security_id"
    security_master_table: str | None = None
    require_stable_security_id: bool = True

    membership_table: str = "qqq_pit_membership_daily"
    membership_security_id_column: str = "security_id"
    require_membership: bool = True

    benchmark_symbol: str = "QQQ"
    benchmark_security_id: str | None = None

    sector_table: str | None = None
    industry_table: str | None = None
    require_sector_for_sector_scan: bool = True
    minimum_sector_members_ex_focal: int = 3
    minimum_peer_members_ex_focal: int = 3

    corporate_actions_path: str | None = (
        "D:/AlgoResearch/Quant Pipeline/reference/"
        "corporate_actions_through_20260430.parquet"
    )
    require_corporate_actions: bool = True
    require_cash_dividends: bool = True
    require_delisting_outcomes: bool = True

    target_horizons_sessions: tuple[int, ...] = DEFAULT_HORIZONS
    return_windows_sessions: tuple[int, ...] = DEFAULT_RETURN_WINDOWS
    trend_windows_sessions: tuple[int, ...] = (5, 10, 20, 40)
    location_windows_sessions: tuple[int, ...] = (5, 10, 20, 60, 252)
    volatility_windows_sessions: tuple[int, ...] = (5, 10, 20, 60)
    volume_windows_sessions: tuple[int, ...] = (5, 10, 20, 60)
    session_component_windows_sessions: tuple[int, ...] = (1, 3, 5, 10, 20)

    minimum_price: float = 3.0
    minimum_prior_20d_median_dollar_volume: float = 10_000_000.0

    minimum_rank_ic_cross_section_size: int = 50
    minimum_quintile_cross_section_size: int = 50
    minimum_decile_cross_section_size: int = 80
    minimum_distinct_values_for_rank_ic: int = 2
    minimum_distinct_values_for_quintiles: int = 5
    minimum_distinct_values_for_deciles: int = 10

    minimum_valid_outcomes_per_extreme_decile: int = 6
    minimum_valid_outcomes_per_extreme_quintile: int = 8
    minimum_target_coverage_fraction_per_bin: float = 0.75
    minimum_middle_target_coverage_fraction: float = 0.75

    minimum_candidate_rank_ic_dates: int = 750
    minimum_candidate_decile_dates: int = 500
    minimum_candidate_symbols: int = 50
    minimum_rank_ic_effect: float = 0.01

    primary_fdr_threshold: float = 0.05
    pre_shortlist_q_threshold: float = 0.20
    pvalue_sidedness: str = "two_sided"

    minimum_neighbor_retention: float = 0.50
    minimum_positive_folds: int = 3
    maximum_top5_symbol_effect_share: float = 0.25
    minimum_remove_top5_effect_retention: float = 0.50

    minimum_effect_bps_h01_h02: float = 8.0
    minimum_effect_bps_h03_h05: float = 12.0
    minimum_effect_bps_h06_h10: float = 18.0
    minimum_effect_bps_h12_h14: float = 22.0
    minimum_effect_bps_h18_h20: float = 25.0

    beta_primary_window_sessions: int = 120
    beta_sensitivity_window_sessions: int = 60
    beta_minimum_observations: int = 60

    # Keep false until a real CUDA parity-tested reducer exists.
    use_cuda: bool = False
    cuda_device: str = "cuda:0"

    memory_budget_fraction: float = 0.70
    feature_block_size: int | None = None
    target_block_size: int | None = None
    cache_value_dtype: str = "float32"
    statistic_accumulation_dtype: str = "float64"
    resume: bool = True

    exact_bootstrap_samples: int = 1000
    exact_cost_scenarios_bps_per_side: tuple[float, ...] = (1.0, 3.0, 5.0)

    panel_schema_version: str = "interday_daily_v2"
    feature_schema_version: str = "interday_2a_features_v2"
    target_schema_version: str = "interday_2a_targets_v2"
    rank_schema_version: str = "interday_2a_ranks_v2"
    scan_schema_version: str = "interday_2a_scan_v2"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InterdayConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown interday config keys: {sorted(unknown)}")

        for field_name in TUPLE_FIELDS:
            if field_name in raw:
                raw[field_name] = tuple(raw[field_name])

        config = cls(**raw)
        config.validate()
        return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_root(self) -> Path:
        return Path(self.output_root) / self.experiment_id

    @property
    def sector_scan_enabled(self) -> bool:
        return bool(self.sector_table)

    @property
    def peer_scan_enabled(self) -> bool:
        return bool(self.industry_table)

    def effect_floor_bps(self, horizon: int | float | None) -> float:
        if horizon is None or pd.isna(horizon):
            return self.minimum_effect_bps_h01_h02

        horizon_int = int(horizon)
        if horizon_int <= 2:
            return self.minimum_effect_bps_h01_h02
        if horizon_int <= 5:
            return self.minimum_effect_bps_h03_h05
        if horizon_int <= 10:
            return self.minimum_effect_bps_h06_h10
        if horizon_int <= 14:
            return self.minimum_effect_bps_h12_h14
        return self.minimum_effect_bps_h18_h20

    def validate(self) -> None:
        source_start = pd.Timestamp(self.source_start)
        analysis_start = pd.Timestamp(self.analysis_start)
        discovery_end = pd.Timestamp(self.discovery_end)
        holdout_start = pd.Timestamp(self.sealed_holdout_start)

        if not source_start < analysis_start:
            raise ValueError("source_start must precede analysis_start")
        if not analysis_start <= discovery_end:
            raise ValueError("analysis_start must not exceed discovery_end")
        if discovery_end >= holdout_start:
            raise ValueError("discovery_end must precede sealed_holdout_start")
        if self.allow_holdout_access:
            raise ValueError("Interday 2A may not access the sealed holdout")

        horizons = tuple(self.target_horizons_sessions)
        if horizons != tuple(sorted(set(horizons))):
            raise ValueError("target_horizons_sessions must be sorted and unique")
        if not horizons or max(horizons) > 20:
            raise ValueError("Interday 2A requires horizons in 1..20")

        if self.require_stable_security_id and not self.security_id_column:
            raise ValueError("A stable security ID column is required")
        if self.require_corporate_actions and not self.corporate_actions_path:
            raise ValueError("Corporate-action path is required")
        if self.require_membership and not self.membership_table:
            raise ValueError("Membership table is required")

        fraction_fields = (
            "minimum_target_coverage_fraction_per_bin",
            "minimum_middle_target_coverage_fraction",
            "memory_budget_fraction",
            "minimum_neighbor_retention",
            "maximum_top5_symbol_effect_share",
            "minimum_remove_top5_effect_retention",
        )
        for name in fraction_fields:
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")

        if self.minimum_decile_cross_section_size < 80:
            raise ValueError("Decile scan requires at least 80 eligible symbols")
        if self.minimum_quintile_cross_section_size < 50:
            raise ValueError("Quintile scan requires at least 50 eligible symbols")
        if self.minimum_distinct_values_for_deciles < 10:
            raise ValueError("Deciles require at least 10 distinct values")
        if self.minimum_distinct_values_for_quintiles < 5:
            raise ValueError("Quintiles require at least 5 distinct values")

        if self.cache_value_dtype != "float32":
            raise ValueError("Cache values must be float32")
        if self.statistic_accumulation_dtype != "float64":
            raise ValueError("Statistical accumulation must be float64")
        if self.pvalue_sidedness != "two_sided":
            raise ValueError("Interday 2A discovery uses two-sided p-values")
