from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 20)
DEFAULT_RETURN_WINDOWS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 40, 60)

@dataclass(frozen=True)
class InterdayConfig:
    experiment_id: str = "interday_2a_single_variable_through_20260430"
    catalog_path: str = "D:/AlgoResearch/data/catalog.duckdb"
    source_table: str = "derived_bars_5m"
    feed: str = "sip"
    adjustment: str = "raw"
    exchange_calendar: str = "XNYS"
    start: str = "2019-06-21"
    source_warmup_start: str | None = "2018-06-21"
    discovery_end: str = "2026-04-30"
    sealed_holdout_start: str = "2026-05-01"
    allow_holdout_access: bool = False
    output_root: str = "D:/AlgoResearch/Quant Pipeline/runs"
    membership_table: str = "qqq_pit_membership_daily"
    require_membership: bool = True
    benchmark_symbol: str = "QQQ"
    sector_table: str | None = None
    industry_table: str | None = None
    require_sector_for_sector_scan: bool = True
    minimum_sector_members_ex_focal: int = 3
    minimum_peer_members_ex_focal: int = 3
    corporate_actions_path: str | None = None
    require_corporate_actions: bool = False
    require_cash_dividends: bool = False
    require_delisting_outcomes: bool = False
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
    minimum_valid_outcomes_per_extreme_decile: int = 6
    minimum_target_coverage_fraction_per_bin: float = 0.75
    minimum_middle_target_coverage_fraction: float = 0.75
    minimum_valid_outcomes_per_extreme_quintile: int = 8
    minimum_candidate_rank_ic_dates: int = 750
    minimum_candidate_decile_dates: int = 500
    minimum_candidate_symbols: int = 50
    primary_fdr_threshold: float = 0.05
    pvalue_sidedness: str = "two_sided"
    pre_shortlist_q_threshold: float = 0.20
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
    use_cuda: bool = True
    cuda_device: str = "cuda:0"
    memory_budget_fraction: float = 0.70
    feature_block_size: int | None = None
    target_block_size: int | None = None
    statistic_accumulation_dtype: str = "float64"
    cache_value_dtype: str = "float32"
    resume: bool = True
    exact_bootstrap_samples: int = 1000
    exact_cost_scenarios_bps_per_side: tuple[float, ...] = (1.0, 3.0, 5.0)
    panel_schema_version: str = "interday_daily_v1"
    feature_schema_version: str = "interday_2a_features_v1"
    target_schema_version: str = "interday_2a_targets_v1"
    rank_schema_version: str = "interday_2a_ranks_v1"
    scan_schema_version: str = "interday_2a_scan_v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InterdayConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown interday config keys: {sorted(unknown)}")
        tuple_fields = {n for n, f in cls.__dataclass_fields__.items() if f.type.startswith("tuple") or n.endswith("_sessions") or n == "exact_cost_scenarios_bps_per_side"}
        for name in tuple_fields:
            if name in raw and isinstance(raw[name], list):
                raw[name] = tuple(raw[name])
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

    def effect_floor_bps(self, horizon: int) -> float:
        if horizon <= 2: return self.minimum_effect_bps_h01_h02
        if horizon <= 5: return self.minimum_effect_bps_h03_h05
        if horizon <= 10: return self.minimum_effect_bps_h06_h10
        if horizon <= 14: return self.minimum_effect_bps_h12_h14
        return self.minimum_effect_bps_h18_h20

    def validate(self) -> None:
        if pd.Timestamp(self.discovery_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("discovery_end must precede sealed_holdout_start")
        if self.allow_holdout_access:
            raise ValueError("Interday 2A may not allow holdout access")
        horizons = tuple(self.target_horizons_sessions)
        if not horizons or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("target_horizons_sessions must be unique and sorted")
        if max(horizons) > 20 or min(horizons) < 1:
            raise ValueError("Interday 2A horizons must be in [1, 20]")
        if self.minimum_decile_cross_section_size < 10:
            raise ValueError("minimum_decile_cross_section_size is too small")
        if not 0 < self.minimum_target_coverage_fraction_per_bin <= 1:
            raise ValueError("minimum_target_coverage_fraction_per_bin must be in (0, 1]")
        if not 0 < self.memory_budget_fraction < 1:
            raise ValueError("memory_budget_fraction must be in (0, 1)")
        if self.pvalue_sidedness != "two_sided":
            raise ValueError("Interday 2A uses two-sided discovery p-values")
        if self.source_warmup_start and pd.Timestamp(self.source_warmup_start) > pd.Timestamp(self.start):
            raise ValueError("source_warmup_start must not follow start")
