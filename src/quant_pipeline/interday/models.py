
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


ScanRole = Literal[
    "cross_sectional_scan",
    "context_only",
    "exact_candidate_only",
    "unavailable",
]

TargetFamily = Literal[
    "daily_terminal",
    "time_of_day",
    "diagnostic_next_gap",
    "exact_only",
]

TestType = Literal[
    "rank_ic",
    "top_minus_bottom_decile",
    "top_decile_minus_middle",
    "middle_minus_bottom_decile",
]


@dataclass(frozen=True)
class InterdayFeatureSpec:
    name: str
    description: str
    family: str
    lookback_sessions: int | None
    minimum_history_sessions: int
    price_basis: str
    availability_cutoff: str
    required_columns: tuple[str, ...]
    missing_value_rule: str
    expected_direction: int
    sector_data_required: bool
    peer_data_required: bool
    redundancy_group: str
    scan_role: ScanRole
    status: str = "requested"
    unavailable_reason: str | None = None
    definition_version: str = "v2"


@dataclass(frozen=True)
class InterdayTargetSpec:
    name: str
    canonical_target_id: str
    description: str
    target_family: TargetFamily
    fdr_family: str
    horizon_sessions: int | None
    future_day: int
    checkpoint: str
    entry_checkpoint: str
    exit_checkpoint: str
    return_basis: str
    return_format: str
    benchmark_definition: str | None
    is_executable: bool
    diagnostic_only: bool
    overlap_sessions: int
    minimum_basket_members: int
    endpoint_order: int = 0
    is_duplicate_reference: bool = False
    definition_version: str = "v2"


@dataclass(frozen=True)
class BlockPlan:
    feature_block_size: int
    target_block_size: int
    estimated_peak_bytes: int
    device: str


@dataclass
class FeatureBuildResult:
    names: list[str]
    values: np.ndarray                 # [feature, date, security]
    valid: np.ndarray                  # same shape
    specs: list[InterdayFeatureSpec]
    build_records: list[dict]


@dataclass
class TargetBuildResult:
    names: list[str]
    total_returns: np.ndarray          # [target, date, security]
    price_returns: np.ndarray          # [target, date, security]
    log_total_returns: np.ndarray      # [target, date, security]
    valid: np.ndarray                  # [target, date, security]
    aligned_market_returns: np.ndarray # [target, date]
    missing_reasons: np.ndarray        # int8 [target, date, security]
    entry_date_ids: np.ndarray         # int32 [target, date]
    exit_date_ids: np.ndarray          # int32 [target, date]
    specs: list[InterdayTargetSpec]
    build_records: list[dict]


@dataclass
class RankBinCache:
    feature_names: list[str]
    percentile_ranks: np.ndarray       # [feature, date, security]
    deciles: np.ndarray                # int8, -1 invalid
    quintiles: np.ndarray              # int8, -1 invalid
    valid_counts: np.ndarray           # [feature, date]
    distinct_counts: np.ndarray        # [feature, date]
    tie_fraction: np.ndarray           # [feature, date]
    persistence: pd.DataFrame


@dataclass
class DailyPairSeries:
    rank_ic: np.ndarray
    top_minus_bottom: np.ndarray
    top_minus_middle: np.ndarray
    middle_minus_bottom: np.ndarray
    quintile_spread: np.ndarray

    ic_cross_section_size: np.ndarray
    top_coverage: np.ndarray
    bottom_coverage: np.ndarray
    middle_coverage: np.ndarray
    quintile_top_coverage: np.ndarray
    quintile_bottom_coverage: np.ndarray
