from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ScanRole = Literal["cross_sectional_scan", "context_only", "exact_candidate_only", "unavailable"]
TargetFamily = Literal["daily_terminal", "time_of_day", "diagnostic_next_gap", "exact_only"]
TestType = Literal["rank_ic", "top_minus_bottom_decile", "top_decile_minus_middle", "middle_minus_bottom_decile"]

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
    definition_version: str = "v1"

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
    is_duplicate_reference: bool = False

@dataclass(frozen=True)
class ScanPairKey:
    feature: str
    target: str
    test_type: TestType

@dataclass(frozen=True)
class MatrixLayout:
    n_dates: int
    n_symbols: int
    date_ids: tuple[int, ...]
    security_ids: tuple[str, ...]

@dataclass(frozen=True)
class BlockPlan:
    feature_block_size: int
    target_block_size: int
    estimated_peak_bytes: int
    device: str
