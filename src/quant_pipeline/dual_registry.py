"""Frozen, target-independent Phase 1B dual-feature plans."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ScanConfig
from .registry import FeatureSpec


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class TransformSpec:
    kind: str = "raw"
    orientation: int = 1


@dataclass(frozen=True)
class ConditionSpec:
    transform: str = "raw"
    comparator: str = "ge"
    threshold: float = 0.0
    lag_bars: int = 0


@dataclass(frozen=True)
class DualFeatureDefinition:
    identifier: str
    feature_a: str
    feature_b: str
    operator: str
    condition_a: ConditionSpec | None = None
    condition_b: ConditionSpec | None = None
    transform_a: TransformSpec | None = None
    transform_b: TransformSpec | None = None
    output_dtype: str = "binary"
    expected_direction: int = 0


@dataclass(frozen=True)
class CompiledDualFeature:
    definition: DualFeatureDefinition
    spec: FeatureSpec
    condition_specs: tuple[FeatureSpec, ...] = ()


def _condition(raw: dict[str, Any] | None) -> ConditionSpec | None:
    return None if raw is None else ConditionSpec(**{k: raw[k] for k in ("transform", "comparator", "threshold", "lag_bars") if k in raw})


def _transform(raw: dict[str, Any] | None) -> TransformSpec | None:
    if raw is None:
        return None
    return TransformSpec(kind=raw.get("kind", raw.get("transform", "raw")), orientation=int(raw.get("orientation", 1)))


def _feature_name(definition: DualFeatureDefinition) -> tuple[str, str]:
    payload = {"id": definition.identifier, "a": definition.feature_a, "b": definition.feature_b,
               "operator": definition.operator, "condition_a": definition.condition_a,
               "condition_b": definition.condition_b, "transform_a": definition.transform_a,
               "transform_b": definition.transform_b, "dtype": definition.output_dtype,
               "direction": definition.expected_direction}
    lineage = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    stem = f"dual__{definition.feature_a}__{definition.feature_b}__{definition.identifier}"
    return f"{stem}__{lineage[:8]}", lineage


def _spec(definition: DualFeatureDefinition) -> FeatureSpec:
    name, lineage = _feature_name(definition)
    params = _canonical({"condition_a": definition.condition_a, "condition_b": definition.condition_b,
                         "transform_a": definition.transform_a, "transform_b": definition.transform_b})
    return FeatureSpec(
        name=name, description=f"Frozen dual factor: {definition.identifier}", family="dual_factor",
        classification="time_series", dtype=definition.output_dtype, discovery_phase="1B", arity=2,
        parent_features=(definition.feature_a, definition.feature_b), operator=definition.operator,
        operator_parameters_json=params, lineage_hash=lineage, redundancy_group=f"dual::{definition.identifier}",
        complexity_units=2, expected_direction=definition.expected_direction,
    )


def compile_dual_plan(path: str | Path, base_features: list[FeatureSpec], config: ScanConfig) -> list[CompiledDualFeature]:
    """Parse a strict, deterministic and target-independent manifest."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != "phase1b_manifest_v1":
        raise ValueError("Unsupported Phase 1B manifest schema_version")
    base_names = {spec.name for spec in base_features if spec.status == "requested"}
    defaults = raw.get("defaults", {})
    compiled: list[CompiledDualFeature] = []
    for item in raw.get("definitions", []):
        unknown = set(item) - {"id", "feature_a", "feature_b", "operator", "condition_a", "condition_b", "transform_a", "transform_b", "output_dtype", "expected_direction"}
        if unknown:
            raise ValueError(f"Unknown dual definition keys: {sorted(unknown)}")
        definition = DualFeatureDefinition(
            identifier=item["id"], feature_a=item["feature_a"], feature_b=item["feature_b"], operator=item["operator"],
            condition_a=_condition(item.get("condition_a")), condition_b=_condition(item.get("condition_b")),
            transform_a=_transform(item.get("transform_a")), transform_b=_transform(item.get("transform_b")),
            output_dtype=item.get("output_dtype", "binary"),
            expected_direction=int(item.get("expected_direction", defaults.get("expected_direction", 0))),
        )
        if definition.operator not in config.dual_factor_allowed_operators:
            raise ValueError(f"Unsupported dual operator: {definition.operator}")
        if definition.feature_a not in base_names or definition.feature_b not in base_names:
            raise ValueError(f"Dual definition {definition.identifier} references unavailable base parent")
        if definition.output_dtype not in {"binary", "continuous"}:
            raise ValueError("Dual output_dtype must be binary or continuous")
        if definition.operator in {"intersection", "persistence_intersection"} and definition.output_dtype != "binary":
            raise ValueError(f"{definition.operator} must emit binary output")
        if definition.operator in {"gated_anchor", "aligned_rank_mean"} and definition.output_dtype != "continuous":
            raise ValueError(f"{definition.operator} must emit continuous output")
        compiled.append(CompiledDualFeature(definition, _spec(definition)))
    if len(compiled) > config.dual_factor_max_generated_features:
        raise ValueError("Compiled dual plan exceeds dual_factor_max_generated_features")
    names = [item.spec.name for item in compiled]
    if len(names) != len(set(names)):
        raise ValueError("Dual plan contains duplicate generated feature names")
    return compiled


def plan_hash(compiled: list[CompiledDualFeature]) -> str:
    return hashlib.sha256(_canonical([item.spec for item in compiled]).encode()).hexdigest()

