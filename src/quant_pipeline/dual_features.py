"""Materialize frozen Phase 1B features from already validated base caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import ROW_KEYS, assert_cache_key_alignment, validate_cache, write_cache_metadata
from .config import ScanConfig
from .dual_registry import CompiledDualFeature, ConditionSpec, TransformSpec
from .holdout import assert_pre_holdout_frame
from .registry import FeatureSpec


def build_feature_cache_index(feature_paths: list[Path], specs_by_chunk: list[list[FeatureSpec]]) -> dict[str, Path]:
    return {spec.name: path for path, specs in zip(feature_paths, specs_by_chunk) for spec in specs}


def _rank(frame: pd.DataFrame, value: pd.Series) -> pd.Series:
    return value.groupby([frame["session_date"], frame["decision_ts"]], sort=False).rank(pct=True, method="average")


def _transform(frame: pd.DataFrame, value: pd.Series, transform: TransformSpec | None) -> pd.Series:
    transform = transform or TransformSpec()
    if transform.kind == "raw":
        out = pd.to_numeric(value, errors="coerce")
    elif transform.kind == "cross_sectional_rank":
        out = _rank(frame, pd.to_numeric(value, errors="coerce"))
    else:
        raise ValueError(f"Unsupported dual transform: {transform.kind}")
    return out * transform.orientation


def _condition(frame: pd.DataFrame, value: pd.Series, condition: ConditionSpec | None) -> pd.Series:
    if condition is None:
        raise ValueError("A dual condition is required for this operator")
    transformed = _transform(frame, value, TransformSpec(condition.transform, 1))
    ops = {"ge": transformed.ge, "gt": transformed.gt, "le": transformed.le, "lt": transformed.lt, "eq": transformed.eq}
    if condition.comparator not in ops:
        raise ValueError(f"Unsupported dual comparator: {condition.comparator}")
    result = ops[condition.comparator](condition.threshold).astype(float)
    return result.where(transformed.notna())


def _persistence(frame: pd.DataFrame, left: pd.Series, right: pd.Series, a: ConditionSpec, b: ConditionSpec) -> pd.Series:
    if a.lag_bars < 0 or b.lag_bars < 0:
        raise ValueError("persistence lag_bars must be nonnegative")
    ordered = frame[["symbol", "session_date", "decision_ts"]].copy()
    ordered["left"] = _condition(frame, left, a); ordered["right"] = _condition(frame, right, b)
    ordered["row"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["symbol", "session_date", "decision_ts"])
    grouped = ordered.groupby(["symbol", "session_date"], sort=False)
    # A five-minute source has no valid persistence link across a bar gap.
    previous = grouped["decision_ts"].shift(a.lag_bars)
    expected = pd.to_timedelta(5 * a.lag_bars, unit="m")
    left_value = grouped["left"].shift(a.lag_bars) if a.lag_bars else ordered["left"]
    if a.lag_bars:
        left_value = left_value.where(pd.to_datetime(ordered["decision_ts"], utc=True).sub(pd.to_datetime(previous, utc=True)).eq(expected))
    right_value = grouped["right"].shift(b.lag_bars) if b.lag_bars else ordered["right"]
    if b.lag_bars:
        previous_b = grouped["decision_ts"].shift(b.lag_bars)
        expected_b = pd.to_timedelta(5 * b.lag_bars, unit="m")
        right_value = right_value.where(pd.to_datetime(ordered["decision_ts"], utc=True).sub(pd.to_datetime(previous_b, utc=True)).eq(expected_b))
    output = (left_value.eq(1) & right_value.eq(1)).astype(float).where(left_value.notna() & right_value.notna())
    return output.set_axis(ordered["row"]).reindex(range(len(frame))).reset_index(drop=True)


def _materialize(frame: pd.DataFrame, compiled: CompiledDualFeature) -> pd.Series:
    item = compiled.definition
    left, right = frame[item.feature_a], frame[item.feature_b]
    if item.operator == "intersection":
        a, b = _condition(frame, left, item.condition_a), _condition(frame, right, item.condition_b)
        return (a.eq(1) & b.eq(1)).astype(float).where(a.notna() & b.notna())
    if item.operator == "gated_anchor":
        gate = _condition(frame, right, item.condition_b)
        anchor = _transform(frame, left, item.transform_a)
        return anchor.where(gate.eq(1) & gate.notna())
    if item.operator == "aligned_rank_mean":
        a = _transform(frame, left, item.transform_a); b = _transform(frame, right, item.transform_b)
        return ((a - 0.5) + (b - 0.5)) / 2
    if item.operator == "persistence_intersection":
        return _persistence(frame, left, right, item.condition_a or ConditionSpec(), item.condition_b or ConditionSpec())
    raise ValueError(f"Unsupported dual operator: {item.operator}")


def build_dual_feature_chunks(compiled: list[CompiledDualFeature], feature_path_by_name: dict[str, Path], config: ScanConfig, output_root: Path, fingerprint_sha: str) -> tuple[list[Path], list[list[FeatureSpec]], list[dict]]:
    """Create resume-safe Phase 1B cache chunks from aligned base caches."""
    output_root.mkdir(parents=True, exist_ok=True)
    chunks = [compiled[i:i + config.dual_factor_feature_chunk_size] for i in range(0, len(compiled), config.dual_factor_feature_chunk_size)]
    paths: list[Path] = []; specs_by_chunk: list[list[FeatureSpec]] = []; coverage: list[dict] = []
    memo: dict[Path, pd.DataFrame] = {}
    for index, chunk in enumerate(chunks):
        path = output_root / f"dual_{index:03d}.parquet"; specs = [item.spec for item in chunk]
        paths.append(path); specs_by_chunk.append(specs)
        if path.exists():
            validate_cache(path, fingerprint_sha, config.sealed_holdout_start)
            frame = pd.read_parquet(path)
        else:
            parents = sorted({parent for item in chunk for parent in item.spec.parent_features})
            parent_paths = {feature_path_by_name[parent] for parent in parents}
            first = next(iter(parent_paths))
            for parent_path in parent_paths:
                assert_cache_key_alignment(first, parent_path)
            base = memo.setdefault(first, pd.read_parquet(first))
            frame = base.loc[:, [col for col in base.columns if col in ROW_KEYS or col in {"session_date", "analysis_eligible", "symbol", "bar_start_ts", "decision_ts"}]].copy()
            for parent in parents:
                parent_path = feature_path_by_name[parent]
                parent_frame = memo.setdefault(parent_path, pd.read_parquet(parent_path, columns=[*ROW_KEYS, parent]))
                frame[parent] = parent_frame[parent].to_numpy()
            assert_pre_holdout_frame(frame, config.sealed_holdout_start, "dual feature materialization")
            for item in chunk:
                frame[item.spec.name] = _materialize(frame, item)
            keep = [col for col in frame.columns if col in ROW_KEYS or col in {"session_date", "analysis_eligible", "symbol", "bar_start_ts", "decision_ts"} or col in {s.name for s in specs}]
            frame = frame.loc[:, list(dict.fromkeys(keep))]
            frame.to_parquet(path, index=False)
            write_cache_metadata(path, frame, fingerprint_sha, config.sealed_holdout_start)
        for spec in specs:
            signal = pd.to_numeric(frame[spec.name], errors="coerce")
            valid = signal.notna(); activation = float(signal[valid].mean()) if spec.dtype == "binary" and valid.any() else np.nan
            coverage.append({"feature": spec.name, "valid_observations": int(valid.sum()), "activation_rate": activation,
                             "status": "built" if valid.any() else "insufficient_data"})
    return paths, specs_by_chunk, coverage
