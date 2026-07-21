from __future__ import annotations
from enum import IntEnum
import numpy as np
import pandas as pd
from .config import InterdayConfig
from .corporate_actions import CorporateActionIndex, interval_total_return
from .cache import write_matrix
from .models import InterdayTargetSpec, TargetBuildResult

class TargetMissingReason(IntEnum):
    NONE=0; BEFORE_ANALYSIS_START=1; NOT_DECISION_ELIGIBLE=2; MISSING_ENTRY=3; MISSING_EXIT=4; CROSSES_HOLDOUT=5; UNRESOLVED_CORPORATE_ACTION=6; INSUFFICIENT_SECTOR_BASKET=7; BENCHMARK_MISSING=8

def future_2d(values: np.ndarray, offset: int) -> np.ndarray:
    if values.ndim != 2: raise ValueError("future_2d expects [date, security]")
    if offset <= 0: raise ValueError("offset must be positive")
    output=np.full_like(values,np.nan); output[:-offset]=values[offset:]; return output

def future_1d(values: np.ndarray, offset: int) -> np.ndarray:
    if values.ndim != 1: raise ValueError("future_1d expects [date]")
    if offset <= 0: raise ValueError("offset must be positive")
    output=np.full_like(values,np.nan); output[:-offset]=values[offset:]; return output

def future_date_ids(length: int, offset: int) -> np.ndarray:
    output=np.full(length,-1,dtype=np.int32); output[:-offset]=np.arange(offset,length,dtype=np.int32); return output

def assign_reason(reasons: np.ndarray, mask: np.ndarray, reason: TargetMissingReason) -> None:
    reasons[mask & (reasons == TargetMissingReason.NONE)] = int(reason)

def decision_date_sector_excess(raw,sector_codes,eligible,minimum_others=3):
    out=np.full_like(raw,np.nan,np.float32)
    for d in range(raw.shape[0]):
        if sector_codes is None: continue
        for code in np.unique(sector_codes[d][eligible[d]&(sector_codes[d]>=0)]):
            mask=eligible[d]&np.isfinite(raw[d])&(sector_codes[d]==code); n=int(mask.sum())
            if n-1>=minimum_others: out[d,mask]=(raw[d,mask]-(raw[d,mask].sum(dtype=np.float64)-raw[d,mask])/(n-1)).astype(np.float32)
    return out

def _target_arrays(checkpoint_arrays, benchmark_checkpoint_arrays, spec, action_index, benchmark_action_index, sessions):
    dates=len(sessions)
    if spec.target_family=="diagnostic_next_gap":
        entry=checkpoint_arrays["close5"]; exit_=future_2d(checkpoint_arrays["open5"],1); be=benchmark_checkpoint_arrays["close5"]; bx=future_1d(benchmark_checkpoint_arrays["open5"],1); entry_ids=np.arange(dates,dtype=np.int32); exit_ids=future_date_ids(dates,1)
    else:
        entry=future_2d(checkpoint_arrays["open5"],1); exit_=future_2d(checkpoint_arrays[spec.exit_checkpoint],spec.future_day); be=future_1d(benchmark_checkpoint_arrays["open5"],1); bx=future_1d(benchmark_checkpoint_arrays[spec.exit_checkpoint],spec.future_day); entry_ids=future_date_ids(dates,1); exit_ids=future_date_ids(dates,spec.future_day)
    total,price,unresolved=interval_total_return(entry_price=entry,exit_price=exit_,entry_date_ids=entry_ids,exit_date_ids=exit_ids,action_index=action_index)
    market_total,_,market_unresolved=interval_total_return(entry_price=be[:,None],exit_price=bx[:,None],entry_date_ids=entry_ids,exit_date_ids=exit_ids,action_index=benchmark_action_index)
    return total,price,unresolved,market_total[:,0],market_unresolved[:,0],entry_ids,exit_ids,entry,exit_,be,bx

def build_targets(*, checkpoint_arrays: dict[str,np.ndarray], benchmark_checkpoint_arrays: dict[str,np.ndarray], decision_eligible: np.ndarray, sessions: pd.DatetimeIndex, action_index: CorporateActionIndex, benchmark_action_index: CorporateActionIndex, target_registry: list[InterdayTargetSpec], config: InterdayConfig, sector_codes: np.ndarray|None) -> TargetBuildResult:
    benchmark_shapes = {
        name: np.asarray(value).shape
        for name, value in benchmark_checkpoint_arrays.items()
    }
    invalid_benchmark_shapes = {
        name: shape for name, shape in benchmark_shapes.items() if len(shape) != 1
    }
    if invalid_benchmark_shapes:
        raise ValueError(
            "Benchmark checkpoint arrays must be [date]: "
            f"{invalid_benchmark_shapes}"
        )
    if benchmark_action_index.split_index.shape[1] != 1:
        raise ValueError("benchmark_action_index must contain exactly one security")

    if len({x.canonical_target_id for x in target_registry}) != len(target_registry): raise ValueError("Duplicate canonical targets")
    sessions=pd.DatetimeIndex(sessions); dates,security_count=decision_eligible.shape; values=[]; prices=[]; logs=[]; markets=[]; reasons=[]; entry_ids=[]; exit_ids=[]; specs=[]; records=[]
    analysis_start = pd.Timestamp(config.analysis_start).normalize()
    holdout_start = pd.Timestamp(config.sealed_holdout_start).normalize()
    decision_session_values = sessions.tz_localize(None).normalize()
    for spec in target_registry:
        total,price,unresolved,market,market_unresolved,e_id,x_id,entry,exit_,be,bx=_target_arrays(checkpoint_arrays,benchmark_checkpoint_arrays,spec,action_index,benchmark_action_index,sessions)
        if spec.return_basis=="sector":
            target=decision_date_sector_excess(total,sector_codes,decision_eligible,spec.minimum_basket_members)
        else: target=total.copy()
        reason=np.zeros((dates,security_count),np.int8)
        before_analysis=decision_session_values < analysis_start
        entry_crosses_holdout=np.zeros(dates,dtype=bool)
        exit_crosses_holdout=np.zeros(dates,dtype=bool)
        valid_entry=e_id>=0; valid_exit=x_id>=0
        entry_crosses_holdout[valid_entry]=decision_session_values[e_id[valid_entry]]>=holdout_start
        exit_crosses_holdout[valid_exit]=decision_session_values[x_id[valid_exit]]>=holdout_start
        crosses_holdout=entry_crosses_holdout|exit_crosses_holdout
        assign_reason(reason,np.broadcast_to(before_analysis[:,None],reason.shape),TargetMissingReason.BEFORE_ANALYSIS_START)
        assign_reason(reason,~decision_eligible,TargetMissingReason.NOT_DECISION_ELIGIBLE)
        assign_reason(reason,~np.isfinite(entry),TargetMissingReason.MISSING_ENTRY)
        assign_reason(reason,~np.isfinite(exit_),TargetMissingReason.MISSING_EXIT)
        assign_reason(reason,np.broadcast_to(crosses_holdout[:,None],reason.shape),TargetMissingReason.CROSSES_HOLDOUT)
        assign_reason(reason,unresolved,TargetMissingReason.UNRESOLVED_CORPORATE_ACTION)
        if spec.return_basis=="sector": assign_reason(reason,np.isfinite(total)&~np.isfinite(target),TargetMissingReason.INSUFFICIENT_SECTOR_BASKET)
        benchmark_missing=~np.isfinite(market)
        assign_reason(reason,np.broadcast_to(benchmark_missing[:,None],reason.shape),TargetMissingReason.BENCHMARK_MISSING)
        assign_reason(reason,np.broadcast_to(market_unresolved[:,None],reason.shape),TargetMissingReason.UNRESOLVED_CORPORATE_ACTION)
        target[reason!=TargetMissingReason.NONE]=np.nan; price[reason!=TargetMissingReason.NONE]=np.nan; log=np.log1p(target)
        values.append(target.astype(np.float32)); prices.append(price.astype(np.float32)); logs.append(log.astype(np.float32)); markets.append(market.astype(np.float32)); reasons.append(reason); entry_ids.append(e_id); exit_ids.append(x_id); specs.append(spec); records.append({"target":spec.name,"status":"built","reason":""})
    return TargetBuildResult([x.name for x in specs],np.stack(values,axis=0),np.stack(prices,axis=0),np.stack(logs,axis=0),np.isfinite(np.stack(values,axis=0)),np.stack(markets,axis=0),np.stack(reasons,axis=0),np.stack(entry_ids,axis=0),np.stack(exit_ids,axis=0),specs,records)


def write_target_artifacts(
    *,
    root,
    target_result: TargetBuildResult,
    sessions: pd.DatetimeIndex,
    security_ids: np.ndarray,
    fingerprint: str,
    config: InterdayConfig,
) -> list:
    artifacts: list = []
    security_matrix_items = (
        ("target_total_returns.npy", target_result.total_returns, "float32"),
        ("target_price_returns.npy", target_result.price_returns, "float32"),
        ("target_log_total_returns.npy", target_result.log_total_returns, "float32"),
        ("target_valid.npy", target_result.valid.astype(np.uint8), "uint8"),
        ("target_missing_reasons.npy", target_result.missing_reasons.astype(np.int8), "int8"),
    )
    for filename, values, expected_dtype in security_matrix_items:
        path = root / filename
        cast = values.astype(expected_dtype, copy=False)
        write_matrix(
            path,
            cast,
            names=target_result.names,
            dates=sessions,
            security_ids=security_ids,
            fingerprint=fingerprint,
            schema_version=config.target_schema_version,
            axis_order=("target", "date", "security"),
            source_start=config.source_start,
            analysis_start=config.analysis_start,
        )
        artifacts.extend([path, path.with_suffix(".json")])

    market_path = root / "target_market_returns.npy"
    write_matrix(
        market_path,
        target_result.aligned_market_returns.astype(np.float32, copy=False),
        names=target_result.names,
        dates=sessions,
        security_ids=np.asarray(["benchmark"]),
        fingerprint=fingerprint,
        schema_version=config.target_schema_version,
        axis_order=("target", "date"),
        source_start=config.source_start,
        analysis_start=config.analysis_start,
    )
    artifacts.extend([market_path, market_path.with_suffix(".json")])

    for filename, values in (("target_entry_date_ids.npy", target_result.entry_date_ids), ("target_exit_date_ids.npy", target_result.exit_date_ids)):
        path = root / filename
        write_matrix(
            path,
            values.astype(np.int32, copy=False),
            names=target_result.names,
            dates=sessions,
            security_ids=np.asarray(["date_id"]),
            fingerprint=fingerprint,
            schema_version=config.target_schema_version,
            axis_order=("target", "date"),
            source_start=config.source_start,
            analysis_start=config.analysis_start,
        )
        artifacts.extend([path, path.with_suffix(".json")])
    return artifacts
