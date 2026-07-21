from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from .config import InterdayConfig
from .models import InterdayTargetSpec
from .corporate_actions import vectorized_total_return

@dataclass
class TargetBuildResult:
    names:list[str]; values:np.ndarray; valid:np.ndarray; aligned_market_returns:np.ndarray; specs:list[InterdayTargetSpec]; build_records:list[dict]; missing_reasons:np.ndarray|None=None; price_return_values:np.ndarray|None=None; log_values:np.ndarray|None=None

class TargetMissingReason(IntEnum):
    NONE=0; BEFORE_ANALYSIS_START=1; NOT_DECISION_ELIGIBLE=2; MISSING_ENTRY=3; MISSING_EXIT=4; CROSSES_HOLDOUT=5; UNRESOLVED_CORPORATE_ACTION=6; INSUFFICIENT_SECTOR_BASKET=7

def future_2d(values,offset):
    if values.ndim!=2: raise ValueError("future_2d expects [dates, symbols]")
    if offset<=0: raise ValueError("offset must be positive")
    out=np.full_like(values,np.nan)
    if offset>0: out[:-offset]=values[offset:]
    else: out[:]=values
    return out

def future_1d(values,offset):
    if values.ndim!=1: raise ValueError("future_1d expects [dates]")
    if offset<=0: raise ValueError("offset must be positive")
    out=np.full_like(values,np.nan)
    if offset>0: out[:-offset]=values[offset:]
    else: out[:]=values
    return out

def decision_date_sector_excess(raw,sector_codes,eligible,minimum_others=3):
    out=np.full_like(raw,np.nan,np.float32)
    for d in range(raw.shape[0]):
        for code in np.unique(sector_codes[d][eligible[d]&(sector_codes[d]>=0)]):
            m=eligible[d]&np.isfinite(raw[d])&(sector_codes[d]==code); n=int(m.sum())
            if n-1>=minimum_others: out[d,m]=(raw[d,m]-(raw[d,m].sum(dtype=np.float64)-raw[d,m])/(n-1)).astype(np.float32)
    return out

def build_targets(checkpoint_arrays,benchmark_checkpoint_arrays,sector_codes,decision_eligible,registry,config,*,sessions=None,actions=None,security_ids=None,symbols=None):
    values=[]; prices=[]; logs=[]; market=[]; specs=[]; records=[]; reasons=[]; sessions=pd.DatetimeIndex(sessions) if sessions is not None else pd.date_range(config.analysis_start,periods=decision_eligible.shape[0],freq="B")
    for spec in registry:
        if spec.return_basis=="sector" and sector_codes is None:
            records.append({"target":spec.name,"status":"skipped","reason":"No validated sector codes"}); continue
        if spec.target_family=="diagnostic_next_gap":
            entry=checkpoint_arrays["close5"]; exit_=future_2d(checkpoint_arrays["open5"],1); be=benchmark_checkpoint_arrays["close5"]; bx=future_1d(benchmark_checkpoint_arrays["open5"],1)
        else:
            entry=future_2d(checkpoint_arrays["open5"],1); exit_=future_2d(checkpoint_arrays[spec.exit_checkpoint],spec.future_day); be=future_1d(benchmark_checkpoint_arrays["open5"],1); bx=future_1d(benchmark_checkpoint_arrays[spec.exit_checkpoint],spec.future_day)
        price_raw=np.divide(exit_,entry,out=np.full_like(entry,np.nan),where=np.isfinite(entry)&np.isfinite(exit_)&(entry>0))-1
        if actions is not None:
            entry_offset = 0 if spec.target_family == "diagnostic_next_gap" else 1
            exit_offset = 1 if spec.target_family == "diagnostic_next_gap" else spec.future_day
            entry_sessions = np.roll(np.asarray(sessions), -entry_offset) if entry_offset else np.asarray(sessions)
            exit_sessions = np.roll(np.asarray(sessions), -exit_offset)
            entry_sessions[-max(entry_offset, 1):] = pd.Timestamp(config.sealed_holdout_start)
            exit_sessions[-exit_offset:] = pd.Timestamp(config.sealed_holdout_start)
            raw, price_raw = vectorized_total_return(
                entry, exit_, np.broadcast_to(entry_sessions[:, None], entry.shape),
                np.broadcast_to(exit_sessions[:, None], exit_.shape), actions,
                security_ids=security_ids, symbols=symbols)
        else:
            raw = price_raw
        mkt=np.divide(bx,be,out=np.full_like(be,np.nan),where=np.isfinite(be)&np.isfinite(bx)&(be>0))-1
        target=raw if spec.return_basis=="raw" else decision_date_sector_excess(raw,sector_codes,decision_eligible,spec.minimum_basket_members)
        reason=np.full(target.shape,TargetMissingReason.NONE,np.int8); reason[~decision_eligible]=TargetMissingReason.NOT_DECISION_ELIGIBLE
        reason[~np.isfinite(entry)]=TargetMissingReason.MISSING_ENTRY; reason[~np.isfinite(exit_)]=TargetMissingReason.MISSING_EXIT
        if actions is not None:
            reason[np.isfinite(entry)&np.isfinite(exit_)&~np.isfinite(raw)]=TargetMissingReason.UNRESOLVED_CORPORATE_ACTION
        if spec.return_basis == "sector":
            reason[np.isfinite(raw)&~np.isfinite(target)]=TargetMissingReason.INSUFFICIENT_SECTOR_BASKET
        analysis=np.asarray(sessions)>=pd.Timestamp(config.analysis_start); reason[~analysis,:]=TargetMissingReason.BEFORE_ANALYSIS_START
        exit_date=np.roll(np.asarray(sessions),-spec.future_day); exit_date[-spec.future_day:]=pd.Timestamp(config.sealed_holdout_start); crosses=exit_date>=pd.Timestamp(config.sealed_holdout_start); reason[crosses,:]=TargetMissingReason.CROSSES_HOLDOUT
        target[reason!=TargetMissingReason.NONE]=np.nan
        log=np.log1p(target); values.append(target.astype(np.float32)); prices.append(price_raw.astype(np.float32)); logs.append(log.astype(np.float32)); market.append(mkt.astype(np.float32)); reasons.append(reason); specs.append(spec); records.append({"target":spec.name,"status":"built","reason":""})
    if not values: return TargetBuildResult([],np.empty((0,*decision_eligible.shape),np.float32),np.empty((0,*decision_eligible.shape),bool),np.empty((0,len(decision_eligible)),np.float32),[],records)
    stacked=np.stack(values,axis=0); price_values=np.stack(prices,axis=0); log_values=np.stack(logs,axis=0); reason_values=np.stack(reasons,axis=0); return TargetBuildResult([s.name for s in specs],stacked,np.isfinite(stacked),np.stack(market,axis=0),specs,records,reason_values,price_values,log_values)
