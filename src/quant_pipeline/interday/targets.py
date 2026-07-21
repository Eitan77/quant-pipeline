from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import InterdayConfig
from .models import InterdayTargetSpec

@dataclass
class TargetBuildResult:
    names:list[str]; values:np.ndarray; valid:np.ndarray; aligned_market_returns:np.ndarray; specs:list[InterdayTargetSpec]; build_records:list[dict]; missing_reasons:np.ndarray|None=None

def future_2d(values,offset):
    if values.ndim!=2: raise ValueError("future_2d expects [dates, symbols]")
    out=np.full_like(values,np.nan)
    if offset>0: out[:-offset]=values[offset:]
    else: out[:]=values
    return out

def future_1d(values,offset):
    if values.ndim!=1: raise ValueError("future_1d expects [dates]")
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

def build_targets(checkpoint_arrays,benchmark_checkpoint_arrays,sector_codes,decision_eligible,registry,config):
    values=[]; market=[]; specs=[]; records=[]
    for spec in registry:
        if spec.return_basis=="sector" and sector_codes is None:
            records.append({"target":spec.name,"status":"skipped","reason":"No validated sector codes"}); continue
        if spec.target_family=="diagnostic_next_gap":
            entry=checkpoint_arrays["close5"]; exit_=future_2d(checkpoint_arrays["open5"],1); be=benchmark_checkpoint_arrays["close5"]; bx=future_1d(benchmark_checkpoint_arrays["open5"],1)
        else:
            entry=future_2d(checkpoint_arrays["open5"],1); exit_=future_2d(checkpoint_arrays[spec.exit_checkpoint],spec.future_day); be=future_1d(benchmark_checkpoint_arrays["open5"],1); bx=future_1d(benchmark_checkpoint_arrays[spec.exit_checkpoint],spec.future_day)
        raw=np.divide(exit_,entry,out=np.full_like(entry,np.nan),where=np.isfinite(entry)&np.isfinite(exit_)&(entry>0))-1; mkt=np.divide(bx,be,out=np.full_like(be,np.nan),where=np.isfinite(be)&np.isfinite(bx)&(be>0))-1
        target=raw if spec.return_basis=="raw" else decision_date_sector_excess(raw,sector_codes,decision_eligible,spec.minimum_basket_members)
        target[~decision_eligible]=np.nan; values.append(target.astype(np.float32)); market.append(np.broadcast_to(mkt[:,None],target.shape)[:,0].astype(np.float32)); specs.append(spec); records.append({"target":spec.name,"status":"built","reason":""})
    if not values: return TargetBuildResult([],np.empty((0,*decision_eligible.shape),np.float32),np.empty((0,*decision_eligible.shape),bool),np.empty((0,len(decision_eligible)),np.float32),[],records)
    return TargetBuildResult([s.name for s in specs],np.stack(values,axis=2),np.isfinite(np.stack(values,axis=2)),np.stack(market,axis=1),specs,records)
