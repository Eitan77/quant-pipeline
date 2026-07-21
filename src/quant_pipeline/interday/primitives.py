from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class DensePanel:
    sessions: pd.DatetimeIndex; security_ids: np.ndarray; symbols: np.ndarray; key_frame: pd.DataFrame; arrays: dict[str,np.ndarray]; valid: np.ndarray

@dataclass
class PrimitiveBundle:
    close_log: np.ndarray; close_return: np.ndarray; overnight_return: np.ndarray; regular_return: np.ndarray; high: np.ndarray; low: np.ndarray; open: np.ndarray; close: np.ndarray; volume: np.ndarray; dollar_volume: np.ndarray; first_60m_return: np.ndarray; last_60m_return: np.ndarray; market_return: np.ndarray; sector_codes: np.ndarray|None; industry_codes: np.ndarray|None; session_vwap: np.ndarray|None = None; open_30m_volume: np.ndarray|None = None; close_30m_volume: np.ndarray|None = None; first_60m_volume: np.ndarray|None = None; last_60m_volume: np.ndarray|None = None; largest_5m_volume: np.ndarray|None = None; midday: np.ndarray|None = None; market_overnight_return: np.ndarray|None = None

def shift(matrix: np.ndarray, periods: int) -> np.ndarray:
    out=np.full_like(matrix,np.nan,dtype=np.float32)
    if periods==0: out[:]=matrix
    elif periods>0: out[periods:]=matrix[:-periods]
    else: out[:periods]=matrix[-periods:]
    return out

def rolling_sum(matrix, window, min_periods=None):
    min_periods=window if min_periods is None else min_periods; x=np.asarray(matrix,float); valid=np.isfinite(x); vals=np.where(valid,x,0); counts=valid.astype(int)
    cs=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(vals,axis=0)]); cn=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(counts,axis=0)])
    out=np.full(x.shape,np.nan,dtype=np.float32); out[window-1:]=np.where(cn[window:]-cn[:-window]>=min_periods,cs[window:]-cs[:-window],np.nan); return out

def rolling_mean(matrix, window, min_periods=None):
    min_periods=window if min_periods is None else min_periods; s=rolling_sum(matrix,window,1).astype(float); n=rolling_sum(np.isfinite(matrix).astype(float),window,1); return np.where(n>=min_periods,s/n,np.nan).astype(np.float32)

def rolling_std(matrix, window, min_periods=None):
    mean=rolling_mean(matrix,window,min_periods); mean_sq=rolling_mean(np.asarray(matrix,float)**2,window,min_periods); return np.sqrt(np.maximum(mean_sq-mean*mean,0)).astype(np.float32)

def rolling_max(matrix, window, min_periods=None):
    min_periods=window if min_periods is None else min_periods; x=np.asarray(matrix,float); out=np.full(x.shape,np.nan,np.float32)
    for i in range(window-1,len(x)):
        z=x[i-window+1:i+1]; valid=np.isfinite(z); out[i]=np.where(valid.any(),np.nanmax(z,axis=0),np.nan); out[i,np.sum(valid,axis=0)<min_periods]=np.nan
    return out

def rolling_min(matrix, window, min_periods=None): return -rolling_max(-np.asarray(matrix),window,min_periods)

def to_dense_panel(panel: pd.DataFrame, *, value_columns: list[str]) -> DensePanel:
    sessions=pd.DatetimeIndex(sorted(pd.to_datetime(panel.session_date).unique())); securities=np.array(sorted(panel.security_id.astype(str).unique()))
    di={v:i for i,v in enumerate(sessions)}; si={v:i for i,v in enumerate(securities)}; d=panel.session_date.map(di).to_numpy(); s=panel.security_id.astype(str).map(si).to_numpy(); shape=(len(sessions),len(securities)); arrays={}
    for c in value_columns:
        a=np.full(shape,np.nan,np.float32); a[d,s]=pd.to_numeric(panel[c],errors="coerce").to_numpy(np.float32); arrays[c]=a
    valid=np.zeros(shape,bool); valid[d,s]=panel.analysis_eligible.to_numpy(bool)
    sm=panel[["security_id","symbol"]].drop_duplicates("security_id").set_index("security_id")["symbol"]; symbols=np.array([sm.get(x,x) for x in securities]); key=panel[["security_id","symbol","session_date"]].sort_values(["session_date","security_id"],kind="stable").reset_index(drop=True)
    return DensePanel(sessions,securities,symbols,key,arrays,valid)

def build_primitives(dense: DensePanel, benchmark_symbol="QQQ", sector_codes=None, industry_codes=None) -> PrimitiveBundle:
    a=dense.arrays; close=a["close"]; op=a["open"]; close_log=np.log(close); close_return=np.expm1(close_log-shift(close_log,1)); overnight=op/shift(close,1)-1; regular=close/op-1; benchmark=np.array(dense.symbols)==benchmark_symbol; market=np.nanmean(close_return[:,benchmark],axis=1) if benchmark.any() else np.full(len(close),np.nan); market_overnight=np.nanmean(overnight[:,benchmark],axis=1) if benchmark.any() else np.full(len(close),np.nan)
    return PrimitiveBundle(close_log,close_return,overnight,regular,a["high"],a["low"],op,close,a["volume"],a["dollar_volume"],a.get("first_60m_return",np.full_like(close,np.nan)),a.get("last_60m_return",np.full_like(close,np.nan)),market,sector_codes,industry_codes,a.get("session_vwap"),a.get("open_30m_volume"),a.get("close_30m_volume"),a.get("first_60m_volume"),a.get("last_60m_volume"),a.get("largest_5m_volume"),a.get("midday"),market_overnight)
