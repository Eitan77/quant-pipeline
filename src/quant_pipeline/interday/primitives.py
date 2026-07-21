from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .corporate_actions import build_adjusted_daily_prices

@dataclass
class DensePanel:
    sessions: pd.DatetimeIndex; security_ids: np.ndarray; symbols: np.ndarray; key_frame: pd.DataFrame; arrays: dict[str,np.ndarray]; valid: np.ndarray

@dataclass
class PrimitiveBundle:
    close_log: np.ndarray; close_return: np.ndarray; overnight_return: np.ndarray; regular_return: np.ndarray; high: np.ndarray; low: np.ndarray; open: np.ndarray; close: np.ndarray; volume: np.ndarray; dollar_volume: np.ndarray; first_60m_return: np.ndarray; last_60m_return: np.ndarray; market_return: np.ndarray; sector_codes: np.ndarray|None; industry_codes: np.ndarray|None; session_vwap: np.ndarray|None = None; open_30m_volume: np.ndarray|None = None; close_30m_volume: np.ndarray|None = None; first_60m_volume: np.ndarray|None = None; last_60m_volume: np.ndarray|None = None; largest_5m_volume: np.ndarray|None = None; midday: np.ndarray|None = None; market_overnight_return: np.ndarray|None = None; open5: np.ndarray|None = None; open15: np.ndarray|None = None; close5: np.ndarray|None = None; close15: np.ndarray|None = None; checkpoint_30m: np.ndarray|None = None; checkpoint_60m: np.ndarray|None = None

def shift_2d(values: np.ndarray, periods: int) -> np.ndarray:
    output = np.full_like(values, np.nan)
    if periods == 0:
        output[:] = values
    elif periods > 0:
        output[periods:] = values[:-periods]
    else:
        output[:periods] = values[-periods:]
    return output


shift = shift_2d


def rolling_sum_2d(values: np.ndarray, window: int, minimum_observations: int | None = None) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("rolling_sum_2d expects [date, security]")
    if window <= 0:
        raise ValueError("window must be positive")
    minimum = window if minimum_observations is None else minimum_observations
    finite = np.isfinite(values)
    numeric = np.where(finite, values, 0.0).astype(np.float64)
    counts = finite.astype(np.int32)
    cumulative = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(numeric, axis=0, out=cumulative[1:])
    cumulative_count = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.int32)
    cumulative_count[0] = 0
    np.cumsum(counts, axis=0, out=cumulative_count[1:])
    sums = cumulative[window:] - cumulative[:-window]
    n = cumulative_count[window:] - cumulative_count[:-window]
    output = np.full(values.shape, np.nan, dtype=np.float32)
    output[window - 1:] = np.where(n >= minimum, sums, np.nan).astype(np.float32)
    return output


rolling_sum = rolling_sum_2d


def rolling_mean_2d(values: np.ndarray, window: int, minimum_observations: int | None = None) -> np.ndarray:
    minimum = window if minimum_observations is None else minimum_observations
    sums = rolling_sum_2d(values, window, minimum_observations=1)
    counts = rolling_sum_2d(np.isfinite(values).astype(np.float32), window, minimum_observations=1)
    output = np.full(values.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(sums) & np.isfinite(counts) & (counts >= minimum)
    output[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return output


rolling_mean = rolling_mean_2d


def rolling_std_2d(values: np.ndarray, window: int, minimum_observations: int | None = None) -> np.ndarray:
    mean = rolling_mean_2d(values, window, minimum_observations)
    mean_square = rolling_mean_2d(values * values, window, minimum_observations)
    variance = mean_square.astype(np.float64) - mean.astype(np.float64) ** 2
    tiny_negative = (variance < 0) & (variance > -1e-12)
    variance[tiny_negative] = 0.0
    variance[variance <= -1e-12] = np.nan
    return np.sqrt(variance).astype(np.float32)


rolling_std = rolling_std_2d

def rolling_max(matrix, window, min_periods=None):
    min_periods=window if min_periods is None else min_periods; x=np.asarray(matrix,float); out=np.full(x.shape,np.nan,np.float32)
    for i in range(window-1,len(x)):
        z=x[i-window+1:i+1]; valid=np.isfinite(z); out[i]=np.where(valid.any(),np.nanmax(z,axis=0),np.nan); out[i,np.sum(valid,axis=0)<min_periods]=np.nan
    return out

def rolling_min(matrix, window, min_periods=None): return -rolling_max(-np.asarray(matrix),window,min_periods)

def to_dense_panel(panel: pd.DataFrame, *, value_columns: list[str]) -> DensePanel:
    keys = panel[["security_id", "session_date"]]
    if keys.duplicated().any():
        raise ValueError("Dense panel contains duplicate security-date keys")
    sessions=pd.DatetimeIndex(sorted(pd.to_datetime(panel.session_date).unique())); securities=np.array(sorted(panel.security_id.astype(str).unique()))
    di={v:i for i,v in enumerate(sessions)}; si={v:i for i,v in enumerate(securities)}; d=panel.session_date.map(di).to_numpy(); s=panel.security_id.astype(str).map(si).to_numpy(); shape=(len(sessions),len(securities)); arrays={}
    for c in value_columns:
        a=np.full(shape,np.nan,np.float32); a[d,s]=pd.to_numeric(panel[c],errors="coerce").to_numpy(np.float32); arrays[c]=a
    valid=np.zeros(shape,bool); valid[d,s]=panel.analysis_eligible.to_numpy(bool)
    sm=panel[["security_id","symbol"]].drop_duplicates("security_id").set_index("security_id")["symbol"]; symbols=np.array([sm.get(x,x) for x in securities]); key=panel[["security_id","symbol","session_date"]].sort_values(["session_date","security_id"],kind="stable").reset_index(drop=True)
    return DensePanel(sessions,securities,symbols,key,arrays,valid)

def build_primitives(dense: DensePanel, benchmark_symbol="QQQ", sector_codes=None, industry_codes=None, action_index=None) -> PrimitiveBundle:
    a=dense.arrays; raw_close=a["close"]; raw_open=a["open"]
    if action_index is not None:
        adjusted=build_adjusted_daily_prices(raw_open,a["high"],a["low"],raw_close,a["volume"],action_index)
        op,high,low,close,volume=adjusted.split_adjusted_open,adjusted.split_adjusted_high,adjusted.split_adjusted_low,adjusted.split_adjusted_close,adjusted.split_adjusted_volume
        close_return=adjusted.daily_total_return; overnight=adjusted.overnight_total_return; regular=adjusted.regular_session_return
    else:
        op,high,low,close,volume=raw_open,a["high"],a["low"],raw_close,a["volume"]; close_log=np.log(close); close_return=np.expm1(close_log-shift(close_log,1)); overnight=op/shift(close,1)-1; regular=close/op-1
    close_log=np.log(close); benchmark=np.array(dense.symbols)==benchmark_symbol; market=np.nanmean(close_return[:,benchmark],axis=1) if benchmark.any() else np.full(len(close),np.nan); market_overnight=np.nanmean(overnight[:,benchmark],axis=1) if benchmark.any() else np.full(len(close),np.nan)
    dollar_volume=close*volume
    return PrimitiveBundle(close_log,close_return,overnight,regular,high,low,op,close,volume,dollar_volume,a.get("first_60m_return",np.full_like(close,np.nan)),a.get("last_60m_return",np.full_like(close,np.nan)),market,sector_codes,industry_codes,a.get("session_vwap"),a.get("open_30m_volume"),a.get("close_30m_volume"),a.get("first_60m_volume"),a.get("last_60m_volume"),a.get("largest_5m_volume"),a.get("midday"),market_overnight,a.get("first_5m_vwap"),a.get("first_15m_vwap"),a.get("close5"),a.get("close15"),a.get("10:00"),a.get("11:00"))
