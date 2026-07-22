from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from .config import InterdayConfig
from .models import InterdayFeatureSpec
from .primitives import (
    PrimitiveBundle,
    rolling_mean,
    rolling_std,
    rolling_max,
    rolling_min,
    rolling_sum,
    shift,
    shift_1d,
    shift_2d,
    rolling_sum_2d,
    rolling_mean_2d,
)

def rolling_median(matrix, window, min_periods=None):
    minimum=window if min_periods is None else min_periods; x=np.asarray(matrix,float); out=np.full(x.shape,np.nan,np.float32)
    for i in range(window-1,len(x)):
        segment=x[i-window+1:i+1]; count=np.isfinite(segment).sum(axis=0); value=np.nanmedian(segment,axis=0); out[i]=np.where(count>=minimum,value,np.nan)
    return out

@dataclass
class FeatureBuildResult:
    names: list[str]; values: np.ndarray; valid: np.ndarray; specs: list[InterdayFeatureSpec]; build_records: list[dict]

def safe_divide(a,b):
    aa,bb=np.broadcast_arrays(np.asarray(a),np.asarray(b)); out=np.full(aa.shape,np.nan,np.float32); mask=np.isfinite(aa)&np.isfinite(bb)&(np.abs(bb)>1e-12); out[mask]=(aa[mask]/bb[mask]).astype(np.float32); return out

def rolling_beta_prior_only(
    stock_log_return: np.ndarray,
    market_log_return: np.ndarray,
    *,
    window: int,
    minimum_observations: int,
    market_condition: np.ndarray | None = None,
) -> np.ndarray:
    if stock_log_return.ndim != 2:
        raise ValueError("stock_log_return must be [date, security]")
    if market_log_return.ndim != 1:
        raise ValueError("market_log_return must be [date]")

    market = np.broadcast_to(market_log_return[:, None], stock_log_return.shape)
    stock_lag = shift_2d(stock_log_return, 1)
    market_lag = shift_2d(market, 1)
    valid = np.isfinite(stock_lag) & np.isfinite(market_lag)

    if market_condition is not None:
        condition = np.broadcast_to(market_condition[:, None], stock_log_return.shape)
        condition_lag = shift_2d(condition.astype(np.float32), 1).astype(bool)
        valid &= condition_lag

    stock = np.where(valid, stock_lag, np.nan)
    benchmark = np.where(valid, market_lag, np.nan)
    mean_stock = rolling_mean_2d(stock, window, minimum_observations)
    mean_market = rolling_mean_2d(benchmark, window, minimum_observations)
    mean_product = rolling_mean_2d(stock * benchmark, window, minimum_observations)
    mean_market_square = rolling_mean_2d(benchmark * benchmark, window, minimum_observations)
    covariance = mean_product - mean_stock * mean_market
    market_variance = mean_market_square - mean_market * mean_market
    output = np.full(stock_log_return.shape, np.nan, dtype=np.float32)
    valid_variance = (
        np.isfinite(covariance)
        & np.isfinite(market_variance)
        & (market_variance > 1e-12)
    )
    output[valid_variance] = (
        covariance[valid_variance] / market_variance[valid_variance]
    ).astype(np.float32)
    return output


def rolling_beta(stock_returns, market_returns, *, window, minimum_observations):
    """Compatibility wrapper using the required log-return definition."""
    return rolling_beta_prior_only(
        np.asarray(stock_returns),
        np.asarray(market_returns),
        window=window,
        minimum_observations=minimum_observations,
    )

def rolling_days_since(values, window, *, maximum=True):
    values=np.asarray(values,float); out=np.full(values.shape,np.nan,np.float32)
    for t in range(window-1,len(values)):
        segment=values[t-window+1:t+1]
        if not np.isfinite(segment).all(): continue
        chosen=np.argmax(segment[::-1],axis=0) if maximum else np.argmin(segment[::-1],axis=0); out[t]=chosen.astype(np.float32)
    return out


def rolling_days_since_extreme(values: np.ndarray, window: int, use_maximum: bool) -> np.ndarray:
    return rolling_days_since(values, window, maximum=use_maximum)

def _window_features(primitives: PrimitiveBundle, specs: list[InterdayFeatureSpec], config: InterdayConfig) -> dict[str,np.ndarray]:
    out={}; close=primitives.close; log=primitives.daily_log_total_return; ret=primitives.daily_total_return; names={s.name for s in specs}; windows=sorted({s.lookback_sessions for s in specs if s.lookback_sessions})
    for n in windows:
        if f"return_{n}" in names: out[f"return_{n}"]=np.expm1(rolling_sum_2d(log,n,minimum_observations=n)).astype(np.float32)
        if f"return_skip1_{n}" in names: out[f"return_skip1_{n}"]=np.expm1(rolling_sum_2d(shift_2d(log,1),n,minimum_observations=n)).astype(np.float32)
        if f"return_vol_scaled_{n}" in names: out[f"return_vol_scaled_{n}"]=safe_divide(np.expm1(rolling_sum_2d(log,n,minimum_observations=n)),shift(rolling_std(ret,20,15),1))
        if f"positive_day_fraction_{n}" in names: out[f"positive_day_fraction_{n}"]=rolling_mean(np.where(np.isfinite(ret), (ret>0).astype(np.float32), np.nan),n,n)
        if f"cumulative_overnight_return_{n}" in names: out[f"cumulative_overnight_return_{n}"]=np.expm1(rolling_sum(np.log1p(primitives.overnight_return),n,n)).astype(np.float32)
        if f"cumulative_regular_session_return_{n}" in names: out[f"cumulative_regular_session_return_{n}"]=np.expm1(rolling_sum(np.log1p(primitives.regular_return),n,n)).astype(np.float32)
        if f"overnight_minus_regular_return_{n}" in names: out[f"overnight_minus_regular_return_{n}"]=out.get(f"cumulative_overnight_return_{n}",np.nan)-out.get(f"cumulative_regular_session_return_{n}",np.nan)
        if f"positive_overnight_fraction_{n}" in names: out[f"positive_overnight_fraction_{n}"]=rolling_mean(np.where(np.isfinite(primitives.overnight_return),(primitives.overnight_return>0).astype(float),np.nan),n,n)
        if f"positive_regular_fraction_{n}" in names: out[f"positive_regular_fraction_{n}"]=rolling_mean(np.where(np.isfinite(primitives.regular_return),(primitives.regular_return>0).astype(float),np.nan),n,n)
        if f"path_efficiency_{n}" in names: out[f"path_efficiency_{n}"]=safe_divide(np.abs(rolling_sum(np.log1p(ret),n,n)),rolling_sum(np.abs(np.log1p(ret)),n,n))
        if f"trend_slope_{n}" in names or f"trend_r2_{n}" in names:
            x=np.linspace(-1,1,n); den=np.sum((x-x.mean())**2); slope=np.full_like(log,np.nan); r2=np.full_like(log,np.nan)
            for i in range(n-1,len(log)):
                y=log[i-n+1:i+1]; slope[i]=np.nansum((x[:,None]-x.mean())*(y-np.nanmean(y,axis=0)),axis=0)/den; r2[i]=np.square(np.divide(np.nansum((x[:,None]-x.mean())*(y-np.nanmean(y,axis=0)),axis=0),np.sqrt(den*np.nansum((y-np.nanmean(y,axis=0))**2,axis=0)),out=np.full(log.shape[1],np.nan),where=np.isfinite(np.nansum((y-np.nanmean(y,axis=0))**2,axis=0))))
            if f"trend_slope_{n}" in names: out[f"trend_slope_{n}"]=slope
            if f"trend_r2_{n}" in names: out[f"trend_r2_{n}"]=r2
        if f"return_acceleration_{n}" in names:
            half=max(n//2,1); newest=log-shift(log,half); oldest=shift(log,half)-shift(log,2*half); out[f"return_acceleration_{n}"]=(np.expm1(newest)-np.expm1(oldest)).astype(np.float32)
        if f"drawdown_from_high_{n}" in names: out[f"drawdown_from_high_{n}"]=safe_divide(close,rolling_max(close,n,n))-1
        if f"distance_from_low_{n}" in names: out[f"distance_from_low_{n}"]=safe_divide(close,rolling_min(close,n,n))-1
        if f"range_position_{n}" in names: out[f"range_position_{n}"]=safe_divide(close-rolling_min(close,n,n),rolling_max(close,n,n)-rolling_min(close,n,n))
        if f"distance_from_sma_{n}" in names: out[f"distance_from_sma_{n}"]=safe_divide(close,rolling_mean(close,n,n))-1
        if f"realized_vol_{n}" in names: out[f"realized_vol_{n}"]=rolling_std(ret,n,n)
        if f"downside_vol_{n}" in names: out[f"downside_vol_{n}"]=np.sqrt(rolling_mean(np.minimum(ret,0.0)**2,n,max(3,int(.75*n))))
        if f"atr_pct_{n}" in names:
            prior = shift_2d(close, 1)
            true_range = np.nanmax(np.stack([primitives.high - primitives.low, np.abs(primitives.high - prior), np.abs(primitives.low - prior)], axis=0), axis=0)
            atr = rolling_mean_2d(true_range, n, minimum_observations=max(3, int(0.75 * n)))
            out[f"atr_pct_{n}"] = safe_divide(atr, close)
        if f"relative_volume_{n}" in names: out[f"relative_volume_{n}"]=safe_divide(primitives.volume,shift(rolling_mean(primitives.volume,n,n),1))
        if f"relative_dollar_volume_{n}" in names: out[f"relative_dollar_volume_{n}"]=safe_divide(primitives.dollar_volume,shift(rolling_mean(primitives.dollar_volume,n,n),1))
        if f"volume_zscore_{n}" in names: out[f"volume_zscore_{n}"]=safe_divide(primitives.volume-shift(rolling_mean(primitives.volume,n,n),1),shift(rolling_std(primitives.volume,n,n),1))
        if f"dollar_volume_zscore_{n}" in names: out[f"dollar_volume_zscore_{n}"]=safe_divide(primitives.dollar_volume-shift(rolling_mean(primitives.dollar_volume,n,n),1),shift(rolling_std(primitives.dollar_volume,n,n),1))
        if f"median_dollar_volume_{n}" in names: out[f"median_dollar_volume_{n}"]=rolling_median(primitives.dollar_volume,n,max(3,int(0.75*n)))
    stock_log_return = primitives.daily_log_total_return
    market_log_return = primitives.benchmark_daily_log_total_return
    beta=rolling_beta_prior_only(stock_log_return,market_log_return,window=config.beta_primary_window_sessions,minimum_observations=config.beta_minimum_observations)
    residual_log=stock_log_return-beta*market_log_return[:,None]
    for s in specs:
        if s.name.startswith("beta_residual_return_"):
            n=int(s.lookback_sessions); out[s.name]=np.expm1(rolling_sum(residual_log,n,n)).astype(np.float32)
        if s.name.startswith("sector_residual_return_") and primitives.sector_codes is not None:
            n=int(s.lookback_sessions)
            raw=rolling_sum_2d(primitives.daily_log_total_return,n,minimum_observations=n).astype(np.float32)
            group=leave_one_out_group_mean(raw,primitives.sector_codes,np.isfinite(raw),config.minimum_sector_members_ex_focal)
            out[s.name]=(raw-group).astype(np.float32)
    return out

def leave_one_out_group_mean(values, group_codes, valid, minimum_others):
    out=np.full_like(values,np.nan,np.float32)
    for d in range(values.shape[0]):
        for code in np.unique(group_codes[d][valid[d] & (group_codes[d]>=0)]):
            m=valid[d]&np.isfinite(values[d])&(group_codes[d]==code); n=int(m.sum())
            if n-1>=minimum_others: out[d,m]=((values[d,m].sum(dtype=np.float64)-values[d,m])/(n-1)).astype(np.float32)
    return out

def _current_features(primitives,specs):
    out={}; close,op=primitives.close,primitives.open; prev_close=shift(close,1); gap=op/prev_close-1; daily_range=(primitives.high-primitives.low)/close
    direct={"opening_gap":gap,"first_60m_return":primitives.first_60m_return,"last_60m_return":primitives.last_60m_return,"daily_range_pct":daily_range,"distance_close_from_session_vwap":safe_divide(close,primitives.split_adjusted_session_vwap)-1,"open_to_midday_return":safe_divide(primitives.split_adjusted_1200,primitives.split_adjusted_open5)-1,"midday_to_close_return":safe_divide(primitives.split_adjusted_close5,primitives.split_adjusted_1200)-1,"close_location_in_daily_range":safe_divide(close-primitives.low,primitives.high-primitives.low),"opening_relative_volume_60m":safe_divide(primitives.first_60m_volume,shift(rolling_mean(primitives.first_60m_volume,20,15),1)),"open_30m_volume_share":safe_divide(primitives.open_30m_volume,primitives.volume),"close_30m_volume_share":safe_divide(primitives.close_30m_volume,primitives.volume),"last_hour_volume_share":safe_divide(primitives.last_60m_volume,primitives.volume),"largest_5m_volume_share":safe_divide(primitives.largest_5m_volume,primitives.volume)}
    for s in specs:
        if s.name in direct: out[s.name]=direct[s.name]
        elif s.name.startswith("gap_fill_fraction_"):
            checkpoint = {"gap_fill_fraction_30m": primitives.checkpoint_30m, "gap_fill_fraction_60m": primitives.checkpoint_60m, "gap_fill_fraction_close": primitives.close5}.get(s.name)
            if checkpoint is None:
                continue
            out[s.name] = gap_fill_fraction(shift(close, 1), primitives.open5 if primitives.open5 is not None else op, checkpoint)
        elif s.name.startswith("return_shock_vs_prior20_vol"): out[s.name]=safe_divide(primitives.close_return,shift(rolling_std(primitives.close_return,20,15),1))
    return out

def gap_fill_fraction(prior_close: np.ndarray, open5: np.ndarray, checkpoint: np.ndarray) -> np.ndarray:
    gap = open5 / prior_close - 1.0
    movement = checkpoint / open5 - 1.0
    output = np.full_like(gap, np.nan, dtype=np.float32)
    valid = np.isfinite(gap) & np.isfinite(movement) & (np.abs(gap) >= 0.001)
    output[valid] = (np.sign(-gap[valid]) * movement[valid] / np.abs(gap[valid])).astype(np.float32)
    return output


def consecutive_direction_count(returns: np.ndarray, *, positive: bool) -> np.ndarray:
    dates, securities = returns.shape
    output = np.full((dates, securities), np.nan, dtype=np.float32)
    for security in range(securities):
        count = 0
        for date in range(dates):
            value = returns[date, security]
            if not np.isfinite(value):
                count = 0
                output[date, security] = np.nan
                continue
            condition = value > 0 if positive else value < 0
            count = count + 1 if condition else 0
            output[date, security] = count
    return output

def _fallback_features(primitives: PrimitiveBundle, specs: list[InterdayFeatureSpec], config: InterdayConfig):
    """Small family fallbacks for registry members whose inputs are optional."""
    out={}; ret=primitives.daily_total_return
    stock_log_return = primitives.daily_log_total_return
    market_log_return = primitives.benchmark_daily_log_total_return
    beta=rolling_beta_prior_only(stock_log_return,market_log_return,window=config.beta_primary_window_sessions,minimum_observations=config.beta_minimum_observations)
    residual=stock_log_return-beta*market_log_return[:,None]
    prior_vol=shift(rolling_std(ret,20,15),1); gap=primitives.open/shift(primitives.close,1)-1; prior_dv=shift(rolling_mean(primitives.dollar_volume,20,20),1)
    for spec in specs:
        n=spec.name
        if n.startswith("idiosyncratic_vol_"):
            w=int(n.rsplit("_",1)[1]); out[n]=rolling_std(residual,w,max(3,w//2))
        elif n.startswith("amihud_illiquidity_"):
            w=int(n.rsplit("_",1)[1]); out[n]=rolling_mean(safe_divide(np.abs(ret),primitives.dollar_volume),w,w)
        elif n.startswith("volume_trend_"):
            w=int(n.rsplit("_",1)[1]); out[n]=safe_divide(primitives.volume,shift(rolling_mean(primitives.volume,w,w),1))-1
        elif n.startswith("up_day_volume_share_") or n.startswith("down_day_volume_share_"):
            w=int(n.rsplit("_",1)[1])
            valid_return_volume=np.isfinite(primitives.daily_total_return)&np.isfinite(primitives.volume)
            valid_volume=np.where(valid_return_volume,primitives.volume,np.nan)
            direction = ret > 0 if n.startswith("up_") else ret < 0
            numerator=np.where(valid_return_volume&direction,primitives.volume,0.0)
            out[n]=safe_divide(rolling_sum(numerator,w,w),rolling_sum(valid_volume,w,w))
        elif n=="market_residual_shock_vs_prior20_vol": out[n]=safe_divide(residual,prior_vol)
        elif n=="sector_residual_shock_vs_prior20_vol" and primitives.sector_codes is not None:
            sector_daily_mean=leave_one_out_group_mean(primitives.daily_log_total_return,primitives.sector_codes,np.isfinite(primitives.daily_log_total_return),config.minimum_sector_members_ex_focal)
            sector_daily_residual=primitives.daily_log_total_return-sector_daily_mean
            out[n]=safe_divide(sector_daily_residual,shift_2d(rolling_std_2d(sector_daily_residual,20,minimum_observations=15),1))
        elif n=="daily_range_shock_vs_prior20": out[n]=safe_divide((primitives.high-primitives.low)/primitives.close,shift(rolling_mean((primitives.high-primitives.low)/primitives.close,20,15),1))
        elif n=="gap_shock_vs_prior20": out[n]=safe_divide(gap,shift(rolling_std(gap,20,15),1))
        elif n=="beta_adjusted_gap":
            market_gap = primitives.market_overnight_return if primitives.market_overnight_return is not None else np.full(gap.shape[0], np.nan)
            out[n]=np.log1p(gap)-beta*np.log1p(np.asarray(market_gap)[:,None])
        elif n=="sector_adjusted_gap" and primitives.sector_codes is not None:
            stock_gap_log=np.where(np.isfinite(primitives.overnight_total_return)&(primitives.overnight_total_return>-1.0),np.log1p(primitives.overnight_total_return),np.nan)
            group=leave_one_out_group_mean(stock_gap_log,primitives.sector_codes,np.isfinite(stock_gap_log),config.minimum_sector_members_ex_focal); out[n]=(stock_gap_log-group).astype(np.float32)
        elif n=="last_hour_volume_share": out[n]=safe_divide(primitives.last_60m_volume,primitives.volume) if primitives.last_60m_volume is not None else np.full(ret.shape,np.nan,np.float32)
        elif n=="largest_5m_volume_share": out[n]=safe_divide(primitives.largest_5m_volume,primitives.volume) if primitives.largest_5m_volume is not None else np.full(ret.shape,np.nan,np.float32)
        elif n.startswith("cumulative_first_60m_return_"):
            w=int(n.rsplit("_",1)[1]); out[n]=np.expm1(rolling_sum(np.log1p(primitives.first_60m_return),w,w)).astype(np.float32)
        elif n.startswith("cumulative_last_60m_return_"):
            w=int(n.rsplit("_",1)[1]); out[n]=np.expm1(rolling_sum(np.log1p(primitives.last_60m_return),w,w)).astype(np.float32)
        elif n.startswith("first_60m_minus_last_60m_"):
            w=int(n.rsplit("_",1)[1]); out[n]=out.get(f"cumulative_first_60m_return_{w}",rolling_sum(primitives.first_60m_return,w,w))-out.get(f"cumulative_last_60m_return_{w}",rolling_sum(primitives.last_60m_return,w,w))
        elif n.startswith("average_close_location_"):
            w=int(n.rsplit("_",1)[1]); out[n]=rolling_mean(safe_divide(primitives.close-primitives.low,primitives.high-primitives.low),w,w)
        elif n.startswith("average_open_30m_volume_share_"):
            w=int(n.rsplit("_",1)[1]); out[n]=rolling_mean(safe_divide(primitives.open_30m_volume,primitives.volume),w,w)
        elif n.startswith("average_close_30m_volume_share_"):
            w=int(n.rsplit("_",1)[1]); out[n]=rolling_mean(safe_divide(primitives.close_30m_volume,primitives.volume),w,w)
        elif n in {"market_beta_60","market_beta_120"}: out[n]=rolling_beta_prior_only(stock_log_return,market_log_return,window=int(n.rsplit("_",1)[1]),minimum_observations=max(20,int(n.rsplit("_",1)[1])//2))
        elif n in {"market_correlation_60","market_correlation_120"}: out[n]=safe_divide(rolling_mean(ret*primitives.market_return[:,None],int(n.rsplit("_",1)[1]),20)-rolling_mean(ret,int(n.rsplit("_",1)[1]),20)*rolling_mean(primitives.market_return[:,None],int(n.rsplit("_",1)[1]),20),rolling_std(ret,int(n.rsplit("_",1)[1]),20)*rolling_std(primitives.market_return[:,None],int(n.rsplit("_",1)[1]),20))
        elif n=="downside_beta_120": out[n]=rolling_beta_prior_only(stock_log_return,market_log_return,window=120,minimum_observations=60,market_condition=np.isfinite(market_log_return)&(market_log_return<0))
        elif n=="consecutive_up_days" or n=="consecutive_down_days":
            out[n]=consecutive_direction_count(ret,positive=n.startswith("consecutive_up"))
        elif n.startswith("days_since_"):
            w=20 if "20d" in n else 60; high="_high" in n; out[n]=rolling_days_since(primitives.close,w,maximum=high)
        elif n.startswith("context_") or n.startswith("market_") or n in {"cross_sectional_return_dispersion","average_pairwise_correlation_20","market_breadth_positive","market_breadth_above_sma20","market_drawdown_20","market_drawdown_60","market_realized_vol_20"}:
            out[n]=np.broadcast_to(primitives.market_return[:,None],ret.shape).astype(np.float32)
    return out

def build_feature_matrix(primitives: PrimitiveBundle, registry: list[InterdayFeatureSpec], config: InterdayConfig) -> FeatureBuildResult:
    requested=[s for s in registry if s.status=="requested" and s.scan_role=="cross_sectional_scan"]; built={}; records=[]; by_family=_window_features(primitives,requested,config); by_family.update(_current_features(primitives,requested)); by_family.update(_fallback_features(primitives,requested,config))
    for s in requested:
        value=by_family.get(s.name)
        if value is None: records.append({"feature":s.name,"status":"unavailable","reason":"No builder for registered feature family"})
        else: built[s.name]=np.asarray(value,dtype=np.float32); records.append({"feature":s.name,"status":"built","reason":""})
    names=sorted(built); values=np.stack([built[n] for n in names],axis=0) if names else np.empty((0,primitives.close.shape[0],primitives.close.shape[1]),np.float32)
    missing=[r for r in records if r["status"] not in {"built","unavailable"}]
    if missing: raise ValueError(f"Requested feature build failures: {missing}")
    requested={s.name for s in registry if s.status=="requested" and s.scan_role=="cross_sectional_scan"}
    built=set(names)
    if requested != built:
        raise RuntimeError(f"Feature build mismatch. Missing={sorted(requested-built)}; unexpected={sorted(built-requested)}")
    for index, name in enumerate(names):
        value = values[index]
        if not any(np.isfinite(value[d]).sum() >= 2 and len(np.unique(value[d][np.isfinite(value[d])])) >= 2 for d in range(value.shape[0])):
            raise RuntimeError(f"Feature {name} has no analysis date with two distinct finite values")
    specs={s.name:s for s in registry}; return FeatureBuildResult(names,values,np.isfinite(values),[specs[n] for n in names],records)

def build_context_matrix(primitives: PrimitiveBundle, registry: list[InterdayFeatureSpec], config: InterdayConfig) -> pd.DataFrame:
    context=[s for s in registry if s.status=="requested" and s.scan_role=="context_only"]; values=_fallback_features(primitives,context,config); frame={"date_id":np.arange(primitives.close.shape[0],dtype=np.int32)}
    for spec in context:
        frame[spec.name]=np.nanmean(values.get(spec.name,np.full_like(primitives.close,np.nan)),axis=1)
    return pd.DataFrame(frame)

def feature_content_hash(values,valid):
    h=hashlib.sha256(); h.update(np.ascontiguousarray(np.where(valid,values,0),dtype=np.float32).view(np.uint8)); h.update(np.ascontiguousarray(valid).view(np.uint8)); return h.hexdigest()

def deduplicate_features(result):
    seen={}; keep=[]; records=[]
    for i,name in enumerate(result.names):
        digest=feature_content_hash(result.values[i],result.valid[i]); canonical=seen.setdefault(digest,name); records.append({"feature":name,"canonical_feature":canonical,"content_hash":digest,"deduplicated":canonical!=name}); keep.append(i) if canonical==name else None
    return FeatureBuildResult([result.names[i] for i in keep],result.values[keep],result.valid[keep],[result.specs[i] for i in keep],result.build_records),records
