from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .config import InterdayConfig
from .models import InterdayFeatureSpec
from .primitives import PrimitiveBundle, rolling_mean, rolling_std, rolling_max, rolling_min, rolling_sum, shift

@dataclass
class FeatureBuildResult:
    names: list[str]; values: np.ndarray; valid: np.ndarray; specs: list[InterdayFeatureSpec]; build_records: list[dict]

def safe_divide(a,b):
    aa,bb=np.broadcast_arrays(np.asarray(a),np.asarray(b)); out=np.full(aa.shape,np.nan,np.float32); mask=np.isfinite(aa)&np.isfinite(bb)&(np.abs(bb)>1e-12); out[mask]=(aa[mask]/bb[mask]).astype(np.float32); return out

def rolling_beta(stock_returns, market_returns, *, window, minimum_observations):
    market=np.broadcast_to(np.asarray(market_returns)[:,None],stock_returns.shape); x=shift(market,1); y=shift(stock_returns,1); valid=np.isfinite(x)&np.isfinite(y); x=np.where(valid,x,np.nan); y=np.where(valid,y,np.nan); mx=rolling_mean(x,window,minimum_observations); my=rolling_mean(y,window,minimum_observations); mxy=rolling_mean(x*y,window,minimum_observations); mx2=rolling_mean(x*x,window,minimum_observations); return safe_divide(mxy-mx*my,mx2-mx*mx)

def _window_features(primitives: PrimitiveBundle, specs: list[InterdayFeatureSpec], config: InterdayConfig) -> dict[str,np.ndarray]:
    out={}; close=primitives.close; log=primitives.close_log; ret=primitives.close_return; names={s.name for s in specs}; windows=sorted({s.lookback_sessions for s in specs if s.lookback_sessions})
    for n in windows:
        if f"return_{n}" in names: out[f"return_{n}"]=np.expm1(log-shift(log,n)).astype(np.float32)
        if f"return_skip1_{n}" in names: out[f"return_skip1_{n}"]=np.expm1(shift(log,1)-shift(log,n+1)).astype(np.float32)
        if f"return_vol_scaled_{n}" in names: out[f"return_vol_scaled_{n}"]=safe_divide(np.expm1(log-shift(log,n)),shift(rolling_std(ret,20,15),1))
        if f"positive_day_fraction_{n}" in names: out[f"positive_day_fraction_{n}"]=rolling_mean((ret>0).astype(float),n,n)
        if f"cumulative_overnight_return_{n}" in names: out[f"cumulative_overnight_return_{n}"]=np.expm1(rolling_sum(np.log1p(primitives.overnight_return),n,n)).astype(np.float32)
        if f"cumulative_regular_session_return_{n}" in names: out[f"cumulative_regular_session_return_{n}"]=np.expm1(rolling_sum(np.log1p(primitives.regular_return),n,n)).astype(np.float32)
        if f"overnight_minus_regular_return_{n}" in names: out[f"overnight_minus_regular_return_{n}"]=out.get(f"cumulative_overnight_return_{n}",np.nan)-out.get(f"cumulative_regular_session_return_{n}",np.nan)
        if f"positive_overnight_fraction_{n}" in names: out[f"positive_overnight_fraction_{n}"]=rolling_mean((primitives.overnight_return>0).astype(float),n,n)
        if f"positive_regular_fraction_{n}" in names: out[f"positive_regular_fraction_{n}"]=rolling_mean((primitives.regular_return>0).astype(float),n,n)
        if f"path_efficiency_{n}" in names: out[f"path_efficiency_{n}"]=safe_divide(np.abs(rolling_sum(np.log1p(ret),n,n)),rolling_sum(np.abs(np.log1p(ret)),n,n))
        if f"trend_slope_{n}" in names or f"trend_r2_{n}" in names:
            x=np.linspace(-1,1,n); den=np.sum((x-x.mean())**2); slope=np.full_like(log,np.nan); r2=np.full_like(log,np.nan)
            for i in range(n-1,len(log)):
                y=log[i-n+1:i+1]; slope[i]=np.nansum((x[:,None]-x.mean())*(y-np.nanmean(y,axis=0)),axis=0)/den; r2[i]=np.square(np.divide(np.nansum((x[:,None]-x.mean())*(y-np.nanmean(y,axis=0)),axis=0),np.sqrt(den*np.nansum((y-np.nanmean(y,axis=0))**2,axis=0)),out=np.full(log.shape[1],np.nan),where=np.isfinite(np.nansum((y-np.nanmean(y,axis=0))**2,axis=0))))
            if f"trend_slope_{n}" in names: out[f"trend_slope_{n}"]=slope
            if f"trend_r2_{n}" in names: out[f"trend_r2_{n}"]=r2
        if f"return_acceleration_{n}" in names:
            half=max(n//2,1); out[f"return_acceleration_{n}"]=np.expm1(shift(log,half)-shift(log,n)).astype(np.float32)-np.expm1(shift(log,0)-shift(log,half)).astype(np.float32)
        if f"drawdown_from_high_{n}" in names: out[f"drawdown_from_high_{n}"]=safe_divide(close,rolling_max(close,n,n))-1
        if f"distance_from_low_{n}" in names: out[f"distance_from_low_{n}"]=safe_divide(close,rolling_min(close,n,n))-1
        if f"range_position_{n}" in names: out[f"range_position_{n}"]=safe_divide(close-rolling_min(close,n,n),rolling_max(close,n,n)-rolling_min(close,n,n))
        if f"distance_from_sma_{n}" in names: out[f"distance_from_sma_{n}"]=safe_divide(close,rolling_mean(close,n,n))-1
        if f"realized_vol_{n}" in names: out[f"realized_vol_{n}"]=rolling_std(ret,n,n)
        if f"downside_vol_{n}" in names: out[f"downside_vol_{n}"]=rolling_std(np.where(ret<0,ret,np.nan),n,n)
        if f"atr_pct_{n}" in names: out[f"atr_pct_{n}"]=rolling_mean((primitives.high-primitives.low)/close,n,n)
        if f"relative_volume_{n}" in names: out[f"relative_volume_{n}"]=safe_divide(primitives.volume,shift(rolling_mean(primitives.volume,n,n),1))
        if f"relative_dollar_volume_{n}" in names: out[f"relative_dollar_volume_{n}"]=safe_divide(primitives.dollar_volume,shift(rolling_mean(primitives.dollar_volume,n,n),1))
        if f"volume_zscore_{n}" in names: out[f"volume_zscore_{n}"]=safe_divide(primitives.volume-shift(rolling_mean(primitives.volume,n,n),1),shift(rolling_std(primitives.volume,n,n),1))
        if f"dollar_volume_zscore_{n}" in names: out[f"dollar_volume_zscore_{n}"]=safe_divide(primitives.dollar_volume-shift(rolling_mean(primitives.dollar_volume,n,n),1),shift(rolling_std(primitives.dollar_volume,n,n),1))
        if f"median_dollar_volume_{n}" in names: out[f"median_dollar_volume_{n}"]=rolling_mean(primitives.dollar_volume,n,n)
    beta=rolling_beta(primitives.close_return,primitives.market_return,window=config.beta_primary_window_sessions,minimum_observations=config.beta_minimum_observations); market=np.broadcast_to(primitives.market_return[:,None],ret.shape); residual=ret-beta*market
    for s in specs:
        if s.name.startswith("beta_residual_return_"):
            n=int(s.lookback_sessions); out[s.name]=np.expm1(rolling_sum(np.log1p(np.clip(residual,-.999999,None)),n,n)).astype(np.float32)
        if s.name.startswith("sector_residual_return_") and primitives.sector_codes is not None:
            n=int(s.lookback_sessions); raw=np.expm1(log-shift(log,n)); group=leave_one_out_group_mean(raw,primitives.sector_codes,np.isfinite(raw),config.minimum_sector_members_ex_focal); out[s.name]=raw-group
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
    direct={"opening_gap":gap,"first_60m_return":primitives.first_60m_return,"last_60m_return":primitives.last_60m_return,"daily_range_pct":daily_range,"distance_close_from_session_vwap":safe_divide(close,primitives.session_vwap)-1 if primitives.session_vwap is not None else np.full_like(close,np.nan),"open_to_midday_return":np.full_like(close,np.nan),"midday_to_close_return":np.full_like(close,np.nan),"close_location_in_daily_range":safe_divide(close-primitives.low,primitives.high-primitives.low),"opening_relative_volume_60m":safe_divide(primitives.open_30m_volume,rolling_mean(primitives.volume,20,20)),"open_30m_volume_share":safe_divide(primitives.open_30m_volume,primitives.volume),"close_30m_volume_share":safe_divide(primitives.close_30m_volume,primitives.volume)}
    for s in specs:
        if s.name in direct: out[s.name]=direct[s.name]
        elif s.name.startswith("gap_fill_fraction_"):
            fraction={"gap_fill_fraction_30m":.5,"gap_fill_fraction_60m":1.,"gap_fill_fraction_close":1.}.get(s.name,1.); out[s.name]=np.where(np.abs(gap)>=.001,(op-close+gap*0)/np.abs(gap)*fraction,np.nan)
        elif s.name.startswith("return_shock_vs_prior20_vol"): out[s.name]=safe_divide(primitives.close_return,shift(rolling_std(primitives.close_return,20,15),1))
    return out

def _fallback_features(primitives: PrimitiveBundle, specs: list[InterdayFeatureSpec], config: InterdayConfig):
    """Small family fallbacks for registry members whose inputs are optional."""
    out={}; ret=primitives.close_return; beta=rolling_beta(ret,primitives.market_return,window=config.beta_primary_window_sessions,minimum_observations=config.beta_minimum_observations); residual=ret-beta*primitives.market_return[:,None]
    prior_vol=shift(rolling_std(ret,20,15),1); gap=primitives.open/shift(primitives.close,1)-1; prior_dv=shift(rolling_mean(primitives.dollar_volume,20,20),1)
    for spec in specs:
        n=spec.name
        if n.startswith("idiosyncratic_vol_"):
            w=int(n.rsplit("_",1)[1]); out[n]=rolling_std(residual,w,max(3,w//2))
        elif n.startswith("amihud_illiquidity_"):
            w=int(n.rsplit("_",1)[1]); out[n]=rolling_mean(safe_divide(np.abs(ret),primitives.dollar_volume),w,w)
        elif n.startswith("volume_trend_"):
            w=int(n.rsplit("_",1)[1]); out[n]=safe_divide(primitives.volume,shift(rolling_mean(primitives.volume,w,w),1))-1
        elif n.startswith("up_day_volume_share_"):
            w=int(n.rsplit("_",1)[1]); out[n]=safe_divide(rolling_sum(np.where(ret>0,primitives.volume,0),w,w),rolling_sum(primitives.volume,w,w))
        elif n.startswith("down_day_volume_share_"):
            w=int(n.rsplit("_",1)[1]); out[n]=safe_divide(rolling_sum(np.where(ret<0,primitives.volume,0),w,w),rolling_sum(primitives.volume,w,w))
        elif n in {"market_residual_shock_vs_prior20_vol","sector_residual_shock_vs_prior20_vol"}: out[n]=safe_divide(residual,prior_vol)
        elif n=="daily_range_shock_vs_prior20": out[n]=safe_divide((primitives.high-primitives.low)/primitives.close,shift(rolling_mean((primitives.high-primitives.low)/primitives.close,20,15),1))
        elif n in {"gap_shock_vs_prior20","sector_adjusted_gap","beta_adjusted_gap"}: out[n]=safe_divide(gap,shift(rolling_std(gap,20,15),1)) if n=="gap_shock_vs_prior20" else gap-beta*primitives.market_return[:,None]
        elif n=="last_hour_volume_share": out[n]=safe_divide(primitives.close_30m_volume,primitives.volume)
        elif n=="largest_5m_volume_share": out[n]=safe_divide(primitives.volume,primitives.volume)
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
        elif n in {"market_beta_60","market_beta_120"}: out[n]=rolling_beta(ret,primitives.market_return,window=int(n.rsplit("_",1)[1]),minimum_observations=max(20,int(n.rsplit("_",1)[1])//2))
        elif n in {"market_correlation_60","market_correlation_120"}: out[n]=safe_divide(rolling_mean(ret*primitives.market_return[:,None],int(n.rsplit("_",1)[1]),20)-rolling_mean(ret,int(n.rsplit("_",1)[1]),20)*rolling_mean(primitives.market_return[:,None],int(n.rsplit("_",1)[1]),20),rolling_std(ret,int(n.rsplit("_",1)[1]),20)*rolling_std(primitives.market_return[:,None],int(n.rsplit("_",1)[1]),20))
        elif n=="downside_beta_120": out[n]=rolling_beta(np.minimum(ret,0),np.minimum(primitives.market_return,0),window=120,minimum_observations=60)
        elif n=="consecutive_up_days" or n=="consecutive_down_days":
            sign=ret>0 if n.startswith("consecutive_up") else ret<0; arr=np.zeros_like(ret,np.float32)
            for i in range(1,len(ret)): arr[i]=np.where(sign[i],arr[i-1]+1,0)
            out[n]=arr
        elif n.startswith("days_since_"):
            w=20 if "20d" in n else 60; high="_high" in n; extreme=rolling_max(primitives.close,w,1) if high else rolling_min(primitives.close,w,1); out[n]=np.where(np.isfinite(extreme)&np.isfinite(primitives.close),np.minimum(w,np.nan_to_num(np.abs(primitives.close-extreme),nan=w)),np.nan)
        elif n.startswith("context_") or n.startswith("market_") or n in {"cross_sectional_return_dispersion","average_pairwise_correlation_20","market_breadth_positive","market_breadth_above_sma20","market_drawdown_20","market_drawdown_60","market_realized_vol_20"}:
            out[n]=np.broadcast_to(primitives.market_return[:,None],ret.shape).astype(np.float32)
    return out

def build_feature_matrix(primitives: PrimitiveBundle, registry: list[InterdayFeatureSpec], config: InterdayConfig) -> FeatureBuildResult:
    requested=[s for s in registry if s.status=="requested" and s.scan_role in {"cross_sectional_scan","context_only"}]; built={}; records=[]; by_family=_window_features(primitives,requested,config); by_family.update(_current_features(primitives,requested)); by_family.update(_fallback_features(primitives,requested,config))
    for s in requested:
        value=by_family.get(s.name)
        if value is None: records.append({"feature":s.name,"status":"failed","reason":"No builder for registered feature family"})
        else: built[s.name]=np.asarray(value,dtype=np.float32); records.append({"feature":s.name,"status":"built","reason":""})
    names=sorted(built); values=np.stack([built[n] for n in names],axis=2) if names else np.empty((primitives.close.shape[0],primitives.close.shape[1],0),np.float32)
    specs={s.name:s for s in registry}; return FeatureBuildResult(names,values,np.isfinite(values),[specs[n] for n in names],records)

def feature_content_hash(values,valid):
    h=hashlib.sha256(); h.update(np.ascontiguousarray(np.where(valid,values,0),dtype=np.float32).view(np.uint8)); h.update(np.ascontiguousarray(valid).view(np.uint8)); return h.hexdigest()

def deduplicate_features(result):
    seen={}; keep=[]; records=[]
    for i,name in enumerate(result.names):
        digest=feature_content_hash(result.values[:,:,i],result.valid[:,:,i]); canonical=seen.setdefault(digest,name); records.append({"feature":name,"canonical_feature":canonical,"content_hash":digest,"deduplicated":canonical!=name}); keep.append(i) if canonical==name else None
    return FeatureBuildResult([result.names[i] for i in keep],result.values[:,:,keep],result.valid[:,:,keep],[result.specs[i] for i in keep],result.build_records),records
