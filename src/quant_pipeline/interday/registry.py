from __future__ import annotations
from dataclasses import asdict
import hashlib,json
from .config import InterdayConfig
from .models import InterdayFeatureSpec,InterdayTargetSpec

def _f(name,family,lookback=None,minimum=None,*,sector=False,peer=False,role="cross_sectional_scan",status="requested",reason=None):
    return InterdayFeatureSpec(name,f"Interday {family} feature {name}",family,lookback,minimum if minimum is not None else (lookback or 0),"split_adjusted","completed_session_available_at",(),"pairwise_exclude",0,sector,peer,f"{family}:{name.rsplit('_',1)[0]}",role,status,reason)

def build_feature_registry(config: InterdayConfig) -> list[InterdayFeatureSpec]:
    out=[]
    def add(names,family,windows,**kw):
        for n in names:
            for w in windows: out.append(_f(f"{n}_{w}",family,w,**kw))
    add(["return","sector_residual_return","beta_residual_return"],"return",config.return_windows_sessions,sector=False)
    add(["return_skip1","return_vol_scaled","positive_day_fraction"],"return",(5,10,20,40,60))
    add(["cumulative_overnight_return","cumulative_regular_session_return","overnight_minus_regular_return","positive_overnight_fraction","positive_regular_fraction"],"session_component",config.session_component_windows_sessions)
    add(["path_efficiency","trend_slope","trend_r2","return_acceleration"],"trend",config.trend_windows_sessions)
    add(["drawdown_from_high","distance_from_low","range_position","distance_from_sma"],"location",config.location_windows_sessions)
    add(["realized_vol","downside_vol","atr_pct","idiosyncratic_vol"],"volatility",config.volatility_windows_sessions)
    add(["relative_volume","relative_dollar_volume","volume_zscore","dollar_volume_zscore","median_dollar_volume"],"volume",config.volume_windows_sessions)
    add(["amihud_illiquidity"],"volume",(20,60)); add(["volume_trend","up_day_volume_share","down_day_volume_share"],"volume",(20,60))
    for n in ["return_shock_vs_prior20_vol","market_residual_shock_vs_prior20_vol","sector_residual_shock_vs_prior20_vol","daily_range_shock_vs_prior20","gap_shock_vs_prior20","opening_gap","sector_adjusted_gap","beta_adjusted_gap","gap_fill_fraction_30m","gap_fill_fraction_60m","gap_fill_fraction_close","first_60m_return","opening_relative_volume_60m","close_location_in_daily_range","distance_close_from_session_vwap","last_60m_return","open_to_midday_return","midday_to_close_return","last_hour_volume_share","daily_range_pct","open_30m_volume_share","close_30m_volume_share","largest_5m_volume_share"]: out.append(_f(n,"intraday_shape" if "gap" not in n else "gap",None,0))
    add(["cumulative_first_60m_return","cumulative_last_60m_return","first_60m_minus_last_60m","average_close_location","average_open_30m_volume_share","average_close_30m_volume_share"],"intraday_shape",(3,5,10,20))
    for n in ["consecutive_up_days","consecutive_down_days","days_since_20d_high","days_since_20d_low","days_since_60d_high","days_since_60d_low"]: out.append(_f(n,"location",None,0))
    for n in ["market_beta_60","market_beta_120","market_correlation_60","market_correlation_120","downside_beta_120"]: out.append(_f(n,"market_sensitivity",None,0))
    for n in ["market_return_1","market_return_5","market_return_20","market_drawdown_20","market_drawdown_60","market_realized_vol_20","market_breadth_positive","market_breadth_above_sma20","cross_sectional_return_dispersion","average_pairwise_correlation_20"]: out.append(_f(n,"context",None,0,role="context_only"))
    if config.sector_scan_enabled:
        for n in ["sector_return_1","sector_return_3","sector_return_5","sector_return_10","sector_return_20","sector_breadth_positive","sector_dispersion"]: out.append(_f(n,"sector",None,0,sector=True))
    else:
        out.extend(_f(n,"sector",None,0,sector=True,status="unavailable",reason="No point-in-time sector table") for n in ["sector_return_1","sector_return_3","sector_return_5","sector_return_10","sector_return_20","sector_breadth_positive","sector_dispersion"])
    for n in ["peer_basket_return_1","peer_basket_return_3","peer_basket_return_5","peer_basket_return_10","peer_basket_return_20","stock_minus_peer_return_1","stock_minus_peer_return_3","stock_minus_peer_return_5","stock_minus_peer_return_10","stock_minus_peer_return_20","peer_spread_zscore_60","peer_correlation_60"]: out.append(_f(n,"peer",None,0,peer=True,status="unavailable",reason="No validated industry peer table"))
    if not config.sector_scan_enabled:
        out = [
            (InterdayFeatureSpec(**{**x.__dict__, "status": "unavailable", "unavailable_reason": "No point-in-time sector table", "sector_data_required": True})
             if x.name.startswith("sector_residual_return_") else x)
            for x in out
        ]
    names=[x.name for x in out]
    if len(names)!=len(set(names)): raise ValueError("Duplicate feature registry names")
    return out

def feature_definition_hash(spec): return hashlib.sha256(json.dumps(asdict(spec),sort_keys=True,default=str).encode()).hexdigest()
def target_definition_hash(spec): return hashlib.sha256(json.dumps(asdict(spec),sort_keys=True,default=str).encode()).hexdigest()

DAILY_CHECKPOINTS={1:("09:40","09:45","10:00","10:15","10:30","11:00","12:00","13:00","14:00","15:00","close5"),2:("open5","09:45","10:00","10:30","12:00","14:00","close5"),**{day:("open5","10:00","12:00","14:00","close5") for day in range(3,11)},12:("open5","12:00","close5"),14:("open5","12:00","close5"),18:("open5","12:00","close5"),20:("open5","12:00","close5")}

CHECKPOINT_ORDER = {
    "open5": 0, "09:40": 1, "09:45": 2, "10:00": 3,
    "10:15": 4, "10:30": 5, "11:00": 6, "12:00": 7,
    "13:00": 8, "14:00": 9, "15:00": 10, "close5": 11,
}


def endpoint_order(*, future_day: int, checkpoint: str, target_family: str) -> int:
    if target_family == "diagnostic_next_gap":
        return future_day * 100
    if checkpoint not in CHECKPOINT_ORDER:
        raise KeyError(f"Unknown executable checkpoint: {checkpoint}")
    return future_day * 100 + CHECKPOINT_ORDER[checkpoint]

def canonical_target_id(day,checkpoint,basis): return f"ret_entry_to_d{day:02d}_{checkpoint.replace(':','')}_{basis}"

def build_target_registry(config: InterdayConfig) -> list[InterdayTargetSpec]:
    bases=["raw"]+(["sector"] if config.sector_scan_enabled else []); out=[]; seen=set()
    for day in config.target_horizons_sessions:
        for cp in DAILY_CHECKPOINTS.get(day,("open5","12:00","close5")):
            for basis in bases:
                name=canonical_target_id(day,cp,basis)
                if name in seen: continue
                seen.add(name); family="daily_terminal" if cp=="close5" else "time_of_day"; out.append(InterdayTargetSpec(name,name,f"Return from D+1 open5 to D+{day} {cp}",family,f"{family}",day,day,cp,"open5",cp,basis,"simple", "aligned_QQQ_interval",True,False,day,config.minimum_sector_members_ex_focal if basis=="sector" else 0,endpoint_order=endpoint_order(future_day=day,checkpoint=cp,target_family=family)))
    for basis in bases:
        name=f"diagnostic_next_gap_{basis}"; out.append(InterdayTargetSpec(name,name,"D close5 to D+1 open5", "diagnostic_next_gap","time_of_day",1,1,"open5","decision_close5","open5",basis,"simple","aligned_QQQ_interval",False,True,1,config.minimum_sector_members_ex_focal if basis=="sector" else 0,endpoint_order=endpoint_order(future_day=1,checkpoint="open5",target_family="diagnostic_next_gap")))
    return out
