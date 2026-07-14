from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ScanConfig
from .holdout import assert_pre_holdout_frame
from .scanner import _cross_sectional_ic, _hac_mean, _signed_decile_spread


def _pair_metrics(frame:pd.DataFrame,feature:str,target:str,direction_sign:int,config:ScanConfig)->dict:
    z=frame[["symbol","session_date","decision_ts",feature,target]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<max(20,config.min_bin_observations):return {"observations":len(z),"sessions":z.session_date.nunique(),"symbols":z.symbol.nunique()}
    spread=_signed_decile_spread(z,feature,target); signed=spread*direction_sign if np.isfinite(spread) else np.nan
    try:bins=pd.qcut(z[feature].rank(method="first"),config.quantiles,labels=False,duplicates="drop")
    except ValueError:bins=pd.Series(np.nan,index=z.index)
    daily=z.assign(_bin=bins).groupby(["session_date","_bin"],observed=True)[target].mean().unstack()
    daily_spread=(daily.iloc[:,-1]-daily.iloc[:,0])*direction_sign if daily.shape[1]>1 else pd.Series(dtype=float)
    _,se,_=_hac_mean(daily_spread); low=signed-1.96*se if np.isfinite(signed) and np.isfinite(se) else np.nan; high=signed+1.96*se if np.isfinite(signed) and np.isfinite(se) else np.nan
    rho=float(z[feature].corr(z[target],method="spearman"))*direction_sign
    ic=_cross_sectional_ic(z,feature,target,config.cross_sectional_min_symbols)
    return {"effect":signed,"raw_effect":spread,"spearman":rho,"cross_sectional_ic":ic.get("ic_mean"),"ci_low":low,"ci_high":high,"observations":len(z),"sessions":z.session_date.nunique(),"symbols":z.symbol.nunique(),"direction_consistent":bool(np.isfinite(signed) and signed>0)}


def recent_period_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:float,config:ScanConfig)->tuple[dict,pd.DataFrame]:
    assert_pre_holdout_frame(frame,config.sealed_holdout_start,"recent-period diagnostic")
    sign=1 if direction>=0 else -1; dates=pd.to_datetime(frame.session_date); end=pd.Timestamp(config.discovery_end)
    starts={"full_discovery":pd.Timestamp(config.start),"recent_5y":end-pd.DateOffset(years=5),"recent_3y":end-pd.DateOffset(years=3),"recent_2y":end-pd.DateOffset(years=2),"recent_12m":end-pd.DateOffset(months=12),"jan_apr_2026":pd.Timestamp("2026-01-01")}
    rows=[]
    for label,start in starts.items():
        sub=frame.loc[dates.ge(start)&dates.le(end)]; metrics=_pair_metrics(sub,feature,target,sign,config); rows.append({"period":label,"start":str(max(start,pd.Timestamp(config.start)).date()),"end":str(end.date()),**metrics})
    for year in sorted(dates.dt.year.dropna().unique()):
        sub=frame.loc[dates.dt.year.eq(year)]; rows.append({"period":f"calendar_{year}","start":f"{year}-01-01","end":str(min(end,pd.Timestamp(f'{year}-12-31')).date()),**_pair_metrics(sub,feature,target,sign,config)})
    table=pd.DataFrame(rows); effects=table.set_index("period").effect if "effect" in table else pd.Series(dtype=float); full=effects.get("full_discovery",np.nan); recent=effects.get("recent_12m",np.nan); ratio=recent/full if np.isfinite(full) and full!=0 and np.isfinite(recent) else np.nan
    recent_row=table.loc[table.period.eq("recent_12m")].iloc[0]; enough=recent_row.get("sessions",0)>=min(config.min_sessions,100) and recent_row.get("symbols",0)>=min(config.min_symbols,20)
    if not enough:classification="insufficient_recent_data"
    elif not np.isfinite(recent) or recent<=0:classification="historically_strong_but_currently_weak" if np.isfinite(full) and full>0 else "regime_unstable"
    elif np.isfinite(ratio) and ratio>=1.5:classification="strengthening_recently"
    elif np.isfinite(ratio) and ratio<.5:classification="weakening_recently"
    elif np.isfinite(full) and full<=0<recent:classification="recently_emerged"
    else:classification="persistent"
    summary={"recent_classification":classification,"recent_to_full_effect_ratio":ratio,"recent_12m_effect":recent}
    for period in ["recent_5y","recent_3y","recent_2y","jan_apr_2026"]:
        summary[f"{period}_effect"]=effects.get(period,np.nan)
    return summary,table


def recency_weighted_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:float,config:ScanConfig)->tuple[dict,pd.DataFrame]:
    assert_pre_holdout_frame(frame,config.sealed_holdout_start,"recency-weighted diagnostic")
    z=frame[["symbol","session_date",feature,target]].replace([np.inf,-np.inf],np.nan).dropna().copy(); sign=1 if direction>=0 else -1
    if z.empty:return {},pd.DataFrame()
    dates=pd.to_datetime(z.session_date); end=pd.Timestamp(config.discovery_end); rows=[]
    for months in config.recency_half_lives_months:
        session_age=(end-dates).dt.total_seconds().div(86400); session_weight=np.exp(-np.log(2)*session_age/(months*365.25/12))
        counts=z.groupby("session_date").symbol.transform("size"); row_weight=session_weight/counts
        xr=z[feature].rank(method="average").to_numpy(float); yr=z[target].rank(method="average").to_numpy(float); w=row_weight.to_numpy(float).copy(); w/=w.sum()
        mx=np.sum(w*xr); my=np.sum(w*yr); covariance=np.sum(w*(xr-mx)*(yr-my)); correlation=covariance/np.sqrt(np.sum(w*(xr-mx)**2)*np.sum(w*(yr-my)**2))
        bins=pd.qcut(z[feature].rank(method="first"),config.quantiles,labels=False,duplicates="drop"); work=z.assign(_bin=bins,_weight=session_weight)
        daily=work.groupby(["session_date","_bin"],observed=True).agg(target=(target,"mean"),weight=("_weight","first")).reset_index()
        means=daily.groupby("_bin",observed=True).apply(lambda q:np.average(q.target,weights=q.weight),include_groups=False); effect=(means.iloc[-1]-means.iloc[0])*sign if len(means)>1 else np.nan
        session_weights=work.drop_duplicates("session_date")._weight.to_numpy(float); effective=(session_weights.sum()**2)/(session_weights@session_weights)
        rows.append({"half_life_months":months,"exploratory":True,"signed_effect":effect,"signed_spearman":correlation*sign,"effective_sessions":effective,"actual_sessions":z.session_date.nunique()})
    table=pd.DataFrame(rows); summary={f"recency_{int(r.half_life_months)}m_effect":r.signed_effect for r in table.itertuples()}; summary["recency_weighted_exploratory"]=True; summary["recency_effective_sessions_min"]=float(table.effective_sessions.min())
    return summary,table


def symbol_and_concentration_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:float,config:ScanConfig)->tuple[dict,dict[str,pd.DataFrame]]:
    assert_pre_holdout_frame(frame,config.sealed_holdout_start,"symbol/concentration diagnostic")
    sign=1 if direction>=0 else -1; optional=[c for c in ["close_raw","volume","sector","industry","market_cap","market_context_return"] if c in frame]; z=frame[["symbol","session_date","decision_ts",feature,target,*optional]].replace([np.inf,-np.inf],np.nan).dropna(subset=[feature,target]).copy(); end=pd.Timestamp(config.discovery_end); rows=[]
    for symbol,group in z.groupby("symbol",sort=False):
        sessions=group.session_date.nunique(); reliable=sessions>=50; spread=_signed_decile_spread(group,feature,target); signed=spread*sign if np.isfinite(spread) else np.nan; years=pd.to_datetime(group.session_date).dt.year
        annual=group.assign(_year=years).groupby("_year").apply(lambda q:_signed_decile_spread(q,feature,target)*sign,include_groups=False).dropna()
        recent12=group.loc[pd.to_datetime(group.session_date).ge(end-pd.DateOffset(months=12))]; recent24=group.loc[pd.to_datetime(group.session_date).ge(end-pd.DateOffset(months=24))]
        try:bins=pd.qcut(group[feature].rank(method="first"),10,labels=False,duplicates="drop"); daily=group.assign(_bin=bins).groupby(["session_date","_bin"],observed=True)[target].mean().unstack(); spread_series=(daily.iloc[:,-1]-daily.iloc[:,0])*sign if daily.shape[1]>1 else pd.Series(dtype=float); _,se,_=_hac_mean(spread_series)
        except ValueError:se=np.nan
        rows.append({"symbol":symbol,"valid_observations":len(group),"unique_sessions":sessions,"mean_target_return":group[target].mean(),"median_target_return":group[target].median(),"spearman":group[feature].corr(group[target],method="spearman"),"signed_quantile_spread":signed,"session_ci_low":signed-1.96*se if np.isfinite(signed) and np.isfinite(se) else np.nan,"session_ci_high":signed+1.96*se if np.isfinite(signed) and np.isfinite(se) else np.nan,"candidate_direction_win_rate":float((group[target]*sign>0).mean()),"profitable_year_pct":float((annual>0).mean()) if len(annual) else np.nan,"recent_12m_effect":_signed_decile_spread(recent12,feature,target)*sign,"recent_24m_effect":_signed_decile_spread(recent24,feature,target)*sign,"direction_matches":bool(np.isfinite(signed) and signed>0),"passes_minimum_sample":reliable})
    symbols=pd.DataFrame(rows)
    if symbols.empty:return {"symbol_breadth_classification":"insufficient_evidence"},{"symbol":symbols}
    contribution=(symbols.signed_quantile_spread.fillna(0)*symbols.valid_observations); absolute=contribution.abs(); total=absolute.sum(); symbols["effect_contribution_pct"]=absolute/total if total else 0; symbols["observation_contribution_pct"]=symbols.valid_observations/symbols.valid_observations.sum()
    ordered=symbols.sort_values("effect_contribution_pct",ascending=False); shares=ordered.effect_contribution_pct
    def removed(k:int)->float:
        names=set(ordered.head(k).symbol); return _signed_decile_spread(z.loc[~z.symbol.isin(names)],feature,target)*sign
    reliable=symbols.loc[symbols.passes_minimum_sample]; expected=float(reliable.direction_matches.mean()) if len(reliable) else np.nan; best=float(shares.head(1).sum()); top3=float(shares.head(3).sum()); hhi=float((shares**2).sum())
    if len(reliable)<3:breadth="insufficient_evidence"
    elif best>=.60:breadth="single_symbol_dominated"
    elif top3>=.75 or hhi>=.25:breadth="highly_concentrated"
    elif top3>=.50 or expected<.60:breadth="moderately_concentrated"
    else:breadth="broad_across_symbols"
    summary={"best_symbol_effect_pct":best,"top3_symbol_effect_pct":top3,"top5_symbol_effect_pct":float(shares.head(5).sum()),"symbol_effect_hhi":hhi,"effect_remove_best_symbol":removed(1),"effect_remove_top3_symbols":removed(3),"effect_remove_top5_symbols":removed(5),"equal_weight_symbol_effect":float(reliable.signed_quantile_spread.mean()) if len(reliable) else np.nan,"capped_symbol_effect":float(reliable.signed_quantile_spread.clip(upper=reliable.signed_quantile_spread.quantile(.95)).mean()) if len(reliable) else np.nan,"median_symbol_effect":float(reliable.signed_quantile_spread.median()) if len(reliable) else np.nan,"lower_quartile_symbol_effect":float(reliable.signed_quantile_spread.quantile(.25)) if len(reliable) else np.nan,"eligible_symbols_expected_direction_pct":expected,"eligible_symbols_meaningful_effect_pct":float((reliable.signed_quantile_spread>=config.minimum_effect_bps/10000).mean()) if len(reliable) else np.nan,"symbol_breadth_classification":breadth}
    local=pd.to_datetime(z.decision_ts,utc=True).dt.tz_convert("America/New_York"); z["time_bucket"]=pd.cut(local.dt.hour*60+local.dt.minute,[0,630,720,900,1440],labels=["open","morning","midday","close"])
    time=z.groupby("time_bucket",observed=True).apply(lambda q:pd.Series({"observations":len(q),"signed_effect":_signed_decile_spread(q,feature,target)*sign}),include_groups=False).reset_index()
    if len(time):summary["time_concentration_classification"]=str(time.loc[time.signed_effect.idxmax(),"time_bucket"])+"_concentrated"
    if breadth=="broad_across_symbols":recommendation="advance_as_broad_cross_sectional_candidate"
    elif breadth in {"moderately_concentrated","highly_concentrated"}:recommendation="advance_for_conditional_phase2_testing"
    elif breadth=="single_symbol_dominated":recommendation="advance_as_symbol_specific_candidate" if len(reliable) else "reject_as_concentrated_or_unstable"
    else:recommendation="retain_for_monitoring_only"
    summary["phase2_recommendation"]=recommendation
    tables={"symbol":symbols,"time_of_day":time}
    if {"close_raw","volume"}.issubset(z):
        z["dollar_volume"]=z.close_raw*z.volume
        for column,name in [("dollar_volume","dollar_volume"),("close_raw","price")]:
            try:z[f"{name}_group"]=pd.qcut(z[column].rank(method="first"),4,labels=["low","mid_low","mid_high","high"])
            except ValueError:continue
            tables[name]=z.groupby(f"{name}_group",observed=True).apply(lambda q:pd.Series({"observations":len(q),"symbols":q.symbol.nunique(),"signed_effect":_signed_decile_spread(q,feature,target)*sign}),include_groups=False).reset_index()
    for column in ["sector","industry"]:
        if column in z:
            tables[column]=z.groupby(column,observed=True).apply(lambda q:pd.Series({"observations":len(q),"symbols":q.symbol.nunique(),"signed_effect":_signed_decile_spread(q,feature,target)*sign}),include_groups=False).reset_index()
    if "market_context_return" in z:
        z["market_regime"]=np.where(z.market_context_return.ge(0),"market_up","market_down")
        tables["market_regime"]=z.groupby("market_regime").apply(lambda q:pd.Series({"observations":len(q),"signed_effect":_signed_decile_spread(q,feature,target)*sign}),include_groups=False).reset_index()
    return summary,tables
