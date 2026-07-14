from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScanConfig
from .diagnostics import recent_period_diagnostics,recency_weighted_diagnostics,symbol_and_concentration_diagnostics
from .holdout import assert_pre_holdout_frame
from .registry import FeatureSpec
from .scanner import scan


def exact_pair(
    feature_path: Path,
    target_path: Path,
    spec: FeatureSpec,
    target: str,
    config: ScanConfig,
    direction_hint: float | None = None,
) -> tuple[dict | None, list[dict], dict[str,list[dict]]]:
    import pyarrow.parquet as pq
    identifiers=["symbol","session_date","bar_start_ts","decision_ts","analysis_eligible"]
    feature_schema=set(pq.ParquetFile(feature_path).schema.names); optional=[c for c in ["close_raw","volume","sector","industry","market_cap"] if c in feature_schema]
    feature_frame=pd.read_parquet(feature_path,columns=[*identifiers,*optional,spec.name])
    target_schema=set(pq.ParquetFile(target_path).schema.names); raw=target.removesuffix("_benchmark_adjusted").removesuffix("_beta_residual"); adjusted=f"{raw}_benchmark_adjusted"; target_columns=["symbol","session_date","bar_start_ts","decision_ts",target]
    for column in [raw,adjusted]:
        if column in target_schema and column not in target_columns:target_columns.append(column)
    target_frame=pd.read_parquet(target_path,columns=target_columns)
    frame=feature_frame.merge(target_frame,on=["symbol","session_date","bar_start_ts","decision_ts"],how="inner",validate="one_to_one")
    if len(frame)!=len(feature_frame):raise ValueError("Feature-target cache key mismatch")
    if raw in frame and adjusted in frame:frame["market_context_return"]=frame[raw]-frame[adjusted]
    assert_pre_holdout_frame(frame,config.sealed_holdout_start,"exact pair")
    eligible=frame.analysis_eligible.fillna(False)
    dates=pd.to_datetime(frame.session_date); full=frame.loc[eligible&dates.le(pd.Timestamp(config.discovery_end))]
    if config.use_separate_confirmation_period:
        if not config.selection_end or not config.confirmation_start:raise ValueError("Separate confirmation mode requires selection_end and confirmation_start")
        discovery=full.loc[pd.to_datetime(full.session_date).le(pd.Timestamp(config.selection_end))]
        confirmation=full.loc[pd.to_datetime(full.session_date).ge(pd.Timestamp(config.confirmation_start))]
    else:
        discovery=full; confirmation=full.iloc[0:0]
    exact,tables=scan(discovery,[spec],[target],replace(config,use_cuda=False),None,skip_dense=False,direction_hint=direction_hint)
    row=exact.iloc[0].to_dict() if not exact.empty else None
    confirmed,_=scan(confirmation,[spec],[target],replace(config,use_cuda=False),None,skip_dense=False,direction_hint=direction_hint) if config.use_separate_confirmation_period else (pd.DataFrame(),{})
    diagnostic_tables={}
    if row is not None:
        if config.use_separate_confirmation_period and not confirmed.empty:
            c=confirmed.iloc[0]
            for column in ["n","valid_sessions","valid_symbols","spearman","top_bottom_spread","two_way_cluster_t","two_way_cluster_p","session_bootstrap_ci_low","session_bootstrap_ci_high","symbol_breadth","time_stability","outlier_worst_signed_spread"]:
                if column in c:row[f"confirmation_{column}"]=c[column]
            expected=np.sign(direction_hint or row.get("spearman",0)); row.update(_confirmation_gate(float(row.get("top_bottom_spread",np.nan)),c.to_dict(),expected,config))
        elif config.use_separate_confirmation_period:row.update(_confirmation_gate(float(row.get("top_bottom_spread",np.nan)),{},np.sign(direction_hint or row.get("spearman",0)),config))
        direction=direction_hint or row.get("spearman",0)
        if config.run_historical_walk_forward_diagnostics:row.update(_historical_subperiod_diagnostics(full,spec.name,target,direction))
        if config.run_recent_period_diagnostics:
            summary,table=recent_period_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables["recent_periods"]=table.to_dict("records")
        if config.run_recency_weighted_diagnostics:
            summary,table=recency_weighted_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables["recency_weighted"]=table.to_dict("records")
        summary,diagnostics=symbol_and_concentration_diagnostics(full,spec.name,target,direction,config); row.update(summary); diagnostic_tables.update({name:table.to_dict("records") for name,table in diagnostics.items()})
    table=tables.get((spec.name,target),pd.DataFrame()).to_dict("records")
    return row,table,diagnostic_tables


def _confirmation_gate(discovery_effect:float,confirmation:dict,expected:float,config:ScanConfig)->dict:
    effect=float(confirmation.get("top_bottom_spread",np.nan)); ratio=abs(effect/discovery_effect) if np.isfinite(discovery_effect) and discovery_effect else np.nan; direction_ok=np.isfinite(effect) and np.sign(effect)==expected
    economic=direction_ok and abs(effect)>=config.confirmation_min_effect_bps/10000; ratio_ok=np.isfinite(ratio) and config.confirmation_min_discovery_ratio<=ratio<=config.confirmation_max_discovery_ratio; coverage=confirmation.get("valid_sessions",0)>=config.confirmation_min_sessions and confirmation.get("valid_symbols",0)>=config.confirmation_min_symbols
    bootstrap=confirmation.get("session_bootstrap_ci_low",np.nan)>0 if expected>0 else confirmation.get("session_bootstrap_ci_high",np.nan)<0; statistical=confirmation.get("two_way_cluster_p",1)<config.confirmation_alpha and bootstrap; robust=confirmation.get("outlier_worst_signed_spread",-np.inf)>0
    status="not_replicated"
    if direction_ok:status="directionally_replicated"
    if direction_ok and economic and ratio_ok:status="economically_replicated"
    if direction_ok and economic and ratio_ok and coverage and statistical:status="statistically_confirmed"
    return {"discovery_effect":discovery_effect,"confirmation_effect":effect,"confirmation_to_discovery_ratio":ratio,"confirmation_clustered_t":confirmation.get("two_way_cluster_t"),"confirmation_p_value":confirmation.get("two_way_cluster_p"),"confirmation_bootstrap_lower":confirmation.get("session_bootstrap_ci_low"),"confirmation_bootstrap_upper":confirmation.get("session_bootstrap_ci_high"),"confirmation_sessions":confirmation.get("valid_sessions"),"confirmation_symbols":confirmation.get("valid_symbols"),"internal_confirmation_direction_match":bool(direction_ok),"internal_confirmation_effect_ratio":ratio,"confirmation_status":status,"confirmation_robust":bool(robust)}


def _historical_subperiod_diagnostics(frame:pd.DataFrame,feature:str,target:str,direction:float)->dict:
    years=pd.to_datetime(frame.session_date).dt.year; folds=[]
    for test_year in [2022,2023,2024,2025,2026]:
        train=frame.loc[years.lt(test_year),feature].dropna()
        test=frame.loc[years.eq(test_year)]
        if test_year==2026:test=test.loc[pd.to_datetime(test.session_date).le(pd.Timestamp("2026-04-30"))]
        if len(test)<100 or len(train)<100:continue
        edges=np.unique(train.quantile(np.linspace(.1,.9,9)).to_numpy())
        if len(edges)<2:continue
        bins=pd.Series(np.searchsorted(edges,test[feature].to_numpy(),side="right"),index=test.index)
        means=test.groupby(bins,observed=True)[target].mean(); spread=float(means.iloc[-1]-means.iloc[0]) if len(means)>1 else np.nan
        folds.append({"year":test_year,"spread":spread})
    signed=[x["spread"]*np.sign(direction) for x in folds if np.isfinite(x["spread"])]
    return {"diagnostic_evidence_label":"historical_subperiod_stability","historical_subperiod_folds":len(signed),"historical_subperiod_mean_signed_effect":float(np.mean(signed)) if signed else np.nan,"historical_subperiod_positive_fold_pct":float(np.mean(np.asarray(signed)>0)) if signed else np.nan,"historical_subperiod_worst_signed_effect":float(np.min(signed)) if signed else np.nan}


def _walk_forward(frame:pd.DataFrame,feature:str,target:str,direction:float)->dict:
    """Backward-compatible name; results are explicitly not OOS evidence."""
    return _historical_subperiod_diagnostics(frame,feature,target,direction)
