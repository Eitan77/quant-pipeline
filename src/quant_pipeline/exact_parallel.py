from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScanConfig
from .registry import FeatureSpec
from .scanner import scan


def exact_pair(
    feature_path: Path,
    target_path: Path,
    spec: FeatureSpec,
    target: str,
    config: ScanConfig,
    direction_hint: float | None = None,
) -> tuple[dict | None, list[dict]]:
    identifiers=["symbol","session_date","decision_ts","scan_eligible","session_grid_eligible"]
    feature_frame=pd.read_parquet(feature_path,columns=[*identifiers,spec.name])
    target_frame=pd.read_parquet(target_path,columns=[target])
    frame=pd.concat([feature_frame.reset_index(drop=True),target_frame.reset_index(drop=True)],axis=1)
    eligible=frame.scan_eligible.fillna(False)&frame.session_grid_eligible.fillna(False)
    discovery=frame.loc[eligible&pd.to_datetime(frame.session_date).le(pd.Timestamp(config.selection_end))]
    confirmation=frame.loc[eligible&pd.to_datetime(frame.session_date).ge(pd.Timestamp(config.confirmation_start))&pd.to_datetime(frame.session_date).le(pd.Timestamp(config.discovery_end))]
    exact,tables=scan(discovery,[spec],[target],replace(config,use_cuda=False),None,skip_dense=False,direction_hint=direction_hint)
    row=exact.iloc[0].to_dict() if not exact.empty else None
    confirmed,_=scan(confirmation,[spec],[target],replace(config,use_cuda=False),None,skip_dense=False,direction_hint=direction_hint)
    if row is not None:
        if not confirmed.empty:
            c=confirmed.iloc[0]
            for column in ["n","valid_sessions","valid_symbols","spearman","top_bottom_spread","two_way_cluster_t","session_bootstrap_ci_low","session_bootstrap_ci_high","symbol_breadth","time_stability"]:
                if column in c:row[f"confirmation_{column}"]=c[column]
            expected=np.sign(direction_hint or row.get("spearman",0)); row["internal_confirmation_direction_match"]=bool(np.sign(c.get("top_bottom_spread",np.nan))==expected) if np.isfinite(c.get("top_bottom_spread",np.nan)) else False
            row["internal_confirmation_effect_ratio"]=abs(c.get("top_bottom_spread",np.nan)/row.get("top_bottom_spread",np.nan)) if row.get("top_bottom_spread") not in [0,None] else np.nan
        row.update(_walk_forward(frame.loc[eligible],spec.name,target,direction_hint or row.get("spearman",0)))
    table=tables.get((spec.name,target),pd.DataFrame()).to_dict("records")
    return row,table


def _walk_forward(frame:pd.DataFrame,feature:str,target:str,direction:float)->dict:
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
    return {"walk_forward_folds":len(signed),"walk_forward_mean_signed_effect":float(np.mean(signed)) if signed else np.nan,"walk_forward_positive_fold_pct":float(np.mean(np.asarray(signed)>0)) if signed else np.nan,"walk_forward_worst_signed_effect":float(np.min(signed)) if signed else np.nan}
