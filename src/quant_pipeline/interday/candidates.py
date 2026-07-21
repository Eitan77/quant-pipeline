from __future__ import annotations
import pandas as pd
import numpy as np
from .horizon import build_horizon_profiles,build_checkpoint_profiles

def select_candidates(scan_results: pd.DataFrame, config) -> pd.DataFrame:
    if scan_results.empty:return scan_results.copy()
    result=scan_results.copy(); p=pd.to_numeric(result.raw_p,errors="coerce"); result["global_fdr"]=p.groupby([result.fdr_family,result.test_type],dropna=False).transform(lambda x: _bh(x)); result["effect_floor_pass"]=result.apply(lambda r: abs(r.effect_bps)>=config.effect_floor_bps(int(r.horizon_sessions or 20)) if pd.notna(r.effect_bps) and pd.notna(r.horizon_sessions) else False,axis=1); result["candidate_status"]=np.where((result.global_fdr<=config.primary_fdr_threshold)&result.effect_floor_pass,"shortlisted","not_shortlisted"); return result.loc[result.candidate_status.eq("shortlisted")].sort_values("global_fdr")

def _bh(x):
    p=x.to_numpy(float); out=np.full(len(p),np.nan); good=np.isfinite(p)
    if good.any():
        q=p[good]; order=np.argsort(q); adj=np.minimum.accumulate((q[order]*len(q)/np.arange(1,len(q)+1))[::-1])[::-1]; out[np.flatnonzero(good)[order]]=np.minimum(adj,1)
    return out

def cluster_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:return candidates.copy()
    out=candidates.copy(); out["candidate_cluster"]=out.feature.str.rsplit("_",n=1).str[0]+"|"+out.test_type.astype(str); return out

def target_safe_fold_mask(decision_dates,entry_dates,exit_dates,fold_start,fold_end):
    return (pd.to_datetime(decision_dates)>=pd.Timestamp(fold_start))&(pd.to_datetime(exit_dates)<pd.Timestamp(fold_end))&(pd.to_datetime(entry_dates)>=pd.Timestamp(fold_start))
