from __future__ import annotations
import pandas as pd
import numpy as np
from .horizon import build_horizon_profiles,build_checkpoint_profiles

def apply_interday_fdr(results: pd.DataFrame) -> pd.DataFrame:
    key=["feature","target","test_type"]
    if results.duplicated(key).any(): raise ValueError("Duplicate hypothesis rows")
    output=results.copy(); output["global_fdr"]=np.nan
    for _,indices in output.groupby("fdr_family",sort=False).groups.items():
        p=output.loc[indices,"raw_p"]; valid=p.notna()
        if valid.any(): output.loc[p.index[valid],"global_fdr"]=_bh_series(p[valid])
    return output

def _bh_series(values):
    p=np.asarray(values,float); order=np.argsort(p); adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; out=np.full(len(p),np.nan); out[order]=np.minimum(adjusted,1); return pd.Series(out,index=values.index)

def select_candidates(scan_results: pd.DataFrame, config) -> pd.DataFrame:
    if scan_results.empty:return scan_results.copy()
    result=apply_interday_fdr(scan_results); result["statistical_pass"]=result.global_fdr.le(config.primary_fdr_threshold)
    result["coverage_pass"]=np.where(result.test_type.eq("rank_ic"),result.valid_dates.ge(config.minimum_candidate_rank_ic_dates),result.valid_dates.ge(config.minimum_candidate_decile_dates))
    result["economic_pass"]=np.where(result.test_type.eq("rank_ic"),result.effect.abs().ge(config.minimum_rank_ic_effect),result.effect_bps.abs().ge(result.horizon_sessions.fillna(20).astype(int).map(config.effect_floor_bps)))
    horizon_profile=build_horizon_profiles(result); checkpoint_profile=build_checkpoint_profiles(result)
    result["neighbor_pass"]=False
    for idx,row in result.iterrows():
        table=checkpoint_profile if row.target_family=="time_of_day" else horizon_profile; match=table.loc[(table.feature==row.feature)&(table.test_type==row.test_type)&(table.return_basis==row.return_basis)]
        if not match.empty:
            retention=float(match.iloc[0].get("best_neighbor_retention",match.iloc[0].get("neighbor_retention",np.nan))); isolated=bool(match.iloc[0].get("isolated_spike",False)); result.loc[idx,"neighbor_pass"]=np.isfinite(retention) and retention>=config.minimum_neighbor_retention and not isolated
    result["candidate_status"]=np.where(result.statistical_pass&result.coverage_pass&result.economic_pass&result.neighbor_pass,"shortlisted","not_shortlisted")
    return result.loc[result.candidate_status.eq("shortlisted")].sort_values("global_fdr")

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
