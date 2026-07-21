from __future__ import annotations
import pandas as pd
import numpy as np
from .horizon import build_horizon_profiles,build_checkpoint_profiles,add_local_neighbor_metrics
from .inference import benjamini_hochberg

def apply_interday_fdr(results: pd.DataFrame) -> pd.DataFrame:
    key=["feature","target","test_type"]
    if results.duplicated(key).any():
        duplicates=results.loc[results.duplicated(key,keep=False),key]
        raise ValueError("Duplicate hypotheses:\n"+duplicates.to_string(index=False))
    output=results.copy(); output["global_fdr"]=np.nan
    if "feature_family" not in output: output["feature_family"]=output["feature"]
    for _,indices in output.groupby("fdr_family",sort=False).groups.items():
        output.loc[indices,"global_fdr"]=benjamini_hochberg(output.loc[indices,"raw_p"])
    output["feature_family_fdr"]=output.groupby(["fdr_family","feature_family"],dropna=False)["raw_p"].transform(benjamini_hochberg)
    return output

def _bh_series(values):
    p=np.asarray(values,float); order=np.argsort(p); adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; out=np.full(len(p),np.nan); out[order]=np.minimum(adjusted,1); return pd.Series(out,index=values.index)

def candidate_passes(row: pd.Series, config) -> bool:
    if bool(row.get("diagnostic_only",False)) or not bool(row.get("is_executable",True)): return False
    if not np.isfinite(row.global_fdr) or row.global_fdr>config.primary_fdr_threshold: return False
    if row.test_type=="rank_ic":
        if row.valid_dates<config.minimum_candidate_rank_ic_dates or abs(row.effect)<config.minimum_rank_ic_effect: return False
    else:
        if row.valid_dates<config.minimum_candidate_decile_dates or abs(row.effect_bps)<config.effect_floor_bps(row.horizon_sessions): return False
    if row.distinct_symbols<config.minimum_candidate_symbols: return False
    return bool(row.get("neighbor_supported",False))

def select_candidates(scan_results: pd.DataFrame, config) -> pd.DataFrame:
    if scan_results.empty:return scan_results.copy()
    result=apply_interday_fdr(scan_results)
    groups=[]
    for keys,group in result.groupby(["feature","test_type","return_basis","target_family"],dropna=False):
        order=["horizon_sessions"] if keys[3]!="time_of_day" else ["future_day","checkpoint"]
        groups.append(add_local_neighbor_metrics(group,order))
    result=pd.concat(groups,ignore_index=True) if groups else result; result["distinct_symbols"]=result["distinct_symbols"].fillna(0); result["candidate_status"]=result.apply(lambda row:"shortlisted" if candidate_passes(row,config) else "not_shortlisted",axis=1)
    return result.loc[result.candidate_status.eq("shortlisted")].sort_values("global_fdr")

def _bh(x):
    p=x.to_numpy(float); out=np.full(len(p),np.nan); good=np.isfinite(p)
    if good.any():
        q=p[good]; order=np.argsort(q); adj=np.minimum.accumulate((q[order]*len(q)/np.arange(1,len(q)+1))[::-1])[::-1]; out[np.flatnonzero(good)[order]]=np.minimum(adj,1)
    return out

def cluster_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:return candidates.copy()
    out=candidates.copy()
    redundancy = out.get("feature_redundancy_group", out.get("redundancy_group", out["feature"]))
    basis = out["return_basis"].astype(str) if "return_basis" in out else pd.Series("raw", index=out.index)
    out["candidate_cluster"] = redundancy.astype(str) + "|" + out["test_type"].astype(str) + "|" + basis
    return out

def target_safe_fold_mask(decision_dates,entry_dates,exit_dates,fold_start,fold_end):
    return (pd.to_datetime(decision_dates)>=pd.Timestamp(fold_start))&(pd.to_datetime(exit_dates)<pd.Timestamp(fold_end))&(pd.to_datetime(entry_dates)>=pd.Timestamp(fold_start))
