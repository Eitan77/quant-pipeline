from __future__ import annotations
import pandas as pd
import numpy as np
from .horizon import build_horizon_profiles,build_checkpoint_profiles,add_local_neighbor_metrics
from .inference import benjamini_hochberg

FOLDS = (
    ("2019_2020", "2019-06-21", "2021-01-01"),
    ("2021_2022", "2021-01-01", "2023-01-01"),
    ("2023_2024", "2023-01-01", "2025-01-01"),
    ("2025_2026_apr", "2025-01-01", "2026-05-01"),
)

RECENT_WINDOWS = (
    ("last_24_months", pd.DateOffset(months=24)),
    ("last_12_months", pd.DateOffset(months=12)),
)


def target_safe_fold_mask(*, sessions, entry_date_ids, exit_date_ids, start: str, stop: str) -> np.ndarray:
    start_ts = pd.Timestamp(start)
    stop_ts = pd.Timestamp(stop)
    decision = pd.DatetimeIndex(sessions)
    mask = ((decision >= start_ts) & (decision < stop_ts) & (entry_date_ids >= 0) & (exit_date_ids >= 0))
    valid = np.flatnonzero(mask)
    if len(valid):
        entry = decision[entry_date_ids[valid]]
        exit_ = decision[exit_date_ids[valid]]
        mask[valid] &= ((entry >= start_ts) & (entry < stop_ts) & (exit_ >= start_ts) & (exit_ < stop_ts))
    return mask


def expected_sign_retained(values: np.ndarray, *, expected_sign: float, minimum_observations: int) -> bool:
    valid = values[np.isfinite(values)]
    return len(valid) >= minimum_observations and np.sign(np.mean(valid)) == np.sign(expected_sign)


def recent_window_mask(sessions, offset: pd.DateOffset, discovery_end: str) -> np.ndarray:
    end = pd.Timestamp(discovery_end)
    return (pd.DatetimeIndex(sessions) >= end - offset) & (pd.DatetimeIndex(sessions) <= end)

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
    target_family = out.get("target_family", pd.Series("unknown", index=out.index))
    out["candidate_cluster"] = redundancy.astype(str) + "|" + out["test_type"].astype(str) + "|" + basis + "|" + target_family.astype(str)
    return out
