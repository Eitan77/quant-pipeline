from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from .models import RankBinCache

def row_information(values, valid):
    selected=values[valid]; count=len(selected)
    if not count: return 0,0,np.nan
    _,counts=np.unique(selected,return_counts=True); return count,len(counts),float(counts.max()/count)

def deterministic_bins(values, security_ids, bins):
    output=np.full(len(values),-1,np.int8); valid=np.isfinite(values); indices=np.flatnonzero(valid)
    if len(indices)<bins: return output
    order=np.lexsort((np.asarray(security_ids,dtype=np.int64)[indices],values[indices])); ordered=indices[order]; base,remainder=divmod(len(ordered),bins); start=0
    for bin_id in range(bins):
        width=base+int(bin_id<remainder); output[ordered[start:start+width]]=bin_id; start+=width
    return output

def equal_weight_tail(bins,tail_bin):
    membership=bins==tail_bin; count=membership.sum(axis=1,keepdims=True); weights=np.zeros(membership.shape,dtype=np.float64); valid_dates=count[:,0]>0; weights[valid_dates]=membership[valid_dates]/count[valid_dates]; return weights

def one_way_turnover(weights):
    output=np.full(weights.shape[0],np.nan,dtype=np.float64); output[1:]=0.5*np.abs(weights[1:]-weights[:-1]).sum(axis=1); return output

def feature_tail_turnover(deciles,quintiles,feature_id):
    feature_deciles=deciles[feature_id]; feature_quintiles=quintiles[feature_id]; decile_long=equal_weight_tail(feature_deciles,9); decile_short=equal_weight_tail(feature_deciles,0); quintile_long=equal_weight_tail(feature_quintiles,4); quintile_short=equal_weight_tail(feature_quintiles,0)
    return {"decile_long_turnover":one_way_turnover(decile_long),"decile_short_turnover":one_way_turnover(decile_short),"decile_long_short_turnover":one_way_turnover(decile_long-decile_short),"quintile_long_short_turnover":one_way_turnover(quintile_long-quintile_short)}


def exact_rank_persistence(
    feature_values: np.ndarray,
    *,
    lag: int,
    minimum_symbols: int,
) -> np.ndarray:
    """
    feature_values: [date, security]

    For each date t, rerank the shared finite universe between t and t+lag.
    """
    if feature_values.ndim != 2:
        raise ValueError("feature_values must be [date, security]")
    if lag <= 0:
        raise ValueError("lag must be positive")

    dates = feature_values.shape[0]
    output = np.full(dates, np.nan, dtype=np.float64)
    for date in range(dates - lag):
        current = feature_values[date]
        future = feature_values[date + lag]
        paired = np.isfinite(current) & np.isfinite(future)
        if paired.sum() < minimum_symbols:
            continue
        current_rank = rankdata(current[paired], method="average")
        future_rank = rankdata(future[paired], method="average")
        if np.std(current_rank) == 0 or np.std(future_rank) == 0:
            continue
        output[date] = np.corrcoef(current_rank, future_rank)[0, 1]
    return output


def build_persistence_table(
    *,
    feature_values: np.ndarray,
    feature_names: list[str],
    deciles: np.ndarray,
    quintiles: np.ndarray,
    minimum_symbols: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for feature_id, feature_name in enumerate(feature_names):
        persistence_1 = exact_rank_persistence(feature_values[feature_id], lag=1, minimum_symbols=minimum_symbols)
        persistence_5 = exact_rank_persistence(feature_values[feature_id], lag=5, minimum_symbols=minimum_symbols)
        turnover = feature_tail_turnover(deciles, quintiles, feature_id)
        rows.append({
            "feature": feature_name,
            "mean_rank_persistence_1d": float(np.nanmean(persistence_1)),
            "mean_rank_persistence_5d": float(np.nanmean(persistence_5)),
            "mean_decile_long_turnover": float(np.nanmean(turnover["decile_long_turnover"])),
            "mean_decile_short_turnover": float(np.nanmean(turnover["decile_short_turnover"])),
            "mean_decile_long_short_turnover": float(np.nanmean(turnover["decile_long_short_turnover"])),
            "mean_quintile_long_short_turnover": float(np.nanmean(turnover["quintile_long_short_turnover"])),
        })
    return pd.DataFrame(rows)

def build_rank_bin_cache(feature_values,decision_eligible,security_ids,feature_names,*,minimum_decile_size=80,minimum_quintile_size=50,minimum_rank_ic_size=50,minimum_distinct_rank_ic=2,minimum_distinct_quintile=5,minimum_distinct_decile=10):
    if feature_values.ndim!=3: raise ValueError("feature_values must be [feature, date, security]")
    f_count,d_count,_=feature_values.shape; ranks=np.full(feature_values.shape,np.nan,np.float32); deciles=np.full(feature_values.shape,-1,np.int8); quintiles=np.full(feature_values.shape,-1,np.int8); counts=np.zeros((f_count,d_count),np.int16); distinct=np.zeros_like(counts); ties=np.full((f_count,d_count),np.nan,np.float32); persistence=[]
    for f in range(f_count):
        for d in range(d_count):
            row=feature_values[f,d]; valid=decision_eligible[d]&np.isfinite(row); count,number_distinct,tie=row_information(row,valid); counts[f,d]=count; distinct[f,d]=number_distinct; ties[f,d]=tie
            if count>=minimum_rank_ic_size and number_distinct>=minimum_distinct_rank_ic: ranks[f,d,valid]=((rankdata(row[valid],method="average")-.5)/count).astype(np.float32)
            if count>=minimum_decile_size and number_distinct>=minimum_distinct_decile: deciles[f,d]=deterministic_bins(np.where(valid,row,np.nan),security_ids,10)
            if count>=minimum_quintile_size and number_distinct>=minimum_distinct_quintile: quintiles[f,d]=deterministic_bins(np.where(valid,row,np.nan),security_ids,5)
        for lag in (1,5):
            exact = exact_rank_persistence(feature_values[f], lag=lag, minimum_symbols=3)
            for d, value in enumerate(exact):
                persistence.append({"feature":feature_names[f],"lag":lag,"date_id":d,"rank_spearman":value})
    return RankBinCache(list(feature_names),ranks,deciles,quintiles,counts,distinct,ties,pd.DataFrame(persistence))
