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
            for d in range(lag,d_count):
                valid=np.isfinite(ranks[f,d])&np.isfinite(ranks[f,d-lag]); persistence.append({"feature":feature_names[f],"lag":lag,"date_id":d,"rank_spearman":float(spearmanr(ranks[f,d,valid],ranks[f,d-lag,valid]).statistic) if valid.sum()>=3 else np.nan})
    return RankBinCache(list(feature_names),ranks,deciles,quintiles,counts,distinct,ties,pd.DataFrame(persistence))
