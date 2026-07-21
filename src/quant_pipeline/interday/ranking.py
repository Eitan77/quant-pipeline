from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import rankdata

@dataclass
class RankBinCache:
    feature_names:list[str]; percentile_ranks:np.ndarray; deciles:np.ndarray; quintiles:np.ndarray; valid_counts:np.ndarray

def deterministic_bins(values,security_ids,bins):
    out=np.full(len(values),-1,np.int8); valid=np.isfinite(values); ix=np.flatnonzero(valid)
    if len(ix)<bins: return out
    order=np.lexsort((np.asarray(security_ids,dtype=np.int64)[ix],values[ix])); ordered=ix[order]; base,rem=divmod(len(ordered),bins); start=0
    for b in range(bins): out[ordered[start:start+base+(b<rem)]]=b; start+=base+(b<rem)
    return out

def build_rank_bin_cache(feature_values,decision_eligible,security_ids,feature_names,*,minimum_decile_size=80,minimum_quintile_size=50):
    dates,symbols,features=feature_values.shape; ranks=np.full(feature_values.shape,np.nan,np.float32); dec=np.full(feature_values.shape,-1,np.int8); quint=np.full(feature_values.shape,-1,np.int8); counts=np.zeros((dates,features),np.int16)
    for f in range(features):
        for d in range(dates):
            row=feature_values[d,:,f]; valid=decision_eligible[d]&np.isfinite(row); n=int(valid.sum()); counts[d,f]=n
            if not n: continue
            ranks[d,valid,f]=((rankdata(row[valid],method="average")-.5)/n).astype(np.float32)
            if n>=minimum_decile_size: dec[d,:,f]=deterministic_bins(np.where(valid,row,np.nan),security_ids,10)
            if n>=minimum_quintile_size: quint[d,:,f]=deterministic_bins(np.where(valid,row,np.nan),security_ids,5)
    return RankBinCache(feature_names,ranks,dec,quint,counts)
