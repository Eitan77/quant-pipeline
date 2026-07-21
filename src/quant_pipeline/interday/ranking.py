from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import rankdata

@dataclass
class RankBinCache:
    feature_names:list[str]; percentile_ranks:np.ndarray; deciles:np.ndarray; quintiles:np.ndarray; valid_counts:np.ndarray; persistence:dict[str,np.ndarray]|None=None

def deterministic_bins(values,security_ids,bins):
    out=np.full(len(values),-1,np.int8); valid=np.isfinite(values); ix=np.flatnonzero(valid)
    if len(ix)<bins: return out
    order=np.lexsort((np.asarray(security_ids,dtype=np.int64)[ix],values[ix])); ordered=ix[order]; base,rem=divmod(len(ordered),bins); start=0
    for b in range(bins): out[ordered[start:start+base+(b<rem)]]=b; start+=base+(b<rem)
    return out

def build_rank_bin_cache(feature_values,decision_eligible,security_ids,feature_names,*,minimum_decile_size=80,minimum_quintile_size=50):
    if feature_values.ndim!=3: raise ValueError("feature_values must be [features, dates, symbols]")
    features,dates,symbols=feature_values.shape; ranks=np.full(feature_values.shape,np.nan,np.float32); dec=np.full(feature_values.shape,-1,np.int8); quint=np.full(feature_values.shape,-1,np.int8); counts=np.zeros((features,dates),np.int16)
    for f in range(features):
        for d in range(dates):
            row=feature_values[f,d]; valid=decision_eligible[d]&np.isfinite(row); n=int(valid.sum()); counts[f,d]=n
            if not n: continue
            ranks[f,d,valid]=((rankdata(row[valid],method="average")-.5)/n).astype(np.float32)
            if n>=minimum_decile_size: dec[f,d]=deterministic_bins(np.where(valid,row,np.nan),security_ids,10)
            if n>=minimum_quintile_size: quint[f,d]=deterministic_bins(np.where(valid,row,np.nan),security_ids,5)
    persistence={"rank_autocorrelation_1d":np.full(features,np.nan),"rank_autocorrelation_5d":np.full(features,np.nan),"top_decile_retention_1d":np.full(features,np.nan),"bottom_decile_retention_1d":np.full(features,np.nan),"top_bottom_turnover":np.full(features,np.nan),"quintile_turnover":np.full(features,np.nan)}
    for f in range(features):
        for lag,key in ((1,"rank_autocorrelation_1d"),(5,"rank_autocorrelation_5d")):
            a=ranks[f,:-lag].ravel(); b=ranks[f,lag:].ravel(); valid=np.isfinite(a)&np.isfinite(b); persistence[key][f]=float(np.corrcoef(a[valid],b[valid])[0,1]) if valid.sum()>2 else np.nan
        for bins,key in ((dec,f"top_decile_retention_1d"),(dec,f"bottom_decile_retention_1d")):
            a=bins[f,:-1]; b=bins[f,1:]; target=9 if key.startswith("top") else 0; den=(a==target).sum(); persistence[key][f]=float(((a==target)&(b==target)).sum()/den) if den else np.nan
        for bins,key in ((dec,"top_bottom_turnover"),(quint,"quintile_turnover")):
            if len(bins)>1:
                weights=np.zeros_like(bins,dtype=float); weights[bins>=0]=1/np.maximum((bins>=0).sum(axis=1,keepdims=True),1); persistence[key][f]=float(np.nanmean(.5*np.abs(weights[1:]-weights[:-1]).sum(axis=1)))
    return RankBinCache(feature_names,ranks,dec,quint,counts,persistence)
