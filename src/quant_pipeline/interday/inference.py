from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

@dataclass(frozen=True)
class MeanInference:
    mean:float; median:float; std:float; hac_standard_error:float; hac_t:float; pvalue:float; positive_fraction:float; n:int

def newey_west_mean_inference(values,*,lag):
    x=np.asarray(values,float); finite=np.isfinite(x); n=int(finite.sum())
    if n<max(20,lag+5): return MeanInference(np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,n)
    mean=float(x[finite].mean()); z=np.where(finite,x-mean,np.nan); l=min(lag,len(x)-1); lr=float(np.nanmean(z*z))
    for k in range(1,l+1):
        pair=np.isfinite(z[k:])&np.isfinite(z[:-k])
        if pair.any(): lr += 2*(1-k/(lag+1))*float(np.mean(z[k:][pair]*z[:-k][pair]))
    se=float(np.sqrt(max(lr/n,0))); t=mean/se if se>0 else np.nan; p=float(2*norm.sf(abs(t))) if np.isfinite(t) else np.nan
    return MeanInference(mean,float(np.nanmedian(x)),float(np.nanstd(x,ddof=1)),se,float(t),p,float(np.mean(x[finite]>0)),n)

def benjamini_hochberg(values):
    import pandas as pd
    x=pd.to_numeric(values,errors="coerce"); valid=x.notna(); p=x[valid].to_numpy(float)
    if np.any((p<0)|(p>1)|~np.isfinite(p)): raise ValueError("BH requires p-values in [0,1]")
    out=pd.Series(np.nan,index=x.index,dtype=float)
    if not len(p): return out
    order=np.argsort(p); adj=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; out.loc[x[valid].index[order]]=np.minimum(adj,1); return out
