from __future__ import annotations
import numpy as np
import pandas as pd

def exact_path_diagnostics(path: pd.DataFrame, *, entry_column="entry", exit_column="exit") -> dict:
    if path.empty:return {"status":"insufficient_data"}
    entry=path[entry_column].to_numpy(float); exits=path[exit_column].to_numpy(float); valid=np.isfinite(entry)&np.isfinite(exits)&(entry>0); ret=np.where(valid,exits/entry-1,np.nan); return {"n":int(np.isfinite(ret).sum()),"mean_return":float(np.nanmean(ret)),"mfe":float(np.nanmax(ret)) if np.isfinite(ret).any() else np.nan,"mae":float(np.nanmin(ret)) if np.isfinite(ret).any() else np.nan,"status":"built"}

def cost_sensitivity(returns, scenarios=(1.,3.,5.)):
    x=np.asarray(returns,float); return pd.DataFrame([{"cost_bps_per_side":float(c),"mean_net_return":float(np.nanmean(x)-2*c/10000)} for c in scenarios])
