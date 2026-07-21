from __future__ import annotations
import numpy as np
import pandas as pd

def ordered_neighbor_indices(length,index): return ([index-1] if index>0 else [])+([index+1] if index+1<length else [])

def neighbor_retention(horizons,effects,peak_index):
    peak=effects[peak_index]
    if not np.isfinite(peak) or peak==0:return np.nan
    valid=[i for i in ordered_neighbor_indices(len(horizons),peak_index) if np.isfinite(effects[i])]
    return max((np.sign(peak)*effects[i] for i in valid),default=np.nan)/abs(peak) if valid else np.nan

def chronological_checkpoint_order(targets):
    def key(spec): return (spec.future_day, -1 if spec.checkpoint=="open5" else 999 if spec.checkpoint=="close5" else int(spec.checkpoint.replace(":","")))
    return sorted(targets,key=key)

def build_horizon_profiles(scan_results: pd.DataFrame) -> pd.DataFrame:
    if scan_results.empty:return pd.DataFrame()
    rows=[]
    for (feature,test,basis),g in scan_results.groupby(["feature","test_type","return_basis"],dropna=False):
        g=g.loc[g.checkpoint.eq("close5")].sort_values("horizon_sessions"); effects=g.effect.to_numpy(float); horizons=g.horizon_sessions.to_numpy(int)
        if not len(g):continue
        peak=int(np.nanargmax(np.abs(effects))) if np.isfinite(effects).any() else 0; neighbors=ordered_neighbor_indices(len(effects),peak); valid_neighbors=[i for i in neighbors if np.isfinite(effects[i])]; retention=neighbor_retention(horizons,effects,peak); rows.append({"feature":feature,"test_type":test,"return_basis":basis,"peak_horizon":int(horizons[peak]),"peak_effect":float(effects[peak]),"neighbor_retention":retention,"best_neighbor_retention":retention,"same_sign_neighbor_fraction":float(np.mean(np.sign(effects[np.isfinite(effects)])==np.sign(effects[peak]))) if np.isfinite(effects).any() else np.nan,"isolated_spike":not bool(valid_neighbors),"effect_auc":float(np.trapezoid(np.nan_to_num(effects),horizons)) if np.isfinite(effects).any() else np.nan,"shape_class":"monotonic" if np.all(np.diff(effects[np.isfinite(effects)])>=0) else "non_monotonic"})
    return pd.DataFrame(rows)

def build_checkpoint_profiles(scan_results: pd.DataFrame) -> pd.DataFrame:
    if scan_results.empty:return pd.DataFrame()
    rows=[]
    for keys,g in scan_results.groupby(["feature","test_type","return_basis"],dropna=False):
        order=g.copy(); order["checkpoint_order"]=[(int(x),-1 if c=="open5" else 999 if c=="close5" else int(c.replace(":",""))) for x,c in zip(order.future_day,order.checkpoint)]; order=order.sort_values("checkpoint_order"); e=order.effect.to_numpy(float); valid=np.isfinite(e)
        if not valid.any():continue
        p=int(np.nanargmax(np.abs(e))); neighbors=[i for i in (p-1,p+1) if 0<=i<len(e) and np.isfinite(e[i])]; sign=np.sign(e[p]); prev=sign*e[p-1]/abs(e[p]) if p>0 and np.isfinite(e[p-1]) else np.nan; nxt=sign*e[p+1]/abs(e[p]) if p+1<len(e) and np.isfinite(e[p+1]) else np.nan; best=np.nanmax([prev,nxt]) if np.isfinite([prev,nxt]).any() else np.nan; rows.append({"feature":keys[0],"test_type":keys[1],"return_basis":keys[2],"peak_endpoint":order.iloc[p].target,"peak_effect":e[p],"previous_endpoint_retention":prev,"next_endpoint_retention":nxt,"best_neighbor_retention":best,"same_sign_endpoint_fraction":float(np.mean(np.sign(e[valid])==sign)),"sign_changes":int(np.sum(np.diff(np.sign(e[valid]))!=0)),"isolated_spike":not bool(neighbors)})
    return pd.DataFrame(rows)
