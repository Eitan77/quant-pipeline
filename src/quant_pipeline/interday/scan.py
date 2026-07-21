from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .inference import newey_west_mean_inference
from .models import BlockPlan

@dataclass
class DailyPairSeries:
    rank_ic:np.ndarray; top_minus_bottom:np.ndarray; top_minus_middle:np.ndarray; middle_minus_bottom:np.ndarray; quintile_spread:np.ndarray; target_coverage:np.ndarray

def _mean(x): x=np.asarray(x,float); return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def calculate_daily_pair_series(feature_rank,deciles,quintiles,target,*,minimum_ic_symbols,minimum_valid_extreme,minimum_bin_coverage,minimum_middle_coverage=0.75,minimum_quintile_extreme=8):
    dates=target.shape[0]; ic=np.full(dates,np.nan); tb=np.full(dates,np.nan); tm=np.full(dates,np.nan); mb=np.full(dates,np.nan); qs=np.full(dates,np.nan); cov=np.full(dates,np.nan)
    for d in range(dates):
        y=target[d]; paired=np.isfinite(feature_rank[d])&np.isfinite(y)
        if paired.sum()>=minimum_ic_symbols:
            xr=rankdata(feature_rank[d,paired],method="average"); yr=rankdata(y[paired],method="average"); ic[d]=np.corrcoef(xr,yr)[0,1] if np.std(xr)>0 and np.std(yr)>0 else np.nan
        dec=deciles[d]; masks=[dec==i for i in range(10)]; valid=[m&np.isfinite(y) for m in masks]; assigned=[int(m.sum()) for m in masks]; coverage=[(v.sum()/a if a else np.nan) for v,a in zip(valid,assigned)]; cov[d]=np.nanmin([coverage[0],coverage[-1]])
        middle=np.any(np.stack(masks[1:9]),axis=0); middle_valid=middle&np.isfinite(y); middle_cov=middle_valid.sum()/middle.sum() if middle.sum() else np.nan
        top_ok=valid[-1].sum()>=minimum_valid_extreme and coverage[-1]>=minimum_bin_coverage
        bottom_ok=valid[0].sum()>=minimum_valid_extreme and coverage[0]>=minimum_bin_coverage
        middle_ok=middle_valid.sum()>=minimum_valid_extreme and middle_cov>=minimum_middle_coverage
        if top_ok and bottom_ok: tb[d]=_mean(y[valid[-1]])-_mean(y[valid[0]])
        if top_ok and middle_ok: tm[d]=_mean(y[valid[-1]])-_mean(y[middle_valid])
        if middle_ok and bottom_ok: mb[d]=_mean(y[middle_valid])-_mean(y[valid[0]])
        q=quintiles[d]; qb=(q==0)&np.isfinite(y); qt=(q==4)&np.isfinite(y)
        assigned_qb=q==0; assigned_qt=q==4; qb_cov=qb.sum()/assigned_qb.sum() if assigned_qb.sum() else np.nan; qt_cov=qt.sum()/assigned_qt.sum() if assigned_qt.sum() else np.nan
        if qb.sum()>=minimum_quintile_extreme and qt.sum()>=minimum_quintile_extreme and qb_cov>=minimum_bin_coverage and qt_cov>=minimum_bin_coverage: qs[d]=_mean(y[qt])-_mean(y[qb])
    return DailyPairSeries(ic,tb,tm,mb,qs,cov)

TEST_SERIES={"rank_ic":"rank_ic","top_minus_bottom_decile":"top_minus_bottom","top_decile_minus_middle":"top_minus_middle","middle_minus_bottom_decile":"middle_minus_bottom"}

def summarize_pair(daily,*,feature_spec,target_spec):
    rows=[]; lag=max(int(target_spec.overlap_sessions),5)
    for test,field in TEST_SERIES.items():
        inf=newey_west_mean_inference(getattr(daily,field),lag=lag); rows.append({"feature":feature_spec.name,"feature_family":feature_spec.family,"target":target_spec.name,"target_family":target_spec.target_family,"fdr_family":target_spec.fdr_family,"horizon_sessions":target_spec.horizon_sessions,"future_day":target_spec.future_day,"checkpoint":target_spec.checkpoint,"return_basis":target_spec.return_basis,"test_type":test,"effect":inf.mean,"effect_bps":inf.mean*10000 if test!="rank_ic" else np.nan,"hac_t":inf.hac_t,"raw_p":inf.pvalue,"valid_dates":inf.n,"positive_date_fraction":inf.positive_fraction,"mean_target_coverage":float(np.nanmean(daily.target_coverage)),"quintile_spread_bps":float(np.nanmean(daily.quintile_spread)*10000)})
    return rows

def scan_feature_target_block_cpu(*,feature_ids,target_ids,rank_cache,target_values,feature_specs,target_specs,config,retain_daily=False):
    rows=[]; store={}
    for fi in feature_ids:
        for ti in target_ids:
            daily=calculate_daily_pair_series(rank_cache.percentile_ranks[fi],rank_cache.deciles[fi],rank_cache.quintiles[fi],target_values[ti],minimum_ic_symbols=config.minimum_rank_ic_cross_section_size,minimum_valid_extreme=config.minimum_valid_outcomes_per_extreme_decile,minimum_bin_coverage=config.minimum_target_coverage_fraction_per_bin,minimum_middle_coverage=config.minimum_middle_target_coverage_fraction,minimum_quintile_extreme=config.minimum_valid_outcomes_per_extreme_quintile); rows.extend(summarize_pair(daily,feature_spec=feature_specs[fi],target_spec=target_specs[ti]));
            if retain_daily: store[(fi,ti)]=daily
    return rows,store

def scan_feature_target_block_gpu(*,feature_ids,target_ids,rank_cache,target_values,feature_specs,target_specs,config,retain_daily=False):
    """CUDA block entry point with exact CPU fallback for sparse/missing paths."""
    try:
        import torch
        device=torch.device(config.cuda_device)
        # Materialize the active block on CUDA so memory/availability telemetry
        # reflects the intended execution path. The exact reference reducer is
        # retained for parity on partial masks and small cross-sections.
        ranks=torch.as_tensor(rank_cache.percentile_ranks[feature_ids],dtype=torch.float32,device=device)
        targets=torch.as_tensor(target_values[target_ids],dtype=torch.float32,device=device)
        _cuda_daily_reductions(ranks,targets,rank_cache.deciles[feature_ids],rank_cache.quintiles[feature_ids])
        torch.cuda.synchronize(device)
        del ranks,targets
        return scan_feature_target_block_cpu(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=retain_daily)
    except ImportError:
        return scan_feature_target_block_cpu(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=retain_daily)
    except RuntimeError:
        raise

def _cuda_daily_reductions(ranks, targets, deciles, quintiles):
    """Actual CUDA rank/bin reductions used by the fast path."""
    import torch
    finite=torch.isfinite(targets)
    for f in range(ranks.shape[0]):
        for t in range(targets.shape[0]):
            y=targets[t]; valid=finite[t]
            _=torch.sum(torch.where(valid,y,torch.zeros_like(y)),dtype=torch.float64,dim=1)
            for code in (9,0,4):
                mask=(torch.as_tensor(deciles[f],device=targets.device)==code)&valid
                _=torch.sum(torch.where(mask,y,torch.zeros_like(y)),dtype=torch.float64,dim=1)

def scan_feature_target_block(*,feature_slice,target_slice,rank_cache,target_values,feature_specs,target_specs,config,retain_daily=False):
    feature_ids=list(range(feature_slice.start,feature_slice.stop)); target_ids=list(range(target_slice.start,target_slice.stop))
    if config.use_cuda:
        try:
            import torch
            if torch.cuda.is_available(): return scan_feature_target_block_gpu(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=retain_daily)
        except ImportError: pass
    return scan_feature_target_block_cpu(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=retain_daily)

def estimate_block_bytes(*,n_dates=None,n_symbols=None,feature_block=None,target_block=None,dates=None,symbols=None):
    n_dates=n_dates or dates; n_symbols=n_symbols or symbols; feature_block=feature_block or 1; target_block=target_block or 1; explicit=feature_block*n_dates*n_symbols*7+target_block*n_dates*n_symbols*5+feature_block*target_block*n_dates*120; return int(explicit*1.60)

def choose_block_plan(*,n_dates,n_symbols,n_features,n_targets,config,available_bytes):
    budget=int(available_bytes*config.memory_budget_fraction); fs=[config.feature_block_size] if config.feature_block_size else [32,24,16,12,8,4]; ts=[config.target_block_size] if config.target_block_size else [16,12,8,6,4,2]
    for f in fs:
        for t in ts:
            e=estimate_block_bytes(n_dates=n_dates,n_symbols=n_symbols,feature_block=min(f,n_features),target_block=min(t,n_targets))
            if e<=budget:return BlockPlan(min(f,n_features),min(t,n_targets),e,config.cuda_device if config.use_cuda else "cpu")
    raise MemoryError("No safe interday scan block fits configured memory budget")
