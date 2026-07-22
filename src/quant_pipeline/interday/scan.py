from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .inference import newey_west_mean_inference
from .registry import feature_definition_hash, target_definition_hash
from .models import BlockPlan


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8).tobytes()).hexdigest()

@dataclass
class DailyPairSeries:
    rank_ic:np.ndarray; top_minus_bottom:np.ndarray; top_minus_middle:np.ndarray; middle_minus_bottom:np.ndarray; quintile_spread:np.ndarray; target_coverage:np.ndarray; ic_cross_section_size:np.ndarray|None=None; top_coverage:np.ndarray|None=None; bottom_coverage:np.ndarray|None=None; middle_coverage:np.ndarray|None=None; quintile_top_coverage:np.ndarray|None=None; quintile_bottom_coverage:np.ndarray|None=None

def _mean(x): x=np.asarray(x,float); return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def calculate_daily_pair_series(feature_rank,deciles,quintiles,target,*,minimum_ic_symbols,minimum_valid_extreme,minimum_bin_coverage,minimum_middle_coverage=0.75,minimum_quintile_extreme=8):
    dates=target.shape[0]; ic=np.full(dates,np.nan); tb=np.full(dates,np.nan); tm=np.full(dates,np.nan); mb=np.full(dates,np.nan); qs=np.full(dates,np.nan); cov=np.full(dates,np.nan); ic_n=np.zeros(dates,np.int16); top_cov=np.full(dates,np.nan); bottom_cov=np.full(dates,np.nan); middle_cov_arr=np.full(dates,np.nan); qtop_cov=np.full(dates,np.nan); qbottom_cov=np.full(dates,np.nan)
    for d in range(dates):
        y=target[d]; paired=np.isfinite(feature_rank[d])&np.isfinite(y)
        ic_n[d]=paired.sum(); distinct_x=len(np.unique(feature_rank[d,paired])); distinct_y=len(np.unique(y[paired]))
        if paired.sum()>=minimum_ic_symbols and distinct_x>=2 and distinct_y>=2:
            xr=rankdata(feature_rank[d,paired],method="average"); yr=rankdata(y[paired],method="average"); ic[d]=np.corrcoef(xr,yr)[0,1] if np.std(xr)>0 and np.std(yr)>0 else np.nan
        dec=deciles[d]; masks=[dec==i for i in range(10)]; valid=[m&np.isfinite(y) for m in masks]; assigned=[int(m.sum()) for m in masks]; coverage=[(v.sum()/a if a else np.nan) for v,a in zip(valid,assigned)]; cov[d]=np.nanmin([coverage[0],coverage[-1]])
        top_cov[d]=coverage[-1]; bottom_cov[d]=coverage[0]; middle=np.any(np.stack(masks[1:9]),axis=0); middle_valid=middle&np.isfinite(y); middle_cov=middle_valid.sum()/middle.sum() if middle.sum() else np.nan; middle_cov_arr[d]=middle_cov
        top_ok=valid[-1].sum()>=minimum_valid_extreme and coverage[-1]>=minimum_bin_coverage
        bottom_ok=valid[0].sum()>=minimum_valid_extreme and coverage[0]>=minimum_bin_coverage
        middle_ok=middle_valid.sum()>=minimum_valid_extreme and middle_cov>=minimum_middle_coverage
        if top_ok and bottom_ok: tb[d]=_mean(y[valid[-1]])-_mean(y[valid[0]])
        if top_ok and middle_ok: tm[d]=_mean(y[valid[-1]])-_mean(y[middle_valid])
        if middle_ok and bottom_ok: mb[d]=_mean(y[middle_valid])-_mean(y[valid[0]])
        q=quintiles[d]; qb=(q==0)&np.isfinite(y); qt=(q==4)&np.isfinite(y)
        assigned_qb=q==0; assigned_qt=q==4; qb_cov=qb.sum()/assigned_qb.sum() if assigned_qb.sum() else np.nan; qt_cov=qt.sum()/assigned_qt.sum() if assigned_qt.sum() else np.nan; qbottom_cov[d]=qb_cov; qtop_cov[d]=qt_cov
        if qb.sum()>=minimum_quintile_extreme and qt.sum()>=minimum_quintile_extreme and qb_cov>=minimum_bin_coverage and qt_cov>=minimum_bin_coverage: qs[d]=_mean(y[qt])-_mean(y[qb])
    return DailyPairSeries(ic,tb,tm,mb,qs,cov,ic_n,top_cov,bottom_cov,middle_cov_arr,qtop_cov,qbottom_cov)

TEST_SERIES={"rank_ic":"rank_ic","top_minus_bottom_decile":"top_minus_bottom","top_decile_minus_middle":"top_minus_middle","middle_minus_bottom_decile":"middle_minus_bottom"}

def summarize_pair(daily,*,feature_spec,target_spec,distinct_symbols):
    rows=[]; lag=max(int(target_spec.overlap_sessions),5)
    for test,field in TEST_SERIES.items():
        inf=newey_west_mean_inference(getattr(daily,field),lag=lag); rows.append({"feature":feature_spec.name,"feature_family":feature_spec.family,"feature_redundancy_group":feature_spec.redundancy_group,"target":target_spec.name,"canonical_target_id":target_spec.canonical_target_id,"target_family":target_spec.target_family,"fdr_family":target_spec.fdr_family,"horizon_sessions":target_spec.horizon_sessions,"future_day":target_spec.future_day,"checkpoint":target_spec.checkpoint,"endpoint_order":target_spec.endpoint_order,"return_basis":target_spec.return_basis,"is_executable":target_spec.is_executable,"diagnostic_only":target_spec.diagnostic_only,"test_type":test,"effect":inf.mean,"effect_bps":inf.mean*10000 if test!="rank_ic" else np.nan,"hac_standard_error":inf.hac_standard_error,"hac_t":inf.hac_t,"raw_p":inf.pvalue,"valid_dates":inf.n,"distinct_symbols":distinct_symbols,"mean_ic_cross_section_size":float(np.nanmean(daily.ic_cross_section_size)) if daily.ic_cross_section_size is not None else np.nan,"minimum_ic_cross_section_size":np.nan,"mean_top_coverage":float(np.nanmean(daily.top_coverage)) if daily.top_coverage is not None else np.nan,"mean_bottom_coverage":float(np.nanmean(daily.bottom_coverage)) if daily.bottom_coverage is not None else np.nan,"mean_middle_coverage":float(np.nanmean(daily.middle_coverage)) if daily.middle_coverage is not None else np.nan,"mean_quintile_top_coverage":float(np.nanmean(daily.quintile_top_coverage)) if daily.quintile_top_coverage is not None else np.nan,"mean_quintile_bottom_coverage":float(np.nanmean(daily.quintile_bottom_coverage)) if daily.quintile_bottom_coverage is not None else np.nan,"positive_date_fraction":inf.positive_fraction,"quintile_spread_bps":float(np.nanmean(daily.quintile_spread)*10000),"feature_definition_hash":feature_definition_hash(feature_spec),"target_definition_hash":target_definition_hash(target_spec),"backend":"cpu_python_reference"})
    for row in rows:
        row["backend"] = "cpu_python_reference"
    return rows


def pair_distinct_symbol_count(
    feature_rank: np.ndarray,
    target: np.ndarray,
) -> int:
    """Count securities contributing at least one valid pair observation."""
    paired = np.isfinite(feature_rank) & np.isfinite(target)
    return int(np.any(paired, axis=0).sum())

def scan_feature_target_block_cpu(*,feature_ids,target_ids,rank_cache,target_values,feature_specs,target_specs,config,retain_daily=False):
    rows=[]; store={}
    for fi in feature_ids:
        for ti in target_ids:
            daily=calculate_daily_pair_series(rank_cache.percentile_ranks[fi],rank_cache.deciles[fi],rank_cache.quintiles[fi],target_values[ti],minimum_ic_symbols=config.minimum_rank_ic_cross_section_size,minimum_valid_extreme=config.minimum_valid_outcomes_per_extreme_decile,minimum_bin_coverage=config.minimum_target_coverage_fraction_per_bin,minimum_middle_coverage=config.minimum_middle_target_coverage_fraction,minimum_quintile_extreme=config.minimum_valid_outcomes_per_extreme_quintile); rows.extend(summarize_pair(daily,feature_spec=feature_specs[fi],target_spec=target_specs[ti],distinct_symbols=pair_distinct_symbol_count(rank_cache.percentile_ranks[fi],target_values[ti])));
            if retain_daily: store[(fi,ti)]=daily
    return rows,store


def scan_part_path(root: Path, *, feature_start: int, feature_stop: int, target_start: int, target_stop: int) -> Path:
    return root / "scan_parts" / f"f{feature_start:04d}_f{feature_stop - 1:04d}_t{target_start:04d}_t{target_stop - 1:04d}.parquet"


def scan_part_manifest_path(path: Path) -> Path:
    return path.with_suffix(".json")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_scan_part(*, path: Path, frame: pd.DataFrame, fingerprint: str, schema_version: str, feature_start: int, feature_stop: int, target_start: int, target_stop: int, backend: str = "cpu_python_reference", device_diagnostics=None, upstream_digests: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    manifest = {
        "fingerprint": fingerprint,
        "schema_version": schema_version,
        "feature_start": feature_start,
        "feature_stop": feature_stop,
        "target_start": target_start,
        "target_stop": target_stop,
        "row_count": int(len(frame)),
        "file_size": int(path.stat().st_size),
        "file_sha256": file_sha256(path),
        "backend": backend,
        "device_name": getattr(device_diagnostics, "device_name", "cpu") if device_diagnostics is not None else "cpu",
        "peak_allocated_bytes": int(getattr(device_diagnostics, "peak_allocated_bytes", 0)) if device_diagnostics is not None else 0,
        "peak_reserved_bytes": int(getattr(device_diagnostics, "peak_reserved_bytes", 0)) if device_diagnostics is not None else 0,
    }
    if upstream_digests:
        manifest.update(upstream_digests)
    write_json_atomic(scan_part_manifest_path(path), manifest)


def validate_scan_part(*, path: Path, fingerprint: str, schema_version: str, feature_start: int, feature_stop: int, target_start: int, target_stop: int, backend: str | None = None, upstream_digests: dict | None = None) -> bool:
    manifest_path = scan_part_manifest_path(path)
    if not path.exists() or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"fingerprint": fingerprint, "schema_version": schema_version, "feature_start": feature_start, "feature_stop": feature_stop, "target_start": target_start, "target_stop": target_stop}
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    if backend is not None and manifest.get("backend") != backend:
        return False
    if upstream_digests and any(manifest.get(key) != value for key, value in upstream_digests.items()):
        return False
    if manifest.get("file_size") != path.stat().st_size or manifest.get("file_sha256") != file_sha256(path):
        return False
    frame = pd.read_parquet(path)
    return manifest.get("row_count") == len(frame)


def run_or_resume_scan_blocks(*, root: Path, rank_cache, target_values: np.ndarray, feature_specs, target_specs, config, plan: BlockPlan, fingerprint: str) -> pd.DataFrame:
    part_paths: list[Path] = []
    for feature_start in range(0, len(feature_specs), plan.feature_block_size):
        feature_stop = min(feature_start + plan.feature_block_size, len(feature_specs))
        for target_start in range(0, len(target_specs), plan.target_block_size):
            target_stop = min(target_start + plan.target_block_size, len(target_specs))
            path = scan_part_path(root, feature_start=feature_start, feature_stop=feature_stop, target_start=target_start, target_stop=target_stop)
            part_paths.append(path)
            upstream_digests = {"target_matrix_sha256": array_sha256(target_values), "rank_matrix_sha256": array_sha256(rank_cache.percentile_ranks)}
            if validate_scan_part(path=path, fingerprint=fingerprint, schema_version=config.scan_schema_version, feature_start=feature_start, feature_stop=feature_stop, target_start=target_start, target_stop=target_stop, backend="cuda_exact_cross_sectional" if config.use_cuda else "cpu_python_reference", upstream_digests=upstream_digests):
                continue
            rows, retained_daily, device_diagnostics = scan_feature_target_block(feature_slice=slice(feature_start, feature_stop), target_slice=slice(target_start, target_stop), rank_cache=rank_cache, target_values=target_values, feature_specs=feature_specs, target_specs=target_specs, config=config, retain_daily=False)
            backend = rows[0]["backend"] if rows else (device_diagnostics.backend if device_diagnostics is not None else "cpu_python_reference")
            write_scan_part(path=path, frame=pd.DataFrame(rows), fingerprint=fingerprint, schema_version=config.scan_schema_version, feature_start=feature_start, feature_stop=feature_stop, target_start=target_start, target_stop=target_stop, backend=backend, device_diagnostics=device_diagnostics, upstream_digests=upstream_digests)
    frames = [pd.read_parquet(path) for path in part_paths]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    key = ["feature", "target", "test_type"]
    if not result.empty and result.duplicated(key).any():
        raise ValueError("Merged scan parts contain duplicate hypotheses")
    return result

def scan_feature_target_block(*,feature_slice,target_slice,rank_cache,target_values,feature_specs,target_specs,config,retain_daily=False):
    feature_ids=list(range(feature_slice.start,feature_slice.stop)); target_ids=list(range(target_slice.start,target_slice.stop))
    if config.use_cuda:
        from .gpu_scan import scan_feature_target_block_cuda
        return scan_feature_target_block_cuda(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=retain_daily)
    if not config.allow_cpu_reference_smoke:
        raise RuntimeError("CPU reference scanner is disabled for this run")
    rows, retained = scan_feature_target_block_cpu(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=retain_daily)
    return rows, retained, None


def assert_gpu_parity(*, rank_cache, target_values, feature_specs, target_specs, config) -> None:
    feature_ids=list(range(min(config.gpu_parity_feature_pairs,len(feature_specs))))
    target_ids=list(range(min(config.gpu_parity_target_pairs,len(target_specs))))
    cpu_rows, cpu_daily = scan_feature_target_block_cpu(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=True)
    from .gpu_scan import scan_feature_target_block_cuda
    gpu_rows, gpu_daily, _ = scan_feature_target_block_cuda(feature_ids=feature_ids,target_ids=target_ids,rank_cache=rank_cache,target_values=target_values,feature_specs=feature_specs,target_specs=target_specs,config=config,retain_daily=True)
    if len(cpu_rows) != len(gpu_rows):
        raise AssertionError("CPU/GPU parity row counts differ")
    for key in cpu_daily:
        left, right = cpu_daily[key], gpu_daily[key]
        for field in ("rank_ic","top_minus_bottom","top_minus_middle","middle_minus_bottom","quintile_spread","top_coverage","bottom_coverage","middle_coverage"):
            np.testing.assert_allclose(getattr(left,field),getattr(right,field),equal_nan=True,atol=config.gpu_parity_atol,rtol=config.gpu_parity_rtol)

def estimate_block_bytes(*,dates=None,securities=None,feature_block=None,target_block=None,n_dates=None,n_symbols=None):
    dates=dates if dates is not None else n_dates; securities=securities if securities is not None else n_symbols; feature_block=feature_block or 1; target_block=target_block or 1
    feature_inputs=feature_block*dates*securities*(4+1+1+1); target_inputs=target_block*dates*securities*(4+1); daily_outputs=feature_block*target_block*dates*(10*8); temporaries=feature_block*target_block*dates*(12*8); return int((feature_inputs+target_inputs+daily_outputs+temporaries)*1.60)


def estimate_gpu_block_bytes(*, dates: int, securities: int, target_block: int) -> int:
    target=target_block*dates*securities*4; pair_mask=target_block*dates*securities; rank_values=2*target_block*dates*securities*8; sort_order=2*target_block*dates*securities*8; group_workspace=4*target_block*dates*securities*8; daily_outputs=14*target_block*dates*8; feature_inputs=dates*securities*(4+1+1)
    return int(1.35*(target+pair_mask+rank_values+sort_order+group_workspace+daily_outputs+feature_inputs))

def choose_block_plan(*,n_dates,n_symbols,n_features,n_targets,config,available_bytes):
    budget=int(available_bytes*config.memory_budget_fraction)
    fs=[config.gpu_feature_block_size or config.feature_block_size] if (config.use_cuda and (config.gpu_feature_block_size or config.feature_block_size)) else ([config.feature_block_size] if config.feature_block_size else [32,24,16,12,8,4])
    ts=[config.gpu_target_block_size or config.target_block_size] if (config.use_cuda and (config.gpu_target_block_size or config.target_block_size)) else ([config.target_block_size] if config.target_block_size else [16,12,8,6,4,2])
    for f in fs:
        for t in ts:
            e=estimate_gpu_block_bytes(dates=n_dates,securities=n_symbols,target_block=min(t,n_targets)) if config.use_cuda else estimate_block_bytes(n_dates=n_dates,n_symbols=n_symbols,feature_block=min(f,n_features),target_block=min(t,n_targets))
            if e<=budget:return BlockPlan(min(f,n_features),min(t,n_targets),e,config.cuda_device if config.use_cuda else "cpu")
    raise MemoryError("No safe interday scan block fits configured memory budget")
