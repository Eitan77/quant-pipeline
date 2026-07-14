from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScanConfig
from .features import build_features
from .parallel_features import build_parallel_block, is_symbol_local
from .registry import feature_registry, registry_frame, target_registry
from .scanner import benjamini_hochberg,scan
from .bulk_scan import cuda_screen, finalize_screen
from .report import write_reports
from .table import add_targets, filter_decision_rows, load_canonical_bars, source_provenance, validate_point_in_time
from .fingerprint import enforce_fingerprint,run_fingerprint


def execute(config: ScanConfig) -> Path:
    run_id=config.experiment_id
    root=Path(config.output_root)/run_id; root.mkdir(parents=True,exist_ok=True)
    sector=bool(config.sector_map_path); requested=feature_registry(config.lookbacks,sector,config.opening_windows_minutes); targets=target_registry(config.target_horizons_minutes,config.primary_target_horizons_minutes)
    fingerprint=run_fingerprint(config,requested,targets,_git_revision()); enforce_fingerprint(root,fingerprint,config.resume)
    import gc
    import pandas as pd
    bars_cache=root/"canonical_bars.parquet"; bars=None
    if not bars_cache.exists():
        bars=load_canonical_bars(config); bars.to_parquet(bars_cache,index=False)
    checkpoint=root/"screen_checkpoint.csv"; journal=root/"screen_journal.csv"; results=pd.DataFrame(); built_names=set(); validation=None
    block_root=root/"blocks"; feature_root=block_root/"features"; target_root=block_root/"targets"; feature_root.mkdir(parents=True,exist_ok=True); target_root.mkdir(parents=True,exist_ok=True)
    eligible=[s for s in requested if s.status=="requested"]
    local_specs=[spec for spec in eligible if is_symbol_local(spec)]
    global_specs=[spec for spec in eligible if not is_symbol_local(spec)]
    chunks=[local_specs[i:i+config.feature_chunk_size] for i in range(0,len(local_specs),config.feature_chunk_size)]
    chunks += [global_specs[i:i+config.feature_chunk_size] for i in range(0,len(global_specs),config.feature_chunk_size)]
    raw_targets=[t for t in targets if t.classification=="raw"]
    target_batches=[]
    for start in range(0,len(raw_targets),config.target_chunk_size):
        raw_batch=raw_targets[start:start+config.target_chunk_size]; names={t.name for t in raw_batch}
        target_batches.append(raw_batch+[t for t in targets if t.classification=="benchmark_adjusted" and t.name.removesuffix("_benchmark_adjusted") in names])
    # Materialize each target block once. They are reused by every feature block.
    target_paths=[]
    for target_index,target_batch in enumerate(target_batches):
        path=target_root/f"target_{target_index:03d}.parquet"; target_paths.append(path)
        if not path.exists():
            if bars is None: bars=pd.read_parquet(bars_cache)
            target_frame=add_targets(bars,target_batch,config.benchmark_symbol)
            if validation is None:validation=validate_point_in_time(target_frame,target_batch,config.sealed_holdout_start)
            target_columns=["entry_ts","entry_open_raw",*[t.name for t in target_batch],*[f"exit_ts__{t.name}" for t in target_batch if t.classification=="raw"],*[f"exit_close_raw__{t.name}" for t in target_batch if t.classification=="raw"],*[f"actual_horizon_minutes__{t.name}" for t in target_batch if t.classification=="raw"]]
            target_frame[target_columns].to_parquet(path,index=False); del target_frame; gc.collect()
        (root/"progress.json").write_text(json.dumps({"stage":"target_cache","completed":target_index+1,"total":len(target_batches),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
    del bars; bars=None; gc.collect()
    # Symbol-local blocks are built by 16 independent processes. Features that
    # require the full cross-section remain full-universe calculations.
    feature_paths=[]; built_by_chunk=[]
    for feature_index,chunk in enumerate(chunks):
        digest=hashlib.sha1("\n".join(s.name for s in chunk).encode()).hexdigest()[:10]
        path=feature_root/f"feature_{feature_index:03d}_{digest}.parquet"; feature_paths.append(path)
        if path.exists():
            import pyarrow.parquet as pq
            columns=set(pq.ParquetFile(path).schema.names); built=[s for s in chunk if s.name in columns]
        elif all(is_symbol_local(spec) for spec in chunk):
            built=build_parallel_block(bars_cache,path,config,chunk,root/"progress.json")
        else:
            if bars is None:
                bars=pd.read_parquet(bars_cache)
                for column in ["open","high","low","close","vwap","volume"]:
                    if column in bars: bars[column]=pd.to_numeric(bars[column],downcast="float")
            feature_frame,built=build_features(bars,config,chunk)
            feature_frame.to_parquet(path,index=False); del feature_frame; gc.collect()
        built_by_chunk.append(built); built_names.update(s.name for s in built)
        (root/"progress.json").write_text(json.dumps({"stage":"feature_cache","completed":feature_index+1,"total":len(chunks),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
    del bars; gc.collect()
    resume_frames=[]
    if config.resume and checkpoint.exists(): resume_frames.append(pd.read_csv(checkpoint))
    if config.resume and journal.exists(): resume_frames.append(pd.read_csv(journal))
    if resume_frames:
        results=pd.concat(resume_frames,ignore_index=True).drop_duplicates(["feature","target"],keep="last")
    total_batches=len(chunks)*len(target_batches); completed_batches=0
    from concurrent.futures import ThreadPoolExecutor
    for feature_index,(feature_path,built) in enumerate(zip(feature_paths,built_by_chunk)):
        completed_pairs={(row.feature,row.target) for row in results.itertuples()}
        pending=[i for i,batch in enumerate(target_batches) if any((spec.name,t.name) not in completed_pairs for spec in built for t in batch)]
        completed_batches += len(target_batches)-len(pending)
        if not pending: continue
        feature_frame=pd.read_parquet(feature_path)
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            future=prefetch.submit(pd.read_parquet,target_paths[pending[0]]) if pending else None
            for position,target_index in enumerate(pending):
                target_batch=target_batches[target_index]; target_frame=future.result()
                future=prefetch.submit(pd.read_parquet,target_paths[pending[position+1]]) if position+1<len(pending) else None
                selection_mask=pd.to_datetime(feature_frame.session_date).le(pd.Timestamp(config.selection_end))&feature_frame.scan_eligible.fillna(False)&feature_frame.session_grid_eligible.fillna(False)
                if config.decision_times_et:
                    local=pd.to_datetime(feature_frame.decision_ts,utc=True).dt.tz_convert("America/New_York"); selection_mask&=local.dt.strftime("%H:%M").isin(config.decision_times_et)
                screen_features=feature_frame.loc[selection_mask].reset_index(drop=True); screen_targets=target_frame.loc[selection_mask].reset_index(drop=True)
                continuous=[s for s in built if s.classification!="categorical"]
                results=cuda_screen(screen_features,screen_targets,continuous,[t.name for t in target_batch],config,results,journal)
                categorical=[s for s in built if s.classification=="categorical"]
                if categorical:
                    combined=pd.concat([screen_features.reset_index(drop=True),screen_targets[[t.name for t in target_batch]].reset_index(drop=True)],axis=1)
                    cat_rows,_=scan(combined,categorical,[t.name for t in target_batch],config,None)
                    results=pd.concat([results,cat_rows],ignore_index=True).drop_duplicates(["feature","target"],keep="last")
                completed_batches+=1
                (root/"progress.json").write_text(json.dumps({"stage":"cuda_screen","completed_batches":completed_batches,"total_batches":total_batches,"completed_feature_chunks":feature_index,"total_feature_chunks":len(chunks),"completed_pairs":len(results),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
                del target_frame; gc.collect()
        del feature_frame; gc.collect()
    results=finalize_screen(results.drop_duplicates(["feature","target"],keep="last")); results.to_csv(checkpoint,index=False)
    if journal.exists(): journal.unlink()
    registry_frame(requested).to_csv(root/"feature_registry.csv",index=False); registry_frame(targets).to_csv(root/"target_registry.csv",index=False); results.to_csv(root/"master_results.csv",index=False)
    skipped=[s.name for s in requested if s.name not in built_names]
    def skip_reason(s):
        if s.unavailable_reason: return s.unavailable_reason
        if s.family=="opening" and s.lookback is not None and s.lookback<5: return "Opening window is shorter than configured 5-minute source bars"
        return "Not constructible from configured point-in-time inputs"
    pd.DataFrame([{"feature": s.name, "status": "built" if s.name not in skipped else "skipped", "reason": "" if s.name not in skipped else skip_reason(s)} for s in requested]).to_csv(root / "scan_coverage.csv", index=False)
    tier={t.name:t.tier for t in targets}; results["target_tier"]=results.target.map(tier); results.to_csv(root/"master_results.csv",index=False)
    primary=results.loc[results.target_tier.eq("primary")].copy(); primary["primary_global_fdr"]=benjamini_hochberg(primary.raw_p)
    primary["family_fdr"]=primary.groupby(["feature_family","target_family"],dropna=False).raw_p.transform(benjamini_hochberg)
    primary["candidate_cluster"]=primary.redundancy_group.fillna(primary.feature)+"__"+primary.target.map(_target_horizon_family)
    primary["cluster_fdr"]=primary.groupby("candidate_cluster",dropna=False).raw_p.transform(benjamini_hochberg)
    results=results.merge(primary[["feature","target","primary_global_fdr"]],on=["feature","target"],how="left"); results.to_csv(root/"master_results.csv",index=False)
    feature_path_by_name={spec.name:feature_paths[index] for index,chunk in enumerate(chunks) for spec in chunk}
    queue=_cluster_candidates(primary,feature_path_by_name,limit=250)
    top_specs=[s for s in requested if s.name in set(queue.feature)]
    detailed=[]; qtabs={}; spec_by_name={s.name:s for s in top_specs}
    target_path_by_name={spec.name:target_paths[index] for index,batch in enumerate(target_batches) for spec in batch}
    from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,ThreadPoolExecutor,wait
    from .exact_parallel import exact_pair
    from .hybrid_exact import gpu_dense_pair
    cpu_rows={}; gpu_rows={}; hints={(row.feature,row.target):float(row.spearman) for row in queue.itertuples()}; screen_fdr={(row.feature,row.target):float(row.primary_global_fdr) for row in queue.itertuples()}
    cpu_done=gpu_done=0
    with ProcessPoolExecutor(max_workers=config.exact_workers) as cpu_pool, ThreadPoolExecutor(max_workers=1) as gpu_pool:
        futures={}
        for row in queue.itertuples():
            key=(row.feature,row.target); feature_path=feature_path_by_name[row.feature]; target_path=target_path_by_name[row.target]
            futures[cpu_pool.submit(exact_pair,feature_path,target_path,spec_by_name[row.feature],row.target,config,hints[key])]=("cpu",key)
            futures[gpu_pool.submit(gpu_dense_pair,feature_path,target_path,row.feature,row.target,config.min_bin_observations,config.selection_end)]=("gpu",key)
        pending=set(futures)
        while pending:
            finished,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                kind,key=futures[future]
                if kind=="cpu":
                    row,table=future.result(); cpu_rows[key]=row; qtabs[key]=pd.DataFrame(table); cpu_done+=1
                else:
                    dense,table=future.result(); gpu_rows[key]=dense; gpu_done+=1
            status={"stage":"exact_diagnostics_hybrid","cpu_completed":cpu_done,"gpu_completed":gpu_done,"total":len(queue),"cpu_workers":config.exact_workers,"gpu_workers":1,"updated_at":datetime.now(timezone.utc).isoformat()}
            (root/"progress.json").write_text(json.dumps(status,indent=2),encoding="utf-8"); (root/"detailed_progress.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    for key,row in cpu_rows.items():
        if row is None or key not in gpu_rows: continue
        dense=gpu_rows[key]; hint=hints[key]
        if np.sign(dense["spearman"])!=np.sign(hint):
            for column in ["year_consistency","symbol_breadth","time_stability"]:
                if pd.notna(row.get(column)): row[column]=1-row[column]
        queued=queue.loc[queue.feature.eq(key[0])&queue.target.eq(key[1])].iloc[0]
        row.update(dense); row["screen_bh_fdr_p_global"]=screen_fdr[key]; row["family_fdr"]=queued.family_fdr; row["cluster_fdr"]=queued.cluster_fdr; row["candidate_cluster"]=queued.candidate_cluster; row["target_tier"]="primary"; detailed.append(row)
    detailed=pd.DataFrame(detailed)
    if not detailed.empty:
        from .scanner import benjamini_hochberg
        detailed["exact_selected_bh_fdr_p"]=detailed.groupby(["feature_family","target_family"],dropna=False).raw_p.transform(benjamini_hochberg)
        detailed["bh_fdr_p"]=detailed["screen_bh_fdr_p_global"]
        detailed=_classify_detailed_candidates(detailed)
        detailed["effect_size_score"]=detailed.top_bottom_spread.abs().rank(pct=True); detailed["global_fdr_score"]=(1-detailed.bh_fdr_p.fillna(1)).clip(0,1); detailed["monotonicity_score"]=detailed.monotonicity.abs().fillna(0); detailed["year_stability_score"]=detailed.year_consistency.fillna(0); detailed["symbol_breadth_score"]=detailed.symbol_breadth.fillna(0); detailed["outlier_robustness_score"]=(detailed.outlier_worst_signed_spread.fillna(0)>0).astype(float); detailed["internal_confirmation_score"]=detailed.internal_confirmation_direction_match.fillna(False).astype(float); detailed["redundancy_penalty"]=detailed.groupby("candidate_cluster").cumcount()*.05; detailed["complexity_penalty"]=detailed.feature.str.count("_")*.005
        detailed["anomaly_score"]=.2*detailed.effect_size_score+.2*detailed.global_fdr_score+.1*detailed.monotonicity_score+.1*detailed.year_stability_score+.1*detailed.symbol_breadth_score+.1*detailed.outlier_robustness_score+.2*detailed.internal_confirmation_score-detailed.redundancy_penalty-detailed.complexity_penalty
        detailed=detailed.sort_values("anomaly_score",ascending=False); detailed.to_csv(root/"detailed_candidates.csv",index=False)
        cluster_report=detailed.groupby("candidate_cluster",dropna=False).agg(pairs=("feature","size"),best_primary_fdr=("bh_fdr_p","min"),best_effect=("top_bottom_spread",lambda s:float(s.loc[s.abs().idxmax()])),confirmed=("status",lambda s:int(s.eq("confirmed_anomaly_candidate").any()))).reset_index()
        cluster_report.to_csv(root/"cluster_level_anomalies.csv",index=False)
    _write_coverage_reports(root,results,requested,feature_paths,built_by_chunk)
    write_reports(detailed if not detailed.empty else results.head(50), qtabs, root, None)
    from .gpu import CorrelationBackend
    manifest={"experiment_id":run_id,"executed_at":datetime.now(timezone.utc).isoformat(),"config":config.as_dict(),"fingerprint":fingerprint["sha256"],"source_provenance":source_provenance(config),"validation":validation,"requested_features":len(requested),"built_features":len(built_names),"skipped_features":skipped,"targets":[t.name for t in targets],"primary_targets":[t.name for t in targets if t.tier=="primary"],"selection_period":[config.start,config.selection_end],"internal_confirmation_period":[config.confirmation_start,config.discovery_end],"sealed_holdout_start":config.sealed_holdout_start,"multiple_testing":"Primary-target global, feature-family, candidate-cluster and exact-pair FDR are reported separately; exploratory horizons cannot promote candidates","statistical_error":"Date-clustered screen; exact pass adds two-way date/symbol clustering, session-block bootstrap, and HAC daily spread/IC inference","correlation_backend":CorrelationBackend(config.use_cuda,config.cuda_device).name,"code_version":_git_revision()}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (root/"progress.json").write_text(json.dumps({"stage":"complete","screened_pairs":len(results),"exact_candidates":len(detailed),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
    return root


def _classify_detailed_candidates(detailed):
    """Apply the global discovery gate before assigning robust labels."""
    result=detailed.copy()
    usable=~result.status.isin(["constant_feature","insufficient_data"])
    exact=result.raw_p.lt(.05)
    global_fdr=result.screen_bh_fdr_p_global.lt(.05)
    stable=result.year_consistency.ge(.6)&result.symbol_breadth.ge(.6)
    economic=result.get("top_bottom_spread",pd.Series(np.inf,index=result.index)).abs().ge(1/10000)
    if {"internal_confirmation_direction_match","walk_forward_positive_fold_pct","outlier_worst_signed_spread"}.issubset(result):confirmed=result.internal_confirmation_direction_match.fillna(False)&result.walk_forward_positive_fold_pct.ge(.6)&result.outlier_worst_signed_spread.gt(0)
    else:confirmed=pd.Series(False,index=result.index)
    result.loc[usable,"status"]="no_meaningful_relationship"
    result.loc[usable&exact&~stable,"status"]="interesting_but_unstable"
    result.loc[usable&exact&stable&~global_fdr,"status"]="interesting_not_global"
    result.loc[usable&global_fdr&~(exact&stable),"status"]="statistically_interesting"
    result.loc[usable&global_fdr&exact&stable&economic,"status"]="robust_anomaly_candidate"
    result.loc[usable&global_fdr&exact&stable&economic&confirmed,"status"]="confirmed_anomaly_candidate"
    return result


def _target_horizon_family(target:str)->str:
    import re
    if "eod" in target:return "long_intraday"
    match=re.search(r"_(\d+)m",target); horizon=int(match.group(1)) if match else 999
    if horizon<=15:return "very_short"
    if horizon<=45:return "short"
    if horizon<=120:return "medium"
    return "long_intraday"


def _cluster_candidates(primary,feature_paths:dict[str,Path]|None=None,limit:int=250):
    if primary.empty:return primary
    eligible=primary.loc[primary.raw_p.notna()].sort_values(["primary_global_fdr","anomaly_score"],ascending=[True,False])
    ranked=pd.concat([eligible.head(100),eligible.loc[eligible.primary_global_fdr.lt(.10)]]).drop_duplicates(["feature","target"]).copy()
    if ranked.empty:return ranked
    features=ranked.feature.drop_duplicates().tolist(); parent={name:name for name in features}
    def find(name):
        while parent[name]!=name:parent[name]=parent[parent[name]]; name=parent[name]
        return name
    def union(left,right):
        a,b=find(left),find(right)
        if a!=b:parent[b]=a
    metadata=ranked.drop_duplicates("feature").set_index("feature")
    for _,group in metadata.groupby("feature_family"):
        names=group.index.tolist()
        by_redundancy={}
        for name in names:by_redundancy.setdefault(group.loc[name,"redundancy_group"],[]).append(name)
        for related in by_redundancy.values():
            for name in related[1:]:union(related[0],name)
    if feature_paths:
        samples={}
        by_path={}
        for feature in features:
            if feature in feature_paths:by_path.setdefault(feature_paths[feature],[]).append(feature)
        for path,names in by_path.items():
            frame=pd.read_parquet(path,columns=names); step=max(1,len(frame)//20_000); frame=frame.iloc[::step]
            for name in names:samples[name]=frame[name].reset_index(drop=True)
        for _,group in metadata.groupby("feature_family"):
            names=[name for name in group.index if name in samples]
            if len(names)<2:continue
            correlation_matrix=pd.concat({name:samples[name] for name in names},axis=1).corr(method="spearman")
            for i,left in enumerate(names):
                for right in names[i+1:]:
                    correlation=correlation_matrix.loc[left,right]
                    if pd.notna(correlation) and abs(correlation)>=.85:union(left,right)
    ranked["feature_cluster"]=ranked.feature.map(find)
    response_sign=np.sign(ranked.monotonicity.fillna(ranked.spearman)).astype(int).astype(str)
    ranked["candidate_cluster"]=ranked.feature_cluster+"__"+ranked.target.map(_target_horizon_family)+"__response_"+response_sign
    ranked["cluster_fdr"]=ranked.groupby("candidate_cluster",dropna=False).raw_p.transform(benjamini_hochberg)
    representatives=[]
    for _,group in ranked.groupby("candidate_cluster",sort=False):
        best=group.head(1)
        simplest=group.assign(complexity=group.feature.str.count("_")+group.feature.str.extract(r"_(\d+)(?:m)?$",expand=False).fillna("0").astype(int)/1000).sort_values("complexity").head(1)
        neighbor=group.iloc[[min(1,len(group)-1)]]
        representatives.append(pd.concat([best,simplest,neighbor]).drop_duplicates(["feature","target"]))
    return pd.concat(representatives,ignore_index=True).drop_duplicates(["feature","target"]).head(limit)


def _write_coverage_reports(root,results,requested,feature_paths,built_by_chunk):
    built={s.name for chunk in built_by_chunk for s in chunk}; requested_names={s.name for s in requested}
    coverage=results.rename(columns={"n":"valid observations","sessions":"valid sessions","symbols":"valid symbols"}).copy()
    targets=results.target.drop_duplicates().tolist()
    missing_rows=[{"feature":spec.name,"target":target,"valid observations":0,"valid sessions":0,"valid symbols":0,"valid_decision_timestamps":0,"valid_years":0,"table_rows":np.nan,"raw_p":np.nan} for spec in requested if spec.name not in built for target in targets]
    if missing_rows:coverage=pd.concat([coverage,pd.DataFrame(missing_rows)],ignore_index=True)
    reasons={s.name:(s.unavailable_reason or "not constructible from configured point-in-time inputs") for s in requested}
    coverage["requested"]=coverage.feature.isin(requested_names); coverage["built"]=coverage.feature.isin(built); coverage["tested"]=coverage.raw_p.notna(); coverage["missing rate"]=1-coverage["valid observations"]/coverage.table_rows.replace(0,np.nan); coverage["skip reason"]=np.where(coverage.built,"",coverage.feature.map(reasons)); coverage["validation failures"]=""
    columns=["feature","target","requested","built","tested","valid observations","valid sessions","valid symbols","valid_decision_timestamps","valid_years","missing rate","skip reason","validation failures"]
    coverage[[c for c in columns if c in coverage]].to_csv(root/"coverage_report.csv",index=False)
    rows=[]
    for path,chunk in zip(feature_paths,built_by_chunk):
        columns=["decision_ts",*[s.name for s in chunk]]; frame=pd.read_parquet(path,columns=columns)
        for spec in chunk:
            values=pd.to_numeric(frame[spec.name],errors="coerce"); finite=values[np.isfinite(values)]
            first=frame.loc[values.notna(),"decision_ts"].min() if values.notna().any() else None
            rows.append({"feature name":spec.name,"feature family":spec.family,"built status":"built","availability start":str(first) if first is not None else None,"missing rate":float(values.isna().mean()),"infinite count":int(np.isinf(values).sum()),"minimum":float(finite.min()) if len(finite) else np.nan,"maximum":float(finite.max()) if len(finite) else np.nan,"median":float(finite.median()) if len(finite) else np.nan,"99th percentile":float(finite.quantile(.99)) if len(finite) else np.nan,"corporate-action alerts":0})
    rows.extend({"feature name":spec.name,"feature family":spec.family,"built status":"skipped","availability start":None,"missing rate":1.0,"infinite count":0,"minimum":np.nan,"maximum":np.nan,"median":np.nan,"99th percentile":np.nan,"corporate-action alerts":np.nan} for spec in requested if spec.name not in built)
    pd.DataFrame(rows).to_csv(root/"feature_build_report.csv",index=False)


def _git_revision() -> str | None:
    repository=Path(__file__).resolve().parents[2]
    try:return subprocess.check_output(["git","-C",str(repository),"rev-parse","HEAD"],text=True).strip()
    except Exception:return None
