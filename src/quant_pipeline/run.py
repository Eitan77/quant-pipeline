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
from .parallel_features import build_parallel_blocks, is_symbol_local
from .registry import feature_registry, registry_frame, target_registry
from .scanner import benjamini_hochberg,binary_scan_batch,categorical_scan_batch,scan
from .bulk_scan import assert_valid_screen_results,build_cuda_feature_context,cuda_screen,finalize_screen
from .report import write_reports
from .table import add_targets, filter_decision_rows, load_canonical_bars, source_provenance, validate_point_in_time
from .fingerprint import enforce_fingerprint,run_fingerprint
from .cache import ROW_KEYS,assert_cache_key_alignment,validate_cache,write_cache_metadata
from .holdout import assert_pre_holdout_frame


def execute(config: ScanConfig) -> Path:
    config.validate()
    run_id=config.experiment_id
    root=Path(config.output_root)/run_id; root.mkdir(parents=True,exist_ok=True)
    sector=bool(config.sector_map_path); base_requested=feature_registry(config.lookbacks,sector,config.opening_windows_minutes); targets=target_registry(config.target_horizons_minutes,config.primary_target_horizons_minutes)
    compiled_dual=[]
    if config.dual_factor_enabled:
        from .dual_registry import compile_dual_plan, plan_hash
        manifest_path=Path(config.dual_factor_manifest_path or "configs/phase1b_dual_factors.yaml")
        if not manifest_path.exists(): manifest_path=Path(__file__).resolve().parents[2]/manifest_path
        compiled_dual=compile_dual_plan(manifest_path,base_requested,config)
        requested=base_requested+[item.spec for item in compiled_dual]
        dual_extra={"dual_manifest_hash":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"dual_plan_hash":plan_hash(compiled_dual),"dual_plan":[item.spec.name for item in compiled_dual]}
    else:
        requested=base_requested; dual_extra={}
    fingerprint=run_fingerprint(config,requested,targets,_git_revision(),extra_components=dual_extra); enforce_fingerprint(root,fingerprint,config.resume)
    if compiled_dual:
        phase1b_root=root/"phase1b"; phase1b_root.mkdir(exist_ok=True)
        manifest_source=Path(config.dual_factor_manifest_path or "configs/phase1b_dual_factors.yaml")
        if not manifest_source.exists(): manifest_source=Path(__file__).resolve().parents[2]/manifest_source
        (phase1b_root/"manifest_source.yaml").write_bytes(manifest_source.read_bytes())
        registry_frame([item.spec for item in compiled_dual]).to_csv(phase1b_root/"dual_feature_registry.csv",index=False)
        pd.DataFrame([item.spec.__dict__ for item in compiled_dual]).to_json(phase1b_root/"compiled_feature_plan.json",orient="records",indent=2,default_handler=str)
    import gc
    import pandas as pd
    fingerprint_sha=fingerprint["sha256"]; bars_cache=root/"canonical_bars.parquet"; bars=None
    if not bars_cache.exists():
        bars=load_canonical_bars(config); bars.to_parquet(bars_cache,index=False); write_cache_metadata(bars_cache,bars,fingerprint_sha,config.sealed_holdout_start)
    else:validate_cache(bars_cache,fingerprint_sha,config.sealed_holdout_start)
    checkpoint=root/"screen_checkpoint.csv"; journal=root/"screen_journal.jsonl"; categorical_journal=root/"categorical_journal.jsonl"; results=pd.DataFrame(); built_names=set(); validation_records=[]
    block_root=root/"blocks"; feature_root=block_root/"features"; target_root=block_root/"targets"; feature_root.mkdir(parents=True,exist_ok=True); target_root.mkdir(parents=True,exist_ok=True)
    eligible=[s for s in base_requested if s.status=="requested"]
    local_specs=[spec for spec in eligible if is_symbol_local(spec)]
    global_specs=[spec for spec in eligible if not is_symbol_local(spec)]
    chunks=[local_specs[i:i+config.feature_chunk_size] for i in range(0,len(local_specs),config.feature_chunk_size)]
    chunks += [global_specs[i:i+config.feature_chunk_size] for i in range(0,len(global_specs),config.feature_chunk_size)]
    raw_targets=[t for t in targets if t.classification=="raw"]
    target_batches=[]
    for start in range(0,len(raw_targets),config.target_chunk_size):
        raw_batch=raw_targets[start:start+config.target_chunk_size]; names={t.name for t in raw_batch}
        target_batches.append(raw_batch+[t for t in targets if (t.classification=="benchmark_adjusted" and t.name.removesuffix("_benchmark_adjusted") in names) or (t.classification=="beta_residual" and t.name.removesuffix("_beta_residual") in names)])
    # Materialize each target block once. They are reused by every feature block.
    target_paths=[target_root/f"target_{target_index:03d}.parquet" for target_index in range(len(target_batches))]
    completed_targets=set(); existing_targets=[(index,path) for index,path in enumerate(target_paths) if path.exists()]
    def validate_indexed_cache(item):
        index,path=item; return index,validate_cache(path,fingerprint_sha,config.sealed_holdout_start)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=config.cache_validation_workers) as pool:
        validated_targets=dict(pool.map(validate_indexed_cache,existing_targets))
    for target_index,(target_batch,path) in enumerate(zip(target_batches,target_paths)):
        if not path.exists():continue
        saved=validated_targets[target_index]; record=saved.get("point_in_time_validation")
        if record is None:
            cached=pd.read_parquet(path); record=validate_point_in_time(cached,target_batch,config.sealed_holdout_start); del cached
        record=dict(record); record["batch"]=target_index; validation_records.append(record)
        completed_targets.add(target_index)
    target_pending=[index for index in range(len(target_batches)) if index not in completed_targets]
    for start in range(0,len(target_pending),config.target_build_batch_chunks):
        indices=target_pending[start:start+config.target_build_batch_chunks]
        if bars is None:bars=pd.read_parquet(bars_cache)
        combined_targets=[target for index in indices for target in target_batches[index]]
        target_frame=add_targets(bars,combined_targets,config.benchmark_symbol,config)
        for target_index in indices:
            target_batch=target_batches[target_index]; path=target_paths[target_index]
            record=validate_point_in_time(target_frame,target_batch,config.sealed_holdout_start); record["batch"]=target_index; validation_records.append(record)
            identifiers=[*ROW_KEYS,"bar_end_ts","available_at_ts","symbol_role","pit_member","scan_eligible","session_grid_eligible","analysis_eligible","benchmark_valid"]
            target_columns=[*identifiers,"entry_ts","entry_open_raw","beta_at_decision",*[t.name for t in target_batch],*[f"exit_ts__{t.name}" for t in target_batch if t.classification=="raw"],*[f"exit_close_raw__{t.name}" for t in target_batch if t.classification=="raw"],*[f"actual_horizon_minutes__{t.name}" for t in target_batch if t.classification=="raw"]]
            target_columns=[column for column in target_columns if column in target_frame]
            cached=target_frame.loc[:,target_columns].copy(); cached.to_parquet(path,index=False); write_cache_metadata(path,cached,fingerprint_sha,config.sealed_holdout_start,record); del cached
            completed_targets.add(target_index)
            (root/"progress.json").write_text(json.dumps({"stage":"target_cache","completed":len(completed_targets),"total":len(target_batches),"build_batch":indices,"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
        del target_frame; gc.collect()
    del bars; bars=None; gc.collect()
    # Symbol-local blocks are built by 16 independent processes. Features that
    # require the full cross-section remain full-universe calculations.
    feature_paths=[]
    for feature_index,chunk in enumerate(chunks):
        digest=hashlib.sha1("\n".join(s.name for s in chunk).encode()).hexdigest()[:10]
        feature_paths.append(feature_root/f"feature_{feature_index:03d}_{digest}.parquet")
    built_by_chunk=[[] for _ in chunks]; completed_features=set()

    def finish_feature_block(feature_index,built):
        path=feature_paths[feature_index]
        if not path.with_suffix(path.suffix+".meta.json").exists():
            metadata_frame=pd.read_parquet(path); write_cache_metadata(path,metadata_frame,fingerprint_sha,config.sealed_holdout_start); del metadata_frame
        built_by_chunk[feature_index]=built; built_names.update(s.name for s in built); completed_features.add(feature_index)
        (root/"progress.json").write_text(json.dumps({"stage":"feature_cache","completed":len(completed_features),"total":len(chunks),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")

    existing_features=[(index,path) for index,path in enumerate(feature_paths) if path.exists()]
    with ThreadPoolExecutor(max_workers=config.cache_validation_workers) as pool:
        validated_features=set(index for index,_ in pool.map(validate_indexed_cache,existing_features))
    for feature_index,(chunk,path) in enumerate(zip(chunks,feature_paths)):
        if path.exists():
            if feature_index not in validated_features:raise RuntimeError(f"Feature cache was not validated: {path}")
            import pyarrow.parquet as pq
            columns=set(pq.ParquetFile(path).schema.names); built=[s for s in chunk if s.name in columns]
            finish_feature_block(feature_index,built)

    local_pending=[index for index,chunk in enumerate(chunks) if index not in completed_features and all(is_symbol_local(spec) for spec in chunk)]
    batch_size=config.feature_build_batch_chunks
    for start in range(0,len(local_pending),batch_size):
        indices=local_pending[start:start+batch_size]
        outputs=[(feature_paths[index],chunks[index]) for index in indices]
        built_batch=build_parallel_blocks(bars_cache,outputs,config,root/"progress.json")
        for index,built in zip(indices,built_batch):finish_feature_block(index,built)
        gc.collect()

    for feature_index,(chunk,path) in enumerate(zip(chunks,feature_paths)):
        if feature_index in completed_features:continue
        if not all(is_symbol_local(spec) for spec in chunk):
            if bars is None:
                bars=pd.read_parquet(bars_cache)
                for column in ["open","high","low","close","vwap","volume"]:
                    if column in bars: bars[column]=pd.to_numeric(bars[column],downcast="float")
            feature_frame,built=build_features(bars,config,chunk)
            feature_frame.to_parquet(path,index=False); del feature_frame; gc.collect()
            finish_feature_block(feature_index,built)
    del bars; gc.collect()
    # Phase 1B materializes only after every parent base cache has passed the
    # same fingerprint/key/holdout validation as Phase 1A.
    if compiled_dual:
        from .dual_features import build_dual_feature_chunks, build_feature_cache_index
        base_index=build_feature_cache_index(feature_paths,built_by_chunk)
        dual_paths,dual_chunks,dual_coverage=build_dual_feature_chunks(compiled_dual,base_index,config,root/config.dual_factor_cache_subdir,fingerprint_sha)
        feature_paths.extend(dual_paths); chunks.extend(dual_chunks); built_by_chunk.extend(dual_chunks)
        built_names.update(spec.name for chunk in dual_chunks for spec in chunk)
        pd.DataFrame(dual_coverage).to_csv(root/"phase1b"/"dual_feature_coverage.csv",index=False)
        from .effects import validate_binary_semantics
        for path, chunk in zip(feature_paths, built_by_chunk):
            validate_binary_semantics(pd.read_parquet(path,columns=[s.name for s in chunk]),chunk,config.binary_semantics_validation)
    resume_frames=[]
    if config.resume and checkpoint.exists(): resume_frames.append(pd.read_csv(checkpoint))
    if config.resume and journal.exists(): resume_frames.append(pd.read_json(journal,lines=True))
    if config.resume and categorical_journal.exists(): resume_frames.append(pd.read_json(categorical_journal,lines=True))
    if resume_frames:
        results=pd.concat(resume_frames,ignore_index=True).drop_duplicates(["feature","target"],keep="last")
        assert_valid_screen_results(results,"resumed screen journal")
    total_batches=len(chunks)*len(target_batches); completed_batches=0
    from concurrent.futures import ThreadPoolExecutor
    for feature_index,(feature_path,built) in enumerate(zip(feature_paths,built_by_chunk)):
        completed_pairs={(row.feature,row.target) for row in results.itertuples()}
        pending=[i for i,batch in enumerate(target_batches) if any((spec.name,t.name) not in completed_pairs for spec in built for t in batch)]
        completed_batches += len(target_batches)-len(pending)
        if not pending: continue
        feature_frame=pd.read_parquet(feature_path)
        for target_path in target_paths:assert_cache_key_alignment(feature_path,target_path)
        screen_end=config.selection_end if config.use_separate_confirmation_period else config.discovery_end
        if not screen_end:raise ValueError("A discovery screen end is required")
        selection_mask=pd.to_datetime(feature_frame.session_date).le(pd.Timestamp(screen_end))&feature_frame.analysis_eligible.fillna(False)
        if config.decision_times_et:
            local=pd.to_datetime(feature_frame.decision_ts,utc=True).dt.tz_convert("America/New_York"); selection_mask&=local.dt.strftime("%H:%M").isin(config.decision_times_et)
        selection_mask=selection_mask.to_numpy()
        screen_features=feature_frame.loc[selection_mask].reset_index(drop=True)
        continuous=[s for s in built if s.classification!="categorical" and s.dtype not in {"categorical","binary"}]
        binary=[s for s in built if s.dtype=="binary"]
        categorical=[s for s in built if s.classification=="categorical"]
        feature_context=build_cuda_feature_context(screen_features,continuous,config) if continuous else None
        target_groups=[pending[i:i+config.cuda_target_batch_group_size] for i in range(0,len(pending),config.cuda_target_batch_group_size)]

        def load_target_group(indices):
            pieces=[]
            for target_index in indices:
                names=[t.name for t in target_batches[target_index]]
                target_values=pd.read_parquet(target_paths[target_index],columns=names)
                pieces.append(target_values.loc[selection_mask].reset_index(drop=True))
            return pd.concat(pieces,axis=1)

        with ThreadPoolExecutor(max_workers=1) as prefetch:
            future=prefetch.submit(load_target_group,target_groups[0]) if target_groups else None
            for position,indices in enumerate(target_groups):
                target_batch=[target for index in indices for target in target_batches[index]]
                screen_targets=future.result()
                future=prefetch.submit(load_target_group,target_groups[position+1]) if position+1<len(target_groups) else None
                results=cuda_screen(screen_features,screen_targets,continuous,[t.name for t in target_batch],config,results,journal,feature_context)
                if binary:
                    combined=pd.concat([screen_features.reset_index(drop=True),screen_targets[[t.name for t in target_batch]].reset_index(drop=True)],axis=1)
                    binary_rows=binary_scan_batch(combined,binary,[t.name for t in target_batch],config)
                    completed_pairs={(row.feature,row.target) for row in results.itertuples()}
                    binary_rows=binary_rows.loc[[((row.feature,row.target) not in completed_pairs) for row in binary_rows.itertuples()]]
                    if not binary_rows.empty:
                        assert_valid_screen_results(binary_rows,"binary screen batch")
                        binary_rows.to_json(journal,orient="records",lines=True,mode="a",double_precision=15)
                    results=pd.concat([results,binary_rows],ignore_index=True).drop_duplicates(["feature","target"],keep="last")
                if categorical:
                    combined=pd.concat([screen_features.reset_index(drop=True),screen_targets[[t.name for t in target_batch]].reset_index(drop=True)],axis=1)
                    cat_rows=categorical_scan_batch(combined,categorical,[t.name for t in target_batch],config)
                    completed_pairs={(row.feature,row.target) for row in results.itertuples()}
                    cat_rows=cat_rows.loc[[((row.feature,row.target) not in completed_pairs) for row in cat_rows.itertuples()]]
                    if not cat_rows.empty:
                        assert_valid_screen_results(cat_rows,"categorical screen batch")
                        cat_rows.to_json(categorical_journal,orient="records",lines=True,mode="a",double_precision=15)
                    results=pd.concat([results,cat_rows],ignore_index=True).drop_duplicates(["feature","target"],keep="last")
                completed_batches+=len(indices)
                (root/"progress.json").write_text(json.dumps({"stage":"cuda_screen","completed_batches":completed_batches,"total_batches":total_batches,"completed_feature_chunks":feature_index,"total_feature_chunks":len(chunks),"completed_pairs":len(results),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
                del screen_targets; gc.collect()
        del feature_context,feature_frame,screen_features; gc.collect()
    results=finalize_screen(results.drop_duplicates(["feature","target"],keep="last"))
    # Registry metadata is the authority for Phase/lineage/redundancy and must
    # not be inferred from generated names.
    registry_metadata=registry_frame(requested).drop(columns=["description","classification","dtype"],errors="ignore")
    results=results.drop(columns=[c for c in registry_metadata.columns if c in results and c != "feature"],errors="ignore").merge(registry_metadata.rename(columns={"name":"feature"}),on="feature",how="left")
    results["scan_kind"]=np.where(results.dtype.eq("binary"),"binary",np.where(results.dtype.eq("categorical"),"categorical","continuous"))
    results["redundancy_group"]=results.redundancy_group.fillna(results.feature)
    results.to_csv(checkpoint,index=False)
    if journal.exists(): journal.unlink()
    if categorical_journal.exists():categorical_journal.unlink()
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
    exploratory=results.loc[results.target_tier.eq("exploratory")].copy()
    exploratory["exploratory_family_fdr"]=exploratory.groupby(["feature_family","target_family"],dropna=False).raw_p.transform(benjamini_hochberg)
    results=results.merge(primary[["feature","target","primary_global_fdr"]],on=["feature","target"],how="left")
    if not exploratory.empty:results=results.merge(exploratory[["feature","target","exploratory_family_fdr"]],on=["feature","target"],how="left")
    else:results["exploratory_family_fdr"]=np.nan
    results["primary_test_count"]=int(primary.raw_p.notna().sum()); results["exploratory_test_count"]=int(exploratory.raw_p.notna().sum()); results.to_csv(root/"master_results.csv",index=False)
    if compiled_dual:
        parent_rows=[]
        for item in compiled_dual:
            dual=results.loc[results.feature.eq(item.spec.name)]
            for parent in item.spec.parent_features:
                parent_result=results.loc[results.feature.eq(parent)]
                for row in dual.itertuples():
                    match=parent_result.loc[parent_result.target.eq(row.target)]
                    parent_rows.append({"dual_feature":item.spec.name,"parent_feature":parent,"target":row.target,"dual_effect":row.top_bottom_spread,"parent_effect":match.top_bottom_spread.iloc[0] if not match.empty else np.nan,"dual_raw_p":row.raw_p,"parent_raw_p":match.raw_p.iloc[0] if not match.empty else np.nan})
        pd.DataFrame(parent_rows).to_csv(root/"phase1b"/"dual_parent_comparison.csv",index=False)
    assert_valid_screen_results(results,"master screen results",check_fdr=True)
    feature_path_by_name={spec.name:feature_paths[index] for index,chunk in enumerate(chunks) for spec in chunk}
    diagnostic_context_path=_build_diagnostic_context(feature_path_by_name,built_names,config,root)
    queue=_cluster_candidates(primary,feature_path_by_name,config,limit=250)
    top_specs=[s for s in requested if s.name in set(queue.feature)]
    detailed=[]; qtabs={}; spec_by_name={s.name:s for s in top_specs}
    target_path_by_name={spec.name:target_paths[index] for index,batch in enumerate(target_batches) for spec in batch}
    from concurrent.futures import FIRST_COMPLETED,ProcessPoolExecutor,ThreadPoolExecutor,wait
    from .exact_parallel import exact_pair
    from .hybrid_exact import gpu_dense_pair
    cpu_rows={}; diagnostic_rows={}; gpu_rows={}; hints={(row.feature,row.target):float(row.spearman) for row in queue.itertuples()}; screen_fdr={(row.feature,row.target):float(row.primary_global_fdr) for row in queue.itertuples()}
    exact_root=root/"exact_journal"; cpu_root=exact_root/"cpu"; gpu_root=exact_root/"gpu"; cpu_root.mkdir(parents=True,exist_ok=True); gpu_root.mkdir(parents=True,exist_ok=True)
    def exact_path(folder,key):
        digest=hashlib.sha1("\0".join(key).encode()).hexdigest();return folder/f"{digest}.pkl"
    def save_exact(path,payload):
        temporary=path.with_suffix(path.suffix+".tmp");pd.to_pickle(payload,temporary);temporary.replace(path)
    for row in queue.itertuples():
        key=(row.feature,row.target); cpu_path=exact_path(cpu_root,key); gpu_path=exact_path(gpu_root,key)
        if cpu_path.exists():
            payload=pd.read_pickle(cpu_path)
            if tuple(payload.get("key",()))!=key:raise RuntimeError(f"Exact CPU journal key mismatch: {cpu_path}")
            cpu_rows[key]=payload["row"];diagnostic_rows[key]=payload["diagnostics"];qtabs[key]=pd.DataFrame(payload["table"])
        if gpu_path.exists():
            payload=pd.read_pickle(gpu_path)
            if tuple(payload.get("key",()))!=key:raise RuntimeError(f"Exact GPU journal key mismatch: {gpu_path}")
            gpu_rows[key]=payload["dense"]
    cpu_done=len(cpu_rows);gpu_done=len(gpu_rows)
    with ProcessPoolExecutor(max_workers=config.exact_workers) as cpu_pool, ThreadPoolExecutor(max_workers=1) as gpu_pool:
        futures={}
        for row in queue.itertuples():
            key=(row.feature,row.target); feature_path=feature_path_by_name[row.feature]; target_path=target_path_by_name[row.target]
            if key not in cpu_rows:futures[cpu_pool.submit(exact_pair,feature_path,target_path,spec_by_name[row.feature],row.target,config,hints[key],diagnostic_context_path)]=("cpu",key)
            exact_end=config.selection_end if config.use_separate_confirmation_period else config.discovery_end
            if key not in gpu_rows:futures[gpu_pool.submit(gpu_dense_pair,feature_path,target_path,row.feature,row.target,config.min_bin_observations,exact_end,row.scan_kind)]=("gpu",key)
        pending=set(futures)
        while pending:
            finished,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                kind,key=futures[future]
                if kind=="cpu":
                    row,table,diagnostics=future.result(); cpu_rows[key]=row; diagnostic_rows[key]=diagnostics; qtabs[key]=pd.DataFrame(table);save_exact(exact_path(cpu_root,key),{"key":key,"row":row,"table":table,"diagnostics":diagnostics});cpu_done+=1
                else:
                    dense,table=future.result(); gpu_rows[key]=dense;save_exact(exact_path(gpu_root,key),{"key":key,"dense":dense});gpu_done+=1
            status={"stage":"exact_diagnostics_hybrid","cpu_completed":cpu_done,"gpu_completed":gpu_done,"total":len(queue),"cpu_workers":config.exact_workers,"gpu_workers":1,"updated_at":datetime.now(timezone.utc).isoformat()}
            (root/"progress.json").write_text(json.dumps(status,indent=2),encoding="utf-8"); (root/"detailed_progress.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    # Preserve the already-selected queue order. Worker completion order must
    # never influence redundancy penalties or the existing ranking formula.
    for queued_item in queue.itertuples(index=False):
        key=(queued_item.feature,queued_item.target); row=cpu_rows.get(key)
        if row is None or key not in gpu_rows: continue
        dense=gpu_rows[key]; hint=hints[key]
        if np.sign(dense["spearman"])!=np.sign(hint):
            for column in ["year_consistency","symbol_breadth","time_stability"]:
                if pd.notna(row.get(column)): row[column]=1-row[column]
        queued=queue.loc[queue.feature.eq(key[0])&queue.target.eq(key[1])].iloc[0]
        row.update(dense); row["screen_bh_fdr_p_global"]=screen_fdr[key]; row["family_fdr"]=queued.family_fdr; row["cluster_fdr"]=queued.cluster_fdr; row["candidate_cluster"]=queued.candidate_cluster; row["target_tier"]="primary"; detailed.append(row)
    detailed=pd.DataFrame(detailed)
    if not detailed.empty:
        detailed["exact_selected_bh_fdr_p"]=detailed.groupby(["feature_family","target_family"],dropna=False).raw_p.transform(benjamini_hochberg)
        detailed["bh_fdr_p"]=detailed["screen_bh_fdr_p_global"]
        detailed=apply_confirmation_fdr(detailed,config)
        detailed=_classify_detailed_candidates(detailed,config)
        detailed["effect_size_score"]=detailed.top_bottom_spread.abs().rank(pct=True); detailed["global_fdr_score"]=(1-detailed.bh_fdr_p.fillna(1)).clip(0,1); detailed["quantile_shape_score"]=np.where(detailed.get("scan_kind",pd.Series("continuous",index=detailed.index)).eq("binary"),1.0,detailed.monotonicity.abs().fillna(0)); detailed["session_stability_score"]=detailed.year_consistency.fillna(0); detailed["symbol_breadth_score"]=detailed.symbol_breadth.fillna(0); detailed["outlier_robustness_score"]=(detailed.outlier_worst_signed_spread.fillna(0)>0).astype(float); detailed["historical_stability_score"]=detailed.get("historical_subperiod_positive_fold_pct",pd.Series(0,index=detailed.index)).fillna(0); detailed["recent_relevance_score"]=detailed.get("recent_12m_effect",pd.Series(0,index=detailed.index)).gt(0).astype(float); detailed["redundancy_penalty"]=detailed.groupby("candidate_cluster").cumcount()*.05; detailed["complexity_penalty"]=detailed.get("complexity_units",pd.Series(1,index=detailed.index)).fillna(1).astype(float)*.005
        detailed["anomaly_score"]=.2*detailed.effect_size_score+.2*detailed.global_fdr_score+.1*detailed.quantile_shape_score+.1*detailed.session_stability_score+.1*detailed.symbol_breadth_score+.1*detailed.outlier_robustness_score+.1*detailed.historical_stability_score+.1*detailed.recent_relevance_score-detailed.redundancy_penalty-detailed.complexity_penalty
        detailed=detailed.sort_values("anomaly_score",ascending=False)
        from .diagnostics import phase2_recommendation
        recommendations=pd.DataFrame([phase2_recommendation(row) for _,row in detailed.iterrows()],index=detailed.index); detailed=pd.concat([detailed.drop(columns=[c for c in recommendations if c in detailed],errors="ignore"),recommendations],axis=1); detailed.to_csv(root/"detailed_candidates.csv",index=False)
        cluster_report=detailed.groupby("candidate_cluster",dropna=False).agg(pairs=("feature","size"),best_primary_fdr=("bh_fdr_p","min"),best_effect=("top_bottom_spread",lambda s:float(s.loc[s.abs().idxmax()])),robust_phase1=("status",lambda s:int(s.eq("robust_phase1_anomaly_candidate").any()))).reset_index()
        cluster_report.to_csv(root/"cluster_level_anomalies.csv",index=False)
        diagnostic_root=root/"candidate_diagnostics"; diagnostic_root.mkdir(exist_ok=True)
        for (feature,target),tables in diagnostic_rows.items():
            stem=f"{feature}__{target}".replace("/","_")
            for name,records in tables.items():
                table=pd.DataFrame(records)
                if not table.empty:
                    assert_pre_holdout_frame(table,config.sealed_holdout_start,f"diagnostic output {name}")
                    table.to_csv(diagnostic_root/f"{stem}__{name}.csv",index=False)
        _write_aggregate_diagnostics(root,diagnostic_rows,config)
    _write_coverage_reports(root,results,requested,feature_paths,built_by_chunk)
    assert_pre_holdout_frame(pd.read_parquet(bars_cache,columns=["session_date","bar_start_ts","decision_ts"]),config.sealed_holdout_start,"report input")
    write_reports(detailed if not detailed.empty else results.head(50), qtabs, root, None, config=config,run_metadata={"fingerprint":fingerprint_sha,"git_commit":_git_revision()})
    from .gpu import CorrelationBackend
    violation_fields=["entry_before_decision_violations","exit_before_entry_violations","target_cross_session_violations","horizon_mismatch_violations","missing_entry_price_rows","missing_exit_price_rows","missing_benchmark_rows","holdout_rows"]
    validation_summary={field:int(sum(record.get(field,0) for record in validation_records)) for field in violation_fields}; provenance=source_provenance(config)
    manifest={"experiment_id":run_id,"cache_schema_version":config.cache_schema_version,"executed_at":datetime.now(timezone.utc).isoformat(),"git_commit":_git_revision(),"config":config.as_dict(),"configuration_hash":hashlib.sha256(json.dumps(config.as_dict(),sort_keys=True).encode()).hexdigest(),"fingerprint":fingerprint_sha,"source_provenance":provenance,"evidence_label":"full_pre_holdout_discovery","discovery_start":config.start,"discovery_end":config.discovery_end,"use_separate_confirmation_period":config.use_separate_confirmation_period,"confirmation_start":config.confirmation_start if config.use_separate_confirmation_period else None,"confirmation_end":config.discovery_end if config.use_separate_confirmation_period else None,"sealed_holdout_start":config.sealed_holdout_start,"holdout_access":config.allow_holdout_access,"requested_features":len(requested),"built_features":len(built_names),"skipped_features":skipped,"target_count_requested":len(targets),"target_count_built":len(targets),"target_batches_requested":len(target_batches),"target_batches_built":len(target_paths),"target_batches_validated":len(validation_records),"targets_validated":sum(len(batch) for batch in target_batches),"target_batch_validation":validation_records,"validation_violations":validation_summary,"broad_screen_pair_count":len(results),"pairs_passing_coverage":int(results.raw_p.notna().sum()),"primary_test_count":int(primary.raw_p.notna().sum()),"exploratory_test_count":int(exploratory.raw_p.notna().sum()),"global_fdr_significant_primary_pairs":int(primary.primary_global_fdr.lt(config.primary_fdr_threshold).sum()),"exploratory_significant_pairs":int(exploratory.exploratory_family_fdr.lt(config.primary_fdr_threshold).sum()) if not exploratory.empty else 0,"candidate_cluster_count":int(detailed.candidate_cluster.nunique()) if not detailed.empty else 0,"exact_candidate_count":len(detailed),"robust_phase1_candidate_count":int(detailed.status.eq("robust_phase1_anomaly_candidate").sum()) if not detailed.empty else 0,"requires_phase2_count":int(detailed.status.eq("requires_phase2_testing").sum()) if not detailed.empty else 0,"primary_targets":[t.name for t in targets if t.tier=="primary"],"multiple_testing":"Global BH across primary prespecified targets; feature-family and cluster FDR reported separately. Exploratory horizons and recency diagnostics cannot promote candidates.","statistical_error":"Date-clustered screen; exact pass adds two-way date/symbol clustering, session-block bootstrap, and HAC daily spread/IC inference","correlation_backend":CorrelationBackend(config.use_cuda,config.cuda_device).name}
    manifest["known_limitations"]=["Point-in-time sector, industry, and market-cap diagnostics are unavailable unless a reviewed sector_map_path is configured.","Spread diagnostics use available liquidity proxies; quote-level execution costs remain Phase 2 work.","Historical annual folds are subperiod stability diagnostics, not independently selected out-of-sample confirmation."]
    manifest["benchmark_source_rows_invalid"]=int(sum(record.get("benchmark_source_rows_invalid",0) for record in validation_records))
    manifest["benchmark_sessions_invalid"]=int(sum(record.get("benchmark_sessions_invalid",0) for record in validation_records))
    manifest["beta_residual_rows_excluded_insufficient_history"]=int(sum(record.get("beta_residual_rows_excluded_insufficient_history",0) for record in validation_records))
    manifest["descriptive_diagnostics_only"]=True
    manifest["regime_diagnostic_candidates"]=int(detailed.regime_summary_label.notna().sum()) if "regime_summary_label" in detailed else 0
    manifest["scope_diagnostic_candidates"]=int(detailed.scope_classification.notna().sum()) if "scope_classification" in detailed else 0
    manifest["exact_time_diagnostic_candidates"]=int(detailed.time_concentration_label.notna().sum()) if "time_concentration_label" in detailed else 0
    manifest["phase2_recommendation_candidates"]=int(detailed.phase2_recommendation.notna().sum()) if "phase2_recommendation" in detailed else 0
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (root/"progress.json").write_text(json.dumps({"stage":"complete","screened_pairs":len(results),"exact_candidates":len(detailed),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
    return root


def _classify_detailed_candidates(detailed,config:ScanConfig|None=None):
    """Assign full-discovery evidence labels; recency alone never promotes."""
    config=config or ScanConfig()
    result=detailed.copy()
    usable=~result.status.isin(["constant_feature","insufficient_data"])
    coverage=result.get("n",pd.Series(0,index=result.index)).ge(config.min_observations)&result.get("valid_sessions",pd.Series(0,index=result.index)).ge(config.min_sessions)&result.get("valid_symbols",pd.Series(0,index=result.index)).ge(config.min_symbols)
    raw_p=result.get("raw_p",pd.Series(np.nan,index=result.index,dtype=float))
    exact=result.get("two_way_cluster_p",raw_p).lt(.05)
    global_fdr=result.get("screen_bh_fdr_p_global",pd.Series(np.nan,index=result.index,dtype=float)).lt(config.primary_fdr_threshold)
    stable=result.get("year_consistency",pd.Series(np.nan,index=result.index,dtype=float)).ge(.6)&result.get("symbol_breadth",pd.Series(np.nan,index=result.index,dtype=float)).ge(.6)
    economic=result.get("top_bottom_spread",pd.Series(np.inf,index=result.index)).abs().ge(config.minimum_effect_bps/10000)
    binary=result.get("scan_kind",pd.Series("continuous",index=result.index)).eq("binary")
    shape=binary|result.get("monotonicity",pd.Series(0,index=result.index)).abs().ge(.5)
    outlier=result.get("outlier_worst_signed_spread",pd.Series(-np.inf,index=result.index)).gt(0)
    breadth=~result.get("symbol_breadth_classification",pd.Series("insufficient_evidence",index=result.index)).isin(["highly_concentrated","single_symbol_dominated","insufficient_evidence"])
    recent=result.get("recent_12m_effect",pd.Series(np.nan,index=result.index)).gt(0)
    result.loc[usable,"status"]="no_meaningful_relationship"
    result.loc[usable&exact&~global_fdr,"status"]="exploratory_relationship"
    result.loc[usable&global_fdr,"status"]="statistically_significant_discovery"
    result.loc[usable&global_fdr&exact&coverage&economic&stable,"status"]="stable_anomaly_candidate"
    result.loc[usable&global_fdr&exact&coverage&economic&recent&~stable,"status"]="recently_relevant_anomaly_candidate"
    robust=usable&global_fdr&exact&coverage&economic&stable&shape&outlier&breadth
    result.loc[robust,"status"]="robust_phase1_anomaly_candidate"
    result.loc[usable&global_fdr&exact&coverage&economic&~robust&result.get("phase2_recommendation_seed",pd.Series("",index=result.index)).str.startswith("advance"),"status"]="requires_phase2_testing"
    result["phase2_required"]=result.status.isin(["robust_phase1_anomaly_candidate","requires_phase2_testing"])
    return result


def apply_confirmation_fdr(detailed:pd.DataFrame,config:ScanConfig)->pd.DataFrame:
    result=detailed.copy()
    if config.use_separate_confirmation_period:
        values=result.get("confirmation_p_value",pd.Series(np.nan,index=result.index,dtype=float))
        result["confirmation_fdr"]=benjamini_hochberg(values)
        result["nominally_confirmed"]=values.lt(config.confirmation_alpha)
        result["confirmation_fdr_confirmed"]=result.confirmation_fdr.lt(config.confirmation_alpha)
    else:
        result["confirmation_fdr"]=np.nan; result["nominally_confirmed"]=False; result["confirmation_fdr_confirmed"]=False
    return result


def _target_horizon_family(target:str)->str:
    import re
    if "eod" in target:return "long_intraday"
    match=re.search(r"_(\d+)m",target); horizon=int(match.group(1)) if match else 999
    if horizon<=15:return "very_short"
    if horizon<=45:return "short"
    if horizon<=120:return "medium"
    return "long_intraday"


def _cluster_candidates(primary,feature_paths:dict[str,Path]|None,config:ScanConfig,limit:int=250):
    if primary.empty:return primary
    eligible=primary.loc[primary.raw_p.notna()].sort_values(["primary_global_fdr","anomaly_score"],ascending=[True,False])
    ranked=pd.concat([eligible.head(100),eligible.loc[eligible.primary_global_fdr.lt(.10)]]).drop_duplicates(["feature","target"]).copy()
    if ranked.empty:return ranked
    ranked=ranked.groupby("feature_family",sort=False).head(config.max_candidates_per_feature_family)
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
    response_sign=np.sign(ranked.monotonicity.fillna(ranked.spearman).fillna(0)).astype(int).astype(str)
    ranked["candidate_cluster"]=ranked.feature_cluster+"__"+ranked.target.map(_target_horizon_family)+"__response_"+response_sign
    ranked["cluster_fdr"]=ranked.groupby("candidate_cluster",dropna=False).raw_p.transform(benjamini_hochberg)
    representatives=[]
    for _,group in ranked.groupby("candidate_cluster",sort=False):
        best=group.head(1)
        simplest=group.assign(complexity=group.feature.str.count("_")+group.feature.str.extract(r"_(\d+)(?:m)?$",expand=False).fillna("0").astype(int)/1000).sort_values("complexity").head(1)
        neighbor=group.iloc[[min(1,len(group)-1)]]
        representatives.append(pd.concat([best,simplest,neighbor]).drop_duplicates(["feature","target"]).head(config.max_candidates_per_cluster))
    promoted=pd.concat(representatives,ignore_index=True).drop_duplicates(["feature","target"])
    return promoted.groupby(promoted.target.map(_target_horizon_family),sort=False).head(config.max_candidates_per_target_family).head(limit)


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


def _build_diagnostic_context(feature_paths:dict[str,Path],built_names:set[str],config:ScanConfig,root:Path)->Path|None:
    """Join already-built causal context features; never rebuild or rescan."""
    context_features={
        "high_market_vol":"high_market_vol",
        "universe_breadth_positive":"universe_breadth_positive",
        "universe_return_dispersion":"universe_return_dispersion",
        "return_since_open":"benchmark_return_since_open",
        "distance_session_vwap":"benchmark_distance_session_vwap",
        "overnight_gap":"benchmark_overnight_gap",
    }
    pieces=[]
    for feature,output in context_features.items():
        if feature not in built_names or feature not in feature_paths:continue
        path=feature_paths[feature]; columns=["decision_ts",feature]
        import pyarrow.parquet as pq
        schema=set(pq.ParquetFile(path).schema.names)
        if feature in {"return_since_open","distance_session_vwap","overnight_gap"}:columns += [c for c in ["symbol","benchmark_valid"] if c in schema]
        else:columns += [c for c in ["analysis_eligible"] if c in schema]
        frame=pd.read_parquet(path,columns=columns)
        if feature in {"return_since_open","distance_session_vwap","overnight_gap"}:
            mask=frame.symbol.eq(config.benchmark_symbol) if "symbol" in frame else pd.Series(False,index=frame.index)
            if "benchmark_valid" in frame:mask&=frame.benchmark_valid.fillna(False)
            frame=frame.loc[mask]
        elif "analysis_eligible" in frame:frame=frame.loc[frame.analysis_eligible.fillna(False)]
        series=frame.dropna(subset=[feature]).drop_duplicates("decision_ts").set_index("decision_ts")[feature].rename(output); pieces.append(series)
    if not pieces:return None
    context=pd.concat(pieces,axis=1).reset_index().sort_values("decision_ts"); assert_pre_holdout_frame(context,config.sealed_holdout_start,"diagnostic context build")
    path=root/"diagnostic_context.parquet"; context.to_parquet(path,index=False); return path


def _write_aggregate_diagnostics(root:Path,diagnostic_rows:dict,config:ScanConfig)->None:
    outputs={"regime":"candidate_regime_diagnostics.csv","sector":"candidate_sector_diagnostics.csv","industry":"candidate_industry_diagnostics.csv","scope":"candidate_scope_classification.csv","exact_time":"effect_by_exact_decision_time.csv"}
    for key,filename in outputs.items():
        records=[record for tables in diagnostic_rows.values() for record in tables.get(key,[])]
        table=pd.DataFrame(records); assert_pre_holdout_frame(table,config.sealed_holdout_start,f"aggregate diagnostic {key}"); table.to_csv(root/filename,index=False)


def _git_revision() -> str | None:
    repository=Path(__file__).resolve().parents[2]
    try:return subprocess.check_output(["git","-C",str(repository),"rev-parse","HEAD"],text=True).strip()
    except Exception:return None
