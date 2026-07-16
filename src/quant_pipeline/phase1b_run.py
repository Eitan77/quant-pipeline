"""Phase 1B-only source-run validation and combined-FDR primitives.

This module deliberately treats Phase 1A caches as immutable inputs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cache import assert_cache_key_alignment, validate_cache
from .config import ScanConfig
from .dual_features import build_dual_feature_chunks
from .dual_registry import compile_dual_plan, plan_hash
from .fingerprint import enforce_fingerprint, phase1b_run_fingerprint
from .holdout import assert_pre_holdout_parquet
from .registry import (
    FeatureSpec, TargetSpec, load_feature_registry, load_target_registry,
    registry_frame, write_registry_json,
)
from .screen_finalization import finalize_phase1_screen
from .screen_orchestration import screen_feature_blocks_against_target_blocks
from .bulk_scan import assert_valid_screen_results


@dataclass(frozen=True)
class Phase1ASource:
    root: Path
    manifest: dict
    fingerprint: dict
    feature_registry: list[FeatureSpec]
    target_registry: list[TargetSpec]
    feature_cache_index: dict[str,Path]
    target_cache_index: dict[str,Path]
    master_results_path: Path
    source_manifest_hash: str
    tree_hash: str
    source_results_hash: str


def _index(paths:list[Path], names:set[str]) -> dict[str,Path]:
    import pyarrow.parquet as pq
    result={}
    for path in paths:
        for name in set(pq.ParquetFile(path).schema.names)&names:
            if name in result:raise RuntimeError(f"Cache column has ambiguous ownership: {name}")
            result[name]=path
    return result


def _tree_hash(root:Path)->str:
    digest=hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).replace("\\","/").encode()); digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _atomic_csv(frame:pd.DataFrame,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp");frame.to_csv(temporary,index=False);temporary.replace(path)


def _atomic_json(payload:dict,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8");temporary.replace(path)


def validate_phase1a_source(source_run: str | Path, config: ScanConfig) -> Phase1ASource:
    root=Path(source_run)
    manifest_path=root/"manifest.json"; fingerprint_path=root/"fingerprint.json"
    progress_path=root/"progress.json"; results_path=root/"master_results.csv"; feature_registry_path=root/"feature_registry.csv"; target_registry_path=root/"target_registry.csv"
    if not all(path.exists() for path in (manifest_path,fingerprint_path,progress_path,results_path,feature_registry_path,target_registry_path)):raise FileNotFoundError("Source Phase 1A is incomplete")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); fingerprint=json.loads(fingerprint_path.read_text(encoding="utf-8"))
    progress=json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("stage")!="complete":raise ValueError("Source Phase 1A run is not complete")
    if manifest.get("allow_holdout_access") or manifest.get("holdout_access") or manifest.get("sealed_holdout_start")!=config.sealed_holdout_start or manifest.get("discovery_end")!=config.discovery_end:raise ValueError("Source run has incompatible discovery or holdout boundaries")
    source_hash=fingerprint.get("sha256")
    if not source_hash:raise ValueError("Source fingerprint is missing its digest")
    feature_paths=sorted((root/"blocks"/"features").glob("*.parquet")); target_paths=sorted((root/"blocks"/"targets").glob("*.parquet"))
    if not feature_paths or not target_paths:raise FileNotFoundError("Source feature or target caches are missing")
    for path in [*feature_paths,*target_paths]:
        validate_cache(path,source_hash,config.sealed_holdout_start); assert_pre_holdout_parquet(path,config.sealed_holdout_start,"phase1b source validation",verify_key_rows=False)
    features=load_feature_registry(feature_registry_path); targets=load_target_registry(target_registry_path)
    source_results=pd.read_csv(results_path)
    required={"feature","target","raw_p"}
    if not required.issubset(source_results):raise ValueError(f"Source master results missing columns: {sorted(required-set(source_results))}")
    if source_results.duplicated(["feature","target"]).any():raise ValueError("Source master results contain duplicate hypotheses")
    assert_valid_screen_results(source_results,"source Phase 1A master results",check_fdr=True)
    unknown_result_features=sorted(set(source_results.feature)-{item.name for item in features})
    unknown_result_targets=sorted(set(source_results.target)-{item.name for item in targets})
    if unknown_result_features:raise ValueError(f"Source results contain unknown features: {unknown_result_features}")
    if unknown_result_targets:raise ValueError(f"Source results contain unknown targets: {unknown_result_targets}")
    feature_index=_index(feature_paths,{item.name for item in features}); target_index=_index(target_paths,{item.name for item in targets})
    missing_features=sorted({item.name for item in features if item.status=="requested"}-set(feature_index))
    missing_targets=sorted({item.name for item in targets}-set(target_index))
    if missing_features:raise FileNotFoundError(f"Source registry features missing from caches: {missing_features}")
    if missing_targets:raise FileNotFoundError(f"Source registry targets missing from caches: {missing_targets}")
    anchor=target_paths[0]
    for path in [*feature_paths,*target_paths[1:]]:assert_cache_key_alignment(path,anchor)
    return Phase1ASource(root,manifest,fingerprint,features,targets,feature_index,target_index,results_path,hashlib.sha256(manifest_path.read_bytes()).hexdigest(),_tree_hash(root),hashlib.sha256(results_path.read_bytes()).hexdigest())


def merge_combined_results(base: pd.DataFrame, dual: pd.DataFrame, targets: list[TargetSpec] | None=None, features:list[FeatureSpec]|None=None, config:ScanConfig|None=None) -> pd.DataFrame:
    base=base.copy();dual=dual.copy()
    if "discovery_phase" not in base:base["discovery_phase"]="1A"
    if "arity" not in base:base["arity"]=1
    result=pd.concat([base,dual],ignore_index=True)
    if result.duplicated(["feature","target"]).any():raise ValueError("Duplicate feature-target rows in combined Phase 1 results")
    if targets is None:return result
    if features is None:
        features=[]
        for row in result.drop_duplicates("feature").itertuples():
            features.append(FeatureSpec(name=row.feature,description="persisted screen feature",family=getattr(row,"feature_family","unknown"),discovery_phase=getattr(row,"discovery_phase","1A") if pd.notna(getattr(row,"discovery_phase","1A")) else "1A",arity=int(getattr(row,"arity",1)) if pd.notna(getattr(row,"arity",1)) else 1,redundancy_group=getattr(row,"redundancy_group",None)))
    return finalize_phase1_screen(result,feature_registry=features,target_registry=targets,config=config or ScanConfig()).master


def _parent_comparison(combined:pd.DataFrame,compiled)->pd.DataFrame:
    rows=[]
    lookup=combined.set_index(["feature","target"])
    for item in compiled:
        for target in combined.loc[combined.feature.eq(item.spec.name),"target"]:
            dual=lookup.loc[(item.spec.name,target)]
            for parent in item.spec.parent_features:
                if (parent,target) not in lookup.index:continue
                base=lookup.loc[(parent,target)]; de=float(dual.get("top_bottom_spread",float("nan"))); pe=float(base.get("top_bottom_spread",float("nan")))
                rows.append({"dual_feature":item.spec.name,"parent_feature":parent,"target":target,"dual_effect":de,"parent_effect":pe,"effect_increment":de-pe,"effect_ratio":de/pe if pe else float("nan"),"dual_raw_p":dual.raw_p,"parent_raw_p":base.raw_p,"dual_primary_global_fdr":dual.get("primary_global_fdr"),"parent_primary_global_fdr":base.get("primary_global_fdr"),"same_direction":bool(pd.notna(de) and pd.notna(pe) and de*pe>0)})
    return pd.DataFrame(rows)


def _run_dual_exact(
    source:Phase1ASource,
    combined:pd.DataFrame,
    dual_paths:list[Path],
    chunks:list[list[FeatureSpec]],
    config:ScanConfig,
    root:Path,
) -> pd.DataFrame:
    """Run authoritative CPU exact diagnostics for newly introduced features."""
    from .exact_parallel import exact_pair
    dual_path_by_name={spec.name:path for path,specs in zip(dual_paths,chunks) for spec in specs}
    specs={spec.name:spec for specs_chunk in chunks for spec in specs_chunk}
    queue=combined.loc[
        combined.discovery_phase.eq("1B")
        & combined.target_tier.eq("primary")
        & combined.primary_global_fdr.le(config.primary_fdr_threshold)
    ].sort_values(["primary_global_fdr","feature","target"],kind="mergesort")
    exact_root=root/"exact_journal"/"cpu";exact_root.mkdir(parents=True,exist_ok=True)
    rows=[];diagnostic_root=root/"candidate_diagnostics";diagnostic_root.mkdir(exist_ok=True)
    for queued in queue.itertuples(index=False):
        key=(queued.feature,queued.target);digest=hashlib.sha1("\0".join(key).encode()).hexdigest();journal=exact_root/f"{digest}.pkl"
        if journal.exists():
            payload=pd.read_pickle(journal)
            if tuple(payload.get("key",()))!=key:raise RuntimeError(f"Exact journal key mismatch: {journal}")
        else:
            row,table,diagnostics=exact_pair(dual_path_by_name[queued.feature],source.target_cache_index[queued.target],specs[queued.feature],queued.target,config,float(getattr(queued,"spearman",0) or 0),None)
            payload={"key":key,"row":row,"table":table,"diagnostics":diagnostics};temporary=journal.with_suffix(".pkl.tmp");pd.to_pickle(payload,temporary);temporary.replace(journal)
        row=payload.get("row")
        if row is None:continue
        row=dict(row);row.update({"feature":queued.feature,"target":queued.target,"screen_bh_fdr_p_global":queued.primary_global_fdr,"primary_global_fdr":queued.primary_global_fdr,"family_fdr":queued.family_fdr,"cluster_fdr":queued.cluster_fdr,"candidate_cluster":queued.candidate_cluster,"target_tier":"primary","discovery_phase":"1B","promotion_inference":"exact_cpu_two_way_date_symbol"});rows.append(row)
        stem=f"{queued.feature}__{queued.target}".replace("/","_")
        for name,records in payload.get("diagnostics",{}).items():
            if records:_atomic_csv(pd.DataFrame(records),diagnostic_root/f"{stem}__{name}.csv")
    detailed=pd.DataFrame(rows)
    source_detail=source.root/"detailed_candidates.csv"
    if source_detail.exists():
        prior=pd.read_csv(source_detail)
        refresh=combined[["feature","target","primary_global_fdr","family_fdr","cluster_fdr","candidate_cluster"]]
        prior=prior.drop(columns=[column for column in refresh.columns if column not in {"feature","target"} and column in prior],errors="ignore").merge(refresh,on=["feature","target"],how="inner")
        prior["discovery_phase"]="1A";prior["screen_bh_fdr_p_global"]=prior.primary_global_fdr
        detailed=pd.concat([prior,detailed],ignore_index=True,sort=False)
    if not detailed.empty:
        from .run import _classify_detailed_candidates
        detailed=_classify_detailed_candidates(detailed,config)
        detailed=detailed.sort_values(["primary_global_fdr","feature","target"],kind="mergesort")
    _atomic_csv(detailed if not detailed.empty else pd.DataFrame(columns=["feature","target","status"]),root/"detailed_candidates.csv")
    return detailed


def run_phase1b(source_run: str | Path, config: ScanConfig) -> Path:
    """Build and screen dual factors from immutable Phase 1A caches only."""
    if not config.dual_factor_enabled:raise ValueError("Phase 1B-only execution requires dual_factor_enabled=true")
    source=validate_phase1a_source(source_run,config)
    manifest_path=Path(config.dual_factor_manifest_path or "configs/phase1b_dual_factors.yaml")
    if not manifest_path.exists():manifest_path=Path(__file__).resolve().parents[2]/manifest_path
    compiled=compile_dual_plan(manifest_path,source.feature_registry,config)
    missing=sorted({parent for item in compiled for parent in item.spec.parent_features if parent not in source.feature_cache_index})
    if missing:raise FileNotFoundError(f"Source caches lack required dual parents: {missing}")
    root=(Path(config.output_root)/config.experiment_id).resolve()
    source_root=source.root.resolve()
    if root==source_root or source_root in root.parents or root in source_root.parents:
        raise ValueError("Phase 1B output must be separate from the immutable Phase 1A source tree")
    extra={"source_fingerprint":source.fingerprint["sha256"],"source_manifest_hash":source.source_manifest_hash,"source_results_hash":source.source_results_hash,"source_tree_hash":source.tree_hash,"dual_manifest_hash":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"dual_plan_hash":plan_hash(compiled)}
    combined_registry=source.feature_registry+[item.spec for item in compiled]
    fingerprint=phase1b_run_fingerprint(config,combined_registry,source.target_registry,extra); enforce_fingerprint(root,fingerprint,config.resume); root.mkdir(parents=True,exist_ok=True)
    phase=root/"phase1b"; phase.mkdir(exist_ok=True)
    manifest_copy=phase/"manifest_source.yaml"; temporary=manifest_copy.with_suffix(".yaml.tmp");temporary.write_bytes(manifest_path.read_bytes());temporary.replace(manifest_copy)
    _atomic_json({"source_run":str(source.root),"fingerprint":source.fingerprint["sha256"],"manifest_sha256":source.source_manifest_hash,"master_results_sha256":source.source_results_hash,"tree_sha256":source.tree_hash},phase/"source_artifact_hashes.json")
    _atomic_json({"plan_hash":plan_hash(compiled),"features":[{"definition":item.definition,"spec":item.spec} for item in compiled]},phase/"compiled_feature_plan.json")
    dual_paths,chunks,coverage=build_dual_feature_chunks(compiled,source.feature_cache_index,config,root/config.dual_factor_cache_subdir,fingerprint["sha256"])
    _atomic_csv(pd.DataFrame(coverage),phase/"dual_feature_coverage.csv"); _atomic_csv(registry_frame([item.spec for item in compiled]),phase/"dual_feature_registry.csv")
    write_registry_json([item.spec for item in compiled],phase/"dual_feature_registry.json")
    target_paths=sorted(set(source.target_cache_index.values()),key=str)
    target_chunks=[[item for item in source.target_registry if source.target_cache_index[item.name]==path] for path in target_paths]
    dual=screen_feature_blocks_against_target_blocks(feature_paths=dual_paths,feature_specs_by_path=chunks,target_paths=target_paths,target_specs_by_path=target_chunks,config=config,run_root=root)
    if compiled and any(chunks) and dual.empty:
        raise RuntimeError("Phase 1B built features but produced no broad-screen result rows")
    _atomic_csv(dual.sort_values(["feature","target"],kind="mergesort"),phase/"dual_screen_results.csv")
    base=pd.read_csv(source.master_results_path); combined=merge_combined_results(base,dual,source.target_registry,combined_registry,config)
    _atomic_csv(combined,root/"master_results.csv")
    _atomic_csv(registry_frame(combined_registry),root/"feature_registry.csv");_atomic_csv(registry_frame(source.target_registry),root/"target_registry.csv")
    write_registry_json(combined_registry,root/"feature_registry.json");write_registry_json(source.target_registry,root/"target_registry.json")
    _atomic_csv(_parent_comparison(combined,compiled),phase/"dual_parent_comparison.csv")
    detailed=_run_dual_exact(source,combined,dual_paths,chunks,config,root)
    expected=set(map(tuple,combined.loc[combined.target_tier.eq("primary")&combined.primary_global_fdr.le(config.primary_fdr_threshold),["feature","target"]].to_numpy()))
    observed=set(map(tuple,detailed[["feature","target"]].to_numpy())) if not detailed.empty else set()
    promotion_ready=expected.issubset(observed)
    if not promotion_ready and not detailed.empty:
        detailed.loc[detailed.status.isin(["robust_phase1_anomaly_candidate","requires_phase2_testing"]),"status"]="exact_diagnostics_incomplete"
        _atomic_csv(detailed,root/"detailed_candidates.csv")
    valid_two_way=int(pd.to_numeric(dual.get("two_way_cluster_p",pd.Series(dtype=float)),errors="coerce").notna().sum())
    limitations=[] if promotion_ready else ["One or more combined-FDR candidates lack reusable or newly computed exact diagnostics."]
    status={"stage":"complete","run_type":"phase1b_derived","source_phase1a_run":str(source.root),"source_phase1a_fingerprint":source.fingerprint["sha256"],"source_manifest_hash":source.source_manifest_hash,"phase1b_fingerprint":fingerprint["sha256"],"discovery_end":config.discovery_end,"sealed_holdout_start":config.sealed_holdout_start,"holdout_access":False,"compiled_dual_features":len(compiled),"built_dual_features":sum(len(c) for c in chunks),"skipped_dual_features":int(sum(row.get("status")!="built" for row in coverage)),"dual_pairs_screened":len(dual),"empty_plan_reason":"manifest_compiled_zero_features" if not compiled else None,"pairs_passing_on_off_coverage":int(dual.raw_p.notna().sum()) if "raw_p" in dual else 0,"pairs_with_valid_two_way_inference":valid_two_way,"combined_primary_test_count":int(combined.get("primary_test_count",pd.Series([0])).max()),"combined_exploratory_test_count":int(combined.get("exploratory_test_count",pd.Series([0])).max()),"exact_candidate_count":len(detailed),"promotion_ready":promotion_ready,"evidence_stage":"exact_diagnostics_complete" if promotion_ready else "exact_diagnostics_incomplete","source_immutable":True,"phase1a_rebuilt":False,"combined_test_count":int(combined.raw_p.notna().sum()),"known_limitations":limitations}
    _atomic_json(status,root/"progress.json"); _atomic_json({**status,"config":config.as_dict(),"multiple_testing":"combined Phase 1A + Phase 1B broad-screen BH"},root/"manifest.json")
    if _tree_hash(source.root)!=source.tree_hash:raise RuntimeError("Phase 1A source changed during derived Phase 1B execution")
    readiness=[f"Phase 1B readiness: {'GO' if promotion_ready else 'NO-GO'}",f"Source Phase 1A run path: {source.root}",f"Source Phase 1A fingerprint: {source.fingerprint['sha256']}",f"Source manifest hash: {source.source_manifest_hash}",f"Derived Phase 1B fingerprint: {fingerprint['sha256']}",f"Discovery end: {config.discovery_end}",f"Sealed holdout start: {config.sealed_holdout_start}","Holdout access: false","Source immutable: true","Phase 1A rebuilt: false",f"Compiled dual features: {len(compiled)}",f"Built dual features: {sum(len(c) for c in chunks)}",f"Dual pairs screened: {len(dual)}",f"Pairs with valid two-way inference: {valid_two_way}",f"Combined primary test count: {status['combined_primary_test_count']}",f"Combined exploratory test count: {status['combined_exploratory_test_count']}",f"Exact candidates: {len(detailed)}",f"Known limitations: {limitations or 'none'}"]
    report=root/"phase1b_readiness_report.txt";tmp=report.with_suffix(".txt.tmp");tmp.write_text("\n".join(readiness)+"\n",encoding="utf-8");tmp.replace(report)
    return root
