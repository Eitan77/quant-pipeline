"""Phase 1B-only source-run validation and combined-FDR primitives.

This module deliberately treats Phase 1A caches as immutable inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .bulk_scan import finalize_screen
from .cache import validate_cache
from .config import ScanConfig
from .holdout import assert_pre_holdout_parquet


def validate_phase1a_source(source_run: str | Path, config: ScanConfig) -> dict:
    root=Path(source_run)
    manifest_path=root/"manifest.json"; fingerprint_path=root/"fingerprint.json"
    if not manifest_path.exists() or not fingerprint_path.exists():raise FileNotFoundError("Source Phase 1A manifest/fingerprint is missing")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); fingerprint=json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if manifest.get("allow_holdout_access") or manifest.get("sealed_holdout_start")!=config.sealed_holdout_start or manifest.get("discovery_end")!=config.discovery_end:raise ValueError("Source run has incompatible discovery or holdout boundaries")
    source_hash=fingerprint.get("sha256")
    if not source_hash:raise ValueError("Source fingerprint is missing its digest")
    feature_paths=sorted((root/"blocks"/"features").glob("*.parquet")); target_paths=sorted((root/"blocks"/"targets").glob("*.parquet"))
    if not feature_paths or not target_paths:raise FileNotFoundError("Source feature or target caches are missing")
    for path in [*feature_paths,*target_paths]:
        validate_cache(path,source_hash,config.sealed_holdout_start); assert_pre_holdout_parquet(path,config.sealed_holdout_start,"phase1b source validation",verify_key_rows=False)
    return {"root":root,"manifest":manifest,"fingerprint":fingerprint,"feature_paths":feature_paths,"target_paths":target_paths,"source_manifest_hash":hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


def merge_combined_results(base: pd.DataFrame, dual: pd.DataFrame) -> pd.DataFrame:
    result=pd.concat([base,dual],ignore_index=True)
    if result.duplicated(["feature","target"]).any():raise ValueError("Duplicate feature-target rows in combined Phase 1 results")
    # Existing Phase 1A q-values are descriptive only after new hypotheses are
    # introduced. Recompute every promotion q-value over the combined rows.
    result=result.drop(columns=[column for column in ("bh_fdr_p","bh_fdr_p_global","bh_fdr_p_group","primary_global_fdr") if column in result],errors="ignore")
    return finalize_screen(result)


def run_phase1b(source_run: str | Path, config: ScanConfig) -> dict:
    """Validate immutable Phase 1A inputs before the launcher builds anything."""
    if not config.dual_factor_enabled:raise ValueError("Phase 1B-only execution requires dual_factor_enabled=true")
    return validate_phase1a_source(source_run,config)
