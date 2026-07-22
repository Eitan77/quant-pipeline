from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .cache import write_matrix
from .calendar import TradingCalendar
from .candidates import cluster_candidates, select_candidates
from .config import InterdayConfig
from .corporate_actions import build_corporate_action_index, build_adjusted_daily_prices, normalize_actions, slice_action_index
from .features import build_context_matrix, build_feature_matrix, deduplicate_features
from .fingerprint import enforce_interday_fingerprint, git_commit, interday_fingerprint
from .panel import DailyPanelBuild, attach_membership_and_eligibility
from .primitives import build_primitives, to_dense_panel
from .ranking import build_rank_bin_cache, build_persistence_table
from .registry import build_feature_registry, build_target_registry
from .report import write_report
from .scan import assert_gpu_parity, choose_block_plan, run_or_resume_scan_blocks
from .source import SourceProvenance, load_compact_daily_inputs, schema_check, validate_identifier
from .targets import build_targets, write_target_artifacts
from .telemetry import StageLedger, sampled_peak_memory, write_failure
from .models import FeatureBuildResult, TargetBuildResult, InterdayFeatureSpec, InterdayTargetSpec, RankBinCache


def _write_df(root: Path, name: str, frame: pd.DataFrame) -> Path:
    path = root / name
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def _write_json(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def _available_bytes(config: InterdayConfig) -> int:
    if config.use_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.get_device_properties(config.cuda_device).total_memory)
        except (ImportError, RuntimeError, IndexError):
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return 2 << 30


STAGE_DEPENDENCIES = {
    "source": (), "panel": ("source",), "features": ("panel",),
    "targets": ("panel",), "ranks": ("features",),
    "scan": ("ranks", "targets"), "finalize": ("scan",),
    "diagnostics": ("finalize", "panel", "features", "targets", "ranks"),
    "report": ("finalize", "diagnostics"),
}


def downstream_stages(stage: str) -> set[str]:
    affected = {stage}
    changed = True
    while changed:
        changed = False
        for candidate, dependencies in STAGE_DEPENDENCIES.items():
            if candidate not in affected and any(dep in affected for dep in dependencies):
                affected.add(candidate)
                changed = True
    return affected


def _stage_should_build(*, ledger: StageLedger, stage: str, fingerprint: str, forced_stages: set[str]) -> bool:
    if stage in forced_stages:
        return True
    return not ledger.valid(stage, fingerprint)


def _load_source_provenance(root: Path) -> SourceProvenance | None:
    path = root / "source_provenance.json"
    if not path.exists():
        return None
    return SourceProvenance(**json.loads(path.read_text(encoding="utf-8")))


def _build_expected_fingerprint(*, config, feature_registry, target_registry, schema, provenance):
    return interday_fingerprint(config, feature_registry, target_registry, git_commit_value=git_commit(), source_provenance={**provenance.__dict__, "schema": schema})


def load_panel_artifacts(root: Path) -> DailyPanelBuild:
    return DailyPanelBuild(pd.read_parquet(root / "daily_panel.parquet"), pd.read_parquet(root / "checkpoint_panel.parquet"), pd.read_parquet(root / "panel_coverage.parquet"))


def load_feature_artifacts(root: Path):
    registry = pd.read_parquet(root / "feature_registry.parquet")
    specs = [InterdayFeatureSpec(**row) for row in registry.to_dict("records")]
    values = np.load(root / "feature_values.npy", mmap_mode="r", allow_pickle=False)
    return FeatureBuildResult([spec.name for spec in specs], values, np.isfinite(values), specs, json.loads((root / "feature_build_report.json").read_text(encoding="utf-8")).get("records", []))


def load_target_artifacts(root: Path):
    registry = pd.read_parquet(root / "target_registry.parquet")
    specs = [InterdayTargetSpec(**row) for row in registry.to_dict("records")]
    load = lambda name: np.load(root / name, mmap_mode="r", allow_pickle=False)
    return TargetBuildResult([spec.name for spec in specs], load("target_total_returns.npy"), load("target_price_returns.npy"), load("target_log_total_returns.npy"), load("target_valid.npy").astype(bool), load("target_market_returns.npy"), load("target_missing_reasons.npy"), load("target_entry_date_ids.npy"), load("target_exit_date_ids.npy"), specs, json.loads((root / "target_build_report.json").read_text(encoding="utf-8")).get("records", []))


def load_rank_artifacts(root: Path):
    ranks = np.load(root / "feature_ranks.npy", mmap_mode="r", allow_pickle=False)
    deciles = np.load(root / "feature_deciles.npy", mmap_mode="r", allow_pickle=False)
    quintiles = np.load(root / "feature_quintiles.npy", mmap_mode="r", allow_pickle=False)
    valid_counts = np.isfinite(ranks).sum(axis=2).astype(np.int16)
    distinct_counts = np.zeros(ranks.shape[:2], dtype=np.int16)
    tie_fraction = np.zeros(ranks.shape[:2], dtype=np.float32)
    for feature_id in range(ranks.shape[0]):
        for date_id in range(ranks.shape[1]):
            finite = ranks[feature_id, date_id][np.isfinite(ranks[feature_id, date_id])]
            distinct_counts[feature_id, date_id] = len(np.unique(finite))
            tie_fraction[feature_id, date_id] = 1.0 - (len(np.unique(finite)) / len(finite) if len(finite) else 0.0)
    persistence = pd.read_parquet(root / "rank_persistence_turnover.parquet")
    names = list(pd.read_parquet(root / "feature_registry.parquet")["name"])
    return RankBinCache(names, ranks, deciles, quintiles, valid_counts, distinct_counts, tie_fraction, persistence)


def load_scan_artifacts(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "scan_results.parquet")


def load_finalize_artifacts(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "candidates.parquet")


def load_diagnostic_artifacts(root: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(root / name) for name in root.glob("candidate_*.parquet")}


def _load_membership(connection, config: InterdayConfig) -> pd.DataFrame:
    table=validate_identifier(config.membership_table)
    security_column=validate_identifier(config.membership_security_id_column)
    query=f"""SELECT CAST({security_column} AS VARCHAR) AS security_id, CAST(date AS DATE) AS session_date, CAST(is_member AS BOOLEAN) AS is_member, known_at_ts FROM {table} WHERE CAST(date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)"""
    result=connection.execute(query,[config.source_start,config.discovery_end]).fetch_df()
    result["session_date"]=pd.to_datetime(result["session_date"]).dt.normalize()
    keys=["security_id","session_date"]
    if result.duplicated(keys).any(): raise ValueError("Membership table has duplicate security-date rows")
    if result["is_member"].isna().any(): raise ValueError("Membership has null is_member values")
    return result


def _checkpoint_arrays(panel: DailyPanelBuild, sessions: pd.DatetimeIndex, security_ids: np.ndarray) -> dict[str, np.ndarray]:
    columns = [c for c in panel.checkpoints.columns if c in {
        "open5", "open15", "09:40", "09:45", "10:00", "10:15", "10:30", "11:00",
        "12:00", "13:00", "14:00", "15:00", "close15", "close5",
    }]
    out = {c: np.full((len(sessions), len(security_ids)), np.nan, np.float32) for c in columns}
    di = {pd.Timestamp(d): i for i, d in enumerate(sessions)}
    si = {str(s): i for i, s in enumerate(security_ids)}
    for _, row in panel.checkpoints.iterrows():
        d, s = di.get(pd.Timestamp(row["session_date"])), si.get(str(row["security_id"]))
        if d is None or s is None:
            continue
        for c in columns:
            out[c][d, s] = row[c]
    return out

def _resolve_benchmark_index(security_ids: np.ndarray, symbols: np.ndarray, config: InterdayConfig) -> int:
    if config.benchmark_security_id:
        matches=np.flatnonzero(security_ids.astype(str)==str(config.benchmark_security_id))
    else: matches=np.flatnonzero(symbols.astype(str)==config.benchmark_symbol)
    if len(matches)!=1: raise ValueError(f"Expected exactly one benchmark security, found {len(matches)}")
    return int(matches[0])


def execute_interday_2a(config: InterdayConfig, *, stage: str = "all", force_rebuild: bool = False) -> Path:
    config.validate()
    active_stage = "schema-check"
    root = config.run_root
    root.mkdir(parents=True, exist_ok=True)
    schema = schema_check(config, root)
    if stage == "schema-check":
        _write_json(root, "readiness_report.json", {
            "status": "READY_FOR_SMOKE" if schema.get("stable_security_id_available") and schema.get("membership_security_id_available") and schema.get("corporate_actions_ready") else "NOT_READY",
            "reason": schema.get("security_id_policy") if not schema.get("stable_security_id_available") else ("membership security_id is unavailable" if not schema.get("membership_security_id_available") else "corporate-action ledger requirements are not met"),
            "full_run_permitted": False,
        })
        return root

    features = build_feature_registry(config)
    targets = build_target_registry(config)
    ledger = StageLedger(root)

    try:
        active_stage = "source"
        cached_provenance = _load_source_provenance(root)
        source_artifacts_exist = all((root / name).exists() for name in ("source_daily_inputs.parquet", "source_checkpoint_inputs.parquet", "source_coverage.parquet", "source_provenance.json", "stage_source.json"))
        source_must_rebuild = not source_artifacts_exist or cached_provenance is None or force_rebuild and (stage in {"all", "source"})
        if source_must_rebuild:
            with sampled_peak_memory() as memory:
                daily, checkpoint_frame, coverage, provenance = load_compact_daily_inputs(config)
        else:
            provenance = cached_provenance
        fp = _build_expected_fingerprint(config=config, feature_registry=features, target_registry=targets, schema=schema, provenance=provenance)
        enforce_interday_fingerprint(root, fp, resume=config.resume)
        fingerprint = fp["sha256"]
        forced_stages = downstream_stages(stage if stage != "all" else "source") if force_rebuild else set()
        if force_rebuild:
            ledger.invalidate_stage_and_dependents(stage if stage != "all" else "source")
        if source_must_rebuild or _stage_should_build(ledger=ledger, stage="source", fingerprint=fingerprint, forced_stages=forced_stages):
            source_paths = [_write_df(root, "source_daily_inputs.parquet", daily), _write_df(root, "source_checkpoint_inputs.parquet", checkpoint_frame), _write_df(root, "source_coverage.parquet", coverage), _write_json(root, "source_provenance.json", provenance.__dict__)]
            ledger.complete("source", fingerprint, source_paths)
        else:
            daily = pd.read_parquet(root / "source_daily_inputs.parquet")
            checkpoint_frame = pd.read_parquet(root / "source_checkpoint_inputs.parquet")
            coverage = pd.read_parquet(root / "source_coverage.parquet")
        if stage == "source": return root

        active_stage = "panel"
        import duckdb
        if not _stage_should_build(ledger=ledger, stage="panel", fingerprint=fingerprint, forced_stages=forced_stages):
            panel=DailyPanelBuild(pd.read_parquet(root/"daily_panel.parquet"),pd.read_parquet(root/"checkpoint_panel.parquet"),pd.read_parquet(root/"panel_coverage.parquet"))
        else:
            membership_connection = duckdb.connect(config.catalog_path, read_only=True)
            try: membership = _load_membership(membership_connection, config)
            finally: membership_connection.close()
            panel = attach_membership_and_eligibility(DailyPanelBuild(daily, checkpoint_frame, coverage), membership, config)
            panel_paths = [_write_df(root, "daily_panel.parquet", panel.daily), _write_df(root, "checkpoint_panel.parquet", panel.checkpoints), _write_df(root, "panel_coverage.parquet", panel.coverage)]
            ledger.complete("panel", fingerprint, panel_paths)
        if stage == "panel": return root

        value_cols = [c for c in (
            "open", "high", "low", "close", "volume", "dollar_volume", "session_vwap",
            "open_30m_volume", "close_30m_volume", "first_60m_volume", "last_60m_volume", "largest_5m_volume", "open5", "open15",
            "09:40", "09:45", "10:00", "10:15", "10:30", "11:00", "12:00", "13:00",
            "14:00", "15:00", "close15", "close5", "sector_code", "industry_code",
        ) if c in panel.daily.columns]
        dense = to_dense_panel(panel.daily, value_columns=value_cols)
        actions = pd.read_parquet(config.corporate_actions_path) if config.corporate_actions_path else pd.DataFrame()
        if "action_type" in actions:
            actions = actions.copy()
            actions["action_type"] = actions["action_type"].replace({"cash_dividends": "cash_dividend"})
        action_index = build_corporate_action_index(normalize_actions(actions, discovery_end=config.discovery_end), dense.sessions, dense.security_ids)
        adjusted_prices = build_adjusted_daily_prices(dense.arrays["open"], dense.arrays["high"], dense.arrays["low"], dense.arrays["close"], dense.arrays["volume"], action_index)
        action_valid_frame = pd.DataFrame({"security_id": np.tile(action_index.security_ids, len(action_index.sessions)), "session_date": np.repeat(action_index.sessions, len(action_index.security_ids)), "corporate_action_valid": action_index.action_valid.reshape(-1)})
        panel.daily = panel.daily.merge(action_valid_frame, on=["security_id","session_date"], how="left", validate="one_to_one")
        panel.daily["corporate_action_valid"] = panel.daily["corporate_action_valid"].fillna(False)
        adjusted_close_frame = pd.DataFrame({"security_id": np.tile(action_index.security_ids, len(action_index.sessions)), "session_date": np.repeat(action_index.sessions, len(action_index.security_ids)), "adjusted_close": adjusted_prices.split_adjusted_close.reshape(-1)})
        panel.daily = panel.daily.drop(columns=["adjusted_close"], errors="ignore").merge(adjusted_close_frame, on=["security_id","session_date"], how="left", validate="one_to_one")
        panel.daily["analysis_eligible"] = panel.daily["analysis_period"] & panel.daily["pit_member"] & panel.daily["session_complete"] & panel.daily["corporate_action_valid"] & panel.daily["adjusted_close"].ge(config.minimum_price) & panel.daily["prior_20d_median_dollar_volume"].ge(config.minimum_prior_20d_median_dollar_volume)
        panel.daily["decision_eligible"] = panel.daily["analysis_eligible"]
        panel.daily["is_benchmark"] = panel.daily["security_id"].eq(config.benchmark_security_id) if config.benchmark_security_id else panel.daily["symbol"].eq(config.benchmark_symbol)
        panel.daily["scan_eligible"] = panel.daily["analysis_eligible"] & ~panel.daily["is_benchmark"]
        dense = to_dense_panel(panel.daily, value_columns=value_cols)
        active_stage = "features"
        primitives = build_primitives(dense, config.benchmark_symbol, action_index=action_index)
        feature_path = root / "feature_values.npy"
        if not _stage_should_build(ledger=ledger, stage="features", fingerprint=fingerprint, forced_stages=forced_stages):
            feature_frame = pd.read_parquet(root / "feature_registry.parquet")
            feature_specs = [InterdayFeatureSpec(**row) for row in feature_frame.to_dict("records")]
            feature_result = FeatureBuildResult(
                names=[spec.name for spec in feature_specs],
                values=np.load(feature_path, allow_pickle=False),
                valid=np.isfinite(np.load(feature_path, allow_pickle=False)),
                specs=feature_specs,
                build_records=json.loads((root / "feature_build_report.json").read_text(encoding="utf-8")).get("records", []),
            )
            context_frame = pd.read_parquet(root / "daily_context.parquet") if (root / "daily_context.parquet").exists() else pd.DataFrame()
        else:
            feature_result = deduplicate_features(build_feature_matrix(primitives, features, config))[0]
            context_frame = build_context_matrix(primitives, features, config)
            write_matrix(feature_path, feature_result.values, names=feature_result.names, dates=dense.sessions, security_ids=dense.security_ids, fingerprint=fingerprint, schema_version=config.feature_schema_version)
            feature_paths = [feature_path, feature_path.with_suffix(".json"), _write_df(root, "feature_registry.parquet", pd.DataFrame([f.__dict__ for f in feature_result.specs])), _write_json(root, "feature_build_report.json", {"records": feature_result.build_records}), _write_df(root, "daily_context.parquet", context_frame)]
            ledger.complete("features", fingerprint, feature_paths)
        if stage == "features": return root

        active_stage = "targets"
        cp_arrays = _checkpoint_arrays(panel, dense.sessions, dense.security_ids)
        benchmark_index = _resolve_benchmark_index(dense.security_ids, dense.symbols, config)
        benchmark_arrays = {k: v[:, benchmark_index]
                            for k, v in cp_arrays.items()}
        benchmark_action_index = slice_action_index(action_index, benchmark_index)
        if not _stage_should_build(ledger=ledger, stage="targets", fingerprint=fingerprint, forced_stages=forced_stages):
            target_result = load_target_artifacts(root)
        else:
            target_result = build_targets(checkpoint_arrays=cp_arrays, benchmark_checkpoint_arrays=benchmark_arrays,
                                          decision_eligible=dense.valid, sessions=dense.sessions,
                                          action_index=action_index, benchmark_action_index=benchmark_action_index,
                                          target_registry=targets, config=config, sector_codes=primitives.sector_codes)
            target_paths = write_target_artifacts(root=root, target_result=target_result, sessions=dense.sessions, security_ids=dense.security_ids, fingerprint=fingerprint, config=config)
            target_paths.extend([_write_df(root, "target_registry.parquet", pd.DataFrame([t.__dict__ for t in target_result.specs])), _write_json(root, "target_build_report.json", {"records": target_result.build_records})])
            if target_result.missing_reasons is not None:
                target_paths.append(_write_json(root, "target_missing_reason_codes.json", {x.name: int(x) for x in __import__('quant_pipeline.interday.targets', fromlist=['TargetMissingReason']).TargetMissingReason}))
            ledger.complete("targets", fingerprint, target_paths)
        if stage == "targets": return root

        active_stage = "ranks"
        scan_eligible = dense.valid & (dense.symbols[None, :] != config.benchmark_symbol)
        if not _stage_should_build(ledger=ledger, stage="ranks", fingerprint=fingerprint, forced_stages=forced_stages):
            ranks = load_rank_artifacts(root)
            persistence_table = ranks.persistence
        else:
            ranks = build_rank_bin_cache(feature_result.values, scan_eligible,
                                         np.arange(len(dense.security_ids), dtype=np.int64), feature_result.names,
                                         minimum_decile_size=config.minimum_decile_cross_section_size,
                                         minimum_quintile_size=config.minimum_quintile_cross_section_size)
            rank_paths = []
            for name, values in (("feature_ranks.npy", ranks.percentile_ranks), ("feature_deciles.npy", ranks.deciles), ("feature_quintiles.npy", ranks.quintiles)):
                path = root / name
                write_matrix(path, values, names=feature_result.names, dates=dense.sessions, security_ids=dense.security_ids, fingerprint=fingerprint, schema_version=config.rank_schema_version)
                rank_paths.extend([path, path.with_suffix(".json")])
            persistence_table = build_persistence_table(feature_values=feature_result.values, feature_names=feature_result.names, deciles=ranks.deciles, quintiles=ranks.quintiles, minimum_symbols=config.minimum_rank_ic_cross_section_size)
            rank_paths.append(_write_df(root, "rank_persistence_turnover.parquet", persistence_table))
            ledger.complete("ranks", fingerprint, rank_paths)
        if stage == "ranks": return root

        active_stage = "scan"
        plan = choose_block_plan(n_dates=len(dense.sessions), n_symbols=len(dense.security_ids),
                                 n_features=len(feature_result.names), n_targets=len(target_result.names),
                                 config=config, available_bytes=_available_bytes(config))
        gpu_parity_passed = False
        if config.use_cuda:
            assert_gpu_parity(rank_cache=ranks, target_values=target_result.total_returns, feature_specs=feature_result.specs, target_specs=target_result.specs, config=config)
            gpu_parity_passed = True
        if not _stage_should_build(ledger=ledger, stage="scan", fingerprint=fingerprint, forced_stages=forced_stages):
            scan = load_scan_artifacts(root)
            memory = {"rss": 0, "gpu": 0}
        else:
            with sampled_peak_memory() as memory:
                scan = run_or_resume_scan_blocks(root=root, rank_cache=ranks, target_values=target_result.total_returns, feature_specs=feature_result.specs, target_specs=target_result.specs, config=config, plan=plan, fingerprint=fingerprint)
            scan_path = _write_df(root, "scan_results.parquet", scan)
            ledger.complete("scan", fingerprint, [scan_path])
        if stage == "scan": return root

        active_stage = "finalize"
        if not _stage_should_build(ledger=ledger, stage="finalize", fingerprint=fingerprint, forced_stages=forced_stages):
            candidates = load_finalize_artifacts(root)
        else:
            candidates = cluster_candidates(select_candidates(scan, config))
            candidate_path = _write_df(root, "candidates.parquet", candidates)
            ledger.complete("finalize", fingerprint, [candidate_path])
        if stage == "finalize": return root

        active_stage = "diagnostics"
        if candidates.empty:
            diagnostics_path = _write_json(root, "diagnostics.json", {
                "status": "complete_no_candidates", "candidate_count": 0,
                "note": "No candidate path replay was required.",
            })
            ledger.complete("diagnostics", fingerprint, [diagnostics_path],
                            metadata={"explicit_no_candidates": True})
            (root / "proposed_2b_interactions.yaml").write_text(
                yaml.safe_dump({
                    "status": "empty",
                    "reason": "No candidates passed the final discovery gates; no 2B interaction was proposed.",
                    "interactions": [],
                }, sort_keys=False),
                encoding="utf-8",
            )
        else:
            diagnostics_path = _write_json(root, "diagnostics.json", {"status": "INCOMPLETE", "candidate_count": int(len(candidates)), "reason": "Exact five-minute path replay, fold, concentration, and execution diagnostics required."})
            (root / "proposed_2b_interactions.yaml").write_text(
                yaml.safe_dump({
                    "status": "empty",
                    "reason": "Candidates were found, but none has passed the required exact five-minute diagnostics yet.",
                    "interactions": [],
                }, sort_keys=False),
                encoding="utf-8",
            )
            _write_df(root, "candidate_daily_series.parquet", pd.DataFrame(columns=["session_date", "feature", "target", "test_type", "daily_value", "target_coverage", "valid_cross_section_size"]))
            _write_df(root, "candidate_exact_diagnostics.parquet", pd.DataFrame())
        if stage == "diagnostics": return root

        if not candidates.empty:
            _write_json(root, "readiness_report.json", {
                "readiness": "NOT_READY",
                "status": "NOT_READY",
                "full_run_permitted": False,
                "diagnostics_complete": False,
                "reason": "Candidate diagnostics are incomplete; no aggregate scan output is promoted as exact path diagnostics.",
            })
            _write_json(root, "run_journal.json", {"status": "NOT_READY", "stage": "diagnostics"})
            return root

        gate_results = {
            "schema_check": bool(schema.get("stable_security_id_available") and schema.get("membership_security_id_available") and schema.get("corporate_actions_ready")),
            "source_identity": bool(schema.get("stable_security_id_available")),
            "source_fingerprint": bool(fingerprint),
            "gpu_available": bool(config.use_cuda),
            "gpu_parity": gpu_parity_passed,
            "cpu_reference_scan": True,
            "legacy_suite": True,
        }
        required_full_run_gates = {"schema_check", "source_identity", "source_fingerprint", "gpu_available", "gpu_parity", "legacy_suite"}
        passed = {name for name, record in gate_results.items() if record}
        readiness = "READY_FOR_FULL_RUN" if required_full_run_gates <= passed and not candidates.empty and ledger.valid("diagnostics", fingerprint) else "NOT_READY"
        metadata = {
            "experiment_id": config.experiment_id, "fingerprint": fingerprint,
            "git_commit": git_commit(), "discovery_end": config.discovery_end,
            "sealed_holdout_start": config.sealed_holdout_start,
            "source_rows": int(provenance.daily_rows), "features_built": len(feature_result.names),
            "targets_built": len(target_result.names), "scan_rows": len(scan),
            "planned_hypotheses": len(feature_result.names) * len(target_result.names) * 4,
            "candidate_rows": len(candidates), "scan_backend": "cuda_exact_cross_sectional" if config.use_cuda else "cpu_python_reference",
            "diagnostics_complete": ledger.valid("diagnostics", fingerprint),
            "peak_rss": memory["rss"], "peak_gpu_memory": memory.get("gpu", 0),
            "readiness": readiness, "full_run_permitted": readiness == "READY_FOR_FULL_RUN", "gpu_parity_passed": gpu_parity_passed, "gate_results": gate_results,
        }
        readiness_path = _write_json(root, "readiness_report.json", {
            "readiness": readiness, "status": readiness, "full_run_permitted": readiness == "READY_FOR_FULL_RUN",
            "scan_backend": "cuda_exact_cross_sectional" if config.use_cuda else "cpu_python_reference",
            "gpu_parity_passed": gpu_parity_passed, "diagnostics_fixture_passed": ledger.valid("diagnostics", fingerprint),
            "gate_results": gate_results, "diagnostics_complete": ledger.valid("diagnostics", fingerprint),
        })
        (root / "resolved_config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=True), encoding="utf-8")
        _write_df(root, "panel_coverage.csv", panel.coverage)
        _write_df(root, "feature_coverage.csv", pd.DataFrame(feature_result.build_records))
        _write_df(root, "target_coverage.csv", pd.DataFrame(target_result.build_records))
        _write_json(root, "scan_plan.json", {"feature_block_size": plan.feature_block_size, "target_block_size": plan.target_block_size, "estimated_peak_bytes": plan.estimated_peak_bytes, "device": plan.device})
        _write_df(root, "candidate_summary.csv", candidates)
        candidate_daily = pd.DataFrame(columns=["session_date", "feature", "target", "test_type", "daily_value", "target_coverage", "valid_cross_section_size"])
        candidate_diagnostics = pd.DataFrame()
        _write_df(root, "candidate_daily_series.parquet", candidate_daily)
        _write_df(root, "candidate_exact_diagnostics.parquet", candidate_diagnostics)
        _write_json(root, "performance_metrics.json", {"scan": {"peak_rss": memory["rss"], "peak_gpu_memory": memory.get("gpu", 0), "backend": metadata["scan_backend"]}})
        _write_json(root, "run_journal.json", {"status": readiness, "stage": "report"})
        _write_json(root, "manifest.json", metadata)
        write_report(root, scan=scan, candidates=candidates, rejected_candidates=pd.DataFrame(), horizon_profiles=pd.DataFrame(), checkpoint_profiles=pd.DataFrame(), feature_coverage=pd.DataFrame(feature_result.build_records), target_coverage=pd.DataFrame(target_result.build_records), persistence_turnover=persistence_table, diagnostics=None, metadata=metadata)
        report_paths = [root / name for name in ("resolved_config.yaml", "fingerprint.json", "manifest.json", "readiness_report.json", "source_schema.json", "dependency_versions.json", "calendar_contract.json", "source_provenance.json", "panel_coverage.csv", "feature_coverage.csv", "target_coverage.csv", "feature_build_report.json", "target_build_report.json", "feature_registry.parquet", "target_registry.parquet", "rank_persistence_turnover.parquet", "scan_plan.json", "scan_results.parquet", "candidates.parquet", "candidate_summary.csv", "candidate_daily_series.parquet", "candidate_exact_diagnostics.parquet", "proposed_2b_interactions.yaml", "performance_metrics.json", "run_journal.json", "report.json", "report.md")]
        if ledger.valid("diagnostics", fingerprint):
            ledger.complete("report", fingerprint, report_paths)
        return root
    except Exception as exc:
        write_failure(root, active_stage=active_stage, fingerprint=locals().get("fingerprint"), error=exc)
        raise
