from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import write_matrix
from .calendar import TradingCalendar
from .candidates import cluster_candidates, select_candidates
from .config import InterdayConfig
from .features import build_feature_matrix, deduplicate_features
from .fingerprint import enforce_interday_fingerprint, git_commit, interday_fingerprint
from .panel import DailyPanelBuild, attach_membership_and_eligibility
from .primitives import build_primitives, to_dense_panel
from .ranking import build_rank_bin_cache
from .registry import build_feature_registry, build_target_registry
from .report import write_report
from .scan import choose_block_plan, scan_feature_target_block
from .source import load_compact_daily_inputs, schema_check, validate_identifier
from .targets import build_targets
from .telemetry import StageLedger, sampled_peak_memory


def _write_df(root: Path, name: str, frame: pd.DataFrame) -> Path:
    path = root / name
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


def _load_membership(config: InterdayConfig) -> pd.DataFrame:
    import duckdb
    validate_identifier(config.membership_table)
    con = duckdb.connect(config.catalog_path, read_only=True)
    try:
        columns = {row[0] for row in con.execute(f"describe {config.membership_table}").fetchall()}
        required = {"date", "symbol", "is_member"}
        if not required.issubset(columns):
            raise ValueError(f"Membership schema missing columns: {sorted(required - columns)}")
        return con.execute(
            f"select date, symbol, is_member from {config.membership_table} "
            "where cast(date as date)>=cast(? as date) and cast(date as date)<cast(? as date)",
            [config.source_start, config.sealed_holdout_start],
        ).fetchdf()
    finally:
        con.close()


def _checkpoint_arrays(panel: DailyPanelBuild, sessions: pd.DatetimeIndex, security_ids: np.ndarray) -> dict[str, np.ndarray]:
    columns = [c for c in panel.checkpoints.columns if c in {
        "open5", "open15", "09:40", "09:45", "10:00", "10:15", "10:30", "11:00",
        "12:00", "13:00", "14:00", "15:00", "close15", "close5",
    }]
    out = {c: np.full((len(sessions), len(security_ids)), np.nan, np.float32) for c in columns}
    di = {pd.Timestamp(d): i for i, d in enumerate(sessions)}
    si = {str(s): i for i, s in enumerate(security_ids)}
    for row in panel.checkpoints.itertuples(index=False):
        d, s = di.get(pd.Timestamp(row.session_date)), si.get(str(row.security_id))
        if d is None or s is None:
            continue
        for c in columns:
            out[c][d, s] = getattr(row, c)
    return out


def execute_interday_2a(config: InterdayConfig, *, stage: str = "all", force_rebuild: bool = False) -> Path:
    config.validate()
    root = config.run_root
    root.mkdir(parents=True, exist_ok=True)
    schema = schema_check(config, root)
    if stage == "schema-check":
        _write_json(root, "readiness_report.json", {
            "status": "READY_FOR_SMOKE" if schema.get("stable_security_id_available") and schema.get("corporate_actions_ready") else "NOT_READY",
            "reason": schema.get("security_id_policy") if not schema.get("stable_security_id_available") else "corporate-action ledger requirements are not met",
            "full_run_permitted": False,
        })
        return root

    features = build_feature_registry(config)
    targets = build_target_registry(config)
    fp = interday_fingerprint(
        config, features, targets, git_commit_value=git_commit(),
        source_provenance={"catalog": config.catalog_path, "source_table": config.source_table},
    )
    enforce_interday_fingerprint(root, fp, resume=config.resume)
    fingerprint = fp["sha256"]
    ledger = StageLedger(root)
    if force_rebuild:
        ledger.invalidate_from(stage if stage != "all" else "source")

    try:
        with sampled_peak_memory() as memory:
            daily, checkpoint_frame, coverage, provenance = load_compact_daily_inputs(config)
        source_paths = [_write_df(root, "source_daily_inputs.parquet", daily),
                        _write_df(root, "source_checkpoint_inputs.parquet", checkpoint_frame),
                        _write_df(root, "source_coverage.parquet", coverage)]
        _write_json(root, "source_provenance.json", provenance.__dict__)
        source_paths.append(root / "source_provenance.json")
        ledger.complete("source", fingerprint, source_paths)
        if stage == "source": return root

        membership = _load_membership(config)
        panel = attach_membership_and_eligibility(
            DailyPanelBuild(daily, checkpoint_frame, coverage), membership, config)
        panel_paths = [_write_df(root, "daily_panel.parquet", panel.daily),
                       _write_df(root, "checkpoint_panel.parquet", panel.checkpoints),
                       _write_df(root, "panel_coverage.parquet", panel.coverage)]
        ledger.complete("panel", fingerprint, panel_paths)
        if stage == "panel": return root

        value_cols = [c for c in (
            "open", "high", "low", "close", "volume", "dollar_volume", "first_60m_return",
            "last_60m_return", "session_vwap", "open_30m_volume", "close_30m_volume",
            "first_60m_volume", "last_60m_volume", "largest_5m_volume", "midday",
        ) if c in panel.daily]
        dense = to_dense_panel(panel.daily, value_columns=value_cols)
        primitives = build_primitives(dense, config.benchmark_symbol)
        feature_result = deduplicate_features(build_feature_matrix(primitives, features, config))[0]
        feature_path = root / "feature_values.npy"
        write_matrix(feature_path, feature_result.values, names=feature_result.names,
                     dates=dense.sessions, security_ids=dense.security_ids,
                     fingerprint=fingerprint, schema_version=config.feature_schema_version)
        feature_paths = [feature_path, feature_path.with_suffix(".json"),
                         _write_df(root, "feature_registry.parquet", pd.DataFrame([f.__dict__ for f in feature_result.specs])),
                         _write_json(root, "feature_build_report.json", {"records": feature_result.build_records})]
        ledger.complete("features", fingerprint, feature_paths)
        if stage == "features": return root

        cp_arrays = _checkpoint_arrays(panel, dense.sessions, dense.security_ids)
        benchmark = np.where(dense.symbols == config.benchmark_symbol)[0]
        benchmark_arrays = {k: (v[:, benchmark[0]] if len(benchmark) else np.full(len(dense.sessions), np.nan, np.float32))
                            for k, v in cp_arrays.items()}
        actions = pd.read_parquet(config.corporate_actions_path) if config.corporate_actions_path else None
        target_result = build_targets(cp_arrays, benchmark_arrays, None, dense.valid, targets, config,
                                      sessions=dense.sessions, actions=actions,
                                      security_ids=dense.security_ids, symbols=dense.symbols)
        target_path = root / "target_values.npy"
        write_matrix(target_path, target_result.values, names=target_result.names,
                     dates=dense.sessions, security_ids=dense.security_ids,
                     fingerprint=fingerprint, schema_version=config.target_schema_version,
                     axis_order=("target", "date", "security"))
        target_paths = [target_path, target_path.with_suffix(".json"),
                        _write_df(root, "target_registry.parquet", pd.DataFrame([t.__dict__ for t in target_result.specs])),
                        _write_json(root, "target_build_report.json", {"records": target_result.build_records})]
        if target_result.missing_reasons is not None:
            target_paths.append(_write_json(root, "target_missing_reason_codes.json", {x.name: int(x) for x in __import__('quant_pipeline.interday.targets', fromlist=['TargetMissingReason']).TargetMissingReason}))
        ledger.complete("targets", fingerprint, target_paths)
        if stage == "targets": return root

        scan_eligible = dense.valid & (dense.symbols[None, :] != config.benchmark_symbol)
        ranks = build_rank_bin_cache(feature_result.values, scan_eligible,
                                     np.arange(len(dense.security_ids), dtype=np.int64), feature_result.names,
                                     minimum_decile_size=config.minimum_decile_cross_section_size,
                                     minimum_quintile_size=config.minimum_quintile_cross_section_size)
        rank_paths = []
        for name, values in (("feature_ranks.npy", ranks.percentile_ranks),
                             ("feature_deciles.npy", ranks.deciles), ("feature_quintiles.npy", ranks.quintiles)):
            path = root / name
            write_matrix(path, values, names=feature_result.names, dates=dense.sessions,
                         security_ids=dense.security_ids, fingerprint=fingerprint,
                         schema_version=config.rank_schema_version)
            rank_paths.extend([path, path.with_suffix(".json")])
        ledger.complete("ranks", fingerprint, rank_paths)
        if stage == "ranks": return root

        plan = choose_block_plan(n_dates=len(dense.sessions), n_symbols=len(dense.security_ids),
                                 n_features=len(feature_result.names), n_targets=len(target_result.names),
                                 config=config, available_bytes=_available_bytes(config))
        rows = []
        with sampled_peak_memory() as memory:
            for f0 in range(0, len(feature_result.names), plan.feature_block_size):
                for t0 in range(0, len(target_result.names), plan.target_block_size):
                    pair, _ = scan_feature_target_block(
                        feature_slice=slice(f0, min(f0 + plan.feature_block_size, len(feature_result.names))),
                        target_slice=slice(t0, min(t0 + plan.target_block_size, len(target_result.names))),
                        rank_cache=ranks, target_values=target_result.values,
                        feature_specs=feature_result.specs, target_specs=target_result.specs,
                        config=config, retain_daily=False)
                    rows.extend(pair)
        scan = pd.DataFrame(rows)
        scan_path = _write_df(root, "scan_results.parquet", scan)
        ledger.complete("scan", fingerprint, [scan_path])
        if stage == "scan": return root

        candidates = cluster_candidates(select_candidates(scan, config))
        candidate_path = _write_df(root, "candidates.parquet", candidates)
        ledger.complete("finalize", fingerprint, [candidate_path])
        if stage == "finalize": return root

        if candidates.empty:
            diagnostics_path = _write_json(root, "diagnostics.json", {
                "status": "complete_no_candidates", "candidate_count": 0,
                "note": "No candidate path replay was required.",
            })
            ledger.complete("diagnostics", fingerprint, [diagnostics_path],
                            metadata={"explicit_no_candidates": True})
        else:
            diagnostics_path = _write_json(root, "diagnostics.json", {
                "status": "INCOMPLETE", "candidate_count": int(len(candidates)),
                "reason": "Exact five-minute path replay, fold, concentration, and execution diagnostics required.",
            })
        if stage == "diagnostics": return root

        readiness = "READY_FOR_SMOKE" if not candidates.empty else "READY_FOR_SMOKE"
        metadata = {
            "experiment_id": config.experiment_id, "fingerprint": fingerprint,
            "git_commit": git_commit(), "discovery_end": config.discovery_end,
            "sealed_holdout_start": config.sealed_holdout_start,
            "source_rows": int(provenance.row_count), "features_built": len(feature_result.names),
            "targets_built": len(target_result.names), "scan_rows": len(scan),
            "candidate_rows": len(candidates), "scan_backend": "cuda_reductions_plus_cpu_reference" if plan.device != "cpu" else "cpu_reference",
            "peak_rss": memory["rss"], "peak_gpu_memory": memory.get("gpu", 0),
            "readiness": readiness, "full_run_permitted": False,
        }
        readiness_path = _write_json(root, "readiness_report.json", {
            "status": readiness, "full_run_permitted": False,
            "diagnostics_complete": ledger.valid("diagnostics", fingerprint),
            "reason": "Smoke gates and exact candidate diagnostics remain mandatory.",
        })
        write_report(root, scan=scan, candidates=candidates, metadata=metadata)
        report_paths = [root / "scan_results.csv", root / "candidates.csv", root / "report.json",
                        root / "report.md", readiness_path]
        ledger.complete("report", fingerprint, report_paths)
        _write_json(root, "manifest.json", metadata)
        return root
    except Exception as exc:
        _write_json(root, "failure.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc), "stage": stage})
        raise
