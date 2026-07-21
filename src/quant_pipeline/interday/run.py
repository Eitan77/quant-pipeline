from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .calendar import TradingCalendar
from .config import InterdayConfig
from .source import load_projected_bars, load_compact_daily_inputs, schema_check
from .panel import build_daily_panel, attach_membership_and_eligibility
from .primitives import to_dense_panel, build_primitives
from .registry import build_feature_registry, build_target_registry
from .features import build_feature_matrix, deduplicate_features
from .targets import build_targets
from .ranking import build_rank_bin_cache
from .scan import scan_feature_target_block_cpu, scan_feature_target_block_gpu
from .candidates import select_candidates, cluster_candidates
from .fingerprint import enforce_interday_fingerprint, interday_fingerprint, git_commit
from .telemetry import StageLedger, sampled_peak_memory
from .report import write_report

def _write_df(root,name,frame):
    path=root/name; frame.to_parquet(path,index=False); return path

def execute_interday_2a(config: InterdayConfig, *, stage="all", force_rebuild=False) -> Path:
    config.validate(); root=config.run_root; root.mkdir(parents=True,exist_ok=True)
    schema_check(config,root)
    if stage=="schema-check":
        return root
    features=build_feature_registry(config); targets=build_target_registry(config); fp=interday_fingerprint(config,features,targets,git_commit_value=git_commit(),source_provenance={"catalog":config.catalog_path,"source_table":config.source_table}); enforce_interday_fingerprint(root,fp,resume=config.resume)
    ledger=StageLedger(root); fingerprint=fp["sha256"]
    if force_rebuild: ledger.invalidate_from(stage if stage!="all" else "source")
    if stage=="schema-check": return root
    with sampled_peak_memory() as memory:
        daily, checkpoint_frame, coverage, provenance=load_compact_daily_inputs(config)
    _write_df(root,"source_daily_inputs.parquet",daily); _write_df(root,"source_checkpoint_inputs.parquet",checkpoint_frame); ledger.complete("source",fingerprint,[root/"source_daily_inputs.parquet",root/"source_checkpoint_inputs.parquet"])
    if stage=="source": return root
    calendar=TradingCalendar(config.exchange_calendar); panel=type("Panel",(),{"daily":daily,"checkpoints":checkpoint_frame,"coverage":coverage})()
    try:
        con=__import__('duckdb').connect(config.catalog_path,read_only=True); membership=con.execute(f"select date,symbol,is_member from {config.membership_table}").fetchdf(); con.close()
    except Exception: membership=pd.DataFrame()
    panel=attach_membership_and_eligibility(panel,membership,config); _write_df(root,"daily_panel.parquet",panel.daily); _write_df(root,"checkpoint_panel.parquet",panel.checkpoints); _write_df(root,"panel_coverage.parquet",panel.coverage); ledger.complete("panel",fingerprint,[root/"daily_panel.parquet",root/"checkpoint_panel.parquet",root/"panel_coverage.parquet"])
    if stage=="panel": return root
    value_cols=[c for c in ["open","high","low","close","volume","dollar_volume","first_60m_return","last_60m_return","session_vwap","open_30m_volume","close_30m_volume"] if c in panel.daily]
    dense=to_dense_panel(panel.daily,value_columns=value_cols); primitives=build_primitives(dense,config.benchmark_symbol); feature_result=deduplicate_features(build_feature_matrix(primitives,features,config))[0]
    np.save(root/"feature_values.npy",feature_result.values); _write_df(root,"feature_registry.parquet",pd.DataFrame([f.__dict__ for f in feature_result.specs])); (root/"feature_build_report.json").write_text(json.dumps(feature_result.build_records,indent=2),encoding="utf-8"); ledger.complete("features",fingerprint,[root/"feature_values.npy",root/"feature_registry.parquet"])
    if stage=="features": return root
    cp_arrays={c:np.full((len(dense.sessions),len(dense.security_ids)),np.nan,np.float32) for c in [x for x in panel.checkpoints.columns if x in ("open5","open15","09:40","09:45","10:00","10:15","10:30","11:00","12:00","13:00","14:00","15:00","close15","close5")]}
    lookup={(r.security_id,pd.Timestamp(r.session_date)):i for i,r in panel.checkpoints.iterrows()}
    for (sid,date),i in lookup.items():
        d=dense.sessions.get_loc(date); s=np.where(dense.security_ids==str(sid))[0][0]
        for c in cp_arrays: cp_arrays[c][d,s]=panel.checkpoints.loc[i,c]
    bench=np.where(dense.symbols==config.benchmark_symbol)[0]
    bench_arrays={c:(cp_arrays[c][:,bench[0]] if len(bench) else np.full(len(dense.sessions),np.nan,np.float32)) for c in cp_arrays}
    tr=build_targets(cp_arrays,bench_arrays,None,dense.valid,targets,config); np.save(root/"target_values.npy",tr.values); np.save(root/"market_returns.npy",tr.aligned_market_returns); _write_df(root,"target_registry.parquet",pd.DataFrame([t.__dict__ for t in tr.specs])); ledger.complete("targets",fingerprint,[root/"target_values.npy",root/"market_returns.npy",root/"target_registry.parquet"])
    if stage=="targets": return root
    ranks=build_rank_bin_cache(feature_result.values,dense.valid,np.arange(len(dense.security_ids),dtype=np.int64),feature_result.names,minimum_decile_size=config.minimum_decile_cross_section_size,minimum_quintile_size=config.minimum_quintile_cross_section_size); np.save(root/"feature_ranks.npy",ranks.percentile_ranks); np.save(root/"feature_deciles.npy",ranks.deciles); np.save(root/"feature_quintiles.npy",ranks.quintiles); ledger.complete("ranks",fingerprint,[root/"feature_ranks.npy",root/"feature_deciles.npy",root/"feature_quintiles.npy"])
    if stage=="ranks": return root
    rows=[]
    try:
        import torch
        use_gpu=bool(config.use_cuda and torch.cuda.is_available())
    except Exception:
        use_gpu=False
    scan_block=scan_feature_target_block_gpu if use_gpu else scan_feature_target_block_cpu
    with sampled_peak_memory() as memory:
        for fi in range(len(feature_result.names)):
            for ti in range(len(tr.names)):
                pair,_=scan_block(feature_ids=[fi],target_ids=[ti],rank_cache=ranks,target_values=tr.values,feature_specs=feature_result.specs,target_specs=tr.specs,config=config,retain_daily=False); rows.extend(pair)
    scan=pd.DataFrame(rows); scan.to_parquet(root/"scan_results.parquet",index=False); ledger.complete("scan",fingerprint,[root/"scan_results.parquet"])
    if stage=="scan": return root
    candidates=cluster_candidates(select_candidates(scan,config)); candidates.to_parquet(root/"candidates.parquet",index=False); ledger.complete("finalize",fingerprint,[root/"candidates.parquet"])
    if stage=="finalize": return root
    ledger.complete("diagnostics",fingerprint,[])
    if stage=="diagnostics": return root
    metadata={"experiment_id":config.experiment_id,"fingerprint":fingerprint,"git_commit":git_commit(),"discovery_end":config.discovery_end,"sealed_holdout_start":config.sealed_holdout_start,"source_rows":int(provenance.row_count),"features_built":len(feature_result.names),"targets_built":len(tr.names),"scan_rows":len(scan),"candidate_rows":len(candidates),"scan_backend":"cuda_block_with_exact_cpu_reference" if use_gpu else "cpu_reference","peak_rss":memory["rss"],"peak_gpu_memory":memory.get("gpu",0),"readiness":"READY_FOR_FULL_RUN" if all(ledger.valid(s,fingerprint) for s in ("source","panel","features","targets","ranks","scan","finalize","diagnostics")) else "NOT_READY"}
    write_report(root,scan=scan,candidates=candidates,metadata=metadata); ledger.complete("report",fingerprint,[root/"scan_results.csv",root/"candidates.csv",root/"report.json"]); (root/"manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); return root
