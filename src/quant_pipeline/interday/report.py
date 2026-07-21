from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def write_report(root: Path, *, scan: pd.DataFrame, candidates: pd.DataFrame, metadata: dict):
    root.mkdir(parents=True,exist_ok=True)
    scan.to_csv(root/"scan_results.csv",index=False)
    candidates.to_csv(root/"candidates.csv",index=False)
    (root/"report.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    required = [
        "resolved_config.yaml", "fingerprint.json", "manifest.json", "readiness_report.json",
        "source_schema.json", "dependency_versions.json", "calendar_contract.json", "source_provenance.json",
        "panel_coverage.csv", "feature_coverage.csv", "target_coverage.csv", "feature_build_report.json",
        "target_build_report.json", "feature_registry.parquet", "target_registry.parquet", "scan_plan.json",
        "scan_results.csv", "candidates.csv", "candidate_summary.csv", "candidate_daily_series.parquet",
        "candidate_exact_diagnostics.parquet", "performance_metrics.json", "run_journal.json", "report.md",
    ]
    (root/"report.md").write_text(
        "# Interday 2A report\n\n"
        f"- Readiness: `{metadata.get('readiness','UNKNOWN')}`\n"
        f"- Features: `{metadata.get('features_built', 0)}`\n"
        f"- Targets: `{metadata.get('targets_built', 0)}`\n"
        f"- Scan rows: `{metadata.get('scan_rows', 0)}`\n"
        f"- Candidates: `{metadata.get('candidate_rows', 0)}`\n\n"
        "Required artifact set:\n" + "\n".join(f"- `{x}`" for x in required) + "\n",
        encoding="utf-8")
