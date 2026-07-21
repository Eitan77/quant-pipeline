from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def write_report(root: Path, *, scan: pd.DataFrame, candidates: pd.DataFrame, metadata: dict):
    root.mkdir(parents=True,exist_ok=True)
    scan.to_csv(root/"scan_results.csv",index=False)
    candidates.to_csv(root/"candidates.csv",index=False)
    (root/"report.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    required = ["source_schema.json", "dependency_versions.json", "calendar_contract.json",
                "feature_registry.parquet", "target_registry.parquet", "scan_results.csv",
                "candidates.csv", "readiness_report.json"]
    (root/"report.md").write_text(
        "# Interday 2A report\n\n"
        f"- Readiness: `{metadata.get('readiness','UNKNOWN')}`\n"
        f"- Features: `{metadata.get('features_built', 0)}`\n"
        f"- Targets: `{metadata.get('targets_built', 0)}`\n"
        f"- Scan rows: `{metadata.get('scan_rows', 0)}`\n"
        f"- Candidates: `{metadata.get('candidate_rows', 0)}`\n\n"
        "Required artifact set:\n" + "\n".join(f"- `{x}`" for x in required) + "\n",
        encoding="utf-8")
