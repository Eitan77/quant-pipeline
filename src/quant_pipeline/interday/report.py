from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def write_report(root: Path, *, scan: pd.DataFrame, candidates: pd.DataFrame, metadata: dict):
    root.mkdir(parents=True,exist_ok=True); scan.to_csv(root/"scan_results.csv",index=False); candidates.to_csv(root/"candidates.csv",index=False); (root/"report.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
