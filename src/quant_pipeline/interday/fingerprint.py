from __future__ import annotations
from dataclasses import asdict
import hashlib,json,subprocess
from pathlib import Path
from .config import InterdayConfig
from .models import InterdayFeatureSpec,InterdayTargetSpec

def git_commit() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: return "unknown"

def interday_fingerprint(config: InterdayConfig, features, targets, *, git_commit_value: str, source_provenance: dict) -> dict:
    payload={"config":config.as_dict(),"features":[asdict(x) for x in features],"targets":[asdict(x) for x in targets],"git_commit":git_commit_value,"source_provenance":source_provenance}
    encoded=json.dumps(payload,sort_keys=True,default=str).encode(); return {"sha256":hashlib.sha256(encoded).hexdigest(),"payload":payload}

def enforce_interday_fingerprint(run_root: Path, fingerprint: dict, *, resume: bool) -> None:
    path=run_root/"fingerprint.json"; run_root.mkdir(parents=True,exist_ok=True)
    if path.exists():
        saved=json.loads(path.read_text(encoding="utf-8"))
        if saved.get("sha256")!=fingerprint["sha256"]: raise RuntimeError("Interday run fingerprint changed; use a new experiment_id or rebuild")
        if not resume: raise RuntimeError("Run exists and resume is disabled")
    else:
        tmp=path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(fingerprint,indent=2,default=str),encoding="utf-8"); tmp.replace(path)
