from __future__ import annotations
from dataclasses import asdict
import hashlib,json,subprocess,importlib.metadata
from pathlib import Path
from .config import InterdayConfig
from .models import InterdayFeatureSpec,InterdayTargetSpec

def git_commit() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: return "unknown"

def _file_sha256(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def interday_fingerprint(config: InterdayConfig, features, targets, *, git_commit_value: str, source_provenance: dict) -> dict:
    dependencies = {}
    for package in ("duckdb", "exchange-calendars", "numpy", "pandas", "pyarrow", "scipy", "torch"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "uninstalled"
    payload={"resolved_config":config.as_dict(),"feature_registry":[asdict(x) for x in features],"target_registry":[asdict(x) for x in targets],"git_commit":git_commit_value,"source_provenance":source_provenance,"stable_id_policy":{"security_id_column":config.security_id_column,"security_master_table":config.security_master_table,"require_stable_security_id":config.require_stable_security_id},"membership_snapshot":{"table":config.membership_table,"security_id_column":config.membership_security_id_column},"corporate_action_ledger":{"path":config.corporate_actions_path,"sha256":_file_sha256(config.corporate_actions_path)},"exchange_calendar":config.exchange_calendar,"dependency_versions":dependencies}
    encoded=json.dumps(payload,sort_keys=True,default=str).encode(); return {"sha256":hashlib.sha256(encoded).hexdigest(),"payload":payload}

def enforce_interday_fingerprint(run_root: Path, fingerprint: dict, *, resume: bool) -> None:
    path=run_root/"fingerprint.json"; run_root.mkdir(parents=True,exist_ok=True)
    if path.exists():
        saved=json.loads(path.read_text(encoding="utf-8"))
        if saved.get("sha256")!=fingerprint["sha256"]: raise RuntimeError("Interday run fingerprint changed; use a new experiment_id or rebuild")
        if not resume: raise RuntimeError("Run exists and resume is disabled")
    else:
        tmp=path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(fingerprint,indent=2,default=str),encoding="utf-8"); tmp.replace(path)
