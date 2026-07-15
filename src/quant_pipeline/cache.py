from __future__ import annotations

import hashlib,json
from pathlib import Path
import pandas as pd

from .holdout import assert_pre_holdout_frame, assert_pre_holdout_parquet

ROW_KEYS=["symbol","session_date","bar_start_ts","decision_ts"]


class CacheFingerprintMismatch(RuntimeError):pass


def row_key_hash(frame:pd.DataFrame)->str:
    keys=frame[ROW_KEYS].copy(); keys["session_date"]=pd.to_datetime(keys.session_date).astype(str)
    for column in ["bar_start_ts","decision_ts"]:keys[column]=pd.to_datetime(keys[column],utc=True).astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(keys,index=False).to_numpy().tobytes()).hexdigest()


def schema_hash(frame:pd.DataFrame)->str:
    return hashlib.sha256(json.dumps([(c,str(t)) for c,t in frame.dtypes.items()]).encode()).hexdigest()


def write_cache_metadata(path:Path,frame:pd.DataFrame,fingerprint:str,sealed_holdout_start:str="2026-05-01")->dict:
    assert_pre_holdout_frame(frame,sealed_holdout_start,f"cache write {path}")
    assert_pre_holdout_parquet(path,sealed_holdout_start,f"cache write {path}")
    persisted=pd.read_parquet(path)
    if persisted.duplicated(ROW_KEYS).any():raise ValueError(f"Duplicate cache keys: {path}")
    metadata={"fingerprint":fingerprint,"row_count":len(persisted),"first_key":[str(x) for x in persisted[ROW_KEYS].iloc[0]] if len(persisted) else None,"last_key":[str(x) for x in persisted[ROW_KEYS].iloc[-1]] if len(persisted) else None,"row_key_hash":row_key_hash(persisted),"column_schema_hash":schema_hash(persisted)}
    path.with_suffix(path.suffix+".meta.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); return metadata


def validate_cache(path:Path,fingerprint:str,sealed_holdout_start:str="2026-05-01")->dict:
    assert_pre_holdout_parquet(path,sealed_holdout_start,f"cache resume {path}")
    metadata_path=path.with_suffix(path.suffix+".meta.json")
    if not metadata_path.exists():raise CacheFingerprintMismatch(f"Cache metadata missing: {path}")
    saved=json.loads(metadata_path.read_text(encoding="utf-8"))
    if saved["fingerprint"]!=fingerprint:raise CacheFingerprintMismatch(f"Cache fingerprint mismatch: {path}")
    frame=pd.read_parquet(path); current={"row_count":len(frame),"row_key_hash":row_key_hash(frame),"column_schema_hash":schema_hash(frame)}
    assert_pre_holdout_frame(frame,sealed_holdout_start,f"cache resume {path}")
    if any(saved[k]!=current[k] for k in current):raise CacheFingerprintMismatch(f"Cache integrity check failed: {path}")
    if frame.duplicated(ROW_KEYS).any() or not frame.sort_values(ROW_KEYS,kind="stable").index.equals(frame.index):raise CacheFingerprintMismatch(f"Cache keys are duplicate or unsorted: {path}")
    return saved


def assert_key_alignment(feature_frame:pd.DataFrame,target_frame:pd.DataFrame)->None:
    if len(feature_frame)!=len(target_frame) or row_key_hash(feature_frame)!=row_key_hash(target_frame):raise CacheFingerprintMismatch("Feature and target cache row keys do not align")


def assert_cache_key_alignment(feature_path:Path,target_path:Path)->None:
    """Use already-validated cache metadata instead of rehashing 12M rows per batch."""
    def metadata(path:Path)->dict:
        metadata_path=path.with_suffix(path.suffix+".meta.json")
        if not metadata_path.exists():raise CacheFingerprintMismatch(f"Cache metadata missing: {path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    feature=metadata(feature_path); target=metadata(target_path)
    if feature.get("row_count")!=target.get("row_count") or feature.get("row_key_hash")!=target.get("row_key_hash"):
        raise CacheFingerprintMismatch("Feature and target cache metadata keys do not align")
