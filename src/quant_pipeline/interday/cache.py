from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

INTERDAY_ROW_KEYS = ("security_id","session_date")
class InterdayCacheMismatch(RuntimeError): pass

def interday_row_key_hash(frame: pd.DataFrame, row_keys=INTERDAY_ROW_KEYS) -> str:
    keys=frame.loc[:,list(row_keys)].copy(); keys["session_date"]=pd.to_datetime(keys.session_date).astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(keys,index=False).to_numpy().tobytes()).hexdigest()

def _digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def write_interday_cache(path: Path, frame: pd.DataFrame, *, fingerprint: str, schema_version: str, sealed_holdout_start: str, row_keys=INTERDAY_ROW_KEYS, extra_metadata=None) -> dict:
    path.parent.mkdir(parents=True,exist_ok=True); keys=frame.loc[:,list(row_keys)]
    if keys.duplicated(list(row_keys)).any(): raise ValueError(f"Duplicate interday cache keys: {path}")
    sorted_frame=frame.sort_values(list(row_keys),kind="stable")
    if not sorted_frame.index.equals(frame.index): raise ValueError(f"Interday cache keys must be sorted: {path}")
    if len(frame) and pd.to_datetime(frame.session_date).max() >= pd.Timestamp(sealed_holdout_start): raise ValueError("Cache reaches sealed holdout")
    tmp=path.with_suffix(path.suffix+".tmp"); frame.to_parquet(tmp,index=False); tmp.replace(path)
    metadata={"fingerprint":fingerprint,"schema_version":schema_version,"row_keys":list(row_keys),"row_count":len(frame),"row_key_hash":interday_row_key_hash(frame,row_keys),"file_size":path.stat().st_size,"file_sha256":_digest(path)}
    if extra_metadata: metadata.update(extra_metadata)
    mp=path.with_suffix(path.suffix+".meta.json"); mt=mp.with_suffix(mp.suffix+".tmp"); mt.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8"); mt.replace(mp); return metadata

def validate_interday_cache(path: Path, *, fingerprint: str, schema_version: str, sealed_holdout_start: str) -> dict:
    mp=path.with_suffix(path.suffix+".meta.json")
    if not path.exists() or not mp.exists(): raise InterdayCacheMismatch(f"Cache or metadata missing: {path}")
    saved=json.loads(mp.read_text(encoding="utf-8"))
    if saved["fingerprint"]!=fingerprint or saved["schema_version"]!=schema_version: raise InterdayCacheMismatch(f"Cache identity mismatch: {path}")
    if saved["file_size"]!=path.stat().st_size or saved["file_sha256"]!=_digest(path): raise InterdayCacheMismatch(f"Cache digest mismatch: {path}")
    keys=pd.read_parquet(path,columns=saved["row_keys"])
    if len(keys) and pd.to_datetime(keys.session_date).max()>=pd.Timestamp(sealed_holdout_start): raise InterdayCacheMismatch("Cache reaches holdout")
    if interday_row_key_hash(keys,saved["row_keys"])!=saved["row_key_hash"]: raise InterdayCacheMismatch("Cache row-key mismatch")
    if pq.ParquetFile(path).metadata.num_rows != saved["row_count"]: raise InterdayCacheMismatch("Cache row count mismatch")
    return saved

def write_matrix(path: Path, array: np.ndarray, *, names: list[str], dates: pd.DatetimeIndex, security_ids: np.ndarray, fingerprint: str, schema_version: str) -> dict:
    path.parent.mkdir(parents=True,exist_ok=True); actual=path if path.suffix==".npy" else path.with_suffix(".npy"); np.save(actual,array,allow_pickle=False)
    meta={"shape":list(array.shape),"dtype":str(array.dtype),"names":names,"date_hash":hashlib.sha256(pd.util.hash_pandas_object(pd.Series(dates),index=False).to_numpy().tobytes()).hexdigest(),"symbol_hash":hashlib.sha256("|".join(map(str,security_ids)).encode()).hexdigest(),"fingerprint":fingerprint,"schema_version":schema_version,"file_sha256":_digest(actual)}
    actual.with_suffix(".json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8"); return meta
