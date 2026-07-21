from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

INTERDAY_ROW_KEYS = ("security_id","session_date")
class InterdayCacheMismatch(RuntimeError): pass

def assert_sorted_unique_keys(frame: pd.DataFrame, row_keys: tuple[str, ...]) -> None:
    keys=frame.loc[:,list(row_keys)].reset_index(drop=True)
    if keys.duplicated(list(row_keys)).any(): raise ValueError(f"Duplicate keys: {row_keys}")
    sorted_keys=keys.sort_values(list(row_keys),kind="stable").reset_index(drop=True)
    if not keys.equals(sorted_keys): raise ValueError(f"Keys are not sorted by {row_keys}")

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
    assert_sorted_unique_keys(frame,row_keys)
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

def write_matrix(path: Path, array: np.ndarray, *, names: list[str], dates: pd.DatetimeIndex,
                 security_ids: np.ndarray, fingerprint: str, schema_version: str,
                 axis_order: tuple[str, ...] = ("feature", "date", "security"), source_start: str | None = None, analysis_start: str | None = None) -> dict:
    """Write a typed matrix and an integrity manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    actual = path if path.suffix == ".npy" else path.with_suffix(".npy")
    tmp = actual.with_suffix(actual.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    tmp.replace(actual)
    names_hash=hashlib.sha256("|".join(map(str,names)).encode()).hexdigest(); date_hash=hashlib.sha256(pd.util.hash_pandas_object(pd.Series(dates),index=False).to_numpy().tobytes()).hexdigest(); security_hash=hashlib.sha256("|".join(map(str,security_ids)).encode()).hexdigest()
    meta = {
        "schema_version": schema_version, "fingerprint": fingerprint,
        "shape": list(array.shape), "dtype": str(array.dtype),
        "axis_order": list(axis_order), "names": list(names),
        "names_hash": names_hash, "date_index_hash": date_hash, "security_index_hash": security_hash, "source_start": source_start, "analysis_start": analysis_start,
        "file_size": actual.stat().st_size, "file_sha256": _digest(actual),
    }
    manifest = actual.with_suffix(".json")
    mtmp = manifest.with_suffix(manifest.suffix + ".tmp")
    mtmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    mtmp.replace(manifest)
    return meta

def validate_matrix(path: Path, *, fingerprint: str, schema_version: str,
                    shape: tuple[int, ...] | None = None,
                    axis_order: tuple[str, ...] | None = None,
                    dtype: str | None = None, names: list[str] | None = None,
                    dates: pd.DatetimeIndex | None = None,
                    security_ids: np.ndarray | None = None) -> dict:
    actual = path if path.suffix == ".npy" else path.with_suffix(".npy")
    manifest = actual.with_suffix(".json")
    if not actual.exists() or not manifest.exists():
        raise InterdayCacheMismatch(f"Matrix or manifest missing: {actual}")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved.get("fingerprint") != fingerprint or saved.get("schema_version") != schema_version:
        raise InterdayCacheMismatch(f"Matrix identity mismatch: {actual}")
    if saved.get("file_size") != actual.stat().st_size or saved.get("file_sha256") != _digest(actual):
        raise InterdayCacheMismatch(f"Matrix digest mismatch: {actual}")
    if shape is not None and tuple(saved.get("shape", ())) != tuple(shape):
        raise InterdayCacheMismatch(f"Matrix shape mismatch: {actual}")
    if axis_order is not None and tuple(saved.get("axis_order", ())) != tuple(axis_order):
        raise InterdayCacheMismatch(f"Matrix axis-order mismatch: {actual}")
    if dtype is not None and saved.get("dtype") != dtype: raise InterdayCacheMismatch(f"Matrix dtype mismatch: {actual}")
    if names is not None and saved.get("names_hash") != hashlib.sha256("|".join(map(str,names)).encode()).hexdigest(): raise InterdayCacheMismatch(f"Matrix names mismatch: {actual}")
    if dates is not None and saved.get("date_index_hash") != hashlib.sha256(pd.util.hash_pandas_object(pd.Series(dates),index=False).to_numpy().tobytes()).hexdigest(): raise InterdayCacheMismatch(f"Matrix dates mismatch: {actual}")
    if security_ids is not None and saved.get("security_index_hash") != hashlib.sha256("|".join(map(str,security_ids)).encode()).hexdigest(): raise InterdayCacheMismatch(f"Matrix security IDs mismatch: {actual}")
    np.load(actual,mmap_mode="r",allow_pickle=False)
    return saved
