from __future__ import annotations
from contextlib import contextmanager
import threading,time
import json, hashlib, traceback
from pathlib import Path
import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None

@contextmanager
def sampled_peak_memory(interval_seconds=0.05):
    process=psutil.Process() if psutil is not None else None; stop=threading.Event(); peak={"rss":process.memory_info().rss if process else 0,"gpu":0,"gpu_reserved":0}
    def sample():
        while not stop.wait(interval_seconds):
            if process is not None: peak["rss"]=max(peak["rss"],process.memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True); thread.start()
    try:
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        except Exception: pass
        yield peak
    finally:
        stop.set(); thread.join()
        if process is not None: peak["rss"]=max(peak["rss"],process.memory_info().rss)
        try:
            import torch
            if torch.cuda.is_available(): peak["gpu"]=int(torch.cuda.max_memory_allocated()); peak["gpu_reserved"]=int(torch.cuda.max_memory_reserved())
        except Exception: pass

class StageLedger:
    STAGES=("source","panel","features","targets","ranks","scan","finalize","diagnostics","report")
    DEPENDENCIES={"source":(),"panel":("source",),"features":("panel",),"targets":("panel",),"ranks":("features",),"scan":("ranks","targets"),"finalize":("scan",),"diagnostics":("finalize","panel","features","targets","ranks"),"report":("finalize","diagnostics")}
    def __init__(self,root): self.root=root; root.mkdir(parents=True,exist_ok=True)
    def marker(self,stage): return self.root/f"stage_{stage}.json"
    def complete(self,stage,fingerprint,artifacts,*,metadata=None):
        import hashlib,json,datetime
        entries=[]
        for item in artifacts:
            path=__import__('pathlib').Path(item)
            if not path.exists(): raise FileNotFoundError(f"Missing {stage} artifact: {path}")
            h=hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
            entries.append({"path":str(path),"size":path.stat().st_size,"sha256":h.hexdigest()})
        if not entries and not (metadata or {}).get("explicit_no_candidates",False):
            raise ValueError(f"Stage {stage} cannot be complete without artifacts")
        payload={"stage":stage,"fingerprint":fingerprint,"status":"complete","artifacts":entries,"completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if metadata: payload["metadata"]=metadata
        p=self.marker(stage); tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); tmp.replace(p)
    def valid(self,stage,fingerprint):
        import json
        p=self.marker(stage)
        if not p.exists(): return False
        import hashlib
        x=json.loads(p.read_text(encoding="utf-8"))
        if x.get("status") != "complete" or x.get("fingerprint") != fingerprint: return False
        for entry in x.get("artifacts",[]):
            path=__import__('pathlib').Path(entry["path"])
            if not path.exists() or path.stat().st_size != entry.get("size"): return False
            h=hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
            if h.hexdigest() != entry.get("sha256"): return False
        return bool(x.get("artifacts")) or bool(x.get("metadata",{}).get("explicit_no_candidates"))
    def downstream_closure(self,stage):
        affected={stage}; changed=True
        while changed:
            changed=False
            for candidate,deps in self.DEPENDENCIES.items():
                if candidate not in affected and any(dep in affected for dep in deps): affected.add(candidate); changed=True
        return affected
    def invalidate_stage_and_dependents(self,stage):
        for name in self.downstream_closure(stage):
            p=self.marker(name)
            if p.exists(): p.unlink()
    def invalidate_from(self,stage): self.invalidate_stage_and_dependents(stage)

class Telemetry:
    def __init__(self,path): self.path=Path(path); self.data={}
    def record(self,stage,**metrics): self.data[stage]=metrics; self.path.write_text(json.dumps(self.data,indent=2,default=str),encoding="utf-8")

def write_failure(root: Path, *, active_stage: str, fingerprint: str|None, error: BaseException) -> None:
    payload={"status":"failed","stage":active_stage,"fingerprint":fingerprint,"exception_class":type(error).__name__,"message":str(error),"traceback":traceback.format_exc(),"timestamp":pd.Timestamp.utcnow().isoformat()}
    path=root/"failure.json"; tmp=path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); tmp.replace(path)
