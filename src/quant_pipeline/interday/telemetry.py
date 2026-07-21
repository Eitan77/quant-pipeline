from __future__ import annotations
from contextlib import contextmanager
import threading,time
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
    def complete(self,stage,fingerprint,artifacts):
        import json,datetime
        p=self.marker(stage); tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps({"stage":stage,"fingerprint":fingerprint,"status":"complete","artifacts":[str(x) for x in artifacts],"completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2),encoding="utf-8"); tmp.replace(p)
    def valid(self,stage,fingerprint):
        import json
        p=self.marker(stage)
        if not p.exists(): return False
        x=json.loads(p.read_text(encoding="utf-8")); return x.get("status")=="complete" and x.get("fingerprint")==fingerprint and all(__import__('pathlib').Path(a).exists() for a in x.get("artifacts",[]))
    def invalidate_from(self,stage):
        order=list(self.STAGES); start=order.index(stage)
        for name in order[start:]:
            p=self.marker(name)
            if p.exists(): p.unlink()
