"""Report completed days and verify the acquisition process is actually alive."""
import json
from pathlib import Path
import psutil


def get_status(root=Path("data/miami_history")):
    root=Path(root)
    run=json.loads((root/"run_status.json").read_text(encoding="utf-8"))
    live=False
    pid=run.get("pid")
    if pid:
        try:
            process=psutil.Process(pid)
            live=process.is_running() and any("acquire_history.py" in arg for arg in process.cmdline())
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    products={}
    for product in run["products"]:
        completed=[]
        for file in (root/product).glob("*.json"):
            if file.name=="schema.json":
                continue
            item=json.loads(file.read_text(encoding="utf-8"))
            if item.get("status")=="complete" and file.with_suffix(".nc").exists():
                completed.append(item)
        completed.sort(key=lambda r:r["start"])
        days=sum(r["requested_days"] for r in completed)
        sources=sum(r["acquired_days"] for r in completed)
        products[product]={"calendar_days_processed":days,"acquired_source_days":sources,
            "missing_source_days":days-sources,"completed_through":completed[-1]["end"] if completed else None}
    return {"process_alive":live,"recorded_state":run["state"],
        "actual_state":"running" if live else ("complete" if run["state"]=="complete" else "stopped"),
        "pid":pid,"heartbeat_utc":run.get("heartbeat_utc"),"phase":run.get("phase"),
        "current_product":run.get("current_product"),"current_start":run.get("current_start"),
        "products":products}


if __name__=="__main__":
    print(json.dumps(get_status(),indent=2))
