"""Read-only inventory of policy exports, native API and protected queue."""
from pathlib import Path
import json
import pickle
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
import torch
torch.set_num_threads(1)
from column_review.common import digest, read
from revision_comparisons.progress import snapshot
import continue_column_queue as queue


def main():
    output = Path(__file__).resolve().parent
    data = snapshot()
    group_dir = Path(read(ROOT / "Distillation/Results/review_recovery/latest.json")["repeats"])
    for group in ("residual", "combined"):
        queue.validate_group(read(group_dir / group / "manifest.json"))
    protected = [r for r in data["jobs"] if r["state"] == "running" and r["study"] not in data["historical_studies"]]
    import win32com.client
    service = win32com.client.GetObject("winmgmts:")
    inventory = [{"pid": int(p.ProcessId), "parent": int(p.ParentProcessId), "name": str(p.Name),
                  "created": str(p.CreationDate), "command": str(p.CommandLine)}
                 for p in service.ExecQuery("SELECT ProcessId,ParentProcessId,Name,CreationDate,CommandLine FROM Win32_Process")]
    pids = {p["pid"] for p in inventory}
    assert all(r["pid"] in pids for r in protected)
    assert data["repeat_queue_coordinator"]["pid"] in pids
    checkpoints = []
    for path in sorted((ROOT / "Polymer/Results").rglob("*.pt")):
        if "threshold" not in str(path).lower():
            continue
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        def tensors(value):
            if isinstance(value, torch.Tensor):
                return value.numel()
            if isinstance(value, dict):
                return sum(tensors(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(map(tensors, value))
            return 0
        checkpoints.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size,
                            "tensor_elements": tensors(loaded), "keys": list(loaded)})
    trajectory = ROOT / "Polymer/Results/comparisons/scenario1/20261014_runtime_recovery_threshold_sg/threshold/training.pkl"
    with trajectory.open("rb") as stream:
        saved = pickle.load(stream)
    def structure(value):
        if isinstance(value, dict):
            return {str(k): structure(v) for k,v in value.items()}
        return {"type": type(value).__name__, "shape": list(value.shape) if hasattr(value, "shape") else None}
    api = {}
    import pythoncom
    import pywintypes
    library = pythoncom.LoadRegTypeLib(pywintypes.IID("{E85043BB-0E9D-11D3-82C4-00104B468DE2}"), 1, 0, 0)
    for index in range(library.GetTypeInfoCount()):
        info = library.GetTypeInfo(index)
        name = library.GetDocumentation(index)[0]
        if not any(key in name for key in ("Document", "Results", "Snapshot", "TimeSettings")):
            continue
        api[name] = {"doc": library.GetDocumentation(index), "functions": []}
        for position in range(info.GetTypeAttr().cFuncs):
            desc = info.GetFuncDesc(position)
            api[name]["functions"].append({"names": info.GetNames(desc.memid), "doc": info.GetDocumentation(desc.memid), "signature": str(desc)})
    report = {"observed_at_utc": data["observed_at_utc"], "protected_workers": protected,
              "coordinator": data["repeat_queue_coordinator"], "queued_source_and_model_guards_pass": True,
              "checkpoint_candidates": checkpoints, "trajectory": {"path": str(trajectory), "sha256": digest(trajectory), "structure": structure(saved)},
              "native_api": api, "process_inventory": [
                  {k:r[k] for k in ("pid", "parent", "name", "created")} for r in inventory
                  if r["pid"] in {j["pid"] for j in protected} | {data["repeat_queue_coordinator"]["pid"]}
                  or r["name"].lower() in {"aspenmodeler.exe", "sim_server.exe", "am_task_server.exe"}]}
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protected_workers": len(protected), "coordinator": report["coordinator"], "checkpoints": checkpoints,
                      "trajectory_possible_policy_keys": [k for k in saved if any(w in k.lower() for w in ("agent", "state_dict", "policy", "checkpoint", "network"))]}, indent=2))


if __name__ == "__main__":
    main()
