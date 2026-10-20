"""Read-only snapshots of the current reviewer studies, using no scientific imports."""
from __future__ import annotations

import collections
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "polymer_repeats": "Polymer/Results/phase2_repeats/20261013_010021_35e44c",
    "column_repeats": "Distillation/Results/phase2_repeats/20261014_154252_a2ed3b",
    "E1_polymer": "Polymer/Results/e1/20261013_192015_916c06",
    "E1_combined_recovery": "Polymer/Results/review_recovery/20261015_181507_3aa70b/e1_combined/20261015_181507_cf7a4e",
    "E2_polymer_SG_recovery": "Polymer/Results/review_recovery/20261015_181507_3aa70b/e2_sg",
    "E3_polymer": "Results/e3/20261015_042751_513bff",
    "E3_column": "Results/e3/20261015_042751_4e03da",
}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def process_info(pid):
    """Query process lifetime and accumulated CPU; never signal a process."""
    if not isinstance(pid, int) or pid <= 0 or os.name != "nt":
        return {"alive": None, "cpu_seconds": None}
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return {"alive": False if ctypes.get_last_error() == 87 else None, "cpu_seconds": None}
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return {"alive": None, "cpu_seconds": None}
        times = [wintypes.FILETIME() for _ in range(4)]
        cpu = None
        if kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            cpu = sum((t.dwHighDateTime << 32) + t.dwLowDateTime for t in times[2:]) / 1e7
        return {"alive": code.value == 259, "cpu_seconds": cpu}
    finally:
        kernel.CloseHandle(handle)


def log_tail(path, count=16000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size-count))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def snapshot(root=ROOT):
    rows = []
    studies = dict(STUDIES)
    recovery = read_json(Path(root) / "Distillation/Results/review_recovery/latest.json")
    if recovery.get("repeats"):
        studies["column_repeats"] = recovery["repeats"]
    for path in sorted((Path(root)/"Polymer/Results/review_recovery").glob("*/concurrent_plan.json")):
        if not read_json(path).get("smoke", True):
            studies["Polymer_remaining_" + path.parent.name] = str(path.parent)
    for path in sorted((Path(root)/"Distillation/Results/column_review").glob("*/prepared_plan.json")):
        studies["Column_review_" + path.parent.name] = str(path.parent)
    for name, relative in studies.items():
        directory = Path(root) / relative
        manifest = read_json(directory / "manifest.json")
        if not manifest:
            if name == "E2_polymer_SG_recovery":
                prior_exit = read_json(directory.parent/"e1_combined_exit.json")
                rows.append({"study": name, "job": "sg", "state": "blocked_by_E1_failure" if prior_exit.get("returncode") else "queued_after_E1_combined",
                             "pid": None, "cpu_seconds": None, "cycle": None})
            continue
        jobs = manifest.get("jobs", [])
        if "jobs" not in manifest and "methods" in manifest:
            jobs = [{"id": method, "state": read_json(directory/method/"status.json").get("state", "pending"),
                     "attempts": [{"directory": str((directory/method).resolve())}]} for method in manifest["methods"]]
        for job in jobs:
            attempts = [a if isinstance(a, dict) else {"directory": a}
                        for a in job.get("attempts", [])]
            attempt = attempts[-1] if attempts else {}
            path = Path(attempt["directory"]) if attempt.get("directory") else None
            if path is not None and not path.is_absolute():
                path = directory / path
            status = {}
            if path:
                status = read_json(path/"status.json") or read_json(path/"worker_status.json")
            execution = read_json(path/"execution.json") if path and name.startswith("Polymer_remaining_") else {}
            state = (status.get("state") or execution.get("state") or job.get("state", "pending")) if execution else job.get("state", status.get("state", "pending"))
            if state == "running" and status.get("state") in ("failed", "completed"):
                state = status["state"]
            pid = attempt.get("pid", status.get("pid", execution.get("pid")))
            info = process_info(pid) if state == "running" else {"alive": None, "cpu_seconds": None}
            if state == "running" and info["alive"] is False:
                state = "stopped_without_final_status"
            log = (path/"worker.log" if (path/"worker.log").exists() else path/"run.log") if path else None
            tail = log_tail(log) if log else ""
            cycles = re.findall(r"Sub_Episode:\s*(\d+)", tail)
            row = {"study": name, "job": job.get("id") or "/".join(str(job.get(k)) for k in ("category", "mode", "seed")),
                   "state": state, "pid": pid, **info, "cycle": int(cycles[-1]) if cycles else None,
                   "directory": str(path) if path else None,
                   "aspen_number": job.get("aspen_number"),
                   "error": status.get("error") or attempt.get("error"),
                   "historical_failed_attempts": sum(a.get("state") == "failed" for a in attempts[:-1]),
                   "last_log_at": datetime.fromtimestamp(log.stat().st_mtime).isoformat() if log and log.exists() else None}
            if path:
                checkpoint = read_json(path/"training_checkpoints/latest.json")
                if checkpoint:
                    row["checkpoint_cycle"] = checkpoint.get("completed_cycles")
                for candidate in (path/"capture"/"decisions.jsonl", path/"decisions"/"decisions.jsonl"):
                    if candidate.exists():
                        records = []
                        for line in candidate.read_text(encoding="utf-8").splitlines():
                            try:
                                records.append(json.loads(line))
                            except ValueError:
                                pass  # the active writer can have a partial final line
                        row["matched_decisions"] = len(records)
                        row["completed_matched_decisions"] = sum(r.get("status") == "completed" for r in records)
                        row["last_matched_step"] = records[-1].get("step") if records else None
                        row["expected_matched_decisions"] = 1600 if job.get("category") == "combined" else 400
                captures = path/"capture"/"records.jsonl"
                if captures.exists():
                    row["prediction_origins"] = sum(bool(line.strip()) for line in captures.open(encoding="utf-8"))
            rows.append(row)
    historical = []
    polymer_current = [r for r in rows if r["study"].startswith("Polymer_remaining_")]
    if polymer_current:
        historical += ["E1_polymer", "E1_combined_recovery", "E2_polymer_SG_recovery"]
    if recovery.get("column"):
        historical += [name for name, path in studies.items() if name.startswith("Column_review_")
                       and Path(path).resolve() != Path(recovery["column"]).resolve()]
    coordinator = {}
    if recovery.get("repeat_recovery_plan"):
        plan_directory = Path(recovery["repeat_recovery_plan"]).parent
        coordinator = read_json(plan_directory / "coordinator_status.json")
        if coordinator:
            info = process_info(coordinator.get("pid"))
            coordinator = {**coordinator, **info, "directory": str(plan_directory)}
            if coordinator["state"] == "running" and info["alive"] is False:
                coordinator["state"] = "stopped_without_final_status"
    return {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": rows,
            "historical_studies": historical, "active_recovery": recovery.get("directory"),
            "repeat_queue_coordinator": coordinator,
            "counts": {name: dict(collections.Counter(r["state"] for r in rows if r["study"] == name)) for name in studies}}


def display(data):
    observed = datetime.fromisoformat(data["observed_at_utc"]).astimezone()
    historical = set(data.get("historical_studies", []))
    print("\nReviewer studies: " + observed.strftime("%Y-%m-%d %H:%M:%S %Z"), flush=True)
    for name, counts in data["counts"].items():
        if name in historical:
            continue
        print(f"  {name}: {counts}", flush=True)
    coordinator = data.get("repeat_queue_coordinator", {})
    if coordinator:
        print(f"  Column queue: {coordinator['state']} stage={coordinator.get('stage')} PID={coordinator.get('pid')}", flush=True)
        if coordinator.get("error"):
            print(f"  Queue requires attention: {coordinator['error']}", flush=True)
    if "E1_polymer" not in historical and data["counts"].get("E1_polymer", {}).get("failed"):
        print("  Original E1 manifest retains the residual export failure; its 400 pairs survive. The separate recovery row tracks combined.", flush=True)
    for row in data["jobs"]:
        if row["study"] in historical:
            continue
        if row["state"] not in ("running", "failed", "stopped_without_final_status"):
            continue
        cpu = str(row['cpu_seconds'])+'s' if row['cpu_seconds'] is not None else 'unavailable'
        detail = f"cycle={row['cycle']} PID={row['pid']} CPU={cpu}"
        if row.get("aspen_number") is not None:
            detail += f" Aspen={row['aspen_number']}"
        if "matched_decisions" in row:
            detail += f" matched={row['completed_matched_decisions']}/{row['expected_matched_decisions']} step={row.get('last_matched_step')}"
        if "prediction_origins" in row:
            detail += f" prediction_origins={row['prediction_origins']}"
        if "checkpoint_cycle" in row:
            detail += f" full_training_checkpoint_cycle={row['checkpoint_cycle']}"
        print(f"  {row['study']} {row['job']}: {row['state']} {detail}", flush=True)
    if historical:
        print(f"  {len(historical)} superseded study records retained in the JSON audit; current recoveries shown above.", flush=True)
    print("Closing this viewer does not stop a study. Failed attempts remain saved.", flush=True)
