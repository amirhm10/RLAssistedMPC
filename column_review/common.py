"""Per-case ownership, immutable provenance and atomic status for new jobs."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {"sg": 12, "threshold": 13, "ls": 14, "bo": 15, "e1_pilot": 16}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def own_sources():
    files = [ROOT / "run_distillation_remaining.py", ROOT / "run_polymer_e3_recovery.py",
             *[ROOT / "column_review" / name for name in ("__init__.py", "common.py", "pilot.py", "typelib.py")]]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in files}


def check_job(job):
    from revision_comparisons.column import source_hashes
    if job["method"] not in MAPPING or job["aspen_number"] != MAPPING[job["method"]]:
        raise ValueError("This batch is restricted to the fixed method-to-case mapping 12--16.")
    if Path(job["aspen"]["aspen_path"]).name != f"C2S_SS_simulation{job['aspen_number']}.dynf":
        raise ValueError("Aspen filename differs from the assigned case number.")
    if job["parallel_source_hashes"] != own_sources() or job["source_hashes"] != source_hashes():
        raise RuntimeError("Prepared scientific/launcher dependencies changed; inspect before a fresh preparation.")
    if digest(job["aspen"]["aspen_path"]) != job["aspen"]["aspen_sha256"]:
        raise RuntimeError("Assigned Aspen model changed after preparation.")


def process_identity(pid):
    """Use creation time as well as PID, so reuse cannot identify another app."""
    if os.name != "nt" or not isinstance(pid, int) or pid <= 0:
        return {"pid": pid, "alive": None, "created": None}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return {"pid": pid, "alive": False if ctypes.get_last_error() == 87 else None, "created": None}
    try:
        code = wintypes.DWORD()
        stamps = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return {"pid": pid, "alive": None, "created": None}
        valid = kernel.GetProcessTimes(handle, *[ctypes.byref(stamp) for stamp in stamps])
        created = ((stamps[0].dwHighDateTime << 32) + stamps[0].dwLowDateTime) if valid else None
        return {"pid": pid, "alive": code.value == 259, "created": created}
    finally:
        kernel.CloseHandle(handle)


def owned_apps_remaining(path):
    unresolved = []
    ownership = read(Path(path).with_name("ownership_status.json"), {})
    if ownership.get("phase") in {"creating", "unverified_creation"}:
        unresolved.append({"ownership_unverified": True, "phase": ownership["phase"]})
    for entry in read(path, []):
        current = process_identity(entry["pid"])
        if current["alive"] is False:
            continue
        if current["alive"] is True and current["created"] is not None and current["created"] != entry["created"]:
            continue  # PID now belongs to another process; it is never touched.
        unresolved.append({"recorded": entry, "current": current})
    return unresolved


@contextmanager
def case_lease(paths):
    """Hold the canonical case lock across ALL BO trials/calibration resets."""
    import run_distillation_phase2_repeats as repeat
    actual = repeat.aspen_file_lock
    target = Path(paths["aspen_path"]).resolve()
    thread = threading.get_ident()

    @contextmanager
    def already_owned(path, directory):
        if Path(path).resolve() != target or threading.get_ident() != thread:
            raise RuntimeError("Worker requested an Aspen resource outside its reserved case/thread.")
        yield

    with actual(target, ROOT / "Distillation/Results/_aspen_file_locks"):
        with patch.object(repeat, "aspen_file_lock", already_owned):
            yield


@contextmanager
def record_owned_apps(output):
    import win32com.client
    original = win32com.client.DispatchEx
    records = []
    ownership_path = Path(output) / "ownership_status.json"
    write(ownership_path, {"phase": "ready", "updated_at": now()})

    def create(progid, *args, **kwargs):
        if str(progid).strip().lower() != "ad application":
            raise RuntimeError(f"Unexpected external application requested: {progid}")
        write(ownership_path, {"phase": "creating", "updated_at": now()})
        try:
            app = original(progid, *args, **kwargs)
        except BaseException:
            write(ownership_path, {"phase": "unverified_creation", "updated_at": now()})
            raise
        try:
            identity = process_identity(int(app.processId))
            if identity["alive"] is not True or identity["created"] is None:
                raise RuntimeError("Cannot establish ownership of the newly created Aspen application.")
            records.append({**identity, "created_at": now()})
            write(Path(output) / "owned_aspen_apps.json", records)
            write(ownership_path, {"phase": "tracked", "updated_at": now()})
        except BaseException:
            write(ownership_path, {"phase": "unverified_creation", "updated_at": now()})
            app.Quit()  # Only the fresh DispatchEx object owned by this call.
            raise
        return app

    with patch.object(win32com.client, "DispatchEx", create):
        yield
