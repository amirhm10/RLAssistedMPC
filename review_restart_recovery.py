"""Recover the nine interrupted reviewer studies through an interactive Windows task.

The task is demand-only, under the current user's interactive token, with no
trigger or time limit. Original scientific workers and settings are reused.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
COLUMN = ROOT / "Distillation/Results/column_review/20261016_prepared_cases12_16"
POLYMER = ROOT / "Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent"
E3 = ROOT / "Results/e3/20261015_042751_4e03da"
STAGES = ("e1_combined", "e2_sg", "e2_arx_repaired")
SURVIVORS = (47608, 43012, 47732, 46060, 32904, 34084, 43516, 15260, 40928, 48508, 49892)
OLD_STUDY_PIDS = (2360, 16924, 41176, 37840, 39000, 21724, 46452, 31404,
                  51024, 27272, 23104, 35276, 22612, 43300, 29248, 40760, 40604, 32748)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def lifetime():
    import win32com.client
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)]
    inside = w.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)):
        raise ctypes.WinError(ctypes.get_last_error())
    session = w.DWORD()
    if not kernel.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
        raise ctypes.WinError(ctypes.get_last_error())
    service_pid = int(win32com.client.GetObject("winmgmts:").Get('Win32_Service.Name="Schedule"').ProcessId)
    return {"pid": os.getpid(), "parent_pid": os.getppid(), "in_windows_job": bool(inside.value),
            "scheduler_service_pid": service_pid, "parent_is_scheduler_service": os.getppid() == service_pid,
            "session_id": session.value, "python": sys.executable, "observed_at": now()}


def isolated_interactive_context(identity):
    # Task Scheduler may use its own job object. Its service parent, outside
    # Codex's process tree, and a nonzero interactive session are the contract.
    return identity.get("parent_is_scheduler_service") is True and identity.get("session_id", 0) > 0


def schedule(name, arguments):
    import win32api
    import win32con
    import win32security
    import win32com.client
    service = win32com.client.Dispatch("Schedule.Service")
    service.Connect()
    folder = service.GetFolder("\\")
    definition = service.NewTask(0)
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    definition.RegistrationInfo.Description = "Immediate author-approved reviewer-study recovery; no recurring trigger."
    definition.Principal.UserId = user
    definition.Principal.LogonType = 3  # TASK_LOGON_INTERACTIVE_TOKEN; no password saved.
    definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.Enabled = True
    settings.AllowDemandStart = True
    settings.DisallowStartIfOnBatteries = False
    settings.StopIfGoingOnBatteries = False
    settings.ExecutionTimeLimit = "PT0S"
    settings.MultipleInstances = 2
    action = definition.Actions.Create(0)
    action.Path = str(Path(sys.executable).with_name("pythonw.exe"))
    action.Arguments = subprocess.list2cmdline(["-B", "-u", str(Path(__file__).resolve()), *arguments])
    action.WorkingDirectory = str(ROOT)
    task = folder.RegisterTaskDefinition(name, definition, 2, user, None, 3)  # create only
    task.Run(None)
    return {"task_name": name, "user": user, "logon_type": "interactive_token",
            "run_level": "current_user", "triggers": 0, "execution_time_limit": "PT0S"}


def preflight():
    from column_review.common import MAPPING, check_job, owned_apps_remaining, process_identity
    import run_e3_prediction_quality as e3
    import run_polymer_remaining as polymer
    from revision_recovery.checkpoints import digest
    identities = {str(pid): process_identity(pid) for pid in (*SURVIVORS, *OLD_STUDY_PIDS)}
    if any(identities[str(pid)]["alive"] is not True for pid in SURVIVORS):
        raise RuntimeError("A surviving repetition worker changed; inspect rather than duplicate it.")
    if any(identities[str(pid)]["alive"] is not False for pid in OLD_STUDY_PIDS):
        raise RuntimeError("An old study process is live or unverified; do not duplicate it.")
    for method in MAPPING:
        stage = COLUMN / method
        check_job(read(stage / "job.json"))
        for attempt in read(stage / "attempts.json"):
            if owned_apps_remaining(Path(attempt["directory"]) / "owned_aspen_apps.json"):
                raise RuntimeError(f"Unresolved Aspen ownership: {method}")
    for stage in STAGES:
        job = read(POLYMER / stage / "job.json")
        if job["recovery_source_hashes"] != polymer.own_sources():
            raise RuntimeError("Original polymer recovery source inventory changed.")
    index = read(POLYMER / "e1_combined/training_checkpoints/latest.json")
    job = read(POLYMER / "e1_combined/job.json")
    if index["source_hashes"] != job["recovery_source_hashes"] | job["source_hashes"]:
        raise RuntimeError("E1 checkpoint source inventory differs.")
    checkpoint = POLYMER / "e1_combined/training_checkpoints" / index["file"]
    if checkpoint.parent != POLYMER / "e1_combined/training_checkpoints" or digest(checkpoint) != index["sha256"]:
        raise RuntimeError("E1 full-state checkpoint checksum differs.")
    manifest = read(E3 / "manifest.json")
    e3.check_sources(manifest["source_hashes"])
    e3.reject_live_attempts(manifest["jobs"])
    for job in manifest["jobs"]:
        if job["state"] == "completed" and not e3.completed_job(job):
            raise RuntimeError("Previously completed E3 artifacts no longer verify.")
    return {"observed_at": now(), "processes": identities,
            "checkpoint": {key: index[key] for key in ("file", "sha256", "completed_steps", "completed_cycles", "bytes")},
            "new_column_cases": list(range(11, 17)), "preserved_repetition_cases": list(range(1, 11)),
            "completed_E3_jobs_reused": [job["id"] for job in manifest["jobs"] if job["state"] == "completed"]}


def command_file(path, title, arguments):
    command = subprocess.list2cmdline([str(Path(sys.executable).with_name("python.exe")), "-B", "-u", *arguments])
    path.write_text("@echo off\nsetlocal\n" + f"title {title}\ncd /d \"{ROOT}\"\n" +
                    'call "C:\\ProgramData\\anaconda3\\Scripts\\activate.bat" rl-assisted-env\n' + command +
                    "\nset result=%errorlevel%\necho.\necho Study exited with code %result%.\npause\nexit /b %result%\n", encoding="utf-8")


def archive_before_restart(directory):
    archive = directory / "preserved_before_restart"
    archive.mkdir()
    for stage in STAGES:
        target = archive / "polymer" / stage
        target.mkdir(parents=True)
        for source in (POLYMER / stage).iterdir():
            if source.is_file():
                shutil.copy2(source, target / source.name)
    for name in ("manifest.json", "visible_launches.json"):
        shutil.copy2(COLUMN / name, archive / f"column_{name}")
    for method in ("sg", "threshold", "ls", "bo", "e1_pilot"):
        target = archive / "column" / method
        target.mkdir(parents=True)
        for name in ("execution.json", "attempts.json", "heartbeat.json"):
            source = COLUMN / method / name
            if source.exists():
                shutil.copy2(source, target / name)
    shutil.copy2(E3 / "manifest.json", archive / "e3_manifest.json")
    # Full E1 checkpoint slots and old scientific attempts remain in place.
    # The existing checkpoint manager separately archives the replayed suffix.


def prepare(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT / "Results/restart_recovery"):
        raise ValueError("Recovery records must stay in the dedicated workspace directory.")
    audit = preflight()
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "preflight.json", audit)
    archive_before_restart(directory)
    command_file(directory / "column_e3.cmd", "Column E3 combined - Aspen 11", [str(Path(__file__).resolve()), "--run-e3", str(E3)])
    commands = [{"id": "polymer_" + stage, "command": str(POLYMER / stage / "run.cmd")} for stage in STAGES]
    commands += [{"id": "column_e3", "command": str(directory / "column_e3.cmd")}]
    commands += [{"id": "column_" + method, "command": str(COLUMN / method / "run.cmd")}
                 for method in ("sg", "threshold", "ls", "bo", "e1_pilot")]
    write(directory / "plan.json", {"created_at": now(), "commands": commands,
        "scope": "Nine previously authorized interrupted jobs only; no repetition restart.",
        "resume": "Polymer E1 resumes cycle-10 full-state checkpoint; other incomplete jobs restart original initial conditions.",
        "python": str(Path(sys.executable).with_name("python.exe"))})
    return directory


def broker(directory):
    from column_review.common import process_identity
    from run_polymer_e3_recovery import environment
    directory = Path(directory).resolve()
    with (directory / "broker.log").open("a", encoding="utf-8", buffering=1) as stream:
        sys.stdout = sys.stderr = stream
        try:
            identity = lifetime()
            write(directory / "broker_identity.json", identity)
            if not isolated_interactive_context(identity):
                raise RuntimeError("Broker is not owned by the interactive Windows scheduler; refusing an app-owned launch.")
            preflight()
            commands = read(directory / "plan.json")["commands"]
            results = []
            for item in commands:
                process = subprocess.Popen(f'cmd.exe /d /c call "{item["command"]}"', cwd=ROOT,
                                           env=environment(), creationflags=subprocess.CREATE_NEW_CONSOLE)
                results.append({**item, "cmd": process_identity(process.pid), "started_at": now()})
                write(directory / "launched.json", results)
            write(directory / "broker_status.json", {"state": "running", "launched": len(results), "updated_at": now()})
            while any(process_identity(item["cmd"]["pid"])["alive"] for item in results):
                write(directory / "broker_heartbeat.json", {"pid": os.getpid(), "updated_at": now(),
                      "visible_consoles_alive": sum(process_identity(item["cmd"]["pid"])["alive"] is True for item in results)})
                time.sleep(30)
            write(directory / "broker_status.json", {"state": "consoles_closed", "updated_at": now()})
        except BaseException as error:
            write(directory / "broker_status.json", {"state": "failed", "error": str(error), "updated_at": now()})
            traceback.print_exc()
            return 1
    return 0


def run_e3(study):
    """Use the original E3 continuation and stream its existing worker log."""
    import run_e3_prediction_quality as e3
    from run_distillation_phase2_repeats import aspen_file_lock
    from run_polymer_e3_recovery import environment
    os.environ.update(environment())
    study = Path(study).resolve()
    if study != E3:
        raise ValueError("Only the interrupted existing case-11 study is supported.")
    manifest = read(study / "manifest.json")
    e3.check_sources(manifest["source_hashes"])
    threads = []

    def launch(command, **kwargs):
        process = subprocess.Popen(command, **kwargs)
        log_path = Path(command[-1]).parent / "worker.log"
        def follow():
            with log_path.open(encoding="utf-8", errors="replace") as reader:
                last = time.monotonic()
                while process.poll() is None:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    if time.monotonic() - last >= 30:
                        print(f"[progress] E3 case 11 worker {process.pid} alive", flush=True)
                        last = time.monotonic()
                    time.sleep(1)
                print(reader.read(), end="", flush=True)
        thread = threading.Thread(target=follow, daemon=True)
        thread.start()
        threads.append(thread)
        return process

    with aspen_file_lock(study / "study", study / "_locks"):
        e3.run_jobs(manifest, study, launch=launch)
        for thread in threads:
            thread.join(timeout=3)
        e3.run_analysis(study)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", type=Path)
    group.add_argument("--schedule-probe", type=Path)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--launch", type=Path)
    group.add_argument("--broker", type=Path)
    group.add_argument("--run-e3", type=Path)
    args = parser.parse_args()
    if args.probe:
        write(args.probe, lifetime())
        return 0
    if args.schedule_probe:
        path = args.schedule_probe.resolve()
        print(json.dumps(schedule("RLAssistedMPC_LifetimeProbe_" + str(time.time_ns()), ["--probe", str(path)]), indent=2))
        return 0
    if args.prepare:
        print(prepare(args.prepare))
        return 0
    if args.launch:
        directory = args.launch.resolve()
        probe = read(directory.parent / "scheduler_probe.json")
        if not isolated_interactive_context(probe):
            raise RuntimeError("Interactive scheduler isolation probe did not pass.")
        if (directory / "scheduler.json").exists():
            raise RuntimeError("Recovery was already dispatched; inspect before another launch.")
        record = schedule("RLAssistedMPC_ReviewerRecovery_" + str(time.time_ns()), ["--broker", str(directory)])
        write(directory / "scheduler.json", record)
        print(json.dumps(record, indent=2))
        return 0
    if args.broker:
        return broker(args.broker)
    return run_e3(args.run_e3)


if __name__ == "__main__":
    raise SystemExit(main())
