"""Restart the verified stopped a3 attempt only; preserve the active queue."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def allowed_owned_binary(name, executable):
    """Only installed Aspen servers and Windows' console host are descendants."""
    expected = {"sim_server.exe": Path("C:/Program Files/AspenTech/AMSystem V14.0/Bin/sim_server.exe"),
                "am_task_server.exe": Path("C:/Program Files/AspenTech/AMSystem V14.0/Bin/am_task_server.exe"),
                "conhost.exe": Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/conhost.exe"}
    return name.lower() in expected and Path(executable).resolve() == expected[name.lower()].resolve()


def verified_owned_children(pid):
    import win32com.client
    from column_review.common import process_identity
    parent = process_identity(pid)
    if parent["alive"] is not True or parent["created"] is None:
        raise RuntimeError("Cannot establish the orphan parent's lifetime.")
    service = win32com.client.GetObject("winmgmts:")
    records = []
    for p in service.ExecQuery(f"SELECT ProcessId,Name,ExecutablePath FROM Win32_Process WHERE ParentProcessId={int(pid)}"):
        identity = process_identity(int(p.ProcessId))
        if (identity["alive"] is not True or identity["created"] is None or identity["created"] < parent["created"]
                or not allowed_owned_binary(str(p.Name), str(p.ExecutablePath))):
            raise RuntimeError("Orphan descendant has an unverified binary, identity or creation time.")
        records.extend(verified_owned_children(identity["pid"]))
        records.append({**identity, "name": str(p.Name), "executable": str(p.ExecutablePath), "parent_pid": pid})
    return records


def prepare(directory):
    import continue_column_queue as queue
    import recover_markov_seed23 as recovery
    import run_distillation_phase2_repeats as repeat
    from column_review.common import read, now, digest
    pointer = read(ROOT / "Distillation/Results/review_recovery/latest.json")
    queue_directory = Path(pointer["repeats"])
    plan = read(queue_directory / "plan.json")
    retry = Path(plan["retry_directory"])
    manifest = read(retry / "manifest.json")
    recovery.check_bootstrap(manifest)
    queue.validate_group(read(queue_directory / "combined/manifest.json"))
    job = manifest["jobs"][0]
    supervisor = read(retry / "supervisor.json")
    if len(manifest["jobs"]) != 1 or recovery.key(job) != recovery.TARGET or job["aspen_number"] != 3:
        raise RuntimeError("Expected only Markov SG seed 23 on case 3.")
    if len(job["attempts"]) != 3 or job["state"] != "running":
        raise RuntimeError("This repair is restricted to the stopped third attempt; inspect another state.")
    attempt = Path(job["attempts"][-1]["directory"])
    if not queue.absent(job["attempts"][-1]["pid"]) or not queue.absent(supervisor["pid"]):
        raise RuntimeError("Worker or supervisor is present; never restart a live or uncertain job.")
    directory.mkdir(parents=True, exist_ok=False)
    recovery.write(directory / "before_manifest.json", manifest)
    recovery.write(directory / "before_worker_status.json", read(attempt / "worker_status.json"))
    recovery.write(directory / "before_supervisor.json", supervisor)
    with repeat.aspen_file_lock(retry / "supervisor", retry / "_locks"):
        with repeat.aspen_file_lock(Path(job["aspen_path"]), ROOT / "Distillation/Results/_aspen_file_locks"):
            with patch.object(recovery, "children_of", verified_owned_children):
                cleanup = recovery.close_orphans(attempt, directory / "orphan_cleanup.json")
        job["attempts"][-1].update(state="stopped", observed_at=now(),
            error="Worker and its supervisor disappeared at cycle 112 without final records; no new Windows crash event found. Cause unconfirmed.")
        job.update(state="pending", recovery_of=str(attempt))
        manifest.update(state="pending", updated_at=now(), stopped_attempt_recovery=str(directory))
        recovery.write(retry / "manifest.json", manifest)
    command = subprocess.list2cmdline([sys.executable, "-B", "-u", str(ROOT / "recover_markov_seed23.py"), "--run-group", str(retry)])
    (directory / "run.cmd").write_text('@echo off\nsetlocal\ntitle ACTIVE Markov SG seed 23 - attempt 4 - Aspen 3\n'
        f'cd /d "{ROOT}"\n{command}\nset result=%errorlevel%\necho Markov worker group exited with code %result%.\npause\nexit /b %result%\n', encoding="utf-8")
    recovery.write(directory / "plan.json", {"prepared_at": now(), "retry_directory": str(retry),
        "stopped_attempt": str(attempt), "original_worker_absent": True, "original_supervisor_absent": True,
        "original_scripts_and_settings_unchanged": True, "full_training_checkpoint_available": False,
        "restarts_original_300_cycles": True, "reason": "No verified learner/plant continuation state survived the stop.",
        "queue_directory": str(queue_directory), "combined_source_and_model_guards_pass": True,
        "launcher_sha256": digest(__file__), "cleanup": cleanup})
    queue.refresh(queue_directory)


def schedule(directory):
    import win32api
    import win32con
    import win32security
    import win32com.client
    from recover_markov_seed23 import write
    service = win32com.client.Dispatch("Schedule.Service")
    service.Connect()
    definition = service.NewTask(0)
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    definition.RegistrationInfo.Description = "Immediate author-approved restart of stopped Markov SG seed 23, isolated from the running queue."
    definition.Principal.UserId, definition.Principal.LogonType = user, 3
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    settings.ExecutionTimeLimit, settings.MultipleInstances = "PT0S", 2
    action = definition.Actions.Create(0)
    action.Path = str(Path(sys.executable).with_name("pythonw.exe"))
    action.Arguments = subprocess.list2cmdline(["-B", "-u", str(Path(__file__).resolve()), "--broker", str(directory)])
    action.WorkingDirectory = str(ROOT)
    name = "RLAssistedMPC-Markov23-Restart-" + uuid.uuid4().hex[:12]
    task = service.GetFolder("\\").RegisterTaskDefinition(name, definition, 2, user, None, 3)
    write(directory / "scheduled.json", {"task_name": name, "triggers": 0, "execution_time_limit": "PT0S"})
    task.Run(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--broker", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.launch:
        schedule(args.launch.resolve())
    elif args.broker:
        from recover_polymer_threshold import broker
        return broker(args.broker.resolve())
    else:
        parser.error("Select a recovery action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
