"""Restart only failed Markov SG seed 23 and hand off the existing queue.

The original supervisor and its live workers retain ownership of their files.
A separate aggregate manifest tracks them alongside the retry. After their
processes finish and owned Aspen cases close, residual and combined each run
once, even if an earlier category failed. Completed work is never rerun.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
from unittest.mock import patch

from recover_review_runs import read, command_file, POINTER
from column_review.common import now, digest, process_identity, owned_apps_remaining

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "Distillation/Results/review_recovery/20261018/repeats"
DEFAULT = ROOT / "Distillation/Results/review_recovery/20261020_openblas"
TARGET = ("markov", "sg", 23)
GROUPS = ("retry", "residual", "combined")


def write(path, value):
    """Atomic status writes tolerate brief Windows reader/OneDrive locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    deadline = time.monotonic() + 5
    try:
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
    finally:
        temporary.unlink(missing_ok=True)


def key(job):
    return job["category"], job["mode"], job["seed"]


def same_process(record):
    current = process_identity(record["pid"])
    if current["alive"] is False:
        return False
    if current["created"] is not None and current["created"] != record["created"]:
        return False  # A recycled PID belongs to someone else.
    if current["alive"] is not True or current["created"] is None:
        raise RuntimeError(f"Cannot verify process lifetime: {record['pid']}")
    return True


def validated_result(job, manifest):
    import run_distillation_phase2_repeats as repeat
    if job["state"] != "completed" or job.get("exit_code") != 0 or not job["attempts"]:
        raise RuntimeError(f"No successful completion record for {key(job)}")
    status = repeat.saved_status(Path(job["attempts"][-1]["directory"]), 0)
    if not status or status.get("episodes") != 300 or status["result_file"] != job.get("result_file"):
        raise RuntimeError(f"Saved completion no longer validates: {key(job)}")
    metadata = read(Path(job["result_file"]).parent / "replication_metadata.json")
    entrypoint = repeat.RUNNERS[job["category"]][1]
    if (key(metadata) != key(job) or metadata.get("episodes") != 300
            or metadata.get("profile") != manifest["profile"]
            or metadata.get("entrypoint_sha256") != manifest["source_hashes"][entrypoint]):
        raise RuntimeError(f"Saved result provenance differs: {key(job)}")
    return {"job": list(key(job)), "path": job["result_file"], "sha256": digest(job["result_file"])}


def check_sources(manifest):
    import run_distillation_phase2_repeats as repeat
    if manifest["source_hashes"] != repeat.source_hashes(ROOT):
        raise RuntimeError("Scientific sources changed; refusing to mix versions or restart live work.")


def children_of(pid):
    """Inventory only descendants of an already verified, owned Aspen app."""
    import win32com.client
    service = win32com.client.GetObject("winmgmts:")
    result = []
    for process in service.ExecQuery(f"SELECT ProcessId,Name FROM Win32_Process WHERE ParentProcessId={int(pid)}"):
        identity = process_identity(int(process.ProcessId))
        if identity["alive"] is not True or identity["created"] is None:
            raise RuntimeError("Cannot establish orphan child identity.")
        if str(process.Name).lower() not in {"sim_server.exe", "am_task_server.exe"}:
            raise RuntimeError(f"Unexpected Aspen child; inspect before cleanup: {process.Name}")
        result.extend(children_of(identity["pid"]))
        result.append({**identity, "name": str(process.Name), "parent_pid": pid})
    return result


def terminate_owned_identity(record):
    """Validate creation and terminate through the SAME handle, avoiding PID reuse."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
    kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.OpenProcess(0x100001 | 0x1000, False, record["pid"])
    if not handle:
        if ctypes.get_last_error() == 87:
            return "already_exited"
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        stamps = [w.FILETIME() for _ in range(4)]
        code = w.DWORD()
        if not kernel.GetProcessTimes(handle, *map(ctypes.byref, stamps)) or not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = (stamps[0].dwHighDateTime << 32) + stamps[0].dwLowDateTime
        if created != record["created"]:
            return "pid_reused_untouched"
        if code.value != 259:
            return "already_exited"
        if not kernel.TerminateProcess(handle, 0xDEAD):
            raise ctypes.WinError(ctypes.get_last_error())
        if kernel.WaitForSingleObject(handle, 5000) != 0:
            raise RuntimeError("Verified orphan did not exit after termination.")
        return "closed_verified_orphan"
    finally:
        kernel.CloseHandle(handle)


def close_orphans(attempt, output):
    """Only called after a failed worker is proven dead and its case is leased."""
    attempt = Path(attempt)
    job = read(attempt / "job.json")
    status = read(attempt / "worker_status.json") if (attempt / "worker_status.json").exists() else {}
    if process_identity(status.get("pid"))["alive"] is not False:
        raise RuntimeError("Cannot close an Aspen app owned by a live or unverified worker.")
    unresolved = owned_apps_remaining(attempt / "owned_aspen_apps.json")
    records = []
    for entry in unresolved:
        if "recorded" not in entry or entry["current"]["alive"] is not True or entry["current"]["created"] != entry["recorded"]["created"]:
            raise RuntimeError("Aspen ownership is unresolved; refusing broad cleanup.")
        parent = entry["recorded"]
        children = children_of(parent["pid"])
        if any(child["created"] < parent["created"] for child in children):
            raise RuntimeError("Aspen child predates its recorded parent.")
        records.extend(children)
        records.append(parent)
    audit = {"attempt": str(attempt), "job": list(key(job)), "failed_worker_pid": status.get("pid"),
             "method": "Terminate only owned orphan identities, verified on the termination handle",
             "records": records, "started_at": now(), "actions": []}
    write(output, audit)  # Save the concrete target set before any signal.
    for record in records:
        audit["actions"].append({"identity": record, "result": terminate_owned_identity(record)})
        write(output, audit)
    if owned_apps_remaining(attempt / "owned_aspen_apps.json"):
        raise RuntimeError("Owned Aspen cleanup did not verify.")
    audit.update(state="closed", finished_at=now())
    write(output, audit)
    return audit


def prepare(directory=DEFAULT):
    import run_distillation_phase2_repeats as repeat
    import review_blas_runtime as runtime
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Recovery already prepared; use its existing launch/coordinate command.")
    old = read(OLD / "manifest.json")
    check_sources(old)
    repeat.validate_jobs(old["jobs"], require_paths=True)
    expected = {(category, mode, seed) for category in repeat.RUNNERS for mode in ("sg", "plain") for seed in repeat.DEFAULT_SEEDS}
    if {key(j) for j in old["jobs"]} != expected or old["episodes_per_run"] != 300 or old["threads_per_run"] != 1:
        raise RuntimeError("Unexpected original study inventory/settings.")
    retained = [validated_result(j, old) for j in old["jobs"] if j["state"] == "completed"]
    target = next(j for j in old["jobs"] if key(j) == TARGET)
    failed = Path(target["attempts"][-1]["directory"])
    failed_pid = target["attempts"][-1]["pid"]
    if process_identity(failed_pid)["alive"] is not False:
        raise RuntimeError("Seed 23 is not proven stopped; refusing a duplicate.")
    tail = (failed / "run.log").read_text(encoding="utf-8", errors="replace")[-16000:]
    if "Windows fatal exception: access violation" not in tail or "prediction_improvement_score" not in tail:
        raise RuntimeError("The failed attempt is not the reviewed native Markov LS crash.")
    supervisor = process_identity(read(OLD / "supervisor.json")["pid"])
    if supervisor["alive"] is not True or supervisor["created"] is None:
        raise RuntimeError("Original live supervisor identity changed; inspect before preparation.")
    external = []
    for job in old["jobs"]:
        if job["state"] == "running" and key(job) != TARGET:
            identity = process_identity(job["attempts"][-1]["pid"])
            if identity["alive"] is not True or identity["created"] is None:
                raise RuntimeError(f"Original worker changed during preparation: {key(job)}")
            external.append({"job": list(key(job)), **identity})
    if len(retained) != 20 or len(external) != 9 or sum(j["state"] == "pending" for j in old["jobs"]) != 20:
        raise RuntimeError("Original inventory changed; inspect before preparing the targeted handoff.")
    runtime_manifest = runtime.DEFAULT / "manifest.json"
    stress = read(ROOT / "change-reports/run_recovery_20261020/private_stress.json")
    reference = read(ROOT / "change-reports/run_recovery_20261020/reference.json")
    if stress["state"] != "passed" or stress["runtime"]["runtime_manifest_sha256"] != digest(runtime_manifest):
        raise RuntimeError("The private runtime has not passed its recorded Markov stress check.")
    if max(abs(a-b) for a, b in zip(stress["reference_z"], reference["reference_z"], strict=True)) > 1e-5:
        raise RuntimeError("Private/reference Markov optimizers differ beyond the reviewed tolerance.")
    directory.mkdir(parents=True)
    write(directory / "original_manifest_before.json", old)
    write(directory / "pointer_before.json", read(POINTER))
    bootstrap = {name: digest(ROOT / name) for name in (Path(__file__).name, "review_blas_runtime.py")}
    plan = {"directory": str(directory), "old_directory": str(OLD), "prepared_at": now(),
            "original_supervisor": supervisor, "external_workers": external,
            "failed_attempt": str(failed), "failed_worker_pid": failed_pid,
            "failed_worker_exit": "0xC0000005", "failed_cycle": 62,
            "retained_results": retained, "runtime_directory": str(runtime.DEFAULT),
            "runtime_manifest_sha256": digest(runtime_manifest), "bootstrap_sha256": bootstrap,
            "restart_cycle": 1, "episodes": 300, "seed": 23, "mode": "sg",
            "scientific_settings_changed": False, "shared_environment_modified": False,
            "queue_policy": "Wait for original workers/supervisor and retry to exit; verify per-case cleanup; run residual then combined even if a prior category failed."}
    write(directory / "plan.json", plan)
    for group in GROUPS:
        manifest = deepcopy(old)
        manifest["jobs"] = [deepcopy(j) for j in old["jobs"] if key(j) == TARGET] if group == "retry" else [deepcopy(j) for j in old["jobs"] if j["category"] == group]
        manifest.update(state="pending", recovery_of=str(OLD), runtime_directory=plan["runtime_directory"],
                        runtime_manifest_sha256=plan["runtime_manifest_sha256"], bootstrap_sha256=bootstrap)
        for job in manifest["jobs"]:
            job["state"] = "pending"
            for field in ("error", "exit_code", "result_file", "started_at", "finished_at"):
                job.pop(field, None)
            if group == "retry":
                job["attempts"][-1].update(state="failed", error="Native OpenBLAS access violation at cycle 62", exit_code=3221225477)
        write(directory / group / "manifest.json", manifest)
    command_file(directory / "recovery.cmd", "SEED 23 RECOVERY AND REMAINING QUEUE", [Path(__file__), "--coordinate", directory])
    command_file(directory / "progress.cmd", "ALL REVIEW RUNS - LIVE", [ROOT / "show_review_progress.py", "--watch"])
    refresh(directory)
    return plan


def check_bootstrap(manifest):
    check_sources(manifest)
    if any(digest(ROOT / relative) != expected for relative, expected in manifest["bootstrap_sha256"].items()):
        raise RuntimeError("Recovery bootstrap changed after preparation.")


def worker(path):
    path = Path(path).resolve()
    job = read(path)
    manifest = read(Path(job["batch_directory"]) / "manifest.json")
    check_bootstrap(manifest)
    if key(job) not in {key(j) for j in manifest["jobs"]}:
        raise RuntimeError("Worker job is not reserved in its recovery group.")
    import review_blas_runtime as runtime
    runtime.activate(manifest["runtime_directory"], manifest["runtime_manifest_sha256"])
    import torch
    import run_distillation_phase2_repeats as repeat
    from column_review.common import record_owned_apps
    torch.set_num_threads(1)
    info = runtime.runtime_info(manifest["runtime_directory"])
    info.update(bootstrap_sha256=manifest["bootstrap_sha256"], seed=job["seed"], mode=job["mode"])
    write(path.parent / "isolated_runtime.json", info)
    with record_owned_apps(path.parent), patch.object(repeat, "write_json", write):
        returncode = repeat.run_worker(path)
    status = read(path.parent / "worker_status.json")
    if status.get("result_file"):
        write(Path(status["result_file"]).parent / "recovery_runtime.json", info)
    return returncode


def run_group(directory):
    import run_distillation_phase2_repeats as repeat
    directory = Path(directory).resolve()
    with repeat.aspen_file_lock(directory / "supervisor", directory / "_locks"):
        manifest = read(directory / "manifest.json")
        check_bootstrap(manifest)
        # Never interpret a stale running manifest as permission for duplicates.
        if manifest["state"] != "pending":
            raise RuntimeError("Recovery group already started; inspect it instead of rerunning.")
        write(directory / "supervisor.json", {**process_identity(os.getpid()), "started_at": now()})
        try:
            with patch.object(repeat, "write_json", write):
                code = repeat.run_batches(manifest, directory, command_factory=lambda python, job:
                                          [python, "-B", "-u", str(Path(__file__).resolve()), "--worker", str(job)])
        except BaseException as error:
            write(directory / "exit.json", {"returncode": 1, "error": str(error), "finished_at": now()})
            raise
        write(directory / "exit.json", {"returncode": code, "finished_at": now()})
        return code


def start_group(directory):
    directory = Path(directory)
    if (directory / "launch.json").exists():
        return
    manifest = read(directory / "manifest.json")
    check_bootstrap(manifest)
    write(directory / "launch.json", {"phase": "spawning", "started_at": now()})
    with (directory / "supervisor.log").open("ab") as log:
        process = subprocess.Popen([manifest["python"], "-B", "-u", str(Path(__file__).resolve()), "--run-group", str(directory)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    write(directory / "launch.json", {"phase": "started", "identity": process_identity(process.pid), "started_at": now()})
    print(f"[START] {directory.name}: {len(manifest['jobs'])} run(s).", flush=True)


def group_finished(directory):
    launch_path = Path(directory) / "launch.json"
    if not launch_path.exists():
        return False
    launch = read(launch_path)
    if not launch.get("identity"):
        raise RuntimeError("Unresolved launch intent; inspect before spawning another worker.")
    if same_process(launch["identity"]):
        return False
    if not (Path(directory) / "exit.json").exists():
        raise RuntimeError("Group supervisor stopped without a final record; inspect before queue handoff.")
    return True


def refresh(directory):
    directory = Path(directory)
    plan = read(directory / "plan.json")
    manifest = deepcopy(read(Path(plan["old_directory"]) / "manifest.json"))
    replacements = {}
    for group in GROUPS:
        for job in read(directory / group / "manifest.json")["jobs"]:
            replacements[key(job)] = job
    manifest["jobs"] = [deepcopy(replacements.get(key(j), j)) for j in manifest["jobs"]]
    manifest.update(recovery_of=plan["old_directory"], recovery_plan=str(directory / "plan.json"),
                    updated_at=now(), state="running", queue_coordinator=str(directory / "coordinator_status.json"))
    status_path = directory / "coordinator_status.json"
    if status_path.exists():
        status = read(status_path)
        if status.get("state") in ("completed", "failed", "blocked"):
            manifest["state"] = status["state"]
    write(directory / "manifest.json", manifest)
    return manifest


def release_group_cases(directory, group):
    """A failed run cannot prevent unrelated categories once its lease is free."""
    import run_distillation_phase2_repeats as repeat
    directory = Path(directory)
    manifest = read(directory / group / "manifest.json")
    with ExitStack() as stack:
        for path in sorted({j["aspen_path"] for j in manifest["jobs"]}):
            stack.enter_context(repeat.aspen_file_lock(Path(path), ROOT / "Distillation/Results/_aspen_file_locks"))
        for job in manifest["jobs"]:
            attempt = Path(job["attempts"][-1]["directory"])
            if job["state"] == "completed":
                validated_result(job, manifest)
                if owned_apps_remaining(attempt / "owned_aspen_apps.json"):
                    raise RuntimeError("Completed worker left an unresolved Aspen application.")
            elif owned_apps_remaining(attempt / "owned_aspen_apps.json"):
                close_orphans(attempt, directory / "cleanup" / f"{group}_{job['mode']}_{job['seed']}.json")


def queue_ready(plan, retry_finished):
    # All original workers are checked, in addition to their supervisor.
    return (not any(same_process(p) for p in plan["external_workers"])
            and not same_process(plan["original_supervisor"]) and retry_finished)


def release_original_cases(directory, old):
    import run_distillation_phase2_repeats as repeat
    jobs = [j for j in old["jobs"] if j["category"] == "markov" and key(j) != TARGET]
    if any(j["state"] in ("running", "pending") for j in jobs):
        raise RuntimeError("Original Markov supervisor has no final per-job records.")
    with ExitStack() as stack:
        for path in sorted({j["aspen_path"] for j in jobs}):
            stack.enter_context(repeat.aspen_file_lock(Path(path), ROOT / "Distillation/Results/_aspen_file_locks"))
        for job in jobs:
            attempt = Path(job["attempts"][-1]["directory"])
            if job["state"] == "completed":
                validated_result(job, old)
                if owned_apps_remaining(attempt / "owned_aspen_apps.json"):
                    raise RuntimeError("Original completed Markov case is still occupied.")
            elif owned_apps_remaining(attempt / "owned_aspen_apps.json"):
                close_orphans(attempt, Path(directory) / "cleanup" / f"original_{job['mode']}_{job['seed']}.json")


def coordinate(directory):
    import run_distillation_phase2_repeats as repeat
    directory = Path(directory).resolve()
    with repeat.aspen_file_lock(directory / "coordinator", directory / "_locks"):
        plan = read(directory / "plan.json")
        check_bootstrap(read(directory / "retry/manifest.json"))
        write(directory / "coordinator_identity.json", {**process_identity(os.getpid()), "parent_pid": os.getppid(), "started_at": now()})
        try:
            if not (directory / "retry/launch.json").exists():
                target = read(Path(plan["failed_attempt"]) / "job.json")
                with repeat.aspen_file_lock(Path(target["aspen_path"]), ROOT / "Distillation/Results/_aspen_file_locks"):
                    close_orphans(plan["failed_attempt"], directory / "cleanup/seed23_old_orphan.json")
                start_group(directory / "retry")
            pointer = read(directory / "pointer_before.json")
            pointer.update(repeats=str(directory), repeat_recovery_plan=str(directory / "plan.json"))
            write(POINTER, pointer)
            stage = "waiting_for_original_workers_and_retry"
            while not queue_ready(plan, group_finished(directory / "retry")):
                write(directory / "coordinator_status.json", {"state": "running", "stage": stage,
                      "pid": os.getpid(), "updated_at": now(), "queued_runs": 20})
                refresh(directory)
                time.sleep(15)
            old = read(Path(plan["old_directory"]) / "manifest.json")
            release_original_cases(directory, old)
            release_group_cases(directory, "retry")
            for group in ("residual", "combined"):
                start_group(directory / group)
                while not group_finished(directory / group):
                    write(directory / "coordinator_status.json", {"state": "running", "stage": group, "pid": os.getpid(), "updated_at": now()})
                    refresh(directory)
                    time.sleep(15)
                release_group_cases(directory, group)
            manifest = refresh(directory)
            complete = all(j["state"] == "completed" for j in manifest["jobs"])
            write(directory / "coordinator_status.json", {"state": "completed" if complete else "failed",
                  "stage": "queue_finished", "queued_runs_dispatched": 20, "finished_at": now(), "pid": os.getpid()})
            refresh(directory)
            print("[DONE] Remaining queue finished; see aggregate manifest for each result.", flush=True)
            return 0 if complete else 1
        except BaseException as error:
            write(directory / "coordinator_status.json", {"state": "blocked", "error": str(error), "traceback": traceback.format_exc(), "updated_at": now(), "pid": os.getpid()})
            refresh(directory)
            raise


def schedule(directory):
    import win32api
    import win32con
    import win32security
    import win32com.client
    service = win32com.client.Dispatch("Schedule.Service")
    service.Connect()
    definition = service.NewTask(0)
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    definition.RegistrationInfo.Description = "Author-approved seed 23 recovery and existing remaining queue; demand only."
    definition.Principal.UserId, definition.Principal.LogonType = user, 3
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    settings.ExecutionTimeLimit, settings.MultipleInstances = "PT0S", 2
    action = definition.Actions.Create(0)
    action.Path = str(Path(sys.executable).with_name("pythonw.exe"))
    action.Arguments = subprocess.list2cmdline(["-B", "-u", str(Path(__file__).resolve()), "--broker", str(directory)])
    action.WorkingDirectory = str(ROOT)
    name = "RLAssistedMPC-Seed23-" + uuid.uuid4().hex[:12]
    task = service.GetFolder("\\").RegisterTaskDefinition(name, definition, 2, user, None, 3)
    task.Run(None)
    return {"task_name": name, "triggers": 0, "execution_time_limit": "PT0S", "logon_type": "interactive_token"}


def broker(directory):
    from review_restart_recovery import lifetime, isolated_interactive_context
    import run_distillation_phase2_repeats as repeat
    directory = Path(directory).resolve()
    with repeat.aspen_file_lock(directory / "broker", directory / "_locks"), (directory / "broker.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        identity = lifetime()
        write(directory / "broker_identity.json", identity)
        if not isolated_interactive_context(identity):
            raise RuntimeError("Recovery must be launched from the independent interactive scheduler.")
        launches = []
        for name in ("recovery", "progress"):
            process = subprocess.Popen(f'cmd.exe /d /c call "{directory / (name + ".cmd")}"', cwd=ROOT,
                                       creationflags=subprocess.CREATE_NEW_CONSOLE)
            launches.append({"console": name, **process_identity(process.pid), "started_at": now()})
            write(directory / "launched.json", launches)
        while any(same_process(p) for p in launches):
            time.sleep(30)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for option in ("prepare", "launch", "broker", "coordinate", "run-group", "worker"):
        actions.add_argument("--" + option, type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.prepare), indent=2))
        return 0
    if args.launch:
        result = schedule(args.launch.resolve())
        write(args.launch / "scheduled.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.broker:
        return broker(args.broker)
    if args.coordinate:
        return coordinate(args.coordinate)
    if args.run_group:
        return run_group(args.run_group)
    return worker(args.worker)


if __name__ == "__main__":
    raise SystemExit(main())
