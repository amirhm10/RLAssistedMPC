"""Run the existing remaining channels while Markov seed 23 keeps case 3.

Only the pending residual/combined SG seed 23 jobs move to identical case 12.
The original scientific sources and the running retry bootstrap remain fixed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
from unittest.mock import patch

import recover_markov_seed23 as recovery
import run_distillation_phase2_repeats as repeat
from column_review.common import digest, now, process_identity, read, owned_apps_remaining

ROOT = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
PREVIOUS = ROOT / "Distillation/Results/review_recovery/20261020_openblas"
DEFAULT = ROOT / "Distillation/Results/review_recovery/20261020_queue_case12"
CHANNELS = ("residual", "combined")
SLOTS = {("sg", 7): 1, ("sg", 11): 2, ("sg", 23): 12, ("sg", 37): 4, ("sg", 53): 5,
         ("plain", 7): 6, ("plain", 11): 7, ("plain", 23): 8, ("plain", 37): 9, ("plain", 53): 10}
write = recovery.write


def absent(pid):
    """Use a fresh process inventory if Windows cannot open a retired PID."""
    current = process_identity(pid)
    if current["alive"] is not None:
        return current["alive"] is False
    import win32com.client
    service = win32com.client.GetObject("winmgmts:")
    return not list(service.ExecQuery(f"SELECT ProcessId FROM Win32_Process WHERE ProcessId={int(pid)}"))


def verified_assignment(job, manifest):
    """The only extra slot is case 12 for the existing SG seed 23 identity."""
    if job["category"] not in CHANNELS or job["aspen_number"] != SLOTS.get((job["mode"], job["seed"])):
        raise RuntimeError("Unapproved channel/seed/Aspen slot assignment.")
    reserved = next((j for j in manifest["jobs"] if recovery.key(j) == recovery.key(job)), None)
    if reserved is None or any(job[k] != reserved[k] for k in ("aspen_number", "aspen_path", "snaps_path")):
        raise RuntimeError("Worker paths differ from the reserved case.")
    if digest(job["aspen_path"]) != manifest["model_hashes"][job["aspen_path"]]:
        raise RuntimeError("Prepared Aspen model changed.")


def validate_group(manifest):
    recovery.check_bootstrap(manifest)
    if len(manifest["jobs"]) != 10 or len({j["category"] for j in manifest["jobs"]}) != 1:
        raise RuntimeError("Expected exactly ten existing repetitions in one channel.")
    if {(j["mode"], j["seed"]) for j in manifest["jobs"]} != set(SLOTS):
        raise RuntimeError("Existing seed/mode inventory differs.")
    for job in manifest["jobs"]:
        verified_assignment(job, manifest)
    # Scoped to this new process/context, with the stronger exact mapping above.
    with patch.object(repeat, "MAX_ASPEN_FILES", 12):
        repeat.validate_jobs(manifest["jobs"], require_paths=True)


def refresh(directory):
    directory = Path(directory)
    plan = read(directory / "plan.json")
    manifest = deepcopy(read(Path(plan["original_directory"]) / "manifest.json"))
    replacements = {recovery.key(j): j for j in read(Path(plan["retry_directory"]) / "manifest.json")["jobs"]}
    for group in CHANNELS:
        replacements.update({recovery.key(j): j for j in read(directory / group / "manifest.json")["jobs"]})
    manifest["jobs"] = [deepcopy(replacements.get(recovery.key(j), j)) for j in manifest["jobs"]]
    status = read(directory / "coordinator_status.json", {})
    manifest.update(state=status.get("state", "prepared"), recovery_of=plan["previous_directory"],
                    queue_coordinator=str(directory / "coordinator_status.json"), updated_at=now())
    write(directory / "manifest.json", manifest)
    return manifest


def prepare(directory):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Queue already prepared; use its existing launch.")
    previous = read(PREVIOUS / "plan.json")
    previous_status = read(PREVIOUS / "coordinator_status.json")
    if not absent(previous_status["pid"]):
        raise RuntimeError("The previous coordinator still exists; refusing competing schedulers.")
    original_directory = Path(previous["old_directory"])
    original = read(original_directory / "manifest.json")
    recovery.check_sources(original)
    if original["state"] != "failed":
        raise RuntimeError("Expected the original scheduler to have finished its failed Markov group.")
    retained = [recovery.validated_result(j, original) for j in original["jobs"] if j["state"] == "completed"]
    if len(retained) != 29:
        raise RuntimeError("Expected 29 validated completed repetitions.")
    retry_directory = PREVIOUS / "retry"
    retry = read(retry_directory / "manifest.json")["jobs"][0]
    retry_identity = process_identity(retry["attempts"][-1]["pid"])
    if recovery.key(retry) != recovery.TARGET or retry["state"] != "running" or retry_identity["alive"] is not True:
        raise RuntimeError("Expected the healthy seed 23 retry to remain running on case 3.")
    case_root = Path(retry["aspen_path"]).parent
    spare = case_root / "C2S_SS_simulation12.dynf"
    if digest(spare) != digest(retry["aspen_path"]):
        raise RuntimeError("Spare case 12 differs from the running case 3 model.")
    # Verify ownership of every live Aspen application; unknown user sessions
    # must be inspected before assigning a supposedly free case.
    from revision_comparisons.progress import snapshot
    current = snapshot()
    live = [r for r in current["jobs"] if r["state"] == "running" and r.get("alive") is True
            and r["study"] not in current["historical_studies"]]
    protected, protected_apps = [], []
    for row in live:
        identity = process_identity(row["pid"])
        protected.append({"job": row["job"], "study": row["study"], **identity})
        for app in read(Path(row["directory"]) / "owned_aspen_apps.json", []):
            current = process_identity(app["pid"])
            if current["alive"] is True and current["created"] == app["created"]:
                protected_apps.append(app)
    import win32com.client
    service = win32com.client.GetObject("winmgmts:")
    current_aspen = {int(p.ProcessId) for p in service.ExecQuery("SELECT ProcessId FROM Win32_Process WHERE Name='AspenModeler.exe'")}
    if current_aspen != {p["pid"] for p in protected_apps}:
        raise RuntimeError("An Aspen application is not accounted for by the protected live jobs.")
    for group in CHANNELS:
        pending = read(PREVIOUS / group / "manifest.json")
        if pending["state"] != "pending" or any(j["attempts"] or j["state"] != "pending" for j in pending["jobs"]):
            raise RuntimeError("A remaining channel already started; refusing duplicates.")
    # Probe all ten distinct physical leases before writing any new attempt.
    paths = [case_root / f"C2S_SS_simulation{slot}.dynf" for slot in SLOTS.values()]
    with ExitStack() as stack:
        for path in sorted(paths):
            stack.enter_context(repeat.aspen_file_lock(path, ROOT / "Distillation/Results/_aspen_file_locks"))
    directory.mkdir(parents=True)
    bootstrap = {name: digest(ROOT / name) for name in (THIS.name, "recover_markov_seed23.py", "review_blas_runtime.py")}
    plan = {"prepared_at": now(), "previous_directory": str(PREVIOUS), "original_directory": str(original_directory),
            "retry_directory": str(retry_directory), "retained_results": retained,
            "protected_workers": protected, "protected_apps": protected_apps,
            "seed23_model": {"source_case": 3, "new_pending_case": 12, "sha256": digest(spare)},
            "previous_coordinator_failure": previous_status, "bootstrap_sha256": bootstrap,
            "policy": "Keep active Markov seed 23 on case 3; start residual now, then combined on cases 1,2,12,4--10.",
            "scientific_settings_changed": False}
    write(directory / "plan.json", plan)
    write(directory / "pointer_before.json", read(recovery.POINTER))
    for group in CHANNELS:
        manifest = deepcopy(read(PREVIOUS / group / "manifest.json"))
        manifest.update(bootstrap_sha256=bootstrap, recovery_of=str(PREVIOUS), model_hashes={})
        for job in manifest["jobs"]:
            if (job["mode"], job["seed"]) == ("sg", 23):
                job.update(aspen_number=12, aspen_path=str(spare), snaps_path=str(directory / "native_snapshots" / group / "case12"))
                Path(job["snaps_path"]).mkdir(parents=True)
            manifest["model_hashes"][job["aspen_path"]] = digest(job["aspen_path"])
        validate_group(manifest)
        write(directory / group / "manifest.json", manifest)
    recovery.command_file(directory / "queue.cmd", "COLUMN RESIDUAL THEN COMBINED - SPARE CASE 12", [THIS, "--coordinate", directory])
    recovery.command_file(directory / "progress.cmd", "ALL REVIEW RUNS - LIVE", [ROOT / "show_review_progress.py", "--watch"])
    refresh(directory)
    return {"directory": str(directory), "retained": len(retained), "protected": protected, "case3_sha256_equals_case12": True}


def worker(path):
    path = Path(path).resolve()
    job = read(path)
    manifest = read(Path(job["batch_directory"]) / "manifest.json")
    verified_assignment(job, manifest)
    with patch.object(repeat, "MAX_ASPEN_FILES", 12):
        return recovery.worker(path)


def run_group(directory):
    directory = Path(directory).resolve()
    with repeat.aspen_file_lock(directory / "supervisor", directory / "_locks"):
        manifest = read(directory / "manifest.json")
        validate_group(manifest)
        if manifest["state"] != "pending":
            raise RuntimeError("This channel already started; refusing duplicate execution.")
        write(directory / "supervisor.json", {**process_identity(os.getpid()), "started_at": now()})
        try:
            with patch.object(repeat, "MAX_ASPEN_FILES", 12), patch.object(repeat, "write_json", write):
                code = repeat.run_batches(manifest, directory, command_factory=lambda python, job:
                                          [python, "-B", "-u", str(THIS), "--worker", str(job)])
            write(directory / "exit.json", {"returncode": code, "finished_at": now()})
            return code
        except BaseException as error:
            write(directory / "exit.json", {"returncode": 1, "error": str(error), "finished_at": now()})
            raise


def start_group(directory):
    directory = Path(directory)
    if (directory / "launch.json").exists():
        return
    manifest = read(directory / "manifest.json")
    validate_group(manifest)
    # Leases are still checked by each worker for its whole simulator lifetime.
    with ExitStack() as stack:
        for path in sorted(manifest["model_hashes"]):
            stack.enter_context(repeat.aspen_file_lock(Path(path), ROOT / "Distillation/Results/_aspen_file_locks"))
    write(directory / "launch.json", {"phase": "spawning", "started_at": now()})
    with (directory / "supervisor.log").open("ab") as log:
        process = subprocess.Popen([manifest["python"], "-B", "-u", str(THIS), "--run-group", str(directory)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    write(directory / "launch.json", {"phase": "started", "identity": process_identity(process.pid), "started_at": now()})
    print(f"[START] {directory.name}: 10 repetitions; SG seed 23 uses Aspen 12. Markov continues on Aspen 3.", flush=True)


def group_finished(directory):
    """A final supervisor record outranks queries of long-retired process IDs."""
    directory = Path(directory)
    if (directory / "exit.json").exists():
        manifest = read(directory / "manifest.json")
        if any(j["state"] in ("pending", "running") for j in manifest["jobs"]):
            raise RuntimeError("Supervisor exited before recording final per-job states.")
        return True
    launch = read(directory / "launch.json", {})
    if launch and (not launch.get("identity") or absent(launch["identity"]["pid"])):
        raise RuntimeError("Channel supervisor stopped without a final record; inspect before retrying.")
    return False


def coordinate(directory):
    directory = Path(directory).resolve()
    with repeat.aspen_file_lock(directory / "coordinator", directory / "_locks"):
        plan = read(directory / "plan.json")
        try:
            if not absent(plan["previous_coordinator_failure"]["pid"]):
                raise RuntimeError("Previous coordinator is still present.")
            write(directory / "coordinator_identity.json", {**process_identity(os.getpid()), "started_at": now()})
            pointer = read(directory / "pointer_before.json")
            pointer.update(repeats=str(directory), repeat_recovery_plan=str(directory / "plan.json"))
            write(recovery.POINTER, pointer)
            for group in CHANNELS:
                start_group(directory / group)
                while not group_finished(directory / group):
                    write(directory / "coordinator_status.json", {"state": "running", "stage": group,
                          "pid": os.getpid(), "updated_at": now(), "seed23_spare_case": 12})
                    refresh(directory)
                    time.sleep(15)
                recovery.release_group_cases(directory, group)
            while read(Path(plan["retry_directory"]) / "manifest.json")["jobs"][0]["state"] == "running":
                write(directory / "coordinator_status.json", {"state": "running", "stage": "queue_finished_waiting_for_markov",
                      "pid": os.getpid(), "updated_at": now(), "queued_runs_dispatched": 20})
                refresh(directory)
                time.sleep(15)
            manifest = refresh(directory)
            completed = all(j["state"] == "completed" for j in manifest["jobs"])
            write(directory / "coordinator_status.json", {"state": "completed" if completed else "failed", "stage": "finished",
                  "pid": os.getpid(), "finished_at": now(), "queued_runs_dispatched": 20})
            refresh(directory)
            return 0 if completed else 1
        except BaseException as error:
            write(directory / "coordinator_status.json", {"state": "blocked", "error": str(error), "pid": os.getpid(),
                  "traceback": traceback.format_exc(), "updated_at": now()})
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
    definition.RegistrationInfo.Description = "Author-approved remaining column queue using free Aspen case 12. Demand only."
    definition.Principal.UserId, definition.Principal.LogonType = user, 3
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    settings.ExecutionTimeLimit, settings.MultipleInstances = "PT0S", 2
    action = definition.Actions.Create(0)
    action.Path = str(Path(sys.executable).with_name("pythonw.exe"))
    action.Arguments = subprocess.list2cmdline(["-B", "-u", str(THIS), "--broker", str(directory)])
    action.WorkingDirectory = str(ROOT)
    name = "RLAssistedMPC-ColumnQueue-" + uuid.uuid4().hex[:12]
    task = service.GetFolder("\\").RegisterTaskDefinition(name, definition, 2, user, None, 3)
    task.Run(None)
    return {"task_name": name, "triggers": 0, "execution_time_limit": "PT0S"}


def broker(directory):
    from review_restart_recovery import lifetime, isolated_interactive_context
    directory = Path(directory).resolve()
    with repeat.aspen_file_lock(directory / "broker", directory / "_locks"), (directory / "broker.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        identity = lifetime()
        write(directory / "broker_identity.json", identity)
        if not isolated_interactive_context(identity):
            raise RuntimeError("Queue must be launched from the independent interactive Windows scheduler.")
        launches = []
        for name in ("queue", "progress"):
            process = subprocess.Popen(f'cmd.exe /d /c call "{directory / (name + ".cmd")}"', cwd=ROOT,
                                       creationflags=subprocess.CREATE_NEW_CONSOLE)
            launches.append({"console": name, **process_identity(process.pid), "started_at": now()})
            write(directory / "launched.json", launches)
        while any(not absent(p["pid"]) for p in launches):
            time.sleep(30)
    return 0


def main():
    import json
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
