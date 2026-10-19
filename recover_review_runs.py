"""Recover the unfinished approved column studies in independent visible consoles.

Preparation is read-only for old studies. New attempts use the original study
settings and separate directories. --launch uses a demand-only interactive
Windows task, so closing Codex does not own the experiment process lifetime.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
OLD_REPEATS = ROOT / "Distillation/Results/phase2_repeats/20261014_154252_a2ed3b"
OLD_COLUMN = ROOT / "Distillation/Results/column_review/20261016_prepared_cases12_16"
POINTER = ROOT / "Distillation/Results/review_recovery/latest.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_stopped(pid, label):
    from column_review.common import process_identity
    if process_identity(pid)["alive"] is not False:
        raise RuntimeError(f"Live or unverified process for {label}; refusing duplicate execution: {pid}")


def repeat_manifest(old, old_directory):
    """Keep validated completed results; reset only unfinished jobs in a copy."""
    import run_distillation_phase2_repeats as repeat
    from column_review.common import digest
    value = deepcopy(old)
    if value["state"] == "running":
        raise RuntimeError("The original repeat scheduler is marked running; inspect it first.")
    current = repeat.source_hashes(ROOT)
    if current != old["source_hashes"]:
        raise RuntimeError("Original repetition scientific sources changed; do not mix versions.")
    reused = []
    for job in value["jobs"]:
        for attempt in job["attempts"]:
            if attempt.get("pid"):
                ensure_stopped(attempt["pid"], f"{job['category']}/{job['mode']}/{job['seed']}")
        if job["state"] == "completed":
            attempt = job["attempts"][-1]
            status = repeat.saved_status(Path(attempt["directory"]), 0)
            if (not status or status.get("episodes") != 300
                    or status.get("result_file") != job.get("result_file")):
                raise RuntimeError("A previously completed repetition no longer validates.")
            reused.append({"category": job["category"], "mode": job["mode"], "seed": job["seed"],
                           "path": job["result_file"], "sha256": digest(job["result_file"])})
        else:
            job["state"] = "pending"
            for key in ("result_file", "error", "finished_at"):
                job.pop(key, None)
    value.update(state="pending", recovery_of=str(old_directory), source_hashes=current,
                 reused_completed_results=reused, recovery_runtime="single-threaded Nehalem OpenBLAS; NumPy X86_V3 disabled")
    return value


def old_column_stopped():
    from column_review.common import MAPPING, owned_apps_remaining
    for method in MAPPING:
        stage = OLD_COLUMN / method
        for attempt in read(stage / "attempts.json"):
            ensure_stopped(attempt["pid"], method)
            if owned_apps_remaining(Path(attempt["directory"]) / "owned_aspen_apps.json"):
                raise RuntimeError(f"An earlier owned Aspen case is unresolved: {method}")


def command_file(path, title, arguments):
    command = subprocess.list2cmdline([sys.executable, "-B", "-u", *map(str, arguments)])
    path.write_text(f'@echo off\nsetlocal\ntitle {title}\ncd /d "{ROOT}"\n{command}\n'
                    'set result=%errorlevel%\necho.\necho Process exited with code %result%.\npause\nexit /b %result%\n',
                    encoding="utf-8")


def hold_snapshot_pilot(column_directory):
    """Do not commission another full-prefix retry of the known native failure."""
    stage = Path(column_directory) / "e1_pilot"
    command = stage / "run.cmd"
    preserved = stage / "run_before_snapshot_hold.cmd"
    if not preserved.exists():
        preserved.write_bytes(command.read_bytes())
    reason = ("Native TakeSnapshot failed again; a short startup snapshot exported, "
              "but its restored state did not advance time. Native restoration remains unvalidated.")
    command.write_text('@echo off\nsetlocal\ntitle Column e1_pilot - HELD pending native restoration validation\n'
                       f'echo {reason}\necho See RUN_RECOVERY_STATUS.md and the bounded snapshot diagnostics.\n'
                       'echo Full-prefix retry is held; other production jobs continue independently.\npause\nexit /b 1\n',
                       encoding="utf-8")
    write(stage / "snapshot_hold.json", {"state": "held", "reason": reason,
                                         "release_requires": "Validated native state restoration, advancing clock and repeated continuations."})


def prepare(directory):
    from column_review.common import now, digest
    from run_distillation_remaining import prepare as prepare_column
    from run_polymer_e3_recovery import RUNTIME
    if POINTER.exists():
        previous = read(POINTER)
        broker = Path(previous["directory"]) / "broker_identity.json"
        if broker.exists():
            ensure_stopped(read(broker)["pid"], "previous recovery broker")
        repeat_path = Path(previous["repeats"]) / "manifest.json"
        if repeat_path.exists() and read(repeat_path)["state"] == "running":
            raise RuntimeError("A recovery repetition scheduler is already running.")
    old_column_stopped()
    manifest = repeat_manifest(read(OLD_REPEATS / "manifest.json"), OLD_REPEATS)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    repeat_directory = directory / "repeats"
    write(repeat_directory / "manifest.json", manifest)
    column_directory = ROOT / "Distillation/Results/column_review" / (directory.name + "_recovered")
    prepare_column(column_directory)
    hold_snapshot_pilot(column_directory)
    command_file(directory / "repeats.cmd", "Column repetitions - 30 remaining - Aspen 1-10",
                 [Path(__file__).resolve(), "--run-repeats", repeat_directory])
    command_file(directory / "progress.cmd", "ALL REVIEW RUNS - live progress",
                 [ROOT / "show_review_progress.py", "--watch"])
    commands = [{"id": "column_repeats", "command": str(directory / "repeats.cmd")}]
    commands += [{"id": "column_" + method, "command": str(column_directory / method / "run.cmd")}
                 for method in ("sg", "threshold", "ls", "bo", "e1_pilot")]
    commands.append({"id": "progress", "command": str(directory / "progress.cmd")})
    result = {"directory": str(directory), "repeats": str(repeat_directory), "column": str(column_directory),
              "commands": commands, "prepared_at": now(), "runtime": RUNTIME,
              "repetitions_reused": len(manifest["reused_completed_results"]),
              "repetitions_remaining": sum(j["state"] != "completed" for j in manifest["jobs"]),
              "old_repeat_manifest_sha256": digest(OLD_REPEATS / "manifest.json"),
              "old_column_manifest_sha256": digest(OLD_COLUMN / "manifest.json"),
              "restart_semantics": "Failed column jobs restart original initial conditions; no full-state continuation exists.",
              "scientific_settings_changed": False, "timed_stack_dumps": False}
    result["snapshot_pilot_hold"] = read(column_directory / "e1_pilot/snapshot_hold.json")
    write(directory / "plan.json", result)
    write(POINTER, result)
    return result


def repeat_worker(path):
    from run_polymer_e3_recovery import environment
    os.environ.update(environment())
    from run_polymer_e3_recovery import runtime_info
    from column_review.common import record_owned_apps
    import run_distillation_phase2_repeats as repeat
    write(Path(path).parent / "isolated_runtime.json", runtime_info())
    with record_owned_apps(Path(path).parent):
        return repeat.run_worker(Path(path))


def run_repeats(directory):
    from run_polymer_e3_recovery import environment
    os.environ.update(environment())
    import run_distillation_phase2_repeats as repeat
    from column_review.common import now
    directory = Path(directory).resolve()
    with repeat.aspen_file_lock(directory / "supervisor", directory / "_locks"):
        manifest = read(directory / "manifest.json")
        if manifest["source_hashes"] != repeat.source_hashes(ROOT):
            raise RuntimeError("Original scientific sources differ from the approved recovery.")
        for job in manifest["jobs"]:
            for attempt in job["attempts"]:
                if attempt.get("pid"):
                    ensure_stopped(attempt["pid"], "repeat worker")
        write(directory / "supervisor.json", {"pid": os.getpid(), "started_at": now()})
        print("20 completed runs retained. Recovering Markov, then residual, then combined.", flush=True)
        return repeat.run_batches(manifest, directory, command_factory=lambda python, job:
                                  [python, "-B", "-u", str(Path(__file__).resolve()), "--repeat-worker", str(job)])


def schedule(directory, entry="--broker"):
    """One immediate interactive launch; no recurring trigger or timeout."""
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
    definition.RegistrationInfo.Description = "Author-approved recovery of remaining RL-assisted MPC experiments. Demand only."
    definition.Principal.UserId = user
    definition.Principal.LogonType = 3
    definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    settings.ExecutionTimeLimit = "PT0S"
    settings.MultipleInstances = 2
    action = definition.Actions.Create(0)
    action.Path = str(Path(sys.executable).with_name("pythonw.exe"))
    action.Arguments = subprocess.list2cmdline(["-B", "-u", str(Path(__file__).resolve()), entry, str(directory)])
    action.WorkingDirectory = str(ROOT)
    name = "RLAssistedMPC-Recovery-" + uuid.uuid4().hex[:12]
    task = service.GetFolder("\\").RegisterTaskDefinition(name, definition, 2, user, None, 3)
    task.Run(None)
    return {"task_name": name, "triggers": 0, "execution_time_limit": "PT0S"}


def broker(directory):
    from review_restart_recovery import lifetime, isolated_interactive_context
    from run_polymer_e3_recovery import environment
    from column_review.common import process_identity, MAPPING, now
    from run_distillation_phase2_repeats import aspen_file_lock
    directory = Path(directory).resolve()
    with aspen_file_lock(directory / "broker", directory / "_locks"), \
         (directory / "broker.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        try:
            identity = lifetime()
            write(directory / "broker_identity.json", identity)
            if not isolated_interactive_context(identity):
                raise RuntimeError("Launch broker must be owned by the interactive Windows scheduler.")
            plan = read(directory / "plan.json")
            old_column_stopped()
            manifest = read(Path(plan["repeats"]) / "manifest.json")
            # Reserve/check every case before launching any session.
            paths = {j["aspen_path"] for j in manifest["jobs"] if j["state"] != "completed"}
            paths.update(read(Path(plan["column"]) / method / "job.json")["aspen"]["aspen_path"] for method in MAPPING)
            with ExitStack() as stack:
                for path in sorted(paths):
                    stack.enter_context(aspen_file_lock(Path(path), ROOT / "Distillation/Results/_aspen_file_locks"))
            launches = []
            for item in plan["commands"]:
                process = subprocess.Popen(f'cmd.exe /d /c call "{item["command"]}"', cwd=ROOT,
                                           env=environment(), creationflags=subprocess.CREATE_NEW_CONSOLE)
                launches.append({**item, "process": process_identity(process.pid), "started_at": now()})
                write(directory / "launched.json", launches)
            write(directory / "broker_status.json", {"state": "launched", "sessions": len(launches), "updated_at": now()})
            while any(process_identity(item["process"]["pid"])["alive"] is True for item in launches):
                write(directory / "broker_heartbeat.json", {"pid": os.getpid(), "updated_at": now()})
                time.sleep(30)
        except BaseException as error:
            write(directory / "broker_status.json", {"state": "failed", "error": str(error), "updated_at": now()})
            traceback.print_exc()
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--launch", type=Path)
    group.add_argument("--broker", type=Path)
    group.add_argument("--run-repeats", type=Path)
    group.add_argument("--repeat-worker", type=Path)
    group.add_argument("--restart-repeats", type=Path)
    group.add_argument("--open-repeats", type=Path)
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
    if args.repeat_worker:
        return repeat_worker(args.repeat_worker)
    if args.restart_repeats:
        result = schedule(args.restart_repeats.resolve(), "--open-repeats")
        write(args.restart_repeats / "repeats_rescheduled.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.open_repeats:
        from run_polymer_e3_recovery import environment
        from review_restart_recovery import lifetime, isolated_interactive_context
        identity = lifetime()
        if not isolated_interactive_context(identity):
            raise RuntimeError("Repetition console must be launched independently of Codex.")
        process = subprocess.Popen(f'cmd.exe /d /c call "{args.open_repeats.resolve() / "repeats.cmd"}"',
                                   cwd=ROOT, env=environment(), creationflags=subprocess.CREATE_NEW_CONSOLE)
        write(args.open_repeats / "repeats_reopened.json", {"broker": identity, "cmd_pid": process.pid})
        return 0
    class Tee:
        def __init__(self, console, log):
            self.console, self.log = console, log
        def write(self, text):
            self.log.write(text)
            return self.console.write(text)
        def flush(self):
            self.log.flush()
            self.console.flush()
    with (args.run_repeats / "supervisor.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            return run_repeats(args.run_repeats)
        except BaseException as error:
            write(args.run_repeats / "supervisor_error.json", {"error": str(error), "traceback": traceback.format_exc()})
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
