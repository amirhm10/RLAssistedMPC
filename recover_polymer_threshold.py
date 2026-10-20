"""Recover missing threshold weights in a new, independently launched run.

Original trajectories/empty exports and running-job dependencies are preserved.
Periodic files are inference weights, never training-resume checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "Polymer/Results/comparisons/scenario1/20261014_runtime_recovery_threshold_sg/threshold"
AGENTS = ("horizon_agent", "weight_agent", "markov_agent", "residual_agent")


class PolicyExportManager:
    """Observe completed cycles in a private compilation, without changing RNG."""
    has_checkpoint = False

    def __init__(self, directory, metadata):
        self.directory, self.metadata = Path(directory), metadata
        self.compiled = None

    def due(self, step, environment):
        completed, size = int(step) + 1, int(environment["time_in_sub_episodes"])
        cycle = completed // size
        return completed % size == 0 and (cycle == 1 or cycle % 10 == 0 or completed == int(environment["nFE"]))

    def save(self, environment, session):
        from comparison_studies.checkpoints import save_inference_checkpoint
        from recover_markov_seed23 import write
        completed = int(environment[self.compiled.step_name]) + 1
        cycle = completed // int(environment["time_in_sub_episodes"])
        agents = {name: environment[name] for name in AGENTS}
        self.directory.mkdir(exist_ok=True)
        record = save_inference_checkpoint(self.directory / f"cycle_{cycle:04d}.pt", agents,
                 metadata={**self.metadata, "completed_steps": completed, "completed_cycles": cycle,
                           "training_resume": False})
        record.update(completed_steps=completed, completed_cycles=cycle)
        write(self.directory / "latest.json", record)
        print(f"Validated periodic policy export: cycle {cycle}, four nonempty agents.", flush=True)


def verify_queue():
    from column_review.common import read
    import continue_column_queue as queue
    directory = Path(read(ROOT / "Distillation/Results/review_recovery/latest.json")["repeats"])
    queue.validate_group(read(directory / "combined/manifest.json"))


def prepare(output, smoke):
    import review_blas_runtime
    review_blas_runtime.activate()
    from comparison_studies.common import source_hashes
    from column_review.common import read, digest, now
    from recover_markov_seed23 import write
    verify_queue()
    if not smoke:
        old_status = read(OLD / "status.json")
        if old_status["state"] != "completed" or digest(OLD / "training.pkl") != old_status["trajectory_sha256"]:
            raise RuntimeError("Historical completed threshold trajectory does not verify.")
        import torch
        payload = torch.load(OLD / "final_inference_networks.pt", weights_only=True, map_location="cpu")
        if set(payload) != set(AGENTS) or any(payload.values()):
            raise RuntimeError("Reinspect the old export: expected four empty agent dictionaries.")
    output.mkdir(parents=True, exist_ok=False)
    job = {"method": "threshold", "seed": 7, "profile": "legacy_gradual_200", "threads": 1,
           "smoke": smoke, "source_hashes": source_hashes(), "recovery_bootstrap_sha256": digest(__file__),
           "runtime_manifest_sha256": digest(review_blas_runtime.DEFAULT / "manifest.json"),
           "reason": "No usable original threshold network weights survived; preserve its valid trajectory and rerun only this method.",
           "original_directory": str(OLD), "prepared_at": now(),
           "periodic_policy_exports": "cycle 1, every 10 cycles and final; inference weights only"}
    write(output / "job.json", job)
    write(output / "manifest.json", {"state": "prepared", "job": str(output / "job.json"), "smoke": smoke, "prepared_at": now()})
    command = subprocess.list2cmdline([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--run-job", str(output)])
    (output / "run.cmd").write_text('@echo off\nsetlocal\ntitle Polymer threshold checkpoint recovery - seed 7\n'
        f'cd /d "{ROOT}"\n{command}\nset result=%errorlevel%\necho Study exited with code %result%.\npause\nexit /b %result%\n', encoding="utf-8")
    return job


def worker(job_path):
    import review_blas_runtime
    # Activate before scientific imports; no installation or shared DLL changes.
    job = json.loads(job_path.read_text(encoding="utf-8"))
    review_blas_runtime.activate(expected_manifest_hash=job["runtime_manifest_sha256"])
    from column_review.common import digest
    if job["method"] != "threshold" or job["seed"] != 7 or job["profile"] != "legacy_gradual_200":
        raise ValueError("This recovery supports only the original Scenario 1 threshold seed 7.")
    if digest(__file__) != job["recovery_bootstrap_sha256"]:
        raise RuntimeError("Recovery bootstrap changed after preparation.")
    verify_queue()
    import torch
    torch.set_num_threads(1)
    from revision_recovery.checkpoints import install_runner
    import review_experiments.e1.runtime as runtime
    import run_polymer_comparisons as comparisons
    manager = PolicyExportManager(job_path.parent / "policy_checkpoints",
                                  {"method": "threshold", "seed": 7, "profile": job["profile"],
                                   "smoke": job["smoke"], "source_hashes": job["source_hashes"]})
    def instrument(category, factory):
        if category != "combined":
            raise ValueError("Threshold recovery expects the unchanged combined runner.")
        return install_runner(category, factory, manager)
    with patch.object(runtime, "install_runner", instrument):
        return comparisons.worker(job_path)


def run_job(directory):
    from column_review.common import read, now, process_identity
    from recover_markov_seed23 import write
    from run_distillation_phase2_repeats import aspen_file_lock
    from run_polymer_e3_recovery import environment
    with aspen_file_lock(directory / "supervisor", directory / "_locks"):
        manifest = read(directory / "manifest.json")
        if manifest["state"] != "prepared":
            raise RuntimeError("Threshold recovery already started; refusing duplicate training.")
        attempt = directory / "a1"
        attempt.mkdir()
        job = read(directory / "job.json")
        write(attempt / "job.json", job)
        with (attempt / "run.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", str(attempt / "job.json")],
                                     cwd=ROOT, env={**environment(), "PYTHONHASHSEED": "7", "MPLBACKEND": "Agg"},
                                     stdout=log, stderr=subprocess.STDOUT)
            manifest.update(state="running", pid=child.pid, identity=process_identity(child.pid), directory=str(attempt), started_at=now())
            write(directory / "manifest.json", manifest)
            print(f"Threshold recovery worker PID={child.pid}; log={attempt / 'run.log'}", flush=True)
            previous_cycle = None
            while child.poll() is None:
                from revision_comparisons.progress import log_tail
                cycles = re.findall(r"Sub_Episode:\s*(\d+)", log_tail(attempt / "run.log", 6000))
                cycle = int(cycles[-1]) if cycles else 0
                if cycle != previous_cycle:
                    checkpoint = read(attempt / "policy_checkpoints/latest.json", {})
                    print(f"Threshold: cycle {cycle}/200; last validated policy cycle={checkpoint.get('completed_cycles', 'pending')}", flush=True)
                    previous_cycle = cycle
                time.sleep(15)
            code = child.wait()
        status = read(attempt / "status.json", {})
        if code == 0 and status.get("state") == "completed":
            from comparison_studies.checkpoints import load_inference_checkpoint
            payload = load_inference_checkpoint(attempt / "final_inference_networks.pt")
            if set(payload["agents"]) != set(AGENTS) or status["steps"] != (800 if job["smoke"] else 160000):
                raise RuntimeError("Recovered threshold final artifacts differ from the required contract.")
            manifest.update(state="completed", checkpoint=status["checkpoint"], steps=status["steps"])
        else:
            manifest.update(state="failed", error=status.get("error", "Worker exited without valid final status."))
        manifest.update(exit_code=code, finished_at=now())
        write(directory / "manifest.json", manifest)
        return code


def schedule(directory):
    import win32api
    import win32con
    import win32security
    import win32com.client
    from recover_markov_seed23 import write
    service = win32com.client.Dispatch("Schedule.Service")
    service.Connect()
    task = service.NewTask(0)
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    task.RegistrationInfo.Description = "Immediate author-approved threshold checkpoint recovery. Demand only."
    task.Principal.UserId, task.Principal.LogonType, task.Principal.RunLevel = user, 3, 0
    settings = task.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    settings.ExecutionTimeLimit, settings.MultipleInstances = "PT0S", 2
    action = task.Actions.Create(0)
    action.Path = str(Path(sys.executable).with_name("pythonw.exe"))
    action.Arguments = subprocess.list2cmdline(["-B", "-u", str(Path(__file__).resolve()), "--broker", str(directory)])
    action.WorkingDirectory = str(ROOT)
    name = "RLAssistedMPC-Threshold-Recovery-" + uuid.uuid4().hex[:12]
    registered = service.GetFolder("\\").RegisterTaskDefinition(name, task, 2, user, None, 3)
    write(directory / "scheduled.json", {"task_name": name, "triggers": 0, "execution_time_limit": "PT0S"})
    registered.Run(None)


def broker(directory):
    from review_restart_recovery import lifetime, isolated_interactive_context
    from recover_markov_seed23 import write
    identity = lifetime()
    if not isolated_interactive_context(identity):
        raise RuntimeError("Recovery broker must belong to the interactive Windows scheduler.")
    write(directory / "broker_identity.json", identity)
    process = subprocess.Popen(f'cmd.exe /d /c call "{directory / "run.cmd"}"', cwd=ROOT,
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
    write(directory / "visible_launch.json", {"cmd_pid": process.pid})
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-job", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--broker", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve(), args.smoke)
    elif args.worker:
        return worker(args.worker.resolve())
    elif args.run_job:
        return run_job(args.run_job.resolve())
    elif args.launch:
        schedule(args.launch.resolve())
    elif args.broker:
        return broker(args.broker.resolve())
    else:
        parser.error("Select an explicit recovery action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
