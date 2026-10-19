"""Prepare five isolated column jobs; live execution requires --launch/--run-job.

Default is a read-only plan. --prepare writes commands without opening Aspen.
Fixed mapping: SG 12, threshold 13, LS 14, BO 15, restoration pilot 16.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import argparse
import faulthandler
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid

from column_review.common import ROOT, MAPPING, read, write, digest, own_sources, now, process_identity, owned_apps_remaining, check_job


def plan(root_override=None):
    from review_experiments.e3.runtime import resolve_aspen
    from column_review.typelib import inspect_api
    from systems.distillation import get_distillation_notebook_defaults
    defaults = get_distillation_notebook_defaults("combined")
    if defaults["td3_post_warm_start_action_freeze_subepisodes"] != 3 or defaults["td3_post_warm_start_actor_freeze_subepisodes"] != 3:
        raise ValueError("The original three-cycle critic-only/action-freeze configuration changed.")
    cases = {method: {"number": number, **resolve_aspen("combined", root_override=root_override, number_override=number)}
             for method, number in MAPPING.items()}
    if len({item["aspen_sha256"] for item in cases.values()}) != 1:
        raise ValueError("Cases 12--16 differ; verify comparable original plant copies before preparation.")
    if len({item["aspen_path"] for item in cases.values()}) != 5 or len({item["snaps_path"] for item in cases.values()}) != 5:
        raise ValueError("Each concurrent study needs its own model and snapshot directory.")
    return {"schema": 1, "plant": "distillation", "mapping": cases,
            "profile": "legacy_200", "seed": 7, "cycles_per_comparison": 200,
            "steps_per_cycle": 400, "warmup_cycles": [1, 10], "critic_only_cycles": [11, 13],
            "policy_release_cycle": 14, "final_evaluation_cycle": 200,
            "phase_wording": "Paper calls cycles 11--13 the handoff phase: critics learn while executed actions and TD3 actors remain frozen. Policy eligibility starts in cycle 14, with native inclusive sample boundaries preserved.",
            "bo_trials": 40, "bo_calibration_steps": 192000,
            "pilot": {"purpose": "Native snapshot restoration, repeated input continuation and observer-copy pilot",
                      "prefix_steps": 400, "continuation_steps": 24, "repeats": 3,
                      "production_E1_started_automatically": False},
            "installed_snapshot_api": inspect_api(),
            "new_processes_started": False, "live_aspen_opened": False,
            "protected_cases": list(range(1, 12))}


def prepare(directory, root_override=None):
    from revision_comparisons.column import source_hashes
    summary = plan(root_override)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    jobs = []
    hashes, launch_hashes = source_hashes(), own_sources()
    for method, number in MAPPING.items():
        output = directory / method
        output.mkdir()
        aspen = {key: value for key, value in summary["mapping"][method].items() if key != "number"}
        job = {"method": method, "aspen_number": number, "aspen": aspen, "seed": 7,
               "profile": "snapshot_pilot" if method == "e1_pilot" else "legacy_200",
               "threads": 1, "bo_trials": 40, "source_hashes": hashes, "parallel_source_hashes": launch_hashes}
        write(output / "job.json", job)
        command = subprocess.list2cmdline([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--run-job", str(output / "job.json")])
        script = (f'@echo off\nsetlocal\ntitle Column {method} - Aspen {number}\ncd /d "{ROOT}"\n'
                  'call "C:\\ProgramData\\anaconda3\\Scripts\\activate.bat" rl-assisted-env\n'
                  + command + '\nset result=%errorlevel%\necho.\necho Job exited with code %result%.\npause\nexit /b %result%\n')
        (output / "run.cmd").write_text(script, encoding="utf-8")
        jobs.append({"id": method, "method": method, "aspen_number": number, "specification": job,
                     "state": "pending", "attempts": []})
    write(directory / "prepared_plan.json", summary)
    write(directory / "manifest.json", {"schema": 1, "state": "prepared", "created_at": now(),
                                        "python": sys.executable, "jobs": jobs,
                                        "source_hashes": hashes, "parallel_source_hashes": launch_hashes})
    return summary


@contextmanager
def metadata_lock(directory):
    from run_distillation_phase2_repeats import aspen_file_lock
    for retry in range(30):
        lock = aspen_file_lock(Path(directory) / "metadata", Path(directory) / "_locks")
        try:
            lock.__enter__()
            break
        except RuntimeError:
            if retry == 29:
                raise
            time.sleep(.1)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def refresh(directory):
    directory = Path(directory)
    with metadata_lock(directory):
        manifest = read(directory / "manifest.json")
        for job in manifest["jobs"]:
            record = read(directory / job["method"] / "execution.json", {})
            job["state"] = record.get("state", "pending")
            job["attempts"] = read(directory / job["method"] / "attempts.json", [])
        states = [job["state"] for job in manifest["jobs"]]
        manifest["state"] = ("running" if "running" in states else "completed" if all(state == "completed" for state in states)
                             else "failed" if "failed" in states else "prepared")
        manifest["updated_at"] = now()
        write(directory / "manifest.json", manifest)
        comparisons = [job for job in manifest["jobs"] if job["method"] != "e1_pilot"]
        if all(job["state"] == "completed" for job in comparisons):
            from run_distillation_comparisons import summarize
            summarize(directory, {"jobs": comparisons})
    return manifest


def worker(path):
    from column_review.common import case_lease, record_owned_apps
    from run_polymer_e3_recovery import runtime_info
    path = Path(path).resolve()
    output, job = path.parent, read(path)
    status = {"state": "running", "pid": os.getpid(), "method": job["method"], "started_at": now()}
    write(output / "status.json", status)
    try:
        check_job(job)
        write(output / "isolated_runtime.json", runtime_info())
        output.joinpath("simulator_cleanup", "AM_owned").mkdir(parents=True, exist_ok=True)
        # Cleanup is confined to this attempt. Previous native case snapshots
        # and intentionally exported pilot snapshots cannot be removed by close.
        job["aspen"]["original_snaps_path"] = job["aspen"]["snaps_path"]
        job["aspen"]["snaps_path"] = str(output / "simulator_cleanup/AM_owned")
        runtime_job = output / "runtime_job.json"
        write(runtime_job, job)
        faulthandler.enable()
        # Progress belongs in the supervisor process. Scheduled all-thread
        # stack dumps coincided with four native crashes in the old attempts.
        # Keep fatal-error diagnostics, without a timed in-worker watchdog.
        with case_lease(job["aspen"]), record_owned_apps(output):
            if job["method"] == "e1_pilot":
                from column_review.pilot import run
                run(output, job["aspen"], job["seed"])
                status.update(state="completed", aspen_closed=True, finished_at=now())
                status["files"] = {file.relative_to(output).as_posix(): digest(file) for file in output.rglob("*")
                                   if file.is_file() and file.name in {"pilot_result.json", "continuations.npz", "python_checkpoint.pkl", "snapshot_api.json"}}
                status["files"].update({file.relative_to(output).as_posix(): digest(file)
                                        for file in (output / "preserved_snapshots").glob("*.snp")})
                write(output / "status.json", status)
                code = 0
            else:
                from run_distillation_comparisons import worker as original_worker
                code = original_worker(runtime_job)
        if code == 0:
            check_job(job)
        return code
    except BaseException as error:
        status.update(state="failed", finished_at=now(), error=str(error), error_type=type(error).__name__)
        write(output / "status.json", status)
        traceback.print_exc()
        return 1


def completed(attempt, expected):
    from run_distillation_comparisons import completed as comparison_completed
    attempt = Path(attempt)
    if not read(attempt / "owned_aspen_apps.json", []) or owned_apps_remaining(attempt / "owned_aspen_apps.json"):
        return False
    if expected["method"] != "e1_pilot":
        return comparison_completed(attempt, expected)
    status = read(attempt / "status.json", {})
    result = read(attempt / "pilot_result.json", {})
    required = {"pilot_result.json", "continuations.npz", "python_checkpoint.pkl", "snapshot_api.json"}
    snapshot = result.get("native_snapshot")
    if not isinstance(snapshot, str) or not (attempt / snapshot).resolve().is_relative_to(attempt.resolve()):
        return False
    required.add(snapshot)
    return (status.get("state") == "completed" and status.get("aspen_closed") is True
            and result.get("pilot_passed") is True and result.get("e1_critic_validation_performed") is False
            and status.get("files", {}).get(snapshot) == result.get("native_snapshot_sha256")
            and required <= status.get("files", {}).keys()
            and all((attempt / name).is_file() and digest(attempt / name) == expected_hash
                    for name, expected_hash in status["files"].items()))


def run_job(path):
    from run_distillation_phase2_repeats import aspen_file_lock
    from revision_comparisons.runtime import numerical_environment
    path = Path(path).resolve()
    stage, job = path.parent, read(path)
    batch = stage.parent
    check_job(job)
    with aspen_file_lock(stage / "supervisor", batch / "_locks"):
        attempts = read(stage / "attempts.json", [])
        for previous in attempts:
            identity = process_identity(previous["pid"])
            if identity["alive"] is None or (identity["alive"] is True and
                    (identity["created"] is None or identity["created"] == previous.get("created"))):
                raise RuntimeError("Previous worker is still live or cannot be checked; refusing duplicate execution.")
            if owned_apps_remaining(Path(previous["directory"]) / "owned_aspen_apps.json"):
                raise RuntimeError("An earlier owned Aspen process remains; inspect its case before restart.")
        if attempts and completed(attempts[-1]["directory"], job):
            print("Verified completed job reused; no duplicate simulation.", flush=True)
            return 0
        attempt = stage / f"a{len(attempts) + 1}"
        attempt.mkdir()
        write(attempt / "job.json", job)
        write(stage / "execution.json", {"state": "running", "pid": os.getpid(), "started_at": now(), "attempt": str(attempt)})
        print(f"{job['method']}: reserved Aspen case {job['aspen_number']}; {sys.executable}\n{attempt}", flush=True)
        print(("Native snapshot pilot: one 400-step MPC prefix and three identical 24-step continuations; no RL training."
               if job["method"] == "e1_pilot" else
               "Cycles 11--13: critic-only training under supervisor control; policy eligibility begins in cycle 14."), flush=True)
        log_path = attempt / "worker.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-job", str(attempt / "job.json")],
                                       cwd=ROOT, env=numerical_environment(1), stdout=log, stderr=subprocess.STDOUT)
            record = {"directory": str(attempt), **process_identity(process.pid), "state": "running", "started_at": now()}
            attempts.append(record)
            write(stage / "attempts.json", attempts)
            refresh(batch)
            print(f"Worker PID {process.pid}; live progress follows.", flush=True)
            last_heartbeat = began = time.monotonic()
            last_log, tail = began, ""
            with log_path.open(encoding="utf-8", errors="replace") as reader:
                try:
                    while process.poll() is None:
                        chunk = reader.read()
                        if chunk:
                            print(chunk, end="", flush=True)
                            tail = (tail + chunk)[-6000:]
                            last_log = time.monotonic()
                        current = time.monotonic()
                        if current - last_heartbeat >= 30:
                            cycles = re.findall(r"Sub_Episode:\s*(\d+)", tail)
                            heartbeat = {"worker_pid": process.pid, "updated_at": now(), "elapsed_seconds": current - began,
                                         "seconds_since_output": current - last_log, "last_cycle": int(cycles[-1]) if cycles else None}
                            write(stage / "heartbeat.json", heartbeat)
                            print(f"[progress] {job['method']} worker alive; last cycle {heartbeat['last_cycle']}; "
                                  f"last output {heartbeat['seconds_since_output']:.0f}s ago", flush=True)
                            last_heartbeat = current
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("Waiting for this worker's normal owned-case cleanup; other jobs continue.", flush=True)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError("Worker is still alive; do not launch another copy. Existing ownership records are retained.")
                print(reader.read(), end="", flush=True)
            code = process.wait()
        # Quit can return shortly before the owned server exits. Never signal it
        # or another Aspen instance; wait briefly, then report unverified cleanup.
        until = time.monotonic() + 20
        while owned_apps_remaining(attempt / "owned_aspen_apps.json") and time.monotonic() < until:
            time.sleep(.5)
        valid = code == 0 and completed(attempt, job)
        record.update(state="completed" if valid else "failed", returncode=code, finished_at=now())
        write(stage / "attempts.json", attempts)
        write(stage / "execution.json", {"state": record["state"], "pid": os.getpid(), "exit_code": code,
                                          "finished_at": now(), "attempt": str(attempt)})
        if not valid:
            status = read(attempt / "status.json", {})
            status.update(state="failed", parent_error="Nonzero worker exit, incomplete artifacts or unverified Aspen cleanup.",
                          returncode=code, owned_apps_remaining=owned_apps_remaining(attempt / "owned_aspen_apps.json"))
            write(attempt / "status.json", status)
        try:
            refresh(batch)
        except Exception as error:
            write(stage / "postprocess_status.json", {"state": "failed", "error": str(error)})
            print(f"Evidence assembly needs attention: {error}", flush=True)
        print(f"{job['method']}: {record['state']}. Failed attempts remain saved; other jobs run independently.", flush=True)
        return 0 if valid else 1


def launch(batch):
    from run_distillation_phase2_repeats import aspen_file_lock
    batch = Path(batch).resolve()
    for method in MAPPING:
        job = read(batch / method / "job.json")
        check_job(job)
        execution = read(batch / method / "execution.json", {})
        if execution.get("state") == "running" and process_identity(execution.get("pid"))["alive"] is not False:
            raise RuntimeError(f"{method} already has a live supervisor.")
        if execution.get("state") == "completed":
            attempts = read(batch / method / "attempts.json", [])
            if not attempts or not completed(attempts[-1]["directory"], job):
                raise RuntimeError(f"{method} is labeled completed but its saved artifacts/cleanup no longer verify.")
    # Check every selected case before opening any CMD. Workers acquire the same
    # locks again for the full lifetime; a later external race fails safely.
    with ExitStack() as stack:
        for method in MAPPING:
            execution = read(batch / method / "execution.json", {})
            if execution.get("state") != "completed":
                job = read(batch / method / "job.json")
                stack.enter_context(aspen_file_lock(Path(job["aspen"]["aspen_path"]), ROOT / "Distillation/Results/_aspen_file_locks"))
    results = []
    for method in MAPPING:
        execution = read(batch / method / "execution.json", {})
        if execution.get("state") == "completed":
            continue
        command = batch / method / "run.cmd"
        process = subprocess.Popen(f'cmd.exe /d /c call "{command}"', cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
        results.append({"method": method, "aspen_number": MAPPING[method], "cmd_pid": process.pid})
    write(batch / "visible_launches.json", results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--aspen-root", type=Path)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-job", type=Path)
    parser.add_argument("--inspect-api", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_job:
        return worker(args.worker_job)
    if args.run_job:
        return run_job(args.run_job)
    if args.inspect_api:
        from column_review.typelib import inspect_api
        print(json.dumps(inspect_api(), indent=2))
        return 0
    if not args.prepare and not args.launch:
        print(json.dumps(plan(args.aspen_root), indent=2))
        return 0
    batch = args.batch or ROOT / "Distillation/Results/column_review" / (time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:6])
    if not args.batch or not (batch / "prepared_plan.json").exists():
        prepare(batch, args.aspen_root)
    else:
        for method in MAPPING:
            check_job(read(batch / method / "job.json"))
    print(f"Prepared batch: {batch.resolve()}", flush=True)
    if args.launch:
        if os.name != "nt":
            raise RuntimeError("Visible CMD launching requires Windows.")
        print(json.dumps(launch(batch), indent=2))
    else:
        print("PREPARATION ONLY: no worker, Aspen application or training started.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
