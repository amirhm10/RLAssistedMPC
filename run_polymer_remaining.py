"""Concurrent, visible recovery of the three unfinished polymer reviewer jobs.

Default: read-only plan. --launch: create a fresh batch and open three CMDs.
Reopen one saved run.cmd to resume its E1 checkpoint without duplicate workers.
No new distillation jobs are started; original scientific sources are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
STAGES = ("e1_combined", "e2_sg", "e2_arx_repaired")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def own_sources():
    files = [Path(__file__).resolve(), *sorted((ROOT / "revision_recovery").glob("*.py")),
             ROOT / "run_polymer_e3_recovery.py"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def environment():
    from run_polymer_e3_recovery import environment as isolated
    return {**isolated(), "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}


def plan(smoke=False):
    import run_polymer_review_recovery as prior
    from revision_comparisons.progress import process_info
    retained = prior.build_plan(sys.executable, smoke=smoke)
    failed = ROOT / "Polymer/Results/review_recovery/20261015_181507_3aa70b/e1_combined/20261015_181507_cf7a4e/combined_s7/a1"
    old_status = read(failed / "status.json")
    if process_info(old_status.get("pid"))["alive"] is not False:
        raise RuntimeError("Cannot prove the prior E1 worker stopped; refusing a duplicate.")
    records_path = failed / "capture/decisions.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    e3 = read(ROOT / "Results/e3/20261015_042751_513bff/manifest.json")
    if any(job["state"] != "completed" for job in e3["jobs"]):
        raise RuntimeError("Polymer E3 inventory changed; inspect before scheduling.")
    for batch in (ROOT / "Polymer/Results/review_recovery").glob("*/concurrent_plan.json"):
        if read(batch).get("smoke"):
            continue
        for name in STAGES:
            path = batch.parent / name / "execution.json"
            if path.is_file() and process_info(read(path).get("pid"))["alive"] is True:
                raise RuntimeError(f"{name} already has a live recovery parent: {path}")
            status = batch.parent / name / "status.json"
            if status.is_file() and read(status).get("state") == "completed":
                raise RuntimeError(f"{name} already completed in {batch.parent}; reuse it, do not retrain.")
    return {"schema": 1, "smoke": smoke, "seed": 7, "python": sys.executable,
            "stages": list(STAGES), "concurrent": True, "new_distillation_jobs": [],
            "e1_cycles": 300, "e2_cycles": 200, "steps_per_cycle": 800,
            "warm_cycles": 10, "critic_only_cycles": [11, 12, 13], "first_policy_cycle": 14,
            "e1_checkpoint_cycles": "1, then every 10, and 300; two atomic rotating slots",
            "old_E1_partial": {"directory": str(failed), "decisions": len(records),
                "completed_decisions": sum(r["status"] == "completed" for r in records),
                "decisions_sha256": digest(records_path), "restart_from_cycle": 1,
                "reason": "Native access violation; old inference snapshots lack optimizers/replay."},
            "retained_evidence": retained["retained_evidence"],
            "polymer_E3": {"state": "completed", "analysis": e3["latest_analysis"]},
            "runtime_environment": {key: environment()[key] for key in
                ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            "scope": "three polymer jobs only; completed jobs and live column jobs are preserved"}


def prepare(output, smoke=False):
    import run_polymer_e1 as e1
    import run_polymer_review_recovery as prior
    import run_polymer_arx_revision as arx
    from comparison_studies.common import source_hashes as comparison_sources
    value = plan(smoke)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    old = read(prior.E1_STUDY / "manifest.json")
    options = dict(old["options"], seed=7)
    if smoke:
        options.update(decisions_per_window=1, rollout_steps=8, horizons=[4, 8],
                       long_rollout_steps=8, long_fraction=0., smoke=True)
    specifications = {
        "e1_combined": {"category": "combined", "seed": 7, "options": options,
                        "source_hashes": e1.source_hashes(), "threads": 1, "smoke": smoke},
        "e2_sg": {"method": "sg", "seed": 7, "profile": "legacy_gradual_200", "threads": 1,
                  "smoke": smoke, "source_hashes": comparison_sources()},
        "e2_arx_repaired": {"method": "arx_regularized_v1", "seed": 7,
                            "smoke": smoke, "source_hashes": arx.source_hashes()},
    }
    manifest = {"schema": 1, "smoke": smoke, "started_at": datetime.now(timezone.utc).isoformat(),
                "python": sys.executable, "jobs": []}
    for stage, job in specifications.items():
        directory = output / stage
        directory.mkdir()
        job.update(recovery_stage=stage, recovery_source_hashes=own_sources())
        write(directory / "job.json", job)
        title = f"Polymer {stage} - {'TECHNICAL SMOKE' if smoke else 'seed 7'}"
        command = subprocess.list2cmdline([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                          "--run-job", str(directory / "job.json")])
        script = ("@echo off\nsetlocal\n" + f"title {title}\ncd /d \"{ROOT}\"\n" +
                  'call "C:\\ProgramData\\anaconda3\\Scripts\\activate.bat" rl-assisted-env\n' +
                  command + "\nset result=%errorlevel%\necho.\necho Study process exited with code %result%.\n"
                  "echo Results and logs are retained beside this CMD file.\npause\nexit /b %result%\n")
        (directory / "run.cmd").write_text(script, encoding="utf-8")
        manifest["jobs"].append({"id": stage, "category": "combined" if stage == "e1_combined" else stage,
                                 "seed": 7, "state": "pending", "attempts": [{"directory": str(directory)}]})
    write(output / "concurrent_plan.json", value)
    write(output / "manifest.json", manifest)
    return value


def worker(job_path, test_stop=None):
    job_path = Path(job_path).resolve()
    job = read(job_path)
    if job["recovery_source_hashes"] != own_sources():
        raise RuntimeError("Recovery implementation changed; inspect provenance before resuming.")
    from run_polymer_e3_recovery import runtime_info
    write(job_path.parent / "isolated_runtime.json", runtime_info())
    stage = job["recovery_stage"]
    if stage == "e1_combined":
        import review_experiments.e1.runtime as original
        from revision_recovery.checkpoints import CheckpointManager, install_runner
        from unittest.mock import patch
        manager = CheckpointManager(job_path.parent / "training_checkpoints", job["recovery_source_hashes"] | job["source_hashes"],
                                    stop_after_checkpoint=test_stop)
        install = lambda category, factory: install_runner(category, factory, manager)
        with patch.object(original, "install_runner", install):
            from run_polymer_e1 import worker as e1_worker
            return e1_worker(job_path)
    if stage == "e2_sg":
        if not job["smoke"]:
            from run_polymer_comparisons import baseline_audit
            from comparison_studies.common import prepare_nominal, episode_bundle
            print("Verifying the existing offset-free MPC baseline (800-step audit).", flush=True)
            namespace = prepare_nominal(job["seed"], job_path.parent)
            report, _ = baseline_audit(namespace, episode_bundle(namespace))
            write(job_path.parent / "baseline_audit.json", report)
            if not report["usable"]:
                raise RuntimeError("The original nominal baseline failed its matching-protocol audit.")
            print("Existing nominal MPC baseline verified: True; starting independent E2 SG.", flush=True)
        from run_polymer_comparisons import worker as sg_worker
        return sg_worker(job_path)
    if stage == "e2_arx_repaired":
        from run_polymer_arx_revision import worker as arx_worker
        return arx_worker(job_path)
    raise ValueError(f"Unexpected recovery stage: {stage}")


def run_job(path):
    from run_distillation_phase2_repeats import aspen_file_lock
    path = Path(path).resolve()
    job = read(path)
    stage, output = job["recovery_stage"], path.parent
    lock_root = ROOT / "Polymer/Results/_review_recovery_locks"
    # Separate locks allow concurrency, but forbid two copies of the same job.
    key = f"{stage}_{'smoke' if job['smoke'] else 'production'}_s{job['seed']}"
    with aspen_file_lock(lock_root / key, lock_root):
        if (output / "status.json").is_file() and read(output / "status.json").get("state") == "completed":
            validate_output(path, read(output / "status.json"))
            print("This job is already completed; no duplicate run started.", flush=True)
            return 0
        write(output / "execution.json", {"state": "running", "pid": os.getpid(),
                                           "started_at": datetime.now(timezone.utc).isoformat()})
        print(f"{stage}: {sys.executable}\nResults: {output}\nOriginal controllers/settings unchanged.", flush=True)
        log_path = output / "worker.log"
        attempts = read(output / "process_attempts.json") if (output / "process_attempts.json").is_file() else []
        for retry in range(2):
            # Preserve a failed export before replaying the end of a checkpoint.
            if stage == "e1_combined" and (output / "training.pkl").exists():
                (output / "training.pkl").replace(output / f"training_before_resume_{time.time_ns()}.pkl")
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-job", str(path)]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== Process attempt {len(attempts)+1}; {datetime.now(timezone.utc).isoformat()} ===\n")
                log.flush()
                with log_path.open(encoding="utf-8", errors="replace") as reader:
                    reader.seek(0, 2)
                    process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
                    print(f"Worker PID={process.pid}; live progress follows.", flush=True)
                    while process.poll() is None:
                        chunk = reader.read()
                        if chunk:
                            print(chunk, end="", flush=True)
                        time.sleep(1)
                    print(reader.read(), end="", flush=True)
                    code = process.wait()
            status_path = output / "status.json"
            status = read(status_path) if status_path.exists() else {}
            attempts.append({"pid": process.pid, "exit_code": code, "status": status,
                             "finished_at": datetime.now(timezone.utc).isoformat()})
            write(output / "process_attempts.json", attempts)
            native = code & 0xffffffff == 0xc0000005
            if retry == 0 and native and stage == "e1_combined" and (output / "training_checkpoints/latest.json").is_file():
                print("Native failure: retaining failed attempt and resuming the last full E1 checkpoint once.", flush=True)
                continue
            valid = code == 0 and status.get("state") == "completed"
            if valid:
                try:
                    validate_output(path, status)
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    valid, code = False, 1
                    status.update(state="failed", artifact_validation_error=str(error))
                    write(status_path, status)
            if not valid and status.get("state") not in ("failed", "completed"):
                status.update(state="failed", error=f"Worker terminated with exit code {code}", native_exit_code=code)
                write(status_path, status)
            write(output / "execution.json", {"state": "completed" if valid else "failed", "pid": os.getpid(),
                                               "exit_code": code, "finished_at": datetime.now(timezone.utc).isoformat()})
            print(f"{stage}: {'COMPLETED and saved trajectory verified' if valid else 'FAILED; inspect worker.log'}", flush=True)
            if valid:
                from revision_recovery.analysis import finalize
                try:
                    # Separate finalizers never alter another live controller.
                    with aspen_file_lock(output.parent / "analysis", output.parent / "_locks"):
                        analysis = finalize(stage, output.parent)
                    write(output / "postprocess_status.json", analysis)
                    print(f"Combined evidence analysis: {analysis['state']}", flush=True)
                except Exception as error:
                    write(output / "postprocess_status.json", {"state": "failed", "error": str(error)})
                    print(f"Simulation is complete; evidence assembly needs attention: {error}", flush=True)
            return 0 if valid else code or 1


def validate_output(path, status):
    path = Path(path)
    job = read(path)
    output = path.parent
    if status.get("trajectory_sha256") != digest(output / "training.pkl"):
        raise RuntimeError("Completed trajectory checksum did not verify.")
    if job["recovery_stage"] == "e1_combined":
        expected = 16 if job["smoke"] else 1600
        capture = read(output / "capture/capture_status.json")
        if (status.get("decisions") != expected or status.get("failed_pairs") != 0
                or capture.get("state") != "completed" or capture.get("saved_decisions") != expected):
            raise ValueError("E1 matched-decision inventory is incomplete or has failed branches.")
    else:
        if status.get("steps") != (800 if job["smoke"] else 160000):
            raise ValueError("E2 trajectory has the wrong control-step count.")
        if job["recovery_stage"] == "e2_sg":
            checkpoint = status.get("checkpoint", {})
            if (not checkpoint.get("round_trip_verified") or len(checkpoint.get("tensor_counts", {})) != 4
                    or not all(checkpoint["tensor_counts"].values())
                    or checkpoint.get("sha256") != digest(output / "final_inference_networks.pt")):
                raise ValueError("E2 SG requires four populated, validated final networks.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Prepare jobs for inspection; do not launch.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-job", type=Path)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-test-stop", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_job:
        return worker(args.worker_job, args.checkpoint_test_stop)
    if args.run_job:
        return run_job(args.run_job)
    if not args.launch and not args.prepare:
        print(json.dumps(plan(args.smoke), indent=2))
        return 0
    output = (args.output_dir or ROOT / "Polymer/Results/review_recovery" /
              (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_concurrent_" + uuid.uuid4().hex[:6])).resolve()
    prepare(output, args.smoke)
    print(f"Prepared three independent polymer jobs: {output}", flush=True)
    if args.launch:
        if os.name != "nt":
            raise RuntimeError("Visible CMD launching requires Windows; saved job commands remain available.")
        launches = []
        for stage in STAGES:
            script = output / stage / "run.cmd"
            process = subprocess.Popen(f'cmd.exe /d /c call "{script}"', cwd=ROOT,
                                       creationflags=subprocess.CREATE_NEW_CONSOLE)
            launches.append({"stage": stage, "cmd_pid": process.pid, "script": str(script)})
            print(f"Opened visible CMD: {stage} (PID {process.pid})", flush=True)
        write(output / "visible_launches.json", launches)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
