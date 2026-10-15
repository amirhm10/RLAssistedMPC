"""E3: four one-seed SG studies and two shared, independent nominal probes.

Use --dry-run to inspect the plan without scientific imports or file creation.
Production training is launched only by explicitly running this command.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import signal
import subprocess
import sys
import traceback
import uuid

from review_experiments.e3.common import (
    ENTRIES, PROFILES, ROOT, check_sources, sha256, source_hashes, utc_now, write_json,
)


def build_jobs(plants, categories, seed):
    jobs = []
    for plant in plants:
        jobs.append({"id": f"{plant}_nominal", "plant": plant, "category": "nominal", "seed": seed,
                     "state": "pending", "attempts": []})
        for category in categories:
            jobs.append({"id": f"{plant}_{category}_s{seed}", "plant": plant, "category": category,
                         "seed": seed, "state": "pending", "attempts": []})
    return jobs


def validate_options(options):
    if options["seed"] < 0 or options["samples_per_window"] < 1 or options["threads"] < 1:
        raise ValueError("Seed must be nonnegative; sampling budget and threads must be positive.")
    if options.get("aspen_number") is not None:
        if options["aspen_number"] < 1:
            raise ValueError("Aspen simulation number must be positive.")
        if "distillation" not in options["plants"]:
            raise ValueError("--aspen-number requires selecting distillation.")
    horizons = options["horizons"]
    if not horizons or sorted(set(horizons)) != list(horizons) or min(horizons) < 1:
        raise ValueError("Horizons must be distinct, increasing positive integers.")
    if options["smoke"] and options["plants"] != ["polymer"]:
        raise ValueError("CLI smoke is polymer-only; distillation is tested with fake plant/COM fixtures.")
    # Validate fixed budgets before creating a study or opening a simulator.
    length = max(horizons)
    for plant in options["plants"]:
        episode = (4 if options["smoke"] else (800 if plant == "polymer" else 400))
        if options["samples_per_window"] > (7 * episode) // length:
            raise ValueError("Sampling budget exceeds complete non-overlapping futures in the release window.")
        nominal_cycle = 32 if options["smoke"] else (800 if plant == "polymer" else 400)
        if nominal_cycle // (2 * length) < (1 if options["smoke"] else 25):
            raise ValueError("Requested forecast horizon exceeds the nominal validation budget.")


def preflight(python, jobs, aspen_root=None, aspen_number=None):
    request = {"categories": sorted({j["category"] for j in jobs if j["plant"] == "distillation"}),
               "aspen_root": str(aspen_root) if aspen_root else None, "aspen_number": aspen_number}
    code = (
        "import json,sys; import numpy,scipy,torch,control; "
        "from review_experiments.e3.runtime import resolve_aspen; "
        "r=json.load(sys.stdin); "
        "print(json.dumps({c:resolve_aspen(c,r['aspen_root'],r['aspen_number']) for c in r['categories']}))"
    )
    checked = subprocess.run([str(python), "-c", code], cwd=ROOT, input=json.dumps(request),
                             text=True, capture_output=True)
    if checked.returncode:
        raise RuntimeError(f"E3 environment/path preflight failed:\n{checked.stderr}")
    paths = json.loads(checked.stdout)
    planned = []
    for job in jobs:
        job = dict(job)
        if job["plant"] == "distillation":
            job["aspen"] = paths[job["category"]]
        planned.append(job)
    return planned


def git_metadata():
    def read(*arguments):
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"head": read("rev-parse", "HEAD"), "working_tree_status": read("status", "--porcelain")}


def _runtime_metadata(namespace, plant, options):
    import numpy as np
    import scipy
    import torch
    data = namespace["system_data"]
    metadata = {
        "python": sys.executable, "python_version": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numerical_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "device": "cuda" if torch.cuda.is_available() else "cpu", "options": options,
        "effective_defaults": namespace["NB"], "delta_t_hours": float(namespace["result_bundle"]["delta_t"]),
        "augmented_model_order": int(np.asarray(namespace["A_aug"]).shape[0]),
        "unaugmented_identified_model_order": int(np.asarray(data["A"]).shape[0]) if data.get("A") is not None else None,
        "scaling": "stored identification min/max; input/output deviations from nominal steady state",
        "data_directory": str(namespace["DATA_DIR"]),
        "identification_artifacts": {p.name: sha256(p) for p in Path(namespace["DATA_DIR"]).iterdir()
                                     if p.is_file() and p.name in {"system_dict", "system_dict.pickle", "scaling_factor.pickle", "min_max_states.pickle"}},
        "identification_procedure": "Existing identified model retained without refitting.",
        "missing_identification_details": ["Complete original excitation and fitting/validation split must be verified from identification records.",
                                           "Order-selection rationale cannot be reconstructed from matrix dimensions alone."],
        "output_units": (["L/g", "K"] if plant == "polymer" else ["fraction", "native simulator temperature; unverified"]),
        "smoke": options["smoke"],
    }
    return metadata


def run_worker(job_path):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output, options = job_path.parent, job["options"]
    status = {"state": "running", "pid": os.getpid(), "started_at": utc_now(),
              "plant": job["plant"], "category": job["category"], "smoke": options["smoke"]}
    write_json(output / "status.json", status)
    try:
        check_sources(job["source_hashes"])
        os.chdir(ROOT)
        import numpy as np
        import torch
        from review_experiments.e3.capture import CaptureOptions, CaptureSession
        from review_experiments.e3.nominal import monitor_nominal, save_nominal
        from review_experiments.e3.runtime import aspen_resources, configure_defaults, execute_entrypoint, passive_capture

        torch.set_num_threads(options["threads"])
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job["seed"])

        def configure(defaults):
            configured = configure_defaults(defaults, job["plant"], job["category"], job["seed"], output, job.get("aspen"))
            if options["smoke"] and job["category"] == "nominal":
                configured["set_points_len_override"] = 16
            return configured

        with ExitStack() as stack:
            if job["plant"] == "distillation":
                stack.enter_context(aspen_resources(job["aspen"]))
            if job["category"] == "nominal":
                diagnostics = stack.enter_context(monitor_nominal(output / "stop_requested"))
            else:
                if options["smoke"]:
                    from review_experiments.e1.smoke import compressed_smoke_clock
                    stack.enter_context(compressed_smoke_clock(job["category"]))
                capture_options = CaptureOptions(seed=job["seed"], horizons=tuple(options["horizons"]),
                                                 samples_per_window=options["samples_per_window"], smoke=options["smoke"])
                session = CaptureSession(output / "capture", job["plant"], job["category"], capture_options,
                                         stop_file=output / "stop_requested")
                stack.enter_context(passive_capture(job["category"], session))
            namespace = execute_entrypoint(job["plant"], job["category"], configure)
        status["aspen_closed"] = job["plant"] == "distillation"
        bundle = namespace["result_bundle"]
        if job["category"] == "nominal":
            artifact = save_nominal(namespace, output, diagnostics, smoke=options["smoke"])
        else:
            if namespace["TRAINING_PROFILE_NAME"] != PROFILES[job["plant"]] or len(bundle["avg_rewards"]) != 300:
                raise RuntimeError("E3 did not preserve the original 300-episode Scenario 2 study.")
            capture_status = output / "capture" / "capture_status.json"
            if not capture_status.is_file() or json.loads(capture_status.read_text())["state"] != "completed":
                raise RuntimeError("Training returned without completing the required E3 predictor capture.")
            artifact = output / "training.pkl"
            with artifact.open("xb") as stream:
                pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(output / "runtime_metadata.json", _runtime_metadata(namespace, job["plant"], options))
        check_sources(job["source_hashes"])
        if job.get("aspen", {}).get("aspen_sha256") and sha256(job["aspen"]["aspen_path"]) != job["aspen"]["aspen_sha256"]:
            raise RuntimeError("Aspen model file changed during the E3 stage; preserve and inspect this attempt.")
        files = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*")
                 if p.is_file() and p.name not in {"status.json", "worker.log", "job.json"}}
        status.update(state="completed", finished_at=utc_now(), artifact=str(artifact), artifact_sha256=sha256(artifact), files=files)
        write_json(output / "status.json", status)
        return 0
    except BaseException as error:
        status.update(state="failed", finished_at=utc_now(), error=str(error), error_type=type(error).__name__)
        write_json(output / "status.json", status)
        traceback.print_exc()
        return 1


def completed_job(job):
    if job.get("state") != "completed" or not job.get("attempts"):
        return False
    directory = Path(job["attempts"][-1]["directory"])
    try:
        status = json.loads((directory / "status.json").read_text())
        artifact = Path(status["artifact"])
        return (status["state"] == "completed" and artifact.is_file() and
                sha256(artifact) == status["artifact_sha256"] and
                bool(status.get("files")) and all((directory / name).is_file() and sha256(directory / name) == digest
                                                for name, digest in status["files"].items()) and
                (job["plant"] != "distillation" or status.get("aspen_closed") is True))
    except (OSError, ValueError, KeyError):
        return False


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # access denied still means not known dead
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError("Cannot determine prior E3 worker state; inspect it before resuming.")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reject_live_attempts(jobs):
    for job in jobs:
        if not job.get("attempts"):
            continue
        attempt = job["attempts"][-1]
        if attempt.get("state") not in {"running", "reserved", "interrupted"}:
            continue
        path = Path(attempt["directory"]) / "status.json"
        status = json.loads(path.read_text()) if path.is_file() else {}
        if status.get("state") == "running" or attempt.get("state") == "running":
            if pid_alive(status.get("pid") or attempt.get("pid")):
                raise RuntimeError(f"Prior worker for {job['id']} is still alive; do not launch a duplicate.")


def run_jobs(manifest, directory, *, launch=subprocess.Popen):
    """Serial stages, preserved attempts, and parent detection of native crashes."""
    directory = Path(directory)
    reject_live_attempts(manifest["jobs"])
    for job in manifest["jobs"]:
        if completed_job(job):
            print(f"[SKIP] {job['id']}: verified saved result", flush=True)
            continue
        check_sources(manifest["source_hashes"])
        attempt_dir = directory / job["id"] / f"a{len(job['attempts']) + 1}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt = {"directory": str(attempt_dir.resolve()), "started_at": utc_now(), "state": "reserved"}
        job["attempts"].append(attempt)
        job["state"] = "running"
        payload = {**job, "options": manifest["options"], "source_hashes": manifest["source_hashes"]}
        write_json(attempt_dir / "job.json", payload)
        write_json(directory / "manifest.json", manifest)
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(manifest["options"]["threads"])
        env.update(PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
        command = [manifest["python"], str(ROOT / "run_e3_prediction_quality.py"), "--worker-job", str(attempt_dir / "job.json")]
        print(f"[RUN] {job['id']} -> {attempt_dir}", flush=True)
        process = None
        try:
            with (attempt_dir / "worker.log").open("x", encoding="utf-8") as log:
                kwargs = {"cwd": ROOT, "stdout": log, "stderr": subprocess.STDOUT, "env": env}
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                process = launch(command, **kwargs)
                attempt.update(state="running", pid=process.pid)
                write_json(directory / "manifest.json", manifest)
                returncode = process.wait()
        except BaseException:
            if process is not None and process.poll() is None:
                # A file request works for hidden Windows processes as well as
                # POSIX workers; the next passive hook raises through cleanup.
                (attempt_dir / "stop_requested").write_text("Graceful stop requested by parent.\n")
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    attempt["cleanup_unconfirmed"] = job["plant"] == "distillation"
            job["state"], attempt["state"] = "interrupted", "interrupted"
            write_json(directory / "manifest.json", manifest)
            raise
        attempt.update(returncode=returncode, finished_at=utc_now())
        job["state"] = "completed" if returncode == 0 else "failed"
        if not completed_job(job):
            job["state"], attempt["state"] = "failed", "failed"
            attempt["error"] = "Worker exit/artifact/cleanup validation failed. Inspect status.json and worker.log."
            manifest["state"] = "failed"
            write_json(directory / "manifest.json", manifest)
            raise RuntimeError(f"E3 stage failed: {job['id']} (exit {returncode}); preserved {attempt_dir}")
        attempt["state"] = "completed"
        write_json(directory / "manifest.json", manifest)
        print(f"[SAVED] {job['id']}", flush=True)
    manifest["state"] = "runs_completed"
    write_json(directory / "manifest.json", manifest)


def _analyse_study(directory):
    from review_experiments.e3.analysis import summarize, write_csv
    from review_experiments.e3.nominal import evaluate_nominal
    import numpy as np
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if not all(completed_job(job) for job in manifest["jobs"]):
        raise ValueError("Complete/verify all study stages before producing the E3 analysis.")
    options = manifest["options"]
    target = directory / "analysis" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    target.mkdir(parents=True)
    inputs, tracking = [], []
    nominal = {job["plant"]: Path(job["attempts"][-1]["directory"]) / "nominal.npz"
               for job in manifest["jobs"] if job["category"] == "nominal"}
    for job in manifest["jobs"]:
        if job["category"] == "nominal":
            continue
        saved = Path(job["attempts"][-1]["directory"])
        probe = target / "nominal_checkpoints" / job["id"]
        probe.parent.mkdir(exist_ok=True)
        evaluate_nominal(saved / "capture", nominal[job["plant"]], probe, options["horizons"],
                         samples=2 if options["smoke"] else 50)
        inputs.extend([saved / "capture", probe])
        with (saved / "training.pkl").open("rb") as stream:
            bundle = pickle.load(stream)
        rewards = np.asarray(bundle["avg_rewards"], float)
        tracking.append({"plant": job["plant"], "category": job["category"], "seed": job["seed"],
                         "smoke": options["smoke"], "episodes": len(rewards),
                         "final_episode_reward": float(rewards[-1]), "source": str(saved / "training.pkl"),
                         "scope": "separate tracking diagnostic; not prediction accuracy"})
    summarize(inputs, target / "prediction", tuple(options["horizons"]))
    write_csv(target / "tracking_reward_separate.csv", tracking)
    manifest.update(state="completed", latest_analysis=str(target), finished_at=utc_now())
    write_json(directory / "manifest.json", manifest)
    print(f"E3 analysis saved: {target}", flush=True)
    return target


def analyse_study(directory):
    try:
        return _analyse_study(directory)
    except BaseException as error:
        path = Path(directory).resolve() / "manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            manifest.update(state="analysis_failed", analysis_error=str(error), analysis_failed_at=utc_now())
            write_json(path, manifest)
        raise


def run_analysis(directory):
    """Run analysis with the saved scientific interpreter, even for a stdlib parent."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(manifest["options"]["threads"])
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    completed = subprocess.run([manifest["python"], str(ROOT / "run_e3_prediction_quality.py"),
                                "--analysis-worker", str(directory)], cwd=ROOT, env=env,
                               capture_output=True, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    log = directory / ("analysis_execution_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6] + ".log")
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"E3 analysis failed; training stages are preserved. Inspect {log} and use --analyse.")
    updated = json.loads((directory / "manifest.json").read_text())
    if updated["state"] != "completed" or not (Path(updated["latest_analysis"]) / "prediction" / "prediction_errors.csv").is_file():
        raise RuntimeError(f"Analysis exited without a verified report. Inspect {log}.")
    print(completed.stdout.strip(), flush=True)
    return Path(updated["latest_analysis"])


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--plants", nargs="+", choices=ENTRIES)
    value.add_argument("--categories", nargs="+", choices=("markov", "combined"))
    value.add_argument("--seed", type=int)
    value.add_argument("--prediction-horizons", nargs="+", type=int)
    value.add_argument("--samples-per-window", type=int)
    value.add_argument("--threads", type=int)
    value.add_argument("--python", dest="python_executable")
    value.add_argument("--aspen-root", type=Path)
    value.add_argument("--aspen-number", type=int,
                       help="Use C2S_SS_simulationN.dynf for every sequential distillation stage.")
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--preflight", action="store_true", help="Check interpreter and Aspen paths without opening Aspen or writing outputs.")
    value.add_argument("--smoke", action="store_true", help="Polymer-only compressed-clock technical run; no scientific evidence.")
    value.add_argument("--resume", type=Path)
    value.add_argument("--analyse", type=Path)
    value.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    value.add_argument("--analysis-worker", type=Path, help=argparse.SUPPRESS)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_job:
        # Ctrl-C remains a Python exception when a caller has an interactive
        # console; hidden workers also support the stop-request file.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        return run_worker(args.worker_job)
    if args.analysis_worker:
        analyse_study(args.analysis_worker)
        return 0
    if args.analyse:
        from run_distillation_phase2_repeats import aspen_file_lock
        with aspen_file_lock(args.analyse.resolve() / "study", args.analyse.resolve() / "_locks"):
            run_analysis(args.analyse)
        return 0
    if args.resume:
        if any(item is not None for item in (args.plants, args.categories, args.seed, args.prediction_horizons,
                                             args.samples_per_window, args.threads, args.python_executable,
                                             args.aspen_root, args.aspen_number, args.output_dir)) or args.smoke:
            raise ValueError("Resume uses its saved options and interpreter; do not change the study settings.")
        directory = args.resume.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        options, jobs = manifest["options"], manifest["jobs"]
        python = manifest["python"]
    else:
        options = {"plants": args.plants or (["polymer"] if args.smoke else list(ENTRIES)),
                   "categories": args.categories or ["markov", "combined"],
                   "seed": 7 if args.seed is None else args.seed,
                   "horizons": args.prediction_horizons or [1, 4, 8],
                   "samples_per_window": args.samples_per_window if args.samples_per_window is not None else (1 if args.smoke else 100),
                   "threads": 1 if args.threads is None else args.threads, "smoke": args.smoke,
                   "aspen_number": args.aspen_number,
                   "aspen_root": str(args.aspen_root.expanduser().resolve()) if args.aspen_root else None}
        if len(set(options["plants"])) != len(options["plants"]) or len(set(options["categories"])) != len(options["categories"]):
            raise ValueError("Plants and categories cannot be repeated.")
        validate_options(options)
        jobs = build_jobs(options["plants"], options["categories"], options["seed"])
        if args.python_executable:
            candidate = Path(args.python_executable).expanduser()
            python = str(candidate.resolve()) if candidate.is_file() else shutil.which(args.python_executable)
            if not python:
                raise FileNotFoundError(f"Python interpreter not found: {args.python_executable}")
        else:
            python = sys.executable
    print(f"E3: {sum(j['category'] != 'nominal' for j in jobs)} SG training runs; "
          f"{sum(j['category'] == 'nominal' for j in jobs)} shared nominal probes; "
          f"seed {options['seed']}; horizons {options['horizons']}; sequential workers.")
    for job in jobs:
        print(f"  {job['id']}: {ENTRIES[job['plant']][job['category']]}")
    if options.get("aspen_number") is not None:
        print(f"  All distillation stages use C2S_SS_simulation{options['aspen_number']}.dynf sequentially.")
    if args.dry_run:
        print("Dry run: no imports of the scientific stack, outputs, Aspen sessions, or training.")
        return 0
    if args.preflight:
        planned = preflight(python, jobs, options.get("aspen_root"), options.get("aspen_number"))
        print(json.dumps(planned, indent=2))
        return 0
    if args.resume:
        check_sources(manifest["source_hashes"])
    else:
        jobs = preflight(python, jobs, options.get("aspen_root"), options.get("aspen_number"))
        directory = (args.output_dir or ROOT / "Results" / "e3" /
                     (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": 1, "state": "pending", "created_at": utc_now(), "python": python,
                    "options": options, "jobs": jobs, "source_hashes": source_hashes(),
                    "git": git_metadata(),
                    "scope": "single-seed prediction diagnostics; no robustness or causal channel attribution"}
        write_json(directory / "manifest.json", manifest)
    print(f"E3 study: {directory}", flush=True)
    from run_distillation_phase2_repeats import aspen_file_lock
    with aspen_file_lock(directory / "study", directory / "_locks"):
        run_jobs(manifest, directory)
        run_analysis(directory)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"E3 error: {error}", file=sys.stderr)
        raise SystemExit(1)
