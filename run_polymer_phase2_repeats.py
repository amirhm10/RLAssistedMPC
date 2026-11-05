"""Repeat the existing polymer Scenario 2 runners without editing their sources.

Five seeded processes run together within one category/mode. The next group
starts only after every process in the current group has saved its results.
The parent uses only the standard library; workers use the existing scientific
Python environment. Use --dry-run to inspect the plan without starting training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time
import traceback
from typing import Callable
import uuid


ROOT = Path(__file__).resolve().parent
PROFILE = "robustness_200_100"
DEFAULT_SEEDS = (7, 11, 23, 37, 53)
RUNNERS = {
    "horizon": ("horizon_standard", "RL_assisted_MPC_horizons_unified.py"),
    "weights": ("weights", "RL_assisted_MPC_weights_unified.py"),
    "markov": ("markov", "RL_assisted_MPC_markov_unified.py"),
    "residual": ("residual", "RL_assisted_MPC_residual_unified.py"),
    "combined": ("combined", "RL_assisted_MPC_combined_unified.py"),
}
AGENT_CONFIG_KEYS = (
    "agent", "td3_agent", "horizon_agent", "markov_td3_agent",
    "weights_td3_agent", "residual_td3_agent",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes(root: Path) -> dict[str, str]:
    paths = {root / name for _, name in RUNNERS.values()}
    paths.add(root / Path(__file__).name)
    for directory in ("Simulation", "DQN", "TD3Agent", "utils", "systems"):
        paths.update((root / directory).rglob("*.py"))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if path.is_file()
    }


def build_jobs(categories, modes, seeds) -> list[dict]:
    return [
        {"category": category, "mode": mode, "seed": seed,
         "state": "pending", "attempts": []}
        for category in categories for mode in modes for seed in seeds
    ]


def configure_defaults(defaults: dict, job: dict) -> dict:
    """Change experiment selection, training seeds, and output routing only."""
    configured = deepcopy(defaults)
    if configured.get("n_tests_override") not in (None, 300):
        raise ValueError("Scenario 2 requires 300 episodes; existing n_tests_override differs.")
    configured["run_mode"] = "disturb"
    configured["training_profile_override"] = PROFILE
    configured["results_dir_override"] = job["results_directory"]
    # Short names keep nested Windows output paths manageable. Category, mode,
    # seed, attempt, and original entrypoint are recorded in the parent path/JSON.
    configured["result_prefix_override"] = "rl"
    configured["compare_prefix_override"] = "comparison"
    if job["category"] == "combined":
        configured["combined_agent_mode"] = job["mode"]
    elif job["category"] == "horizon":
        configured["agent_kind"] = "sg_dqn" if job["mode"] == "sg" else "dqn"
    else:
        configured["agent_kind"] = "sg_td3" if job["mode"] == "sg" else "td3"
    for key in AGENT_CONFIG_KEYS:
        if isinstance(configured.get(key), dict):
            configured[key]["seed"] = job["seed"]
    return configured


def dependency_check(python: str) -> None:
    code = (
        "import importlib.util,sys; "
        "names=('numpy','scipy','torch','matplotlib'); "
        "missing=[n for n in names if importlib.util.find_spec(n) is None]; "
        "print('Python:',sys.executable); print('Missing packages:',', '.join(missing) or 'none'); "
        "sys.exit(bool(missing))"
    )
    checked = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if checked.stdout:
        print(checked.stdout.strip(), flush=True)
    if checked.returncode:
        raise RuntimeError(
            "Use the same Python environment as your working polymer runners, "
            "or pass --python PATH_TO_THAT_PYTHON. No packages were installed.\n"
            + checked.stderr.strip()
        )


def run_worker(job_file: Path) -> int:
    job = json.loads(job_file.read_text(encoding="utf-8"))
    attempt_dir = job_file.parent
    status = {"state": "running", "started_at": utc_now(), "pid": os.getpid(),
              "category": job["category"], "mode": job["mode"], "seed": job["seed"]}
    write_json(attempt_dir / "worker_status.json", status)
    try:
        root = Path(job["repo_root"]).resolve()
        sys.path.insert(0, str(root))
        os.chdir(root)
        import numpy as np
        import torch
        import systems.polymer as polymer

        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job["seed"])
        torch.set_num_threads(job["threads_per_run"])

        family, entrypoint = RUNNERS[job["category"]]
        original_provider = polymer.get_polymer_notebook_defaults

        def worker_defaults(requested_family):
            defaults = original_provider(requested_family)
            if str(requested_family).strip().lower() == family:
                return configure_defaults(defaults, job)
            return defaults

        # A worker-local adapter also covers the combined entrypoint, which
        # does not expose NB_CONFIGURE. No source or shared-default file is edited.
        polymer.get_polymer_notebook_defaults = worker_defaults
        try:
            namespace = runpy.run_path(str(root / entrypoint), run_name="__main__")
        finally:
            polymer.get_polymer_notebook_defaults = original_provider

        actual_profile = namespace.get("TRAINING_PROFILE_NAME")
        episodes = len(namespace["result_bundle"]["avg_rewards"])
        if actual_profile != PROFILE or episodes != 300:
            raise RuntimeError(f"Unexpected study: profile={actual_profile!r}, episodes={episodes}.")
        result_dir = Path(namespace["out_dir_rl"]).resolve()
        result_file = result_dir / "input_data.pkl"
        expected_root = Path(job["results_directory"]).resolve()
        if not result_dir.is_relative_to(expected_root) or not result_file.is_file():
            raise RuntimeError("Runner did not save input_data.pkl inside its isolated output directory.")
        metadata = {
            "schema_version": 1, "category": job["category"], "mode": job["mode"],
            "seed": job["seed"], "profile": actual_profile, "episodes": episodes,
            "entrypoint": entrypoint, "entrypoint_sha256": job["entrypoint_sha256"],
            "source_commit": job.get("source_commit"), "python": sys.executable,
            "threads_per_run": job["threads_per_run"], "batch_directory": job["batch_directory"],
            "worker_job": str(job_file), "created_at": utc_now(),
            "runtime_changes": ["SG/plain selection", "Scenario 2 profile selection",
                                "training random seeds", "output directory and prefixes"],
        }
        write_json(result_dir / "replication_metadata.json", metadata)
        status.update(state="completed", finished_at=utc_now(), episodes=episodes,
                      result_file=str(result_file), comparison_directory=str(namespace.get("out_dir_cmp")))
        write_json(attempt_dir / "worker_status.json", status)
        return 0
    except Exception as error:
        status.update(state="failed", finished_at=utc_now(), error=str(error))
        write_json(attempt_dir / "worker_status.json", status)
        traceback.print_exc()
        return 1


def worker_command(python: str, job_file: Path) -> list[str]:
    return [python, "-u", str(Path(__file__).resolve()), "--worker-job", str(job_file)]


def completed_result(job: dict) -> bool:
    result = job.get("result_file")
    return job["state"] == "completed" and bool(result) and Path(result).is_file()


def saved_status(attempt_dir: Path, returncode: int | None) -> dict:
    status_file = attempt_dir / "worker_status.json"
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result = status.get("result_file")
    if returncode == 0 and status.get("state") == "completed" and result and Path(result).is_file():
        return status
    return {}


def run_batches(manifest: dict, batch_dir: Path,
                command_factory: Callable[[str, Path], list[str]] = worker_command) -> int:
    """Launch each whole seed group before waiting; never overlap groups."""
    manifest_path = batch_dir / "manifest.json"
    groups = list(dict.fromkeys((job["category"], job["mode"]) for job in manifest["jobs"]))
    manifest.update(state="running", updated_at=utc_now())
    write_json(manifest_path, manifest)
    for category, mode in groups:
        jobs = [job for job in manifest["jobs"] if (job["category"], job["mode"]) == (category, mode)]
        pending = [job for job in jobs if not completed_result(job)]
        if not pending:
            print(f"[SKIP] {category}/{mode}: all results already saved.", flush=True)
            continue
        print(f"[START] {category}/{mode}: {len(pending)} parallel repetitions.", flush=True)
        running = []
        try:
            for job in pending:
                attempt_number = len(job["attempts"]) + 1
                attempt_dir = batch_dir / f"{category}_{mode}" / f"s{job['seed']}" / f"a{attempt_number}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                payload = {
                    **{key: job[key] for key in ("category", "mode", "seed")},
                    "repo_root": manifest["repo_root"], "batch_directory": str(batch_dir),
                    "results_directory": str(attempt_dir),
                    "threads_per_run": manifest["threads_per_run"],
                    "source_commit": manifest.get("source_commit"),
                    "entrypoint_sha256": manifest["source_hashes"][RUNNERS[category][1]],
                }
                job_file = attempt_dir / "job.json"
                write_json(job_file, payload)
                attempt = {"directory": str(attempt_dir), "pid": None}
                job["attempts"].append(attempt)
                # Record the reserved directory even if creating its process
                # fails, so resume uses a new attempt instead of overwriting it.
                write_json(manifest_path, manifest)
                environment = dict(os.environ)
                environment.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                                   PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(job["seed"]))
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[key] = str(manifest["threads_per_run"])
                log = (attempt_dir / "run.log").open("wb")
                try:
                    process = subprocess.Popen(command_factory(manifest["python"], job_file),
                                               cwd=manifest["repo_root"], env=environment,
                                               stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                attempt["pid"] = process.pid
                job.update(state="running", started_at=utc_now())
                running.append((job, process, log, attempt_dir))
            write_json(manifest_path, manifest)
            # The launch loop above finishes before any worker is awaited.
            while any(process.poll() is None for _, process, _, _ in running):
                time.sleep(0.25)
            for job, process, log, attempt_dir in running:
                log.close()
                status = saved_status(attempt_dir, process.returncode)
                job.update(exit_code=process.returncode, finished_at=utc_now())
                result = status.get("result_file")
                if status:
                    job.update(state="completed", result_file=result)
                    print(f"[SAVED] {category}/{mode} seed={job['seed']}: {result}", flush=True)
                else:
                    job.update(state="failed", error="Worker exited without a saved result; see run.log/worker_status.json.")
                    print(f"[FAILED] {category}/{mode} seed={job['seed']}; see {attempt_dir / 'run.log'}", flush=True)
        except BaseException:
            for job, process, log, attempt_dir in running:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                log.close()
                status = saved_status(attempt_dir, process.returncode)
                if status:
                    job.update(state="completed", result_file=status["result_file"], finished_at=utc_now())
                else:
                    job.update(state="interrupted", finished_at=utc_now())
            manifest.update(state="interrupted", updated_at=utc_now())
            write_json(manifest_path, manifest)
            raise
        manifest["updated_at"] = utc_now()
        if any(job["state"] != "completed" for job in jobs):
            manifest["state"] = "failed"
            write_json(manifest_path, manifest)
            print("Stopped before the next group. Saved results are preserved; use --resume.", flush=True)
            return 1
        write_json(manifest_path, manifest)
    manifest.update(state="completed", updated_at=utc_now())
    write_json(manifest_path, manifest)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=tuple(RUNNERS))
    parser.add_argument("--modes", nargs="+", choices=("sg", "plain"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--python", dest="python_executable", help="Python from your working scientific environment.")
    parser.add_argument("--threads-per-run", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_job:
        return run_worker(args.worker_job)
    if args.resume:
        if any(value is not None for value in (args.categories, args.modes, args.seeds,
                                               args.output_dir, args.threads_per_run)):
            parser.error("--resume uses the saved categories, modes, seeds, and settings.")
        batch_dir = args.resume.expanduser().resolve()
        manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("state") == "running":
            parser.error("This batch is marked running. Stop its current launcher before resuming.")
        if Path(manifest["repo_root"]).resolve() != ROOT:
            parser.error("The saved batch belongs to a different repository.")
        if manifest["source_hashes"] != source_hashes(ROOT):
            parser.error("Sources changed since this batch. Start a new batch to avoid mixing versions.")
        if args.python_executable:
            manifest["python"] = args.python_executable
    else:
        categories = args.categories or list(RUNNERS)
        modes = args.modes or ["sg", "plain"]
        seeds = args.seeds or list(DEFAULT_SEEDS)
        for label, values in (("categories", categories), ("modes", modes), ("seeds", seeds)):
            if len(set(values)) != len(values):
                parser.error(f"Duplicate {label} are not allowed.")
        if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
            parser.error("Seeds must be in NumPy's unsigned 32-bit seed range.")
        threads = 1 if args.threads_per_run is None else args.threads_per_run
        if threads < 1:
            parser.error("--threads-per-run must be positive.")
        output = (args.output_dir or ROOT / "Polymer" / "Results" / "phase2_repeats").expanduser().resolve()
        batch_dir = output / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        manifest = {
            "schema_version": 1, "profile": PROFILE, "episodes_per_run": 300,
            "repo_root": str(ROOT), "python": args.python_executable or sys.executable,
            "threads_per_run": threads, "created_at": utc_now(), "state": "pending",
            "source_hashes": source_hashes(ROOT), "jobs": build_jobs(categories, modes, seeds),
        }
        try:
            manifest["source_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["source_commit"] = None
    print(f"Profile: {PROFILE} (200 + 100 episodes per repetition)", flush=True)
    print(f"Python: {manifest['python']}\nBatch directory: {batch_dir}", flush=True)
    for category, mode in dict.fromkeys((job["category"], job["mode"]) for job in manifest["jobs"]):
        seeds = [job["seed"] for job in manifest["jobs"] if (job["category"], job["mode"]) == (category, mode)]
        print(f"  {category}/{mode}: parallel seeds {seeds}", flush=True)
    print(f"Total repetitions: {len(manifest['jobs'])}", flush=True)
    if args.dry_run:
        return 0
    dependency_check(manifest["python"])
    if not args.resume:
        batch_dir.mkdir(parents=True, exist_ok=False)
        write_json(batch_dir / "manifest.json", manifest)
    return run_batches(manifest, batch_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; saved outputs remain available for --resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Batch launcher error: {error}", file=sys.stderr)
        raise SystemExit(1)
