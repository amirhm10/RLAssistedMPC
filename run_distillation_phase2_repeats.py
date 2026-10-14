"""Repeat the existing distillation Scenario 2 runners without changing sources.

Five SG and five plain seeded processes run together within each channel.
Channels run sequentially: horizon, weights, markov, residual, combined.
Each channel uses Aspen simulations 1 through 10 exactly once concurrently.
Use --dry-run to inspect the plan without importing Aspen or starting training.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import signal
import subprocess
import sys
import time
import traceback
from typing import Callable
import uuid


ROOT = Path(__file__).resolve().parent
PROFILE = "temperature_flip_200_100"
DEFAULT_SEEDS = (7, 11, 23, 37, 53)
MAX_ASPEN_FILES = 10
RUNNERS = {
    "horizon": ("horizon_standard", "distillation_RL_assisted_MPC_horizons_unified.py"),
    "weights": ("weights", "distillation_RL_assisted_MPC_weights_unified.py"),
    "markov": ("markov", "distillation_RL_assisted_MPC_markov_unified.py"),
    "residual": ("residual", "distillation_RL_assisted_MPC_residual_unified.py"),
    "combined": ("combined", "distillation_RL_assisted_MPC_combined_unified.py"),
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
    jobs = []
    for category in categories:
        for number, (mode, seed) in enumerate(
            ((mode, seed) for mode in modes for seed in seeds), start=1
        ):
            jobs.append({"category": category, "mode": mode, "seed": seed,
                         "aspen_number": number, "state": "pending", "attempts": []})
    validate_jobs(jobs)
    return jobs


def validate_jobs(jobs: list[dict], require_paths: bool = False) -> None:
    """Reject collisions before launching anything, including on resume."""
    if not jobs:
        raise ValueError("The study must contain at least one repetition.")
    identities = set()
    numbers = {}
    paths = {}
    snapshots = {}
    for job in jobs:
        category, mode, seed = (job[key] for key in ("category", "mode", "seed"))
        if category not in RUNNERS or mode not in ("sg", "plain"):
            raise ValueError("Unknown channel or SG/plain mode in the study.")
        if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seeds must be in NumPy's unsigned 32-bit seed range.")
        identity = (category, mode, seed)
        if identity in identities:
            raise ValueError(f"Duplicate repetition: {identity}.")
        identities.add(identity)
        number = job["aspen_number"]
        if not isinstance(number, int) or not 1 <= number <= MAX_ASPEN_FILES:
            raise ValueError("Each channel supports at most ten runs using Aspen files 1 through 10.")
        used = numbers.setdefault(category, set())
        if number in used:
            raise ValueError(f"Aspen simulation {number} was assigned twice in {category}.")
        used.add(number)
        if require_paths:
            dyn_path = Path(job["aspen_path"]).resolve()
            snaps_path = Path(job["snaps_path"]).resolve()
            if dyn_path.name != f"C2S_SS_simulation{number}.dynf":
                raise ValueError(f"Aspen path does not match assigned simulation {number}: {dyn_path}.")
            for target, seen in ((dyn_path, paths), (snaps_path, snapshots)):
                key = os.path.normcase(str(target))
                category_paths = seen.setdefault(category, set())
                if key in category_paths:
                    raise ValueError(f"Concurrent workers would share an Aspen model/snapshot path: {target}.")
                category_paths.add(key)


def seeded_defaults(defaults: dict, seed: int) -> dict:
    configured = deepcopy(defaults)
    for key in AGENT_CONFIG_KEYS:
        if isinstance(configured.get(key), dict):
            configured[key]["seed"] = seed
    return configured


def configure_defaults(defaults: dict, job: dict) -> dict:
    """Select the existing study/mode, seeds, outputs, and exclusive Aspen slot."""
    configured = seeded_defaults(defaults, job["seed"])
    if configured.get("n_tests_override") not in (None, 300):
        raise ValueError("Scenario 2 requires 300 episodes; existing n_tests_override differs.")
    configured["run_mode"] = "disturb"
    configured["disturbance_profile"] = "fluctuation"
    configured["training_profile_override"] = PROFILE
    configured["results_dir_override"] = job["results_directory"]
    configured["result_prefix_override"] = "rl"
    configured["compare_prefix_override"] = "comparison"
    if job["category"] == "combined":
        configured["combined_agent_mode"] = job["mode"]
    else:
        # The distillation entrypoints derive agent_kind from agent_mode.
        configured["agent_mode"] = job["mode"]
    configured["aspen_preset"] = job["aspen_number"]
    # Explicit paths take precedence over family defaults, including Markov's
    # usual simulation11 and any existing shared manual/snapshot overrides.
    configured["aspen_path_override"] = job["aspen_path"]
    configured["snaps_path_override"] = job["snaps_path"]
    return configured


def resolve_job_paths(jobs: list[dict], aspen_root: str | None = None) -> list[dict]:
    """Read existing Aspen roots with the worker's scientific interpreter."""
    from systems.distillation import get_distillation_notebook_defaults
    from systems.distillation.config import DEFAULT_ASPEN_ROOT, resolve_snapshot_dir

    planned = deepcopy(jobs)
    for job in planned:
        family, _ = RUNNERS[job["category"]]
        defaults = get_distillation_notebook_defaults(family)
        if defaults.get("n_tests_override") not in (None, 300):
            raise ValueError(f"{family}: Scenario 2 requires 300 episodes.")
        if aspen_root:
            root = Path(aspen_root)
        elif defaults.get("aspen_path_override"):
            root = Path(defaults["aspen_path_override"]).expanduser().parent
        else:
            root = Path(defaults.get("aspen_root_override") or DEFAULT_ASPEN_ROOT)
        dyn_path = (root.expanduser().resolve() / f"C2S_SS_simulation{job['aspen_number']}.dynf").resolve()
        job.update(aspen_path=str(dyn_path), snaps_path=str(resolve_snapshot_dir(dyn_path).resolve()))
    validate_jobs(planned, require_paths=True)
    return planned


def dependency_check(python: str) -> None:
    code = (
        "import importlib.util,os,sys; "
        "names=('numpy','scipy','torch','matplotlib','win32com','pythoncom'); "
        "missing=[n for n in names if importlib.util.find_spec(n) is None]; "
        "print('Python:',sys.executable); print('Missing packages:',', '.join(missing) or 'none'); "
        "print('Platform:',os.name); sys.exit(bool(missing) or os.name!='nt')"
    )
    checked = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if checked.stdout:
        print(checked.stdout.strip(), flush=True)
    if checked.returncode:
        raise RuntimeError(
            "Use Windows and the Python environment from your working distillation runners "
            "(including pywin32), or pass --python PATH_TO_THAT_PYTHON.\n" + checked.stderr.strip()
        )


def aspen_preflight(python: str, jobs: list[dict], aspen_root: Path | None) -> list[dict]:
    code = (
        "import json,sys; import run_distillation_phase2_repeats as launcher; "
        "request=json.load(sys.stdin); "
        "print(json.dumps(launcher.resolve_job_paths(request['jobs'],request['aspen_root'])))"
    )
    checked = subprocess.run(
        [python, "-c", code], cwd=ROOT, capture_output=True, text=True,
        input=json.dumps({"jobs": jobs, "aspen_root": str(aspen_root) if aspen_root else None}),
    )
    if checked.returncode:
        raise RuntimeError("Could not resolve the existing distillation Aspen settings.\n" + checked.stderr.strip())
    planned = json.loads(checked.stdout)
    check_aspen_files(planned)
    return planned


def check_aspen_files(jobs: list[dict]) -> None:
    validate_jobs(jobs, require_paths=True)
    missing = sorted({job["aspen_path"] for job in jobs if not Path(job["aspen_path"]).is_file()})
    if missing:
        raise FileNotFoundError("Required Aspen model files are missing:\n" + "\n".join(missing))


@contextmanager
def aspen_file_lock(dyn_path: Path, lock_directory: Path):
    """An OS file lock prevents other wrapper workers opening the same model.

    Locks are held until cleanup finishes and released by the OS on process exit.
    Keep the lock files: removing them could bypass another process's live lock.
    """
    lock_directory.mkdir(parents=True, exist_ok=True)
    identity = os.path.normcase(str(dyn_path.resolve()))
    lock_path = lock_directory / (hashlib.sha256(identity.encode()).hexdigest() + ".lock")
    with lock_path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Aspen model is reserved by another wrapper worker: {dyn_path}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def worker_adapters(distillation, plant, job: dict):
    """Apply worker-local defaults and ensure this worker closes its Aspen app."""
    family, _ = RUNNERS[job["category"]]
    original_provider = distillation.get_distillation_notebook_defaults
    original_column = plant.DistillationColumnAspen
    original_export = distillation.DistillationColumnAspen
    columns = []

    def worker_defaults(requested_family):
        defaults = original_provider(requested_family)
        if str(requested_family).strip().lower() == family:
            return configure_defaults(defaults, job)
        # Combined validates its agent settings against standalone defaults.
        # Seed those copies too, retaining all original controller/agent values.
        return seeded_defaults(defaults, job["seed"])

    class WorkerColumn(original_column):
        def __init__(self, *args, **kwargs):
            self._replication_closed = False
            columns.append(self)
            super().__init__(*args, **kwargs)

        def close(self, snaps_path=None, prefix="snp"):
            if not self._replication_closed:
                super().close(snaps_path or job["snaps_path"], prefix=prefix)
                self._replication_closed = True

    distillation.get_distillation_notebook_defaults = worker_defaults
    distillation.DistillationColumnAspen = plant.DistillationColumnAspen = WorkerColumn
    try:
        yield
    finally:
        try:
            errors = []
            for column in columns:
                if getattr(column, "ad", None) is None:
                    continue
                try:
                    column.close(job["snaps_path"])
                except Exception as error:
                    errors.append(error)
            if errors:
                raise RuntimeError("Aspen cleanup failed; close this worker's document before resuming.") from errors[0]
        finally:
            distillation.get_distillation_notebook_defaults = original_provider
            plant.DistillationColumnAspen = original_column
            distillation.DistillationColumnAspen = original_export


def interrupt_worker(_signal, _frame):
    raise KeyboardInterrupt


def run_worker(job_file: Path) -> int:
    job = json.loads(job_file.read_text(encoding="utf-8"))
    attempt_dir = job_file.parent
    status = {"state": "running", "started_at": utc_now(), "pid": os.getpid(),
              **{key: job[key] for key in ("category", "mode", "seed", "aspen_number", "aspen_path", "snaps_path")}}
    write_json(attempt_dir / "worker_status.json", status)
    try:
        validate_jobs([job], require_paths=True)
        root = Path(job["repo_root"]).resolve()
        sys.path.insert(0, str(root))
        os.chdir(root)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, interrupt_worker)
        import numpy as np
        import torch
        import systems.distillation as distillation
        import systems.distillation.plant as plant

        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job["seed"])
        torch.set_num_threads(job["threads_per_run"])

        _, entrypoint = RUNNERS[job["category"]]
        with aspen_file_lock(Path(job["aspen_path"]), root / "Distillation" / "Results" / "_aspen_file_locks"):
            with worker_adapters(distillation, plant, job):
                namespace = runpy.run_path(str(root / entrypoint), run_name="__main__")
            # Cleanup must succeed before this worker reports a saved result.
            status["aspen_closed"] = True

        actual_profile = namespace.get("TRAINING_PROFILE_NAME")
        actual_mode = namespace.get("COMBINED_AGENT_MODE" if job["category"] == "combined" else "AGENT_MODE")
        episodes = len(namespace["result_bundle"]["avg_rewards"])
        if actual_profile != PROFILE or episodes != 300 or actual_mode != job["mode"]:
            raise RuntimeError(f"Unexpected study: profile={actual_profile!r}, episodes={episodes}, mode={actual_mode!r}.")
        if Path(namespace["DYN_PATH"]).resolve() != Path(job["aspen_path"]).resolve():
            raise RuntimeError("Runner did not use its assigned Aspen file.")
        result_dir = Path(namespace["out_dir_rl"]).resolve()
        result_file = result_dir / "input_data.pkl"
        expected_root = Path(job["results_directory"]).resolve()
        if not result_dir.is_relative_to(expected_root) or not result_file.is_file():
            raise RuntimeError("Runner did not save input_data.pkl inside its isolated output directory.")
        metadata = {
            "schema_version": 1, "category": job["category"], "mode": job["mode"],
            "seed": job["seed"], "profile": actual_profile, "episodes": episodes,
            "aspen_number": job["aspen_number"], "aspen_path": job["aspen_path"], "snaps_path": job["snaps_path"],
            "entrypoint": entrypoint, "entrypoint_sha256": job["entrypoint_sha256"],
            "source_commit": job.get("source_commit"), "python": sys.executable,
            "threads_per_run": job["threads_per_run"], "batch_directory": job["batch_directory"],
            "worker_job": str(job_file), "created_at": utc_now(),
            "runtime_changes": ["SG/plain selection", "Scenario 2 profile selection",
                                "training random seeds", "output directory and prefixes",
                                "Aspen file and matching snapshot directory", "Aspen cleanup guard"],
        }
        write_json(result_dir / "replication_metadata.json", metadata)
        status.update(state="completed", finished_at=utc_now(), episodes=episodes,
                      result_file=str(result_file), comparison_directory=str(namespace.get("out_dir_cmp")))
        write_json(attempt_dir / "worker_status.json", status)
        return 0
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        status.update(state="interrupted" if interrupted else "failed", finished_at=utc_now(), error=str(error))
        write_json(attempt_dir / "worker_status.json", status)
        traceback.print_exc()
        return 130 if interrupted else 1


def worker_command(python: str, job_file: Path) -> list[str]:
    return [python, "-u", str(Path(__file__).resolve()), "--worker-job", str(job_file)]


def completed_result(job: dict) -> bool:
    result = job.get("result_file")
    return job["state"] == "completed" and bool(result) and Path(result).is_file()


def saved_status(attempt_dir: Path, returncode: int | None) -> dict:
    try:
        status = json.loads((attempt_dir / "worker_status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result = status.get("result_file")
    if (returncode == 0 and status.get("state") == "completed" and status.get("aspen_closed") is True
            and result and Path(result).is_file()):
        return status
    return {}


def stop_workers(running) -> None:
    for _, process, _, _ in running:
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            except OSError:
                process.terminate()
    deadline = time.monotonic() + 10
    while any(process.poll() is None for _, process, _, _ in running) and time.monotonic() < deadline:
        time.sleep(0.1)
    for _, process, _, _ in running:
        if process.poll() is None:
            process.kill()
        process.wait()


def run_batches(manifest: dict, batch_dir: Path,
                command_factory: Callable[[str, Path], list[str]] = worker_command) -> int:
    """Launch all selected modes/seeds together; never overlap channels."""
    validate_jobs(manifest["jobs"], require_paths=True)
    manifest_path = batch_dir / "manifest.json"
    groups = list(dict.fromkeys(job["category"] for job in manifest["jobs"]))
    manifest.update(state="running", updated_at=utc_now())
    write_json(manifest_path, manifest)
    for category in groups:
        jobs = [job for job in manifest["jobs"] if job["category"] == category]
        pending = [job for job in jobs if not completed_result(job)]
        if not pending:
            print(f"[SKIP] {category}: all results already saved.", flush=True)
            continue
        print(f"[START] {category}: {len(pending)} parallel repetitions across selected modes.", flush=True)
        running = []
        try:
            for job in pending:
                attempt_number = len(job["attempts"]) + 1
                attempt_dir = batch_dir / f"{category}_{job['mode']}" / f"s{job['seed']}" / f"a{attempt_number}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                payload = {
                    **{key: job[key] for key in ("category", "mode", "seed", "aspen_number", "aspen_path", "snaps_path")},
                    "repo_root": manifest["repo_root"], "batch_directory": str(batch_dir),
                    "results_directory": str(attempt_dir), "threads_per_run": manifest["threads_per_run"],
                    "source_commit": manifest.get("source_commit"),
                    "entrypoint_sha256": manifest["source_hashes"][RUNNERS[category][1]],
                }
                job_file = attempt_dir / "job.json"
                write_json(job_file, payload)
                attempt = {"directory": str(attempt_dir), "pid": None}
                job["attempts"].append(attempt)
                # Reserve attempts before spawning so resume never overwrites.
                write_json(manifest_path, manifest)
                environment = dict(os.environ)
                environment.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                                   PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(job["seed"]))
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[key] = str(manifest["threads_per_run"])
                log = (attempt_dir / "run.log").open("wb")
                try:
                    process = subprocess.Popen(
                        command_factory(manifest["python"], job_file), cwd=manifest["repo_root"],
                        env=environment, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                    )
                except BaseException:
                    log.close()
                    raise
                attempt["pid"] = process.pid
                job.update(state="running", started_at=utc_now())
                running.append((job, process, log, attempt_dir))
            write_json(manifest_path, manifest)
            while any(process.poll() is None for _, process, _, _ in running):
                time.sleep(0.25)
            for job, process, log, attempt_dir in running:
                log.close()
                status = saved_status(attempt_dir, process.returncode)
                job.update(exit_code=process.returncode, finished_at=utc_now())
                if status:
                    job.update(state="completed", result_file=status["result_file"])
                    print(f"[SAVED] {category}/{job['mode']} seed={job['seed']} Aspen={job['aspen_number']}: {job['result_file']}", flush=True)
                else:
                    job.update(state="failed", error="Worker exited without a saved result/confirmed Aspen cleanup; see run.log/worker_status.json.")
                    print(f"[FAILED] {category}/{job['mode']} seed={job['seed']}; see {attempt_dir / 'run.log'}", flush=True)
        except BaseException:
            stop_workers(running)
            for job, process, log, attempt_dir in running:
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
            print("Stopped before the next channel. Saved results are preserved; use --resume.", flush=True)
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
    parser.add_argument("--aspen-root", type=Path, help="Directory containing C2S_SS_simulation1.dynf through simulation10.dynf.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_job:
        return run_worker(args.worker_job)
    if args.resume:
        if any(value is not None for value in (args.categories, args.modes, args.seeds,
                                               args.output_dir, args.threads_per_run, args.aspen_root)):
            parser.error("--resume uses the saved channels, modes, seeds, and Aspen settings.")
        batch_dir = args.resume.expanduser().resolve()
        manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("state") == "running":
            parser.error("This batch is marked running. Stop its current launcher before resuming.")
        if Path(manifest["repo_root"]).resolve() != ROOT or manifest.get("profile") != PROFILE:
            parser.error("The saved batch belongs to a different repository or study.")
        if manifest["source_hashes"] != source_hashes(ROOT):
            parser.error("Sources changed since this batch. Start a new batch to avoid mixing versions.")
        validate_jobs(manifest["jobs"], require_paths=True)
        if args.python_executable:
            manifest["python"] = args.python_executable
    else:
        categories = args.categories or list(RUNNERS)
        modes = args.modes or ["sg", "plain"]
        seeds = args.seeds or list(DEFAULT_SEEDS)
        for label, values in (("categories", categories), ("modes", modes), ("seeds", seeds)):
            if len(set(values)) != len(values):
                parser.error(f"Duplicate {label} are not allowed.")
        try:
            jobs = build_jobs(categories, modes, seeds)
        except ValueError as error:
            parser.error(str(error))
        threads = 1 if args.threads_per_run is None else args.threads_per_run
        if threads < 1:
            parser.error("--threads-per-run must be positive.")
        output = (args.output_dir or ROOT / "Distillation" / "Results" / "phase2_repeats").expanduser().resolve()
        batch_dir = output / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        manifest = {
            "schema_version": 1, "profile": PROFILE, "episodes_per_run": 300,
            "repo_root": str(ROOT), "python": args.python_executable or sys.executable,
            "threads_per_run": threads, "created_at": utc_now(), "state": "pending",
            "source_hashes": source_hashes(ROOT), "jobs": jobs,
        }
        try:
            manifest["source_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["source_commit"] = None
    print(f"Profile: {PROFILE} (200 + 100 episodes per repetition)", flush=True)
    print(f"Python: {manifest['python']}\nBatch directory: {batch_dir}", flush=True)
    for category in dict.fromkeys(job["category"] for job in manifest["jobs"]):
        jobs = [job for job in manifest["jobs"] if job["category"] == category]
        print(f"  {category}: {len(jobs)} parallel repetitions", flush=True)
        for job in jobs:
            print(f"    {job['mode']} seed={job['seed']}: C2S_SS_simulation{job['aspen_number']}.dynf", flush=True)
    print(f"Total repetitions: {len(manifest['jobs'])}", flush=True)
    if args.dry_run:
        return 0
    dependency_check(manifest["python"])
    if args.resume:
        check_aspen_files(manifest["jobs"])
    else:
        manifest["jobs"] = aspen_preflight(manifest["python"], manifest["jobs"], args.aspen_root)
        batch_dir.mkdir(parents=True, exist_ok=False)
        write_json(batch_dir / "manifest.json", manifest)
    return run_batches(manifest, batch_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; saved outputs remain available for --resume. Close any Aspen documents left after a forced stop before resuming.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Batch launcher error: {error}", file=sys.stderr)
        raise SystemExit(1)
