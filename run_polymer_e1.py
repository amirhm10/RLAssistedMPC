"""One-seed E1 study using the unchanged publication training/controller sources.

CMD: python run_polymer_e1.py --dry-run
Technical validation: python run_polymer_e1.py --smoke
Full study: python run_polymer_e1.py
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parent
ENTRIES = {
    "horizon": "RL_assisted_MPC_horizons_unified.py",
    "weights": "RL_assisted_MPC_weights_unified.py",
    "markov": "RL_assisted_MPC_markov_unified.py",
    "residual": "RL_assisted_MPC_residual_unified.py",
    "combined": "RL_assisted_MPC_combined_unified.py",
}
SEED_KEYS = ("agent", "td3_agent", "horizon_agent", "markov_td3_agent", "weights_td3_agent", "residual_td3_agent")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes(root=None):
    """Fingerprint E1 dependencies, excluding unrelated reviewer experiments.

    Additions or edits inside E1 and the shared implementation remain guarded.
    A new E3 experiment must not invalidate an unchanged E1 training run.
    """
    root = Path(root) if root is not None else ROOT
    files = {root / name for name in ENTRIES.values()}
    files.update(root / name for name in ("run_polymer_e1.py", "review_experiments/__init__.py"))
    for directory in ("Simulation", "DQN", "TD3Agent", "utils", "systems", "review_experiments/e1"):
        files.update((root / directory).rglob("*.py"))
    files.update(root / "Polymer" / "Data" / name for name in ("system_dict", "scaling_factor.pickle", "min_max_states.pickle"))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()}


def configure_defaults(defaults, category, seed, output, smoke=False):
    configured = deepcopy(defaults)
    configured["run_mode"] = "disturb"
    configured["training_profile_override"] = "robustness_200_100"
    configured["results_dir_override"] = str(output)
    configured["result_prefix_override"] = "training"
    configured["compare_prefix_override"] = "comparison"
    if category == "combined":
        configured["combined_agent_mode"] = "sg"
    else:
        configured["agent_kind"] = "sg_dqn" if category == "horizon" else "sg_td3"
    for key in SEED_KEYS:
        if isinstance(configured.get(key), dict):
            configured[key]["seed"] = seed
    return configured


def execute_entrypoint(category, configure):
    """Execute the existing entrypoint through its original training call.

    Its configuration, plant/model setup, agents and runner call are reused.
    Only later plotting/comparison statements are omitted; E1 owns its outputs.
    """
    path = ROOT / ENTRIES[category]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    end = None
    for index, statement in enumerate(tree.body):
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "result_bundle" for target in statement.targets
        ):
            end = index + 1
            break
    if end is None:
        raise ValueError("The existing entrypoint no longer has a recognized training result assignment.")
    prefix = ast.Module(body=tree.body[:end], type_ignores=[])
    # This is the same process-local defaults interface used by the existing
    # repeat wrapper. Combined has no NB_CONFIGURE hook, so adapting only that
    # hook would accidentally leave it in plain mode.
    import systems.polymer as polymer
    original_provider = polymer.get_polymer_notebook_defaults
    family = "horizon_standard" if category == "horizon" else category

    def provider(requested_family):
        defaults = original_provider(requested_family)
        return configure(defaults) if requested_family == family else defaults

    namespace = {"__name__": "__e1_training__", "__file__": str(path)}
    polymer.get_polymer_notebook_defaults = provider
    try:
        exec(compile(prefix, str(path), "exec"), namespace, namespace)
    finally:
        polymer.get_polymer_notebook_defaults = original_provider
    return namespace


def worker(job_path):
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = job_path.parent
    status = {"state": "running", "started_at": utc_now(), "pid": os.getpid()}
    write_json(output / "status.json", status)
    try:
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("Scientific/E1 sources changed after the job was prepared.")
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        from review_experiments.e1.analysis import summarize
        from review_experiments.e1.runtime import install_runner
        from review_experiments.e1.smoke import compressed_smoke_clock
        from review_experiments.e1.study import StudyOptions, StudySession
        from contextlib import ExitStack

        torch.set_num_threads(job["threads"])
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job["seed"])
        options = StudyOptions(**{**job["options"], "horizons": tuple(job["options"]["horizons"])})
        records = []
        factory = lambda compiled, args, kwargs: StudySession(compiled, output / "capture", options, records=records,
                                                              source_hashes=job["source_hashes"])
        configure = lambda defaults: configure_defaults(defaults, job["category"], job["seed"], output, job["smoke"])
        with ExitStack() as stack:
            if job["smoke"]:
                stack.enter_context(compressed_smoke_clock(job["category"]))
            stack.enter_context(install_runner(job["category"], factory))
            namespace = execute_entrypoint(job["category"], configure)
        if namespace.get("TRAINING_PROFILE_NAME") != "robustness_200_100":
            raise RuntimeError("Entrypoint did not select the requested Scenario 2 profile.")
        # Preserve the completed raw trajectory even if final validation fails.
        # Only the completed status certifies source validation and analysis.
        with (output / "training.pkl").open("xb") as stream:
            pickle.dump(namespace["result_bundle"], stream, protocol=pickle.HIGHEST_PROTOCOL)
        status["trajectory_sha256"] = hashlib.sha256((output / "training.pkl").read_bytes()).hexdigest()
        status["trajectory_saved"] = True
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("Scientific/E1 sources changed during this capture; raw trajectory retained for diagnosis.")
        summarize(records, output / "analysis", options.practical_tolerance)
        status.update(state="completed", finished_at=utc_now(), decisions=len(records),
                      failed_pairs=sum(r["status"] != "completed" for r in records),
                      python=sys.executable, numpy=np.__version__, torch=torch.__version__,
                      device="cuda" if torch.cuda.is_available() else "cpu")
        write_json(output / "status.json", status)
        return 0
    except BaseException as error:
        status.update(state="failed", finished_at=utc_now(), error=str(error), error_type=type(error).__name__)
        write_json(output / "status.json", status)
        traceback.print_exc()
        return 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--categories", nargs="+", choices=ENTRIES, default=list(ENTRIES))
    value.add_argument("--seeds", nargs="+", type=int, default=[7])
    value.add_argument("--decisions-per-window", type=int, default=100)
    value.add_argument("--rollout-steps", type=int, default=200)
    value.add_argument("--horizons", nargs="+", type=int, default=[20, 80, 200])
    value.add_argument("--long-rollout-steps", type=int, default=800)
    value.add_argument("--long-fraction", type=float, default=0.1)
    value.add_argument("--snapshots-per-window", type=int, default=1)
    value.add_argument("--continuation", choices=["gated", "policy", "supervisor"], default="gated")
    value.add_argument("--return-tolerance", type=float, default=0.0)
    value.add_argument("--workers", type=int, default=1)
    value.add_argument("--threads", type=int, default=1)
    value.add_argument("--python", default=sys.executable)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--resume", type=Path)
    value.add_argument("--analyse", type=Path)
    value.add_argument("--replay-snapshot", type=Path)
    value.add_argument("--replay-steps", type=int)
    value.add_argument("--margin-offsets", nargs="+", type=float, default=[0.0])
    value.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_job:
        return worker(args.worker_job)
    if args.replay_snapshot:
        from review_experiments.e1.snapshots import replay_snapshot
        destination = (args.output_dir or args.replay_snapshot.parent.parent / "snapshot_replays").resolve() / (
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        )
        results = replay_snapshot(args.replay_snapshot, destination, steps=args.replay_steps,
                                  scientific_hashes=source_hashes())
        print(f"Snapshot replay saved: {destination}")
        return int(any(value["state"] != "completed" for value in results.values()))
    if args.analyse:
        from review_experiments.e1.analysis import load_records, summarize
        records = load_records(args.analyse)
        if not records:
            raise ValueError("No matched-decision records found.")
        summarize(records, args.analyse / "analysis", args.return_tolerance, tuple(args.margin_offsets))
        print(f"Summarized {len(records)} decisions.")
        return 0
    if len(set(args.categories)) != len(args.categories) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate categories/seeds are not allowed.")
    if not 1 <= args.workers <= 10 or args.threads < 1 or any(seed < 0 for seed in args.seeds):
        raise ValueError("Workers must be 1–10, threads positive and seeds nonnegative.")
    options = {
        "decisions_per_window": 1 if args.smoke else args.decisions_per_window,
        "rollout_steps": 8 if args.smoke else args.rollout_steps,
        "horizons": [4, 8] if args.smoke else args.horizons,
        "long_rollout_steps": 8 if args.smoke else args.long_rollout_steps,
        "long_fraction": 0.0 if args.smoke else args.long_fraction,
        "snapshots_per_window": args.snapshots_per_window,
        "continuation": args.continuation, "practical_tolerance": args.return_tolerance,
        "smoke": args.smoke,
    }
    if options["decisions_per_window"] < 1 or options["rollout_steps"] < 1 or not options["horizons"]:
        raise ValueError("Decision count, rollout length and report horizons must be positive.")
    if min(options["horizons"]) < 1 or max(options["horizons"]) > options["rollout_steps"]:
        raise ValueError("Report horizons must lie within the primary rollout.")
    if not 0 <= options["long_fraction"] <= 1 or options["long_rollout_steps"] < options["rollout_steps"]:
        raise ValueError("Invalid long-rollout settings.")
    if options["snapshots_per_window"] < 0 or options["practical_tolerance"] < 0:
        raise ValueError("Snapshot count and return tolerance must be nonnegative.")
    if args.resume:
        directory = args.resume.resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_hashes"] != source_hashes():
            raise ValueError("Cannot resume after a scientific/E1 source change.")
    else:
        matched = sum(4 if category == "combined" else 1 for category in args.categories) * 4 * options["decisions_per_window"] * len(args.seeds)
        print(f"E1: {len(args.seeds)} training seed(s), {len(args.categories) * len(args.seeds)} capture runs, {matched} matched channel decisions.")
        print("Original warm-up, critic-only hand-off and episode-201 plant change are retained.")
        print(f"Continuation: frozen {args.continuation}; primary branch steps: {options['rollout_steps']}.")
        if args.dry_run:
            print("Dry run: no output directory or training process created.")
            return 0
        preflight = subprocess.run([args.python, "-B", "-c",
            "import importlib.util,sys; names=('numpy','scipy','torch','control','cloudpickle','matplotlib'); "
            "missing=[n for n in names if importlib.util.find_spec(n) is None]; "
            "print('Interpreter:',sys.executable); print('Missing:',missing); sys.exit(bool(missing))"],
            capture_output=True, text=True)
        if preflight.returncode:
            raise RuntimeError("Activate rl-assisted-env before starting E1.\n" + preflight.stdout + preflight.stderr)
        directory = (args.output_dir or ROOT / "Polymer" / "Results" / "e1").resolve() / (
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        )
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": 1, "started_at": utc_now(), "source_hashes": source_hashes(),
                    "python": args.python, "workers": args.workers, "threads": args.threads,
                    "smoke": args.smoke, "options": options, "jobs": [
                        {"category": category, "seed": seed, "state": "pending", "attempts": []}
                        for category in args.categories for seed in args.seeds
                    ]}
        write_json(directory / "manifest.json", manifest)
    print(f"E1 output: {directory}", flush=True)
    live = {}
    active_category = None
    try:
        pending = [job for job in manifest["jobs"] if job["state"] != "completed"]
        while pending or live:
            if not live and pending:
                active_category = pending[0]["category"]
            while pending and len(live) < manifest["workers"] and pending[0]["category"] == active_category:
                job = pending.pop(0)
                attempt = directory / f"{job['category']}_s{job['seed']}" / f"a{len(job['attempts']) + 1}"
                attempt.mkdir(parents=True, exist_ok=False)
                specification = {"category": job["category"], "seed": job["seed"],
                                 "options": {**manifest["options"], "seed": job["seed"]},
                                 "source_hashes": manifest["source_hashes"], "threads": manifest["threads"],
                                 "smoke": manifest["smoke"]}
                write_json(attempt / "job.json", specification)
                environment = dict(os.environ)
                environment.update(MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                                   OMP_NUM_THREADS=str(manifest["threads"]), OPENBLAS_NUM_THREADS=str(manifest["threads"]),
                                   MKL_NUM_THREADS=str(manifest["threads"]), NUMEXPR_NUM_THREADS=str(manifest["threads"]))
                log = (attempt / "worker.log").open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen([manifest["python"], "-B", "-u", str(Path(__file__).resolve()),
                                                "--worker-job", str(attempt / "job.json")],
                                               cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                job["state"] = "running"
                job["attempts"].append(str(attempt.relative_to(directory)))
                live[process.pid] = (process, log, job, attempt)
                print(f"Started {job['category']} seed {job['seed']} (PID {process.pid}).", flush=True)
                write_json(directory / "manifest.json", manifest)
            for pid, (process, log, job, attempt) in list(live.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                status = json.loads((attempt / "status.json").read_text()) if (attempt / "status.json").exists() else {}
                job["state"] = "completed" if code == 0 and status.get("state") == "completed" else "failed"
                job["failed_pairs"] = status.get("failed_pairs")
                print(f"{job['category']} seed {job['seed']}: {job['state']}; see {attempt / 'worker.log'}", flush=True)
                del live[pid]
                write_json(directory / "manifest.json", manifest)
                if job["state"] == "failed":
                    pending.clear()
            if live:
                time.sleep(0.5)
        return int(any(job["state"] != "completed" for job in manifest["jobs"]))
    except KeyboardInterrupt:
        print("Stopping owned E1 workers; saved results are retained.", flush=True)
        for process, log, job, attempt in live.values():
            process.terminate()
            process.wait(timeout=20)
            log.close()
            job["state"] = "interrupted"
        write_json(directory / "manifest.json", manifest)
        return 130
    except BaseException:
        for process, log, job, attempt in live.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            log.close()
            job["state"] = "interrupted"
        write_json(directory / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
