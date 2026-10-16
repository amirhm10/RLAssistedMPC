"""Isolated numerical-runtime recovery of the failed polymer E3 combined job.

No controller, E3 implementation file or installed package is modified. The
default prints the plan. --run resumes only the failed combined stage, using
the original E3 scheduler and preserving completed nominal/matrix stages.
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

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "Results/e3/20261015_042751_513bff"
RUNTIME = {"NPY_DISABLE_CPU_FEATURES": "X86_V3,AVX2,FMA3", "OPENBLAS_CORETYPE": "NEHALEM",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"}


def environment(base=None):
    return {**(os.environ if base is None else base), **RUNTIME}


def runtime_info():
    """Validate actual dispatch, rather than merely recording requested flags."""
    import numpy as np
    import scipy
    from numpy._core import _multiarray_umath
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
    blas = [pool for pool in pools if pool["internal_api"] == "openblas"]
    features = _multiarray_umath.__cpu_features__
    if features.get("X86_V3") is not False:
        raise RuntimeError("The isolated worker did not disable NumPy X86_V3 dispatch.")
    if not blas or any(pool["architecture"].lower() != "nehalem" or pool["num_threads"] != 1 for pool in blas):
        raise RuntimeError("The isolated worker did not select single-threaded Nehalem OpenBLAS.")
    return {"python": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__,
            "requested_environment": RUNTIME, "numpy_x86_v3_enabled": features.get("X86_V3"),
            "threadpools": pools, "scope": "process-local numerical dispatch; unchanged equations/settings",
            "interpretation": "native-crash mitigation; exact underlying binary/hardware trigger unproven"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def plan(study):
    import run_e3_prediction_quality as e3
    manifest = read(study / "manifest.json")
    if manifest["options"]["plants"] != ["polymer"] or manifest["options"]["smoke"]:
        raise ValueError("Recovery is restricted to the production polymer E3 study.")
    jobs = {job["category"]: job for job in manifest["jobs"]}
    if set(jobs) != {"nominal", "markov", "combined"}:
        raise ValueError("Unexpected E3 stage inventory.")
    for category in ("nominal", "markov"):
        if not e3.completed_job(jobs[category]):
            raise ValueError(f"Previously completed {category} artifacts do not verify; do not rerun silently.")
    combined = jobs["combined"]
    if e3.completed_job(combined):
        raise ValueError("Combined E3 is already complete; refusing a duplicate run.")
    if combined["state"] != "failed" or not combined["attempts"]:
        raise ValueError("Expected a failed combined attempt; inspect current status before restarting.")
    if e3.pid_alive(combined["attempts"][-1].get("pid")):
        raise RuntimeError("The prior combined PID is still alive; refusing to interrupt or duplicate it.")
    e3.check_sources(manifest["source_hashes"])
    return manifest, {"study": str(study), "restart_only": combined["id"],
        "skip_verified": [jobs[c]["id"] for c in ("nominal", "markov")],
        "options_unchanged": manifest["options"], "python_unchanged": manifest["python"],
        "runtime_override": RUNTIME, "restart_from_cycle": 1,
        "reason": "No exact training checkpoint survived the native access violation.",
        "controller_source_changes": [], "installed_package_changes": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check_runtime:
        print(json.dumps(runtime_info(), indent=2))
        return 0
    if args.worker_job:
        job = read(args.worker_job)
        if job["plant"] != "polymer" or job["category"] != "combined":
            raise ValueError("Isolated worker supports only polymer combined; no Aspen sessions are opened.")
        from review_experiments.e3.common import write_json
        info = runtime_info()
        info["bootstrap_sha256"] = digest(__file__)
        write_json(args.worker_job.parent / "isolated_runtime.json", info)
        import run_e3_prediction_quality as e3
        return e3.run_worker(args.worker_job)
    study = args.study.resolve()
    manifest, summary = plan(study)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.run:
        print("Plan only: no training, package changes, or study writes.")
        return 0
    import run_e3_prediction_quality as e3
    from run_distillation_phase2_repeats import aspen_file_lock
    with aspen_file_lock(study / "study", study / "_locks"):
        # Recheck inside the study lock before making a new attempt.
        manifest, summary = plan(study)
        record = study / "recovery" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        record.mkdir(parents=True, exist_ok=False)
        (record / "manifest_before.json").write_bytes((study / "manifest.json").read_bytes())
        e3.write_json(record / "recovery.json", {**summary, "bootstrap_sha256": digest(__file__)})
        manifest["state"] = "running"
        manifest["isolated_recovery"] = str(record / "recovery.json")
        e3.write_json(study / "manifest.json", manifest)

        def launch(command, **kwargs):
            job_path = Path(command[-1])
            job = read(job_path)
            if job["id"] != summary["restart_only"]:
                raise RuntimeError(f"Refusing an unexpected recovery job: {job['id']}")
            kwargs["env"] = environment(kwargs["env"])
            actual = [command[0], "-B", "-u", str(Path(__file__).resolve()), "--worker-job", str(job_path)]
            e3.write_json(job_path.parent / "recovery_launch.json", {
                "command": actual, "runtime_environment": RUNTIME, "original_command": command})
            return subprocess.Popen(actual, **kwargs)

        e3.run_jobs(manifest, study, launch=launch)
        e3.run_analysis(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
