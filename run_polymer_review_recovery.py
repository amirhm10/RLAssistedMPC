"""Prepare or run only the missing polymer E1/E2 studies in fresh directories.

The default is a read-only plan. --run explicitly starts E1 combined, followed
by E2 critic-gated. Historical failed statuses and completed data are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
E1_STUDY = ROOT / "Polymer/Results/e1/20261013_192015_916c06"
E2_STUDY = ROOT / "Polymer/Results/comparisons/scenario1/20261014_runtime_recovery_threshold_sg"
COMPARISONS = {
    "ls": "20261013_194218_ls_production/ls",
    "arx": "20261014_033927_fd26b6/arx",
    "bo": "20261014_033927_fd26b6/bo",
    "threshold": "20261014_runtime_recovery_threshold_sg/threshold",
}
REPAIRED_WRAPPERS = {"run_polymer_e1.py", "run_polymer_comparisons.py", "comparison_studies/common.py"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audit_historical_sources(expected):
    """Permit only the documented wrapper fixes, never scientific changes."""
    changed_wrappers = {}
    for name, expected_hash in expected.items():
        path = ROOT / name
        actual = sha256(path) if path.is_file() else None
        if actual == expected_hash:
            continue
        if actual is None or name not in REPAIRED_WRAPPERS:
            raise ValueError(f"Historical dependency changed or disappeared: {name}")
        changed_wrappers[name] = {"before": expected_hash, "after": actual}
    return {"checked_files": len(expected), "documented_wrapper_changes": changed_wrappers}


def build_plan(python, *, smoke=False):
    manifest = read_json(E1_STUDY / "manifest.json")
    if manifest["smoke"] or {job["seed"] for job in manifest["jobs"]} != {7}:
        raise ValueError("Recovery expects the approved production seed-7 E1 study.")
    source_audit = audit_historical_sources(manifest["source_hashes"])
    reused = {}
    for job in manifest["jobs"]:
        category = job["category"]
        if category == "combined":
            if job["state"] != "pending" or job["attempts"]:
                raise ValueError("E1 combined is no longer unstarted; inspect its current results before launching another run.")
            continue
        attempt = E1_STUDY / job["attempts"][-1]
        status = read_json(attempt / "status.json")
        capture = read_json(attempt / "capture/capture_status.json")
        if capture["state"] != "completed" or capture["saved_decisions"] != 400:
            raise ValueError(f"Incomplete historical E1 captures: {category}")
        lines = (attempt / "capture/decisions.jsonl").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        counts = Counter(record["window"] for record in records)
        if (len(records) != 400 or any(record["status"] != "completed" for record in records)
                or len({(r["channel"], r["step"]) for r in records}) != 400
                or sorted(counts.values()) != [100, 100, 100, 100]):
            raise ValueError(f"Historical E1 decision inventory is invalid: {category}")
        if category == "residual":
            if status["state"] != "failed" or "sources changed during this capture" not in status.get("error", ""):
                raise ValueError("Residual failure is not the audited export-only source-guard failure.")
        elif job["state"] != "completed" or status["state"] != "completed" or not (attempt / "training.pkl").is_file():
            raise ValueError(f"Historical E1 training is incomplete: {category}")
        reused[f"e1_{category}"] = {"directory": str(attempt), "decisions": len(records),
            "decisions_sha256": sha256(attempt / "capture/decisions.jsonl"),
            "original_state": status["state"], "trajectory_available": (attempt / "training.pkl").is_file()}
    if set(reused) != {"e1_horizon", "e1_weights", "e1_markov", "e1_residual"}:
        raise ValueError("Historical E1 inventory does not contain all four standalone channels.")
    for method, relative in COMPARISONS.items():
        directory = ROOT / "Polymer/Results/comparisons/scenario1" / relative
        job, status = read_json(directory / "job.json"), read_json(directory / "status.json")
        if (job["method"] != method or job["seed"] != 7 or job["smoke"]
                or job["profile"] != "legacy_gradual_200" or status["state"] != "completed"
                or status["steps"] != 160000 or status["trajectory_sha256"] != sha256(directory / "training.pkl")):
            raise ValueError(f"Historical {method} comparison provenance failed.")
        reused[f"e2_{method}"] = {"directory": str(directory), "steps": status["steps"],
            "trajectory_sha256": status["trajectory_sha256"],
            "source_audit": audit_historical_sources(job["source_hashes"])}
    sg_job, sg_status = read_json(E2_STUDY / "sg/job.json"), read_json(E2_STUDY / "sg/status.json")
    if sg_status["state"] != "failed" or "Sources changed during the run" not in sg_status.get("error", ""):
        raise ValueError("Historical SG failure is not the audited source-guard export failure.")
    sg_audit = audit_historical_sources(sg_job["source_hashes"])
    options = manifest["options"]
    e1 = [str(python), "-B", "-u", str(ROOT / "run_polymer_e1.py"), "--categories", "combined",
          "--seeds", "7", "--workers", "1", "--threads", str(manifest["threads"]), "--python", str(python),
          "--decisions-per-window", str(options["decisions_per_window"]),
          "--rollout-steps", str(options["rollout_steps"]), "--horizons", *map(str, options["horizons"]),
          "--long-rollout-steps", str(options["long_rollout_steps"]), "--long-fraction", str(options["long_fraction"]),
          "--snapshots-per-window", str(options["snapshots_per_window"]),
          "--continuation", options["continuation"], "--return-tolerance", str(options["practical_tolerance"])]
    e2 = [str(python), "-B", "-u", str(ROOT / "run_polymer_comparisons.py"), "--methods", "sg",
          "--seed", "7", "--threads", str(sg_job["threads"]), "--python", str(python)]
    if smoke:
        e1.append("--smoke")
        e2.append("--smoke")
    return {"schema": 1, "smoke": smoke, "python": str(python),
            "e1_source_audit": source_audit, "e2_sg_source_audit": sg_audit,
            "retained_evidence": reused, "stages": [{"name": "e1_combined", "command": e1},
                                                      {"name": "e2_sg", "command": e2}],
            "limitations": ["Residual matched pairs are retained; its original training trajectory is unavailable.",
                "Old empty threshold/SG network files cannot restore trained weights.",
                "No residual, threshold, LS, ARX or BO training is scheduled.",
                "New exports are separate attempts, not recovered historical trajectories."]}


def tail_logs(directory, offsets):
    for path in sorted(directory.rglob("*.log")):
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(offsets.get(path, 0))
            chunk = stream.read()
            offsets[path] = stream.tell()
        if chunk:
            print(f"\n[{path.relative_to(directory)}]\n{chunk}", end="", flush=True)


def run_stage(command, directory, name, offsets):
    environment = dict(os.environ, PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    print(f"\nStarting {name}; command: {subprocess.list2cmdline(command)}", flush=True)
    with (directory / f"{name}_launcher.log").open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                tail_logs(directory, offsets)
                time.sleep(1)
            tail_logs(directory, offsets)
            return process.returncode
        except KeyboardInterrupt:
            # The shared CMD console also sends Ctrl+C to owned child Python
            # processes. Give their normal handlers time to preserve results.
            print("Stopping recovery; retaining saved results.", flush=True)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
                else:
                    process.terminate()
            raise


def verify_stage_outputs(name, output, smoke):
    """Exit zero alone is insufficient: require complete saved artifacts."""
    if name == "e1_combined":
        statuses = list(output.glob("*/combined_s7/a1/status.json"))
        if len(statuses) != 1:
            raise ValueError("Cannot identify the new E1 combined attempt.")
        path = statuses[0]
        status = read_json(path)
        capture = read_json(path.parent / "capture/capture_status.json")
        expected = 16 if smoke else 1600
        if (status["state"] != "completed" or status.get("failed_pairs") != 0
                or status.get("decisions") != expected or capture["state"] != "completed"
                or capture["saved_decisions"] != expected):
            raise ValueError("E1 combined has incomplete or failed matched decisions.")
    else:
        path = output / "sg/status.json"
        status = read_json(path)
        checkpoint = status.get("checkpoint", {})
        if (status["state"] != "completed" or status.get("steps") != (800 if smoke else 160000)
                or not checkpoint.get("round_trip_verified") or len(checkpoint.get("tensor_counts", {})) != 4
                or not all(checkpoint["tensor_counts"].values())
                or checkpoint.get("sha256") != sha256(path.parent / "final_inference_networks.pt")):
            raise ValueError("E2 SG lacks a complete trajectory and validated four-agent checkpoint.")
    if status.get("trajectory_sha256") != sha256(path.parent / "training.pkl"):
        raise ValueError("The saved training trajectory hash does not match its completed status.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Technical compressed-clock checks only.")
    parser.add_argument("--stages", nargs="+", choices=("e1_combined", "e2_sg"),
                        default=["e1_combined", "e2_sg"], help="Select only a still-needed stage after a partial recovery.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    plan = build_plan(sys.executable, smoke=args.smoke)
    if len(set(args.stages)) != len(args.stages):
        raise ValueError("Duplicate recovery stages are not allowed.")
    plan["stages"] = [stage for stage in plan["stages"] if stage["name"] in args.stages]
    if not args.run:
        print(json.dumps(plan, indent=2))
        print("Plan only: no output directory, training or Aspen session was created.")
        return 0
    directory = (args.output_dir or ROOT / "Polymer/Results/review_recovery" /
                 (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for stage in plan["stages"]:
        output = directory / stage["name"]
        stage["command"].extend(["--output-dir", str(output)])
    plan["launcher_sha256"] = sha256(__file__)
    write_json(directory / "plan.json", plan)
    print(f"Recovery output: {directory}\nStages are sequential; live worker logs follow.", flush=True)
    offsets = {}
    try:
        for stage in plan["stages"]:
            code = run_stage(stage["command"], directory, stage["name"], offsets)
            exit_status = {"returncode": code}
            if not code:
                try:
                    verify_stage_outputs(stage["name"], directory / stage["name"], args.smoke)
                    exit_status["saved_artifacts_verified"] = True
                except (ValueError, OSError, KeyError) as error:
                    code = 1
                    exit_status.update(returncode=code, validation_error=str(error))
                    print(f"Result validation failed: {error}", flush=True)
            write_json(directory / f"{stage['name']}_exit.json", exit_status)
            if code:
                print(f"Stopped after {stage['name']} failed. Review {directory}; historical studies are untouched.", flush=True)
                return code
    except KeyboardInterrupt:
        return 130
    print(f"Recovery stages finished. Results and retained-evidence index: {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
