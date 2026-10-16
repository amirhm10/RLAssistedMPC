"""Prepare/run the original 200-cycle column alternatives in isolated workers.

Printing a plan is the default. --run explicitly starts the long Aspen study.
--resume DIRECTORY preserves successful jobs and creates new attempts for
unfinished jobs, after checking that no earlier worker remains live.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import uuid

from revision_comparisons.runtime import numerical_environment

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=["threshold", "ls", "bo", "sg", "nominal"],
                   default=["threshold", "ls", "bo"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--aspen-number", type=int, default=1,
                   help="Case must be free; active repeat/E3 case locks are respected.")
    p.add_argument("--aspen-root", type=Path)
    p.add_argument("--bo-trials", type=int, default=40)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--sg-reference", type=Path, help="Completed matching new-runner SG job directory.")
    p.add_argument("--run", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    return p


def completed(directory, expected):
    from review_experiments.e3.common import sha256
    directory = Path(directory)
    try:
        job = json.loads((directory/"job.json").read_text(encoding="utf-8"))
        status = json.loads((directory/"status.json").read_text(encoding="utf-8"))
        required = {"training.pkl", "metrics.json", "timing.json", "resolved_defaults.json",
                    "baseline_audit.json", "baseline_metrics.json", "runtime_metadata.json"}
        if expected.get("method") in ("sg", "threshold"):
            required.add("final_inference_networks.pt")
        return (status["state"] == "completed" and status.get("aspen_closed") is True
                and all(job.get(k) == expected.get(k) for k in ("method", "seed", "source_hashes", "profile", "bo_trials"))
                and required.issubset(status.get("files", {}))
                and all((directory/name).is_file() and sha256(directory/name) == digest
                        for name, digest in status["files"].items()))
    except (OSError, ValueError, KeyError):
        return False


def summarize(directory, manifest):
    import csv
    from review_experiments.e3.common import write_json
    rows, sources = [], []
    for item in manifest["jobs"]:
        path = Path(item["reference"] if item.get("reference") else item["attempts"][-1]["directory"])
        if not completed(path, item["specification"]):
            raise RuntimeError("Only verified completed comparison attempts may be summarized.")
        rows.extend(json.loads((path/"metrics.json").read_text(encoding="utf-8")))
        sources.append(str(path))
    if sources:
        path = Path(sources[0])/"baseline_metrics.json"
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
    fields = sorted({key for row in rows for key in row})
    with (directory/"comparison_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory/"analysis_metadata.json", {
        "sources": sources, "alignment": "after-action y[k+1] minus reference[k]",
        "nominal_reference": "Saved original trajectory, re-scored with the active common reward; see baseline_audit.json",
        "replication": "one training seed; separate repeat study supplies replication evidence",
        "temperature_error_unit": "kelvin difference, numerically equal to Celsius difference; original raw measurements are preserved"})


def worker(path):
    import numpy as np
    import torch
    from comparison_studies.common import solver_audit
    from comparison_studies.checkpoints import save_inference_checkpoint
    from review_experiments.e3.common import sha256, utc_now, write_json
    from revision_comparisons.column import source_hashes, run_method, metrics
    path = Path(path)
    job, output = json.loads(path.read_text(encoding="utf-8")), path.parent
    status = {"state": "running", "pid": os.getpid(), "started_at": utc_now()}
    write_json(output/"status.json", status)
    began = time.perf_counter()
    try:
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("Comparison dependencies changed after preparation.")
        torch.set_num_threads(job["threads"])
        from revision_comparisons.runtime import metadata
        write_json(output/"runtime_metadata.json", metadata())
        with solver_audit() as audit:
            result, agents = run_method(job["method"], output, job["aspen"], job["seed"], bo_trials=job["bo_trials"])
        status["aspen_closed"] = True
        result = {k: v for k, v in result.items() if not k.startswith("_")}
        if result["nFE"] != 80000 or len(result["avg_rewards"]) != 200:
            raise RuntimeError("The original complete 200-cycle Scenario 1 is required.")
        result.update(comparison_method=job["method"], comparison_seed=job["seed"],
                      comparison_smoke=False, comparison_profile="legacy_200",
                      comparison_source_hashes=job["source_hashes"], comparison_solver_audit=np.asarray(audit),
                      comparison_worker_seconds=time.perf_counter()-began)
        with (output/"training.pkl").open("xb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if agents is not None:
            expected = {"horizon_agent", "weight_agent", "markov_agent", "residual_agent"}
            if set(agents) != expected:
                raise RuntimeError("Combined inference export must include all four agents.")
            save_inference_checkpoint(output/"final_inference_networks.pt", agents,
                                      metadata={"plant": "distillation", "profile": "legacy_200", "seed": job["seed"],
                                                "method": job["method"], "source_hashes": job["source_hashes"]})
        write_json(output/"metrics.json", metrics(result))
        write_json(output/"timing.json", result.get("comparison_timing"))
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("Comparison dependencies changed; raw trajectory retained for diagnosis.")
        if sha256(job["aspen"]["aspen_path"]) != job["aspen"]["aspen_sha256"]:
            raise RuntimeError("The original Aspen case file changed during execution.")
        files = {p.relative_to(output).as_posix(): sha256(p) for p in output.iterdir()
                 if p.is_file() and p.name not in {"worker.log", "job.json", "status.json"}}
        status.update(state="completed", finished_at=utc_now(), files=files,
                      solver_calls=len(audit), solver_failed_calls=sum(not r[1] for r in audit),
                      duration_seconds=time.perf_counter()-began)
        write_json(output/"status.json", status)
        return 0
    except BaseException as error:
        status.update(state="failed", error=str(error), error_type=type(error).__name__, finished_at=utc_now())
        write_json(output/"status.json", status)
        traceback.print_exc()
        return 1


def schedule(directory, manifest):
    from run_distillation_phase2_repeats import aspen_file_lock
    from run_e3_prediction_quality import pid_alive
    from review_experiments.e3.common import utc_now, write_json
    from revision_comparisons.column import source_hashes
    directory = Path(directory)
    with aspen_file_lock(directory/"study", directory/"_locks"):
        if source_hashes() != manifest["source_hashes"]:
            raise RuntimeError("Saved study dependencies differ. Prepare a separate new study.")
        for item in manifest["jobs"]:
            last = item["attempts"][-1] if item["attempts"] else None
            if last and last.get("state") == "running" and pid_alive(last.get("pid")):
                raise RuntimeError("A prior comparison worker is still live; no duplicate will be launched.")
        manifest["state"] = "running"
        write_json(directory/"manifest.json", manifest)
        for item in manifest["jobs"]:
            if item.get("reference"):
                if not completed(item["reference"], item["specification"]):
                    raise RuntimeError("The saved reference no longer passes its hash/provenance audit.")
                item["state"] = "completed"
                continue
            last = item["attempts"][-1] if item["attempts"] else None
            if last and completed(last["directory"], item["specification"]):
                item["state"] = "completed"
                continue
            attempt = directory/item["method"]/f"a{len(item['attempts'])+1}"
            attempt.mkdir(parents=True, exist_ok=False)
            spec = dict(item["specification"])
            write_json(attempt/"job.json", spec)
            print(f"Starting {item['method']}: {attempt}", flush=True)
            with (attempt/"worker.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                         "--worker-job", str(attempt/"job.json")], cwd=ROOT,
                                        env=numerical_environment(manifest["threads"]), stdout=log, stderr=subprocess.STDOUT)
                record = {"directory": str(attempt), "pid": proc.pid, "state": "running", "started_at": utc_now()}
                item["attempts"].append(record)
                item["state"] = "running"
                write_json(directory/"manifest.json", manifest)
                with (attempt/"worker.log").open(encoding="utf-8", errors="replace") as reader:
                    while proc.poll() is None:
                        chunk = reader.read()
                        if chunk:
                            print(chunk, end="", flush=True)
                        time.sleep(1.)
                    print(reader.read(), end="", flush=True)
                code = proc.wait()
            record.update(state="completed" if code == 0 and completed(attempt, spec) else "failed",
                          returncode=code, finished_at=utc_now())
            item["state"] = record["state"]
            if item["state"] != "completed":
                manifest["state"] = "failed"
                write_json(directory/"manifest.json", manifest)
                print("Stopped after the failed attempt. Preserve it and resume after diagnosis.", flush=True)
                return 1
            write_json(directory/"manifest.json", manifest)
        summarize(directory, manifest)
        manifest.update(state="completed", finished_at=utc_now())
        write_json(directory/"manifest.json", manifest)
    print(f"Column comparisons finished: {directory}", flush=True)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_job:
        return worker(args.worker_job)
    if not 0 <= args.seed <= 2**32-1 or args.threads < 1 or args.bo_trials < 5:
        raise ValueError("Require a valid unsigned seed, positive threads and at least five BO trials.")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Select each method once.")
    if args.resume:
        if not args.run or args.dry_run:
            print(json.dumps({"resume": str(args.resume), "action": "plan only; add --run to resume"}, indent=2))
            return 0
        manifest = json.loads((args.resume/"manifest.json").read_text(encoding="utf-8"))
        return schedule(args.resume.resolve(), manifest)
    from revision_comparisons.column import source_hashes
    from review_experiments.e3.runtime import resolve_aspen
    hashes = source_hashes()
    aspen = resolve_aspen("combined", root_override=args.aspen_root, number_override=args.aspen_number)
    methods = list(args.methods)
    reference = None
    if "threshold" in methods and "sg" not in methods:
        if args.sg_reference:
            expected = {"method": "sg", "seed": args.seed, "source_hashes": hashes,
                        "profile": "legacy_200", "bo_trials": args.bo_trials}
            if not completed(args.sg_reference, expected):
                raise ValueError("The SG reference lacks matching completed provenance and hashes.")
            reference = str(args.sg_reference.resolve())
        else:
            methods.insert(methods.index("threshold")+1, "sg")
    plan = {"plant": "distillation", "methods": methods, "seed": args.seed,
            "profile": "legacy_200", "cycles": 200, "steps_per_cycle": 400, "main_steps": 80000,
            "warmup_cycles": [1, 10], "handoff_cycles": [11, 13], "policy_release_cycle": 14,
            "final_evaluation_cycle": 200, "bo_trials": args.bo_trials,
            "bo_objective_steps_per_trial": 800, "bo_replayed_prefix_steps_per_trial": 4000,
            "bo_total_calibration_steps": args.bo_trials*4800, "threshold_calibration_steps": 4000,
            "aspen": aspen, "threads": args.threads, "sg_reference": reference,
            "source_hashes": hashes, "new_production_runs_started": False}
    if args.dry_run or not args.run:
        print(json.dumps({k: v for k, v in plan.items() if k != "source_hashes"}, indent=2))
        return 0
    from run_distillation_phase2_repeats import aspen_file_lock
    # Fail before allocating a study if the selected case is currently owned.
    with aspen_file_lock(Path(aspen["aspen_path"]), ROOT/"Distillation/Results/_aspen_file_locks"):
        pass
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
    directory = (args.output_dir or ROOT/"Distillation/Results/comparisons/scenario1"/stamp).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from review_experiments.e3.common import write_json, utc_now
    manifest = dict(plan, state="pending", new_production_runs_started=True, started_at=utc_now(), python=sys.executable,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    git_dirty=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True), jobs=[])
    if reference:
        methods.append("sg")
    for method in methods:
        spec = {k: plan[k] for k in ("seed", "source_hashes", "profile", "bo_trials", "aspen", "threads")}
        spec["method"] = method
        manifest["jobs"].append({"method": method, "specification": spec, "state": "pending",
                                 "attempts": [], "reference": reference if method == "sg" else None})
    write_json(directory/"manifest.json", manifest)
    return schedule(directory, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
