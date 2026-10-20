"""One-off read-only status/artifact audit; never opens Aspen or starts a job."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from column_review.common import digest, read, process_identity
from revision_comparisons.progress import snapshot, log_tail


def main():
    import review_blas_runtime as runtime
    runtime.activate()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from run_e3_prediction_quality import completed_job
    from run_polymer_remaining import validate_output
    from comparison_studies.analysis import trajectories
    import recover_markov_seed23 as recovery

    issues, limitations, completed = [], [], {}
    result = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Current approved studies; superseded attempts retained separately."}
    for study, path in {
        "polymer_repetitions": "Polymer/Results/phase2_repeats/20261013_010021_35e44c",
        "column_repetitions": read(ROOT / "Distillation/Results/review_recovery/latest.json")["repeats"],
    }.items():
        directory = ROOT / path
        manifest = read(directory / "manifest.json")
        retained_hashes = {tuple(r["job"]): r["sha256"] for r in read(directory / "plan.json", {}).get("retained_results", [])}
        verified = []
        for job in manifest["jobs"]:
            if job["state"] != "completed":
                continue
            try:
                attempt = Path(job["attempts"][-1]["directory"])
                status = read(attempt / "worker_status.json")
                ok = (job.get("exit_code") == 0 and status.get("state") == "completed" and status.get("episodes") == 300
                      and status.get("result_file") == job.get("result_file") and Path(job["result_file"]).is_file())
                if study == "column_repetitions":
                    artifact = recovery.validated_result(job, manifest)
                    if recovery.key(job) in retained_hashes and artifact["sha256"] != retained_hashes[recovery.key(job)]:
                        raise ValueError("Retained result differs from its pre-launch SHA-256 record.")
                    ok = ok and status.get("aspen_closed") is True
                if not ok:
                    raise ValueError("Saved worker completion contract differs.")
                verified.append({"category": job["category"], "mode": job["mode"], "seed": job["seed"],
                                 "path": job["result_file"], "bytes": Path(job["result_file"]).stat().st_size})
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                issues.append({"study": study, "job": [job.get(k) for k in ("category", "mode", "seed")], "error": str(error)})
        completed[study] = {"verified": len(verified), "total": len(manifest["jobs"]), "runs": verified}

    polymer = ROOT / "Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent"
    pipeline = {}
    for stage in ("e1_combined", "e2_sg", "e2_arx_repaired"):
        status = read(polymer / stage / "status.json")
        try:
            validate_output(polymer / stage / "job.json", status)
            valid = status["state"] == "completed"
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            valid = False
            issues.append({"study": stage, "error": str(error)})
        pipeline[stage] = {"status": status, "artifact_contract_verified": valid,
                           "postprocess_marker": read(polymer / stage / "postprocess_status.json", {})}
    completed["polymer_recovery"] = pipeline
    e1 = read(polymer / "analysis/e1/provenance.json")
    pair_counts = {}
    identities = set()
    for source in e1["sources"]:
        path = Path(source["path"])
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        valid = digest(path) == source["sha256"] and all(r["status"] == "completed" for r in records)
        identities.update((r["seed"], r["category"], r["channel"], r["step"]) for r in records)
        pair_counts[str(path)] = {"records": len(records), "hash_and_completed_pairs_verified": valid}
        if not valid:
            issues.append({"study": "polymer_E1", "path": str(path), "error": "Matched-pair data did not verify."})
    completed["polymer_E1"] = {"matched_pairs": sum(v["records"] for v in pair_counts.values()),
                               "unique_pairs": len(identities), "sources": pair_counts}
    if len(identities) != 3200:
        issues.append({"study": "polymer_E1", "error": "Expected 3200 unique completed matched pairs."})
    limitations.append({"study": "polymer_E1", "issue": "Historical residual final trajectory export is absent; its 400 matched pairs remain verified and are included without rerunning them."})

    e2 = read(polymer / "analysis/e2/provenance.json")
    sources = {}
    for method, source in e2["sources"].items():
        path = Path(source["path"])
        valid = digest(path) == source["sha256"]
        with path.open("rb") as stream:
            data = pickle.load(stream)
        try:
            trajectories(data)  # Includes the canonical baseline's y_mpc/u_mpc schema.
            finite = True
        except (ValueError, KeyError) as error:
            finite = False
            issues.append({"study": "polymer_E2", "method": method, "error": str(error)})
        sources[method] = {"path": str(path), "hash_verified": valid, "finite_primary_arrays": bool(finite), "steps": int(data.get("nFE", 0))}
        if not valid or not finite:
            issues.append({"study": "polymer_E2", "method": method, "error": "Trajectory hash or primary finite arrays failed."})
        del data
    with (polymer / "analysis/e2/comparison_metrics.csv").open(newline="") as stream:
        metric_rows = sum(1 for _ in csv.DictReader(stream))
    completed["polymer_E2"] = {"sources": sources, "metric_rows": metric_rows, "analysis_state": e2["state"]}
    if metric_rows != 30 or e2["state"] != "completed":
        issues.append({"study": "polymer_E2", "error": "Shared six-controller analysis is incomplete."})
    threshold_path = ROOT / "Polymer/Results/comparisons/scenario1/20261014_runtime_recovery_threshold_sg/threshold/final_inference_networks.pt"
    state = torch.load(threshold_path, map_location="cpu", weights_only=True)
    def tensor_count(value):
        if isinstance(value, torch.Tensor):
            return int(value.numel())
        if isinstance(value, dict):
            return sum(tensor_count(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_count(v) for v in value)
        return 0
    count = tensor_count(state)
    limitations.extend([
        {"study": "polymer_threshold", "issue": "Historical inference export is empty; completed trajectory is usable, but frozen-policy evaluation cannot restore this policy.",
         "checkpoint": str(threshold_path), "tensor_elements": count, "sha256": digest(threshold_path)},
        {"study": "polymer_ARX_postprocess", "issue": "Local ARX marker still says waiting; the authoritative shared E2 provenance/30-row analysis is complete and includes validated ARX/SG trajectories."},
    ])
    result["polymer_threshold_policy_tensor_elements"] = count

    e3 = {}
    for plant, relative in (("polymer", "Results/e3/20261015_042751_513bff"), ("column", "Results/e3/20261015_042751_4e03da")):
        manifest = read(ROOT / relative / "manifest.json")
        e3[plant] = [{"id": j["id"], "category": j["category"], "artifact_inventory_verified": completed_job(j)} for j in manifest["jobs"]]
        issues.extend({"study": "E3_" + plant, "job": j["id"], "error": "Saved artifact inventory failed."} for j in e3[plant] if not j["artifact_inventory_verified"])
    completed["E3"] = e3

    column = ROOT / read(ROOT / "Distillation/Results/review_recovery/latest.json")["column"]
    comparisons = {}
    for method in ("sg", "threshold", "ls", "bo"):
        attempt = Path(read(column / method / "attempts.json")[-1]["directory"])
        status = read(attempt / "status.json")
        row = {"state": status["state"], "directory": str(attempt), "status": status}
        if status["state"] == "completed":
            valid = (status.get("aspen_closed") is True and status.get("cycles") == 200 and status.get("steps") == 80000
                     and bool(status.get("files")) and all((attempt / p).is_file() and digest(attempt / p) == sha for p, sha in status["files"].items()))
            row["artifact_inventory_verified"] = valid
            if not valid:
                issues.append({"study": "column_comparisons", "method": method, "error": "Completed artifact inventory failed."})
        if method == "bo":
            trials = read(attempt / "bo_trials.json", [])
            row.update(calibration_trials_complete=len(trials), calibration_trials_planned=read(column / method / "job.json")["bo_trials"],
                       valid_saved_trials=sum(bool(t.get("valid")) for t in trials), final_evaluation_cycles=200,
                       phase="calibration" if len(trials) < 40 else "inspect_final_evaluation")
        comparisons[method] = row
    completed["column_comparisons"] = comparisons
    result["snapshot_pilot_hold"] = read(column / "e1_pilot/snapshot_hold.json")

    import win32com.client
    service = win32com.client.GetObject("winmgmts:")
    inventory = []
    for process in service.ExecQuery("SELECT ProcessId,ParentProcessId,CommandLine FROM Win32_Process WHERE Name='python.exe'"):
        command = str(process.CommandLine)
        if str(ROOT).lower() in command.lower():
            inventory.append({"pid": int(process.ProcessId), "parent_pid": int(process.ParentProcessId),
                              "worker": "--worker" in command, "coordinator": "--coordinate" in command})
    observed = snapshot()
    current = [r for r in observed["jobs"] if r["study"] not in observed["historical_studies"]]
    active = [r for r in current if r["state"] == "running"]
    registered = {r["pid"] for r in active}
    actual = {r["pid"] for r in inventory if r["worker"]}
    result["process_inventory"] = inventory
    result["unregistered_worker_pids"] = sorted(actual - registered)
    if actual - registered:
        issues.append({"study": "inventory", "error": "An additional repository worker exists.", "pids": sorted(actual - registered)})
    health = []
    for row in active:
        directory = Path(row["directory"])
        log = directory / ("worker.log" if (directory / "worker.log").exists() else "run.log")
        tail = log_tail(log, 16000)
        fatal = "Windows fatal exception" in tail or "Traceback (most recent call last)" in tail
        info = {"job": row["job"], "study": row["study"], "process": process_identity(row["pid"]), "cycle": row["cycle"],
                "recent_fatal_log": fatal, "last_log_at": row["last_log_at"], "aspen_number": row["aspen_number"]}
        health.append(info)
        if fatal or info["process"]["alive"] is not True:
            issues.append({"study": row["study"], "job": row["job"], "error": "Worker health requires inspection."})
    production = [r for r in current if r["job"] != "e1_pilot"]
    totals = Counter(r["state"] for r in production)
    result.update(observed_at_utc=observed["observed_at_utc"], current_snapshot=observed,
                  production_totals=dict(totals), remaining_production=sum(n for s, n in totals.items() if s != "completed"),
                  remaining_running=totals["running"], remaining_queued=totals["pending"], failed_held_pilots=1,
                  active_worker_health=health, completed=completed, verification_issues=issues, known_limitations=limitations,
                  planned_work_outside_current_queue=["Full column E1 after native restoration is validated", "Frozen-policy unseen-reference evaluations", "Separately scoped E4 joint/causal diagnostics and E5 supervisory sensitivity"],
                  audit_script_sha256=digest(__file__))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(__file__).with_name(f"all_runs_{stamp}.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"observed": result["observed_at_utc"], "remaining_production": result["remaining_production"],
                      "production_totals": dict(totals), "held_pilots": 1, "BO_trials": comparisons["bo"]["calibration_trials_complete"],
                      "verification_issues": issues, "threshold_policy_elements": count,
                      "unregistered_workers": result["unregistered_worker_pids"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
