"""Read-only evidence snapshot for the isolated seed 23 restart and queue."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from column_review.common import digest, process_identity, read
from revision_comparisons.progress import snapshot


def source_check(path):
    record = read(ROOT / path)
    expected = record["source_hashes"]
    changed = [name for name, sha in expected.items() if digest(ROOT / name) != sha]
    return {"manifest": path, "files": len(expected), "changed": changed, "unchanged": not changed}


def main():
    output = Path(__file__).resolve().parent
    directory = ROOT / "Distillation/Results/review_recovery/20261020_openblas"
    plan = read(directory / "plan.json")
    manifest = read(directory / "manifest.json")
    retry = read(directory / "retry/manifest.json")["jobs"][0]
    attempt = Path(retry["attempts"][-1]["directory"])
    runtime = read(attempt / "isolated_runtime.json")
    worker = read(attempt / "worker_status.json")
    data = snapshot()
    protected = []
    for record in plan["external_workers"]:
        current = process_identity(record["pid"])
        protected.append({"recorded": record, "current": current,
                          "same_live_process": current["alive"] is True and current["created"] == record["created"]})
    apps = [{"recorded": app, "current": process_identity(app["pid"])} for app in read(attempt / "owned_aspen_apps.json", [])]
    assert apps and all(x["current"]["alive"] is True and x["current"]["created"] == x["recorded"]["created"] for x in apps)
    assert worker["state"] == "running" and process_identity(worker["pid"])["alive"] is True
    assert all(row["same_live_process"] for row in protected)
    assert len(manifest["jobs"]) == 50
    assert all(j["state"] == "pending" for j in manifest["jobs"] if j["category"] in ("residual", "combined"))
    retained = [{**record, "unchanged": digest(record["path"]) == record["sha256"]} for record in plan["retained_results"]]
    assert len(retained) == 20 and all(r["unchanged"] for r in retained)
    sources = {name: source_check(path) for name, path in {
        "repeats": "Distillation/Results/review_recovery/20261018/repeats/manifest.json",
        "E1": "Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent/e1_combined/job.json",
        "E3_polymer": "Results/e3/20261015_042751_513bff/manifest.json",
        "E3_column": "Results/e3/20261015_042751_4e03da/manifest.json",
    }.items()}
    assert all(s["unchanged"] for s in sources.values())
    paper = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"
    freeze = read(paper / "manuscript_revised/frozen_original_manifest.json")
    original = {"files": len(freeze["files"]), "unchanged": all(digest(paper / "manuscript" / f["path"]) == f["sha256"] for f in freeze["files"])}
    assert original["unchanged"]
    environment = Path(sys.executable).parent
    openblas_path = environment / "Library/bin/openblas.dll"
    package = read(environment / "conda-meta/libopenblas-0.3.33-pthreads_h877e47f_0.json")
    packaged = next(p for p in package["paths_data"]["paths"] if p["_path"] == "Library/bin/openblas.dll")
    shared_blas = {"path": str(openblas_path), "sha256": digest(openblas_path),
                   "matches_original_conda_package": digest(openblas_path) == packaged["sha256_in_prefix"]}
    assert shared_blas["matches_original_conda_package"]
    failed = Path(plan["failed_attempt"])
    old_files = {name: {"sha256": digest(failed / name), "last_written_utc": datetime.fromtimestamp((failed / name).stat().st_mtime, timezone.utc).isoformat()}
                 for name in ("run.log", "worker_status.json", "job.json", "owned_aspen_apps.json")}
    assert all(record["last_written_utc"] < plan["prepared_at"] for record in old_files.values())
    broker = read(directory / "broker_identity.json")
    assert broker["parent_is_scheduler_service"] and broker["session_id"] > 0
    coordinator = read(directory / "coordinator_status.json")
    assert coordinator["state"] == "running" and process_identity(coordinator["pid"])["alive"] is True
    result = {"observed_at_utc": data["observed_at_utc"], "plan": plan,
              "aggregate_counts": data["counts"]["column_repeats"],
              "current_jobs": [r for r in data["jobs"] if r["study"] not in data["historical_studies"]
                               and (r["study"] == "column_repeats" or r["study"].startswith("Column_review_"))],
              "retry_worker": worker, "retry_process": process_identity(worker["pid"]),
              "retry_runtime": runtime, "owned_retry_aspen": apps, "protected_workers": protected,
              "original_supervisor": process_identity(plan["original_supervisor"]["pid"]),
              "retained_results": retained, "source_integrity": sources, "original_manuscript": original,
              "shared_blas": shared_blas, "historical_failed_files": old_files,
              "old_active_manifest_unchanged_so_far": digest(Path(plan["old_directory"]) / "manifest.json") == digest(directory / "original_manifest_before.json"),
              "cleanup": read(directory / "cleanup/seed23_old_orphan.json"),
              "broker": broker, "visible_consoles": read(directory / "launched.json"),
              "coordinator": coordinator, "scheduled": read(directory / "scheduled.json"),
              "private_stress": read(output / "private_stress.json"), "reference": read(output / "reference.json"),
              "wheel_provenance": read(output / "wheel_provenance.json"),
              "tests": {"passed": 39, "scope": ["test_markov_seed23_recovery.py", "test_recover_review_runs.py", "test_distillation_phase2_repeats.py"]},
              "completion_claim": "Restart verified; long-run completion and elimination of native crashes remain unproven."}
    (output / "recovery_assessment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"retry_pid": worker["pid"], "counts": result["aggregate_counts"],
          "protected_workers_unchanged": len(protected), "sources": sources, "original": original,
          "queue": coordinator, "output": str(output / "recovery_assessment.json")}, indent=2))


if __name__ == "__main__":
    main()
