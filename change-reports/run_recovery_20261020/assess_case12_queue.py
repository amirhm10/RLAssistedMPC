"""Read-only launch audit: residual now, combined next, live Markov retained."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from column_review.common import digest, now, process_identity, read
from revision_comparisons.progress import snapshot
import continue_column_queue as queue


def main():
    directory = queue.DEFAULT
    plan = read(directory / "plan.json")
    group = read(directory / "residual/manifest.json")
    queue.validate_group(group)
    protected = [{"recorded": p, "current": process_identity(p["pid"])} for p in plan["protected_workers"]]
    assert all(p["current"]["alive"] is True and p["current"]["created"] == p["recorded"]["created"] for p in protected)
    workers = []
    for job in group["jobs"]:
        attempt = Path(job["attempts"][-1]["directory"])
        runtime = read(attempt / "isolated_runtime.json")
        status = read(attempt / "worker_status.json")
        current = process_identity(status["pid"])
        owned = [{"recorded": app, "current": process_identity(app["pid"])} for app in read(attempt / "owned_aspen_apps.json", [])]
        assert current["alive"] is True and status["state"] == "running"
        assert owned and all(p["current"]["alive"] is True and p["current"]["created"] == p["recorded"]["created"] for p in owned)
        assert all(p["architecture"] == "Nehalem" and p["num_threads"] == 1 for p in runtime["private_blas"])
        assert runtime["runtime_manifest_sha256"] == group["runtime_manifest_sha256"]
        log = (attempt / "run.log").read_text(encoding="utf-8", errors="replace")
        assert "Traceback (most recent call last)" not in log and "Windows fatal exception" not in log
        workers.append({"job": list(queue.recovery.key(job)), "aspen_number": job["aspen_number"],
                        "directory": str(attempt), "status": status, "process": current,
                        "owned_aspen": owned, "runtime": runtime, "startup_error": False})
    assert len(workers) == 10 and len({j["aspen_number"] for j in workers}) == 10
    assert next(j for j in workers if j["job"] == ["residual", "sg", 23])["aspen_number"] == 12
    assert all(j["aspen_number"] != 3 for j in workers)
    retained = [{**r, "unchanged": digest(r["path"]) == r["sha256"]} for r in plan["retained_results"]]
    assert len(retained) == 29 and all(r["unchanged"] for r in retained)
    sources = {}
    for name, path in {
        "repeats": "Distillation/Results/review_recovery/20261018/repeats/manifest.json",
        "E1": "Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent/e1_combined/job.json",
        "E3_polymer": "Results/e3/20261015_042751_513bff/manifest.json",
        "E3_column": "Results/e3/20261015_042751_4e03da/manifest.json",
    }.items():
        expected = read(ROOT / path)["source_hashes"]
        changed = [p for p, sha in expected.items() if digest(ROOT / p) != sha]
        sources[name] = {"files": len(expected), "changed": changed, "unchanged": not changed}
    assert all(s["unchanged"] for s in sources.values())
    paper = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"
    freeze = read(paper / "manuscript_revised/frozen_original_manifest.json")
    original = {"files": len(freeze["files"]), "unchanged": all(digest(paper / "manuscript" / f["path"]) == f["sha256"] for f in freeze["files"])}
    assert original["unchanged"]
    broker = read(directory / "broker_identity.json")
    assert broker["parent_is_scheduler_service"] and broker["session_id"] > 0
    data = snapshot()
    counts = data["counts"]["column_repeats"]
    assert counts == {"completed": 29, "running": 11, "pending": 10}
    status = read(directory / "coordinator_status.json")
    assert status["state"] == "running" and status["stage"] == "residual" and process_identity(status["pid"])["alive"] is True
    result = {"observed_at_utc": now(), "plan": plan, "counts": counts, "workers": workers,
              "protected_workers": protected, "retained_results": retained, "source_integrity": sources,
              "original_manuscript": original, "coordinator": status, "broker": broker,
              "visible_consoles": read(directory / "launched.json"), "scheduled": read(directory / "scheduled.json"),
              "markov_retry": next(r for r in data["jobs"] if r["study"] == "column_repeats" and r["job"] == "markov/sg/23"),
              "combined_queued": read(directory / "combined/manifest.json"),
              "tests": {"passed": 49, "additional": "Case-3 lease integration rerun passed after using the production lock directory."},
              "scope": "Next existing channel started; active retry preserved; long-run completion remains pending."}
    # Do not duplicate scientific source inventories in the compact copied record.
    result["combined_queued"] = {"state": result["combined_queued"]["state"],
                                 "jobs": [{k: j[k] for k in ("category", "mode", "seed", "aspen_number", "state")} for j in result["combined_queued"]["jobs"]]}
    destination = Path(__file__).with_name("case12_queue_assessment.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "workers": [{"job": w["job"], "pid": w["process"]["pid"], "aspen": w["aspen_number"]} for w in workers],
                      "protected": protected, "coordinator": status, "evidence": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
