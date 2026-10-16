"""Read-only startup capture for the five author-approved column jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from column_review.common import MAPPING, check_job, process_identity, read

BATCH = ROOT / "Distillation/Results/column_review/20261016_prepared_cases12_16"
DESTINATION = ROOT / "change-reports/column_parallel_launch_20261016/launch_assessment.json"
EXISTING_WORKERS = [43012, 47732, 46060, 32904, 34084, 43516, 15260, 40928,
                    48508, 49892, 16924]
EXISTING_ASPEN = [47544, 40408, 30076, 19364, 38988, 24740, 24996, 45080,
                  16548, 27364, 19208]
EXISTING_POLYMER = [21724, 46452, 31404]


def main():
    launches = read(BATCH / "visible_launches.json")
    jobs = {}
    for method, number in MAPPING.items():
        stage = BATCH / method
        job = read(stage / "job.json")
        check_job(job)
        execution = read(stage / "execution.json")
        attempt = Path(execution["attempt"])
        attempts = read(stage / "attempts.json")
        worker = process_identity(attempts[-1]["pid"])
        owned = []
        for entry in read(attempt / "owned_aspen_apps.json", []):
            current = process_identity(entry["pid"])
            owned.append({"recorded": entry, "current": current,
                          "same_process": current["created"] == entry["created"]})
        command = next(item for item in launches if item["method"] == method)
        runtime = read(attempt / "isolated_runtime.json")
        jobs[method] = {
            "aspen_number": number, "execution": execution,
            "supervisor": process_identity(execution["pid"]), "worker": worker,
            "cmd": process_identity(command["cmd_pid"]),
            "ownership": read(attempt / "ownership_status.json"), "owned_apps": owned,
            "runtime": runtime, "status": read(attempt / "status.json"),
            "log_tail": (attempt / "worker.log").read_text(encoding="utf-8", errors="replace").splitlines()[-8:],
            "prepared_sources_and_model_verified": True,
        }
        assert execution["state"] == "running", (method, execution)
        assert worker["alive"] and jobs[method]["supervisor"]["alive"] and jobs[method]["cmd"]["alive"]
        assert owned and any(entry["same_process"] and entry["current"]["alive"] for entry in owned), method
        assert runtime["python"] == sys.executable
        assert runtime["numpy_x86_v3_enabled"] is False
    existing = {"cases_1_to_11_workers": [process_identity(pid) for pid in EXISTING_WORKERS],
                "cases_1_to_11_aspen": [process_identity(pid) for pid in EXISTING_ASPEN],
                "polymer_workers": [process_identity(pid) for pid in EXISTING_POLYMER]}
    assert all(item["alive"] for values in existing.values() for item in values)
    audit = read(DESTINATION.with_name("source_and_study_audit.json"))
    assert audit["original_manuscript"]["unchanged"]
    assert all(item["unchanged"] for item in audit["source_integrity"].values())
    result = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User explicitly authorized all five simultaneous visible CMD runs on cases 12--16 using rl-assisted-env; preserve cases 1--11.",
        "code_commit_at_launch": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "batch": str(BATCH), "jobs": jobs, "existing_processes": existing,
        "source_integrity": audit["source_integrity"], "original_manuscript": audit["original_manuscript"],
        "interpretation": "Verified startup only. Four E2 comparisons and one native restoration pilot; no completed performance or full E1 critic-validation claim.",
    }
    with DESTINATION.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(DESTINATION), "workers": {key: value["worker"]["pid"] for key, value in jobs.items()},
                      "existing_case_workers_alive": len(existing["cases_1_to_11_workers"])}, indent=2))


if __name__ == "__main__":
    main()
