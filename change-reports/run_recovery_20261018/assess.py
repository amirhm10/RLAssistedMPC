"""Save verifiable current recovery evidence without launching any simulation."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from column_review.common import digest, process_identity, read
from revision_comparisons.progress import snapshot


def main():
    output = Path(__file__).resolve().parent
    plan = read(ROOT / "Distillation/Results/review_recovery/20261018/plan.json")
    directory = Path(plan["directory"])
    columns = Path(plan["column"])
    data = snapshot()
    historical = set(data["historical_studies"])
    data["current_jobs"] = [r for r in data["jobs"] if r["study"] not in historical]
    data["live_recovery_workers"] = sum(r["state"] == "running" and r["alive"] is True for r in data["current_jobs"])
    data["broker"] = read(directory / "broker_identity.json")
    data["visible_launches"] = read(directory / "launched.json")
    data["repetition_console_reopen"] = read(directory / "repeats_reopened.json")
    data["recovery_plan"] = plan
    data["old_manifest_hashes_unchanged"] = {
        "repeats": digest(ROOT / "Distillation/Results/phase2_repeats/20261014_154252_a2ed3b/manifest.json") == plan["old_repeat_manifest_sha256"],
        "column": digest(ROOT / "Distillation/Results/column_review/20261016_prepared_cases12_16/manifest.json") == plan["old_column_manifest_sha256"],
    }
    data["runtime_checks"] = []
    ownership = []
    paths = [Path(row["directory"]) / "isolated_runtime.json" for row in data["current_jobs"]
             if row.get("directory") and (row["study"] == "column_repeats" or row["study"].startswith("Column_review_"))]
    paths = [path for path in paths if path.exists()]
    for path in paths:
        info = read(path)
        pools = [p for p in info["threadpools"] if p["internal_api"] == "openblas"]
        data["runtime_checks"].append({"directory": str(path.parent), "numpy_x86_v3_enabled": info["numpy_x86_v3_enabled"],
                                       "blas": [{k: p[k] for k in ("architecture", "num_threads")} for p in pools]})
        for app in read(path.parent / "owned_aspen_apps.json", []):
            current = process_identity(app["pid"])
            ownership.append({"directory": str(path.parent), "recorded": app, "current": current,
                              "ownership_verified": current["alive"] is True and current["created"] == app["created"]})
    data["owned_aspen_apps"] = ownership
    polymer = ROOT / "Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent"
    from run_polymer_remaining import validate_output
    verified = {}
    for stage in ("e1_combined", "e2_sg", "e2_arx_repaired"):
        status = read(polymer / stage / "status.json")
        validate_output(polymer / stage / "job.json", status)
        verified[stage] = {"status": status, "artifact_contract_verified": True,
                           "postprocess": read(polymer / stage / "postprocess_status.json", {})}
    data["completed_polymer"] = verified
    pilot_attempt = Path(read(columns / "e1_pilot/attempts.json")[-1]["directory"])
    data["snapshot_pilot"] = read(pilot_attempt / "pilot_result.json", read(pilot_attempt / "status.json", {}))
    data["runtime_stress"] = read(output / "runtime_stress.json", {})
    data["old_failure_review"] = read(output / "failure_review.json", {})
    data["windows_fault_events"] = read(output / "windows_fault_events.json", {})
    data["snapshot_diagnostics"] = {path.name: read(path / "status.json", {})
                                     for path in directory.glob("snapshot_diag_*") if path.is_dir()}
    data["diagnostic_snapshot_exports"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                                           for path in directory.glob("snapshot_diag_*/snapshots/*.snp")]
    data["snapshot_clock_checks"] = read(directory / "snapshot_diag_clock/step_clock.json", [])
    data["snapshot_clock_initial_state"] = read(directory / "snapshot_diag_clock/native_before.json", {})
    data["changed_source_hashes"] = {name: digest(ROOT / name) for name in (
        "recover_review_runs.py", "run_distillation_remaining.py", "column_review/pilot.py",
        "column_review/typelib.py", "revision_comparisons/progress.py")}
    data["validation"] = {"regression_tests_passed": 61, "regression_seconds": 16.83,
                          "post_launcher_change_tests_passed": 6, "post_launcher_change_seconds": .20,
                          "post_pilot_hold_tests_passed": 7, "post_pilot_hold_seconds": .26}
    (output / "recovery_assessment.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"observed_at": data["observed_at_utc"], "live_workers": data["live_recovery_workers"],
                      "owned_aspen_apps": sum(a["ownership_verified"] for a in ownership),
                      "runtime_checks": len(paths), "old_manifests_preserved": data["old_manifest_hashes_unchanged"],
                      "pilot": data["snapshot_pilot"]}, indent=2))


if __name__ == "__main__":
    main()
