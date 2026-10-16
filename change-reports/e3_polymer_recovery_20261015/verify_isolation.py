"""Verify no original controllers, failed results or other study sources changed."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protected = []
    for row in read(HERE / "protected_other_study_sources.json"):
        changed = [name for name, digest in row["source_hashes"].items() if sha(ROOT / name) != digest]
        protected.append({"manifest": row["manifest"], "files_checked": len(row["source_hashes"]), "changed": changed})
    failed = ROOT / "Results/e3/20261015_042751_513bff/polymer_combined_s7/a1"
    sealed = read(HERE / "failed_attempt_hashes.json")
    failed_changes = [name for name, digest in sealed.items() if sha(failed / name) != digest]
    snapshot = read(ROOT / "change-reports/2026-10-13_comparison_original_hashes.json")[str(ROOT)]
    # The E1 wrapper was explicitly repaired in the preceding completed task.
    original = {name: digest for name, digest in snapshot.items() if name.endswith(".py") and name != "run_polymer_e1.py"}
    changed_original = [name for name, digest in original.items() if sha(ROOT / name) != digest]
    frozen = read(PAPER / "manuscript_revised/frozen_original_manifest.json")["files"]
    changed_manuscript = [row["path"] for row in frozen if sha(PAPER / "manuscript" / row["path"]) != row["sha256"]]
    study = ROOT / "Results/e3/20261015_042751_513bff"
    manifest = read(study / "manifest.json")
    attempt = next(job for job in manifest["jobs"] if job["category"] == "combined")["attempts"][-1]
    new = Path(attempt["directory"])
    status, runtime = read(new / "status.json"), read(new / "isolated_runtime.json")
    report = {"protected_other_studies": protected, "old_failed_files_checked": len(sealed),
        "old_failed_files_changed": failed_changes, "original_scientific_python_checked": len(original),
        "original_scientific_python_changed": changed_original, "frozen_manuscript_files_checked": len(frozen),
        "frozen_manuscript_changed": changed_manuscript, "new_attempt": str(new), "new_worker": status,
        "isolated_runtime": runtime,
        "bootstrap_file_matches_executed_hash": sha(ROOT / "run_polymer_e3_recovery.py") == runtime["bootstrap_sha256"],
        "nominal_matrix_not_retrained": {job["category"]: len(job["attempts"]) for job in manifest["jobs"] if job["category"] != "combined"}}
    (HERE / "isolation_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protected_sources_unchanged": all(not row["changed"] for row in protected),
        "failed_files_unchanged": not failed_changes, "original_controllers_unchanged": not changed_original,
        "manuscript_unchanged": not changed_manuscript, "new_worker_pid": status.get("pid"),
        "new_worker_state": status["state"], "runtime_bootstrap_matches": report["bootstrap_file_matches_executed_hash"]}))
    assert all(not row["changed"] for row in protected)
    assert not failed_changes and not changed_original and not changed_manuscript
    assert report["bootstrap_file_matches_executed_hash"]


if __name__ == "__main__": main()
