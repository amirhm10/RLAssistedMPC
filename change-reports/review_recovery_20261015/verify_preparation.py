"""Read-only preservation/guard audit; writes derived preparation records only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from run_polymer_e1 import source_hashes
    from run_polymer_review_recovery import build_plan, REPAIRED_WRAPPERS
    original = read(ROOT / "change-reports/2026-10-13_comparison_original_hashes.json")[str(ROOT)]
    python_files = {name: value for name, value in original.items() if name.endswith(".py")}
    changed_snapshot = [name for name, value in python_files.items() if sha(ROOT / name) != value]
    changed_original = [name for name in changed_snapshot if name not in REPAIRED_WRAPPERS]
    frozen = read(PAPER / "manuscript_revised/frozen_original_manifest.json")["files"]
    changed_manuscript = [item["path"] for item in frozen
                          if sha(PAPER / "manuscript" / item["path"]) != item["sha256"]]
    running = []
    for entry in read(HERE / "running_study_sources_before.json"):
        expected = entry["source_hashes"]
        changed = [name for name, value in expected.items() if sha(ROOT / name) != value]
        running.append({"manifest": entry["manifest"], "checked_files": len(expected), "changed": changed})
    before = read(HERE / "wrapper_hashes_before.json")
    plan = build_plan(sys.executable)
    (HERE / "production_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    smoke_exits = {name: read(HERE / "smoke_suite" / f"{name}_exit.json")
                   for name in ("e1_combined", "e2_sg")
                   if (HERE / "smoke_suite" / f"{name}_exit.json").is_file()}
    report = {
        "initial_snapshot_python_files_checked": len(python_files),
        "documented_snapshot_wrapper_repairs": [name for name in changed_snapshot if name in REPAIRED_WRAPPERS],
        "original_scientific_python_files_checked": len(set(python_files) - REPAIRED_WRAPPERS),
        "original_scientific_python_changed": changed_original,
        "original_manuscript_files_checked": len(frozen), "original_manuscript_changed": changed_manuscript,
        "running_study_source_inventories": running,
        "wrapper_changes": {name: {"before": value, "after": sha(ROOT / name)} for name, value in before.items()},
        "e1_fingerprint_files": len(source_hashes()),
        "unrelated_e3_excluded_from_e1_fingerprint": not any(name.startswith("review_experiments/e3/") for name in source_hashes()),
        "production_stages": [stage["name"] for stage in plan["stages"]],
        "production_launched_by_this_preparation": False,
        "retained_matched_decisions": sum(item.get("decisions", 0) for item in plan["retained_evidence"].values()),
        "smoke_exits": smoke_exits,
        "both_smokes_complete": len(smoke_exits) == 2 and all(
            item["returncode"] == 0 and item.get("saved_artifacts_verified") for item in smoke_exits.values()),
    }
    (HERE / "preparation_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if changed_original or changed_manuscript or any(item["changed"] for item in running):
        raise SystemExit("Preservation or active-study source audit failed.")


if __name__ == "__main__":
    main()
