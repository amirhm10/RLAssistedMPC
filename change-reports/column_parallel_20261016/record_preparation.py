"""Verify preparation, tests and freezes; never open Aspen or launch a worker."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from column_review.common import MAPPING, read, digest, check_job


def main():
    directory = Path(__file__).parent
    batch = ROOT / "Distillation/Results/column_review/20261016_prepared_cases12_16"
    manifest, plan = read(batch / "manifest.json"), read(batch / "prepared_plan.json")
    assert manifest["state"] == "prepared" and len(manifest["jobs"]) == 5
    assert all(job["state"] == "pending" and not job["attempts"] for job in manifest["jobs"])
    assert not (batch / "visible_launches.json").exists()
    assert not list(batch.glob("*/execution.json"))
    assert not list(batch.glob("*/a*/job.json"))
    for method in MAPPING:
        check_job(read(batch / method / "job.json"))
    offline = read(directory / "offline_methods_a/summary.json")
    assert offline["state"] == "completed" and offline["live_aspen_opened"] is False
    assert offline["owned_fake_cases_created"] == offline["owned_fake_cases_closed"] == 12
    assert {row["method"] for row in offline["methods"]} == {"nominal", "ls", "bo", "threshold", "sg"}
    for row in offline["methods"]:
        if row["method"] in ("sg", "threshold"):
            assert row["checkpoint"]["round_trip_verified"] and len(row["checkpoint"]["tensor_counts"]) == 4
            assert all(row["checkpoint"]["tensor_counts"].values())
    tests = directory / "tests_final.log"
    assert "61 passed" in tests.read_text(encoding="utf-8")
    audit = read(directory / "active_run_integrity.json")
    assert all(value["unchanged"] for value in audit["source_integrity"].values())
    assert audit["original_manuscript"]["unchanged"]
    paper = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"
    phase_sources = [paper / "manuscript/sections/results.tex", paper / "manuscript/sections/supporting_information_content.tex",
                     ROOT / "utils/phase1_hidden_release.py", ROOT / "utils/combined_runner.py",
                     ROOT / "systems/distillation/notebook_params.py"]
    result = {"scope": "Development and offline checks only; live launch explicitly deferred by author",
              "prepared_batch": str(batch), "mapping": MAPPING, "state": "prepared",
              "new_workers_started": 0, "new_aspen_applications_opened": 0,
              "plan_sha256": digest(batch / "prepared_plan.json"),
              "manifest_sha256": digest(batch / "manifest.json"),
              "phase": {"paper_label": "handoff phase", "technical_meaning": "critic-only learning under supervisor/MPC control",
                        "critic_only_cycles": [11, 13], "policy_eligibility_cycle": 14,
                        "original_inclusive_boundaries_preserved": True,
                        "sources": [{"path": str(path), "sha256": digest(path)} for path in phase_sources]},
              "tests": {"passed": 61, "log": str(tests), "sha256": digest(tests)},
              "offline_integration": offline,
              "snapshot_api": plan["installed_snapshot_api"],
              "source_integrity": audit["source_integrity"], "original_manuscript": audit["original_manuscript"],
              "existing_run_counts": audit["counts"], "observed_at_utc": audit["observed_at_utc"],
              "live_E1_pilot_pending": True, "full_E1_critic_validation_implemented": False,
              "unseen_reference_evaluation_pending": True}
    output = directory / "validation.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
