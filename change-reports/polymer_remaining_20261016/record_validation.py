"""Record the launched batch and verified recovery artifacts without simulations."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_polymer_remaining import own_sources, read, digest, validate_output
from revision_comparisons.progress import snapshot


def main():
    batch = ROOT / "Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent"
    checks = ROOT / "Polymer/Results/review_recovery/20261016_checkpoint_validation_a"
    smoke = {}
    for stage in ("e1_combined", "e2_sg", "e2_arx_repaired"):
        path = checks / stage / "job.json"
        status = read(path.parent / "status.json")
        validate_output(path, status)
        assert status["state"] == "completed" and read(path)["smoke"]
        smoke[stage] = {"directory": str(path.parent), "status": status}
    parent = ROOT / "Polymer/Results/review_recovery/20261016_checkpoint_validation_b/e2_arx_repaired"
    assert read(parent / "execution.json")["state"] == "completed"
    assert "Resumed full training state after step 4" in (checks / "e1_combined/full_smoke.log").read_text(encoding="utf-8")
    live = {}
    for stage in ("e1_combined", "e2_sg", "e2_arx_repaired"):
        directory = batch / stage
        job = read(directory / "job.json")
        assert job["recovery_source_hashes"] == own_sources()
        assert all(digest(ROOT / relative) == expected for relative, expected in job["source_hashes"].items())
        runtime = read(directory / "isolated_runtime.json")
        assert runtime["numpy_x86_v3_enabled"] is False
        assert all(pool["architecture"].lower() == "nehalem" and pool["num_threads"] == 1
                   for pool in runtime["threadpools"] if pool["internal_api"] == "openblas")
        live[stage] = {"directory": str(directory), "status": read(directory / "status.json"),
                       "runtime": runtime, "recovery_dependencies_verified": True,
                       "scientific_dependencies_verified": True}
    checkpoint = read(batch / "e1_combined/training_checkpoints/latest.json")
    assert checkpoint["completed_cycles"] >= 1
    checkpoint_path = batch / "e1_combined/training_checkpoints" / checkpoint["file"]
    assert digest(checkpoint_path) == checkpoint["sha256"]
    plan = read(batch / "concurrent_plan.json")
    assert digest(Path(plan["old_E1_partial"]["directory"]) / "capture/decisions.jsonl") == plan["old_E1_partial"]["decisions_sha256"]
    audit = read(Path(__file__).parent / "source_and_run_audit.json")
    assert all(record["unchanged"] for record in audit["source_integrity"].values())
    assert audit["original_manuscript"]["unchanged"]
    result = {"production_batch": str(batch), "live_snapshot": snapshot(), "jobs": live,
              "E1_full_checkpoint": checkpoint, "retained_old_E1_partial": plan["old_E1_partial"],
              "technical_smokes": smoke,
              "parent_smoke": {"directory": str(parent), "state": "completed"},
              "tests": {"passed": 48, "seconds": 10.78,
                        "modules": ["test_revision_recovery", "test_polymer_review_recovery", "test_polymer_e3_recovery", "test_revision_comparisons"],
                        "record": "Local pytest output, exec session 74176, final chunk a2d9fe.",
                        "learning_resume": "Exact CPU trajectory, RNG, optimizer, replay and pending n-step equality; real CUDA E1 compressed run resumed and completed 16 pairs."},
              "audit": {"path": str(Path(__file__).parent / "source_and_run_audit.json"),
                        "sha256": digest(Path(__file__).parent / "source_and_run_audit.json")},
              "new_distillation_studies_started": False,
              "original_manuscript": audit["original_manuscript"]}
    output = Path(__file__).parent / "validation.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
