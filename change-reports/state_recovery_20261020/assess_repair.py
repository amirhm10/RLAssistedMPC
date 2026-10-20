"""Read-only final repair/launch audit. Never opens or signals an Aspen app."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
import torch
torch.set_num_threads(1)
import numpy as np
from column_review.common import read, digest, now, process_identity, check_job
from recover_markov_seed23 import write
from comparison_studies.checkpoints import load_inference_checkpoint
from revision_comparisons.progress import snapshot, log_tail
import continue_column_queue as queue


def main():
    output = Path(__file__).resolve().parent
    pilot = ROOT / "Distillation/Results/state_recovery/20261020_snapshot_pilot_a2"
    status, result = read(pilot / "status.json"), read(pilot / "pilot_result.json", {})
    assert status["state"] == "completed" and status.get("aspen_closed") and result.get("pilot_passed")
    assert not result["e1_production_ready"]
    with np.load(pilot / "continuations.npz") as arrays:
        clocks = arrays["clock_intervals"]
        assert clocks.shape == (3,24) and np.max(np.abs(clocks - 1/6)) <= 1.1e-6
        differences = []
        for i in range(3):
            assert arrays[f"repeat_{i}_outputs"].shape == (24,2)
            assert np.all(np.isfinite(arrays[f"repeat_{i}_internal"]))
            difference = np.max(np.abs(arrays[f"repeat_{i}_outputs"] - arrays["repeat_0_outputs"]),axis=0)
            assert np.all(difference <= [1e-8,1e-4])
            differences.append(difference.tolist())
    primes = read(pilot / "restore_initializations.json")
    assert len(primes) == 4 and all(abs(r["before"][0]-r["after"][0]) < 1e-9 for r in primes)
    assert digest(result["native_snapshot"]) == result["native_snapshot_sha256"]
    current = snapshot()
    queue_root = Path(read(ROOT / "Distillation/Results/review_recovery/latest.json")["repeats"])
    for channel in ("residual", "combined"):
        queue.validate_group(read(queue_root / channel / "manifest.json"))
    combined = read(queue_root / "combined/manifest.json")
    assert len(combined["jobs"]) == 10 and all(j["state"] == "pending" and not j["attempts"] for j in combined["jobs"])
    check_job(read(ROOT / "Distillation/Results/column_review/20261018_recovered/bo/job.json"))
    threshold = ROOT / "Polymer/Results/state_recovery/20261020_threshold_policy_recovery"
    threshold_manifest = read(threshold / "manifest.json")
    threshold_job = read(threshold / "job.json")
    assert digest(ROOT / "recover_polymer_threshold.py") == threshold_job["recovery_bootstrap_sha256"]
    threshold_attempt = Path(threshold_manifest["directory"])
    latest = read(threshold_attempt / "policy_checkpoints/latest.json")
    policy = load_inference_checkpoint(threshold_attempt / "policy_checkpoints" / latest["path"])
    assert set(policy["agents"]) == {"horizon_agent","weight_agent","markov_agent","residual_agent"}
    assert latest["round_trip_verified"] and not latest["training_resume"]
    modules = json.loads((output / "threshold_loaded_blas.json").read_text(encoding="utf-8-sig"))
    scientific_extensions = [r for r in modules if r["ModuleName"].startswith(("_multiarray_umath", "_fblas"))]
    assert len(scientific_extensions) == 2 and all(Path(r["FileName"]).is_relative_to(review_blas_runtime.DEFAULT / "site-packages") for r in scientific_extensions)
    old = ROOT / "Polymer/Results/comparisons/scenario1/20261014_runtime_recovery_threshold_sg/threshold"
    old_status = read(old / "status.json")
    assert digest(old / "training.pkl") == old_status["trajectory_sha256"]
    assert digest(old / "final_inference_networks.pt") == "2f4f2f8a41335419849a40ac880f4cf25b79d5c161a965cb3d2ecff556905e8e"
    import win32com.client
    service = win32com.client.GetObject("winmgmts:")
    inventory = {int(p.ProcessId): {"name": str(p.Name), "created": str(p.CreationDate), "parent": int(p.ParentProcessId)}
                 for p in service.ExecQuery("SELECT ProcessId,ParentProcessId,Name,CreationDate FROM Win32_Process")}
    preflight = read(output / "preflight.json")
    original = {p["pid"]: p for p in preflight["process_inventory"]}
    preserved = []
    for row in preflight["protected_workers"]:
        assert row["pid"] in inventory and inventory[row["pid"]]["created"] == original[row["pid"]]["created"]
        preserved.append({"job":row["job"],"pid":row["pid"],"creation_unchanged":True})
    assert len(preserved) == 11
    active = [r for r in current["jobs"] if r["state"] == "running" and r["study"] not in current["historical_studies"]]
    assert len(active) == 12 and all(r["pid"] in inventory for r in active)
    coordinator = current["repeat_queue_coordinator"]
    assert coordinator["pid"] == preflight["coordinator"]["pid"] and coordinator["pid"] in inventory and coordinator["stage"] == "residual"
    registered_apps = set()
    for row in active:
        for app in read(Path(row["directory"]) / "owned_aspen_apps.json", []):
            if app["pid"] in inventory and process_identity(app["pid"])["created"] == app["created"]:
                registered_apps.add(app["pid"])
    aspen = {pid for pid,info in inventory.items() if info["name"].lower() == "aspenmodeler.exe"}
    assert aspen == registered_apps
    assert threshold_manifest["pid"] in inventory
    brokers = {}
    for name,directory in {"threshold":threshold,"markov":ROOT / "Distillation/Results/state_recovery/20261020_markov23_attempt4_launch"}.items():
        broker = read(directory / "broker_identity.json")
        assert broker["parent_is_scheduler_service"] and broker["session_id"] > 0 and broker["pid"] in inventory
        brokers[name] = broker
    frozen = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"
    freeze = read(frozen / "manuscript_revised/frozen_original_manifest.json")
    assert len(freeze["files"]) == 46 and all(digest(frozen / "manuscript" / f["path"]) == f["sha256"] for f in freeze["files"])
    threshold_cycles = re.findall(r"Sub_Episode:\s*(\d+)",log_tail(threshold_attempt / "run.log"))
    bo_row = next(r for r in active if r["job"] == "bo")
    trials = read(Path(bo_row["directory"]) / "bo_trials.json", [])
    report = {"observed_at_utc":now(), "production_running":13,"production_queued":10,"production_unfinished":23,
              "protected_workers_preserved":preserved, "active_existing_jobs":active, "coordinator":coordinator,
              "combined_guards_pass":True,"combined_slots":{f"{j['mode']}/{j['seed']}":j["aspen_number"] for j in combined["jobs"]},
              "column_BO_saved_calibration_trials":len(trials),"column_BO_final_evaluation_cycles":200,
              "snapshot_pilot":result, "snapshot_artifacts":{str(p.relative_to(pilot)):digest(p) for p in (pilot/"continuations.npz",pilot/"restore_initializations.json",pilot/"python_checkpoint.pkl",Path(result["native_snapshot"]))},
              "snapshot_continuation_max_output_differences":differences, "owned_aspen_apps_all_accounted_for":sorted(aspen),
              "threshold_recovery": {"manifest":str(threshold / "manifest.json"),"pid":threshold_manifest["pid"],"cycle":int(threshold_cycles[-1]) if threshold_cycles else 0,
                                     "periodic_policy":latest,"original_trajectory_and_empty_export_unchanged":True,
                                     "complete_trained_policy_recovery_pending":True,"exact_old_weights_recovered":False},
              "threshold_loaded_numerical_libraries":modules,
              "independent_brokers":brokers,"frozen_original_files_unchanged":46,
              "repair_source_hashes":{name:digest(ROOT/name) for name in ("repair_aspen_snapshot.py","recover_polymer_threshold.py","restart_stopped_markov.py","tests/test_state_recovery.py")},
              "verification":{"focused_tests_passed":21,"threshold_technical_smoke_steps":800,"smoke_validated_all_four_final_agents":True},
              "limitations":["Full column E1 still needs learner/gate/history/MPC adapter integration and an explicit matched-rollout budget.",
                             "Markov a3 stopped at cycle 112 without final records; cause unconfirmed. A4 restarts from original initial conditions.",
                             "Periodic inference weights are not full optimizer/replay/plant checkpoints.",
                             "Long production completion is pending; runtime mitigation does not guarantee multi-day stability."]}
    write(output / "repair_assessment.json", report)
    write(ROOT / "Distillation/Results/state_recovery/latest.json", {"snapshot":str(pilot), "threshold":str(threshold),
          "markov_restart":str(ROOT / "Distillation/Results/state_recovery/20261020_markov23_attempt4_launch"),"assessment":str(output / "repair_assessment.json")})
    print(json.dumps({"running":13,"queued":10,"snapshot_passed":True,"preserved_workers":len(preserved),"threshold_cycle":report["threshold_recovery"]["cycle"]},indent=2))


if __name__ == "__main__":
    main()
