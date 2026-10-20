"""Write the verified recovery overview from the saved assessment."""
from datetime import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    data = json.loads((HERE / "repair_assessment.json").read_text())
    when = datetime.fromisoformat(data["observed_at_utc"]).astimezone().strftime("%H:%M Toronto")
    workers = data["active_existing_jobs"]
    residual = [r["cycle"] for r in workers if r["job"].startswith("residual/") and r["cycle"] is not None]
    markov = next(r for r in workers if r["job"] == "markov/sg/23")
    threshold = data["threshold_recovery"]
    residual_progress = f"cycle {min(residual)}" if min(residual) == max(residual) else f"cycles {min(residual)}--{max(residual)}"
    progress = (f"At {when}: **23 production jobs remain: 13 running and 10 queued**. "
        f"The ten residual repetitions are at {residual_progress}/300. "
        f"Markov SG seed 23 attempt 4 is live (PID {markov['pid']}, "
        + (f"cycle {markov['cycle']}/300" if markov['cycle'] is not None else "first cycle in progress")
        + f"). Threshold recovery is at cycle {threshold['cycle']}/200, PID {threshold['pid']}; "
        f"its latest validated four-agent inference export is from cycle {threshold['periodic_policy']['completed_cycles']}. "
        f"Column BO has {data['column_BO_saved_calibration_trials']}/40 calibration trials saved; its 200-cycle final evaluation follows.")
    snapshot = ("The repaired native snapshot pilot passed the original 400-step prefix and three 24-step open-loop continuations. "
        "With short snapshot descriptions, TakeSnapshot and binary export now succeed. Repeated diagnosis established that the first Step after LoadSnapshot "
        "performs zero-time initialization with either optional-script setting. The new adapter completes that operation before applying a branch action "
        "and rejects any time advance or physical-profile change. The subsequent original plant steps advance by one sampling interval and reproduce "
        "controlled outputs/internal profiles within the saved tolerances. The COM time check allows its observed micro-hour quantization (1.1e-6 h); "
        "the controller interval and integrator settings are unchanged. This validates native plant continuation and observer copying. Full column E1 "
        "still needs integration of all learners, gates, action histories and MPC state plus its matched-rollout budget; the pilot is not critic evidence.")
    policy = ("No original usable threshold weights survived: all candidate exports were empty, and the valid historical trajectory contained no network "
        "state dictionaries. Only threshold was rerun, using the original seed 7, 200-cycle Scenario 1, calibration, warm-up, handoff and final freeze. "
        "The corrected registered-module exporter passed an 800-step technical smoke and saves finite four-agent weights at completed cycle 1, every "
        "ten cycles and at completion. These are inference exports, not optimizer/replay/plant checkpoints. The final trained policy is pending. The "
        "original trajectory and empty file remain byte-identical. The new trained weights will belong to this fresh trajectory, not the old run.")
    restart = ("Fresh process inventory found the former Markov a3 worker and its supervisor absent at cycle 112 without final records. Recent Windows "
        "application/resource-exhaustion events did not establish a cause; do not label this as another proven BLAS crash. Failed/stopped records remain "
        "preserved. Case-3 cleanup closed only that attempt's saved Aspen identity and verified Aspen-server/System32 console-host descendants, with "
        "creation-time validation on termination handles. Attempt 4 runs through its own independent interactive Windows scheduler broker with the "
        "unchanged private BLAS runtime and full 300-cycle budget. It restarts original initial conditions because no complete learner/plant checkpoint survived.")
    queue = ("The ten residual workers, BO worker and queue coordinator retain their original process identities. The combined group's scientific source, "
        "bootstrap, exact seed/mode assignments and model hashes verify. Ten combined jobs remain pending with no attempts and will start automatically "
        "after residual; SG seed 23 uses case 12, while Markov uses case 3. The new threshold/Markov brokers have scheduler-service parents, interactive "
        "sessions, zero triggers and no execution timeout. Every current Aspen application is accounted for. No shared package, DLL, frozen scientific "
        "source or completed trajectory was changed. All 46 original manuscript files verify unchanged.")
    files = ("Inspected: active repeat/group manifests and source/model hashes; BO job/trials; worker/ownership/process records; the stopped Markov attempt; "
        "threshold candidate exports, trajectory keys and configuration; installed Aspen V14 type-library metadata and local DynamicsHtmlHelp.exe "
        "automation reference (Step, Run, snapshot/rewind and time settings); bounded diagnosis records; full pilot artifacts; and the frozen-original manifest. "
        "Changed: three isolated recovery wrappers, tests/test_state_recovery.py, this report and audit scripts/evidence, RUN_RECOVERY_STATUS.md, and the "
        "affected paper workflow records. The unrelated existing run_polymer_phase2_repeats.py edit is preserved.")
    text = ("# Snapshot and threshold recovery — October 20, 2026\n\n" + progress + "\n\n## Snapshot repair\n\n" + snapshot
        + "\n\n## Threshold policy recovery\n\n" + policy + "\n\n## Stopped Markov attempt\n\n" + restart
        + "\n\n## Queue and preservation\n\n" + queue
        + "\n\n## Verification and files\n\n21 focused tests passed. The 800-step threshold technical smoke completed with all four final network state dictionaries "
          "nonempty, finite and disk-round-trip verified. The full native pilot's snapshot hash, three continuation arrays, advancing clock and four zero-time "
          "restore initializations verify. A prior full pilot completed its clock checks but hit a NumPy tolerance-formatting error before comparison; "
          "that harness error is fixed with a regression test, and each raw continuation now exports before validation. Earlier failed diagnostic "
          "attempts remain saved. Launch health is a current observation; long production completion remains pending.\n\n" + files
        + "\n\nEvidence: repair_assessment.json and preflight.json. Raw diagnostics/pilot are under Distillation/Results/state_recovery; threshold recovery is under "
          "Polymer/Results/state_recovery/20261020_threshold_policy_recovery. Use the visible active-run CMD windows and existing queue viewer for progress. "
          "In VS Code, open a run.log and press Ctrl+End to inspect the latest cycle; reading it does not control a worker.\n")
    (HERE / "RECOVERY.md").write_text(text, encoding="utf-8")
    overview = ROOT / "RUN_RECOVERY_STATUS.md"
    heading, _, rest = overview.read_text(encoding="utf-8").partition("\n")
    marker = "## Snapshot/threshold repair and stopped-run recovery — October 20"
    if marker in rest:
        before,after = rest.split(marker,1)
        following = after.find("\n## ")
        rest = before + (after[following:] if following >= 0 else "")
    block = (marker + "\n\n" + progress + "\n\nThe native snapshot continuation pilot passed; full column E1 adapter integration remains separate. "
        "Threshold's original empty export is preserved and its fresh policy-recovery run is underway with validated periodic exports. "
        "The former Markov attempt stopped at cycle 112 for an unconfirmed reason; attempt 4 has its own independent visible session. "
        "All eleven other production workers and the residual/combined coordinator retain their identities; combined remains automatic. "
        "Details: change-reports/state_recovery_20261020/RECOVERY.md and repair_assessment.json.\n\n")
    overview.write_text(heading + "\n\n" + block + rest.lstrip("\n"), encoding="utf-8")
    print(progress)


if __name__ == "__main__":
    main()
