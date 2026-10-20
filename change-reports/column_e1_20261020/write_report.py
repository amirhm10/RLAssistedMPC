"""Publish only the already-verified E1 integration and launch audit."""
import json
from pathlib import Path

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[1]

def main():
    data=json.loads((DIRECTORY/'launch_assessment.json').read_text(encoding='utf-8'))
    assert data['technical']['passed'] and data['production']['state']=='running'
    prod,tech=data['production'],data['technical']
    tests=data['verification']['focused_tests_passed']
    report=f'''# Column E1 integration and launch — October 20, 2026

At {data['observed_at_utc']}, combined E1 seed 7 is running on source case 16, PID {prod['pid']}, cycle {prod['cycle']}/300. Four standalone E1 configurations follow automatically on the same source-case lease. All 13 pre-existing workers retain their original PID/creation identities; the independent residual/combined coordinator is live. Combined's sources, bootstraps, model hashes and reserved cases 1,2,4–10,12 verify unchanged.

The five-configuration E1 budget is the existing one-seed protocol: original 300-cycle temperature-flip profile, 100 sampled decisions in each of four windows per channel, 200 primary rollout steps and 10% extended to 800. This produces 3200 expected matched pairs and approximately 2,264,000 total Aspen advances including main training. Production completion and scientific interpretation remain pending.

The integration uses a second private Aspen application under the case-16 lease for counterfactuals. Python observer/MPC state, conditioning, action/gate histories and frozen inference agents are cloned using the existing E1 runner adapter. Training networks, replay, optimizer/n-step state and random streams retain their normal main-trajectory updates. Diagnostic branches never load or initialize the training application's native state. Exact live-profile and training-state hash checks pass for all four release-point gate channels.

Three native defects were isolated. ExportasBinary creates a result that another application cannot rewind. A saved initialized private document plus its snapshot database preserves rewindable rows. Reopened documents can retain locks through old COM references, so each transfer uses a fresh private working directory. LoadSnapshot also leaves fixed feed/actuator specifications unchanged; the adapter reapplies the captured native feed, reflux and reboiler specifications before checking the restored state. Original desktop model files and running-job dependencies are preserved.

Aspen performs zero-time algebraic reinitialization after loading a snapshot. Fidelity is checked separately from repeated-rollout reproducibility: controlled composition tolerance 1e-6, temperature tolerance 0.002 in native units (1% of the minimum tracking band), internal composition tolerance 0.0001, and no time advance. This is a numerical restoration contract, not a bitwise integrator-state guarantee. The largest measured controlled-output initialization differences in the four-channel check are {tech['maximum_reinitialization_output_difference']}. Both policy and supervisor branches repeat within the tighter saved-trace tolerances. Counterfactual failures remain explicit diagnostic records; restoration-integrity failures stop E1 and retain evidence.

Full paired native/Python checkpoints contain four learners, real replay, optimizers, targets, n-step queues, histories, controller state, sampling state and all random streams. The technical checkpoint resumed from step 80 in a fresh process and advanced to step 88. Complete Python training-state hashes match on restore. Production's first full checkpoint is cycle {prod['checkpoint']['completed_cycles']}; files rotate between two Python slots. Checkpoints resume inside the original attempt using a fresh native working session. These files are trusted local experiment artifacts.

Verification: {tests} focused tests; four live matched decisions with eight repeated branch checks; fresh-process full-state restart; native project transfer; current worker/ownership/queue auditing; all 46 original manuscript hashes unchanged. The private pinned NumPy/SciPy runtime is selected before scientific imports. The independent Windows scheduler broker has no execution time limit. The E1 run window is visible; the global progress viewer also includes E1 and threshold-policy recovery. Closing the read-only viewer does not stop studies.

Files inspected: both AGENTS.md instructions and revision instructions; research/Aspen/testing/software workflow skills; run_polymer_e1.py; review_experiments/e1 runtime/study/snapshots/prediction/smoke modules; revision_recovery/checkpoints.py; revision_comparisons/column.py and progress.py; column_review/common.py and pilot.py; review_experiments/e3 runtime/common modules; five original column entrypoints and their runner contracts; systems/distillation plant/config/scenario/default files; repair_aspen_snapshot.py; recover_polymer_threshold.py; original queue/recovery manifests; installed Aspen automation help; current paper-memory and freeze records.

Files changed: run_column_e1.py; column_e1 package; tests/test_column_e1.py; revision_comparisons/progress.py; RUN_RECOVERY_STATUS.md; this report and scoped audit/verification scripts and JSON/XML evidence in this directory. Unrelated run_polymer_phase2_repeats.py edits are preserved. No shared scientific runner, original model, old result bundle or manuscript source is edited.

Evidence: launch_assessment.json, technical_assessment.json, tests.xml, preflight.json, python_checkpoint_inspection.json and saved technical checkpoint/ownership records. Raw attempts, native projects and traces remain under Distillation/Results/column_e1. Failed technical probes are retained separately; they are not production performance evidence. A private runtime and validated checkpoints mitigate prior failures, while multi-day completion remains pending.
'''
    (DIRECTORY/'INTEGRATION.md').write_text(report,encoding='utf-8')
    path=ROOT/'RUN_RECOVERY_STATUS.md'
    heading,sep,body=path.read_text(encoding='utf-8').partition('\n')
    marker='## Column E1 integrated and started — October 20'
    assert marker not in body
    block=(marker+'\n\n'+f'At {data["observed_at_utc"]}: column E1 combined seed 7 is live on source case 16 (PID {prod["pid"]}, cycle {prod["cycle"]}/300), '
           f'with four standalone configurations queued. The {tests} focused tests, four-channel native/learner checks and fresh-process full-state restart pass. '
           'A second private case-16 application handles rollouts; the training application remains untouched by restores. Full native/Python checkpoints are enabled.\n\n'
           'All 13 existing worker identities and the automatic combined queue are preserved. The ten combined repetitions still reserve cases 1,2,4–10,12. '
           'Including the five newly authorized E1 configurations, 28 production jobs remain: 14 running and 14 queued. Production E1 results are pending. '
           'Details: change-reports/column_e1_20261020/INTEGRATION.md and launch_assessment.json.\n\n')
    path.write_text(heading+sep+'\n'+block+body.lstrip('\n'),encoding='utf-8')

if __name__=='__main__':main()
