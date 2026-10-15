# Prepare the missing polymer E1/E2 runs

Amir requested repair and preparation before starting the remaining studies.
This task changes added wrappers and exports only. No full production study,
Aspen session, original controller/agent/plant/defaults, or manuscript TeX is
started or changed by this preparation.

## Defects and repairs

The E1 residual and E2 SG simulations reached their final cycles. Their final
source checks failed because the old E1 fingerprint recursively included every
review experiment. Seven new, unused E3 files expanded that inventory. Every
previously hashed dependency still matched at the evidence intake. The residual
400 matched decisions survive; its final training trajectory and SG's final
trajectory were never written. Residual snapshots are frozen inference states
without optimizer/replay, not exact training-resume checkpoints. SG's episode
log cannot reconstruct missing physical trajectories or final weights.

`run_polymer_e1.source_hashes` now includes E1 and shared dependencies while
excluding unrelated E3 additions. Edits, removals and additions inside the
guarded implementation still fail integrity validation. E2 uses that same
scope plus its comparison modules. The workers save completed raw trajectories
before final source/checkpoint validation. A failed status remains failed and
is not automatically reusable as validated evidence.

The old comparison checkpoint exporter inspected `vars(agent)`. DQN/TD3 agents
are PyTorch modules whose registered child networks are stored in the module
registry, so both old combined exports contained only four empty dictionaries.
`comparison_studies/checkpoints.py` exports complete CPU copies of each agent's
`state_dict`, records source/configuration metadata, checks nonempty finite
tensors, and compares disk-loaded values exactly before finalizing the file.
The worker requires all four combined agents and records checkpoint hashes and
tensor counts. Export does not change live networks or random streams. These
files contain inference network weights and metadata, not complete training or
plant-state checkpoints. Construct matching agents and configure evaluation
state/eligibility explicitly when using them in future frozen-policy tests.

## Recovery scope and commands

`run_polymer_review_recovery.py` first validates original source provenance and
retained decision/trajectory inventories. The only permitted historical source
differences are the three documented wrapper files; changes to original
scientific dependencies are rejected. Actual new workers receive fresh, strict
fingerprints. Old manifests, failed statuses and artifacts are untouched.

The default queue is sequential, one seed (7), one CPU math thread per worker:

1. Combined E1: original 300-cycle Scenario 2 profile, 100 decisions for each
   channel/window, four windows and four channels, 1,600 new matched decisions.
   Rollouts 200 steps, 10 percent extended to 800, reports at 20/80/200 and the
   long subset, one snapshot per channel/window, gated frozen continuation.
2. Combined critic-gated E2: original 200-cycle Scenario 1 profile, 160,000
   control steps, 10 warm-up cycles, original three-cycle critic-only hand-off,
   policy release at cycle 14, original final evaluation at cycle 200.

The 1,600 existing standalone E1 pairs and completed LS/ARX/BO/threshold
trajectories are retained. Residual is not rerun merely to recreate its missing
main export. Threshold's old empty policy file remains unusable, but its full
trajectory remains valid; no threshold retraining is scheduled. ARX's severe
deterioration remains a scientific diagnosis task rather than an overwritten
result. Distillation Phase 2 and E3 jobs are not relaunched or reconfigured.

In CMD from the scientific repository:

```cmd
python run_polymer_review_recovery.py --dry-run
run_remaining_polymer_reviews.cmd
```

The CMD file activates `rl-assisted-env`, runs the queue and keeps the window
open at the end. The recovery launcher follows worker logs live, stops after a
failure, refuses an existing output directory, and verifies saved artifacts
after exit zero. New outputs go to `Polymer/Results/review_recovery/<run>/`, with
a `plan.json` retaining old evidence locations and separate child directories
`e1_combined/` and `e2_sg/`. E1 adds its normal timestamp directory. Nothing is
copied over historical runs. After a partial recovery, `--stages e2_sg` can
select only SG. Do not resume the old E1 manifest after changing its wrapper
hash; the new launcher preserves the old manifest and prepares a fresh attempt.

## Verification and traceability

The preparation record is `change-reports/review_recovery_20261015/`:

- `wrapper_hashes_before.json`: the three pre-fix wrapper hashes.
- `running_study_sources_before.json`: source inventories of both E3 studies
  and the distillation repeat campaign before editing.
- `production_plan.json`: read-only production plan and retained-data audit.
- `preparation_audit.json`: final preservation/technical-check status.
- `verify_preparation.py`: reproducible preservation/guard audit.

38 targeted E1/comparator/recovery checks passed. Three additional recovery
tests verify stopping before the next stage on failure, refusing an existing
output directory, and rejecting incomplete matched decisions despite exit
zero. Expensive unchanged full-training parity tests are excluded; original
implementation hashes and existing configuration contracts are checked.

Short end-to-end checks run the original combined architectures under explicitly
compressed technical clocks, sequential E1 then SG. They exercise actual
trajectory/checkpoint exports; results are recorded in the preparation audit.
They are excluded from scientific performance evidence. Their generated
trajectories, model files, figures and logs are not committed.

The initial snapshot has 115 Python files, including the E1 wrapper deliberately
repaired here. The other 114 scientific Python files and all 46 frozen manuscript
files remain unchanged. Both E3 source inventories (103 files each) and the
distillation repeat inventory (86 files) remain byte-identical to their running
manifests. The pre-existing dirty `run_polymer_phase2_repeats.py` is excluded from
this task and commit.

## Files inspected and changed

Inspected the two wrappers, comparison setup/native/threshold code, E1 runtime/
study/snapshot/analysis code, DQN/TD3 registered-module definitions, relevant
tests, old jobs/statuses/capture inventories, runtime records and source hashes,
E3/distillation guard inventories, and frozen manuscript manifest. No original
scientific implementation or raw historical output was edited.

Changed `run_polymer_e1.py`, `run_polymer_comparisons.py`,
`comparison_studies/common.py`, `E1_POLYMER_STUDY.md`, and
`POLYMER_SCENARIO1_COMPARISONS.md`. Added the checkpoint helper, recovery launcher,
CMD entrypoint, regression tests, this report and small preparation records.
Paper workflow records document preparation rather than claiming new results.
