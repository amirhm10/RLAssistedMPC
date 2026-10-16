# Prepared parallel column reviewer studies: cases 12--16

Amir authorized code preparation and the mapping below on October 16, 2026.
**Execution is deferred until Amir explicitly says to start.** No live Aspen
session or scientific run was started during this preparation.

| Case | Job | Protocol |
|---:|---|---|
| 12 | Combined SG reference | Seed 7, original Scenario 1, 200 cycles |
| 13 | Combined threshold-supervised RL | Same original protocol and networks |
| 14 | LS-only MPC | Original accepted LS correction, no RL learner |
| 15 | Bayesian weight-tuned MPC | 40 calibration trials, then 200 main cycles |
| 16 | E1 native-state restoration pilot | 400-step nominal prefix, three identical 24-step continuations |

Cases 1--11 and their existing workers are excluded from this launcher. All five
assigned `.dynf` files exist and have identical SHA-256 values. No model file is
rewritten. The original controller, identification, reward and training sources
remain unchanged. New study wrappers are under `column_review/` and
`run_distillation_remaining.py/.cmd`.

## What cycles 11--13 mean

The submitted paper's `sections/results.tex`, line 11, calls episodes 11--13 the
**handoff phase**, explicitly describing critic learning while MPC retains
control. Its Supporting Information release table likewise says critic learning
and action freeze. Thus handoff is the paper's label for a **critic-only
preparation window**, with TD3 actors and RL execution authority frozen.
Policy eligibility and actor learning begin in episode 14. Horizon DQN has a
Q network and no separate actor. New logs use this precise wording.

The code retains the original three-cycle action/actor freeze. For a 400-step
cycle, its inclusive native boundaries end at zero-based step 5200 and first
permit learned action selection at 5201, within cycle 14. No boundary is shifted
to make the label look cleaner. Warm-up is 1--10 and final evaluation is 200.

## Prepared location and commands

Batch, relative to this scientific repository:

`Distillation/Results/column_review/20261016_prepared_cases12_16`

It contains a manifest, a read-only protocol/API plan and five job specifications
with individual `run.cmd` scripts. Every job is pending, with zero attempts.

The default command displays a read-only plan. It does not open Aspen:

```cmd
call "C:\ProgramData\anaconda3\Scripts\activate.bat" rl-assisted-env
python run_distillation_remaining.py
```

**After authorization**, the launch command will open five independent visible
CMD sessions:

```cmd
run_distillation_remaining.cmd --launch --batch "Distillation\Results\column_review\20261016_prepared_cases12_16"
```

That launch command has not been executed. Do not open a job's saved `run.cmd`
until execution is authorized. The root CMD without arguments prints the plan.

The same launch command can later reuse verified completed jobs and start fresh
attempts for failed jobs, provided their original source/environment/model
provenance still matches. A method's training or BO calibration restarts at its
original initial condition; this workflow does not promise an exact full
training or BO-trial resume. Successful final inference weights are preserved
for later frozen-policy evaluations.

## Isolation, cleanup and progress

- Separate model files, private attempt outputs and private cleanup/export paths.
  Existing native snapshot caches are preserved. The E1 exported `.snp` is
  outside the directory used by normal cleanup.
- The canonical model lock is held across an entire worker, including gaps
  between all BO trial initializations; nested original contexts reuse that
  worker's lease. All five cases are checked before opening any new CMD.
- DispatchEx ownership records include both PID and process creation time.
  Cleanup verification cannot mistake a reused PID for an owned Aspen instance.
  Failed or untracked COM creation requires inspection before restart.
- Only the original owned objects close their documents. No unrelated process
  is terminated or controlled. A native failure leaves a failed attempt with
  ownership records, rather than a false running/completed status.
- Workers use the tested process-local NumPy/OpenBLAS runtime and one numerical
  thread. Actual dispatch is validated before simulation imports/training.
- Live worker output is tailed, with 30-second supervisor heartbeats. Quiet
  output is reported as quiet output, not proof of a hung solver. Periodic
  Python stacks are saved separately from the sealed scientific artifacts.
- Source/model guards, full artifact hashes, cycle/step counts and case cleanup
  must verify before a job is complete. One failure does not stop other jobs.
- The original four-comparison CSV is assembled only from verified completed
  jobs, with one consistently re-scored existing MPC baseline. The restoration
  pilot is excluded from that scientific comparison table.

## E1 pilot scope

The installed Aspen type library confirms `Simulation.Results.TakeSnapshot`,
`FindSnapshot`, `LoadSnapshot` and snapshot `ExportasBinary`/`Converged` support.
This was checked through registered metadata without creating or attaching to
an Aspen application.

The future live pilot will create its own case-16 application, run one original
nominal MPC cycle, export a converged native snapshot, restore native and Python
observer state, and repeat a bounded input/disturbance suffix three times. It
checks time, feed, native manipulated values, temperatures and ethane fractions
at trays 1, 12, 24, 40, 60 and 85, as well as both measured outputs. A step must
advance by the original sampling interval. Tolerances are fixed in the pilot
before execution: immediate temperature differences 1e-5 K and composition
differences 1e-9; repeated-continuation differences 1e-4 K and 1e-8 respectively.

This is a plant-restoration/observer-copy pilot, not E1 critic ranking evidence
and not a complete all-agent training checkpoint. Its success does not start
full E1 automatically. Integrating the native snapshot with all frozen learners,
normalization, gate/action histories and MPC state remains the next E1 step.
Unseen-reference policy evaluations likewise follow availability/audit of the
new final policies. The current five-job preparation does not silently add them.

## Verification and corrections before launch

61 focused tests pass, including shared recovery regressions, original freeze
boundaries, model identity, allocation/locking, PID reuse, missing exports,
snapshot/hidden-state checks and no-process/no-COM preparation. A compressed
offline integration check exercised all five underlying comparator methods with
original networks and entrypoints on an identified-model stand-in. All 12 owned
fake cases closed. None of these checks opened Aspen or produced performance
evidence for the physical column.

Two pre-launch omissions were corrected in isolated comparison infrastructure:
standalone SG now creates mandatory baseline-audit sidecars before training, and
SG/threshold exports retain common reward/tracking-band parameters for joint-band
metrics. Completion now explicitly requires 80,000 steps and 200 cycles. No
original controller or agent implementation was changed.

Detailed verification: `change-reports/column_parallel_20261016/validation.json`.
Engineering record: `change-reports/2026-10-16_column_parallel_preparation.md`.
`show_review_progress.cmd` discovers this prepared batch and later shows its
running/failed/completed jobs with Aspen case numbers.
