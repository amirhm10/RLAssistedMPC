# Polymer E1--E3 continuation, 2026-10-16

Amir authorized fixing the failed E1 combined job and starting all unfinished,
nonrunning polymer E1--E3 work concurrently, with visible CMD progress. New
distillation comparisons remain for discussion. Existing column jobs continue.

## Started production jobs

Batch: `Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent`.

| Job | Original profile | Seed | Worker PID at launch |
|---|---|---:|---:|
| E1 combined | Scenario 2, 300 cycles, 800 steps per cycle | 7 | 21724 |
| E2 SG reference | Scenario 1, 200 cycles, 800 steps per cycle | 7 | 46452 |
| E2 regularized ARX | Scenario 1, 200 cycles, 800 steps per cycle | 7 | 31404 |

Three independent CMD sessions were opened at approximately 12:42 Toronto time.
The batch's `visible_launches.json` records CMD PIDs 26388, 31652 and 47408.
The Python interpreter is the existing `rl-assisted-env`; no package is changed.
Original references, disturbances, warm-up, critic-only cycles 11--13, policy
release in cycle 14, learning, replay and evaluation settings remain unchanged.
At initial verification, E1 saved a full checkpoint after cycle 1, SG passed its
800-step nominal-baseline audit and entered its main training, and ARX advanced.

## Failure and isolated mitigation

The superseded E1 combined process died at cycle 198 with a native Windows
access violation (0xC0000005) during the existing LS correction's prediction
error calculation, inside SciPy finite-difference/L-BFGS-B evaluation. Python
exception handling could not finish its status/export. The precise underlying
binary or hardware trigger is not established.

That process predated the E3 recovery bootstrap and did not inherit its startup
CPU-dispatch controls. The same isolated NumPy/OpenBLAS runtime that completed
polymer E3 now applies to each new worker: disable grouped NumPy X86_V3 plus
AVX2/FMA3 dispatch, use Nehalem OpenBLAS and one numerical thread. Workers
validate the actual loaded dispatch and save `isolated_runtime.json`.
No controller, optimizer formulation, installed package or active column source
is edited. This is a runtime mitigation, not proof of the native fault's cause.

The failed E1 attempt retains 793 completed pairs and all its snapshots/traces.
They are not combined with the fresh training trajectory. Its inference-only
snapshots cannot restore optimizers/replay, so the new E1 starts at cycle 1.

## Recovery and completion

`run_polymer_remaining.py` and `revision_recovery/` are opt-in infrastructure.
They invoke the existing E1, SG and revised-ARX workers. The E1 adapter compiles
the original runner privately, adds only completed-cycle checkpoint yields and
a restore entry point, and preserves the original statements/gate sites/order.

E1 checkpoints retain the full plant/observer/controller histories, live and
target networks, optimizers, replay arrays, pending transitions, agent counters,
RNG streams and matched-decision sampling/log state. They are written after
cycle 1, every tenth cycle, and cycle 300, in two atomic slots with checksums and
source guards. The first production checkpoint is about 786 MiB and took 1.5 s.
Reopening that job's saved `run.cmd` restores its latest complete checkpoint.
An E1 native access violation permits one automatic checkpoint recovery; Python
or source-validation errors do not trigger blind retries. A per-stage OS lock
prevents duplicate production copies while allowing the three stages to overlap.
Post-checkpoint suffix records and orphan files are archived before replay.

SG and ARX have separate parents and final exports. They are independent of
E1's completion and have no new periodic training-resume promise. Each parent's
completion checks require the saved trajectory hash and expected step count;
SG also requires four populated, round-trip-validated inference networks, and
E1 requires all 1600 pairs with no failed branch. Native exit codes cannot leave
a dead process represented as currently running by the new parent.

After E1 completes, the finalizer combines exactly 3200 unique pairs from the
four retained standalone captures and the new combined capture. After both E2
jobs complete, it assembles six-controller/five-window metrics using the existing
OF-MPC/LS/BO/threshold trajectories and the new SG/repaired-ARX exports. The
old drifting ARX remains a diagnosis artifact. No new superiority or statistical
claim is inferred from launching these single-seed studies.

## Existing completed/running work

Polymer Phase 2 repeats: 50/50 complete. Polymer E3: nominal, Markov and combined
complete; final analysis saved at `Results/e3/20261015_042751_513bff/analysis/20261016_112339_43cb78`.
The final E3 completion is 07:23 Toronto, October 16; its old console simply
shows the final lines. No polymer E3 is rerun.

Standalone E1 horizon/weights/Markov/residual each retain 400 valid pairs;
residual's old main trajectory is unavailable, but the matched pairs suffice
for this E1 analysis. Existing LS, BO and threshold E2 trajectories are retained.
Column repeats and column E3 continue under their existing parents. No new
Aspen document or distillation comparison is opened by this recovery.

## Verification

- 48 focused tests passed: recovery, prior export recovery, E3 runtime and
  revised comparisons. Exact CPU interrupted/uninterrupted learning parity
  includes optimizer state, replay, pending n-step data, helpers and RNG.
- Real CUDA E1 compressed test stopped after a durable checkpoint, resumed,
  completed all 300 compressed cycles and exported all 16 matched decisions.
- SG and ARX 800-step compressed checks completed with valid exports; the
  updated parent/lock/log/validation path also completed an ARX smoke.
- The first full production E1 checkpoint is saved and its hash verifies.
- E3 inventories (103 files each), E1 inventory (97), column repeat inventory
  (86), all completed E3 artifacts and the frozen 46 manuscript files verify.
- `polymer_remaining_20261016/validation.json` records paths, hashes, actual
  runtime metadata, launch/process state and the source/run audit. These are
  timestamped status/technical records, not completed production outcomes.

## Files inspected and changed

Inspected: original E1 compiler/session/worker, combined controller loop,
comparison workers/runtime/checkpoint/analysis code, E3 recovery bootstrap,
ARX repair, historical manifests/logs, active column inventories and manuscript
freeze. Original scientific sources and the user's dirty repeat wrapper are
untouched.

Added: `run_polymer_remaining.py/.cmd`, `revision_recovery/`,
`tests/test_revision_recovery.py`, this report and its bounded audit records.
Updated only `revision_comparisons/progress.py` to discover the concurrent
batch and show its checkpoint progress. Its old historical failure rows remain
visible for provenance; `Polymer_remaining_*` rows represent the new jobs.

For live progress use `show_review_progress.cmd`. To resume one interrupted
job, open its existing `run.cmd`; do not create a second full batch.
