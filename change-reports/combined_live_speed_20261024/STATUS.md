# Combined live scheduling and remaining-time check

October 24, 2026, 23:52 Toronto. The user asked why 75% completed comparisons still leave 20–26 hours and requested acceleration without interruption or lost progress. Scope: inspect the current accepted run and apply only reversible operating-system scheduling changes. No training restart, new simulation, scientific-budget reduction or source change is authorized by this performance check.

## Why substantial work remains

The 75% figure is 240 of 320 selected matched comparisons. It does not measure overall wall time. All 80 release, 80 mature and 80 plant-change comparisons are complete. The final 80 belong to the adapted window later in the original 300-cycle training trajectory. Their native states and learned policies do not yet exist and cannot be evaluated early without changing the study.

The measured remaining-work model at step 86000 separates these costs:

| Remaining component | Work | Estimated hours |
| --- | --- | ---: |
| Advancing training | 34,000 control steps to step 120000 | 10.19 |
| Adapted-window matched comparisons | 80 decisions, including eight longer rollouts, already using two concurrent branches | 11.50 |
| Total | Original scientific budget preserved | 21.69 |

The 20–26-hour range is a planning range, not a confidence interval. Main training has since advanced to step 86800/cycle 217, leaving 33,200 steps. The earlier estimate remains suitable at this precision. Additional free Aspen cases cannot divide this single coupled four-learner trajectory while preserving its exact state/update sequence. The current synchronous adapter uses case15/17 for the two independent comparison branches. A larger asynchronous or distributed training change would need separate implementation, fidelity/restoration validation and a checkpoint migration; it cannot be injected as an uninterrupted scheduling change.

## Observations and live action

The 12-second before sample found approximately 4.76% mean CPU use across 24 permitted logical processors. All relevant processes already had access to every processor. Windows CPU-set records distinguish eight class-1 and sixteen class-0 processors. The CPU was using the Balanced power plan; no power setting was changed. These observations do not show general CPU contention or a restrictive affinity cap. Low aggregate CPU use alone does not identify every per-step bottleneck; the source shows an ordered Python controller/learner loop with synchronous native plant stepping.

All three accepted Python workers were below normal priority (16384). Aspen applications were already normal priority (32). With the other experiments complete, the user-authorized live adjustment restored NORMAL_PRIORITY_CLASS for precisely:

- parent 59504 / creation 134373496370785599;
- case15 worker 38188 / creation 134373499744270962;
- case17 worker 46276 / creation 134373499745265459.

The identity-bound plan and guarded normalize_priority.py check name, creation generation, current priority/affinity and accepted production state before changing anything. Readback verifies priority 32 for all three and unchanged affinity. The rollback option restores the recorded prior priority only for these same live generations and expected scheduling state; the plan must never be applied to replacement processes. No healthy worker was paused or stopped. The two retention helpers retain below-normal priority.

This removes unnecessary scheduling disadvantage; it is not a demonstrated large speedup. The short after sample measured 4.91% aggregate CPU use. Low contention means the benefit may be small. The next checkpoint interval was 474.15 seconds for 400 steps and straddled the adjustment and evolving controller choices, so it is not a controlled before/after throughput measurement. No shorter ETA is claimed from the priority change.

## Verification and preservation

The same parent and both child creation identities remain live. A subsequent complete checkpoint advanced 86400 to 86800, cycle216 to217. All four learner train-step counters advanced exactly400 and all four replay sizes remain40000. The scientific source map is identical. The normal-priority and unchanged-affinity readbacks pass. Standard post-change audit verifies every current native ownership, all143 sealed sources, completed/unique capture keys, and no observation requiring diagnosis. Other configurations and repetition/coordinator/horizon states remain completed.

Original retention remains live and independently verified86400/86000 at the full audit. During the later86800 write it deferred conservatively with every file retained. The new86800 Python/native checkpoint committed normally; no cleanup was forced. Side retention remains live with66 archived retired working folders, both newest primary working comparisons unpacked, and all results/checkpoints/inputs retained. The existing viewer and queue registration are unchanged. Final resource sample has35.8GiB commitment headroom and265.0GiB free disk.

Applied actions: three priority changes. Zero restarts, process signals, simulator calls, affinity changes, solver/numerical-thread changes, source/shared-queue edits, data deletions or administrator operations. Global packages, power plan, OneDrive, scientific results, recovery data, frozen analysis and manuscript/responses remain intact.

## Evidence and files

Inspected: current root/parallel status, immutable acceptance and ETA records, recover_parallel.py, session_adapter.py, systems/distillation/plant.py, column_e1/checkpoints.py, the original timing model, current stable manifest/index/ownership/retention aliases, append-only production log, actual process scheduling/CPU/affinity/memory and Windows CPU-set/power-plan queries.

Changed: this isolated folder (guarded scheduling helper, prior-state plan, before/after resource samples, applied/readback and continuation receipts, status), the new post-change hourly audit receipt, and concise root/hourly/parallel operational notes. The unrelated dirty run_polymer_phase2_repeats.py remains untouched.

Evidence: resource_before.json, priority_plan.json, applied.json, resource_after.json, checkpoint_continuation.json, final_state.json, and ../hourly_run_checks_20261024/audit_20261025T0350_priority.json. The helper parses and its real generation/priority/affinity readbacks pass; live400-step four-learner continuation is the functional verification. No simulation or training test was launched. No paper or reviewer comment was edited.
