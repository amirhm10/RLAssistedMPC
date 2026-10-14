# Polymer run intake and native-crash recovery

Author request, 2026-10-14: inspect the completed repeat/E1/E2 campaigns, then repair the failures and resume unfinished work without changing the scientific settings.

## Verified failure evidence

The 50 Scenario 2 repetitions completed. E1 and the Scenario 1 comparator batch stopped early.

| Worker | Last saved progress | Windows Application Error, Toronto time | Faulting module | Error |
| --- | --- | --- | --- | --- |
| E1 standalone dynamic matrix, PID 24244 | 196 logged training cycles; 172 matched decisions; no final training bundle | 2026-10-14 08:23:01 | NumPy `_multiarray_umath.cp313-win_amd64.pyd` | `0xc0000005` |
| E2 combined threshold, PID 44336 | 174 logged cycles; no final training bundle | 2026-10-14 09:18:06 | `openblas.dll` | `0xc0000005` |

Windows event 1000 identifies both processes and modules. These are native access violations; the worker cannot execute its Python exception handler, so its initial `running` status remains stale. The E1 parent records failure and leaves residual/combined pending. The E2 parent records threshold failure and never starts its matched SG reference. The exact instruction-level trigger or upstream defect has not been established. Fault location is evidence for repairing the numerical runtime, not proof of a particular NumPy/OpenBLAS bug or memory-exhaustion mechanism. No matching Windows resource-exhaustion event was found in the inspected interval.

## Runtime repair

The Conda solver was first inspected with `--dry-run`. Only six numerical packages were replaced in `rl-assisted-env`: NumPy 2.5.4 to 2.4.6, SciPy 1.18.1 to 1.17.1, OpenBLAS 0.3.34 to 0.3.33, and the matching libblas/libcblas/liblapack builds from build 11 to build 8. Python remains 3.13.2 and PyTorch remains 2.13.0+cu130. The readable environment recipe and Windows lock were updated; SciPy is declared as its actual Conda dependency in the lock rather than as a pip override.

Before/after package records, explicit rollback specifications, the Conda transaction, and actual imported runtime/thread pools are saved in `change-reports/runtime_recovery_20261014/`. All original training, reward, bounds, margins, networks, seed, warm-up, handoff, operating-change and evaluation settings are retained. The existing one-thread-per-worker setting is retained. `PYTHONFAULTHANDLER=1` is inherited by both resumed runners and their workers to capture Python stack diagnostics if another native crash occurs.

Changing numerical libraries can change floating-point trajectories. Record the recovery runtime alongside the original-run runtime; do not imply identical hardware/library provenance or guaranteed full-duration crash freedom.

## Verification

- 35 E1 preservation and comparator tests passed, with four expected GP kernel-bound warnings; 846.77 seconds. The first sandboxed test invocation could not use its default temporary directory. A fresh workspace-local `--basetemp` resolved that test-harness issue.
- 2,500 matrix solve/eigenvalue/SVD/CPU–CUDA comparisons and 100 bounded SLSQP solves passed. Maximum linear-system residual: 1.24e-14.
- E1 dynamic-matrix smoke completed all four sampled windows with the original network sizes, compressed technical clock, and zero failed matched pairs.
- Threshold and matched SG technical smoke runs completed. These are software checks, excluded from scientific performance evidence.
- The repaired runtime passed the full nominal-baseline schedule/scaling/reward audit and one original-cycle controller replay. All audit checks are true; maximum reward recomputation difference is 8.53e-14.
- All 115 Python files in the original scientific snapshot are unchanged. The 116-item snapshot also contains the intentionally updated environment recipe. E1's 97 and the comparators' 105 scientific-source/data hashes still match their original manifests. All 46 frozen original manuscript files are unchanged. The open LaTeX protocol was not edited.

## Production restart

E1 resumes `Polymer/Results/e1/20261013_192015_916c06`: completed horizon and weights are skipped; dynamic matrix restarts in `markov_s7/a2`, then residual and combined run sequentially with the original one-seed, 100-decisions-per-window study budget. Earlier attempts are retained. Frozen diagnostic snapshots omit mutable replay and optimizer state, so they do not support continuing the interrupted online-training trajectory at cycle 196.

E2 restarts only threshold and its matched SG reference in `Polymer/Results/comparisons/scenario1/20261014_runtime_recovery_threshold_sg`. Completed LS, ARX and BO stay in their original folders. LS and nominal MPC references are reused through the original wrapper. The comparator wrapper has no training-checkpoint resume, so threshold restarts from the original initial condition.

Launch provenance and parent/worker logs are in `runtime_recovery_20261014/production_launch.json`, `e1_resume_attempt2.log` and `e2_resume_attempt2.log`. Parent PIDs at successful launch: E1 19460 and E2 36136. The first recovery launch had a PowerShell argument-quoting error and exited before any scientific worker began; its records are retained, and quoting was corrected before the successful restart. Both successful workers were observed advancing after launch. Completion remains pending.

## Completed evidence and readiness

Read-only intake of all 50 repeat pickles found full 240,000-step/300-episode trajectories, finite primary outputs/inputs/rewards, matching episode-average rewards, the episode-201 operating change and final evaluation flags, matching paired saved profile metadata, and correct seeds in the external replication metadata. The original bundles may have an unset top-level `seed`; the worker job and `replication_metadata.json`, together with the launcher's verified seeding path, establish the repetition identity. Exact unlogged disturbance histories are not reconstructed from a profile label.

The intake script and full seed/window results are in `runtime_recovery_20261014/audit_completed_runs.py` and `repeat_intake.json`. Final evaluation reward improvements from SG are mixed across channels and seeds; the repeated results do not support universal SG superiority. The completed E1 horizon and weights jobs provide 800 matched decisions with no failed pairs. The old interrupted dynamic-matrix job's 172 pairs are retained for diagnosis but must not be merged with its restarted trajectory as independent evidence.

LS, ARX and BO each completed 160,000 Scenario 1 steps. ARX's saved tracking errors deteriorate severely late in the run (evaluation temperature RMSE about 121.5 K). This is a separate scientific/numerical validation flag, not one of the two process crashes. Preserve it and audit its predictors/actions before attributing it to the reidentification method or writing a comparison claim. No ARX tuning or rerun was performed in this recovery.

Before additional tests, finish the restarted E1/E2 evidence, investigate ARX's late deterioration, and review the paired output-wise results. Unseen-reference evaluation, distillation Scenario 2 replication, joint-channel tests, and stress/sensitivity evidence remain separate roadmap items. No new scientific test design was launched here.

## Files inspected and changed

Inspected: original wrapper/worker paths, manifests and statuses for all three campaigns; both failed logs; Windows crash events; Conda package records; source hashes; original-manuscript snapshot; all 50 repeat bundles and metadata; completed comparator metrics; preservation/comparison tests; nominal baseline and existing protocol documentation.

Changed: the two environment specifications; this dated report; new derived recovery/intake files under `change-reports/runtime_recovery_20261014/`; synchronized paper workflow records. Original controllers, agents, plant, runners, raw results and manuscript package remain preserved. Long-running restarts write only fresh attempt/result folders.

## Visible progress viewers

At Amir's request, two visible CMD windows follow the existing production logs. `runtime_recovery_20261014/live_progress.ps1` reads parent and active-worker logs, follows subsequent categories/methods automatically, and reports launcher exit. `view_E1_progress.cmd` and `view_E2_progress.cmd` reopen the viewers. Each script passed a read-only `-Once` check before launch. Closing a viewer leaves the simulations running. No training process or scientific source was changed by opening these windows.
