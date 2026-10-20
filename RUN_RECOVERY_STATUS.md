# Reviewer-run recovery — latest update October 20, 2026

## Column E1 integrated and started — October 20

At 2026-10-20T21:26:28.530174+00:00: column E1 combined seed 7 is live on source case 16 (PID 34440, cycle 1/300), with four standalone configurations queued. The 47 focused tests, four-channel native/learner checks and fresh-process full-state restart pass. A second private case-16 application handles rollouts; the training application remains untouched by restores. Full native/Python checkpoints are enabled.

All 13 existing worker identities and the automatic combined queue are preserved. The ten combined repetitions still reserve cases 1,2,4–10,12. Including the five newly authorized E1 configurations, 28 production jobs remain: 14 running and 14 queued. Production E1 results are pending. Details: change-reports/column_e1_20261020/INTEGRATION.md and launch_assessment.json.

## Snapshot/threshold repair and stopped-run recovery — October 20

At 13:57 Toronto: **23 production jobs remain: 13 running and 10 queued**. The ten residual repetitions are at cycle 11/300. Markov SG seed 23 attempt 4 is live (PID 23064, cycle 2/300). Threshold recovery is at cycle 18/200, PID 45524; its latest validated four-agent inference export is from cycle 10. Column BO has 36/40 calibration trials saved; its 200-cycle final evaluation follows.

The native snapshot continuation pilot passed; full column E1 adapter integration remains separate. Threshold's original empty export is preserved and its fresh policy-recovery run is underway with validated periodic exports. The former Markov attempt stopped at cycle 112 for an unconfirmed reason; attempt 4 has its own independent visible session. All eleven other production workers and the residual/combined coordinator retain their identities; combined remains automatic. Details: change-reports/state_recovery_20261020/RECOVERY.md and repair_assessment.json.

## All-study audit — October 20, 12:59 Toronto

**22 production jobs remain: 12 running and 10 queued.** Running: Markov SG seed 23 (cycle 108/300), ten residual repetitions (cycle 2/300), and column BO (35/40 calibration trials saved; 200-cycle evaluation still follows). The ten combined repetitions are queued after residual. The independent coordinator and every current worker are live; an independent Windows inventory finds no unregistered experiment worker or new failure.

Finished evidence verifies: 50/50 polymer repetitions, 29/50 column repetitions, 3,200 polymer E1 matched pairs, all five polymer E2 method trajectories plus baseline/shared analysis, six E3 stages across both plants, and column SG/threshold/LS comparisons. The separate case-16 snapshot pilot remains failed/held and blocks full column E1. The historical polymer threshold policy export is empty and needs recovery before frozen-policy testing; the ARX local waiting marker is superseded by the completed shared E2 analysis. These are separate from the 22 existing production jobs.

Recommended order and exact evidence: `change-reports/run_recovery_20261020/ALL_RUNS_STATUS.md` and `all_runs_20261020_165926.json`. Keep the live jobs/automatic queue running, repair native restoration and the unusable policy export separately, and audit/assemble finished column results before extending the study scope. No study was launched or stopped during this audit.

## Residual started using spare case 12 — October 20, 12:40 Toronto

The nine original healthy Markov runs are complete; **29 completed results are retained**. The active Markov SG seed 23 retry continues on case 3 with its existing process and progress (cycle 104 at inspection). To start the next channel immediately, the pending residual SG seed 23 job now uses free, byte-identical Aspen case 12. The pending combined SG seed 23 job uses case 12 too. Cases 1, 2, 4--10 serve the other seeds. BO on case 15 continues unchanged.

The old replacement coordinator stopped on an inaccessible retired worker PID after the original scheduler had stopped at the historical failed attempt. A new independent coordinator validates completed records/artifacts, confirms free case leases and starts residual now, then combined after residual. Final per-group records supersede retired PID queries. Source/model/ownership checks remain enforced. The old live retry bootstrap and original scientific sources are unchanged.

**Current launch inventory: 29 completed, 11 running, 10 queued** column repetitions. All ten residual workers started, including SG seed 23 on case 12. The queue and live-progress consoles are open. There are 49 passing focused tests, including all twenty fake pending runs with case 3 held occupied. Details and ownership/runtime verification: `change-reports/run_recovery_20261020/CASE12_QUEUE.md` and `case12_queue_assessment.json`. The earlier sections below are dated historical recovery records.

## Markov SG seed 23 restarted; remaining queue armed — October 20

Seed 23 failed again at cycle 62 with Windows access violation `0xC0000005` in Conda's `openblas.dll`, at offset `0x276a6`, despite the earlier conservative dispatch. A new a3 attempt started at 01:07 Toronto on October 20, PID 28652, with original seed 23, SG mode, Scenario 2 profile and 300-cycle budget. It restarts at cycle 1 because there is no verified full learner/simulator checkpoint. Its owned Aspen case is 3. The nine healthy Markov workers and original supervisor retain their process identities.

New processes load private official NumPy 2.4.6 and SciPy 1.17.1 Windows wheels, with their bundled OpenBLAS builds (0.3.31.188.0 and 0.3.30). Their wheel hashes match official PyPI metadata; all 2,465 extracted files are checked before each new worker. The shared environment, original Conda DLL, scientific sources, training settings and saved results are unchanged. This is a stronger isolated crash mitigation; multi-day stability remains unproven. Other optional dependencies can still load Conda BLAS.

The independent coordinator (PID 48692) tracks the old workers and new retry in a fresh 50-job aggregate manifest. It waits for their exits and confirmed Aspen cleanup, then starts ten residual runs followed by ten combined runs. A failed preceding category no longer strands unrelated queued work once its resources are verified free. It preserves original failures and completed results. Source checks, per-case leases, creation-time process checks and duplicate guards remain enforced. New processes also retry brief Windows manifest replacement locks, a failure reproduced in the queue test.

At the launch audit: **20 completed, 10 running, 20 queued** column repetitions; all nine protected workers unchanged. The 39 focused tests passed, including fake-worker handoff with injected retry/residual failures. The isolated runtime passed 2,738 actual Markov L-BFGS-B fits in 40 seconds and 8,214 linear batches, with fitted parameters identical to the reference build on the synthetic fixture. These checks do not establish production completion or the exact underlying binary defect.

The **ALL REVIEW RUNS - LIVE** and **SEED 23 RECOVERY AND REMAINING QUEUE** consoles are open through a demand-only interactive Windows task, independent of Codex, with no execution timeout. `show_review_progress.cmd` opens another read-only view. Detailed records: `change-reports/run_recovery_20261020/README.md`, `recovery_assessment.json`; current repetition manifest: `Distillation/Results/review_recovery/20261020_openblas/manifest.json`. The separate native snapshot pilot remains held.

## Historical recovery — October 18, 2026

The approved unfinished column work was restarted in visible CMD windows at 8:30–8:31 p.m. Toronto. The native snapshot pilot failed again at 8:38 p.m. and is held for separate bounded diagnostics; the other 14 production workers continue. Windows Task Scheduler owns the production launch process independently of Codex; it has no recurring trigger or execution timeout. This file is a saved recovery record. The **ALL REVIEW RUNS - live progress** console refreshes every 30 seconds.

Open `show_review_progress.cmd` for another read-only live viewer. Closing that viewer does not stop experiments. Each comparison also has a visible console streaming its worker log and heartbeat.

## Completed work retained

- Polymer repetitions: 50/50 completed.
- Polymer E1 combined: 300 cycles and 1,600/1,600 completed matched pairs, zero failed pairs. Its merged analysis contains 3,200 unique completed pairs across standalone and combined contexts.
- Polymer SG and repaired ARX comparisons: both 200 cycles; validated trajectories and the populated SG inference checkpoint are present. E2 postprocessing completed.
- Polymer and column E3: all three stages in each study completed; all six saved artifact inventories verify.
- Column repetitions: 20/50 completed, covering horizon and weights. Their completed worker records, 300-cycle inventories and result checksums were validated and reused.

## Remaining work started or queued

| Work | Aspen cases | Recovery |
|---|---|---|
| Five SG and five plain Markov repetitions | 1–10 | Ten workers started together; 300 cycles each |
| Residual repetitions | 1–10 | Ten queued after Markov |
| Combined repetitions | 1–10 | Ten queued after residual |
| Combined SG comparison | 12 | Restarted; original 200-cycle Scenario 1 |
| Threshold comparison | 13 | Restarted; original calibration and 200-cycle Scenario 1 |
| LS-only comparison | 14 | Restarted; original 200-cycle Scenario 1 |
| BO comparison | 15 | Restarted; 40 calibration trials, then original 200-cycle evaluation |
| Native snapshot restoration pilot | 16 | Failed again at TakeSnapshot; full-prefix retry held, bounded API diagnosis in progress |

Column E3 has completed and case 11 is free. All recovered column jobs restart from their original initial conditions: the failed runs did not save a verified complete learner/simulator continuation state. The old partial runs, logs and manifests are preserved separately.

## Failure review and repairs

1. **Markov repetitions:** all ten old workers exited with Windows access violation `0xC0000005`. The earlier Windows event audit identified `openblas.dll` for SG seed 23; the exact crashing module is unconfirmed for the other nine. Recovery workers now use the same conservative process-local numerical dispatch that completed the polymer recoveries: single-threaded Nehalem OpenBLAS and disabled NumPy X86_V3/AVX2/FMA3 dispatch. All ten workers verified the actual dispatch at startup. Installed packages, controller equations, seeds and training settings were unchanged. Long-run effectiveness still requires completion.
2. **Four comparison jobs:** old crashes coincided with the recovery wrapper's repeating 30-minute all-thread stack dumps. The last stack-file writes were within 0.40–0.54 seconds of exact 30-minute multiples after worker startup in all four jobs. This is strong diagnostic evidence, not a reproduced causal proof. The timed `faulthandler.dump_traceback_later` monitor was removed. Fatal-error diagnostics remain enabled; progress and heartbeats run in the separate supervisor process. An [upstream CPython crash report](https://github.com/python/cpython/issues/140815) concerns a different platform and an extremely small timeout; it does not establish the cause of these Windows crashes.
3. **Snapshot pilot:** the old failure occurred at `Results.TakeSnapshot`. The adapter omitted the documented `Results.Refresh()` prerequisite. It now refreshes before taking a snapshot and again before finding the new snapshot. The installed type library confirms the method. **The live retry still failed at TakeSnapshot, so Refresh alone is insufficient and restoration remains unvalidated.** A separate case-16 diagnostic tests native state/description/history handling immediately after initialization, avoiding another eight-minute full-prefix replay of a known failure. Other workers' source files are left fixed. [Aspen Modeler Reference, Results object](https://pdfcoffee.com/acm-aspen-modeler-reference-pdf-free.html).
4. **Progress display:** the current recovery supersedes old failed/blocked study rows in the console. Historical records remain in the JSON audit. A separate repetition-console startup was retried once with persistent diagnostics; the ten workers subsequently started and their numerical settings verified. That transient startup cause was not established.

The subsequent bounded snapshot diagnosis exported a nonempty native snapshot using a shorter description at startup. This does not establish that description length caused the original post-prefix server exception. A separate clock check measured an ordinary pre-restoration step from 40.0 to 40.166747 hours, close to the declared 1/6-hour interval. After `LoadSnapshot`, native time remained at 40.166747 on stepping; explicitly setting Dynamic mode, the communication interval and a future end time still did not advance it. The failing restoration branch must not be accepted as a matched continuation. The full pilot command is held, with its original command preserved as `run_before_snapshot_hold.cmd`, to prevent another blind eight-minute replay. The original production stepper is unchanged; this diagnostic has not established a stopped clock in the other jobs. A future validated adapter also needs a timing tolerance consistent with the observed native solver overshoot.

## Validation and evidence

- 61 focused regression tests passed; six recovery tests passed again after launcher logging changes, followed by seven recovery tests including the new pilot-hold guard.
- A separate 40-second numerical-runtime stress passed 438,521 solve/inverse pairs and 43,853 SLSQP/Lyapunov/PyTorch batches. It verified the exact `Library/bin/openblas.dll` named in the Windows crash record, using Nehalem dispatch and one thread. This bounded diagnostic is not a guarantee for multi-day training. Details: `change-reports/run_recovery_20261018/runtime_stress.json`.
- Old crash timings/exit codes and the preserved Windows fault event are in `change-reports/run_recovery_20261018/failure_review.json` and `windows_fault_events.json`.
- Original E1/E3/column-repeat scientific inventories verify unchanged: 97, 103, 103 and 86 files respectively.
- All 46 frozen original manuscript files verify unchanged. No TeX or PDF was edited.
- Original repetition and column-comparison manifest checksums match the pre-recovery values.
- Detailed current worker identities, owned Aspen identities, runtime checks and completed-polymer validation: `change-reports/run_recovery_20261018/recovery_assessment.json`.
- Full prior-study artifact/source audit: `change-reports/run_recovery_20261018/artifact_audit.json`.
- New attempt plan and commands: `Distillation/Results/review_recovery/20261018/plan.json`.
- New comparisons: `Distillation/Results/column_review/20261018_recovered/`.
- New repetition scheduler log: `Distillation/Results/review_recovery/20261018/repeats/supervisor.log`.
- Bounded native snapshot/clock diagnostics: `column_snapshot_diagnostics.py` and `Distillation/Results/review_recovery/20261018/snapshot_diag_short/`, `snapshot_diag_clock/`. These are technical failed diagnostics, with preserved exports and clock traces, and are not E1 validation evidence.

Files inspected include the original repeat/comparison manifests, worker logs and crash records; `run_distillation_phase2_repeats.py`, `run_distillation_remaining.py`, `review_restart_recovery.py`, `column_review/pilot.py`, `revision_comparisons/runtime.py`, `revision_comparisons/progress.py`, the completed polymer status/postprocess records, and `audit_revision_studies.py`. Changes are confined to recovery infrastructure, snapshot access, progress presentation, regression tests and workflow records. The unrelated pre-existing edit to `run_polymer_phase2_repeats.py` was preserved.

Full column E1 learner/gate-history integration, unseen-reference evaluations and E4/E5 extensions remain separate work. This recovery executes the existing unfinished study queue; it does not establish completed column evidence or close reviewer comments.
