# All-run audit — October 21, 2026, 20:50 Toronto

There are **22 unfinished production jobs: four running, ten awaiting automatic launch, four E1 configurations held behind a failed job, and four failed jobs**. This count covers the existing authorized campaigns. Historical failed attempts and superseded technical pilots do not add new jobs to it.

The artifact audit started at 2026-10-22T00:47:35.661970+00:00. The final process/queue check was at 2026-10-22T00:50:04.097697+00:00. These correspond to October 21 in Toronto.

## Finished and verified

- Polymer repetitions: **50/50**, with completed 300-cycle worker records and saved result checksums.
- Column repetitions: **34/50**: horizon 10, weights 10, Markov 9, residual SG 5. The five residual SG workers finished since the last audit. Their 300-cycle outputs, provenance, zero process exit codes and Aspen cleanup verify. The channel supervisor will finalize its aggregate manifest after the other children exit; its current ten `running` labels do not override completed child records.
- Polymer E1: **3,200 unique completed matched pairs**. Registered capture hashes verify. The historical residual final-trajectory export remains absent; its matched-pair evidence survives.
- Polymer E2: **all five methods plus the baseline** have completed trajectories and shared analysis. Registered trajectory hashes and recovery-stage artifact contracts verify. The old threshold policy export remains empty; the separate policy-recovery attempt is discussed below.
- Polymer E3 and column E3: **3/3 stages each**, with all six saved artifact inventories verified.
- Column E2: **SG, threshold, LS and BO all complete**, each with 200 cycles/80,000 steps and verified final artifact hashes and Aspen cleanup. BO finished October 21 at 12:36 Toronto after all 40 valid calibration trials and its full evaluation. Its four failed solver calls among 272,105 calls are recorded recoverable calls; the completed job is not classified as failed.

## Still running

Four residual plain repetitions remain live, with matching launch-time process identities, increasing CPU/progress, and no fatal marker in their recent log tails:

- Seed 7: **296/300 cycles**, PID 32632, case 6.
- Seed 11: **291/300**, PID 47360, case 7.
- Seed 37: **295/300**, PID 29360, case 9.
- Seed 53: **297/300**, PID 31656, case 10.

## Queued and held

The ten combined repetitions have not launched. Their independent coordinator PID 36308 and residual supervisor PID 15980 are live with verified creation times. Residual and combined source, model, assignment and bootstrap guards pass. Cases 1, 2, 12, 4–10 remain reserved for this queue; SG seed 23 uses case 12 independently of the failed Markov case-3 retry.

The handoff preflight discovered that the existing cleanup helper rejects `conhost.exe` descendants of the crashed residual case-8 Aspen application. Without intervention it would block before combined. Under the author's existing instruction to preserve automatic queue progress, this audit cleared only the two crashed runs' verified orphan trees on cases 8 and 3: two owned Aspen applications, their four Aspen servers, and four System32 console hosts. Every executable path, parent relationship and process creation time was checked; canonical case leases and the existing same-handle termination guard were used. All four healthy worker identities and both supervisors were preserved. No scientific source, runtime, checkpoint or queued manifest was changed. Both failed runs now have zero remaining owned Aspen applications. This clears the known current handoff blocker; the helper's general handling of a future crash remains unchanged.

The separate column E1 supervisor exited after combined failed. **Horizon, weights, Markov and residual E1 are held**, not advancing automatically.

## Current failures

1. **Markov SG seed 23, attempt 4:** stopped after cycle **135/300**, October 21 around 04:52 Toronto. The supervisor records exit `0xC0000005`. Windows identifies the private NumPy `_multiarray_umath.cp313-win_amd64.pyd` extension as the faulting module. There is no completed result or verified full training-resume checkpoint.
2. **Residual plain seed 23:** stopped after **190/300**, October 21 around 10:15 Toronto. Windows and the exited process confirm `0xC0000005` in the same NumPy extension. Its worker status remains stale `running`; it is a confirmed failure. The channel supervisor has not yet finalized the group.
3. **Polymer threshold policy recovery:** crashed after **129/200**, October 20 at 18:32 Toronto, with the same NumPy extension/exception. The last saved **cycle-120** four-agent policy checkpoint loads with finite, nonempty tensors and a matching checksum. It contains inference weights only; no full optimizer/replay/plant resume state, final 200-cycle trajectory or final policy exists. Do not represent it as the final original threshold policy.
4. **Column E1 combined:** failed after cycle **13/300**, at the first sampled branch near step 5202. Native initialization changed controlled composition by approximately 1.75e-6 and temperature by 0.00305, exceeding the declared 1e-6 and 0.002 bounds. **Zero completed production matched pairs** were saved. The full cycle-10 native/Python checkpoint and its native project hashes verify. A repair must reproduce and resolve the fidelity failure before restoring production. Aspen also has a crash event during this failure/cleanup; it is distinct from the three NumPy crashes.

The previous OpenBLAS mitigation has not eliminated native crashes. The present Windows records identify NumPy's extension rather than the original Conda OpenBLAS DLL. They do not by themselves establish the underlying corruption mechanism. Queued combined workers use this same numerical runtime, so successful source guards do not guarantee completion.

## Time remaining and recommended work

- Healthy residual workers: approximately **1–2 hours** for the slowest worker and final exports; linear training estimates span 0.3–1.0 hours.
- Combined cohort: allow **36–60 hours after launch**, roughly **2–3 days from this audit** if handoff succeeds and no further crashes occur. Comparable completed Markov/residual column jobs took 30.8–35.9 hours; a complete same-profile combined column cohort has not yet supplied a direct duration measurement.
- Failed column repetitions: allow roughly **31–36 hours each after a reliable full restart**, plus unknown repair time. Spare-case assignment determines whether they overlap the combined cohort. No restart was launched during this audit.
- Threshold: repair time is unknown. Cycle-120 inference weights are usable as partial policy evidence but cannot resume the original training state; do not estimate completion as simply the remaining 71 cycles.
- Full column E1: **several weeks after repair with the present sequential budget**. Five configurations require 600,000 main steps plus approximately 1,664,000 branch steps. Measured main-run throughput, including startup, was 1.014 seconds/step, giving a rough **26.6-day step-only projection**. Production branch throughput remains unmeasured, and document transfer/learning overhead can change this estimate. Warm-up cycle percentages exclude most branch work. This is a planning projection, not a promised finish date.

Keep the healthy workers and combined queue running. Investigate the recurring native NumPy failure before blind retries; repair and validate E1 restoration, then use its verified checkpoint. Preserve the cycle-120 threshold policy and all finished bundles. Changing E1 concurrency or sampling budget is a separate execution/protocol decision. There is no reliable overall finish date while these failures are unresolved.

## Evidence and changed files

`audit_runs.py` reproduces the read-only status/artifact audit. `audit.json` records all current and historical rows, normalized failures, completed artifact verification, queue guards, checkpoint checks, process identities and ETA inputs. `completed_artifact_verification.json`, `snapshot_initial.json`, `process_inventory.json` and `windows_fault_events.json` retain the underlying observations. `final_state.json` records the post-cleanup health check. `orphan_cleanup_preflight.json`, `orphan_cleanup_plan.json` and `orphan_cleanup_applied.json` document the concrete blocker, reviewed target set and resulting actions; `clear_verified_orphans.py` is the narrowly scoped helper used.

Inspected sources include the active pointers/manifests and worker logs, `revision_comparisons/progress.py`, `run_distillation_phase2_repeats.py`, `continue_column_queue.py`, `recover_markov_seed23.py`, `recover_polymer_threshold.py`, `run_column_e1.py`, `column_e1/checkpoints.py`, the inference-checkpoint validator, completed E1/E2/E3 provenance, previous launch records and paper-memory/freeze records. Changes are limited to the dated audit/helper/evidence/report and workflow-status records. No numerical runtime or scientific execution source was changed. The unrelated pre-existing edit to `run_polymer_phase2_repeats.py` is preserved. All 46 original manuscript hashes verify unchanged; no TeX/PDF was edited.
