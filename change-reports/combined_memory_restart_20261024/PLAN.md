# Combined E1 memory restart prepared for review

Later reconciliation at 13:36 UTC: the user approval remains pending and this plan is still review-only. The weights/Markov creation identities below are superseded: weights now has accepted production PID 47172, and Markov has an independent memory-gated recovery. Combined's complete checkpoint pair has advanced to 80400/80000. Before any approved pause, refresh the full backups, trace hashes and actual peer identities and apply the tested generation-safe ownership helper from `../weights_pid_reuse_recovery_20261024/ownership_fix.py` in the new combined worker. Re-seal and test that adaptation and require actual eight-step four-learner/native validation before production. Do not execute the older prepared adapter/plan unchanged. No combined worker has been interrupted by this reconciliation.

At 09:56 UTC on October 24 (05:56 Toronto), the four remaining E1 workers are progressing with unchanged creation identities. Combined has 154/320 completed comparisons; the campaign has 414/640. No current scientific failure is established. The combined comparison application, PID 9316, uses 60,991,504,384 private bytes (56.8 GiB). Commitment headroom is 1,519,091,712 bytes (1.41 GiB), below the 6 GiB recovery launch threshold. All eight current Aspen applications belong to active workers; none is an unowned stale instance.

## Decision and scope

The restart plan is `prepared_for_review_only`. No worker, simulator or helper was signalled; no recovery task, runtime directory or viewer manifest was registered. The user has authorized recovery of failed jobs but also instructed: **"Do not interrupt healthy workers."** A preventive checkpoint restart of the currently healthy combined worker requires an explicit exception to that instruction. If combined actually fails before a decision, the existing failed-run recovery authorization applies after confirming the failure and preserving its state.

The proposed action affects only combined E1, seed 7, case 16. Its worker is PID 22588, creation identity 134371156883220647. Its registered native applications are PID 9316 (comparison) and PID 44800 (training); each has its exact creation identity recorded in `restart_plan.json`. Weights, Markov and residual, their six applications, the completed horizon/repetition evidence and automatic retention remain protected. The original combined budget remains 320 comparisons and 300 training cycles, with 20 selected decisions per window. No scientific source, shared queue plan or global package change is proposed.

## Saved recovery evidence

The refreshed local backup is outside OneDrive at:

`C:\Users\hamediaa\RL_MPC_local_archive\memory_preservation_20261024T0940_combined`

It contains 168 independently hash-verified files, 2,178,199,889 bytes (2.03 GiB): both complete matched Python/native checkpoints at steps 79200/78800, the full learner/controller recovery state, job/index/status records, the append-only decision record and all 154 then-completed raw comparisons. Both Python checkpoint digests match the independent retention anchors. The backup manifest SHA-256 is `e4087b0b926e18f2303f7d8be89b8787f54e09215318d3f3973b134276f219c0`. The original files, older verified backup and all lossless archives remain intact.

This is saved-file integrity evidence. A cold combined restore has not yet been attempted because the original worker still owns case 16 and headroom is insufficient. The newest full checkpoint and all additional completed traces must be refreshed and backed up again immediately before an approved pause. Unsaved steps after the newest full checkpoint may need replay; completed scientific trace files must remain unchanged.

## Execution after approval or confirmed failure

1. Recheck the current process creation identities, current manifests and all three protected peers. Verify and back up the latest two complete matched checkpoint sets and every newly completed trace; refresh the reviewed plan's checkpoint and trace anchors. Do not use the original 100-decision job: use the sealed `budget20_job.json` recorded in the plan.
2. After an approved pause, retire only the exact combined worker identity, then close only its registered native applications after confirming ownership again. Preserve its active ancestor helpers and every peer process. If the worker failed naturally, verify that fact before orphan cleanup. Check that its case lease no longer belongs to a live owner using the existing recovery wrapper; do not remove a live lease or alter a shared queue plan.
3. Require at least 6 GiB commitment headroom and an empty retired combined case. Set plan state to the appropriate explicitly authorized execution state only after those conditions are established. The guarded adapter itself cannot stop a process.
4. Launch the isolated normal-user supervisor with the existing private numerical runtime and fresh-owned-process loader. It must first perform an eight-step bounded continuation and verify all four learners/controllers, Python/native checksums and native restoration. The loader must leave the full Python state digest unchanged during native import. No administrator access, numerical package reinstall or fresh training initialization is needed.
5. Production may start only from the verified eight-step checkpoint. Recheck memory and peer identities before each launch, retain all frozen comparison hashes, and register the recovery manifest in the existing progress view without replacing its other entries. If validation fails or memory is insufficient, retain the state and report the exact blocker; do not duplicate a live or waiting launch.

## Validation and limitations

Thirteen standard-library unit tests pass in `tests.log`. They cover review-only refusal before any process/resource access, live/reused/inaccessible former workers, remaining or inaccessible owned applications, the exact 6 GiB launch threshold, invalid validation lengths and adapting only the intended case/budget row in a deep copy. The refusal path imports no NumPy, Torch or numerical bootstrap.

The actual read-only preflight in `review_preflight.json` verifies all 143 sealed source hashes, the adapter/protocol/job/backup seals, all 154 frozen trace hashes and the three peers plus retention. The real plan refuses execution because it is review-only. An authorized copy made only in memory also refuses because the original worker is live. No simulator was called, and the recovery runtime directory was not created. These checks validate the guarded preparation; they do not prove an unperformed cold restore or guarantee elimination of native memory growth.

Windows increased the commitment limit automatically during this check. Read-only paging-file usage reports a 43,115 MiB allocation; no Windows setting was changed. OneDrive syncing remains closed. Hourly monitoring remains active, and healthy workers continue while the preventive restart decision is pending.

Inspected: current worker/checkpoint/ownership records, capture decisions and traces, append-only logs, retention status/independent hash anchors, current recovery manifests, existing fresh-process recovery wrappers, source seals and Windows counters. Created: this plan, `restart_plan.json`, `fresh_process_protocol.json`, `recover_combined.py`, `test_recover_combined.py`, `tests.log` and `review_preflight.json`; the full preserved data remain outside OneDrive. Updated: the hourly audit/status and root recovery status. No scientific result or manuscript was edited.
