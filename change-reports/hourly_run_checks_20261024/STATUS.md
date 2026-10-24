# Hourly experiment check — October 24, 2026

## Follow-up at 05:32 UTC (01:32 Toronto)

Five E1 workers remain active with their exact original creation identities. The total advanced from 377 to 387 completed comparisons: horizon 74/80, weights 60/80, Markov 60/80, residual 60/80 and combined 133/320. Initial checkpoint cycles were 295, 236, 221, 226 and 196 respectively. During follow-up, weights saved cycle 237 and Markov cycle 222. All new completed NPZs pass ZIP-member CRC checks and have recorded SHA-256 hashes; previous trace files remain present. Decisions are unique within channel/window/step and every saved decision has completed status. The other standalone pair counts remain at sixty because training is progressing toward their next selected capture window.

All five Python workers are alive. The combined Python worker had zero CPU change in the twelve-second sample while in a native call. Its owned comparison app, PID 9316, then accrued 49.95 CPU seconds during follow-up, and its completed-pair/checkpoint counts have advanced since the previous check. An idle Python sample therefore does not establish a hang. All ten native simulator instances in the initial inventory match active E1 ownership by exact creation identity. Horizon's supervisor is live in production. The expected native proxy RPC diagnostics were not classified as scientific failures.

The retired queue coordinator remains completed. All fifty previously hashed repetition result files remain present at their recorded sizes. This hourly check did not rehash the approximately 10 GB of unchanged completed result files; their original hashes remain in `audit.json`. All 143 sealed source files were reverified. There are no pending launches or current scientific worker failures. The OneDrive sync client remains closed.

Retention remains active and has replaced 642 obsolete folders (180,837,351,656 raw bytes) with 20,457,876,798 compressed bytes. Its initial combined-job deferral was explicitly waiting for a stable newly committed checkpoint. A later pass verified both combined checkpoints at 78400/78000; a concurrent weights save briefly deferred that job instead, retaining all files. Other later verified checkpoint steps were horizon 118000/117600, Markov 88800/88400 and residual 90400/90000. Weights had verified 94400/94000 before its next save. These are normal conservative save deferrals, not deletion or checkpoint failures.

At 05:32:19 UTC, C: had about 387.75 GiB free and commitment headroom was 6.53 GiB. The combined comparison app's private memory fluctuated from about 43.86 to 43.09 GiB during the check, above the previous hour's approximately 39.95 GiB. Memory remains tight, and no spare stale simulator was identified. All currently owned simulators were preserved. No job was launched or stopped, no simulator API was called, and no package, sealed source, queue plan, result or archive was changed. Hourly monitoring continues because the five E1 runs remain unfinished.

Evidence created: `audit_20261024T0530.json`, containing the CPU/identity samples, current progress, ten new trace hashes, append-only log summaries and retention observations. Inspected files were the existing root recovery status, current E1 worker/checkpoint/ownership metadata, horizon supervisor/current manifests, completed coordinator status, retention status, five append-only logs, current capture decision streams and previously indexed completed outputs. Changes are limited to this progress record, the root status and the new audit. The initial detailed metadata snapshot is retained locally as ignored diagnostic material. No implementation change required a new simulation or unit test.

## Initial check at 04:34 UTC (00:34 Toronto)

At 04:34:37 UTC (00:34 Toronto), all fifty column repetition jobs have completed. The five E1 configurations remain running; no current scientific job is failed or queued. This is an operational check, not an interpretation of E1 performance.

## Progress and verification

- Column repetitions: 50/50 current worker completion records, nonempty result files and file SHA-256 hashes verified. The result files total 10,048,944,555 bytes. Polymer repetitions were previously verified complete and were not rehashed in this check.
- E1 horizon: case 13, PID 38752, checkpoint cycle 293, 68/80 completed pairs.
- E1 weights: case 14, PID 28560, checkpoint cycle 227, 60/80 completed pairs.
- E1 Markov: case 15, PID 47812, checkpoint cycle 213, 60/80 completed pairs.
- E1 residual: case 17, PID 50420, checkpoint cycle 217, 60/80 completed pairs.
- E1 combined: case 16, PID 22588, checkpoint cycle 195, 129/320 completed pairs.

E1 totals 377/640 pairs, leaving 263 comparisons plus unfinished training. Every completed decision is unique within its channel/window/step key, and its saved NPZ exists, passes ZIP-member CRC checks and has a recorded SHA-256. No claim about numerical performance, safety or statistical confidence follows from these storage checks. The five Python worker identities match their established creation identities, with positive CPU changes across a twelve-second sample. Horizon's supervisor remains live in production. All ten AspenModeler instances match the five active ownership journals by PID and exact creation time. An idle comparison or training application was not treated as stale. All 143 sealed source hashes verify unchanged.

## Finished coordinator status reconciled

The original coordinator (PID 36308) finished at 04:11:30 UTC after dispatching all twenty queued repetitions. Its frozen aggregate contains 48 completed rows and two historical failed rows: Markov SG seed 23 and residual plain seed 23. The independent recovery outputs for both seeds are completed and verified, at 15:31:50 and 15:35:49 UTC on October 23 respectively. No additional run is needed for either seed.

The exact original coordinator status bytes and SHA-256 are retained in `coordinator_status_before.json` and `coordinator_reconciliation.json`. Only the retired coordinator's mutable final status was reconciled to completed, with the original process final state and recovery provenance included. The queue plan and final aggregate manifest hashes are unchanged, so historical failures remain inspectable. The existing progress view now receives the reconciled completion status. The actual horizon protected-process check passes after reconciliation, preventing the stale coordinator flag from causing a false completion error. No scientific source, result, queue plan, active worker or native application was changed or restarted.

## Resources and retention

At the audit there were 420,585,603,072 free bytes on C: (391.7 GiB) and 9,792,425,984 bytes of commitment headroom (9.1 GiB). Commitment remains high. The combined comparison app, PID 9316, uses approximately 39.95 GiB of private memory and is actively owned; it was preserved. No additional worker launch or native process recycling was attempted.

The retention worker (PID 58264) is live. Each E1 job has two verified complete matched checkpoints. Retention has replaced 604 obsolete folders, representing 163,864,594,193 original bytes, with 18,539,449,742 compressed bytes. Its protected checkpoints, pinned dependencies, results and local archives remain preserved. Checkpoint steps recorded in this audit are combined 78000/77600, horizon 117200/116800, weights 90800/90400, Markov 85200/84800 and residual 86800/86400.

Hourly monitoring remains enabled because all five E1 configurations are unfinished. No scientific failure required recovery during this check. The check did not call Aspen, signal a process, reopen OneDrive, change installed packages or require Windows administrator access.

## Evidence and files

Inspected: `RUN_RECOVERY_STATUS.md`; the current recovery pointers/manifests and worker completion records; fifty repetition result files; five E1 status, checkpoint, ownership and decision streams; current append-only logs; retention state; the finished coordinator records; source hash inventories; and actual Windows process identities and resource counters. Operational schemas were checked against `aspen_retention.py`, `continue_column_queue.py`, `run_column_e1_horizon_recovery.py`, `run_column_e1_process_recovery.py`, `column_review/common.py` and `revision_comparisons/progress.py`.

Created evidence: `initial_snapshot.json`, `reconciliation_inputs.json`, `audit.json`, `coordinator_status_before.json` and `coordinator_reconciliation.json`, plus this note. Updated: the root recovery status and the retired coordinator's final status. Mutable metadata was read through stable snapshot aliases; append-only logs and immutable completed files used shared reads. The unrelated dirty polymer runner remains untouched. Validation consists of actual artifact/identity/hash checks, preserved queue hashes, status readback and the real horizon protected-process check; no simulation was launched for testing.
