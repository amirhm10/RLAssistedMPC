# Automatic Aspen retention — October 23, 2026

Automatic retention is enabled as an independent, low-priority current-user Windows task. It preserves both complete Python/native checkpoint slots for each of the five E1 configurations and every scientific result directory. Scientific workers, their sealed sources, Aspen applications and the automatic combined coordinator are not controlled by this helper.

Live status: `C:\Users\hamediaa\RL_MPC_local_archive\aspen_retention_20261023\STATUS.md`. Machine-readable state, archive receipts and compressed files are alongside it. The existing task is `RLAssistedMPC-AspenRetention-337f010bee3f`; its worker resumed as PID 58264 at 22:01:59 UTC (18:01 Toronto).

## Retention behavior

Before selecting an old native checkpoint, the helper verifies both Python files structurally without executing pickle code, source provenance, the latest writer checksum, native model/snapshot hashes and their metadata linkage. Prior verified Python generations have persistent checksum anchors. A legacy previous slot without a recorded writer checksum is first adopted using complete structural/provenance/native linkage checks and a fresh checksum; this is not a native simulation restart test. An incomplete save or changing generation defers cleanup and retains everything.

Only older step-numbered native folders and old closed private `load_*` copies are eligible. The latest two complete native sets, ten explicit saved recovery dependencies, newest two private copies per role, active document markers and result artifacts are excluded. Selected E1 comparison snapshots and raw traces remain in place. Known private sessions come from setup metadata and parsed checkpoint plant state; maintenance does not read mutable queue manifests, take queue/case locks or enumerate worker status directories at runtime.

Each eligible directory is compressed outside OneDrive. Every archive member is decompressed and SHA-256 compared with its source; the ZIP also gets a SHA-256 receipt. Sources are rechecked before a same-parent rename into an owned retirement directory. Only enumerated, hash-verified files in that directory are removed. Crash receipts resume unfinished cleanup. Clearing Windows READONLY applies only inside these verified retirement directories.

The task runs without administrator access or an execution time limit. It handles up to twelve folders per pass, polls every thirty seconds, waits for two stable checkpoint observations and skips busy or ambiguous data. It does not load NumPy, PyTorch or Aspen. Archive and verification I/O consumes resources, but the worker runs below normal priority with bounded memory. Results and retained ZIPs still grow; this policy removes the large duplicate-native-copy accumulation rather than imposing a hard disk quota.

## Validation and current runs

All 33 preservation tests and fifteen repeated concurrent writer/queue checks pass with the production two-second checkpoint quiescence guard. Tests cover interrupted saves/removal, corrupt archives, read-only OneDrive directories, path boundaries, source changes, active documents and preservation across successive generations. Earlier rapid-write experiments also encountered a Windows replacement error in a writer-only control; they do not establish an observer cause. Final evidence is in `tests_desktop_final.log` and `concurrency_stress_final.log`.

Actual native and private archive extractions into fresh stages passed all eight restored file hashes, with no simulator calls or existing-file replacements (`actual_staged_restores.json`). The 910 previously verified historical ZIPs, their restoration manifests and the five full offline recovery backups are retained.

At the 22:05 UTC activation audit, 71 completed archives represented 7.25 GiB of old raw files with 0.82 GiB of ZIPs, including new cleanup beyond the repaired pending batch. C: had approximately 326.1 GiB free. All completed ZIP hashes were independently rechecked. The five retained checkpoint pairs were combined 76400/76000, horizon 116000/115600, weights 78800/78400, Markov 78400/78000 and residual 78000/77600. The ten saved dependencies remain present; all 139 scientific source hashes and 200 frozen trace hashes are unchanged. Nine original workers have unchanged identities and positive CPU deltas, and the original combined coordinator remains alive. Cleanup and disk counters continue changing; consult live status for current totals.

One run failure was discovered during the activation audit: E1 horizon ended at 21:44:44 UTC (17:44 Toronto). The underlying traceback is an Aspen `openDocument` RPC failure, followed by the secondary `AD application.Quit` error seen in previous failed attempts. Its checkpoint pair is protected. The cause of the native failure is not established by this maintenance change. See `activation_audit.json` for all process identities, source/trace hashes, protected checkpoint steps and archive totals; do not interpret retention activation as proof that every experiment is healthy.

## Restore or stop maintenance

Restore a directory into a NEW stage (choose an existing receipt and a destination that does not exist):

```powershell
& 'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\python.exe' -B 'C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC\run_aspen_retention.py' --restore-receipt 'C:\Users\hamediaa\RL_MPC_local_archive\aspen_retention_20261023\receipts\RECEIPT.json' --destination 'C:\Users\hamediaa\RL_MPC_local_archive\new_staged_restore'
```

The helper verifies the ZIP and every extracted file and refuses an existing destination. A native ZIP requires its matching Python/learner state for full training resumption. Never extract over a running job. These archives remain on C: and are not an independent-device backup; OneDrive stays intentionally stopped.

Creating `stop.request` in the maintenance state directory gracefully stops only maintenance. Inspect the task and its worker before removing that flag and starting the existing task again. The one-time `resume_retention.py` records this repair's guarded resume; do not rerun it against a live worker. No recurring task or experiment launch was added.

Inspected: sealed `column_e1/checkpoints.py`, `column_e1/runtime.py`, existing recovery metadata/manifests, paired saved checkpoint/native files, baseline worker identities and frozen trace hashes. Changed: three new retention modules, their focused test file, scoped audit/restore helpers and reports, preservation guide, and recovery-status addendum. The unrelated `run_polymer_phase2_repeats.py` edits are preserved. No scientific source or manuscript was changed and no remote push was made.
