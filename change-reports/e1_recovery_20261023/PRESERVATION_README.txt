VERIFIED LOCAL DATA - UPDATED AFTER APPROVED DISK RECLAIM, 2026-10-23

Historical native checkpoint data is retained locally in 910 lossless ZIPs (18.42 GiB) under compressed_storage_20261023/. These ZIPs represent all 3640 files, 162.59 GiB uncompressed. Every original file and ZIP was SHA-256 verified before removal, and every retained ZIP was rehashed after removal. The user explicitly approved replacing the duplicate uncompressed historical folders with these verified ZIPs.

storage_20261023/: the 910 duplicate historical native folders have been removed. The original-path restore_manifest.json and preservation metadata remain here. Earlier receipts showing originals present describe the earlier verification time. raw_duplicate_removal_20261023.json in the compressed directory records the subsequent approved removal. The immutable compression manifest remains unchanged; its originals_deleted=0 field describes compression time, not the present state.

compressed_storage_20261023/: keep all 910 ZIPs, restore_manifest.json, receipts.jsonl, restore_checkpoint.py, and the verification/removal records together. None of these retained files was deleted.

e1_recovery_20261023/: complete Python/native cycle-190 backups for weights, Markov and residual, preservation mappings, scientific source/protocol ZIP, and the 200-pair reviewer-data ZIP remain intact.
healthy_e1_checkpoints_20261023/: complete saved combined/horizon backups and their mappings remain intact.
Current E1 checkpoint directories in the project were excluded. All five latest Python/native checkpoint hashes and both Python slots were checked after cleanup. No live simulator was called or terminated by the cleanup.

RESTORATION

Historical ZIPs contain native Aspen checkpoint files. Full training resumption additionally requires the matching Python/learner checkpoint. Retain the complete current checkpoint sets and the five full backup sets, including their indexes and preservation mappings.

restore_checkpoint.py extracts a selected archive into a NEW staging directory only. It verifies the ZIP and every extracted file hash, refuses an existing destination, and never replaces active data. Example PowerShell command (choose a staging directory that does not exist):
& 'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\python.exe' -S -B 'C:\Users\hamediaa\RL_MPC_local_archive\compressed_storage_20261023\restore_checkpoint.py' --index 0 --destination 'C:\Users\hamediaa\RL_MPC_local_archive\my_staged_restore'

The compressed manifest maps ZIP contents to their historical local archive directories. storage_20261023/restore_manifest.json maps those archive directories back to their original project paths. Do not restore over a newer checkpoint or a running worker.
An actual fresh extraction after removal (archive index 909, four files) passed every SHA-256 check under restore_test_after_cleanup_20261023/. The earlier index-0 test remains under restore_test_20261023/.

SPACE AND RETENTION

At the final disk check, C: had 345,369,247,744 bytes free (321.7 GiB). Current files continue changing as jobs run.
Automatic Aspen checkpoint retention has NOT been changed or enabled. The existing writer rotates two Python checkpoint slots but keeps creating step-numbered native folders. Closed private load copies also accumulate. This approved historical cleanup frees space; it does not stop that future growth.
Keep the newest two complete matched Python/native checkpoints. Any later automatic archive/retention implementation must also preserve selected E1 snapshots, raw traces, pending recovery dependencies, and all files still used by live Aspen applications.

These local backups share the C: drive. No independent-device copy or new cloud upload was performed. OneDrive remains intentionally stopped. No current raw scientific result was removed.

Audit: change-reports/e1_disk_20261023/STATUS.md in the RL_assisted_MPC repository.
