> Historical pre-approval plan. The approved archive is now complete; see STATUS.md.

# Proposed OneDrive storage relief

The code repository contains 343.8 GiB of logical file content; the paper repository contains 0.67 GiB. Local C: still has about 233 GiB free. These are local file measurements, not a measurement of the cloud account quota or version history.

The main growth is native Aspen checkpoint export: the Python state rotates between two files, but a new native folder is retained for every completed checkpoint cycle. Native .dynf files alone total about 192.6 GiB. No file has been moved or deleted.

Proposed archive: 910 old native checkpoint folders, 162.6 GiB, under `C:\Users\hamediaa\RL_MPC_local_archive\storage_20261023` outside OneDrive.

For each checkpoint directory, retain the earliest three and latest four native folders per slot, every folder referenced by latest.json, and every folder modified within one hour. Preserve all Python checkpoints, E1 capture data, trajectories, policies, result analysis, source/models and active native sessions. Immediately before any move, recheck current indexes, age, unchanged contents, paths and worker identities. Preserve relative paths and a restoration manifest; verify every moved file. Abort if any guard fails.

This retains the archived files locally and removes their synced copies from OneDrive. It changes where these historical backups live; they would no longer be cloud backed up. Same-volume relocation does not itself add local disk space. A second local copy or external backup can be arranged later. Cloud quota recovery depends on which files actually synced and OneDrive recycle-bin handling; do not claim it is already freed.

Candidate checkpoint parents and approximate archive amounts:

- `Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1/training_checkpoints`: 145 folders, 32.76 GiB; latest cycle 165.
- `Distillation/Results/state_recovery/20261021_remaining_active/e1/horizon/a1/training_checkpoints`: 222 folders, 46.31 GiB; latest cycle 246.
- `Distillation/Results/state_recovery/20261021_remaining_active/e1/markov/a1/training_checkpoints`: 168 folders, 27.34 GiB; latest cycle 190.
- `Distillation/Results/state_recovery/20261021_remaining_active/e1/residual/a1/training_checkpoints`: 168 folders, 27.32 GiB; latest cycle 190.
- `Distillation/Results/state_recovery/20261021_remaining_active/e1/weights/a1/training_checkpoints`: 169 folders, 27.57 GiB; latest cycle 190.
- `Distillation/Results/state_recovery/20261021_remaining_active/repetitions/markov_sg/s23/a1/training_checkpoints`: 19 folders, 0.65 GiB; latest cycle 300.
- `Distillation/Results/state_recovery/20261021_remaining_active/repetitions/residual_plain/s23/a1/training_checkpoints`: 19 folders, 0.65 GiB; latest cycle 300.

Current run check: twelve jobs are running (ten combined repetitions plus E1 combined/horizon); three E1 jobs (weights, Markov, residual) failed overnight with Aspen openDocument RPC errors, followed by a cleanup AttributeError. Markov SG seed 23 and residual plain seed 23 completed. This is not evidence that cloud quota caused the Aspen failures. No job has been interrupted by this audit.

Files inspected: repository instructions and current status, size-only directory metadata, checkpoint implementations/current indexes, live progress, failed E1 status/tracebacks, Windows disk/process/memory counters. Files changed: only this proposed plan and read-only inventory/candidate evidence in this report directory.

References: [Microsoft guidance for retaining local files outside OneDrive](https://support.microsoft.com/en-us/onedrive/delete-files-or-folders-in-onedrive); [school/work storage and recycle-bin behavior](https://support.microsoft.com/en-gb/onedrive/manage-your-onedrive-for-work-or-school-storage).
