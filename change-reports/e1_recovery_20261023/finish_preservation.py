"""Finalize portable offline restoration instructions after all ZIPs verify."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive')
COMPRESSED = ARCHIVE / 'compressed_storage_20261023'
receipt = json.loads((HERE / 'compressed_archive_verified.json').read_text(encoding='utf-8'))
assert receipt['verified'] and receipt['folders'] == 910 and receipt['original_files'] == 3640
shutil.copyfile(HERE / 'restore_verified_checkpoint.py', COMPRESSED / 'restore_checkpoint.py')
shutil.copyfile(COMPRESSED / 'restore_manifest.json', HERE / 'compression_restore_manifest.json')
text = '''VERIFIED LOCAL DATA - 2026-10-23

No original archived folder has been deleted.

storage_20261023/: all 910 historical native-checkpoint folders, 3640 original files, 162.6 GiB. Every file was reread locally and SHA-256 matched.
compressed_storage_20261023/: 910 lossless ZIPs plus restore_manifest.json. Every restored file hash was checked during compression, then every complete ZIP hash was independently rechecked. Originals remain intact.
e1_recovery_20261023/: complete Python/native cycle-190 checkpoints for weights, Markov and residual, their preservation mappings, the scientific source/protocol ZIP, and the 200-pair reviewer data ZIP.
healthy_e1_checkpoints_20261023/: copied complete saved checkpoints for combined (cycle 180) and horizon (cycle 265), with source/copy/hash mappings. No live Aspen application was called to create these copies.

RESTORATION

The 910 historical folders contain native Aspen checkpoint files. Full training-resume state also needs its matching Python/learner checkpoint; the five E1 backup sets retain those full dependencies. Keep their latest.json and preservation mappings together.

For historical ZIPs, restore_checkpoint.py extracts a selected archive into a NEW staging directory only. It verifies the ZIP hash and every extracted file hash, refuses an existing destination and never overwrites active data. It works with an archive directory copied to another drive because ZIPs are located by verified manifest filename.

Example PowerShell command (change the staging directory if it already exists):
& 'C:\\Users\\hamediaa\\AppData\\Local\\PythonEnvs\\rl-assisted-env\\python.exe' -S -B 'C:\\Users\\hamediaa\\RL_MPC_local_archive\\compressed_storage_20261023\\restore_checkpoint.py' --index 0 --destination 'C:\\Users\\hamediaa\\RL_MPC_local_archive\\my_staged_restore'

The compressed manifest maps ZIPs to the original LOCAL ARCHIVE directories. storage_20261023/restore_manifest.json separately maps those archive directories back to their original OneDrive project paths. Never replace a newer checkpoint or restore over a running worker. Extraction into staging does not start Aspen or resume training.

BACKUP LOCATION

These local copies all occupy the same C: drive. Copy the verified compressed archive, the two E1 backup directories and their manifests to an external device or an explicitly approved cloud destination for protection independent of that drive. Verify hashes again on the independent copy. No external upload or local-original deletion was scheduled.

Cloud cleanup removed only 878 whitelisted archived folders found in both recycle-bin stages; 32 other local archive folders were not found in the cloud bins. OneDrive Storage Metrics showed 151.05 GB free of 1024 GB afterward. Current results and current/recent checkpoints were retained.
'''
(ARCHIVE / 'PRESERVATION_README_20261023.txt').write_text(text, encoding='utf-8')
(HERE / 'PRESERVATION_README.txt').write_text(text, encoding='utf-8')
print(json.dumps(dict(verified=True, at_utc=datetime.now(timezone.utc).isoformat(),
                      instructions=str(ARCHIVE / 'PRESERVATION_README_20261023.txt'), originals_deleted=0)))
