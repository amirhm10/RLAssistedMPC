"""Read-only acceptance audit for the horizon-only recovery."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from aspen_retention import atomic_json, digest, read_json
import run_column_e1_horizon_recovery as recovery
from column_review.common import process_identity

plan = recovery.verify(recovery.DEFAULT)
recovery.protected_check(plan)
identity = read_json(recovery.DEFAULT / 'supervisor_identity.json')
assert identity['parent_is_scheduler_service'] and identity['session_id'] > 0
assert psutil.pid_exists(identity['pid'])
backup = read_json(recovery.BACKUP / 'preservation.json')
assert backup['verified']
for row in backup['files']:
    assert digest(Path(row['local_copy'])) == row['sha256']
for path, checksum in plan['initial_trace_sha256'].items():
    assert digest(Path(path)) == checksum
validation = read_json(recovery.DEFAULT / 'validation.json')
assert validation['passed'] and validation['index']['completed_steps'] == 116008
assert validation['restore']['completed_steps'] == 116000
status = read_json(recovery.OUTPUT / 'status.json')
assert plan['stage'] == 'production' and plan['validation_passed']
assert status['state'] in ('running', 'completed')
assert status['category'] == 'horizon' and status['case'] == 13
worker = process_identity(status['pid'])
assert worker['alive'] is True or status['state'] == 'completed'
restored = read_json(recovery.OUTPUT / 'training_checkpoints/last_restore.json')
assert restored['completed_steps'] >= 116008 and restored['python_checksum_verified']
assert restored['native_checksum_verified']
events = []
with (recovery.OUTPUT / 'native_process_recovery.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        if line.strip():
            events.append(json.loads(line))
python_restores = [event for event in events if event['event'] == 'full_python_restore_verified']
assert len(python_restores) >= 2
assert all(event['before'] == event['after'] for event in python_restores)
retention_config = read_json(recovery.RETENTION_CONFIG)
retention = read_json(Path(retention_config['state_root']) / 'status.json')
assert psutil.pid_exists(retention['pid'])
pointer = recovery.read(ROOT / 'Distillation/Results/live_recovery/latest.json')
before = read_json(recovery.DEFAULT / 'progress_pointer_before.json')
assert str(recovery.DEFAULT / 'manifest.json') in pointer['additional_manifests']
assert {key: value for key, value in before.items() if key != 'additional_manifests'} == {
    key: value for key, value in pointer.items() if key != 'additional_manifests'}
result = dict(at_utc=datetime.now(timezone.utc).isoformat(), worker=worker,
              worker_state=status['state'], supervisor=identity, validation_passed=True,
              initial_checkpoint_step=116000, validation_checkpoint_step=116008,
              production_restored_step=restored['completed_steps'],
              initial_comparisons=60, original_trace_hashes_preserved=len(plan['initial_trace_sha256']),
              verified_backup_files=len(backup['files']), full_python_restore_checks=len(python_restores),
              protected_processes_verified=len(plan['protected']),
              protected_workers=[dict(job=row['job'], identity=process_identity(row['identity']['pid']))
                                 for row in plan['protected'] if row['role'] == 'scientific_worker'],
              source_hashes_verified=len(plan['source_sha256']), queue_pointer_fields_preserved=True,
              retention_state=retention['state'], free_disk_bytes=retention['free_disk_bytes'])
atomic_json(HERE / 'acceptance.json', result)
print(json.dumps(result, indent=2))
