"""Restore two actual maintenance archives into fresh, isolated directories."""
from pathlib import Path
import json
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from aspen_retention import atomic_json, now, read_json, restore

state = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\aspen_retention_20261023')
receipts = [(path, read_json(path)) for path in (state / 'receipts').glob('*.json')]
results = []
for kind in ('native', 'private'):
    eligible = [(path, record) for path, record in receipts
                if record['state'] == 'raw_removed' and record['kind'] == kind]
    assert eligible, 'No completed archive of type ' + kind
    path, record = min(eligible, key=lambda pair: sum(row['bytes'] for row in pair[1]['files']))
    destination = state / ('restore_test_' + kind + '_' + uuid.uuid4().hex[:8])
    result = restore(path, destination)
    assert result['verified'] and result['simulator_calls'] == result['existing_files_replaced'] == 0
    result['kind'] = kind
    results.append(result)
audit = dict(at_utc=now(), actual_restores=results, all_verified=True,
             files_verified=sum(row['files'] for row in results), experiment_process_actions=0)
atomic_json(HERE / 'actual_staged_restores.json', audit)
print(json.dumps(audit, indent=2))
