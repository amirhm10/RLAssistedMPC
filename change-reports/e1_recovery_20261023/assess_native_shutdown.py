"""Match Windows faults to exact disposable validation identities and saves."""
from datetime import datetime
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_process'
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
plan = read(PLAN / 'manifest.json')
opened = {}
for job in plan['jobs']:
    path = Path(job['directory']) / 'native_process_recovery.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('event') == 'opened_fresh_application':
            identity = row['current']
            opened[identity['pid'], identity['created']] = dict(category=job['category'], **row)
validations = [read(path) for path in PLAN.glob('validation_*.json')]
events = read(HERE / 'windows_events_since_recovery.json')
matched = []
for event in events['events']:
    if event['id'] != 1000:
        continue
    pid = int(re.search(r'Faulting process id:\s*(0x[0-9a-fA-F]+)', event['message'])[1], 16)
    created = int(re.search(r'Faulting application start time:\s*(0x[0-9a-fA-F]+)', event['message'])[1], 16)
    identity = (pid, created)
    assert identity in opened, identity
    assert identity not in {(r['pid'], r['created']) for r in plan['protected']}
    app = opened[identity]
    at = datetime.fromisoformat(event['at']).timestamp()
    receipt = min((r for r in validations if r['category'] == app['category']),
                  key=lambda r: abs(r['index']['saved_at_unix'] - at))
    delay = at - receipt['index']['saved_at_unix']
    assert receipt['passed'] is True and 0 <= delay < 30
    matched.append(dict(pid=pid, created=created, category=app['category'],
                        saved_step=receipt['index']['completed_steps'], saved_checkpoint_passed=True,
                        fault_at=event['at'], seconds_after_checkpoint_save=delay,
                        original_protected_process=False, fault_message=event['message']))
assert len(matched) == 4
result = dict(application_error_events=4, total_error_reporting_records=events['count'],
              exact_pid_and_creation_matches=True, original_protected_processes_affected=0,
              production_recovery_errors_in_this_event_sample=0,
              matches=matched, inference='All four native faults occurred in disposable validation applications within 30 seconds after a verified full checkpoint save, during final cleanup. Cause remains unresolved; successful restart/trace checks do not certify clean native shutdown or multi-day stability.')
(HERE / 'native_shutdown_assessment.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(application_errors=len(matched), original_protected_processes_affected=0,
                      validation_checkpoints_preserved=True, shutdown_issue_unresolved=True)))
