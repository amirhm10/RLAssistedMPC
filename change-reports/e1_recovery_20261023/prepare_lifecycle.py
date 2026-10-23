"""Register a supervisor-only guard; sealed scientific workers remain unchanged."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import run_column_e1_process_recovery as recovery
from column_review.common import digest

directory = recovery.DEFAULT
plan = recovery.verify(directory)
assert plan['state'] == 'prepared'
recovery.protected_check(plan)
for category in ('weights', 'markov', 'residual'):
    receipts = list(directory.glob(f'validation_{category}_*.json'))
    assert receipts and all(recovery.read(path)['passed'] is True for path in receipts)
source = HERE / 'e1_supervisor_lifecycle.py'
recovery.write(directory / 'supervisor_lifecycle.json', dict(source=str(source), sha256=digest(source),
    sealed_worker_source_sha256=recovery.dependencies(), at_utc=datetime.now(timezone.utc).isoformat(),
    purpose='Accept only verified natural completion of the original combined queue coordinator; no worker/source/protocol changes.'))
command = subprocess.list2cmdline([str(recovery.PYTHON), '-B', '-u', str(source), str(directory)])
target = directory / 'run.cmd'
original = target.read_text(encoding='utf-8')
assert 'run_column_e1_process_recovery.py" --supervise' in original
(directory / 'run_original.cmd.txt').write_text(original, encoding='utf-8')
target.write_text('@echo off\nsetlocal\ntitle E1 recovery - fresh Aspen and memory queue\ncd /d "' + str(ROOT) + '"\n' + command + '\necho E1 recovery supervisor finished.\npause\n', encoding='utf-8')
print('Registered tested supervisor lifecycle guard; all three full-state validations passed.', flush=True)
