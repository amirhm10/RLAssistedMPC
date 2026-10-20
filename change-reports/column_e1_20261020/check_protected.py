"""Verify original process identities and queue guards without COM attachment."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
from column_review.common import read, write, process_identity, now
from revision_comparisons.progress import snapshot
from run_column_e1 import guard_queue

baseline = read(Path(__file__).with_name('preflight.json'))
rows = []
for original in baseline['protected_workers']:
    current = process_identity(original['identity']['pid'])
    assert current == original['identity'], (original['job'], current)
    rows.append(dict(job=original['job'], **current, unchanged=True))
guard_queue()
current = snapshot()
assert current['repeat_queue_coordinator']['pid'] == baseline['coordinator']['pid']
assert current['repeat_queue_coordinator']['alive'] is True
result = dict(observed_at_utc=now(), preserved=rows, queue_guards_pass=True, current=current)
write(Path(__file__).with_name('protected_current.json'), result)
print(json.dumps({'preserved_workers':len(rows), 'queue':current['repeat_queue_coordinator'],
                  'current':[{k:r.get(k) for k in ('job','state','cycle','pid')} for r in current['jobs']
                             if r['state']=='running' and r['study'] not in current['historical_studies']]},indent=2))
