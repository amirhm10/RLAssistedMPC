"""Read-only queue, worker and case audit before the isolated column E1 job."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
from column_review.common import read, write, now, process_identity, check_job
from revision_comparisons.progress import snapshot
import continue_column_queue as queue
import win32com.client

def main():
    current = snapshot()
    directory = Path(read(ROOT / 'Distillation/Results/review_recovery/latest.json')['repeats'])
    groups = {name: read(directory / name / 'manifest.json') for name in ('residual', 'combined')}
    for group in groups.values():
        queue.validate_group(group)
    check_job(read(ROOT / 'Distillation/Results/column_review/20261018_recovered/bo/job.json'))
    protected = [dict(job=r['job'], directory=r['directory'], identity=process_identity(r['pid']))
                 for r in current['jobs'] if r['state'] == 'running' and r['study'] not in current['historical_studies']]
    threshold = read(ROOT / 'Polymer/Results/state_recovery/20261020_threshold_policy_recovery/manifest.json')
    protected.append(dict(job='polymer_threshold_recovery', directory=threshold['directory'], identity=process_identity(threshold['pid'])))
    for row in protected:
        assert row['identity']['alive'] is True and row['identity']['created'] is not None, row
    service = win32com.client.GetObject('winmgmts:')
    apps = [dict(pid=int(p.ProcessId), parent=int(p.ParentProcessId), created=str(p.CreationDate), name=str(p.Name))
            for p in service.ExecQuery("SELECT ProcessId,ParentProcessId,CreationDate,Name FROM Win32_Process WHERE Name='AspenModeler.exe'")]
    original = read(ROOT / 'Distillation/Results/column_review/20261018_recovered/e1_pilot/job.json')['aspen']
    from run_distillation_phase2_repeats import aspen_file_lock
    with aspen_file_lock(Path(original['aspen_path']), ROOT / 'Distillation/Results/_aspen_file_locks'):
        free_lease_verified = True
    forbidden = sorted({int(j['aspen_number']) for g in groups.values() for j in g['jobs']})
    assert 16 not in forbidden
    result = dict(observed_at_utc=now(), protected_workers=protected, coordinator=current['repeat_queue_coordinator'],
                  combined_reserved_cases=forbidden, combined_guards_pass=True, case16_lease_free=free_lease_verified,
                  candidate=original, aspen_processes=apps, current=current)
    write(Path(__file__).with_name('preflight.json'), result)
    print(json.dumps({k:result[k] for k in ('observed_at_utc','protected_workers','coordinator','combined_reserved_cases','case16_lease_free')}, indent=2))

if __name__ == '__main__':
    main()
