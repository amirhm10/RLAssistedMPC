"""Permit verified natural queue completion without changing sealed E1 workers."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def coordinator_completed(plan, read, identity):
    queue = plan['queue_coordinator']
    old = next(row for row in plan['protected'] if row['pid'] == queue['pid'])
    current = identity(old['pid'])
    if current == old:
        return False
    if current.get('alive') is not False:
        raise RuntimeError('Original coordinator identity changed or was reused')
    directory = Path(queue['directory'])
    recorded = read(directory / 'coordinator_identity.json')
    if any(recorded.get(key) != old.get(key) for key in ('pid', 'created')):
        raise RuntimeError('Coordinator completion belongs to another process')
    status = read(directory / 'coordinator_status.json')
    if not (status.get('state') == 'completed' and status.get('stage') == 'finished'
            and status.get('pid') == old['pid'] and status.get('queued_runs_dispatched') == 20
            and status.get('finished_at', '') >= plan['prepared_at']):
        raise RuntimeError('Original coordinator stopped without verified completion')
    jobs = read(directory / 'combined/manifest.json')['jobs']
    if len(jobs) != 10 or not all(job['state'] == 'completed' for job in jobs):
        raise RuntimeError('Combined queue has unfinished jobs')
    return True


def guarded_check(plan):
    import run_column_e1_process_recovery as recovery
    from column_review.common import process_identity
    import run_remaining_recovery
    run_remaining_recovery.queue_guard()
    if coordinator_completed(plan, recovery.read, process_identity):
        reduced = dict(plan, protected=[row for row in plan['protected']
                                       if row['pid'] != plan['queue_coordinator']['pid']])
        return ORIGINAL_CHECK(reduced)
    return ORIGINAL_CHECK(plan)


def main(directory):
    import run_column_e1_process_recovery as recovery
    from column_review.common import digest
    receipt = recovery.read(directory / 'supervisor_lifecycle.json')
    assert receipt['sha256'] == digest(Path(__file__))
    assert receipt['sealed_worker_source_sha256'] == recovery.dependencies()
    global ORIGINAL_CHECK
    ORIGINAL_CHECK = recovery.protected_check
    with patch.object(recovery, 'protected_check', guarded_check):
        return recovery.supervise(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    main(args.directory.resolve())
