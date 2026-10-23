"""Verify original process identities, frozen evidence and completed archives."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from aspen_retention import atomic_json, digest, now, read_json


def observed(pid):
    try:
        process = psutil.Process(pid)
        cpu = process.cpu_times()
        return dict(pid=pid, alive=True, created=process.create_time(), cpu=cpu.user+cpu.system)
    except psutil.NoSuchProcess:
        return dict(pid=pid, alive=False)


def completion(row):
    directory = Path(row['directory'])
    name = 'status.json' if row['study'] == 'E1_column' else 'worker_status.json'
    path = directory / name
    # Only inspect a dead worker's settled final record, never a mutable queue.
    if time.time()-path.stat().st_mtime < 30:
        return dict(state='awaiting settled completion record')
    status = read_json(path)
    result = dict(state=status.get('state'), finished_at=status.get('finished_at'), error=status.get('error'),
                  error_type=status.get('error_type'), traceback=status.get('traceback'))
    if result['state'] == 'completed':
        file = status.get('result_file') or status.get('results_file')
        result['result_file_exists'] = Path(file).is_file() if file else None
    return result


def before():
    prior = read_json(ROOT / 'change-reports/e1_disk_20261023/post_cleanup_audit.json')
    assessment = read_json(ROOT / 'change-reports/e1_recovery_20261023/recovery_assessment.json')
    directories = {(row['study'], row['job']): row['directory'] for row in assessment['jobs']}
    rows = []
    for old in prior['preserved_workers']:
        record = dict(study=old['study'], job=old['job'], directory=directories[(old['study'],old['job'])],
                      identity=observed(old['pid']))
        if record['identity']['alive']:
            assert record['identity']['created'] == old['created']
        else:
            record['completion'] = completion(record)
            assert record['completion']['state'] == 'completed', record
        rows.append(record)
    evidence = read_json(ROOT / 'change-reports/e1_recovery_20261023/partial_results_audit.json')
    traces = evidence['trace_artifacts']
    assert all(digest(Path(row['path'])) == row['sha256'] for row in traces)
    config = read_json(HERE / 'config.json')
    assert all(digest(ROOT / path) == sha for path,sha in config['scientific_sha256'].items())
    result = dict(at_utc=now(), workers=rows, combined_coordinator=observed(36308),
                  frozen_trace_count=len(traces), frozen_trace_hashes_verified=True,
                  scientific_files_verified=len(config['scientific_sha256']))
    atomic_json(HERE / 'before_runtime.json', result)
    print(json.dumps(dict(at_utc=result['at_utc'], live_workers=sum(row['identity']['alive'] for row in rows),
                          completed_before_maintenance=sum(not row['identity']['alive'] for row in rows),
                          combined_coordinator=result['combined_coordinator'], traces_verified=len(traces))))


def after():
    baseline = read_json(HERE / 'before_runtime.json')
    rows = []
    failed_runs = []
    for old in baseline['workers']:
        current = observed(old['identity']['pid'])
        record = dict(study=old['study'], job=old['job'], current=current)
        if old['identity']['alive'] and current['alive']:
            assert current['created'] == old['identity']['created'], record
            record['cpu_delta_seconds'] = current['cpu']-old['identity']['cpu']
            assert record['cpu_delta_seconds'] > 0, record
        elif not current['alive']:
            record['completion'] = completion(old)
            assert record['completion']['state'] in ('completed', 'failed'), record
            if record['completion']['state'] == 'failed':
                # Report failures explicitly; do not turn their disappearance
                # into an assertion that all experiments remain healthy.
                failed_runs.append(dict(job=old['job'], study=old['study'], **record['completion']))
        rows.append(record)
    queue = observed(36308)
    if baseline['combined_coordinator']['alive'] and queue['alive']:
        assert queue['created'] == baseline['combined_coordinator']['created']
    elif not queue['alive']:
        assert all(row.get('completion',{}).get('state') == 'completed'
                   for row in rows if row['study'] == 'column_repeats')
    config = read_json(HERE / 'config.json')
    assert all(digest(ROOT / path) == sha for path,sha in config['scientific_sha256'].items())
    assert all(digest(ROOT / path) == sha for path,sha in config['bootstrap_sha256'].items())
    traces = read_json(ROOT / 'change-reports/e1_recovery_20261023/partial_results_audit.json')['trace_artifacts']
    assert all(digest(Path(row['path'])) == row['sha256'] for row in traces)
    state = Path(config['state_root'])
    status = read_json(state / 'status.json')
    identity = read_json(state / 'worker_identity.json')
    live = observed(identity['pid'])
    assert live['alive'] and identity['parent_is_scheduler_service'] and identity['session_id'] > 0
    receipts = [read_json(p) for p in (state/'receipts').glob('*.json')]
    completed = [r for r in receipts if r['state'] == 'raw_removed']
    kinds = {r['kind'] for r in completed}
    assert kinds == {'native','private'}, kinds
    for record in completed:
        assert digest(Path(record['compressed'])) == record['compressed_sha256']
        assert record['restored_hashes_verified'] is True
        assert not Path(record['original_directory']).exists()
    assert len(status['jobs']) == 5
    assert all(len(row.get('checkpoints',[])) == 2 for row in status['jobs'])
    protected = []
    for row in status['jobs']:
        protected.append(dict(job=row['label'], steps=[p['completed_steps'] for p in row['checkpoints']],
                              native_directories_present=all(Path(p['native_directory']).is_dir() for p in row['checkpoints'])))
    assert all(row['native_directories_present'] for row in protected)
    assert all(Path(path).is_dir() for path in config['pinned_native_directories'])
    result = dict(at_utc=now(), worker_identity=identity, processes=rows, combined_coordinator=queue,
                  scientific_source_hashes_unchanged=True, frozen_trace_hashes_unchanged=True,
                  frozen_trace_count=len(traces), current_checkpoints=protected,
                  completed_archives_checked=len(completed), completed_archive_kinds=sorted(kinds),
                  live_original_workers=sum(row['current']['alive'] for row in rows),
                  failed_runs=failed_runs, pinned_native_directories_preserved=len(config['pinned_native_directories']),
                  retention_totals=status['totals'], resources=read_json(state/'resources.json'),
                  status_updated_at=status['updated_at'], free_disk_bytes=status['free_disk_bytes'])
    atomic_json(HERE / 'activation_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('before','after'))
    args = parser.parse_args()
    before() if args.mode == 'before' else after()
