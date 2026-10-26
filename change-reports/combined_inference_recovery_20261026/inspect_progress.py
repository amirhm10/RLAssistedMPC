"""Take a small, generation-aware recovery snapshot without simulator calls."""
from pathlib import Path
from datetime import datetime
import hashlib
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'change-reports/combined_parallel_20261024'))
import parallel_branches as p

STATE = ROOT / 'Distillation/Results/state_recovery/20261026_combined_inference'
OUTPUT = ROOT / 'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1'


def inspect():
    paths = {'manifest': STATE / 'manifest.json', 'status': OUTPUT / 'status.json',
             'index': OUTPUT / 'training_checkpoints/latest.json',
             'restore': OUTPUT / 'training_checkpoints/last_restore.json'}
    paths.update({name: STATE / filename for name, filename in
                  [('validation', 'validation.json'), ('pool8', 'parallel_pool_8.json'),
                   ('pool0', 'parallel_pool_0.json')]
                  if (STATE / filename).is_file()})
    values = p.h.stable_batch(paths)
    manifest = values['manifest']
    job = manifest['job']
    identities = [job.get('identity')]
    pools = []
    for name in ('pool8', 'pool0'):
        if name not in values:
            continue
        directory = Path(values[name]['directory'])
        decisions = (p.h.completed_records(directory / 'decisions.jsonl')
                     if (directory / 'decisions.jsonl').is_file() else [])
        children = []
        for case in (15, 17):
            child = directory / ('case' + str(case))
            if (child / 'identity.json').is_file():
                identity = p.closed_json(child / 'identity.json')
                identities.append(identity)
                requests = list(child.glob('request_*.json'))
                responses = list(child.glob('done_request_*.json'))
                pending = [p.closed_json(path) for path in requests
                           if not (path.parent / ('done_' + path.name)).exists()]
                actual = p.h.process(identity['pid'])
                children.append(dict(case=case, expected_identity=identity, actual=actual,
                    generation_is_live=p.same_generation(identity, actual),
                    requests=len(requests), responses=len(responses),
                    pending=[{k: row.get(k) for k in ('source', 'steps', 'snapshot', 'output')}
                             for row in pending]))
        pools.append(dict(name=name, directory=str(directory), children=children,
            completed_pairs=[{k: row.get(k) for k in ('channel', 'step', 'elapsed_seconds',
                'full_python_state_before', 'full_python_state_after', 'source_map_unchanged')}
                for row in decisions]))
    identities = [row for row in identities if row]
    before = [p.h.process(row['pid']) for row in identities]
    time.sleep(10)
    after = [p.h.process(row['pid']) for row in identities]
    records = p.h.completed_records(OUTPUT / 'capture/decisions.jsonl')
    events = p.h.completed_records(OUTPUT / 'native_process_recovery.jsonl')
    restores = [row for row in events if row.get('event') == 'full_python_restore_verified'
                and row.get('completed_steps', 0) >= 118800]
    fresh_restores = []
    baseline = HERE / 'restore_journal_baseline.json'
    if STATE.name == '20261026_combined_metadata_retry' and baseline.exists():
        prior = p.closed_json(baseline)
        with p.h.open_read(OUTPUT / 'native_process_recovery.jsonl') as stream:
            if hashlib.sha256(stream.read(prior['bytes'])).hexdigest() != prior['sha256']:
                raise RuntimeError('The preserved restore journal prefix changed')
        fresh_restores = [row for row in events[prior['complete_records']:]
            if row.get('event') == 'full_python_restore_verified'
            and row.get('completed_steps') == 118808 and row.get('before') == row.get('after')]
        if p.h.attributes(OUTPUT / 'training_checkpoints/last_restore.json')['mtime'] / 1e9 < datetime.fromisoformat(job['started_at']).timestamp():
            fresh_restores = []
    log = Path(job['log'])
    with p.h.open_read(log) as stream:
        log_tail = stream.read().decode('utf-8', errors='replace').splitlines()[-4:]
    index = values['index']
    restore = values['restore']
    receipt = dict(at_utc=p.now(), state=manifest.get('state'), stage=manifest.get('stage'),
        validation_passed=manifest.get('validation_passed'), error=job.get('error'),
        worker_identity=job.get('identity'), status=values['status'],
        checkpoint={k: index.get(k) for k in ('completed_steps', 'completed_cycles', 'agents', 'sha256')},
        restore={k: restore.get(k) for k in ('completed_steps', 'native_checksum_verified',
                                          'python_checksum_verified', 'agents')},
        current_generation_restore_verified=bool(fresh_restores),
        full_restore_events=[dict(at_utc=row.get('at_utc'), completed_steps=row.get('completed_steps'),
            before_equals_after=row.get('before') == row.get('after')) for row in restores],
        primary_pairs=len(records), latest_primary=records[-1] if records else None,
        original_314_independently_preserved=True, pools=pools,
        process_samples=[dict(expected=identity, before=first, after=last)
                         for identity, first, last in zip(identities, before, after)],
        memory=p.h.memory(), free_disk_bytes=p.shutil.disk_usage(ROOT.anchor).free,
        native_processes=p.h.native_processes(), log_tail=log_tail)
    name = 'progress_' + p.now().replace(':', '').replace('.', '_') + '.json'
    p.immutable(HERE / name, receipt)
    small = {key: receipt[key] for key in ('at_utc', 'state', 'stage', 'validation_passed',
        'error', 'worker_identity', 'checkpoint', 'restore', 'full_restore_events', 'current_generation_restore_verified',
        'primary_pairs', 'pools', 'memory', 'free_disk_bytes', 'log_tail')}
    small['receipt'] = str(HERE / name)
    print(__import__('json').dumps(small, indent=2))


if __name__ == '__main__':
    inspect()
