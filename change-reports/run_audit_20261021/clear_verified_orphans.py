"""Clear only the two audited crashed workers' owned Aspen process trees."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from column_review.common import read, process_identity, owned_apps_remaining, now, write
from recover_markov_seed23 import terminate_owned_identity
from run_distillation_phase2_repeats import aspen_file_lock

ATTEMPTS = [ROOT / 'Distillation/Results/review_recovery/20261020_queue_case12/residual/residual_plain/s23/a1',
            ROOT / 'Distillation/Results/review_recovery/20261020_openblas/retry/markov_sg/s23/a4']
BIN = Path(r'C:\Program Files\AspenTech\AMSystem V14.0\Bin')
ALLOWED = {name.lower(): os.path.normcase(str((BIN / name).resolve()))
           for name in ('AspenModeler.exe', 'sim_server.exe', 'am_task_server.exe')}
ALLOWED['conhost.exe'] = os.path.normcase(str(Path(r'C:\Windows\System32\conhost.exe').resolve()))


def descendants(service, parent):
    records = []
    for p in service.ExecQuery(f'SELECT ProcessId,ParentProcessId,Name,ExecutablePath FROM Win32_Process WHERE ParentProcessId={parent["pid"]}'):
        name, path = str(p.Name).lower(), str(p.ExecutablePath)
        if name not in ALLOWED or os.path.normcase(str(Path(path).resolve())) != ALLOWED[name]:
            raise RuntimeError(f'Unverified descendant executable: {name} {path}')
        identity = process_identity(int(p.ProcessId))
        if identity['alive'] is not True or identity['created'] is None or identity['created'] < parent['created']:
            raise RuntimeError('Descendant lifetime cannot be verified')
        record = dict(identity, name=name, executable=path, parent_pid=int(p.ParentProcessId))
        records.extend(descendants(service, record))
        records.append(record)
    return records


def protected():
    audit = read(OUT / 'audit.json')
    records = [r['identity'] for r in audit['live_worker_health']]
    records += [audit['queue']['coordinator_identity'], audit['queue']['residual_supervisor_identity']]
    for record in records:
        current = process_identity(record['pid'])
        if current['alive'] is not True or current['created'] != record['created']:
            raise RuntimeError('A protected live process changed; reassess before cleanup')
    return records


def prepare():
    import win32com.client
    service = win32com.client.GetObject('winmgmts:')
    guard = protected()
    groups = []
    for attempt in ATTEMPTS:
        status, job = read(attempt / 'worker_status.json'), read(attempt / 'job.json')
        if process_identity(status['pid'])['alive'] is not False:
            raise RuntimeError('Failed worker is not confirmed absent')
        if job['seed'] != 23 or job['aspen_number'] not in (3, 8):
            raise RuntimeError('Attempt is outside the audited two-worker scope')
        apps = owned_apps_remaining(attempt / 'owned_aspen_apps.json')
        records = []
        for app in apps:
            parent = app['recorded']
            if app['current']['alive'] is not True or app['current']['created'] != parent['created']:
                raise RuntimeError('Owned Aspen creation time differs')
            process = list(service.ExecQuery(f'SELECT Name,ExecutablePath FROM Win32_Process WHERE ProcessId={parent["pid"]}'))
            if len(process) != 1 or str(process[0].Name).lower() != 'aspenmodeler.exe':
                raise RuntimeError('Owned application is not Aspen')
            if os.path.normcase(str(Path(str(process[0].ExecutablePath)).resolve())) != ALLOWED['aspenmodeler.exe']:
                raise RuntimeError('Owned Aspen executable path differs')
            records.extend(descendants(service, parent))
            records.append(dict(parent, name='aspenmodeler.exe', executable=str(process[0].ExecutablePath)))
        if not records:
            raise RuntimeError('No orphan remains; reassess rather than prepare a redundant action')
        groups.append(dict(attempt=str(attempt), failed_worker_pid=status['pid'], case=job['aspen_number'],
                           aspen_path=job['aspen_path'], records=records))
    if {r['pid'] for g in groups for r in g['records']} & {r['pid'] for r in guard}:
        raise RuntimeError('Cleanup overlaps a protected worker')
    return dict(prepared_at=now(), authorization='Existing author request to recover failed runs and preserve the automatic combined queue',
                protected_processes=guard, groups=groups)


def apply():
    saved = read(OUT / 'orphan_cleanup_plan.json')
    fresh = prepare()
    if fresh['groups'] != saved['groups']:
        raise RuntimeError('Concrete orphan targets changed after review')
    result = dict(started_at=now(), plan='orphan_cleanup_plan.json', actions=[])
    target = OUT / 'orphan_cleanup_applied.json'
    if target.exists():
        raise FileExistsError(target)
    with ExitStack() as stack:
        for group in saved['groups']:
            stack.enter_context(aspen_file_lock(Path(group['aspen_path']), ROOT / 'Distillation/Results/_aspen_file_locks'))
        protected()
        for group in saved['groups']:
            if process_identity(group['failed_worker_pid'])['alive'] is not False:
                raise RuntimeError('A failed worker is no longer absent')
        write(target, result)
        for group in saved['groups']:
            for record in group['records']:
                action = terminate_owned_identity(record)
                result['actions'].append(dict(record=record, action=action, case=group['case']))
                write(target, result)
            if owned_apps_remaining(Path(group['attempt']) / 'owned_aspen_apps.json'):
                raise RuntimeError('Orphan cleanup did not verify')
        result.update(finished_at=now(), state='cleared', protected_processes_after=protected(),
                      scientific_sources_changed=False, queued_manifests_changed=False)
        write(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    value = apply() if args.apply else prepare()
    if not args.apply:
        path = OUT / 'orphan_cleanup_plan.json'
        if path.exists():
            raise FileExistsError(path)
        write(path, value)
    print(json.dumps(value, indent=2))
