"""Finish a guarded partial retirement without signaling console processes."""
import argparse
from pathlib import Path
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import lifecycle
p = lifecycle.p
r = lifecycle.recovery.r
ALLOWED_SERVERS = {'sim_server.exe', 'am_task_server.exe'}


def prepare():
    plan = r.verify()
    intent = p.closed_json(HERE / 'retirement_intent.json')
    if any(p.same_generation(row, p.h.process(row['pid'])) for row in intent['failed_python_generations']):
        raise RuntimeError('The stalled Python generations must already be retired')
    servers, ignored = [], []

    def descendants(parent):
        if not p.same_generation(parent, p.h.process(parent['pid'])):
            return
        for process in psutil.process_iter(['name', 'ppid']):
            if process.info['ppid'] != parent['pid']:
                continue
            identity = p.h.process(process.pid)
            if identity.get('alive') is not True or identity.get('created') is None:
                continue
            row = dict(identity=identity, name=process.info['name'], parent=parent)
            if identity['created'] < parent['created'] or (process.info['name'] or '').lower() not in ALLOWED_SERVERS:
                ignored.append(dict(row, action='leave_untouched'))
                continue
            descendants(identity)
            servers.append(row)

    for native in intent['owned_native_generations']:
        descendants(native)
    result = dict(at_utc=p.now(), failed_plan_sha256=p.h.digest(HERE / 'plan.json'),
        failed_backup_sha256=plan['backup_manifest_sha256'], intent_sha256=p.h.digest(HERE / 'retirement_intent.json'),
        source_sha256=p.h.digest(Path(__file__)), servers=servers, ignored_descendants=ignored,
        owned_native_generations=intent['owned_native_generations'], other_healthy_workers_to_signal=[])
    p.immutable(HERE / 'remaining_native_retirement_plan.json', result)
    print(__import__('json').dumps(result, indent=2))


def execute():
    plan = r.verify()
    reviewed = p.closed_json(HERE / 'remaining_native_retirement_plan.json')
    if (p.h.digest(Path(__file__)) != reviewed['source_sha256']
        or p.h.digest(HERE / 'plan.json') != reviewed['failed_plan_sha256']
        or p.h.digest(plan['backup_manifest']) != reviewed['failed_backup_sha256']):
        raise RuntimeError('Reviewed cleanup or independent preservation changed')
    if (HERE / 'retirement.json').exists():
        raise FileExistsError('Never repeat completed cleanup')
    intent = p.closed_json(HERE / 'retirement_intent.json')
    if any(p.same_generation(row, p.h.process(row['pid'])) for row in intent['failed_python_generations']):
        raise RuntimeError('A former Python generation is still live')
    from recover_markov_seed23 import terminate_owned_identity
    actions = []
    for row in reviewed['servers']:
        if row['name'].lower() not in ALLOWED_SERVERS or row['identity']['created'] < row['parent']['created']:
            raise RuntimeError('Only generation-verified Aspen servers may be retired')
        actions.append(dict(identity=row['identity'], action=terminate_owned_identity(row['identity'])))
    for row in reviewed['owned_native_generations']:
        actions.append(dict(identity=row, action=terminate_owned_identity(row)))
    if any(p.same_generation(row, p.h.process(row['pid'])) for row in reviewed['owned_native_generations']):
        raise RuntimeError('An original owned native generation remains')
    saved = p.closed_json(HERE / 'preservation.json')
    full = p.closed_json(saved['manifest'])
    pool = Path(full['pending'][0]['request']).parent.parent
    roots = {Path(x) for point in full['pair']['checkpoints'] for x in point['private_roots']}
    for case in (15, 17):
        opened = [row for row in p.h.completed_records(pool / ('case' + str(case)) / 'native_process_recovery.jsonl')
                  if row.get('event') == 'opened_fresh_application']
        if opened:
            roots.add(Path(opened[-1]['model']).parent)
    destination = Path(saved['backup']) / 'retired_working'
    copied = []
    for folder in sorted(roots):
        if not folder.resolve().is_relative_to(r.OUTPUT.resolve()) and not folder.resolve().is_relative_to(pool.resolve()):
            raise RuntimeError('A working copy escaped the failed attempt')
        for source in sorted(folder.rglob('*')):
            if source.is_file():
                copied.append(lifecycle.preserve.copied_generation(source, destination / source.relative_to(ROOT).parent))
    for row in full['deferred_owned_working_files']:
        source = Path(row['source'])
        if any(source.is_relative_to(folder) for folder in roots):
            continue
        for attempt in range(30):
            try:
                copied.append(lifecycle.preserve.copied_generation(source, destination / source.relative_to(ROOT).parent))
                break
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(1)
    p.immutable(destination / 'preservation.json', dict(verified=True, at_utc=p.now(), copies=copied,
        formerly_locked_working_files_copied=True, all_originals_retained=True))
    p.immutable(HERE / 'retirement.json', dict(at_utc=p.now(), initial_partial_intent=str(HERE / 'retirement_intent.json'),
        completed_native_actions=actions, ignored_descendants=reviewed['ignored_descendants'],
        only_failed_combined_was_signaled=True, other_healthy_workers_interrupted=0,
        former_working_preservation=str(destination / 'preservation.json'),
        former_working_sha256=p.h.digest(destination / 'preservation.json')))
    r.protected_check(plan)
    r.start_guard(plan)
    print('Remaining owned Aspen cleanup and all formerly locked working backups verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'execute'))
    globals()[parser.parse_args().action]()
