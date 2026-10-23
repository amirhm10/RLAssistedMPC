"""Recover the three failed cycle-190 E1 jobs, with a guarded memory queue."""
import argparse
from copy import deepcopy
from datetime import datetime
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_process'
PYTHON = ROOT / 'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'
REPORT = ROOT / 'change-reports/e1_recovery_20261023'
BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\e1_recovery_20261023')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    from recover_markov_seed23 import write as atomic
    return atomic(Path(path), value)


def dependencies():
    import run_column_e1_budget_lazy_recovery as previous
    from column_review.common import digest
    return dict(previous.dependencies(), **{name: digest(ROOT / name) for name in
                ('column_e1_process_recovery.py', 'run_column_e1_process_recovery.py')})


def memory():
    class Info(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                    ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
                     'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
                    (name, wintypes.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]
    record = Info(); record.cb = ctypes.sizeof(record)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetPerformanceInfo.argtypes = [ctypes.POINTER(Info), wintypes.DWORD]
    if not api.GetPerformanceInfo(ctypes.byref(record), record.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(committed_bytes=record.CommitTotal * record.PageSize,
                limit_bytes=record.CommitLimit * record.PageSize,
                headroom_bytes=(record.CommitLimit - record.CommitTotal) * record.PageSize)


def protected_check(plan):
    from column_review.common import process_identity
    for old in plan['protected']:
        current = process_identity(old['pid'])
        if current != old:
            # Natural completion is permitted, interruption or PID reuse is not.
            group = plan['protected_groups'].get(str(old['pid']))
            completed = False
            if group and current['alive'] is False:
                target = Path(group)
                for name in ('worker_status.json', 'status.json'):
                    if (target / name).exists() and read(target / name).get('state') == 'completed':
                        completed = True
            if not completed:
                raise RuntimeError('Protected identity changed: ' + str(old['pid']))
    import run_remaining_recovery
    run_remaining_recovery.queue_guard()


def prepare(directory):
    import run_column_e1_budget_lazy_recovery as previous
    from column_review.common import digest, owned_apps_remaining, process_identity, now
    from revision_comparisons.progress import snapshot
    old = previous.verify(previous.DEFAULT)
    native_test = ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_probe_a2/acceptance.json'
    assert read(native_test)['passed'] is True
    assert not directory.exists() and not BACKUP.exists()
    progress = snapshot()
    protected, protected_groups = {}, {}
    for row in progress['jobs']:
        if row['state'] != 'running' or row['study'] in progress['historical_studies']:
            continue
        protected[row['pid']] = process_identity(row['pid'])
        protected_groups[str(row['pid'])] = row['directory']
        for app in owned_apps_remaining(Path(row['directory']) / 'owned_aspen_apps.json'):
            protected[app['recorded']['pid']] = app['current']
            protected_groups[str(app['recorded']['pid'])] = row['directory']
    q = progress['repeat_queue_coordinator']
    protected[q['pid']] = process_identity(q['pid'])
    assert len(protected) == 27 and all(r['alive'] is True for r in protected.values())
    directory.mkdir(); (directory / 'logs').mkdir(); (directory / 'originals').mkdir()
    BACKUP.mkdir(parents=True)
    protocol = dict(id='column-e1-fresh-owned-process-v1', prepared_at=now(),
                    scope='Fresh private native process on document load; scientific state and bounds unchanged',
                    preserved_budget_sha256=read(previous.prior.DEFAULT / 'manifest.json')['protocol_sha256'],
                    preserved_lazy_sha256=old['protocol_sha256'], bootstrap_sha256=dependencies(),
                    minimum_start_headroom_bytes=6 * 1024**3, backup=str(BACKUP))
    write(directory / 'protocol.json', protocol)
    jobs = []
    for selected in old['jobs']:
        assert selected['category'] in ('weights', 'markov', 'residual') and selected['state'] == 'failed'
        out = Path(selected['directory']); status = read(out / 'status.json')
        assert status['state'] == 'failed' and process_identity(status['pid'])['alive'] is False
        assert not owned_apps_remaining(out / 'owned_aspen_apps.json')
        index = read(out / 'training_checkpoints/latest.json')
        assert index['completed_cycles'] == 190 and digest(out / 'training_checkpoints' / index['file']) == index['sha256']
        backup = BACKUP / selected['category']; backup.mkdir()
        for source in (out / 'training_checkpoints').glob('training_state_*.pkl'):
            shutil.copyfile(source, backup / source.name)
            assert digest(source) == digest(backup / source.name)
        shutil.copyfile(out / 'training_checkpoints/latest.json', backup / 'latest.json')
        local = directory / 'originals' / selected['category']; local.mkdir()
        for source in (out / 'status.json', Path(selected['log']), out / 'owned_aspen_apps.json'):
            shutil.copyfile(source, local / source.name)
        for name in ('sampling.json', 'decisions.jsonl', 'native_restore_checks.jsonl'):
            source = out / 'capture' / name
            if source.exists(): shutil.copyfile(source, local / name)
        job = deepcopy(selected)
        job.update(state='pending', attempts=[], prior_failure=deepcopy(selected), pid=None,
                   log=str(directory / 'logs' / (selected['category'] + '.log')), initial_checkpoint=index,
                   checkpoint_backup=str(backup), error=None)
        jobs.append(job)
    plan = dict(state='prepared', prepared_at=now(), protocol_sha256=digest(directory / 'protocol.json'),
                bootstrap_sha256=dependencies(), jobs=jobs, protected=list(protected.values()), protected_groups=protected_groups,
                queue_coordinator=q, expected_matched_pairs=640, memory_gate_bytes=6 * 1024**3)
    write(directory / 'manifest.json', plan)
    command = subprocess.list2cmdline([str(PYTHON), '-B', '-u', str(Path(__file__)), '--supervise', str(directory)])
    (directory / 'run.cmd').write_text('@echo off\nsetlocal\ntitle E1 recovery - fresh Aspen and memory queue\ncd /d "' + str(ROOT) + '"\n' + command + '\necho E1 recovery supervisor finished.\npause\n', encoding='utf-8')
    # Reuse the existing progress viewer instead of opening another CMD window.
    (directory / 'progress.cmd').write_text('@echo off\nexit /b 0\n', encoding='utf-8')
    print('Prepared three full-state E1 resumes, protected 27 identities, and verified local checkpoint backups.', flush=True)


def verify(directory):
    from column_review.common import digest
    import run_column_e1_budget_lazy_recovery as previous
    plan = read(directory / 'manifest.json')
    assert dependencies() == plan['bootstrap_sha256']
    assert digest(directory / 'protocol.json') == plan['protocol_sha256']
    previous.verify(previous.DEFAULT)
    return plan


def worker(directory, category, validation_steps=0):
    import review_numeric_recovery
    review_numeric_recovery.install()
    import run_column_e1_budget_lazy_recovery as previous
    import column_e1_budget as budget
    import column_e1_process_recovery as recovery
    from column_review.common import owned_apps_remaining
    plan = verify(directory)
    entry = next(j for j in plan['jobs'] if j['category'] == category)
    out = Path(entry['directory'])
    assert not owned_apps_remaining(out / 'owned_aspen_apps.json')
    original_verify = previous.verify
    def selected_verify(target):
        observed = original_verify(target)
        selected = next(j for j in observed['jobs'] if j['category'] == category)
        assert selected['state'] == 'failed'
        selected['state'] = 'ready'
        return observed
    factory = recovery.manager_factory(budget.manager_class, plan['protocol_sha256'], out)
    with patch.object(previous, 'verify', selected_verify), \
         patch.object(budget, 'manager_class', factory), recovery.installed(out):
        code = previous.worker(Path(entry['worker']), validation_steps)
    if validation_steps:
        index = read(out / 'training_checkpoints/latest.json')
        status = read(out / 'status.json')
        assert code == 0 and status['state'] == 'technical_checkpoint'
        assert status['resumed_from_step'] + validation_steps == index['completed_steps']
        assert not owned_apps_remaining(out / 'owned_aspen_apps.json')
        write(directory / f'validation_{category}_{index["completed_steps"]}.json',
              dict(passed=True, category=category, index=index,
                   restore=read(out / 'training_checkpoints/last_restore.json')))
    return code


def supervise(directory):
    from column_review.common import process_identity, now
    from run_distillation_phase2_repeats import aspen_file_lock
    import run_remaining_recovery as original
    with aspen_file_lock(directory / 'supervisor', directory / '_locks'):
        plan = verify(directory)
        assert plan['state'] == 'prepared'
        for category in ('weights', 'markov', 'residual'):
            assert list(directory.glob(f'validation_{category}_*.json'))
        protected_check(plan)
        plan.update(state='running', supervisor=process_identity(os.getpid()), started_at=now())
        pointer_path = ROOT / 'Distillation/Results/live_recovery/latest.json'
        pointer = read(pointer_path)
        pointer['additional_manifests'].append(str(directory / 'manifest.json'))
        write(pointer_path, pointer)
        running = []
        while running or any(j['state'] == 'pending' for j in plan['jobs']):
            protected_check(plan)
            resources = memory()
            # Stagger launches and require a completed restore before considering
            # another start; subsequent loop observations measure actual headroom.
            restored = all((Path(j['directory']) / 'training_checkpoints/last_restore.json').exists()
                           and (Path(j['directory']) / 'training_checkpoints/last_restore.json').stat().st_mtime
                               >= datetime.fromisoformat(j['started_at']).timestamp()
                           for j, child, stream in running)
            pending = next((j for j in plan['jobs'] if j['state'] == 'pending'), None)
            if pending and restored and resources['headroom_bytes'] >= plan['memory_gate_bytes']:
                job = pending; stream = Path(job['log']).open('xb')
                command = [str(PYTHON), '-B', '-u', str(Path(__file__)), '--worker', str(directory), '--category', job['category']]
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                job.update(state='running', pid=child.pid, started_at=now(), error=None)
                job['attempts'].append(dict(pid=child.pid, directory=job['directory'], started_at=job['started_at']))
                running.append((job, child, stream))
                print('Resuming', job['category'], 'from full checkpoint; memory headroom GiB', round(resources['headroom_bytes'] / 1024**3, 2), flush=True)
            for item in list(running):
                job, child, stream = item; status = read(Path(job['directory']) / 'status.json')
                if status.get('started_at', '') >= job['started_at']:
                    job['pid'] = status['pid']; job['attempts'][-1]['pid'] = status['pid']
                    job['identity'] = process_identity(status['pid'])
                if child.poll() is None: continue
                stream.close(); running.remove(item)
                job.update(state='completed' if child.returncode == 0 and status.get('state') == 'completed' else 'failed',
                           error=status.get('error'), exit_code=child.returncode, finished_at=now())
                job['attempts'][-1].update(state=job['state'], exit_code=child.returncode)
                print(job['category'], job['state'], flush=True)
            plan.update(updated_at=now(), memory=resources)
            write(directory / 'manifest.json', plan)
            write(directory / 'progress.json', dict(updated_at=now(), jobs=[dict(category=j['category'], state=j['state'], pid=j.get('pid')) for j in plan['jobs']], memory=resources))
            if running or any(j['state'] == 'pending' for j in plan['jobs']): time.sleep(30)
        plan.update(state='completed' if all(j['state'] == 'completed' for j in plan['jobs']) else 'failed', finished_at=now())
        write(directory / 'manifest.json', plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare', 'worker', 'supervise', 'launch'): actions.add_argument('--' + action, type=Path)
    parser.add_argument('--category', choices=('weights', 'markov', 'residual'))
    parser.add_argument('--validation-steps', type=int, default=0)
    args = parser.parse_args()
    for action in ('prepare', 'worker', 'supervise', 'launch'):
        target = getattr(args, action)
        if target:
            if action == 'worker': raise SystemExit(worker(target.resolve(), args.category, args.validation_steps))
            elif action == 'launch':
                import run_remaining_recovery
                run_remaining_recovery.launch(target.resolve())
            else: globals()[action](target.resolve())
