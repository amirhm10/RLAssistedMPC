"""Guarded combined E1 continuation. This adapter cannot stop any process.

The review plan must be explicitly authorized and refreshed before use. The
former worker and all of its owned native apps must already be retired. Only
then can the ordinary-user supervisor run validation and production.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aspen_retention import attributes, digest, now, read_json, version
from column_review.common import process_identity
from run_column_e1_process_recovery import memory

HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1'
STATE = ROOT / 'Distillation/Results/state_recovery/20261024_combined_native_process'
PYTHON = ROOT / 'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'
BASE_PYTHON = Path(r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\pythonw.exe')


def read(path):
    path = Path(path)
    for _ in range(6):
        stamp = version(path)
        time.sleep(2.1)
        if version(path) != stamp:
            continue
        result = read_json(path)
        if version(path) == stamp:
            return result
    raise RuntimeError('Metadata remains changing: ' + str(path))


def write(path, value):
    from recover_markov_seed23 import write as atomic
    return atomic(Path(path), value)


def selected_only(base, job):
    """Change one operational row in memory, never the original plan file."""
    result = deepcopy(base)
    rows = [row for row in result['jobs'] if Path(row['job']) == Path(job)]
    if len(rows) != 1 or rows[0]['category'] != 'combined' or rows[0]['aspen_number'] != 16:
        raise RuntimeError('Only the original combined case 16 row is eligible')
    if rows[0]['state'] not in ('running', 'failed', 'ready'):
        raise RuntimeError('The selected combined row is not recoverable')
    if rows[0]['expected_matched_pairs'] != 320:
        raise RuntimeError('Combined comparison budget differs')
    rows[0]['state'] = 'ready'
    return result


def start_guard(plan, identity=process_identity, resources=memory, applications=None):
    if plan['state'] not in ('execution_authorized_after_user_approval',
                              'execution_authorized_after_confirmed_failure'):
        raise RuntimeError('The restart is prepared for review only; execution is not authorized')
    old = identity(plan['current_worker']['pid'])
    if old.get('alive') is not False:
        raise RuntimeError('The former worker is live, reused or inaccessible; do not launch')
    if applications is None:
        owners = read(OUTPUT / 'owned_aspen_apps.json')
        applications = []
        for row in owners:
            current = identity(row['pid'])
            if current.get('alive') is False:
                continue
            if current.get('alive') is True and current.get('created') != row['created']:
                continue
            applications.append(row)  # Inaccessible ownership also blocks launch.
    if applications:
        raise RuntimeError('Former combined-owned native applications remain')
    if resources()['headroom_bytes'] < plan['minimum_start_headroom_bytes']:
        raise RuntimeError('At least 6 GiB commitment headroom is required')


def protected_check(plan):
    for peer in plan['protected_peers']:
        actual = process_identity(peer['identity']['pid'])
        if actual == peer['identity']:
            continue
        folder = ROOT / ('Distillation/Results/state_recovery/20261021_remaining_active/e1/'
                         + peer['category'] + '/a1')
        if actual['alive'] is False:
            status = read(folder / 'status.json')
            if status.get('state') == 'completed' and status.get('trajectory_sha256'):
                if digest(folder / 'training.pkl') == status['trajectory_sha256']:
                    continue
        raise RuntimeError('A protected peer changed without verified completion')
    if process_identity(plan['retention_identity']['pid']) != plan['retention_identity']:
        raise RuntimeError('Automatic retention identity changed')


def verify():
    plan = read(HERE / 'restart_plan.json')
    if (plan['category'], plan['seed'], plan['case'], plan['expected_matched_pairs']) != ('combined', 7, 16, 320):
        raise RuntimeError('Restart scope differs')
    if plan['main_training_cycles'] != 300 or plan['decisions_per_window'] != 20:
        raise RuntimeError('Scientific schedule differs')
    if digest(Path(__file__)) != plan['worker_adapter_sha256']:
        raise RuntimeError('Reviewed recovery adapter changed')
    if digest(Path(plan['job'])) != plan['job_sha256']:
        raise RuntimeError('Original combined job changed')
    for filename, checksum in plan['source_sha256'].items():
        if digest(ROOT / filename) != checksum:
            raise RuntimeError('Sealed scientific/recovery source changed: ' + filename)
    if digest(Path(plan['backup_manifest'])) != plan['backup_manifest_sha256']:
        raise RuntimeError('Verified backup manifest changed')
    if digest(Path(plan['loader_protocol_file'])) != plan['loader_protocol_sha256']:
        raise RuntimeError('Reviewed fresh-process protocol changed')
    return plan


def frozen_traces(plan):
    for filename, checksum in plan['initial_trace_sha256'].items():
        if digest(Path(filename)) != checksum:
            raise RuntimeError('An already completed combined trace changed')


def worker(validation_steps):
    if validation_steps not in (0, 8):
        raise ValueError('Only an eight-step validation or full continuation is allowed')
    plan = verify()
    # All refusal checks precede numerical imports and simulator access.
    start_guard(plan)
    protected_check(plan)
    index = read(OUTPUT / 'training_checkpoints/latest.json')
    expected = plan['verified_checkpoint_steps'][0] + (0 if validation_steps else 8)
    if index['completed_steps'] != expected or len(index['agents']) != 4:
        raise RuntimeError('Refresh the reviewed checkpoint before any continuation')
    if validation_steps:
        if index['sha256'] != plan['checkpoint_latest_sha256']:
            raise RuntimeError('The prepared complete checkpoint changed')
    else:
        validation = read(STATE / 'validation.json')
        if not validation['passed'] or validation['index']['sha256'] != index['sha256']:
            raise RuntimeError('Production requires the verified eight-step checkpoint')
    frozen_traces(plan)
    import review_numeric_recovery
    review_numeric_recovery.install()
    import run_column_e1_budget as original
    import column_e1_budget as budget
    import column_e1_process_recovery as fresh
    import column_review.common as common
    previous_verify = original.verify

    def local_verify(directory):
        return selected_only(previous_verify(directory), plan['job'])

    def local_write(path, value):
        path = Path(path)
        if path.parent == original.DEFAULT and path.name.startswith('validation_'):
            path = STATE / 'validation.json'
        write(path, value)

    factory = fresh.manager_factory(budget.manager_class, plan['loader_protocol_sha256'], OUTPUT)
    with patch.object(original, 'read', read), patch.object(original, 'verify', local_verify), \
         patch.object(original, 'write', local_write), patch.object(common, 'read', read), \
         patch.object(budget, 'manager_class', factory), fresh.installed(OUTPUT):
        return original.worker(Path(plan['job']), validation_steps)


def supervise():
    from run_aspen_retention import single_instance
    from run_polymer_e3_recovery import environment
    from review_restart_recovery import lifetime, isolated_interactive_context
    plan = verify()
    start_guard(plan)
    protected_check(plan)
    identity = lifetime()
    if not isolated_interactive_context(identity):
        raise RuntimeError('Use an independent current-user scheduled supervisor')
    STATE.mkdir(parents=True, exist_ok=True)
    with single_instance(STATE):
        if (STATE / 'manifest.json').exists():
            raise RuntimeError('Inspect the existing recovery before any duplicate launch')
        write(STATE / 'supervisor_identity.json', identity)
        entry = dict(id='e1/combined', kind='e1', category='combined', seed=7, aspen_number=16,
                     job=plan['job'], directory=str(OUTPUT), state='pending', attempts=[],
                     expected_matched_pairs=320, protocol_id='column-e1-20-clock-strata-v1')
        manifest = dict(state='prepared', job=entry, jobs=[entry], initial_checkpoint_steps=plan['verified_checkpoint_steps'])
        pointer_path = ROOT / 'Distillation/Results/live_recovery/latest.json'
        stamp = version(pointer_path)
        pointer = read(pointer_path)
        before_pointer = deepcopy(pointer)
        manifests = pointer.setdefault('additional_manifests', [])
        if str(STATE / 'manifest.json') not in manifests:
            manifests.append(str(STATE / 'manifest.json'))
        if version(pointer_path) != stamp:
            raise RuntimeError('Progress pointer changed during registration')
        write(STATE / 'progress_pointer_before.json', before_pointer)
        write(pointer_path, pointer)
        logs = STATE / 'logs'
        logs.mkdir()
        for stage, steps in (('validation', 8), ('production', 0)):
            while memory()['headroom_bytes'] < plan['minimum_start_headroom_bytes']:
                protected_check(plan)
                manifest.update(state='waiting_for_memory', stage=stage, memory=memory(), updated_at=now())
                write(STATE / 'manifest.json', manifest)
                time.sleep(20)
            verify()
            protected_check(plan)
            start_guard(plan)
            started = now()
            command = [str(PYTHON), '-B', '-u', str(Path(__file__)), '--worker', '--validation-steps', str(steps)]
            log = logs / (stage + '.log')
            with log.open('xb') as stream:
                child = subprocess.Popen(command, cwd=ROOT,
                         env=dict(environment(), PYTHONHASHSEED='7', MPLBACKEND='Agg'),
                         stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                entry.update(state='running', pid=child.pid, started_at=started, log=str(log), error=None)
                entry['attempts'].append(dict(pid=child.pid, started_at=started, directory=str(OUTPUT), stage=stage))
                manifest.update(state='running', stage=stage)
                while True:
                    status = read(OUTPUT / 'status.json')
                    if status.get('started_at', '') >= started:
                        entry['pid'] = status['pid']
                        entry['attempts'][-1]['pid'] = status['pid']
                    manifest.update(updated_at=now(), memory=memory())
                    write(STATE / 'manifest.json', manifest)
                    if child.poll() is not None:
                        break
                    time.sleep(10)
            status = read(OUTPUT / 'status.json')
            expected = 'technical_checkpoint' if steps else 'completed'
            if child.returncode != 0 or status.get('state') != expected:
                entry.update(state='failed', error=status.get('error'), exit_code=child.returncode)
                manifest.update(state='failed', finished_at=now())
                write(STATE / 'manifest.json', manifest)
                return
            if steps:
                validation = read(STATE / 'validation.json')
                checkpoint = validation['index']
                if (not validation['passed'] or checkpoint['completed_steps'] != plan['verified_checkpoint_steps'][0] + 8
                        or len(checkpoint['agents']) != 4 or not validation['restore']['python_checksum_verified']
                        or not validation['restore']['native_checksum_verified']
                        or validation['restore']['completed_steps'] != plan['verified_checkpoint_steps'][0]):
                    raise RuntimeError('Eight-step combined full-state continuation did not verify')
                frozen_traces(plan)
                manifest['validation_passed'] = True
            else:
                if status.get('matched_pairs') != 320 or status.get('failed_pairs') != 0:
                    raise RuntimeError('The original combined comparison budget did not complete')
                entry.update(state='completed', finished_at=now(), exit_code=0)
                manifest.update(state='completed', finished_at=now())
                write(STATE / 'manifest.json', manifest)


def launch():
    plan = verify()
    start_guard(plan)
    protected_check(plan)
    if (STATE / 'scheduled.json').exists() or (STATE / 'manifest.json').exists():
        raise RuntimeError('An existing launch must be inspected before another attempt')
    import win32api
    import win32con
    import win32security
    import win32com.client
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    service = win32com.client.Dispatch('Schedule.Service')
    service.Connect()
    definition = service.NewTask(0)
    definition.Principal.UserId = user
    definition.Principal.LogonType = 3
    definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.ExecutionTimeLimit = 'PT0S'
    settings.MultipleInstances = 2
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    action = definition.Actions.Create(0)
    action.Path = str(BASE_PYTHON)
    action.Arguments = subprocess.list2cmdline(['-B', str(Path(__file__)), '--supervise'])
    action.WorkingDirectory = str(ROOT)
    task = 'RLAssistedMPC-CombinedRecovery-20261024'
    service.GetFolder('\\').RegisterTaskDefinition(task, definition, 2, user, None, 3)
    write(STATE / 'scheduled.json', dict(task_name=task, administrator_required=False,
          execution_time_limit='PT0S', triggers=0, launch_at_utc=now()))
    service.GetFolder('\\').GetTask(task).Run(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + name, action='store_true')
    parser.add_argument('--validation-steps', type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.validation_steps))
    elif args.supervise:
        try:
            supervise()
        except BaseException:
            write(STATE / 'supervisor_error.json', dict(at_utc=now(), traceback=traceback.format_exc()))
            raise
    else:
        launch()
