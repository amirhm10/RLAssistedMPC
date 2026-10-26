"""Retry only failed production startup from the already verified eight-step checkpoint."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recover_parallel as prior
import parallel_branches as p
import metadata_io

r = prior.r
BASE_VERIFY = r.verify
BASE_SCOPE = r.scope
r.STATE = ROOT / 'Distillation/Results/state_recovery/20261026_combined_metadata_retry'
r.SCRIPT = Path(__file__).resolve()


def scope(plan):
    if not plan.get('failed_startup_retry'):
        return BASE_SCOPE(plan)
    if tuple(plan[k] for k in ('category', 'case', 'seed', 'expected_matched_pairs',
        'decisions_per_window', 'main_training_cycles')) != ('combined', 16, 7, 320, 20, 300):
        raise RuntimeError('The original scientific budget must remain unchanged')
    if (plan['authorization'] != 'confirmed_failed_run_recovery'
            or not plan['failure_evidence']['confirmed_ownership_atomic_replace_failure']
            or plan['failed_worker'] != plan['failure_evidence']['parent_identity']):
        raise RuntimeError('Only the confirmed failed ownership startup may retry')


def verify():
    BASE_VERIFY()
    seal = p.closed_json(HERE / 'retry_seal.json')
    if p.h.digest(HERE / 'retry_plan.json') != seal['retry_plan_sha256']:
        raise RuntimeError('The fresh retry plan changed')
    plan = p.closed_json(HERE / 'retry_plan.json')
    scope(plan)
    for name, checksum in plan['adapter_sha256'].items():
        if p.h.digest(HERE / name) != checksum:
            raise RuntimeError('A reviewed retry adapter changed: ' + name)
    for path, checksum in ((plan['backup_manifest'], plan['backup_manifest_sha256']),
        (plan['failure_evidence']['path'], plan['failure_evidence']['sha256']),
        (plan['bounded_acceptance'], plan['bounded_acceptance_sha256'])):
        if p.h.digest(path) != checksum:
            raise RuntimeError('Fresh preservation/failure/bounded evidence changed')
    return plan


r.scope = scope
r.verify = verify


def worker():
    # Only the child entrypoint and ownership-status I/O change. Existing
    # scientific runner, native restoration and checkpoint protocols are reused.
    with metadata_io.installed([r.OUTPUT]), patch.object(p, '__file__', str(HERE / 'branch_metadata_worker.py')):
        return prior.execution.worker(0)


def supervise():
    from run_aspen_retention import single_instance
    from run_polymer_e3_recovery import environment
    from review_restart_recovery import lifetime, isolated_interactive_context
    plan = verify()
    r.protected_check(plan)
    identity = lifetime()
    if not isolated_interactive_context(identity):
        raise RuntimeError('An independent ordinary-user supervisor is required')
    with single_instance(r.STATE):
        if (r.STATE / 'manifest.json').exists():
            raise RuntimeError('Never duplicate a current or memory-gated retry')
        r.write(r.STATE / 'supervisor_identity.json', identity)
        entry = dict(id='e1/combined', kind='e1', category='combined', seed=7, aspen_number=16,
            directory=str(r.OUTPUT), job=plan['job'], expected_matched_pairs=320, state='pending',
            attempts=[plan['accepted_validation_attempt']], error=None)
        manifest = dict(state='prepared', stage='production', validation_passed=True,
            validation_reused_from=plan['bounded_acceptance'], recovery_plan=str(HERE / 'retry_plan.json'),
            job=entry, jobs=[entry], replay_archive=plan['original_replay_archive'])
        pointer_path = ROOT / 'Distillation/Results/live_recovery/latest.json'
        stamp = r.version(pointer_path)
        pointer = r.read(pointer_path)
        before = deepcopy(pointer)
        pointer.setdefault('additional_manifests', []).append(str(r.STATE / 'manifest.json'))
        if r.version(pointer_path) != stamp:
            raise RuntimeError('The viewer pointer changed; do not overwrite it')
        r.write(r.STATE / 'progress_pointer_before.json', before)
        r.write(pointer_path, pointer)
        while r.memory()['headroom_bytes'] < r.MINIMUM_HEADROOM:
            r.protected_check(plan)
            manifest.update(state='waiting_for_memory', memory=r.memory(), updated_at=r.now())
            r.write(r.STATE / 'manifest.json', manifest)
            time.sleep(20)
        verify()
        r.protected_check(plan)
        r.start_guard(plan)
        (r.STATE / 'logs').mkdir()
        log = r.STATE / 'logs/production.log'
        started = r.now()
        with log.open('xb') as stream:
            child = subprocess.Popen([str(r.PYTHON), '-B', '-u', str(r.SCRIPT), '--worker'], cwd=ROOT,
                env=dict(environment(), PYTHONHASHSEED='7', MPLBACKEND='Agg'), stdout=stream,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            entry.update(state='running', pid=child.pid, identity=r.process_identity(child.pid),
                started_at=started, log=str(log))
            entry['attempts'].append(dict(pid=child.pid, identity=entry['identity'], directory=str(r.OUTPUT),
                stage='production', started_at=started))
            manifest.update(state='running')
            while True:
                status = r.read(r.OUTPUT / 'status.json', {})
                if status.get('started_at', '') >= started:
                    entry.update(pid=status['pid'], identity=r.process_identity(status['pid']))
                    entry['attempts'][-1].update(pid=entry['pid'], identity=entry['identity'])
                manifest.update(updated_at=r.now(), memory=r.memory())
                r.write(r.STATE / 'manifest.json', manifest)
                if child.poll() is not None:
                    break
                time.sleep(10)
        status = r.read(r.OUTPUT / 'status.json', {})
        if (child.returncode != 0 or status.get('state') != 'completed'
                or status.get('matched_pairs') != 320 or status.get('failed_pairs') != 0):
            entry.update(state='failed', exit_code=child.returncode, error=status.get('error'))
            manifest.update(state='failed', finished_at=r.now())
            r.write(r.STATE / 'manifest.json', manifest)
            return
        entry.update(state='completed', exit_code=0, finished_at=r.now())
        manifest.update(state='completed', finished_at=r.now())
        r.write(r.STATE / 'manifest.json', manifest)
        r.protected_check(plan)


def launch():
    plan = verify()
    r.protected_check(plan)
    r.start_guard(plan)
    if (r.STATE / 'scheduled.json').exists() or (r.STATE / 'manifest.json').exists():
        raise RuntimeError('Never duplicate a current or memory-gated production retry')
    import win32api
    import win32con
    import win32security
    import win32com.client
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    scheduler = win32com.client.Dispatch('Schedule.Service')
    scheduler.Connect()
    task = scheduler.NewTask(0)
    task.Principal.UserId = user
    task.Principal.LogonType = 3
    task.Principal.RunLevel = 0
    task.Settings.Enabled = task.Settings.AllowDemandStart = True
    task.Settings.ExecutionTimeLimit = 'PT0S'
    task.Settings.MultipleInstances = 2
    task.Settings.DisallowStartIfOnBatteries = task.Settings.StopIfGoingOnBatteries = False
    action = task.Actions.Create(0)
    action.Path = str(r.BASE_PYTHON)
    action.Arguments = subprocess.list2cmdline(['-B', str(r.SCRIPT), '--supervise'])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-CombinedMetadataRetry-20261026'
    scheduler.GetFolder('\\').RegisterTaskDefinition(name, task, 2, user, None, 3)
    r.write(r.STATE / 'scheduled.json', dict(task_name=name, administrator_required=False,
        at_utc=r.now(), plan_sha256=p.h.digest(HERE / 'retry_plan.json'), production_only=True))
    scheduler.GetFolder('\\').GetTask(name).Run(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('worker', 'supervise', 'launch'):
        group.add_argument('--' + name, action='store_true')
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker())
    elif args.supervise:
        try:
            supervise()
        except BaseException:
            r.write(r.STATE / 'supervisor_error.json', dict(at_utc=r.now(), traceback=__import__('traceback').format_exc()))
            raise
    else:
        launch()
