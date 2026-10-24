"""Resume only the retired failed weights worker with generation-safe ownership."""
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
from aspen_retention import digest, now, read_json, version
from column_review.common import process_identity
from run_column_e1_process_recovery import memory
import ownership_fix

HERE = Path(__file__).resolve().parent
CATEGORY = 'weights'
CASE = 14
SCRIPT = Path(__file__).resolve()
STATE = ROOT / 'Distillation/Results/state_recovery/20261024_weights_pid_reuse'
ORIGINAL = ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_process'
OUTPUT = ROOT / 'Distillation/Results/state_recovery/20261021_remaining_active/e1/weights/a1'
PYTHON = ROOT / 'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'
BASE_PYTHON = Path(r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\pythonw.exe')
MINIMUM_HEADROOM = 6 * 1024**3


def read(path, default=None):
    path = Path(path)
    for _ in range(6):
        try:
            stamp = version(path)
        except FileNotFoundError:
            return default
        time.sleep(2.1)
        if version(path) != stamp:
            continue
        value = read_json(path)
        if version(path) == stamp:
            return value
    raise RuntimeError('Metadata remains changing: ' + str(path))


def write(path, value):
    from recover_markov_seed23 import write as atomic
    return atomic(Path(path), value)


def scope(plan):
    if (plan['category'], plan['case'], plan['seed'], plan['expected_matched_pairs'],
            plan['decisions_per_window'], plan['main_training_cycles']) != (CATEGORY, CASE, 7, 80, 20, 300):
        raise RuntimeError('Recovery scope or scientific budget differs')
    if plan['authorization'] != 'confirmed_failed_run_recovery':
        raise RuntimeError('Only an actually failed weights run is authorized')


def verify():
    plan = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
    scope(plan)
    for name, checksum in plan['source_sha256'].items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError('Sealed scientific/recovery source differs: ' + name)
    for name, checksum in plan['adapter_sha256'].items():
        if digest(HERE / name) != checksum:
            raise RuntimeError('Reviewed local recovery adapter differs: ' + name)
    for path, checksum in ((plan['job'], plan['job_sha256']),
                           (plan['backup_manifest'], plan['backup_manifest_sha256']),
                           (str(ORIGINAL / 'protocol.json'), plan['original_process_protocol_sha256'])):
        if digest(Path(path)) != checksum:
            raise RuntimeError('Recovery job/protocol/preservation seal differs')
    return plan


def owned_remaining():
    phase = read(OUTPUT / 'ownership_status.json', {})
    if phase.get('phase') in ('creating', 'unverified_creation'):
        raise RuntimeError('Native creation ownership is unresolved')
    result = []
    for record in read(OUTPUT / 'owned_aspen_apps.json', []):
        current = process_identity(record['pid'])
        if current['alive'] is False:
            continue
        if current['alive'] is True and current['created'] != record['created']:
            continue
        result.append(dict(recorded=record, current=current))
    return result


def start_guard(plan, resources=memory, identity=process_identity, applications=None, status=None):
    scope(plan)
    if identity(plan['failed_worker']['pid']).get('alive') is not False:
        raise RuntimeError('Former weights worker is live, reused or inaccessible')
    if status is None:
        status = read(OUTPUT / 'status.json', {})
    if status.get('pid') and status['pid'] != plan['failed_worker']['pid']:
        current = identity(status['pid'])
        if current.get('alive') is not False:
            raise RuntimeError('Another weights worker is live or inaccessible; do not duplicate it')
    if applications is None:
        applications = owned_remaining()
    if applications:
        raise RuntimeError('The failed weights worker still owns native applications')
    if resources()['headroom_bytes'] < MINIMUM_HEADROOM:
        raise RuntimeError('At least 6 GiB commitment headroom is required')


def protected_check(plan):
    for peer in plan['protected']:
        current = process_identity(peer['identity']['pid'])
        if current == peer['identity']:
            continue
        status = read(Path(peer['directory']) / 'status.json', {})
        if (current['alive'] is False and status.get('state') == 'completed'
                and status.get('trajectory_sha256')
                and digest(Path(peer['directory']) / 'training.pkl') == status['trajectory_sha256']):
            continue
        raise RuntimeError('A protected peer changed without verified completion: ' + peer['category'])
    if process_identity(plan['retention_identity']['pid']) != plan['retention_identity']:
        raise RuntimeError('Automatic retention identity changed')


def preserved_traces(plan, after_validation=False):
    replayed = []
    for trace in plan['traces']:
        if digest(Path(trace['backup'])) != trace['sha256']:
            raise RuntimeError('A preserved original comparison changed')
        if trace['committed']:
            if digest(Path(trace['source'])) != trace['sha256']:
                raise RuntimeError('A checkpoint-prefix comparison changed')
        elif after_validation:
            candidates = list((OUTPUT / 'capture/replayed_suffixes').glob('*/traces/' + Path(trace['source']).name))
            matching = [str(path) for path in candidates if digest(path) == trace['sha256']]
            if not matching:
                raise RuntimeError('Post-checkpoint comparison was not preserved in the replay archive')
            replayed.extend(matching)
    return replayed


def worker(validation_steps):
    if validation_steps not in (0, 8):
        raise ValueError('Only eight validation steps or full continuation are permitted')
    plan = verify()
    start_guard(plan)
    protected_check(plan)
    index = read(OUTPUT / 'training_checkpoints/latest.json')
    expected = plan['initial_steps'] + (0 if validation_steps else 8)
    if index['completed_steps'] != expected or len(index['agents']) != 1:
        raise RuntimeError('The verified weights recovery checkpoint differs')
    if validation_steps and index['sha256'] != plan['initial_checkpoint_sha256']:
        raise RuntimeError('The prepared full checkpoint changed')
    if not validation_steps:
        result = read(STATE / 'validation.json')
        if not result['passed'] or result['index']['sha256'] != index['sha256']:
            raise RuntimeError('Production requires the verified bounded checkpoint')
    preserved_traces(plan, after_validation=not validation_steps)
    # No numerical bootstrap or native access occurs before the launch guards.
    import review_numeric_recovery
    review_numeric_recovery.install()
    import run_column_e1_process_recovery as original
    import run_column_e1_budget_lazy_recovery as lazy
    import run_column_e1_budget as budget_runner
    import column_e1_process_recovery as fresh
    import column_review.common as common

    def redirected_write(path, value):
        path = Path(path)
        if path.parent == ORIGINAL and path.name.startswith('validation_' + CATEGORY + '_'):
            path = STATE / 'validation.json'
        elif path.name.startswith('validation_') and path.parent != STATE:
            path = STATE / 'legacy_receipts' / path.name
        write(path, value)

    write(STATE / ('worker_' + str(validation_steps) + '_identity.json'),
          dict(identity=process_identity(__import__('os').getpid()), at_utc=now(),
               source_sha256=plan['adapter_sha256'], existing_scientific_protocols_unchanged=True))
    with patch.object(original, 'read', read), patch.object(original, 'write', redirected_write), \
         patch.object(lazy, 'read', read), patch.object(lazy, 'write', redirected_write), \
         patch.object(budget_runner, 'read', read), patch.object(budget_runner, 'write', redirected_write), \
         patch.object(common, 'read', read), ownership_fix.installed(fresh, read, process_identity):
        return original.worker(ORIGINAL, CATEGORY, validation_steps)


def supervise():
    from run_aspen_retention import single_instance
    from run_polymer_e3_recovery import environment
    from review_restart_recovery import lifetime, isolated_interactive_context
    plan = verify()
    protected_check(plan)
    identity = lifetime()
    if not isolated_interactive_context(identity):
        raise RuntimeError('An independent current-user supervisor is required')
    STATE.mkdir(parents=True, exist_ok=True)
    with single_instance(STATE):
        if (STATE / 'manifest.json').exists():
            raise RuntimeError('Inspect the existing supervisor before any duplicate launch')
        write(STATE / 'supervisor_identity.json', identity)
        entry = dict(id='e1/' + CATEGORY, kind='e1', category=CATEGORY, seed=7, aspen_number=CASE,
                     directory=str(OUTPUT), job=plan['job'], expected_matched_pairs=80,
                     state='pending', attempts=[], error=None)
        manifest = dict(state='prepared', job=entry, jobs=[entry], recovery_plan=str(HERE / 'plan.json'))
        pointer_path = ROOT / 'Distillation/Results/live_recovery/latest.json'
        stamp = version(pointer_path)
        pointer = read(pointer_path)
        before = deepcopy(pointer)
        manifests = pointer.setdefault('additional_manifests', [])
        if str(STATE / 'manifest.json') not in manifests:
            manifests.append(str(STATE / 'manifest.json'))
        if version(pointer_path) != stamp:
            raise RuntimeError('Progress pointer changed during registration')
        write(STATE / 'progress_pointer_before.json', before)
        write(pointer_path, pointer)
        (STATE / 'logs').mkdir()
        for stage, steps in (('validation', 8), ('production', 0)):
            while memory()['headroom_bytes'] < MINIMUM_HEADROOM:
                protected_check(plan)
                manifest.update(state='waiting_for_memory', stage=stage, memory=memory(), updated_at=now())
                write(STATE / 'manifest.json', manifest)
                time.sleep(20)
            verify()
            protected_check(plan)
            start_guard(plan)
            log = STATE / 'logs' / (stage + '.log')
            started = now()
            command = [str(PYTHON), '-B', '-u', str(SCRIPT), '--worker', '--validation-steps', str(steps)]
            with log.open('xb') as stream:
                child = subprocess.Popen(command, cwd=ROOT,
                          env=dict(environment(), PYTHONHASHSEED='7', MPLBACKEND='Agg'),
                          stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                entry.update(state='running', pid=child.pid, identity=process_identity(child.pid),
                             started_at=started, log=str(log), error=None)
                entry['attempts'].append(dict(pid=child.pid, identity=entry['identity'], directory=str(OUTPUT),
                                              stage=stage, started_at=started))
                manifest.update(state='running', stage=stage)
                while True:
                    status = read(OUTPUT / 'status.json', {})
                    if status.get('started_at', '') >= started:
                        entry.update(pid=status['pid'], identity=process_identity(status['pid']))
                        entry['attempts'][-1].update(pid=entry['pid'], identity=entry['identity'])
                    manifest.update(updated_at=now(), memory=memory())
                    write(STATE / 'manifest.json', manifest)
                    if child.poll() is not None:
                        break
                    time.sleep(10)
            status = read(OUTPUT / 'status.json', {})
            expected = 'technical_checkpoint' if steps else 'completed'
            if child.returncode != 0 or status.get('state') != expected:
                entry.update(state='failed', exit_code=child.returncode, error=status.get('error'))
                manifest.update(state='failed', finished_at=now())
                write(STATE / 'manifest.json', manifest)
                return
            if steps:
                result = read(STATE / 'validation.json')
                if (not result['passed'] or result['index']['completed_steps'] != plan['initial_steps'] + 8
                        or result['restore']['completed_steps'] != plan['initial_steps']
                        or not result['restore']['native_checksum_verified']
                        or not result['restore']['python_checksum_verified'] or len(result['index']['agents']) != 1):
                    raise RuntimeError('Eight-step full-state validation did not verify')
                manifest.update(validation_passed=True, replay_archive=preserved_traces(plan, after_validation=True))
                entry.update(state='pending', exit_code=0)
            else:
                if status.get('matched_pairs') != 80 or status.get('failed_pairs') != 0:
                    raise RuntimeError('Original weights comparison budget did not complete')
                entry.update(state='completed', exit_code=0, finished_at=now())
                manifest.update(state='completed', finished_at=now())
                write(STATE / 'manifest.json', manifest)
        protected_check(plan)


def launch():
    plan = verify()
    protected_check(plan)
    if process_identity(plan['failed_worker']['pid'])['alive'] is not False or owned_remaining():
        raise RuntimeError('The failed worker and its owned applications must be retired first')
    if (STATE / 'scheduled.json').exists() or (STATE / 'manifest.json').exists():
        raise RuntimeError('A live or memory-gated recovery already exists')
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
    action.Arguments = subprocess.list2cmdline(['-B', str(SCRIPT), '--supervise'])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-' + CATEGORY.title() + 'PIDRecovery-20261024'
    service.GetFolder('\\').RegisterTaskDefinition(name, definition, 2, user, None, 3)
    write(STATE / 'scheduled.json', dict(task_name=name, administrator_required=False,
          execution_time_limit='PT0S', triggers=0, launch_at_utc=now()))
    service.GetFolder('\\').GetTask(name).Run(None)


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
