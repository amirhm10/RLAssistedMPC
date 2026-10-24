"""Resume only failed horizon using the existing owned-process Aspen repair."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from aspen_retention import attributes, digest, now, read_json, version, Verifier

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'Distillation/Results/state_recovery/20261023_horizon_native_process'
REPORT = ROOT / 'change-reports/horizon_recovery_20261023'
BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\horizon_recovery_20261023')
OUTPUT = ROOT / 'Distillation/Results/state_recovery/20261021_remaining_active/e1/horizon/a1'
PYTHON = ROOT / 'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'
BASE_PYTHON = Path(r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\pythonw.exe')
RETENTION_CONFIG = ROOT / 'change-reports/aspen_retention_20261023/config.json'
MINIMUM_HEADROOM = 6 * 1024**3


def atomic_json(path, value):
    # The established writer retries brief Windows replacement conflicts.
    from recover_markov_seed23 import write
    return write(Path(path), value)


def read(path, default=None):
    try:
        return read_json(Path(path))
    except FileNotFoundError:
        return default


def horizon_only(plan, path):
    """Adapt one operational row locally; never write the shared base manifest."""
    result = deepcopy(plan)
    selected = [row for row in result['jobs'] if Path(row['job']) == Path(path)]
    if len(selected) != 1 or selected[0]['category'] != 'horizon' or selected[0]['state'] != 'failed':
        raise RuntimeError('Only the failed horizon row can be recovered')
    selected[0]['state'] = 'ready'
    return result


def completed_result(folder, study):
    status = read(Path(folder) / ('worker_status.json' if study == 'column_repeats' else 'status.json'))
    if not status or status.get('state') != 'completed':
        return False
    target = status.get('result_file') or status.get('results_file')
    if target:
        return attributes(Path(target))['size'] > 0
    return bool(status.get('trajectory_sha256')) and (Path(folder) / 'training.pkl').is_file()


def protected_check(plan):
    from column_review.common import process_identity
    for row in plan['protected']:
        current = process_identity(row['identity']['pid'])
        if current == row['identity']:
            continue
        if current['alive'] is False and row.get('directory') and completed_result(row['directory'], row['study']):
            continue
        if current['alive'] is False and row['role'] == 'combined_coordinator':
            status = read(Path(row['queue_directory']) / 'coordinator_status.json')
            if (status.get('state') == 'completed' and status.get('stage') == 'finished'
                    and status.get('pid') == row['identity']['pid']
                    and status.get('queued_runs_dispatched') == 20):
                continue
        raise RuntimeError('A protected process changed without verified completion: ' + str(row))


def verify(directory):
    plan = read(directory / 'manifest.json')
    # JSON does not retain the in-memory alias used by progress readers.
    plan['jobs'] = [plan['job']]
    if Path(plan['job']['directory']) != OUTPUT or plan['job']['category'] != 'horizon':
        raise RuntimeError('Recovery is restricted to the original horizon output')
    if digest(directory / 'protocol.json') != plan['protocol_sha256']:
        raise RuntimeError('Horizon recovery protocol changed')
    for name, checksum in plan['source_sha256'].items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError('Sealed scientific/recovery source changed: ' + name)
    if digest(Path(plan['job']['job'])) != plan['job']['job_sha256']:
        raise RuntimeError('The original horizon job changed')
    return plan


def prepare(directory):
    import review_numeric_recovery
    review_numeric_recovery.install()
    import psutil
    import run_column_e1_budget as budget_runner
    from column_review.common import owned_apps_remaining, process_identity
    from run_aspen_retention import verify as verify_retention
    from run_distillation_phase2_repeats import aspen_file_lock

    if directory.exists() or BACKUP.exists() or REPORT.exists():
        raise FileExistsError('An existing horizon recovery must be inspected before another preparation')
    original = budget_runner.verify(budget_runner.DEFAULT)
    entry = next(row for row in original['jobs'] if row['category'] == 'horizon')
    status = read(OUTPUT / 'status.json')
    job = read(Path(entry['job']))
    horizon_only(original, entry['job'])
    if (status['state'] != 'failed' or process_identity(status['pid'])['alive'] is not False
            or owned_apps_remaining(OUTPUT / 'owned_aspen_apps.json')):
        raise RuntimeError('Horizon is not a retired failed worker')
    if (Path(entry['directory']) != OUTPUT or entry['aspen_number'] != 13
            or job['category'] != 'horizon' or job['options']['decisions_per_window'] != 20
            or entry['expected_matched_pairs'] != 80):
        raise RuntimeError('Original horizon case/budget differs')
    config = read(RETENTION_CONFIG)
    verify_retention(config)
    state = read(Path(config['state_root']) / 'status.json')
    previous = next(row for row in state['jobs'] if row['label'] == 'horizon')
    if len(previous.get('checkpoints', [])) != 2:
        raise RuntimeError('Two prior verified horizon checkpoints are required')
    source = Path(job['source_case']['aspen_path'])
    if source.name != 'C2S_SS_simulation13.dynf' or digest(source) != job['source_case']['aspen_sha256']:
        raise RuntimeError('The assigned source case differs')
    # Probe only horizon's case reservation; never acquire queue/other cases.
    with aspen_file_lock(source, ROOT / 'Distillation/Results/_aspen_file_locks'):
        pass
    directory.mkdir(parents=True)
    (directory / 'logs').mkdir()
    REPORT.mkdir(parents=True)
    BACKUP.mkdir(parents=True)
    verification = REPORT / 'checkpoint_verification'
    verification.mkdir()
    anchors = {str(OUTPUT / 'training_checkpoints') + f'::{row["slot"]}::{row["completed_steps"]}': row['sha256']
               for row in previous['checkpoints']}
    atomic_json(verification / 'checkpoint_hashes.json', anchors)
    pair = Verifier(verification).pair(OUTPUT / 'training_checkpoints', ROOT / 'Distillation/Results')
    if pair['checkpoints'][0]['completed_steps'] != 116000:
        raise RuntimeError('Expected the cycle-290 horizon recovery point')
    paths = {OUTPUT / 'status.json', OUTPUT / 'owned_aspen_apps.json', Path(entry['job']),
             OUTPUT / 'training_checkpoints/latest.json', Path(entry['log'])}
    for checkpoint in pair['checkpoints']:
        paths.add(Path(checkpoint['python_file']))
        paths.update(path for path in Path(checkpoint['native_directory']).rglob('*') if path.is_file())
    paths.update(path for path in (OUTPUT / 'capture').glob('*.json*') if path.is_file())
    paths.update(path for path in (OUTPUT / 'capture/traces').rglob('*') if path.is_file())
    copies = []
    for path in sorted(paths):
        relative = path.relative_to(OUTPUT) if path.is_relative_to(OUTPUT) else Path('prior_logs') / path.name
        target = BACKUP / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(path)
        shutil.copyfile(path, target)
        if digest(target) != checksum or digest(path) != checksum:
            raise RuntimeError('Backup file differs: ' + str(path))
        copies.append(dict(source=str(path), local_copy=str(target), sha256=checksum, bytes=attributes(path)['size']))
    traces = {str(path): digest(path) for path in paths if path.suffix == '.npz'}
    decisions = [json.loads(line) for line in (OUTPUT / 'capture/decisions.jsonl').read_text().splitlines() if line.strip()]
    if len(decisions) != 60 or not all(row.get('status') == 'completed' for row in decisions):
        raise RuntimeError('The original sixty completed horizon comparisons differ')
    baseline = read(ROOT / 'change-reports/aspen_retention_20261023/before_runtime.json')
    protected = []
    for row in baseline['workers']:
        if row['study'] == 'E1_column' and row['job'] == 'horizon':
            continue
        identity = process_identity(row['identity']['pid'])
        if identity['alive'] is not True:
            if completed_result(row['directory'], row['study']):
                continue
            raise RuntimeError('A peer requires inspection: ' + row['job'])
        if psutil.Process(identity['pid']).create_time() != row['identity']['created']:
            raise RuntimeError('Peer PID was reused')
        protected.append(dict(role='scientific_worker', study=row['study'], directory=row['directory'],
                              job=row['job'], identity=identity))
    coordinator = process_identity(baseline['combined_coordinator']['pid'])
    if coordinator['alive'] is True:
        protected.append(dict(role='combined_coordinator', identity=coordinator,
                              queue_directory=str(ROOT / 'Distillation/Results/review_recovery/20261020_queue_case12')))
    retention = read(Path(config['state_root']) / 'worker_identity.json')
    protected.append(dict(role='retention', identity=process_identity(retention['pid'])))
    if protected[-1]['identity']['alive'] is not True:
        raise RuntimeError('Automatic retention is not active')
    protocol = dict(id='column-e1-horizon-fresh-owned-process-v1', prepared_at=now(),
                    budget_protocol_sha256=original['protocol_sha256'], seed=7, case=13,
                    main_training_cycles=300, expected_matched_pairs=80, validation_steps=8,
                    start_headroom_bytes=MINIMUM_HEADROOM,
                    scope='Existing fresh-owned-process loader and complete Python restore; scientific schedule unchanged')
    atomic_json(directory / 'protocol.json', protocol)
    entry = deepcopy(entry)
    entry.update(state='pending', attempts=[], pid=None, error=None, initial_checkpoint=pair['index'],
                 kind='e1', log=str(directory / 'logs/production.log'), prior_failure_status=status)
    sources = dict(config['scientific_sha256'], **config['bootstrap_sha256'])
    sources[Path(__file__).name] = digest(Path(__file__))
    plan = dict(state='prepared', prepared_at=now(), protocol_sha256=digest(directory / 'protocol.json'),
                source_sha256=sources, job=entry, jobs=[entry], protected=protected,
                initial_trace_sha256=traces, initial_pairs=60, checkpoint_pair=pair['checkpoints'],
                minimum_headroom_bytes=MINIMUM_HEADROOM, backup=str(BACKUP))
    protected_check(plan)
    atomic_json(BACKUP / 'preservation.json', dict(verified=True, at_utc=now(), files=copies,
                checkpoint_steps=[row['completed_steps'] for row in pair['checkpoints']], original_pairs=60))
    atomic_json(directory / 'manifest.json', plan)
    atomic_json(REPORT / 'prepared.json', dict(at_utc=now(), directory=str(directory), backup=str(BACKUP),
                checkpoint_steps=[row['completed_steps'] for row in pair['checkpoints']], backup_files=len(copies),
                protected_workers=len([row for row in protected if row['role'] == 'scientific_worker'])))
    print('Prepared horizon only; verified both full checkpoints and local backups.', flush=True)


def worker(directory, validation_steps=0):
    if validation_steps not in (0, 8):
        raise ValueError('Only the reviewed eight-step validation or full resume is supported')
    import review_numeric_recovery
    review_numeric_recovery.install()
    import run_column_e1_budget as original
    import column_e1_budget as budget
    import column_e1_process_recovery as fresh
    import column_review.common as common
    from column_review.common import owned_apps_remaining
    plan = verify(directory)
    if owned_apps_remaining(OUTPUT / 'owned_aspen_apps.json'):
        raise RuntimeError('A horizon-owned application is already live')
    original_verify = original.verify
    def selected_verify(target):
        return horizon_only(original_verify(target), plan['job']['job'])
    def local_write(target, value):
        if Path(target).parent == original.DEFAULT and Path(target).name.startswith('validation_'):
            target = directory / 'validation.json'
        atomic_json(Path(target), value)
    factory = fresh.manager_factory(budget.manager_class, plan['protocol_sha256'], OUTPUT)
    # All adaptations exist only in this new process. The original scientific
    # sources, shared operational manifest and peers are never modified.
    with patch.object(original, 'read', read), patch.object(original, 'verify', selected_verify), \
         patch.object(original, 'write', local_write), patch.object(common, 'read', read), \
         patch.object(budget, 'manager_class', factory), fresh.installed(OUTPUT):
        return original.worker(Path(plan['job']['job']), validation_steps)


def supervise(directory):
    from run_column_e1_process_recovery import memory
    from run_aspen_retention import single_instance
    from run_polymer_e3_recovery import environment
    from review_restart_recovery import lifetime, isolated_interactive_context
    identity = lifetime()
    if not isolated_interactive_context(identity):
        raise RuntimeError('Horizon supervisor needs an independent current-user lifetime')
    atomic_json(directory / 'supervisor_identity.json', identity)
    with single_instance(directory):
        plan = verify(directory)
        if plan['state'] != 'prepared':
            raise RuntimeError('This horizon recovery was already started')
        pointer_path = ROOT / 'Distillation/Results/live_recovery/latest.json'
        stamp = version(pointer_path)
        pointer = read(pointer_path)
        atomic_json(directory / 'progress_pointer_before.json', pointer)
        manifests = pointer.setdefault('additional_manifests', [])
        if str(directory / 'manifest.json') not in manifests:
            manifests.append(str(directory / 'manifest.json'))
        if version(pointer_path) != stamp:
            raise RuntimeError('Progress pointer changed during recovery registration')
        atomic_json(pointer_path, pointer)
        for stage, steps in (('validation', 8), ('production', 0)):
            while True:
                verify(directory)
                protected_check(plan)
                resources = memory()
                plan.update(state='waiting_for_memory', stage=stage, memory=resources, updated_at=now())
                atomic_json(directory / 'manifest.json', plan)
                if resources['headroom_bytes'] >= plan['minimum_headroom_bytes']:
                    break
                time.sleep(20)
            log = directory / 'logs' / (stage + '.log')
            started = now()
            command = [str(PYTHON), '-B', '-u', str(Path(__file__)), '--worker', str(directory)]
            if steps:
                command += ['--validation-steps', str(steps)]
            with log.open('xb') as output:
                child = subprocess.Popen(command, cwd=ROOT,
                        env=dict(environment(), PYTHONHASHSEED='7', MPLBACKEND='Agg'),
                        stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                entry = plan['job']
                entry.update(state='running', pid=child.pid, started_at=started, log=str(log), error=None)
                entry['attempts'].append(dict(pid=child.pid, directory=str(OUTPUT), started_at=started, stage=stage))
                plan.update(state='running', stage=stage)
                while True:
                    status = read(OUTPUT / 'status.json')
                    if status.get('started_at', '') >= started:
                        entry['pid'] = status['pid']
                        entry['attempts'][-1]['pid'] = status['pid']
                    plan.update(updated_at=now(), memory=memory())
                    atomic_json(directory / 'manifest.json', plan)
                    if child.poll() is not None:
                        break
                    time.sleep(10)
            status = read(OUTPUT / 'status.json')
            expected = 'technical_checkpoint' if steps else 'completed'
            if child.returncode != 0 or status.get('state') != expected:
                entry.update(state='failed', error=status.get('error'), exit_code=child.returncode)
                plan.update(state='failed', finished_at=now())
                atomic_json(directory / 'manifest.json', plan)
                return
            if steps:
                result = read(directory / 'validation.json')
                if (not result['passed'] or result['index']['completed_steps'] != 116008
                        or result['restore']['completed_steps'] != 116000):
                    raise RuntimeError('Bounded full-state horizon continuation did not verify')
                for path, checksum in plan['initial_trace_sha256'].items():
                    if digest(Path(path)) != checksum:
                        raise RuntimeError('An original horizon trace changed')
                entry['state'] = 'pending'
                plan.update(validation_passed=True)
            else:
                entry.update(state='completed', exit_code=0, finished_at=now())
                plan.update(state='completed', finished_at=now())
                atomic_json(directory / 'manifest.json', plan)
        protected_check(plan)


def launch(directory):
    import win32api
    import win32con
    import win32security
    import win32com.client
    plan = verify(directory)
    if (directory / 'scheduled.json').exists() or plan['state'] != 'prepared':
        raise RuntimeError('Horizon recovery already launched')
    protected_check(plan)
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
    action.Arguments = subprocess.list2cmdline(['-B', str(Path(__file__)), '--supervise', str(directory)])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-HorizonRecovery-20261023'
    service.GetFolder('\\').RegisterTaskDefinition(name, definition, 2, user, None, 3)
    atomic_json(directory / 'scheduled.json', dict(task_name=name, administrator_required=False,
                execution_time_limit='PT0S', triggers=0, launch_at_utc=now()))
    service.GetFolder('\\').GetTask(name).Run(None)
    print('Started current-user horizon recovery; memory-gated validation then production.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'launch', 'worker', 'supervise'):
        group.add_argument('--' + name, type=Path)
    parser.add_argument('--validation-steps', type=int, default=0)
    args = parser.parse_args()
    for name in ('prepare', 'launch', 'worker', 'supervise'):
        selected = getattr(args, name)
        if selected is None:
            continue
        try:
            result = worker(selected.resolve(), args.validation_steps) if name == 'worker' else globals()[name](selected.resolve())
        except BaseException:
            if name == 'supervise':
                atomic_json(selected / 'supervisor_error.json', dict(at_utc=now(), traceback=traceback.format_exc()))
            raise
        if name == 'worker':
            raise SystemExit(result)
