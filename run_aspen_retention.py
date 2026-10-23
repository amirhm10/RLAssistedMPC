"""Run automatic retention independently of all Aspen/combined queue workers."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time
import traceback
import uuid

from aspen_retention import Retainer, SESSION, atomic_json, digest, native_paths_in, now, read_json, restore

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'change-reports/aspen_retention_20261023'
CONFIG = REPORT / 'config.json'
PYTHON = Path(r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\python.exe')


def prepare():
    if CONFIG.exists():
        raise FileExistsError('Existing retention configuration will not be replaced')
    assessment = read_json(ROOT / 'change-reports/e1_recovery_20261023/recovery_assessment.json')
    jobs = [dict(label=row['job'], directory=row['directory']) for row in assessment['jobs']
            if row['study'] == 'E1_column']
    if len(jobs) != 5:
        raise ValueError('Expected the five existing E1 configurations')
    scientific = {}
    seeds = {row['directory']: row['index'] for row in read_json(REPORT / 'seed_indexes.json')}
    for row in jobs:
        row['private_sessions'] = [str(path) for path in Path(row['directory']).iterdir()
                                   if SESSION.fullmatch(path.name) and path.is_dir()]
        index = seeds[str(Path(row['directory']) / 'training_checkpoints')]
        for name, checksum in index['source_hashes'].items():
            path = ROOT / name
            if path.is_file() and path.resolve().is_relative_to(ROOT):
                if digest(path) != checksum:
                    raise ValueError('Sealed scientific source differs: ' + name)
                scientific[name] = checksum
    results_root = ROOT / 'Distillation/Results'
    pins = set()
    # These are immutable saved handoff/backup records, never live queue files.
    frozen_roots = [Path(r'C:\Users\hamediaa\RL_MPC_local_archive\e1_recovery_20261023'),
                    Path(r'C:\Users\hamediaa\RL_MPC_local_archive\healthy_e1_checkpoints_20261023')]
    frozen_roots.extend(ROOT / 'Distillation/Results/state_recovery' / name / 'originals' for name in
                        ('20261021_e1_budget20', '20261022_e1_budget_lazy', '20261023_e1_native_process'))
    for frozen in frozen_roots:
        if frozen.exists():
            for path in frozen.rglob('*.json'):
                pins.update(native_paths_in(read_json(path), results_root))
    pins = {path for path in pins if path.is_dir()}
    config = dict(version=1, prepared_at=now(), results_root=str(results_root),
                  state_root=r'C:\Users\hamediaa\RL_MPC_local_archive\aspen_retention_20261023',
                  jobs=jobs, pinned_native_directories=sorted(str(path) for path in pins),
                  queue_policy='No live queue files, queue locks or Aspen case locks are opened by retention',
                  minimum_age_seconds=180, stable_seconds=20, checkpoint_quiet_seconds=2, poll_seconds=30,
                  maximum_folders_per_pass=12, io_mib_per_second=128,
                  bootstrap_sha256={name: digest(ROOT / name) for name in
                                    ('aspen_retention.py', 'aspen_retention_pickle.py', 'run_aspen_retention.py')},
                  scientific_sha256=scientific,
                  authorization='Fix automatic Aspen retention; keep latest two verified complete checkpoints and all results; do not interrupt workers or combined queue.')
    atomic_json(CONFIG, config)
    print(json.dumps(dict(prepared=True, jobs=len(jobs), scientific_files_sealed=len(scientific),
                          state_root=config['state_root'])), flush=True)


def verify(config):
    if Path(config['results_root']).absolute() != ROOT / 'Distillation/Results':
        raise ValueError('Unexpected production results root')
    if Path(config['state_root']).absolute() != Path(r'C:\Users\hamediaa\RL_MPC_local_archive\aspen_retention_20261023'):
        raise ValueError('Unexpected production archive root')
    for name, checksum in dict(config['scientific_sha256'], **config['bootstrap_sha256']).items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Retention/source integrity check failed: ' + name)


@contextmanager
def single_instance(state):
    import msvcrt
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'retention.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def run(config_path):
    import psutil
    config = read_json(config_path)
    verify(config)
    state = Path(config['state_root'])
    launch_record = read_json(state / 'launch.json')
    if digest(config_path) != launch_record['config_sha256']:
        raise ValueError('The approved retention configuration changed after launch')
    with single_instance(state):
        own = psutil.Process()
        own.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        from review_restart_recovery import lifetime, isolated_interactive_context
        identity = lifetime()
        if not isolated_interactive_context(identity):
            raise RuntimeError('Retention must have a lifetime independent of the tool executor')
        atomic_json(state / 'worker_identity.json', identity)
        worker = Retainer(config)
        worker.recover_pending()
        while not (state / 'stop.request').exists():
            started = time.monotonic()
            try:
                verify(config)
                result = worker.scan(execute=True)
                atomic_json(state / 'resources.json', dict(updated_at=now(), pid=own.pid,
                            rss_bytes=own.memory_info().rss, threads=own.num_threads(),
                            cpu_seconds=sum(own.cpu_times()[:2]), last_pass_seconds=time.monotonic()-started))
                with (state / 'worker.log').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(at_utc=now(), state=result['state'], totals=result['totals'])) + '\n')
            except Exception:
                message = traceback.format_exc()
                worker.errors.append(dict(at_utc=now(), message=message))
                worker.status('deferred_after_error; original data preserved')
                with (state / 'worker.log').open('a', encoding='utf-8') as stream:
                    stream.write(message + '\n')
            # The stop file stops only this maintenance worker, never a run.
            for _ in range(config.get('poll_seconds', 30)):
                if (state / 'stop.request').exists():
                    break
                time.sleep(1)
        worker.status('stopped_by_request')


def launch(config_path):
    import win32api
    import win32con
    import win32security
    import win32com.client
    config = read_json(config_path)
    verify(config)
    state = Path(config['state_root'])
    state.mkdir(parents=True, exist_ok=True)
    if (state / 'launch.json').exists():
        raise FileExistsError('An existing retention launch must be inspected before another start')
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    service = win32com.client.Dispatch('Schedule.Service')
    service.Connect()
    definition = service.NewTask(0)
    definition.RegistrationInfo.Description = 'Keep two verified complete E1 checkpoints; losslessly archive obsolete native and closed private copies. No simulator/process control.'
    definition.Principal.UserId = user
    definition.Principal.LogonType = 3
    definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.ExecutionTimeLimit = 'PT0S'
    settings.MultipleInstances = 2
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    action = definition.Actions.Create(0)
    action.Path = str(PYTHON.with_name('pythonw.exe'))
    action.Arguments = subprocess.list2cmdline(['-B', str(Path(__file__)), '--run', str(config_path)])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-AspenRetention-' + uuid.uuid4().hex[:12]
    atomic_json(state / 'launch.json', dict(task_name=name, created_at=now(), administrator_required=False,
                                         current_user=True, execution_time_limit='PT0S', triggers=0,
                                         config_sha256=digest(config_path), bootstrap_sha256=config['bootstrap_sha256']))
    task = service.GetFolder('\\').RegisterTaskDefinition(name, definition, 2, user, None, 3)
    task.Run(None)
    print(json.dumps(dict(started=True, task_name=name, administrator_required=False, state_root=str(state))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--launch', type=Path)
    group.add_argument('--run', type=Path)
    group.add_argument('--dry-run', type=Path)
    group.add_argument('--restore-receipt', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        launch(args.launch)
    elif args.run:
        try:
            run(args.run)
        except BaseException:
            state = Path(read_json(args.run)['state_root'])
            atomic_json(state / 'startup_error.json', dict(at_utc=now(), error=traceback.format_exc()))
            raise
    elif args.dry_run:
        selected = read_json(args.dry_run)
        verify(selected)
        result = Retainer(selected).scan(execute=False)
        print(json.dumps(result, indent=2), flush=True)
    else:
        if not args.destination:
            parser.error('--destination is required for a staged restore')
        print(json.dumps(restore(args.restore_receipt, args.destination)), flush=True)
