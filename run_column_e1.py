"""Run the existing E1 protocol on isolated Aspen case 16, one job at a time."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
CATEGORIES = ('combined', 'horizon', 'weights', 'markov', 'residual')
ENTRIES = {name: f'distillation_RL_assisted_MPC_{"horizons" if name == "horizon" else name}_unified.py' for name in CATEGORIES}
PROFILE = 'temperature_flip_200_100'


def sources():
    from revision_comparisons.column import source_hashes
    from column_review.common import digest
    value = source_hashes()
    for path in [ROOT / 'run_column_e1.py', ROOT / 'repair_aspen_snapshot.py', ROOT / 'review_blas_runtime.py',
                 ROOT / 'revision_recovery/checkpoints.py', *(ROOT / 'column_e1').glob('*.py'),
                 *(ROOT / name for name in ENTRIES.values())]:
        value[path.relative_to(ROOT).as_posix()] = digest(path)
    return value


def guard_queue():
    from column_review.common import read
    import continue_column_queue as queue
    directory = Path(read(ROOT / 'Distillation/Results/review_recovery/latest.json')['repeats'])
    for group in ('residual', 'combined'):
        manifest = read(directory / group / 'manifest.json')
        queue.validate_group(manifest)
        if any(int(job['aspen_number']) == 16 for job in manifest['jobs']):
            raise RuntimeError('Case 16 is reserved by the residual/combined queue.')


def configure(defaults, category, output, aspen, technical=False):
    from run_polymer_e1 import SEED_KEYS
    value = deepcopy(defaults)
    value.update(run_mode='disturb', disturbance_profile='fluctuation', training_profile_override=PROFILE,
                 results_dir_override=str(output), result_prefix_override='column_e1', compare_prefix_override='column_e1',
                 aspen_path_override=aspen['aspen_path'], snaps_path_override=aspen['snaps_path'])
    value['combined_agent_mode' if category == 'combined' else 'agent_mode'] = 'sg'
    for key in SEED_KEYS:
        if isinstance(value.get(key), dict):
            value[key]['seed'] = 7
            if technical:
                if 'batch_size' in value[key]:
                    value[key]['batch_size'] = 8
                for field in ('hidden_layers', 'actor_hidden', 'critic_hidden'):
                    if field in value[key]:
                        value[key][field] = [32, 32]
    return value


def execute(category, configure_defaults):
    """Use the original column setup and loop, retaining original finally blocks."""
    import systems.distillation as distillation
    from review_experiments.e3.runtime import _prefix
    from run_polymer_e1 import SEED_KEYS
    path = ROOT / ENTRIES[category]
    body, found = _prefix(ast.parse(path.read_text(encoding='utf-8'), filename=str(path)).body)
    if not found:
        raise RuntimeError('Column entrypoint no longer has its recognized training assignment.')
    original = distillation.get_distillation_notebook_defaults

    def provider(family):
        defaults = original(family)
        if family == ('horizon_standard' if category == 'horizon' else category):
            return configure_defaults(defaults)
        seeded = deepcopy(defaults)
        selected = configure_defaults(defaults)
        for key in SEED_KEYS:
            if isinstance(seeded.get(key), dict):
                seeded[key] = selected[key]
        return seeded

    namespace = {'__name__': '__column_e1__', '__file__': str(path)}
    distillation.get_distillation_notebook_defaults = provider
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), namespace)
    finally:
        distillation.get_distillation_notebook_defaults = original
    return namespace


def prepare(directory, technical=False):
    from column_review.common import read, write, digest, now
    from review_experiments.e1.study import StudyOptions
    guard_queue()
    original = read(ROOT / 'Distillation/Results/column_review/20261018_recovered/e1_pilot/job.json')['aspen']
    if Path(original['aspen_path']).name != 'C2S_SS_simulation16.dynf' or digest(original['aspen_path']) != original['aspen_sha256']:
        raise RuntimeError('Original case 16 does not verify.')
    directory.mkdir(parents=True, exist_ok=False)
    options = StudyOptions(smoke=technical, **(dict(decisions_per_window=1, rollout_steps=4, horizons=(4,),
                          long_rollout_steps=4, long_fraction=0, snapshots_per_window=1) if technical else {}))
    from dataclasses import asdict
    categories = ('combined',) if technical else CATEGORIES
    jobs = []
    for category in categories:
        output = directory / category / 'a1'
        output.mkdir(parents=True)
        model_dir = output / 'private_model'
        model_dir.mkdir()
        model = model_dir / Path(original['aspen_path']).name
        shutil.copyfile(original['aspen_path'], model)
        snaps = model_dir / 'AM_C2S_SS_simulation16'
        snaps.mkdir()
        aspen = dict(original, aspen_path=str(model), snaps_path=str(snaps))
        branch_dir = output / 'branch_model'
        branch_dir.mkdir()
        branch_model = branch_dir / model.name
        shutil.copyfile(original['aspen_path'], branch_model)
        branch_snaps = branch_dir / 'AM_C2S_SS_simulation16'
        branch_snaps.mkdir()
        branch_aspen = dict(original, aspen_path=str(branch_model), snaps_path=str(branch_snaps))
        job = dict(category=category, seed=7, profile=PROFILE, technical=technical, options=asdict(options),
                   source_case=original, aspen=aspen, branch_aspen=branch_aspen, sources=sources(), prepared_at=now(),
                   runtime_manifest_sha256=digest(ROOT / 'change-reports/run_recovery_20261020/runtime/manifest.json'))
        write(output / 'job.json', job)
        jobs.append(dict(id=category, category=category, seed=7, aspen_number=16,
                         state='pending', directory=str(output), job=str(output / 'job.json')))
    write(directory / 'manifest.json', dict(state='prepared', prepared_at=now(), technical=technical,
                seed=7, case=16, categories=list(categories), jobs=jobs, expected_matched_pairs=4 if technical else 3200,
                budget='300 original cycles per configuration; 100 matched decisions per channel/window; 200 primary steps; 10% extended to 800; one seed',
                protected_queue='Original automatic combined queue; cases 1,2,4-10,12 reserved.'))
    command = subprocess.list2cmdline([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--run', str(directory)])
    (directory / 'run.cmd').write_text('@echo off\nsetlocal\ntitle Column E1 - seed 7 - case 16\n'
        f'cd /d "{ROOT}"\n{command}\nset result=%errorlevel%\necho Column E1 exited with code %result%.\npause\nexit /b %result%\n', encoding='utf-8')
    return directory


def worker(path, resume_steps=0):
    import review_blas_runtime
    job = json.loads(path.read_text(encoding='utf-8'))
    review_blas_runtime.activate(expected_manifest_hash=job['runtime_manifest_sha256'])
    import torch
    torch.set_num_threads(1)
    import numpy as np
    from column_review.common import read, write, digest, now, record_owned_apps, owned_apps_remaining
    from column_e1.runtime import ColumnStudySession, column_copying
    from column_e1.checkpoints import NativeCheckpointManager
    from revision_recovery.checkpoints import install_runner, CheckpointStop
    from revision_comparisons.runtime import seed_all
    from review_experiments.e1.study import StudyOptions, json_safe
    from review_experiments.e1.analysis import summarize
    from review_experiments.e1.smoke import compressed_smoke_clock
    from review_experiments.e3.runtime import aspen_resources
    from run_distillation_phase2_repeats import aspen_file_lock
    output, technical = path.parent, job['technical']
    status = dict(state='running', pid=os.getpid(), started_at=now(), category=job['category'], case=16,
                  technical=technical, output=str(output), profile=PROFILE)
    write(output / 'status.json', status)
    code = 1
    try:
        if sources() != job['sources']:
            raise RuntimeError('Column E1 source dependencies changed after preparation.')
        guard_queue()
        if digest(job['source_case']['aspen_path']) != job['source_case']['aspen_sha256']:
            raise RuntimeError('Original case changed.')
        seed_all(7)
        # SaveDocument writes only these process-specific working copies. The
        # prepared pristine models remain available for any checkpoint restart.
        active = {}
        session_root = output / ('native_' + uuid.uuid4().hex[:8])
        for role,key in (('main','aspen'),('branch','branch_aspen')):
            directory = session_root / role
            directory.mkdir(parents=True)
            model = directory / Path(job[key]['aspen_path']).name
            shutil.copyfile(job[key]['aspen_path'], model)
            snaps = directory / 'AM_C2S_SS_simulation16'
            snaps.mkdir()
            active[key] = dict(job[key], aspen_path=str(model), snaps_path=str(snaps))
        status['native_sessions'] = active
        write(output / 'status.json', status)
        records, sessions = [], []
        branch_columns = []
        options = StudyOptions(**dict(job['options'], horizons=tuple(job['options']['horizons'])))
        manager = NativeCheckpointManager(output / 'training_checkpoints', job['sources'],
                        technical_stop=80 if technical else None, technical_resume_steps=resume_steps)
        if technical and manager.has_checkpoint and not resume_steps:
            raise ValueError('Technical restart needs an explicit bounded continuation length.')

        def factory(compiled, args, kwargs):
            import inspect
            from column_e1.runtime import DistillationColumnAspen
            bindings = inspect.signature(compiled.original).bind(*args, **kwargs).arguments
            context = next(value for value in bindings.values() if isinstance(value, dict) and 'system' in value)
            main = context['system']
            if not branch_columns:
                branch_paths = active['branch_aspen']
                if digest(branch_paths['aspen_path']) != job['source_case']['aspen_sha256']:
                    raise RuntimeError('Private branch model differs from the assigned source case.')
                stack.enter_context(aspen_file_lock(Path(branch_paths['aspen_path']), ROOT / 'Distillation/Results/_aspen_file_locks'))
                branch = object.__new__(DistillationColumnAspen)
                branch_columns.append(branch)
                def close_branch():
                    if getattr(branch, 'ad', None) is not None:
                        branch.close(branch_paths['snaps_path'])
                stack.callback(close_branch)
                # The first capture supplies a fully initialized native case.
                # Do not repeat the scientific main plant's SS/dynamic prefix.
                import win32com.client
                branch.path = branch_paths['aspen_path']
                branch.delta_t = main.delta_t
                branch.ss_inputs = main.ss_inputs.copy()
                branch.initialization_point = main.initialization_point.copy()
                branch.y_ss = main.y_ss.copy()
                branch.current_input = main.current_input.copy()
                branch.current_output = main.current_output.copy()
                branch.ad = win32com.client.DispatchEx('AD application')
                branch.ad.NewDocument()
                branch.ad.Visible = True
                branch._e1_document_open = True
            session = ColumnStudySession(compiled, output / 'capture', options, records=records,
                        source_hashes=job['sources'], branch_plant=branch_columns[0],
                        verify_full_state=technical, repeat_branches=technical)
            sessions.append(session)
            return session

        with aspen_file_lock(Path(job['source_case']['aspen_path']), ROOT / 'Distillation/Results/_aspen_file_locks'):
            with ExitStack() as stack:
                stack.enter_context(record_owned_apps(output))
                stack.enter_context(aspen_resources(active['aspen']))
                stack.enter_context(column_copying())
                if technical:
                    stack.enter_context(compressed_smoke_clock(job['category'], stride=100))
                stack.enter_context(install_runner(job['category'], factory, manager))
                namespace = execute(job['category'], lambda defaults: configure(defaults, job['category'], output, active['aspen'], technical))
        if namespace['TRAINING_PROFILE_NAME'] != PROFILE:
            raise RuntimeError('Wrong column E1 training profile.')
        with (output / 'training.pkl').open('xb') as stream:
            pickle.dump(namespace['result_bundle'], stream, protocol=5)
        if sources() != job['sources']:
            raise RuntimeError('E1 source guard changed during training.')
        summarize(records, output / 'analysis', options.practical_tolerance)
        expected = 1600 if job['category'] == 'combined' else 400
        if len(records) != expected:
            raise RuntimeError('Incomplete matched-pair evidence.')
        status.update(state='completed', matched_pairs=len(records),
                      failed_pairs=sum(r['status']!='completed' for r in records), trajectory_sha256=digest(output / 'training.pkl'))
        code = 0
    except CheckpointStop:
        if not technical:
            raise
        checkpoint = read(output / 'training_checkpoints/latest.json')
        checks = sessions[0].integration_checks
        status.update(state='technical_checkpoint', completed_steps=checkpoint['completed_steps'],
                      matched_pairs=len(records), channels=sorted({r['channel'] for r in checks}),
                      full_state_checks=len(checks), resumed_from_step=manager.restored_step,
                      checkpoint_has_four_learners=len(checkpoint['agents']) == 4)
        code = 0
    except BaseException as error:
        status.update(state='failed', error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        traceback.print_exc()
    # Release retained failed-branch automation references before checking Quit.
    for session in locals().get('sessions', []):
        session.native = session.native_plant = session.snapshot = session.branch_plant = None
    locals().get('branch_columns', []).clear()
    import gc
    gc.collect()
    remaining = owned_apps_remaining(output / 'owned_aspen_apps.json')
    for _ in range(10):
        if not remaining: break
        time.sleep(1)
        remaining = owned_apps_remaining(output / 'owned_aspen_apps.json')
    status.update(finished_at=now(), owned_apps_remaining=remaining)
    if status['owned_apps_remaining']:
        status.update(state='failed', cleanup_error='Owned Aspen application remains; inspect before any restart.')
        code = 1
    write(output / 'status.json', json_safe(status))
    return code


def run(directory):
    from column_review.common import read, write, now, process_identity
    from run_distillation_phase2_repeats import aspen_file_lock
    from run_polymer_e3_recovery import environment
    from revision_comparisons.progress import log_tail
    with aspen_file_lock(directory / 'supervisor', directory / '_locks'):
        manifest = read(directory / 'manifest.json')
        if manifest['state'] != 'prepared':
            raise RuntimeError('E1 already started; refusing a duplicate supervisor.')
        guard_queue()
        manifest.update(state='running', supervisor=process_identity(os.getpid()), started_at=now())
        write(directory / 'manifest.json', manifest)
        for job in manifest['jobs']:
            output = Path(job['directory'])
            with (output / 'worker.log').open('xb') as log:
                child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', job['job']],
                    cwd=ROOT, env={**environment(), 'PYTHONHASHSEED': '7', 'MPLBACKEND': 'Agg'}, stdout=log, stderr=subprocess.STDOUT)
                job.update(state='running', pid=child.pid, identity=process_identity(child.pid), started_at=now())
                write(directory / 'manifest.json', manifest)
                previous = None
                while child.poll() is None:
                    text = log_tail(output / 'worker.log', 8000)
                    cycles = re.findall(r'Sub_Episode:\s*(\d+)', text)
                    cycle = int(cycles[-1]) if cycles else 0
                    decisions = output / 'capture/decisions.jsonl'
                    count = sum(1 for _ in decisions.open(encoding='utf-8')) if decisions.exists() else 0
                    progress = (cycle, count)
                    if progress != previous:
                        print(f'Column E1 {job["category"]}: cycle {cycle}/300; {count} matched pairs; PID {child.pid}; case 16', flush=True)
                        write(directory / 'progress.json', dict(updated_at=now(), category=job['category'], cycle=cycle,
                                   total_cycles=300, matched_pairs=count, pid=child.pid, case=16,
                                   completed_configurations=sum(j['state']=='completed' for j in manifest['jobs']),
                                   pending_configurations=sum(j['state']=='pending' for j in manifest['jobs'])))
                        previous = progress
                    time.sleep(15)
                code = child.wait()
            status = read(output / 'status.json', {})
            job.update(state='completed' if code == 0 and status.get('state') == 'completed' else 'failed',
                       exit_code=code, finished_at=now())
            write(directory / 'manifest.json', manifest)
            if job['state'] == 'failed':
                manifest.update(state='failed', error=status.get('error', 'Worker stopped without complete E1 evidence.'))
                write(directory / 'manifest.json', manifest)
                return 1
            guard_queue()
        manifest.update(state='completed', finished_at=now())
        write(directory / 'manifest.json', manifest)
        return 0


def schedule(directory):
    import win32api, win32con, win32security, win32com.client
    from column_review.common import write
    service = win32com.client.Dispatch('Schedule.Service')
    service.Connect()
    task = service.NewTask(0)
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    task.RegistrationInfo.Description = 'Author-requested column E1; case 16; one sequential worker; demand only.'
    task.Principal.UserId, task.Principal.LogonType, task.Principal.RunLevel = user, 3, 0
    task.Settings.Enabled = task.Settings.AllowDemandStart = True
    task.Settings.DisallowStartIfOnBatteries = task.Settings.StopIfGoingOnBatteries = False
    task.Settings.ExecutionTimeLimit, task.Settings.MultipleInstances = 'PT0S', 2
    action = task.Actions.Create(0)
    action.Path = str(Path(sys.executable).with_name('pythonw.exe'))
    action.Arguments = subprocess.list2cmdline(['-B', '-u', str(Path(__file__).resolve()), '--broker', str(directory)])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-Column-E1-' + uuid.uuid4().hex[:12]
    registered = service.GetFolder('\\').RegisterTaskDefinition(name, task, 2, user, None, 3)
    write(directory / 'scheduled.json', dict(task_name=name, triggers=0, execution_time_limit='PT0S'))
    registered.Run(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepare', 'worker', 'run', 'launch', 'broker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--technical', action='store_true')
    parser.add_argument('--resume-steps', type=int, default=0)
    args = parser.parse_args()
    import review_blas_runtime
    review_blas_runtime.activate()
    if args.prepare:
        prepare(args.prepare.resolve(), args.technical)
    elif args.worker:
        return worker(args.worker.resolve(), args.resume_steps)
    elif args.run:
        return run(args.run.resolve())
    elif args.launch:
        schedule(args.launch.resolve())
    elif args.broker:
        from recover_polymer_threshold import broker
        return broker(args.broker.resolve())
    else:
        parser.error('Select an explicit E1 action.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
