"""Read-only status, saved-artifact and queue audit. Never opens or signals Aspen."""
from collections import Counter
from copy import deepcopy
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from column_review.common import digest, read, process_identity, owned_apps_remaining
from revision_comparisons.progress import snapshot, log_tail


def exit_code(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        return code.value if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
    finally:
        kernel.CloseHandle(handle)


def inventory(directory, files):
    return bool(files) and all((Path(directory) / name).is_file()
                              and digest(Path(directory) / name) == expected
                              for name, expected in files.items())


def main():
    import review_blas_runtime
    review_blas_runtime.activate()
    import torch
    torch.set_num_threads(1)
    import continue_column_queue as queue
    import recover_markov_seed23 as recovery
    from run_e3_prediction_quality import completed_job
    from run_polymer_remaining import validate_output
    from comparison_studies.checkpoints import load_inference_checkpoint

    issues, completed = [], {}
    start = snapshot()
    previous = read(OUT / 'snapshot_initial.json')
    before = {(r['study'], r['job']): r for r in previous['jobs']}
    current = [deepcopy(r) for r in start['jobs']
               if r['study'] not in start['historical_studies'] and r['job'] != 'e1_pilot']

    events = json.loads((OUT / 'windows_fault_events.json').read_text(encoding='utf-8-sig'))
    faults = []
    for event in events:
        if event['Id'] != 1000:
            continue
        message = event['Message']
        fields = {key: (m.group(1) if (m := re.search(pattern, message)) else None)
                  for key, pattern in {
                      'application': r'Faulting application name: ([^,]+)',
                      'module': r'Faulting module name: ([^,]+)',
                      'exception': r'Exception code: (\S+)',
                      'pid_hex': r'Faulting process id: (\S+)',
                      'module_path': r'Faulting module path: ([^\r\n]+)',
                  }.items()}
        fields['pid'] = int(fields['pid_hex'], 16) if fields['pid_hex'] else None
        faults.append(dict(time=event['TimeCreated'], **fields))
    failures, health = [], []
    for row in current:
        directory = Path(row['directory']) if row.get('directory') else None
        related = [f for f in faults if f['pid'] == row.get('pid')]
        if row['state'] == 'stopped_without_final_status' and related:
            row['recorded_state'] = row['state']
            row['state'] = 'failed'
            row['failure_classification'] = 'Windows application crash; no final worker status'
        if row['state'] == 'failed':
            row['windows_faults'] = related
            row['exit_code_read_only'] = exit_code(row['pid'])
            row['remaining_owned_aspen_apps'] = owned_apps_remaining(directory / 'owned_aspen_apps.json')
            failures.append(row)
        if row['state'] == 'running':
            earlier = before.get((row['study'], row['job']), {})
            identity = process_identity(row['pid'])
            log = directory / ('worker.log' if (directory / 'worker.log').exists() else 'run.log')
            status = read(directory / 'worker_status.json', read(directory / 'status.json', {}))
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(status['started_at'])).total_seconds()
            mean_cycle_seconds = elapsed / row['cycle']
            observed = dict(job=row['job'], pid=row['pid'], identity=identity,
                            cycle=row['cycle'], aspen_number=row.get('aspen_number'),
                            cpu_delta_seconds=row['cpu_seconds'] - earlier.get('cpu_seconds', row['cpu_seconds']),
                            cycle_delta=row['cycle'] - earlier.get('cycle', row['cycle']),
                            mean_cycle_seconds=mean_cycle_seconds,
                            remaining_hours_linear=(300-row['cycle'])*mean_cycle_seconds/3600,
                            recent_fatal_log=any(marker in log_tail(log) for marker in
                                                 ('Windows fatal exception', 'Traceback (most recent call last)')))
            health.append(observed)
            if identity['alive'] is not True or observed['recent_fatal_log']:
                issues.append(dict(study=row['study'], job=row['job'], error='Live-worker health check failed'))

    pointer = read(ROOT / 'Distillation/Results/review_recovery/latest.json')
    repetitions = {}
    for name, directory in [('polymer', ROOT / 'Polymer/Results/phase2_repeats/20261013_010021_35e44c'),
                            ('column', Path(pointer['repeats']))]:
        manifest = read(directory / 'manifest.json')
        retained = {tuple(r['job']): r['sha256'] for r in read(directory / 'plan.json', {}).get('retained_results', [])}
        verified = []
        for row in [r for r in current if r['study'] == name+'_repeats' and r['state'] == 'completed']:
            attempt = Path(row['directory'])
            status = read(attempt / 'worker_status.json')
            job = next(j for j in manifest['jobs'] if recovery.key(j) ==
                       (status['category'], status['mode'], status['seed']))
            # Retired PIDs can belong to a different process days later.
            # Only a still-unfinalized child needs an OS exit-code check.
            code = exit_code(row['pid']) if job['state'] != 'completed' else None
            record = dict(job=row['job'], pid=row['pid'], worker_state=status['state'],
                          recorded_manifest_state=job['state'], os_exit_code=code,
                          result=status['result_file'], episodes=status['episodes'])
            try:
                if status['state'] != 'completed' or status['episodes'] != 300:
                    raise ValueError('No full-budget completed worker record')
                if job['state'] == 'completed' and job.get('exit_code') != 0:
                    raise ValueError('Saved supervisor exit code contradicts completion')
                if code not in (None, 0):
                    raise ValueError('Worker exit code contradicts completion')
                if name == 'column':
                    # A channel supervisor finalizes its manifest only after all ten
                    # children exit. Validate already completed children without
                    # modifying that manifest or claiming it has finalized.
                    normalized = dict(job, state='completed', exit_code=0, result_file=status['result_file'])
                    artifact = recovery.validated_result(normalized, manifest)
                    if recovery.key(job) in retained and artifact['sha256'] != retained[recovery.key(job)]:
                        raise ValueError('Retained result checksum changed')
                    record.update(sha256=artifact['sha256'], path=artifact['path'],
                                  aspen_closed=status.get('aspen_closed'),
                                  remaining_owned_apps=owned_apps_remaining(attempt / 'owned_aspen_apps.json'))
                    if status.get('aspen_closed') is not True or record['remaining_owned_apps']:
                        raise ValueError('Completed column cleanup did not verify')
                else:
                    if not Path(status['result_file']).is_file():
                        raise ValueError('Missing saved polymer result')
                    record['sha256'] = digest(status['result_file'])
                record['verified'] = True
                record['duration_hours'] = (datetime.fromisoformat(status['finished_at'])-
                                            datetime.fromisoformat(status['started_at'])).total_seconds()/3600
                verified.append(record)
            except (KeyError, ValueError, RuntimeError, OSError) as error:
                issues.append(dict(study=name+'_repeats', job=row['job'], error=str(error)))
        repetitions[name] = dict(total=50, completed_verified=len(verified), runs=verified)
        print(f'Verified {name} repetitions: {len(verified)}/50', flush=True)
    completed['repetitions'] = repetitions

    column = Path(pointer['column'])
    comparisons = {}
    for method in ('sg', 'threshold', 'ls', 'bo'):
        attempt = Path(read(column / method / 'attempts.json')[-1]['directory'])
        status = read(attempt / 'status.json')
        valid = (status['state'] == 'completed' and status.get('aspen_closed') is True
                 and status.get('cycles') == 200 and status.get('steps') == 80000
                 and inventory(attempt, status.get('files', {})))
        comparisons[method] = dict(state=status['state'], verified=valid, directory=str(attempt),
                                   cycles=status.get('cycles'), steps=status.get('steps'),
                                   finished_at=status.get('finished_at'), files=len(status.get('files', {})))
        if method == 'bo':
            trials = read(attempt / 'bo_trials.json')
            comparisons[method].update(calibration_trials=len(trials), valid_trials=sum(bool(t.get('valid')) for t in trials),
                                        solver_calls=status.get('solver_calls'), solver_failed_calls=status.get('solver_failed_calls'))
        if not valid:
            issues.append(dict(study='column_E2', method=method, error='Completed artifact inventory did not verify'))
    completed['column_E2'] = comparisons
    print('Verified completed column E2 artifact inventories', flush=True)

    e3 = {}
    for plant, directory in [('polymer', 'Results/e3/20261015_042751_513bff'),
                             ('column', 'Results/e3/20261015_042751_4e03da')]:
        e3[plant] = [{'id': j['id'], 'verified': completed_job(j)} for j in read(ROOT / directory / 'manifest.json')['jobs']]
        if not all(j['verified'] for j in e3[plant]):
            issues.append(dict(study=plant+'_E3', error='Completed artifact inventory failed'))
    completed['E3'] = e3
    print('Verified six completed E3 artifact inventories', flush=True)
    polymer = ROOT / 'Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent'
    pipeline = {}
    for stage in ('e1_combined', 'e2_sg', 'e2_arx_repaired'):
        status = read(polymer / stage / 'status.json')
        try:
            validate_output(polymer / stage / 'job.json', status)
            valid = status['state'] == 'completed'
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            valid = False
            issues.append(dict(study=stage, error=str(error)))
        pipeline[stage] = dict(state=status['state'], verified=valid)
    completed['polymer_recovery'] = pipeline
    provenance = read(polymer / 'analysis/e1/provenance.json')
    identities, pairs = set(), []
    for source in provenance['sources']:
        path = Path(source['path'])
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        valid = digest(path) == source['sha256'] and all(r['status'] == 'completed' for r in records)
        identities.update((r['seed'], r['category'], r['channel'], r['step']) for r in records)
        pairs.append(dict(path=str(path), records=len(records), verified=valid))
    completed['polymer_E1'] = dict(unique_pairs=len(identities), sources=pairs)
    if len(identities) != 3200 or not all(p['verified'] for p in pairs):
        issues.append(dict(study='polymer_E1', error='Expected 3200 verified unique completed pairs'))
    provenance = read(polymer / 'analysis/e2/provenance.json')
    e2 = {method: dict(path=source['path'], verified=digest(source['path']) == source['sha256'])
          for method, source in provenance['sources'].items()}
    completed['polymer_E2'] = dict(state=provenance['state'], sources=e2,
                                  validation='Registered trajectory checksums; primary-array validity audited in the prior all-study audit')
    print('Verified completed polymer E1/E2 evidence', flush=True)
    if provenance['state'] != 'completed' or not all(x['verified'] for x in e2.values()):
        issues.append(dict(study='polymer_E2', error='Shared provenance/checksum mismatch'))
    (OUT / 'completed_artifact_verification.json').write_text(
        json.dumps(dict(observed_at_utc=start['observed_at_utc'], completed=completed,
                        verification_issues=issues), indent=2)+'\n', encoding='utf-8')

    guards = {}
    for name in ('residual', 'combined'):
        manifest = read(Path(pointer['repeats']) / name / 'manifest.json')
        try:
            queue.validate_group(manifest)
            valid, error = True, None
        except (OSError, KeyError, ValueError, RuntimeError) as failure:
            valid, error = False, str(failure)
            issues.append(dict(study='queue_'+name, error=error))
        guards[name] = dict(valid=valid, error=error, states=dict(Counter(j['state'] for j in manifest['jobs'])),
                            reserved_cases=[j['aspen_number'] for j in manifest['jobs']])
    queue_root = Path(pointer['repeats'])
    coordinator = read(queue_root / 'coordinator_identity.json')
    supervisor = read(queue_root / 'residual/supervisor.json')
    coordinator_now, supervisor_now = process_identity(coordinator['pid']), process_identity(supervisor['pid'])
    guards.update(coordinator_identity=coordinator_now, residual_supervisor_identity=supervisor_now,
                  coordinator_creation_matches=coordinator_now['created'] == coordinator['created'],
                  supervisor_creation_matches=supervisor_now['created'] == supervisor['created'],
                  residual_group_exit_record=(queue_root / 'residual/exit.json').exists(),
                  combined_launch_record=(queue_root / 'combined/launch.json').exists(),
                  handoff='Live coordinator waits for residual group finalization, verifies releases, and then starts combined even if a residual worker failed.')

    e1_root = ROOT / read(ROOT / 'Distillation/Results/column_e1/latest.json')['directory']
    e1_dir = e1_root / 'combined/a1'
    checkpoint = read(e1_dir / 'training_checkpoints/latest.json')
    native = checkpoint['native']
    checkpoint_valid = (digest(e1_dir / 'training_checkpoints' / checkpoint['file']) == checkpoint['sha256']
                        and digest(native['file']) == native['file_sha256']
                        and digest(native['project_file']) == native['project_file_sha256'])
    e1 = dict(state=read(e1_root / 'manifest.json')['state'], checkpoint_cycle=checkpoint['completed_cycles'],
              checkpoint_hashes_verified=checkpoint_valid, training_resume=checkpoint['training_resume'],
              completed_pairs=0, expected_pairs=3200,
              held_configurations=[j['category'] for j in read(e1_root / 'manifest.json')['jobs'] if j['state']=='pending'],
              first_restore_failure=read(e1_dir / 'status.json')['error'])
    decisions = e1_dir / 'capture/decisions.jsonl'
    if decisions.exists():
        e1['completed_pairs'] = sum(json.loads(l).get('status') == 'completed' for l in decisions.read_text().splitlines() if l.strip())
    if not checkpoint_valid:
        issues.append(dict(study='column_E1', error='Checkpoint checksum failed'))

    threshold_dir = ROOT / read(ROOT / 'Distillation/Results/state_recovery/latest.json')['threshold'] / 'a1'
    policy = read(threshold_dir / 'policy_checkpoints/latest.json')
    path = threshold_dir / 'policy_checkpoints' / policy['path']
    loaded = load_inference_checkpoint(path)
    threshold = dict(cycle=policy['completed_cycles'], sha256_verified=digest(path) == policy['sha256'],
                     agents=list(loaded['agents']), training_resume=policy['training_resume'],
                     final_checkpoint_exists=(threshold_dir / 'final_inference_networks.pt').exists(),
                     final_training_exists=(threshold_dir / 'training.pkl').exists())

    import win32com.client
    service = win32com.client.GetObject('winmgmts:')
    processes = [{'pid': int(p.ProcessId), 'parent': int(p.ParentProcessId), 'command': str(p.CommandLine)}
                 for p in service.ExecQuery("SELECT ProcessId,ParentProcessId,CommandLine FROM Win32_Process WHERE Name='python.exe'")]
    actual_workers = {p['pid'] for p in processes if str(ROOT).lower() in p['command'].lower() and '--worker' in p['command']}
    registered_workers = {r['pid'] for r in current if r['state'] == 'running'}
    unexpected = sorted(actual_workers - registered_workers)
    if unexpected:
        issues.append(dict(study='process_inventory', error='Unexpected experiment worker', pids=unexpected))
    baseline = read(ROOT / 'change-reports/column_e1_20261020/launch_assessment.json')
    baseline_workers = baseline['protected_workers_preserved']
    for observed in health:
        old = next(p for p in baseline_workers if p['identity']['pid'] == observed['pid'])
        observed['creation_matches_launch_record'] = observed['identity']['created'] == old['identity']['created']
        if not observed['creation_matches_launch_record']:
            issues.append(dict(study='process_identity', job=observed['job'], error='Worker PID creation time differs'))
    frozen_path = Path(r'C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\Rl_assisted_paper\RL_assisted_paper\manuscript_revised\frozen_original_manifest.json')
    frozen = read(frozen_path)
    frozen_base = frozen_path.parent.parent / 'manuscript'
    original_ok = all(digest(frozen_base / record['path']) == record['sha256'] for record in frozen['files'])

    totals = Counter(r['state'] for r in current)
    durations = [r['duration_hours'] for r in repetitions['column']['runs'] if r['job'].startswith(('markov/', 'residual/'))]
    e1_start = datetime.fromisoformat(read(e1_dir / 'status.json')['started_at']).timestamp()
    cycle10_time = checkpoint['saved_at_unix']
    main_seconds_per_step_including_startup = (cycle10_time - e1_start) / checkpoint['completed_steps']
    eta = dict(healthy_residual_linear_hours={r['job']: r['remaining_hours_linear'] for r in health},
               healthy_residual_planning_range_hours=[0.5, 2.0],
               observed_full_column_run_hours=[min(durations), max(durations)],
               combined_cohort_planning_range_hours_after_launch=[36, 60],
               combined_projection_basis='Adjacent 300-cycle Markov/residual runs took 30.8-35.9 h; allow extra combined-agent work. No completed same-profile combined column cohort yet.',
               repairs_and_retries='Unknown repair time, followed by about 31-36 h each if full restarts remain necessary; concurrent execution depends on spare-case assignment.',
               e1_main_steps=600000, e1_branch_steps=1664000, e1_total_aspen_steps=2264000,
               e1_measured_main_seconds_per_step_including_startup=main_seconds_per_step_including_startup,
               e1_step_only_days_projection=2264000 * main_seconds_per_step_including_startup / 86400,
               e1_planning='Several weeks with current one-worker sequential protocol after the restore defect is fixed. Branch throughput remains unmeasured in production; no firm finish date.')
    result = dict(observed_at_utc=start['observed_at_utc'],
                  observed_at_toronto=datetime.fromisoformat(start['observed_at_utc']).astimezone(ZoneInfo('America/Toronto')).isoformat(),
                  scope='Existing authorized production campaigns; no worker or scientific source was modified.',
                  snapshot=start, normalized_current_jobs=current, normalized_totals=dict(totals),
                  remaining=sum(n for state, n in totals.items() if state != 'completed'),
                  completed=completed, failures=failures, live_worker_health=health, queue=guards,
                  column_E1=e1, threshold_policy=threshold, relevant_fault_events=faults,
                  windows_python_processes=processes, unexpected_worker_pids=unexpected,
                  eta=eta, original_manuscript_files=len(frozen['files']), original_manuscript_unchanged=original_ok,
                  verification_issues=issues, script_sha256=digest(__file__))
    target = OUT / 'audit.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(observed=result['observed_at_toronto'], remaining=result['remaining'],
                         totals=dict(totals), completed_repetitions={k:v['completed_verified'] for k,v in repetitions.items()},
                         failures=[dict(job=f['job'], study=f['study'], cycle=f['cycle'], exit=f['exit_code_read_only']) for f in failures],
                         live=health, queue_guards=guards, E1=e1, threshold=threshold, eta=eta,
                         frozen_unchanged=original_ok, verification_issues=issues), indent=2))


if __name__ == '__main__':
    main()
