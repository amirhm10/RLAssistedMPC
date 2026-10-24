"""Preserve a retired guard-only attempt and re-arm the same Markov recovery."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_markov
r = recover_markov.engine


def retired(expected):
    current = r.process_identity(expected['pid'])
    if current['alive'] is False:
        return True
    return current['alive'] is True and current['created'] is not None and current['created'] != expected['created']


def rearm():
    plan = r.verify()
    source = r.STATE.resolve()
    allowed = (r.ROOT/'Distillation/Results/state_recovery').resolve()
    target = source.with_name(source.name+'_retired_peer_guard_attempt1').resolve()
    if source.parent != allowed or target.parent != allowed or target.exists():
        raise RuntimeError('Resolved archival paths must be exact unused workspace children')
    if r.OUTPUT.resolve().is_relative_to(source):
        raise RuntimeError('Scientific output must not be inside the retired supervisor folder')
    manifest = r.read(source/'manifest.json')
    supervisor = r.read(source/'supervisor_identity.json')
    attempt = manifest['job']['attempts'][-1]
    if (manifest['state'] != 'failed' or manifest['stage'] != 'validation'
            or not retired(supervisor) or not retired(attempt['identity'])):
        raise RuntimeError('The old supervisor and guard attempt are not both retired')
    status = r.read(r.OUTPUT/'status.json')
    index = r.read(r.OUTPUT/'training_checkpoints/latest.json')
    if (status['pid'] != plan['failed_worker']['pid'] or status['state'] != 'failed'
            or index['completed_steps'] != 116000 or index['sha256'] != plan['initial_checkpoint_sha256']
            or r.owned_remaining()):
        raise RuntimeError('The original Markov state changed or owns an application')
    log = source/'logs/validation.log'
    text = log.read_text(encoding='utf-8',errors='replace')
    cause = 'A protected peer changed without verified completion: residual'
    if cause not in text:
        raise RuntimeError('The attempt needs a different diagnosis')
    residual = r.ROOT/'change-reports/residual_native_recovery_20261024'
    preservation = json.loads((residual/'preservation.json').read_text(encoding='utf-8'))
    prepared = json.loads((residual/'preparation.json').read_text(encoding='utf-8'))
    if (not preservation['verified'] or prepared['status']['state'] != 'failed'
            or r.process_identity(prepared['status']['pid'])['alive'] is not False
            or r.digest(Path(preservation['manifest'])) != preservation['manifest_sha256']):
        raise RuntimeError('The failed residual peer is not independently preserved')
    updated = dict(plan)
    updated['protected'] = [p for p in plan['protected'] if p['category'] != 'residual']
    updated['preserved_failed_peer_transitions'] = [dict(category='residual',
        failure=prepared['status']['error'], failed_at=prepared['status']['finished_at'],
        backup_manifest=preservation['manifest'], backup_manifest_sha256=preservation['manifest_sha256'],
        checkpoint_steps=[117600,117200], original_pairs=69)]
    r.protected_check(updated)
    r.preserved_traces(updated)
    import win32com.client
    service = win32com.client.Dispatch('Schedule.Service')
    service.Connect()
    folder = service.GetFolder('\\')
    scheduled = r.read(source/'scheduled.json')
    task_name = scheduled['task_name']
    if task_name != 'RLAssistedMPC-MarkovPIDRecovery-20261024':
        raise RuntimeError('Unexpected task name')
    task = folder.GetTask(task_name)
    definition = task.Definition
    expected_arguments = subprocess.list2cmdline(['-B',str(r.SCRIPT),'--supervise'])
    if (task.State == 4 or definition.Actions.Count != 1 or definition.Triggers.Count != 0
            or Path(definition.Actions.Item(1).Path).resolve() != r.BASE_PYTHON.resolve()
            or definition.Actions.Item(1).Arguments != expected_arguments):
        raise RuntimeError('The exact one-shot task is not safely retired')
    r.HERE.joinpath('attempt1_task.xml').write_text(task.Xml,encoding='utf-8')
    shutil.copyfile(source/'manifest.json',r.HERE/'attempt1_manifest_before.json')
    r.write(r.HERE/'rearm_receipt.json',dict(state='prepared',at_utc=r.now(),
        resolved_source=str(source),resolved_archive=str(target),scientific_output=str(r.OUTPUT),
        retired_supervisor=supervisor,retired_attempt=attempt,actual_exit_cause=cause,
        original_manifest_sha256=r.digest(source/'manifest.json'),log_sha256=r.digest(log),
        native_restore_started=False,checkpoint_unchanged=True,task_name=task_name))
    entry = manifest['job']
    entry['inherited_original_failure_error'] = entry.get('error')
    entry.update(state='superseded_peer_guard_attempt',error=cause,error_type='prelaunch_peer_guard')
    manifest.update(state='superseded_peer_guard_attempt',job=entry,jobs=[entry],
                    native_restore_started=False,superseded_at=r.now())
    r.write(source/'manifest.json',manifest)
    folder.DeleteTask(task_name,0)  # Exact completed one-shot task; no process signal.
    source.rename(target)  # Both resolved paths checked above; no scientific data inside.
    shutil.copyfile(r.HERE/'plan.json',r.HERE/'plan_before_residual_failure.json')
    updated['retired_guard_attempt'] = str(target/'manifest.json')
    updated['adapter_sha256'][str(Path(__file__).resolve())] = r.digest(Path(__file__).resolve())
    r.write(r.HERE/'plan.json',updated)
    receipt = r.read(r.HERE/'rearm_receipt.json')
    receipt.update(state='rearmed',finished_at=r.now(),archived_manifest=str(target/'manifest.json'),
                   healthy_workers_interrupted=0,source_checkpoint_or_result_edits=0)
    r.write(r.HERE/'rearm_receipt.json',receipt)
    r.verify()
    r.launch()
    print('Preserved guard-only attempt; reconciled backed-up residual failure; relaunched Markov once with memory gating.')


if __name__ == '__main__':
    rearm()
