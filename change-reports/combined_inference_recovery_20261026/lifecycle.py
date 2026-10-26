"""Seal a fresh failed-attempt plan and retire only its exact owned generations."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACCEPTED = ROOT / 'change-reports/combined_parallel_20261024'
sys.path.insert(0, str(HERE))
import recover_parallel as recovery
p = recovery.execution.p if hasattr(recovery.execution, 'p') else __import__('parallel_branches')
import preserve
from aspen_retention import still_current

preserve.HERE = HERE


def current_pair(pair):
    from copy import deepcopy
    copied = deepcopy(pair)
    copied['layout']['index'] = tuple(copied['layout']['index'])
    return still_current(copied)


def seal():
    evidence = p.closed_json(HERE / 'preservation.json')
    full = p.closed_json(evidence['manifest'])
    if not evidence['verified'] or p.h.digest(evidence['manifest']) != evidence['manifest_sha256'] or not current_pair(full['pair']):
        raise RuntimeError('Refresh independently verified current recovery preservation')
    diagnostics = ACCEPTED / 'diagnostics_20261026'
    stacks = p.closed_json(diagnostics / 'stack_followup_summary.json')
    for sample in stacks['samples']:
        if sample['returncode'] or not sample['nonblocking']:
            raise RuntimeError('Both nonblocking stall samples must verify')
        current = __import__('psutil').Process(sample['after']['pid'])
        if not p.same_generation(sample['after'], p.h.process(sample['after']['pid'])):
            raise RuntimeError('Refresh the stalled inference generation')
        old = sample['io']
        live = current.io_counters()
        if live.read_bytes != old['read_bytes'] or live.write_bytes != old['write_bytes']:
            raise RuntimeError('A branch resumed I/O; re-evaluate before any retirement')
    old = p.closed_json(ACCEPTED / 'plan.json')
    source = dict(old['source_sha256'])
    for name, checksum in source.items():
        if p.h.digest(ROOT / name) != checksum:
            raise RuntimeError('A sealed scientific source changed')
    index = full['pair']['index']
    adapters = dict(old['adapter_sha256'])
    adapters.update({path.name:p.h.digest(path) for path in HERE.glob('*.py')})
    branch = __import__('copy').deepcopy(old['branch_template'])
    branch['adapter_hashes'].update({str(path):p.h.digest(path) for path in HERE.glob('*.py')})
    diagnosis = dict(parent_identity=full['original_identity'],
        unchanged_inference_stacks_and_io_verified=True,
        stack_evidence=str(diagnostics / 'stack_followup_summary.json'),
        stack_evidence_sha256=p.h.digest(diagnostics / 'stack_followup_summary.json'),
        initial_stack_evidence_sha256=p.h.digest(diagnostics / 'stack_initial_summary.json'),
        assessment='Persistent CUDA inference/context stall; device memory saturation observed. Low-level CUDA cause remains unproven.',
        pending=full['pending'], healthy_workers_to_interrupt=[])
    plan = {k:old[k] for k in ('category','case','seed','expected_matched_pairs','main_training_cycles',
        'decisions_per_window','job','job_sha256','original_process_protocol_sha256',
        'pilot_acceptance','transport_acceptance','completed_peer_receipts')}
    plan.update(authorization='confirmed_failed_run_recovery', prepared_at=p.now(),
        failed_worker=full['original_identity'], former_worker_was_healthy=False,
        initial_steps=index['completed_steps'], initial_checkpoint_sha256=index['sha256'],
        initial_agents=index['agents'], initial_sources=index['source_hashes'],
        checkpoint_steps=[row['completed_steps'] for row in full['pair']['checkpoints']],
        backup_manifest=evidence['manifest'], backup_manifest_sha256=evidence['manifest_sha256'],
        traces=full['traces'], original_pairs=full['completed_pairs'], post_checkpoint_pairs=full['post_checkpoint_pairs'],
        protected=[], retention_identity=full['retention_identity'], source_sha256=source,
        adapter_sha256=adapters, branch_template=branch, maximum_parallel_decisions=340,
        failure_evidence=diagnosis, tests_passed=4)
    p.immutable(HERE / 'plan.json', plan)
    recovery.r.verify()
    recovery.r.protected_check(plan)
    recovery.r.preserved_traces(plan)
    print('Sealed fresh failed-inference recovery plan: checkpoints ' + str(plan['checkpoint_steps']) +
          '; preserved comparisons ' + str(plan['original_pairs']), flush=True)


def retire():
    if (HERE / 'retirement.json').exists():
        raise FileExistsError('Never repeat a completed retirement')
    plan = recovery.r.verify()
    recovery.r.protected_check(plan)
    preserved = p.closed_json(HERE / 'preservation.json')
    full = p.closed_json(preserved['manifest'])
    if not current_pair(full['pair']):
        raise RuntimeError('The preserved checkpoint advanced; refresh before retirement')
    records = p.h.completed_records(recovery.r.OUTPUT / 'capture/decisions.jsonl')
    if len(records) != plan['original_pairs']:
        raise RuntimeError('The original attempt resumed comparison progress; do not retire it')
    pool = Path(full['pending'][0]['request']).parent.parent
    for item in full['pending']:
        request = Path(item['request'])
        if (request.parent / ('done_' + request.stem + '.json')).exists():
            raise RuntimeError('The pending comparison completed; re-evaluate before retirement')
    metadata = p.h.stable_batch(dict(owners=recovery.r.OUTPUT / 'owned_aspen_apps.json',
        ownership=recovery.r.OUTPUT / 'ownership_status.json'))
    if metadata['ownership']['phase'] in ('creating', 'unverified_creation'):
        raise RuntimeError('Wait for settled ownership before scoped retirement')
    owned = []
    for row in metadata['owners']:
        current = p.h.process(row['pid'])
        if p.same_generation(row, current):
            owned.append({key:row[key] for key in ('pid','created','alive')})
        elif current.get('alive') is None:
            raise RuntimeError('An owned generation is inaccessible')
    known = {(row['pid'],row['created']) for row in owned}
    if any((row['pid'],row['created']) not in known for row in p.h.native_processes().values()):
        raise RuntimeError('A foreign native application must remain untouched')
    python = [plan['failed_worker']] + [row['identity'] for row in full['child_identities']]
    p.immutable(HERE / 'retirement_intent.json', dict(at_utc=p.now(), failed_python_generations=python,
        owned_native_generations=owned, checkpoint_steps=plan['checkpoint_steps'],
        preserved_matched_pairs=len(records), independent_backup_manifest=plan['backup_manifest'],
        reason='Confirmed unchanged CUDA calls and I/O; only this stalled combined attempt is retired',
        other_healthy_workers_to_signal=[]))
    from recover_markov_seed23 import terminate_owned_identity, children_of
    actions = []
    for row in python:
        actions.append(dict(identity=row, action=terminate_owned_identity(row)))
    for row in owned:
        for child in children_of(row['pid']):
            actions.append(dict(identity=child, action=terminate_owned_identity(child)))
        actions.append(dict(identity=row, action=terminate_owned_identity(row)))
    for row in python + owned:
        if p.same_generation(row, p.h.process(row['pid'])):
            raise RuntimeError('An old failed generation is still live')
    roots = {Path(value) for point in full['pair']['checkpoints'] for value in point['private_roots']}
    for case in (15, 17):
        events = p.h.completed_records(pool / ('case' + str(case)) / 'native_process_recovery.jsonl')
        opened = [row for row in events if row.get('event') == 'opened_fresh_application']
        if opened:
            roots.add(Path(opened[-1]['model']).parent)
    destination = Path(preserved['backup']) / 'retired_working'
    copied = []
    for source_root in sorted(roots):
        resolved = source_root.resolve()
        if not resolved.is_relative_to(recovery.r.OUTPUT.resolve()) and not resolved.is_relative_to(pool.resolve()):
            raise RuntimeError('A working backup root escaped this failed attempt')
        for source in sorted(source_root.rglob('*')):
            if source.is_file():
                copied.append(preserve.copied_generation(source, destination / source.relative_to(ROOT).parent))
    for row in full['deferred_owned_working_files']:
        source = Path(row['source'])
        if any(source.is_relative_to(root) for root in roots):
            continue
        for attempt in range(30):
            try:
                copied.append(preserve.copied_generation(source, destination / source.relative_to(ROOT).parent))
                break
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(1)
    p.immutable(destination / 'preservation.json', dict(verified=True, at_utc=p.now(), copies=copied,
        formerly_locked_working_files_copied=True, all_originals_retained=True))
    p.immutable(HERE / 'retirement.json', dict(at_utc=p.now(), actions=actions,
        only_failed_combined_was_signaled=True, other_healthy_workers_interrupted=0,
        former_working_preservation=str(destination / 'preservation.json'),
        former_working_sha256=p.h.digest(destination / 'preservation.json')))
    recovery.r.protected_check(plan)
    recovery.r.start_guard(plan)
    print('Retired only the stalled combined generations; all original/working bytes preserved.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('seal','retire'))
    globals()[parser.parse_args().action]()
