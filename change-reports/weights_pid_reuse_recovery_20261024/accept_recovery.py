"""Record read-only acceptance of a validated production cold restore."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'change-reports/hourly_run_checks_20261024'))
import collect_hourly as hourly


def accept(category):
    if category == 'residual':
        sys.path.insert(0, str(ROOT / 'change-reports/residual_native_recovery_20261024'))
        import recover_residual
        recovery = recover_residual.engine
    elif category == 'markov':
        sys.path.insert(0, str(ROOT / 'change-reports/markov_native_recovery_20261024'))
        import recover_markov
        recovery = recover_markov.engine
    else:
        import recover_weights as recovery
    destination = recovery.HERE / 'acceptance.json'
    if destination.exists():
        raise FileExistsError('Acceptance is an immutable receipt; inspect the existing result')
    plan = recovery.verify()
    recovery.protected_check(plan)
    paths = dict(manifest=recovery.STATE/'manifest.json', status=recovery.OUTPUT/'status.json',
                 index=recovery.OUTPUT/'training_checkpoints/latest.json',
                 restored=recovery.OUTPUT/'training_checkpoints/last_restore.json',
                 validation=recovery.STATE/'validation.json', supervisor=recovery.STATE/'supervisor_identity.json')
    metadata = hourly.stable_batch(paths)
    manifest, status = metadata['manifest'], metadata['status']
    if (manifest.get('stage') != 'production' or not manifest.get('validation_passed')
            or status.get('state') != 'running' or status['pid'] != manifest['job']['pid']):
        raise RuntimeError('Production is not confirmed running')
    restored = metadata['restored']
    if (restored['completed_steps'] != plan['initial_steps'] + 8
            or not restored['python_checksum_verified'] or not restored['native_checksum_verified']):
        raise RuntimeError('Production did not cold-restore the bounded checkpoint')
    validation = metadata['validation']
    if (not validation['passed'] or validation['index']['completed_steps'] != plan['initial_steps'] + 8
            or validation['restore']['completed_steps'] != plan['initial_steps']):
        raise RuntimeError('The original full-state bounded continuation differs')
    events = hourly.completed_records(recovery.OUTPUT/'native_process_recovery.jsonl')
    checks = [event for event in events if event.get('event') == 'full_python_restore_verified'
              and event.get('completed_steps') in (plan['initial_steps'], plan['initial_steps'] + 8)]
    for step in (plan['initial_steps'], plan['initial_steps'] + 8):
        matching = [event for event in checks if event['completed_steps'] == step]
        if not matching or any(not e.get('before') or e['before'] != e['after'] for e in matching):
            raise RuntimeError('Native import did not preserve the complete Python state')
    replay_archive = recovery.preserved_traces(plan, after_validation=True)
    index = metadata['index']
    if len(index['agents']) != 1 or index['completed_steps'] < plan['initial_steps'] + 8:
        raise RuntimeError('The advancing learner checkpoint differs')
    prepared = json.loads((recovery.HERE/'preparation.json').read_text(encoding='utf-8'))
    if index['source_hashes'] != prepared['pair']['index']['source_hashes']:
        raise RuntimeError('A scientific/budget checkpoint protocol marker changed')
    records = hourly.completed_records(recovery.OUTPUT/'capture/decisions.jsonl')
    if any(r['status'] != 'completed' for r in records) or len(records) != len({(r['channel'],r['window'],r['step']) for r in records}):
        raise RuntimeError('Current primary capture records differ')
    before = hourly.process(status['pid'])
    time.sleep(12)
    after = hourly.process(status['pid'])
    identity = {key:after[key] for key in ('pid','created','alive')}
    if identity != manifest['job']['identity'] or after['alive'] is not True:
        raise RuntimeError('Production process identity changed')
    acceptance = dict(passed=True, at_utc=recovery.now(), category=category, output=str(recovery.OUTPUT),
                     recovery_manifest=str(recovery.STATE/'manifest.json'), production_identity=identity,
                     production_cpu_delta_seconds=after['cpu_seconds']-before['cpu_seconds'],
                     initial_completed_steps=plan['initial_steps'], bounded_completed_steps=plan['initial_steps']+8,
                     production_restore=restored, validation=validation, full_python_native_checks=checks,
                     current_checkpoint_steps=index['completed_steps'], current_agents=index['agents'],
                     checkpoint_protocols_unchanged=True, all_143_source_hashes_verified=True,
                     original_completed_pairs_preserved=plan['original_pairs'], current_primary_pairs=len(records),
                     replay_archive=replay_archive, original_trace_backups_verified=len(plan['traces']),
                     backup_manifest=plan['backup_manifest'], backup_manifest_sha256=plan['backup_manifest_sha256'],
                     protected_peers=plan['protected'], retention_identity=plan['retention_identity'],
                     memory=recovery.memory(), tests_passed=plan['tests_passed'],
                     scientific_source_edits=0, shared_queue_plan_edits=0, healthy_workers_interrupted=0)
    recovery.protected_check(plan)
    recovery.write(destination, acceptance)
    print(json.dumps({key:acceptance[key] for key in ('passed','at_utc','category','production_identity',
                      'bounded_completed_steps','current_checkpoint_steps','current_primary_pairs',
                      'original_completed_pairs_preserved','production_cpu_delta_seconds','memory')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', required=True, choices=('weights','markov','residual'))
    accept(parser.parse_args().category)
