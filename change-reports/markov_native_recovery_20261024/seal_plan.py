"""Seal case-15 recovery only after weights has a verified production restore."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recover_markov
r = recover_markov.engine


def seal():
    if (HERE / 'plan.json').exists() or (r.STATE / 'scheduled.json').exists():
        raise FileExistsError('Inspect the existing plan/launch before repeating')
    weights = ROOT / 'Distillation/Results/state_recovery/20261024_weights_pid_reuse'
    manifest = r.read(weights / 'manifest.json', {})
    output = ROOT / 'Distillation/Results/state_recovery/20261021_remaining_active/e1/weights/a1'
    status = r.read(output / 'status.json', {})
    restored = r.read(output / 'training_checkpoints/last_restore.json', {})
    if (manifest.get('stage') != 'production' or not manifest.get('validation_passed')
            or status.get('state') != 'running' or restored.get('completed_steps') != 117208
            or not restored.get('python_checksum_verified') or not restored.get('native_checksum_verified')):
        raise RuntimeError('Weights production restoration is not yet verified; do not launch Markov')
    current = r.process_identity(status['pid'])
    if current['alive'] is not True or current != manifest['job']['identity']:
        raise RuntimeError('Weights production identity is not confirmed')
    preparation = json.loads((HERE / 'preparation.json').read_text(encoding='utf-8'))
    preservation = json.loads((HERE / 'preservation.json').read_text(encoding='utf-8'))
    shared = ROOT / 'change-reports/weights_pid_reuse_recovery_20261024'
    original_weights_plan = json.loads((shared / 'plan.json').read_text(encoding='utf-8'))
    former = json.loads((ROOT / 'change-reports/hourly_run_checks_20261024/audit_20261024T1033.json').read_text(encoding='utf-8'))
    former = next(v['after'] for v in former['E1'] if v['category'] == 'markov')
    protocol = dict(id='markov-isolated-native-retry-v1', prepared_at=r.now(), case=15, seed=7,
                    expected_matched_pairs=80, main_training_cycles=300, decisions_per_window=20,
                    validation_steps=8, minimum_headroom_bytes=r.MINIMUM_HEADROOM,
                    original_process_protocol_sha256=preparation['original_process_protocol_sha256'],
                    scope='Fresh-owned-process full-state retry after native document-open failure; generation-safe ownership lookup in this worker; existing scientific checkpoint protocol markers unchanged')
    r.write(HERE / 'protocol.json', protocol)
    protected = preparation['protected'] + [dict(category='weights', directory=str(output), identity=current)]
    files = ('recover_markov.py', 'prepare_markov.py', 'seal_plan.py', 'protocol.json',
             '../weights_pid_reuse_recovery_20261024/recover_weights.py',
             '../weights_pid_reuse_recovery_20261024/ownership_fix.py',
             '../weights_pid_reuse_recovery_20261024/prepare_recovery.py',
             '../weights_pid_reuse_recovery_20261024/test_recovery.py')
    plan = dict(authorization='confirmed_failed_run_recovery', prepared_at=r.now(), category='markov', case=15,
                seed=7, expected_matched_pairs=80, main_training_cycles=300, decisions_per_window=20,
                failed_worker={k:former[k] for k in ('pid', 'created', 'alive')},
                job=preparation['job_path'], job_sha256=preparation['original_entry']['job_sha256'],
                initial_steps=116000, initial_checkpoint_sha256=preparation['pair']['index']['sha256'],
                initial_agents=preparation['pair']['index']['agents'], checkpoint_steps=[116000, 115600],
                backup_manifest=preservation['manifest'], backup_manifest_sha256=preservation['manifest_sha256'],
                backup_files=preservation['files'], backup_bytes=preservation['bytes'], original_pairs=60,
                post_checkpoint_pairs=0, traces=preparation['traces'], protected=protected,
                retention_identity=preparation['retention_identity'],
                original_process_protocol_sha256=preparation['original_process_protocol_sha256'],
                source_sha256=original_weights_plan['source_sha256'],
                adapter_sha256={str((HERE/name).resolve()):r.digest((HERE/name).resolve()) for name in files}, tests_passed=17,
                weights_production_restore=restored)
    r.write(HERE / 'plan.json', plan)
    verified = r.verify()
    r.protected_check(verified)
    r.preserved_traces(verified)
    if r.owned_remaining():
        raise RuntimeError('An owned Markov application remains')
    from run_distillation_phase2_repeats import aspen_file_lock
    job = r.read(Path(plan['job']))
    canonical = Path(job['source_case']['aspen_path'])
    if canonical.name != 'C2S_SS_simulation15.dynf' or r.digest(canonical) != job['source_case']['aspen_sha256']:
        raise RuntimeError('The reserved source case differs')
    with aspen_file_lock(canonical, ROOT/'Distillation/Results/_aspen_file_locks'):
        pass
    r.write(HERE/'preflight.json', dict(at_utc=r.now(), full_source_seals_verified=143, tests_passed=17,
            checkpoint_steps=[116000,115600], original_pairs=60, case_lease_available=True,
            live_peer_categories=[p['category'] for p in protected], memory=r.memory(),
            numerical_bootstrap_imported='review_numeric_recovery' in sys.modules, simulator_calls=0))
    print('Markov plan and backup seals verify; case 15 available; weights production restore and healthy peers protected.')


if __name__ == '__main__':
    seal()
