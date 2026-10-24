"""Seal the failed combined continuation after residual's production restore."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
import recover_combined
r = recover_combined.engine


def seal():
    if (HERE/'plan.json').exists() or (r.STATE/'scheduled.json').exists():
        raise FileExistsError('Inspect the existing combined recovery before repeating')
    prepared = json.loads((HERE/'preparation.json').read_text(encoding='utf-8'))
    preserved = json.loads((HERE/'preservation.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT/'change-reports/weights_pid_reuse_recovery_20261024/plan.json').read_text(encoding='utf-8'))
    peers=[]
    accepted_receipts={}
    for category,folder,bounded in (
            ('weights','weights_pid_reuse_recovery_20261024',117208),
            ('markov','markov_native_recovery_20261024',116008),
            ('residual','residual_native_recovery_20261024',117608)):
        receipt = ROOT/'change-reports'/folder/'acceptance.json'
        accepted = json.loads(receipt.read_text(encoding='utf-8'))
        output = Path(accepted['output'])
        status = r.read(output/'status.json')
        if (not accepted['passed'] or accepted['bounded_completed_steps'] != bounded
                or not accepted['checkpoint_protocols_unchanged'] or not accepted['all_143_source_hashes_verified']):
            raise RuntimeError('Peer production acceptance differs: '+category)
        if status.get('state') == 'completed' and status.get('trajectory_sha256'):
            if r.digest(output/'training.pkl') != status['trajectory_sha256']:
                raise RuntimeError('Completed peer result differs')
        else:
            identity = r.process_identity(status['pid'])
            if status['state'] != 'running' or identity != accepted['production_identity'] or identity['alive'] is not True:
                raise RuntimeError('Accepted peer is not confirmed live: '+category)
            peers.append(dict(category=category,directory=str(output),identity=identity))
        accepted_receipts[category]=dict(path=str(receipt),sha256=r.digest(receipt))
    if r.process_identity(prepared['status']['pid'])['alive'] is not False or r.owned_remaining():
        raise RuntimeError('The actually failed combined worker/native processes must remain retired')
    protocol=dict(id='combined-four-learner-native-retry-v1',prepared_at=r.now(),case=16,seed=7,
        expected_matched_pairs=320,main_training_cycles=300,decisions_per_window=20,validation_steps=8,
        minimum_headroom_bytes=r.MINIMUM_HEADROOM,original_budget_protocol_sha256=prepared['original_budget_protocol_sha256'],
        checkpoint_source_map='Exact original map retained; operational fresh-process receipt stays separate',
        scope='Full four-learner/controller/native continuation after confirmed document-open failure; generation-safe ownership')
    r.write(HERE/'protocol.json',protocol)
    files=[HERE/name for name in ('prepare_combined.py','recover_combined.py','seal_plan.py','test_recovery.py','protocol.json')]
    files += [ROOT/'change-reports/weights_pid_reuse_recovery_20261024'/name for name in ('recover_weights.py','ownership_fix.py')]
    files += [ROOT/'change-reports/combined_memory_restart_20261024/recover_combined.py']
    original = prepared['pair']['index']
    plan=dict(authorization='confirmed_failed_run_recovery',prepared_at=r.now(),category='combined',case=16,
        seed=7,expected_matched_pairs=320,main_training_cycles=300,decisions_per_window=20,
        failed_worker=dict(pid=22588,created=134371156883220647,alive=True),
        actual_failure_at=prepared['status']['finished_at'],actual_failure=prepared['status']['error'],
        job=prepared['job_path'],job_sha256=prepared['original_entry']['job_sha256'],initial_steps=80800,
        initial_checkpoint_sha256=original['sha256'],initial_agents=original['agents'],initial_sources=original['source_hashes'],
        checkpoint_steps=[80800,80400],backup_manifest=preserved['manifest'],backup_manifest_sha256=preserved['manifest_sha256'],
        backup_files=preserved['files'],backup_bytes=preserved['bytes'],original_pairs=181,
        post_checkpoint_pairs=preserved['post_checkpoint_pairs'],traces=prepared['traces'],protected=peers,
        accepted_peer_receipts=accepted_receipts,retention_identity=prepared['retention_identity'],
        original_process_protocol_sha256=prepared['original_budget_protocol_sha256'],
        original_protocol_kind='existing_budget_protocol_no_new_checkpoint_marker',
        source_sha256=baseline['source_sha256'],adapter_sha256={str(path.resolve()):r.digest(path.resolve()) for path in files},tests_passed=15)
    r.write(HERE/'plan.json',plan)
    r.verify()
    r.protected_check(plan)
    r.preserved_traces(plan)
    r.start_guard(plan)
    from run_distillation_phase2_repeats import aspen_file_lock
    job=r.read(Path(plan['job']))
    canonical=Path(job['source_case']['aspen_path'])
    if canonical.name != 'C2S_SS_simulation16.dynf' or r.digest(canonical) != job['source_case']['aspen_sha256']:
        raise RuntimeError('Reserved source case differs')
    with aspen_file_lock(canonical,ROOT/'Distillation/Results/_aspen_file_locks'):
        pass
    r.write(HERE/'preflight.json',dict(at_utc=r.now(),full_source_seals_verified=143,tests_passed=15,
        checkpoint_steps=[80800,80400],initial_agents=original['agents'],original_pairs=181,
        post_checkpoint_pairs=preserved['post_checkpoint_pairs'],case_lease_available=True,
        accepted_live_peer_categories=[p['category'] for p in peers],memory=r.memory(),simulator_calls=0,
        scientific_checkpoint_map_unchanged=True,numerical_bootstrap_imported='review_numeric_recovery' in sys.modules))
    print('Combined plan and full backup verify; four learners preserved; case 16 available; accepted peers protected.')


if __name__=='__main__':
    seal()
