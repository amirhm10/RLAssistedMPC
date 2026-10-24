"""Seal residual's recovery after Markov has a verified production restore."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import recover_residual
r=recover_residual.engine


def seal():
    if (HERE/'plan.json').exists() or (r.STATE/'scheduled.json').exists():
        raise FileExistsError('Inspect the existing residual recovery before repeating')
    current=[]
    for category,folder,step in (('weights','20261024_weights_pid_reuse',117208),
                                 ('markov','20261024_markov_native_retry',116008)):
        state=ROOT/'Distillation/Results/state_recovery'/folder
        manifest=r.read(state/'manifest.json',{})
        output=ROOT/('Distillation/Results/state_recovery/20261021_remaining_active/e1/'+category+'/a1')
        status=r.read(output/'status.json',{})
        restored=r.read(output/'training_checkpoints/last_restore.json',{})
        if status.get('state')=='completed' and status.get('trajectory_sha256'):
            if r.digest(output/'training.pkl')==status['trajectory_sha256']:
                continue
        if (manifest.get('stage')!='production' or not manifest.get('validation_passed')
                or status.get('state')!='running' or restored.get('completed_steps')!=step
                or not restored.get('python_checksum_verified') or not restored.get('native_checksum_verified')):
            raise RuntimeError(category+' production restoration is not yet verified')
        identity=r.process_identity(status['pid'])
        if identity['alive'] is not True or identity!=manifest['job']['identity']:
            raise RuntimeError('Current peer production identity differs')
        current.append(dict(category=category,directory=str(output),identity=identity))
    preparation=json.loads((HERE/'preparation.json').read_text(encoding='utf-8'))
    preserved=json.loads((HERE/'preservation.json').read_text(encoding='utf-8'))
    shared=ROOT/'change-reports/weights_pid_reuse_recovery_20261024'
    baseline=json.loads((shared/'plan.json').read_text(encoding='utf-8'))
    before=json.loads((ROOT/'change-reports/hourly_run_checks_20261024/audit_20261024T1033.json').read_text(encoding='utf-8'))
    former=next(v['after'] for v in before['E1'] if v['category']=='residual')
    protocol=dict(id='residual-isolated-native-retry-v1',prepared_at=r.now(),case=17,seed=7,
        expected_matched_pairs=80,main_training_cycles=300,decisions_per_window=20,
        validation_steps=8,minimum_headroom_bytes=r.MINIMUM_HEADROOM,
        original_process_protocol_sha256=preparation['original_process_protocol_sha256'],
        scope='Fresh-owned-process full-state retry after document-open failure, generation-safe ownership, existing scientific checkpoint markers unchanged')
    r.write(HERE/'protocol.json',protocol)
    protected=[v for v in preparation['protected'] if v['category']=='combined']+current
    files=('prepare_residual.py','recover_residual.py','seal_plan.py','test_entrypoint.py','protocol.json',
           '../weights_pid_reuse_recovery_20261024/recover_weights.py',
           '../weights_pid_reuse_recovery_20261024/ownership_fix.py',
           '../weights_pid_reuse_recovery_20261024/prepare_recovery.py',
           '../weights_pid_reuse_recovery_20261024/test_recovery.py')
    plan=dict(authorization='confirmed_failed_run_recovery',prepared_at=r.now(),category='residual',case=17,
        seed=7,expected_matched_pairs=80,main_training_cycles=300,decisions_per_window=20,
        failed_worker={k:former[k] for k in ('pid','created','alive')},job=preparation['job_path'],
        job_sha256=preparation['original_entry']['job_sha256'],initial_steps=117600,
        initial_checkpoint_sha256=preparation['pair']['index']['sha256'],
        initial_agents=preparation['pair']['index']['agents'],checkpoint_steps=[117600,117200],
        backup_manifest=preserved['manifest'],backup_manifest_sha256=preserved['manifest_sha256'],
        backup_files=preserved['files'],backup_bytes=preserved['bytes'],original_pairs=69,
        post_checkpoint_pairs=1,traces=preparation['traces'],protected=protected,
        retention_identity=preparation['retention_identity'],
        original_process_protocol_sha256=preparation['original_process_protocol_sha256'],
        source_sha256=baseline['source_sha256'],
        adapter_sha256={str((HERE/f).resolve()):r.digest((HERE/f).resolve()) for f in files},tests_passed=18)
    r.write(HERE/'plan.json',plan)
    r.verify();r.protected_check(plan);r.preserved_traces(plan)
    if r.owned_remaining():
        raise RuntimeError('A residual-owned application remains')
    from run_distillation_phase2_repeats import aspen_file_lock
    job=r.read(Path(plan['job']));canonical=Path(job['source_case']['aspen_path'])
    if canonical.name!='C2S_SS_simulation17.dynf' or r.digest(canonical)!=job['source_case']['aspen_sha256']:
        raise RuntimeError('Reserved source case differs')
    with aspen_file_lock(canonical,ROOT/'Distillation/Results/_aspen_file_locks'):
        pass
    r.write(HERE/'preflight.json',dict(at_utc=r.now(),full_source_seals_verified=143,tests_passed=18,
        checkpoint_steps=[117600,117200],original_pairs=69,post_checkpoint_pairs=1,
        case_lease_available=True,live_peer_categories=[p['category'] for p in protected],
        memory=r.memory(),numerical_bootstrap_imported='review_numeric_recovery' in sys.modules,
        simulator_calls=0))
    print('Residual plan/backup seals verify; case 17 available; verified Markov/weights production peers protected.')


if __name__=='__main__':
    seal()
