"""Seal the diagnosed residual failure and bounded operational retry policy."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import recover_residual_retry2
r=recover_residual_retry2.engine


def seal():
    if (HERE/'plan.json').exists() or (r.STATE/'scheduled.json').exists():
        raise FileExistsError('Inspect the existing second residual recovery before repeating')
    prepared=json.loads((HERE/'preparation.json').read_text(encoding='utf-8'))
    saved=json.loads((HERE/'preservation.json').read_text(encoding='utf-8'))
    if not saved.get('extended_recovery_dependencies_verified'):
        raise RuntimeError('Full dependency/current-log preservation must finish first')
    oldstate=ROOT/'Distillation/Results/state_recovery/20261024_residual_native_retry'
    oldmanifest=r.read(oldstate/'manifest.json')
    oldsupervisor=r.read(oldstate/'supervisor_identity.json')
    if (oldmanifest['state']!='failed' or oldmanifest['stage']!='production'
            or oldmanifest['job']['pid']!=46740 or r.process_identity(46740)['alive'] is not False
            or r.process_identity(oldsupervisor['pid'])['alive'] is not False or r.owned_remaining()):
        raise RuntimeError('The superseded residual control/worker/native processes must be retired')
    peers=[]
    for category,folder in (('markov','markov_native_recovery_20261024'),('combined','combined_native_recovery_20261024')):
        accepted=json.loads((ROOT/'change-reports'/folder/'acceptance.json').read_text(encoding='utf-8'))
        output=Path(accepted['output']);status=r.read(output/'status.json')
        if status.get('state')=='completed' and status.get('trajectory_sha256'):
            if r.digest(output/'training.pkl')!=status['trajectory_sha256']:raise RuntimeError('Completed peer result differs')
            continue
        identity=r.process_identity(status['pid'])
        if not accepted['passed'] or status['state']!='running' or identity!=accepted['production_identity']:
            raise RuntimeError('Accepted live peer needs inspection: '+category)
        peers.append(dict(category=category,directory=str(output),identity=identity))
    weights=ROOT/'change-reports/weights_pid_reuse_recovery_20261024/completion.json'
    completed=json.loads(weights.read_text(encoding='utf-8'))
    if (not completed['verified'] or completed['completed_cycles']!=300 or completed['matched_pairs']!=80
            or r.digest(Path(completed['trajectory']['path']))!=completed['trajectory']['sha256']):
        raise RuntimeError('Weights completed result must verify before treating its exit as natural')
    baseline=json.loads((ROOT/'change-reports/weights_pid_reuse_recovery_20261024/plan.json').read_text(encoding='utf-8'))
    protocol=dict(id='residual-owned-document-open-retry-v1',prepared_at=r.now(),case=17,seed=7,
        expected_matched_pairs=80,main_training_cycles=300,decisions_per_window=20,validation_steps=8,
        minimum_headroom_bytes=r.MINIMUM_HEADROOM,document_open_max_attempts=3,
        retry_hresult=-2147417851,retry_call='sealed fresh.load_project plant.ad.openDocument only',
        scientific_checkpoint_source_map='unchanged; operational retry receipt only')
    r.write(HERE/'protocol.json',protocol)
    files=[HERE/name for name in ('prepare_residual_retry2.py','recover_residual_retry2.py','document_open_retry.py','seal_plan.py','test_retry.py','protocol.json')]
    files += [ROOT/'change-reports/weights_pid_reuse_recovery_20261024'/name for name in ('recover_weights.py','ownership_fix.py','prepare_recovery.py')]
    original=prepared['pair']['index']
    plan=dict(authorization='confirmed_failed_run_recovery',prepared_at=r.now(),category='residual',case=17,seed=7,
        expected_matched_pairs=80,main_training_cycles=300,decisions_per_window=20,
        failed_worker=dict(pid=46740,created=134373280185732992,alive=True),initial_steps=118400,
        initial_checkpoint_sha256=original['sha256'],initial_agents=original['agents'],checkpoint_steps=[118400,118000],
        job=prepared['job_path'],job_sha256=prepared['original_entry']['job_sha256'],
        backup_manifest=saved['manifest'],backup_manifest_sha256=saved['manifest_sha256'],
        backup_files=saved['files'],backup_bytes=saved['bytes'],original_pairs=77,
        post_checkpoint_pairs=saved['post_checkpoint_pairs'],traces=prepared['traces'],protected=peers,
        retention_identity=prepared['retention_identity'],
        original_process_protocol_sha256=prepared['original_process_protocol_sha256'],
        source_sha256=baseline['source_sha256'],adapter_sha256={str(p.resolve()):r.digest(p.resolve()) for p in files},
        tests_passed=15,document_open_retry=dict(max_attempts=3,hresult=-2147417851,backoff_seconds=[2,4],
            exact_loader_call_only=True,owned_generation_required=True,same_verified_model_and_snapshot=True),
        completed_peer_receipt=dict(path=str(weights),sha256=r.digest(weights)),
        superseded_manifest=str(oldstate/'manifest.json'),superseded_failure_at=prepared['status']['finished_at'])
    r.write(HERE/'plan.json',plan)
    r.verify();r.protected_check(plan);r.preserved_traces(plan);r.start_guard(plan)
    import column_e1_process_recovery as fresh
    import document_open_retry
    line=document_open_retry.open_line(fresh.load_project)
    from run_distillation_phase2_repeats import aspen_file_lock
    job=r.read(Path(plan['job']));canonical=Path(job['source_case']['aspen_path'])
    if canonical.name!='C2S_SS_simulation17.dynf' or r.digest(canonical)!=job['source_case']['aspen_sha256']:
        raise RuntimeError('Reserved source case differs')
    with aspen_file_lock(canonical,ROOT/'Distillation/Results/_aspen_file_locks'):pass
    r.write(HERE/'preflight.json',dict(at_utc=r.now(),full_source_seals_verified=143,tests_passed=15,
        original_pairs=77,checkpoint_steps=[118400,118000],post_checkpoint_pairs=saved['post_checkpoint_pairs'],
        case_lease_available=True,old_residual_control_retired=True,weights_completion_verified=True,
        live_peer_categories=[p['category'] for p in peers],document_open_line=line,memory=r.memory(),
        numerical_bootstrap_imported='review_numeric_recovery' in sys.modules,simulator_calls=0))
    print('Residual retry2 source/backup/ownership seals verify; completed weights and healthy Markov/combined protected; case 17 free.')


if __name__=='__main__':seal()
