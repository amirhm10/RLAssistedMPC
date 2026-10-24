"""Seal and execute one conditional combined-only parallel handoff.

Native fidelity/benefit receipts, independent full checkpoint/results copies,
generation-safe process ownership and a second working-tree backup are gates.
There is no global cleanup, training reset or shared-plan mutation.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import parallel_branches as p
import recover_parallel as activation
import preserve
from aspen_retention import still_current,Deferred
from transport_pilot import remaining_work,benefit


def persisted_pair_current(pair):
    from copy import deepcopy
    pair=deepcopy(pair)
    pair['layout']['index']=tuple(pair['layout']['index'])
    return still_current(pair)


def refreshed_benefit():
    accepted=p.closed_json(HERE/'attempt4/pilot_acceptance.json')
    exported=p.closed_json(HERE/'transport_acceptance.json')
    work=remaining_work()
    elapsed=(datetime.now(timezone.utc)-datetime(2026,10,24,19,51,54,tzinfo=timezone.utc)).total_seconds()/3600
    prediction=benefit(work,accepted['comparison_speedup'],exported['input_receipt']['elapsed_seconds'],elapsed_hours=elapsed)
    # Keep all private side data, even after each request completes. Forecast
    # includes the replay suffix, technical checks and a 48-GiB reserve.
    side=(work['remaining_pairs']+len(work['replay_pairs'])+4)*exported['projected_side_data_bytes_per_pair']+48*2**30
    free=p.shutil.disk_usage(ROOT.anchor).free
    if not prediction['enough_time_saved'] or free<side:
        raise RuntimeError('Refreshed time or all-data-preserved disk gate is insufficient')
    return work,dict(prediction=prediction,required_disk_bytes=side,free_disk_bytes=free,at_utc=p.now())


def seal():
    if (HERE/'plan.json').exists():raise FileExistsError('Inspect the sealed handoff before repeating')
    preserved=p.closed_json(HERE/'preservation.json')
    full=p.closed_json(preserved['manifest'])
    if not preserved['verified'] or p.h.digest(preserved['manifest'])!=preserved['manifest_sha256']:
        raise RuntimeError('Preservation receipt changed')
    pair=full['pair']
    if not persisted_pair_current(pair):raise Deferred('Refresh the newest two complete checkpoints before handoff')
    work,estimate=refreshed_benefit()
    if work['index']['sha256']!=pair['index']['sha256']:raise Deferred('The verified checkpoint advanced')
    # Refresh short append-only scientific results after the long state copy.
    # Every already-complete trace must have a verified independent copy.
    saved={r['source']:r for r in full['files']};extra=[]
    folder=Path(preserved['backup'])/'final_results'
    for source in sorted((preserve.OUTPUT/'capture').rglob('*')):
        if not source.is_file():continue
        row=saved.get(str(source))
        if row and tuple(row['source_version'])==tuple(p.h.version(source)):continue
        row=preserve.copied_generation(source,folder/source.relative_to(preserve.OUTPUT).parent)
        saved[str(source)]=row;extra.append(row)
    records=p.h.completed_records(preserve.OUTPUT/'capture/decisions.jsonl')
    traces=[]
    for row in records:
        source=(preserve.OUTPUT/'capture'/row['trace']).resolve();copied=saved.get(str(source))
        if not copied or p.h.digest(copied['local_copy'])!=copied['sha256']:raise Deferred('A new trace needs preservation')
        traces.append(dict(source=str(source),backup=copied['local_copy'],sha256=copied['sha256'],
                           step=row['step'],committed=row['step']<pair['index']['completed_steps']))
    peers={}
    for category,report in (('weights','weights_pid_reuse_recovery_20261024'),
        ('markov','markov_native_recovery_20261024'),('residual','residual_native_retry2_20261024')):
        path=ROOT/'change-reports'/report/'completion.json';receipt=p.closed_json(path)
        if not receipt.get('verified',receipt.get('passed',False)):
            raise RuntimeError('Completed peer evidence needs inspection: '+category)
        peers[category]=dict(path=str(path),sha256=p.h.digest(path))
    old=p.closed_json(ROOT/'change-reports/combined_native_recovery_20261024/plan.json')
    pilot=p.closed_json(HERE/'attempt4/pilot_plan.json')
    template={k:pilot[k] for k in ('category','cases','science_hashes','snapshot_source_hashes','adapter_hashes')}
    files=list(HERE.glob('*.py'))+[ROOT/'change-reports/combined_native_recovery_20261024/recover_combined.py',
        ROOT/'change-reports/weights_pid_reuse_recovery_20261024/recover_weights.py',
        ROOT/'change-reports/weights_pid_reuse_recovery_20261024/ownership_fix.py',
        ROOT/'change-reports/combined_memory_restart_20261024/recover_combined.py']
    hashes={str(path.resolve()):p.h.digest(path) for path in files}
    template['adapter_hashes']=dict(template['adapter_hashes'],**hashes)
    index=pair['index']
    plan=dict(authorization='user_approved_verified_parallel_migration',prepared_at=p.now(),category='combined',case=16,
        seed=7,expected_matched_pairs=320,main_training_cycles=300,decisions_per_window=20,
        failed_worker=full['original_identity'],former_worker_was_healthy=True,
        job=old['job'],job_sha256=old['job_sha256'],initial_steps=index['completed_steps'],
        initial_checkpoint_sha256=index['sha256'],initial_agents=index['agents'],initial_sources=index['source_hashes'],
        checkpoint_steps=[r['completed_steps'] for r in pair['checkpoints']],
        backup_manifest=preserved['manifest'],backup_manifest_sha256=preserved['manifest_sha256'],
        traces=traces,original_pairs=len(records),post_checkpoint_pairs=sum(not x['committed'] for x in traces),
        protected=[],completed_peer_receipts=peers,retention_identity=full['retention_identity'],
        original_process_protocol_sha256=old['original_process_protocol_sha256'],source_sha256=old['source_sha256'],
        adapter_sha256=hashes,branch_template=template,maximum_parallel_decisions=340,
        net_saving_after_development_and_replay_hours=estimate['prediction']['conservative_net_saved_hours'],
        migration_benefit=estimate,tests_passed=40)
    for label,path in (('pilot_acceptance',HERE/'attempt4/pilot_acceptance.json'),
                       ('transport_acceptance',HERE/'transport_acceptance.json')):
        plan[label]=dict(path=str(path),sha256=p.h.digest(path))
    p.immutable(HERE/'final_result_copies.json',dict(at_utc=p.now(),copies=extra))
    p.immutable(HERE/'plan.json',plan)
    activation.r.verify();activation.r.protected_check(plan);activation.r.preserved_traces(plan)
    print('Sealed conditional handoff: checkpoints '+str(plan['checkpoint_steps'])+'; net hours saved '+str(round(plan['net_saving_after_development_and_replay_hours'],2)),flush=True)


def retire():
    if (HERE/'retirement.json').exists():raise FileExistsError('Never repeat a recorded retirement')
    plan=activation.r.verify();activation.r.protected_check(plan)
    preserved=p.closed_json(HERE/'preservation.json');full=p.closed_json(preserved['manifest'])
    if not persisted_pair_current(full['pair']):raise Deferred('Refresh the checkpoint before stopping combined')
    work,estimate=refreshed_benefit()
    expected=plan['failed_worker'];actual=p.h.process(expected['pid'])
    if not p.same_generation(expected,actual):raise RuntimeError('The original combined generation changed')
    current=p.h.stable_batch({'owners':preserve.OUTPUT/'owned_aspen_apps.json',
        'ownership':preserve.OUTPUT/'ownership_status.json','status':preserve.OUTPUT/'status.json'})
    if current['ownership']['phase'] in ('creating','unverified_creation') or current['status']['state']!='running':
        raise Deferred('Combined ownership/status is changing; wait')
    records=p.h.completed_records(preserve.OUTPUT/'capture/decisions.jsonl')
    if len(records)!=plan['original_pairs']:raise Deferred('An additional primary result needs preservation and resealing')
    from recover_markov_seed23 import terminate_owned_identity,children_of
    owned=[]
    for row in current['owners']:
        now=p.h.process(row['pid'])
        if p.same_generation(row,now):owned.append({k:row[k] for k in ('pid','created','alive')})
        elif now.get('alive') is None:raise RuntimeError('Owned native generation is inaccessible')
    # Reviewable immutable intent exists before the sole Python signal.
    p.immutable(HERE/'retirement_intent.json',dict(at_utc=p.now(),original=expected,native_owned=owned,
        checkpoint_steps=plan['checkpoint_steps'],estimate=estimate,all_results_preserved=len(records),
        complete_native_python_checkpoints_verified=True,other_workers_to_signal=[]))
    actions=[dict(identity=expected,action=terminate_owned_identity(expected))]
    for row in owned:
        for child in children_of(row['pid']):actions.append(dict(identity=child,action=terminate_owned_identity(child)))
        actions.append(dict(identity=row,action=terminate_owned_identity(row)))
    for row in owned:
        if p.same_generation(row,p.h.process(row['pid'])):raise RuntimeError('An original owned native application remains')
    # All opened-working originals remain in place. With their owner retired,
    # copy the entire pinned roots, including formerly locked databases.
    copies=[];folder=Path(preserved['backup'])/'retired_working'
    roots={Path(x) for point in full['pair']['checkpoints'] for x in point['private_roots']}
    for root in roots:
        for source in sorted(root.rglob('*')):
            if source.is_file():copies.append(preserve.copied_generation(source,folder/source.relative_to(ROOT).parent))
    for row in full['deferred_owned_working_files']:
        source=Path(row['source'])
        if any(source.is_relative_to(root) for root in roots):continue
        # The old independent supervisor observes its child exit and releases
        # its one-byte mutex normally. Never signal that supervisor.
        for attempt in range(30):
            try:
                copies.append(preserve.copied_generation(source,folder/source.relative_to(ROOT).parent));break
            except PermissionError:
                if attempt==29:raise
                time.sleep(1)
    p.immutable(folder/'preservation.json',dict(verified=True,at_utc=p.now(),copies=copies,
        formerly_locked_working_files_copied=True,all_originals_retained=True))
    p.immutable(HERE/'retirement.json',dict(at_utc=p.now(),actions=actions,only_combined_was_signaled=True,
        former_working_preservation=str(folder/'preservation.json'),former_working_sha256=p.h.digest(folder/'preservation.json')))
    activation.r.protected_check(plan);activation.r.start_guard(plan)
    print('Only original combined retired; complete checkpoint/results and all formerly opened working files preserved.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('seal','retire'))
    args=parser.parse_args();globals()[args.action]()
