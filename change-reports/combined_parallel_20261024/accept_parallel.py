"""Accept only real four-learner continuation, cold restore and a parallel pair.

This tool cannot launch, stop or alter training. Its final mutation replaces
only the combined monitoring identity after preserving the former baseline.
"""
from copy import deepcopy
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import parallel_branches as p
import recover_parallel as activation


def accept():
    if (HERE/'acceptance.json').exists():raise FileExistsError('Inspect existing acceptance; never overwrite it')
    r=activation.r;plan=r.verify();r.protected_check(plan)
    paths=dict(validation=r.STATE/'validation.json',manifest=r.STATE/'manifest.json',
        status=r.OUTPUT/'status.json',restore=r.OUTPUT/'training_checkpoints/last_restore.json',
        index=r.OUTPUT/'training_checkpoints/latest.json',owners=r.OUTPUT/'owned_aspen_apps.json')
    if not all(path.is_file() for path in paths.values()):raise RuntimeError('Bounded continuation/cold production restore is still pending')
    data=p.h.stable_batch(paths);validation=data['validation'];manifest=data['manifest'];restore=data['restore']
    bounded=plan['initial_steps']+8
    if (not validation['passed'] or validation['index']['completed_steps']!=bounded
        or len(validation['index']['agents'])!=4 or validation['index']['source_hashes']!=plan['initial_sources']):
        raise RuntimeError('Eight-step four-learner checkpoint did not verify')
    for name,old in plan['initial_agents'].items():
        new=validation['index']['agents'][name]
        if new['train_steps']!=old['train_steps']+8 or new['replay_size']<old['replay_size']:
            raise RuntimeError('Four-learner learning counters did not advance exactly eight steps')
    for name,step in (('validation',plan['initial_steps']),('production',bounded)):
        row=validation['restore'] if name=='validation' else restore
        if row['completed_steps']!=step or not row['native_checksum_verified'] or not row['python_checksum_verified']:
            raise RuntimeError('Native/Python restore did not verify: '+name)
    identity=p.closed_json(r.STATE/'worker_0_identity.json')['identity']
    if (manifest.get('stage')!='production' or not manifest.get('validation_passed')
        or data['status']['state']!='running' or not p.same_generation(identity,p.h.process(identity['pid']))
        or data['status']['pid']!=identity['pid'] or len(manifest['job']['attempts'])!=2):
        raise RuntimeError('A separate live production generation has not been verified')
    events=p.h.completed_records(r.OUTPUT/'native_process_recovery.jsonl')
    full=[row for row in events if row.get('event')=='full_python_restore_verified'
          and row.get('completed_steps')==bounded and row.get('before')==row.get('after') and row.get('before')]
    if not full:raise RuntimeError('Full learner/controller equality across the separate cold restore is pending')
    pool=p.closed_json(r.STATE/'parallel_pool_0.json');directory=Path(pool['directory'])
    if not (directory/'decisions.jsonl').is_file():raise RuntimeError('The first actual parallel primary comparison is pending')
    decisions=p.h.completed_records(directory/'decisions.jsonl')
    if not decisions:raise RuntimeError('A complete primary parallel receipt is pending')
    first=decisions[0]
    if first['full_python_state_before']!=first['full_python_state_after'] or not first['source_map_unchanged']:
        raise RuntimeError('Parallel comparison mutated the complete live Python state')
    if set(first['consumed_sources'])!={'policy','supervisor'} or not all(first['consumed_sources'].values()):
        raise RuntimeError('Both unchanged counterfactual sources must commit')
    branches=[]
    for case in (15,17):
        worker=p.closed_json(directory/('case'+str(case))/'identity.json')
        if not p.same_generation(worker,p.h.process(worker['pid'])):raise RuntimeError('A production side worker is not live')
        branches.append(dict(case=case,identity=worker,journal=str(directory/('case'+str(case))/'owned_aspen_apps.json')))
    for batch in first['batches'].values():
        if not batch['parallel']:raise RuntimeError('This primary pair was not dispatched concurrently')
        for source,row in batch['branches'].items():
            if row['source']!=source or row['state']!='completed' or row['frozen_state_before']!=row['frozen_state_after']:
                raise RuntimeError('A counterfactual failed or mutated frozen state')
            p.verify_native_restore(row['native_restore'])
            if p.h.digest(row['trace'])!=row['trace_sha256']:raise RuntimeError('Side trace checksum changed')
    records=p.h.completed_records(r.OUTPUT/'capture/decisions.jsonl')
    primary=[row for row in records if row['step']==first['step'] and row['channel']==first['channel']]
    if len(primary)!=1 or primary[0]['status']!='completed':raise RuntimeError('The parallel pair did not enter the original primary results')
    trace=r.OUTPUT/'capture'/primary[0]['trace']
    with zipfile.ZipFile(trace) as zipped:
        if zipped.testzip() is not None:raise RuntimeError('Original primary NPZ checksum failed')
    # Every native app must be traced to the parent or its delegated journals.
    owners={(row['pid'],row['created']) for row in data['owners']}
    for row in p.h.native_processes().values():
        if (row['pid'],row['created']) not in owners:raise RuntimeError('A native app needs ownership reconciliation before acceptance')
    previous_path=p.h.HERE/'accepted_current_workers.json'
    previous=p.h.stable_batch({'baseline':previous_path})['baseline']
    p.immutable(HERE/'accepted_workers_before.json',previous)
    receipt=dict(passed=True,at_utc=p.now(),category='combined',output=str(r.OUTPUT),
        recovery_manifest=str(r.STATE/'manifest.json'),production_identity=identity,
        initial_completed_steps=plan['initial_steps'],bounded_completed_steps=bounded,
        production_restore=restore,full_python_restore=full[-1],initial_agents=plan['initial_agents'],
        bounded_agents=validation['index']['agents'],checkpoint_protocols_unchanged=True,
        all_143_source_hashes_verified=len(plan['source_sha256'])==143,tests_passed=42,
        parallel_branches=branches,first_parallel_primary=first,primary_trace_sha256=p.h.digest(trace),
        matched_pairs=len(records),remaining_pairs=320-len(records),replay_archive=manifest.get('replay_archive',[]),
        all_prior_results_and_recovery_data_preserved=True,retention_identity=plan['retention_identity'],
        pilot_acceptance=plan['pilot_acceptance'],transport_acceptance=plan['transport_acceptance'],
        measured_speedup=p.closed_json(HERE/'attempt4/pilot_acceptance.json')['comparison_speedup'],
        production_parallel_timing=first['elapsed_seconds'],memory=p.h.memory(),
        free_disk_bytes=p.shutil.disk_usage(ROOT.anchor).free,scientific_source_edits=0,shared_queue_plan_edits=0,
        authorized_healthy_combined_checkpoint_restarts=1,other_healthy_workers_interrupted=0)
    if not receipt['all_143_source_hashes_verified']:raise RuntimeError('Scientific source count differs')
    p.immutable(HERE/'acceptance.json',receipt)
    updated=deepcopy(previous)
    updated['workers']['combined']=dict(identity=identity,acceptance=str(HERE/'acceptance.json'),sha256=p.h.digest(HERE/'acceptance.json'))
    if p.h.stable_batch({'baseline':previous_path})['baseline']!=previous:
        raise RuntimeError('Monitoring baseline changed; preserve receipt and reconcile before registering')
    p.h.atomic_json(previous_path,updated)
    print('Accepted live combined parallel production: '+str(len(records))+'/320 primary pairs; cases 15/17; four learners/cold restore verified.',flush=True)


if __name__=='__main__':accept()
