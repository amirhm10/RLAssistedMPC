"""Validate the actual durable CPU/native transport and refreshed benefit gate."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import math
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import parallel_branches as p


def remaining_work():
    folder=p.h.OUTPUTS['combined']
    data=p.h.stable_batch({'sampling':folder/'capture/sampling.json','index':folder/'training_checkpoints/latest.json'})
    records=p.h.completed_records(folder/'capture/decisions.jsonl')
    done={(r['channel'],r['step']) for r in records};longs={tuple(x) for x in data['sampling']['long_steps']}
    pending=[]
    for site,steps in data['sampling']['selected_steps'].items():
        for step in steps:
            if (site.split(':')[0],step) not in done:pending.append((site,step,(site,step) in longs))
    if len(records)+len(pending)!=320:raise RuntimeError('Remaining scientific schedule does not reconcile')
    return dict(records=records,index=data['index'],remaining_pairs=len(pending),
                remaining_long_pairs=sum(r[2] for r in pending),
                branch_steps=sum(1600 if r[2] else 400 for r in pending),
                main_steps=120000-records[-1]['step']-1,
                replay_pairs=[r for r in records if r['step']>=data['index']['completed_steps']])


def benefit(work,speedup,transport_seconds,*,elapsed_hours):
    baseline=p.closed_json(ROOT/'change-reports/combined_eta_capacity_20261024/assessment.json')['timing_model']
    main=work['main_steps']*baseline['main_seconds_per_step']/3600
    comparison=(work['branch_steps']*baseline['branch_seconds_per_step']+
                work['remaining_pairs']*baseline['fixed_seconds_per_pair'])/3600
    transport=work['remaining_pairs']*max(60.,transport_seconds)/3600
    replay=sum((2*r['rollout_steps']*baseline['branch_seconds_per_step']+baseline['fixed_seconds_per_pair'])/
               speedup+max(60.,transport_seconds) for r in work['replay_pairs'])/3600
    saved=comparison*(1-1/speedup)-transport-replay-1.-elapsed_hours
    return dict(serial_future_hours=main+comparison,parallel_future_hours=main+comparison/speedup+transport+replay+1.,
        main_training_hours=main,remaining_comparison_hours=comparison,
        transport_allowance_hours=transport,replay_allowance_hours=replay,
        restart_validation_allowance_hours=1.,elapsed_development_and_tests_hours=elapsed_hours,
        conservative_net_saved_hours=saved,
        enough_time_saved=speedup>=1.25 and saved>=3.)


def run():
    run=HERE/'transport_test'
    if run.exists():raise FileExistsError('Inspect the existing transport test; do not duplicate it')
    plan=p.closed_json(HERE/'attempt4/pilot_plan.json')
    acceptance=p.closed_json(HERE/'attempt4/pilot_acceptance.json')
    cleanup=p.closed_json(HERE/'attempt4/pilot_cleanup.json')
    if not acceptance['passed'] or not acceptance['enough_time_saved'] or not cleanup['all_owned_worker_processes_retired']:
        raise RuntimeError('The full serial/parallel native benchmark must pass and retire first')
    template={k:plan[k] for k in ('category','cases','protected_main','science_hashes','snapshot_source_hashes','adapter_hashes')}
    template['adapter_hashes']=dict(template['adapter_hashes'],**{str(HERE/n):p.h.digest(HERE/n)
        for n in ('session_adapter.py','transport_pilot.py')})
    probe=dict(template,case=15,parent_identity={k:p.h.process(__import__('os').getpid())[k] for k in ('pid','created','alive')},
        canonical_model=plan['cases']['15']['source_model'],canonical_sha256=plan['cases']['15']['source_sha256'])
    p.guard(probe)
    run.mkdir()
    import review_numeric_recovery
    review_numeric_recovery.install()
    import torch
    torch.set_num_threads(1)
    from review_experiments.e1.snapshots import load_snapshot,portable_environment
    from review_experiments.e1.study import StudyOptions
    import session_adapter
    from pilot import compare
    payload,compiled,environment=load_snapshot(plan['snapshot'],scientific_hashes=plan['snapshot_source_hashes'])
    options=StudyOptions(**dict(payload['options'],horizons=tuple(payload['options']['horizons'])))
    session=SimpleNamespace(snapshot=environment,compiled=compiled,record=payload['record'],options=options,
        source_hashes=plan['snapshot_source_hashes'])
    site=compiled.sites[payload['site']]
    snapshot,input_receipt=session_adapter.export_transport(session,site,plan['sample_step'],payload['rng'],
                                                           run/'durable_input',portable_environment)
    pool=p.BranchPool(run/'workers',template)
    try:
        pool.start()
        result=pool.pair(snapshot,20,parallel=True)
        checks={source:compare(row['trace'],plan['baseline_trace'],source,20) for source,row in result['branches'].items()}
        work=remaining_work()
        elapsed=(datetime.now(timezone.utc)-datetime(2026,10,24,19,51,54,tzinfo=timezone.utc)).total_seconds()/3600
        prediction=benefit(work,acceptance['comparison_speedup'],input_receipt['elapsed_seconds'],elapsed_hours=elapsed)
        native_bytes=sum(row['bytes'] for row in input_receipt['native_inputs'])
        per_pair=3*native_bytes+math.ceil(1.7*input_receipt['snapshot_bytes'])+1024**2
        required=(work['remaining_pairs']+4)*per_pair+48*2**30
        free=p.shutil.disk_usage(ROOT.anchor).free
        enough_disk=free>required
        receipt=dict(passed=True,at_utc=p.now(),original_trace_fidelity_verified=True,
            payload_and_native_inputs_retained=True,input_receipt=input_receipt,result=result,trace_checks=checks,
            remaining_pairs=work['remaining_pairs'],remaining_long_pairs=work['remaining_long_pairs'],
            current_checkpoint_steps=work['index']['completed_steps'],post_checkpoint_pairs=len(work['replay_pairs']),
            measured_comparison_speedup=acceptance['comparison_speedup'],prediction=prediction,
            projected_side_data_bytes_per_pair=per_pair,worst_case_side_data_plus_reserve_bytes=required,
            free_disk_bytes=free,enough_disk_to_preserve_all_side_data=enough_disk,
            ready_for_conditional_migration=prediction['enough_time_saved'] and enough_disk,
            active_combined_interrupted=False,scientific_source_or_result_edits=0)
        p.immutable(HERE/'transport_acceptance.json',receipt)
        print('Durable transport matches original native traces; export seconds='+str(round(input_receipt['elapsed_seconds'],2)),flush=True)
        print('Refreshed net hours saved='+str(round(prediction['conservative_net_saved_hours'],2))+'; migration gate='+str(receipt['ready_for_conditional_migration']),flush=True)
    except BaseException:
        p.immutable(run/'error.json',dict(at_utc=p.now(),traceback=traceback.format_exc()))
        raise
    finally:
        pool.close()
        p.immutable(run/'cleanup.json',dict(at_utc=p.now(),all_owned_worker_processes_retired=True,
            protected_combined=p.h.process(plan['protected_main']['pid'])))


if __name__=='__main__':run()
