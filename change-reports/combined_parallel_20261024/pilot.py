"""Benchmark equivalent serial/parallel counterfactuals while training continues."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE
sys.path.insert(0,str(HERE))
import parallel_branches as p


def compare(trace_path,baseline_path,source,steps):
    import numpy as np
    with np.load(trace_path,allow_pickle=False) as actual,np.load(baseline_path,allow_pickle=False) as stored:
        expected_keys={name.split('__',1)[1] for name in stored.files if name.startswith(source+'__') and '__forecast_' not in name}
        if set(actual.files)!=expected_keys:
            raise RuntimeError('Replay trace fields differ: '+str((set(actual.files),expected_keys)))
        details={}
        tolerances={'y':[1e-8,1e-4],'u':1e-3,'reward':1e-5,'reference':1e-10,'previous_input':1e-3}
        for name in sorted(actual.files):
            left=actual[name];right=stored[source+'__'+name]
            if name!='previous_input':right=right[:steps]
            if left.shape!=right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise RuntimeError('Replay trace shape/finite contract differs: '+name)
            tolerance=tolerances.get(name,1e-6)
            matches=np.array_equal(left,right) if left.dtype.kind in 'biu' else np.allclose(left,right,rtol=1e-7,atol=tolerance)
            details[name]={'max_abs_difference':float(np.max(np.abs(left-right))) if left.size else 0.,'passed':bool(matches)}
        if not all(v['passed'] for v in details.values()):
            p.immutable(Path(trace_path).parent/'fidelity_failed.json',details)
            raise RuntimeError('Original serial trace fidelity failed: '+str([k for k,v in details.items() if not v['passed']]))
        return details


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    destination=RUN/'pilot_plan.json'
    if destination.exists():raise FileExistsError('Pilot plan exists; inspect it instead of duplicating a launch')
    anchors=p.closed_json(ROOT/'change-reports/hourly_run_checks_20261024/reconciliation_inputs.json')
    science=anchors['metadata']['horizon']['source_sha256']
    for filename,checksum in science.items():
        if p.h.digest(ROOT/filename)!=checksum:raise RuntimeError('Scientific seal differs: '+filename)
    free=p.closed_json(ROOT/'change-reports/combined_eta_capacity_20261024/free_cases.json')
    cases={str(row['case']):{k:row[k] for k in ('source_model','source_sha256')} for row in free['cases']}
    if set(cases)!={'15','17'}:raise RuntimeError('Pilot must use the two completed free cases')
    for row in free['cases']:
        if p.h.digest(row['completion_receipt'])!=row['completion_sha256']:raise RuntimeError('A completed peer receipt differs')
    output=p.h.OUTPUTS['combined'];meta=p.h.stable_batch({'status':output/'status.json','owned':output/'owned_aspen_apps.json','index':output/'training_checkpoints/latest.json'})
    original=p.closed_json(ROOT/'change-reports/combined_native_recovery_20261024/plan.json')
    if meta['index']['source_hashes']!=original['initial_sources']:
        raise RuntimeError('Current scientific/protocol checkpoint map differs from accepted combined')
    current=p.h.process(meta['status']['pid']);expected={'pid':15940,'alive':True,'created':134373287822018420}
    if meta['status']['state']!='running' or not p.same_generation(expected,current):raise RuntimeError('Advancing combined identity changed')
    records=p.h.completed_records(output/'capture/decisions.jsonl')
    selected=[r for r in records if r.get('snapshot') and r['channel']=='weights' and r['window']=='plant_change']
    if len(selected)!=1:raise RuntimeError('Select exactly the existing frozen weights plant-change diagnostic')
    row=selected[0];snapshot=output/'capture'/row['snapshot'];trace=output/'capture'/row['trace']
    files=[HERE/'parallel_branches.py',HERE/'pilot.py',ROOT/'change-reports/residual_native_retry2_20261024/document_open_retry.py',ROOT/'change-reports/weights_pid_reuse_recovery_20261024/ownership_fix.py']
    plan=dict(at_utc=p.now(),authorization='User requested tested parallel branches on free 15/17; restart only if equivalence and savings verify',
        category='combined',cases=cases,protected_main=expected,science_hashes=science,
        snapshot_source_hashes=meta['index']['source_hashes'],
        adapter_hashes={str(f):p.h.digest(f) for f in files},snapshot=str(snapshot),snapshot_sha256=p.h.digest(snapshot),
        baseline_trace=str(trace),baseline_trace_sha256=p.h.digest(trace),sample_step=row['step'],
        sample_rollout_steps=row['rollout_steps'],serial_and_parallel_steps=200,
        warmup_steps=20,minimum_comparison_speedup=1.25,minimum_net_saved_hours=3.,
        maximum_per_pair_transport_allowance_seconds=60.,runtime=str(p.PYTHON),
        original_pairs=len(records),no_active_job_interruption=True)
    p.immutable(destination,plan)
    print('Pilot sealed; saved weights snapshot/trace, all143 scientific sources, cases15/17 and protected live combined verified.',flush=True)


def run():
    plan=p.closed_json(RUN/'pilot_plan.json')
    template={k:plan[k] for k in ('category','cases','protected_main','science_hashes','snapshot_source_hashes','adapter_hashes')}
    runtime=RUN/'pilot_runtime'
    if runtime.exists():raise FileExistsError('Pilot runtime exists; inspect current process and receipts')
    # The heavy scientific imports occur only after plan/identity/resource seals.
    probe=dict(template,case=15,parent_identity={k:p.h.process(__import__('os').getpid())[k] for k in ('pid','created','alive')},
               canonical_model=plan['cases']['15']['source_model'],canonical_sha256=plan['cases']['15']['source_sha256'])
    p.guard(probe)
    import review_numeric_recovery
    review_numeric_recovery.install()
    import torch
    torch.set_num_threads(1)
    started=time.monotonic();pool=p.BranchPool(runtime,template)
    p.immutable(RUN/'pilot_identity.json',dict(identity=p.h.process(__import__('os').getpid()),at_utc=p.now()))
    timings={};checks={}
    try:
        pool.start()
        for label,steps,parallel in (('warmup',20,True),('serial',200,False),('parallel',200,True)):
            print('Starting '+label+' diagnostic pair; original combined continues.',flush=True)
            result=pool.pair(plan['snapshot'],steps,parallel=parallel)
            checks[label]={source:compare(row['trace'],plan['baseline_trace'],source,steps) for source,row in result['branches'].items()}
            timings[label]=result
            p.immutable(RUN/('pilot_'+label+'.json'),dict(passed=True,at_utc=p.now(),result=result,trace_checks=checks[label]))
            print(label+' passed native/frozen-state/serial-trace checks in '+str(round(result['elapsed_seconds'],1))+' seconds.',flush=True)
        speedup=timings['serial']['elapsed_seconds']/timings['parallel']['elapsed_seconds']
        assessment=p.closed_json(ROOT/'change-reports/combined_eta_capacity_20261024/assessment.json')
        remaining=assessment['remaining_pairs']
        serial_hours=assessment['serial_estimate_hours']['total']
        comparison_hours=assessment['serial_estimate_hours']['branch_rollouts']+assessment['serial_estimate_hours']['pair_overhead']
        gross=comparison_hours*(1-1/speedup)
        transport_allowance=remaining*plan['maximum_per_pair_transport_allowance_seconds']/3600
        pilot_elapsed=(time.monotonic()-started)/3600
        conservative_net=gross-transport_allowance-pilot_elapsed-1.0
        enough=speedup>=plan['minimum_comparison_speedup'] and conservative_net>=plan['minimum_net_saved_hours']
        p.immutable(RUN/'pilot_acceptance.json',dict(passed=True,at_utc=p.now(),all_trace_fidelity_checks_passed=True,
            exact_original_fidelity_bounds_preserved=True,comparison_speedup=speedup,
            serial_seconds=timings['serial']['elapsed_seconds'],parallel_seconds=timings['parallel']['elapsed_seconds'],
            gross_projected_saved_hours=gross,transport_allowance_hours=transport_allowance,
            pilot_elapsed_hours=pilot_elapsed,additional_restart_validation_allowance_hours=1.,
            conservative_projected_net_saved_hours=conservative_net,enough_time_saved=enough,
            active_combined_interrupted=False,primary_results_changed=False,
            original_serial_eta_hours=serial_hours,limitation='One complete 200-step decision pair; extra long/other-channel tests may be required before migration.'))
        print('Pilot complete. Speedup='+str(round(speedup,3))+'; conservative net hours saved='+str(round(conservative_net,2))+'; migration benefit gate='+str(enough),flush=True)
    except BaseException:
        import traceback
        p.immutable(RUN/'pilot_error.json',dict(at_utc=p.now(),traceback=traceback.format_exc(),
                                               protected_combined_identity=p.h.process(plan['protected_main']['pid'])))
        raise
    finally:
        pool.close()
        p.immutable(RUN/'pilot_cleanup.json',dict(at_utc=p.now(),all_owned_worker_processes_retired=True,
                     protected_combined=p.h.process(plan['protected_main']['pid']),simulator_processes=p.h.native_processes()))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--attempt',default=None)
    args=parser.parse_args()
    if args.attempt:
        if not args.attempt.isalnum():raise ValueError('Use a simple unique attempt name')
        RUN=HERE/args.attempt
    prepare() if args.action=='prepare' else run()
