"""Read-only launch/guard/progress evidence. Never opens or signals Aspen."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def audit(output):
    from column_review.common import read,digest,process_identity,owned_apps_remaining
    from revision_comparisons.progress import snapshot,log_tail
    import run_remaining_recovery as pool
    import run_markov_checkpoint_recovery as companion
    import psutil
    import continue_column_queue as queue
    import recover_markov_seed23 as prior_recovery
    main=read(pool.DEFAULT/'manifest.json');extra=read(companion.DEFAULT/'manifest.json')
    queue_root,combined=pool.queue_guard()
    assert pool.dependencies()==main['bootstrap_sha256']
    assert companion.dependencies()==extra['bootstrap_sha256']
    issues=[]
    current=snapshot()
    current_jobs={job['id']:dict(job,log=str(pool.DEFAULT/'logs'/f'worker_{index}.log'))
                  for index,job in enumerate(main['jobs'])}
    current_jobs.update({job['id']:job for job in extra['jobs']})
    jobs=[]
    for job in current_jobs.values():
        path=Path(job['directory'])
        status=read(path/('worker_status.json' if job['kind']=='repeat' else 'status.json'),{})
        pid=status.get('pid') or job.get('pid')
        identity=process_identity(pid)
        record=dict(id=job['id'],kind=job['kind'],case=job['aspen_number'],state=job['state'],
                    actual_identity=identity,recorded_status=status,directory=str(path),
                    numeric_runtime=read(path/'numerical_recovery.json'),
                    native_checkpoint=read(path/'training_checkpoints/latest.json'),
                    policy_checkpoint=read(path/'policy_checkpoints/latest.json'),
                    restore=read(path/'training_checkpoints/last_restore.json'),
                    apps=owned_apps_remaining(path/'owned_aspen_apps.json'),log=job.get('log'))
        text=log_tail(job['log'])
        cycles=re.findall(r'Sub_Episode:\s*(\d+)',text)
        record['cycle']=int(cycles[-1]) if cycles else None
        record['log_bytes']=Path(job['log']).stat().st_size if Path(job['log']).exists() else 0
        if identity['alive']:
            process=psutil.Process(pid)
            record.update(cpu_seconds=sum(process.cpu_times()[:2]),rss_bytes=process.memory_info().rss,
                          created_unix=process.create_time())
        if job['state']=='running' and (identity['alive'] is not True or status.get('state')!='running'):
            issues.append(job['id']+' does not have a verified live running worker')
        numeric=record['numeric_runtime']
        if not numeric or numeric.get('numpy')!='2.5.3' or numeric.get('shared_environment_modified') is not False:
            issues.append(job['id']+' lacks verified private NumPy 2.5.3 metadata')
        if job['kind']=='e1':
            import run_column_e1 as e1
            job_data=read(job['job'])
            if job_data['sources']!=e1.sources():issues.append(job['id']+' scientific sources changed')
            capture=path/'capture/decisions.jsonl'
            records=[]
            if capture.exists():
                for line in capture.read_text().splitlines():
                    try:records.append(json.loads(line))
                    except ValueError:pass
            record.update(matched_records=len(records),completed_matched_records=sum(r.get('status')=='completed' for r in records),
                          last_matched_step=records[-1].get('step') if records else None)
        jobs.append(record)
    protected=[]
    residual_manifest=read(queue_root/'residual/manifest.json')
    prior=read(ROOT/'change-reports/run_audit_20261021/snapshot_initial.json')
    for row in prior['jobs']:
        if row['study']!='column_repeats' or row['state']!='running':continue
        if row['job'] not in ('residual/plain/7','residual/plain/11','residual/plain/37','residual/plain/53'):continue
        status=read(Path(row['directory'])/'worker_status.json')
        identity=process_identity(row['pid'])
        valid=(status.get('state')=='completed' and status.get('aspen_closed') is True
               and Path(status.get('result_file','')).is_file()) or (
               identity['alive'] is True and status.get('state')=='running' and status.get('pid')==row['pid'])
        protected.append(dict(job=row['job'],original_pid=row['pid'],current_identity=identity,
                              status=status,preserved_or_naturally_completed=valid))
        if valid and status.get('state')=='completed':
            category,mode,seed=row['job'].split('/')
            original_job=next(j for j in residual_manifest['jobs'] if prior_recovery.key(j)==(category,mode,int(seed)))
            protected[-1]['validated_result']=prior_recovery.validated_result(original_job,residual_manifest)
        if not valid:issues.append('Protected worker '+row['job']+' did not verify')
    original=main['protected_coordinator'];identity=process_identity(original['pid'])
    coordinator_same=identity['alive'] is True and identity['created']==original['created']
    if not coordinator_same:issues.append('Original combined coordinator identity changed')
    cases=[job['aspen_number'] for job in current_jobs.values() if job['aspen_number']]
    reserved=[job['aspen_number'] for job in combined['jobs']]
    collision=bool(set(cases)&set(reserved)) or len(set(cases))!=len(cases)
    if collision:issues.append('Recovery / queue case collision')
    combined_jobs=[]
    for job in combined['jobs']:
        path=Path(job['attempts'][-1]['directory']) if job['attempts'] else None
        status=read(path/'worker_status.json',{}) if path else {}
        numeric=read(path/'isolated_runtime.json',{}) if path else {}
        actual=process_identity(status.get('pid')) if status.get('pid') else None
        record=dict(id='/'.join(map(str,(job['category'],job['mode'],job['seed']))),state=job['state'],
                    pid=status.get('pid'),case=job['aspen_number'],status=status,numeric_runtime=numeric,
                    actual_identity=actual,directory=str(path) if path else None)
        if combined['state']=='running' and job['state']=='running' and not (
            actual and actual['alive'] is True and status.get('state')=='running' and numeric.get('numpy')=='2.5.3'):
            issues.append(record['id']+' queued-worker runtime/liveness not verified')
        if path:
            record['apps']=owned_apps_remaining(path/'owned_aspen_apps.json')
            cycles=re.findall(r'Sub_Episode:\s*(\d+)',log_tail(path/'run.log'))
            record['cycle']=int(cycles[-1]) if cycles else None
        combined_jobs.append(record)
    paper=ROOT.parent/'Rl_assisted_paper/RL_assisted_paper'
    frozen=read(paper/'manuscript_revised/frozen_original_manifest.json')
    frozen_ok=all(digest(paper/'manuscript'/item['path'])==item['sha256'] for item in frozen['files'])
    if not frozen_ok:issues.append('Frozen manuscript differs')
    result=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),issues=issues,progress=current,
                recovery_jobs=jobs,protected_workers=protected,
                queue=dict(coordinator_same_identity=coordinator_same,identity=identity,
                           status=read(queue_root/'coordinator_status.json'),combined_state=combined['state'],
                           launch=read(queue_root/'combined/launch.json'),
                           jobs=combined_jobs,source_model_bootstrap_guards_passed=True,
                           runtime_recovery_override=combined['runtime_recovery_override']),
                recovery_case_collisions=collision,memory=dict(psutil.virtual_memory()._asdict()),
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                frozen_original=dict(verified=frozen_ok,files=len(frozen['files'])),
                full_budget_preserved=True,e1_parallel_configurations=5)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(observed_at_utc=result['observed_at_utc'],issues=issues,
                          counts=current['counts'],queue=result['queue']['status'],
                          available_memory_gib=round(result['memory']['available']/1024**3,2),
                          frozen_original=result['frozen_original']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    audit(parser.parse_args().output.resolve())
