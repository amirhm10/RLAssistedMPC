"""Resume only the three E1 handoffs whose pre-gate sampling state was empty."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

import run_column_e1_budget as prior

ROOT=prior.ROOT
DEFAULT=ROOT/'Distillation/Results/state_recovery/20261022_e1_budget_lazy'
read,write=prior.read,prior.write


def dependencies():
    from column_review.common import digest
    return dict(prior.dependencies(), **{name:digest(ROOT/name) for name in
                ('column_e1_budget_lazy_recovery.py','run_column_e1_budget_lazy_recovery.py')})


def prepare(directory):
    from column_review.common import process_identity,digest,now,owned_apps_remaining
    base=prior.verify(prior.DEFAULT);prior.protected_check(base)
    if directory.exists():raise FileExistsError(directory)
    jobs=[]
    for job in base['jobs']:
        if job['category'] not in ('weights','markov','residual'):continue
        out=Path(job['directory']);status=read(out/'status.json')
        assert job['state']=='failed' and status['state']=='failed' and status['error_type']=='KeyError'
        assert not owned_apps_remaining(out/'owned_aspen_apps.json')
        index=read(out/'training_checkpoints/latest.json');authorized=read(job['job'])['authorized_checkpoint']
        assert index==authorized and digest(out/'training_checkpoints'/index['file'])==index['sha256']
        jobs.append(dict(job,state='ready',attempts=[],prior_failure=deepcopy(job)))
    assert len(jobs)==3
    peers=[j for j in base['jobs'] if j['category'] in ('combined','horizon')]
    protected=list(base['protected'])
    for job in peers:
        status=read(Path(job['directory'])/'status.json');identity=process_identity(status['pid'])
        assert status['state']=='running' and identity['alive'] is True
        protected.append(dict(row=dict(job=job['id'],directory=job['directory']),identity=identity))
    directory.mkdir();(directory/'logs').mkdir();(directory/'originals').mkdir()
    protocol=dict(kind='empty-pre-gate-schedule-initialization',base_protocol_sha256=base['protocol_sha256'],
                  bootstrap_sha256=dependencies(),scope='Rebuild original prospective schedule, verify archived points, then apply unchanged 20-point migration')
    write(directory/'protocol.json',protocol);signature=digest(directory/'protocol.json')
    for job in jobs:
        category=job['category'];out=Path(job['directory']);backup=directory/'originals'/category;backup.mkdir()
        for source,name in ((out/'status.json','failed_status.json'),(Path(job['log']),'failed.log')):
            shutil.copyfile(source,backup/name)
        job.update(worker=str(directory/f'worker_{category}.json'),log=str(directory/'logs'/f'{category}.log'))
        descriptor=dict(job=job['job'],category=category,repair_directory=str(directory),repair_protocol_sha256=signature)
        write(Path(job['worker']),descriptor);job['worker_sha256']=digest(job['worker'])
    write(directory/'manifest.json',dict(state='prepared',prepared_at=now(),protocol_sha256=signature,
          bootstrap_sha256=dependencies(),jobs=jobs,protected=protected,peers=peers,queue_coordinator=base['queue_coordinator']))
    command=subprocess.list2cmdline([str(prior.PYTHON),'-B','-u',str(Path(__file__)),'--supervise',str(directory)])
    (directory/'run.cmd').write_text('@echo off\nsetlocal\ntitle COLUMN E1 REDUCED - checkpoint initialization repair\ncd /d "'+str(ROOT)+'"\n'+command+'\npause\n',encoding='utf-8')
    shutil.copyfile(prior.DEFAULT/'progress.cmd',directory/'progress.cmd')
    print('Prepared three isolated E1 resumes; 15 healthy production identities protected.',flush=True)


def verify(directory):
    from column_review.common import digest
    plan=read(directory/'manifest.json');prior.verify(prior.DEFAULT)
    assert dependencies()==plan['bootstrap_sha256'] and digest(directory/'protocol.json')==plan['protocol_sha256']
    for job in plan['jobs']:assert digest(job['worker'])==job['worker_sha256']
    return plan


def worker(path,validation_steps=0):
    import column_e1_budget as budget
    import column_e1_budget_lazy_recovery as fix
    descriptor=read(path);directory=Path(descriptor['repair_directory']);plan=verify(directory)
    entry=next(j for j in plan['jobs'] if j['worker']==str(path))
    assert entry['state'] in ('ready','running') and descriptor['repair_protocol_sha256']==plan['protocol_sha256']
    original_verify=prior.verify;original_write=prior.write
    def local_verify(target):
        observed=original_verify(target)
        selected=next(j for j in observed['jobs'] if j['job']==entry['job'])
        assert selected['state']=='failed' and selected['category']==descriptor['category']
        # Only this process sees its retired operational row as runnable. All
        # immutable original job/protocol/source verifications still execute.
        selected['state']='ready'
        return observed
    def local_write(target,value):
        target=Path(target)
        if target.parent==prior.DEFAULT and target.name.startswith('validation_'):
            target=directory/f'validation_{entry["category"]}_{value["index"]["completed_steps"]}.json'
            value=dict(value,lazy_recovery_protocol_sha256=plan['protocol_sha256'])
        elif target==Path(entry['directory'])/'budget_runtime.json':
            value=dict(value,lazy_recovery_protocol_sha256=plan['protocol_sha256'],lazy_bootstrap_sha256=plan['bootstrap_sha256'])
        return original_write(target,value)
    archived=read(prior.DEFAULT/'originals'/entry['category']/'capture/sampling.json')
    factory=fix.manager_factory(budget.manager_class,budget.migrate_session,archived,plan['protocol_sha256'])
    with patch.object(prior,'verify',local_verify),patch.object(prior,'write',local_write),patch.object(budget,'manager_class',factory):
        return prior.worker(Path(entry['job']),validation_steps)


def supervise(directory):
    from column_review.common import process_identity,now
    from run_distillation_phase2_repeats import aspen_file_lock
    import run_remaining_recovery as base
    with aspen_file_lock(directory/'supervisor',directory/'_locks'):
        plan=verify(directory);assert plan['state']=='prepared'
        assert read(directory/'validation_weights_4008.json')['passed'] is True
        prior.protected_check(plan)
        plan.update(state='running',supervisor=process_identity(os.getpid()),started_at=now())
        pointer=read(ROOT/'Distillation/Results/live_recovery/latest.json')
        pointer['additional_manifests'].append(str(directory/'manifest.json'))
        write(ROOT/'Distillation/Results/live_recovery/latest.json',pointer)
        running=[]
        def start(job):
            verify(directory);stream=Path(job['log']).open('xb')
            child=subprocess.Popen([str(prior.PYTHON),'-B','-u',str(Path(__file__)),'--worker',job['worker']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            job.update(state='running',pid=child.pid,identity=process_identity(child.pid),started_at=now(),error=None)
            job['attempts'].append(dict(directory=job['directory'],pid=child.pid,started_at=job['started_at']))
            running.append((job,child,stream));write(directory/'manifest.json',plan)
        for job in plan['jobs']:start(job)
        while True:
            for item in list(running):
                job,child,stream=item;status=read(Path(job['directory'])/'status.json',{})
                if status.get('started_at','')>=job['started_at'] and status.get('pid')!=job['pid']:
                    job.update(launcher_pid=child.pid,pid=status['pid'],identity=process_identity(status['pid']))
                    job['attempts'][-1]['pid']=status['pid']
                if child.poll() is None:continue
                stream.close();running.remove(item)
                job.update(state='completed' if child.returncode==0 and status.get('state')=='completed' else 'failed',exit_code=child.returncode,error=status.get('error'),finished_at=now())
                job['attempts'][-1].update(state=job['state'],exit_code=child.returncode)
                if child.returncode in (3221225477,-1073741819) and len(job['attempts'])<3:
                    try:
                        base.clear_owned_orphans(job)
                        job['log']=str(directory/'logs'/f'{job["category"]}_retry_{len(job["attempts"])}.log');start(job)
                    except Exception as error:job['recovery_error']=str(error)
            write(directory/'manifest.json',plan)
            if not running:
                if any(j['state']!='completed' for j in plan['jobs']):break
                peers=[read(Path(j['directory'])/'status.json',{}) for j in plan['peers']]
                if all(s.get('state')=='completed' for s in peers):break
                if read(prior.DEFAULT/'manifest.json')['state']=='failed' and any(s.get('state')!='completed' for s in peers):break
            time.sleep(30)
        all_jobs=plan['jobs']+plan['peers']
        plan.update(state='completed' if all(read(Path(j['directory'])/'status.json').get('state')=='completed' for j in all_jobs) else 'failed',finished_at=now())
        if plan['state']=='completed':
            from review_experiments.e1.analysis import summarize
            records=[]
            for job in all_jobs:
                rows=[json.loads(line) for line in (Path(job['directory'])/'capture/decisions.jsonl').read_text().splitlines() if line.strip()]
                assert len(rows)==job['expected_matched_pairs'];records.extend(rows)
            assert len(records)==640
            summarize(records,directory/'analysis')
            for stratum in ('early','middle','late'):
                summarize([r for r in records if r['sampling_stratum']==stratum],directory/'analysis_by_clock_stratum'/stratum)
            write(directory/'analysis_manifest.json',dict(primary_pairs=640,supplemental_pairs=sum(j['supplemental_pairs'] for j in all_jobs),scope='Conditional selected-state diagnostics',base_protocol_sha256=read(prior.DEFAULT/'manifest.json')['protocol_sha256'],lazy_recovery_protocol_sha256=plan['protocol_sha256']))
        write(directory/'manifest.json',plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','worker','supervise','launch'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--validation-steps',type=int,default=0);args=parser.parse_args()
    for action in ('prepare','worker','supervise','launch'):
        target=getattr(args,action)
        if target:
            target=target.resolve()
            if action=='worker':raise SystemExit(worker(target,args.validation_steps))
            elif action=='launch':
                from run_remaining_recovery import launch
                launch(target)
            else:globals()[action](target)
