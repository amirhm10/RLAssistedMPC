"""Guarded companion for two Markov startup failures; no active pool edits."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'Distillation/Results/state_recovery/20261021_markov_checkpoint_a2'
POOL=ROOT/'Distillation/Results/state_recovery/20261021_remaining_active'
PYTHON=ROOT/'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'


def dependencies():
    import run_remaining_recovery as base
    from column_review.common import digest
    return dict(base.dependencies(), **{name:digest(ROOT/name) for name in
        ('run_markov_checkpoint_recovery.py','review_markov_checkpoint_recovery.py')})


def prepare(directory):
    from column_review.common import read,write,now,process_identity,owned_apps_remaining
    import run_remaining_recovery as base
    if directory.exists():raise FileExistsError(directory)
    source=read(POOL/'manifest.json')
    assert base.dependencies()==source['bootstrap_sha256']
    base.queue_guard()
    jobs=[]
    for index in (0,5):
        previous=source['jobs'][index]
        assert previous['category']=='markov' and previous['state']=='failed'
        assert process_identity(previous['pid'])['alive'] is False
        old_status=read(Path(previous['directory'])/('worker_status.json' if previous['kind']=='repeat' else 'status.json'))
        assert process_identity(old_status['pid'])['alive'] is False
        assert not owned_apps_remaining(Path(previous['directory'])/'owned_aspen_apps.json')
        assert not (Path(previous['directory'])/'training_checkpoints/latest.json').exists()
        jobs.append(dict(deepcopy(previous),state='pending',attempts=[],original_worker=str(POOL/f'worker_{index}.json'),
                         previous_failure=deepcopy(previous)))
    directory.mkdir();(directory/'logs').mkdir()
    for job in jobs:
        output=Path(job['directory'])
        archive=directory/('startup_failure_'+job['kind']);archive.mkdir()
        for name in ('status.json','worker_status.json','owned_aspen_apps.json'):
            if (output/name).exists():shutil.copyfile(output/name,archive/name)
        job['log']=str(output/'checkpoint_recovery_a2.log')
        write(output/'markov_compiler_recovery.json',dict(bootstrap_sha256=dependencies(),previous_failure=job['previous_failure']))
    manifest=dict(state='prepared',prepared_at=now(),jobs=jobs,bootstrap_sha256=dependencies())
    write(directory/'manifest.json',manifest)
    command=subprocess.list2cmdline([str(PYTHON),'-B','-u',str(Path(__file__)),'--supervise',str(directory)])
    (directory/'run.cmd').write_text('@echo off\nsetlocal\ntitle Markov checkpoint recovery\ncd /d "'+str(ROOT)+'"\n'+command+'\necho Supervisor finished.\npause\n',encoding='utf-8')
    command=subprocess.list2cmdline([str(PYTHON),'-B','-u',str(ROOT/'show_review_progress.py'),'--watch','--interval','30'])
    (directory/'progress.cmd').write_text('@echo off\ntitle CURRENT all runs with Markov recovery\ncd /d "'+str(ROOT)+'"\n'+command+'\n',encoding='utf-8')
    pointer=ROOT/'Distillation/Results/live_recovery/latest.json'
    live=read(pointer);live['additional_manifests']=[str(directory/'manifest.json')];write(pointer,live)
    return manifest


def worker(path):
    from column_review.common import read,record_owned_apps
    plan=read(path.parent/'manifest.json')
    if dependencies()!=plan['bootstrap_sha256']:raise RuntimeError('Markov companion dependencies changed')
    import review_markov_checkpoint_recovery as compiler
    compiler.install()
    import run_remaining_recovery as base
    job=read(path)
    if job['kind']=='repeat':
        with record_owned_apps(Path(job['directory'])):
            return base.worker(Path(job['original_worker']))
    return base.worker(Path(job['original_worker']))


def supervise(directory):
    from column_review.common import read,write,now,process_identity,owned_apps_remaining
    from run_distillation_phase2_repeats import aspen_file_lock
    import run_remaining_recovery as base
    with aspen_file_lock(directory/'supervisor',directory/'_locks'):
        plan=read(directory/'manifest.json')
        assert plan['state']=='prepared' and dependencies()==plan['bootstrap_sha256']
        plan.update(state='running',supervisor=process_identity(os.getpid()),started_at=now())
        running=[]
        def start(index,job,retry=False):
            assert dependencies()==plan['bootstrap_sha256']
            base.queue_guard()
            old=job.get('identity') or job['previous_failure']['identity']
            current=process_identity(old['pid'])
            assert not (current['alive'] and current['created']==old['created'])
            assert not owned_apps_remaining(Path(job['directory'])/'owned_aspen_apps.json')
            target=directory/f'worker_{index}.json';write(target,job)
            log=Path(job['log'])
            if retry:log=log.with_name(f'checkpoint_recovery_retry_{len(job["attempts"])}.log')
            job['log']=str(log)
            stream=log.open('xb')
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(Path(__file__)),'--worker',str(target)],
                                   cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            job.update(state='running',pid=child.pid,launcher_pid=child.pid,identity=process_identity(child.pid),started_at=now(),error=None)
            job['attempts'].append(dict(directory=job['directory'],pid=child.pid,started_at=job['started_at'],log=str(log)))
            running.append((index,job,child,stream));write(directory/'manifest.json',plan)
            print('Started',job['id'],'case',job['aspen_number'],flush=True)
        for index,job in enumerate(plan['jobs']):start(index,job)
        while running:
            for item in list(running):
                index,job,child,stream=item
                status=read(Path(job['directory'])/('worker_status.json' if job['kind']=='repeat' else 'status.json'),{})
                if status.get('pid') not in (None,job['previous_failure']['pid']):
                    actual=process_identity(status['pid'])
                    if actual['alive']:
                        job['pid']=status['pid'];job['identity']=actual;job['attempts'][-1]['pid']=status['pid']
                if child.poll() is None:continue
                stream.close();running.remove(item)
                job.update(state='completed' if child.returncode==0 and status.get('state')=='completed' else 'failed',
                           exit_code=child.returncode,finished_at=now(),error=status.get('error'))
                if status.get('result_file'):job['result_file']=status['result_file']
                job['attempts'][-1].update(state=job['state'],exit_code=child.returncode)
                transient=child.returncode in (3221225477,-1073741819) or status.get('error_type')=='PermissionError'
                if transient and len(job['attempts'])<3 and (Path(job['directory'])/'training_checkpoints/latest.json').exists():
                    try:base.clear_owned_orphans(job);start(index,job,True)
                    except Exception as error:job['recovery_error']=str(error)
            write(directory/'manifest.json',plan)
            if running:time.sleep(30)
        plan.update(state='completed' if all(j['state']=='completed' for j in plan['jobs']) else 'failed',finished_at=now())
        write(directory/'manifest.json',plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','worker','supervise','launch'):actions.add_argument('--'+action,type=Path)
    args=parser.parse_args()
    for action in ('prepare','worker','supervise','launch'):
        path=getattr(args,action)
        if path:
            if action=='launch':
                from run_remaining_recovery import launch
                launch(path.resolve())
            else:
                result=globals()[action](path.resolve())
                if action=='worker':raise SystemExit(result)
