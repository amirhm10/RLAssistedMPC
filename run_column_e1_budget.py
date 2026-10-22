"""Reduce only column E1, with signed checkpoints and an isolated handoff."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'Distillation/Results/state_recovery/20261021_e1_budget20'
POOL=ROOT/'Distillation/Results/state_recovery/20261021_remaining_active'
COMPANION=ROOT/'Distillation/Results/state_recovery/20261021_markov_checkpoint_a2'
PYTHON=ROOT/'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'


def dependencies():
    from column_review.common import digest
    import run_markov_checkpoint_recovery as previous
    return dict(previous.dependencies(),**{name:digest(ROOT/name) for name in ('column_e1_budget.py','run_column_e1_budget.py')})


def write(path,value):
    from recover_markov_seed23 import write as atomic
    from review_experiments.e1.study import json_safe
    return atomic(path,json_safe(value))


def read(path,default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def prepare(directory):
    from column_review.common import digest,now,process_identity
    from revision_comparisons.progress import snapshot
    import run_remaining_recovery as base
    import column_e1_budget as budget
    base.queue_guard()
    if directory.exists():raise FileExistsError(directory)
    main=read(POOL/'manifest.json');extra=read(COMPANION/'manifest.json')
    originals={j['id']:dict(j,parent_manifest=str(POOL/'manifest.json')) for j in main['jobs']}
    originals.update({j['id']:dict(j,parent_manifest=str(COMPANION/'manifest.json')) for j in extra['jobs']})
    active=snapshot()
    protected=[dict(row=r,identity=process_identity(r['pid'])) for r in active['jobs']
               if r['state']=='running' and r['study'] not in active['historical_studies'] and r['study']!='E1_column']
    assert len(protected)==13 and all(p['identity']['alive'] is True for p in protected)
    directory.mkdir();(directory/'logs').mkdir();(directory/'originals').mkdir()
    protocol=dict(id=budget.PROTOCOL_ID,decisions_per_window=20,expected_pairs=640,
                  original_decisions_per_window=100,reference_block_steps=200,strata=budget.STRATA,
                  primary_rollout_steps=200,long_rollout_steps=800,long_fraction=.1,
                  main_training_cycles=300,training_seed=7,bootstrap_sha256=dependencies(),
                  timing='Early/middle/late clock positions; no claim that late points are physically settled',
                  evidence='Conditional selected-state diagnostics; no unweighted overall failure-frequency claim',
                  preserved_work='Pre-handoff completed captures retained as supplemental evidence')
    write(directory/'protocol.json',protocol);signature=digest(directory/'protocol.json')
    jobs=[]
    for category in ('combined','horizon','weights','markov','residual'):
        previous=originals['e1/'+category];out=Path(previous['directory'])
        status=read(out/'status.json');identity=process_identity(status['pid'])
        assert previous['state']=='running' and status['state']=='running' and identity['alive'] is True
        index=read(out/'training_checkpoints/latest.json');sample=read(out/'capture/sampling.json')
        first=min(min(values) for values in sample['selected_steps'].values())
        assert index['schema']==2 and index['completed_steps']<first
        assert digest(out/'training_checkpoints'/index['file'])==index['sha256']
        job=deepcopy(read(previous['job']));assert job['options']['decisions_per_window']==100
        job['options']['decisions_per_window']=20
        job.update(budget_directory=str(directory),budget_protocol_sha256=signature,authorized_checkpoint=index)
        path=out/'budget20_job.json'
        if path.exists():raise FileExistsError(path)
        write(path,job)
        jobs.append(dict(id='e1/'+category,kind='e1',category=category,seed=7,aspen_number=previous['aspen_number'],
                         directory=str(out),job=str(path),job_sha256=digest(path),state='pending_handoff',attempts=[],
                         original=previous,original_identity=identity,expected_matched_pairs=320 if category=='combined' else 80,
                         protocol_id=budget.PROTOCOL_ID,log=str(directory/'logs'/f'{category}.log')))
    plan=dict(state='prepared',prepared_at=now(),protocol_sha256=signature,bootstrap_sha256=dependencies(),
              jobs=jobs,protected=protected,queue_coordinator=read(base.queue_guard()[0]/'coordinator_identity.json'))
    write(directory/'manifest.json',plan)
    command=subprocess.list2cmdline([str(PYTHON),'-B','-u',str(Path(__file__)),'--supervise',str(directory)])
    (directory/'run.cmd').write_text('@echo off\nsetlocal\ntitle COLUMN E1 REDUCED BUDGET 640 PAIRS\ncd /d "'+str(ROOT)+'"\n'+command+'\necho E1 supervisor exited.\npause\n',encoding='utf-8')
    command=subprocess.list2cmdline([str(PYTHON),'-B','-u',str(ROOT/'show_review_progress.py'),'--watch','--interval','30'])
    (directory/'progress.cmd').write_text('@echo off\ntitle CURRENT reviewer progress - E1 target 640\ncd /d "'+str(ROOT)+'"\n'+command+'\n',encoding='utf-8')
    print('Prepared five checkpoint handoffs; 640 paired tests, unchanged training/rollout horizons.',flush=True)


def verify(directory):
    from column_review.common import digest
    plan=read(directory/'manifest.json')
    assert dependencies()==plan['bootstrap_sha256']
    assert digest(directory/'protocol.json')==plan['protocol_sha256']
    for job in plan['jobs']:assert digest(job['job'])==job['job_sha256']
    return plan


def protected_check(plan):
    from column_review.common import process_identity
    result=[]
    for item in plan['protected']:
        old=item['identity'];current=process_identity(old['pid']);row=item['row'];path=Path(row['directory'])
        status=read(path/'worker_status.json',{}) or read(path/'status.json',{})
        preserved=(current['alive'] is True and current['created']==old['created'])
        completed=status.get('state')=='completed' and bool(status.get('result_file') or status.get('files'))
        if not (preserved or completed):raise RuntimeError('Protected job did not verify: '+row['job'])
        result.append(dict(job=row['job'],preserved=preserved,naturally_completed=completed,identity=current))
    return result


def stop(directory,category):
    from column_review.common import process_identity,owned_apps_remaining,digest,now
    from recover_markov_seed23 import terminate_owned_identity
    from run_distillation_phase2_repeats import aspen_file_lock
    import win32com.client
    import psutil
    plan=verify(directory);assert plan['state']=='prepared' and not (directory/'scheduled.json').exists()
    selected=[j for j in plan['jobs'] if (category=='all' or j['category']==category) and j['state']=='pending_handoff']
    for job in selected:
        protected_check(plan)
        out=Path(job['directory']);status=read(out/'status.json');old=job['original_identity']
        assert status['state']=='running' and status['pid']==old['pid']
        assert process_identity(old['pid'])==old and status.get('error_type')!='PermissionError'
        ownership=owned_apps_remaining(out/'owned_aspen_apps.json')
        assert len(ownership)==2 and all('recorded' in row and row['current']=={k:row['recorded'][k] for k in ('pid','alive','created')} for row in ownership)
        helper_path=ROOT/'change-reports/run_audit_20261021/clear_verified_orphans.py'
        spec=importlib.util.spec_from_file_location('e1_owned_tree',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        service=win32com.client.GetObject('winmgmts:');records=[]
        for row in ownership:
            app=dict(row['recorded'],name=psutil.Process(row['recorded']['pid']).name(),executable=psutil.Process(row['recorded']['pid']).exe())
            assert app['name'].lower()=='aspenmodeler.exe' and 'AspenTech' in app['executable'] and app['created']>old['created']
            records.extend(helper.descendants(service,app));records.append(app)
        index=read(out/'training_checkpoints/latest.json');authorized=read(job['job'])['authorized_checkpoint']
        assert index==authorized and digest(out/'training_checkpoints'/index['file'])==index['sha256']
        backup=directory/'originals'/job['category'];backup.mkdir()
        for name in ('status.json','job.json','owned_aspen_apps.json'):
            shutil.copyfile(out/name,backup/name)
        shutil.copyfile(out/'training_checkpoints/latest.json',backup/'checkpoint_index.json')
        shutil.copyfile(out/'training_checkpoints'/index['file'],backup/index['file'])
        concrete=dict(reason='Explicit author reduction of E1 budget',worker=old,command=psutil.Process(old['pid']).cmdline(),
                      owned_tree=records,checkpoint_sha256=index['sha256'],protected=protected_check(plan))
        write(backup/'stop_plan.json',concrete)
        actions=[dict(pid=old['pid'],result=terminate_owned_identity(old))]
        canonical=read(job['job'])['source_case']['aspen_path']
        with aspen_file_lock(Path(canonical),ROOT/'Distillation/Results/_aspen_file_locks'):
            for record in records:actions.append(dict(pid=record['pid'],result=terminate_owned_identity(record)))
        assert not owned_apps_remaining(out/'owned_aspen_apps.json')
        capture=out/'capture';preserved=backup/'capture';preserved.mkdir()
        for name in ('decisions.jsonl','sampling.json','capture_status.json'):
            if (capture/name).exists():shutil.copyfile(capture/name,preserved/name)
        for name in ('traces','snapshots'):
            if (capture/name).exists():shutil.copytree(capture/name,preserved/name)
        records_kept=0
        if (preserved/'decisions.jsonl').exists():
            for line in (preserved/'decisions.jsonl').read_text().splitlines():
                try:json.loads(line);records_kept+=1
                except ValueError:pass
        status.update(state='superseded_budget_handoff',finished_at=now(),error_type=None)
        write(out/'status.json',status)
        job.update(state='ready',supplemental_pairs=records_kept)
        write(backup/'stop_applied.json',dict(actions=actions,checkpoint_retained=True,protected=protected_check(plan),supplemental_pairs=records_kept))
        write(directory/'manifest.json',plan)
        print('Preserved checkpoint and',records_kept,'completed captures for',job['category'],flush=True)
    deadline=time.monotonic()+45
    while selected:
        retired=[]
        for job in selected:
            previous=read(job['original']['parent_manifest'])
            old=next(j for j in previous['jobs'] if j['id']==job['id'])
            if old['state']!='running':retired.append(job)
        selected=[j for j in selected if j not in retired]
        if not selected:break
        if time.monotonic()>deadline:raise RuntimeError('Old supervisor has not acknowledged the handoff yet')
        time.sleep(1)


def worker(path,validation_steps=0):
    from column_review.common import digest,owned_apps_remaining
    import run_remaining_recovery as base
    import column_e1_budget as budget
    import column_e1_recovery
    import column_e1.checkpoints as native
    import column_e1.runtime as runtime
    import column_review.common as common
    import run_column_e1 as original
    job=read(path);directory=Path(job['budget_directory']);plan=verify(directory)
    entry=next(j for j in plan['jobs'] if Path(j['job'])==path)
    assert entry['state'] in ('ready','running') and not owned_apps_remaining(path.parent/'owned_aspen_apps.json')
    base.queue_guard();base.install_atomic_writes();column_e1_recovery.install()
    if job['category']=='markov':
        import review_markov_checkpoint_recovery
        review_markov_checkpoint_recovery.install()
    manager=budget.manager_class(plan['protocol_sha256'],job,validation_steps=validation_steps)
    original_write=common.write
    def status_write(target,value):
        if Path(target)==path.parent/'status.json':
            value.update(case=entry['aspen_number'],budget_protocol_id=budget.PROTOCOL_ID,
                         expected_matched_pairs=entry['expected_matched_pairs'],budget_directory=str(directory))
        return original_write(target,value)
    import review_blas_runtime
    write(path.parent/'budget_runtime.json',dict(review_blas_runtime.runtime_info(),protocol_sha256=plan['protocol_sha256'],bootstrap_sha256=plan['bootstrap_sha256']))
    with patch.object(native,'NativeCheckpointManager',manager),patch.object(runtime,'ColumnStudySession',budget.BudgetSession),patch.object(common,'write',status_write):
        code=budget.adapted_worker(original.worker,validation_steps=validation_steps)(path)
    if validation_steps:
        index=read(path.parent/'training_checkpoints/latest.json');status=read(path.parent/'status.json')
        assert code==0 and status['state']=='technical_checkpoint' and status['resumed_from_step']+validation_steps==index['completed_steps']
        assert not owned_apps_remaining(path.parent/'owned_aspen_apps.json')
        write(directory/f'validation_{index["completed_steps"]}.json',dict(passed=True,category=entry['category'],index=index,restore=read(path.parent/'training_checkpoints/last_restore.json')))
    return code


def supervise(directory):
    from column_review.common import process_identity,now
    from run_distillation_phase2_repeats import aspen_file_lock
    import run_remaining_recovery as base
    with aspen_file_lock(directory/'supervisor',directory/'_locks'):
        plan=verify(directory);assert plan['state']=='prepared' and all(j['state']=='ready' for j in plan['jobs'])
        assert len(list(directory.glob('validation_*.json')))>=2
        plan.update(state='running',supervisor=process_identity(os.getpid()),started_at=now())
        protected_check(plan)
        pointer_path=ROOT/'Distillation/Results/live_recovery/latest.json';pointer=read(pointer_path)
        pointer['additional_manifests']=pointer.get('additional_manifests',[])+[str(directory/'manifest.json')]
        write(pointer_path,pointer)
        running=[]
        def start(job):
            verify(directory);stream=Path(job['log']).open('xb')
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(Path(__file__)),'--worker',job['job']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            job.update(state='running',pid=child.pid,identity=process_identity(child.pid),started_at=now(),error=None)
            job['attempts'].append(dict(directory=job['directory'],pid=child.pid,started_at=job['started_at']))
            running.append((job,child,stream));write(directory/'manifest.json',plan)
        for job in plan['jobs']:start(job)
        while running:
            for item in list(running):
                job,child,stream=item;status=read(Path(job['directory'])/'status.json',{})
                if status.get('budget_protocol_id')=='column-e1-20-clock-strata-v1' and status.get('started_at','')>=job['started_at'] and status.get('pid')!=job['pid']:
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
            if running:time.sleep(30)
        plan.update(state='completed' if all(j['state']=='completed' for j in plan['jobs']) else 'failed',finished_at=now())
        if plan['state']=='completed':
            from review_experiments.e1.analysis import summarize
            records=[]
            for job in plan['jobs']:
                rows=[json.loads(line) for line in (Path(job['directory'])/'capture/decisions.jsonl').read_text().splitlines() if line.strip()]
                assert len(rows)==job['expected_matched_pairs'];records.extend(rows)
            assert len(records)==640
            summarize(records,directory/'analysis')
            for name in ('early','middle','late'):
                summarize([r for r in records if r['sampling_stratum']==name],directory/'analysis_by_clock_stratum'/name)
            write(directory/'analysis_manifest.json',dict(primary_pairs=640,supplemental_pairs=sum(j['supplemental_pairs'] for j in plan['jobs']),scope='Selected-condition diagnostics; clock strata reported separately'))
        write(directory/'manifest.json',plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','stop','worker','supervise','launch'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--category',default='all');parser.add_argument('--validation-steps',type=int,default=0)
    args=parser.parse_args()
    for action in ('prepare','stop','worker','supervise','launch'):
        path=getattr(args,action)
        if path:
            path=path.resolve()
            if action=='stop':stop(path,args.category)
            elif action=='worker':raise SystemExit(worker(path,args.validation_steps))
            elif action=='launch':
                from run_remaining_recovery import launch
                launch(path)
            else:globals()[action](path)
