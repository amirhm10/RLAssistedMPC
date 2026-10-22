"""Recover only failed studies; run five unchanged E1 configurations in parallel."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'Distillation/Results/state_recovery/20261021_remaining_active'
PYTHON=ROOT/'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'
BOOTSTRAPS=('run_remaining_recovery.py','review_numeric_recovery.py','review_training_recovery.py','column_e1_recovery.py')


def install_atomic_writes():
    """Use the already validated Windows replacement retry in new processes."""
    from recover_markov_seed23 import write
    import column_review.common as common
    import revision_recovery.checkpoints as checkpoints
    import review_experiments.e1.study as study
    import column_e1.checkpoints as native
    def converted(path,value):
        return write(path,study.json_safe(value))
    common.write=write
    checkpoints.write_json=study.write_json=native.write_json=converted
    install_native_aliases()


def install_native_aliases():
    """Native loading must retain the restored controller's array identities."""
    import column_e1.checkpoints as native
    if getattr(native.import_native,'_preserves_python_aliases',False):return
    original=native.import_native
    def restore(plant,metadata):
        current_input,current_output=plant.current_input,plant.current_output
        result=original(plant,metadata)
        import numpy as np
        if not np.array_equal(plant.current_input,current_input) or not np.array_equal(plant.current_output,current_output):
            raise RuntimeError('Native load changed the restored Python controller samples')
        plant.current_input,plant.current_output=current_input,current_output
        return result
    restore._preserves_python_aliases=True
    native.import_native=restore


def dependencies():
    from column_review.common import digest
    return {name:digest(ROOT/name) for name in BOOTSTRAPS}


def queue_guard():
    from column_review.common import read
    import continue_column_queue as q
    folder=Path(read(ROOT/'Distillation/Results/review_recovery/latest.json')['repeats'])
    q.validate_group(read(folder/'combined/manifest.json'))
    return folder,read(folder/'combined/manifest.json')


def prepare(directory):
    from column_review.common import read,write,digest,process_identity,owned_apps_remaining,now
    from revision_comparisons.progress import snapshot
    from run_distillation_phase2_repeats import aspen_file_lock
    import run_column_e1 as e1
    import recover_polymer_threshold as threshold
    queue_root,combined=queue_guard()
    old=ROOT/'Distillation/Results/column_e1/20261020_seed7_native_v2'
    old_manifest=read(old/'manifest.json')
    assert old_manifest['state']=='failed'
    source=read(old/'combined/a1/job.json')['source_case']
    model_root=Path(source['aspen_path']).parent
    cases={n:model_root/f'C2S_SS_simulation{n}.dynf' for n in (3,11,13,14,15,16)}
    assert not set(cases)&{j['aspen_number'] for j in combined['jobs']}
    if any(digest(path)!=source['aspen_sha256'] for path in cases.values()):
        raise RuntimeError('A spare case differs from the original source model')
    observed=snapshot()
    protected=[dict(job=r['job'],study=r['study'],identity=process_identity(r['pid']),directory=r['directory'])
               for r in observed['jobs'] if r['state']=='running' and r['study'] not in observed['historical_studies']]
    with ExitStack() as stack:
        for path in cases.values():
            if path.name=='C2S_SS_simulation16.dynf':
                continue  # owned bounded validation uses this case until the E1 gate passes
            stack.enter_context(aspen_file_lock(path,ROOT/'Distillation/Results/_aspen_file_locks'))
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'private_sources').mkdir()
        cases[17]=directory/'private_sources/C2S_SS_simulation17.dynf'
        shutil.copyfile(source['aspen_path'],cases[17])
        jobs=[]
        repeat_manifest=read(queue_root/'combined/manifest.json')
        repeat_root=directory/'repetitions';repeat_root.mkdir()
        for category,mode,case,old_attempt in (
            ('markov','sg',3,ROOT/'Distillation/Results/review_recovery/20261020_openblas/retry/markov_sg/s23/a4'),
            ('residual','plain',11,queue_root/'residual/residual_plain/s23/a1')):
            assert not owned_apps_remaining(old_attempt/'owned_aspen_apps.json')
            import continue_column_queue as previous_queue
            old_status=read(old_attempt/'worker_status.json');assert previous_queue.absent(old_status['pid'])
            output=repeat_root/f'{category}_{mode}/s23/a1';output.mkdir(parents=True)
            job=deepcopy(read(old_attempt/'job.json'))
            job.update(aspen_number=case,aspen_path=str(cases[case]),snaps_path=str(cases[case].parent/f'AM_C2S_SS_simulation{case}'),
                       batch_directory=str(repeat_root),results_directory=str(output))
            write(output/'job.json',job)
            jobs.append(dict(id=f'{category}/{mode}/23',kind='repeat',category=category,mode=mode,seed=23,
                             aspen_number=case,directory=str(output),job=str(output/'job.json'),state='pending',attempts=[]))
        write(repeat_root/'manifest.json',dict(source_hashes=repeat_manifest['source_hashes'],profile=repeat_manifest['profile'],jobs=deepcopy(jobs)))
        for category,case in zip(e1.CATEGORIES,(16,13,14,15,17)):
            if category=='combined':
                output=old/'combined/a1'
                backup=directory/'initial_failures/e1_combined';backup.mkdir(parents=True)
                shutil.copyfile(output/'status.json',backup/'status.json')
                index=read(output/'training_checkpoints/latest.json')
                shutil.copyfile(output/'training_checkpoints/latest.json',backup/'checkpoint_index.json')
                shutil.copyfile(output/'training_checkpoints'/index['file'],backup/index['file'])
                job_path=output/'job.json'
            else:
                output=directory/'e1'/category/'a1';output.mkdir(parents=True)
                job=deepcopy(read(old/category/'a1/job.json'))
                selected=dict(source,aspen_path=str(cases[case]),snaps_path=str(cases[case].parent/f'AM_C2S_SS_simulation{case}'))
                job['source_case']=selected
                for key,folder in (('aspen','private_model'),('branch_aspen','branch_model')):
                    model=output/folder/cases[case].name;model.parent.mkdir()
                    shutil.copyfile(cases[case],model)
                    snaps=model.parent/f'AM_C2S_SS_simulation{case}';snaps.mkdir()
                    job[key]=dict(selected,aspen_path=str(model),snaps_path=str(snaps))
                job['sources']=e1.sources()
                job_path=output/'job.json';write(job_path,job)
            jobs.append(dict(id='e1/'+category,kind='e1',category=category,seed=7,aspen_number=case,
                             directory=str(output),job=str(job_path),state='pending',attempts=[]))
        threshold_root=directory/'threshold'
        threshold.prepare(threshold_root,False)
        threshold_output=threshold_root/'a1';threshold_output.mkdir()
        shutil.copyfile(threshold_root/'job.json',threshold_output/'job.json')
        jobs.append(dict(id='threshold/seed7',kind='threshold',category='threshold',seed=7,aspen_number=None,
                         directory=str(threshold_output),job=str(threshold_output/'job.json'),state='pending',attempts=[]))
        plan=dict(state='prepared',prepared_at=now(),jobs=jobs,bootstrap_sha256=dependencies(),
                  protected_workers=protected,protected_coordinator=read(queue_root/'coordinator_identity.json'),
                  reserved_combined_cases=[j['aspen_number'] for j in combined['jobs']],
                  e1_budget='Original one-seed 300-cycle protocol; 3200 pairs; five independent configurations in parallel',
                  case17='Private byte-identical source copy, separate lease; no additional scientific configuration')
        write(directory/'manifest.json',plan)
        (directory/'logs').mkdir()
        command=subprocess.list2cmdline([str(PYTHON),'-B','-u',str(Path(__file__)),'--supervise',str(directory)])
        (directory/'run.cmd').write_text('@echo off\nsetlocal\ntitle Remaining RL-MPC recovery and parallel E1\ncd /d "'+str(ROOT)+'"\n'+command+'\necho Recovery supervisor exited.\npause\n',encoding='utf-8')
        viewer=subprocess.list2cmdline([str(PYTHON),'-B','-u',str(ROOT/'show_review_progress.py'),'--watch','--interval','30'])
        (directory/'progress.cmd').write_text('@echo off\ntitle CURRENT - all reviewer runs\ncd /d "'+str(ROOT)+'"\n'+viewer+'\n',encoding='utf-8')
    return plan


def worker(path):
    install_atomic_writes()
    from column_review.common import read,write,digest
    directory=path.parent
    plan=read(directory/'manifest.json')
    if dependencies()!=plan['bootstrap_sha256']:
        raise RuntimeError('Recovery implementation changed after preparation')
    job=read(path)
    queue_guard()
    import review_blas_runtime
    info=review_blas_runtime.runtime_info()
    write(Path(job['directory'])/'numerical_recovery.json',dict(info,bootstrap_sha256=plan['bootstrap_sha256']))
    if job['kind']=='repeat':
        from review_training_recovery import repeat_checkpoints
        import run_distillation_phase2_repeats as repeat
        with patch.object(repeat,'MAX_ASPEN_FILES',17),repeat_checkpoints(job['job']):
            return repeat.run_worker(Path(job['job']))
    if job['kind']=='e1':
        import column_e1_recovery
        column_e1_recovery.install()
        import column_review.common as common
        import run_column_e1 as e1
        original=common.write
        def actual_case(target,value):
            if Path(target).name=='status.json' and Path(target).parent==Path(job['directory']):
                value.update(case=job['aspen_number'],recovery_bootstrap_sha256=plan['bootstrap_sha256'])
            return original(target,value)
        with patch.object(common,'write',actual_case):
            return e1.worker(Path(job['job']))
    import recover_polymer_threshold as threshold
    from review_training_recovery import full_policy_manager
    with patch.object(threshold,'PolicyExportManager',full_policy_manager()):
        return threshold.worker(Path(job['job']))


def publish(directory,manifest):
    from column_review.common import write,read
    write(directory/'manifest.json',manifest)
    e1_jobs=[dict(j,id=j['category']) for j in manifest['jobs'] if j['kind']=='e1']
    write(directory/'e1/manifest.json',dict(state=manifest['state'],jobs=e1_jobs,expected_matched_pairs=3200,
                                         concurrency=5,bootstrap_sha256=manifest['bootstrap_sha256']))
    threshold=next(j for j in manifest['jobs'] if j['kind']=='threshold')
    scalar=read(directory/'threshold/manifest.json')
    scalar.update(state=threshold['state'],pid=threshold.get('pid'),directory=threshold['directory'])
    write(directory/'threshold/manifest.json',scalar)


def clear_owned_orphans(job):
    """Only a stopped child with an exclusive canonical lease reaches this."""
    from column_review.common import read,write,owned_apps_remaining,process_identity
    from recover_markov_seed23 import terminate_owned_identity
    from run_distillation_phase2_repeats import aspen_file_lock
    import importlib.util
    import win32com.client
    source=ROOT/'change-reports/run_audit_20261021/clear_verified_orphans.py'
    spec=importlib.util.spec_from_file_location('verified_orphan_reader',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output=Path(job['directory'])
    owned=owned_apps_remaining(output/'owned_aspen_apps.json')
    if not owned:return []
    data=read(job['job']);canonical=data.get('aspen_path',data.get('source_case',{}).get('aspen_path'))
    if not canonical:raise RuntimeError('Unexpected Aspen ownership in a non-Aspen job')
    service=win32com.client.GetObject('winmgmts:')
    records=[]
    with aspen_file_lock(Path(canonical),ROOT/'Distillation/Results/_aspen_file_locks'):
        current=process_identity(job['pid'])
        if current['alive'] is True and current['created']==job['identity']['created']:
            raise RuntimeError('Refusing cleanup for a live worker')
        for app in owned:
            if 'recorded' not in app or app['current']['alive'] is not True or app['current']['created']!=app['recorded']['created']:
                raise RuntimeError('Unverified orphan application identity')
            records.extend(module.descendants(service,app['recorded']))
            records.append(app['recorded'])
        target=output/f'orphan_retry_{len(job["attempts"])}.json'
        result=dict(records=records,actions=[]);write(target,result)
        for record in records:
            result['actions'].append(dict(pid=record['pid'],result=terminate_owned_identity(record)))
            write(target,result)
        if owned_apps_remaining(output/'owned_aspen_apps.json'):raise RuntimeError('Orphan cleanup did not verify')
    return records


def supervise(directory):
    from column_review.common import read,now,process_identity
    from recover_markov_seed23 import write
    from run_distillation_phase2_repeats import aspen_file_lock
    with aspen_file_lock(directory/'supervisor',directory/'_locks'):
        manifest=read(directory/'manifest.json')
        assert manifest['state']=='prepared'
        assert dependencies()==manifest['bootstrap_sha256']
        manifest.update(state='running',supervisor=process_identity(os.getpid()),started_at=now())
        running=[]
        def start(index,job):
            target=directory/f'worker_{index}.json';write(target,job)
            stream=(directory/'logs'/f'worker_{index}.log').open('xb')
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(Path(__file__)),'--worker',str(target)],cwd=ROOT,
                                   stdout=stream,stderr=subprocess.STDOUT)
            job.update(state='running',pid=child.pid,identity=process_identity(child.pid),started_at=now())
            job['attempts'].append(dict(directory=job['directory'],pid=child.pid,started_at=now()))
            running.append((job,child,stream,index))
            publish(directory,manifest)
            print('Started',job['id'],'PID',child.pid,'case',job['aspen_number'],flush=True)
        pointer=read(ROOT/'Distillation/Results/state_recovery/latest.json')
        pointer.update(threshold=str(directory/'threshold'),active_recovery=str(directory))
        write(ROOT/'Distillation/Results/state_recovery/latest.json',pointer)
        write(ROOT/'Distillation/Results/column_e1/latest.json',dict(directory=str(directory/'e1'),parallel_configurations=5,recovery=str(directory)))
        live_pointer=ROOT/'Distillation/Results/live_recovery';live_pointer.mkdir(exist_ok=True)
        write(live_pointer/'latest.json',dict(directory=str(directory)))
        while running or any(j['state']=='pending' for j in manifest['jobs']):
            validation=read(directory/'e1_validated.json',{})
            for index,job in enumerate(manifest['jobs']):
                ready=(job['kind']!='e1' or (validation.get('passed') is True and validation.get('bootstrap_sha256')==manifest['bootstrap_sha256']))
                if job['id']=='residual/plain/23':
                    ready=ready and (Path(job['directory'])/'technical_resume_passed.json').exists()
                if job['state']=='pending' and ready:
                    if dependencies()!=manifest['bootstrap_sha256']:
                        raise RuntimeError('Recovery dependencies changed before a queued launch')
                    start(index,job)
            for item in list(running):
                job,child,stream,index=item
                observed=read(Path(job['directory'])/('worker_status.json' if job['kind']=='repeat' else 'status.json'),{})
                if observed.get('pid') and observed.get('started_at','')>=job.get('started_at','') and observed['pid']!=job['pid']:
                    job['launcher_pid']=child.pid
                    job['pid']=observed['pid'];job['identity']=process_identity(observed['pid'])
                    job['attempts'][-1]['pid']=observed['pid']
                if child.poll() is None:continue
                stream.close()
                status=read(Path(job['directory'])/('worker_status.json' if job['kind']=='repeat' else 'status.json'),{})
                job.update(state='completed' if child.returncode==0 and status.get('state')=='completed' else 'failed',
                           exit_code=child.returncode,finished_at=now(),error=status.get('error'))
                if status.get('result_file'):job['result_file']=status['result_file']
                job['attempts'][-1].update(state=job['state'],exit_code=child.returncode)
                running.remove(item)
                print('Finished',job['id'],job['state'],flush=True)
                transient=(child.returncode in (3221225477,-1073741819) or status.get('error_type')=='PermissionError')
                if transient and len(job['attempts'])<3 and (Path(job['directory'])/'training_checkpoints/latest.json').exists():
                    try:
                        clear_owned_orphans(job)
                        if job['kind']=='threshold':
                            cycle=read(Path(job['directory'])/'training_checkpoints/latest.json')['completed_cycles']
                            for policy in (Path(job['directory'])/'policy_checkpoints').glob('cycle_*.pt'):
                                if int(policy.stem.split('_')[-1])>cycle:
                                    archive=policy.parent/f'abandoned_after_restart_{len(job["attempts"])}';archive.mkdir(exist_ok=True)
                                    policy.replace(archive/policy.name)
                        stream=(directory/'logs'/f'worker_{index}_retry_{len(job["attempts"])}.log').open('xb')
                        child=subprocess.Popen([str(PYTHON),'-B','-u',str(Path(__file__)),'--worker',str(directory/f'worker_{index}.json')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                        job.update(state='running',pid=child.pid,identity=process_identity(child.pid),error=None,started_at=now())
                        job['attempts'].append(dict(directory=job['directory'],pid=child.pid,started_at=now(),checkpoint_resume=True))
                        running.append((job,child,stream,index))
                        print('Resuming verified full checkpoint:',job['id'],flush=True)
                    except Exception as error:
                        job['recovery_error']=str(error)
            publish(directory,manifest)
            print('Remaining recovery:',{state:sum(j['state']==state for j in manifest['jobs']) for state in ('running','pending','completed','failed')},flush=True)
            if running or any(j['state']=='pending' for j in manifest['jobs']):time.sleep(30)
        manifest.update(state='completed' if all(j['state']=='completed' for j in manifest['jobs']) else 'failed',finished_at=now())
        publish(directory,manifest)


def launch(directory):
    import win32api,win32con,win32security,win32com.client
    from column_review.common import write
    service=win32com.client.Dispatch('Schedule.Service');service.Connect();definition=service.NewTask(0)
    token=win32security.OpenProcessToken(win32api.GetCurrentProcess(),win32con.TOKEN_QUERY)
    try:user=win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token,win32security.TokenUser)[0])
    finally:token.Close()
    definition.RegistrationInfo.Description='Author-requested failed-run recovery and five parallel E1 configurations; demand only.'
    definition.Principal.UserId=user;definition.Principal.LogonType=3
    settings=definition.Settings;settings.Enabled=settings.AllowDemandStart=True
    settings.ExecutionTimeLimit='PT0S';settings.DisallowStartIfOnBatteries=settings.StopIfGoingOnBatteries=False
    # Windows venv redirectors add an intermediary process, so use the proven
    # base interpreter for the broker's direct scheduler-parent check. Every
    # supervisor/worker command still explicitly selects the private runtime.
    action=definition.Actions.Create(0);action.Path=r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\pythonw.exe'
    action.Arguments=subprocess.list2cmdline(['-B','-u',str(Path(__file__)),'--broker',str(directory)])
    action.WorkingDirectory=str(ROOT)
    name='RLAssistedMPC-Remaining-'+uuid.uuid4().hex[:12]
    task=service.GetFolder('\\').RegisterTaskDefinition(name,definition,2,user,None,3)
    write(directory/'scheduled.json',dict(task_name=name,triggers=0,execution_time_limit='PT0S'))
    task.Run(None)


def broker(directory):
    from review_restart_recovery import lifetime,isolated_interactive_context
    from column_review.common import write
    identity=lifetime();assert isolated_interactive_context(identity)
    write(directory/'broker_identity.json',identity)
    process=subprocess.Popen(f'cmd.exe /d /c call "{directory / "run.cmd"}"',cwd=ROOT,creationflags=subprocess.CREATE_NEW_CONSOLE)
    subprocess.Popen(f'cmd.exe /d /c call "{directory / "progress.cmd"}"',cwd=ROOT,creationflags=subprocess.CREATE_NEW_CONSOLE)
    process.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','worker','supervise','launch','broker'):actions.add_argument('--'+action,type=Path)
    args=parser.parse_args()
    for action in ('prepare','worker','supervise','launch','broker'):
        path=getattr(args,action)
        if path:
            result=globals()[action](path.resolve())
            if action=='worker':raise SystemExit(result)
