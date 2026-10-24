"""Losslessly retain older retired branch working copies without disk growth.

Only isolated case15/case17 working/load_* folders are eligible. Keep the two
newest primary comparisons unpacked, every scientific result and every durable
frozen/native input. Each older folder is ZIP/CRC/SHA/readback verified before
removing its duplicate files. Current or inaccessible process generations refuse.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import parallel_branches as p
from aspen_retention import attributes,version,safe_child,open_read

ARCHIVE=Path(r'C:\Users\hamediaa\RL_MPC_local_archive\combined_parallel_20261024\retired_branch_zips')
STATE=ROOT/'Distillation/Results/state_recovery/20261024_combined_parallel'


def compress_verified(folder,destination,*,prune=False,allowed_root=None):
    folder=Path(folder).resolve();destination=Path(destination)
    if allowed_root is not None:safe_child(folder,Path(allowed_root).resolve())
    if prune and (allowed_root is None or not folder.name.startswith('load_')):
        raise RuntimeError('Pruning requires an exact isolated working/load folder')
    files=[]
    for parent,dirs,names in os.walk(folder,followlinks=False):
        for name in dirs:safe_child(Path(parent)/name,folder)
        for name in names:
            source=safe_child(Path(parent)/name,folder)
            files.append(dict(relative=source.relative_to(folder).as_posix(),version=version(source)))
    files.sort(key=lambda row:row['relative'])
    if not files:raise RuntimeError('A retired branch folder must contain recoverable bytes')
    destination.mkdir(parents=True,exist_ok=True)
    token=hashlib.sha256(str(folder).encode()).hexdigest()[:24]
    archive=destination/(token+'.zip');receipt=destination/(token+'.json')
    if receipt.exists():return p.closed_json(receipt)
    if archive.exists():raise FileExistsError('Inspect the retained incomplete archive before retrying')
    rows=[];started=time.monotonic()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as zipped:
        for source_record in files:
            source=folder/source_record['relative'];before=version(source);checksum=hashlib.sha256()
            with open_read(source) as stream,zipped.open(source_record['relative'],'w',force_zip64=True) as target:
                while block:=stream.read(4*1024**2):target.write(block);checksum.update(block)
            if version(source)!=before:raise RuntimeError('Retired source changed during compression')
            rows.append(dict(relative=source_record['relative'],sha256=checksum.hexdigest(),
                             bytes=attributes(source)['size'],version=before))
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:raise RuntimeError('Lossless archive CRC failed')
        for row in rows:
            checksum=hashlib.sha256()
            with zipped.open(row['relative']) as stream:
                while block:=stream.read(4*1024**2):checksum.update(block)
            if checksum.hexdigest()!=row['sha256']:raise RuntimeError('Independent ZIP member SHA differs')
    result=dict(verified=True,at_utc=p.now(),source=str(folder),archive=str(archive),archive_sha256=p.h.digest(archive),
        files=rows,source_bytes=sum(r['bytes'] for r in rows),archive_bytes=attributes(archive)['size'],
        elapsed_seconds=time.monotonic()-started,all_original_bytes_recoverable_by_standard_zip=True,
        scientific_results_or_durable_inputs_removed=0,duplicate_working_files_removed=False)
    # Durable verification evidence precedes any removal. Preserve the ZIP and
    # manifest permanently, including when a subsequent guard defers cleanup.
    p.immutable(receipt,result)
    if prune:
        for row in rows:
            source=safe_child(folder/row['relative'],folder)
            if version(source)!=tuple(row['version']) or p.h.digest(source)!=row['sha256']:
                raise RuntimeError('Source changed after archive verification; leave all remaining files')
        for row in rows:
            source=safe_child(folder/row['relative'],Path(allowed_root).resolve())
            source.unlink()
        for directory in sorted((x for x in folder.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):
            safe_child(directory,Path(allowed_root).resolve());directory.rmdir()
        safe_child(folder,Path(allowed_root).resolve());folder.rmdir()
        p.immutable(receipt.with_suffix('.pruned.json'),dict(at_utc=p.now(),verified_archive=str(archive),
            archive_sha256=result['archive_sha256'],only_verified_duplicate_working_files_removed=True,
            bytes_freed=result['source_bytes'],all_results_inputs_and_checkpoints_retained=True))
    return result


def eligible_pool(directory):
    directory=Path(directory).resolve();safe_child(directory,STATE)
    if not directory.name.startswith('branches_0_'):raise RuntimeError('Only accepted production branch pools are eligible')
    path=directory/'decisions.jsonl'
    if not path.exists():return []
    decisions=p.h.completed_records(path)
    if len(decisions)<=2:return []
    keep={(r['channel'],r['step']) for r in decisions[-2:]};eligible=set()
    for row in decisions[:-2]:
        if (row['channel'],row['step']) in keep:continue
        for batch in row['batches'].values():
            for result in batch['branches'].values():eligible.add((result['native_identity']['pid'],result['native_identity']['created']))
    candidates=[]
    for case in (15,17):
        child=directory/('case'+str(case));events=p.h.completed_records(child/'native_process_recovery.jsonl')
        for row in events:
            if row.get('event')!='opened_fresh_application':continue
            identity=row['current'];key=(identity['pid'],identity['created'])
            if key not in eligible:continue
            actual=p.h.process(identity['pid'])
            if actual.get('alive') is None or p.same_generation(identity,actual):continue
            folder=Path(row['model']).parent.resolve();working=child/'working'
            if folder.is_dir() and folder.name.startswith('load_'):
                safe_child(folder,working)
                candidates.append((folder,working,identity))
    return candidates


def service():
    import psutil
    from run_aspen_retention import single_instance
    from review_restart_recovery import lifetime,isolated_interactive_context
    identity=lifetime()
    if not isolated_interactive_context(identity):raise RuntimeError('Use the independent current-user archive task')
    own=psutil.Process();own.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    state=STATE/'retired_side_retention'
    with single_instance(state):
        p.immutable(state/'identity.json',dict(lifetime=identity,identity=p.h.process(own.pid),
            source_sha256=p.h.digest(Path(__file__)),administrator_required=False))
        while True:
            acceptance=HERE/'acceptance.json'
            if not acceptance.exists():raise RuntimeError('Actual parallel primary acceptance is required first')
            accepted=p.closed_json(acceptance)
            if not accepted['passed']:raise RuntimeError('Parallel production is not accepted')
            pools=sorted(STATE.glob('branches_0_*'));archived=[]
            for pool in pools:
                for folder,working,owner in eligible_pool(pool):
                    archived.append(compress_verified(folder,ARCHIVE,prune=True,allowed_root=working))
            completed=[]
            for receipt in ARCHIVE.glob('*.pruned.json') if ARCHIVE.exists() else []:
                completed.append(p.closed_json(receipt))
            parent=accepted['production_identity'];alive=p.same_generation(parent,p.h.process(parent['pid']))
            p.h.atomic_json(state/'status.json',dict(state='running' if alive else 'completed',updated_at=p.now(),
                pid=own.pid,identity=p.h.process(own.pid),archived_working_folders=len(completed),
                duplicate_bytes_removed=sum(x['bytes_freed'] for x in completed),archive_root=str(ARCHIVE),
                newest_two_primary_comparisons_unpacked=True,all_results_checkpoints_and_inputs_retained=True,
                free_disk_bytes=p.shutil.disk_usage(ROOT.anchor).free))
            if not alive:return
            time.sleep(45)


def launch():
    if not p.closed_json(HERE/'acceptance.json')['passed']:raise RuntimeError('Actual production acceptance is required first')
    state=STATE/'retired_side_retention'
    if (state/'identity.json').exists() or (state/'scheduled.json').exists():
        raise RuntimeError('Inspect an existing archive service before a duplicate launch')
    import subprocess
    import win32api
    import win32con
    import win32security
    import win32com.client
    token=win32security.OpenProcessToken(win32api.GetCurrentProcess(),win32con.TOKEN_QUERY)
    try:user=win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token,win32security.TokenUser)[0])
    finally:token.Close()
    scheduler=win32com.client.Dispatch('Schedule.Service');scheduler.Connect();definition=scheduler.NewTask(0)
    definition.Principal.UserId=user;definition.Principal.LogonType=3;definition.Principal.RunLevel=0
    settings=definition.Settings;settings.Enabled=settings.AllowDemandStart=True
    settings.ExecutionTimeLimit='PT0S';settings.MultipleInstances=2
    settings.DisallowStartIfOnBatteries=settings.StopIfGoingOnBatteries=False
    action=definition.Actions.Create(0)
    action.Path=r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\pythonw.exe'
    action.Arguments=subprocess.list2cmdline(['-B',str(Path(__file__).resolve()),'--service'])
    action.WorkingDirectory=str(ROOT);name='RLAssistedMPC-CombinedRetiredSideArchive-20261024'
    scheduler.GetFolder('\\').RegisterTaskDefinition(name,definition,2,user,None,3)
    p.immutable(state/'scheduled.json',dict(task_name=name,administrator_required=False,
        execution_time_limit='PT0S',at_utc=p.now(),source_sha256=p.h.digest(Path(__file__))))
    scheduler.GetFolder('\\').GetTask(name).Run(None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--service',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--benchmark',type=Path);args=parser.parse_args()
    if args.launch:launch()
    elif args.service:
        try:service()
        except BaseException:
            import traceback
            p.h.atomic_json(STATE/'retired_side_retention/status.json',dict(state='failed',at_utc=p.now(),
                pid=__import__('os').getpid(),error=traceback.format_exc(),existing_archives_preserved=True))
            raise
    elif args.benchmark:
        print(json.dumps(compress_verified(args.benchmark,ARCHIVE/'benchmark'),indent=2),flush=True)
    else:parser.error('Choose a benchmark or the independent service')
