"""Copy combined recovery/results without holding original replaceable handles.

This preparer cannot signal processes. It preserves every copied generation,
verifies readback, and refuses a checkpoint that changed during preparation.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import parallel_branches as p
from aspen_retention import Verifier,Deferred,private_snapshot,version,still_current,digest_stream

ARCHIVE=Path(r'C:\Users\hamediaa\RL_MPC_local_archive\combined_parallel_20261024')
OUTPUT=p.h.OUTPUTS['combined']


def copied_generation(source,directory,*,attempts=4):
    """An alias cannot block replacement of the original name. Keep attempts."""
    source=Path(source);directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for _ in range(attempts):
        before=version(source)
        target=directory/(source.name+'.'+uuid.uuid4().hex[:12]+'.copy')
        checksum=hashlib.sha256()
        with private_snapshot(source,HERE/'backup_read_aliases') as stream,target.open('xb') as saved:
            while block:=stream.read(4*1024**2):saved.write(block);checksum.update(block)
        value=checksum.hexdigest()
        if p.h.digest(target)!=value:raise RuntimeError('Independent backup readback differs')
        # Win32 can preserve creation times during rapid replacements, and
        # short same-size writes can share a timestamp. Confirm content via a
        # second private alias as well as attributes; never trust size/time
        # alone for a backup that permits a checkpoint handoff.
        with private_snapshot(source,HERE/'backup_read_aliases') as stream:
            current_checksum=digest_stream(stream)
        if version(source)==before and current_checksum==value:
            return dict(source=str(source),local_copy=str(target),bytes=p.h.attributes(target)['size'],
                        sha256=value,source_version=before,source_content_reverified=True)
        p.immutable(target.with_suffix('.changed.json'),dict(source=str(source),copy_sha256=value,
                    before=before,after=version(source),not_a_committed_backup=True))
        time.sleep(2.2)
    raise Deferred('A continuously changing input needs a later stable copy: '+str(source))


def capture_files(pair):
    files=set()
    # All existing scientific/technical files stay at their original paths.
    # Save all result/capture generations and both complete matched slots.
    for name in ('capture','native_snapshots'):
        files.update(x for x in (OUTPUT/name).rglob('*') if x.is_file())
    files.update(x for x in OUTPUT.iterdir() if x.is_file())
    files.add(OUTPUT/'training_checkpoints/latest.json')
    for point in pair['checkpoints']:
        files.add(Path(point['python_file']))
        files.update(x for x in Path(point['native_directory']).rglob('*') if x.is_file())
        for value in point['private_roots']:
            path=Path(value).resolve()
            if not path.is_relative_to(OUTPUT.resolve()):raise RuntimeError('Pinned dependency leaves combined output')
            if not path.exists():raise RuntimeError('Pinned recovery dependency is absent')
            files.update(x for x in path.rglob('*') if x.is_file()) if path.is_dir() else files.add(path)
    old=ROOT/'Distillation/Results/state_recovery/20261024_combined_native_retry'
    files.update(x for x in old.rglob('*') if x.is_file())
    files.update(x for x in (ROOT/'change-reports/combined_native_recovery_20261024').iterdir() if x.is_file())
    return sorted(files)


def prepare():
    if (HERE/'preservation.json').exists():raise FileExistsError('Inspect existing preservation before repeating')
    acceptance=p.closed_json(HERE/'transport_acceptance.json')
    if not acceptance['ready_for_conditional_migration']:raise RuntimeError('Native equivalence/time/disk gates did not pass')
    for name in ('attempt4/pilot_cleanup.json','transport_test/cleanup.json'):
        if not p.closed_json(HERE/name)['all_owned_worker_processes_retired']:
            raise RuntimeError('Test-owned workers must retire before preparation')
    identity=p.closed_json(HERE/'attempt4/pilot_plan.json')['protected_main']
    if not p.same_generation(identity,p.h.process(identity['pid'])):raise RuntimeError('Original combined generation changed')
    metadata=p.h.stable_batch(dict(anchors=p.h.RETENTION/'checkpoint_hashes.json',
        status=OUTPUT/'status.json',owners=OUTPUT/'owned_aspen_apps.json',ownership=OUTPUT/'ownership_status.json',
        retention=p.h.RETENTION/'status.json'))
    if metadata['status']['state']!='running' or metadata['ownership']['phase'] in ('creating','unverified_creation'):
        raise RuntimeError('Original worker/ownership is not stable')
    cache=HERE/('preservation_verify_'+uuid.uuid4().hex[:8]);cache.mkdir()
    p.h.atomic_json(cache/'checkpoint_hashes.json',metadata['anchors'])
    pair=Verifier(cache).pair(OUTPUT/'training_checkpoints',ROOT/'Distillation/Results')
    if len(pair['index']['agents'])!=4:raise RuntimeError('Four complete learners are required')
    for point in pair['checkpoints']:
        key=str(OUTPUT/'training_checkpoints')+f'::{point["slot"]}::{point["completed_steps"]}'
        if metadata['anchors'].get(key)!=point['sha256']:raise Deferred('Wait for independent retention verification of both slots')
    backup=ARCHIVE/('before_'+p.now().replace(':','').replace('.','_'))
    backup.mkdir(parents=True,exist_ok=False)
    p.immutable(backup/'checkpoint_pair.json',pair)
    p.immutable(backup/'metadata.json',metadata)
    copied=[];deferred=[]
    previous={}
    for earlier in list(ARCHIVE.glob('before_*/incomplete.json'))+list(ARCHIVE.glob('before_*/preservation.json')):
        for row in p.closed_json(earlier).get('verified_copies',[]):previous[row['source']]=row
        for row in p.closed_json(earlier).get('files',[]):previous[row['source']]=row
    pinned=[Path(value).resolve() for point in pair['checkpoints'] for value in point['private_roots']]
    try:
        for source in capture_files(pair):
            relative=source.relative_to(ROOT)
            old=previous.get(str(source))
            if old and tuple(old['source_version'])==tuple(version(source)) and p.h.digest(old['local_copy'])==old['sha256']:
                copied.append(old);continue
            try:copied.append(copied_generation(source,backup/'files'/relative.parent))
            except PermissionError as error:
                supervisor_lock=ROOT/'Distillation/Results/state_recovery/20261024_combined_native_retry/retention.lock'
                if source!=supervisor_lock and not any(source.resolve().is_relative_to(path) for path in pinned):raise
                # The opened working database is not the matched checkpoint.
                # Its separate sealed model/snapshot pair is copied below.
                # Retire only combined after those recovery/results copies
                # verify, then copy the unchanged original working tree before
                # any relaunch. Never delete or substitute the locked file.
                deferred.append(dict(source=str(source),version=version(source),reason='owned_live_native_lock',error=repr(error)))
        if not still_current(pair):raise Deferred('Latest checkpoint changed during backup; retain this backup and refresh')
        current=p.h.stable_batch({'index':OUTPUT/'training_checkpoints/latest.json',
                                  'owners':OUTPUT/'owned_aspen_apps.json','ownership':OUTPUT/'ownership_status.json'})
        if current['index']['sha256']!=pair['index']['sha256']:raise Deferred('Original checkpoint advanced during preservation')
        if current['ownership']['phase'] in ('creating','unverified_creation'):raise Deferred('Owned native replacement is in progress')
        records=p.h.completed_records(OUTPUT/'capture/decisions.jsonl')
        bysource={row['source']:row for row in copied}
        traces=[]
        for row in records:
            source=(OUTPUT/'capture'/row['trace']).resolve()
            saved=bysource[str(source)]
            traces.append(dict(source=str(source),backup=saved['local_copy'],sha256=saved['sha256'],
                               step=row['step'],committed=row['step']<pair['index']['completed_steps']))
        for row in copied:
            if p.h.digest(row['local_copy'])!=row['sha256']:raise RuntimeError('Final independent backup verification differs')
        if not still_current(pair):raise Deferred('Checkpoint changed during final backup readback; refresh')
        receipt=dict(verified=True,at_utc=p.now(),files=copied,pair=pair,original_identity=identity,
            owners=current['owners'],retention_identity={k:p.h.process(metadata['retention']['pid'])[k] for k in ('pid','created','alive')},
            completed_pairs=len(records),post_checkpoint_pairs=sum(not r['committed'] for r in traces),traces=traces,
            original_files_retained=True,all_results_and_complete_checkpoint_dependencies_copied=True,
            deferred_owned_working_files=deferred,working_files_must_copy_before_relaunch=bool(deferred),
            process_signals=0,simulator_calls=0)
        p.immutable(backup/'preservation.json',receipt)
        p.immutable(HERE/'preservation.json',dict(verified=True,manifest=str(backup/'preservation.json'),
            manifest_sha256=p.h.digest(backup/'preservation.json'),backup=str(backup),files=len(copied),
            bytes=sum(r['bytes'] for r in copied),checkpoint_steps=[r['completed_steps'] for r in pair['checkpoints']],
            completed_pairs=len(records),post_checkpoint_pairs=receipt['post_checkpoint_pairs'],
            deferred_live_working_files=len(deferred)))
        print('Independent preservation passed: '+str(len(copied))+' files, checkpoints '+str([r['completed_steps'] for r in pair['checkpoints']]),flush=True)
    except BaseException as error:
        p.immutable(backup/'incomplete.json',dict(at_utc=p.now(),error=repr(error),verified_copies=copied,
                                                original_combined_not_interrupted=True))
        raise


if __name__=='__main__':prepare()
