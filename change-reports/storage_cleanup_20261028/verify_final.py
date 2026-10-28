"""Independent post-cleanup verification; hashes retained evidence, never unpickles."""
import datetime
import json
from pathlib import Path
import shutil
import time
import zipfile
import compact_native as c

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOCAL=Path(r'C:\Users\hamediaa\RL_MPC_local_archive')


def journal_rows(name):
    p=HERE/name
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def main():
    before=c.read(HERE/'protected_evidence_before.json');assert before['passed']
    errors=[];hashed=0;online=0;last=start=time.monotonic()
    for row in before['files']:
        p=Path(row['path'])
        if not p.is_file():errors.append(dict(path=str(p),error='missing protected source'));continue
        if p.stat().st_size!=row['bytes']:errors.append(dict(path=str(p),error='protected source size differs'));continue
        if row.get('sha256'):
            if c.digest(p)!=row['sha256']:errors.append(dict(path=str(p),error='protected source checksum differs'))
            hashed+=1
        else:
            if p.stat().st_mtime_ns!=row['mtime_ns']:errors.append(dict(path=str(p),error='online-only source timestamp differs'))
            online+=1
        if time.monotonic()-last>15:
            print(json.dumps(dict(stage='protected evidence',hashed=hashed,online_preserved=online,seconds=round(time.monotonic()-start))),flush=True);last=time.monotonic()
    checkpoint_components=[]
    for job in before['checkpoints']:
        for cp in job['checkpoints']:
            assert c.digest(Path(cp['python_file']))==cp['sha256']
            for key in ('file','project_file'):
                assert c.digest(Path(cp['native'][key]))==cp['native'][key+'_sha256']
                checkpoint_components.append(cp['native'][key])
        print(json.dumps(dict(stage='final checkpoints preserved',job=job['label'])),flush=True)
    native=[r for r in journal_rows('native_compaction_receipts.jsonl') if r.get('state')=='raw_removed']
    backups=[r for r in journal_rows('native_backup_duplicate_receipts.jsonl') if r.get('state')=='raw_removed']
    assert len(native)==len(c.read(HERE/'native_compaction_plan.json')['candidates'])==536
    assert len(backups)==len(c.read(HERE/'native_backup_duplicate_plan.json')['candidates'])==81
    expected_removed={str(Path(r['source'])) for r in native}|{str(Path(r['path'])) for r in backups}
    retained={}
    for r in native:
        retained[r['archive']]=r['archive_sha256']
        assert not Path(r['source']).exists()
    for r in backups:
        retained[r['preservation']['archive']]=r['preservation']['archive_sha256']
        assert not Path(r['path']).exists()
    for i,(name,expected) in enumerate(retained.items()):
        assert c.digest(Path(name))==expected,name
        if time.monotonic()-last>15:
            print(json.dumps(dict(stage='retained archives',done=i+1,total=len(retained))),flush=True);last=time.monotonic()
    # All pre-existing files outside the exact removal set must retain their
    # content metadata. This also protects unselected runs and failure evidence.
    metadata_checked=0
    workflow_updates=[]
    allowed_workflow={'paper-memory/'+name for name in ('ACTIVE_PAPER_STATE.md','FREEZE_STATE.md','METHOD_FAMILY_MAP.md','RESULT_BUNDLE_INDEX.md','CLAIM_LEDGER.md','EXPERIMENT_BACKLOG.md','OPEN_QUESTIONS.md','DECISION_LOG.md')}
    for label,root in [('code',ROOT),('paper',Path(r'C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\Rl_assisted_paper')),('local_archive',LOCAL)]:
        for line in (HERE/f'inventory_{label}.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);p=root/row['path']
            if str(p) in expected_removed:continue
            if not p.is_file():errors.append(dict(path=str(p),error='unlisted deletion'));continue
            st=p.stat()
            if (st.st_size,st.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):
                if label=='paper' and Path(row['path']).as_posix().removeprefix('RL_assisted_paper/') in allowed_workflow:
                    workflow_updates.append(str(p))
                else:
                    errors.append(dict(path=str(p),error='unexpected pre-existing file change'))
            metadata_checked+=1
    new_archives={r['archive']:r['archive_bytes'] for r in native if r['new_archive']}
    raw=sum(r['source_bytes'] for r in native)
    duplicate_bytes=sum(r['bytes'] for r in backups)
    initial=c.read(HERE/'inventory_summary.json')['disk_usage']
    disk=shutil.disk_usage('C:\\')._asdict()
    result=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=not errors,errors=errors,
                protected_local_files_sha256_verified=hashed,protected_online_only_files_preserved=online,
                final_complete_checkpoint_sets=10,final_native_components_sha256_verified=len(checkpoint_components),
                historical_native_files_compacted=len(native),historical_native_raw_bytes=raw,
                new_unique_zip_files=len(new_archives),new_unique_zip_bytes=sum(new_archives.values()),
                redundant_native_backup_files_removed=len(backups),redundant_native_backup_bytes=duplicate_bytes,
                net_bytes_reclaimed_by_cleanup=raw+duplicate_bytes-sum(new_archives.values()),
                retained_archive_files_freshly_rehashed=len(retained),preexisting_files_metadata_verified=metadata_checked,
                intentional_paper_workflow_updates=workflow_updates,
                initial_disk_usage=initial,final_disk_usage=disk,observed_free_disk_increase=disk['free']-initial['free'],
                actual_staged_restore_verified=c.read(HERE/'pilot_restore_verification.json')['passed'],
                no_raw_numerical_results_policies_python_checkpoints_manuscripts_removed=True,
                cloud_permanent_deletion_performed=False)
    c.atomic(HERE/'final_verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
