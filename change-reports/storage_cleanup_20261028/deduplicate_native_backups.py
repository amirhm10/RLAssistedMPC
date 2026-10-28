"""Remove only raw native backup duplicates already covered by verified ZIPs.

Plan performs read-only verification. Execution is confined to the explicitly
named local archive and the exact preverified file list; final-generation copies
and all non-native files are excluded. Nothing is recursively removed.
"""
import argparse
import datetime
import json
from pathlib import Path
import time
import compact_native as c

HERE=Path(__file__).resolve().parent
LOCAL=Path(r'C:\Users\hamediaa\RL_MPC_local_archive')


def allowed(path):
    resolved=path.resolve(strict=True)
    assert resolved.is_relative_to(LOCAL.resolve()) and resolved!=LOCAL.resolve()
    assert path.suffix.lower()=='.dynf' or ('.dynf.' in path.name.lower() and path.suffix.lower()=='.copy')
    assert not any(p.startswith('final_') for p in path.relative_to(LOCAL).parts)
    current=path
    while current!=LOCAL.parent:
        assert getattr(current.lstat(),'st_reparse_tag',0) not in (0xA0000003,0xA000000C)
        assert not current.is_symlink()
        current=current.parent
    assert not getattr(path.stat(),'st_file_attributes',0)&(0x1000|0x40000|0x400000)


def plan():
    proof=c.read(HERE/'protected_evidence_before.json');assert proof['passed']
    final_hashes={cp['native']['project_file_sha256'] for j in proof['checkpoints'] for cp in j['checkpoints']}
    lookup=c.catalogue();verified=set();rows=[];unmatched=[];examined=0;start=last=time.monotonic()
    for line in (HERE/'inventory_local_archive.jsonl').read_text().splitlines():
        r=json.loads(line);p=LOCAL/r['path']
        if not (p.suffix.lower()=='.dynf' or ('.dynf.' in p.name.lower() and p.suffix.lower()=='.copy')):continue
        if any(x.startswith('final_') for x in p.relative_to(LOCAL).parts):continue
        allowed(p);version=c.stamp(p)
        h=c.digest(p);assert c.stamp(p)==version
        examined+=1
        if h in final_hashes:continue
        matches=[e for e in lookup.get(h,[]) if e['bytes']==r['bytes']]
        if not matches:
            unmatched.append(dict(path=str(p),bytes=r['bytes'],sha256=h));continue
        entry=matches[0]
        c.verify_archive(entry,h,r['bytes'],verified)
        rows.append(dict(path=str(p),bytes=r['bytes'],version=version,sha256=h,preservation=entry))
        if time.monotonic()-last>15:
            print(json.dumps(dict(examined=examined,verified_duplicates=len(rows),gib=sum(x['bytes'] for x in rows)/1024**3,seconds=round(time.monotonic()-start))),flush=True);last=time.monotonic()
    report=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),allowed_root=str(LOCAL.resolve()),
                candidates=rows,bytes=sum(r['bytes'] for r in rows),unmatched_retained=unmatched,
                final_generation_hashes_excluded=sorted(final_hashes),all_final_backup_directories_excluded=True,
                no_python_results_policies_or_existing_zips_targeted=True)
    c.atomic(HERE/'native_backup_duplicate_plan.json',report)
    print(json.dumps(dict(candidates=len(rows),gib=report['bytes']/1024**3,unmatched_retained=len(unmatched),examined=examined)),flush=True)


def execute():
    plan=c.read(HERE/'native_backup_duplicate_plan.json')
    assert Path(plan['allowed_root'])==LOCAL.resolve()
    verified=set();removed=[]
    for row in plan['candidates']:
        p=Path(row['path']);allowed(p)
        assert c.stamp(p)==row['version']
        c.verify_archive(row['preservation'],row['sha256'],row['bytes'],verified)
        with c.open_read(p,exclusive=True) as f:assert c.digest_stream(f)==row['sha256']
        assert c.stamp(p)==row['version'];allowed(p)
        receipt=dict(row,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='verified_before_removal')
        with (HERE/'native_backup_duplicate_receipts.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(receipt)+'\n');out.flush()
            import os;os.fsync(out.fileno())
        p.unlink()
        assert not p.exists()
        with (HERE/'native_backup_duplicate_receipts.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(dict(receipt,state='raw_removed'))+'\n');out.flush()
            os.fsync(out.fileno())
        removed.append(row)
        print(json.dumps(dict(removed=len(removed),total=len(plan['candidates']),gib=sum(r['bytes'] for r in removed)/1024**3)),flush=True)
    c.atomic(HERE/'native_backup_duplicate_result.json',dict(passed=True,files=len(removed),bytes=sum(r['bytes'] for r in removed),all_retained_archives_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','execute']);args=p.parse_args()
    plan() if args.action=='plan' else execute()
