"""Losslessly compact closed historical Aspen files, with durable restore maps.

Only local .dynf files in Distillation/Results are eligible. Raw scientific data,
policies, sources, source models and both final checkpoints per job are excluded.
Every removal requires a source hash, full ZIP-member decompression/hash check,
an unchanged source generation and a flushed receipt. No directories are deleted.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ALLOWED=ROOT/'Distillation/Results'
ARCHIVE=ROOT/'archive/manuscript_native_states_20261028'
LOCAL=Path(r'C:\Users\hamediaa\RL_MPC_local_archive')
sys.path.insert(0,str(ROOT))
from aspen_retention import open_read


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stamp(path):
    s=path.stat()
    return [s.st_size,s.st_mtime_ns,s.st_ino]


def digest_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(4*1024**2),b''):
        h.update(block)
    return h.hexdigest()


def digest(path):
    with open_read(path) as stream:
        return digest_stream(stream)


def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def journal(obj):
    with (HERE/'native_compaction_receipts.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(obj)+'\n')
        stream.flush();os.fsync(stream.fileno())


def eligible(path,protected):
    resolved=path.resolve(strict=True)
    if not resolved.is_relative_to(ALLOWED.resolve()) or path.suffix.lower()!='.dynf':
        raise ValueError('Outside approved file type/root: '+str(path))
    for root in protected:
        if resolved.is_relative_to(Path(root).resolve()):
            raise ValueError('Protected final checkpoint: '+str(path))
    current=path
    while current!=ALLOWED.parent:
        stat=current.lstat()
        if getattr(stat,'st_reparse_tag',0) in (0xA0000003,0xA000000C) or current.is_symlink():
            raise ValueError('Junction/symlink refused: '+str(current))
        current=current.parent
    if getattr(path.stat(),'st_file_attributes',0)&(0x1000|0x40000|0x400000):
        raise ValueError('Online-only source is excluded: '+str(path))
    return resolved


def make_plan():
    proof=read(HERE/'protected_evidence_before.json')
    assert proof['passed'] and not proof['errors']
    protected=proof['protected_native_directories']
    rows=[];excluded=[]
    for line in (HERE/'inventory_code.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);p=ROOT/r['path']
        if p.suffix.lower()!='.dynf' or not p.is_relative_to(ALLOWED):
            continue
        try:
            eligible(p,protected)
        except ValueError as exc:
            excluded.append(dict(path=str(p),reason=str(exc)))
            continue
        assert stamp(p)[:2]==[r['bytes'],r['mtime_ns']],str(p)
        rows.append(dict(path=str(p),bytes=r['bytes'],version=stamp(p)))
    rows.sort(key=lambda r:r['bytes'],reverse=True)
    result=dict(version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                allowed_root=str(ALLOWED.resolve()),protected=protected,candidates=rows,
                candidate_bytes=sum(r['bytes'] for r in rows),excluded=excluded,
                preservation='Native states remain losslessly recoverable from verified ZIP members. No raw numerical result, policy, source model, Python checkpoint or manuscript is a target.')
    atomic(HERE/'native_compaction_plan.json',result)
    print(json.dumps(dict(files=len(rows),raw_gib=round(result['candidate_bytes']/1024**3,3),
                          protected_or_online=len(excluded),plan=str(HERE/'native_compaction_plan.json'))),flush=True)


def catalogue():
    result=collections.defaultdict(list)
    def visit(obj):
        if isinstance(obj,dict):
            archive=obj.get('archive') or obj.get('compressed')
            ah=obj.get('archive_sha256') or obj.get('compressed_sha256')
            if isinstance(archive,str) and archive.lower().endswith('.zip') and isinstance(ah,str):
                q=Path(archive)
                if q.is_relative_to(LOCAL) and q.is_file():
                    for f in obj.get('files',[]):
                        if isinstance(f,dict) and 'sha256' in f and 'relative' in f and f['relative'].lower().endswith('.dynf'):
                            result[f['sha256']].append(dict(archive=str(q),archive_sha256=ah,member=f['relative'].replace('\\','/'),bytes=f['bytes']))
            for key,value in obj.items():
                if key!='files':visit(value)
        elif isinstance(obj,list):
            for value in obj:visit(value)
    for line in (HERE/'inventory_local_archive.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if Path(r['path']).suffix.lower()=='.json' and r['bytes']<20*1024**2:
            try:visit(read(LOCAL/r['path']))
            except (ValueError,OSError):continue
    # Own previously completed archives make interrupted execution resumable.
    log=HERE/'native_compaction_receipts.jsonl'
    if log.exists():
        for line in log.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('verified'):
                result[row['source_sha256']].append(dict(archive=row['archive'],archive_sha256=row['archive_sha256'],member=row['member'],bytes=row['source_bytes']))
    return result


def verify_archive(entry,source_hash,source_size,cache):
    archive=Path(entry['archive'])
    version=stamp(archive)
    key=(str(archive),tuple(version),entry['member'],source_hash)
    if key not in cache:
        assert digest(archive)==entry['archive_sha256'],'Archive SHA-256 differs: '+str(archive)
        with zipfile.ZipFile(archive) as z:
            info=z.getinfo(entry['member'])
            assert info.file_size==source_size
            with z.open(info) as f:
                assert digest_stream(f)==source_hash,'Decompressed content differs'
        assert stamp(archive)==version
        cache.add(key)


def execute(limit):
    plan=read(HERE/'native_compaction_plan.json')
    proof=read(HERE/'protected_evidence_before.json')
    assert proof['passed'] and not proof['errors']
    assert Path(plan['allowed_root'])==ALLOWED.resolve()
    assert plan['protected']==proof['protected_native_directories']
    lookup=catalogue();cache=set()
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    completed=0;freed=0;new_bytes=0;start=time.monotonic()
    for row in plan['candidates']:
        path=Path(row['path'])
        if not path.exists():
            continue
        eligible(path,plan['protected'])
        assert stamp(path)==row['version'],'Source changed since plan: '+str(path)
        # Exclusive read refuses files still open in a native application.
        with open_read(path,exclusive=True) as stream:
            source_hash=digest_stream(stream)
        assert stamp(path)==row['version']
        entry=None
        for known in lookup.get(source_hash,[]):
            if known['bytes']==row['bytes']:
                verify_archive(known,source_hash,row['bytes'],cache)
                entry=known
                break
        created=False
        if entry is None:
            archive=ARCHIVE/(source_hash+'.zip')
            if archive.exists():
                entry=dict(archive=str(archive),archive_sha256=digest(archive),member='state.dynf',bytes=row['bytes'])
            else:
                temporary=archive.with_suffix('.zip.tmp')
                assert not temporary.exists()
                with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
                    with open_read(path,exclusive=True) as source,z.open('state.dynf','w',force_zip64=True) as target:
                        shutil.copyfileobj(source,target,length=4*1024**2)
                os.replace(temporary,archive)
                entry=dict(archive=str(archive),archive_sha256=digest(archive),member='state.dynf',bytes=row['bytes'])
                created=True
            verify_archive(entry,source_hash,row['bytes'],cache)
            lookup[source_hash].append(entry)
        # Re-read source independently after the decompression verification.
        with open_read(path,exclusive=True) as stream:
            assert digest_stream(stream)==source_hash
        eligible(path,plan['protected'])
        assert stamp(path)==row['version'],'Source generation changed before removal'
        receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     state='verified_before_removal',verified=True,source=str(path),
                     source_bytes=row['bytes'],source_sha256=source_hash,source_version=row['version'],
                     archive=entry['archive'],archive_sha256=entry['archive_sha256'],member=entry['member'],
                     archive_bytes=Path(entry['archive']).stat().st_size,new_archive=created,
                     full_decompression_sha256_verified=True)
        journal(receipt)
        # Deliberately one verified file unlink, never a recursive deletion.
        path.unlink()
        assert not path.exists()
        journal(dict(receipt,state='raw_removed'))
        completed+=1;freed+=row['bytes'];new_bytes+=receipt['archive_bytes'] if created else 0
        atomic(HERE/'native_compaction_progress.json',dict(files_this_pass=completed,raw_bytes_removed=freed,
               new_archive_bytes=new_bytes,net_bytes_reclaimed=freed-new_bytes,
               elapsed_seconds=round(time.monotonic()-start,1),last_source=str(path)))
        print(json.dumps(dict(done=completed,total=len(plan['candidates']),source_gib=round(row['bytes']/1024**3,3),
                              new_archive=created,reclaimed_gib=round((freed-new_bytes)/1024**3,3),seconds=round(time.monotonic()-start,1))),flush=True)
        if limit and completed>=limit:break


def restore(source,destination):
    """Restore to a new staging path only; existing paths are never overwritten."""
    rows=[json.loads(x) for x in (HERE/'native_compaction_receipts.jsonl').read_text(encoding='utf-8').splitlines()]
    matches=[r for r in rows if r.get('state')=='raw_removed' and Path(r['source'])==Path(source)]
    assert matches,'No verified removal receipt for the requested original'
    row=matches[-1]
    target=Path(destination)
    assert not target.exists(),'Choose a new destination'
    assert target.parent.exists(),'Destination parent must already exist'
    verify_archive(dict(archive=row['archive'],archive_sha256=row['archive_sha256'],member=row['member']),row['source_sha256'],row['source_bytes'],set())
    with zipfile.ZipFile(row['archive']) as z,z.open(row['member']) as f,target.open('xb') as out:
        shutil.copyfileobj(f,out,length=4*1024**2)
    assert digest(target)==row['source_sha256']
    print(json.dumps(dict(restored=str(target),sha256=row['source_sha256'],bytes=target.stat().st_size)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','execute','restore'])
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--source')
    parser.add_argument('--destination')
    args=parser.parse_args()
    if args.action=='plan':make_plan()
    elif args.action=='execute':execute(args.limit)
    else:
        assert args.source and args.destination
        restore(args.source,args.destination)
