"""Read-only local archive verification and exact cloud recycle-bin allowlist."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOCAL=Path(r'C:\Users\hamediaa\RL_MPC_local_archive')
PREFIX='personal/hamedia_mcmaster_ca/Documents/PythonProjects/RL_assisted_MPC/'
NAMES=re.compile(r'^(?:native_[01]_\d+|load_[0-9a-f]{8})$')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    objects=[]
    for line in (HERE/'inventory_local_archive.jsonl').read_text().splitlines():
        row=json.loads(line);p=LOCAL/row['path']
        if p.suffix.lower()!='.json' or not p.is_relative_to(LOCAL/'aspen_retention_20261023/receipts') and not p.is_relative_to(LOCAL/'combined_parallel_20261024/retired_branch_zips') and not p.is_relative_to(LOCAL/'combined_inference_recovery_20261026/retired_branch_zips'):
            continue
        try:d=read(p)
        except (OSError,ValueError):continue
        source=d.get('original_directory') or d.get('source')
        archive=d.get('compressed') or d.get('archive')
        checksum=d.get('compressed_sha256') or d.get('archive_sha256')
        if not all(isinstance(x,str) for x in (source,archive,checksum)):continue
        source=Path(source);archive=Path(archive)
        if not source.is_relative_to(ROOT/'Distillation/Results') or not NAMES.fullmatch(source.name):continue
        if source.exists():continue
        if not archive.is_relative_to(LOCAL) or not archive.is_file():continue
        files=d.get('files',[])
        if not files or not all(isinstance(f,dict) and 'relative' in f and 'sha256' in f and 'bytes' in f for f in files):continue
        if any(Path(f['relative']).suffix.lower() in ('.pkl','.pt','.pth','.npz','.npy','.csv','.tsv','.png','.pdf') for f in files):continue
        if d.get('state') not in ('raw_removed','verified') and not d.get('verified'):continue
        objects.append(dict(cloud_path=PREFIX+source.relative_to(ROOT).as_posix(),kind='folder',
                            original=str(source),archive=str(archive),archive_sha256=checksum,
                            files=files,raw_bytes=sum(f['bytes'] for f in files)))
    log=HERE/'native_compaction_receipts.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line)
            if r['state']!='raw_removed':continue
            source=Path(r['source'])
            objects.append(dict(cloud_path=PREFIX+source.relative_to(ROOT).as_posix(),kind='file',
                                original=str(source),archive=r['archive'],archive_sha256=r['archive_sha256'],
                                files=[dict(relative=r['member'],sha256=r['source_sha256'],bytes=r['source_bytes'])],raw_bytes=r['source_bytes']))
    # A temporary export can have a different pathname from its byte-identical
    # archived private load copy. Immutable snapshot metadata supplies the full
    # original SHA-256; the catalogue supplies the matching verified ZIP member.
    matches=HERE/'cloud_review/metadata_archive_matches.json'
    if matches.exists():
        for r in read(matches):
            source=Path(r['source']);entry=r['archive']
            if source.exists() or not source.is_relative_to(ROOT/'Distillation/Results'):continue
            objects.append(dict(cloud_path=PREFIX+source.relative_to(ROOT).as_posix(),kind='file',
                original=str(source),archive=entry['archive'],archive_sha256=entry['archive_sha256'],
                files=[dict(relative=entry['member'],sha256=r['sha256'],bytes=entry['bytes'])],
                raw_bytes=entry['bytes'],original_identity_source='saved native metadata project_file_sha256'))
    # Fresh archive digests, central-directory names and uncompressed sizes
    # establish that the previously fully verified lossless archives are intact.
    # Current cleanup archives additionally passed full decompression before removal.
    cache={};result=[];errors=[];last=time.monotonic();start=last
    for row in objects:
        p=Path(row['archive']);key=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
        if key not in cache:
            actual=sha(p)
            if actual!=row['archive_sha256']:
                errors.append(dict(archive=str(p),error='fresh archive checksum differs'));continue
            with zipfile.ZipFile(p) as z:
                cache[key]={x.filename:x.file_size for x in z.infolist()}
        members=cache[key]
        if not all(members.get(f['relative'].replace('\\','/'))==f['bytes'] for f in row['files']):
            errors.append(dict(archive=str(p),error='ZIP member manifest differs'));continue
        row['fresh_archive_sha256_verified']=True
        result.append(row)
        if time.monotonic()-last>15:
            print(json.dumps(dict(verified=len(result),total=len(objects),seconds=round(time.monotonic()-start))),flush=True);last=time.monotonic()
    target=HERE/'cloud_review';target.mkdir(exist_ok=True)
    report=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                entries={r['cloud_path']:r for r in result},errors=errors,
                note='Exact paths only; no whole-bin deletion. Protected results and current checkpoints are excluded. UI must be matched to this list before action-time confirmation and selective permanent deletion.')
    (target/'whitelist.json').write_text(json.dumps(report),encoding='utf-8')
    print(json.dumps(dict(allowed_paths=len(report['entries']),fresh_archives_verified=len(cache),errors=errors,
                          raw_gib=sum(r['raw_bytes'] for r in report['entries'].values())/1024**3,
                          seconds=round(time.monotonic()-start))),flush=True)


if __name__=='__main__':main()
