"""Lossless compression of the fixed 910-folder archive; never deletes originals."""
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\storage_20261023').resolve()
DESTINATION = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\compressed_storage_20261023').resolve()


def digest_stream(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    assert kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    rows = json.loads((ROOT / 'change-reports/storage_20261023_1306/archive_prepared.json').read_text(encoding='utf-8-sig'))
    assert len(rows) == 910 and sum(r['bytes'] for r in rows) == 174583311764
    assert not DESTINATION.exists()
    for row in rows:
        folder = Path(row['destination']).resolve()
        assert folder.is_relative_to(ARCHIVE) and folder != ARCHIVE
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        assert actual == {r['relative'] for r in row['files']}
        for file in row['files']:
            path = (folder / file['relative']).resolve()
            assert path.is_relative_to(folder) and path.stat().st_size == file['bytes']
    DESTINATION.mkdir()
    started = time.monotonic()
    receipts = []
    for number, row in enumerate(rows):
        folder = Path(row['destination']).resolve()
        target = DESTINATION / f'{number:04d}_{folder.name}.zip'
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for file in row['files']:
                path = folder / file['relative']; check = hashlib.sha256()
                with path.open('rb') as source, archive.open(file['relative'], 'w', force_zip64=True) as saved:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        check.update(block); saved.write(block)
                assert check.hexdigest() == file['sha256']
        with zipfile.ZipFile(target) as archive:
            assert set(archive.namelist()) == {f['relative'] for f in row['files']}
            for file in row['files']:
                with archive.open(file['relative']) as stream:
                    assert digest_stream(stream) == file['sha256']
        with target.open('rb') as stream: compressed_sha = digest_stream(stream)
        receipt = dict(original_directory=str(folder), compressed=str(target),
                       original_bytes=row['bytes'], compressed_bytes=target.stat().st_size,
                       compressed_sha256=compressed_sha, files=row['files'], restored_hashes_verified=True)
        receipts.append(receipt)
        with (DESTINATION / 'receipts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(receipt) + '\n')
        status = dict(state='compressing', at_utc=datetime.now(timezone.utc).isoformat(),
                      verified_folders=len(receipts), total_folders=910,
                      original_bytes=sum(r['original_bytes'] for r in receipts),
                      compressed_bytes=sum(r['compressed_bytes'] for r in receipts),
                      seconds=time.monotonic() - started, originals_deleted=0)
        save(DESTINATION / 'status.json', status)
        if len(receipts) % 25 == 0: print(json.dumps(status), flush=True)
    status.update(state='verified_complete', at_utc=datetime.now(timezone.utc).isoformat())
    save(DESTINATION / 'status.json', status)
    save(DESTINATION / 'restore_manifest.json', dict(format='ZIP DEFLATE level 6; lossless', archives=receipts,
        restoration='Extract each ZIP only into its recorded original_directory if missing; never overwrite newer files. Validate every file SHA-256 against this manifest.',
        originals_deleted=0))
    (DESTINATION / 'README.txt').write_text('All 910 ZIP archives have verified restored file hashes.\nOriginal directories have NOT been deleted.\nSee restore_manifest.json for exact file paths and SHA-256 values.\n', encoding='utf-8')
    save(HERE / 'archive_compression_result.json', dict(status, destination=str(DESTINATION)))
    print(json.dumps(status), flush=True)


if __name__ == '__main__': main()
