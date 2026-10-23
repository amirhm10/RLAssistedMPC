"""Independently rehash completed ZIPs and verify the complete restoration map."""
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
DESTINATION = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\compressed_storage_20261023').resolve()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    assert kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    original = read(ROOT / 'change-reports/storage_20261023_1306/archive_prepared.json')
    expected = {str(Path(row['destination']).resolve()): row for row in original}
    checked = []
    started = time.monotonic()
    while len(checked) < 910:
        lines = (DESTINATION / 'receipts.jsonl').read_text(encoding='utf-8').splitlines(keepends=True)
        available = [json.loads(line) for line in lines if line.endswith('\n')]
        assert len(available) <= 910
        for record in available[len(checked):]:
            key = str(Path(record['original_directory']).resolve())
            assert key in expected and key not in {r['original_directory'] for r in checked}
            row = expected[key]
            assert record['files'] == row['files'] and record['original_bytes'] == row['bytes']
            assert record['restored_hashes_verified'] is True
            path = Path(record['compressed']).resolve()
            assert path.is_relative_to(DESTINATION) and path.stat().st_size == record['compressed_bytes']
            assert digest(path) == record['compressed_sha256']
            with zipfile.ZipFile(path) as archive:
                assert {p.filename for p in archive.infolist()} == {f['relative'] for f in row['files']}
                assert all(archive.getinfo(f['relative']).file_size == f['bytes'] for f in row['files'])
            for file in row['files']:
                assert (Path(key) / file['relative']).stat().st_size == file['bytes']
            checked.append(record)
        status = dict(at_utc=datetime.now(timezone.utc).isoformat(), checked_zips=len(checked),
                      original_files=sum(len(r['files']) for r in checked),
                      compressed_bytes=sum(r['compressed_bytes'] for r in checked),
                      seconds=time.monotonic() - started, local_originals_deleted=0)
        save(HERE / 'compressed_verification_progress.json', status)
        if len(checked) < 910:
            time.sleep(5)
    while not (DESTINATION / 'restore_manifest.json').exists():
        time.sleep(1)
    manifest = read(DESTINATION / 'restore_manifest.json')
    compression = read(DESTINATION / 'status.json')
    assert compression['state'] == 'verified_complete' and compression['verified_folders'] == 910
    assert manifest['archives'] == checked and manifest['originals_deleted'] == 0
    assert status['original_files'] == 3640
    status.update(verified=True, folders=910, original_bytes=174583311764,
        all_original_paths_present=True, all_zip_hashes_rechecked=True,
        restored_file_hashes_verified_by_compressor=True,
        restoration_manifest=str(DESTINATION / 'restore_manifest.json'),
        restoration_manifest_sha256=digest(DESTINATION / 'restore_manifest.json'),
        at_utc=datetime.now(timezone.utc).isoformat())
    save(HERE / 'compressed_archive_verified.json', status)
    save(DESTINATION / 'independent_verification.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
