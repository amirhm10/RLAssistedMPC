"""Read every archived byte and preserve full failed-E1 checkpoint dependencies."""
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import shutil
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\storage_20261023').resolve()
BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\e1_recovery_20261023').resolve()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
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
    started = time.monotonic()
    backups = []
    for category in ('weights', 'markov', 'residual'):
        folder = BACKUP / category
        index = json.loads((folder / 'latest.json').read_text(encoding='utf-8-sig'))
        assert digest(folder / index['file']) == index['sha256']
        native = index['native']
        source = Path(native['project_file']).parent.parent.resolve()
        assert source.is_relative_to(ROOT) and source.name.startswith('native_')
        destination = folder / source.name
        assert not destination.exists(), destination
        shutil.copytree(source, destination)
        files = []
        for original in source.rglob('*'):
            if not original.is_file():
                continue
            target = destination / original.relative_to(source)
            before = digest(original)
            assert digest(target) == before
            files.append(dict(source=str(original), local_copy=str(target), bytes=target.stat().st_size, sha256=before))
        assert digest(Path(native['project_file'])) == native['project_file_sha256']
        assert digest(Path(native['file'])) == native['file_sha256']
        for path in folder.glob('training_state_*.pkl'):
            files.append(dict(local_copy=str(path), bytes=path.stat().st_size, sha256=digest(path)))
        record = dict(category=category, cycle=index['completed_cycles'], step=index['completed_steps'],
                      latest_index=str(folder / 'latest.json'), files=files,
                      restoration='Keep original latest.json unchanged as provenance. Restore its Python slot and the recorded native directory to their recorded paths only when missing; never overwrite a live/newer checkpoint.')
        save(folder / 'offline_preservation.json', record)
        backups.append(record)
    save(HERE / 'failed_e1_offline_preservation.json', dict(verified=True, backups=backups, at_utc=datetime.now(timezone.utc).isoformat()))
    rows = json.loads((ROOT / 'change-reports/storage_20261023_1306/archive_prepared.json').read_text(encoding='utf-8-sig'))
    assert len(rows) == 910
    count = total = 0
    for number, row in enumerate(rows, 1):
        folder = Path(row['destination']).resolve()
        assert folder.is_relative_to(ARCHIVE) and folder != ARCHIVE
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        assert actual == {file['relative'] for file in row['files']}
        for file in row['files']:
            path = folder / file['relative']
            assert path.stat().st_size == file['bytes'] and digest(path) == file['sha256'], path
            count += 1
            total += file['bytes']
        if number % 25 == 0:
            save(HERE / 'local_preservation_progress.json', dict(verified_folders=number, files=count, bytes=total, at_utc=datetime.now(timezone.utc).isoformat()))
    assert count == 3640 and total == 174583311764
    receipt = dict(verified=True, folders=len(rows), files=count, bytes=total,
                   original_archive=str(ARCHIVE), failed_e1_backups=backups,
                   originals_deleted=0, seconds=time.monotonic() - started, at_utc=datetime.now(timezone.utc).isoformat(),
                   scope='All original archive files read locally and SHA-256 matched. Failed E1 native/Python state copied and hash verified outside OneDrive. No local files removed.')
    save(HERE / 'local_preservation_verified.json', receipt)
    save(ARCHIVE / 'local_preservation_verified_20261023.json', receipt)
    print(json.dumps(dict(verified=True, folders=len(rows), files=count, bytes=total, originals_deleted=0)), flush=True)


if __name__ == '__main__':
    main()
