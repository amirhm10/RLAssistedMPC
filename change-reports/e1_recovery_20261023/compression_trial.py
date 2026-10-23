"""Measure verified lossless ZIP savings on inactive checkpoint copies."""
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
OUT = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\compression_trial_20261023')
OUT.mkdir(exist_ok=False)
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
prepared = json.loads((ROOT / 'change-reports/storage_20261023_1306/archive_prepared.json').read_text(encoding='utf-8-sig'))
chosen = []
for parent in ('column_e1/20261020', '/e1/horizon/'):
    eligible = [r for r in prepared if parent in r['source'].replace('\\', '/')]
    row = eligible[0] if parent.startswith('column_e1') else max(eligible, key=lambda r: r['bytes'])
    file = next(r for r in row['files'] if r['relative'].endswith('.dynf'))
    chosen.append((Path(row['destination']) / file['relative'], file['sha256']))
baseline = json.loads((HERE / 'before.json').read_text())
job = next(r for r in baseline['jobs'] if r['study'] == 'E1_column' and r['job'] == 'weights')
chosen.append((Path(job['directory']) / 'training_checkpoints' / job['latest_checkpoint']['file'], job['latest_checkpoint']['sha256']))
rows = []
for number, (source, expected) in enumerate(chosen):
    target = OUT / f'{number}_{source.name}.zip'
    started = time.monotonic(); original = hashlib.sha256()
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        with source.open('rb') as stream, archive.open(source.name, 'w', force_zip64=True) as saved:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                original.update(block); saved.write(block)
    assert original.hexdigest() == expected
    restored = hashlib.sha256()
    with zipfile.ZipFile(target) as archive:
        with archive.open(source.name) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): restored.update(block)
    assert restored.hexdigest() == expected
    rows.append(dict(source=str(source), compressed=str(target), source_bytes=source.stat().st_size,
                     compressed_bytes=target.stat().st_size, sha256=expected, restored_sha256=restored.hexdigest(),
                     fraction_saved=1 - target.stat().st_size / source.stat().st_size,
                     seconds=time.monotonic() - started, original_untouched=True))
    print(json.dumps(rows[-1]), flush=True)
result = dict(at_utc=datetime.now(timezone.utc).isoformat(), format='ZIP DEFLATE level 6; lossless',
              files=rows, originals_deleted=0, active_files_modified=0,
              caveat='Three representative files; no account-wide compression ratio is established')
(HERE / 'compression_trial.json').write_text(json.dumps(result, indent=2) + '\n')
