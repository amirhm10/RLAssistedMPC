"""Read-only metadata inventory; never opens file content or follows junctions."""
import collections
import ctypes
import datetime
import json
import os
from pathlib import Path
import shutil
import time

OUT = Path(__file__).resolve().parent
ROOTS = {
    'code': Path(r'C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC'),
    'paper': Path(r'C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\Rl_assisted_paper'),
    'local_archive': Path(r'C:\Users\hamediaa\RL_MPC_local_archive'),
}
get_size = ctypes.WinDLL('kernel32', use_last_error=True).GetCompressedFileSizeW
get_size.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
get_size.restype = ctypes.c_ulong


def allocated(path):
    high = ctypes.c_ulong()
    ctypes.set_last_error(0)
    low = get_size(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        return None
    return (high.value << 32) | low


def main():
    errors, skipped = [], []
    summary = {}
    for label, root in ROOTS.items():
        totals = collections.defaultdict(lambda: dict(files=0, bytes=0, allocated=0, offline_bytes=0))
        biggest = []
        count = 0
        started = time.monotonic()
        target = OUT / f'inventory_{label}.jsonl'
        with target.open('x', encoding='utf-8') as stream:
            if not root.exists():
                errors.append(dict(root=str(root), error='missing root'))
                continue
            stack = [root]
            while stack:
                directory = stack.pop()
                try:
                    entries = list(os.scandir(directory))
                except OSError as exc:
                    errors.append(dict(path=str(directory), error=str(exc)))
                    continue
                for entry in entries:
                    try:
                        st = entry.stat(follow_symlinks=False)
                        tag = getattr(st, 'st_reparse_tag', 0)
                        if tag in (0xA0000003, 0xA000000C) or entry.is_symlink():
                            skipped.append(dict(path=entry.path, tag=tag))
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in ('.git', '.agents', '.codex'):
                                continue
                            stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        rel = Path(entry.path).relative_to(root)
                        if OUT == Path(entry.path).parent:
                            continue
                        attrs = getattr(st, 'st_file_attributes', 0)
                        alloc = allocated(entry.path)
                        row = dict(path=str(rel), bytes=st.st_size, allocated=alloc,
                                   attributes=attrs, reparse_tag=tag, mtime_ns=st.st_mtime_ns,
                                   inode=st.st_ino, links=st.st_nlink)
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        count += 1
                        for key in ('ALL', str(Path(*rel.parts[:1])), str(Path(*rel.parts[:2]))):
                            target_totals = totals[key]
                            target_totals['files'] += 1
                            target_totals['bytes'] += st.st_size
                            target_totals['allocated'] += alloc or 0
                            if attrs & (0x1000 | 0x40000 | 0x400000):
                                target_totals['offline_bytes'] += st.st_size
                        biggest.append(row)
                        if len(biggest) > 400:
                            biggest = sorted(biggest, key=lambda x: x['bytes'], reverse=True)[:100]
                    except OSError as exc:
                        errors.append(dict(path=entry.path, error=str(exc)))
        summary[label] = dict(root=str(root), totals=dict(totals),
                              biggest=sorted(biggest, key=lambda x: x['bytes'], reverse=True)[:60],
                              duration_seconds=time.monotonic()-started)
        print(json.dumps(dict(root=label, files=count, totals=totals.get('ALL'),
                              seconds=round(time.monotonic()-started, 1))), flush=True)
    report = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  disk_usage=shutil.disk_usage('C:\\')._asdict(), roots=summary,
                  errors=errors, skipped_links=skipped)
    (OUT / 'inventory_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(errors=len(errors), skipped_links=len(skipped), disk=report['disk_usage'])), flush=True)


if __name__ == '__main__':
    main()
