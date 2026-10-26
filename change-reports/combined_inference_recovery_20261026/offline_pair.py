"""Verify a closed checkpoint whose only newer entry is its native restore audit.

The shared retention guard is unchanged. This exception requires an exited
failed parent and no native applications, and still performs all original
Python/native/component checksum and linkage checks in Verifier.pair.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'change-reports/combined_parallel_20261024'))
import parallel_branches as p
from aspen_retention import Deferred, NATIVE, safe_child


def closed_layout(directory, failed):
    if p.h.process(failed['pid']).get('alive') is not False or p.h.native_processes():
        raise Deferred('Offline verification requires the failed process and every native app to be closed')
    directory = Path(directory)
    directory_mtime = p.h.attributes(directory)['mtime']
    latest = p.h.version(directory / 'latest.json')
    slots = [p.h.version(directory / f'training_state_{slot}.pkl') for slot in (0, 1)]
    if any(stamp[3] > latest[3] for stamp in slots):
        raise Deferred('A Python slot is newer than its index')
    entries = []
    for path in directory.iterdir():
        if path.name.startswith('training_state_') and path.name.endswith('.tmp'):
            raise Deferred('A checkpoint temporary exists')
        if not (NATIVE.fullmatch(path.name) and path.is_dir()):
            continue
        safe_child(path, directory)
        children = [child for child in path.iterdir() if child.is_dir()]
        for child in children:
            safe_child(child, directory)
        entries.append((path.name, tuple(sorted(child.name for child in children))))
        for folder in [path] + children:
            if p.h.attributes(folder)['mtime'] <= latest[3]:
                continue
            audits = []
            for child in folder.iterdir():
                if child.is_dir():
                    if p.h.attributes(child)['mtime'] > latest[3]:
                        raise Deferred('A newer native subdirectory is not an audit-only change')
                    continue
                if p.h.version(child)[3] <= latest[3]:
                    continue
                if child.name != 'snapshot.restores.jsonl':
                    raise Deferred('A newer non-audit native file needs investigation')
                records = p.h.completed_records(child)
                p.verify_native_restore(records)
                audits.append(child)
            if not audits:
                raise Deferred('A native directory changed without a verified restore audit')
    if p.h.attributes(directory)['mtime'] != directory_mtime or p.h.version(directory / 'latest.json') != latest:
        raise Deferred('Closed checkpoint attributes changed during verification')
    return dict(index=latest, slots=slots, directory_mtime=directory_mtime,
                native_entries=tuple(sorted(entries)))
