"""Restore normal CPU priority through exact-generation handles; never pause a worker."""
import argparse
import ctypes
from ctypes import wintypes as w
from contextlib import ExitStack
from pathlib import Path
import sys
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'change-reports/combined_parallel_20261024'))
import parallel_branches as p


def main(rollback=False):
    accepted = p.closed_json(HERE / 'acceptance.json')
    if not accepted['passed']:
        raise RuntimeError('Accept the real production recovery first')
    paths = {'manifest': Path(accepted['recovery_manifest']),
             'baseline': p.h.HERE / 'accepted_current_workers.json'}
    values = p.h.stable_batch(paths)
    if (values['manifest']['state'] != 'running' or values['manifest']['stage'] != 'production'
            or values['baseline']['workers']['combined']['identity'] != accepted['production_identity']):
        raise RuntimeError('Only the current accepted production is eligible')
    targets = [accepted['production_identity']] + [row['identity'] for row in accepted['parallel_branches']]
    if len({(row['pid'], row['created']) for row in targets}) != 3:
        raise RuntimeError('Exactly three distinct accepted workers are required')
    receipt = HERE / ('priority_rollback.json' if rollback else 'priority_applied.json')
    if receipt.exists():
        raise FileExistsError('Never replay a completed scheduling action')
    previous = p.closed_json(HERE / 'priority_plan.json') if rollback else None
    if previous and previous['targets'] != targets:
        raise RuntimeError('Never reuse scheduling evidence for replacement workers')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    kernel.GetPriorityClass.argtypes = [w.HANDLE]
    kernel.GetPriorityClass.restype = w.DWORD
    kernel.SetPriorityClass.argtypes = [w.HANDLE, w.DWORD]
    kernel.CloseHandle.argtypes = [w.HANDLE]

    def checked(handle, target):
        times = [w.FILETIME() for _ in range(4)]
        code = w.DWORD()
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]) or not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        if code.value != 259 or created != target['created']:
            raise RuntimeError('An exact accepted generation retired; leave replacements untouched')
        return int(kernel.GetPriorityClass(handle))

    with ExitStack() as stack:
        opened = []
        for target in targets:
            handle = kernel.OpenProcess(0x1200, False, target['pid'])
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            stack.callback(kernel.CloseHandle, handle)
            old = checked(handle, target)
            proc = psutil.Process(target['pid'])
            if proc.name().lower() != 'python.exe' or old not in (32, 16384):
                raise RuntimeError('Unexpected process or user scheduling state; do not change it')
            opened.append((handle, target, old, proc.cpu_affinity()))
        if not rollback:
            p.immutable(HERE / 'priority_plan.json', dict(at_utc=p.now(), targets=targets,
                old_priorities=[row[2] for row in opened], affinity=[row[3] for row in opened],
                receipt_sha256=p.h.digest(HERE / 'acceptance.json'), generation_bound=True))
        changed = []
        try:
            for position, (handle, target, old, affinity) in enumerate(opened):
                if checked(handle, target) != old:
                    raise RuntimeError('Scheduling changed after preparation')
                desired = previous['old_priorities'][position] if rollback else 32
                if not kernel.SetPriorityClass(handle, desired):
                    raise ctypes.WinError(ctypes.get_last_error())
                changed.append((handle, old))
                if checked(handle, target) != desired or psutil.Process(target['pid']).cpu_affinity() != affinity:
                    raise RuntimeError('Scheduling readback or unchanged affinity did not verify')
        except BaseException:
            for handle, old in reversed(changed):
                kernel.SetPriorityClass(handle, old)
            raise
        p.immutable(receipt, dict(passed=True, at_utc=p.now(), rollback=rollback, targets=targets,
            resulting_priorities=[checked(handle, target) for handle, target, _, _ in opened],
            affinity_unchanged=True, process_signals=0, pauses=0, restarts=0, simulator_calls=0,
            scientific_changes=0, same_handle_creation_checks=True))
    print('Verified exact-generation CPU priority; no pause, restart or affinity change.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollback', action='store_true')
    main(parser.parse_args().rollback)
