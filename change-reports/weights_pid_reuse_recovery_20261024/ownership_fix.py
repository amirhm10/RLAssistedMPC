"""Select one worker-owned native process generation; never signal a process."""
from contextlib import contextmanager
from unittest.mock import patch


def select_identity(pid, remembered, records, current):
    if current.get('alive') is None:
        raise RuntimeError('Native process identity is inaccessible')
    candidates = [r for r in records if r.get('pid') == pid]
    if remembered and remembered.get('pid') == pid:
        expected = remembered['created']
    elif current.get('alive') is True:
        expected = current.get('created')
    else:
        generations = {r.get('created') for r in candidates}
        if len(generations) != 1:
            raise RuntimeError('Exited native process generation is ambiguous')
        expected = next(iter(generations))
    if expected is None or not any(r.get('created') == expected for r in candidates):
        raise RuntimeError('Native process generation is absent from this worker ownership journal')
    if current.get('alive') is True and current.get('created') != expected:
        raise RuntimeError('Native PID has been reused; the new generation must not be touched')
    return dict(pid=pid, alive=True, created=expected)


@contextmanager
def installed(recovery, reader, identity):
    def owned_identity(plant, output):
        from pathlib import Path
        remembered = recovery._PROCESS_IDENTITIES.get(id(plant))
        try:
            pid = int(plant.ad.processId)
        except Exception:
            if remembered is None:
                raise RuntimeError('Cannot identify the owned native process')
            pid = remembered['pid']
        records = reader(Path(output) / 'owned_aspen_apps.json', [])
        return select_identity(pid, remembered, records, identity(pid))
    with patch.object(recovery, 'owned_identity', owned_identity):
        yield
