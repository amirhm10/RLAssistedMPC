"""Retry transient atomic ownership-metadata replacement; never change scientific files."""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import time
import uuid
from unittest.mock import patch

NAMES = frozenset(('ownership_status.json', 'owned_aspen_apps.json'))


def eligible(path, roots):
    path = Path(path).resolve()
    return path.name in NAMES and any(path.is_relative_to(Path(root).resolve()) for root in roots)


def atomic_ownership(path, value, roots, *, timeout=60., replace=os.replace, pause=time.sleep):
    path = Path(path).resolve()
    if not eligible(path, roots):
        raise ValueError('Only this attempt ownership metadata is eligible')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic() + timeout
    while True:
        try:
            replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                # Retain both the previous metadata and this unique temporary
                # on permanent failure, so ownership evidence is not lost.
                raise
            pause(.2)


@contextmanager
def installed(roots):
    import column_review.common as common
    import recover_markov_seed23 as recovery
    with ExitStack() as stack:
        for module in (common, recovery):
            original = module.write

            def guarded(path, value, original=original):
                if eligible(path, roots):
                    return atomic_ownership(path, value, roots)
                return original(path, value)

            stack.enter_context(patch.object(module, 'write', guarded))
        yield
