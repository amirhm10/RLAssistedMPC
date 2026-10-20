"""Reattach a saved E1 inference/controller snapshot to an owned native case."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from review_experiments.e1.runtime import clone_environment, rng_state, restore_rng
from review_experiments.e1.snapshots import load_snapshot
from review_experiments.e1.study import run_branch, write_json
from .runtime import DetachedColumn, COM_FIELDS, copy_column, import_native, column_copying


def replay_snapshot(path, plant, output, *, scientific_hashes, steps=None):
    """Caller owns the canonical case lease and the fresh initialized Aspen app."""
    payload, compiled, detached = load_snapshot(path, scientific_hashes=scientific_hashes)
    exported = detached['system']
    if not isinstance(exported, DetachedColumn) or not hasattr(exported, 'native_metadata'):
        raise ValueError('Not a portable native column E1 snapshot.')
    metadata = exported.native_metadata
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def attach(saved, memo):
        branch = copy_column(plant, {})
        memo[id(saved)] = branch
        branch.__dict__.update(deepcopy({key:value for key,value in vars(saved).items()
                                        if key not in COM_FIELDS and key not in {'native_metadata','native_snapshot'}}, memo))
        return branch

    initial_rng = rng_state()
    results = {}
    try:
        with column_copying(), patch.object(DetachedColumn, '__deepcopy__', attach, create=True):
            for source in ('policy', 'supervisor'):
                import_native(plant, metadata)
                # Reopening a document replaces every COM variable object.
                # Bind each branch to the newly opened document, never to the
                # previous branch's now-closed automation handles.
                environment = clone_environment(detached, compiled)
                restore_rng(payload['rng'])
                branch = run_branch(environment, compiled, compiled.sites[payload['site']], payload['record'],
                                    source, int(steps or payload['options']['rollout_steps']), payload['continuation'])
                np.savez_compressed(output / f'{source}.npz', **branch.pop('trace'))
                results[source] = branch
            import_native(plant, metadata)
    finally:
        restore_rng(initial_rng)
    write_json(output / 'replay.json', {'snapshot':str(Path(path).resolve()), 'branches':results})
    return results
