"""Expose child ownership and close the pool before the original cleanup check."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import time

import parallel_branches as p


def merge_owned(parent_records, child_records, *, case, journal, worker_identity):
    result=deepcopy(parent_records)
    known={(r['pid'],r['created']) for r in result}
    for row in child_records:
        key=(row['pid'],row['created'])
        if key not in known:
            result.append(dict(row,role='parallel_branch',case=case,
                               source_ownership_journal=str(journal),branch_worker=worker_identity))
            known.add(key)
    return result


class ManagedBranchPool(p.BranchPool):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.next_owner_sync=0.

    def sync_owners(self,*,force=False):
        target=self.template.get('aggregate_journal')
        if not target or not force and time.monotonic()<self.next_owner_sync:return
        target=Path(target).resolve()
        expected=p.h.OUTPUTS['combined']/'owned_aspen_apps.json'
        if target!=expected.resolve():raise RuntimeError('Only this combined parent ownership journal may aggregate child ownership')
        paths={case:w['directory']/'owned_aspen_apps.json' for case,w in self.workers.items()
               if (w['directory']/'owned_aspen_apps.json').exists()}
        if not paths:return
        paths['parent']=target
        values=p.h.stable_batch(paths)
        before=values.pop('parent');merged=before
        for case,rows in values.items():
            identity=p.closed_json(self.workers[case]['directory']/'identity.json')
            merged=merge_owned(merged,rows,case=case,journal=paths[case],worker_identity=identity)
        if merged!=before:
            from recover_markov_seed23 import write
            write(target,merged)
        self.next_owner_sync=time.monotonic()+5.

    def check_processes(self):
        super().check_processes()
        self.sync_owners()

    def close(self):
        if self.closed:return
        super().close()
        self.sync_owners(force=True)


def closing_stack(pool):
    class PoolClosingStack(ExitStack):
        def __enter__(self):
            super().__enter__()
            # Registered first, so the original plant/branch cleanup runs
            # before this callback; child retirement still precedes the
            # original worker's final owned_apps_remaining verification.
            self.callback(pool.close)
            return self
    return PoolClosingStack
