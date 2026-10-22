"""Restore saved native values after zero-time Aspen integrator priming.

CopyValues is the documented full-variable snapshot copy. Unlike LoadSnapshot,
it retains the initialized integrator clock. Fidelity bounds are not relaxed.
"""
from copy import deepcopy
import json
import numpy as np


def restore_snapshot(self):
    from column_review.pilot import profile
    from review_experiments.e1.runtime import RestorationError
    snapshot = self.restored_snapshot
    plant = self.plant
    plant.sim.Results.LoadSnapshot(snapshot.native)
    plant.feed.FmR.Value=float(snapshot.internal[1])
    plant.col.Reflux.FmR.Value=float(snapshot.internal[2])
    plant.col.QRebR.Value=float(snapshot.internal[3])
    plant.current_input=snapshot.python_state['current_input'].copy()
    before=profile(plant)[1]
    tolerance=np.array([1e-9,1e-3,1e-3,1e-6]+[v for _ in range(6) for v in (1e-5,1e-9)])
    if np.any(np.abs(before-snapshot.internal)>tolerance):
        raise RestorationError('Immediate native restoration differs from its saved values')
    plant.sim.Step(True,False)
    primed=profile(plant)[1]
    if abs(primed[0]-before[0])>1e-9:
        raise RestorationError('Native priming advanced time')
    plant.sim.Results.CopyValues(snapshot.native,True,True,True,True,True,True,'','~')
    after=profile(plant)[1]
    row=dict(before=before.tolist(),primed=primed.tolist(),after=after.tolist(),
             difference=(after-before).tolist(),priming_difference=(primed-before).tolist(),
             full_variable_copy=True)
    self.priming_records.append(row)
    with self.snapshot.path.with_suffix('.restores.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    if np.any(np.abs(after-snapshot.internal)>tolerance):
        raise RestorationError('Full native variable copy failed the original immediate-state fidelity bounds')
    plant.current_output=self.snapshot.python_state['current_output'].copy()
    return deepcopy(snapshot.python_state['controller'])


def install():
    from column_e1.runtime import E1NativeSnapshot
    E1NativeSnapshot.restore=restore_snapshot
