"""Reproduce E1 initialization with pending versus last-integrated feed."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
import numpy as np
import win32com.client
from column_review.common import read, write, record_owned_apps, owned_apps_remaining
from column_review.pilot import profile
from column_e1.runtime import load_project
from systems.distillation.plant import DistillationColumnAspen
from systems.distillation.scenarios import _generate_feed_fluctuation_segment
from systems.distillation.config import DISTILLATION_NOMINAL_CONDITIONS
from run_distillation_phase2_repeats import aspen_file_lock, aspen_com_compatibility

OUT=ROOT/'Distillation/Results/state_recovery/20261021_e1_feed_probe_a2'
OUT.mkdir(parents=True,exist_ok=False)
source=ROOT/'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1/capture/temporary_native'
metadata=read(next(source.glob('*/metadata.json')))
expected=np.asarray(metadata['internal'])
nominal=float(DISTILLATION_NOMINAL_CONDITIONS[0])
schedule,_=_generate_feed_fluctuation_segment(total_steps=80000,nominal_feed=nominal,current_feed=nominal,rng=np.random.RandomState(42))
indices=np.flatnonzero(np.isclose(schedule,expected[1],rtol=0,atol=1e-8))
print('Saved pending feed matches original schedule at',indices[:12].tolist(),flush=True)
assert 5201 in indices
case=read(ROOT/'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1/job.json')['source_case']
rows=[]
with aspen_file_lock(Path(case['aspen_path']),ROOT/'Distillation/Results/_aspen_file_locks'):
 with aspen_com_compatibility(),record_owned_apps(OUT):
  plant=object.__new__(DistillationColumnAspen)
  plant.path=str(OUT/'C2S_SS_simulation16.dynf');plant.delta_t=1/6
  plant.current_input=np.asarray(metadata['python_state']['current_input']);plant.current_output=np.asarray(metadata['python_state']['current_output'])
  plant.ad=win32com.client.DispatchEx('AD application');plant.ad.NewDocument();plant.ad.Visible=True
  try:
   native=load_project(plant,metadata)
   for solved in (None,float(schedule[5200]),float(schedule[5200])):
    plant.sim.Results.LoadSnapshot(native)
    plant.feed.FmR.Value=expected[1] if solved is None else solved
    plant.col.Reflux.FmR.Value=expected[2];plant.col.QRebR.Value=expected[3]
    before=profile(plant)[1]
    plant.sim.Step(True,False)
    after=profile(plant)[1]
    plant.feed.FmR.Value=expected[1]
    corrected=profile(plant)[1]
    row=dict(last_solved_feed=solved,pending_feed=float(expected[1]),before=before.tolist(),after=after.tolist(),
             restored_pending=corrected.tolist(),difference=(corrected-expected).tolist())
    rows.append(row);write(OUT/'trials.json',rows)
    print('Last solved feed:',solved,'controlled differences:',(corrected-expected)[[9,14]].tolist(),flush=True)
   tolerance=np.array([1e-9,1e-3,1e-3,1e-6]+[v for _ in range(6) for v in (2e-3,1e-4)]);tolerance[9]=1e-6
   passed=all(np.all(np.abs(np.asarray(r['difference']))<=tolerance) for r in rows[1:])
   write(OUT/'result.json',dict(passed=bool(passed),solved_feed=float(schedule[5200]),pending_feed=float(expected[1]),
                              original_failure_reproduced=bool(np.any(np.abs(rows[0]['difference'])>tolerance)),rows=rows))
   assert passed
  finally:
   plant.close()
assert not owned_apps_remaining(OUT/'owned_aspen_apps.json')
