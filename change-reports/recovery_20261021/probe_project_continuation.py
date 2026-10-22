"""Compare opening the saved native project with explicit snapshot loading."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
import numpy as np
import win32com.client
from column_review.common import read,write,record_owned_apps
from column_review.pilot import profile
from column_e1.runtime import load_project
from systems.distillation.plant import DistillationColumnAspen
from run_distillation_phase2_repeats import aspen_file_lock,aspen_com_compatibility

out=ROOT/'Distillation/Results/state_recovery/20261021_project_continuation_a1';out.mkdir(parents=True)
base=ROOT/'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1'
metadata=read(next((base/'capture/temporary_native').glob('*/metadata.json')))
job=read(base/'job.json')
with aspen_file_lock(Path(job['source_case']['aspen_path']),ROOT/'Distillation/Results/_aspen_file_locks'):
 with aspen_com_compatibility(),record_owned_apps(out):
  p=object.__new__(DistillationColumnAspen);p.path=str(out/'C2S_SS_simulation16.dynf');p.delta_t=1/6
  p.current_input=np.asarray(metadata['python_state']['current_input']);p.current_output=np.asarray(metadata['python_state']['current_output'])
  p.ad=win32com.client.DispatchEx('AD application');p.ad.NewDocument();p.ad.Visible=True
  rows=[]
  try:
   for load in (False,True):
    snap=load_project(p,metadata)
    if load:p.sim.Results.LoadSnapshot(snap)
    before=profile(p)[1];print('Explicit snapshot load',load,'time on open',before[0],flush=True)
    p.sim.Step(True)
    after=profile(p)[1]
    row=dict(explicit_load=load,before=before.tolist(),after=after.tolist(),difference=(after-before).tolist())
    rows.append(row);write(out/'result.json',rows)
    print('First step time change',after[0]-before[0], 'controlled change',(after-before)[[9,14]].tolist(),flush=True)
  finally:p.close()
