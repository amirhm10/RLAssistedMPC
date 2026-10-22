"""Test documented full-variable copying after Aspen's zero-time priming."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
import numpy as np
import win32com.client
from column_review.common import read,write,record_owned_apps
from column_review.pilot import profile
from column_e1.runtime import load_project
from systems.distillation.plant import DistillationColumnAspen
from run_distillation_phase2_repeats import aspen_file_lock,aspen_com_compatibility
out=ROOT/'Distillation/Results/state_recovery/20261021_copyvalues_a1';out.mkdir(parents=True,exist_ok=False)
base=ROOT/'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1'
meta=read(next((base/'capture/temporary_native').glob('*/metadata.json')));job=read(base/'job.json');expected=np.asarray(meta['internal'])
rows=[]
with aspen_file_lock(Path(job['source_case']['aspen_path']),ROOT/'Distillation/Results/_aspen_file_locks'):
 with aspen_com_compatibility(),record_owned_apps(out):
  p=object.__new__(DistillationColumnAspen);p.path=str(out/'C2S_SS_simulation16.dynf');p.delta_t=1/6
  p.current_input=np.asarray(meta['python_state']['current_input']);p.current_output=np.asarray(meta['python_state']['current_output'])
  p.ad=win32com.client.DispatchEx('AD application');p.ad.NewDocument();p.ad.Visible=True
  try:
   snap=load_project(p,meta)
   for repeat in range(3):
    p.sim.Results.LoadSnapshot(snap)
    p.feed.FmR.Value=expected[1];p.col.Reflux.FmR.Value=expected[2];p.col.QRebR.Value=expected[3]
    p.sim.Step(True,False);primed=profile(p)[1]
    p.sim.Results.CopyValues(snap,True,True,True,True,True,True,'','~')
    copied=profile(p)[1];trace=[]
    print('Repeat',repeat,'post-copy difference',(copied-expected)[[0,9,14]].tolist(),flush=True)
    for step in range(4):
     p.step();trace.append(profile(p)[1].tolist())
    print('First advance',trace[0][0]-copied[0],flush=True)
    row=dict(primed=primed.tolist(),copied=copied.tolist(),difference=(copied-expected).tolist(),trace=trace)
    rows.append(row);write(out/'trials.json',rows)
   for row in rows:
    np.testing.assert_allclose(np.asarray(row['copied'])[[0,9,14]],expected[[0,9,14]],atol=1e-8,rtol=0)
    assert abs(row['trace'][0][0]-expected[0]-1/6)<1.1e-6
   for row in rows[1:]:np.testing.assert_allclose(row['trace'],rows[0]['trace'],atol=1e-7,rtol=0)
   write(out/'result.json',dict(passed=True,repeats=3,steps_each=4,rows=rows))
  finally:p.close()
