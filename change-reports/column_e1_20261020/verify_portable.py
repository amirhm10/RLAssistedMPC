"""Replay a captured Markov decision in a fresh owned case without training."""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
import numpy as np
from column_review.common import read,write,record_owned_apps,owned_apps_remaining
from column_e1.runtime import DistillationColumnAspen,column_copying
from column_e1.replay import replay_snapshot
from run_column_e1 import sources,guard_queue
from run_distillation_phase2_repeats import aspen_file_lock,aspen_com_compatibility
import win32com.client

capture=ROOT/'Distillation/Results/column_e1/20261020_integration_a10/combined/a1/capture'
records=[json.loads(line) for line in (capture/'decisions.jsonl').read_text().splitlines()]
record=next(row for row in records if row['channel']=='markov')
output=ROOT/'Distillation/Results/column_e1/20261020_portable_replay_a1'
output.mkdir()
original=read(Path(__file__).with_name('preflight.json'))['candidate']
model=output/Path(original['aspen_path']).name
shutil.copyfile(original['aspen_path'],model)
guard_queue()
with aspen_file_lock(Path(original['aspen_path']),ROOT/'Distillation/Results/_aspen_file_locks'):
    with aspen_com_compatibility(),record_owned_apps(output),column_copying():
        plant=object.__new__(DistillationColumnAspen)
        plant.path=str(model)
        plant.delta_t=1/6
        plant.ad=win32com.client.DispatchEx('AD application')
        plant.ad.NewDocument()
        plant.ad.Visible=True
        plant._e1_document_open=True
        try:
            results=replay_snapshot(capture/record['snapshot'],plant,output/'replay',scientific_hashes=sources())
            assert all(row['state']=='completed' for row in results.values())
            differences={}
            with np.load(capture/record['trace']) as expected:
                for source in ('policy','supervisor'):
                    with np.load(output/'replay'/f'{source}.npz') as actual:
                        for field,atol in (('y',[1e-8,1e-4]),('u',1e-3),('reward',1e-5),('reference',1e-10)):
                            assert np.allclose(actual[field],expected[f'{source}__{field}'],rtol=1e-7,atol=atol),(source,field)
                        differences[source]=np.max(np.abs(actual['y']-expected[f'{source}__y']),axis=0).tolist()
            write(output/'result.json',{'passed':True,'channel':'markov','steps_each':4,'maximum_output_differences':differences})
            print('Fresh native/Python inference snapshot replays both original branches.',flush=True)
        finally:
            plant.close()
