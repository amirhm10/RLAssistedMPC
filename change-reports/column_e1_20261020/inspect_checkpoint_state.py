"""Check complete Python serialization and native-cache alias preservation offline."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
from column_review.common import read,write
from column_e1.runtime import DistillationColumnAspen,digest_state,integrity_payload
from column_e1.checkpoints import restoring_into
from revision_recovery.checkpoints import load_payload,CheckpointRunner
from utils.combined_runner import run_combined_supervisor

directory=ROOT/'Distillation/Results/column_e1/20261020_integration_a10/combined/a1/training_checkpoints'
index=read(directory/'latest.json')
compiled=CheckpointRunner(run_combined_supervisor,'combined')
environment=dict(compiled.original.__globals__)
plant=object.__new__(DistillationColumnAspen)
plant.path='offline_probe.dynf'
with restoring_into(plant): payload=load_payload(directory/index['file'],environment)
environment.update(payload['environment'])
expected=payload['technical_python_state_sha256']
actual=digest_state(integrity_payload(environment,compiled))
aliases={field:[name for name,value in environment.items() if name in compiled.local_names and value is getattr(plant,field)]
         for field in ('current_input','current_output')}
for field in aliases:
    setattr(plant,field,payload['native']['python_state'][field].copy())
changed=digest_state(integrity_payload(environment,compiled))
result={'source':str(directory/index['file']),'python_round_trip_exact':actual==expected,'cache_replacement_preserves_complete_state':changed==expected,
        'aliases':aliases,'expected':expected,'actual':actual,'after_cache_replacement':changed}
write(Path(__file__).with_name('python_checkpoint_inspection.json'),result)
print(result)
