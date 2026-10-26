"""Load this recovery's acceptance helper by exact path, avoiding legacy name collisions."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recover_production as activation
import parallel_branches as p

source = HERE / 'accept_recovery.py'
spec = spec_from_file_location('current_combined_recovery_acceptance', source)
accepted = module_from_spec(spec)
spec.loader.exec_module(accepted)
accepted.activation = activation


def accept():
    expected = HERE.parents[1] / 'Distillation/Results/state_recovery/20261026_combined_metadata_retry'
    if accepted.HERE != HERE or activation.r.STATE != expected:
        raise RuntimeError('Acceptance resolved to a legacy helper/state')
    proof = p.closed_json(HERE / 'fresh_production_restore.json')
    identity = p.closed_json(expected / 'worker_0_identity.json')['identity']
    if not proof['passed'] or proof['current_generation'] != identity:
        raise RuntimeError('The current generation needs its own verified cold restore')
    accepted.accept()


if __name__ == '__main__':
    accept()
