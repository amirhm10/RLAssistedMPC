"""Accept only after the formerly stalled Markov pair completes in fresh workers."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recover_parallel as activation
import parallel_branches as p


def accept():
    plan = activation.r.verify()
    values = p.h.stable_batch({'pool':activation.r.STATE / 'parallel_pool_0.json'})
    pool = Path(values['pool']['directory'])
    decisions = p.h.completed_records(pool / 'decisions.jsonl')
    recovered = [r for r in decisions if r['channel'] == 'markov' and r['step'] == 118860]
    if not recovered or recovered[-1]['full_python_state_before'] != recovered[-1]['full_python_state_after']:
        raise RuntimeError('The formerly stalled Markov comparison has not yet verified')
    spec = spec_from_file_location('accepted_parallel_acceptance', ROOT / 'change-reports/combined_parallel_20261024/accept_parallel.py')
    original = module_from_spec(spec)
    spec.loader.exec_module(original)
    original.HERE = HERE
    original.activation = activation
    immutable = p.immutable

    def recorded(path, value):
        if Path(path) == HERE / 'acceptance.json':
            value = dict(value)
            value['authorized_healthy_combined_checkpoint_restarts'] = 0
            value['confirmed_failed_inference_recovery'] = True
            value['previous_worker_identity'] = plan['failed_worker']
            value['failure_evidence'] = plan['failure_evidence']
            value['formerly_stalled_markov_pair'] = recovered[-1]
            value['independent_preservation_manifest'] = plan['backup_manifest']
            value['independent_preservation_sha256'] = plan['backup_manifest_sha256']
        return immutable(path, value)

    with patch.object(p, 'immutable', recorded):
        original.accept()


if __name__ == '__main__':
    accept()
