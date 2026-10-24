"""Preserve only the confirmed failed Markov E1 run on its reserved case 15."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'change-reports/weights_pid_reuse_recovery_20261024'))
import prepare_recovery as preparation

if __name__ == '__main__':
    preparation.HERE = HERE
    preparation.CATEGORY = 'markov'
    preparation.CASE = 15
    preparation.EXPECTED_STEPS = [116000, 115600]
    records = preparation.hourly.completed_records(preparation.hourly.OUTPUTS['markov'] / 'capture/decisions.jsonl')
    preparation.EXPECTED_PAIRS = len(records)
    preparation.EXPECTED_ERROR = "(-2147417851, 'The server threw an exception.', None, None)"
    preparation.PEER_CATEGORIES = ('residual', 'combined')
    preparation.OUTPUT = preparation.hourly.OUTPUTS['markov']
    preparation.STATE = ROOT / 'Distillation/Results/state_recovery/20261024_markov_native_retry'
    preparation.BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\markov_native_recovery_20261024')
    preparation.prepare()
