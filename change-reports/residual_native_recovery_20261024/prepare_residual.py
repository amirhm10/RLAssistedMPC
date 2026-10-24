"""Preserve the confirmed failed residual worker; no simulator/process action."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'change-reports/weights_pid_reuse_recovery_20261024'))
import prepare_recovery as preparation

if __name__ == '__main__':
    preparation.HERE = HERE
    preparation.CATEGORY = 'residual'
    preparation.CASE = 17
    preparation.EXPECTED_STEPS = [117600,117200]
    preparation.EXPECTED_PAIRS = 69
    preparation.EXPECTED_ERROR = "(-2147417851, 'The server threw an exception.', None, None)"
    preparation.PEER_CATEGORIES = ('weights','combined')
    preparation.OUTPUT = preparation.hourly.OUTPUTS['residual']
    preparation.STATE = ROOT/'Distillation/Results/state_recovery/20261024_residual_native_retry'
    preparation.BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\residual_native_recovery_20261024')
    preparation.prepare()
