"""Reserved-case-17 entrypoint for the isolated failed-run recovery engine."""
import argparse
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'change-reports/weights_pid_reuse_recovery_20261024'))
import recover_weights as engine

engine.HERE = HERE
engine.CATEGORY = 'residual'
engine.CASE = 17
engine.SCRIPT = Path(__file__).resolve()
engine.OUTPUT = ROOT/'Distillation/Results/state_recovery/20261021_remaining_active/e1/residual/a1'
engine.STATE = ROOT/'Distillation/Results/state_recovery/20261024_residual_native_retry'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('launch','supervise','worker'):
        group.add_argument('--'+name,action='store_true')
    parser.add_argument('--validation-steps',type=int,default=0)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(engine.worker(args.validation_steps))
    elif args.supervise:
        try:
            engine.supervise()
        except BaseException:
            engine.write(engine.STATE/'supervisor_error.json',dict(at_utc=engine.now(),traceback=traceback.format_exc()))
            raise
    else:
        engine.launch()
