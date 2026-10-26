"""Fresh-process recovery of the verified stalled combined inference attempt."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACCEPTED = ROOT / 'change-reports/combined_parallel_20261024'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ACCEPTED))
spec = spec_from_file_location('accepted_parallel_recovery', ACCEPTED / 'recover_parallel.py')
execution = module_from_spec(spec)
spec.loader.exec_module(execution)
r = execution.r
r.HERE = execution.HERE = HERE
r.SCRIPT = Path(__file__).resolve()
r.STATE = ROOT / 'Distillation/Results/state_recovery/20261026_combined_inference'


def scope(plan):
    expected = ('combined', 16, 7, 320, 20, 300)
    actual = tuple(plan[k] for k in ('category', 'case', 'seed', 'expected_matched_pairs',
                                    'decisions_per_window', 'main_training_cycles'))
    if actual != expected or plan['authorization'] != 'confirmed_failed_run_recovery':
        raise RuntimeError('Only the preserved stalled combined attempt may resume')
    if plan['failed_worker'] != plan['failure_evidence']['parent_identity']:
        raise RuntimeError('Stall evidence and retired parent generations differ')
    if not plan['failure_evidence']['unchanged_inference_stacks_and_io_verified']:
        raise RuntimeError('Actual stalled inference evidence is required')


r.scope = scope


def launch():
    plan = r.verify()
    r.protected_check(plan)
    r.start_guard(plan)
    if (r.STATE / 'scheduled.json').exists() or (r.STATE / 'manifest.json').exists():
        raise RuntimeError('Do not duplicate a live or memory-gated recovery')
    import win32api
    import win32con
    import win32security
    import win32com.client
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    service = win32com.client.Dispatch('Schedule.Service')
    service.Connect()
    definition = service.NewTask(0)
    definition.Principal.UserId = user
    definition.Principal.LogonType = 3
    definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.ExecutionTimeLimit = 'PT0S'
    settings.MultipleInstances = 2
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    action = definition.Actions.Create(0)
    action.Path = str(r.BASE_PYTHON)
    action.Arguments = subprocess.list2cmdline(['-B', str(r.SCRIPT), '--supervise'])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-CombinedInferenceRecovery-20261026'
    service.GetFolder('\\').RegisterTaskDefinition(name, definition, 2, user, None, 3)
    r.write(r.STATE / 'scheduled.json', dict(task_name=name, administrator_required=False,
            execution_time_limit='PT0S', triggers=0, launch_at_utc=r.now(), cases=[15, 17]))
    service.GetFolder('\\').GetTask(name).Run(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + name, action='store_true')
    parser.add_argument('--validation-steps', type=int, choices=(0, 8), default=0)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(execution.worker(args.validation_steps))
    elif args.supervise:
        try:
            execution.supervise()
        except BaseException:
            r.write(r.STATE / 'supervisor_error.json', dict(at_utc=r.now(), traceback=__import__('traceback').format_exc()))
            raise
    else:
        launch()
