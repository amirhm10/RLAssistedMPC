"""Late full-state residual recovery, with bounded owned document-open retry."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'change-reports/weights_pid_reuse_recovery_20261024'))
sys.path.insert(0,str(HERE))
import recover_weights as engine

engine.HERE=HERE
engine.CATEGORY='residual'
engine.CASE=17
engine.SCRIPT=Path(__file__).resolve()
engine.OUTPUT=ROOT/'Distillation/Results/state_recovery/20261021_remaining_active/e1/residual/a1'
engine.STATE=ROOT/'Distillation/Results/state_recovery/20261024_residual_native_retry2'
TASK='RLAssistedMPC-ResidualNativeRetry2-20261024'


def worker(steps):
    if steps not in (0,8):raise ValueError('Only eight-step validation or full continuation is permitted')
    # These imports are standard-library-only. The existing worker performs
    # every scope/source/process/memory check before numerical/native work.
    import column_e1_process_recovery as fresh
    import document_open_retry
    with document_open_retry.installed(fresh):return engine.worker(steps)


def launch():
    plan=engine.verify()
    engine.protected_check(plan)
    if engine.process_identity(plan['failed_worker']['pid'])['alive'] is not False or engine.owned_remaining():
        raise RuntimeError('The failed residual worker and its applications must be retired')
    if (engine.STATE/'scheduled.json').exists() or (engine.STATE/'manifest.json').exists():
        raise RuntimeError('Inspect an existing launch; never duplicate it')
    import win32api,win32con,win32security,win32com.client
    token=win32security.OpenProcessToken(win32api.GetCurrentProcess(),win32con.TOKEN_QUERY)
    try:user=win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token,win32security.TokenUser)[0])
    finally:token.Close()
    service=win32com.client.Dispatch('Schedule.Service');service.Connect()
    definition=service.NewTask(0)
    definition.Principal.UserId=user
    definition.Principal.LogonType=3
    definition.Principal.RunLevel=0
    settings=definition.Settings
    settings.Enabled=settings.AllowDemandStart=True
    settings.ExecutionTimeLimit='PT0S';settings.MultipleInstances=2
    settings.DisallowStartIfOnBatteries=settings.StopIfGoingOnBatteries=False
    action=definition.Actions.Create(0)
    action.Path=str(engine.BASE_PYTHON)
    action.Arguments=subprocess.list2cmdline(['-B',str(engine.SCRIPT),'--supervise'])
    action.WorkingDirectory=str(ROOT)
    service.GetFolder('\\').RegisterTaskDefinition(TASK,definition,2,user,None,3)
    engine.write(engine.STATE/'scheduled.json',dict(task_name=TASK,administrator_required=False,
        execution_time_limit='PT0S',triggers=0,launch_at_utc=engine.now()))
    service.GetFolder('\\').GetTask(TASK).Run(None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('launch','supervise','worker'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--validation-steps',type=int,default=0)
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.validation_steps))
    elif args.supervise:
        try:engine.supervise()
        except BaseException:
            engine.write(engine.STATE/'supervisor_error.json',dict(at_utc=engine.now(),traceback=traceback.format_exc()))
            raise
    else:launch()
