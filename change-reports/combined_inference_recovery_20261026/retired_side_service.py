"""Apply the unchanged verified side-archive algorithm to the fresh pool only."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / 'change-reports/combined_parallel_20261024/retired_side_archive.py'
sys.path.insert(0, str(ORIGINAL.parent))
import parallel_branches as p
spec = spec_from_file_location('verified_side_archive', ORIGINAL)
archive = module_from_spec(spec)
spec.loader.exec_module(archive)
archive.HERE = HERE
archive.STATE = ROOT / 'Distillation/Results/state_recovery/20261026_combined_inference'
archive.ARCHIVE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\combined_inference_recovery_20261026\retired_branch_zips')


def launch():
    if not p.closed_json(HERE / 'acceptance.json')['passed']:
        raise RuntimeError('Accepted production recovery is required first')
    state = archive.STATE / 'retired_side_retention'
    if (state / 'identity.json').exists() or (state / 'scheduled.json').exists():
        raise RuntimeError('Never duplicate an existing archive service')
    if p.h.digest(ORIGINAL) != '7abdfc436206e968e3696e56b13ed5ba6a7634a02c92bccaa0a23d9809d861df':
        raise RuntimeError('The tested archival algorithm changed')
    import win32api
    import win32con
    import win32security
    import win32com.client
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally:
        token.Close()
    scheduler = win32com.client.Dispatch('Schedule.Service')
    scheduler.Connect()
    definition = scheduler.NewTask(0)
    definition.Principal.UserId = user
    definition.Principal.LogonType = 3
    definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.ExecutionTimeLimit = 'PT0S'
    settings.MultipleInstances = 2
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    action = definition.Actions.Create(0)
    action.Path = r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\pythonw.exe'
    action.Arguments = subprocess.list2cmdline(['-B', str(Path(__file__).resolve()), '--service'])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-CombinedRecoveredSideArchive-20261026'
    scheduler.GetFolder('\\').RegisterTaskDefinition(name, definition, 2, user, None, 3)
    p.immutable(state / 'scheduled.json', dict(task_name=name, administrator_required=False,
        execution_time_limit='PT0S', at_utc=p.now(), source_sha256=p.h.digest(ORIGINAL),
        wrapper=str(Path(__file__)), wrapper_sha256=p.h.digest(Path(__file__))))
    scheduler.GetFolder('\\').GetTask(name).Run(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--service', action='store_true')
    group.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        try:
            archive.service()
        except BaseException:
            p.h.atomic_json(archive.STATE / 'retired_side_retention/status.json', dict(state='failed',
                at_utc=p.now(), error=__import__('traceback').format_exc(), existing_archives_preserved=True))
            raise
