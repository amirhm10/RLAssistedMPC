"""Start the author-approved one-time compression under the current user."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path(r'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\python.exe')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def launch():
    import win32api, win32con, win32security, win32com.client
    helper = HERE / 'compress_verified_archive.py'
    plan = HERE / 'compression_launch.json'
    assert not plan.exists()
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try: user = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
    finally: token.Close()
    service = win32com.client.Dispatch('Schedule.Service'); service.Connect()
    definition = service.NewTask(0)
    definition.RegistrationInfo.Description = 'Author-approved one-time verified lossless archive compression; originals retained.'
    definition.Principal.UserId = user; definition.Principal.LogonType = 3; definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.Enabled = settings.AllowDemandStart = True
    settings.ExecutionTimeLimit = 'PT0S'; settings.MultipleInstances = 2
    settings.DisallowStartIfOnBatteries = settings.StopIfGoingOnBatteries = False
    action = definition.Actions.Create(0)
    action.Path = str(BASE.with_name('pythonw.exe'))
    action.Arguments = subprocess.list2cmdline(['-B', str(Path(__file__)), '--run'])
    action.WorkingDirectory = str(ROOT)
    name = 'RLAssistedMPC-ArchiveCompression-' + uuid.uuid4().hex[:12]
    write(plan, dict(task_name=name, current_user=True, administrator_required=False, triggers=0,
                    helper=str(helper), helper_sha256=digest(helper), launcher_sha256=digest(Path(__file__)),
                    authorization='User selected Compress the full local archive (Recommended)',
                    originals_deleted=0))
    task = service.GetFolder('\\').RegisterTaskDefinition(name, definition, 2, user, None, 3)
    task.Run(None)
    print(json.dumps(dict(started=True, task_name=name, administrator_required=False)))


def run():
    sys.path.insert(0, str(ROOT))
    from review_restart_recovery import lifetime, isolated_interactive_context
    identity = lifetime(); assert isolated_interactive_context(identity)
    write(HERE / 'compression_identity.json', identity)
    plan = json.loads((HERE / 'compression_launch.json').read_text())
    helper = Path(plan['helper'])
    assert digest(helper) == plan['helper_sha256'] and digest(Path(__file__)) == plan['launcher_sha256']
    try:
        with (HERE / 'archive_compression.log').open('xb') as stream:
            result = subprocess.run([str(BASE), '-S', '-B', '-u', str(helper)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        write(HERE / 'compression_exit.json', dict(returncode=result.returncode, originals_deleted=0))
    except BaseException:
        write(HERE / 'compression_exit.json', dict(error=traceback.format_exc(), originals_deleted=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true')
    if parser.parse_args().run: run()
    else: launch()
