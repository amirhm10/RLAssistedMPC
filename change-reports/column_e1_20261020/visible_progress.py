"""Demand-only independent read-only progress window; never controls studies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',type=Path)
    parser.add_argument('--broker',type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT))
    if args.broker:
        from review_restart_recovery import lifetime,isolated_interactive_context
        identity=lifetime()
        assert isolated_interactive_context(identity)
        (args.broker/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        child=subprocess.Popen(f'cmd.exe /d /c call "{args.broker / "viewer.cmd"}"',cwd=ROOT,
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
        (args.broker/'visible.json').write_text(json.dumps({'cmd_pid':child.pid})+'\n')
        return child.wait()
    assert args.launch
    directory=args.launch.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    command=subprocess.list2cmdline([sys.executable,'-B','-u',str(ROOT/'show_review_progress.py'),'--watch','--interval','30'])
    (directory/'viewer.cmd').write_text('@echo off\nsetlocal\ntitle RL MPC all studies progress - read only\n'
        +f'cd /d "{ROOT}"\n{command}\n',encoding='utf-8')
    import win32api,win32con,win32security,win32com.client
    service=win32com.client.Dispatch('Schedule.Service')
    service.Connect()
    task=service.NewTask(0)
    token=win32security.OpenProcessToken(win32api.GetCurrentProcess(),win32con.TOKEN_QUERY)
    try:
        user=win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token,win32security.TokenUser)[0])
    finally:token.Close()
    task.Principal.UserId,task.Principal.LogonType,task.Principal.RunLevel=user,3,0
    task.Settings.Enabled=task.Settings.AllowDemandStart=True
    task.Settings.DisallowStartIfOnBatteries=task.Settings.StopIfGoingOnBatteries=False
    task.Settings.ExecutionTimeLimit,task.Settings.MultipleInstances='PT0S',2
    action=task.Actions.Create(0)
    action.Path=str(Path(sys.executable).with_name('pythonw.exe'))
    action.Arguments=subprocess.list2cmdline(['-B',str(Path(__file__).resolve()),'--broker',str(directory)])
    action.WorkingDirectory=str(ROOT)
    name='RLAssistedMPC-Progress-'+uuid.uuid4().hex[:12]
    registered=service.GetFolder('\\').RegisterTaskDefinition(name,task,2,user,None,3)
    (directory/'scheduled.json').write_text(json.dumps({'task_name':name,'triggers':0,'read_only':True})+'\n')
    registered.Run(None)
    return 0

if __name__=='__main__':raise SystemExit(main())
