"""Stop only the misbounded private technical test, with saved identity proof."""
import argparse
import ctypes
from ctypes import wintypes
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent

def windows():
    user=ctypes.WinDLL('user32',use_last_error=True)
    callback=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    rows=[]
    @callback
    def visit(handle,unused):
        size=user.GetWindowTextLengthW(handle)
        if size:
            title=ctypes.create_unicode_buffer(size+1);user.GetWindowTextW(handle,title,size+1)
            if 'C2S_SS_simulation3' in title.value and not any('C2S_SS_simulation'+str(n) in title.value for n in range(30,40)):
                pid=wintypes.DWORD();user.GetWindowThreadProcessId(handle,ctypes.byref(pid))
                rows.append(dict(pid=pid.value,title=title.value))
        return True
    user.EnumWindows(visit,0)
    return rows

def main(apply):
    from column_review.common import read,write,process_identity
    from run_distillation_phase2_repeats import aspen_file_lock
    from recover_markov_seed23 import terminate_owned_identity
    import run_markov_checkpoint_recovery as companion
    import psutil
    import win32com.client
    plan_path=OUT/'bounded_test_stop_plan.json'
    output=companion.POOL/'repetitions/markov_sg/s23/a1'
    worker=process_identity(54084)
    source=read(output/'native_working_case.json')
    state=read(output/'worker_status.json')
    index=read(output/'training_checkpoints/latest.json')
    assert read(companion.DEFAULT/'manifest.json')['state']=='prepared'
    assert not (companion.DEFAULT/'scheduled.json').exists()
    assert index['completed_steps']==8 and state['pid']==worker['pid'] and worker['alive']
    process=psutil.Process(worker['pid'])
    assert 'check_markov_native_restart.py' in ' '.join(process.cmdline()) and process.cmdline()[-1]=='16'
    assert source['canonical_case'].endswith('C2S_SS_simulation3.dynf')
    matches=[row for row in windows() if psutil.Process(row['pid']).create_time()>process.create_time()
             and psutil.Process(row['pid']).name().lower()=='aspenmodeler.exe']
    assert len(matches)==1,matches
    app=dict(process_identity(matches[0]['pid']),name='AspenModeler.exe',path=psutil.Process(matches[0]['pid']).exe(),title=matches[0]['title'])
    assert app['alive'] and 'AspenTech' in app['path']
    helper_path=ROOT/'change-reports/run_audit_20261021/clear_verified_orphans.py'
    spec=importlib.util.spec_from_file_location('owned_descendants',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    descendants=helper.descendants(win32com.client.GetObject('winmgmts:'),app)
    plan=dict(worker=worker,command=process.cmdline(),case=source,index_sha256=index['sha256'],
              app=app,descendants=descendants,protected_active_recovery=read(companion.POOL/'manifest.json')['jobs'])
    if not apply:
        write(plan_path,plan);print(json.dumps(plan,indent=2));return
    saved=read(plan_path)
    assert saved['worker']==worker and saved['app']==app and saved['descendants']==descendants
    actions=[dict(pid=worker['pid'],result=terminate_owned_identity(worker))]
    with aspen_file_lock(Path(source['canonical_case']),ROOT/'Distillation/Results/_aspen_file_locks'):
        for record in [*descendants,app]:
            actions.append(dict(pid=record['pid'],result=terminate_owned_identity(record)))
    assert read(output/'training_checkpoints/latest.json')['sha256']==saved['index_sha256']
    write(OUT/'bounded_test_stop_applied.json',dict(actions=actions,checkpoint_retained=True))
    print(json.dumps(actions,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');main(parser.parse_args().apply)
