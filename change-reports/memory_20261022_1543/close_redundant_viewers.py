"""Close only proved read-only duplicate progress consoles; preserve live jobs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from column_review.common import process_identity,owned_apps_remaining
from revision_comparisons.progress import snapshot
from recover_markov_seed23 import terminate_owned_identity


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(name,value):
    path=HERE/name
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    inventory=read(HERE/'before.json');processes={p['pid']:p for p in inventory['processes']}
    current=snapshot();workers=[r for r in current['jobs'] if r['study'] not in current['historical_studies'] and r['state']=='running']
    assert len(workers)==17
    protected={row['pid']:process_identity(row['pid']) for row in workers}
    native=[]
    for row in workers:
        native.extend(owned_apps_remaining(Path(row['directory'])/'owned_aspen_apps.json'))
    for item in native:
        assert item['current']=={key:item['recorded'][key] for key in ('pid','alive','created')}
        protected[item['current']['pid']]=item['current']
    actual={p['pid'] for p in processes.values() if p['name'].lower()=='aspenmodeler.exe'}
    assert actual=={item['current']['pid'] for item in native} and len(actual)==22
    supervisors=[]
    for folder in ('20261021_remaining_active','20261021_markov_checkpoint_a2','20261021_e1_budget20','20261022_e1_budget_lazy'):
        plan=read(ROOT/'Distillation/Results/state_recovery'/folder/'manifest.json')
        identity=process_identity(plan['supervisor']['pid']);assert identity==plan['supervisor']
        protected[identity['pid']]=identity;supervisors.append(identity)
        broker=read(ROOT/'Distillation/Results/state_recovery'/folder/'broker_identity.json')
        protected[broker['pid']]=process_identity(broker['pid'])
    queue=current['repeat_queue_coordinator'];protected[queue['pid']]=process_identity(queue['pid'])
    assert all(identity['alive'] is True for identity in protected.values())
    for pid in list(protected):
        while pid in processes and processes[pid]['parent'] in processes:
            parent=processes[pid]['parent']
            if parent in protected:break
            protected[parent]=process_identity(parent);pid=parent
    viewers=[]
    for process in processes.values():
        if process['name'].lower()!='cmd.exe':continue
        match=re.fullmatch(r'cmd\.exe /d /c call "([^"]+)"',process['command'],re.I)
        if not match:continue
        path=Path(match.group(1))
        if not path.is_relative_to(ROOT/'Distillation/Results') or path.name not in ('progress.cmd','viewer.cmd'):continue
        text=path.read_text(encoding='utf-8')
        assert 'show_review_progress.py' in text and '--watch' in text
        tree=[]
        def descend(parent,depth=0):
            children=[p for p in processes.values() if p['parent']==parent]
            for child in children:
                assert datetime.fromisoformat(child['created'])>=datetime.fromisoformat(process['created'])
                descend(child['pid'],depth+1);tree.append(dict(child,depth=depth+1))
        descend(process['pid']);tree.append(dict(process,depth=0))
        assert not ({p['pid'] for p in tree}&set(protected))
        for child in tree:
            assert child['name'].lower() in ('cmd.exe','conhost.exe','python.exe','pythonw.exe')
            if child['name'].lower().startswith('python'):
                assert str(ROOT/'show_review_progress.py') in child['command'] and '--watch' in child['command']
                assert all(flag not in child['command'].split() for flag in ('--worker','--supervise','--broker'))
        viewers.append(dict(root=process['pid'],path=str(path),tree=tree,created=process['created']))
    assert len(viewers)>=2
    keep=max(viewers,key=lambda viewer:viewer['created']);close=[v for v in viewers if v is not keep]
    for viewer in close:
        for process in viewer['tree']:
            process['identity']=process_identity(process['pid']);assert process['identity']['alive'] is True
    plan=dict(reason='Author requested closing unnecessary CMD windows',protected=protected,
              keep_current_viewer=keep['root'],close=close,aspen_count=len(actual),stale_aspen=[])
    save('console_cleanup_plan.json',plan)
    actions=[]
    for viewer in close:
        for process in viewer['tree']:
            identity=process['identity'];now=process_identity(identity['pid'])
            if now['alive'] is True:
                assert now==identity
                actions.append(dict(viewer=viewer['root'],pid=identity['pid'],result=terminate_owned_identity(identity)))
        for pid,identity in protected.items():assert process_identity(pid)==identity
    time.sleep(2)
    assert process_identity(keep['root'])['alive'] is True
    save('console_cleanup_result.json',dict(closed_viewer_windows=len(close),actions=actions,
          kept_viewer=keep['root'],all_protected_identities_unchanged=True,stale_aspen_count=0,
          finished_project_run_consoles=0,observed_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(closed_duplicate_progress_windows=len(close),kept_current_viewer=keep['root'],
                         protected_workers=len(workers),protected_aspen=len(actual),stale_aspen=0),indent=2))


if __name__=='__main__':main()
