"""Correct one proven stale case-11 ownership record without signaling a process."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from column_review.common import process_identity

HERE=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

proof=read(HERE/'ownership_candidate.json');audit=read(HERE/'audit.json')
assert (datetime.now(timezone.utc)-datetime.fromisoformat(proof['observed_at_utc'])).total_seconds()<120
assert proof['native']['title']=='C2S_SS_simulation11.dynf - Aspen Plus Dynamics V14 - aspenONE'
assert proof['native']['executable']==r'C:\Program Files\AspenTech\AMSystem V14.0\Bin\AspenModeler.exe'
worker=process_identity(proof['worker']['pid']);native=process_identity(proof['native']['pid'])
assert worker==dict(proof['worker'],alive=True) and native==dict(pid=proof['native']['pid'],created=proof['native']['created'],alive=True)
assert native['created']>worker['created'] and native['created']-worker['created']<10*10**7
target=ROOT/'Distillation/Results/state_recovery/20261021_remaining_active/repetitions/residual_plain/s23/a1'
assert read(target/'worker_status.json')['pid']==worker['pid']
working=read(target/'native_working_case.json')
assert Path(working['canonical_case']).name==Path(working['private_model']).name=='C2S_SS_simulation11.dynf'
old=read(target/'owned_aspen_apps.json');assert len(old)==1 and process_identity(old[0]['pid'])['alive'] is False
known={a['current']['pid'] for row in audit['workers'] for a in row['owned_native'] if 'current' in a}
resources=read(HERE/'resources_final.json')
assert {p['ProcessId'] for p in resources['aspen']}-known=={native['pid']}
preserved={row['pid']:process_identity(row['pid']) for row in audit['workers']}
assert all(identity==row['identity'] for row in audit['workers'] for pid,identity in preserved.items() if pid==row['pid'])
backup=HERE/'residual_original_owned_aspen_apps.json';assert not backup.exists()
backup.write_bytes((target/'owned_aspen_apps.json').read_bytes())
new=[dict(native,created_at=proof['observed_at_utc'],reconciled_at=proof['observed_at_utc'],
          evidence=str(HERE/'ownership_candidate.json'),reason='Previous record belonged to the completed technical process')]
temporary=target/'owned_aspen_apps.reconciled.tmp'
temporary.write_text(json.dumps(new,indent=2)+'\n',encoding='utf-8');temporary.replace(target/'owned_aspen_apps.json')
after={pid:process_identity(pid) for pid in preserved};assert after==preserved
(HERE/'ownership_reconciled.json').write_text(json.dumps(dict(proof=proof,previous=old,current=new,workers_unchanged=True,
    protected_workers=after,scientific_sources_changed=False,processes_signaled=0),indent=2)+'\n',encoding='utf-8')
print('Corrected only the residual case-11 ownership record; all 17 workers retain their identities.')
