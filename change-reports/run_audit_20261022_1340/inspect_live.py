"""Read-only live-run audit using standard-library imports and no Torch runtime."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PAPER=ROOT.parent/'Rl_assisted_paper/RL_assisted_paper'
sys.path.insert(0,str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def audit(target):
    from column_review.common import process_identity,owned_apps_remaining
    from revision_comparisons.progress import snapshot
    progress=snapshot();issues=[];warnings=[];cache={}
    def digest(path):
        path=Path(path).resolve()
        if path not in cache:
            result=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
            cache[path]=result.hexdigest()
        return cache[path]
    def hashes(mapping,label):
        for name,expected in mapping.items():
            path=ROOT/name
            if not path.is_file() or digest(path)!=expected:issues.append(label+': '+name+' differs')
    baseline=read(HERE/'cpu_first.json');resources=read(HERE/'resources_final.json')
    first={p['Id']:p for p in baseline['processes']};last={p['Id']:p for p in resources['processes']}
    live=[r for r in progress['jobs'] if r['study'] not in progress['historical_studies'] and r['state']=='running']
    unexpected=[r for r in progress['jobs'] if r['study'] not in progress['historical_studies'] and r['state'] in ('failed','stopped_without_final_status')]
    issues.extend(r['study']+'/'+r['job']+': '+r['state'] for r in unexpected)
    owned_ids=set();workers=[]
    for row in live:
        out=Path(row['directory']);identity=process_identity(row['pid'])
        owned=owned_apps_remaining(out/'owned_aspen_apps.json')
        if not identity['alive']:issues.append(row['job']+': process not verified alive')
        if len(owned)!=(2 if row['study']=='E1_column' else 1):issues.append(row['job']+': native ownership count differs')
        for item in owned:
            if item.get('recorded') and item['current']=={key:item['recorded'][key] for key in ('pid','alive','created')}:
                owned_ids.add(item['current']['pid'])
            else:issues.append(row['job']+': native process creation identity differs')
        pids=[row['pid']]+[a['current']['pid'] for a in owned if 'current' in a]
        activity=sum((last[pid]['CPU'] or 0)-(first[pid]['CPU'] or 0) for pid in pids if pid in first and pid in last)
        if activity<=0:warnings.append(row['job']+': no CPU increase in the short sample')
        status_path=out/('worker_status.json' if row['study']=='column_repeats' else 'status.json')
        status=read(status_path)
        checkpoint=read(out/'training_checkpoints/latest.json') if (out/'training_checkpoints/latest.json').exists() else None
        workers.append(dict(row,identity=identity,status=status['state'],owned_native=owned,
                            sampled_cpu_increase_seconds=activity,
                            checkpoint_steps=checkpoint.get('completed_steps') if checkpoint else None,
                            log_age_seconds=(datetime.now().timestamp()-datetime.fromisoformat(row['last_log_at']).timestamp()) if row.get('last_log_at') else None))
    actual_aspen={p['ProcessId'] for p in resources['aspen']}
    if owned_ids!=actual_aspen:issues.append('Aspen inventory differs from recorded ownership')
    e1_base=ROOT/'Distillation/Results/state_recovery/20261021_e1_budget20'
    e1=read(e1_base/'manifest.json');lazy_dir=ROOT/'Distillation/Results/state_recovery/20261022_e1_budget_lazy'
    lazy=read(lazy_dir/'manifest.json');pairs=[]
    for folder,plan in ((e1_base,e1),(lazy_dir,lazy)):
        hashes(plan['bootstrap_sha256'],str(folder.name))
        if digest(folder/'protocol.json')!=plan['protocol_sha256']:issues.append(folder.name+': protocol changed')
    for job in e1['jobs']:
        out=Path(job['directory']);definition=read(job['job']);hashes(definition['sources'],job['id'])
        if digest(job['job'])!=job['job_sha256']:issues.append(job['id']+': job metadata changed')
        rows=[json.loads(line) for line in (out/'capture/decisions.jsonl').read_text().splitlines() if line.strip()]
        counts=dict(Counter(r['status'] for r in rows));sampling=read(out/'capture/sampling.json')
        if counts.get('completed',0)!=len(rows):issues.append(job['id']+': failed/incomplete matched pairs')
        if any(r['budget_protocol_sha256']!=e1['protocol_sha256'] for r in rows):issues.append(job['id']+': record protocol differs')
        if sampling['options']['decisions_per_window']!=20:issues.append(job['id']+': reduced budget differs')
        if len({(r['channel'],r['step']) for r in rows})!=len(rows):issues.append(job['id']+': duplicated matched pairs')
        pairs.append(dict(category=job['category'],expected=job['expected_matched_pairs'],recorded=len(rows),states=counts,windows=dict(Counter(r['window'] for r in rows))))
    threshold=ROOT/'Distillation/Results/state_recovery/20261021_remaining_active/threshold/a1'
    status=read(threshold/'status.json');policy=read(threshold/'policy_checkpoints/latest.json')
    final_policy=status['checkpoint'];export_valid=(digest(threshold/'training.pkl')==status['trajectory_sha256']
          and digest(threshold/final_policy['path'])==final_policy['sha256']
          and digest(threshold/'policy_checkpoints'/policy['path'])==policy['sha256']
          and final_policy['round_trip_verified'] and len(final_policy['parameter_elements'])==4
          and all(n>0 for n in final_policy['parameter_elements'].values()))
    if status['state']!='completed' or policy['completed_cycles']!=200 or not export_valid:issues.append('Threshold completion artifacts did not verify')
    combined=ROOT/'Distillation/Results/review_recovery/20261020_queue_case12/combined/manifest.json'
    queue_manifest=read(combined);hashes(queue_manifest['source_hashes'],'combined sources');hashes(queue_manifest['bootstrap_sha256'],'combined bootstrap')
    coordinator=progress['repeat_queue_coordinator'];expected_coordinator=e1['queue_coordinator']
    if process_identity(coordinator['pid'])!={k:expected_coordinator[k] for k in ('pid','alive','created')}:issues.append('Combined coordinator identity differs')
    supervisors=[]
    for folder in ('20261021_remaining_active','20261021_markov_checkpoint_a2','20261021_e1_budget20','20261022_e1_budget_lazy'):
        plan=read(ROOT/'Distillation/Results/state_recovery'/folder/'manifest.json')
        current=process_identity(plan['supervisor']['pid'])
        if current!=plan['supervisor']:issues.append(folder+': supervisor identity differs')
        supervisors.append(dict(pool=folder,identity=current))
    memory=resources['memory'];headroom=memory['CommitLimit']-memory['CommittedBytes']
    if headroom<2*1024**3:warnings.append('Windows memory commitment near limit; no system setting or worker changed')
    crashes=[event for event in resources['events'] if 'APPCRASH' in event['Message'] or event['Id']==1000]
    if crashes:issues.append('New Python/Aspen crash events after the previous verified audit')
    freeze=read(PAPER/'manuscript_revised/frozen_original_manifest.json')
    frozen=all(digest(PAPER/'manuscript'/entry['path'])==entry['sha256'] for entry in freeze['files'])
    if not frozen:issues.append('Frozen original manuscript changed')
    result=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),issues=issues,warnings=warnings,
        counts=progress['counts'],historical_studies=progress['historical_studies'],workers=workers,
        primary_pairs=pairs,primary_completed=sum(p['states'].get('completed',0) for p in pairs),primary_expected=640,
        threshold=dict(status=status,policy=policy,final_artifacts_hash_verified=export_valid),
        queue=coordinator,supervisors=supervisors,aspen_inventory=dict(count=len(actual_aspen),all_owned=actual_aspen==owned_ids),
        activity_window=dict(start=baseline['observed_at'],end=resources['observed_at']),
        memory=dict(resources['memory'],headroom_bytes=headroom,automatic_pagefile=resources['automatic_pagefile']),
        crash_events=crashes,resource_events=resources['events'],frozen_original=dict(verified=frozen,files=len(freeze['files'])),
        remaining_unlaunched_scope='Frozen-policy unseen-reference follow-ups and separately scoped E4/E5 remain outside the current active queue',
        inspected=['live progress/current manifests','worker status/checkpoint indexes and source hashes','owned Aspen identities and sampled CPU','E1 primary records/sampling/protocols','threshold final trajectory/policy hashes','Windows resources and events','frozen original manuscript manifest'])
    if (HERE/'ownership_reconciled.json').exists():result['ownership_reconciliation']=read(HERE/'ownership_reconciled.json')
    save(target,result)
    print(json.dumps(dict(observed_at_utc=result['observed_at_utc'],issues=issues,warnings=warnings,active_workers=len(workers),
                         primary_completed=result['primary_completed'],threshold_completed=status['state'],all_aspen_owned=result['aspen_inventory']),indent=2))


if __name__=='__main__':audit(Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'audit.json')
