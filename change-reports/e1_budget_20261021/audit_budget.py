"""Read-only reduced-E1 and protected-run acceptance evidence."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def audit(target):
    import run_column_e1_budget as recovery
    from column_review.common import process_identity,digest,owned_apps_remaining
    from revision_comparisons.progress import snapshot
    import run_remaining_recovery as original
    import run_column_e1 as scientific
    plan=recovery.verify(recovery.DEFAULT)
    import run_column_e1_budget_lazy_recovery as lazy
    lazy_plan=lazy.verify(lazy.DEFAULT) if (lazy.DEFAULT/'manifest.json').exists() else None
    current={job['id']:job for job in plan['jobs']}
    if lazy_plan:current.update({job['id']:job for job in lazy_plan['jobs']})
    protected=recovery.protected_check(plan)
    lazy_protected=recovery.protected_check(lazy_plan) if lazy_plan else []
    original.queue_guard()
    issues=[];jobs=[]
    for job in current.values():
        output=Path(job['directory']);status=recovery.read(output/'status.json',{})
        identity=process_identity(status.get('pid'));sampling=recovery.read(output/'capture/sampling.json',{})
        migration=recovery.read(output/'training_checkpoints/budget_migration.json',{})
        numeric=recovery.read(output/'budget_runtime.json',{})
        job_data=recovery.read(job['job'])
        if not (job['state']=='running' and status.get('state')=='running' and identity['alive'] is True):
            issues.append(job['id']+' not verified running')
        if not (sampling.get('protocol_sha256')==plan['protocol_sha256'] and sampling.get('options',{}).get('decisions_per_window')==20):
            issues.append(job['id']+' sampler migration pending or different')
        if not (migration.get('controller_unchanged') is True and migration.get('protocol_sha256')==plan['protocol_sha256']):
            issues.append(job['id']+' full-state migration not verified')
        if numeric.get('numpy')!='2.5.3':issues.append(job['id']+' numerical runtime not verified')
        if lazy_plan and job['category'] in ('weights','markov','residual'):
            if not (migration.get('lazy_recovery_protocol_sha256')==lazy_plan['protocol_sha256']
                    and migration.get('original_schedule_rebuilt_and_verified') is True
                    and numeric.get('lazy_recovery_protocol_sha256')==lazy_plan['protocol_sha256']):
                issues.append(job['id']+' lazy initialization recovery not verified')
        if job_data['sources']!=scientific.sources():issues.append(job['id']+' original scientific sources changed')
        selected=sampling.get('selected_steps',{})
        if sum(len(points) for points in selected.values())!=job['expected_matched_pairs']:
            issues.append(job['id']+' wrong sampling count')
        checkpoint=recovery.read(output/'training_checkpoints/latest.json')
        restored=recovery.read(output/'training_checkpoints/last_restore.json')
        restored_current=((output/'training_checkpoints/last_restore.json').stat().st_mtime
                          >=datetime.fromisoformat(status['started_at']).timestamp())
        if not restored_current:issues.append(job['id']+' current native/Python restore pending')
        elapsed=checkpoint['saved_at_unix']-(output/'training_checkpoints/last_restore.json').stat().st_mtime
        advanced=checkpoint['completed_steps']-restored['completed_steps']
        first_selected=min((step for points in selected.values() for step in points),default=0)
        timing=(dict(advances=advanced,elapsed_seconds=elapsed,seconds_per_advance=elapsed/advanced)
                if advanced>0 and elapsed>0 and checkpoint['completed_steps']<first_selected else None)
        jobs.append(dict(category=job['category'],case=job['aspen_number'],identity=identity,state=status.get('state'),
                         expected_pairs=job['expected_matched_pairs'],supplemental_pairs=job['supplemental_pairs'],
                         sampling=sampling,migration=migration,numeric=numeric,
                         restore=restored,restored_in_current_process=restored_current,
                         checkpoint=checkpoint,pre_capture_timing=timing,
                         apps=owned_apps_remaining(output/'owned_aspen_apps.json')))
    coordinator=process_identity(plan['queue_coordinator']['pid'])
    same=coordinator['alive'] is True and coordinator['created']==plan['queue_coordinator']['created']
    if not same:issues.append('Original combined coordinator identity differs')
    import psutil
    active_plan=lazy_plan or plan;active_directory=lazy.DEFAULT if lazy_plan else recovery.DEFAULT
    original_supervisor=process_identity(plan['supervisor']['pid'])
    if original_supervisor!=plan['supervisor']:issues.append('Original budget supervisor identity differs')
    supervisor=process_identity(active_plan['supervisor']['pid'])
    if supervisor!=active_plan['supervisor']:issues.append('Budget supervisor identity differs')
    broker=recovery.read(active_directory/'broker_identity.json',{})
    broker_identity=process_identity(broker.get('pid'))
    if not (broker_identity['alive'] is True and broker.get('parent_is_scheduler_service') is True):
        issues.append('Independent scheduler broker not verified')
    viewers=[]
    if broker_identity['alive'] is True:
        for child in psutil.Process(broker['pid']).children(recursive=True):
            try:command=child.cmdline()
            except (psutil.NoSuchProcess,psutil.AccessDenied):continue
            if '--watch' in command and any(arg.endswith('show_review_progress.py') for arg in command):
                viewers.append(dict(identity=process_identity(child.pid),command=command))
    if not viewers:issues.append('Live progress viewer not verified')
    paper=ROOT.parent/'Rl_assisted_paper/RL_assisted_paper'
    freeze=recovery.read(paper/'manuscript_revised/frozen_original_manifest.json')
    frozen=all(digest(paper/'manuscript'/item['path'])==item['sha256'] for item in freeze['files'])
    if not frozen:issues.append('Frozen original manuscript differs')
    result=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),issues=issues,protocol=recovery.read(recovery.DEFAULT/'protocol.json'),
                protocol_sha256=plan['protocol_sha256'],jobs=jobs,protected_jobs=protected,
                original_coordinator_preserved=same,coordinator=coordinator,progress=snapshot(),
                supervisor=supervisor,broker=broker,broker_identity=broker_identity,progress_viewers=viewers,
                original_budget_supervisor=original_supervisor,
                lazy_protected_jobs=lazy_protected,
                lazy_recovery_protocol=recovery.read(lazy.DEFAULT/'protocol.json') if lazy_plan else None,
                lazy_recovery_protocol_sha256=lazy_plan['protocol_sha256'] if lazy_plan else None,
                native_tests=[recovery.read(path) for directory in (recovery.DEFAULT,lazy.DEFAULT)
                              for path in sorted(directory.glob('validation_*.json'))],
                frozen_original=dict(verified=frozen,files=len(freeze['files'])))
    if target.exists():raise FileExistsError(target)
    recovery.write(target,result)
    print(json.dumps(dict(observed_at_utc=result['observed_at_utc'],issues=issues,expected_primary_pairs=sum(j['expected_pairs'] for j in jobs),
                          preserved_supplemental_pairs=sum(j['supplemental_pairs'] for j in jobs),protected_jobs=len(protected),
                          original_coordinator_preserved=same,frozen_original=result['frozen_original']),indent=2))
    return result


if __name__=='__main__':audit(Path(sys.argv[1]).resolve())
