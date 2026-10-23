"""Read-only production/preservation audit and an optional offline source bundle."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT.parent / 'Rl_assisted_paper/RL_assisted_paper'
sys.path.insert(0, str(ROOT))
from column_review.common import process_identity, owned_apps_remaining
from revision_comparisons.progress import snapshot
import run_column_e1_process_recovery as recovery


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def offline_sources(plan):
    backup = recovery.BACKUP
    files = dict(plan['bootstrap_sha256'])
    for job in plan['jobs']:
        files.update({name: sha for name, sha in job['initial_checkpoint']['source_hashes'].items()
                      if (ROOT / name).is_file()})
    package = backup / 'scientific_sources_and_protocols.zip'
    assert not package.exists()
    with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, sha in files.items():
            assert digest(ROOT / name) == sha, name
            archive.write(ROOT / name, 'sources/' + name)
        for name in ('manifest.json', 'protocol.json', 'supervisor_lifecycle.json'):
            archive.write(recovery.DEFAULT / name, 'recovery/' + name)
        archive.write(HERE / 'e1_supervisor_lifecycle.py', 'recovery/e1_supervisor_lifecycle.py')
        for name in ('partial_results_audit.json', 'failed_e1_offline_preservation.json', 'local_preservation_verified.json'):
            archive.write(HERE / name, 'preservation/' + name)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        for name, sha in files.items():
            assert hashlib.sha256(archive.read('sources/' + name)).hexdigest() == sha
    reviewer = HERE / 'partial_analysis/reviewer_data_200_pairs.zip'
    copy = backup / reviewer.name
    assert not copy.exists()
    shutil.copyfile(reviewer, copy)
    assert digest(copy) == digest(reviewer)
    with zipfile.ZipFile(copy) as archive:
        assert archive.testzip() is None
    receipt = dict(verified=True, at_utc=datetime.now(timezone.utc).isoformat(),
                   source_files=len(files), source_package=str(package), source_package_sha256=digest(package),
                   reviewer_package=str(copy), reviewer_package_sha256=digest(copy))
    save(backup / 'evidence_preservation.json', receipt)
    save(HERE / 'evidence_preservation.json', receipt)


def main():
    plan = recovery.verify(recovery.DEFAULT)
    recovery.protected_check(plan)
    assert digest(HERE / 'e1_supervisor_lifecycle.py') == read(recovery.DEFAULT / 'supervisor_lifecycle.json')['sha256']
    if '--copy-evidence' in sys.argv:
        offline_sources(plan)
    before = read(HERE / 'before.json')
    progress = snapshot()
    rows = [r for r in progress['jobs'] if r['study'] not in progress['historical_studies']]
    current = {r['pid']: r for r in rows if r.get('pid')}
    activity = []
    for old in before['jobs']:
        if old['state'] != 'running':
            continue
        observed = current.get(old['pid'])
        assert observed is not None, old['pid']
        assert observed['state'] in ('running', 'completed')
        if observed['state'] == 'running':
            assert observed['alive'] is True and process_identity(old['pid']) == old['identity']
            delta = observed['cpu_seconds'] - old['cpu_seconds']
            assert delta > 0, old['pid']
            activity.append(dict(job=old['job'], pid=old['pid'], python_cpu_delta_seconds=delta))
    sources = read(HERE / 'partial_analysis/analysis_manifest.json')['sources']
    pairs = []
    for source in sources:
        records = [json.loads(line) for line in (Path(source['directory']) / 'decisions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        keys = [(r['category'], r['channel'], r['step']) for r in records]
        assert len(keys) == len(set(keys)) and all(r['status'] == 'completed' for r in records)
        pairs.append(dict(category=source['category'], pairs=len(records)))
    audit = read(HERE / 'partial_results_audit.json')
    for trace in audit['trace_artifacts']:
        assert digest(trace['path']) == trace['sha256']
    frozen = read(PAPER / 'manuscript_revised/frozen_original_manifest.json')
    assert len(frozen['files']) == 46
    assert all(digest(PAPER / 'manuscript' / row['path']) == row['sha256'] for row in frozen['files'])
    owned = {}
    for row in rows:
        if row['state'] != 'running':
            continue
        for app in owned_apps_remaining(Path(row['directory']) / 'owned_aspen_apps.json'):
            assert 'recorded' in app and app['current']['alive'] is True
            assert all(app['current'][key] == app['recorded'][key] for key in ('pid', 'created'))
            owned[app['recorded']['pid']] = dict(job=row['job'], identity=app['current'])
    inventory = read(HERE / 'aspen_inventory.json')
    assert {int(row['pid']) for row in inventory['aspen']} == set(owned), 'Native inventory differs from live ownership'
    result = dict(at_utc=datetime.now(timezone.utc).isoformat(), current_state_counts=dict(Counter(r['state'] for r in rows)),
                  jobs=rows, counts={k: v for k, v in progress['counts'].items() if k not in progress['historical_studies']},
                  original_worker_activity=activity, protected_identities=plan['protected'],
                  protected_identities_verified=True, queue=progress['repeat_queue_coordinator'],
                  pairs=pairs, primary_pairs=sum(r['pairs'] for r in pairs), planned_pairs=640,
                  original_200_trace_hashes_unchanged=True, owned_aspen=owned,
                  aspen_inventory=inventory, unmatched_aspen=[],
                  recovery_manifest=dict(state=plan['state'], supervisor=plan.get('supervisor'),
                                         jobs=[dict(category=j['category'], state=j['state'], pid=j.get('pid')) for j in plan['jobs']]),
                  memory=recovery.memory(), archive_preservation=read(HERE / 'local_preservation_verified.json'),
                  compression=read(Path(r'C:\Users\hamediaa\RL_MPC_local_archive\compressed_storage_20261023\status.json')),
                  frozen_original_verified=True, frozen_original_files=46, issues=[])
    save(HERE / 'recovery_assessment.json', result)
    print(json.dumps(dict(at_utc=result['at_utc'], current_state_counts=result['current_state_counts'], pairs=result['pairs'],
                          originals_verified=True, all_original_worker_cpu_positive=True, memory=result['memory'])), flush=True)


if __name__ == '__main__':
    main()
