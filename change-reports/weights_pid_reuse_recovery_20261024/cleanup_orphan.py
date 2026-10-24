"""Close only the backed-up failed weights worker's exact owned orphan."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_weights as recovery
from recover_markov_seed23 import terminate_owned_identity
from run_distillation_phase2_repeats import aspen_file_lock


def cleanup():
    plan = recovery.verify()
    recovery.protected_check(plan)
    if (recovery.HERE / 'orphan_cleanup.json').exists():
        raise FileExistsError('Inspect the existing cleanup receipt; never repeat a stale target')
    status = recovery.read(recovery.OUTPUT / 'status.json')
    if (status['state'] != 'failed' or status['pid'] != plan['failed_worker']['pid']
            or recovery.process_identity(status['pid'])['alive'] is not False):
        raise RuntimeError('Weights is not the retired failed worker')
    preservation = json.loads(Path(plan['backup_manifest']).read_text(encoding='utf-8'))
    for row in preservation['files']:
        if recovery.digest(Path(row['local_copy'])) != row['sha256']:
            raise RuntimeError('Full recovery backup no longer verifies')
    unresolved = recovery.owned_remaining()
    if len(unresolved) != 1:
        raise RuntimeError('The exact one-orphan cleanup plan needs reconciliation')
    target = unresolved[0]['recorded']
    if (target['pid'], target['created']) != (37392, 134373167992691453):
        raise RuntimeError('The failed worker orphan identity differs')
    for peer in plan['protected']:
        owners = recovery.read(Path(peer['directory']) / 'owned_aspen_apps.json', [])
        if any((r['pid'], r['created']) == (target['pid'], target['created']) for r in owners):
            raise RuntimeError('Target is also recorded by a protected live peer')
    job = recovery.read(Path(plan['job']))
    canonical = Path(job['source_case']['aspen_path'])
    if canonical.name != 'C2S_SS_simulation14.dynf' or recovery.digest(canonical) != job['source_case']['aspen_sha256']:
        raise RuntimeError('The reserved source case differs')
    receipt = dict(state='prepared', at_utc=recovery.now(), failed_worker=plan['failed_worker'],
                   target=target, verified_backup_files=len(preservation['files']),
                   before_memory=recovery.memory(), actions=[])
    destination = recovery.HERE / 'orphan_cleanup.json'
    recovery.write(destination, receipt)
    with aspen_file_lock(canonical, recovery.ROOT / 'Distillation/Results/_aspen_file_locks'):
        recovery.protected_check(plan)
        if recovery.process_identity(status['pid'])['alive'] is not False:
            raise RuntimeError('The former worker identity is not retired')
        receipt['actions'].append(dict(target=target, result=terminate_owned_identity(target)))
        recovery.write(destination, receipt)
    if recovery.owned_remaining():
        raise RuntimeError('An owned application remains; do not launch')
    recovery.protected_check(plan)
    receipt.update(state='closed', finished_at=recovery.now(), after_memory=recovery.memory(),
                   all_peer_and_retention_identities_preserved=True)
    recovery.write(destination, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    cleanup()
