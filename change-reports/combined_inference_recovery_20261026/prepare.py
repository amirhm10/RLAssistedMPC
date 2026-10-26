"""Independently verify and preserve the current stalled attempt; never signal it."""
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACCEPTED = ROOT / 'change-reports/combined_parallel_20261024'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ACCEPTED))
import parallel_branches as p
import preserve
from aspen_retention import Verifier, Deferred, still_current, version

preserve.HERE = HERE
OUTPUT = p.h.OUTPUTS['combined']
OLD_STATE = ROOT / 'Distillation/Results/state_recovery/20261024_combined_parallel'
ARCHIVE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\combined_inference_recovery_20261026')


def prepare():
    if (HERE / 'preservation.json').exists():
        raise FileExistsError('Never overwrite a completed preservation receipt')
    accepted = p.closed_json(ACCEPTED / 'acceptance.json')
    parent = accepted['production_identity']
    if not p.same_generation(parent, p.h.process(parent['pid'])):
        raise RuntimeError('Refresh the failure identities before preservation')
    paths = dict(index=OUTPUT / 'training_checkpoints/latest.json',
        owners=OUTPUT / 'owned_aspen_apps.json', ownership=OUTPUT / 'ownership_status.json',
        status=OUTPUT / 'status.json', anchors=p.h.RETENTION / 'checkpoint_hashes.json',
        retention=p.h.RETENTION / 'status.json', pool=OLD_STATE / 'parallel_pool_0.json')
    metadata = p.h.stable_batch(paths)
    pool = Path(metadata['pool']['directory'])
    if metadata['ownership']['phase'] in ('creating', 'unverified_creation'):
        raise Deferred('Wait for a settled native ownership generation')
    verifier = HERE / ('pair_verifier_' + uuid.uuid4().hex[:8])
    verifier.mkdir()
    p.h.atomic_json(verifier / 'checkpoint_hashes.json', metadata['anchors'])
    pair = Verifier(verifier).pair(OUTPUT / 'training_checkpoints', ROOT / 'Distillation/Results')
    if len(pair['index']['agents']) != 4:
        raise RuntimeError('Four complete learner/controller states are required')
    for point in pair['checkpoints']:
        key = str(OUTPUT / 'training_checkpoints') + f'::{point["slot"]}::{point["completed_steps"]}'
        if metadata['anchors'].get(key) != point['sha256']:
            raise Deferred('Both slots require the independent retention verification')
    files = set(preserve.capture_files(pair))
    files.update(x for x in OLD_STATE.iterdir() if x.is_file())
    files.update(x for x in (OLD_STATE / 'logs').rglob('*') if x.is_file())
    files.update(x for x in pool.iterdir() if x.is_file())
    files.update(x for x in (pool / 'frozen_decisions').rglob('input_receipt.json'))
    files.update(x for x in (pool / 'frozen_decisions').rglob('primary_commit.json'))
    child_identities = []
    pending = []
    for case in (15, 17):
        child = pool / ('case' + str(case))
        identity = p.closed_json(child / 'identity.json')
        if not p.same_generation(identity, p.h.process(identity['pid'])):
            raise RuntimeError('Refresh the stalled child generation before preservation')
        child_identities.append(dict(case=case, identity=identity))
        files.update(x for x in child.rglob('*') if x.is_file() and 'working' not in x.relative_to(child).parts)
        for request in child.glob('request_*.json'):
            if (child / ('done_' + request.stem + '.json')).exists():
                continue
            row = p.closed_json(request)
            pending.append(dict(case=case, request=str(request), value=row))
            files.update(x for x in Path(row['snapshot']).parent.rglob('*') if x.is_file())
    if len(pending) != 2 or len({r['value']['snapshot_sha256'] for r in pending}) != 1:
        raise RuntimeError('The two stalled requests must share one preserved frozen decision')
    backup = ARCHIVE / ('before_' + p.now().replace(':', '').replace('.', '_'))
    backup.mkdir(parents=True, exist_ok=False)
    p.immutable(backup / 'checkpoint_pair.json', pair)
    p.immutable(backup / 'metadata.json', metadata)
    total = sum(p.h.attributes(x)['size'] for x in files)
    if p.shutil.disk_usage(ROOT.anchor).free < total + 24 * 2**30:
        raise RuntimeError('Insufficient room to preserve every recovery dependency and result')
    previous = {}
    older = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\combined_parallel_20261024')
    for manifest in older.glob('before_*/preservation.json'):
        for row in p.closed_json(manifest).get('files', []):
            previous[row['source']] = row
    copied, deferred = [], []
    pinned = [Path(x).resolve() for point in pair['checkpoints'] for x in point['private_roots']]
    try:
        for number, source in enumerate(sorted(files), 1):
            old = previous.get(str(source))
            if old and tuple(old['source_version']) == tuple(version(source)) and p.h.digest(old['local_copy']) == old['sha256']:
                copied.append(old)
            else:
                try:
                    copied.append(preserve.copied_generation(source, backup / 'files' / source.relative_to(ROOT).parent))
                except PermissionError as error:
                    permitted_lock = source.name == 'retention.lock' and source.is_relative_to(ROOT / 'Distillation/Results/state_recovery')
                    if not permitted_lock and not any(source.resolve().is_relative_to(path) for path in pinned):
                        raise
                    deferred.append(dict(source=str(source), version=version(source), error=repr(error), reason='owned_working_lock'))
            if number % 100 == 0:
                print(f'Preserved {number}/{len(files)} source files', flush=True)
        if not still_current(pair):
            raise Deferred('Checkpoint changed; retain this backup and refresh')
        records = p.h.completed_records(OUTPUT / 'capture/decisions.jsonl')
        saved = {row['source']: row for row in copied}
        traces = []
        for row in records:
            source = (OUTPUT / 'capture' / row['trace']).resolve()
            item = saved[str(source)]
            if p.h.digest(item['local_copy']) != item['sha256']:
                raise RuntimeError('Independent result backup changed')
            traces.append(dict(source=str(source), backup=item['local_copy'], sha256=item['sha256'],
                               step=row['step'], committed=row['step'] < pair['index']['completed_steps']))
        for row in copied:
            if p.h.digest(row['local_copy']) != row['sha256']:
                raise RuntimeError('Final independent backup readback differs')
        if not still_current(pair):
            raise Deferred('Checkpoint changed during final readback')
        receipt = dict(verified=True, at_utc=p.now(), files=copied, pair=pair, original_identity=parent,
            child_identities=child_identities, pending=pending, owners=metadata['owners'],
            retention_identity={k:p.h.process(metadata['retention']['pid'])[k] for k in ('pid', 'created', 'alive')},
            completed_pairs=len(records), traces=traces, post_checkpoint_pairs=sum(not x['committed'] for x in traces),
            original_files_retained=True, all_results_and_complete_checkpoint_dependencies_copied=True,
            deferred_owned_working_files=deferred, working_files_must_copy_before_relaunch=bool(deferred),
            process_signals=0, simulator_calls=0)
        p.immutable(backup / 'preservation.json', receipt)
        p.immutable(HERE / 'preservation.json', dict(verified=True, manifest=str(backup / 'preservation.json'),
            manifest_sha256=p.h.digest(backup / 'preservation.json'), backup=str(backup), files=len(copied),
            bytes=sum(x['bytes'] for x in copied), checkpoint_steps=[x['completed_steps'] for x in pair['checkpoints']],
            completed_pairs=len(records), post_checkpoint_pairs=receipt['post_checkpoint_pairs'],
            deferred_live_working_files=len(deferred)))
        print('Verified independent recovery/results preservation: ' + str(len(records)) + ' pairs; ' + str(len(copied)) + ' files', flush=True)
    except BaseException as error:
        p.immutable(backup / 'incomplete.json', dict(at_utc=p.now(), error=repr(error), verified_copies=copied,
                    original_attempt_not_interrupted=True))
        raise


if __name__ == '__main__':
    prepare()
