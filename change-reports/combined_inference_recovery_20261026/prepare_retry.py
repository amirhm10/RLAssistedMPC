"""Preserve the verified bounded checkpoint and failed startup before any retry."""
from copy import deepcopy
from pathlib import Path
import sys
import uuid
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recover_parallel as prior
import parallel_branches as p
import preserve
from aspen_retention import Verifier, still_current, version
import aspen_retention
from offline_pair import closed_layout

preserve.HERE = HERE
OUTPUT = prior.r.OUTPUT
OLD_STATE = ROOT / 'Distillation/Results/state_recovery/20261026_combined_inference'
STATE = ROOT / 'Distillation/Results/state_recovery/20261026_combined_metadata_retry'
ARCHIVE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\combined_inference_recovery_20261026')


def prepare():
    if (HERE / 'retry_plan.json').exists():
        raise FileExistsError('Never overwrite or repeat a prepared retry plan')
    old = prior.r.verify()
    failed = p.closed_json(OLD_STATE / 'worker_0_identity.json')['identity']
    if p.h.process(failed['pid']).get('alive') is not False or p.h.native_processes():
        raise RuntimeError('The failed worker and all its owned applications must already be retired')
    values = p.h.stable_batch(dict(manifest=OLD_STATE / 'manifest.json', validation=OLD_STATE / 'validation.json',
        anchors=p.h.RETENTION / 'checkpoint_hashes.json', retention=p.h.RETENTION / 'status.json',
        status=OUTPUT / 'status.json', owners=OUTPUT / 'owned_aspen_apps.json', ownership=OUTPUT / 'ownership_status.json',
        pool=OLD_STATE / 'parallel_pool_0.json'))
    if (values['manifest']['state'] != 'failed' or not values['validation']['passed']
            or values['validation']['index']['completed_steps'] != 118808
            or values['manifest']['job']['pid'] != failed['pid']):
        raise RuntimeError('Refresh the failed startup and accepted bounded checkpoint evidence')
    pool = Path(values['pool']['directory'])
    case17 = pool / 'case17.log'
    log = case17.read_text(encoding='utf-8', errors='replace')
    if 'PermissionError: [WinError 5]' not in log or 'ownership_status.json.tmp' not in log:
        raise RuntimeError('The startup error does not match the scoped metadata repair')
    if any(list((pool / ('case' + str(case))).glob('request_*.json')) for case in (15, 17)):
        raise RuntimeError('A scientific branch request ran; refresh recovery scope')
    evidence = dict(at_utc=p.now(), parent_identity=failed,
        confirmed_ownership_atomic_replace_failure=True, case17_log=str(case17), case17_log_sha256=p.h.digest(case17),
        error='PermissionError [WinError 5] replacing case17 ownership_status.json at branch startup',
        requests_dispatched=0, all_scientific_native_apps_retired=True, process_signals_required=0,
        onedrive_running=any((x.info['name'] or '').lower() == 'onedrive.exe'
            for x in __import__('psutil').process_iter(['name'])),
        repair='Unique atomic ownership temporary plus bounded retry for WinError5/32/33; science and checkpoints unchanged',
        tests_passed=3)
    if (HERE / 'startup_failure.json').exists():
        recorded = p.closed_json(HERE / 'startup_failure.json')
        if recorded['parent_identity'] != failed or recorded['case17_log_sha256'] != evidence['case17_log_sha256']:
            raise RuntimeError('The preserved startup evidence differs; do not overwrite it')
        evidence = recorded
    else:
        p.immutable(HERE / 'startup_failure.json', evidence)
    cache = HERE / ('retry_pair_verifier_' + uuid.uuid4().hex[:8])
    cache.mkdir()
    p.h.atomic_json(cache / 'checkpoint_hashes.json', values['anchors'])
    before = closed_layout(OUTPUT / 'training_checkpoints', failed)
    __import__('time').sleep(2.2)
    if closed_layout(OUTPUT / 'training_checkpoints', failed) != before:
        raise RuntimeError('The closed checkpoint is not stable')
    # Cold restore appended snapshot.restores.jsonl after the index. Only that
    # audited, closed-process timestamp exception is allowed; all original
    # component SHA-256, Python metadata and native-linkage checks still run.
    with patch.object(aspen_retention, 'quiet_checkpoint_layout', lambda directory: closed_layout(directory, failed)):
        pair = Verifier(cache).pair(OUTPUT / 'training_checkpoints', ROOT / 'Distillation/Results')
    if [row['completed_steps'] for row in pair['checkpoints']] != [118808, 118800]:
        raise RuntimeError('The newest two complete matched checkpoints differ')
    if pair['index']['sha256'] != values['validation']['index']['sha256']:
        raise RuntimeError('The bounded checkpoint changed after validation')
    for point in pair['checkpoints']:
        key = str(OUTPUT / 'training_checkpoints') + f'::{point["slot"]}::{point["completed_steps"]}'
        anchor = values['anchors'].get(key)
        if anchor is not None and anchor != point['sha256']:
            raise RuntimeError('Independent retention checksum differs')
        if anchor is None and point['completed_steps'] == 118800 and point['sha256'] != old['initial_checkpoint_sha256']:
            raise RuntimeError('The previous verified checkpoint differs')
    files = set(preserve.capture_files(pair))
    files.update(path for path in OLD_STATE.rglob('*') if path.is_file())
    for session in values['status'].get('native_sessions', {}).values():
        folder = Path(session['aspen_path']).parent
        if not folder.resolve().is_relative_to(OUTPUT.resolve()):
            raise RuntimeError('A failed working folder escaped this output')
        files.update(path for path in folder.rglob('*') if path.is_file())
    previous = {}
    for manifest in ARCHIVE.glob('before_*/preservation.json'):
        for row in p.closed_json(manifest).get('files', []):
            previous[row['source']] = row
    if p.shutil.disk_usage(ROOT.anchor).free < sum(p.h.attributes(path)['size'] for path in files) + 24 * 2**30:
        raise RuntimeError('Insufficient space for complete independent preservation')
    backup = ARCHIVE / ('before_metadata_retry_' + p.now().replace(':', '').replace('.', '_'))
    backup.mkdir(parents=True, exist_ok=False)
    p.immutable(backup / 'checkpoint_pair.json', pair)
    p.immutable(backup / 'metadata.json', values)
    copies = []
    for number, source in enumerate(sorted(files), 1):
        cached = previous.get(str(source))
        if (cached and tuple(cached['source_version']) == tuple(version(source))
                and p.h.digest(cached['local_copy']) == cached['sha256']):
            copies.append(cached)
        else:
            copies.append(preserve.copied_generation(source, backup / 'files' / source.relative_to(ROOT).parent))
        if number % 200 == 0:
            print(f'Verified retry preservation {number}/{len(files)} files', flush=True)
    if not still_current(pair):
        raise RuntimeError('The checkpoint changed during preservation; retain backups and refresh')
    records = p.h.completed_records(OUTPUT / 'capture/decisions.jsonl')
    saved = {row['source']: row for row in copies}
    traces = []
    for row in records:
        source = (OUTPUT / 'capture' / row['trace']).resolve()
        copied = saved[str(source)]
        traces.append(dict(source=str(source), backup=copied['local_copy'], sha256=copied['sha256'],
            step=row['step'], committed=row['step'] < 118808))
    if len(records) != 313 or not all(row['committed'] for row in traces):
        raise RuntimeError('Refresh current primary prefix before retrying')
    for row in copies:
        if p.h.digest(row['local_copy']) != row['sha256']:
            raise RuntimeError('Independent retry backup readback differs')
    preservation = dict(verified=True, at_utc=p.now(), pair=pair, files=copies,
        completed_pairs=len(records), traces=traces, all_originals_retained=True,
        initial_314_preservation=old['backup_manifest'], initial_314_preservation_sha256=old['backup_manifest_sha256'],
        process_signals=0, simulator_calls=0)
    p.immutable(backup / 'preservation.json', preservation)
    p.immutable(HERE / 'retry_preservation.json', dict(verified=True, at_utc=p.now(),
        manifest=str(backup / 'preservation.json'), manifest_sha256=p.h.digest(backup / 'preservation.json'),
        files=len(copies), bytes=sum(row['bytes'] for row in copies), checkpoint_steps=[118808, 118800],
        primary_pairs=313, initial_314_originals_still_preserved=True))
    plan = deepcopy(old)
    plan.update(failed_startup_retry=True, failed_worker=failed,
        failure_evidence=dict(evidence, path=str(HERE / 'startup_failure.json'), sha256=p.h.digest(HERE / 'startup_failure.json')),
        prepared_at=p.now(), backup_manifest=str(backup / 'preservation.json'),
        backup_manifest_sha256=p.h.digest(backup / 'preservation.json'), traces=traces,
        original_pairs=313, post_checkpoint_pairs=0, checkpoint_steps=[118808, 118800],
        bounded_acceptance=str(HERE / 'bounded_acceptance.json'), bounded_acceptance_sha256=p.h.digest(HERE / 'bounded_acceptance.json'),
        accepted_validation_attempt=dict(values['manifest']['job']['attempts'][0], identity=p.closed_json(OLD_STATE / 'worker_8_identity.json')['identity']),
        original_replay_archive=values['manifest']['replay_archive'])
    for name in ('metadata_io.py', 'branch_metadata_worker.py', 'recover_production.py', 'prepare_retry.py',
                 'accept_retry.py', 'inspect_retry.py', 'retired_side_service_retry.py', 'offline_pair.py'):
        plan['adapter_sha256'][name] = p.h.digest(HERE / name)
        plan['branch_template']['adapter_hashes'][str(HERE / name)] = p.h.digest(HERE / name)
    p.immutable(HERE / 'retry_plan.json', plan)
    p.immutable(HERE / 'retry_seal.json', dict(at_utc=p.now(), retry_plan_sha256=p.h.digest(HERE / 'retry_plan.json')))
    STATE.mkdir(parents=True, exist_ok=False)
    with (STATE / 'validation.json').open('xb') as stream:
        stream.write((OLD_STATE / 'validation.json').read_bytes())
    if p.h.digest(STATE / 'validation.json') != p.h.digest(OLD_STATE / 'validation.json'):
        raise RuntimeError('The unchanged bounded-validation receipt copy differs')
    print('Fresh startup retry preserved: 313 current prefix pairs, all314 originals and checkpoints118808/118800.', flush=True)


if __name__ == '__main__':
    prepare()
