"""Preserve the retired failed weights worker; no process or simulator action."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'change-reports/hourly_run_checks_20261024'))
import collect_hourly as hourly
from aspen_retention import Verifier, atomic_json, attributes, digest, now, private_snapshot
from column_review.common import process_identity

HERE = Path(__file__).resolve().parent
CATEGORY = 'weights'
CASE = 14
EXPECTED_STEPS = [117200, 116800]
EXPECTED_PAIRS = 67
EXPECTED_ERROR = 'Refusing to retire an unowned native application'
PEER_CATEGORIES = ('markov', 'residual', 'combined')
OUTPUT = hourly.OUTPUTS['weights']
BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\weights_pid_reuse_recovery_20261024')
STATE = ROOT / 'Distillation/Results/state_recovery/20261024_weights_pid_reuse'


def prepare():
    if BACKUP.exists() or STATE.exists() or (HERE / 'preservation.json').exists():
        raise FileExistsError('Inspect existing preparation before any duplicate backup or launch')
    paths = dict(status=OUTPUT / 'status.json', index=OUTPUT / 'training_checkpoints/latest.json',
                 owners=OUTPUT / 'owned_aspen_apps.json', retention=hourly.RETENTION / 'status.json',
                 anchors=hourly.RETENTION / 'checkpoint_hashes.json',
                 base=ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_process/manifest.json',
                 budget=ROOT / 'Distillation/Results/state_recovery/20261021_e1_budget20/manifest.json')
    metadata = hourly.stable_batch(paths)
    status = metadata['status']
    if (status['state'] != 'failed' or status['category'] != CATEGORY or status['case'] != CASE
            or process_identity(status['pid'])['alive'] is not False):
        raise RuntimeError('Only the retired failed weights worker may be prepared')
    if status['error'] != EXPECTED_ERROR:
        raise RuntimeError('Failure requires a new diagnosis')
    entry = next(j for j in metadata['budget']['jobs'] if j['category'] == CATEGORY)
    job_path = Path(entry['job'])
    job = hourly.stable_batch({'job': job_path})['job']
    if (entry['aspen_number'] != CASE or entry['expected_matched_pairs'] != 80
            or job['category'] != CATEGORY or job['options']['decisions_per_window'] != 20
            or digest(job_path) != entry['job_sha256']):
        raise RuntimeError('Original case, job or comparison budget differs')
    verification = HERE / 'checkpoint_verification'
    verification.mkdir()
    atomic_json(verification / 'checkpoint_hashes.json', metadata['anchors'])
    pair = Verifier(verification).pair(OUTPUT / 'training_checkpoints', ROOT / 'Distillation/Results')
    if [r['completed_steps'] for r in pair['checkpoints']] != EXPECTED_STEPS:
        raise RuntimeError('Expected the two complete cycle-293/292 weights checkpoints')
    for r in pair['checkpoints']:
        key = str(OUTPUT / 'training_checkpoints') + f'::{r["slot"]}::{r["completed_steps"]}'
        if metadata['anchors'].get(key) != r['sha256']:
            raise RuntimeError('Independent retention checkpoint anchor differs')
    records = hourly.completed_records(OUTPUT / 'capture/decisions.jsonl')
    if len(records) != EXPECTED_PAIRS or any(r['status'] != 'completed' for r in records):
        raise RuntimeError('The expected 67 completed comparisons differ')
    files = {OUTPUT / name for name in ('status.json', 'owned_aspen_apps.json', 'ownership_status.json',
                                      'native_process_recovery.jsonl', 'training_checkpoints/latest.json')}
    for name in ('capture', 'native_snapshots'):
        files.update(p for p in (OUTPUT / name).rglob('*') if p.is_file())
    for checkpoint in pair['checkpoints']:
        files.add(Path(checkpoint['python_file']))
        files.update(p for p in Path(checkpoint['native_directory']).rglob('*') if p.is_file())
    files.add(job_path)
    log = ROOT / ('Distillation/Results/state_recovery/20261023_e1_native_process/logs/' + CATEGORY + '.log')
    files.add(log)
    copies = []
    BACKUP.mkdir()
    for source in sorted(files):
        if not source.is_file():
            continue
        relative = source.relative_to(OUTPUT) if source.is_relative_to(OUTPUT) else Path('prior_logs') / source.name
        target = BACKUP / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(source)
        if source.suffix.lower() in ('.json', '.pkl'):
            with private_snapshot(source) as stream, target.open('xb') as destination:
                shutil.copyfileobj(stream, destination)
        else:
            shutil.copyfile(source, target)
        if digest(target) != checksum or digest(source) != checksum:
            raise RuntimeError('Backup checksum differs: ' + str(source))
        copies.append(dict(source=str(source), local_copy=str(target), bytes=attributes(target)['size'], sha256=checksum))
    for row in copies:
        if digest(Path(row['local_copy'])) != row['sha256']:
            raise RuntimeError('Independent backup readback differs')
    trace_records = []
    for record in records:
        source = (OUTPUT / 'capture' / record['trace']).resolve()
        if not source.is_relative_to((OUTPUT / 'capture').resolve()):
            raise RuntimeError('Saved trace leaves its capture directory')
        trace_records.append(dict(source=str(source), sha256=digest(source),
                                  backup=str(BACKUP / source.relative_to(OUTPUT)),
                                  step=record['step'], committed=record['step'] < pair['index']['completed_steps']))
    protected = []
    for category in PEER_CATEGORIES:
        peer = hourly.stable_batch({'status': hourly.OUTPUTS[category] / 'status.json'})['status']
        identity = process_identity(peer['pid'])
        if peer['state'] != 'running' or identity['alive'] is not True:
            raise RuntimeError('A peer needs inspection before recovery: ' + category)
        protected.append(dict(category=category, directory=str(hourly.OUTPUTS[category]), identity=identity))
    retention_identity = process_identity(metadata['retention']['pid'])
    if retention_identity['alive'] is not True:
        raise RuntimeError('Retention is not confirmed live')
    preservation = dict(verified=True, at_utc=now(), backup=str(BACKUP), files=copies,
                        checkpoint_steps=[r['completed_steps'] for r in pair['checkpoints']],
                        completed_pairs=EXPECTED_PAIRS, post_checkpoint_pairs=sum(not r['committed'] for r in trace_records),
                        original_files_retained=True, actions=dict(process_signals=0, simulator_calls=0))
    atomic_json(BACKUP / 'preservation.json', preservation)
    atomic_json(HERE / 'preservation.json', dict(verified=True, at_utc=now(), backup=str(BACKUP),
                manifest=str(BACKUP / 'preservation.json'), manifest_sha256=digest(BACKUP / 'preservation.json'),
                files=len(copies), bytes=sum(r['bytes'] for r in copies), completed_pairs=EXPECTED_PAIRS,
                post_checkpoint_pairs=preservation['post_checkpoint_pairs']))
    atomic_json(HERE / 'preparation.json', dict(status=status, ownership=metadata['owners'], pair=pair,
                original_entry=entry, original_process_entry=next(j for j in metadata['base']['jobs'] if j['category']==CATEGORY),
                original_process_protocol_sha256=metadata['base']['protocol_sha256'], protected=protected,
                retention_identity=retention_identity, traces=trace_records, job=job, job_path=str(job_path)))
    print(json.dumps(json.loads((HERE / 'preservation.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    prepare()
