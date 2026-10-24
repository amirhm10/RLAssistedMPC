"""Preserve only the confirmed retired combined failure; no simulator actions."""
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'change-reports/hourly_run_checks_20261024'))
import collect_hourly as h
from aspen_retention import Verifier, atomic_json, attributes, digest, now, private_snapshot
from column_review.common import process_identity

OUTPUT = h.OUTPUTS['combined']
BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\combined_native_recovery_20261024')
STATE = ROOT/'Distillation/Results/state_recovery/20261024_combined_native_retry'


def prepare():
    if BACKUP.exists() or STATE.exists() or (HERE/'preservation.json').exists():
        raise FileExistsError('Inspect existing preservation/recovery before repeating')
    paths = dict(status=OUTPUT/'status.json', index=OUTPUT/'training_checkpoints/latest.json',
                 owners=OUTPUT/'owned_aspen_apps.json', ownership=OUTPUT/'ownership_status.json',
                 retention=h.RETENTION/'status.json', anchors=h.RETENTION/'checkpoint_hashes.json',
                 budget=ROOT/'Distillation/Results/state_recovery/20261021_e1_budget20/manifest.json')
    metadata = h.stable_batch(paths)
    status = metadata['status']
    if (status.get('state') != 'failed' or status.get('category') != 'combined' or status.get('case') != 16
            or status.get('pid') != 22588 or process_identity(status['pid'])['alive'] is not False
            or status.get('error') != "(-2147417851, 'The server threw an exception.', None, None)"
            or 'openDocument' not in status.get('traceback', '')):
        raise RuntimeError('Only the diagnosed, retired combined failure is eligible')
    if metadata['ownership'].get('phase') in ('creating', 'unverified_creation'):
        raise RuntimeError('Failed-run native creation ownership is unresolved')
    for old in metadata['owners']:
        actual = process_identity(old['pid'])
        if actual['alive'] is not False and not (actual['alive'] is True and actual['created'] != old['created']):
            raise RuntimeError('A failed-run owned application remains live or inaccessible')
    entry = next(j for j in metadata['budget']['jobs'] if j['category'] == 'combined')
    job_path = Path(entry['job'])
    job = h.stable_batch({'job': job_path})['job']
    if (entry['aspen_number'] != 16 or entry['expected_matched_pairs'] != 320 or job['category'] != 'combined'
            or job['options']['decisions_per_window'] != 20 or digest(job_path) != entry['job_sha256']):
        raise RuntimeError('Original combined job/case/budget differs')
    verification = HERE/'checkpoint_verification'
    verification.mkdir()
    atomic_json(verification/'checkpoint_hashes.json', metadata['anchors'])
    pair = Verifier(verification).pair(OUTPUT/'training_checkpoints', ROOT/'Distillation/Results')
    if [r['completed_steps'] for r in pair['checkpoints']] != [80800, 80400] or len(pair['index']['agents']) != 4:
        raise RuntimeError('Expected two complete combined checkpoints with four learners')
    for point in pair['checkpoints']:
        key = str(OUTPUT/'training_checkpoints') + f'::{point["slot"]}::{point["completed_steps"]}'
        if metadata['anchors'].get(key) != point['sha256']:
            raise RuntimeError('Independent retention checkpoint anchor differs')
    records = h.completed_records(OUTPUT/'capture/decisions.jsonl')
    if (len(records) != 181 or any(r['status'] != 'completed' for r in records)
            or len(records) != len({(r['channel'],r['window'],r['step']) for r in records})):
        raise RuntimeError('The 181 saved primary comparisons differ')
    protected = []
    for category in ('weights', 'markov'):
        receipt = ROOT/f'change-reports/{"weights_pid_reuse" if category=="weights" else "markov_native"}_recovery_20261024/acceptance.json'
        accepted = json.loads(receipt.read_text(encoding='utf-8'))
        peer = h.stable_batch({'status': h.OUTPUTS[category]/'status.json'})['status']
        identity = process_identity(peer['pid'])
        if (not accepted['passed'] or peer['state'] != 'running'
                or identity != accepted['production_identity'] or identity['alive'] is not True):
            raise RuntimeError('Accepted production peer needs inspection: '+category)
        protected.append(dict(category=category, directory=str(h.OUTPUTS[category]), identity=identity))
    retention_identity = process_identity(metadata['retention']['pid'])
    if retention_identity['alive'] is not True:
        raise RuntimeError('Automatic retention is not live')
    files = {OUTPUT/name for name in ('status.json','owned_aspen_apps.json','ownership_status.json',
                                      'training_checkpoints/latest.json')}
    for name in ('capture','native_snapshots'):
        files.update(p for p in (OUTPUT/name).rglob('*') if p.is_file())
    for point in pair['checkpoints']:
        files.add(Path(point['python_file']))
        files.update(p for p in Path(point['native_directory']).rglob('*') if p.is_file())
        for value in point.get('private_roots', []):
            dependency = Path(value).resolve()
            if not dependency.is_relative_to(OUTPUT.resolve()):
                raise RuntimeError('A pinned private dependency leaves the isolated output')
            if dependency.is_dir():
                files.update(p for p in dependency.rglob('*') if p.is_file())
            elif dependency.is_file():
                files.add(dependency)
            else:
                raise RuntimeError('A pinned recovery dependency is absent')
    files.add(job_path)
    files.add(ROOT/'Distillation/Results/state_recovery/20261021_e1_budget20/logs/combined.log')
    copies = []
    BACKUP.mkdir()
    for source in sorted(files):
        if not source.is_file():
            raise RuntimeError('A required recovery file is absent: '+str(source))
        relative = source.relative_to(OUTPUT) if source.is_relative_to(OUTPUT) else Path('prior_logs')/source.name
        target = BACKUP/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(source)
        if source.suffix.lower() in ('.json','.pkl'):
            with private_snapshot(source) as stream, target.open('xb') as destination:
                shutil.copyfileobj(stream,destination)
        else:
            shutil.copyfile(source,target)
        if digest(target) != checksum or digest(source) != checksum:
            raise RuntimeError('Backup/source checksum differs: '+str(source))
        copies.append(dict(source=str(source),local_copy=str(target),bytes=attributes(target)['size'],sha256=checksum))
    for record in copies:
        if digest(Path(record['local_copy'])) != record['sha256']:
            raise RuntimeError('Independent backup readback differs')
    traces = []
    for record in records:
        source = (OUTPUT/'capture'/record['trace']).resolve()
        if not source.is_relative_to((OUTPUT/'capture').resolve()):
            raise RuntimeError('A comparison trace leaves the capture directory')
        traces.append(dict(source=str(source),backup=str(BACKUP/source.relative_to(OUTPUT)),sha256=digest(source),
                           step=record['step'],committed=record['step'] < 80800))
    preserved = dict(verified=True,at_utc=now(),backup=str(BACKUP),files=copies,checkpoint_steps=[80800,80400],
                     completed_pairs=181,post_checkpoint_pairs=sum(not t['committed'] for t in traces),
                     original_files_retained=True,actions=dict(process_signals=0,simulator_calls=0))
    atomic_json(BACKUP/'preservation.json',preserved)
    atomic_json(HERE/'preservation.json',dict(verified=True,at_utc=now(),backup=str(BACKUP),
        manifest=str(BACKUP/'preservation.json'),manifest_sha256=digest(BACKUP/'preservation.json'),
        files=len(copies),bytes=sum(row['bytes'] for row in copies),completed_pairs=181,
        post_checkpoint_pairs=preserved['post_checkpoint_pairs']))
    atomic_json(HERE/'preparation.json',dict(status=status,ownership=metadata['owners'],pair=pair,original_entry=entry,
        original_budget_protocol_sha256=metadata['budget']['protocol_sha256'],protected=protected,
        retention_identity=retention_identity,traces=traces,job=job,job_path=str(job_path)))
    print(json.dumps(json.loads((HERE/'preservation.json').read_text(encoding='utf-8')),indent=2),flush=True)


if __name__ == '__main__':
    prepare()
