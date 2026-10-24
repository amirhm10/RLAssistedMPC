"""Preserve the retired residual production failure and all recovery pins."""
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'change-reports/weights_pid_reuse_recovery_20261024'))
import prepare_recovery as original


def prepare():
    original.HERE=HERE
    original.CATEGORY='residual'
    original.CASE=17
    original.EXPECTED_STEPS=[118400,118000]
    original.EXPECTED_PAIRS=77
    original.EXPECTED_ERROR="(-2147417851, 'The server threw an exception.', None, None)"
    original.PEER_CATEGORIES=('markov','combined')
    original.OUTPUT=original.hourly.OUTPUTS['residual']
    original.STATE=ROOT/'Distillation/Results/state_recovery/20261024_residual_native_retry2'
    original.BACKUP=Path(r'C:\Users\hamediaa\RL_MPC_local_archive\residual_native_retry2_20261024')
    original.prepare()
    prepared=json.loads((HERE/'preparation.json').read_text(encoding='utf-8'))
    receipt=json.loads((HERE/'preservation.json').read_text(encoding='utf-8'))
    manifest_path=Path(receipt['manifest'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if prepared['status']['pid']!=46740 or 'openDocument' not in prepared['status']['traceback']:
        raise RuntimeError('This backup must describe the diagnosed second residual production failure')
    files={ROOT/'Distillation/Results/state_recovery/20261024_residual_native_retry/logs/production.log'}
    for checkpoint in prepared['pair']['checkpoints']:
        for name in checkpoint.get('private_roots',[]):
            path=Path(name).resolve()
            if not path.is_relative_to(original.OUTPUT.resolve()):
                raise RuntimeError('A pinned private dependency leaves the residual output')
            if path.is_dir():files.update(p for p in path.rglob('*') if p.is_file())
            elif path.is_file():files.add(path)
            else:raise RuntimeError('A pinned private dependency is absent')
    known={row['source'] for row in manifest['files']}
    for source in sorted(files):
        if str(source) in known:continue
        relative=source.relative_to(original.OUTPUT) if source.is_relative_to(original.OUTPUT) else Path('prior_logs/latest_production.log')
        target=original.BACKUP/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        checksum=original.digest(source)
        if source.suffix.lower() in ('.json','.pkl'):
            with original.private_snapshot(source) as stream,target.open('xb') as copied:
                shutil.copyfileobj(stream,copied)
        else:shutil.copyfile(source,target)
        if original.digest(source)!=checksum or original.digest(target)!=checksum:
            raise RuntimeError('Pinned dependency/current log copy differs')
        manifest['files'].append(dict(source=str(source),local_copy=str(target),bytes=original.attributes(target)['size'],sha256=checksum))
    for row in manifest['files']:
        if original.digest(Path(row['local_copy']))!=row['sha256']:
            raise RuntimeError('Independent full backup readback differs')
    current=original.hourly.stable_batch({'index':original.OUTPUT/'training_checkpoints/latest.json'})['index']
    if current!=prepared['pair']['index']:
        raise RuntimeError('Failed-run checkpoint changed during preservation')
    manifest.update(extended_recovery_dependencies_verified=True,current_failure_pid=46740,
                    current_failure_at=prepared['status']['finished_at'])
    original.atomic_json(manifest_path,manifest)
    receipt.update(manifest_sha256=original.digest(manifest_path),files=len(manifest['files']),
        bytes=sum(row['bytes'] for row in manifest['files']),extended_recovery_dependencies_verified=True)
    original.atomic_json(HERE/'preservation.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':prepare()
