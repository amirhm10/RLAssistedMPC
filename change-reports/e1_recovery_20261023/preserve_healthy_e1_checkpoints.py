"""Copy complete saved healthy-E1 checkpoints; never attach to live Aspen."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKUP = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\healthy_e1_checkpoints_20261023')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert not BACKUP.exists()
    BACKUP.mkdir()
    jobs = [row for row in read(HERE / 'before.json')['jobs']
            if row['study'] == 'E1_column' and row['job'] in ('combined', 'horizon')]
    assert len(jobs) == 2
    records = []
    for job in jobs:
        source = Path(job['directory']) / 'training_checkpoints'
        index = read(source / 'latest.json')
        output = BACKUP / job['job']
        output.mkdir()
        primary = source / index['file']
        shutil.copyfile(primary, output / primary.name)
        assert digest(output / primary.name) == index['sha256']
        native_source = Path(index['native']['project_file']).parent.parent.resolve()
        assert native_source.is_relative_to(ROOT) and native_source.name.startswith('native_')
        native_copy = output / native_source.name
        shutil.copytree(native_source, native_copy)
        files = [dict(source=str(primary), copy=str(output / primary.name),
                      sha256=index['sha256'], bytes=(output / primary.name).stat().st_size)]
        for path in native_source.rglob('*'):
            if not path.is_file():
                continue
            target = native_copy / path.relative_to(native_source)
            sha = digest(path)
            assert digest(target) == sha
            files.append(dict(source=str(path), copy=str(target), sha256=sha, bytes=target.stat().st_size))
        for name in ('file', 'project_file'):
            original = Path(index['native'][name])
            copied = native_copy / original.relative_to(native_source)
            assert digest(copied) == index['native'][name + '_sha256']
        (output / 'latest.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
        records.append(dict(category=job['job'], completed_steps=index['completed_steps'],
                            completed_cycles=index['completed_cycles'], files=files,
                            saved_checkpoint_at_unix=index['saved_at_unix']))
    receipt = dict(verified=True, at_utc=datetime.now(timezone.utc).isoformat(),
                   destination=str(BACKUP), records=records, original_files_modified=0,
                   live_simulator_calls=0, restored=False,
                   restoration='Timestamped saved-state copies. Restore only missing files when their worker is stopped; never overwrite a newer checkpoint. latest.json retains original absolute paths as provenance; files maps original paths to verified local copies.')
    for path in (BACKUP / 'preservation_receipt.json', HERE / 'healthy_e1_preservation.json'):
        path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(verified=True, categories=[r['category'] for r in records],
                          cycles=[r['completed_cycles'] for r in records], live_simulator_calls=0)))


if __name__ == '__main__':
    main()
