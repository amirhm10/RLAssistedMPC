"""Restore one verified ZIP into a NEW staging directory; never overwrite data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore(root, index, destination):
    root = root.resolve()
    manifest = json.loads((root / 'restore_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['archives']) == 910 and 0 <= index < 910
    record = manifest['archives'][index]
    source = (root / Path(record['compressed']).name).resolve()
    assert source.is_relative_to(root) and source.is_file()
    assert digest(source) == record['compressed_sha256']
    destination = destination.resolve()
    assert not destination.exists(), 'Destination must be new; existing data will never be overwritten'
    destination.mkdir(parents=True)
    restored = []
    with zipfile.ZipFile(source) as archive:
        assert set(archive.namelist()) == {file['relative'] for file in record['files']}
        for file in record['files']:
            relative = PurePosixPath(file['relative'])
            assert not relative.is_absolute() and '..' not in relative.parts
            target = destination.joinpath(*relative.parts).resolve()
            assert target.is_relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(file['relative']) as saved, target.open('xb') as output:
                for block in iter(lambda: saved.read(1024 * 1024), b''):
                    output.write(block)
            assert target.stat().st_size == file['bytes'] and digest(target) == file['sha256']
            restored.append(dict(path=str(target), bytes=file['bytes'], sha256=file['sha256']))
    receipt = dict(verified=True, at_utc=datetime.now(timezone.utc).isoformat(), index=index,
                   archive=str(source), original_directory=record['original_directory'],
                   staging_directory=str(destination), restored_files=restored,
                   original_files_modified=0, original_files_deleted=0,
                   scope='Staged extraction only. No simulator opened and no active/recent checkpoint replaced.')
    (destination / 'restoration_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=Path(r'C:\Users\hamediaa\RL_MPC_local_archive\compressed_storage_20261023'))
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    restore(args.archive_root, args.index, args.destination)
