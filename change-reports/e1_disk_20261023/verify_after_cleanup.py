"""Read-only checks after the explicitly approved raw-archive replacement."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive')
MANIFEST_SHA = '97af00c0dfbf371c8f2141d417d26b4ed09e6d272a0e609cb43be208745f96a9'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    removal = read(HERE / 'removal_result.json')
    assert removal['removed_folders'] == 910
    assert removal['removed_files'] == 3640
    assert removal['removed_bytes'] == 174583311764
    receipts = [json.loads(line) for line in (HERE / 'deletion_receipts.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    assert len(receipts) == len({row['directory'] for row in receipts}) == 910
    before = read(HERE / 'before.json')
    identities = {row['pid']: row for row in before['processes']}
    old_assessment = read(ROOT / 'change-reports/e1_recovery_20261023/recovery_assessment.json')
    jobs = [row for row in old_assessment['jobs']
            if row.get('state') == 'running' or row.get('study') == 'E1_column']
    processes = []
    completions = []
    for row in jobs:
        directory = Path(row['directory'])
        status_path = directory / ('status.json' if row['study'] == 'E1_column' else 'worker_status.json')
        status = read(status_path)
        if status['state'] == 'completed':
            assert status.get('episodes') == 300 and Path(status['result_file']).is_file()
            completions.append(dict(study=row['study'], job=row['job'], finished_at=status['finished_at'],
                                    result_file=status['result_file']))
            continue
        assert status['state'] == 'running', (row['job'], status)
        pid = int(status['pid'] if row['study'] == 'E1_column' else row['pid'])
        process = psutil.Process(pid)
        previous = identities[pid]
        previous_time = datetime.fromisoformat(previous['created_utc'].replace('Z', '+00:00')).timestamp()
        assert abs(process.create_time() - previous_time) < .01
        cpu = process.cpu_times()
        processes.append(dict(study=row['study'], job=row['job'], pid=pid,
                              state=status['state'], created=process.create_time(),
                              cpu_before=cpu.user + cpu.system))
    coordinator = psutil.Process(36308)
    coordinator_start = datetime.fromisoformat(identities[36308]['created_utc'].replace('Z', '+00:00')).timestamp()
    assert abs(coordinator.create_time() - coordinator_start) < .01

    manifest_path = ARCHIVE / 'compressed_storage_20261023/restore_manifest.json'
    assert digest(manifest_path) == MANIFEST_SHA
    manifest = read(manifest_path)
    for i, row in enumerate(manifest['archives'], 1):
        assert not Path(row['original_directory']).exists()
        assert digest(Path(row['compressed'])) == row['compressed_sha256']
        if i % 200 == 0:
            print(f'After removal: {i}/910 retained ZIP hashes verified', flush=True)
    assert (ARCHIVE / 'storage_20261023/restore_manifest.json').is_file()

    checkpoints = []
    for row in jobs:
        if row.get('study') != 'E1_column':
            continue
        directory = Path(row['directory']) / 'training_checkpoints'
        for attempt in range(3):
            index_bytes = (directory / 'latest.json').read_bytes()
            index = json.loads(index_bytes)
            python_path = directory / index['file']
            native = index['native']
            hashes_ok = digest(python_path) == index['sha256']
            for key in ('file', 'project_file'):
                path = Path(native[key])
                assert not path.resolve().is_relative_to((ARCHIVE / 'storage_20261023').resolve())
                hashes_ok = hashes_ok and digest(path) == native[key + '_sha256']
            if (directory / 'latest.json').read_bytes() != index_bytes:
                continue
            assert hashes_ok, row['job']
            assert (directory / 'training_state_0.pkl').is_file()
            assert (directory / 'training_state_1.pkl').is_file()
            checkpoints.append(dict(job=row['job'], completed_steps=index['completed_steps'],
                                    latest_python_and_native_hashes_verified=True,
                                    both_python_slots_present=True))
            break
        else:
            raise RuntimeError(f'Checkpoint index kept changing: {row["job"]}')
    for row in processes:
        process = psutil.Process(row['pid'])
        assert process.create_time() == row['created']
        cpu = process.cpu_times()
        row['cpu_delta_seconds'] = cpu.user + cpu.system - row.pop('cpu_before')
    result = dict(at_utc=datetime.now(timezone.utc).isoformat(),
                  retained_zips_verified=910, historical_native_files_in_zips=3640,
                  raw_folders_removed=910, original_path_manifest_retained=True,
                  immutable_compression_manifest_sha256=MANIFEST_SHA,
                  current_e1_checkpoints=checkpoints, preserved_workers=processes,
                  natural_completions=completions,
                  combined_coordinator_identity_preserved=True,
                  free_bytes=shutil.disk_usage('C:/').free,
                  automatic_retention_enabled=False, live_simulator_calls=0)
    (HERE / 'post_cleanup_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
