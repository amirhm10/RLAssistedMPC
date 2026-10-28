"""Verify the two completed, explicitly approved cloud-copy removal batches.

Reads retained archives and protected-source metadata; never deletes or hydrates
online-only source files. Earlier full decompression/member checks remain valid
only if the supporting ZIP's complete SHA-256 still matches.
"""
import datetime
import json
from pathlib import Path
import time

import compact_native as c

HERE = Path(__file__).resolve().parent
CLOUD = HERE / 'cloud_review'
ROOT = HERE.parents[1]
PAPER = ROOT.parent / 'Rl_assisted_paper' / 'RL_assisted_paper'
PREFIX = 'personal/hamedia_mcmaster_ca/Documents/PythonProjects/RL_assisted_MPC/'


def read(name):
    return json.loads((CLOUD / name).read_text(encoding='utf-8'))


def main():
    first = read('deletion_operation.json')
    second = read('additional_deletion_operation.json')
    first_check = read('cloud_verification.json')
    quota = read('quota_final.json')
    plan = read('remaining_removal_plan.json')['candidates']
    before = c.read(HERE / 'protected_evidence_before.json')
    assert before['passed'] and first_check['all_archive_representations_retained']
    assert first_check['approved_files'] == 211
    assert first_check['all_4271_protected_file_versions_unchanged_after_cloud_cleanup']
    for operation, count in ((first, 211), (second, 354)):
        assert operation['approved_count'] == count
        assert len(set(operation['approved_paths'])) == count
        assert operation['first_stage_completed'] == count
        assert operation['second_stage_completed'] == count
        assert operation['permanent_completion_verified']
        assert operation['second_stage_empty_verified']
    assert first['user_action_time_confirmation'] == 'Permanently remove these 211 copies'
    assert second['user_action_time_confirmation'] == 'Permanently remove these additional 354 copies'
    assert set(first['approved_paths']).isdisjoint(second['approved_paths'])
    assert set(first['second_stage_paths']) | set(first['two_paths_not_initially_present_in_second_stage']) == set(first['approved_paths'])
    assert set(second['second_stage_paths']) == set(second['approved_paths'])
    assert second['second_stage_batches'] == [351, 3]
    assert not second['remaining_approved_paths']
    assert len(plan) == 354
    assert {row['path'] for row in plan} == set(second['approved_paths'])
    assert sum(row['bytes'] for row in plan) == second['approved_bytes']
    retained = {}
    protected_dirs = [Path(p).resolve() for p in before['protected_native_directories']]
    for row in plan:
        assert row['kind'] == 'file' and row['path'].endswith('.dynf')
        assert row['path'].startswith(PREFIX)
        source = (ROOT / row['path'][len(PREFIX):]).resolve()
        assert source.is_relative_to(ROOT.resolve())
        assert not source.exists(), f'Previously compacted path reappeared: {source}'
        assert all(not source.is_relative_to(p) for p in protected_dirs)
        prior = retained.setdefault(row['archive'], row['archive_sha256'])
        assert prior == row['archive_sha256']
    last = time.monotonic()
    for i, (name, expected) in enumerate(retained.items(), 1):
        assert c.digest(Path(name)) == expected, f'Retained ZIP changed: {name}'
        if time.monotonic() - last > 15:
            print(json.dumps({'stage': 'post-cloud archive hashes', 'done': i, 'total': len(retained)}), flush=True)
            last = time.monotonic()
    for row in before['files']:
        stat = Path(row['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']), row['path']
    frozen = c.read(PAPER / 'manuscript_revised/frozen_original_manifest.json')
    assert len(frozen['files']) == 46
    for row in frozen['files']:
        assert c.digest(PAPER / 'manuscript' / row['path']) == row['sha256']
    result = dict(
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        passed=True, errors=[],
        additional_approved_files=354, additional_approved_bytes=second['approved_bytes'],
        additional_first_stage_deleted=354, additional_second_stage_deleted=354,
        additional_second_stage_batches=[351, 3],
        additional_retained_archive_files_freshly_rehashed_after_cloud_cleanup=len(retained),
        all_4271_protected_file_versions_unchanged_after_both_batches=True,
        frozen_original_hashes_verified_after_both_batches=46,
        total_approved_files=565,
        total_approved_bytes=first['approved_bytes'] + second['approved_bytes'],
        total_first_stage_deleted=565, total_second_stage_deleted=565,
        second_stage_empty_verified=True, unrelated_recycle_bin_items_removed=0,
        all_archive_representations_retained=True,
        quota_before_gb_free=0, quota_after_first_batch_gb_free=19.83,
        quota_after_gb_free=quota['quota_after_gb_free'],
        quota_total_gb=quota['quota_total_gb'], quota_text=quota['quota_text'],
        quota_observed_at_utc=quota['at_utc'],
        first_batch_verification='cloud_verification.json',
        action_time_confirmations=[first['user_action_time_confirmation'], second['user_action_time_confirmation']],
        note='Quota is the observed live value; raw bytes removed are recorded separately. No further cloud items are authorized.'
    )
    c.atomic(CLOUD / 'additional_cloud_verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
