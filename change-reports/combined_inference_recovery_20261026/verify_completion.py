"""Verify the finished accepted campaign; never launch or signal a process.

Only operational receipts and private checkpoint-read aliases are written.
Scientific sources, result bytes, native applications and queues are untouched.
"""
from collections import Counter
from pathlib import Path
import json
import pickle
import sys
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOUR = ROOT / 'change-reports/hourly_run_checks_20261024'
sys.path.insert(0, str(ROOT / 'change-reports/combined_parallel_20261024'))
import parallel_branches as p
from aspen_retention import Verifier, still_current

STATE = ROOT / 'Distillation/Results/state_recovery/20261026_combined_metadata_retry'
OUTPUT = p.h.OUTPUTS['combined']


def verify():
    import numpy as np

    target = HERE / 'completion.json'
    if target.exists():
        raise FileExistsError('Do not replace a completed verification receipt')
    side = STATE / 'retired_side_retention'
    paths = dict(manifest=STATE / 'manifest.json', status=OUTPUT / 'status.json',
        pool=STATE / 'parallel_pool_0.json', owners=OUTPUT / 'owned_aspen_apps.json',
        side_status=side / 'status.json', side_identity=side / 'identity.json',
        side_scheduled=side / 'scheduled.json', anchors=p.h.RETENTION / 'checkpoint_hashes.json',
        retention=p.h.RETENTION / 'status.json',
        viewer=ROOT / 'Distillation/Results/live_recovery/latest.json')
    v = p.h.stable_batch(paths)
    accepted = p.closed_json(HERE / 'acceptance.json')
    expected = accepted['production_identity']
    worker = p.closed_json(STATE / 'worker_0_identity.json')
    assert worker['identity'] == expected
    assert v['manifest']['state'] == v['manifest']['job']['state'] == v['status']['state'] == 'completed'
    assert v['manifest']['job']['exit_code'] == 0 and not v['manifest']['job'].get('error')
    assert v['status']['pid'] == expected['pid'] and not p.same_generation(expected, p.h.process(expected['pid']))
    assert v['status']['matched_pairs'] == 320 and v['status']['failed_pairs'] == 0
    assert v['status']['owned_apps_remaining'] == []
    assert Path(v['viewer']['additional_manifests'][-1]).resolve() == (STATE / 'manifest.json').resolve()
    assert not p.h.native_processes()
    for name, checksum in worker['source_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else HERE / name
        assert p.h.digest(path) == checksum, name

    audit = p.closed_json(HOUR / 'audit_20261026T0426.json')
    assert audit['completed_pairs'] == 640 and audit['remaining_scientific_runs'] == 0
    assert audit['sources_verified'] == 143 and audit['observations_needing_review'] == []
    assert audit['coordinator_state'] == audit['horizon_manifest_state'] == 'completed'
    assert audit['column_repetitions']['completed'] == audit['column_repetitions']['total'] == 50
    assert all(row['state'] == 'completed' and row['checkpoint_cycles'] == 300 for row in audit['E1'])

    pair_path = HERE / 'completion_checkpoint_pair.json'
    if pair_path.exists():
        pair = p.closed_json(pair_path)
        # JSON stores Win32 version tuples as lists; restore the reader's type.
        pair['layout']['index'] = tuple(pair['layout']['index'])
        assert still_current(pair), 'The previously verified final pair changed'
    else:
        cache = HERE / ('completion_verifier_' + uuid.uuid4().hex[:8])
        cache.mkdir()
        p.h.atomic_json(cache / 'checkpoint_hashes.json', v['anchors'])
        pair = Verifier(cache).pair(OUTPUT / 'training_checkpoints', ROOT / 'Distillation/Results')
    assert [x['completed_steps'] for x in pair['checkpoints']] == [120000, 119600]
    assert pair['index']['completed_cycles'] == 300 and len(pair['index']['agents']) == 4
    assert all(x['train_steps'] == 115600 and x['replay_size'] == 40000 for x in pair['index']['agents'].values())
    for point in pair['checkpoints']:
        key = str(OUTPUT / 'training_checkpoints') + f'::{point["slot"]}::{point["completed_steps"]}'
        assert v['anchors'][key] == point['sha256']
    assert still_current(pair)
    if not pair_path.exists():
        p.immutable(pair_path, pair)

    records = p.h.completed_records(OUTPUT / 'capture/decisions.jsonl')
    assert len(records) == 320 and all(x['status'] == 'completed' for x in records)
    assert len({(x['channel'], x['window'], x['step']) for x in records}) == 320
    cells = Counter((x['channel'], x['window']) for x in records)
    assert len(cells) == 16 and all(n == 20 for n in cells.values())
    known = {}
    audits = [p.closed_json(HOUR / 'audit.json')]
    audits += [p.closed_json(x) for x in HOUR.glob('audit_*.json')]
    for previous in sorted(audits, key=lambda x: x.get('at_utc', '')):
        traces = previous.get('new_trace_hashes', []) + previous.get('final_check', {}).get('new_trace_hashes', [])
        traces += [x for row in previous.get('E1', []) for x in row.get('trace_files', [])]
        for row in traces:
            if row.get('sha256'):
                known[row['path']] = row['sha256']
    hashes = []
    for row in records:
        trace = (OUTPUT / 'capture' / row['trace']).resolve()
        assert trace.is_relative_to((OUTPUT / 'capture').resolve())
        with zipfile.ZipFile(trace) as archive:
            assert archive.testzip() is None
        checksum = p.h.digest(trace)
        assert known[str(trace)] == checksum, str(trace)
        hashes.append(dict(channel=row['channel'], window=row['window'], step=row['step'], path=str(trace), sha256=checksum))

    trajectory = OUTPUT / 'training.pkl'
    trajectory_hash = p.h.digest(trajectory)
    assert trajectory_hash == v['status']['trajectory_sha256']
    with p.h.open_read(trajectory) as stream:
        result = pickle.load(stream)
    assert result['nFE'] == 120000 and result['time_in_sub_episodes'] == 400
    assert result['training_profile_name'] == v['status']['profile']
    expected_shapes = dict(y=(120001, 2), u=(120000, 2), avg_rewards=(300,),
                           rewards_step=(120000,), yhat=(2, 120000), xhatdhat=(7, 120001))
    arrays = {}
    for name, shape in expected_shapes.items():
        value = np.asarray(result[name])
        assert value.shape == shape and np.isfinite(value).all(), name
        arrays[name] = dict(shape=list(value.shape), all_finite=True)
    assert set(result['replay_buffer_snapshots']) == {'horizon', 'markov', 'weights', 'residual'}
    del result

    pool = Path(v['pool']['directory'])
    decisions = p.h.completed_records(pool / 'decisions.jsonl')
    assert len(decisions) == 7
    assert all(r['full_python_state_before'] == r['full_python_state_after'] and r['source_map_unchanged'] for r in decisions)
    children, models = [], []
    for case in (15, 17):
        child = pool / ('case' + str(case))
        identity = p.closed_json(child / 'identity.json')
        actual = p.h.process(identity['pid'])
        requests = list(child.glob('request_*.json'))
        assert not p.same_generation(identity, actual)
        assert len(requests) == 7 and all((child / ('done_' + f.name)).is_file() for f in requests)
        children.append(dict(case=case, expected=identity, actual=actual, requests=7, responses=7, pending=0))
        events = p.h.completed_records(child / 'native_process_recovery.jsonl')
        opened = {(r['current']['pid'], r['current']['created']): r for r in events if r.get('event') == 'opened_fresh_application'}
        for row in decisions[-2:]:
            matching = []
            for batch in row['batches'].values():
                for branch in batch['branches'].values():
                    native = branch['native_identity']
                    event = opened.get((native['pid'], native['created']))
                    if event:
                        model = Path(event['model'])
                        assert model.is_relative_to(child / 'working') and model.is_file()
                        matching.append(dict(case=case, channel=row['channel'], step=row['step'], model=str(model), bytes=p.h.attributes(model)['size']))
            assert matching
            models.extend(matching)

    assert v['side_status']['state'] == 'completed'
    assert not p.same_generation(v['side_identity']['identity'], p.h.process(v['side_identity']['identity']['pid']))
    assert p.h.digest(ROOT / 'change-reports/combined_parallel_20261024/retired_side_archive.py') == v['side_identity']['source_sha256'] == v['side_scheduled']['source_sha256']
    assert p.h.digest(Path(v['side_scheduled']['wrapper'])) == v['side_scheduled']['wrapper_sha256']
    archive_root = Path(v['side_status']['archive_root'])
    pruned = [p.closed_json(x) for x in archive_root.glob('*.pruned.json')]
    assert len(pruned) == v['side_status']['archived_working_folders'] == 10
    archive_sizes = []
    for row in pruned:
        archive = Path(row['verified_archive'])
        receipt = p.closed_json(archive.with_suffix('.json'))
        assert receipt['verified'] and receipt['scientific_results_or_durable_inputs_removed'] == 0
        assert p.h.attributes(archive)['size'] == receipt['archive_bytes']
        archive_sizes.append(receipt['archive_bytes'])
    latest_archive = max(pruned, key=lambda x: x['at_utc'])
    path = Path(latest_archive['verified_archive'])
    assert p.h.digest(path) == latest_archive['archive_sha256']
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    assert v['retention']['state'] == 'running'
    assert p.same_generation(audit['retention']['identity'], p.h.process(v['retention']['pid']))
    assert all(row['state'] == 'two complete checkpoints verified' for row in audit['retention']['jobs'])
    analysis = [dict(path=str(x), bytes=p.h.attributes(x)['size'], sha256=p.h.digest(x)) for x in sorted((OUTPUT / 'analysis').glob('*')) if x.is_file()]
    assert len(analysis) >= 3 and all(x['bytes'] for x in analysis)
    receipt = dict(verified=True, at_utc=p.now(), category='combined', output=str(OUTPUT),
        finished_at=v['manifest']['job']['finished_at'], current_manifest=str(STATE / 'manifest.json'),
        exit_code=0, accepted_identity=expected, retired_process=p.h.process(expected['pid']),
        completed_cycles=300, completed_steps=120000, matched_pairs=320, failed_pairs=0,
        balanced_channel_window_counts=[dict(channel=k[0], window=k[1], count=n) for k,n in sorted(cells.items())],
        all_320_current_trace_hashes_match_prior_verified_generations=True, trace_files=hashes,
        trajectory=dict(path=str(trajectory), bytes=p.h.attributes(trajectory)['size'], sha256=trajectory_hash, arrays=arrays),
        checkpoint_pair=str(HERE / 'completion_checkpoint_pair.json'), checkpoint_steps=[120000, 119600],
        four_final_agent_buffers_and_train_counters=pair['index']['agents'],
        current_parallel_pairs=len(decisions), all_parallel_state_and_source_guards_passed=True,
        children=children, remaining_native_applications=[], side_status=v['side_status'],
        final_side_archives=len(pruned), final_side_archive_bytes=sum(archive_sizes),
        latest_side_archive_sha256_crc_passed=True, newest_two_unpacked_working_models=models,
        main_retention_live=True, combined_retention_deferral_cleared=True,
        analysis_files=analysis, final_campaign_audit=str(HOUR / 'audit_20261026T0426.json'),
        total_column_E1_pairs=640, completed_column_repetitions=50,
        all_authorized_campaign_jobs_completed=True, all_143_scientific_source_seals_verified=True,
        viewer_current_manifest_verified=True, rpc_cleanup_diagnostics_do_not_override_verified_outputs=True,
        original_results_replays_backups_pins_archives_retained=True,
        frozen_reviewer_analysis_and_manuscript_unchanged=True, simulator_calls=0, process_signals=0,
        memory=p.h.memory(), free_disk_bytes=p.shutil.disk_usage(ROOT.anchor).free)
    p.immutable(target, receipt)
    print(json.dumps({k:receipt[k] for k in ('verified','at_utc','finished_at','completed_cycles','completed_steps','matched_pairs','failed_pairs','checkpoint_steps','final_side_archives','total_column_E1_pairs','all_authorized_campaign_jobs_completed','free_disk_bytes')}, indent=2))


if __name__ == '__main__':
    verify()
