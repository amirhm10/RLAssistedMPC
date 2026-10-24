"""Read-only ETA/capacity assessment and immutable late-run completion receipts.

Uses existing handle-free stability/snapshot readers. Never opens a simulator,
signals a process, changes a scientific source, or edits a scientific result.
"""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'change-reports/hourly_run_checks_20261024'))
import collect_hourly as h


def now():
    return datetime.now(timezone.utc).isoformat()


def record_file(path):
    return dict(path=str(path), bytes=h.attributes(path)['size'], sha256=h.digest(path))


def verify_finished(category, report, state, source_hashes):
    destination = report / 'completion.json'
    if destination.exists():
        raise FileExistsError('Completion receipts are immutable; inspect the existing receipt')
    folder = h.OUTPUTS[category]
    paths = dict(status=folder/'status.json', index=folder/'training_checkpoints/latest.json',
                 owned=folder/'owned_aspen_apps.json', manifest=state/'manifest.json',
                 analysis=folder/'analysis/summary.json', capture=folder/'capture/capture_status.json')
    values = h.stable_batch(paths)
    status, index, manifest = values['status'], values['index'], values['manifest']
    accepted = json.loads((report/'acceptance.json').read_text(encoding='utf-8'))
    identity = accepted['production_identity']
    actual = h.process(status['pid'])
    if (status['state'] != 'completed' or status.get('matched_pairs') != 80
            or status.get('failed_pairs') != 0 or status['pid'] != identity['pid']
            or status.get('error') or status.get('owned_apps_remaining')
            or index['completed_cycles'] != 300 or index['completed_steps'] != 120000
            or actual['alive'] is True and actual['created'] == identity['created']
            or manifest['state'] != 'completed' or manifest['job']['state'] != 'completed'):
        raise RuntimeError('Completion/retirement checks differ for ' + category)
    remaining = []
    for saved in values['owned']:
        current = h.process(saved['pid'])
        if current['alive'] is True and current['created'] == saved['created']:
            remaining.append(current)
    if remaining:
        raise RuntimeError('A completed worker still owns a live native application')
    trajectory = record_file(folder/'training.pkl')
    if trajectory['sha256'] != status['trajectory_sha256']:
        raise RuntimeError('Final saved trajectory checksum differs')
    checkpoint = record_file(folder/'training_checkpoints'/index['file'])
    if checkpoint['sha256'] != index['sha256'] or checkpoint['bytes'] != index['bytes']:
        raise RuntimeError('Final Python checkpoint checksum/size differs')
    prepared = json.loads((report/'preparation.json').read_text(encoding='utf-8'))
    if index['source_hashes'] != prepared['pair']['index']['source_hashes']:
        raise RuntimeError('Final scientific checkpoint source map changed')
    native = []
    for key in ('file', 'project_file'):
        checked = record_file(Path(index['native'][key]))
        if checked['sha256'] != index['native'][key+'_sha256']:
            raise RuntimeError('Final native checkpoint checksum differs')
        native.append(checked)
    records = h.completed_records(folder/'capture/decisions.jsonl')
    identities = {(r['channel'], r['window'], r['step']) for r in records}
    if (len(records) != 80 or len(identities) != 80
            or any(r['status'] != 'completed' for r in records)
            or any(v != 20 for v in Counter(r['window'] for r in records).values())
            or values['capture']['state'] != 'completed'
            or values['capture']['saved_decisions'] != 80
            or values['analysis']['total_decisions'] != 80
            or values['analysis']['failed_pairs'] != 0):
        raise RuntimeError('Final matched-pair/capture/analysis evidence differs')
    traces = []
    for row in records:
        path = (folder/'capture'/row['trace']).resolve()
        if not path.is_relative_to((folder/'capture/traces').resolve()):
            raise RuntimeError('Trace path escaped the capture directory')
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('A final trace CRC differs')
        traces.append(record_file(path))
    analysis_files = [record_file(p) for p in sorted((folder/'analysis').iterdir()) if p.is_file()]
    if not (folder/'analysis/summary.csv').stat().st_size:
        raise RuntimeError('Final analysis summary CSV is empty')
    for name, checksum in source_hashes.items():
        if h.digest(ROOT/name) != checksum:
            raise RuntimeError('A sealed source differs: '+name)
    supervisor_error = state/'supervisor_error.json'
    diagnostic = json.loads(supervisor_error.read_text(encoding='utf-8')) if supervisor_error.exists() else None
    receipt = dict(verified=True, at_utc=now(), category=category, output=str(folder),
                   finished_at=status['finished_at'], completed_cycles=300, completed_steps=120000,
                   matched_pairs=80, failed_pairs=0, accepted_production_identity=identity,
                   retired_process_observation=actual, owned_native_remaining=remaining,
                   trajectory=trajectory, latest_checkpoint=index, checkpoint=checkpoint,
                   final_native_files=native, final_capture=values['capture'], traces=traces,
                   analysis_files=analysis_files, analysis_summary_verified=True,
                   empty_prediction_csv_expected_no_prediction_groups=(
                       not values['analysis']['prediction_groups']),
                   all_143_source_hashes_verified=len(source_hashes)==143,
                   supervisor_postcompletion_diagnostic=diagnostic,
                   diagnostic_is_scientific_failure=False, scientific_results_preserved=True,
                   process_signals=0, simulator_calls=0)
    h.atomic_json(destination, receipt)
    return {key:receipt[key] for key in ('verified', 'category', 'finished_at', 'matched_pairs',
                                        'failed_pairs', 'completed_cycles', 'trajectory')}


def estimate():
    folder = h.OUTPUTS['combined']
    data = h.stable_batch(dict(status=folder/'status.json', index=folder/'training_checkpoints/latest.json',
                               sampling=folder/'capture/sampling.json', owned=folder/'owned_aspen_apps.json'))
    records = h.completed_records(folder/'capture/decisions.jsonl')
    sampling = data['sampling']
    done = {(r['channel'], r['step']) for r in records}
    longs = {tuple(row) for row in sampling['long_steps']}
    pending = []
    for site, steps in sampling['selected_steps'].items():
        for step in steps:
            channel = site.split(':')[0]
            if (channel, step) not in done:
                pending.append(dict(channel=channel, step=step, long=(site,step) in longs))
    if len(pending) + len(records) != 320:
        raise RuntimeError('Exact remaining schedule does not reconcile')
    observations = []
    since = datetime(2026,10,24,15,26,tzinfo=timezone.utc).timestamp()
    for row in records:
        path = folder/'capture'/row['trace']
        stamp = h.attributes(path)['mtime']/1e9
        if stamp >= since:
            observations.append(dict(channel=row['channel'],step=row['step'],
                                     branch_steps=2*row['rollout_steps'],timestamp=stamp))
    intervals = [dict(main_steps=b['step']-a['step'],branch_steps=b['branch_steps'],
                      seconds=b['timestamp']-a['timestamp'],channel=b['channel'],step=b['step'])
                 for a,b in zip(observations,observations[1:])]
    if any(row['main_steps'] < 0 or row['seconds'] <= 0 for row in intervals):
        raise RuntimeError('Replay boundary or timestamp inversion in ETA sample')
    ordinary = [r for r in intervals if r['branch_steps']==400]
    extended = [r for r in intervals if r['branch_steps']==1600]
    if len(ordinary) < 8 or not extended:
        raise RuntimeError('Insufficient timing observations for separate training/comparison estimate')
    main_rate, ordinary_cost = statistics.linear_regression(
        [r['main_steps'] for r in ordinary], [r['seconds'] for r in ordinary])
    branch_rate = statistics.median((r['seconds']-main_rate*r['main_steps']-ordinary_cost)/1200
                                    for r in extended)
    pair_overhead = ordinary_cost-400*branch_rate
    if min(main_rate,branch_rate,pair_overhead) < 0:
        raise RuntimeError('Timing model is not physically interpretable')
    main_remaining = 120000-records[-1]['step']-1
    branch_remaining = sum(1600 if r['long'] else 400 for r in pending)
    training_hours = main_remaining*main_rate/3600
    branch_hours = branch_remaining*branch_rate/3600
    overhead_hours = len(pending)*pair_overhead/3600
    serial = training_hours+branch_hours+overhead_hours
    # Counterfactual capacity scenarios, not benchmarked speedup measurements.
    two_branches_overhead_unchanged = training_hours+branch_hours/2+overhead_hours
    two_branches_all_comparison_cost_halved = training_hours+(branch_hours+overhead_hours)/2
    native = h.native_processes()
    owner_ids = {(r['pid'],r['created']) for r in data['owned']}
    active_owned = [v for v in native.values() if (v['pid'],v['created']) in owner_ids]
    actual = h.process(data['status']['pid'])
    if (data['status']['state'] != 'running' or actual['alive'] is not True
            or (actual['pid'],actual['created']) != (15940,134373287822018420)
            or len(active_owned) != len(native)):
        raise RuntimeError('Current combined generation/native ownership needs review')
    return dict(at_utc=now(),completed_pairs=len(records),expected_pairs=320,remaining_pairs=len(pending),
                completed_windows=dict(Counter(r['window'] for r in records)),
                checkpoint_cycles=data['index']['completed_cycles'],latest_captured_step=records[-1]['step'],
                remaining_training_steps_upper_bound=main_remaining,
                remaining_long_pairs=sum(r['long'] for r in pending),remaining_branch_steps=branch_remaining,
                timing_since_utc=datetime.fromtimestamp(since,timezone.utc).isoformat(),
                timing_observations=observations,timing_intervals=intervals,
                timing_model=dict(main_seconds_per_step=main_rate,branch_seconds_per_step=branch_rate,
                    fixed_seconds_per_pair=pair_overhead,normal_pairs=len(ordinary),long_pairs=len(extended),
                    normal_fit_residual_seconds_rms=(statistics.mean(
                        (r['seconds']-main_rate*r['main_steps']-ordinary_cost)**2 for r in ordinary)**0.5)),
                serial_estimate_hours=dict(training=training_hours,branch_rollouts=branch_hours,
                    pair_overhead=overhead_hours,total=serial),
                serial_planning_range_hours=[30,40],planning_range_is_not_a_confidence_interval=True,
                two_branch_projection_hours=dict(fixed_pair_overhead_unchanged=two_branches_overhead_unchanged,
                    all_comparison_cost_halved=two_branches_all_comparison_cost_halved),
                parallel_speedup_benchmarked=False,
                uncertainty=['Only one extended pair in the current-generation interval sample.',
                    'Trace timestamps include native restore/checkpoint/export/serialization costs.',
                    'Training coefficient is inferred from short within-window step gaps and extrapolated.',
                    'The long no-capture training interval can have a different rate.',
                    'Markov/residual retirement may improve available resources.',
                    'Future Aspen failures or a retrofit/replay delay are not included.'],
                current_identity=actual,native_apps=active_owned,memory=h.memory(),
                free_disk_bytes=h.shutil.disk_usage(ROOT.anchor).free,
                execution=dict(policy_and_supervisor_branches='serial',main_training='one coupled four-learner trajectory',
                    hot_worker_attachment=False,retrofitting_requires='new isolated worker pool and guarded combined checkpoint migration'),
                jobs_launched=0,process_signals=0,simulator_calls=0,scientific_source_edits=0,queue_plan_edits=0)


def main():
    destination = HERE/'assessment.json'
    if destination.exists():
        raise FileExistsError('Assessment is an immutable observation; use a new report for refresh')
    anchors = json.loads((h.HERE/'reconciliation_inputs.json').read_text(encoding='utf-8'))
    sources = anchors['metadata']['horizon']['source_sha256']
    completions = []
    for category,report_name,state_name in (
        ('markov','markov_native_recovery_20261024','20261024_markov_native_retry'),
        ('residual','residual_native_retry2_20261024','20261024_residual_native_retry2')):
        completions.append(verify_finished(category,ROOT/'change-reports'/report_name,
                           ROOT/'Distillation/Results/state_recovery'/state_name,sources))
    result = estimate()
    result['newly_verified_completions'] = completions
    h.atomic_json(destination,result)
    print(json.dumps({k:result[k] for k in ('at_utc','completed_pairs','remaining_pairs','remaining_long_pairs',
        'timing_model','serial_estimate_hours','two_branch_projection_hours','memory',
        'newly_verified_completions')},indent=2))


if __name__ == '__main__':
    main()
