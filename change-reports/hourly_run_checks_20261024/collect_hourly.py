"""Collect read-only hourly evidence; never launch, stop or repair a worker.

Run from the repository with the base scientific interpreter and -B. Windows
snapshot aliases/process queries need the ordinary user's unsandboxed context,
not administrator rights. Diagnose flagged observations before taking action.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import psutil
from aspen_retention import atomic_json, attributes, digest, open_read, read_json, version
from column_review.common import process_identity
from run_column_e1_process_recovery import memory

HERE = Path(__file__).resolve().parent
E1 = ROOT / 'Distillation/Results/state_recovery/20261021_remaining_active/e1'
OUTPUTS = {name: E1 / name / 'a1' for name in ('horizon', 'weights', 'markov', 'residual')}
OUTPUTS['combined'] = ROOT / 'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1'
HORIZON = ROOT / 'Distillation/Results/state_recovery/20261023_horizon_native_process'
RETENTION = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\aspen_retention_20261023')


def stable_batch(paths):
    result = {}
    for _ in range(6):
        pending = {key: path for key, path in paths.items() if key not in result}
        if not pending:
            break
        stamps = {key: version(path) for key, path in pending.items() if path.is_file()}
        time.sleep(2.2)
        for key, stamp in stamps.items():
            path = pending[key]
            if version(path) != stamp:
                continue
            try:
                value = read_json(path)
                if version(path) == stamp:
                    result[key] = value
            except (OSError, ValueError):
                continue
    missing = set(paths) - set(result)
    if missing:
        raise RuntimeError('Metadata unavailable or still changing: ' + ', '.join(sorted(missing)))
    return result


def process(pid):
    result = process_identity(pid)
    if result['alive'] is True:
        try:
            proc = psutil.Process(pid)
            mem = proc.memory_info()
            result.update(cpu_seconds=sum(proc.cpu_times()[:2]),
                          private_bytes=getattr(mem, 'private', 0), rss_bytes=mem.rss)
        except psutil.NoSuchProcess:
            return process_identity(pid)
    return result


def native_processes():
    result = {}
    for proc in psutil.process_iter(['name']):
        if (proc.info['name'] or '').lower() == 'aspenmodeler.exe':
            result[proc.pid] = process(proc.pid)
    return result


def completed_records(path):
    with open_read(path) as stream:
        lines = stream.read().splitlines(keepends=True)
    # An incomplete trailing append is not a failed comparison.
    return [json.loads(line) for line in lines if line.endswith(b'\n') and line.strip()]


def previous_evidence():
    # These are closed audit artifacts, never live shared queue manifests.
    base = json.loads((HERE / 'audit.json').read_text(encoding='utf-8'))
    traces = {trace['path'] for row in base['E1'] for trace in row['trace_files']}
    anchors = json.loads((HERE / 'reconciliation_inputs.json').read_text(encoding='utf-8'))
    for path in sorted(HERE.glob('audit_*.json')):
        saved = json.loads(path.read_text(encoding='utf-8'))
        traces.update(row['path'] for row in saved.get('new_trace_hashes', []))
    return base, traces, anchors['metadata']['horizon']['source_sha256']


def collect(sample_seconds):
    baseline, known_traces, sources = previous_evidence()
    paths = {}
    for name, folder in OUTPUTS.items():
        for filename in ('status.json', 'training_checkpoints/latest.json', 'owned_aspen_apps.json'):
            paths[name + '/' + filename] = folder / filename
    paths.update(retention=RETENTION / 'status.json', horizon_manifest=HORIZON / 'manifest.json',
                 horizon_supervisor=HORIZON / 'supervisor_identity.json',
                 coordinator=ROOT / 'Distillation/Results/review_recovery/20261020_queue_case12/coordinator_status.json')
    metadata = stable_batch(paths)
    before = {name: process(metadata[name + '/status.json']['pid']) for name in OUTPUTS}
    native_before = native_processes()
    memory_before = memory()
    time.sleep(sample_seconds)
    rows, new_traces, logs, owners, issues = [], [], {}, {}, []
    for name, folder in OUTPUTS.items():
        status = metadata[name + '/status.json']
        checkpoint = metadata[name + '/training_checkpoints/latest.json']
        current = process(status['pid'])
        baseline_identity = next(row['identity'] for row in baseline['E1'] if row['category'] == name)
        identity_matches = all(current.get(key) == baseline_identity.get(key)
                               for key in ('pid', 'created', 'alive'))
        records = completed_records(folder / 'capture/decisions.jsonl')
        unique = len(records) == len({(r['channel'], r['window'], r['step']) for r in records})
        bad_pairs = [r['status'] for r in records if r['status'] != 'completed']
        for record in records:
            path = (folder / 'capture' / record['trace']).resolve()
            if not path.is_relative_to((folder / 'capture').resolve()):
                raise RuntimeError('Trace path escaped its capture folder')
            size = attributes(path)['size']
            if not size:
                raise RuntimeError('Empty saved trace: ' + str(path))
            if str(path) not in known_traces:
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        raise RuntimeError('Trace CRC mismatch: ' + str(path))
                new_traces.append(dict(category=name, path=str(path), bytes=size, sha256=digest(path)))
        for recorded in metadata[name + '/owned_aspen_apps.json']:
            actual = process_identity(recorded['pid'])
            if actual['alive'] is True and actual['created'] == recorded['created']:
                owners[recorded['pid']] = name
        log = HORIZON / 'logs/production.log' if name == 'horizon' else ROOT / (
            'Distillation/Results/state_recovery/20261021_e1_budget20/logs/combined.log' if name == 'combined'
            else 'Distillation/Results/state_recovery/20261023_e1_native_process/logs/' + name + '.log')
        with open_read(log) as stream:
            stream.seek(max(0, attributes(log)['size'] - 16000))
            tail = stream.read().decode('utf-8', errors='replace')
        cycles = re.findall(r'Sub_Episode:\s*(\d+)', tail)
        logs[name] = dict(path=str(log), version=version(log), last_cycle=int(cycles[-1]) if cycles else None,
                          tail=tail.splitlines()[-7:])
        delta = current.get('cpu_seconds', 0) - before[name].get('cpu_seconds', 0)
        row = dict(category=name, directory=str(folder), state=status['state'], error=status.get('error'),
                   before=before[name], after=current, identity_matches_initial_worker=identity_matches,
                   cpu_delta_seconds=delta, checkpoint_cycles=checkpoint['completed_cycles'],
                   checkpoint_steps=checkpoint['completed_steps'], completed_pairs=len(records) - len(bad_pairs),
                   expected_pairs=status['expected_matched_pairs'], last_pair_step=records[-1]['step'],
                   last_pair_episode=records[-1]['episode'], unique=unique, other_pair_states=bad_pairs)
        rows.append(row)
        if status['state'] == 'running' and current['alive'] is not True:
            issues.append(name + ': current worker is not confirmed live; inspect before recovery')
        if status['state'] == 'running' and not identity_matches:
            issues.append(name + ': worker identity differs; reconcile current recovery attempt')
        if status['state'] == 'failed' or not unique or bad_pairs:
            issues.append(name + ': failure or capture inconsistency needs diagnosis')
    native = []
    for pid, current in native_processes().items():
        prior = native_before.get(pid)
        current.update(owner=owners.get(pid), cpu_delta_seconds=(
            current.get('cpu_seconds', 0) - prior.get('cpu_seconds', 0)
            if prior and prior['created'] == current['created'] else None))
        native.append(current)
        if current['owner'] is None:
            issues.append('Native PID ' + str(pid) + ': recheck ownership; it may have just been created')
    for row in rows:
        if row['state'] == 'running' and row['cpu_delta_seconds'] <= 0:
            if not any(app['owner'] == row['category'] and (app['cpu_delta_seconds'] or 0) > 0 for app in native):
                issues.append(row['category'] + ': short CPU sample quiet; compare longer progress before diagnosing a hang')
    for row in baseline['column_repetitions']['results']:
        if attributes(Path(row['result_file']))['size'] != row['result_bytes']:
            raise RuntimeError('Previously completed repetition output changed size')
    for filename, checksum in sources.items():
        if digest(ROOT / filename) != checksum:
            raise RuntimeError('Sealed source differs: ' + filename)
    retention = metadata['retention']
    retention_identity = process(retention['pid'])
    if retention_identity['alive'] is not True:
        issues.append('Retention process is not confirmed live')
    onedrive = [proc.info for proc in psutil.process_iter(['pid', 'name'])
                if (proc.info['name'] or '').lower() in ('onedrive.exe', 'onedrive.sync.service.exe', 'filecoauth.exe')]
    result = dict(at_utc=datetime.now(timezone.utc).isoformat(), E1=rows,
                  completed_pairs=sum(row['completed_pairs'] for row in rows), expected_pairs=640,
                  pairs_by_category={row['category']: row['completed_pairs'] for row in rows},
                  remaining_scientific_runs=sum(row['state'] != 'completed' for row in rows),
                  new_trace_hashes=new_traces, previous_trace_hashes_in='audit.json and prior audit_*.json',
                  column_repetitions=dict(completed=50, total=50, all_prior_outputs_present_same_size=True,
                                          original_hashes_in='audit.json'),
                  status_records={name: metadata[name + '/status.json'] for name in OUTPUTS},
                  horizon_manifest_state=metadata['horizon_manifest']['state'],
                  horizon_supervisor=process(metadata['horizon_supervisor']['pid']),
                  coordinator_state=metadata['coordinator']['state'],
                  retention=dict(state=retention['state'], identity=retention_identity,
                                 updated_at=retention['updated_at'], totals=retention['totals'],
                                 jobs=[dict(category=j['label'], state=j['state'], reason=j.get('reason'),
                                            checkpoint_steps=[p['completed_steps'] for p in j.get('checkpoints', [])])
                                       for j in retention['jobs']]),
                  native_apps=native, logs=logs, sources_verified=len(sources), memory_before=memory_before,
                  memory=memory(), free_disk_bytes=shutil.disk_usage(ROOT.anchor).free,
                  OneDrive_processes=onedrive, observations_needing_review=issues,
                  actions=dict(jobs_launched=0, jobs_stopped=0, process_signals=0,
                               live_simulator_calls=0, scientific_source_edits=0, queue_plan_edits=0, data_deletions=0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-seconds', type=int, default=12, choices=range(10, 31))
    args = parser.parse_args()
    target = args.output.resolve()
    if target.parent != HERE or target.exists():
        raise RuntimeError('Use a new audit filename inside this report folder')
    result = collect(args.sample_seconds)
    atomic_json(target, result)
    print(json.dumps({key: result[key] for key in (
        'at_utc', 'completed_pairs', 'pairs_by_category', 'remaining_scientific_runs', 'memory',
        'free_disk_bytes', 'coordinator_state', 'retention', 'observations_needing_review')}, indent=2), flush=True)
    for row in result['E1']:
        print(json.dumps({key: row[key] for key in ('category', 'state', 'checkpoint_cycles',
              'completed_pairs', 'expected_pairs', 'cpu_delta_seconds')}, sort_keys=True), flush=True)
    print('Verified new trace files:', len(result['new_trace_hashes']), flush=True)


if __name__ == '__main__':
    main()
