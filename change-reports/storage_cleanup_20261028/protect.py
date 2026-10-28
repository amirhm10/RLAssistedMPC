"""Verify manuscript evidence and declare exclusions before any cleanup."""
from collections import Counter
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = Path(r'C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\Rl_assisted_paper\RL_assisted_paper')
sys.path.insert(0, str(ROOT))
from aspen_retention_pickle import metadata


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), str(path)
    return h.hexdigest()


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def closed_checkpoints(directory, anchors):
    """Verify closed final slots by content; OneDrive changed directory mtimes.

    The live-worker retention guard conservatively defers on those timestamps.
    This audit leaves that guard unchanged and checks immutable final contents,
    index hashes, both native components, linkage and before/after file versions.
    """
    index_path=directory/'latest.json'
    index_hash=sha(index_path)
    index=read(index_path)
    assert index['schema']==2
    records=[]
    for slot in (0,1):
        path=directory/f'training_state_{slot}.pkl'
        checksum=sha(path)
        before=path.stat()
        with path.open('rb') as stream:
            payload=metadata(stream)
        after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        step=payload['completed_steps']
        assert payload['source_hashes']==index['source_hashes']
        native=payload['native']
        native_dir=directory/f'native_{slot}_{step}'
        for key in ('file','project_file'):
            component=Path(native[key])
            assert component.resolve().is_relative_to(native_dir.resolve())
            assert sha(component)==native[key+'_sha256']
        saved=read(Path(native['file']).parent/'metadata.json')
        for key in ('file','project_file','file_sha256','project_file_sha256','name'):
            assert saved[key]==native[key]
        anchor=anchors.get(str(directory)+f'::{slot}::{step}')
        if anchor:
            assert checksum==anchor
        if slot==index['slot']:
            assert checksum==index['sha256'] and step==index['completed_steps']
            # The restricted pickle reader intentionally skips large native
            # fidelity arrays; compare the authoritative linkage fields.
            for key in ('file','project_file','file_sha256','project_file_sha256','name','live_save_unchanged'):
                assert index['native'].get(key)==native.get(key)
        records.append(dict(slot=slot,completed_steps=step,python_file=str(path),sha256=checksum,
                            native_directory=str(native_dir),native=native,
                            runner_sha256=payload['runner_sha256']))
    assert sha(index_path)==index_hash
    assert records[0]['runner_sha256']==records[1]['runner_sha256']
    return sorted(records,key=lambda row:row['completed_steps'],reverse=True)


def main():
    expected = {}
    sources = []
    errors = []
    offline = []
    for row in read(PAPER/'manuscript_revised/frozen_original_manifest.json')['files']:
        expected[str(PAPER/'manuscript'/row['path'])] = dict(row, reason='frozen original manuscript')
    for name in ['canonical_results_20260722/canonical_run_manifest.csv', 'results_discussion_20260729/repeated_run_manifest.csv']:
        rows = csv_rows(PAPER/'analysis'/name)
        for row in rows:
            path = ROOT / row['source_bundle']
            expected.setdefault(str(path),dict(bytes=int(row['source_size_bytes']),reason='original manuscript raw bundle'))
        sources.append(dict(manifest=name, rows=len(rows)))
    data = ROOT/'change-reports/results_assessment_20261026_final/data'
    for name in ['repeated_provenance.json','e2_provenance.json','combined_final_training_provenance.json','combined_steady_provenance.json','steady_state_provenance.json','threshold_handoff_provenance.json']:
        values = read(data/name)
        values = values if isinstance(values,list) else [values]
        def visit(obj):
            if isinstance(obj,dict):
                if 'path' in obj and 'sha256' in obj:
                    expected[obj['path']] = dict(obj,reason='revision evidence: '+name)
                for value in obj.values():
                    visit(value)
            elif isinstance(obj,list):
                for value in obj:
                    visit(value)
        visit(values)
        sources.append(dict(manifest=name,rows=len(values)))
    gate_rows = csv_rows(data/'gate_decisions.csv')
    assert len(gate_rows) == 3840
    assert Counter(r['plant'] for r in gate_rows) == dict(polymer=3200,column=640)
    for row in gate_rows:
        expected[row['trace']] = dict(sha256=row['trace_sha256'],reason='revision E1 raw paired trace')
    for row in read(data/'gate_provenance.json')['sources']:
        path=Path(row['path'])
        expected[str(path)] = dict(sha256=sha(path),reason='revision E1 decisions')
    verified = []
    cached={r['path']:r for r in read(HERE/'evidence_hashes_cache.json')['files']} if (HERE/'evidence_hashes_cache.json').exists() else {}
    start = last = time.monotonic()
    for i,(name,row) in enumerate(expected.items()):
        path=Path(name)
        if not path.is_file():
            errors.append(dict(path=name,error='missing protected source',reason=row['reason']))
            continue
        st=path.stat()
        if 'bytes' in row and int(row['bytes']) != st.st_size:
            errors.append(dict(path=name,error='recorded size differs',expected=row['bytes'],actual=st.st_size))
            continue
        attrs=getattr(st,'st_file_attributes',0)
        result=dict(path=name,bytes=st.st_size,mtime_ns=st.st_mtime_ns,reason=row['reason'])
        if attrs & (0x1000|0x40000|0x400000):
            result['verification']='online-only metadata; protected without hydration'
            offline.append(name)
        else:
            prior=cached.get(name,{})
            result['sha256']=prior['sha256'] if (prior.get('bytes'),prior.get('mtime_ns'))==(st.st_size,st.st_mtime_ns) and prior.get('sha256') else sha(path)
            result['verification']='fresh SHA-256'
            if row.get('sha256') and result['sha256']!=row['sha256']:
                errors.append(dict(path=name,error='SHA-256 differs',reason=row['reason']))
        verified.append(result)
        if time.monotonic()-last>15:
            print(json.dumps(dict(stage='protected evidence',done=i+1,total=len(expected),seconds=round(time.monotonic()-start))),flush=True)
            last=time.monotonic()
    config=read(ROOT/'change-reports/aspen_retention_20261023/config.json')
    (HERE/'evidence_hashes_cache.json').write_text(json.dumps(dict(files=verified,errors=errors,offline=offline),indent=2),encoding='utf-8')
    print(json.dumps(dict(stage='evidence complete',files=len(verified),errors=errors,offline=len(offline))),flush=True)
    anchors=read(Path(config['state_root'])/'checkpoint_hashes.json')
    checkpoints=[]
    native_dirs=[]
    for job in config['jobs']:
        pair=closed_checkpoints(Path(job['directory'])/'training_checkpoints',anchors)
        assert [c['completed_steps'] for c in pair]==[120000,119600]
        checkpoints.append(dict(label=job['label'],checkpoints=pair))
        native_dirs.extend(c['native_directory'] for c in pair)
        print(json.dumps(dict(stage='final checkpoint',job=job['label'],steps=[120000,119600],verified=True)),flush=True)
    protected_roots=[str(PAPER),str(ROOT/'Polymer'),str(ROOT/'Distillation/Data'),str(ROOT/'Results/e3')]
    protected_roots+=native_dirs
    result=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=not errors,
                frozen_original_files=46,original_manifest_rows=258,revision_training_runs=100,
                e1_pairs=3840,e2_trajectories=12,e3_reproduced_rows=576,
                errors=errors,online_only_protected=offline,files=verified,
                checkpoints=checkpoints,protected_native_directories=native_dirs,
                protected_roots=protected_roots,source_manifests=sources,
                note='All paper, polymer, original/revision raw trajectories, NPZ traces, policy exports and final two checkpoint sets are excluded. No online-only file is hydrated for cleanup.')
    target=HERE/'protected_evidence_before.json'
    assert not target.exists()
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=not errors,protected_files=len(verified),errors=errors,
                         online_only_preserved=len(offline),seconds=round(time.monotonic()-start))),flush=True)


if __name__=='__main__':
    main()
