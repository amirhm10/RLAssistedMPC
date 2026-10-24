"""Isolated, owned Aspen branch workers; no training or source edits here.

The parent synchronously waits for both counterfactuals. Each worker imports a
frozen portable decision using the existing native and controller code. Files
are unique immutable requests/responses; all input/working/result data remain.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'change-reports/hourly_run_checks_20261024'))
import collect_hourly as h

PYTHON = ROOT/'change-reports/recovery_20261021/runtime/env/Scripts/python.exe'
CASES = (15, 17)
MINIMUM_HEADROOM = 6*2**30
MINIMUM_DISK = 32*2**30


def now():
    return datetime.now(timezone.utc).isoformat()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Immutable operational receipt already exists: '+str(path))
    h.atomic_json(path, value)


def closed_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def same_generation(expected, actual):
    return (actual.get('alive') is True and
            all(expected.get(k) == actual.get(k) for k in ('pid', 'created')))


def guard(config, *, headroom=True):
    if config.get('case') not in CASES or config.get('category') != 'combined':
        raise RuntimeError('Only the two approved combined branch cases are allowed')
    if not same_generation(config['parent_identity'], h.process(config['parent_identity']['pid'])):
        raise RuntimeError('This branch worker parent has retired or changed generation')
    protected = config['protected_main']
    if not same_generation(protected, h.process(protected['pid'])):
        raise RuntimeError('The protected advancing combined worker changed generation')
    if headroom and h.memory()['headroom_bytes'] < MINIMUM_HEADROOM:
        raise RuntimeError('Six GiB commitment headroom is required')
    if shutil.disk_usage(ROOT.anchor).free < MINIMUM_DISK:
        raise RuntimeError('Keep at least 32 GiB disk headroom')
    for key in ('science_hashes', 'adapter_hashes'):
        for filename, checksum in config[key].items():
            path = Path(filename) if Path(filename).is_absolute() else ROOT/filename
            if h.digest(path) != checksum:
                raise RuntimeError('A sealed dependency differs: '+filename)
    if h.digest(Path(config['canonical_model'])) != config['canonical_sha256']:
        raise RuntimeError('Reserved canonical case checksum differs')


def safe_read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return h.stable_batch({'value':path})['value']


def snapshot_sources(config):
    """Retain symbolic checkpoint protocol seals as well as file hashes."""
    sources=config.get('snapshot_source_hashes',{})
    if 'column_e1_budget_protocol' not in sources:
        raise RuntimeError('The frozen snapshot requires its budget protocol seal')
    if any(name in sources and sources[name]!=checksum for name,checksum in config['science_hashes'].items()):
        raise RuntimeError('Frozen snapshot file/source seals conflict')
    return sources


def restore_log_redirect(original, directory):
    """Change only audit-log destinations, never input snapshots or tolerances."""
    def restore(native):
        previous = native.snapshot.path
        native.snapshot.path = Path(directory)/'native_restore_audit.snp'
        try:
            return original(native)
        finally:
            native.snapshot.path = previous
    return restore


def one_branch(request, config, worker_directory, plant):
    import numpy as np
    from review_experiments.e1.snapshots import load_snapshot
    from review_experiments.e1.runtime import clone_environment, restore_rng, rng_state
    from review_experiments.e1.study import run_branch, json_safe
    import column_e1.runtime as runtime
    from column_e1.runtime import (DetachedColumn, COM_FIELDS, copy_column,
                                   digest_state, integrity_payload)
    source = request['source']
    if source not in ('policy', 'supervisor') or request['steps'] not in (20, 200, 800):
        raise ValueError('Only an original source and bounded/full original horizons are allowed')
    snapshot = Path(request['snapshot'])
    if h.digest(snapshot) != request['snapshot_sha256']:
        raise RuntimeError('Frozen input snapshot checksum differs')
    target = Path(request['output']).resolve()
    if not target.is_relative_to(Path(worker_directory).resolve()) or target.exists():
        raise RuntimeError('Use a new output owned by this branch worker')
    target.mkdir()
    started = time.monotonic()
    payload, compiled, detached = load_snapshot(snapshot, scientific_hashes=snapshot_sources(config))
    if payload['category'] != 'combined' or payload['continuation'] != 'gated':
        raise RuntimeError('Frozen decision changed the combined/gated protocol')
    exported = detached['system']
    if not isinstance(exported, DetachedColumn) or not hasattr(exported, 'native_metadata'):
        raise RuntimeError('A detached native snapshot is required')
    metadata = exported.native_metadata
    if Path(metadata['project_file']).name != Path(plant.path).name:
        raise RuntimeError('Native document basename changed')
    metadata_before = digest_state(metadata)
    frozen_before = digest_state(integrity_payload(detached,compiled))
    immutable(target/'request.json',request)
    original_restore = runtime.E1NativeSnapshot.restore
    initial_rng = rng_state()
    try:
        with patch.object(runtime.E1NativeSnapshot,'restore',restore_log_redirect(original_restore,target)):
            primes = runtime.import_native(plant,metadata)
        def attach(saved,memo):
            branch = copy_column(plant,{})
            memo[id(saved)] = branch
            branch.__dict__.update(deepcopy({key:value for key,value in vars(saved).items()
                if key not in COM_FIELDS and key not in {'native_metadata','native_snapshot'}},memo))
            return branch
        with runtime.column_copying(),patch.object(DetachedColumn,'__deepcopy__',attach,create=True):
            environment = clone_environment(detached,compiled)
            restore_rng(payload['rng'])
            result = run_branch(environment,compiled,compiled.sites[payload['site']],payload['record'],
                                source,request['steps'],payload['continuation'])
        if result['state'] != 'completed' or result['completed_steps'] != request['steps']:
            raise RuntimeError('The original counterfactual did not complete: '+str(result.get('error')))
        frozen_after = digest_state(integrity_payload(detached,compiled))
        if frozen_before != frozen_after or digest_state(metadata) != metadata_before:
            raise RuntimeError('A branch mutated the original frozen input state')
        trace = result.pop('trace')
        trace_path = target/'trace.npz'
        np.savez_compressed(trace_path,**trace)
        response = dict(state='completed',at_utc=now(),source=source,steps=request['steps'],
            elapsed_seconds=time.monotonic()-started,trace=str(trace_path),trace_sha256=h.digest(trace_path),
            result=json_safe(result),native_restore=primes,
            frozen_state_before=frozen_before,frozen_state_after=frozen_after,
            snapshot_sha256=request['snapshot_sha256'],worker_identity=h.process(os.getpid()),
            native_identity=__import__('column_review.common',fromlist=['process_identity']).process_identity(int(plant.ad.processId)))
        immutable(target/'response.json',response)
        return response
    finally:
        restore_rng(initial_rng)


def worker(config_path):
    config = closed_json(config_path)
    guard(config)
    directory = Path(config['worker_directory']).resolve()
    if not directory.is_relative_to(ROOT) or (directory/'identity.json').exists():
        raise RuntimeError('Do not duplicate an existing private branch worker')
    directory.mkdir(parents=True,exist_ok=True)
    import review_numeric_recovery
    review_numeric_recovery.install()
    import torch
    torch.set_num_threads(1)
    import win32com.client
    import column_review.common as common
    import column_e1_process_recovery as fresh
    import column_e1.runtime as runtime
    sys.path.insert(0,str(ROOT/'change-reports/weights_pid_reuse_recovery_20261024'))
    import ownership_fix
    sys.path.insert(0,str(ROOT/'change-reports/residual_native_retry2_20261024'))
    import document_open_retry
    from run_distillation_phase2_repeats import aspen_file_lock, aspen_com_compatibility
    from systems.distillation.plant import DistillationColumnAspen
    identity = common.process_identity(os.getpid())
    immutable(directory/'identity.json',identity)
    plant = object.__new__(DistillationColumnAspen)
    plant.path = str(directory/'working'/'C2S_SS_simulation16.dynf')
    Path(plant.path).parent.mkdir()
    plant._e1_project_root = str(Path(plant.path).parent)
    plant._e1_document_open = False
    original_step = DistillationColumnAspen.step
    def guarded_step(obj,*args,**kwargs):
        if (directory/'STOP').exists() or not same_generation(config['parent_identity'],common.process_identity(config['parent_identity']['pid'])):
            raise runtime.RestorationError('Branch parent stopped; retire only this owned worker')
        return original_step(obj,*args,**kwargs)
    processed = set()
    with aspen_com_compatibility(),aspen_file_lock(Path(config['canonical_model']),ROOT/'Distillation/Results/_aspen_file_locks'), \
         common.record_owned_apps(directory),ownership_fix.installed(fresh,safe_read,common.process_identity), \
         document_open_retry.installed(fresh), \
         fresh.installed(directory),patch.object(DistillationColumnAspen,'step',guarded_step):
        try:
            plant.ad = win32com.client.DispatchEx('AD application')
            plant.ad.NewDocument()
            plant.ad.Visible = True
            fresh._PROCESS_IDENTITIES[id(plant)] = common.process_identity(int(plant.ad.processId))
            immutable(directory/'ready.json',dict(at_utc=now(),identity=identity,case=config['case']))
            while not (directory/'STOP').exists():
                if not same_generation(config['parent_identity'],common.process_identity(config['parent_identity']['pid'])):
                    break
                for request_path in sorted(directory.glob('request_*.json')):
                    if request_path.name in processed:
                        continue
                    processed.add(request_path.name)
                    guard(config)
                    request = closed_json(request_path)
                    response_file = directory/('done_'+request_path.stem+'.json')
                    try:
                        response = one_branch(request,config,directory,plant)
                        immutable(response_file,response)
                    except BaseException:
                        immutable(response_file,dict(state='failed',at_utc=now(),traceback=traceback.format_exc(),
                            worker_identity=identity,request=request))
                        raise
                time.sleep(.2)
        finally:
            if getattr(plant,'ad',None) is not None:
                fresh.retire(plant,directory)
            immutable(directory/'retired.json',dict(at_utc=now(),identity=identity,
                owned_remaining=common.owned_apps_remaining(directory/'owned_aspen_apps.json')))


class BranchPool:
    """Two persistent processes, synchronously completing one decision pair."""
    def __init__(self, directory, template, *, popen=subprocess.Popen):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=False)
        self.template = deepcopy(template)
        self.parent_identity = h.process(os.getpid())
        self.workers = {}
        self.logs = []
        self.popen = popen
        self.closed = False

    def start(self):
        from run_polymer_e3_recovery import environment
        for case in CASES:
            config = deepcopy(self.template)
            config.update(case=case,parent_identity={k:self.parent_identity[k] for k in ('pid','created','alive')},
                          worker_directory=str(self.directory/('case'+str(case))))
            case_info = config.pop('cases')[str(case)]
            config.update(canonical_model=case_info['source_model'],canonical_sha256=case_info['source_sha256'])
            guard(config)
            path = self.directory/('config'+str(case)+'.json')
            immutable(path,config)
            log = (self.directory/('case'+str(case)+'.log')).open('xb')
            self.logs.append(log)
            child = self.popen([str(PYTHON),'-B','-u',str(Path(__file__).resolve()),'--worker',str(path)],
                cwd=ROOT,env=dict(environment(),PYTHONHASHSEED='7',MPLBACKEND='Agg'),stdout=log,
                stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            self.workers[case] = dict(process=child,config=config,directory=Path(config['worker_directory']))
        self.wait_ready()

    def wait_ready(self,timeout=240):
        deadline=time.monotonic()+timeout
        while True:
            if all((w['directory']/'ready.json').exists() for w in self.workers.values()):
                return
            self.check_processes()
            if time.monotonic()>deadline:
                raise TimeoutError('Branch workers did not become ready')
            time.sleep(.5)

    def check_processes(self):
        for w in self.workers.values():
            if w['process'].poll() is not None:
                raise RuntimeError('Private branch worker exited; inspect '+str(w['directory']))

    def submit(self,case,snapshot,steps,source):
        if self.closed or case not in self.workers or source not in ('policy','supervisor'):
            raise RuntimeError('Invalid branch request scope')
        w=self.workers[case]
        token=uuid.uuid4().hex
        name='request_'+token
        request=dict(snapshot=str(Path(snapshot).resolve()),snapshot_sha256=h.digest(snapshot),
                     steps=int(steps),source=source,output=str(w['directory']/('result_'+token)))
        immutable(w['directory']/(name+'.json'),request)
        return w['directory']/('done_'+name+'.json')

    def wait(self,requests,timeout=3600):
        deadline=time.monotonic()+timeout
        while not all(path.exists() for path in requests.values()):
            self.check_processes()
            if time.monotonic()>deadline:
                raise TimeoutError('Private branch pair exceeded its bounded wait; preserve all data')
            time.sleep(.5)
        results={name:closed_json(path) for name,path in requests.items()}
        for source,result in results.items():
            if result.get('state')!='completed' or result.get('source')!=source:
                raise RuntimeError('Parallel branch failed: '+str(result))
            if h.digest(result['trace'])!=result['trace_sha256'] or result['frozen_state_before']!=result['frozen_state_after']:
                raise RuntimeError('Branch result or frozen-state integrity differs')
        return results

    def pair(self,snapshot,steps,*,parallel=True):
        started=time.monotonic()
        if parallel:
            requests={source:self.submit(case,snapshot,steps,source)
                      for source,case in zip(('policy','supervisor'),CASES)}
            results=self.wait(requests)
        else:
            results={}
            for source,case in zip(('policy','supervisor'),CASES):
                results.update(self.wait({source:self.submit(case,snapshot,steps,source)}))
        return dict(elapsed_seconds=time.monotonic()-started,branches=results,parallel=parallel)

    def close(self):
        if self.closed:
            return
        self.closed=True
        for w in self.workers.values():
            w['directory'].mkdir(parents=True,exist_ok=True)
            (w['directory']/'STOP').write_text('Retire this owned branch worker.\n',encoding='utf-8')
        deadline=time.monotonic()+60
        for w in self.workers.values():
            remaining=max(1,deadline-time.monotonic())
            w['process'].wait(timeout=remaining)
            retired=closed_json(w['directory']/'retired.json')
            if retired['owned_remaining']:
                raise RuntimeError('A private worker still owns an Aspen application')
        for log in self.logs:
            log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    args=parser.parse_args()
    try:
        worker(args.worker)
    except BaseException:
        traceback.print_exc()
        raise
