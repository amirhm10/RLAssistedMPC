"""Process-local scheduling adapter around the unchanged E1 scientific code.

One synchronous decision at a time; two branch processes; all four advancing
learners stay together. Native source captures and original outputs retain the
existing semantics. No asynchronous pending results enter a training checkpoint.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid
from unittest.mock import patch

import parallel_branches as p


def export_transport(session, site, step, state, directory, exporter):
    """Save independent CPU inference state and durable native inputs once."""
    import cloudpickle
    from review_experiments.e1.study import json_safe
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    exported=exporter(session.snapshot,session.compiled)
    metadata=deepcopy(exported['system'].native_metadata)
    input_files=[]
    for key in ('file','project_file'):
        source=Path(metadata[key])
        target=directory/source.name
        if target.exists():raise RuntimeError('Native input filenames collide')
        if p.h.digest(source)!=metadata[key+'_sha256']:raise RuntimeError('Captured native input checksum differs')
        shutil.copyfile(source,target)
        if p.h.digest(target)!=metadata[key+'_sha256']:raise RuntimeError('Durable native input copy differs')
        metadata[key]=str(target)
        input_files.append(dict(path=str(target),bytes=p.h.attributes(target)['size'],sha256=metadata[key+'_sha256']))
    exported['system'].native_metadata=metadata
    payload={'schema':1,'category':session.compiled.category,'site':site.key,
        'local_names':sorted(session.compiled.local_names),
        'environment':{key:exported[key] for key in session.compiled.local_names if key in exported},
        'record':session.record,'rng':state,'options':asdict(session.options),
        'source_sha256':hashlib.sha256(session.compiled.source.encode()).hexdigest(),
        'scientific_hashes':{key:value for key,value in session.source_hashes.items()
                             if not key.startswith('review_experiments/') and key!='run_polymer_e1.py'},
        'continuation':session.options.continuation}
    snapshot=directory/'frozen.pkl.gz'
    with gzip.open(snapshot,'xb',compresslevel=1) as stream:cloudpickle.dump(payload,stream)
    receipt=dict(at_utc=p.now(),channel=site.channel,step=step,snapshot=str(snapshot),
        snapshot_sha256=p.h.digest(snapshot),snapshot_bytes=p.h.attributes(snapshot)['size'],
        native_inputs=input_files,elapsed_seconds=time.monotonic()-started,
        all_inputs_retained=True,scientific_source_map_unchanged=True)
    p.immutable(directory/'input_receipt.json',json_safe(receipt))
    return snapshot,receipt


class DecisionDispatch:
    """A repeated integration check uses a second complete paired execution."""
    def __init__(self, run_pair, load_trace):
        self.run_pair=run_pair
        self.load_trace=load_trace
        self.counts={'policy':0,'supervisor':0}
        self.batches={}
        self.receipts=[]

    def branch(self, source):
        if source not in self.counts:raise RuntimeError('Unexpected counterfactual source')
        repeat=self.counts[source]
        self.counts[source]+=1
        if repeat not in self.batches:
            self.batches[repeat]=self.run_pair()
        response=self.batches[repeat]['branches'][source]
        if response['frozen_state_before']!=response['frozen_state_after']:
            raise RuntimeError('Frozen branch state mutated')
        result=dict(response['result'],trace=self.load_trace(response))
        self.receipts.append(response)
        return result

    def complete(self):
        if self.counts['policy']!=self.counts['supervisor'] or not self.counts['policy']:
            raise RuntimeError('One counterfactual result was not consumed')


@contextmanager
def installed(pool, *, maximum_pairs=340):
    """Patch only this new process; preserve original files and source markers."""
    import numpy as np
    import column_e1_budget as budget
    import column_e1.runtime as native_runtime
    import review_experiments.e1.study as study
    import review_experiments.e1.snapshots as portable
    from review_experiments.e1.runtime import rng_state
    from column_e1.runtime import digest_state,integrity_payload,copying_into,E1NativeSnapshot
    original_after=budget.BudgetSession.after
    decisions=[]
    transport_root=pool.directory/'frozen_decisions'

    def before(session,site,environment):
        # Capture/save only. No restore or zero-time initialization of the
        # advancing native plant is permitted by this adapter.
        main=environment['system'];session.native_plant=main
        step=int(environment[session.compiled.step_name])
        selected=(site.key,step) in session.snapshot_steps
        folder=session.output/('native_snapshots' if selected else 'temporary_native')
        session.native=E1NativeSnapshot(main,{},folder)
        with copying_into(main):study.StudySession.before(session,site,environment)

    def after(session,site,environment):
        if len(decisions)>=maximum_pairs:raise RuntimeError('Parallel request budget exhausted')
        # Match the original detachment boundary before hashing agent state.
        session.detach()
        full_before=digest_state(integrity_payload(environment,session.compiled))
        step=int(environment[session.compiled.step_name]);native=session.native
        initial_rng=rng_state();snapshot=None;input_receipt=None;branch_steps=None
        started=time.monotonic()
        def paired(steps):
            nonlocal snapshot,input_receipt,branch_steps
            if snapshot is None:
                path=transport_root/(site.channel+'_'+str(step)+'_'+uuid.uuid4().hex[:8])
                # The original ColumnStudySession.after installs its native
                # detaching exporter before this first branch call.
                snapshot,input_receipt=export_transport(session,site,step,initial_rng,path,portable.portable_environment)
                branch_steps=steps
            if steps!=branch_steps:raise RuntimeError('A decision changed its rollout horizon')
            return pool.pair(snapshot,steps,parallel=True)
        def load_trace(response):
            with np.load(response['trace'],allow_pickle=False) as arrays:
                return {name:arrays[name].copy() for name in arrays.files}
        dispatcher=DecisionDispatch(lambda:paired(branch_steps),load_trace)
        def branch(snapshot_arg,compiled_arg,site_arg,record_arg,source,steps,continuation):
            nonlocal branch_steps
            if compiled_arg is not session.compiled or site_arg.key!=site.key or continuation!=session.options.continuation:
                raise RuntimeError('A dispatched branch changed decision provenance')
            if branch_steps is None:branch_steps=int(steps)
            if int(steps)!=branch_steps:raise RuntimeError('A branch changed its original horizon')
            result=dispatcher.branch(source)
            native.priming_records.extend(deepcopy(dispatcher.receipts[-1]['native_restore']))
            return result
        # Native restoration happens exclusively in the two child processes.
        # Keep the original recording/metrics/source tags, full live-state
        # checks, selected snapshot export, release and primary result writer.
        with patch.object(study,'run_branch',branch),patch.object(native,'restore',lambda:None):
            original_after(session,site,environment)
        dispatcher.complete()
        full_after=digest_state(integrity_payload(environment,session.compiled))
        if full_before!=full_after:raise RuntimeError('Parallel comparison mutated the advancing learner/controller')
        row=dict(at_utc=p.now(),channel=site.channel,step=step,steps=branch_steps,
            elapsed_seconds=time.monotonic()-started,input_receipt=input_receipt,
            full_python_state_before=full_before,full_python_state_after=full_after,
            source_map_unchanged=True,consumed_sources=dispatcher.counts,
            batches=dispatcher.batches,primary_result=session.all_records[-1])
        p.immutable(Path(snapshot).parent/'primary_commit.json',study.json_safe(row))
        decisions.append(row)
        with (pool.directory/'decisions.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(study.json_safe(row))+'\n')

    with patch.object(budget.BudgetSession,'before',before),patch.object(budget.BudgetSession,'after',after):
        yield decisions
