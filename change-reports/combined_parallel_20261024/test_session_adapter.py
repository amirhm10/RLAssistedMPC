"""Main-plant isolation, repeated-dispatch and durable-transport contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import gzip

import cloudpickle
import numpy as np
import pytest

import parallel_branches as p
import session_adapter as a


def response(source,batch=0,trace=None):
    return dict(source=source,frozen_state_before='same',frozen_state_after='same',
                result={'state':'completed','completed_steps':20},trace=str(trace or 'trace'),
                native_restore=[{'full_variable_copy':True}],batch=batch)


def test_repeated_native_validation_uses_two_independent_complete_pairs():
    calls=[]
    def paired():
        index=len(calls);calls.append(index)
        return {'branches':{s:response(s,index) for s in ('policy','supervisor')}}
    dispatcher=a.DecisionDispatch(paired,lambda r:r['batch'])
    results=[dispatcher.branch(s)['trace'] for s in ('policy','policy','supervisor','supervisor')]
    assert results==[0,1,0,1] and calls==[0,1]
    dispatcher.complete()


def test_one_unconsumed_source_cannot_commit():
    dispatcher=a.DecisionDispatch(lambda:{'branches':{s:response(s) for s in ('policy','supervisor')}},lambda r:None)
    dispatcher.branch('policy')
    with pytest.raises(RuntimeError,match='not consumed'):dispatcher.complete()


def test_durable_transport_copies_native_inputs_without_editing_capture(tmp_path):
    from review_experiments.e1.study import StudyOptions
    source=tmp_path/'original';source.mkdir()
    model=source/'C2S_SS_simulation16.dynf';model.write_bytes(b'original-model')
    native=source/'snapshot.snp';native.write_bytes(b'original-native')
    metadata=dict(file=str(native),file_sha256=p.h.digest(native),project_file=str(model),project_file_sha256=p.h.digest(model))
    snapshot={'system':SimpleNamespace(native_metadata=metadata),'i':4}
    compiled=SimpleNamespace(category='combined',source='unchanged controller',local_names={'system','i'})
    session=SimpleNamespace(snapshot=snapshot,compiled=compiled,record={'decision':'same'},
        options=StudyOptions(decisions_per_window=20),source_hashes={'scientific.py':'sealed','column_e1_budget_protocol':'signed'})
    exported,receipt=a.export_transport(session,SimpleNamespace(key='weights:1',channel='weights'),4,
                                        {'rng':'same'},tmp_path/'transport',lambda env,compiled:deepcopy(env))
    with gzip.open(exported,'rb') as stream:payload=cloudpickle.load(stream)
    new=payload['environment']['system'].native_metadata
    assert Path(new['file']).parent==tmp_path/'transport' and Path(new['project_file']).parent==tmp_path/'transport'
    assert snapshot['system'].native_metadata==metadata
    assert native.read_bytes()==b'original-native' and model.read_bytes()==b'original-model'
    assert payload['scientific_hashes']==session.source_hashes and receipt['all_inputs_retained']


@pytest.mark.parametrize('mutate',[False,True])
def test_parent_adapter_never_restores_or_steps_main_and_rejects_mutation(monkeypatch,tmp_path,mutate):
    import column_e1_budget as budget
    import review_experiments.e1.study as study
    pool_dir=tmp_path/'pool';pool_dir.mkdir()
    transport=tmp_path/'transport';transport.mkdir()
    trace=tmp_path/'trace.npz';np.savez_compressed(trace,y=np.zeros((20,2)))
    pool=SimpleNamespace(directory=pool_dir,pair=lambda *args,**kwargs:{'branches':{s:response(s,trace=trace) for s in ('policy','supervisor')}})
    restores=[];environment={'i':5}
    native=SimpleNamespace(restore=lambda:restores.append('UNSAFE_MAIN_RESTORE'),priming_records=[])
    compiled=SimpleNamespace(category='combined',step_name='i',local_names={'i'})
    site=SimpleNamespace(key='weights:1',channel='weights')
    session=SimpleNamespace(native=native,compiled=compiled,options=SimpleNamespace(continuation='gated'),
        snapshot={},record={'original':'decision'},source_hashes={'same':'map'},all_records=[],detach=lambda:None)
    def original_after(obj,site,env):
        obj.native.restore()
        for source in ('policy','supervisor'):
            branch=study.run_branch(obj.snapshot,obj.compiled,site,obj.record,source,20,'gated')
            assert branch['trace']['y'].shape==(20,2)
        if mutate:env['i']+=1
        obj.all_records.append({'status':'completed'})
    monkeypatch.setattr(budget.BudgetSession,'after',original_after)
    monkeypatch.setattr(a,'export_transport',lambda *args,**kwargs:(transport/'frozen.pkl.gz',{'same':'inputs'}))
    with a.installed(pool):
        if mutate:
            with pytest.raises(RuntimeError,match='advancing'):budget.BudgetSession.after(session,site,environment)
        else:budget.BudgetSession.after(session,site,environment)
    assert restores==[] and len(native.priming_records)==2 and session.source_hashes=={'same':'map'}
    assert (transport/'primary_commit.json').exists() is (not mutate)
