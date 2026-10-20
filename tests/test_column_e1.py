"""Native-handle isolation, complete learner serialization and failure restores."""
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest
import torch

from systems.distillation.plant import DistillationColumnAspen
from column_e1.runtime import (COM_FIELDS, BranchColumn, ColumnStudySession, E1NativeSnapshot, column_copying,
                               digest_state, python_plant_state, import_native, copying_into)
from column_e1.checkpoints import NativeTrainingPickler, restoring_into
from review_experiments.e1.study import StudyOptions, StudySession
from review_experiments.e1.runtime import rng_state
import run_column_e1 as cli


class OpaqueCOM:
    def __deepcopy__(self, memo):
        raise AssertionError('COM handle must never be deep-copied')


def column():
    plant = object.__new__(DistillationColumnAspen)
    for field in COM_FIELDS:
        setattr(plant, field, OpaqueCOM())
    plant.current_input = np.array([1., 2.])
    plant.current_output = np.array([.01, 300.])
    plant.delta_t = 1/6
    plant.path = 'private_case16.dynf'
    return plant


def test_native_copy_preserves_aliases_and_isolates_all_python_arrays():
    plant = column()
    with column_copying():
        copied = deepcopy({'system': plant, 'ctx': {'system': plant}})
    branch = copied['system']
    assert isinstance(branch, BranchColumn) and copied['ctx']['system'] is branch
    for field in COM_FIELDS:
        assert getattr(branch, field) is getattr(plant, field)
    branch.current_input[0] = 77
    assert plant.current_input[0] == 1
    with pytest.raises(RuntimeError, match='cannot close'):
        branch.close()


def test_native_pickle_rehydrates_aliases_learners_replay_optimizer_and_rng():
    plant, fresh = column(), column()
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)
    net(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    payload = {'plant': plant, 'ctx': {'system': plant}, 'learner': {'network': net, 'optimizer': optimizer,
                'replay': np.arange(12).reshape(6, 2), 'nstep': [(1, 2, 3)], 'train_steps': 1}, 'rng': rng_state()}
    stream = io.BytesIO()
    NativeTrainingPickler(stream, {}).dump(payload)
    with restoring_into(fresh):
        loaded = cloudpickle.loads(stream.getvalue())
    assert loaded['plant'] is loaded['ctx']['system'] is fresh
    assert digest_state(loaded['learner']) == digest_state(payload['learner'])
    assert digest_state(loaded['rng']) == digest_state(payload['rng'])
    assert loaded['learner']['optimizer'].param_groups[0]['params'][0] is loaded['learner']['network'].weight


def test_digest_detects_optimizer_replay_and_gate_state_changes():
    value = {'replay': np.zeros((4, 2)), 'gate': {'previous': [1, 2]}, 'optimizer': {'step': torch.tensor(4)}}
    baseline = digest_state(value)
    for key in value:
        changed = deepcopy(value)
        if key == 'replay': changed[key][0, 0] = 1
        elif key == 'gate': changed[key]['previous'][0] = 4
        else: changed[key]['step'] += 1
        assert digest_state(changed) != baseline


def test_failed_branch_leaves_live_native_state_untouched_and_restores_rng(monkeypatch, tmp_path):
    native_file = tmp_path / 'owned.snp'
    native_file.write_bytes(b'trusted native snapshot')
    restored, deleted = [], []
    native = SimpleNamespace(restore=lambda: restored.append(True), release=lambda:deleted.append(True), priming_records=[],
                             snapshot=SimpleNamespace(path=native_file, native=object(), python_state={'current_output':np.ones(2)}))
    compiled = SimpleNamespace(local_names={'i', 'system'}, step_name='i')
    session = ColumnStudySession(compiled, tmp_path / 'capture', StudyOptions())
    session.native = native
    session.native_plant = SimpleNamespace(sim=SimpleNamespace(Results=SimpleNamespace(Delete=lambda obj:deleted.append(obj))))
    import column_review.pilot as pilot
    monkeypatch.setattr(pilot, 'profile', lambda plant:([],np.zeros(16)))
    environment = {'i': 10, 'system': SimpleNamespace(current_input=np.ones(2))}
    before = digest_state(rng_state())
    def fail(*args):
        np.random.rand(6)
        raise RuntimeError('branch error')
    monkeypatch.setattr(StudySession, 'after', fail)
    with pytest.raises(RuntimeError, match='branch error'):
        session.after(SimpleNamespace(channel='markov', key='m'), environment)
    assert restored == [] and len(deleted) == 1
    assert before == digest_state(rng_state())
    assert native_file.exists()  # failed evidence is retained


def test_complete_network_hash_checks_later_parameters_and_temporary_tensor_views():
    net = torch.nn.Sequential(torch.nn.Linear(3, 20), torch.nn.Linear(20, 1))
    baseline = digest_state(net)
    assert baseline == digest_state(net)
    with torch.no_grad():
        net[1].bias.add_(1)
    assert baseline != digest_state(net)


def test_branch_copy_routes_native_operations_to_private_application_only():
    main, scratch = column(), column()
    scratch.current_input = np.array([500., 600.])
    with column_copying(), copying_into(scratch):
        branch = deepcopy({'system':main,'ctx':{'system':main}})
    assert branch['system'] is branch['ctx']['system']
    for field in COM_FIELDS:
        assert getattr(branch['system'],field) is getattr(scratch,field)
        assert getattr(branch['system'],field) is not getattr(main,field)
    np.testing.assert_array_equal(branch['system'].current_input, main.current_input)
    branch['system'].current_input[0] = 99
    assert main.current_input[0] == 1 and scratch.current_input[0] == 500


def test_full_integrity_hash_includes_nested_controller_context_without_com_handles():
    from column_e1.runtime import integrity_payload
    main = column()
    environment = {'system':main, 'ctx':{'system':main,'observer':np.ones(4),'mpc_guess':np.zeros(6)}}
    compiled = SimpleNamespace(local_names=set(environment))
    before = digest_state(integrity_payload(environment, compiled))
    environment['ctx']['mpc_guess'][0] = 1
    assert before != digest_state(integrity_payload(environment, compiled))


@pytest.mark.parametrize('coordinate,change', [(0, 1/6), (9, 2e-6), (14, .01)])
def test_branch_reinitialization_refuses_time_advance_or_out_of_bound_state_drift(coordinate, change, monkeypatch, tmp_path):
    import column_review.pilot as pilot
    before,after = np.zeros(16),np.zeros(16)
    after[coordinate] = change
    rows = iter([before,before,after])
    monkeypatch.setattr(pilot,'profile',lambda plant:([],next(rows)))
    native = object.__new__(E1NativeSnapshot)
    native.snapshot=SimpleNamespace(path=tmp_path/'owned.snp')
    native.restored_snapshot=SimpleNamespace(native=object(),internal=before,
        python_state={'current_input':np.ones(2),'current_output':np.zeros(2),'controller':{}})
    native.plant=SimpleNamespace(sim=SimpleNamespace(Step=lambda *args:None,Results=SimpleNamespace(LoadSnapshot=lambda obj:None)),
        feed=SimpleNamespace(FmR=SimpleNamespace(Value=0)),
        col=SimpleNamespace(Reflux=SimpleNamespace(FmR=SimpleNamespace(Value=0)),QRebR=SimpleNamespace(Value=0)))
    native.priming_records=[]
    with pytest.raises(RuntimeError,match='changed profile'):
        native.restore()
    assert (tmp_path/'owned.restores.jsonl').is_file()


def test_loader_requires_both_verified_native_files_before_closing_owned_document(tmp_path):
    from column_e1.runtime import load_project
    from column_review.common import digest
    bundle=tmp_path/'bundle'
    bundle.mkdir()
    source=bundle/'C2S_SS_simulation16.dynf'
    database=bundle/'snapshot.snp'
    source.write_bytes(b'initialized private model')
    database.write_bytes(b'rewindable native snapshot database')
    working=tmp_path/'working'
    working.mkdir()
    model=working/source.name
    model.write_bytes(b'pristine private input')
    events=[]
    native=object()
    results=SimpleNamespace(Refresh=lambda:events.append('refresh'), FindSnapshot=lambda name:native)
    sim=SimpleNamespace(Results=results,Flowsheet=SimpleNamespace(Streams=lambda name:'feed',Blocks=lambda name:'column'))
    def opened(path):
        events.append('open')
        active=Path(path)
        assert active!=model and active.read_bytes()==source.read_bytes()
        assert (active.parent/'AM_C2S_SS_simulation16/snapshot.snp').read_bytes()==database.read_bytes()
    app=SimpleNamespace(CloseDocument=lambda save:events.append(('close',save)),openDocument=opened,Simulation=sim)
    plant=SimpleNamespace(path=str(model),ad=app)
    metadata={'file':str(database),'file_sha256':digest(database),'project_file':str(source),
              'project_file_sha256':digest(source),'name':'review_e1_test'}
    with pytest.raises(RuntimeError,match='checksum'):
        load_project(plant,dict(metadata,project_file_sha256='wrong'))
    assert not events and model.read_bytes()==b'pristine private input'
    assert load_project(plant,metadata) is native
    assert events==[('close',False),'open','refresh']
    assert model.read_bytes()==b'pristine private input'
    first=plant.path
    load_project(plant,metadata)
    assert plant.path!=first


def test_capture_saves_native_database_instead_of_nonrewindable_export(monkeypatch,tmp_path):
    from column_e1.runtime import native_metadata
    from repair_aspen_snapshot import ReadyNativeSnapshot
    import column_review.pilot as pilot
    model=tmp_path/'main/C2S_SS_simulation16.dynf'
    model.parent.mkdir()
    model.write_bytes(b'private initialized document')
    work=model.parent/'AM_C2S_SS_simulation16'
    work.mkdir()
    (work/'snapshot.bak').write_bytes(b'actual rewindable database')
    output=tmp_path/'capture'
    output.mkdir()
    plant=SimpleNamespace(path=str(model),ad=SimpleNamespace(SaveDocument=lambda:None))
    def initialize(self,plant,controller,output):
        exported=output/'review_e1_test.snp'
        exported.write_bytes(b'exported result cannot rewind')
        self.snapshot=SimpleNamespace(path=exported,name='review_e1_test',internal=np.zeros(16),python_state={},native=object())
    monkeypatch.setattr(ReadyNativeSnapshot,'__init__',initialize)
    monkeypatch.setattr(pilot,'profile',lambda plant:([],np.zeros(16)))
    native=E1NativeSnapshot(plant,{},output)
    metadata=native_metadata(native)
    assert native.snapshot.path.read_bytes()==b'actual rewindable database'
    assert native.snapshot.exported_result_path.read_bytes()==b'exported result cannot rewind'
    assert Path(metadata['project_file']).read_bytes()==model.read_bytes()


def test_restore_recovers_physical_specifications_and_preserves_pending_python_control(monkeypatch,tmp_path):
    import column_review.pilot as pilot
    expected=np.zeros(16)
    expected[[0,1,2,3,9,14]]=[5.,3.,10.,2.,.1,-20.]
    plant=SimpleNamespace(feed=SimpleNamespace(FmR=SimpleNamespace(Value=0.)),
        col=SimpleNamespace(Reflux=SimpleNamespace(FmR=SimpleNamespace(Value=0.)),QRebR=SimpleNamespace(Value=0.)),
        sim=SimpleNamespace(Results=SimpleNamespace(LoadSnapshot=lambda obj:None),Step=lambda *args:None))
    def profile(plant):
        value=expected.copy()
        value[1:4]=[plant.feed.FmR.Value,plant.col.Reflux.FmR.Value,plant.col.QRebR.Value]
        return [],value
    monkeypatch.setattr(pilot,'profile',profile)
    native=object.__new__(E1NativeSnapshot)
    snapshot=SimpleNamespace(path=tmp_path/'native.snp',native=object(),internal=expected,
        python_state={'current_input':np.array([99.,88.]),'current_output':np.array([.1,-20.]),'controller':{'observer':[1,2]}})
    native.snapshot=native.restored_snapshot=snapshot
    native.plant=plant
    native.priming_records=[]
    assert native.restore()=={'observer':[1,2]}
    np.testing.assert_array_equal(profile(plant)[1],expected)
    np.testing.assert_array_equal(plant.current_input,[99.,88.])


def test_failed_counterfactual_is_retained_without_stopping_unchanged_main(monkeypatch,tmp_path):
    import column_review.pilot as pilot
    monkeypatch.setattr(pilot,'profile',lambda plant:([],np.zeros(16)))
    compiled=SimpleNamespace(local_names={'i','system'},step_name='i')
    session=ColumnStudySession(compiled,tmp_path/'capture',StudyOptions())
    native_file=tmp_path/'native.snp'
    native_file.write_bytes(b'failed diagnostic state retained')
    released=[]
    session.native=SimpleNamespace(snapshot=SimpleNamespace(path=native_file),priming_records=[],release=lambda:released.append(True))
    session.native_plant=SimpleNamespace()
    environment={'i':10,'system':SimpleNamespace()}
    monkeypatch.setattr(StudySession,'after',lambda self,site,environment:self.all_records.append({'status':'branch_failed'}))
    session.after(SimpleNamespace(channel='markov',key='m'),environment)
    assert session.all_records==[{'status':'branch_failed'}] and released==[True]
    assert native_file.is_file() and session.integration_checks[-1]['live_native_unchanged']


@pytest.mark.parametrize('category', cli.CATEGORIES)
def test_full_column_configuration_preserves_science_and_original_clock(category, tmp_path):
    from systems.distillation import get_distillation_notebook_defaults
    defaults = get_distillation_notebook_defaults('horizon_standard' if category == 'horizon' else category)
    before = repr(defaults)
    configured = cli.configure(defaults, category, tmp_path, {'aspen_path':'private.dynf', 'snaps_path':'private_snaps'})
    assert repr(defaults) == before
    assert configured['training_profile_override'] == 'temperature_flip_200_100'
    for name in ('set_points_len_override', 'warm_start_override', 'n_tests_override', 'test_cycle_override'):
        assert configured[name] == defaults[name]
    for name in ('controller', 'system_setup', 'episode', 'mpc'):
        if name in defaults: assert repr(configured[name]) == repr(defaults[name])
    for key in ('td3_agent', 'horizon_agent', 'markov_td3_agent', 'weights_td3_agent', 'residual_td3_agent', 'agent'):
        if isinstance(defaults.get(key), dict):
            assert repr({k:v for k,v in defaults[key].items() if k!='seed'}) == repr({k:v for k,v in configured[key].items() if k!='seed'})


def test_exported_checkpoint_rejects_changed_native_binary(tmp_path):
    path = tmp_path / 'changed.snp'
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='checksum'):
        import_native(SimpleNamespace(), {'file':str(path), 'file_sha256':'wrong'})


@pytest.mark.parametrize('category', cli.CATEGORIES)
def test_technical_setup_reaches_native_creation_with_correct_original_defaults(category, monkeypatch, tmp_path):
    import systems.distillation as module
    import systems.distillation.plant as plant_module
    class SetupReached(Exception): pass
    def stop(*args, **kwargs): raise SetupReached()
    monkeypatch.setattr(module, 'build_distillation_system', stop)
    monkeypatch.setattr(plant_module, 'build_distillation_system', stop)
    with pytest.raises(SetupReached):
        cli.execute(category, lambda defaults: cli.configure(defaults, category, tmp_path,
                    {'aspen_path':'private.dynf','snaps_path':'private_snaps'}, technical=True))


def test_progress_viewer_reads_new_isolated_queue_without_starting_it(monkeypatch, tmp_path):
    import json
    from revision_comparisons import progress
    directory = tmp_path / 'study'
    attempt = directory / 'combined/a1'
    attempt.mkdir(parents=True)
    pointer = tmp_path / 'Distillation/Results/column_e1/latest.json'
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({'directory':str(directory)}))
    (directory / 'manifest.json').write_text(json.dumps({'jobs':[{'category':'combined','state':'running','pid':42,'directory':str(attempt)}]}))
    (attempt / 'status.json').write_text(json.dumps({'state':'running','pid':42}))
    (attempt / 'worker.log').write_text('Sub_Episode: 17\n')
    monkeypatch.setattr(progress, 'process_info', lambda pid: {'alive':True,'cpu_seconds':2})
    row = next(row for row in progress.snapshot(tmp_path)['jobs'] if row['study']=='E1_column')
    assert row['state']=='running' and row['cycle']==17 and row['pid']==42
