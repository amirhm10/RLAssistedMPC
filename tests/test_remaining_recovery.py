from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from column_e1_recovery import restore_snapshot
from review_experiments.e1.runtime import RestorationError
from review_training_recovery import TrainingSession


def native_fixture(monkeypatch,tmp_path,*,copy=True,advance=False,bad_load=False):
    import column_review.pilot
    expected=np.zeros(16)
    expected[0]=907.;expected[1:4]=[150997.,368210.,124.]
    expected[9]=.023;expected[14]=-22.16
    value=expected.copy()
    def load(_):
        value[:]=expected
        if bad_load:value[9]+=.01
    def step(*_):
        value[9]+=1.75e-6;value[14]+=.00305
        if advance:value[0]+=1/6
    def copy_values(*_):
        if copy:value[:]=expected
    results=SimpleNamespace(LoadSnapshot=load,CopyValues=copy_values)
    variable=lambda:SimpleNamespace(Value=0.)
    plant=SimpleNamespace(sim=SimpleNamespace(Results=results,Step=step),feed=SimpleNamespace(FmR=variable()),
                          col=SimpleNamespace(Reflux=SimpleNamespace(FmR=variable()),QRebR=variable()))
    monkeypatch.setattr(column_review.pilot,'profile',lambda p:([],value.copy()))
    data=SimpleNamespace(internal=expected.copy(),native=object(),python_state=dict(current_input=np.ones(2),current_output=np.ones(2),controller={'observer':np.ones(3)}))
    owner=SimpleNamespace(plant=plant,restored_snapshot=data,snapshot=SimpleNamespace(path=tmp_path/'native.snp',python_state=data.python_state),priming_records=[])
    return owner,value,expected


def test_real_failure_magnitude_is_corrected_without_relaxing_fidelity(monkeypatch,tmp_path):
    owner,value,expected=native_fixture(monkeypatch,tmp_path)
    controller=restore_snapshot(owner)
    np.testing.assert_array_equal(value,expected)
    assert owner.priming_records[0]['priming_difference'][14]>.003
    assert owner.priming_records[0]['difference']==[0.]*16
    controller['observer'][0]=99
    assert owner.restored_snapshot.python_state['controller']['observer'][0]==1


@pytest.mark.parametrize('kwargs',[{'copy':False},{'advance':True},{'bad_load':True}])
def test_native_integrity_failures_still_stop_recovery(monkeypatch,tmp_path,kwargs):
    owner,_,_=native_fixture(monkeypatch,tmp_path,**kwargs)
    with pytest.raises(RestorationError):restore_snapshot(owner)


def test_repetition_checkpoint_session_never_adds_counterfactual_sampling(tmp_path):
    session=TrainingSession(object(),tmp_path/'capture')
    assert not session.wants('gate',{}) and not session.is_active('gate')
    assert session.all_records==[] and session.native is None


def test_threshold_exports_and_complete_training_state_round_trip(tmp_path):
    import time
    import torch
    from review_training_recovery import full_policy_manager
    names=('horizon_agent','weight_agent','markov_agent','residual_agent')
    agents={name:torch.nn.Linear(2,1) for name in names}
    optimizer=torch.optim.Adam(agents['weight_agent'].parameters(),lr=.01)
    agents['weight_agent'](torch.ones(1,2)).sum().backward();optimizer.step()
    environment=dict(agents,i=3,time_in_sub_episodes=4,nFE=8,optimizer=optimizer)
    compiled=SimpleNamespace(step_name='i',source='original runner',local_names=set(environment))
    session=TrainingSession(compiled,tmp_path/'session')
    manager=full_policy_manager()(tmp_path/'policy_checkpoints',{'source_hashes':{}})
    manager.compiled=compiled
    expected=agents['weight_agent'].weight.detach().clone()
    manager.save(environment,session)
    assert manager.has_checkpoint
    restored={}
    manager.restore(restored,session)
    torch.testing.assert_close(restored['weight_agent'].weight,expected)
    assert restored['optimizer'].param_groups[0]['params'][0] is restored['weight_agent'].weight
    assert restored['__recovery_next_step']==4


def test_native_reload_preserves_controller_input_output_aliases(monkeypatch):
    import column_e1.checkpoints as native
    from run_remaining_recovery import install_native_aliases
    plant=SimpleNamespace(current_input=np.ones(2),current_output=np.zeros(2))
    controller={'u':plant.current_input,'y':plant.current_output}
    def load(p,metadata):
        p.current_input=p.current_input.copy();p.current_output=p.current_output.copy()
        return ['native fidelity passed']
    monkeypatch.setattr(native,'import_native',load)
    install_native_aliases()
    assert native.import_native(plant,{})==['native fidelity passed']
    assert plant.current_input is controller['u'] and plant.current_output is controller['y']


def test_atomic_checkpoint_metadata_accepts_numpy_arrays(tmp_path,monkeypatch):
    import json
    import column_review.common as common
    import revision_recovery.checkpoints as checkpoints
    import review_experiments.e1.study as study
    import column_e1.checkpoints as native
    from run_remaining_recovery import install_atomic_writes
    for module,name in ((common,'write'),(checkpoints,'write_json'),(study,'write_json'),(native,'write_json'),(native,'import_native')):
        monkeypatch.setattr(module,name,getattr(module,name))
    install_atomic_writes()
    path=tmp_path/'native_metadata.json'
    native.write_json(path,{'profile':np.array([1.,2.]),'count':np.int64(3)})
    assert json.loads(path.read_text())=={'profile':[1.,2.],'count':3}
