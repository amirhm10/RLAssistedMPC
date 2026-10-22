from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
from column_e1_budget import choose_steps,migrate_session,adapted_worker,PROTOCOL_ID
from review_experiments.e1.study import StudyOptions,WINDOWS


def test_nested_sampling_covers_clock_strata_without_outcomes():
    original=list(range(5200,7200,20))
    state=np.random.get_state()
    chosen,strata,allocation=choose_steps(original,7,'combined','weights','release')
    assert len(chosen)==20 and len(set(chosen))==20 and set(chosen)<=set(original)
    assert Counter(strata.values())=={'early':10,'middle':5,'late':5}
    assert chosen==choose_steps(original,7,'combined','weights','release')[0]
    after=np.random.get_state()
    assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]


def test_stratum_shortage_is_explicitly_reallocated():
    original=[i*200+100 for i in range(100)]
    chosen,strata,allocation=choose_steps(original,7,'standalone')
    assert len(chosen)==20 and set(strata.values())=={'late'}
    assert allocation['quotas']=={'early':0,'middle':0,'late':20}


def session_fixture(tmp_path):
    site=SimpleNamespace(key='weights:42',channel='weights')
    schedules=set()
    for _,first,last in WINDOWS:
        schedules.update((first-1)*400+i*20 for i in range(100))
    output=tmp_path/'capture';output.mkdir()
    return SimpleNamespace(output=output,compiled=SimpleNamespace(category='weights',sites={site.key:site}),
          schedules={site.key:schedules},options=StudyOptions(),counts=Counter(),all_records=[])


def test_migration_keeps_controller_objects_and_full_phase_coverage(tmp_path):
    session=session_fixture(tmp_path)
    learner=np.arange(10.);environment={'time_in_sub_episodes':400,'learner':learner,'system':object()}
    original=dict(environment)
    migrate_session(session,environment,'signed')
    assert environment.keys()==original.keys()
    assert all(environment[key] is original[key] for key in original)
    assert session.options.decisions_per_window==20
    assert len(session.schedules['weights:42'])==80 and len(session.long_steps)==8 and len(session.snapshot_steps)==4
    assert Counter(session.window_for_step.values())=={window:20 for window,_,_ in WINDOWS}
    assert session.options.rollout_steps==200 and session.options.long_rollout_steps==800
    assert session.budget_protocol_sha256=='signed'


def test_migration_rejects_checkpoint_with_captured_prefix(tmp_path):
    session=session_fixture(tmp_path);session.all_records=[{'step':5200}]
    with pytest.raises(RuntimeError,match='pre-capture'):migrate_session(session,{'time_in_sub_episodes':400},'signed')


def test_original_worker_adaptation_keeps_source_and_supports_bounded_handoff():
    import run_column_e1
    original=run_column_e1.worker
    adapted=adapted_worker(original,validation_steps=8)
    assert adapted.__code__.co_filename==original.__code__.co_filename
    assert adapted.__code__.co_firstlineno==original.__code__.co_firstlineno
    assert adapted.__globals__['__budget_validation_steps']==8
    assert run_column_e1.worker is original


def test_protocol_tags_match_disk_and_checkpoint_record(monkeypatch):
    import json
    import review_experiments.e1.study as study
    import column_e1.runtime as runtime
    from column_e1_budget import BudgetSession
    session=object.__new__(BudgetSession)
    session.budget_protocol_sha256='signed';session.budget_strata={('weights:42',5200):'early'}
    session.all_records=[]
    saved=[]
    def parent(self,site,environment):
        row={'gate_source_line':42,'metrics':{},'step':5200}
        saved.append(json.dumps(study.json_safe(row)))
        self.all_records.append(study.json_safe(row))
    monkeypatch.setattr(runtime.ColumnStudySession,'after',parent)
    session.after(SimpleNamespace(key='weights:42'),{})
    assert json.loads(saved[0])==session.all_records[0]
    assert session.all_records[0]['budget_protocol_id']==PROTOCOL_ID
    assert session.all_records[0]['sampling_stratum']=='early'
