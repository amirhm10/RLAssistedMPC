from types import SimpleNamespace
import pytest
from column_e1_budget_lazy_recovery import migration_function


def test_empty_checkpoint_rebuilds_and_checks_original_schedule_before_migration():
    calls=[]
    session=SimpleNamespace(schedules={},options=SimpleNamespace(decisions_per_window=100))
    def initialize(environment):
        calls.append(('initialize',environment))
        session.schedules={'weights:470':{3,1,2}}
    session._initialize=initialize
    def migrate(current,environment,sha):
        assert current.schedules=={'weights:470':{1,2,3}}
        calls.append(('migrate',sha));return 'ok'
    wrapper=migration_function(migrate,{'selected_steps':{'weights:470':[1,2,3]}})
    assert wrapper(session,{'clock':400},'signed')=='ok'
    assert session.budget_lazy_initialized and [c[0] for c in calls]==['initialize','migrate']


def test_rebuilt_schedule_mismatch_is_rejected():
    session=SimpleNamespace(schedules={},options=SimpleNamespace(decisions_per_window=100))
    session._initialize=lambda environment:session.schedules.update({'weights:470':{1}})
    wrapper=migration_function(lambda *args:pytest.fail('must not migrate'),{'selected_steps':{'weights:470':[2]}})
    with pytest.raises(RuntimeError,match='differs'):wrapper(session,{},'signed')


def test_existing_schedule_is_preserved_without_reinitialization():
    schedule={'weights:470':{1,2}}
    session=SimpleNamespace(schedules=schedule,options=SimpleNamespace(decisions_per_window=100))
    session._initialize=lambda environment:pytest.fail('already initialized')
    wrapper=migration_function(lambda current,*args:current.schedules,{})
    assert wrapper(session,{},'signed') is schedule
    assert session.budget_lazy_initialized is False
