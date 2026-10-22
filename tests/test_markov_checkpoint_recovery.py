from types import SimpleNamespace
import pytest
from review_markov_checkpoint_recovery import MarkovCheckpointRunner


def select_action_with_supervisor(step):
    return step * 2


def guarded_runner(ctx):
    nFE = ctx['nFE']
    trace = ctx['trace']
    trace.append('outer')
    try:
        trace.append('inner')
        for step in range(nFE):
            result = select_action_with_supervisor(step)
            trace.append(result)
        trace.append('tail')
        return trace
    finally:
        trace.append('closed')


class Manager:
    has_checkpoint = False
    def due(self, step, environment):
        return True
    def restore(self, environment, session):
        environment.update(self.saved)
        environment['__recovery_next_step'] = 2


def generator(manager, trace):
    compiled = MarkovCheckpointRunner(guarded_runner, 'markov')
    env = compiled.environment(SimpleNamespace(wants=lambda *a:False, is_active=lambda *a:False),
                               ({'nFE':4, 'trace':trace, 'time_in_sub_episodes':2},), {})
    env.update(__recovery_manager=manager, __recovery_next_step=0)
    return env, env['guarded_runner']()


def test_checkpoint_resume_retains_order_and_finally():
    manager = Manager()
    trace = []
    env, running = generator(manager, trace)
    assert next(running) == ('checkpoint', None)
    assert next(running) == ('checkpoint', None)
    manager.saved = {name:env[name] for name in ('ctx','nFE','trace','step','result')}
    manager.saved['trace'] = list(trace)
    running.close()
    assert trace == ['outer','inner',0,2,'closed']
    manager.has_checkpoint = True
    resumed, running = generator(manager, [])
    assert next(running) == ('checkpoint', None)
    assert next(running) == ('checkpoint', None)
    with pytest.raises(StopIteration): next(running)
    assert resumed['trace'] == ['outer','inner',0,2,4,6,'tail','closed']


def test_actual_markov_runner_compiles_without_editing_science():
    import utils.markov_runner as original
    compiled = MarkovCheckpointRunner(original.run_single_closed_loop, 'markov')
    assert compiled.step_name == 'step'
    assert len(compiled.sites) == 1
    session=SimpleNamespace()
    environment=compiled.environment(session,({}, {'time_in_sub_episodes':400},None,None,None,None),{'use_markov':True})
    assert environment['time_in_sub_episodes']==400
    assert 'time_in_sub_episodes' in compiled.local_names
