"""Private restart compiler for the authoritative Markov try/finally runner."""
import ast
from pathlib import Path

from review_experiments.e1.runtime import (
    CompiledRunner, _GlobalizeNonlocal, _Suspensions, _as_list,
)
from revision_recovery.checkpoints import CheckpointRunner as OriginalCheckpointRunner


class MarkovCheckpointRunner(CompiledRunner):
    def __init__(self, function, category):
        super().__init__(function, category)
        self.local_names.add('time_in_sub_episodes')
        root = ast.parse(self.source).body[0]
        wrappers = [(i, node) for i, node in enumerate(root.body) if isinstance(node, ast.Try)]
        if len(wrappers) != 1:
            raise ValueError('Expected the authoritative single Markov try/finally.')
        outer_index, wrapper = wrappers[0]
        if wrapper.handlers or wrapper.orelse or not wrapper.finalbody or outer_index != len(root.body)-1:
            raise ValueError('Unsupported Markov cleanup structure.')
        loops = [(i, node) for i, node in enumerate(wrapper.body)
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                 and node.target.id == self.step_name
                 and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                 and node.iter.func.id == 'range' and len(node.iter.args) == 1
                 and isinstance(node.iter.args[0], ast.Name) and node.iter.args[0].id == 'nFE']
        if len(loops) != 1:
            raise ValueError('Expected one zero-based Markov range(nFE) loop.')
        index, loop = loops[0]
        root.decorator_list = []
        root.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
        suspension = _Suspensions(list(self.sites.values()), self.source_start-1)
        loop.body = [part for statement in loop.body for part in _as_list(suspension.visit(statement))]
        loop.body.append(ast.parse(
            f"if __recovery_manager.due({self.step_name}, globals()):\n"
            "    yield ('checkpoint', None)\n").body[0])
        loop.iter.args.insert(0, ast.Name(id='__recovery_next_step', ctx=ast.Load()))
        # Keep every controller statement, helper, tail and cleanup in order.
        # On resume, the saved environment supplies both initialization blocks.
        prefix = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=ast.Attribute(
            value=ast.Name(id='__recovery_manager', ctx=ast.Load()), attr='has_checkpoint', ctx=ast.Load())),
            body=wrapper.body[:index] or [ast.Pass()], orelse=[])
        wrapper.body = [prefix, loop, *wrapper.body[index+1:]]
        choice = ast.If(test=ast.Attribute(value=ast.Name(id='__recovery_manager', ctx=ast.Load()),
                         attr='has_checkpoint', ctx=ast.Load()),
                        body=ast.parse('__recovery_manager.restore(globals(), __e1_session)').body,
                        orelse=root.body[:outer_index] or [ast.Pass()])
        root.body = [ast.Global(names=sorted(self.local_names)), choice, wrapper]
        root = _GlobalizeNonlocal(self.local_names).visit(root)
        tree = ast.fix_missing_locations(ast.Module(body=[root], type_ignores=[]))
        ast.increment_lineno(tree, self.source_start-1)
        self.generator_code = compile(tree, self.filename, 'exec')

    def environment(self, session, args, kwargs):
        environment = super().environment(session, args, kwargs)
        # This runner reads its schedule from ctx instead of binding a local.
        # Expose the same value to checkpoint accounting without editing it.
        environment['time_in_sub_episodes'] = int(environment['ctx']['time_in_sub_episodes'])
        return environment


def install():
    """Affects this process only; existing workers retain their loaded code."""
    import revision_recovery.checkpoints as checkpoints
    import column_e1.checkpoints as native
    from column_review.common import digest
    def runner(function, category):
        return (MarkovCheckpointRunner if category == 'markov' else OriginalCheckpointRunner)(function, category)
    checkpoints.CheckpointRunner = runner
    if getattr(native.NativeCheckpointManager.__init__, '_markov_recovery', False):
        return
    original = native.NativeCheckpointManager.__init__
    signature = digest(Path(__file__))
    def initialize(self, directory, sources, **kwargs):
        original(self, directory, dict(sources, review_markov_checkpoint_recovery=signature), **kwargs)
    initialize._markov_recovery = True
    native.NativeCheckpointManager.__init__ = initialize
