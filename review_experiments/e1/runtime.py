"""Pause existing runner calculations and execute isolated counterfactual suffixes.

The source files are never written. A private, process-local compilation exposes
the runner's local state in a namespace and adds suspension points around its
existing gate calls. The original AST statements, including warm-up, hand-off,
replay, learning and disturbance handling, are executed in their original order.
Branch continuations reuse those same statements, with frozen inference agents.
Unsupported source structures fail explicitly rather than approximating a loop.
"""
from __future__ import annotations

import ast
from copy import copy, deepcopy
from dataclasses import dataclass, replace
import functools
import inspect
import random
import textwrap
import types
from typing import Any

import numpy as np
import torch


RUNNER_TARGETS = {
    "horizon": ("utils.horizon_runner", "run_dqn_mpc_horizon_supervisor"),
    "weights": ("utils.weights_runner", "run_weight_multiplier_supervisor"),
    "residual": ("utils.residual_runner", "run_residual_supervisor"),
    "markov": ("utils.markov_runner", "run_single_closed_loop"),
    "combined": ("utils.combined_runner", "run_combined_supervisor"),
}
AGENT_NAMES = {
    "horizon_decision": ("horizon", "horizon_agent"),
    "weight_action_decision": ("weights", "weight_agent"),
    "markov_decision": ("markov", "markov_agent"),
    "residual_action_decision": ("residual", "residual_agent"),
}
GATE_CALLS = {
    "select_action_with_supervisor",
    "select_supervisor_gated_horizon_action",
    "select_supervisor_gated_continuous_action",
}


class RestorationError(RuntimeError):
    """A capture/reconstruction inconsistency, not a plant or gate failure."""


@dataclass(frozen=True)
class GateSite:
    key: str
    channel: str
    agent_name: str
    result_name: str
    line: int
    suffix: Any


def rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _call_name(node: ast.Call) -> str:
    return node.func.id if isinstance(node.func, ast.Name) else (
        node.func.attr if isinstance(node.func, ast.Attribute) else ""
    )


def _step_code(statements: list[ast.stmt], filename: str, line_offset=0):
    # A one-item loop retains the original outer-loop continue/break semantics.
    tree = ast.Module(body=[ast.For(
        target=ast.Name(id="__e1_once", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
        body=deepcopy(statements), orelse=[],
    )], type_ignores=[])
    tree = ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, line_offset)
    return compile(tree, filename, "exec")


def _find_sites(body, category, filename, source_offset, tail=()):
    sites = []
    for index, statement in enumerate(body):
        remaining = list(body[index + 1:]) + list(tail)
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            if _call_name(statement.value) in GATE_CALLS:
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise ValueError("E1 requires a named gate-result assignment.")
                result_name = statement.targets[0].id
                if category == "combined":
                    channel, agent_name = AGENT_NAMES[result_name]
                else:
                    channel, agent_name = category, "rl_agent" if category == "markov" else "agent"
                key = f"{channel}:{source_offset + statement.lineno}"
                sites.append(GateSite(
                    key, channel, agent_name, result_name,
                    source_offset + statement.lineno,
                    _step_code([statement] + remaining, filename, source_offset),
                ))
        if isinstance(statement, ast.If):
            sites.extend(_find_sites(statement.body, category, filename, source_offset, remaining))
            sites.extend(_find_sites(statement.orelse, category, filename, source_offset, remaining))
        elif isinstance(statement, (ast.For, ast.While, ast.Try, ast.With)):
            # The active gate sites are nested only inside if branches. Revisit
            # this contract explicitly if a future controller changes that.
            for descendant in ast.walk(statement):
                if isinstance(descendant, ast.Call) and _call_name(descendant) in GATE_CALLS:
                    raise ValueError("E1 does not support a gate inside a nested loop/try/with.")
    return sites


class _GlobalizeNonlocal(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Nonlocal(self, node):
        if not set(node.names).issubset(self.names):
            raise ValueError("Unsupported inner nonlocal binding in E1 adapter.")
        return ast.copy_location(ast.Global(names=node.names), node)


class _Suspensions(ast.NodeTransformer):
    def __init__(self, sites, source_offset):
        self.by_line = {site.line - source_offset: site for site in sites}

    def visit_FunctionDef(self, node):
        # Local numerical helpers are not suspension points.
        return node

    def visit_Assign(self, node):
        site = self.by_line.get(node.lineno)
        if site is None:
            return node
        before = ast.parse(
            f"if __e1_session.wants({site.key!r}, globals()):\n"
            f"    yield ('before', {site.key!r})\n"
        ).body[0]
        after = ast.parse(
            f"if __e1_session.is_active({site.key!r}):\n"
            f"    yield ('after', {site.key!r})\n"
        ).body[0]
        return [ast.copy_location(before, node), node, ast.copy_location(after, node)]


class CompiledRunner:
    """Reusable AST and state contract for one authoritative runner function."""

    def __init__(self, function, category):
        self.original = function
        self.category = category
        source, start = inspect.getsourcelines(function)
        self.source_start = start
        self.source_end = start + len(source) - 1
        self.filename = inspect.getsourcefile(function) or "<e1-runner>"
        self.source = textwrap.dedent("".join(source))
        root = ast.parse(self.source).body[0]
        self.local_names = set(function.__code__.co_varnames) | set(function.__code__.co_cellvars)
        self.local_names.update(function.__code__.co_freevars)
        loops = [node for node in ast.walk(root) if isinstance(node, ast.For)
                 and isinstance(node.iter, ast.Call) and _call_name(node.iter) == "range"
                 and any(isinstance(part, ast.Name) and part.id == "nFE" for part in ast.walk(node.iter))]
        candidates = []
        for loop in loops:
            found = _find_sites(loop.body, category, self.filename, start - 1)
            if found:
                candidates.append((loop, found))
        if len(candidates) != 1:
            raise ValueError(f"Expected one E1 control loop for {category}; found {len(candidates)}.")
        loop, sites = candidates[0]
        self.step_name = loop.target.id
        self.sites = {site.key: site for site in sites}
        expected = 4 if category == "combined" else 1
        if len(sites) != expected:
            raise ValueError(f"Expected {expected} gate sites for {category}, found {len(sites)}.")
        self.step_code = _step_code(loop.body, self.filename, start - 1)
        transformed = deepcopy(root)
        transformed.decorator_list = []
        transformed.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
        transformed.body.insert(0, ast.Global(names=sorted(self.local_names)))
        transformed = _GlobalizeNonlocal(self.local_names).visit(transformed)
        # Visit the runner body explicitly; nested numerical helper definitions
        # retain their ordinary argument-local scopes.
        suspension = _Suspensions(sites, start - 1)
        for node in ast.walk(transformed):
            if isinstance(node, ast.For) and node.lineno == loop.lineno:
                node.body = [item for statement in node.body
                             for item in _as_list(suspension.visit(statement))]
                break
        generator_tree = ast.fix_missing_locations(ast.Module(body=[transformed], type_ignores=[]))
        ast.increment_lineno(generator_tree, start - 1)
        self.generator_code = compile(generator_tree, self.filename, "exec")

    def environment(self, session, args, kwargs):
        environment = dict(self.original.__globals__)
        bound = inspect.signature(self.original).bind(*args, **kwargs)
        bound.apply_defaults()
        environment.update(bound.arguments)
        if self.original.__closure__:
            environment.update(zip(self.original.__code__.co_freevars,
                                   (cell.cell_contents for cell in self.original.__closure__)))
        environment["__e1_session"] = session
        exec(self.generator_code, environment, environment)
        return environment


def _as_list(value):
    return value if isinstance(value, list) else [value]


class BufferDimensions:
    """Inference-only buffer interface: dimensions/counts without mutable replay."""

    def __init__(self, buffer):
        for name in ("action_dim", "state_dim", "capacity", "size", "ptr", "pos"):
            if hasattr(buffer, name):
                setattr(self, name, getattr(buffer, name))

    def __len__(self):
        return int(getattr(self, "size", 0))


def _nothing(*args, **kwargs):
    return None


def _selected(result, source):
    action = result.policy_action if source == "policy" else result.supervisor_action
    if not np.all(np.isfinite(np.asarray(action))):
        raise ValueError(f"Cannot force a nonfinite {source} proposal.")
    return replace(result, action=deepcopy(action), selected_source=2 if source == "policy" else 1)


def frozen_agent(agent, *, initial=None, initial_source=None, continuation="gated"):
    """Clone all mutable agent attributes; share read-only network parameters.

    The training runner is suspended while branches run. Sharing its networks
    therefore preserves exact decision-time weights without copying optimizers
    or allocating another set of CUDA models. Branch push/train methods are
    disabled. Saved snapshots serialize independent copies of these weights.
    """
    clone = copy(agent)
    agent_type = type(agent)
    memo = {id(agent): clone}
    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            memo[id(value)] = value
        elif isinstance(value, torch.optim.Optimizer):
            memo[id(value)] = None
    buffer = BufferDimensions(agent.buffer)
    memo[id(agent.buffer)] = buffer
    clone.__dict__ = deepcopy(vars(agent), memo)
    clone.buffer = buffer
    for name in ("push", "push_supervised", "train_step", "flush_pending_transitions", "flush_nstep"):
        if hasattr(clone, name):
            setattr(clone, name, _nothing)
    selection = getattr(agent_type, "select_action_with_supervisor")
    action_selection = getattr(agent_type, "select_action", None)
    first = {"used": False}

    def choose(self, state, *args, **kwargs):
        if initial is not None and not first["used"]:
            first["used"] = True
            if not np.allclose(np.asarray(state), initial["state"], rtol=1e-6, atol=1e-7):
                raise RestorationError("Restored critic observation differs from the captured decision.")
            supplied = kwargs.get("supervisor_action", args[0] if args else None)
            if supplied is None or not np.allclose(np.asarray(supplied),
                                                 np.asarray(initial["decision"].supervisor_action),
                                                 rtol=1e-6, atol=1e-7):
                raise RestorationError("Restored supervisor proposal differs from the captured decision.")
            return _selected(initial["decision"], initial_source)
        kwargs["explore"] = False
        kwargs["test"] = True
        result = selection(self, state, *args, **kwargs)
        return _selected(result, continuation) if continuation in {"policy", "supervisor"} else result

    def act(self, *args, **kwargs):
        if action_selection is None:
            raise AttributeError("Agent has no select_action method.")
        signature = inspect.signature(action_selection)
        if "explore" in signature.parameters:
            kwargs["explore"] = False
        if "test" in signature.parameters:
            kwargs["test"] = True
        return action_selection(self, *args, **kwargs)

    clone.select_action_with_supervisor = types.MethodType(choose, clone)
    if action_selection is not None:
        clone.select_action = types.MethodType(act, clone)
    if hasattr(agent, "act_train"):
        def act_train(self, state, *args, **kwargs):
            return agent_type.act_eval(self, state)
        clone.act_train = types.MethodType(act_train, clone)
    return clone


def _private_functions(value, environment, compiled, found, seen):
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, types.FunctionType):
        if value.__globals__ is environment or (
            value.__code__.co_filename == compiled.filename
            and (value.__dict__.get("_e1_runner_owner") == compiled.original.__qualname__
                 or compiled.source_start <= value.__code__.co_firstlineno <= compiled.source_end)
        ):
            found.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            _private_functions(child, environment, compiled, found, seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _private_functions(child, environment, compiled, found, seen)


def agents_in(environment, local_names):
    found = {}
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if hasattr(value, "select_action_with_supervisor") and hasattr(value, "buffer"):
            found[id(value)] = value
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for name in local_names:
        if name in environment:
            visit(environment[name])
    return found


def clone_environment(environment, compiled, *, selected_agent=None, initial=None,
                      initial_source=None, continuation="gated"):
    cloned = dict(environment)
    memo = {id(environment): cloned}
    for identity, agent in agents_in(environment, compiled.local_names).items():
        memo[identity] = frozen_agent(
            agent, initial=initial if agent is selected_agent else None,
            initial_source=initial_source, continuation=continuation,
        )
    functions = []
    seen = set()
    for name in compiled.local_names:
        if name in environment:
            _private_functions(environment[name], environment, compiled, functions, seen)
    for function in functions:
        if function.__closure__:
            raise ValueError("Unexpected closure in a globalized runner helper.")
        rebound = types.FunctionType(function.__code__, cloned, function.__name__)
        rebound.__qualname__ = function.__qualname__
        rebound._e1_runner_owner = compiled.original.__qualname__
        memo[id(function)] = rebound
    payload = {name: environment[name] for name in compiled.local_names if name in environment}
    cloned.update(deepcopy(payload, memo))
    for function in functions:
        memo[id(function)].__defaults__ = deepcopy(function.__defaults__, memo)
        memo[id(function)].__kwdefaults__ = deepcopy(function.__kwdefaults__, memo)
    return cloned


def network_modes(environment, compiled):
    modes = {}
    for agent in agents_in(environment, compiled.local_names).values():
        for value in vars(agent).values():
            if isinstance(value, torch.nn.Module):
                for module in value.modules():
                    modes[id(module)] = (module, module.training)
    return list(modes.values())


def freeze_normalization(environment, compiled):
    """Only validation branches freeze the existing running normalizers."""
    from utils.observation_conditioning import PhysicalXhatStateConditioner
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, PhysicalXhatStateConditioner):
            original = value.transform

            def transform(self, *args, __original=original, **kwargs):
                kwargs["update"] = False
                return __original(*args, **kwargs)
            value.transform = types.MethodType(transform, value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
    for name in compiled.local_names:
        if name in environment:
            visit(environment[name])


def execute_rollout(environment, compiled, site, steps):
    """Finish the sampled decision, then reuse the full control-step AST."""
    freeze_normalization(environment, compiled)
    start = int(environment[compiled.step_name])
    n_steps = int(environment.get("nFE", environment.get("ctx", {}).get("nFE", 0)))
    if start + steps > n_steps:
        raise ValueError("Rollout would exceed the original reference/disturbance schedule.")
    exec(site.suffix, environment, environment)
    for step in range(start + 1, start + steps):
        environment[compiled.step_name] = step
        exec(compiled.step_code, environment, environment)
    return environment


def install_runner(category, session_factory):
    """Return a context manager for a process-local runner replacement."""
    import importlib
    from contextlib import contextmanager

    @contextmanager
    def installed():
        module_name, function_name = RUNNER_TARGETS[category]
        module = importlib.import_module(module_name)
        original = getattr(module, function_name)
        compiled = CompiledRunner(original, category)

        @functools.wraps(original)
        def run(*args, **kwargs):
            session = session_factory(compiled, args, kwargs)
            environment = compiled.environment(session, args, kwargs)
            generator = environment[original.__name__]()
            try:
                while True:
                    kind, key = next(generator)
                    if kind == "before":
                        session.before(compiled.sites[key], environment)
                    else:
                        session.after(compiled.sites[key], environment)
            except StopIteration as stopped:
                session.finish(environment)
                return stopped.value
            finally:
                session.detach()
                generator.close()

        setattr(module, function_name, run)
        try:
            yield compiled
        finally:
            setattr(module, function_name, original)
    return installed()
