"""Full E1 training checkpoints at completed control-cycle boundaries.

The original statements and gate suspensions execute in their original order.
Only the private compilation gains a boundary yield and a restart entry point.
These are trusted local pickle files, for the same environment and source hash.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import types

import cloudpickle

from review_experiments.e1.runtime import (
    CompiledRunner, RUNNER_TARGETS, _GlobalizeNonlocal, _Suspensions, _as_list,
    agents_in, rng_state, restore_rng,
)

_LOAD_ENV = None


def _helper(code, name, defaults, kwdefaults, attributes):
    if _LOAD_ENV is None:
        raise RuntimeError("Runner helper loaded without a restoration environment.")
    function = types.FunctionType(code, _LOAD_ENV, name, defaults)
    function.__kwdefaults__ = kwdefaults
    function.__dict__.update(attributes)
    return function


class TrainingPickler(cloudpickle.CloudPickler):
    """Bind private numerical helpers to the one restored, advancing namespace."""
    def __init__(self, stream, environment):
        super().__init__(stream, protocol=5)
        self.environment = environment

    def reducer_override(self, value):
        if isinstance(value, types.FunctionType) and value.__globals__ is self.environment:
            if value.__closure__:
                raise ValueError(f"Unsupported closure in runner helper {value.__name__}.")
            return (_helper, (value.__code__, value.__name__, value.__defaults__,
                              value.__kwdefaults__, value.__dict__))
        return super().reducer_override(value)


def load_payload(path, environment):
    global _LOAD_ENV
    previous, _LOAD_ENV = _LOAD_ENV, environment
    try:
        with Path(path).open("rb") as stream:
            return cloudpickle.load(stream)
    finally:
        _LOAD_ENV = previous


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class CheckpointRunner(CompiledRunner):
    def __init__(self, function, category):
        super().__init__(function, category)
        root = ast.parse(self.source).body[0]
        loops = [(index, node) for index, node in enumerate(root.body)
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                 and node.target.id == self.step_name
                 and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                 and node.iter.func.id == "range" and len(node.iter.args) == 1
                 and isinstance(node.iter.args[0], ast.Name) and node.iter.args[0].id == "nFE"]
        if len(loops) != 1:
            raise ValueError("Recovery requires one top-level zero-based range(nFE) loop.")
        index, loop = loops[0]
        # No controller AST statement is edited. The only additions are the
        # restore/prologue choice, loop starting index, and a completed-step yield.
        root.decorator_list = []
        root.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
        prologue = root.body[:index]
        tail = root.body[index + 1:]
        suspension = _Suspensions(list(self.sites.values()), self.source_start - 1)
        loop.body = [item for statement in loop.body for item in _as_list(suspension.visit(statement))]
        loop.body.append(ast.parse(
            f"if __recovery_manager.due({self.step_name}, globals()):\n"
            "    yield ('checkpoint', None)\n").body[0])
        loop.iter.args.insert(0, ast.Name(id="__recovery_next_step", ctx=ast.Load()))
        choice = ast.If(
            test=ast.Attribute(value=ast.Name(id="__recovery_manager", ctx=ast.Load()), attr="has_checkpoint", ctx=ast.Load()),
            body=ast.parse("__recovery_manager.restore(globals(), __e1_session)").body,
            orelse=prologue,
        )
        root.body = [ast.Global(names=sorted(self.local_names)), choice, loop, *tail]
        root = _GlobalizeNonlocal(self.local_names).visit(root)
        tree = ast.fix_missing_locations(ast.Module(body=[root], type_ignores=[]))
        ast.increment_lineno(tree, self.source_start - 1)
        self.generator_code = compile(tree, self.filename, "exec")


class CheckpointManager:
    def __init__(self, directory, sources, *, every_cycles=10, stop_after_checkpoint=None):
        self.directory = Path(directory).resolve()
        self.sources = sources
        self.every_cycles = every_cycles
        self.stop_after_checkpoint = stop_after_checkpoint
        self.index = self.directory / "latest.json"
        self.has_checkpoint = self.index.is_file()
        self.compiled = None

    def due(self, step, environment):
        size = int(environment["time_in_sub_episodes"])
        completed = int(step) + 1
        cycle = completed // size
        return completed % size == 0 and (cycle == 1 or cycle % self.every_cycles == 0
                                          or completed == int(environment["nFE"]))

    def save(self, environment, session):
        if getattr(session, "active_key", None) or getattr(session, "method_restore", None):
            raise RuntimeError("Cannot checkpoint an active counterfactual branch or temporary agent method.")
        self.directory.mkdir(parents=True, exist_ok=True)
        rng = rng_state()
        step = int(environment[self.compiled.step_name]) + 1
        started = time.perf_counter()
        # Retain real replay arrays, n-step queues, optimizers, target networks,
        # counters, plant, observer, histories and gate/session sampling state.
        local_state = {name: environment[name] for name in self.compiled.local_names if name in environment}
        session_state = {name: value for name, value in vars(session).items()
                         if name not in {"compiled", "started"}}
        payload = {"schema": 1, "completed_steps": step, "source_hashes": self.sources,
                   "runner_sha256": hashlib.sha256(self.compiled.source.encode()).hexdigest(),
                   "environment": local_state, "session": session_state, "rng": rng,
                   "session_elapsed": time.perf_counter() - getattr(session, "started", time.perf_counter())}
        # Rotate two complete, atomic slots; a failed write leaves latest intact.
        slot = 1 - int(json.loads(self.index.read_text())["slot"]) if self.index.is_file() else 0
        path = self.directory / f"training_state_{slot}.pkl"
        temporary = path.with_suffix(".pkl.tmp")
        try:
            with temporary.open("wb") as stream:
                TrainingPickler(stream, environment).dump(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            agent_report = {}
            for identity, agent in agents_in(environment, self.compiled.local_names).items():
                name = next((key for key in ("horizon_agent", "markov_agent", "weight_agent", "residual_agent", "agent")
                             if environment.get(key) is agent), str(identity))
                optimizers = [key for key, val in vars(agent).items()
                              if hasattr(val, "state_dict") and val.__class__.__module__.startswith("torch.optim")]
                agent_report[name] = {"train_steps": int(getattr(agent, "train_steps", 0)),
                                      "replay_size": len(agent.buffer) if hasattr(agent, "buffer") else None,
                                      "optimizers": optimizers}
            write_json(self.index, {"schema": 1, "slot": slot, "file": path.name, "sha256": digest(path),
                                   "completed_steps": step,
                                   "completed_cycles": step // int(environment["time_in_sub_episodes"]),
                                   "bytes": path.stat().st_size, "agents": agent_report,
                                   "source_hashes": self.sources, "saved_at_unix": time.time()})
            self.has_checkpoint = True
            print(f"Full training checkpoint: cycle {step // int(environment['time_in_sub_episodes'])}; "
                  f"step {step}; {path.stat().st_size / 1024**2:.1f} MiB; "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
        finally:
            # Disk I/O must not consume any training random stream.
            restore_rng(rng)
        if self.stop_after_checkpoint is not None and step >= self.stop_after_checkpoint:
            raise CheckpointStop("Technical checkpoint/resume validation stop.")

    def restore(self, environment, session):
        index = json.loads(self.index.read_text(encoding="utf-8"))
        if index["source_hashes"] != self.sources:
            raise RuntimeError("Checkpoint dependencies differ; refusing to resume another implementation.")
        path = self.directory / index["file"]
        if path.name not in {"training_state_0.pkl", "training_state_1.pkl"} or path.parent != self.directory:
            raise ValueError("Invalid checkpoint slot.")
        if digest(path) != index["sha256"]:
            raise RuntimeError("Checkpoint checksum differs; preserve it for diagnosis.")
        payload = load_payload(path, environment)
        if payload["schema"] != 1 or payload["source_hashes"] != self.sources:
            raise RuntimeError("Checkpoint schema or source provenance differs.")
        if payload["runner_sha256"] != hashlib.sha256(self.compiled.source.encode()).hexdigest():
            raise RuntimeError("Checkpoint runner source differs.")
        environment.update(payload["environment"])
        environment["__recovery_next_step"] = payload["completed_steps"]
        state = payload["session"]
        # Preserve the records list owned by the original worker/analysis caller.
        records = getattr(session, "all_records", None)
        if records is not None:
            records[:] = state["all_records"]
            state["all_records"] = records
        if Path(state["output"]).resolve() != Path(session.output).resolve():
            raise RuntimeError("A training checkpoint must resume inside its original new-attempt directory.")
        vars(session).update(state)
        session.started = time.perf_counter() - payload["session_elapsed"]
        self._reconcile_capture(session)
        restore_rng(payload["rng"])
        print(f"Resumed full training state after step {payload['completed_steps']}; "
              f"{len(session.all_records)} matched decisions retained.", flush=True)

    @staticmethod
    def _reconcile_capture(session):
        """Archive post-checkpoint suffix files before deterministic replay."""
        path = Path(session.output) / "decisions.jsonl"
        records = session.all_records
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        current = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                current.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1 or len(current) < len(records):
                    raise RuntimeError("Decision log is corrupt before its checkpoint prefix.")
        if current[:len(records)] != records:
            raise RuntimeError("Saved decision log does not match the checkpoint prefix.")
        retained = {str((Path(session.output) / record[key]).resolve())
                    for record in records for key in ("trace", "snapshot") if record.get(key)}
        orphans = [source for folder in ("traces", "snapshots")
                   for source in (Path(session.output) / folder).glob("*")
                   if source.is_file() and str(source.resolve()) not in retained]
        if len(current) == len(records) and len(lines) == len(records) and not orphans:
            return
        archive = Path(session.output) / "replayed_suffixes" / str(time.time_ns())
        archive.mkdir(parents=True, exist_ok=False)
        (archive / "decisions.jsonl").write_text("\n".join(lines[len(records):]) + "\n", encoding="utf-8")
        for record in current[len(records):]:
            for key in ("trace", "snapshot"):
                if record.get(key):
                    source = (Path(session.output) / record[key]).resolve()
                    if not source.is_relative_to(Path(session.output).resolve()):
                        raise ValueError("Capture path leaves its attempt directory.")
                    if source.exists():
                        destination = archive / record[key]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source.replace(destination)
        # Native interruption can leave a trace not yet in decisions.jsonl.
        for folder in ("traces", "snapshots"):
            for source in (Path(session.output) / folder).glob("*"):
                if source.is_file() and str(source.resolve()) not in retained:
                    destination = archive / folder / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(destination)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        temporary.replace(path)


class CheckpointStop(RuntimeError):
    """Controlled technical test stop after a durable checkpoint."""


def install_runner(category, session_factory, manager):
    @contextmanager
    def installed():
        module_name, name = RUNNER_TARGETS[category]
        module = importlib.import_module(module_name)
        original = getattr(module, name)
        compiled = CheckpointRunner(original, category)
        manager.compiled = compiled

        @functools.wraps(original)
        def run(*args, **kwargs):
            session = session_factory(compiled, args, kwargs)
            environment = compiled.environment(session, args, kwargs)
            environment.update(__recovery_manager=manager, __recovery_next_step=0)
            generator = environment[original.__name__]()
            try:
                while True:
                    kind, key = next(generator)
                    if kind == "checkpoint":
                        manager.save(environment, session)
                    elif kind == "before":
                        session.before(compiled.sites[key], environment)
                    else:
                        session.after(compiled.sites[key], environment)
            except StopIteration as stopped:
                session.finish(environment)
                return stopped.value
            finally:
                session.detach()
                generator.close()

        setattr(module, name, run)
        try:
            yield compiled
        finally:
            setattr(module, name, original)
    return installed()
