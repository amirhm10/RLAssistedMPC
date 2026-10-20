"""Atomic paired Python training state and native Aspen binary checkpoints."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import cloudpickle

from systems.distillation.plant import DistillationColumnAspen
from revision_recovery.checkpoints import (
    CheckpointManager, CheckpointStop, TrainingPickler, load_payload, digest,
)
from review_experiments.e1.runtime import agents_in, rng_state, restore_rng
from review_experiments.e1.study import write_json, json_safe
from .runtime import (python_plant_state, import_native, digest_state, integrity_payload,
                      E1NativeSnapshot, native_metadata)

_RESTORING_PLANT = None


def restore_plant(state):
    if _RESTORING_PLANT is None:
        raise RuntimeError('Native training state loaded without an owned initialized column.')
    active_path = _RESTORING_PLANT.path
    active_root = getattr(_RESTORING_PLANT, '_e1_project_root', str(Path(active_path).parent))
    vars(_RESTORING_PLANT).update(dict(state, path=active_path, _e1_project_root=active_root))
    return _RESTORING_PLANT


class NativeTrainingPickler(TrainingPickler):
    def reducer_override(self, value):
        if isinstance(value, DistillationColumnAspen):
            return restore_plant, (python_plant_state(value),)
        return super().reducer_override(value)


@contextmanager
def restoring_into(plant):
    global _RESTORING_PLANT
    previous, _RESTORING_PLANT = _RESTORING_PLANT, plant
    try:
        yield
    finally:
        _RESTORING_PLANT = previous


class NativeCheckpointManager(CheckpointManager):
    def __init__(self, *args, technical_stop=None, technical_resume_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.technical_stop = technical_stop
        self.technical_resume_steps = technical_resume_steps
        self.restored_step = None

    def due(self, step, environment):
        if self.technical_stop is not None:
            completed = int(step) + 1
            target = (self.restored_step + self.technical_resume_steps) if self.restored_step is not None else self.technical_stop
            return completed == target
        return super().due(step, environment)

    def save(self, environment, session):
        if session.active_key or session.method_restore or session.native is not None:
            raise RuntimeError('Cannot checkpoint an active branch.')
        self.directory.mkdir(parents=True, exist_ok=True)
        rng = rng_state()
        step = int(environment[self.compiled.step_name]) + 1
        slot = 1 - json.loads(self.index.read_text())['slot'] if self.index.exists() else 0
        path = self.directory / f'training_state_{slot}.pkl'
        native = E1NativeSnapshot(environment['system'], {}, self.directory / f'native_{slot}_{step}')
        native_record = native_metadata(native)
        state = {name: environment[name] for name in self.compiled.local_names if name in environment}
        saved_session = {name: value for name, value in vars(session).items() if name not in {'compiled', 'started', 'branch_plant'}}
        payload = {'schema': 2, 'completed_steps': step, 'source_hashes': self.sources,
                   'runner_sha256': hashlib.sha256(self.compiled.source.encode()).hexdigest(),
                   'environment': state, 'session': saved_session, 'rng': rng, 'native': native_record,
                   'session_elapsed': time.perf_counter() - session.started}
        if self.technical_stop is not None:
            payload['technical_python_state_sha256'] = digest_state(integrity_payload(environment, self.compiled))
        try:
            temporary = path.with_suffix('.pkl.tmp')
            with temporary.open('wb') as stream:
                NativeTrainingPickler(stream, environment).dump(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            agents = {}
            for identity, agent in agents_in(environment, self.compiled.local_names).items():
                name = next((key for key in sorted(self.compiled.local_names) if environment.get(key) is agent), str(identity))
                agents[name] = {'train_steps': int(agent.train_steps), 'replay_size': int(agent.buffer.size)}
            write_json(self.index, {'schema': 2, 'slot': slot, 'file': path.name, 'sha256': digest(path),
                       'completed_steps': step, 'completed_cycles': step // int(environment['time_in_sub_episodes']),
                       'bytes': path.stat().st_size, 'agents': agents, 'native': native_record,
                       'source_hashes': self.sources, 'training_resume': True, 'saved_at_unix': time.time()})
            self.has_checkpoint = True
            print(f'Full native/Python E1 checkpoint: step {step}; {path.stat().st_size / 1024**2:.1f} MiB', flush=True)
        finally:
            environment['system'].sim.Results.Delete(native.snapshot.native)
            restore_rng(rng)
        if self.technical_stop is not None:
            raise CheckpointStop('Bounded native integration checkpoint stop.')

    def restore(self, environment, session):
        index = json.loads(self.index.read_text())
        if index['schema'] != 2 or index['source_hashes'] != self.sources:
            raise RuntimeError('Native checkpoint source provenance differs.')
        path = self.directory / index['file']
        if path.name not in {'training_state_0.pkl', 'training_state_1.pkl'} or digest(path) != index['sha256']:
            raise RuntimeError('Native Python checkpoint slot/checksum differs.')
        # The entrypoint has initialized one new owned case before this runner.
        contexts = [environment.get(name) for name in ('runtime_ctx', 'ctx')]
        plant = next((value['system'] for value in contexts if isinstance(value, dict) and 'system' in value), None)
        if plant is None:
            raise RuntimeError('Checkpoint restore has no initialized owned column.')
        with restoring_into(plant):
            payload = load_payload(path, environment)
        if payload['schema'] != 2 or payload['source_hashes'] != self.sources or payload['runner_sha256'] != hashlib.sha256(self.compiled.source.encode()).hexdigest():
            raise RuntimeError('Native checkpoint payload provenance differs.')
        environment.update(payload['environment'])
        environment['__recovery_next_step'] = payload['completed_steps']
        state = payload['session']
        if Path(state['output']).resolve() != session.output.resolve():
            raise RuntimeError('Native resume must use its original isolated attempt.')
        session.all_records[:] = state['all_records']
        state['all_records'] = session.all_records
        vars(session).update(state)
        session.started = time.perf_counter() - payload['session_elapsed']
        self._reconcile_capture(session)
        primes = import_native(plant, payload['native'])
        full_hash = payload.get('technical_python_state_sha256')
        if full_hash and digest_state(integrity_payload(environment, self.compiled)) != full_hash:
            raise RuntimeError('Restored complete learner/controller state differs from its saved checkpoint.')
        restore_rng(payload['rng'])
        self.restored_step = payload['completed_steps']
        write_json(self.directory / 'last_restore.json', {'completed_steps': self.restored_step,
                   'native_checksum_verified': True, 'python_checksum_verified': True,
                   'zero_time_initializations': primes, 'agents': index['agents'], 'full_python_state_sha256': full_hash})
        print(f'Resumed native Aspen plus learner/optimizer/replay/controller state at step {self.restored_step}', flush=True)
