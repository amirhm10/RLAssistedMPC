"""Serial native restores around unchanged E1 controller/learner branches.

Every branch uses a separate private Aspen application and owns its Python
arrays, observer, optimizer warm starts, histories, conditioners and frozen
inference agents. The training application's native state is never restored or
initialized by diagnostics, including failure paths.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from unittest.mock import patch

import numpy as np
import torch

from systems.distillation.plant import DistillationColumnAspen
from review_experiments.e1.runtime import agents_in, rng_state, restore_rng, RestorationError
from review_experiments.e1.study import StudySession, json_safe
from repair_aspen_snapshot import ReadyNativeSnapshot

COM_FIELDS = frozenset(('ad', 'sim', 'fsheet', 'streams', 'block', 'col', 'feed'))
_COPY_TARGET = None


def load_project(plant, metadata):
    """Reopen a saved initialized private document with its snapshot database.

    ExportasBinary produces a result, which LoadSnapshot rejects. The saved
    document plus its unlocked snapshot.bak preserves rewindable native rows.
    This operation is restricted to the caller's owned private application.
    """
    from systems.distillation.config import resolve_snapshot_dir
    from column_review.common import digest
    for key in ('file', 'project_file'):
        if digest(metadata[key]) != metadata[key + '_sha256']:
            raise RestorationError('Native project checkpoint checksum differs.')
    previous_model = Path(plant.path)
    if previous_model.name != Path(metadata['project_file']).name:
        raise RestorationError('Native project model filename differs.')
    # Old COM references can retain the previous document's snapshot file even
    # after CloseDocument. A fresh owned directory avoids overwriting that file.
    root = Path(getattr(plant, '_e1_project_root', previous_model.parent))
    plant._e1_project_root = str(root)
    directory = root / ('load_' + uuid.uuid4().hex[:8])
    directory.mkdir()
    model = directory / previous_model.name
    plant.ad.CloseDocument(False)
    plant._e1_document_open = False
    for field in COM_FIELDS - {'ad'}:
        vars(plant).pop(field, None)
    import gc
    gc.collect()
    shutil.copyfile(metadata['project_file'], model)
    destination = Path(resolve_snapshot_dir(model)) / 'snapshot.snp'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metadata['file'], destination)
    plant.ad.openDocument(str(model))
    plant.path = str(model)
    plant._e1_document_open = True
    plant.sim = plant.ad.Simulation
    plant.fsheet = plant.sim.Flowsheet
    plant.streams, plant.block = plant.fsheet.Streams, plant.fsheet.Blocks
    plant.col, plant.feed = plant.block('C2S'), plant.streams('Feed')
    results = plant.sim.Results
    results.Refresh()
    native = results.FindSnapshot(metadata['name'])
    if isinstance(native, (int, np.integer)):
        native = results.GetSnapshot(int(native))
    return native


class E1NativeSnapshot(ReadyNativeSnapshot):
    """Retain transient initialization diagnostics before enforcing the contract."""
    def __init__(self, plant, controller, output, *, target=None):
        super().__init__(plant, controller, output)
        self.capture_plant = plant
        self.plant = target or plant
        from column_review.pilot import profile
        from column_review.common import digest
        from systems.distillation.config import resolve_snapshot_dir
        before = profile(plant)[1]
        plant.ad.SaveDocument()
        plant.ad.SaveDocument()  # backup must contain the just-flushed snapshot database
        after = profile(plant)[1]
        if not np.array_equal(before, after):
            raise RestorationError('Saving the private document changed live native state.')
        directory = self.snapshot.path.with_suffix('')
        directory.mkdir(exist_ok=False)
        project = directory / Path(plant.path).name
        database = directory / 'snapshot.snp'
        shutil.copyfile(plant.path, project)
        shutil.copyfile(Path(resolve_snapshot_dir(plant.path)) / 'snapshot.bak', database)
        self.snapshot.exported_result_path = self.snapshot.path
        self.snapshot.path = database
        self.metadata = {'file':str(database), 'file_sha256':digest(database),
                         'project_file':str(project), 'project_file_sha256':digest(project),
                         'name':self.snapshot.name, 'python_state':self.snapshot.python_state,
                         'internal':self.snapshot.internal.tolist(),
                         'reinitialization_output_tolerance':[1e-6,2e-3], 'live_save_unchanged':True}
        (directory/'metadata.json').write_text(json.dumps(json_safe(self.metadata),indent=2)+'\n',encoding='utf-8')
        self.restored_snapshot = self.snapshot
        if self.plant is not plant:
            from copy import copy
            self.restored_snapshot = copy(self.snapshot)
            self.restored_snapshot.plant = self.plant
            self.restored_snapshot.native = load_project(self.plant, self.metadata)

    def restore(self):
        from column_review.pilot import profile
        snapshot=self.restored_snapshot
        self.plant.sim.Results.LoadSnapshot(snapshot.native)
        raw=profile(self.plant)[1]
        # Native snapshots do not reliably overwrite fixed equipment
        # specifications after reopening a saved case. Restore the exact
        # captured physical feed and two actuator specifications as well.
        self.plant.feed.FmR.Value=float(snapshot.internal[1])
        self.plant.col.Reflux.FmR.Value=float(snapshot.internal[2])
        self.plant.col.QRebR.Value=float(snapshot.internal[3])
        self.plant.current_input=snapshot.python_state['current_input'].copy()
        before=profile(self.plant)[1]
        immediate=np.array([1e-9,1e-3,1e-3,1e-6]+[v for _ in range(6) for v in (1e-5,1e-9)])
        loaded={'expected':snapshot.internal.tolist(),'raw_loaded':raw.tolist(),'corrected_loaded':before.tolist()}
        with self.snapshot.path.with_suffix('.loaded.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(loaded)+'\n')
        bad=np.flatnonzero(np.abs(before-snapshot.internal)>immediate)
        if len(bad):
            raise RestorationError(f'Immediate native restore mismatch at coordinates {bad.tolist()}: {(before-snapshot.internal)[bad].tolist()}')
        self.plant.sim.Step(True, False)
        after = profile(self.plant)[1]
        row = {'before': before.tolist(), 'after': after.tolist(), 'difference': (after-before).tolist()}
        self.priming_records.append(row)
        with self.snapshot.path.with_suffix('.restores.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        # Reinitialization reconstructs the integrator's algebraic state. These
        # bounds are explicit physical fidelity limits, separate from the much
        # tighter repeated-continuation checks. The live plant is never loaded
        # or initialized for a diagnostic branch.
        # 0.002 temperature units is 1% of the controller's minimum 0.2
        # tracking band. Internal compositions use a separate 0.0001 bound;
        # the controlled tray-24 composition remains much tighter at 1e-6.
        tolerance = np.array([1e-9, 1e-3, 1e-3, 1e-6] + [v for _ in range(6) for v in (2e-3, 1e-4)])
        tolerance[9] = 1e-6  # controlled tray-24 ethane composition
        bad = np.flatnonzero(np.abs(after-before) > tolerance)
        if len(bad):
            raise RestorationError(f'Native initialization changed profile coordinates {bad.tolist()}; differences={(after-before)[bad].tolist()}')
        self.plant.current_output = self.snapshot.python_state['current_output'].copy()
        return deepcopy(snapshot.python_state['controller'])

    def release(self):
        if self.plant is not self.capture_plant:
            self.plant.sim.Results.Delete(self.restored_snapshot.native)
        self.capture_plant.sim.Results.Delete(self.snapshot.native)


def native_metadata(native):
    return native.metadata


def python_plant_state(plant):
    return {key: value for key, value in vars(plant).items() if key not in COM_FIELDS}


def copy_column(plant, memo):
    clone = object.__new__(BranchColumn)
    memo[id(plant)] = clone
    target = _COPY_TARGET or plant
    clone.__dict__ = {key: getattr(target, key) if key in COM_FIELDS else deepcopy(value, memo)
                      for key, value in vars(plant).items()}
    return clone


class BranchColumn(DistillationColumnAspen):
    __deepcopy__ = copy_column

    def step(self, disturbances=None):
        before = float(self.sim.Time)
        super().step(disturbances)
        if abs(float(self.sim.Time) - before - self.delta_t) > 1.1e-6:
            raise RestorationError('Counterfactual plant did not advance one original communication interval.')

    def close(self, *args, **kwargs):
        raise RestorationError('A counterfactual branch cannot close the owned native application.')


class DetachedColumn:
    """Serializable Python plant data; attach a verified native snapshot to run."""
    def step(self, *args, **kwargs):
        raise RestorationError('This portable column requires native Aspen snapshot reattachment.')


@contextmanager
def column_copying():
    original_close = DistillationColumnAspen.close
    def close(plant, *args, **kwargs):
        if getattr(plant, '_e1_document_open', True) is False:
            plant.ad.Quit()
            return
        return original_close(plant, *args, **kwargs)
    with patch.object(DistillationColumnAspen, '__deepcopy__', copy_column, create=True), \
         patch.object(DistillationColumnAspen, 'close', close):
        yield


@contextmanager
def copying_into(target):
    global _COPY_TARGET
    previous, _COPY_TARGET = _COPY_TARGET, target
    try:
        yield
    finally:
        _COPY_TARGET = previous


def digest_state(value):
    """Stable recursive evidence hash, including tensors, replay and optimizers.

    Used for bounded integration validation, not repeatedly on full-size replay.
    """
    h = hashlib.sha256()
    seen = {}

    def visit(item):
        if item is None or isinstance(item, (bool, int, float, str, bytes)):
            h.update((type(item).__name__ + ':' + repr(item)).encode())
            return
        if isinstance(item, np.generic):
            visit(item.item())
            return
        if id(item) in seen:
            h.update(f'ref:{seen[id(item)][0]}'.encode())
            return
        # Retain temporary state_dict/NumPy views until hashing finishes: their
        # IDs can otherwise be reused and silently skip later parameters.
        seen[id(item)] = (len(seen), item)
        if isinstance(item, torch.Tensor):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, item.shape)).encode())
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, (torch.nn.Module, torch.optim.Optimizer)):
            visit(item.state_dict())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, DistillationColumnAspen):
            visit({key:value for key,value in python_plant_state(item).items()
                   if key not in {'path','_e3_closed','_e1_project_root','_e1_document_open'}})
        elif callable(item):
            h.update(getattr(item, '__qualname__', type(item).__name__).encode())
        elif hasattr(item, '__dict__'):
            visit(vars(item))
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def learner_summary(environment, compiled):
    return {name: {'steps': int(getattr(agent, 'steps', 0)), 'train_steps': int(getattr(agent, 'train_steps', 0)),
                   'replay_size': int(agent.buffer.size), 'replay_ptr': int(getattr(agent.buffer, 'ptr', 0)),
                   'tensor_versions': [tensor._version for value in vars(agent).values()
                        if isinstance(value, torch.nn.Module) for tensor in value.state_dict(keep_vars=True).values()]}
            for identity, agent in agents_in(environment, compiled.local_names).items()
            for name in [next((key for key in sorted(compiled.local_names) if environment.get(key) is agent), str(identity))]}


def integrity_payload(environment, compiled):
    """Full bounded-test evidence without native objects or helper globals."""
    return {name: value
            for name, value in environment.items() if name in compiled.local_names
            and not callable(value)}


class ColumnStudySession(StudySession):
    def __init__(self, *args, branch_plant=None, verify_full_state=False, repeat_branches=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify_full_state = verify_full_state
        self.repeat_branches = repeat_branches
        self.native = None
        self.native_plant = None
        self.integration_checks = []
        self.branch_plant = branch_plant

    def before(self, site, environment):
        self.native_plant = environment['system']
        step = int(environment[self.compiled.step_name])
        selected = (site.key, step) in self.snapshot_steps
        folder = self.output / ('native_snapshots' if selected else 'temporary_native')
        if self.branch_plant is None or self.branch_plant is self.native_plant:
            raise RestorationError('E1 requires a separate private native branch application.')
        self.native = E1NativeSnapshot(self.native_plant, {}, folder, target=self.branch_plant)
        with copying_into(self.branch_plant):
            super().before(site, environment)

    def after(self, site, environment):
        import review_experiments.e1.study as study
        import review_experiments.e1.snapshots as portable
        original_branch = study.run_branch
        original_export = portable.portable_environment
        # Detach the recording method before hashing the real learners.
        self.detach()
        summary = learner_summary(environment, self.compiled)
        complete_hash = digest_state(integrity_payload(environment, self.compiled)) if self.verify_full_state else None
        initial_rng = rng_state()
        native = self.native
        branch_checks = []
        from column_review.pilot import profile
        live_before = profile(self.native_plant)[1]

        def branch(*args, **kwargs):
            native.restore()
            result = original_branch(*args, **kwargs)
            # Save raw evidence before assertions or repeated-continuation checks.
            if self.repeat_branches:
                label = args[4]
                folder = self.output / 'integration_traces'
                folder.mkdir(exist_ok=True)
                stem = f'{site.channel}_{int(environment[self.compiled.step_name]):07d}_{label}'
                np.savez_compressed(folder / (stem + '_first.npz'), **result['trace'])
                branch_rng = rng_state()
                restore_rng(initial_rng)
                native.restore()
                again = original_branch(*args, **kwargs)
                np.savez_compressed(folder / (stem + '_repeat.npz'), **again['trace'])
                if result['state'] != 'completed' or again['state'] != 'completed':
                    raise RestorationError(f'Repeated {site.channel}/{label} rollout failed: {result}, {again}')
                for field, atol in (('y', [1e-8, 1e-4]), ('u', 1e-3), ('reward', 1e-5), ('reference', 1e-10)):
                    if not np.allclose(result['trace'][field], again['trace'][field], rtol=1e-7, atol=atol):
                        raise RestorationError(f'Repeated native/controller rollout differs: {site.channel}/{label}/{field}')
                restore_rng(branch_rng)
                branch_checks.append({'source': label, 'repeated': True, 'steps': result['completed_steps']})
            return result

        def export(snapshot, compiled):
            exported = original_export(snapshot, compiled)
            plant = exported['system']
            state = python_plant_state(plant)
            plant.__class__ = DetachedColumn
            plant.__dict__ = deepcopy(state)
            plant.native_snapshot = str(native.snapshot.path)
            plant.native_metadata = native_metadata(native)
            return exported

        success = False
        try:
            with patch.object(study, 'run_branch', branch), patch.object(portable, 'portable_environment', export):
                super().after(site, environment)
            # Preserve the original E1 protocol: failed counterfactuals are
            # explicit diagnostic records. Restoration errors still propagate.
            success = self.all_records[-1]['status']=='completed'
        finally:
            live_after = profile(self.native_plant)[1]
            restore_rng(initial_rng)
            if not np.array_equal(live_before, live_after):
                raise RestorationError('A diagnostic branch changed the live native plant.')
            if learner_summary(environment, self.compiled) != summary:
                raise RestorationError('A diagnostic branch mutated a live learner counter, replay pointer or network tensor.')
            after_hash = digest_state(integrity_payload(environment, self.compiled)) if self.verify_full_state else None
            if complete_hash is not None and complete_hash != after_hash:
                raise RestorationError('A diagnostic branch mutated the live Python controller/learner state.')
            row = {'channel': site.channel, 'step': int(environment[self.compiled.step_name]),
                   'live_learner_unchanged': True, 'full_state_before': complete_hash, 'full_state_after': after_hash,
                   'zero_time_initializations': native.priming_records, 'repeat_checks': branch_checks,
                   'live_native_before':live_before.tolist(), 'live_native_after':live_after.tolist(),
                   'live_native_unchanged':True,
                   'native_file': str(native.snapshot.path), 'native_sha256': hashlib.sha256(native.snapshot.path.read_bytes()).hexdigest()}
            self.integration_checks.append(row)
            with (self.output / 'native_restore_checks.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(json_safe(row), allow_nan=False) + '\n')
            # Delete only this adapter's in-memory snapshot. Its selected binary
            # exports persist alongside the portable Python snapshot.
            native.release()
            self.native = self.native_plant = None
            selected = (site.key, row['step']) in self.snapshot_steps
            if success and not selected:
                native.snapshot.path.unlink()
                Path(native.metadata['project_file']).unlink()
                native.snapshot.exported_result_path.unlink()


def import_native(plant, metadata):
    """Attach a trusted exported snapshot to a fresh, initialized owned case."""
    from column_review.common import digest
    from column_review.pilot import NativeSnapshot
    if digest(metadata['file']) != metadata['file_sha256']:
        raise RestorationError('Native checkpoint checksum differs.')
    snap = load_project(plant, metadata)
    results = plant.sim.Results
    base = object.__new__(NativeSnapshot)
    base.plant, base.native = plant, snap
    base.python_state = metadata['python_state']
    base.internal = np.asarray(metadata['internal'])
    ready = object.__new__(E1NativeSnapshot)
    base.path = Path(metadata['file'])
    ready.snapshot, ready.plant, ready.priming_records = base, plant, []
    ready.restored_snapshot = base
    ready.restore()
    # Native measurements were independently checked above. Preserve the exact
    # cached Python sample that the original observer/controller would consume.
    plant.current_output = np.asarray(metadata['python_state']['current_output']).copy()
    results.Delete(snap)
    return ready.priming_records
