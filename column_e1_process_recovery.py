"""Fresh owned Aspen documents for recovered E1 workers only.

The scientific state, fidelity bounds, sampling schedule and original sources
remain sealed. A saved project is opened in a new native process, rather than
reusing the process whose document was closed. Every retirement is restricted
to an identity in this worker's ownership journal.
"""
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import shutil
import time
import uuid
from unittest.mock import patch

_PROCESS_IDENTITIES = {}


def append(output, value):
    with (Path(output) / 'native_process_recovery.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')


def owned_identity(plant, output):
    from column_review.common import read, process_identity
    previous = _PROCESS_IDENTITIES.get(id(plant))
    try:
        pid = int(plant.ad.processId)
    except Exception:
        if previous is None:
            raise RuntimeError('Cannot identify the owned native process')
        pid = previous['pid']
    owners = read(Path(output) / 'owned_aspen_apps.json', [])
    identity = next((r for r in owners if r['pid'] == pid), None)
    if identity is None or (previous and previous['pid'] == pid and previous['created'] != identity['created']):
        raise RuntimeError('Refusing to retire an unowned native application')
    current = process_identity(pid)
    if current['alive'] is True and current['created'] != identity['created']:
        raise RuntimeError('Native PID has been reused')
    return {key: identity[key] for key in ('pid', 'alive', 'created')}


def retire(plant, output):
    from column_review.common import process_identity
    identity = owned_identity(plant, output)
    errors = []
    if process_identity(identity['pid'])['alive'] is True:
        for method, args in (('CloseDocument', (False,)), ('Quit', ())):
            try:
                getattr(plant.ad, method)(*args)
            except Exception as error:
                errors.append(str(error))
        deadline = time.monotonic() + 8
        while process_identity(identity['pid'])['alive'] is True and time.monotonic() < deadline:
            time.sleep(.1)
        if process_identity(identity['pid'])['alive'] is True:
            # The snapshot was verified and copied before retiring this process.
            # This is exclusively the current recovery worker's disposable app.
            from recover_markov_seed23 import terminate_owned_identity
            terminate_owned_identity(identity)
    if process_identity(identity['pid'])['alive'] is not False:
        raise RuntimeError('Owned native application did not exit')
    append(output, dict(event='retired_owned_application', identity=identity, cleanup_errors=errors))
    return identity


def load_project(plant, metadata, output):
    import numpy as np
    import win32com.client
    from column_review.common import digest, process_identity
    from column_e1.runtime import COM_FIELDS
    from systems.distillation.config import resolve_snapshot_dir
    from review_experiments.e1.runtime import RestorationError
    for key in ('file', 'project_file'):
        if digest(metadata[key]) != metadata[key + '_sha256']:
            raise RestorationError('Native project checkpoint checksum differs')
    previous = Path(plant.path)
    if previous.name != Path(metadata['project_file']).name:
        raise RestorationError('Native project model filename differs')
    root = Path(getattr(plant, '_e1_project_root', previous.parent))
    plant._e1_project_root = str(root)
    directory = root / ('load_' + uuid.uuid4().hex[:8])
    directory.mkdir()
    model = directory / previous.name
    shutil.copyfile(metadata['project_file'], model)
    database = Path(resolve_snapshot_dir(model)) / 'snapshot.snp'
    database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metadata['file'], database)
    if digest(model) != metadata['project_file_sha256'] or digest(database) != metadata['file_sha256']:
        raise RestorationError('Private native checkpoint copy differs')
    old = retire(plant, output)
    for field in COM_FIELDS:
        vars(plant).pop(field, None)
    gc.collect()
    plant._e1_document_open = False
    plant.ad = win32com.client.DispatchEx('AD application')
    identity = process_identity(int(plant.ad.processId))
    _PROCESS_IDENTITIES[id(plant)] = identity
    if identity['alive'] is not True:
        raise RuntimeError('Fresh native application did not verify')
    plant.ad.Visible = True
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
    # Aspen locks snapshot.snp after opening it. Both private copies were
    # hashed above, before the native application acquired that lock.
    append(output, dict(event='opened_fresh_application', retired=old,
                        current=identity, native_model_sha256=metadata['project_file_sha256'],
                        native_snapshot_sha256=metadata['file_sha256'],
                        private_copy_hashes_verified_before_open=True, model=str(model)))
    return native


@contextmanager
def installed(output):
    import column_e1.runtime as runtime
    from systems.distillation.plant import DistillationColumnAspen
    from column_review.common import process_identity
    copying = runtime.column_copying

    @contextmanager
    def guarded_copying():
        with copying():
            original = DistillationColumnAspen.close
            def close(plant, *args, **kwargs):
                identity = owned_identity(plant, output)
                try:
                    return original(plant, *args, **kwargs)
                except Exception:
                    if process_identity(identity['pid'])['alive'] is not False:
                        raise
                    append(output, dict(event='already_exited_during_cleanup', identity=identity))
            with patch.object(DistillationColumnAspen, 'close', close):
                yield

    with patch.object(runtime, 'load_project', lambda plant, metadata: load_project(plant, metadata, output)), \
         patch.object(runtime, 'column_copying', guarded_copying):
        yield


def manager_factory(original, recovery_sha, receipts):
    def factory(protocol_sha, job, *, validation_steps=0):
        base = original(protocol_sha, job, validation_steps=validation_steps)
        class ProcessManager(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sources = dict(self.sources, column_e1_process_protocol=recovery_sha)

            def restore(self, environment, session):
                import column_e1.checkpoints as native
                from column_review.common import read
                from column_e1.runtime import digest_state, integrity_payload
                index = read(self.index)
                current = dict(self.sources)
                legacy = {k: v for k, v in current.items() if k != 'column_e1_process_protocol'}
                if index['source_hashes'] == legacy:
                    self.sources = legacy
                elif index['source_hashes'] != current:
                    raise RuntimeError('Recovery checkpoint protocol differs')
                original_import = native.import_native
                def checked_import(plant, metadata):
                    before = digest_state(integrity_payload(environment, self.compiled))
                    result = original_import(plant, metadata)
                    after = digest_state(integrity_payload(environment, self.compiled))
                    if before != after:
                        raise RuntimeError('Native restore mutated full learner/controller state')
                    append(receipts, dict(event='full_python_restore_verified', before=before, after=after,
                                          completed_steps=index['completed_steps']))
                    return result
                try:
                    with patch.object(native, 'import_native', checked_import):
                        super().restore(environment, session)
                finally:
                    self.sources = current
                session.source_hashes = current
        return ProcessManager
    return factory
