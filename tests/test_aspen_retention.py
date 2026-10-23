"""Storage/lifecycle invariants; only disposable files, no NumPy or Aspen."""
import io
import json
import os
from pathlib import Path
import pickle
import stat
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import aspen_retention as retention
from aspen_retention_pickle import metadata, InspectionError


def forbidden_side_effect(path):
    Path(path).write_text('REDUCER EXECUTED')


class DangerousReducer:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return forbidden_side_effect, (self.marker,)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(os.environ.get('ASPEN_RETENTION_TEST_SCRATCH',
                            str(Path(__file__).resolve().parents[1] / '.rt')))
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='rt-', dir=self.scratch)
        self.base = Path(self.temporary.name).resolve()
        assert self.base.parent == self.scratch.resolve()
        assert self.base.name.startswith('rt-')
        self.results = self.base / 'Distillation/Results'
        self.job = self.results / 'e1/weights/a1'
        self.ck = self.job / 'training_checkpoints'
        self.ck.mkdir(parents=True)
        self.clock = time.time() + 1000
        self.config = dict(results_root=str(self.results), state_root=str(self.base / 'archive'),
                           jobs=[dict(label='weights', directory=str(self.job))],
                           minimum_age_seconds=0, stable_seconds=1,
                           checkpoint_quiet_seconds=0,
                           maximum_folders_per_pass=100, io_mib_per_second=1_000_000)
        self.checkpoint(0, 100, commit=False)
        self.checkpoint(1, 200)
        self.result = self.job / 'capture/raw_result.npz'
        self.result.parent.mkdir()
        self.result.write_bytes(b'IRREPLACEABLE SCIENTIFIC RESULT')

    def tearDown(self):
        assert self.base.parent == self.scratch.resolve()
        assert self.base.name.startswith('rt-')
        self.temporary.cleanup()
        try:
            self.scratch.rmdir()
        except OSError:
            pass

    def checkpoint(self, slot, step, *, commit=True, source='a' * 64, extra=None):
        folder = self.ck / f'native_{slot}_{step}'
        sub = folder / ('review_e1_' + str(step))
        sub.mkdir(parents=True, exist_ok=True)
        project = sub / 'C2S_SS_simulation14.dynf'
        snapshot = sub / 'snapshot.snp'
        project.write_bytes((f'FULL MODEL AT {step}\n'.encode()) * 250)
        snapshot.write_bytes((f'INITIALIZED NATIVE STATE {step}\n'.encode()) * 100)
        native = dict(file=str(snapshot), file_sha256=retention.digest(snapshot),
                      project_file=str(project), project_file_sha256=retention.digest(project),
                      name='review_e1_' + str(step), live_save_unchanged=True)
        retention.atomic_json(sub / 'metadata.json', native)
        (folder / ('review_e1_' + str(step) + '.snp')).write_bytes(b'EXPORTED RESULT')
        payload = dict(schema=2, completed_steps=step, source_hashes={'runner.py': source},
                       runner_sha256='b' * 64, environment=extra or {'big_array': bytearray(100_000)},
                       native=native)
        temporary = self.ck / f'training_state_{slot}.pkl.tmp'
        temporary.write_bytes(pickle.dumps(payload, protocol=5))
        python_path = self.ck / f'training_state_{slot}.pkl'
        os.replace(temporary, python_path)
        index = dict(schema=2, slot=slot, file=python_path.name, sha256=retention.digest(python_path),
                     completed_steps=step, source_hashes=payload['source_hashes'], native=native)
        if commit:
            retention.atomic_json(self.ck / 'latest.json', index)
        return folder, index

    def old_native(self, step=50):
        folder = self.ck / f'native_0_{step}'
        folder.mkdir()
        (folder / 'old.dynf').write_bytes(b'OLD MODEL, ALL BYTES MUST SURVIVE' * 100)
        (folder / 'snapshot.snp').write_bytes(b'OLD SNAPSHOT')
        # This fixture represents a historical save made before the current
        # index, rather than an in-flight native write after that index.
        stamp = (self.ck / 'latest.json').stat().st_mtime_ns - 10_000_000
        os.utime(folder, ns=(stamp, stamp))
        return folder

    def loads(self):
        parent = self.job / 'native_1234abcd/branch'
        parent.mkdir(parents=True)
        self.config['jobs'][0]['private_sessions'] = [str(parent.parent)]
        folders = []
        for i in range(4):
            path = parent / f'load_{i:08x}'
            path.mkdir()
            (path / 'model.dynf').write_bytes(b'PRIVATE MODEL' * 100)
            (path / 'snapshot.bak').write_bytes(b'PRIVATE DATABASE')
            folders.append(path)
            time.sleep(.003)
        return folders

    def worker(self):
        return retention.Retainer(self.config, clock=lambda: self.clock)

    def observe_then_retire(self, worker, folder, kind='native'):
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(folder, kind, self.job, pair)
        self.clock += 2
        return worker.retire(folder, kind, self.job, pair)

    def test_never_executes_reducers(self):
        marker = self.base / 'must_not_exist'
        self.checkpoint(1, 200, extra={'danger': DangerousReducer(marker)})
        pair = self.worker().verifier.pair(self.ck, self.results)
        self.assertEqual([r['completed_steps'] for r in pair['checkpoints']], [200, 100])
        self.assertFalse(marker.exists())

    def test_verification_never_opens_live_slot_or_index_by_original_name(self):
        originals = {self.ck / name for name in ('latest.json', 'training_state_0.pkl', 'training_state_1.pkl')}
        actual = retention.open_read
        actual_stat = Path.stat
        def guarded(path, **kwargs):
            self.assertNotIn(Path(path), originals)
            return actual(path, **kwargs)
        def guarded_stat(path, *args, **kwargs):
            if os.name == 'nt':
                self.assertNotIn(Path(path), originals)
            return actual_stat(path, *args, **kwargs)
        with patch.object(retention, 'open_read', guarded), patch.object(Path, 'stat', guarded_stat):
            self.assertEqual(len(self.worker().verifier.pair(self.ck, self.results)['checkpoints']), 2)

    def test_large_arrays_are_seeked_not_read(self):
        class Tracked(io.BytesIO):
            maximum = 0
            def read(self, n=-1):
                self.maximum = max(self.maximum, n)
                return super().read(n)
        stream = Tracked(pickle.dumps(dict(schema=2, completed_steps=100,
                         environment={'huge': bytearray(5_000_000)}, native={'name': 'native'}), protocol=5))
        result = metadata(stream)
        self.assertEqual(result['completed_steps'], 100)
        self.assertLess(stream.maximum, 1000)

    def test_new_private_session_is_discovered_without_scanning_worker_outputs(self):
        folders = self.loads()
        self.config['jobs'][0]['private_sessions'] = []
        self.checkpoint(1, 200, extra={'plant': {'_e1_project_root': str(folders[0].parent)}})
        actual = Path.iterdir
        def guarded(path):
            self.assertNotEqual(path, self.job)
            return actual(path)
        with patch.object(Path, 'iterdir', guarded):
            result = self.worker().scan(execute=True)
        self.assertEqual(result['jobs'][0]['old_private'], 2)
        self.assertTrue(all(path.exists() for path in folders))

    def test_truncation_blocks_all_source_removal(self):
        old = self.old_native()
        path = self.ck / 'training_state_0.pkl'
        path.write_bytes(path.read_bytes()[:-1])
        state = self.worker().scan(execute=True)
        self.assertIn('deferred', state['jobs'][0]['state'])
        self.assertTrue(old.is_dir())
        self.assertTrue(self.result.exists())

    def test_oversized_opcode_is_rejected_before_allocation(self):
        # PROTO 5, LONG4 with a malicious length, not an actual allocation.
        with self.assertRaises((InspectionError, ValueError)):
            metadata(io.BytesIO(b'\x80\x05\x8b\xff\xff\xff\x7f'))

    def test_corrupt_native_pair_blocks_cleanup(self):
        old = self.old_native()
        index = retention.read_json(self.ck / 'latest.json')
        Path(index['native']['file']).write_bytes(b'CORRUPT SNAPSHOT')
        self.assertIn('deferred', self.worker().scan(execute=True)['jobs'][0]['state'])
        self.assertTrue(old.exists())

    def test_source_provenance_mismatch_blocks_cleanup(self):
        old = self.old_native()
        self.checkpoint(0, 100, commit=False, source='c' * 64)
        self.assertIn('deferred', self.worker().scan(execute=True)['jobs'][0]['state'])
        self.assertTrue(old.exists())

    def test_uncommitted_new_slot_is_never_pruned(self):
        old = self.old_native()
        future, _ = self.checkpoint(0, 300, commit=False)
        self.assertIn('deferred', self.worker().scan(execute=True)['jobs'][0]['state'])
        self.assertTrue(future.exists())
        self.assertTrue(old.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows atomic replacement sharing contract')
    def test_slot_reader_allows_atomic_writer_replacement(self):
        path = self.ck / 'training_state_0.pkl'
        original = path.read_bytes()
        with retention.private_snapshot(path, self.base / 'read_aliases') as stream:
            temporary = self.ck / 'replacement.tmp'
            temporary.write_bytes(b'NEW GENERATION')
            os.replace(temporary, path)
            self.assertEqual(stream.read(), original)
        self.assertEqual(path.read_bytes(), b'NEW GENERATION')

    @unittest.skipUnless(os.name == 'nt', 'Windows concurrent manifest replacement contract')
    def test_queue_manifest_reader_does_not_block_repeated_commits(self):
        queue = self.results / 'combined_queue'
        queue.mkdir()
        source = queue / 'live_manifest.json'
        retention.atomic_json(source, {'generation': 0})
        errors = []
        def writer():
            try:
                for generation in range(1, 101):
                    retention.atomic_json(source, {'generation': generation})
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=writer)
        thread.start()
        worker = self.worker()
        while thread.is_alive():
            # Maintenance must not open the queue's mutable manifest at all.
            worker.scan(execute=True)
        thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(retention.read_json(source)['generation'], 100)

    @unittest.skipUnless(os.name == 'nt', 'Windows native-checkpoint commit contract')
    def test_native_writer_can_commit_while_retention_observes(self):
        # Use the production quiescence interval: the observer must defer
        # throughout a burst of native saves, then resume when it is quiet.
        self.config['checkpoint_quiet_seconds'] = 2
        errors = []
        def writer():
            try:
                for i in range(1, 41):
                    step = 200 + i * 100
                    self.checkpoint(0 if i % 2 else 1, step)
            except Exception as error:
                import traceback
                errors.append(traceback.format_exc())
        worker = self.worker()
        thread = threading.Thread(target=writer)
        thread.start()
        while thread.is_alive():
            try:
                worker.verifier.pair(self.ck, self.results)
            except (retention.Deferred, OSError):
                pass  # The observer, rather than the writer, must defer.
        thread.join()
        self.assertEqual(errors, [])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                pair = worker.verifier.pair(self.ck, self.results)
                break
            except retention.Deferred:
                time.sleep(.06)
        else:
            self.fail('Observer did not resume after the writer became quiet')
        self.assertEqual(pair['checkpoints'][0]['completed_steps'], 4200)

    def test_root_escape_and_parent_traversal_are_rejected(self):
        for path in (self.results, self.base, self.results / '..'):
            with self.subTest(path=path), self.assertRaises(retention.Deferred):
                retention.safe_child(path, self.results)

    def test_reparse_point_guard(self):
        actual = retention.attributes
        def reparse(path):
            result = actual(path)
            result['flags'] |= 0x400
            return result
        with patch.object(retention, 'attributes', reparse):
            with self.assertRaises(retention.Deferred):
                retention.safe_child(self.ck, self.results)

    def test_latest_two_and_future_native_folders_are_protected(self):
        old = self.old_native()
        pair = self.worker().verifier.pair(self.ck, self.results)
        future = self.ck / 'native_1_300'
        future.mkdir()
        self.assertEqual(retention.native_candidates(pair), [old])
        self.assertFalse(retention.still_current(pair))

    def test_two_newest_working_copies_are_retained(self):
        folders = self.loads()
        self.assertEqual(set(retention.load_candidates(self.job)), set(folders[:2]))

    def test_closed_private_copy_cleanup_keeps_two_newest_and_all_results(self):
        folders = self.loads()
        record = self.observe_then_retire(self.worker(), folders[0], 'private')
        self.assertEqual(record['kind'], 'private')
        self.assertFalse(folders[0].exists())
        self.assertTrue(all(folder.exists() for folder in folders[1:]))
        self.assertTrue((self.ck / 'native_0_100').exists())
        self.assertTrue((self.ck / 'native_1_200').exists())
        self.assertTrue(self.result.exists())

    def test_active_native_markers_exclude_old_working_copy(self):
        folders = self.loads()
        (folders[0] / 'AspenAMLock.aml').write_bytes(b'LOCK')
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(folders[0], 'private', self.job, pair)
        self.assertTrue(folders[0].exists())

    def test_unexpected_result_in_candidate_is_preserved(self):
        old = self.old_native()
        (old / 'raw_trace.npz').write_bytes(b'KEEP RESULT')
        stamp = (self.ck / 'latest.json').stat().st_mtime_ns - 10_000_000
        os.utime(old, ns=(stamp, stamp))
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(old, 'native', self.job, pair)
        self.assertTrue((old / 'raw_trace.npz').exists())

    def test_archive_restore_and_latest_two_remain_untouched(self):
        old = self.old_native()
        originals = {p.name: p.read_bytes() for p in old.iterdir()}
        checkpoints = {p: p.read_bytes() for p in self.ck.glob('training_state_?.pkl')}
        record = self.observe_then_retire(self.worker(), old)
        self.assertFalse(old.exists())
        self.assertEqual(record['state'], 'raw_removed')
        receipt = next((self.base / 'archive/receipts').glob('*.json'))
        stage = self.base / 'new_staged_restore'
        self.assertTrue(retention.restore(receipt, stage)['verified'])
        self.assertEqual({p.name: p.read_bytes() for p in stage.iterdir() if p.name in originals}, originals)
        self.assertEqual({p: p.read_bytes() for p in checkpoints}, checkpoints)
        self.assertEqual(self.result.read_bytes(), b'IRREPLACEABLE SCIENTIFIC RESULT')

    @unittest.skipUnless(os.name == 'nt', 'OneDrive READONLY directory attributes')
    def test_readonly_native_directories_are_removed_only_after_verified_archive(self):
        old = self.old_native()
        nested = old / 'native_project'
        nested.mkdir()
        (nested / 'extra.snp').write_bytes(b'ADDITIONAL PRESERVED SNAPSHOT')
        stamp = (self.ck / 'latest.json').stat().st_mtime_ns - 10_000_000
        os.utime(nested, ns=(stamp, stamp))
        os.utime(old, ns=(stamp, stamp))
        nested.chmod(stat.S_IREAD)
        old.chmod(stat.S_IREAD)
        protected = {p: p.stat().st_file_attributes for p in
                     (self.ck/'native_0_100', self.ck/'native_1_200')}
        record = self.observe_then_retire(self.worker(), old)
        self.assertEqual(record['state'], 'raw_removed')
        self.assertFalse(old.exists())
        self.assertEqual({p:p.stat().st_file_attributes for p in protected}, protected)
        receipt = next((self.base/'archive/receipts').glob('*.json'))
        stage = self.base/'readonly_restore'
        retention.restore(receipt, stage)
        self.assertEqual((stage/'native_project/extra.snp').read_bytes(), b'ADDITIONAL PRESERVED SNAPSHOT')

    def test_directory_enumeration_error_is_not_silently_ignored(self):
        old = self.old_native()
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        def denied(path, **kwargs):
            kwargs['onerror'](PermissionError('synthetic unreadable nested directory'))
            return iter(())
        with patch.object(retention.os, 'walk', denied):
            with self.assertRaises(PermissionError):
                worker.retire(old, 'native', self.job, pair)
        self.assertTrue(old.is_dir())
        self.assertTrue(self.result.is_file())

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing violation guard')
    def test_open_file_blocks_retirement_without_interrupting_reader(self):
        old = self.old_native()
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(old, 'native', self.job, pair)
        self.clock += 2
        with retention.open_read(old / 'old.dynf') as reader:
            with self.assertRaises(OSError):
                worker.retire(old, 'native', self.job, pair)
            self.assertIn(b'ALL BYTES MUST SURVIVE', reader.read())
        self.assertTrue(old.exists())

    def test_queue_manifest_dependency_is_preserved(self):
        old = self.old_native()
        manifest = self.results / 'queue/manifest.json'
        retention.atomic_json(manifest, {'pending': {'native': {'project_file': str(old / 'old.dynf')}}})
        pointer = self.results / 'latest.json'
        retention.atomic_json(pointer, {'directory': str(manifest.parent)})
        before = manifest.read_bytes()
        self.config['pinned_native_directories'] = [str(old)]
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(old, 'native', self.job, pair)
        self.assertTrue(old.exists())
        self.assertEqual(manifest.read_bytes(), before)

    def test_checkpoint_commit_during_compression_defers_raw_retirement(self):
        old = self.old_native()
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(old, 'native', self.job, pair)
        self.clock += 2
        original = worker._archive
        def commit_between(*args, **kwargs):
            result = original(*args, **kwargs)
            self.checkpoint(0, 300)
            return result
        with patch.object(worker, '_archive', commit_between):
            with self.assertRaises(retention.Deferred):
                worker.retire(old, 'native', self.job, pair)
        self.assertTrue(old.exists())
        self.assertEqual(retention.read_json(self.ck / 'latest.json')['completed_steps'], 300)

    def test_corrupted_zip_never_removes_raw_source(self):
        old = self.old_native()
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(old, 'native', self.job, pair)
        self.clock += 2
        original = worker._archive
        def corrupt_between(*args, **kwargs):
            path, record = original(*args, **kwargs)
            Path(record['compressed']).write_bytes(b'DAMAGED ZIP')
            return path, record
        with patch.object(worker, '_archive', corrupt_between):
            with self.assertRaises(retention.Deferred):
                worker.retire(old, 'native', self.job, pair)
        self.assertTrue(old.exists())
        self.assertTrue((old / 'old.dynf').exists())

    def test_failed_archive_keeps_source_and_discards_only_own_partial(self):
        old = self.old_native()
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(old, 'native', self.job, pair)
        self.clock += 2
        with patch.object(worker.limiter, 'consume', side_effect=OSError('synthetic full disk')):
            with self.assertRaises(OSError):
                worker.retire(old, 'native', self.job, pair)
        self.assertTrue(old.exists())
        self.assertFalse(list((self.base / 'archive/archives').iterdir()))

    def test_partial_removal_resumes_from_verified_zip_receipt(self):
        old = self.old_native()
        worker = self.worker()
        pair = worker.verifier.pair(self.ck, self.results)
        with self.assertRaises(retention.Deferred):
            worker.retire(old, 'native', self.job, pair)
        self.clock += 2
        original_unlink = Path.unlink
        calls = []
        def fail_second(path, *args, **kwargs):
            if any(p.name.startswith('.aspen_retired_') for p in path.parents):
                calls.append(path)
                if len(calls) == 2:
                    raise OSError('synthetic interruption after first unlink')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', fail_second):
            with self.assertRaises(OSError):
                worker.retire(old, 'native', self.job, pair)
        recovered = self.worker()
        recovered.recover_pending()
        self.assertFalse(list(self.ck.glob('.aspen_retired_*')))
        self.assertEqual(recovered.totals['removed_folders'], 1)
        self.assertTrue((self.ck / 'native_0_100').exists())
        self.assertTrue((self.ck / 'native_1_200').exists())

    def test_repeated_new_generations_keep_exactly_latest_two_native_sets(self):
        worker = self.worker()
        for step in (300, 400, 500):
            slot = 0 if step % 200 else 1
            self.checkpoint(slot, step)
            worker.scan(execute=True)
            self.clock += 2
            worker.scan(execute=True)
            present = sorted(int(retention.NATIVE.fullmatch(p.name)[2]) for p in self.ck.iterdir()
                             if retention.NATIVE.fullmatch(p.name))
            self.assertEqual(present, [step - 100, step])
            self.assertTrue(self.result.exists())

    def test_one_bad_job_does_not_block_retention_for_the_other_job(self):
        old = self.old_native()
        bad = self.results / 'e1/bad/a1'
        bad.mkdir(parents=True)
        self.config['jobs'].append(dict(label='bad', directory=str(bad)))
        worker = self.worker()
        worker.scan(execute=True)
        self.clock += 2
        result = worker.scan(execute=True)
        self.assertFalse(old.exists())
        self.assertIn('deferred', result['jobs'][1]['state'])
        self.assertTrue(self.result.exists())

    def test_a_previously_verified_previous_slot_cannot_silently_change(self):
        worker = self.worker()
        worker.verifier.pair(self.ck, self.results)
        path = self.ck / 'training_state_0.pkl'
        data = bytearray(path.read_bytes())
        offset = data.find(b'\x00' * 100)
        self.assertGreater(offset, 0)
        data[offset + 50] = 1
        path.write_bytes(data)
        with self.assertRaises(retention.Deferred):
            worker.verifier.pair(self.ck, self.results)

    def test_restore_refuses_existing_destination(self):
        old = self.old_native()
        self.observe_then_retire(self.worker(), old)
        receipt = next((self.base / 'archive/receipts').glob('*.json'))
        with self.assertRaises(FileExistsError):
            retention.restore(receipt, self.result.parent)
        self.assertTrue(self.result.exists())

    def test_restore_rejects_parent_escape(self):
        source = self.base / 'malicious.zip'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('../outside.txt', b'BAD')
        receipt = self.base / 'malicious_receipt.json'
        retention.atomic_json(receipt, dict(compressed=str(source), compressed_sha256=retention.digest(source),
             files=[dict(relative='../outside.txt', bytes=3, sha256='a'*64)], original_directory='historical'))
        with self.assertRaises(ValueError):
            retention.restore(receipt, self.base / 'safe_stage')
        self.assertFalse((self.base / 'outside.txt').exists())


if __name__ == '__main__':
    unittest.main()
