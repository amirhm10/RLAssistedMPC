"""Cheap regression/ownership/preservation guards; no simulator or numerics."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ownership_fix as fix
import recover_weights as recovery


class OwnershipGeneration(unittest.TestCase):
    def setUp(self):
        self.old = dict(pid=7, alive=True, created=100)
        self.new = dict(pid=7, alive=True, created=200)

    def test_exact_reproduction_old_code_refuses_correct_recycled_generation(self):
        import column_e1_process_recovery as fresh
        import column_review.common as common
        plant = SimpleNamespace(ad=SimpleNamespace(processId=7))
        saved = dict(fresh._PROCESS_IDENTITIES)
        try:
            fresh._PROCESS_IDENTITIES[id(plant)] = self.new
            with patch.object(common, 'read', return_value=[self.old, self.new]), \
                 patch.object(common, 'process_identity', return_value=self.new):
                with self.assertRaisesRegex(RuntimeError, 'unowned'):
                    fresh.owned_identity(plant, HERE)
            with fix.installed(fresh, lambda path, default: [self.old, self.new], lambda pid: self.new):
                self.assertEqual(fresh.owned_identity(plant, HERE), self.new)
        finally:
            fresh._PROCESS_IDENTITIES.clear()
            fresh._PROCESS_IDENTITIES.update(saved)

    def test_journal_order_does_not_change_matching_generation(self):
        for records in ([self.old, self.new], [self.new, self.old]):
            self.assertEqual(fix.select_identity(7, self.new, records, self.new), self.new)

    def test_no_remembered_identity_still_requires_current_generation(self):
        self.assertEqual(fix.select_identity(7, None, [self.old, self.new], self.new), self.new)
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            fix.select_identity(7, None, [self.old], self.new)

    def test_old_proxy_cannot_retire_reused_live_pid(self):
        with self.assertRaisesRegex(RuntimeError, 'reused'):
            fix.select_identity(7, self.old, [self.old, self.new], self.new)

    def test_unowned_and_inaccessible_processes_are_refused(self):
        with self.assertRaises(RuntimeError):
            fix.select_identity(7, self.new, [], self.new)
        with self.assertRaisesRegex(RuntimeError, 'inaccessible'):
            fix.select_identity(7, self.new, [self.new], dict(pid=7, alive=None, created=None))

    def test_dead_ambiguous_generation_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            fix.select_identity(7, None, [self.old, self.new], dict(pid=7, alive=False, created=None))

    def test_remembered_dead_generation_can_complete_cleanup(self):
        actual = dict(pid=7, alive=False, created=None)
        self.assertEqual(fix.select_identity(7, self.new, [self.old, self.new], actual), self.new)


class RecoveryGuards(unittest.TestCase):
    def plan(self):
        return dict(category='weights', case=14, seed=7, expected_matched_pairs=80,
                    decisions_per_window=20, main_training_cycles=300,
                    authorization='confirmed_failed_run_recovery', failed_worker=dict(pid=123))

    def guard(self, **kwargs):
        args = dict(resources=lambda: dict(headroom_bytes=6*1024**3),
                    identity=lambda pid: dict(pid=pid, alive=False, created=None),
                    applications=[], status=dict(pid=123, state='failed'))
        args.update(kwargs)
        return recovery.start_guard(self.plan(), **args)

    def test_live_or_inaccessible_old_worker_refused(self):
        for alive in (True, None):
            with self.subTest(alive=alive), self.assertRaisesRegex(RuntimeError, 'Former'):
                self.guard(identity=lambda pid: dict(pid=pid, alive=alive, created=1))

    def test_other_live_worker_blocks_duplicate_even_before_native_creation(self):
        with self.assertRaisesRegex(RuntimeError, 'Another weights worker'):
            self.guard(status=dict(pid=456, state='running'),
                       identity=lambda pid: dict(pid=pid, alive=pid==456, created=1))

    def test_owned_native_application_blocks_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'still owns'):
            self.guard(applications=[dict(pid=777)])

    def test_memory_threshold_and_exact_boundary(self):
        with self.assertRaisesRegex(RuntimeError, '6 GiB'):
            self.guard(resources=lambda: dict(headroom_bytes=6*1024**3-1))
        self.guard()

    def test_wrong_case_budget_or_authorization_is_refused(self):
        for key, value in (('case', 16), ('expected_matched_pairs', 320),
                           ('decisions_per_window', 100), ('authorization', 'healthy_restart')):
            plan = dict(self.plan(), **{key:value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                recovery.scope(plan)

    def test_invalid_continuation_stops_before_verification(self):
        with patch.object(recovery, 'verify', side_effect=AssertionError('Verification called')):
            with self.assertRaises(ValueError):
                recovery.worker(16)

    def test_markov_wrapper_scope_is_separate_from_weights(self):
        with patch.object(recovery, 'CATEGORY', 'markov'), patch.object(recovery, 'CASE', 15):
            recovery.scope(dict(self.plan(), category='markov', case=15))
            with self.assertRaises(RuntimeError):
                recovery.scope(self.plan())

    def test_guard_refusal_precedes_numerical_import(self):
        before = set(sys.modules)
        with patch.object(recovery, 'verify', return_value=self.plan()), \
             patch.object(recovery, 'start_guard', side_effect=RuntimeError('blocked')):
            with self.assertRaisesRegex(RuntimeError, 'blocked'):
                recovery.worker(8)
        self.assertFalse({'numpy', 'torch', 'review_numeric_recovery'} & (set(sys.modules)-before))


class EvidencePreservation(unittest.TestCase):
    def test_committed_prefix_must_remain_at_original_path(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder); original=root/'trace.npz'; backup=root/'saved.npz'
            original.write_bytes(b'trace'); backup.write_bytes(b'trace')
            record=dict(source=str(original),backup=str(backup),sha256=recovery.digest(backup),committed=True)
            recovery.preserved_traces(dict(traces=[record]))
            original.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'prefix'):
                recovery.preserved_traces(dict(traces=[record]))

    def test_post_checkpoint_trace_must_exist_in_verified_replay_archive(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder); backup=root/'saved.npz'; backup.write_bytes(b'trace')
            original=root/'capture/traces/trace.npz'
            record=dict(source=str(original),backup=str(backup),sha256=recovery.digest(backup),committed=False)
            with patch.object(recovery, 'OUTPUT', root):
                with self.assertRaisesRegex(RuntimeError, 'replay archive'):
                    recovery.preserved_traces(dict(traces=[record]),after_validation=True)
                archived=root/'capture/replayed_suffixes/1/traces/trace.npz'
                archived.parent.mkdir(parents=True); archived.write_bytes(b'trace')
                self.assertEqual(recovery.preserved_traces(dict(traces=[record]),after_validation=True),[str(archived)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
