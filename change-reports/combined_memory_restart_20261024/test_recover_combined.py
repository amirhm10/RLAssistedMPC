"""Launch refusal checks; no numerical imports, native calls or process signals."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('recover_combined.py')
SPEC = importlib.util.spec_from_file_location('combined_recovery_review', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LaunchGuards(unittest.TestCase):
    def plan(self, authorized=True):
        return dict(state='execution_authorized_after_user_approval' if authorized else 'prepared_for_review_only',
                    current_worker=dict(pid=123, alive=True, created=456),
                    minimum_start_headroom_bytes=6 * 1024**3)

    def dead(self, pid):
        return dict(pid=pid, alive=False, created=None)

    def enough(self):
        return dict(headroom_bytes=6 * 1024**3)

    def test_review_plan_refuses_before_resource_or_process_access(self):
        with self.assertRaisesRegex(RuntimeError, 'review only'):
            MODULE.start_guard(self.plan(False), identity=lambda pid: self.fail('Identity queried'),
                               resources=lambda: self.fail('Resources queried'), applications=[])

    def test_live_original_is_never_duplicated(self):
        plan = self.plan()
        with self.assertRaisesRegex(RuntimeError, 'former worker'):
            MODULE.start_guard(plan, identity=lambda pid: plan['current_worker'],
                               resources=self.enough, applications=[])

    def test_reused_pid_blocks_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'former worker'):
            MODULE.start_guard(self.plan(), identity=lambda pid: dict(pid=pid, alive=True, created=999),
                               resources=self.enough, applications=[])

    def test_inaccessible_original_blocks_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'former worker'):
            MODULE.start_guard(self.plan(), identity=lambda pid: dict(pid=pid, alive=None, created=None),
                               resources=self.enough, applications=[])

    def test_remaining_owned_application_blocks_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'native applications remain'):
            MODULE.start_guard(self.plan(), identity=self.dead, resources=self.enough,
                               applications=[dict(pid=777, ownership_unverified=True)])

    def test_inaccessible_owned_application_is_not_assumed_closed(self):
        def identity(pid):
            return self.dead(pid) if pid == 123 else dict(pid=pid, alive=None, created=None)
        with patch.object(MODULE, 'read', return_value=[dict(pid=777, created=999)]):
            with self.assertRaisesRegex(RuntimeError, 'native applications remain'):
                MODULE.start_guard(self.plan(), identity=identity, resources=self.enough)

    def test_insufficient_commitment_headroom_blocks_launch(self):
        with self.assertRaisesRegex(RuntimeError, '6 GiB'):
            MODULE.start_guard(self.plan(), identity=self.dead, resources=lambda: dict(headroom_bytes=6*1024**3-1),
                               applications=[])

    def test_exact_threshold_accepts_only_retired_empty_case(self):
        MODULE.start_guard(self.plan(), identity=self.dead, resources=self.enough, applications=[])

    def test_adaptation_preserves_other_rows_and_original(self):
        plan = dict(jobs=[dict(job='combined.json', category='combined', aspen_number=16,
                               expected_matched_pairs=320, state='running'),
                          dict(job='weights.json', category='weights', aspen_number=14, state='running')])
        result = MODULE.selected_only(plan, Path('combined.json'))
        self.assertEqual(plan['jobs'][0]['state'], 'running')
        self.assertEqual(result['jobs'][0]['state'], 'ready')
        self.assertEqual(result['jobs'][1], plan['jobs'][1])

    def test_wrong_case_or_budget_is_rejected(self):
        for case, budget in ((13, 320), (16, 1600)):
            with self.subTest(case=case, budget=budget), self.assertRaises(RuntimeError):
                MODULE.selected_only(dict(jobs=[dict(job='combined.json', category='combined',
                                     aspen_number=case, expected_matched_pairs=budget, state='running')]),
                                     Path('combined.json'))

    def test_duplicate_scientific_row_is_rejected(self):
        row = dict(job='combined.json', category='combined', aspen_number=16,
                   expected_matched_pairs=320, state='failed')
        with self.assertRaises(RuntimeError):
            MODULE.selected_only(dict(jobs=[row, dict(row)]), Path('combined.json'))

    def test_review_worker_cannot_import_numerical_runtime(self):
        previous = set(sys.modules)
        with patch.object(MODULE, 'verify', return_value=self.plan(False)):
            with self.assertRaisesRegex(RuntimeError, 'review only'):
                MODULE.worker(8)
        self.assertFalse({'numpy', 'torch', 'review_numeric_recovery'} & (set(sys.modules) - previous))

    def test_invalid_validation_budget_is_rejected_before_verification(self):
        with patch.object(MODULE, 'verify', side_effect=AssertionError('Verification called')):
            with self.assertRaises(ValueError):
                MODULE.worker(16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
