"""The recovery boundary permits only a retired failed horizon job."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

import run_column_e1_horizon_recovery as recovery


class HorizonBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.path = Path('horizon.json')
        self.plan = {'jobs': [dict(category='horizon', job=str(self.path), state='failed'),
                              dict(category='combined', job='combined.json', state='running')]}

    def test_only_horizon_row_changes_in_a_private_copy(self):
        original = deepcopy(self.plan)
        actual = recovery.horizon_only(self.plan, self.path)
        self.assertEqual(self.plan, original)
        self.assertEqual(actual['jobs'][0]['state'], 'ready')
        self.assertEqual(actual['jobs'][1], original['jobs'][1])

    def test_running_horizon_is_rejected(self):
        self.plan['jobs'][0]['state'] = 'running'
        with self.assertRaisesRegex(RuntimeError, 'failed horizon'):
            recovery.horizon_only(self.plan, self.path)

    def test_another_configuration_cannot_be_selected(self):
        self.plan['jobs'][1]['state'] = 'failed'
        with self.assertRaisesRegex(RuntimeError, 'failed horizon'):
            recovery.horizon_only(self.plan, Path('combined.json'))

    def test_duplicate_operational_rows_are_rejected(self):
        self.plan['jobs'].append(dict(self.plan['jobs'][0]))
        with self.assertRaisesRegex(RuntimeError, 'failed horizon'):
            recovery.horizon_only(self.plan, self.path)

    def test_unchanged_peer_identity_does_not_open_its_status(self):
        from column_review import common
        identity = dict(pid=123, alive=True, created=10)
        plan = dict(protected=[dict(role='scientific_worker', identity=identity)])
        with patch.object(common, 'process_identity', return_value=dict(identity)), \
             patch.object(recovery, 'read', side_effect=AssertionError('Live peer status must not be opened')):
            recovery.protected_check(plan)

    def test_reused_peer_pid_blocks_launch_without_signalling_it(self):
        from column_review import common
        old = dict(pid=123, alive=True, created=10)
        plan = dict(protected=[dict(role='scientific_worker', identity=old)])
        with patch.object(common, 'process_identity', return_value=dict(old, created=11)):
            with self.assertRaisesRegex(RuntimeError, 'protected process'):
                recovery.protected_check(plan)


if __name__ == '__main__':
    unittest.main()
