"""Check that the new entrypoint can select only residual's case 17."""
import unittest
import recover_residual
r = recover_residual.engine

class Scope(unittest.TestCase):
    def test_case17_residual_only(self):
        plan=dict(category='residual',case=17,seed=7,expected_matched_pairs=80,
                  decisions_per_window=20,main_training_cycles=300,authorization='confirmed_failed_run_recovery')
        r.scope(plan)
        for changed in (dict(category='weights',case=14),dict(category='markov',case=15)):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                r.scope(dict(plan,**changed))

if __name__=='__main__':
    unittest.main(verbosity=2)
