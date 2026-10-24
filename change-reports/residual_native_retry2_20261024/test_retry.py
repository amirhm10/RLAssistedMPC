"""Document-open retries are bounded, owned and restricted to the exact call."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import document_open_retry as retry
import recover_residual_retry2 as recovery


class Fault(Exception):
    def __init__(self,code=retry.SERVER_FAULT):
        super().__init__(code,'server fault');self.hresult=code


def loader(plant,metadata,output):
    return plant.ad.openDocument('same-verified-model')


def other_phase_loader(plant,metadata,output):
    plant.ad.prepare()
    return plant.ad.openDocument('same-verified-model')


def wrong_contract(plant,metadata,output):
    return 'no native call'


class RetryContract(unittest.TestCase):
    def exercise(self,failures,*,code=retry.SERVER_FAULT,original=loader,owned=None):
        calls=[];sleeps=[];events=[];identities=[]
        def opening(model):
            calls.append(model)
            if len(calls)<=failures:raise Fault(code)
            return 'native-snapshot'
        plant=SimpleNamespace(ad=SimpleNamespace(openDocument=opening,prepare=lambda:None))
        def identity(plant,output):
            value=dict(pid=5,created=100+len(calls),alive=True)
            identities.append(value)
            return value
        result=retry.retry(original,plant,{'project_file_sha256':'model','file_sha256':'snapshot'},HERE,
            owned or identity,lambda output,event:events.append(event),sleeper=sleeps.append)
        return result,calls,sleeps,events,identities

    def test_success_is_not_retried_or_subject_to_extra_ownership_queries(self):
        result,calls,sleeps,events,identities=self.exercise(0)
        self.assertEqual(result,'native-snapshot');self.assertEqual(len(calls),1)
        self.assertEqual((sleeps,events,identities),([],[],[]))

    def test_exact_server_fault_retries_same_model_in_owned_generations(self):
        result,calls,sleeps,events,identities=self.exercise(2)
        self.assertEqual(result,'native-snapshot');self.assertEqual(len(calls),3)
        self.assertEqual(set(calls),{'same-verified-model'});self.assertEqual(sleeps,[2,4])
        self.assertEqual([v['event'] for v in events],['document_open_retry','document_open_retry','document_open_retry_recovered'])
        self.assertEqual([v['identity'] for v in events[:2]],identities)
        self.assertTrue(all(v['native_model_sha256']=='model' and v['native_snapshot_sha256']=='snapshot' for v in events))

    def test_repeated_fault_exhausts_after_three_without_a_fourth_call(self):
        calls=[];sleeps=[];events=[]
        def opening(model):calls.append(model);raise Fault()
        plant=SimpleNamespace(ad=SimpleNamespace(openDocument=opening))
        with self.assertRaises(Fault):
            retry.retry(loader,plant,{'project_file_sha256':'model','file_sha256':'snapshot'},HERE,
                lambda *args:dict(pid=7,created=1,alive=True),lambda out,event:events.append(event),sleeper=sleeps.append)
        self.assertEqual(len(calls),3);self.assertEqual(sleeps,[2,4])
        self.assertEqual(events[-1]['event'],'document_open_retry_exhausted')

    def test_other_hresult_is_not_retried(self):
        with self.assertRaises(Fault):self.exercise(1,code=-2147023174)

    def test_same_hresult_at_a_different_loader_line_is_not_retried(self):
        def failing():raise Fault()
        plant=SimpleNamespace(ad=SimpleNamespace(prepare=failing,openDocument=lambda *args:self.fail('opened')))
        with self.assertRaises(Fault):
            retry.retry(other_phase_loader,plant,{},HERE,lambda *args:self.fail('ownership queried'),
                        lambda *args:self.fail('recorded'),sleeper=lambda *args:self.fail('slept'))

    def test_ownership_refusal_blocks_retry_before_sleep_or_record(self):
        def refused(*args):raise RuntimeError('not this worker generation')
        with self.assertRaisesRegex(RuntimeError,'generation'):self.exercise(1,owned=refused)

    def test_wrong_loader_contract_is_refused_before_any_native_call(self):
        with self.assertRaisesRegex(RuntimeError,'contract'):
            retry.retry(wrong_contract,None,{},HERE,None,None)

    def test_modified_attempt_limit_is_refused(self):
        with self.assertRaises(ValueError):retry.retry(loader,None,{},HERE,None,None,limit=4)

    def test_non_com_failure_is_not_retried(self):
        def failing(model):raise RuntimeError('native metadata mismatch')
        plant=SimpleNamespace(ad=SimpleNamespace(openDocument=failing))
        with self.assertRaisesRegex(RuntimeError,'metadata mismatch'):
            retry.retry(loader,plant,{},HERE,lambda *args:self.fail('ownership queried'),lambda *args:self.fail('recorded'))


class LaunchScope(unittest.TestCase):
    def plan(self):
        return dict(category='residual',case=17,seed=7,expected_matched_pairs=80,
            decisions_per_window=20,main_training_cycles=300,authorization='confirmed_failed_run_recovery',
            failed_worker=dict(pid=46740))

    def test_case17_original_budget_only(self):
        recovery.engine.scope(self.plan())
        for field,value in (('case',16),('category','combined'),('expected_matched_pairs',320),('authorization','healthy_restart')):
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                recovery.engine.scope(dict(self.plan(),**{field:value}))

    def test_invalid_validation_length_precedes_worker_or_simulator_work(self):
        with patch.object(recovery.engine,'worker',side_effect=AssertionError('worker called')):
            with self.assertRaises(ValueError):recovery.worker(16)

    def test_worker_refusal_imports_no_numerical_runtime(self):
        before=set(sys.modules)
        with patch.object(recovery.engine,'worker',side_effect=RuntimeError('guard blocked')):
            with self.assertRaisesRegex(RuntimeError,'guard blocked'):recovery.worker(8)
        self.assertFalse({'numpy','torch','review_numeric_recovery'} & (set(sys.modules)-before))

    def test_memory_gate_retains_six_gib_threshold(self):
        with self.assertRaisesRegex(RuntimeError,'6 GiB'):
            recovery.engine.start_guard(self.plan(),resources=lambda:dict(headroom_bytes=6*1024**3-1),
                identity=lambda pid:dict(pid=pid,alive=False,created=None),applications=[],status=dict(pid=46740,state='failed'))

    def test_new_task_does_not_replace_old_recovery_history(self):
        self.assertEqual(recovery.TASK,'RLAssistedMPC-ResidualNativeRetry2-20261024')
        self.assertEqual(recovery.engine.STATE.name,'20261024_residual_native_retry2')

    def test_duplicate_launch_is_refused_before_task_registration(self):
        with patch.object(recovery.engine,'verify',return_value=self.plan()), \
             patch.object(recovery.engine,'protected_check'), \
             patch.object(recovery.engine,'process_identity',return_value=dict(alive=False)), \
             patch.object(recovery.engine,'owned_remaining',return_value=[]),patch.object(Path,'exists',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'duplicate'):recovery.launch()


if __name__=='__main__':unittest.main(verbosity=2)
