"""Guard and full-state/provenance regressions, without simulators/numerics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('combined_failed_recovery',HERE/'recover_combined.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Guards(unittest.TestCase):
    def plan(self):
        return dict(category='combined',case=16,seed=7,expected_matched_pairs=320,
            decisions_per_window=20,main_training_cycles=300,authorization='confirmed_failed_run_recovery',
            failed_worker=dict(pid=123))

    def guard(self,**overrides):
        options=dict(resources=lambda:dict(headroom_bytes=6*1024**3),
            identity=lambda pid:dict(pid=pid,alive=False,created=None),applications=[],
            status=dict(pid=123,state='failed'))
        options.update(overrides)
        return r.engine.start_guard(self.plan(),**options)

    def test_wrong_case_category_budget_and_healthy_authorization_refused(self):
        for key,value in (('case',17),('category','residual'),('seed',23),('expected_matched_pairs',80),
                          ('decisions_per_window',100),('main_training_cycles',200),('authorization','healthy_restart')):
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                r.scope(dict(self.plan(),**{key:value}))

    def test_live_reused_or_inaccessible_original_blocks_launch(self):
        for alive in (True,None):
            with self.subTest(alive=alive),self.assertRaises(RuntimeError):
                self.guard(identity=lambda pid:dict(pid=pid,alive=alive,created=987))

    def test_other_live_worker_blocks_duplicate_before_native_creation(self):
        with self.assertRaises(RuntimeError):
            self.guard(status=dict(pid=456,state='running'),
                identity=lambda pid:dict(pid=pid,alive=pid==456,created=999))

    def test_remaining_native_application_blocks_launch(self):
        with self.assertRaises(RuntimeError):
            self.guard(applications=[dict(pid=999)])

    def test_memory_below_threshold_is_refused_exact_threshold_passes(self):
        with self.assertRaisesRegex(RuntimeError,'6 GiB'):
            self.guard(resources=lambda:dict(headroom_bytes=6*1024**3-1))
        self.guard()

    def test_invalid_validation_length_precedes_verification(self):
        with patch.object(r.engine,'verify',side_effect=AssertionError('verification called')):
            with self.assertRaises(ValueError):
                r.worker(16)

    def test_guard_refusal_precedes_numerical_imports(self):
        before=set(sys.modules)
        with patch.object(r.engine,'verify',return_value=self.plan()), \
             patch.object(r.engine,'start_guard',side_effect=RuntimeError('blocked')):
            with self.assertRaisesRegex(RuntimeError,'blocked'):
                r.worker(8)
        self.assertFalse({'numpy','torch','review_numeric_recovery'} & (set(sys.modules)-before))


class NativeImport(unittest.TestCase):
    def test_unchanged_full_state_keeps_return_value_and_records_equality(self):
        records=[]
        result=r.checked_import(lambda plant,meta:'restored','plant',{},lambda:'full-state',
                                lambda before,after:records.append((before,after)))
        self.assertEqual(result,'restored')
        self.assertEqual(records,[('full-state','full-state')])

    def test_any_learner_or_controller_mutation_is_refused(self):
        state=dict(horizon=1,markov=2,weights=3,residual=4,controller=5)
        original=dict(state)
        for component in original:
            state.update(original)
            def importer(plant,meta):
                state[component]+=1
            with self.subTest(component=component),self.assertRaisesRegex(RuntimeError,'mutated'):
                r.checked_import(importer,None,{},lambda:json.dumps(state,sort_keys=True),
                                 lambda *args:self.fail('invalid receipt recorded'))

    def test_empty_full_state_digest_is_refused(self):
        with self.assertRaises(RuntimeError):
            r.checked_import(lambda *args:None,None,{},lambda:'',lambda *args:None)


class ManagerContract(unittest.TestCase):
    def exercise(self,*,agents=4,index_sources=None,mutate_sources=False,wrong_session=False):
        sources={'sealed_runner':'unchanged','column_e1_budget_protocol':'original'}
        env=dict(horizon=1,markov=2,weights=3,residual=4,controller=dict(observer=5))
        session=SimpleNamespace()
        events=[]
        native=ModuleType('column_e1.checkpoints')
        native.import_native=lambda *args:'native'
        runtime=ModuleType('column_e1.runtime')
        runtime.integrity_payload=lambda environment,compiled:environment
        runtime.digest_state=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        package=ModuleType('column_e1')
        package.checkpoints=native
        package.runtime=runtime
        fresh=ModuleType('column_e1_process_recovery')
        fresh.append=lambda output,event:events.append(event)
        class Base:
            def __init__(self):
                self.sources=dict(sources);self.index=HERE/'unused-index';self.compiled=object()
            def restore(self,environment,restored_session):
                native.import_native('plant',{})
                if mutate_sources:
                    self.sources['unexpected']='new'
                restored_session.source_hashes={} if wrong_session else dict(self.sources)
        modules={'column_e1':package,'column_e1.checkpoints':native,'column_e1.runtime':runtime,
                 'column_e1_process_recovery':fresh}
        index=dict(completed_steps=80800,source_hashes=sources if index_sources is None else index_sources,
                   agents={str(i):{} for i in range(agents)})
        with patch.dict(sys.modules,modules),patch.object(r.engine,'read',return_value=index):
            manager=r.source_preserving_manager(lambda *args,**kwargs:Base,HERE)('original',{},validation_steps=8)()
            manager.restore(env,session)
        return manager,session,events,sources

    def test_original_scientific_protocol_map_remains_exact(self):
        manager,session,events,sources=self.exercise()
        self.assertEqual(manager.sources,sources)
        self.assertEqual(session.source_hashes,sources)
        self.assertEqual(events[0]['before'],events[0]['after'])
        self.assertEqual(events[0]['completed_steps'],80800)

    def test_missing_fourth_learner_refused(self):
        with self.assertRaisesRegex(RuntimeError,'four-learner'):
            self.exercise(agents=3)

    def test_wrong_source_map_refused(self):
        with self.assertRaisesRegex(RuntimeError,'provenance'):
            self.exercise(index_sources={'changed':'runner'})

    def test_added_checkpoint_marker_refused(self):
        with self.assertRaisesRegex(RuntimeError,'markers changed'):
            self.exercise(mutate_sources=True)

    def test_session_source_map_mutation_refused(self):
        with self.assertRaisesRegex(RuntimeError,'markers changed'):
            self.exercise(wrong_session=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
