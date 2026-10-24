"""Refusal, ownership and concurrent-dispatch contracts; no numerical imports."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import parallel_branches as p


class ParallelContracts(unittest.TestCase):
    def config(self):
        return dict(case=15,category='combined',parent_identity={'pid':1,'created':11},
                    protected_main={'pid':2,'created':22},science_hashes={},adapter_hashes={},
                    canonical_model='model.dynf',canonical_sha256='ok')

    def process(self,pid):
        return dict(pid=pid,created=11 if pid==1 else 22,alive=True)

    def guarded(self,config,headroom=p.MINIMUM_HEADROOM,disk=p.MINIMUM_DISK):
        with patch.object(p.h,'process',side_effect=self.process),patch.object(p.h,'memory',return_value={'headroom_bytes':headroom}), \
             patch.object(p.shutil,'disk_usage',return_value=SimpleNamespace(free=disk)),patch.object(p.h,'digest',return_value='ok'):
            return p.guard(config)

    def test_generation_requires_pid_creation_and_access(self):
        a=dict(pid=2,created=22)
        self.assertTrue(p.same_generation(a,dict(a,alive=True)))
        for b in (dict(a,alive=None),dict(a,alive=False),dict(a,created=23,alive=True),dict(a,pid=3,alive=True)):
            self.assertFalse(p.same_generation(a,b))

    def test_wrong_case_refuses_before_any_process_access(self):
        for case in (13,14,16,18):
            c=self.config();c['case']=case
            with patch.object(p.h,'process',side_effect=AssertionError('No access')):
                with self.assertRaisesRegex(RuntimeError,'approved'):p.guard(c)

    def test_other_training_category_refuses(self):
        c=self.config();c['category']='markov'
        with self.assertRaises(RuntimeError):p.guard(c)

    def test_six_gib_boundary(self):
        self.guarded(self.config())
        with self.assertRaisesRegex(RuntimeError,'Six GiB'):self.guarded(self.config(),headroom=p.MINIMUM_HEADROOM-1)

    def test_disk_reserve_boundary(self):
        self.guarded(self.config())
        with self.assertRaisesRegex(RuntimeError,'32 GiB'):self.guarded(self.config(),disk=p.MINIMUM_DISK-1)

    def test_wrong_model_hash_refuses(self):
        c=self.config();c['canonical_sha256']='changed'
        with self.assertRaisesRegex(RuntimeError,'checksum'):self.guarded(c)

    def test_scientific_or_adapter_change_refuses(self):
        for key in ('science_hashes','adapter_hashes'):
            c=self.config();c[key]={'x':'changed'}
            with self.assertRaisesRegex(RuntimeError,'sealed'):self.guarded(c)

    def test_parent_generation_change_refuses(self):
        c=self.config();c['parent_identity']['created']=99
        with self.assertRaisesRegex(RuntimeError,'parent'):self.guarded(c)

    def test_protected_training_change_refuses(self):
        c=self.config();c['protected_main']['created']=99
        with self.assertRaisesRegex(RuntimeError,'advancing'):self.guarded(c)

    def test_parallel_submits_both_before_waiting(self):
        pool=object.__new__(p.BranchPool);events=[]
        pool.submit=lambda case,snapshot,steps,source: events.append(('submit',case,source)) or Path(source)
        pool.wait=lambda requests:events.append(('wait',tuple(requests))) or {}
        pool.pair('x',200,parallel=True)
        self.assertEqual(events,[('submit',15,'policy'),('submit',17,'supervisor'),('wait',('policy','supervisor'))])

    def test_serial_benchmark_waits_between_sources(self):
        pool=object.__new__(p.BranchPool);events=[]
        pool.submit=lambda case,snapshot,steps,source:events.append(('submit',source)) or Path(source)
        pool.wait=lambda requests:events.append(('wait',tuple(requests))) or {}
        pool.pair('x',200,parallel=False)
        self.assertEqual(events,[('submit','policy'),('wait',('policy',)),('submit','supervisor'),('wait',('supervisor',))])

    def test_result_checksum_must_match(self):
        pool=object.__new__(p.BranchPool);response={'state':'completed','source':'policy','trace':'x','trace_sha256':'good','frozen_state_before':'a','frozen_state_after':'a'}
        with patch.object(p,'closed_json',return_value=response),patch.object(p.h,'digest',return_value='bad'):
            with self.assertRaisesRegex(RuntimeError,'integrity'):pool.wait({'policy':Path(__file__)})

    def test_frozen_input_mutation_must_fail(self):
        pool=object.__new__(p.BranchPool);response={'state':'completed','source':'policy','trace':'x','trace_sha256':'good','frozen_state_before':'a','frozen_state_after':'b'}
        with patch.object(p,'closed_json',return_value=response),patch.object(p.h,'digest',return_value='good'):
            with self.assertRaisesRegex(RuntimeError,'integrity'):pool.wait({'policy':Path(__file__)})

    def test_wrong_branch_identity_must_fail(self):
        pool=object.__new__(p.BranchPool)
        with patch.object(p,'closed_json',return_value={'state':'completed','source':'supervisor'}):
            with self.assertRaisesRegex(RuntimeError,'failed'):pool.wait({'policy':Path(__file__)})

    def test_log_redirect_changes_only_path_and_restores_on_error(self):
        native=SimpleNamespace(snapshot=SimpleNamespace(path=Path('original.snp')))
        def original(value):
            self.assertEqual(value.snapshot.path,Path('private/native_restore_audit.snp'))
            raise ValueError('Injected restore error')
        with self.assertRaises(ValueError):p.restore_log_redirect(original,'private')(native)
        self.assertEqual(native.snapshot.path,Path('original.snp'))

    def test_snapshot_requires_symbolic_budget_seal(self):
        with self.assertRaisesRegex(RuntimeError,'budget protocol'):
            p.snapshot_sources({'science_hashes':{'x':'one'},'snapshot_source_hashes':{'x':'one'}})
        sources={'x':'one','column_e1_budget_protocol':'signed-protocol'}
        self.assertEqual(p.snapshot_sources({'science_hashes':{'x':'one'},'snapshot_source_hashes':sources}),sources)

    def test_snapshot_rejects_conflicting_file_seal(self):
        with self.assertRaisesRegex(RuntimeError,'conflict'):
            p.snapshot_sources({'science_hashes':{'x':'one'},'snapshot_source_hashes':{'x':'two','column_e1_budget_protocol':'signed-protocol'}})

    def test_native_restore_requires_full_copy_and_original_clock(self):
        p.verify_native_restore([{'full_variable_copy':True,'before':[10.],'after':[10.]}])
        for records in ([],[{'before':[10.],'after':[10.]}],[{'full_variable_copy':True,'before':[10.],'after':[10.1]}]):
            with self.assertRaises(RuntimeError):p.verify_native_restore(records)


if __name__=='__main__':unittest.main(verbosity=2)
