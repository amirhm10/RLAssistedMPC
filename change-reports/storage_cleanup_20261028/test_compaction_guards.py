"""Exercise deletion boundaries, exact recovery and corruption refusal on fixtures."""
import hashlib
import json
from pathlib import Path
import unittest
import compact_native as c


class Guards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=c.HERE/'guard_test_fixtures'
        assert not cls.fixture.exists()
        cls.fixture.mkdir()
        assert cls.fixture.resolve().is_relative_to(c.HERE.resolve())
        c.HERE=cls.fixture
        c.ALLOWED=cls.fixture/'closed_results'
        c.ALLOWED.mkdir()
        c.ARCHIVE=cls.fixture/'archives'
        cls.protected=c.ALLOWED/'final_checkpoint'
        cls.protected.mkdir()
        cls.final=cls.protected/'final.dynf'
        cls.final.write_bytes(b'keep final native state')
        cls.proof=dict(passed=True,errors=[],protected_native_directories=[str(cls.protected)])
        c.atomic(cls.fixture/'protected_evidence_before.json',cls.proof)
        (cls.fixture/'inventory_local_archive.jsonl').write_text('',encoding='utf-8')

    def test_1_refuse_protected(self):
        with self.assertRaises(ValueError):c.eligible(self.final,[str(self.protected)])

    def test_2_refuse_outside(self):
        p=self.fixture/'outside.dynf';p.write_bytes(b'keep')
        with self.assertRaises(ValueError):c.eligible(p,[])

    def test_3_refuse_scientific_data(self):
        p=c.ALLOWED/'training.pkl';p.write_bytes(b'keep raw scientific data')
        with self.assertRaises(ValueError):c.eligible(p,[])

    def test_4_lossless_removal_and_restore(self):
        p=c.ALLOWED/'obsolete.dynf';payload=b'exact original native state\x00\xff'*10000;p.write_bytes(payload)
        row=dict(path=str(p),bytes=len(payload),version=c.stamp(p))
        c.atomic(self.fixture/'native_compaction_plan.json',dict(allowed_root=str(c.ALLOWED.resolve()),protected=[str(self.protected)],candidates=[row]))
        c.execute(1)
        self.assertFalse(p.exists());self.assertTrue(self.final.exists())
        target=self.fixture/'restored.dynf';c.restore(str(p),str(target))
        self.assertEqual(target.read_bytes(),payload)
        with self.assertRaises(AssertionError):c.restore(str(p),str(target))

    def test_5_corruption_refused(self):
        z=next(c.ARCHIVE.glob('*.zip'))
        entry=dict(archive=str(z),archive_sha256='0'*64,member='state.dynf')
        with self.assertRaises(AssertionError):c.verify_archive(entry,'0'*64,1,set())

    def test_6_changed_source_refused(self):
        p=c.ALLOWED/'changed.dynf';p.write_bytes(b'before')
        row=dict(path=str(p),bytes=p.stat().st_size,version=c.stamp(p))
        c.atomic(self.fixture/'native_compaction_plan.json',dict(allowed_root=str(c.ALLOWED.resolve()),protected=[str(self.protected)],candidates=[row]))
        p.write_bytes(b'changed after inventory')
        with self.assertRaises(AssertionError):c.execute(1)
        self.assertTrue(p.exists())


if __name__=='__main__':
    unittest.main()
