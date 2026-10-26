import json
import os
import pytest
import offline_pair as offline
from aspen_retention import Deferred


def layout_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'training_checkpoints'
    native = root / 'native_1_118808'
    child = native / 'review_e1_fixture'
    child.mkdir(parents=True)
    for path in [root / 'latest.json', root / 'training_state_0.pkl', root / 'training_state_1.pkl',
                 child / 'metadata.json', child / 'snapshot.snp', child / 'C2S_SS_simulation16.dynf']:
        path.write_bytes(b'fixture')
        os.utime(path, (100, 100))
    audit = child / 'snapshot.restores.jsonl'
    audit.write_text(json.dumps(dict(full_variable_copy=True, before=[10.], after=[10.])) + '\n')
    os.utime(audit, (200, 200))
    os.utime(child, (200, 200))
    os.utime(native, (100, 100))
    monkeypatch.setattr(offline.p.h, 'process', lambda _: {'alive': False})
    monkeypatch.setattr(offline.p.h, 'native_processes', lambda: {})
    return root, child


def test_only_verified_closed_restore_audit_may_be_newer(tmp_path, monkeypatch):
    root, _ = layout_fixture(tmp_path, monkeypatch)
    layout = offline.closed_layout(root, {'pid': 1})
    assert layout['native_entries'] == (('native_1_118808', ('review_e1_fixture',)),)


def test_live_failed_or_native_process_refuses_exception(tmp_path, monkeypatch):
    root, _ = layout_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(offline.p.h, 'native_processes', lambda: {2: {'alive': True}})
    with pytest.raises(Deferred, match='closed'):
        offline.closed_layout(root, {'pid': 1})


def test_newer_scientific_native_component_refuses_exception(tmp_path, monkeypatch):
    root, child = layout_fixture(tmp_path, monkeypatch)
    os.utime(child / 'snapshot.snp', (200, 200))
    with pytest.raises(Deferred, match='non-audit'):
        offline.closed_layout(root, {'pid': 1})
