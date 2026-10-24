"""Backups and refreshed time saving must support a conditional handoff."""
from pathlib import Path
import pytest
import preserve
from transport_pilot import benefit


def test_alias_backup_keeps_original_and_independently_reads_copy(tmp_path,monkeypatch):
    from contextlib import contextmanager
    @contextmanager
    def local_snapshot(path,aliases):
        with Path(path).open('rb') as stream:yield stream
    monkeypatch.setattr(preserve,'private_snapshot',local_snapshot)
    source=tmp_path/'training.pkl';source.write_bytes(b'complete-state')
    receipt=preserve.copied_generation(source,tmp_path/'backup')
    assert source.read_bytes()==b'complete-state'
    assert Path(receipt['local_copy']).read_bytes()==b'complete-state'
    assert receipt['sha256']==preserve.p.h.digest(source)


def test_replaced_source_cannot_be_certified_as_original_generation(tmp_path,monkeypatch):
    from contextlib import contextmanager
    @contextmanager
    def replacing_snapshot(path,aliases):
        with Path(path).open('rb') as stream:yield stream
        replacement=Path(path).with_suffix('.replacement')
        replacement.write_bytes(b'new-generation')
        replacement.replace(path)
    monkeypatch.setattr(preserve,'private_snapshot',replacing_snapshot)
    source=tmp_path/'latest.json';source.write_bytes(b'old-generation')
    with pytest.raises(preserve.Deferred):preserve.copied_generation(source,tmp_path/'backup',attempts=1)
    assert len(list((tmp_path/'backup').glob('*.changed.json')))==1
    assert source.read_bytes()==b'new-generation'


def test_net_benefit_subtracts_export_development_restart_and_replay():
    work=dict(main_steps=36000,branch_steps=64000,remaining_pairs=120,
              replay_pairs=[dict(rollout_steps=200)])
    result=benefit(work,1.8,7.,elapsed_hours=1.)
    assert result['transport_allowance_hours']==2.
    assert result['restart_validation_allowance_hours']==1.
    assert result['elapsed_development_and_tests_hours']==1.
    assert result['replay_allowance_hours']>0
    assert result['parallel_future_hours']<result['serial_future_hours']
    assert benefit(work,1.1,7.,elapsed_hours=1.)['enough_time_saved'] is False
    assert benefit(work,1.8,7.,elapsed_hours=10.)['enough_time_saved'] is False
