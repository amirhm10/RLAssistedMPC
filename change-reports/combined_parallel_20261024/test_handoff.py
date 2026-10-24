"""A changed checkpoint or worker generation must refuse before any signal."""
import pytest
import handoff


def minimal(monkeypatch,tmp_path):
    monkeypatch.setattr(handoff,'HERE',tmp_path)
    plan={'failed_worker':{'pid':15940,'created':10,'alive':True}}
    full={'pair':{'sealed':'pair','layout':{'index':['version']}}}
    monkeypatch.setattr(handoff.activation.r,'verify',lambda:plan)
    monkeypatch.setattr(handoff.activation.r,'protected_check',lambda plan:None)
    monkeypatch.setattr(handoff.p,'closed_json',lambda path:{'manifest':'backup.json'} if str(path).endswith('preservation.json') else full)
    return plan


def test_checkpoint_change_blocks_before_reading_or_signaling_process(monkeypatch,tmp_path):
    minimal(monkeypatch,tmp_path)
    monkeypatch.setattr(handoff,'still_current',lambda pair:False)
    monkeypatch.setattr(handoff,'refreshed_benefit',lambda:pytest.fail('No launch checks after a changed checkpoint'))
    with pytest.raises(handoff.Deferred,match='Refresh the checkpoint'):handoff.retire()
    assert not (tmp_path/'retirement_intent.json').exists()


def test_pid_reuse_blocks_before_native_inventory_or_signal(monkeypatch,tmp_path):
    minimal(monkeypatch,tmp_path)
    monkeypatch.setattr(handoff,'still_current',lambda pair:True)
    monkeypatch.setattr(handoff,'refreshed_benefit',lambda:({},{}))
    monkeypatch.setattr(handoff.p.h,'process',lambda pid:{'pid':pid,'created':11,'alive':True})
    monkeypatch.setattr(handoff.p.h,'stable_batch',lambda paths:pytest.fail('No inventory after PID reuse'))
    with pytest.raises(RuntimeError,match='generation changed'):handoff.retire()
    assert not (tmp_path/'retirement_intent.json').exists()
