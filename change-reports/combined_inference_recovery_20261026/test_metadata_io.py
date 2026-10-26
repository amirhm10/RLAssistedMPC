import json
from pathlib import Path
import pytest
from metadata_io import atomic_ownership


def denied():
    error = PermissionError('temporary Windows sharing/replace denial')
    error.winerror = 5
    return error


def test_transient_denial_keeps_previous_metadata_until_atomic_success(tmp_path):
    target = tmp_path / 'ownership_status.json'
    target.write_text('{"phase":"before"}')
    attempts = []

    def replace(source, destination):
        attempts.append(Path(source))
        if len(attempts) < 3:
            assert json.loads(target.read_text()) == {'phase': 'before'}
            raise denied()
        Path(source).replace(destination)

    value = {'phase': 'tracked', 'identity': {'pid': 12, 'created': 134374542030451776}}
    atomic_ownership(target, value, [tmp_path], replace=replace, pause=lambda _: None)
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert json.loads(target.read_text()) == value
    assert not attempts[-1].exists()


def test_permanent_denial_preserves_old_and_pending_metadata(tmp_path):
    target = tmp_path / 'owned_aspen_apps.json'
    target.write_text('[]')

    def replace(*_):
        raise denied()

    with pytest.raises(PermissionError):
        atomic_ownership(target, [{'pid': 1}], [tmp_path], timeout=0, replace=replace)
    assert target.read_text() == '[]'
    pending = list(tmp_path.glob('*.tmp'))
    assert len(pending) == 1 and json.loads(pending[0].read_text()) == [{'pid': 1}]


def test_scientific_or_foreign_targets_refuse_before_writing(tmp_path):
    owned = tmp_path / 'owned'
    owned.mkdir()
    for target in (owned / 'latest.json', tmp_path / 'ownership_status.json'):
        with pytest.raises(ValueError):
            atomic_ownership(target, {'state': 'untouched'}, [owned])
        assert not target.exists()
    assert not list(tmp_path.rglob('*.tmp'))
