from pathlib import Path
from types import SimpleNamespace
import json

import pytest

import column_e1_process_recovery as recovery


def test_unowned_application_is_never_closed(tmp_path):
    events = []
    app = SimpleNamespace(processId=123, CloseDocument=lambda *_: events.append('close'),
                          Quit=lambda: events.append('quit'))
    (tmp_path / 'owned_aspen_apps.json').write_text('[]')
    with pytest.raises(RuntimeError, match='unowned'):
        recovery.retire(SimpleNamespace(ad=app), tmp_path)
    assert events == []


def test_reused_native_pid_is_never_closed(tmp_path, monkeypatch):
    import column_review.common as common
    events = []
    old = dict(pid=123, alive=True, created=11)
    (tmp_path / 'owned_aspen_apps.json').write_text(json.dumps([old]))
    monkeypatch.setattr(common, 'process_identity', lambda pid: dict(pid=pid, alive=True, created=22))
    app = SimpleNamespace(processId=123, CloseDocument=lambda *_: events.append('close'),
                          Quit=lambda: events.append('quit'))
    with pytest.raises(RuntimeError, match='reused'):
        recovery.retire(SimpleNamespace(ad=app), tmp_path)
    assert events == []


def test_invalid_checkpoint_is_rejected_before_application_access(tmp_path):
    model = tmp_path / 'C2S_SS_simulation14.dynf'; model.write_bytes(b'private model')
    database = tmp_path / 'snapshot.snp'; database.write_bytes(b'private snapshot')
    metadata = dict(file=str(database), file_sha256='invalid', project_file=str(model),
                    project_file_sha256='invalid', name='owned')
    # No application attribute is supplied: any simulator access would fail.
    from review_experiments.e1.runtime import RestorationError
    with pytest.raises(RestorationError, match='checksum'):
        recovery.load_project(SimpleNamespace(path=str(model)), metadata, tmp_path)


def test_different_case_name_is_rejected_before_retiring_application(tmp_path):
    from column_review.common import digest
    from review_experiments.e1.runtime import RestorationError
    model = tmp_path / 'C2S_SS_simulation14.dynf'; model.write_bytes(b'private model')
    database = tmp_path / 'snapshot.snp'; database.write_bytes(b'private snapshot')
    metadata = dict(file=str(database), file_sha256=digest(database), project_file=str(model),
                    project_file_sha256=digest(model), name='owned')
    with pytest.raises(RestorationError, match='filename'):
        recovery.load_project(SimpleNamespace(path=str(tmp_path / 'C2S_SS_simulation16.dynf')), metadata, tmp_path)
