import importlib.util
from pathlib import Path
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'change-reports/e1_recovery_20261023/e1_supervisor_lifecycle.py'
SPEC = importlib.util.spec_from_file_location('e1_supervisor_lifecycle', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_records():
    old = dict(pid=123, created=456, alive=True)
    plan = dict(queue_coordinator=dict(pid=123, directory='unused'), protected=[old], prepared_at='2026-10-23T18:00:00')
    values = {
        'coordinator_identity.json': old,
        'coordinator_status.json': dict(pid=123, state='completed', stage='finished', queued_runs_dispatched=20,
                                        finished_at='2026-10-23T19:00:00'),
        'manifest.json': dict(jobs=[dict(state='completed') for _ in range(10)]),
    }
    return plan, values, lambda path: values[path.name]


def test_live_original_is_still_protected():
    plan, _, read = fixture_records()
    assert MODULE.coordinator_completed(plan, read, lambda pid: plan['protected'][0]) is False


def test_verified_natural_completion_can_release_guard():
    plan, _, read = fixture_records()
    assert MODULE.coordinator_completed(plan, read, lambda pid: dict(pid=pid, alive=False)) is True


@pytest.mark.parametrize('problem', ['pid_reused', 'incomplete_queue', 'failed_coordinator', 'wrong_creation'])
def test_incomplete_or_reused_coordinator_remains_blocked(problem):
    plan, values, read = fixture_records()
    current = dict(pid=123, alive=False)
    if problem == 'pid_reused': current.update(alive=True, created=789)
    elif problem == 'incomplete_queue': values['manifest.json']['jobs'][0]['state'] = 'running'
    elif problem == 'failed_coordinator': values['coordinator_status.json']['state'] = 'failed'
    elif problem == 'wrong_creation': values['coordinator_identity.json'] = dict(pid=123, created=999)
    with pytest.raises(RuntimeError):
        MODULE.coordinator_completed(plan, read, lambda pid: current)
