"""Recovery lifecycle regressions; fake workers never open a simulator."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

import recover_markov_seed23 as recovery
import run_distillation_phase2_repeats as repeat
import review_blas_runtime as runtime


def write(path, value):
    recovery.write(path, value)


def completion(job, manifest, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "rl/input_data.pkl"
    result.parent.mkdir(exist_ok=True)
    result.write_bytes(b"retained scientific result")
    write(result.parent / "replication_metadata.json", {**{k: job[k] for k in ("category", "mode", "seed")},
          "episodes": 300, "profile": manifest["profile"],
          "entrypoint_sha256": manifest["source_hashes"][repeat.RUNNERS[job["category"]][1]]})
    write(directory / "worker_status.json", {"state": "completed", "episodes": 300, "aspen_closed": True,
                                            "result_file": str(result), "pid": 987654321})
    job.update(state="completed", exit_code=0, result_file=str(result), attempts=[{"directory": str(directory), "pid": 987654321}])


def fixture(tmp_path):
    directory = tmp_path / "recovery"
    old_directory = tmp_path / "original"
    jobs = repeat.build_jobs(list(repeat.RUNNERS), ["sg", "plain"], repeat.DEFAULT_SEEDS)
    manifest = {"jobs": jobs, "state": "failed", "profile": repeat.PROFILE, "episodes_per_run": 300,
                "repo_root": str(tmp_path), "python": sys.executable, "threads_per_run": 1,
                "source_hashes": {v[1]: "frozen" for v in repeat.RUNNERS.values()}}
    for job in jobs:
        job.update(aspen_path=str(tmp_path / f"C2S_SS_simulation{job['aspen_number']}.dynf"),
                   snaps_path=str(tmp_path / f"AM_C2S_SS_simulation{job['aspen_number']}"))
        Path(job["aspen_path"]).touch(exist_ok=True)
        Path(job["snaps_path"]).mkdir(exist_ok=True)
        if job["category"] in ("horizon", "weights", "markov"):
            completion(job, manifest, old_directory / f"{job['category']}_{job['mode']}" / str(job["seed"]) / "a1")
    target = next(j for j in jobs if recovery.key(j) == recovery.TARGET)
    target.update(state="failed", exit_code=3221225477)
    target.pop("result_file")
    write(old_directory / "manifest.json", manifest)
    write(Path(target["attempts"][-1]["directory"]) / "job.json", target)
    plan = {"old_directory": str(old_directory), "failed_attempt": target["attempts"][-1]["directory"],
            "external_workers": [{"pid": 123, "created": 456}], "original_supervisor": {"pid": 321, "created": 654}}
    write(directory / "plan.json", plan)
    write(directory / "pointer_before.json", {"repeats": str(old_directory), "column": "keep-comparisons"})
    for group in recovery.GROUPS:
        copy = deepcopy(manifest)
        copy["jobs"] = [deepcopy(target)] if group == "retry" else [deepcopy(j) for j in jobs if j["category"] == group]
        copy["state"] = "pending"
        for job in copy["jobs"]:
            job["state"] = "pending"
            job.pop("exit_code", None)
        write(directory / group / "manifest.json", copy)
    return directory, old_directory


FAKE_WORKER = '''
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone
p=Path(sys.argv[1]); j=json.loads(p.read_text()); d=p.parent
m=json.loads((Path(j['batch_directory'])/'manifest.json').read_text())
failed=sys.argv[2]=='fail'
status=dict(state='failed' if failed else 'completed',pid=os.getpid(),episodes=300,aspen_closed=True)
if not failed:
    result=d/'rl'/'input_data.pkl'; result.parent.mkdir(); result.write_bytes(b'new result')
    status['result_file']=str(result)
    meta={k:j[k] for k in ('category','mode','seed')}
    meta.update(episodes=300,profile=m['profile'],entrypoint_sha256=j['entrypoint_sha256'])
    (result.parent/'replication_metadata.json').write_text(json.dumps(meta))
status['finished_at']=datetime.now(timezone.utc).isoformat()
(d/'worker_status.json').write_text(json.dumps(status))
sys.exit(1 if failed else 0)
'''


@pytest.mark.parametrize("fail_retry,fail_residual", [(False, False), (True, True)])
def test_existing_queue_continues_after_native_failure_and_preserves_saved_work(tmp_path, fail_retry, fail_residual):
    directory, old_directory = fixture(tmp_path)
    old_bytes = (old_directory / "manifest.json").read_bytes()
    old_results = {str(p): recovery.digest(p) for p in old_directory.rglob("input_data.pkl")}
    fake = tmp_path / "fake_worker.py"
    fake.write_text(FAKE_WORKER)
    dispatched = []

    def launch(group_dir):
        manifest = recovery.read(group_dir / "manifest.json")
        def command(python, path):
            job = recovery.read(path)
            failed = ((group_dir.name == "retry" and fail_retry)
                      or (group_dir.name == "residual" and job["mode"] == "plain" and job["seed"] == 53 and fail_residual))
            dispatched.append(recovery.key(job))
            return [python, "-B", str(fake), str(path), "fail" if failed else "pass"]
        with patch.object(repeat, "write_json", recovery.write):
            result = repeat.run_batches(manifest, group_dir, command_factory=command)
        write(group_dir / "launch.json", {"phase": "started", "identity": {"pid": 111, "created": 222}})
        write(group_dir / "exit.json", {"returncode": result})

    with patch.object(recovery, "POINTER", tmp_path / "latest.json"), \
         patch.object(recovery, "check_bootstrap"), \
         patch.object(recovery, "close_orphans", return_value={"state": "closed"}), \
         patch.object(recovery, "same_process", return_value=False), \
         patch.object(recovery, "start_group", side_effect=launch):
        result = recovery.coordinate(directory)
    assert result == (1 if fail_retry else 0)
    assert len(dispatched) == 21 and len(set(dispatched)) == 21
    assert dispatched[0] == recovery.TARGET
    assert all(k[0] == "residual" for k in dispatched[1:11])
    assert all(k[0] == "combined" for k in dispatched[11:21])
    assert (old_directory / "manifest.json").read_bytes() == old_bytes
    assert all(recovery.digest(p) == sha for p, sha in old_results.items())
    final = recovery.read(directory / "manifest.json")
    assert sum(j["state"] == "completed" for j in final["jobs"]) == (48 if fail_retry else 50)
    assert recovery.read(directory / "coordinator_status.json")["queued_runs_dispatched"] == 20
    assert recovery.read(tmp_path / "latest.json")["column"] == "keep-comparisons"


def test_queue_waits_for_both_original_workers_and_seed_retry():
    plan = {"external_workers": [{"pid": 1}], "original_supervisor": {"pid": 2}}
    with patch.object(recovery, "same_process", side_effect=lambda p: p["pid"] == 1):
        assert not recovery.queue_ready(plan, True)
    with patch.object(recovery, "same_process", return_value=False):
        assert not recovery.queue_ready(plan, False)
        assert recovery.queue_ready(plan, True)


def test_recycled_pid_is_never_treated_as_the_original_worker():
    with patch.object(recovery, "process_identity", return_value={"pid": 12, "created": 999, "alive": True}):
        assert recovery.same_process({"pid": 12, "created": 123}) is False
    # Also cover a reuse between the inventory query and opening a kill handle.
    kernel = MagicMock()
    kernel.OpenProcess.return_value = 42
    def stamps(handle, created, exited, system, user):
        created._obj.dwLowDateTime = 999
        return 1
    def exit_code(handle, code):
        code._obj.value = 259
        return 1
    kernel.GetProcessTimes.side_effect = stamps
    kernel.GetExitCodeProcess.side_effect = exit_code
    with patch.object(recovery.ctypes, "WinDLL", return_value=kernel):
        assert recovery.terminate_owned_identity({"pid": 12, "created": 123}) == "pid_reused_untouched"
    kernel.TerminateProcess.assert_not_called()
    kernel.CloseHandle.assert_called_once_with(42)


@pytest.mark.parametrize("alive", [True, None])
def test_orphan_cleanup_refuses_live_or_unverified_worker(tmp_path, alive):
    write(tmp_path / "job.json", {"category": "markov", "mode": "sg", "seed": 23})
    write(tmp_path / "worker_status.json", {"pid": 55})
    with patch.object(recovery, "process_identity", return_value={"alive": alive}), \
         patch.object(recovery, "terminate_owned_identity") as terminate:
        with pytest.raises(RuntimeError, match="live or unverified"):
            recovery.close_orphans(tmp_path, tmp_path / "cleanup.json")
        terminate.assert_not_called()


def test_cleanup_targets_only_recorded_orphan_and_verified_children(tmp_path):
    write(tmp_path / "job.json", {"category": "markov", "mode": "sg", "seed": 23})
    write(tmp_path / "worker_status.json", {"pid": 55, "state": "running"})
    before = (tmp_path / "worker_status.json").read_bytes()
    parent = {"pid": 7, "created": 100}
    child = {"pid": 8, "created": 101, "parent_pid": 7}
    unresolved = [{"recorded": parent, "current": {**parent, "alive": True}}]
    with patch.object(recovery, "process_identity", return_value={"alive": False}), \
         patch.object(recovery, "owned_apps_remaining", side_effect=[unresolved, []]), \
         patch.object(recovery, "children_of", return_value=[child]), \
         patch.object(recovery, "terminate_owned_identity", return_value="closed_verified_orphan") as terminate:
        audit = recovery.close_orphans(tmp_path, tmp_path / "cleanup.json")
    assert [c.args[0] for c in terminate.call_args_list] == [child, parent]
    assert audit["state"] == "closed"
    assert (tmp_path / "worker_status.json").read_bytes() == before


def test_source_change_blocks_new_group_before_any_spawn(tmp_path):
    write(tmp_path / "manifest.json", {"source_hashes": {"original.py": "old"}, "bootstrap_sha256": {}})
    with patch.object(repeat, "source_hashes", return_value={"original.py": "changed"}), \
         patch.object(recovery.subprocess, "Popen") as popen:
        with pytest.raises(RuntimeError, match="Scientific sources changed"):
            recovery.start_group(tmp_path)
        popen.assert_not_called()
    assert not (tmp_path / "launch.json").exists()


def test_runtime_rejects_modified_manifest_before_scientific_import(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    write(tmp_path / "manifest.json", {"wheels": runtime.WHEELS, "files": {}})
    with pytest.raises(RuntimeError, match="manifest changed"):
        runtime.activate(tmp_path, "wrong hash")


def test_runtime_rejects_changed_private_binary(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    binary = tmp_path / "site-packages/numpy.libs/private.dll"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"changed")
    write(tmp_path / "manifest.json", {"wheels": runtime.WHEELS,
          "files": {"numpy.libs/private.dll": hashlib.sha256(b"original").hexdigest()}})
    with pytest.raises(RuntimeError, match="runtime file changed"):
        runtime.activate(tmp_path)


def test_progress_shows_a_stopped_queue_coordinator(tmp_path):
    from revision_comparisons.progress import snapshot
    recovery_dir = tmp_path / "current"
    write(tmp_path / "Distillation/Results/review_recovery/latest.json", {
          "repeats": str(recovery_dir), "repeat_recovery_plan": str(recovery_dir / "plan.json")})
    write(recovery_dir / "coordinator_status.json", {"state": "running", "stage": "waiting", "pid": 222})
    with patch("revision_comparisons.progress.process_info", return_value={"alive": False, "cpu_seconds": 1}):
        data = snapshot(tmp_path)
    assert data["repeat_queue_coordinator"]["state"] == "stopped_without_final_status"


def test_atomic_manifest_write_retries_brief_windows_reader_locks(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"state":"old"}')
    original = Path.replace
    attempts = []
    def replace(source, target):
        attempts.append(source)
        if len(attempts) <= 2:
            assert json.loads(path.read_text())["state"] == "old"
            raise PermissionError("injected Windows sharing violation")
        return original(source, target)
    with patch.object(Path, "replace", replace):
        recovery.write(path, {"state": "new"})
    assert len(attempts) == 3 and recovery.read(path) == {"state": "new"}
    assert not list(tmp_path.glob("*.tmp"))
