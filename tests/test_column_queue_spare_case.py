"""Existing queue proceeds with case 3 occupied; no live Aspen in these tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

import continue_column_queue as queue
import recover_markov_seed23 as recovery
import run_distillation_phase2_repeats as repeat


def fixture(tmp_path):
    directory = tmp_path / "queue"
    original = tmp_path / "old"
    retry_dir = tmp_path / "retry"
    jobs = repeat.build_jobs(list(repeat.RUNNERS), ["sg", "plain"], repeat.DEFAULT_SEEDS)
    manifest = {"jobs": jobs, "state": "failed", "profile": repeat.PROFILE, "episodes_per_run": 300,
                "repo_root": str(tmp_path), "python": sys.executable, "threads_per_run": 1,
                "source_hashes": {v[1]: "frozen" for v in repeat.RUNNERS.values()}}
    for job in jobs:
        job.update(aspen_path=str(tmp_path / f"C2S_SS_simulation{job['aspen_number']}.dynf"),
                   snaps_path=str(tmp_path / f"snaps{job['aspen_number']}"))
        Path(job["aspen_path"]).write_bytes(b"identical original model")
        Path(job["snaps_path"]).mkdir(exist_ok=True)
        if job["category"] not in queue.CHANNELS:
            job["state"] = "completed"
    target = next(j for j in jobs if recovery.key(j) == recovery.TARGET)
    target["state"] = "failed"
    recovery.write(original / "manifest.json", manifest)
    live_retry = deepcopy(target)
    live_retry.update(state="running", attempts=[{"pid": 99999, "directory": str(retry_dir)}])
    recovery.write(retry_dir / "manifest.json", {**manifest, "jobs": [live_retry], "state": "running"})
    recovery.write(directory / "plan.json", {"original_directory": str(original), "retry_directory": str(retry_dir),
                   "previous_directory": str(original), "previous_coordinator_failure": {"pid": 22}})
    recovery.write(directory / "pointer_before.json", {"repeats": str(original), "column": "keep-other-runs"})
    spare = tmp_path / "C2S_SS_simulation12.dynf"
    spare.write_bytes(b"identical original model")
    for group in queue.CHANNELS:
        group_manifest = deepcopy(manifest)
        group_manifest.update(jobs=[deepcopy(j) for j in jobs if j["category"] == group], state="pending", model_hashes={})
        for job in group_manifest["jobs"]:
            if (job["mode"], job["seed"]) == ("sg", 23):
                job.update(aspen_number=12, aspen_path=str(spare), snaps_path=str(tmp_path / "snaps12" / group))
            group_manifest["model_hashes"][job["aspen_path"]] = recovery.digest(job["aspen_path"])
        recovery.write(directory / group / "manifest.json", group_manifest)
    return directory, original, retry_dir


FAKE = '''
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); j=json.loads(p.read_text()); d=p.parent
m=json.loads((Path(j['batch_directory'])/'manifest.json').read_text())
out=d/'rl'/'input_data.pkl'; out.parent.mkdir(); out.write_bytes(b'new completed output')
meta={k:j[k] for k in ('category','mode','seed')}
meta.update(episodes=300,profile=m['profile'],entrypoint_sha256=j['entrypoint_sha256'])
(out.parent/'replication_metadata.json').write_text(json.dumps(meta))
(d/'worker_status.json').write_text(json.dumps(dict(state='completed',episodes=300,aspen_closed=True,result_file=str(out),pid=os.getpid())))
'''


def test_twenty_queued_workers_run_in_order_while_case3_lease_stays_occupied(tmp_path):
    directory, original, retry_dir = fixture(tmp_path)
    before = (original / "manifest.json").read_bytes()
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    dispatched = []

    def launch(group_dir):
        manifest = recovery.read(group_dir / "manifest.json")
        queue.validate_group(manifest)
        assert recovery.read(retry_dir / "manifest.json")["jobs"][0]["state"] == "running"
        def command(python, path):
            job = recovery.read(path)
            dispatched.append((job["category"], job["mode"], job["seed"], job["aspen_number"]))
            assert job["aspen_number"] != 3
            return [python, "-B", str(fake), str(path)]
        with patch.object(repeat, "MAX_ASPEN_FILES", 12), patch.object(repeat, "write_json", recovery.write):
            code = repeat.run_batches(manifest, group_dir, command_factory=command)
        recovery.write(group_dir / "exit.json", {"returncode": code})
        if group_dir.name == "combined":
            retry = recovery.read(retry_dir / "manifest.json")
            retry["jobs"][0]["state"] = "completed"
            recovery.write(retry_dir / "manifest.json", retry)

    # The exact resource which blocked the former queue remains held throughout.
    with repeat.aspen_file_lock(tmp_path / "C2S_SS_simulation3.dynf", queue.ROOT / "Distillation/Results/_aspen_file_locks"), \
         patch.object(recovery, "POINTER", tmp_path / "latest.json"), \
         patch.object(recovery, "check_bootstrap"), \
         patch.object(queue, "absent", return_value=True), \
         patch.object(queue, "start_group", side_effect=launch):
        assert queue.coordinate(directory) == 0
    assert len(dispatched) == 20 and len(set(dispatched)) == 20
    assert all(j[0] == "residual" for j in dispatched[:10])
    assert all(j[0] == "combined" for j in dispatched[10:])
    assert [(j[0], j[3]) for j in dispatched if j[1:3] == ("sg", 23)] == [("residual", 12), ("combined", 12)]
    assert (original / "manifest.json").read_bytes() == before
    assert repeat.MAX_ASPEN_FILES == 10
    assert recovery.read(tmp_path / "latest.json")["column"] == "keep-other-runs"


def test_final_completion_record_avoids_querying_inaccessible_retired_pid(tmp_path):
    recovery.write(tmp_path / "exit.json", {"returncode": 0})
    recovery.write(tmp_path / "manifest.json", {"jobs": [{"state": "completed"}]})
    recovery.write(tmp_path / "launch.json", {"identity": {"pid": 43244}})
    with patch.object(queue, "absent", side_effect=AssertionError("Retired PID must not block a saved completion")):
        assert queue.group_finished(tmp_path)


def test_uncertain_pid_uses_fresh_windows_inventory():
    service = MagicMock()
    service.ExecQuery.return_value = []
    with patch.object(queue, "process_identity", return_value={"alive": None}), \
         patch("win32com.client.GetObject", return_value=service):
        assert queue.absent(43244) is True
        service.ExecQuery.return_value = [object()]
        assert queue.absent(43244) is False


@pytest.mark.parametrize("slot", [3, 11, 13, 15])
def test_spare_slot_allowance_cannot_assign_other_or_live_cases(tmp_path, slot):
    directory, _, _ = fixture(tmp_path)
    manifest = recovery.read(directory / "residual/manifest.json")
    target = next(j for j in manifest["jobs"] if j["mode"] == "sg" and j["seed"] == 23)
    target["aspen_number"] = slot
    with patch.object(recovery, "check_bootstrap"), pytest.raises(RuntimeError, match="Unapproved"):
        queue.validate_group(manifest)
    assert repeat.MAX_ASPEN_FILES == 10


def test_changed_model_is_rejected_before_new_process(tmp_path):
    directory, _, _ = fixture(tmp_path)
    (tmp_path / "C2S_SS_simulation12.dynf").write_bytes(b"changed")
    with patch.object(recovery, "check_bootstrap"), patch.object(queue.subprocess, "Popen") as popen:
        with pytest.raises(RuntimeError, match="model changed"):
            queue.start_group(directory / "residual")
        popen.assert_not_called()


def test_worker_uses_only_scoped_spare_case_validation(tmp_path):
    directory, _, _ = fixture(tmp_path)
    manifest = recovery.read(directory / "residual/manifest.json")
    job = next(j for j in manifest["jobs"] if j["mode"] == "sg" and j["seed"] == 23)
    attempt = directory / "residual/attempt"
    recovery.write(attempt / "job.json", {**job, "batch_directory": str(directory / "residual")})
    def delegate(path):
        assert repeat.MAX_ASPEN_FILES == 12
        return 0
    with patch.object(recovery, "worker", side_effect=delegate):
        assert queue.worker(attempt / "job.json") == 0
    assert repeat.MAX_ASPEN_FILES == 10


def test_exit_record_cannot_hide_incomplete_channel_states(tmp_path):
    recovery.write(tmp_path / "exit.json", {"returncode": 1})
    recovery.write(tmp_path / "manifest.json", {"jobs": [{"state": "running"}]})
    with pytest.raises(RuntimeError, match="final per-job"):
        queue.group_finished(tmp_path)
