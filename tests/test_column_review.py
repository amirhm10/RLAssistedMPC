"""Offline preparation, isolation, native-snapshot and failure contracts."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import types
from unittest.mock import patch

import numpy as np
import pytest

import run_distillation_remaining as runner
from systems.distillation import get_distillation_notebook_defaults
from column_review import common
from column_review.pilot import NativeSnapshot, paired_continuations


def fake_cases(tmp_path):
    def resolve(category, root_override=None, number_override=None):
        path = tmp_path / f"C2S_SS_simulation{number_override}.dynf"
        path.write_bytes(b"identical private offline fixture")
        return {"aspen_path": str(path), "snaps_path": str(tmp_path / f"AM_{path.stem}"),
                "aspen_sha256": common.digest(path)}
    return resolve


def test_preparation_uses_only_cases12_to16_without_any_process_or_com(tmp_path):
    with patch("review_experiments.e3.runtime.resolve_aspen", fake_cases(tmp_path)), \
         patch("column_review.typelib.inspect_api", return_value={"live_aspen_opened": False}), \
         patch("subprocess.Popen", side_effect=AssertionError("No worker in preparation")), \
         patch("win32com.client.DispatchEx", side_effect=AssertionError("No COM app in preparation")):
        plan = runner.prepare(tmp_path / "batch")
    assert {k: v["number"] for k, v in plan["mapping"].items()} == common.MAPPING
    assert plan["critic_only_cycles"] == [11, 13] and plan["policy_release_cycle"] == 14
    assert plan["new_processes_started"] is False
    manifest = common.read(tmp_path / "batch/manifest.json")
    assert manifest["state"] == "prepared"
    assert len(manifest["jobs"]) == 5 and all(item["attempts"] == [] for item in manifest["jobs"])
    assert {item["method"] for item in manifest["jobs"]} == {"sg", "threshold", "ls", "bo", "e1_pilot"}
    assert all((tmp_path / "batch" / method / "run.cmd").exists() for method in common.MAPPING)
    assert not (tmp_path / "batch/visible_launches.json").exists()


def test_nonidentical_plant_copies_are_rejected_before_batch_creation(tmp_path):
    resolve = fake_cases(tmp_path)
    def changed(*args, **kwargs):
        result = resolve(*args, **kwargs)
        if kwargs["number_override"] == 14:
            result["aspen_sha256"] = "different"
        return result
    with patch("review_experiments.e3.runtime.resolve_aspen", changed):
        with pytest.raises(ValueError, match="differ"):
            runner.prepare(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_case_lease_stays_reserved_between_nested_aspen_sessions(tmp_path, monkeypatch):
    import run_distillation_phase2_repeats as repeat
    monkeypatch.setattr(common, "ROOT", tmp_path)
    original = repeat.aspen_file_lock
    case = tmp_path / "C2S_SS_simulation15.dynf"
    lock_dir = tmp_path / "Distillation/Results/_aspen_file_locks"
    with common.case_lease({"aspen_path": str(case)}):
        for _ in range(3):
            with repeat.aspen_file_lock(case, lock_dir):
                pass
            with pytest.raises(RuntimeError, match="reserved"):
                with original(case, lock_dir):
                    pass
        with pytest.raises(RuntimeError, match="outside"):
            with repeat.aspen_file_lock(tmp_path / "C2S_SS_simulation1.dynf", lock_dir):
                pass
    with original(case, lock_dir):
        pass  # released only after the full worker lease


def test_launch_checks_all_case_locks_before_opening_any_cmd(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for method, number in common.MAPPING.items():
        (tmp_path / method).mkdir()
        common.write(tmp_path / method / "job.json", {"aspen": {"aspen_path": str(tmp_path / f"{number}.dynf")}})
    @contextmanager
    def lock(path, directory):
        if Path(path).stem == "14":
            raise RuntimeError("occupied")
        yield
    with patch.object(runner, "check_job"), patch("run_distillation_phase2_repeats.aspen_file_lock", lock), \
         patch("subprocess.Popen", side_effect=AssertionError("Do not start a partial batch")):
        with pytest.raises(RuntimeError, match="occupied"):
            runner.launch(tmp_path)


def test_process_id_reuse_does_not_claim_an_unrelated_application(tmp_path):
    path = tmp_path / "apps.json"
    common.write(path, [{"pid": 12, "created": 100}])
    with patch.object(common, "process_identity", return_value={"pid": 12, "alive": True, "created": 200}):
        assert common.owned_apps_remaining(path) == []
    with patch.object(common, "process_identity", return_value={"pid": 12, "alive": True, "created": 100}):
        assert len(common.owned_apps_remaining(path)) == 1


def test_new_dispatch_records_owned_pid_before_document_work(tmp_path):
    app = types.SimpleNamespace(processId=123, Quit=lambda: None)
    with patch("win32com.client.DispatchEx", return_value=app), \
         patch.object(common, "process_identity", return_value={"pid": 123, "alive": True, "created": 456}):
        with common.record_owned_apps(tmp_path):
            import win32com.client
            assert win32com.client.DispatchEx("AD application") is app
            assert common.read(tmp_path / "owned_aspen_apps.json")[0]["created"] == 456


def test_failed_com_creation_is_not_misreported_as_verified_cleanup(tmp_path):
    with patch("win32com.client.DispatchEx", side_effect=RuntimeError("COM startup failed")):
        with common.record_owned_apps(tmp_path):
            import win32com.client
            with pytest.raises(RuntimeError, match="startup"):
                win32com.client.DispatchEx("AD application")
    remaining = common.owned_apps_remaining(tmp_path / "owned_aspen_apps.json")
    assert remaining[0]["ownership_unverified"] is True


def test_completed_manifest_label_cannot_skip_missing_result_files(tmp_path):
    for method in common.MAPPING:
        (tmp_path / method).mkdir()
        common.write(tmp_path / method / "job.json", {})
    common.write(tmp_path / "sg/execution.json", {"state": "completed"})
    common.write(tmp_path / "sg/attempts.json", [{"directory": str(tmp_path / "sg/a1")}])
    with patch.object(runner, "check_job"), patch.object(runner, "completed", return_value=False), \
         patch("subprocess.Popen", side_effect=AssertionError("Do not launch around invalid completed evidence")):
        with pytest.raises(RuntimeError, match="artifacts"):
            runner.launch(tmp_path)


def test_pilot_completed_contract_requires_the_exported_native_snapshot(tmp_path):
    for name in ("pilot_result.json", "continuations.npz", "python_checkpoint.pkl", "snapshot_api.json"):
        (tmp_path / name).write_text("{}")
    common.write(tmp_path / "owned_aspen_apps.json", [{"pid": 12, "created": 1}])
    common.write(tmp_path / "pilot_result.json", {"pilot_passed": True, "e1_critic_validation_performed": False,
        "native_snapshot": "preserved_snapshots/test.snp", "native_snapshot_sha256": "missing"})
    common.write(tmp_path / "status.json", {"state": "completed", "aspen_closed": True,
        "files": {name: common.digest(tmp_path / name) for name in
                  ("pilot_result.json", "continuations.npz", "python_checkpoint.pkl", "snapshot_api.json")}})
    with patch.object(runner, "owned_apps_remaining", return_value=[]):
        assert not runner.completed(tmp_path, {"method": "e1_pilot"})


class Stage:
    def __init__(self, plant, index):
        self.plant, self.index = plant, index
    @property
    def T(self):
        return types.SimpleNamespace(Value=self.plant.state[0] + .01 * self.index + .1 * self.plant.state[2])
    def y(self, name):
        return types.SimpleNamespace(Value=self.plant.state[1] + 1e-5 * self.index + 1e-5 * self.plant.state[2])


class Snapshot:
    def __init__(self, plant):
        self.state, self.time, self.feed = plant.state.copy(), plant.sim.Time, plant.feed.FmR.Value
        self.inputs = np.array([plant.col.Reflux.FmR.Value, plant.col.QRebR.Value])
        self.Converged = True
    def ExportasBinary(self, path):
        Path(path).write_bytes(b"native snapshot fixture")


class Results:
    def __init__(self, plant):
        self.plant = plant
    def TakeSnapshot(self, name):
        self.snapshot = Snapshot(self.plant)
    def FindSnapshot(self, name):
        return self.snapshot
    def LoadSnapshot(self, snapshot):
        self.plant.state = snapshot.state.copy()
        self.plant.sim.Time = snapshot.time
        self.plant.feed.FmR.Value = snapshot.feed
        self.plant.col.Reflux.FmR.Value, self.plant.col.QRebR.Value = snapshot.inputs


class Plant:
    def __init__(self):
        self.state = np.array([-23., .01, .2])
        self.sim = types.SimpleNamespace(Time=40.)
        self.sim.Results = Results(self)
        self.feed = types.SimpleNamespace(FmR=types.SimpleNamespace(Value=1e5))
        self.col = types.SimpleNamespace(Stage=lambda index: Stage(self, index))
        self.col.Reflux = types.SimpleNamespace(FmR=types.SimpleNamespace(Value=3e5))
        self.col.QRebR = types.SimpleNamespace(Value=120.)
        self.delta_t = 1 / 6
        self.current_input = np.array([3e5, 120.])
        self.current_output = np.array([self.col.Stage(24).y("C2H6").Value, self.col.Stage(85).T.Value])
    def step(self, disturbances=None):
        self.col.Reflux.FmR.Value, self.col.QRebR.Value = self.current_input
        self.state += np.array([.01, 1e-6, .001])
        self.sim.Time += 1 / 6
        self.feed.FmR.Value = float(disturbances[0])
        self.current_output = np.array([self.col.Stage(24).y("C2H6").Value, self.col.Stage(85).T.Value])


def test_native_snapshot_restores_internal_profiles_and_repeated_continuations(tmp_path):
    plant = Plant()
    state = {"observer": np.array([.2, .4]), "held_action": np.array([0., 0.])}
    snapshot = NativeSnapshot(plant, state, tmp_path)
    state["observer"][:] = 999
    runs = paired_continuations(plant, snapshot, np.ones((24, 2)), np.full(24, 1.1e5))
    np.testing.assert_array_equal(runs[0]["outputs"], runs[2]["outputs"])
    np.testing.assert_array_equal(runs[1]["controller"]["observer"], [.2, .4])
    assert snapshot.path.exists()


def test_matching_only_two_cached_outputs_cannot_pass_internal_restore_check(tmp_path):
    plant = Plant()
    snapshot = NativeSnapshot(plant, {"observer": np.zeros(2)}, tmp_path)
    plant.state[2] = 10.
    # Deliberately broken native restore leaves the hidden state unchanged.
    with patch.object(plant.sim.Results, "LoadSnapshot", lambda native: None):
        with pytest.raises(RuntimeError, match="internal-profile"):
            snapshot.restore()


def test_pilot_rejects_a_step_that_never_advances_simulation_time(tmp_path):
    plant = Plant()
    snapshot = NativeSnapshot(plant, {"observer": np.zeros(2)}, tmp_path)
    with patch.object(plant, "step", lambda **kwargs: None):
        with pytest.raises(RuntimeError, match="sampling interval"):
            paired_continuations(plant, snapshot, np.ones((2, 2)), np.ones(2))


def test_standalone_sg_creates_baseline_audits_before_training(tmp_path):
    from revision_comparisons.column import run_method
    events = []
    def prepare(seed, output, aspen):
        events.append("audit")
        for name in ("baseline_audit.json", "baseline_metrics.json"):
            (output / name).write_text("{}")
    @contextmanager
    def install(category, factory):
        session = factory(object(), (), {})
        session.agents = {"horizon_agent": object()}
        yield
    def execute(*args):
        events.append("train")
        return {"result_bundle": {}, "NB": {}, "reward_params": {"k_rel": [.1, .1], "band_floor_phys": [.01, 1.]}}
    with patch("review_experiments.e3.runtime.aspen_resources", lambda paths: nullcontext()), \
         patch("revision_comparisons.column.prepare", prepare), \
         patch("revision_comparisons.column.timed_columns", lambda clock: nullcontext()), \
         patch("review_experiments.e1.runtime.install_runner", install), \
         patch("review_experiments.e3.runtime.execute_entrypoint", execute):
        result, _ = run_method("sg", tmp_path, {}, 7)
    assert events == ["audit", "train"]
    assert result["reward_params"]["band_floor_phys"] == [.01, 1.]


def test_original_critic_only_phase_boundaries_are_preserved():
    from utils.phase1_hidden_release import build_phase1_schedule
    schedule = build_phase1_schedule(agent_kind="sg_td3", warm_start_step=4000, time_in_sub_episodes=400,
        n_steps=80000, test_train_dict={0: False, 79600: True}, action_freeze_subepisodes=3,
        actor_freeze_subepisodes=3, batch_size=256, train_start_step=4000)
    assert schedule["action_freeze_end_step"] == 5200
    assert schedule["actor_freeze_end_step"] == 5200
    assert schedule["first_live_action_step"] == 5201
    assert schedule["hidden_window_active_log"][5199] == 1
    assert schedule["hidden_window_active_log"][5201] == 0
