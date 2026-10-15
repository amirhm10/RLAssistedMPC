from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_e3_prediction_quality as cli
from review_experiments.e3.capture import CaptureOptions, CaptureSession, sampling_schedule
from review_experiments.e3.common import PREDICTORS, ROOT, WINDOWS, sha256, source_hashes, write_json
from review_experiments.e3.nominal import evaluate_nominal, monitor_nominal, nominal_origins, save_nominal
from review_experiments.e3.prediction import forecasts, observer_history
from review_experiments.e3.runtime import configure_defaults, execute_entrypoint, passive_capture
from review_experiments.e1.smoke import compressed_smoke_clock


@pytest.fixture(autouse=True)
def prohibit_live_aspen(monkeypatch):
    import systems.distillation.plant as plant
    if plant.win32com is not None:
        def blocked(*args, **kwargs):
            raise AssertionError("E3 technical tests must never create a live Aspen COM application.")
        monkeypatch.setattr(plant.win32com.client, "DispatchEx", blocked)


def oracle_model():
    return {"A": np.diag([.8, .6]), "B": np.array([[1., .2], [.1, .7]]), "C": np.eye(2),
            "L": np.diag([.1, .15]), "data_min": np.zeros(4), "data_max": np.ones(4),
            "ss_scaled_inputs": np.zeros(2), "y_ss_scaled": np.zeros(2), "basis_family": "io_pair_gain",
            "observer_mode": "legacy_previous_measurement", "observer_float32": False,
            "corrections": np.array([[0.] * 4, [.1] * 4, [.2] * 4, [.2] * 4])}


def test_forecasts_against_independent_nonzero_state_changing_input_oracle():
    model = oracle_model()
    initial = np.array([.4, -.3])
    u = np.array([[.1 * (k % 3), -.03 * k] for k in range(8)])
    state, truth = initial.copy(), []
    for move in u:
        state = model["A"] @ state + 1.2 * model["B"] @ move
        truth.append(model["C"] @ state)
    predicted, failed = forecasts(model, initial, u)
    assert not failed
    np.testing.assert_allclose(predicted["rl_proposed"], truth, atol=1e-12)
    assert np.linalg.norm(predicted["nominal"] - truth) > 0
    # Suffix inputs change only corresponding future leads.
    altered = u.copy()
    altered[4:] += 10
    changed, _ = forecasts(model, initial, altered)
    np.testing.assert_allclose(changed["executed"][:4], predicted["executed"][:4])
    assert not np.allclose(changed["executed"][4:], predicted["executed"][4:])


@pytest.mark.parametrize("mode", ["legacy_previous_measurement", "current_measurement_corrector", "predictor_corrector_current"])
def test_nominal_observer_is_causal_and_uses_captured_convention(mode):
    model = oracle_model()
    model["observer_mode"] = mode
    u, y = np.ones((10, 2)) * .1, np.arange(22).reshape(11, 2) * .01
    states = observer_history(model, y, u)
    changed = y.copy()
    changed[6:] += 100
    revised = observer_history(model, changed, u)
    np.testing.assert_array_equal(states[:6], revised[:6])
    if mode == "legacy_previous_measurement":
        expected = model["B"] @ u[0] + model["L"] @ y[0]
    else:
        prediction = model["B"] @ u[0]
        expected = prediction + model["L"] @ (y[1] - prediction)
    np.testing.assert_allclose(states[1], expected)


def test_invalid_proposed_predictor_is_retained_without_hiding_nominal_result():
    model = oracle_model()
    model["corrections"][2] = np.nan
    values, failures = forecasts(model, np.zeros(2), np.zeros((8, 2)))
    assert set(failures) == {"rl_proposed"}
    assert np.isnan(values["rl_proposed"]).all()
    assert np.isfinite(values["nominal"]).all()


def test_sampling_has_complete_futures_no_overlap_and_no_training_rng_consumption():
    options = CaptureOptions()
    state = np.random.get_state()
    schedule = sampling_schedule(400, 120000, options, "distillation/combined")
    repeated = sampling_schedule(400, 120000, options, "distillation/combined")
    assert schedule == repeated and len(schedule) == 400
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    for window, first, last in WINDOWS:
        origins = np.array([k for k, name in schedule.items() if name == window])
        assert len(origins) == 100 and np.min(np.diff(origins)) >= 8
        assert origins[0] >= (first - 1) * 400 and origins[-1] + 8 <= last * 400
    with pytest.raises(ValueError, match="Insufficient"):
        sampling_schedule(4, 1200, CaptureOptions(samples_per_window=100), "test")


def test_dry_run_is_standard_library_only_and_creates_nothing(tmp_path):
    code = ("import sys; import run_e3_prediction_quality as c; "
            f"c.main(['--dry-run','--output-dir',{str(tmp_path / 'unused')!r}]); "
            "assert not any(n in sys.modules for n in ('numpy','scipy','torch','win32com.client'))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "4 SG training runs; 2 shared nominal probes" in result.stdout
    assert not (tmp_path / "unused").exists()


def test_aspen_number_dry_run_is_standard_library_only_and_creates_nothing(tmp_path):
    code = ("import sys; import run_e3_prediction_quality as c; "
            f"c.main(['--dry-run','--plants','distillation','--aspen-number','11','--output-dir',{str(tmp_path / 'unused')!r}]); "
            "assert not any(n in sys.modules for n in ('numpy','scipy','torch','win32com.client'))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "2 SG training runs; 1 shared nominal probes" in result.stdout
    assert "All distillation stages use C2S_SS_simulation11.dynf sequentially." in result.stdout
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("number", range(11, 17))
def test_aspen_number_preflight_selects_all_stage_paths_and_matching_snapshots(number, tmp_path):
    model = tmp_path / f"C2S_SS_simulation{number}.dynf"
    model.write_bytes(b"Fake Aspen model for path preflight only; never opened by COM.")
    snapshots = tmp_path / f"AM_C2S_SS_simulation{number}"
    snapshots.mkdir()
    jobs = cli.preflight(sys.executable, cli.build_jobs(["distillation"], ["markov", "combined"], 7),
                         tmp_path, number)
    from systems.distillation import get_distillation_notebook_defaults
    for job in jobs:
        paths = job["aspen"]
        assert paths == {"aspen_path": str(model), "snaps_path": str(snapshots),
                         "path_source": "e3-number-override", "aspen_sha256": sha256(model)}
        family = "baseline" if job["category"] == "nominal" else job["category"]
        configured = configure_defaults(get_distillation_notebook_defaults(family), "distillation",
                                        job["category"], 7, tmp_path, paths)
        assert configured["aspen_path_override"] == str(model)
        assert configured["snaps_path_override"] == str(snapshots)


def test_resume_cannot_override_saved_aspen_number(tmp_path):
    with pytest.raises(ValueError, match="Resume uses its saved options"):
        cli.main(["--resume", str(tmp_path), "--aspen-number", "12"])


@pytest.mark.parametrize("arguments", [["--seed", "-1"], ["--samples-per-window", "999999"],
                                       ["--prediction-horizons", "8", "1"], ["--smoke", "--plants", "distillation"],
                                       ["--aspen-number", "0"], ["--aspen-number", "-1"],
                                       ["--plants", "polymer", "--aspen-number", "11"]])
def test_invalid_budgets_fail_before_outputs(arguments, tmp_path):
    with pytest.raises(ValueError):
        cli.main([*arguments, "--output-dir", str(tmp_path / "unused")])
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("plant", ["polymer", "distillation"])
@pytest.mark.parametrize("category", ["markov", "combined"])
def test_full_configuration_preserves_science_except_mode_seed_output(plant, category, tmp_path):
    import importlib
    provider = getattr(importlib.import_module(f"systems.{plant}"), f"get_{plant}_notebook_defaults")
    original = provider(category)
    configured = configure_defaults(original, plant, category, 7, tmp_path)
    allowed = {"run_mode", "disturbance_profile", "training_profile_override", "results_dir_override",
               "result_prefix_override", "compare_prefix_override", "agent_kind", "agent_mode", "combined_agent_mode"}
    from review_experiments.e3.common import SEED_KEYS
    for key in SEED_KEYS:
        if isinstance(configured.get(key), dict) and isinstance(original.get(key), dict):
            if "seed" in original[key]:
                configured[key]["seed"] = original[key]["seed"]
            else:
                configured[key].pop("seed", None)
    for key in original:
        if key not in allowed:
            assert pickle.dumps(original[key]) == pickle.dumps(configured[key]), key


class FakeColumn:
    """Deterministic physical-coordinate contract fixture; never opens COM."""
    def __init__(self, path, ss_inputs, initialization_point, delta_t, visible=True):
        self.path, self.delta_t = path, delta_t
        self.ss_inputs = np.asarray(ss_inputs).copy()
        self.current_input = self.ss_inputs.copy()
        self.y_ss = np.array([.02, -22.])
        self.current_output = self.y_ss.copy()
        self.feed = SimpleNamespace(FmR=SimpleNamespace(Value=float(initialization_point[0])))
        self.closed = False

    def step(self, disturbances=None):
        du = self.current_input - self.ss_inputs
        target = self.y_ss + np.array([du[0] * 1e-7 - du[1] * .0001, du[0] * 1e-6 + du[1] * .02])
        self.current_output = .85 * self.current_output + .15 * target
        if disturbances is not None:
            self.feed.FmR.Value = float(np.asarray(disturbances).reshape(-1)[0])

    def close(self, *args, **kwargs):
        self.closed = True


def small_agents(defaults):
    from review_experiments.e3.common import SEED_KEYS
    configured = deepcopy(defaults)
    for key in SEED_KEYS:
        if isinstance(configured.get(key), dict):
            if "batch_size" in configured[key]:
                configured[key]["batch_size"] = 8
            for field in ("hidden_layers", "actor_hidden", "critic_hidden"):
                if field in configured[key]:
                    configured[key][field] = [16, 16]
    return configured


def small_configuration(defaults, plant, category, output):
    configured = small_agents(configure_defaults(defaults, plant, category, 7, output))
    if category == "nominal":
        configured["set_points_len_override"] = 16
    return configured


_SMALL_CACHE = {}


def run_small(plant, category, output, capture, monkeypatch, *, reuse=False):
    cache_key = (plant, category, capture)
    if reuse and cache_key in _SMALL_CACHE:
        return _SMALL_CACHE[cache_key]
    import systems.distillation.plant as distplant
    from utils import markov_runner
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
    torch.set_num_threads(1)
    created = []
    original_factory = markov_runner.make_td3_markov_agent

    def make_agent(*args, **kwargs):
        agent = original_factory(*args, **kwargs)
        created.append(agent)
        return agent

    session = CaptureSession(output / "capture", plant, category, CaptureOptions(samples_per_window=1, smoke=True))
    with monkeypatch.context() as patch:
        # Tiny test networks run faster on CPU and do not compete with the
        # author's production CUDA jobs. Both sides use the same test device.
        patch.setattr(torch.cuda, "is_available", lambda: False)
        # The production distillation entrypoint checks combined settings
        # against all standalone defaults. Reduce those test copies equally.
        import importlib
        system = importlib.import_module(f"systems.{plant}")
        provider_name = f"get_{plant}_notebook_defaults"
        original_provider = getattr(system, provider_name)
        patch.setattr(system, provider_name, lambda family: small_agents(original_provider(family)))
        patch.setattr(markov_runner, "make_td3_markov_agent", make_agent)
        if plant == "distillation":
            fake_factory = lambda **kwargs: FakeColumn(**kwargs)
            patch.setattr(distplant, "build_distillation_system", fake_factory)
            patch.setattr(system, "build_distillation_system", fake_factory)
        with compressed_smoke_clock(category, stride=200 if plant == "polymer" else 100):
            with passive_capture(category, session) if capture else nullcontext():
                namespace = execute_entrypoint(plant, category, lambda defaults: small_configuration(defaults, plant, category, output))
    result = namespace["result_bundle"]
    evidence = {key: value.copy() for key, value in result.items() if isinstance(value, np.ndarray)}
    agents = created + list(namespace.get("agents", {}).values())
    for index, agent in enumerate(agents):
        evidence[f"agent_{index}_steps"] = np.array([agent.steps, agent.train_steps, agent.buffer.size, agent.buffer.ptr])
        for field in ("states", "actions", "rewards", "next_states"):
            evidence[f"replay_{index}_{field}"] = getattr(agent.buffer, field)[:agent.buffer.size].copy()
        for name, module in vars(agent).items():
            if isinstance(module, torch.nn.Module):
                for key, tensor in module.state_dict().items():
                    evidence[f"network_{index}_{name}_{key}"] = tensor.detach().cpu().numpy().copy()
    evidence["rng_numpy"] = np.random.get_state()[1].copy()
    evidence["rng_torch"] = torch.random.get_rng_state().numpy().copy()
    evidence["rng_python"] = np.asarray(random.getstate()[1])
    result = (evidence, session, namespace)
    if reuse:
        _SMALL_CACHE[cache_key] = result
    return result


@pytest.mark.parametrize("plant", ["polymer", "distillation"])
@pytest.mark.parametrize("category", ["markov", "combined"])
def test_passive_capture_preserves_training_replay_networks_and_rng(plant, category, tmp_path, monkeypatch):
    reference, _, _ = run_small(plant, category, tmp_path / "reference", False, monkeypatch, reuse=True)
    captured, session, _ = run_small(plant, category, tmp_path / "captured", True, monkeypatch, reuse=True)
    assert set(reference) == set(captured)
    for key in reference:
        np.testing.assert_allclose(reference[key], captured[key], rtol=1e-8, atol=1e-9,
                                   equal_nan=True, err_msg=f"Training changed: {plant}/{category}/{key}")
    assert len(session.records) == 4
    assert not any(row["predictor_failures"] for row in session.records)
    for row in session.records:
        with np.load(session.output / row["trace"]) as arrays:
            assert arrays["u_future"].shape == arrays["y_future"].shape == (8, 2)
            assert arrays["forecast_executed"].shape == (8, 2)
            if plant == "distillation":
                # Diagnostics extend beyond its standalone six-step horizon.
                assert len(arrays["forecast_nominal"]) > 6


@pytest.mark.parametrize("plant", ["polymer", "distillation"])
def test_shared_nominal_probe_and_checkpoint_analysis(plant, tmp_path, monkeypatch):
    from review_experiments.e3.analysis import summarize
    import systems.distillation.plant as distplant
    nominal_dir = tmp_path / "nominal"
    nominal_dir.mkdir()
    with monkeypatch.context() as patch:
        if plant == "distillation":
            patch.setattr(distplant, "build_distillation_system", lambda **kwargs: FakeColumn(**kwargs))
        with monitor_nominal() as diagnostics:
            namespace = execute_entrypoint(plant, "nominal", lambda defaults: small_configuration(defaults, plant, "nominal", nominal_dir))
    nominal_file = save_nominal(namespace, nominal_dir, diagnostics, smoke=True)
    inputs, hashes = [], set()
    for category in ("markov", "combined"):
        _, session, _ = run_small(plant, category, tmp_path / category, True, monkeypatch, reuse=True)
        checkpoint_dir = tmp_path / f"{category}_nominal"
        records = evaluate_nominal(session.output, nominal_file, checkpoint_dir, (1, 4, 8), samples=2)
        hashes.update(r["nominal_dataset_sha256"] for r in records)
        inputs.extend([session.output, checkpoint_dir])
    assert hashes == {sha256(nominal_file)}
    rows = summarize(inputs, tmp_path / "analysis", plots=True)
    assert len(rows) == 2 * 6 * 4 * 3 * 2  # configurations, conditions, predictors, leads, outputs
    assert all(row["total_origins"] == row["valid_origins"] for row in rows)
    assert list((tmp_path / "analysis").glob("*.png"))


def test_nominal_origins_exclude_initialization_and_stay_inside_each_hold():
    origins = nominal_origins(400, 8)
    assert len(origins) == 50 and min(origins) >= 400 and max(origins) + 8 <= 800
    assert min(np.diff(origins)) >= 8
    assert all(origin + 8 <= (600 if origin < 600 else 800) for origin in origins)


def test_lead_metrics_do_not_average_prefix_and_keep_failed_denominators(tmp_path):
    from review_experiments.e3.analysis import summarize
    directory = tmp_path / "capture"
    directory.mkdir()
    truth = np.zeros((8, 2))
    arrays = {f"forecast_{p}": np.arange(1, 9)[:, None] * np.ones((8, 2)) for p in PREDICTORS}
    arrays["forecast_rl_proposed"] = np.full((8, 2), np.nan)
    np.savez(directory / "trace.npz", y_future=truth, data_min=np.zeros(4), data_max=np.ones(4), **arrays)
    record = {"plant": "polymer", "category": "markov", "dataset": "training", "window": "adapted",
              "smoke": True, "step": 1, "executed_source_code": 7, "trace": "trace.npz"}
    (directory / "records.jsonl").write_text(json.dumps(record) + "\n")
    rows = summarize([directory], tmp_path / "summary", plots=False)
    lead8 = next(r for r in rows if r["horizon"] == 8 and r["predictor"] == "nominal")
    assert lead8["rmse_physical"] == 8.
    failed = next(r for r in rows if r["predictor"] == "rl_proposed")
    assert failed["total_origins"] == 1 and failed["valid_origins"] == 0 and failed["rmse_physical"] is None


@pytest.mark.parametrize("partial_failure", [False, True])
def test_aspen_context_owns_cleanup_and_restores_exports(tmp_path, monkeypatch, partial_failure):
    from review_experiments.e3.runtime import aspen_resources
    import run_distillation_phase2_repeats as repeats
    import systems.distillation as distillation
    import systems.distillation.plant as plant
    instances = []

    class Owned:
        def __init__(self):
            self.ad = object()
            self.close_calls = 0
            instances.append(self)
            if partial_failure:
                raise ValueError("partial construction")

        def close(self, snaps_path=None, prefix="snp"):
            self.close_calls += 1

    monkeypatch.setattr(plant, "DistillationColumnAspen", Owned)
    monkeypatch.setattr(distillation, "DistillationColumnAspen", Owned)
    monkeypatch.setattr(repeats, "aspen_com_compatibility", nullcontext)
    paths = {"aspen_path": str(tmp_path / "fake.dynf"), "snaps_path": str(tmp_path / "fake_snaps")}
    if partial_failure:
        with pytest.raises(ValueError, match="partial construction"):
            with aspen_resources(paths):
                plant.DistillationColumnAspen()
    else:
        with aspen_resources(paths):
            column = plant.DistillationColumnAspen()
            column.close()
            column.close()
    assert instances[0].close_calls == 1
    assert plant.DistillationColumnAspen is Owned and distillation.DistillationColumnAspen is Owned


def test_aspen_lock_prevents_second_owner(tmp_path):
    from run_distillation_phase2_repeats import aspen_file_lock
    with aspen_file_lock(tmp_path / "model.dynf", tmp_path / "locks"):
        with pytest.raises(RuntimeError, match="reserved"):
            with aspen_file_lock(tmp_path / "model.dynf", tmp_path / "locks"):
                pytest.fail("A second owner acquired a live model lock.")


def test_parent_preserves_failed_attempts_and_resumes_only_missing_stage(tmp_path):
    import textwrap
    helper = tmp_path / "fake_worker.py"
    helper.write_text(textwrap.dedent('''
        import hashlib,json,os,pathlib,sys
        job_path=pathlib.Path(sys.argv[1]); job=json.loads(job_path.read_text()); out=job_path.parent
        if job['category']=='markov' and out.name=='a1':
            (out/'status.json').write_text(json.dumps({'state':'running','pid':os.getpid()}))
            sys.exit(3)
        result=out/('nominal.npz' if job['category']=='nominal' else 'training.pkl')
        result.write_bytes(b'verified fake artifact')
        digest=hashlib.sha256(result.read_bytes()).hexdigest()
        status={'state':'completed','artifact':str(result),'artifact_sha256':digest,'files':{result.name:digest}}
        (out/'status.json').write_text(json.dumps(status))
    '''))
    study = tmp_path / "study"
    study.mkdir()
    manifest = {"python": sys.executable, "options": {"threads": 1}, "source_hashes": source_hashes(),
                "jobs": cli.build_jobs(["polymer"], ["markov"], 7)}
    calls = []

    def launch(command, **kwargs):
        calls.append(command[-1])
        return subprocess.Popen([sys.executable, str(helper), command[-1]], **kwargs)

    with pytest.raises(RuntimeError, match="stage failed"):
        cli.run_jobs(manifest, study, launch=launch)
    assert len(calls) == 2 and manifest["jobs"][0]["state"] == "completed"
    failed = Path(manifest["jobs"][1]["attempts"][0]["directory"])
    cli.run_jobs(manifest, study, launch=launch)
    assert len(calls) == 3 and all(cli.completed_job(j) for j in manifest["jobs"])
    assert len(manifest["jobs"][0]["attempts"]) == 1
    assert len(manifest["jobs"][1]["attempts"]) == 2
    assert (failed / "status.json").is_file()
    artifact = Path(manifest["jobs"][1]["attempts"][-1]["directory"]) / "training.pkl"
    artifact.write_bytes(b'corrupted')
    assert not cli.completed_job(manifest["jobs"][1])


def test_resume_rejects_live_worker_without_starting_duplicate(tmp_path):
    import os
    job = cli.build_jobs(["polymer"], ["markov"], 7)[0]
    job["attempts"] = [{"directory": str(tmp_path), "state": "running", "pid": os.getpid()}]
    with pytest.raises(RuntimeError, match="still alive"):
        cli.reject_live_attempts([job])


def test_incomplete_capture_is_explicit_and_stop_request_unwinds(tmp_path):
    session = CaptureSession(tmp_path / "capture", "polymer", "markov", CaptureOptions(samples_per_window=1))
    with pytest.raises(RuntimeError, match="incomplete"):
        session.finish()
    status = json.loads((session.output / "capture_status.json").read_text())
    assert status["state"] == "incomplete" and status["expected"] == 4
    stop = tmp_path / "stop"
    stop.write_text("stop")
    session = CaptureSession(tmp_path / "second", "polymer", "markov", CaptureOptions(), stop_file=stop)
    with pytest.raises(KeyboardInterrupt, match="graceful"):
        session.before({})


@pytest.mark.parametrize("plant", ["polymer", "distillation"])
def test_nominal_worker_writes_hashed_reusable_artifacts(plant, tmp_path, monkeypatch):
    import systems.distillation.plant as distplant
    import review_experiments.e3.runtime as runtime
    if plant == "distillation":
        monkeypatch.setattr(distplant, "build_distillation_system", lambda **kwargs: FakeColumn(**kwargs))
        monkeypatch.setattr(runtime, "aspen_resources", lambda paths: nullcontext())
    job = {"plant": plant, "category": "nominal", "seed": 7, "source_hashes": source_hashes(),
           "options": {"threads": 1, "smoke": True, "horizons": [1, 4, 8], "samples_per_window": 1},
           "aspen": {"aspen_path": str(tmp_path / "fake.dynf"), "snaps_path": str(tmp_path / "snapshots")}}
    write_json(tmp_path / "job.json", job)
    assert cli.run_worker(tmp_path / "job.json") == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "completed" and status["files"]
    metadata = json.loads((tmp_path / "runtime_metadata.json").read_text())
    assert metadata["delta_t_hours"] == (.5 if plant == "polymer" else 1 / 6)
    parent_job = {"plant": plant, "state": "completed", "attempts": [{"directory": str(tmp_path)}]}
    assert cli.completed_job(parent_job)


def test_analysis_runs_with_saved_interpreter_and_reuses_raw_evidence(tmp_path):
    """A real analysis subprocess over a tiny synthetic saved study; no training."""
    jobs = cli.build_jobs(["polymer"], ["markov"], 7)
    model = oracle_model()
    raw_hashes = {}
    for job in jobs:
        directory = tmp_path / job["id"]
        directory.mkdir()
        if job["category"] == "nominal":
            artifact = directory / "nominal.npz"
            np.savez(artifact, y=np.zeros((65, 2)), u=np.zeros((64, 2)), cycle_steps=32,
                     A_aug=model["A"], B_aug=model["B"], C_aug=model["C"],
                     data_min=model["data_min"], data_max=model["data_max"])
            write_json(directory / "nominal_metadata.json", {"state": "completed", "sha256": sha256(artifact),
                                                              "solver_failures": 0, "smoke": True})
        else:
            artifact = directory / "training.pkl"
            with artifact.open("wb") as stream:
                pickle.dump({"avg_rewards": np.zeros(300)}, stream)
            capture = directory / "capture"
            capture.mkdir()
            records = []
            for index, (window, _, _) in enumerate(WINDOWS):
                path = capture / f"{window}.npz"
                np.savez(path, **model, x0=np.zeros(2), y_future=np.zeros((8, 2)),
                         **{f"forecast_{p}": np.zeros((8, 2)) for p in PREDICTORS})
                records.append({"plant": "polymer", "category": "markov", "dataset": "training",
                                "seed": 7, "smoke": True, "window": window, "step": index,
                                "executed_source_code": 7, "gate_source_code": 1, "trace": path.name})
            (capture / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            write_json(capture / "capture_status.json", {"state": "completed", "checkpoints": {
                "prechange_mature": "prechange_mature.npz", "adapted": "adapted.npz"}})
        files = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*") if p.is_file()}
        write_json(directory / "status.json", {"state": "completed", "artifact": str(artifact),
                                                "artifact_sha256": sha256(artifact), "files": files})
        job.update(state="completed", attempts=[{"directory": str(directory), "state": "completed"}])
        raw_hashes.update({str(directory / name): digest for name, digest in files.items()})
    write_json(tmp_path / "manifest.json", {"python": sys.executable, "jobs": jobs,
                                            "options": {"threads": 1, "smoke": True, "horizons": [1, 4, 8]}})
    first = cli.run_analysis(tmp_path)
    assert (first / "prediction" / "prediction_errors.csv").is_file()
    second = cli.run_analysis(tmp_path)
    assert first != second
    assert all(sha256(path) == digest for path, digest in raw_hashes.items())
