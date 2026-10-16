"""Revision comparator contracts. All tests run offline without an Aspen session."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import pickle
import types
from unittest.mock import patch

import numpy as np
import pytest

from comparison_studies.arx import IncrementalARX, nominal_coefficients
from comparison_studies.common import jsonable
from revision_comparisons.arx import RegularizedARX, dc_gains
from revision_comparisons.column import ROOT, baseline, calibration_window, configure, metrics, run_method


@pytest.fixture
def polymer_theta():
    with (ROOT/"Polymer/Data/system_dict").open("rb") as stream:
        data = pickle.load(stream)
    return nominal_coefficients(data["A"], data["B"], data["C"])


def test_stable_but_reversed_temperature_gain_is_rejected(polymer_theta):
    model = RegularizedARX(polymer_theta, np.zeros(2), np.zeros(2))
    candidate = polymer_theta[1].copy()
    candidate[4:] *= -1
    assert IncrementalARX.valid(candidate)  # historical admission rule misses the defect
    assert dc_gains(candidate, 1)[1] < 0 < dc_gains(polymer_theta[1], 1)[1]
    assert model.admission_reason(candidate, 1) == "reversed_gain_direction"


def test_stable_but_excessive_gain_is_rejected(polymer_theta):
    model = RegularizedARX(polymer_theta, np.zeros(2), np.zeros(2))
    candidate = polymer_theta[1].copy()
    candidate[2:] *= 5
    assert IncrementalARX.valid(candidate)
    assert model.admission_reason(candidate, 1) == "dc_gain_outside_nominal_bounds"


def test_persistent_prior_survives_weak_closed_loop_excitation(polymer_theta):
    model = RegularizedARX(polymer_theta, np.zeros(2), np.zeros(2))
    rng = np.random.default_rng(7)
    for _ in range(1600):
        proposed = rng.uniform(-1e-4, 1e-4, 2)
        y = model.predict(proposed[None], 1)[0]
        model.observe(y, proposed)
    for h, h0 in zip(model.information, model.prior_information):
        assert np.linalg.eigvalsh(h-h0).min() >= -1e-10
    for i, theta in enumerate(model.theta):
        assert model.admission_reason(theta, i) == "accepted"


def test_rejected_model_does_not_replace_the_controller_predictor(polymer_theta):
    model = RegularizedARX(polymer_theta, np.zeros(2), np.zeros(2))
    before = deepcopy(model.theta)
    with patch.object(model, "admission_reason", return_value="reversed_gain_direction"):
        for _ in range(8):
            model.observe(model.current_y+np.array([.01, -.01]), model.current_u+np.array([.02, .01]))
    for a, b in zip(before, model.theta):
        np.testing.assert_array_equal(a, b)
    assert model.rejections["reversed_gain_direction"] > 0


def test_regularized_rls_learns_a_known_excited_mimo_model():
    theta = [np.array([.4, .1, .2, .1, -.08, .3, -.1]),
             np.array([.3, .15, -.2, .04, .25, .09])]
    true = IncrementalARX(theta, np.zeros(2), np.zeros(2), adaptive=False)
    initial = [np.r_[t[:2]*.9, t[2:]*.5] for t in theta]
    fitted = RegularizedARX(initial, np.zeros(2), np.zeros(2))
    rng = np.random.default_rng(12)
    for _ in range(2000):
        proposed = rng.uniform(-.5, .5, 2)
        y = true.predict(proposed[None], 1)[0]
        true.observe(y, proposed)
        fitted.observe(y, proposed)
    # With forgetting, the persistent ridge intentionally leaves a finite
    # nominal bias. Test identification improvement and held-out predictions,
    # rather than incorrectly asserting asymptotically unbiased coefficients.
    for expected, actual, start in zip(theta, fitted.theta, initial):
        assert np.linalg.norm(actual-expected) < .05*np.linalg.norm(start-expected)
    fitted.adaptive = False
    errors, increments = [], []
    for _ in range(400):
        proposed = rng.uniform(-.5, .5, 2)
        y = true.predict(proposed[None], 1)[0]
        errors.append(fitted.predict(proposed[None], 1)[0]-y)
        increments.append(y-true.current_y)
        true.observe(y, proposed)
        fitted.observe(y, proposed)
    nrmse = np.sqrt(np.mean(np.asarray(errors)**2, axis=0)/np.mean(np.asarray(increments)**2, axis=0))
    assert np.all(nrmse < .01)


def test_nominal_prediction_and_history_alignment_are_preserved(polymer_theta):
    with (ROOT/"Polymer/Data/system_dict").open("rb") as stream:
        data = pickle.load(stream)
    A, B, C = (np.asarray(data[k]) for k in ("A", "B", "C"))
    model = RegularizedARX(polymer_theta, np.zeros(2), np.zeros(2), adaptive=False)
    x = np.zeros(A.shape[0])
    inputs = np.random.default_rng(11).uniform(-.1, .1, (100, 2))
    for k in range(90):
        xx, expected = x.copy(), []
        for u in inputs[k:k+10]:
            xx = A@xx+B@u
            expected.append(C@xx)
        np.testing.assert_allclose(model.predict(inputs[k:k+10], 10), expected, rtol=1e-6, atol=1e-10)
        x = A@x+B@inputs[k]
        model.observe(C@x, inputs[k])


@pytest.mark.parametrize("method", ["ls", "nominal", "bo", "threshold", "sg"])
def test_column_config_keeps_native_networks_rewards_and_controller(method):
    from systems.distillation import get_distillation_notebook_defaults
    family = "combined" if method in ("sg", "threshold") else "markov"
    original = get_distillation_notebook_defaults(family)
    before = deepcopy(original)
    configured = configure(original, method, 7, Path("out"), {"aspen_path": "case.dynf", "snaps_path": "snaps"})
    assert configured["training_profile_override"] == "legacy_200"
    assert configured["n_tests_override"] == 200
    assert configured["set_points_len_override"] == 200
    assert configured["warm_start_override"] == 10
    for key, value in before["controller"].items():
        if method not in ("sg", "threshold") and key in ("run_rl_proposal", "rl_save_agent_checkpoint"):
            assert configured["controller"][key] is False
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(configured["controller"][key], value)
        else:
            assert configured["controller"][key] == value
    assert jsonable(configured["reward"]) == jsonable(before["reward"])
    assert original["n_tests_override"] == before["n_tests_override"]
    for key in ("td3_agent", "horizon_agent", "markov_td3_agent", "weights_td3_agent", "residual_td3_agent"):
        if key in configured:
            expected = dict(before[key], seed=7)
            assert jsonable(configured[key]) == jsonable(expected)


def bundle(steps=80000):
    return {"nFE": steps, "time_in_sub_episodes": 400, "warm_start_step": 4000,
            "y_sp": np.zeros((steps, 2)), "qi": np.zeros(steps), "qs": np.zeros(steps), "ha": np.zeros(steps),
            "disturbance_schedule": np.linspace(1e5, 1.1e5, steps),
            "sub_episodes_changes_dict": {}, "sub_episode_changes_dict": {}, "test_train_dict": {}}


def test_column_calibration_slices_the_original_feed_schedule():
    full = bundle()
    window = calibration_window(full, 4000, 4800)
    np.testing.assert_array_equal(window["disturbance_schedule"], full["disturbance_schedule"][4000:4800])
    assert window["nFE"] == 800
    assert full["nFE"] == 80000


def test_baseline_preserves_column_weight_ratios_and_uses_sliced_disturbances():
    full = bundle()
    model = types.SimpleNamespace(Q_out=np.array([2., 7.]), R_in=np.array([3., 5.]))
    plant = types.SimpleNamespace(step=lambda: None)
    ns = {"markov_cfg": {"observer_update_alignment": "legacy_previous_measurement"},
          "runtime_ctx": {"system": plant, "disturbance_schedule": full["disturbance_schedule"]},
          "EPISODE_BUNDLE": full, "MPC_obj": model, "reward_params": {},
          "SYS": {"input_bounds": {"u_min": np.zeros(2), "u_max": np.ones(2)}}}
    seen = {}
    def run(cfg, ctx):
        seen.update(cfg=cfg, ctx=ctx)
        return {}
    with patch("utils.mpc_baseline_runner.run_offsetfree_mpc", run):
        baseline(ns, calibration_window(full, 4000, 4800), alpha=1.5, beta=.75)
    np.testing.assert_array_equal(seen["ctx"]["MPC_obj"].Q_out, [3., 7.])
    np.testing.assert_array_equal(seen["ctx"]["MPC_obj"].R_in, [2.25, 3.75])
    np.testing.assert_array_equal(seen["ctx"]["disturbance_schedule"], full["disturbance_schedule"][4000:4800])
    np.testing.assert_array_equal(model.Q_out, [2., 7.])


def test_bo_reinitializes_each_trial_replays_nominal_prefix_and_restores_observer(tmp_path):
    full = bundle()
    prepared, prefixes, trials = [], [], []
    @contextmanager
    def resources(aspen):
        yield
    def prepare(seed, output, aspen):
        ns = {"EPISODE_BUNDLE": full, "markov_cfg": {"observer_update_alignment": "legacy_previous_measurement"},
              "runtime_ctx": {"system": object()}, "NB": {},
              "MPC_obj": types.SimpleNamespace(Q_out=np.array([2., 7.]), R_in=np.array([3., 5.])),
              "SYS": {"input_bounds": {"u_min": np.zeros(2), "u_max": np.ones(2)}}}
        prepared.append(ns)
        return ns
    def base(ns, selected=None, **kwargs):
        if selected is not None:
            prefixes.append((ns["runtime_ctx"]["system"], selected))
            return {"xhatdhat": np.array([[.1, .2], [.3, .4]])}
        return {"main_weights": kwargs}
    def trial(cfg, ctx):
        trials.append(ctx)
        np.testing.assert_array_equal(ctx["comparison_initial_xhat"], [.2, .4])
        np.testing.assert_array_equal(ctx["disturbance_schedule"], full["disturbance_schedule"][4000:4800])
        return {"y": np.zeros((801, 2)), "rewards_step": np.full(800, 3.)}
    with patch("review_experiments.e3.runtime.aspen_resources", resources), \
         patch("revision_comparisons.column.prepare", prepare), \
         patch("revision_comparisons.column.baseline", base), \
         patch("comparison_studies.native.baseline_with_initial_state", trial):
        result, agents = run_method("bo", tmp_path, {}, 7, bo_trials=5)
    assert len(prepared) == 6 and len(trials) == 5 and len(prefixes) == 5
    assert len({id(ns["runtime_ctx"]["system"]) for ns in prepared}) == 6
    assert result["comparison_bo_calibration"]["total_calibration_steps"] == 24000
    assert agents is None


def test_column_metrics_label_composition_and_after_action_temperature_correctly():
    from comparison_studies.analysis import trajectories
    with (ROOT/"Distillation/Data/mpc_results_disturb_fluctuation.pickle").open("rb") as stream:
        data = pickle.load(stream)
    rows = metrics(data)
    y, _, rewards, refs = trajectories(data)
    row = next(r for r in rows if r["window"] == "evaluation_200")
    np.testing.assert_allclose(row["tray85_temperature_rmse_K"], np.sqrt(np.mean((y[79601:, 1]-refs[79600:, 1])**2)))
    assert "eta_rmse_L_per_g" not in row
    np.testing.assert_allclose(row["mean_reward"], rewards[-400:].mean())


def test_progress_accepts_relative_legacy_attempts_and_partial_json_line(tmp_path):
    from revision_comparisons import progress
    study = tmp_path/"study"
    attempt = study/"combined_s7/a1"
    (attempt/"capture").mkdir(parents=True)
    (study/"manifest.json").write_text(json.dumps({"jobs": [{"category": "combined", "seed": 7,
        "state": "running", "attempts": ["combined_s7/a1"]}]}))
    (attempt/"status.json").write_text(json.dumps({"state": "running", "pid": 12}))
    (attempt/"worker.log").write_text("Sub_Episode: 191 | reward: -1\n")
    (attempt/"capture/decisions.jsonl").write_text('{"status":"completed","step":153412}\n{"stat')
    with patch.dict(progress.STUDIES, {"test": "study"}, clear=True), \
         patch.object(progress, "process_info", return_value={"alive": True, "cpu_seconds": 20.}):
        row = progress.snapshot(tmp_path)["jobs"][0]
    assert row["cycle"] == 191 and row["pid"] == 12
    assert row["completed_matched_decisions"] == 1 and row["last_matched_step"] == 153412


def test_column_families_use_the_same_fixed_reward_and_nominal_mpc():
    from systems.distillation import get_distillation_notebook_defaults
    ls = get_distillation_notebook_defaults("markov")
    sg = get_distillation_notebook_defaults("combined")
    assert jsonable(ls["reward"]) == jsonable(sg["reward"])
    for key in ("predict_h", "cont_h", "Q1_penalty", "Q2_penalty", "R1_penalty", "R2_penalty", "observer_update_alignment"):
        assert ls["controller"][key] == sg["controller"][key]


@pytest.mark.parametrize("changed_during_run", [False, True])
def test_column_worker_seals_success_or_retains_raw_data_before_guard_failure(tmp_path, changed_during_run):
    from run_distillation_comparisons import worker, completed
    from review_experiments.e3.common import sha256
    case = tmp_path/"offline_case.dynf"
    case.write_text("not an Aspen case; fake worker test only")
    spec = {"method": "nominal", "seed": 7, "source_hashes": {"fake": "hash"}, "profile": "legacy_200",
            "bo_trials": 40, "threads": 1, "aspen": {"aspen_path": str(case), "aspen_sha256": sha256(case)}}
    (tmp_path/"job.json").write_text(json.dumps(spec))
    def method(*args, **kwargs):
        with (ROOT/"Distillation/Data/mpc_results_disturb_fluctuation.pickle").open("rb") as stream:
            result = pickle.load(stream)
        (tmp_path/"resolved_defaults.json").write_text("{}")
        (tmp_path/"baseline_audit.json").write_text("{}")
        (tmp_path/"baseline_metrics.json").write_text("[]")
        return result, None
    hashes = [{"fake": "hash"}, {"fake": "changed" if changed_during_run else "hash"}]
    with patch("revision_comparisons.column.source_hashes", side_effect=hashes), \
         patch("revision_comparisons.column.run_method", method):
        code = worker(tmp_path/"job.json")
    assert (tmp_path/"training.pkl").is_file()
    assert code == int(changed_during_run)
    assert completed(tmp_path, spec) is (not changed_during_run)
    if not changed_during_run:
        with (tmp_path/"metrics.json").open("a") as stream:
            stream.write(" ")
        assert not completed(tmp_path, spec)  # reject a modified reference


def test_column_plan_does_not_open_an_aspen_session_or_launch_a_worker(capsys):
    import run_distillation_comparisons as runner
    with patch("subprocess.Popen", side_effect=AssertionError("No process may be launched by a plan")), \
         patch("revision_comparisons.column.source_hashes", return_value={}), \
         patch("review_experiments.e3.runtime.resolve_aspen", return_value={"aspen_path": "fake.dynf"}):
        assert runner.main(["--dry-run"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["methods"] == ["threshold", "sg", "ls", "bo"]
    assert data["bo_total_calibration_steps"] == 192000
    assert data["new_production_runs_started"] is False


def test_busy_column_case_stops_before_output_or_worker_creation(tmp_path):
    import run_distillation_comparisons as runner
    with patch("revision_comparisons.column.source_hashes", return_value={}), \
         patch("review_experiments.e3.runtime.resolve_aspen", return_value={"aspen_path": "fake.dynf"}), \
         patch("run_distillation_phase2_repeats.aspen_file_lock", side_effect=RuntimeError("case is occupied")), \
         patch("subprocess.Popen", side_effect=AssertionError("Must not launch")):
        with pytest.raises(RuntimeError, match="occupied"):
            runner.main(["--run", "--methods", "ls", "--output-dir", str(tmp_path/"not_created")])
    assert not (tmp_path/"not_created").exists()
