from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation import (
    DISTILLATION_RL_SETPOINTS_PHYS,
    DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
    TEMPERATURE_FLIP_PHASE2_SETPOINTS_PHYS,
    build_distillation_training_profile,
    get_distillation_notebook_defaults,
)
from systems.distillation.data_io import canonical_baseline_path
from systems.distillation.scenarios import (
    _generate_feed_fluctuation_segment,
    generate_feed_fluctuation,
)
from utils.episode_profiles import validate_episode_bundle
from utils.helpers import apply_min_max, reverse_min_max
from utils import plotting_core


DATA_MIN = np.asarray([300_000.0, 100.0, 0.002, -26.0], dtype=float)
DATA_MAX = np.asarray([460_000.0, 150.0, 0.05, -16.0], dtype=float)
STEADY_OUTPUTS = np.asarray([0.02, -22.0], dtype=float)
NOMINAL_FEED = 150_032.484


def _build_profile():
    return build_distillation_training_profile(
        profile_name=DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
        run_mode="disturb",
        disturbance_profile="fluctuation",
        y_sp_scenario_phys=DISTILLATION_RL_SETPOINTS_PHYS,
        n_tests=300,
        set_points_len=200,
        warm_start=10,
        test_cycle=[False, False, False, False, False],
        steady_outputs=STEADY_OUTPUTS,
        data_min=DATA_MIN,
        data_max=DATA_MAX,
        n_inputs=2,
        nominal_feed=NOMINAL_FEED,
        seed=42,
    )


def _setpoints_phys(profile):
    y_ss_scaled = apply_min_max(STEADY_OUTPUTS, DATA_MIN[2:], DATA_MAX[2:])
    return reverse_min_max(profile["y_sp"] + y_ss_scaled, DATA_MIN[2:], DATA_MAX[2:])


def test_temperature_flip_profile_preserves_phase1_and_continues_feed_rng():
    profile = _build_profile()
    legacy_feed = generate_feed_fluctuation(
        80_000,
        nominal_feed=NOMINAL_FEED,
        seed=42,
    )
    setpoints_phys = _setpoints_phys(profile)

    assert profile["nFE"] == 120_000
    assert profile["time_in_sub_episodes"] == 400
    assert profile["phase_switch_step"] == 80_000
    assert profile["phase_switch_episode"] == 201
    np.testing.assert_array_equal(profile["disturbance_schedule"][:80_000], legacy_feed)
    assert profile["disturbance_schedule"][80_000] == profile["disturbance_schedule"][79_999]

    rng = np.random.RandomState(42)
    expected_phase1, terminal_feed = _generate_feed_fluctuation_segment(
        total_steps=80_000,
        nominal_feed=NOMINAL_FEED,
        current_feed=NOMINAL_FEED,
        rng=rng,
    )
    expected_phase2, _ = _generate_feed_fluctuation_segment(
        total_steps=40_000,
        nominal_feed=NOMINAL_FEED,
        current_feed=terminal_feed,
        rng=rng,
    )
    np.testing.assert_array_equal(profile["disturbance_schedule"][:80_000], expected_phase1)
    np.testing.assert_array_equal(profile["disturbance_schedule"][80_000:], expected_phase2)

    np.testing.assert_allclose(
        setpoints_phys[:80_000],
        np.tile(
            np.repeat(DISTILLATION_RL_SETPOINTS_PHYS, 200, axis=0),
            (200, 1),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        setpoints_phys[80_000:],
        np.tile(
            np.repeat(TEMPERATURE_FLIP_PHASE2_SETPOINTS_PHYS, 200, axis=0),
            (100, 1),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert profile["test_train_dict"][79_600] is False
    assert profile["test_train_dict"][80_000] is False
    assert profile["test_train_dict"][119_600] is True
    assert profile["exploration_freeze_step"] == 80_000
    assert profile["phase2_episode_status"] == {
        "learning_episode_start": 201,
        "learning_episode_end": 299,
        "evaluation_only_episodes": [300],
    }


def test_legacy_profile_remains_selectable_and_matches_current_generator():
    legacy = build_distillation_training_profile(
        profile_name="legacy_200",
        run_mode="disturb",
        disturbance_profile="fluctuation",
        y_sp_scenario_phys=DISTILLATION_RL_SETPOINTS_PHYS,
        n_tests=200,
        set_points_len=200,
        warm_start=10,
        test_cycle=[False] * 5,
        steady_outputs=STEADY_OUTPUTS,
        data_min=DATA_MIN,
        data_max=DATA_MAX,
        n_inputs=2,
        nominal_feed=NOMINAL_FEED,
        seed=42,
    )
    assert legacy["nFE"] == 80_000
    assert legacy["phase_switch_step"] is None
    np.testing.assert_array_equal(
        legacy["disturbance_schedule"],
        generate_feed_fluctuation(80_000, nominal_feed=NOMINAL_FEED, seed=42),
    )
    assert legacy["test_train_dict"][79_600] is True


def test_active_distillation_defaults_use_temperature_flip_and_sg():
    baseline = get_distillation_notebook_defaults("baseline")
    horizon = get_distillation_notebook_defaults("horizon_standard")
    markov = get_distillation_notebook_defaults("markov")
    weights = get_distillation_notebook_defaults("weights")
    residual = get_distillation_notebook_defaults("residual")
    combined = get_distillation_notebook_defaults("combined")

    assert baseline["run_profiles"][("disturb", "fluctuation")]["n_tests"] == 300
    assert horizon["agent_mode"] == "sg"
    assert horizon["agent_kind"] == "sg_dqn"
    assert horizon["episode_defaults"]["profile_name"] == DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE
    assert horizon["episode_defaults"]["n_tests"] == 300
    for settings in (markov, weights, residual):
        assert settings["agent_mode"] == "sg"
        assert settings["agent_kind"] == "sg_td3"
        run_profile = settings["run_profiles"][("sg_td3", "disturb", "fluctuation")]
        assert run_profile["profile_name"] == DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE
        assert run_profile["n_tests"] == 300
    assert combined["combined_agent_mode"] == "sg"
    assert combined["episode_defaults"]["n_tests"] == 300

    markov_source = (ROOT / "distillation_RL_assisted_MPC_markov_unified.py").read_text(
        encoding="utf-8"
    )
    assert '"disturbance_labels": DISTILLATION_SYSTEM_METADATA.get("disturbance_labels")' in markov_source


def test_profile_aware_baseline_path_is_distinct():
    legacy = canonical_baseline_path(ROOT, "disturb", "fluctuation")
    profile = canonical_baseline_path(
        ROOT,
        "disturb",
        "fluctuation",
        training_profile_name=DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
    )
    assert legacy.name == "mpc_results_disturb_fluctuation.pickle"
    assert profile.name == "mpc_results_disturb_fluctuation_temperature_flip_200_100.pickle"
    assert profile != legacy


def test_validated_episode_bundle_rejects_wrong_schedule_length():
    profile = _build_profile()
    validated = validate_episode_bundle(
        profile,
        expected_n_tests=300,
        expected_n_outputs=2,
    )
    assert validated["nFE"] == 120_000
    broken = dict(profile)
    broken["y_sp"] = profile["y_sp"][:-1]
    try:
        validate_episode_bundle(broken, expected_n_tests=300, expected_n_outputs=2)
    except ValueError as exc:
        assert "y_sp" in str(exc)
    else:
        raise AssertionError("A truncated setpoint schedule must be rejected.")


def test_synthetic_two_phase_plots_cover_all_active_families(monkeypatch):
    profile = _build_profile()
    y_sp_phys = _setpoints_phys(profile)
    bundle = {
        **profile,
        "y_line_full": np.vstack([y_sp_phys[0], y_sp_phys]),
        "u_step_full": np.zeros((profile["nFE"], 2), dtype=float),
        "n_inputs": 2,
        "n_outputs": 2,
        "steady_states": {"y_ss": STEADY_OUTPUTS.copy(), "ss_inputs": np.asarray([320_000.0, 110.0])},
        "data_min": DATA_MIN.copy(),
        "data_max": DATA_MAX.copy(),
        "disturbance_profile": {"Feed flow (kg/h)": profile["disturbance_schedule"]},
        "rewards_step": np.zeros(profile["nFE"], dtype=float),
        "effective_exploration_step_log": np.zeros(profile["nFE"], dtype=float),
        "system_metadata": {},
    }
    saved_names = []

    def record_figure(fig, out_dir, fname_base, save_pdf=False):
        del out_dir, save_pdf
        saved_names.append(fname_base)
        plt.close(fig)

    monkeypatch.setattr(plotting_core, "_save_fig", record_figure)
    for family in ("mpc", "horizon", "markov", "weights", "residual", "combined"):
        plotting_core._plot_two_phase_study(bundle, "unused", family, save_pdf=False)
        assert f"fig_{family}_robustness_outputs_inputs" in saved_names
        assert f"fig_{family}_robustness_disturbances" in saved_names

    summary = bundle["phase_window_summary"]
    assert set(summary) == {
        "phase1_tail_episodes_191_200",
        "phase2_entry_episodes_201_210",
        "phase2_tail_episodes_291_300",
    }
    for values in summary.values():
        np.testing.assert_allclose(values["tracking_rmse_phys"], 0.0, atol=1.0e-12)
        assert values["average_reward"] == 0.0


@pytest.mark.parametrize("changed_schedule", ["setpoint", "feed"])
def test_baseline_comparison_rejects_changed_schedule(monkeypatch, changed_schedule):
    base = {
        "y": np.zeros((3, 2), dtype=float),
        "u": np.zeros((2, 2), dtype=float),
        "nFE": 2,
        "delta_t": 1.0,
        "time_in_sub_episodes": 1,
        "y_sp": np.zeros((2, 2), dtype=float),
        "data_min": DATA_MIN.copy(),
        "data_max": DATA_MAX.copy(),
        "training_profile_name": DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
        "phase_switch_step": 80_000,
        "phase_switch_episode": 201,
        "disturbance_profile": {"Feed flow (kg/h)": np.asarray([150_000.0, 150_001.0])},
    }
    baseline = dict(base)
    if changed_schedule == "setpoint":
        baseline["y_sp"] = np.asarray([[0.0, 0.0], [0.0, 1.0e-6]])
    else:
        baseline["disturbance_profile"] = {
            "Feed flow (kg/h)": np.asarray([150_000.0, 150_002.0])
        }

    monkeypatch.setattr(
        plotting_core,
        "load_pickle",
        lambda path: base if path == "rl" else baseline,
    )
    with pytest.warns(RuntimeWarning, match="schedule mismatch"):
        result = plotting_core.compare_mpc_rl_from_dirs_core(
            rl_dir="rl",
            mpc_path_or_dir="baseline",
            reward_fn=lambda *args, **kwargs: 0.0,
            directory="unused",
            prefix_name="distillation_mismatch",
            allow_missing_baseline=True,
            expected_training_profile_name=DISTILLATION_TEMPERATURE_FLIP_TRAINING_PROFILE,
        )
    assert result is None
