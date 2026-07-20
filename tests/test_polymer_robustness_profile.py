from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DQN.dqn_agent import DQNAgent
from TD3Agent.agent import TD3Agent
from systems.polymer import get_polymer_notebook_defaults
from systems.polymer.data_io import canonical_baseline_path
from systems.polymer.scenarios import (
    POLYMER_LEGACY_TRAINING_PROFILE,
    POLYMER_ROBUSTNESS_TRAINING_PROFILE,
    ROBUSTNESS_FOULED_HA,
    build_polymer_training_profile,
    default_polymer_profile_episode_count,
)
from utils.exploration_freeze import (
    effective_agent_exploration_value,
    maybe_freeze_agent_exploration,
)
from utils.helpers import apply_min_max, generate_setpoints_training_rl_gradually, reverse_min_max
import utils.plotting_core as plotting_core
from utils.plotting_core import (
    _plot_polymer_robustness_study,
    compare_mpc_rl_from_dirs_core,
)


DATA_MIN = np.asarray([0.0, 0.0, 3.0, 320.0], dtype=float)
DATA_MAX = np.asarray([1.0, 1.0, 5.0, 325.0], dtype=float)
STEADY_OUTPUTS = np.asarray([4.0, 322.0], dtype=float)
PHASE1_SETPOINTS_PHYS = np.asarray([[4.5, 324.0], [3.4, 321.0]], dtype=float)


def _scaled_phase1_setpoints():
    return apply_min_max(PHASE1_SETPOINTS_PHYS, DATA_MIN[2:], DATA_MAX[2:]) - apply_min_max(
        STEADY_OUTPUTS,
        DATA_MIN[2:],
        DATA_MAX[2:],
    )


def _build_profile():
    return build_polymer_training_profile(
        profile_name=POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        y_sp_scenario=_scaled_phase1_setpoints(),
        n_tests=300,
        set_points_len=400,
        warm_start=10,
        test_cycle=[False] * 5,
        nominal_qi=108.0,
        nominal_qs=459.0,
        nominal_ha=1.05e6,
        qi_change=0.85,
        qs_change=1.3,
        ha_change=0.85,
        steady_outputs=STEADY_OUTPUTS,
        data_min=DATA_MIN,
        data_max=DATA_MAX,
        n_inputs=2,
    )


def test_phase1_matches_legacy_schedule_sample_for_sample():
    profile = _build_profile()
    legacy = generate_setpoints_training_rl_gradually(
        _scaled_phase1_setpoints(),
        200,
        400,
        10,
        [False] * 5,
        108.0,
        459.0,
        1.05e6,
        0.85,
        1.3,
        0.85,
    )

    assert profile["nFE"] == 240_000
    assert profile["time_in_sub_episodes"] == 800
    np.testing.assert_array_equal(profile["y_sp"][:160_000], legacy[0])
    np.testing.assert_array_equal(profile["qi"][:160_000], legacy[6])
    np.testing.assert_array_equal(profile["qs"][:160_000], legacy[7])
    np.testing.assert_array_equal(profile["ha"][:160_000], legacy[8])


def test_legacy_200_episode_profile_remains_explicitly_selectable():
    assert default_polymer_profile_episode_count(POLYMER_LEGACY_TRAINING_PROFILE) == 200
    assert default_polymer_profile_episode_count(POLYMER_ROBUSTNESS_TRAINING_PROFILE) == 300
    legacy = build_polymer_training_profile(
        profile_name=POLYMER_LEGACY_TRAINING_PROFILE,
        y_sp_scenario=_scaled_phase1_setpoints(),
        n_tests=200,
        set_points_len=400,
        warm_start=10,
        test_cycle=[False] * 5,
        nominal_qi=108.0,
        nominal_qs=459.0,
        nominal_ha=1.05e6,
        qi_change=0.85,
        qs_change=1.3,
        ha_change=0.85,
        steady_outputs=STEADY_OUTPUTS,
        data_min=DATA_MIN,
        data_max=DATA_MAX,
        n_inputs=2,
    )
    assert legacy["training_profile_name"] == POLYMER_LEGACY_TRAINING_PROFILE
    assert legacy["nFE"] == 160_000
    assert legacy["test_train_dict"][159_200] is True


def test_phase2_setpoints_continuous_disturbances_and_persistent_fouling():
    profile = _build_profile()
    phase2_start = 160_000
    y_ss_scaled = apply_min_max(STEADY_OUTPUTS, DATA_MIN[2:], DATA_MAX[2:])
    phase2_phys = reverse_min_max(
        profile["y_sp"][phase2_start:] + y_ss_scaled,
        DATA_MIN[2:],
        DATA_MAX[2:],
    )

    np.testing.assert_allclose(phase2_phys[:400], np.tile([4.0, 321.5], (400, 1)))
    np.testing.assert_allclose(phase2_phys[400:800], np.tile([3.3, 324.5], (400, 1)))
    assert profile["qi"][phase2_start] == pytest.approx(91.8)
    assert profile["qi"][-1] == pytest.approx(102.6)
    assert profile["qs"][phase2_start] == pytest.approx(596.7)
    assert profile["qs"][-1] == pytest.approx(481.95)
    np.testing.assert_array_equal(
        profile["ha"][phase2_start:],
        np.full(80_000, ROBUSTNESS_FOULED_HA),
    )
    assert profile["phase_switch_step"] == profile["exploration_freeze_step"] == 160_000
    assert profile["phase_switch_episode"] == 201
    assert profile["fouling_active"] is True
    assert profile["exploration_freeze_settings"]["last_training_episode"] == 299
    assert profile["exploration_freeze_settings"]["evaluation_episode"] == 300
    assert profile["phase2_episode_status"]["learning_episode_start"] == 201
    assert profile["phase2_episode_status"]["learning_episode_end"] == 299
    assert profile["phase2_episode_status"]["evaluation_only_episodes"] == [300]
    assert profile["test_train_dict"][159_200] is False
    assert profile["test_train_dict"][160_000] is False
    assert profile["test_train_dict"][238_400] is False
    assert profile["test_train_dict"][239_200] is True


def test_exploration_freeze_holds_dqn_and_both_td3_noise_schedules():
    dqn = DQNAgent(
        state_dim=3,
        action_dim=2,
        hidden_dim=[8],
        batch_size=2,
        buffer_size=16,
        eps_start=0.8,
        eps_end=0.2,
        eps_decay_steps=100,
        eps_decay_mode="linear",
        device=torch.device("cpu"),
    )
    dqn.steps = 75
    dqn_expected = dqn.eps_schedule.value(75)
    info = maybe_freeze_agent_exploration(dqn, environment_step=160_000, freeze_step=160_000)
    assert info["value"] == pytest.approx(dqn_expected)
    dqn.steps = 10_000
    assert effective_agent_exploration_value(dqn) == pytest.approx(dqn_expected)
    assert effective_agent_exploration_value(dqn, test=True) == 0.0

    for mode in ("gaussian", "param_noise"):
        td3 = TD3Agent(
            state_dim=3,
            action_dim=2,
            actor_hidden=[8],
            critic_hidden=[8],
            batch_size=2,
            buffer_size=16,
            std_start=0.5,
            std_end=0.1,
            std_decay_steps=100,
            std_decay_mode="linear",
            param_noise_std_start=0.4,
            param_noise_std_end=0.05,
            param_noise_decay_steps=100,
            param_noise_decay_mode="linear",
            exploration_mode=mode,
            device=torch.device("cpu"),
            use_adamw=False,
        )
        td3.steps = 60
        expected = td3.effective_exploration_schedule_value()
        maybe_freeze_agent_exploration(td3, environment_step=160_000, freeze_step=160_000)
        td3.steps = 50_000
        assert effective_agent_exploration_value(td3) == pytest.approx(expected)
        assert effective_agent_exploration_value(td3, test=True) == 0.0


def test_polymer_defaults_use_robustness_profile_sg_agents_and_distinct_names():
    baseline = get_polymer_notebook_defaults("baseline")
    horizon = get_polymer_notebook_defaults("horizon_standard")
    markov = get_polymer_notebook_defaults("markov")
    weights = get_polymer_notebook_defaults("weights")
    residual = get_polymer_notebook_defaults("residual")
    combined = get_polymer_notebook_defaults("combined")

    assert baseline["run_profiles"]["disturb"]["profile_name"] == POLYMER_ROBUSTNESS_TRAINING_PROFILE
    assert horizon["agent_kind"] == "sg_dqn"
    for settings in (markov, weights, residual):
        assert settings["agent_kind"] == "sg_td3"
    assert combined["combined_agent_mode"] == "sg"
    for settings in (horizon, markov, weights, residual, combined):
        assert settings["episode_defaults"]["profile_name"] == POLYMER_ROBUSTNESS_TRAINING_PROFILE
        assert settings["episode_defaults"]["n_tests"] == 300

    robust_path = canonical_baseline_path(
        ".",
        "disturb",
        training_profile_name=POLYMER_ROBUSTNESS_TRAINING_PROFILE,
    )
    legacy_path = canonical_baseline_path(
        ".",
        "disturb",
        training_profile_name=POLYMER_LEGACY_TRAINING_PROFILE,
    )
    assert robust_path.name == "mpc_results_dist_robustness_200_100.pickle"
    assert legacy_path.name == "mpc_results_dist.pickle"
    assert robust_path != legacy_path


def _synthetic_robustness_bundle():
    profile = _build_profile()
    y_ss_scaled = apply_min_max(STEADY_OUTPUTS, DATA_MIN[2:], DATA_MAX[2:])
    y_sp_phys = reverse_min_max(profile["y_sp"] + y_ss_scaled, DATA_MIN[2:], DATA_MAX[2:])
    exploration = np.concatenate(
        [
            np.linspace(1.0, 0.1, 160_000),
            np.full(79_200, 0.1),
            np.zeros(800),
        ]
    )
    return {
        **profile,
        "y_line_full": np.vstack([y_sp_phys[0], y_sp_phys]),
        "u_step_full": np.zeros((profile["nFE"], 2), dtype=float),
        "n_inputs": 2,
        "n_outputs": 2,
        "steady_states": {"y_ss": STEADY_OUTPUTS.copy(), "ss_inputs": np.asarray([0.5, 0.5])},
        "data_min": DATA_MIN.copy(),
        "data_max": DATA_MAX.copy(),
        "disturbance_profile": {
            "qi": profile["qi"],
            "qs": profile["qs"],
            "ha": profile["ha"],
        },
        "rewards_step": np.zeros(profile["nFE"], dtype=float),
        "effective_exploration_step_log": exploration,
        "system_metadata": {},
    }


def test_synthetic_300_episode_robustness_plots_and_metric_windows(monkeypatch):
    bundle = _synthetic_robustness_bundle()
    saved_names = []

    def record_figure(fig, out_dir, fname_base, save_pdf=False):
        del out_dir, save_pdf
        saved_names.append(f"{fname_base}.png")
        plt.close(fig)

    monkeypatch.setattr(plotting_core, "_save_fig", record_figure)
    _plot_polymer_robustness_study(bundle, "unused", "synthetic", save_pdf=False)

    expected = {
        "fig_synthetic_robustness_outputs_inputs.png",
        "fig_synthetic_robustness_disturbances.png",
        "fig_synthetic_robustness_reward_exploration.png",
        "fig_synthetic_robustness_entry_final_outputs.png",
        "fig_synthetic_robustness_entry_final_inputs.png",
    }
    assert expected.issubset(set(saved_names))
    summary = bundle["robustness_window_summary"]
    assert set(summary) == {
        "phase1_tail_episodes_191_200",
        "phase2_entry_episodes_201_210",
        "phase2_tail_episodes_291_300",
    }
    for values in summary.values():
        np.testing.assert_allclose(values["tracking_rmse_phys"], 0.0, atol=1.0e-12)
        assert values["average_reward"] == 0.0


def _minimal_comparison_bundle():
    return {
        "y": np.zeros((3, 2), dtype=float),
        "u": np.zeros((2, 2), dtype=float),
        "nFE": 2,
        "delta_t": 1.0,
        "time_in_sub_episodes": 1,
        "y_sp": np.zeros((2, 2), dtype=float),
        "data_min": DATA_MIN.copy(),
        "data_max": DATA_MAX.copy(),
        "training_profile_name": POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        "phase_switch_step": 160_000,
        "phase_switch_episode": 201,
        "disturbance_profile": {
            "qi": np.asarray([91.8, 91.9]),
            "qs": np.asarray([596.7, 596.6]),
            "ha": np.asarray([892_500.0, 892_500.0]),
        },
    }


def test_missing_or_schedule_mismatched_baseline_is_skipped_with_warning(monkeypatch):
    rl_bundle = _minimal_comparison_bundle()

    def missing_loader(path):
        if path == "rl":
            return rl_bundle
        raise FileNotFoundError(path)

    monkeypatch.setattr(plotting_core, "load_pickle", missing_loader)
    with pytest.warns(RuntimeWarning, match="no baseline bundle exists"):
        assert compare_mpc_rl_from_dirs_core(
            rl_dir="rl",
            mpc_path_or_dir="missing",
            reward_fn=lambda *args, **kwargs: 0.0,
            directory="unused",
            prefix_name="missing_compare",
            allow_missing_baseline=True,
            expected_training_profile_name=POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        ) is None

    baseline = _minimal_comparison_bundle()
    baseline["disturbance_profile"] = dict(baseline["disturbance_profile"])
    baseline["disturbance_profile"]["qi"] = np.asarray([91.8, 102.6])

    def mismatch_loader(path):
        return rl_bundle if path == "rl" else baseline

    monkeypatch.setattr(plotting_core, "load_pickle", mismatch_loader)
    with pytest.warns(RuntimeWarning, match="schedule mismatch"):
        assert compare_mpc_rl_from_dirs_core(
            rl_dir="rl",
            mpc_path_or_dir="baseline",
            reward_fn=lambda *args, **kwargs: 0.0,
            directory="unused",
            prefix_name="mismatch_compare",
            allow_missing_baseline=True,
            expected_training_profile_name=POLYMER_ROBUSTNESS_TRAINING_PROFILE,
        ) is None
