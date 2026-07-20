from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation import (
    get_distillation_notebook_defaults,
    resolve_distillation_combined_agent_kinds,
)
from utils.multiplier_mapping import map_centered_bounds_to_action


def _assert_equal(actual, expected):
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float))
    else:
        assert actual == expected


def test_distillation_combined_defaults_use_active_sg_standalone_parity():
    nb = get_distillation_notebook_defaults("combined")
    horizon = get_distillation_notebook_defaults("horizon_standard")
    markov = get_distillation_notebook_defaults("markov")
    weights = get_distillation_notebook_defaults("weights")
    residual = get_distillation_notebook_defaults("residual")

    assert nb["combined_agent_mode"] == "sg"
    assert nb["run_mode"] == "disturb"
    assert nb["disturbance_profile"] == "fluctuation"
    assert nb["enable_horizon"] is True
    assert nb["enable_markov"] is True
    assert nb["enable_weights"] is True
    assert nb["enable_residual"] is True
    assert nb["enable_matrix"] is False

    assert resolve_distillation_combined_agent_kinds(nb["combined_agent_mode"]) == {
        "horizon_agent_kind": "sg_dqn",
        "markov_agent_kind": "sg_td3",
        "weights_agent_kind": "sg_td3",
        "residual_agent_kind": "sg_td3",
    }
    assert nb["horizon_agent_kind"] == "sg_dqn"
    assert nb["markov_agent_kind"] == "sg_td3"
    assert nb["weights_agent_kind"] == "sg_td3"
    assert nb["residual_agent_kind"] == "sg_td3"

    assert nb["horizon_state_mode"] == horizon["state_mode"]
    assert nb["markov_state_mode"] == markov["state_mode"]
    assert nb["weights_state_mode"] == weights["state_mode"]
    assert nb["residual_state_mode"] == residual["state_mode"]

    _assert_equal(nb["horizon_agent"], horizon["agent"])
    _assert_equal(nb["markov_td3_agent"], markov["td3_agent"])
    _assert_equal(nb["weights_td3_agent"], weights["td3_agent"])
    _assert_equal(nb["residual_td3_agent"], residual["td3_agent"])
    _assert_equal(nb["horizon_supervisor_gate"], horizon["supervisor_gate"])
    _assert_equal(nb["markov_supervisor_gate"], markov["supervisor_gate"])
    _assert_equal(nb["weights_supervisor_gate"], weights["supervisor_gate"])
    _assert_equal(nb["residual_supervisor_gate"], residual["supervisor_gate"])
    _assert_equal(nb["horizon_safety"], horizon["horizon_safety"])
    _assert_equal(nb["weight_safety"], weights["weight_safety"])
    _assert_equal(nb["residual_safety"], residual["residual_safety"])

    assert nb["horizon_post_warm_start_action_freeze_subepisodes"] == horizon[
        "post_warm_start_action_freeze_subepisodes"
    ]
    assert nb["td3_post_warm_start_action_freeze_subepisodes"] == 3
    assert nb["td3_post_warm_start_actor_freeze_subepisodes"] == 3

    ctrl = nb["controller"]
    for key in (
        "decision_interval",
        "predict_grid",
        "control_grid",
        "predict_h",
        "cont_h",
        "Q1_penalty",
        "Q2_penalty",
        "R1_penalty",
        "R2_penalty",
        "mismatch_clip",
        "innovation_scale_mode",
        "innovation_scale_ref",
        "tracking_scale_mode",
        "tracking_eta_tol",
        "tracking_scale_floor",
        "tracking_scale_floor_mode",
        "base_state_norm_mode",
        "base_state_running_norm_clip",
        "base_state_running_norm_eps",
        "mismatch_feature_transform_mode",
        "mismatch_transform_tanh_scale",
        "mismatch_transform_post_clip",
        "observer_update_alignment",
    ):
        _assert_equal(ctrl[key], horizon["controller"][key])

    for key in (
        "basis_family",
        "z_bound",
        "z_safety",
        "prediction_window",
        "lambda_z",
        "s_pred_min",
        "gain_drift_max",
        "nominal_cost_relative_tol",
        "nominal_cost_absolute_tol",
        "run_adaptive_ls",
        "run_live_corrected_mpc",
        "run_rl_proposal",
        "rl_fallback_to_ls",
        "force_td3_execute",
        "force_td3_respects_warm_start",
        "rl_store_executed_action_in_replay",
        "td3_priority_fallback",
        "markov_shadow_safety",
    ):
        _assert_equal(ctrl[key], markov["controller"].get(key))

    _assert_equal(ctrl["weights_low"], weights["controller"]["low_coef"])
    _assert_equal(ctrl["weights_high"], weights["controller"]["high_coef"])
    _assert_equal(ctrl["residual_low"], residual["controller"]["low_coef"])
    _assert_equal(ctrl["residual_high"], residual["controller"]["high_coef"])
    assert np.all(ctrl["model_high"] > ctrl["model_low"])
    assert np.all(ctrl["model_low"] < 1.0)
    assert np.all(ctrl["model_high"] > 1.0)
    model_baseline_raw = map_centered_bounds_to_action(
        np.ones_like(ctrl["model_low"]),
        ctrl["model_low"],
        ctrl["model_high"],
        nominal=1.0,
    )
    np.testing.assert_allclose(model_baseline_raw, np.zeros_like(model_baseline_raw))

    assert nb["residual_authority_enabled"] is False
    assert nb["append_rho_to_state"] is False
    assert nb["authority_use_rho"] is False
    assert nb["use_rho_authority"] is False
    assert nb["residual_zero_deadband_enabled"] is False

    for stale_key in (
        "sac_agent",
        "markov_sac_agent",
        "weights_sac_agent",
        "residual_sac_agent",
        "td7_agent",
        "horizon_dueling_agent",
    ):
        assert stale_key not in nb


def test_distillation_combined_plain_mode_resolves_all_plain_agents():
    expected = {
        "horizon_agent_kind": "dqn",
        "markov_agent_kind": "td3",
        "weights_agent_kind": "td3",
        "residual_agent_kind": "td3",
    }
    assert resolve_distillation_combined_agent_kinds("plain") == expected
    assert resolve_distillation_combined_agent_kinds("without_sg") == expected
    assert resolve_distillation_combined_agent_kinds("no-sg") == expected


def test_distillation_combined_profiles_use_simple_future_prefixes():
    nb = get_distillation_notebook_defaults("combined")
    profile = nb["run_profiles"][("disturb", "fluctuation")]

    assert (
        profile["result_prefix_template"].format(mode="sg")
        == "distillation_combined_sg_disturb_fluctuation"
    )
    assert (
        profile["result_prefix_template"].format(mode="plain")
        == "distillation_combined_plain_disturb_fluctuation"
    )
    assert (
        profile["compare_prefix_template"].format(mode="sg")
        == "distillation_compare_combined_sg_disturb_fluctuation"
    )


def test_distillation_combined_root_runner_is_active_sg_plain_only():
    source = (ROOT / "distillation_RL_assisted_MPC_combined_unified.py").read_text(encoding="utf-8")

    assert "resolve_distillation_combined_agent_kinds" in source
    assert "run_combined_supervisor" in source
    assert "build_distillation_system" in source
    assert "distillation_system_stepper" in source
    assert "build_distillation_training_profile" in source
    assert '"episode_bundle": EPISODE_BUNDLE' in source
    assert '"training_profile_name": TRAINING_PROFILE_NAME' in source
    assert "validate_standalone_parity" in source
    assert "validate_disabled_matrix_bounds" in source
    assert "legacy matrix branch disabled" in source
    assert "residual rho authority disabled" in source
    assert 'result_bundle["disturbance_profile_name"] = DISTURBANCE_PROFILE' in source
    assert 'result_bundle["disturbance_profile"] = DISTURBANCE_PROFILE' not in source
    assert "SupervisorGatedDQNAgent" in source
    assert "SupervisorGatedTD3Agent" in source
    assert "SACAgent" not in source
    assert "DuelingDQN" not in source
    assert "TD7" not in source
    assert "sg_sac" not in source
    assert "td7" not in source.lower()
    assert "dueling" not in source.lower()


def test_distillation_combined_runner_ignores_stale_direct_agent_kind_defaults():
    source = (ROOT / "distillation_RL_assisted_MPC_combined_unified.py").read_text(encoding="utf-8")

    assert 'HORIZON_AGENT_KIND = RESOLVED_AGENT_KINDS["horizon_agent_kind"]' in source
    assert 'MARKOV_AGENT_KIND = RESOLVED_AGENT_KINDS["markov_agent_kind"]' in source
    assert 'WEIGHTS_AGENT_KIND = RESOLVED_AGENT_KINDS["weights_agent_kind"]' in source
    assert 'RESIDUAL_AGENT_KIND = RESOLVED_AGENT_KINDS["residual_agent_kind"]' in source
    assert 'NB.get("horizon_agent_kind"' not in source
    assert 'NB.get("markov_agent_kind"' not in source
    assert 'NB.get("weights_agent_kind"' not in source
    assert 'NB.get("residual_agent_kind"' not in source


def run_direct():
    test_distillation_combined_defaults_use_active_sg_standalone_parity()
    test_distillation_combined_plain_mode_resolves_all_plain_agents()
    test_distillation_combined_profiles_use_simple_future_prefixes()
    test_distillation_combined_root_runner_is_active_sg_plain_only()
    test_distillation_combined_runner_ignores_stale_direct_agent_kind_defaults()
    print("distillation combined runner tests passed")


if __name__ == "__main__":
    run_direct()
