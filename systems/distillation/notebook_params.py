from __future__ import annotations

from copy import deepcopy

import numpy as np

from .config import (
    DELTA_T_HOURS,
    DISTILLATION_ACTIVE_HORIZON_RUN_PROFILES,
    DISTILLATION_ACTIVE_MARKOV_RUN_PROFILES,
    DISTILLATION_ACTIVE_RESIDUAL_RUN_PROFILES,
    DISTILLATION_ACTIVE_WEIGHT_RUN_PROFILES,
    DISTILLATION_BASELINE_RUN_PROFILES,
    DISTILLATION_COMBINED_RUN_PROFILES,
    DISTILLATION_COMBINED_SETPOINTS_PHYS,
    DISTILLATION_INPUT_BOUNDS,
    DISTILLATION_MARKOV_RUN_PROFILES,
    DISTILLATION_MATRIX_RUN_PROFILES,
    DISTILLATION_MATRIX_ALPHA_DEFAULT_HIGH,
    DISTILLATION_MATRIX_ALPHA_DEFAULT_LOW,
    DISTILLATION_MATRIX_ALPHA_UPPER_CAP,
    DISTILLATION_DEFAULT_MULTIPLIER_LOW,
    DISTILLATION_DEFAULT_MULTIPLIER_HIGH,
    DISTILLATION_HORIZON_RUN_PROFILES,
    DISTILLATION_NOMINAL_CONDITIONS,
    DISTILLATION_OBSERVER_POLES,
    DISTILLATION_REIDENTIFICATION_RUN_PROFILES,
    DISTILLATION_RESIDUAL_RUN_PROFILES,
    DISTILLATION_RL_SETPOINTS_PHYS,
    DISTILLATION_SETPOINT_RANGE_PHYS,
    DISTILLATION_SS_INPUTS,
    DISTILLATION_SS_INPUTS_SYSTEM_ID,
    DISTILLATION_WEIGHT_RUN_PROFILES,
    HORIZON_CONTROL_GRID,
    HORIZON_PREDICT_GRID,
    MATRIX_MULTIPLIER_BOUNDS,
    RESIDUAL_BOUNDS,
    RL_REWARD_DEFAULTS as _RL_REWARD_DEFAULTS,
    WEIGHT_MULTIPLIER_BOUNDS,
)


DISTILLATION_DEFAULT_DQN_HIDDEN = [512, 512, 512, 512, 512]
DISTILLATION_DEFAULT_ACTOR_HIDDEN = [512, 512, 512, 512, 512]
DISTILLATION_DEFAULT_CRITIC_HIDDEN = [512, 512, 512, 512, 512]
DISTILLATION_DEFAULT_GAMMA = 0.99


def _copy_reward_defaults():
    return {k: deepcopy(v) for k, v in _RL_REWARD_DEFAULTS.items()}


def _distillation_disabled_matrix_bounds(n_inputs: int):
    low = np.full(1 + int(n_inputs), DISTILLATION_MATRIX_ALPHA_DEFAULT_LOW, dtype=float)
    high = np.full(1 + int(n_inputs), DISTILLATION_MATRIX_ALPHA_DEFAULT_HIGH, dtype=float)
    high[0] = min(high[0], DISTILLATION_MATRIX_ALPHA_UPPER_CAP)
    return low, high


def _copy_replay_defaults():
    return {
        # Replay buffer controls:
        # - buffer_size: total transition capacity
        # - replay_frac_per / replay_frac_recent: fraction of each batch drawn
        #   from PER and the recent window; the remainder is uniform
        # - replay_recent_window_mult: notebook helper multiplier used to derive
        #   the effective recent window as min(buffer_size, mult * set_points_len)
        # - replay_recent_window: explicit override; None keeps the derived value
        # - replay_alpha / replay_beta_*: standard PER priority and IS-weight controls
        "buffer_size": 40_000,
        "replay_frac_per": 0.5,
        "replay_frac_recent": 0.2,
        "replay_recent_window_mult": 5,
        "replay_recent_window": None,
        "replay_alpha": 0.6,
        "replay_beta_start": 0.4,
        "replay_beta_end": 1.0,
        "replay_beta_steps": 50_000,
    }


def _copy_active_replay_defaults():
    defaults = _copy_replay_defaults()
    defaults.update(
        {
            "buffer_size": 40_000,
            "replay_frac_per": 0.4,
            "replay_frac_recent": 0.3,
            "replay_recent_window_mult": 10,
            "replay_recent_window": None,
        }
    )
    return defaults


def _copy_active_td3_noise_defaults():
    return {
        "target_policy_smoothing_noise_std": 0.1,
        "noise_clip": 0.2,
        "std_start": 0.2,
        "std_end": 0.02,
        "param_noise_std_start": 0.2,
        "param_noise_std_end": 0.02,
        "std_decay_rate": 0.99995,
        "std_decay_mode": "exp",
        "exploration_mode": "gaussian",
        "param_noise_resample_interval": 4,
    }


def _copy_offline_multiplier_diagnostic_defaults(enabled=False):
    return {
        "enabled": bool(enabled),
        "epsilon_log": 0.02,
        "n_random_samples": 2_000,
        "seed": 42,
        "rho_target": 0.995,
        "gain_threshold": 0.25,
        "save_outputs": True,
        "make_plots": True,
        "apply_suggested_caps": False,
    }


def _copy_release_protected_advisory_cap_defaults(enabled=False):
    return {
        "enabled": bool(enabled),
        "use_offline_diagnostic_bounds": True,
        "protected_live_subepisodes": 15,
        "authority_ramp_subepisodes": 30,
        "store_executed_action_in_replay": True,
        "log_policy_and_executed_multipliers": True,
        "fail_if_diagnostic_missing": True,
    }


def _copy_mpc_acceptance_fallback_defaults(enabled=False):
    return {
        "enabled": bool(enabled),
        "relative_tolerance": 0.0,
        "absolute_tolerance": 1e-8,
        "fallback_on_candidate_solve_failure": True,
        "store_executed_action_in_replay": True,
        "log_policy_candidate_and_executed": True,
    }


def _copy_mpc_dual_cost_shadow_defaults(enabled=False):
    return {
        "enabled": bool(enabled),
        "relative_tolerance": 1e-4,
        "absolute_tolerance": 1e-8,
        "benefit_tolerance": 0.0,
        "fallback_on_candidate_solve_failure": True,
    }


def _copy_mpc_usefulness_gate_defaults(enabled=False):
    return {
        "enabled": bool(enabled),
        "relative_tolerance": 1e-4,
        "absolute_tolerance": 1e-8,
        "benefit_tolerance": 0.0,
        "b_penalty_lambda": 0.02,
        "b_label_weights": {"B_col_1": 1.0, "B_col_2": 2.0},
        "safe_threshold_scales_by_phase": {
            "protected": 0.75,
            "ramp": 1.0,
            "full": 1.25,
        },
        "benefit_threshold_offsets_by_phase": {
            "protected": 0.0010,
            "ramp": 0.0005,
            "full": 0.0,
        },
        "gain_drift_thresholds_by_phase": {
            "protected": 0.10,
            "ramp": 0.15,
            "full": 0.22,
        },
        "fallback_on_candidate_solve_failure": True,
        "store_executed_action_in_replay": True,
        "log_policy_candidate_and_executed": True,
    }


def _copy_behavioral_cloning_defaults(
    enabled=False,
    *,
    target_mode="nominal_only",
    lambda_bc_start=0.1,
    lambda_bc_end=0.0,
    decay_mode="exp",
    active_subepisodes=10,
    start_after_warm_start=True,
    coordinate_weights=None,
    label_weight_overrides=None,
    action_gap_tolerance=0.0,
    release_gate=None,
    tail_anchor=None,
    handoff=None,
):
    if release_gate is None:
        release_gate = {
            "enabled": False,
            "window_subepisodes": 1,
            "mean_action_gap_max": 0.25,
            "max_coordinate_gap_max": 0.20,
            "min_window_fraction": 1.0,
        }
    if handoff is None:
        handoff = {
            "enabled": False,
            "mode": "raw_action_blend",
            "start_authority": 1.0,
            "end_authority": 1.0,
            "active_subepisodes": 0,
        }
    return {
        "enabled": bool(enabled),
        "target_mode": str(target_mode),
        "lambda_bc_start": float(lambda_bc_start),
        "lambda_bc_end": float(lambda_bc_end),
        "decay_mode": str(decay_mode),
        "active_subepisodes": int(active_subepisodes),
        "start_after_warm_start": bool(start_after_warm_start),
        "log_diagnostics": True,
        "coordinate_weights": None if coordinate_weights is None else np.asarray(coordinate_weights, float).copy(),
        "label_weight_overrides": {} if label_weight_overrides is None else dict(label_weight_overrides),
        "action_gap_tolerance": float(action_gap_tolerance),
        "release_gate": dict(release_gate),
        "tail_anchor": None if tail_anchor is None else dict(tail_anchor),
        "handoff": dict(handoff),
    }


def _copy_protected_bc_defaults(*, target_mode, action_gap_tolerance=0.0):
    return _copy_behavioral_cloning_defaults(
        enabled=True,
        target_mode=target_mode,
        lambda_bc_start=1.0,
        lambda_bc_end=0.05,
        decay_mode="exp",
        active_subepisodes=10,
        start_after_warm_start=False,
        action_gap_tolerance=action_gap_tolerance,
        release_gate={
            "enabled": False,
            "window_subepisodes": 1,
            "mean_action_gap_max": 0.25,
            "max_coordinate_gap_max": 0.20,
            "min_window_fraction": 1.0,
        },
        handoff={
            "enabled": True,
            "mode": "raw_action_blend",
            "start_authority": 0.1,
            "end_authority": 1.0,
            "active_subepisodes": 10,
        },
    )


def _copy_td3_authority_ramp_defaults(kind):
    kind = str(kind).strip().lower()
    if kind == "weights":
        return {
            "enabled": False,
            "mode": "multiplier_deviation_cap",
            "units": "physical_multiplier_deviation_from_identity",
            "start_cap": 0.05,
            "end_cap": 0.25,
            "protected_subepisodes": 0,
            "ramp_subepisodes": 30,
            "diagnostic_release_gate_only": True,
        }
    if kind == "residual":
        return {
            "enabled": False,
            "mode": "residual_delta_u_cap",
            "units": "scaled_input_delta",
            "start_cap": 0.005,
            "end_cap": 0.02,
            "protected_subepisodes": 0,
            "ramp_subepisodes": 30,
            "diagnostic_release_gate_only": True,
        }
    if kind == "markov":
        return {
            "enabled": False,
            "mode": "z_safety_live_release",
            "units": "z_safety_controls_authority",
            "start_cap": 0.0,
            "end_cap": 0.0,
            "protected_subepisodes": 0,
            "ramp_subepisodes": 1,
            "diagnostic_release_gate_only": True,
        }
    raise ValueError(f"Unknown TD3 authority ramp kind: {kind}")


def _copy_active_td3_agent_defaults():
    return {
        "actor_hidden": list(DISTILLATION_DEFAULT_ACTOR_HIDDEN),
        "critic_hidden": list(DISTILLATION_DEFAULT_CRITIC_HIDDEN),
        **_copy_active_replay_defaults(),
        "gamma": DISTILLATION_DEFAULT_GAMMA,
        "n_step": 1,
        "multistep_mode": "one_step",
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 2,
        **_copy_active_td3_noise_defaults(),
        "max_action": 1.0,
        "tau": 0.005,
        "actor_freeze": 0,
        "loss_type": "huber",
    }


def _copy_active_sac_agent_defaults():
    return {
        "actor_hidden": list(DISTILLATION_DEFAULT_ACTOR_HIDDEN),
        "critic_hidden": list(DISTILLATION_DEFAULT_CRITIC_HIDDEN),
        **_copy_active_replay_defaults(),
        "gamma": DISTILLATION_DEFAULT_GAMMA,
        "n_step": 1,
        "multistep_mode": "one_step",
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "alpha_lr": 1e-4,
        "batch_size": 128,
        "grad_clip_norm": 10.0,
        "init_alpha": 0.01,
        "learn_alpha": True,
        "target_entropy": "auto_negative_action_dim",
        "target_update": "soft",
        "tau": 0.005,
        "hard_update_interval": 10_000,
        "activation": "relu",
        "use_layernorm": False,
        "dropout": 0.0,
        "max_action": 1.0,
        "use_adamw": True,
        "actor_freeze": 0,
        "alpha_freeze": "actor_freeze",
        "actor_q_mode": "min",
        "loss_type": "huber",
    }


def _copy_active_td7_agent_defaults():
    replay_defaults = _copy_active_replay_defaults()
    replay_defaults["replay_alpha"] = 0.4
    return {
        "actor_hidden": list(DISTILLATION_DEFAULT_ACTOR_HIDDEN),
        "critic_hidden": list(DISTILLATION_DEFAULT_CRITIC_HIDDEN),
        "encoder_hidden": list(DISTILLATION_DEFAULT_CRITIC_HIDDEN),
        "zs_dim": 256,
        **replay_defaults,
        "gamma": DISTILLATION_DEFAULT_GAMMA,
        "n_step": 1,
        "multistep_mode": "one_step",
        "lambda_value": None,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "encoder_lr": 1e-4,
        "batch_size": 128,
        "grad_clip_norm": 10.0,
        "policy_delay": 2,
        "target_update_rate": 250,
        "target_policy_smoothing_noise_std": 0.2,
        "noise_clip": 0.5,
        "max_action": 1.0,
        "actor_activation": "relu",
        "critic_activation": "elu",
        "encoder_activation": "elu",
        "use_layernorm": False,
        "dropout": 0.0,
        "std_start": 0.2,
        "std_end": 0.02,
        "std_decay_rate": 0.99995,
        "std_decay_steps": 100_000,
        "std_decay_mode": "exp",
        "bc_lambda_scale": 1.0,
        "actor_freeze": 0,
        "use_adamw": False,
        "min_priority": 1.0,
        "use_checkpoints": False,
    }


def _copy_td3_priority_fallback_defaults(enabled=True):
    return {
        "enabled": bool(enabled),
        "protected_subepisodes": 5,
        "ramp_subepisodes": 10,
        "score_hard_min": None,
        "gain_drift_max": 0.10,
        "cost_caps": {
            "protected": {"absolute": 0.02, "relative": 5.0},
            "ramp": {"absolute": 0.05, "relative": 20.0},
            "full": {"absolute": 0.10, "relative": 50.0},
        },
        "authority_ramp": {
            "enabled": bool(enabled),
            "protected_scale": 0.25,
            "ramp_start_scale": 0.25,
            "ramp_end_scale": 1.0,
            "full_scale": 1.0,
        },
        "reward_probation": {
            "enabled": False,
            "reference_warm_episodes": 3,
            "collapse_threshold": 5.0,
            "cooldown_subepisodes": 2,
            "cooldown_scale": 0.25,
        },
    }


def _copy_mismatch_defaults():
    return {
        "mismatch_clip": 3.0,
        "base_state_norm_mode": "fixed_minmax",
        "base_state_running_norm_clip": 10.0,
        "base_state_running_norm_eps": 1e-8,
        "innovation_scale_mode": "band_ref",
        "innovation_scale_ref": None,
        "tracking_scale_mode": "eta_band",
        "tracking_eta_tol": 0.3,
        "tracking_scale_floor": None,
        "tracking_scale_floor_mode": "half_eta_band_ref",
        "mismatch_feature_transform_mode": "signed_log",
        "mismatch_transform_tanh_scale": 3.0,
        "mismatch_transform_post_clip": None,
        "observer_update_alignment": "legacy_previous_measurement",
    }


def _copy_residual_authority_defaults(action_dim):
    return {
        "append_rho_to_state": False,
        "authority_use_rho": True,
        "authority_beta_res": np.full(int(action_dim), 0.3, dtype=float),
        "authority_du0_res": np.full(int(action_dim), 0.003, dtype=float),
        "authority_eta_tol": 0.3,
        "authority_rho_floor": 0.2,
        "authority_rho_power": 1.0,
        "rho_mapping_mode": "exp_raw_tracking",
        "authority_rho_k": 0.55,
        "residual_zero_deadband_enabled": True,
        "residual_zero_tracking_raw_threshold": 0.1,
        "residual_zero_innovation_raw_threshold": 0.1,
    }


# -----------------------------------------------------------------------------
# Distillation notebook defaults
# -----------------------------------------------------------------------------
# This file is the notebook-facing source of truth for the distillation case.
# Every active distillation notebook should read its editable defaults from
# here. Change the dictionaries below if you want those defaults to propagate
# into the notebooks automatically.
#
# Parameter guidance:
# - String controls document the valid option set inline.
# - Numeric defaults mirror the current unified/archived study settings.
# - Arrays are stored in physical plant units unless a comment says otherwise.
# -----------------------------------------------------------------------------

DISTILLATION_COMMON_PATH_DEFAULTS = {
    # Canonical folder overrides:
    #   None -> use Distillation/Data and Distillation/Results
    #   Path/string -> redirect a notebook to another data or result root
    "data_dir_override": None,
    "results_dir_override": None,
    # Output naming / baseline overrides:
    #   None -> use the notebook-family default path/prefix
    #   Path/string -> force a custom saved name/location
    "result_prefix_override": None,
    "compare_prefix_override": None,
    "baseline_mpc_path_override": None,
    "baseline_save_path_override": None,
}

DISTILLATION_COMMON_DISPLAY_DEFAULTS = {
    # STYLE_PROFILE options:
    #   "hybrid" -> default research/debug mix
    #   "paper"  -> cleaner compact export styling
    #   "debug"  -> most verbose diagnostics
    "style_profile": "hybrid",
    # SAVE_PDF:
    #   False -> PNG only
    #   True  -> PNG and PDF
    "save_pdf": False,
}

DISTILLATION_COMMON_OVERRIDE_DEFAULTS = {
    # Leave these as None to use the notebook-family run-profile defaults.
    "n_tests_override": None,
    "set_points_len_override": None,
    "warm_start_override": None,
    "test_cycle_override": None,
    "plot_start_episode_override": None,
    "compare_start_episode_override": None,
}

DISTILLATION_ASPEN_DEFAULTS = {
    # ASPEN_PRESET:
    #   "default" -> use the family/profile mapping from systems.distillation.config
    #   integer/int-like string -> use C2S_SS_simulation{N}.dynf
    "aspen_preset": "default",
    # Manual path overrides:
    #   None -> resolve from ASPEN_PRESET/family
    #   Path/string -> use exactly this dynf/snaps path
    "aspen_path_override": None,
    "snaps_path_override": None,
    "aspen_root_override": None,
    # Visible Aspen window:
    #   True  -> keep Aspen visible
    #   False -> run hidden/background if Aspen permits it
    "distillation_visible": True,
}

DISTILLATION_SYSTEM_SETUP = {
    "delta_t_hours": float(DELTA_T_HOURS),
    "nominal_conditions": np.asarray(DISTILLATION_NOMINAL_CONDITIONS, float).copy(),
    "ss_inputs": np.asarray(DISTILLATION_SS_INPUTS, float).copy(),
    "ss_inputs_system_id": np.asarray(DISTILLATION_SS_INPUTS_SYSTEM_ID, float).copy(),
    "input_bounds": {
        "u_min": np.asarray(DISTILLATION_INPUT_BOUNDS["u_min"], float).copy(),
        "u_max": np.asarray(DISTILLATION_INPUT_BOUNDS["u_max"], float).copy(),
    },
    "setpoint_range_phys": np.asarray(DISTILLATION_SETPOINT_RANGE_PHYS, float).copy(),
    # Use one shared supervisory setpoint pair across the baseline and RL
    # notebooks so all distillation studies compare against the same targets.
    "rl_setpoints_phys": np.asarray(DISTILLATION_RL_SETPOINTS_PHYS, float).copy(),
    "combined_setpoints_phys": np.asarray(DISTILLATION_COMBINED_SETPOINTS_PHYS, float).copy(),
    "observer_poles": np.asarray(DISTILLATION_OBSERVER_POLES, float).copy(),
}

DISTILLATION_SYSTEM_IDENTIFICATION_DEFAULTS = {
    # RUN_NEW_EXPERIMENTS:
    #   True  -> rerun Aspen step tests and regenerate the canonical files
    #   False -> reuse the stored Distillation/Data CSVs and rebuild from them
    "run_new_experiments": False,
    # USE_RHP_ZERO:
    #   True  -> keep right-half-plane-zero handling enabled
    #   False -> disable it for alternate identification studies
    "use_rhp_zero": True,
    "show_fopdt_plots": True,
    "show_validation_plots": True,
    "carry_forward_min_max_name": "min_max_states.pickle",
    "step_tests": [
        {"step_channel": 0, "step_value": -40000.0, "save_filename": "Reflux.csv"},
        {"step_channel": 1, "step_value": -15.0, "save_filename": "Reboiler.csv"},
    ],
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_BASELINE_DEFAULTS = {
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",  # "none" | "ramp" | "fluctuation"
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    "n_tests_override": None,
    "set_points_len_override": None,
    "test_cycle_override": None,
    "plot_start_episode_override": None,
    "run_profiles": deepcopy(DISTILLATION_BASELINE_RUN_PROFILES),
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "use_shifted_mpc_warm_start": False,
        # Distillation baseline disturbances are injected through the explicit
        # schedule/stepper path, so these remain neutral placeholders.
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_HORIZON_STANDARD_DEFAULTS = {
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    "state_mode": "mismatch",  # "standard" | "mismatch"
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "episode_defaults": {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False]},
    "post_warm_start_action_freeze_subepisodes": 3,
    "controller": {
        "predict_grid": list(HORIZON_PREDICT_GRID),
        "control_grid": list(HORIZON_CONTROL_GRID),
        "decision_interval": 4,
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "horizon_safety": {
        "enabled": False,
        "release_filter": {
            "enabled": False,
            "protected": {
                "subepisodes": 3,
                "predict_min": 4,
                "predict_max": 12,
                "control_min": 2,
                "control_max": 6,
            },
            "ramp": {
                "end_subepisode": 10,
                "predict_min": 4,
                "predict_max": 14,
                "control_min": 2,
                "control_max": 10,
            },
        },
        "reward_probation": {
            "enabled": False,
            "reference_warm_episodes": 3,
            "collapse_threshold": 5.0,
            "cooldown_subepisodes": 2,
        },
        "shadow_default_mpc": {
            "enabled": False,
            "diagnostic_stride": 4,
        },
    },
    "agent": {
        "hidden_layers": list(DISTILLATION_DEFAULT_DQN_HIDDEN),
        **_copy_active_replay_defaults(),
        "gamma": DISTILLATION_DEFAULT_GAMMA,
        "n_step": 1,  # Positive integer. Keep 1 for the baseline; common DDQN ablations use 3.
        "multistep_mode": "one_step",  # Options: "one_step" | "n_step" | "lambda" | "retrace"
        "lambda_value": 0.9,
        "lr": 1e-4,
        "batch_size": 128,
        "grad_clip_norm": 10.0,
        "double_dqn": True,
        "target_update": "soft",
        "tau": 0.01,
        "hard_update_interval": 10_000,
        "activation": "relu",
        "use_layer_norm": False,
        "dropout": 0.0,
        "target_combine": "q1",
        "exploration_mode": "epsilon",
        "noisy_sigma_init": 0.5,
        "loss_type": "huber",
        "eps_start": 0.2,
        "eps_end": 0.02,
        "eps_decay_rate": 0.99999,
        "eps_decay_mode": "linear",
        "eps_decay_steps": 50_000,
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}
DISTILLATION_HORIZON_STANDARD_DEFAULTS["run_profiles"] = deepcopy(DISTILLATION_HORIZON_RUN_PROFILES)

DISTILLATION_HORIZON_DUELING_DEFAULTS = {
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    "state_mode": "mismatch",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["run_profiles"]),
    "episode_defaults": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["episode_defaults"]),
    "post_warm_start_action_freeze_subepisodes": DISTILLATION_HORIZON_STANDARD_DEFAULTS[
        "post_warm_start_action_freeze_subepisodes"
    ],
    "controller": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["controller"]),
    "horizon_safety": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["horizon_safety"]),
    "agent": {
        "seed": 7,
        "hidden_layers": list(DISTILLATION_DEFAULT_DQN_HIDDEN),
        **_copy_active_replay_defaults(),
        "gamma": DISTILLATION_DEFAULT_GAMMA,
        "n_step": 1,
        "multistep_mode": "n_step",
        "lambda_value": 0.9,
        "lr": 1e-4,
        "batch_size": 128,
        "grad_clip_norm": 10.0,
        "double_dqn": True,
        "target_update": "hard",
        "tau": 0.005,
        "hard_update_interval": 2_000,
        "activation": "relu",
        "use_layer_norm": False,
        "dropout": 0.0,
        "target_combine": "q1",
        "exploration_mode": "epsilon",
        "noisy_sigma_init": 0.5,
        "loss_type": "huber",
        "eps_start": 0.20,
        "eps_end": 0.02,
        "eps_decay_rate": 0.99995,
        "eps_decay_mode": "linear",
        "eps_decay_steps": 50_000,
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_MATRIX_DEFAULTS = {
    "agent_kind": "td3",  # "td3" | "sac"
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    "state_mode": "mismatch",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    # Distillation scalar matrix default: keep the wide A/B search,
    # retain the protected Step 2 release cap, keep the post-warm-start
    # action/actor freeze enabled, disable live Step 3 logic, and use the
    # conservative Step 4G BC schedule.
    "behavioral_cloning": _copy_behavioral_cloning_defaults(
        enabled=True,
        lambda_bc_start=0.3,
        active_subepisodes=20,
    ),
    "run_profiles": deepcopy(DISTILLATION_MATRIX_RUN_PROFILES),
    "post_warm_start_action_freeze_subepisodes": 5,
    "post_warm_start_actor_freeze_subepisodes": 5,
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "decision_interval": 1,
        "recalculate_observer_on_matrix_change": True,  # Recompute the observer on live matrix updates for the default every-step distillation matrix reruns.
        "recalculate_observer_each_step": True,  # Force an observer redesign each MPC step, even if the executed assisted model repeats.
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef_by_agent": {
            key: np.asarray(value["low"], float).copy() for key, value in MATRIX_MULTIPLIER_BOUNDS.items()
        },
        "high_coef_by_agent": {
            key: np.asarray(value["high"], float).copy() for key, value in MATRIX_MULTIPLIER_BOUNDS.items()
        },
        "offline_multiplier_diagnostics": _copy_offline_multiplier_diagnostic_defaults(enabled=True),
        "release_protected_advisory_caps": _copy_release_protected_advisory_cap_defaults(enabled=True),
        "mpc_acceptance_fallback": _copy_mpc_acceptance_fallback_defaults(enabled=False),
        "mpc_dual_cost_shadow": _copy_mpc_dual_cost_shadow_defaults(enabled=False),
        "mpc_usefulness_gate": _copy_mpc_usefulness_gate_defaults(enabled=False),
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "td3_agent": {
        "actor_hidden": list(DISTILLATION_DEFAULT_ACTOR_HIDDEN),
        "critic_hidden": list(DISTILLATION_DEFAULT_CRITIC_HIDDEN),
        **_copy_replay_defaults(),
        "gamma": DISTILLATION_DEFAULT_GAMMA,
        "n_step": 1,  # Positive integer. Typical TD3 studies here use 1, 3, or 5.
        "multistep_mode": "one_step",  # Options: "one_step" | "n_step" | "lambda"
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 2,
        "target_policy_smoothing_noise_std": 0.01,
        "noise_clip": 0.2,
        "max_action": 1.0,
        "tau": 0.005,
        "std_start": 0.01,
        "std_end": 0.0,
        "param_noise_std_start": 0.01,
        "param_noise_std_end": 0.01,
        "std_decay_rate": 0.99995,
        "std_decay_mode": "exp",
        "actor_freeze": 0,
        "exploration_mode": "gaussian",
        "loss_type": "huber",
        "param_noise_resample_interval": 4,
    },
    "sac_agent": {
        "actor_hidden": list(DISTILLATION_DEFAULT_ACTOR_HIDDEN),
        "critic_hidden": list(DISTILLATION_DEFAULT_CRITIC_HIDDEN),
        **_copy_replay_defaults(),
        "gamma": DISTILLATION_DEFAULT_GAMMA,
        "n_step": 1,  # Positive integer. SAC often uses 3-step as the first extension.
        "multistep_mode": "one_step",  # Options: "one_step" | "n_step" | "sac_n" | "lambda"
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "alpha_lr": 1e-4,
        "batch_size": 128,
        "grad_clip_norm": 10.0,
        "init_alpha": 0.01,
        "learn_alpha": True,
        "target_entropy": "auto_negative_action_dim",
        "target_update": "soft",
        "tau": 0.005,
        "hard_update_interval": 10_000,
        "activation": "relu",
        "use_layernorm": False,
        "dropout": 0.0,
        "max_action": 1.0,
        "use_adamw": True,
        "actor_freeze": 0,
        "alpha_freeze": "actor_freeze",
        "actor_q_mode": "min",
        "loss_type": "huber",
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_STRUCTURED_MATRIX_DEFAULTS = {
    "agent_kind": "td3",  # "td3" | "sac"
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    "state_mode": "mismatch",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    # Distillation structured matrix default: mirror the scalar live path,
    # including the protected Step 2 release cap, the post-warm-start
    # action/actor freeze, and the conservative small-noise TD3 setup,
    # while keeping the structured range family and BC rollout without
    # structured label weighting.
    "behavioral_cloning": _copy_behavioral_cloning_defaults(
        enabled=True,
        lambda_bc_start=0.3,
        active_subepisodes=20,
    ),
    "run_profiles": deepcopy(DISTILLATION_MATRIX_RUN_PROFILES),
    "post_warm_start_action_freeze_subepisodes": 5,
    "post_warm_start_actor_freeze_subepisodes": 5,
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "decision_interval": 1,
        "recalculate_observer_on_matrix_change": True,  # Recompute the observer on live structured-matrix updates for the default every-step distillation reruns.
        "recalculate_observer_each_step": True,  # Force an observer redesign each MPC step, even if the executed assisted model repeats.
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "update_family": "block",  # Options: "block" | "band". Block-lite is the primary first experiment.
        "range_profile": "wide",  # Options: "tight" | "default" | "wide". Wide is the active default for cross-system structured analysis.
        "a_low_override": DISTILLATION_DEFAULT_MULTIPLIER_LOW,  # Keep the A-side wide, with the high side capped by the analyzed distillation alpha limit.
        "a_high_override": min(DISTILLATION_DEFAULT_MULTIPLIER_HIGH, DISTILLATION_MATRIX_ALPHA_UPPER_CAP),
        "b_low_override": DISTILLATION_DEFAULT_MULTIPLIER_LOW,  # Scalar or array override for B-side structured bounds.
        "b_high_override": DISTILLATION_DEFAULT_MULTIPLIER_HIGH,  # Keep B-side wide for gain-authority studies.
        "offline_multiplier_diagnostics": _copy_offline_multiplier_diagnostic_defaults(enabled=True),
        "release_protected_advisory_caps": _copy_release_protected_advisory_cap_defaults(enabled=True),
        "mpc_acceptance_fallback": _copy_mpc_acceptance_fallback_defaults(enabled=False),
        "mpc_dual_cost_shadow": _copy_mpc_dual_cost_shadow_defaults(enabled=False),
        "mpc_usefulness_gate": _copy_mpc_usefulness_gate_defaults(enabled=False),
        "prediction_fallback_on_solve_failure": True,  # Use the shared structured-runner fallback instead of stopping on an assisted MPC solve failure.
        "block_group_count": 3,  # Positive integer. Used only when block_groups is None.
        "block_groups": None,  # Optional explicit 0-based physical-state partition.
        "band_offsets": [0, 1, 2],  # Non-negative offsets used in band mode. Must include 0.
        "log_spectral_radius": True,  # Options: False | True. True logs the physical-model spectral radius each step.
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "td3_agent": deepcopy(DISTILLATION_MATRIX_DEFAULTS["td3_agent"]),
    "sac_agent": deepcopy(DISTILLATION_MATRIX_DEFAULTS["sac_agent"]),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_MARKOV_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "behavioral_cloning": _copy_behavioral_cloning_defaults(
        enabled=True,
        target_mode="ls_action",
        lambda_bc_start=1.0,
        lambda_bc_end=0.05,
        decay_mode="exp",
        active_subepisodes=10,
        start_after_warm_start=False,
        release_gate={
            "enabled": True,
            "diagnostic_only": True,
            "window_subepisodes": 1,
            "mean_action_gap_max": 0.25,
            "max_coordinate_gap_max": 0.20,
            "min_window_fraction": 1.0,
        },
        handoff={
            "enabled": True,
            "mode": "raw_action_blend",
            "start_authority": 0.1,
            "end_authority": 1.0,
            "active_subepisodes": 10,
            "start_after_warm_start": True,
        },
    ),
    "run_profiles": deepcopy(DISTILLATION_MARKOV_RUN_PROFILES),
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "decision_interval": 1,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
        "basis_family": "io_pair_gain",
        "z_bound": 0.04,
        "z_safety": {
            "enabled": True,
            "protected_cap": 0.025,
            "ramp_start_cap": 0.03,
            "ramp_end_cap": 0.04,
            "full_cap": 0.04,
            "probation_cap": 0.025,
            "vector_norm_cap": {
                "enabled": True,
                "max_norm": 0.06,
            },
        },
        "prediction_window": 20,
        "lambda_z": 1.0e-3,
        "s_pred_min": 1.0e-6,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.10,
        "nominal_cost_absolute_tol": 1.0e-8,
        "nominal_solver_mode": "lifted_g0_prototype",
        "run_adaptive_ls": True,
        "run_live_corrected_mpc": True,
        "run_rl_proposal": True,
        "rl_fallback_to_ls": True,
        "force_td3_execute": False,
        "td3_authority_ramp": _copy_td3_authority_ramp_defaults("markov"),
        "rl_store_executed_action_in_replay": True,
        "td3_priority_fallback": _copy_td3_priority_fallback_defaults(enabled=True),
        "rl_save_agent_checkpoint": True,
        "debug_validate_lifted": False,
        "debug_run_shadow_ls": False,
    },
    "td3_agent": _copy_active_td3_agent_defaults(),
    "sac_agent": _copy_active_sac_agent_defaults(),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_WEIGHT_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    "state_mode": "mismatch",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_WEIGHT_RUN_PROFILES),
    "post_warm_start_action_freeze_subepisodes": 0,
    "post_warm_start_actor_freeze_subepisodes": 0,
    "behavioral_cloning": _copy_behavioral_cloning_defaults(
        enabled=True,
        target_mode="nominal_only",
        lambda_bc_start=1.0,
        lambda_bc_end=0.05,
        decay_mode="exp",
        active_subepisodes=10,
        start_after_warm_start=False,
        release_gate={
            "enabled": True,
            "diagnostic_only": True,
            "window_subepisodes": 1,
            "mean_action_gap_max": 0.25,
            "max_coordinate_gap_max": 0.20,
            "min_window_fraction": 1.0,
        },
        handoff={
            "enabled": True,
            "mode": "raw_action_blend",
            "start_authority": 0.1,
            "end_authority": 1.0,
            "active_subepisodes": 10,
            "start_after_warm_start": True,
        },
    ),
    "td3_authority_ramp": {
        **_copy_td3_authority_ramp_defaults("weights"),
        "enabled": True,
        "start_cap": 0.10,
        "end_cap": 1.00,
        "ramp_subepisodes": 30,
        "diagnostic_release_gate_only": False,
    },
    "weight_safety": {
        "enabled": True,
        "reward_probation": {
            "enabled": False,
            "reference_warm_episodes": 3,
            "collapse_threshold": 5.0,
            "cooldown_subepisodes": 2,
            "cooldown_multiplier_cap": 0.10,
        },
        "fallback_to_identity_on_nonfinite": True,
        "fallback_to_identity_on_solve_failure": True,
        "shadow_identity_mpc": {
            "enabled": True,
            "diagnostic_stride": 5,
        },
    },
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.asarray(WEIGHT_MULTIPLIER_BOUNDS["low"], float).copy(),
        "high_coef": np.asarray(WEIGHT_MULTIPLIER_BOUNDS["high"], float).copy(),
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "td3_agent": _copy_active_td3_agent_defaults(),
    "sac_agent": _copy_active_sac_agent_defaults(),
    "supervisor_gate": {
        "score_uncertainty_weight": 0.5,
        "score_previous_action_weight": 0.01,
        "score_supervisor_action_weight": 0.05,
        "advantage_margin": 0.5,
        "default_to_supervisor": True,
        "actor_q_mode": "mean",
        "supervisor_bc_weight": 0.0,
        "supervisor_bc_temperature": 1.0,
        "smooth_action_weight": 0.0,
        "detach_supervisor_weight": True,
        "enable_supervisor_actor_loss": False,
        "min_train_steps_before_policy_gate": 0,
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_RESIDUAL_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    "state_mode": "mismatch",
    **_copy_residual_authority_defaults(action_dim=2),
    "residual_authority_enabled": False,
    "authority_use_rho": False,
    "use_rho_authority": False,
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_RESIDUAL_RUN_PROFILES),
    "post_warm_start_action_freeze_subepisodes": 0,
    "post_warm_start_actor_freeze_subepisodes": 0,
    "behavioral_cloning": _copy_behavioral_cloning_defaults(
        enabled=True,
        target_mode="nominal_only",
        lambda_bc_start=1.0,
        lambda_bc_end=0.05,
        decay_mode="exp",
        active_subepisodes=10,
        start_after_warm_start=False,
        action_gap_tolerance=1e-6,
        release_gate={
            "enabled": True,
            "diagnostic_only": True,
            "window_subepisodes": 1,
            "mean_action_gap_max": 0.25,
            "max_coordinate_gap_max": 0.20,
            "min_window_fraction": 1.0,
        },
        handoff={
            "enabled": True,
            "mode": "raw_action_blend",
            "start_authority": 0.1,
            "end_authority": 1.0,
            "active_subepisodes": 10,
            "start_after_warm_start": True,
        },
    ),
    "td3_authority_ramp": {
        **_copy_td3_authority_ramp_defaults("residual"),
        "enabled": True,
        "diagnostic_release_gate_only": False,
    },
    "residual_safety": {
        "enabled": True,
        "reward_probation": {
            "enabled": False,
            "reference_warm_episodes": 3,
            "collapse_threshold": 5.0,
            "cooldown_subepisodes": 2,
            "cooldown_residual_cap": 0.005,
        },
        "fallback_to_zero_on_nonfinite": True,
        "shadow_rho_authority": {
            "enabled": True,
        },
        "shadow_residual_deadband": {
            "enabled": True,
        },
        "shadow_direction_risk": {
            "enabled": True,
        },
        "early_release_guard": {
            "enabled": True,
            "post_warm_subepisodes": 20,
            "objective_relative_tolerance": 0.05,
            "objective_absolute_tolerance": 1e-8,
            "error_norm_tolerance": 0.05,
            "shrink_scales": [0.5, 0.25, 0.1, 0.0],
        },
    },
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.asarray(RESIDUAL_BOUNDS["low"], float).copy(),
        "high_coef": np.asarray(RESIDUAL_BOUNDS["high"], float).copy(),
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "td3_agent": _copy_active_td3_agent_defaults(),
    "supervisor_gate": {
        "score_uncertainty_weight": 0.5,
        "score_previous_action_weight": 0.01,
        "score_supervisor_action_weight": 0.05,
        "advantage_margin": 0.5,
        "default_to_supervisor": True,
        "actor_q_mode": "mean",
        "supervisor_bc_weight": 0.0,
        "supervisor_bc_temperature": 1.0,
        "smooth_action_weight": 0.0,
        "detach_supervisor_weight": True,
        "enable_supervisor_actor_loss": False,
        "min_train_steps_before_policy_gate": 0,
    },
    "td7_agent": _copy_active_td7_agent_defaults(),
    "sac_agent": _copy_active_sac_agent_defaults(),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_REIDENTIFICATION_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "disturb",
    "disturbance_profile": "fluctuation",
    "state_mode": "mismatch",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_REIDENTIFICATION_RUN_PROFILES),
    "post_warm_start_action_freeze_subepisodes": 5,
    "post_warm_start_actor_freeze_subepisodes": 5,
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "reidentification": {
        "basis_family": "lowrank_distillation",
        "id_component_mode": "AB",
        "observer_update_alignment": "legacy_previous_measurement",
        "candidate_guard_mode": "fro_only",
        "normalize_blend_extras": False,
        "blend_extra_clip": 1.0e6,
        "blend_residual_scale": 1.0e6,
        "log_theta_clipping": True,
        "id_solver": "ridge_closed_form",
        "rank_A": 5,
        "rank_B": 1,
        "offline_window": 80,
        "offline_stride": 80,
        "lambda_A_off": 1e-4,
        "lambda_B_off": 1e-3,
        "id_window": 160,
        "id_update_period": 20,
        "lambda_prev_A": 1e-2,
        "lambda_prev_B": 5e-1,
        "lambda_0_A": 1e-4,
        "lambda_0_B": 5e-3,
        "theta_low_A": -0.15,
        "theta_high_A": 0.15,
        "theta_low_B": -0.04,
        "theta_high_B": 0.04,
        "delta_A_max": 0.10,
        "delta_B_max": 0.10,
        "eta_tau_A": 1.0,
        "eta_tau_B": 1.0,
        "freeze_identification_during_warm_start": True,
        "guard_validation_fraction": 0.25,
        "guard_min_validation_samples": 32,
        "guard_min_train_samples": 80,
        "max_theta_clipped_fraction": 0.05,
        "max_condition_number": 3.0e4,
        "max_validation_residual_ratio": 0.995,
        "max_full_residual_ratio": 1.0,
        "blend_validity_mode": "off",
        "blend_validity_scale_floor": 1.0,
        "blend_validity_residual_soft": 0.04,
        "blend_validity_residual_hard": 0.12,
        "blend_validity_clipped_soft": 0.02,
        "blend_validity_clipped_hard": 0.06,
        "blend_validity_condition_soft": 1.5e4,
        "blend_validity_condition_hard": 3.0e4,
        "blend_validity_fallback_scale": 1.0,
        "blend_validity_invalid_candidate_scale": 1.0,
        "observer_refresh_enabled": False,
        "observer_refresh_every_episodes": 10,
        "rho_obs": 0.25,
        "force_eta_constant": None,
        "disable_identification": False,
    },
    "td3_agent": deepcopy(DISTILLATION_MATRIX_DEFAULTS["td3_agent"]),
    "sac_agent": deepcopy(DISTILLATION_MATRIX_DEFAULTS["sac_agent"]),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

def resolve_distillation_agent_kind(family: str, agent_mode: str) -> str:
    """Return the active distillation agent kind for an SG/plain mode."""
    family_key = str(family).strip().lower()
    mode = str(agent_mode).strip().lower().replace("-", "_")
    if mode in {"without_sg", "no_sg"}:
        mode = "plain"
    if mode not in {"sg", "plain"}:
        raise ValueError("agent_mode must be 'sg' or 'plain'.")

    if family_key in {"horizon", "horizon_standard"}:
        return "sg_dqn" if mode == "sg" else "dqn"
    if family_key in {"markov", "weights", "residual"}:
        return "sg_td3" if mode == "sg" else "td3"
    raise ValueError("family must be one of 'horizon', 'markov', 'weights', or 'residual'.")


def resolve_distillation_combined_agent_kinds(combined_agent_mode: str) -> dict:
    """Return the active combined runner agent kinds for an SG/plain mode."""
    mode = str(combined_agent_mode).strip().lower().replace("-", "_")
    if mode in {"without_sg", "no_sg"}:
        mode = "plain"
    if mode == "sg":
        return {
            "horizon_agent_kind": "sg_dqn",
            "markov_agent_kind": "sg_td3",
            "weights_agent_kind": "sg_td3",
            "residual_agent_kind": "sg_td3",
        }
    if mode == "plain":
        return {
            "horizon_agent_kind": "dqn",
            "markov_agent_kind": "td3",
            "weights_agent_kind": "td3",
            "residual_agent_kind": "td3",
        }
    raise ValueError("combined_agent_mode must be 'sg' or 'plain'.")


def _disable_behavioral_cloning(nb: dict) -> None:
    bc_cfg = deepcopy(nb.get("behavioral_cloning", {}))
    bc_cfg["enabled"] = False
    for key in ("release_gate", "handoff", "tail_anchor"):
        if not isinstance(bc_cfg.get(key), dict):
            bc_cfg[key] = {}
    bc_cfg["release_gate"]["enabled"] = False
    bc_cfg["release_gate"]["diagnostic_only"] = False
    bc_cfg["handoff"]["enabled"] = False
    bc_cfg["handoff"]["start_authority"] = 1.0
    bc_cfg["handoff"]["end_authority"] = 1.0
    bc_cfg["handoff"]["active_subepisodes"] = 0
    bc_cfg["tail_anchor"]["enabled"] = False
    nb["behavioral_cloning"] = bc_cfg


def _apply_active_runner_defaults() -> None:
    horizon_gate = {
        "advantage_margin": 0.0,
        "default_to_supervisor": True,
        "min_train_steps_before_policy_gate": 0,
    }
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["agent_mode"] = "sg"
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["agent_kind"] = "sg_dqn"
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["run_profiles"] = deepcopy(DISTILLATION_ACTIVE_HORIZON_RUN_PROFILES)
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["supervisor_gate"] = deepcopy(horizon_gate)
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["post_warm_start_action_freeze_subepisodes"] = 3
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["controller"]["predict_grid"] = list(range(6, 12))
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["controller"]["control_grid"] = list(range(3, 12))
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["agent"]["eps_decay_steps"] = 18_600
    DISTILLATION_HORIZON_STANDARD_DEFAULTS["agent"]["supervisor_gate"] = deepcopy(horizon_gate)

    DISTILLATION_MARKOV_DEFAULTS["agent_mode"] = "sg"
    DISTILLATION_MARKOV_DEFAULTS["agent_kind"] = "sg_td3"
    DISTILLATION_MARKOV_DEFAULTS["state_mode"] = "mismatch"
    DISTILLATION_MARKOV_DEFAULTS["run_profiles"] = deepcopy(DISTILLATION_ACTIVE_MARKOV_RUN_PROFILES)
    DISTILLATION_MARKOV_DEFAULTS["post_warm_start_action_freeze_subepisodes"] = 3
    DISTILLATION_MARKOV_DEFAULTS["post_warm_start_actor_freeze_subepisodes"] = 3
    DISTILLATION_MARKOV_DEFAULTS["markov_supervisor_mode"] = "ls_else_mpc"
    DISTILLATION_MARKOV_DEFAULTS["markov_live_safety_mode"] = "shadow_only"
    active_bc_handoff = deepcopy(DISTILLATION_MARKOV_DEFAULTS.get("behavioral_cloning", {}).get("handoff", {}))
    _disable_behavioral_cloning(DISTILLATION_MARKOV_DEFAULTS)
    markov_ctrl = DISTILLATION_MARKOV_DEFAULTS["controller"]
    active_z_safety = deepcopy(markov_ctrl.get("z_safety", {}))
    active_priority = deepcopy(markov_ctrl.get("td3_priority_fallback", {}))
    markov_ctrl["rl_fallback_to_ls"] = False
    markov_ctrl["force_td3_respects_warm_start"] = True
    markov_ctrl["markov_supervisor_mode"] = "ls_else_mpc"
    markov_ctrl["markov_live_safety_mode"] = "shadow_only"
    markov_ctrl["z_bound"] = 0.05
    markov_ctrl["z_safety"] = {"enabled": False}
    markov_ctrl["td3_priority_fallback"] = {"enabled": False}
    markov_ctrl["td3_authority_ramp"] = {"enabled": False}
    markov_ctrl["markov_shadow_safety"] = {
        "enabled": True,
        "compute_ls_candidate": False,
        "z_safety": active_z_safety,
        "td3_priority_fallback": active_priority,
        "bc_handoff": active_bc_handoff,
    }
    markov_td3 = DISTILLATION_MARKOV_DEFAULTS["td3_agent"]
    markov_td3["exploration_mode"] = "param_noise"
    markov_td3["param_noise_std_start"] = 0.05
    markov_td3["param_noise_std_end"] = 0.02
    markov_td3["param_noise_resample_interval"] = 4
    DISTILLATION_MARKOV_DEFAULTS["supervisor_gate"] = {
        "score_uncertainty_weight": 0.5,
        "score_previous_action_weight": 0.01,
        "score_supervisor_action_weight": 0.02,
        "advantage_margin": 0.5,
        "default_to_supervisor": True,
        "actor_q_mode": "mean",
        "supervisor_bc_weight": 0.0,
        "supervisor_bc_temperature": 1.0,
        "smooth_action_weight": 0.0,
        "detach_supervisor_weight": True,
        "enable_supervisor_actor_loss": False,
        "min_train_steps_before_policy_gate": 0,
    }

    DISTILLATION_WEIGHT_DEFAULTS["agent_mode"] = "sg"
    DISTILLATION_WEIGHT_DEFAULTS["agent_kind"] = "sg_td3"
    DISTILLATION_WEIGHT_DEFAULTS["run_profiles"] = deepcopy(DISTILLATION_ACTIVE_WEIGHT_RUN_PROFILES)
    DISTILLATION_WEIGHT_DEFAULTS["post_warm_start_action_freeze_subepisodes"] = 3
    DISTILLATION_WEIGHT_DEFAULTS["post_warm_start_actor_freeze_subepisodes"] = 3
    _disable_behavioral_cloning(DISTILLATION_WEIGHT_DEFAULTS)
    DISTILLATION_WEIGHT_DEFAULTS["td3_authority_ramp"]["enabled"] = False
    DISTILLATION_WEIGHT_DEFAULTS["td3_authority_ramp"]["diagnostic_release_gate_only"] = False
    DISTILLATION_WEIGHT_DEFAULTS["weight_safety"]["fallback_to_identity_on_solve_failure"] = False
    DISTILLATION_WEIGHT_DEFAULTS["weight_safety"]["reward_probation"]["enabled"] = False
    DISTILLATION_WEIGHT_DEFAULTS["weight_safety"]["shadow_identity_mpc"]["enabled"] = False
    weight_td3 = DISTILLATION_WEIGHT_DEFAULTS["td3_agent"]
    weight_td3["exploration_mode"] = "gaussian"
    weight_td3["std_start"] = 0.15
    weight_td3["std_end"] = 0.03
    DISTILLATION_WEIGHT_DEFAULTS["supervisor_gate"]["advantage_margin"] = 0.0
    DISTILLATION_WEIGHT_DEFAULTS["supervisor_gate"]["score_supervisor_action_weight"] = 0.01
    DISTILLATION_WEIGHT_DEFAULTS["supervisor_gate"]["enable_supervisor_actor_loss"] = False

    DISTILLATION_RESIDUAL_DEFAULTS["agent_mode"] = "sg"
    DISTILLATION_RESIDUAL_DEFAULTS["agent_kind"] = "sg_td3"
    DISTILLATION_RESIDUAL_DEFAULTS["run_profiles"] = deepcopy(DISTILLATION_ACTIVE_RESIDUAL_RUN_PROFILES)
    DISTILLATION_RESIDUAL_DEFAULTS["post_warm_start_action_freeze_subepisodes"] = 3
    DISTILLATION_RESIDUAL_DEFAULTS["post_warm_start_actor_freeze_subepisodes"] = 3
    DISTILLATION_RESIDUAL_DEFAULTS["residual_authority_enabled"] = False
    DISTILLATION_RESIDUAL_DEFAULTS["authority_use_rho"] = False
    DISTILLATION_RESIDUAL_DEFAULTS["use_rho_authority"] = False
    DISTILLATION_RESIDUAL_DEFAULTS["append_rho_to_state"] = False
    DISTILLATION_RESIDUAL_DEFAULTS["residual_zero_deadband_enabled"] = False
    _disable_behavioral_cloning(DISTILLATION_RESIDUAL_DEFAULTS)
    DISTILLATION_RESIDUAL_DEFAULTS["td3_authority_ramp"]["enabled"] = False
    DISTILLATION_RESIDUAL_DEFAULTS["td3_authority_ramp"]["diagnostic_release_gate_only"] = False
    residual_td3 = DISTILLATION_RESIDUAL_DEFAULTS["td3_agent"]
    residual_td3["exploration_mode"] = "param_noise"
    residual_td3["param_noise_std_start"] = 0.10
    residual_td3["param_noise_std_end"] = 0.02
    residual_td3["param_noise_resample_interval"] = 4
    residual_safety = DISTILLATION_RESIDUAL_DEFAULTS["residual_safety"]
    residual_safety["fallback_to_zero_on_nonfinite"] = True
    residual_safety["reward_probation"]["enabled"] = False
    for key in ("shadow_rho_authority", "shadow_residual_deadband", "shadow_direction_risk", "early_release_guard"):
        residual_safety[key]["enabled"] = False
    DISTILLATION_RESIDUAL_DEFAULTS["supervisor_gate"]["advantage_margin"] = 0.5
    DISTILLATION_RESIDUAL_DEFAULTS["supervisor_gate"]["score_supervisor_action_weight"] = 0.05
    DISTILLATION_RESIDUAL_DEFAULTS["supervisor_gate"]["enable_supervisor_actor_loss"] = False


_apply_active_runner_defaults()


def _copy_active_combined_run_profiles() -> dict:
    profiles = {}
    for (run_mode, profile), settings in DISTILLATION_COMBINED_RUN_PROFILES.items():
        combined_settings = dict(settings)
        combined_settings.update(
            {
                "result_prefix_template": f"distillation_combined_{{mode}}_{run_mode}_{profile}",
                "compare_prefix_template": f"distillation_compare_combined_{{mode}}_{run_mode}_{profile}",
                "compare_mode": run_mode,
            }
        )
        profiles[(run_mode, profile)] = combined_settings
    return profiles


def _build_active_combined_defaults() -> dict:
    horizon_ctrl = DISTILLATION_HORIZON_STANDARD_DEFAULTS["controller"]
    markov_ctrl = DISTILLATION_MARKOV_DEFAULTS["controller"]
    weights_ctrl = DISTILLATION_WEIGHT_DEFAULTS["controller"]
    residual_ctrl = DISTILLATION_RESIDUAL_DEFAULTS["controller"]
    resolved = resolve_distillation_combined_agent_kinds("sg")
    n_inputs = int(np.asarray(DISTILLATION_SYSTEM_SETUP["ss_inputs"], float).size)
    model_low, model_high = _distillation_disabled_matrix_bounds(n_inputs)

    return {
        "run_mode": "disturb",
        "disturbance_profile": "fluctuation",
        "combined_agent_mode": "sg",
        **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
        **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
        **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
        **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
        "enable_horizon": True,
        "horizon_agent_kind": resolved["horizon_agent_kind"],
        "horizon_state_mode": DISTILLATION_HORIZON_STANDARD_DEFAULTS["state_mode"],
        "enable_markov": True,
        "markov_agent_kind": resolved["markov_agent_kind"],
        "markov_state_mode": DISTILLATION_MARKOV_DEFAULTS["state_mode"],
        "enable_matrix": False,
        "matrix_agent_kind": "td3",
        "matrix_state_mode": "mismatch",
        "enable_weights": True,
        "weights_agent_kind": resolved["weights_agent_kind"],
        "weights_state_mode": DISTILLATION_WEIGHT_DEFAULTS["state_mode"],
        "enable_residual": True,
        "residual_agent_kind": resolved["residual_agent_kind"],
        "residual_state_mode": DISTILLATION_RESIDUAL_DEFAULTS["state_mode"],
        **_copy_residual_authority_defaults(action_dim=n_inputs),
        "residual_authority_enabled": DISTILLATION_RESIDUAL_DEFAULTS["residual_authority_enabled"],
        "append_rho_to_state": DISTILLATION_RESIDUAL_DEFAULTS["append_rho_to_state"],
        "authority_use_rho": DISTILLATION_RESIDUAL_DEFAULTS["authority_use_rho"],
        "use_rho_authority": DISTILLATION_RESIDUAL_DEFAULTS["use_rho_authority"],
        "residual_zero_deadband_enabled": DISTILLATION_RESIDUAL_DEFAULTS["residual_zero_deadband_enabled"],
        "run_profiles": _copy_active_combined_run_profiles(),
        "episode_defaults": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["episode_defaults"]),
        "horizon_post_warm_start_action_freeze_subepisodes": DISTILLATION_HORIZON_STANDARD_DEFAULTS[
            "post_warm_start_action_freeze_subepisodes"
        ],
        "td3_post_warm_start_action_freeze_subepisodes": 3,
        "td3_post_warm_start_actor_freeze_subepisodes": 3,
        "horizon_safety": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["horizon_safety"]),
        "weight_safety": deepcopy(DISTILLATION_WEIGHT_DEFAULTS["weight_safety"]),
        "residual_safety": deepcopy(DISTILLATION_RESIDUAL_DEFAULTS["residual_safety"]),
        "controller": {
            "decision_interval": horizon_ctrl["decision_interval"],
            "predict_grid": list(horizon_ctrl["predict_grid"]),
            "control_grid": list(horizon_ctrl["control_grid"]),
            "predict_h": horizon_ctrl["predict_h"],
            "cont_h": horizon_ctrl["cont_h"],
            "Q1_penalty": horizon_ctrl["Q1_penalty"],
            "Q2_penalty": horizon_ctrl["Q2_penalty"],
            "R1_penalty": horizon_ctrl["R1_penalty"],
            "R2_penalty": horizon_ctrl["R2_penalty"],
            "basis_family": markov_ctrl["basis_family"],
            "z_bound": markov_ctrl["z_bound"],
            "z_safety": deepcopy(markov_ctrl["z_safety"]),
            "prediction_window": markov_ctrl["prediction_window"],
            "lambda_z": markov_ctrl["lambda_z"],
            "s_pred_min": markov_ctrl["s_pred_min"],
            "gain_drift_max": markov_ctrl["gain_drift_max"],
            "nominal_cost_relative_tol": markov_ctrl["nominal_cost_relative_tol"],
            "nominal_cost_absolute_tol": markov_ctrl["nominal_cost_absolute_tol"],
            "run_adaptive_ls": markov_ctrl["run_adaptive_ls"],
            "run_live_corrected_mpc": markov_ctrl["run_live_corrected_mpc"],
            "run_rl_proposal": markov_ctrl["run_rl_proposal"],
            "rl_fallback_to_ls": markov_ctrl["rl_fallback_to_ls"],
            "force_td3_execute": markov_ctrl["force_td3_execute"],
            "force_td3_respects_warm_start": markov_ctrl.get("force_td3_respects_warm_start", False),
            "rl_store_executed_action_in_replay": markov_ctrl["rl_store_executed_action_in_replay"],
            "td3_priority_fallback": deepcopy(markov_ctrl["td3_priority_fallback"]),
            "markov_shadow_safety": deepcopy(markov_ctrl["markov_shadow_safety"]),
            "model_low": model_low,
            "model_high": model_high,
            "weights_low": weights_ctrl["low_coef"].copy(),
            "weights_high": weights_ctrl["high_coef"].copy(),
            "residual_low": residual_ctrl["low_coef"].copy(),
            "residual_high": residual_ctrl["high_coef"].copy(),
            **_copy_mismatch_defaults(),
            "use_shifted_mpc_warm_start": False,
            "nominal_qi": 0.0,
            "nominal_qs": 0.0,
            "nominal_ha": 0.0,
            "qi_change": 1.0,
            "qs_change": 1.0,
            "ha_change": 1.0,
        },
        "horizon_agent": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["agent"]),
        "markov_td3_agent": deepcopy(DISTILLATION_MARKOV_DEFAULTS["td3_agent"]),
        "weights_td3_agent": deepcopy(DISTILLATION_WEIGHT_DEFAULTS["td3_agent"]),
        "residual_td3_agent": deepcopy(DISTILLATION_RESIDUAL_DEFAULTS["td3_agent"]),
        "horizon_supervisor_gate": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["supervisor_gate"]),
        "markov_supervisor_gate": deepcopy(DISTILLATION_MARKOV_DEFAULTS["supervisor_gate"]),
        "weights_supervisor_gate": deepcopy(DISTILLATION_WEIGHT_DEFAULTS["supervisor_gate"]),
        "residual_supervisor_gate": deepcopy(DISTILLATION_RESIDUAL_DEFAULTS["supervisor_gate"]),
        "reward": _copy_reward_defaults(),
        "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
    }


DISTILLATION_COMBINED_DEFAULTS = _build_active_combined_defaults()


DISTILLATION_NOTEBOOK_DEFAULTS = {
    "system_identification": DISTILLATION_SYSTEM_IDENTIFICATION_DEFAULTS,
    "baseline": DISTILLATION_BASELINE_DEFAULTS,
    "horizon_standard": DISTILLATION_HORIZON_STANDARD_DEFAULTS,
    "markov": DISTILLATION_MARKOV_DEFAULTS,
    "weights": DISTILLATION_WEIGHT_DEFAULTS,
    "residual": DISTILLATION_RESIDUAL_DEFAULTS,
    "combined": DISTILLATION_COMBINED_DEFAULTS,
}


def get_distillation_notebook_defaults(family: str) -> dict:
    """
    Return a deep-copied parameter dictionary for the requested distillation
    notebook family so notebooks can mutate local settings safely.
    """

    key = str(family).strip().lower()
    if key not in DISTILLATION_NOTEBOOK_DEFAULTS:
        raise KeyError(f"Unknown distillation notebook family: {family}")
    return deepcopy(DISTILLATION_NOTEBOOK_DEFAULTS[key])


__all__ = [
    "DISTILLATION_DEFAULT_ACTOR_HIDDEN",
    "DISTILLATION_DEFAULT_CRITIC_HIDDEN",
    "DISTILLATION_DEFAULT_DQN_HIDDEN",
    "DISTILLATION_DEFAULT_GAMMA",
    "DISTILLATION_NOTEBOOK_DEFAULTS",
    "get_distillation_notebook_defaults",
    "resolve_distillation_agent_kind",
    "resolve_distillation_combined_agent_kinds",
]
