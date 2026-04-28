from __future__ import annotations

from copy import deepcopy

import numpy as np

from .config import (
    HORIZON_CONTROL_GRID,
    HORIZON_PREDICT_GRID,
    POLYMER_DELTA_T_HOURS,
    POLYMER_DESIGN_PARAMS,
    POLYMER_INPUT_BOUNDS,
    POLYMER_OBSERVER_POLES,
    POLYMER_RL_SETPOINTS_PHYS,
    POLYMER_SETPOINT_RANGE_PHYS,
    POLYMER_SS_INPUTS,
    POLYMER_SYSTEM_PARAMS,
    RL_REWARD_DEFAULTS as _RL_REWARD_DEFAULTS,
)


def _copy_reward_defaults():
    return {k: deepcopy(v) for k, v in _RL_REWARD_DEFAULTS.items()}


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
        "buffer_size": 150_000,
        "replay_frac_per": 0.5,
        "replay_frac_recent": 0.2,
        "replay_recent_window_mult": 5,
        "replay_recent_window": None,
        "replay_alpha": 0.6,
        "replay_beta_start": 0.4,
        "replay_beta_end": 1.0,
        "replay_beta_steps": 50_000,
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
        # Step 3B: use the nominal MPC objective as a small trust-region budget.
        # Strict 0.0 tolerance rejected almost every live candidate and reproduced MPC.
        "relative_tolerance": 1e-4 if enabled else 0.0,
        "absolute_tolerance": 1e-8,
        "fallback_on_candidate_solve_failure": True,
        "store_executed_action_in_replay": True,
        "log_policy_candidate_and_executed": True,
    }


def _copy_behavioral_cloning_defaults(
    enabled=False,
    *,
    lambda_bc_start=0.1,
    lambda_bc_end=0.0,
    active_subepisodes=10,
):
    return {
        "enabled": bool(enabled),
        "target_mode": "nominal_only",
        "lambda_bc_start": float(lambda_bc_start),
        "lambda_bc_end": float(lambda_bc_end),
        "decay_mode": "exp",
        "active_subepisodes": int(active_subepisodes),
        "start_after_warm_start": True,
        "log_diagnostics": True,
    }


def _copy_mismatch_defaults():
    return {
        "mismatch_clip": 3.0,
        "base_state_norm_mode": "running_zscore_physical_xhat",
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


def _copy_residual_authority_defaults():
    return {
        "append_rho_to_state": True,
        "authority_use_rho": True,
        "authority_beta_res": np.array([0.5, 0.5], float),
        "authority_du0_res": np.array([0.001, 0.001], float),
        "authority_eta_tol": 0.3,
        "authority_rho_floor": 0.15,
        "authority_rho_power": 1.0,
        "rho_mapping_mode": "exp_raw_tracking",
        "authority_rho_k": 0.55,
        "residual_zero_deadband_enabled": True,
        "residual_zero_tracking_raw_threshold": 0.1,
        "residual_zero_innovation_raw_threshold": 0.1,
    }


POLYMER_MATRIX_ALPHA_UPPER_CAP = 1.0566
POLYMER_DEFAULT_MULTIPLIER_LOW = 0.6
POLYMER_DEFAULT_MULTIPLIER_HIGH = 1.3


def _polymer_matrix_multiplier_bounds():
    low = np.full(3, POLYMER_DEFAULT_MULTIPLIER_LOW, dtype=float)
    high = np.array(
        [
            min(POLYMER_DEFAULT_MULTIPLIER_HIGH, POLYMER_MATRIX_ALPHA_UPPER_CAP),
            POLYMER_DEFAULT_MULTIPLIER_HIGH,
            POLYMER_DEFAULT_MULTIPLIER_HIGH,
        ],
        dtype=float,
    )
    return low, high


# -----------------------------------------------------------------------------
# Polymer notebook defaults
# -----------------------------------------------------------------------------
# These dictionaries are the notebook-facing source of truth for the polymer
# case. The active notebooks import from here and then derive runtime objects.
#
# Editing guidance:
# - String options are documented inline next to the default values.
# - Numeric values are the current research defaults. The useful range depends
#   on the method family and plant scaling, so comments describe the practical
#   operating range rather than forcing a hard validator here.
# - Arrays are stored in physical or scaled units exactly as the notebook uses
#   them. Change them here if you want the notebooks to pick up new defaults.
# -----------------------------------------------------------------------------

POLYMER_COMMON_PATH_DEFAULTS = {
    # Directory overrides:
    #   None -> use canonical Polymer/Data and Polymer/Results
    #   Path/string -> point the notebook at another data or result root
    "data_dir_override": None,
    "results_dir_override": None,
    # Name/path overrides:
    #   None -> use family/run-mode default names
    #   string/path -> force custom prefix or baseline path in the notebook
    "result_prefix_override": None,
    "compare_prefix_override": None,
    "baseline_mpc_path_override": None,
    "baseline_save_path_override": None,
}

POLYMER_COMMON_DISPLAY_DEFAULTS = {
    # STYLE_PROFILE options:
    #   "hybrid" -> default mixed research/debug plotting style
    #   "paper"  -> compact publication-oriented styling
    #   "debug"  -> maximal diagnostics and labels
    "style_profile": "hybrid",
    # SAVE_PDF:
    #   False -> PNG only
    #   True  -> save both PNG and PDF
    "save_pdf": False,
}

POLYMER_COMMON_OVERRIDE_DEFAULTS = {
    # These override values are intentionally None by default so the notebooks
    # fall back to the family-specific run-profile tables below.
    "n_tests_override": None,
    "set_points_len_override": None,
    "warm_start_override": None,
    "test_cycle_override": None,
    "plot_start_episode_override": None,
    "compare_start_episode_override": None,
}

POLYMER_SYSTEM_SETUP = {
    # Core plant initialization:
    # - delta_t_hours: sample time in hours
    # - system_params / design_params: polymer CSTR model parameters
    # - ss_inputs: steady-state manipulated inputs
    "delta_t_hours": float(POLYMER_DELTA_T_HOURS),
    "system_params": np.asarray(POLYMER_SYSTEM_PARAMS, float).copy(),
    "design_params": np.asarray(POLYMER_DESIGN_PARAMS, float).copy(),
    "ss_inputs": np.asarray(POLYMER_SS_INPUTS, float).copy(),
    # Input and setpoint ranges used by data loading and supervisory scaling.
    "input_bounds": {
        "u_min": np.asarray(POLYMER_INPUT_BOUNDS["u_min"], float).copy(),
        "u_max": np.asarray(POLYMER_INPUT_BOUNDS["u_max"], float).copy(),
    },
    "setpoint_range_phys": np.asarray(POLYMER_SETPOINT_RANGE_PHYS, float).copy(),
    "rl_setpoints_phys": np.asarray(POLYMER_RL_SETPOINTS_PHYS, float).copy(),
    # Observer poles:
    #   Current polymer notebooks use this 9-pole vector.
    "observer_poles": np.asarray(POLYMER_OBSERVER_POLES, float).copy(),
}

POLYMER_SYSTEM_IDENTIFICATION_DEFAULTS = {
    # RUN_NEW_EXPERIMENTS:
    #   True  -> regenerate step-test data and identified files
    #   False -> reuse the existing Polymer/Data CSVs and saved bundles
    "run_new_experiments": True,
    # SAVE_SYSTEM_DICT_NO_EXTENSION:
    #   True preserves the legacy no-extension polymer file alongside pickle.
    "save_system_dict_no_extension": True,
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
}

POLYMER_BASELINE_DEFAULTS = {
    "run_mode": "disturb",  # Options: "nominal" | "disturb"
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    # Baseline notebooks only expose these overrides.
    "n_tests_override": None,
    "set_points_len_override": None,
    "test_cycle_override": None,
    "plot_start_episode_override": None,
    "run_profiles": {
        # Run-profile keys:
        # - n_tests, set_points_len, test_cycle: episode schedule
        # - plot_start_episode: first episode shown in summary plots
        # - nominal_qi/qs/ha and *_change: disturbance-model schedule inputs
        "nominal": {
            "use_disturbance": False,
            "result_prefix": "mpc_offsetfree_nominal_unified",
            "plot_start_episode": 1,
            "n_tests": 2,
            "set_points_len": 400,
            "warm_start": 0,
            "test_cycle": [False, False],
            "nominal_qi": 108.0,
            "nominal_qs": 459.0,
            "nominal_ha": 1.05e6,
            "qi_change": 0.85,
            "qs_change": 1.3,
            "ha_change": 0.85,
        },
        "disturb": {
            "use_disturbance": True,
            "result_prefix": "mpc_offsetfree_disturb_unified",
            "plot_start_episode": 2,
            "n_tests": 200,
            "set_points_len": 400,
            "warm_start": 0,
            "test_cycle": [False, False, False, False, False],
            "nominal_qi": 108.0,
            "nominal_qs": 459.0,
            "nominal_ha": 1.05e6,
            "qi_change": 0.85,
            "qs_change": 1.3,
            "ha_change": 0.85,
        },
    },
    "controller": {
        # MPC horizon lengths:
        #   predict_h >= cont_h and both positive integers.
        "predict_h": 9,
        "cont_h": 3,
        # Penalties are positive scalars. Larger Q tracks outputs harder;
        # larger R penalizes input movement more strongly.
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        # False matches legacy zero-initialized MPC solves.
        "use_shifted_mpc_warm_start": False,
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_HORIZON_STANDARD_DEFAULTS = {
    "run_mode": "disturb",  # Options: "nominal" | "disturb"
    "state_mode": "mismatch",  # Options: "standard" | "mismatch"
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        "nominal": {
            "use_disturbance": False,
            "result_prefix": "horizon_nominal_unified",
            "compare_prefix": "nominal_compare_horizon_unified",
            "plot_start_episode": 5,
            "compare_start_episode": 1,
            "compare_mode": "nominal",
        },
        "disturb": {
            "use_disturbance": True,
            "result_prefix": "horizon_disturb_unified",
            "compare_prefix": "disturb_compare_horizon_unified",
            "plot_start_episode": 5,
            "compare_start_episode": 2,
            "compare_mode": "disturb",
        },
    },
    "episode_defaults": {
        "n_tests": 200,
        "set_points_len": 400,
        "warm_start": 10,
        "test_cycle": [False, False, False, False, False],
    },
    "controller": {
        "predict_grid": list(HORIZON_PREDICT_GRID),
        "control_grid": list(HORIZON_CONTROL_GRID),
        "decision_interval": 4,  # Positive integer number of MPC steps per RL decision
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "agent": {
        "hidden_layers": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.99,
        "n_step": 1,  # Positive integer. Keep 1 for the baseline; common research values are 3 or decision_interval.
        "multistep_mode": "one_step",  # Options: "one_step" | "n_step" | "lambda" | "retrace"
        "lambda_value": 0.9,  # Used when multistep_mode is "lambda" or "retrace"
        "lr": 1e-4,
        "batch_size": 128,
        "grad_clip_norm": 10.0,
        "double_dqn": True,
        "target_update": "soft",  # Options: "soft" | "hard"
        "tau": 0.01,
        "hard_update_interval": 10_000,
        "activation": "relu",  # Options depend on agent implementation; "relu" is current baseline
        "use_layer_norm": False,
        "dropout": 0.0,
        "target_combine": "q1",  # Retained for compatibility; the cleaned DDQN path ignores alternate combines.
        "exploration_mode": "epsilon",  # Options: "epsilon" | "noisy"
        "loss_type": "huber",  # Options: "huber" | "mse"
        "eps_start": 0.3,
        "eps_end": 0.01,
        "eps_decay_rate": 0.99999,
        "eps_decay_mode": "exp",  # Options: "linear" | "exp" | "cosine"
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_HORIZON_DUELING_DEFAULTS = {
    "run_mode": "disturb",
    "state_mode": "mismatch",
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        "nominal": {
            "use_disturbance": False,
            "result_prefix": "dueling_horizon_nominal_unified",
            "compare_prefix": "nominal_compare_dueling_horizon_unified",
            "plot_start_episode": 5,
            "compare_start_episode": 1,
            "compare_mode": "nominal",
        },
        "disturb": {
            "use_disturbance": True,
            "result_prefix": "dueling_horizon_disturb_unified",
            "compare_prefix": "disturb_compare_dueling_horizon_unified",
            "plot_start_episode": 5,
            "compare_start_episode": 2,
            "compare_mode": "disturb",
        },
    },
    "episode_defaults": {
        "n_tests": 200,
        "set_points_len": 400,
        "warm_start": 10,
        "test_cycle": [False, False, False, False, False],
    },
    "controller": {
        "predict_grid": list(HORIZON_PREDICT_GRID),
        "control_grid": list(HORIZON_CONTROL_GRID),
        "decision_interval": 4,
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "agent": {
        "seed": 7,
        "hidden_layers": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.99,
        "n_step": 1,
        "multistep_mode": "n_step",  # Dueling defaults to the multistep research path
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
        "exploration_mode": "noisy",  # Dueling default is the upgraded NoisyNet path
        "loss_type": "huber",
        "eps_start": 0.30,
        "eps_end": 0.01,
        "eps_decay_rate": 0.99995,
        "eps_decay_mode": "linear",
        "eps_decay_steps": 15_000,
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_MATRIX_DEFAULTS = {
    "agent_kind": "td3",  # Options: "td3" | "sac"
    "run_mode": "disturb",
    "state_mode": "mismatch",  # Options: "standard" | "mismatch". The latter feeds the authority error to the agent and normalizes it in the same way as the state features.
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        ("td3", "nominal"): {"result_prefix": "td3_multipliers_nominal", "compare_prefix": "nominal_compare_td3_multipliers", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("td3", "disturb"): {"result_prefix": "td3_multipliers_disturb", "compare_prefix": "disturb_compare_td3_multipliers", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "nominal"): {"result_prefix": "sac_multipliers_nominal", "compare_prefix": "nominal_compare_sac_multipliers", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "disturb"): {"result_prefix": "sac_multipliers_disturb", "compare_prefix": "disturb_compare_sac_multipliers", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
    },
    "episode_defaults": {"n_tests": 200, "set_points_len": 400, "warm_start": 10, "test_cycle": [False, False, False, False, False]},
    "post_warm_start_action_freeze_subepisodes": 0,
    "post_warm_start_actor_freeze_subepisodes": 0,
    # Step 4 polymer scalar matrix default: stronger and longer nominal-anchor BC.
    "behavioral_cloning": _copy_behavioral_cloning_defaults(
        enabled=True,
        lambda_bc_start=0.3,
        active_subepisodes=20,
    ),
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": _polymer_matrix_multiplier_bounds()[0],
        "high_coef": _polymer_matrix_multiplier_bounds()[1],
        "offline_multiplier_diagnostics": _copy_offline_multiplier_diagnostic_defaults(enabled=False),
        "release_protected_advisory_caps": _copy_release_protected_advisory_cap_defaults(enabled=False),
        "mpc_acceptance_fallback": _copy_mpc_acceptance_fallback_defaults(enabled=False),
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "td3_agent": {
        "actor_hidden": [256, 256],
        "critic_hidden": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,  # Positive integer. Typical TD3 ablations use 1, 3, or 5.
        "multistep_mode": "one_step",  # Options: "one_step" | "n_step" | "lambda"
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 2,
        "target_policy_smoothing_noise_std": 0.1,
        "noise_clip": 0.2,
        "max_action": 1.0,
        "tau": 0.005,
        "std_start": 0.2,
        "std_end": 0.02,
        "std_decay_rate": 0.99995,
        "std_decay_mode": "exp",
        "actor_freeze": 0,
        "exploration_mode": "param_noise",  # Options: "gaussian" | "param_noise"
        "loss_type": "huber",
        "param_noise_resample_interval": 4,
    },
    "sac_agent": {
        "actor_hidden": [256, 256],
        "critic_hidden": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,  # Positive integer. SAC often benefits from 3-step returns in this repo.
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
        "loss_type": "huber",
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_STRUCTURED_MATRIX_DEFAULTS = {
    "agent_kind": "td3",  # Options: "td3" | "sac"
    "run_mode": "disturb",
    "state_mode": "mismatch",  # Options: "standard" | "mismatch"
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        ("td3", "nominal"): {"result_prefix": "td3_structured_matrices_nominal", "compare_prefix": "nominal_compare_td3_structured_matrices", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("td3", "disturb"): {"result_prefix": "td3_structured_matrices_disturb", "compare_prefix": "disturb_compare_td3_structured_matrices", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "nominal"): {"result_prefix": "sac_structured_matrices_nominal", "compare_prefix": "nominal_compare_sac_structured_matrices", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "disturb"): {"result_prefix": "sac_structured_matrices_disturb", "compare_prefix": "disturb_compare_sac_structured_matrices", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
    },
    "episode_defaults": deepcopy(POLYMER_MATRIX_DEFAULTS["episode_defaults"]),
    "post_warm_start_action_freeze_subepisodes": 0,
    "post_warm_start_actor_freeze_subepisodes": 0,
    # Step 4 polymer structured default: stronger than scalar because the action
    # space is larger and the first BC-only handoff stayed much farther from nominal.
    "behavioral_cloning": _copy_behavioral_cloning_defaults(
        enabled=True,
        lambda_bc_start=0.6,
        active_subepisodes=25,
    ),
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "update_family": "block",  # Options: "block" | "band". Block-lite is the primary first experiment.
        "range_profile": "wide",  # Options: "tight" | "default" | "wide". Wide is the active polymer default for analysis.
        "a_low_override": POLYMER_DEFAULT_MULTIPLIER_LOW,  # Scalar or array override for A-side structured bounds.
        "a_high_override": min(POLYMER_DEFAULT_MULTIPLIER_HIGH, POLYMER_MATRIX_ALPHA_UPPER_CAP),  # Cap A-side widening at the analyzed alpha limit.
        "b_low_override": POLYMER_DEFAULT_MULTIPLIER_LOW,  # Scalar or array override for B-side structured bounds.
        "b_high_override": POLYMER_DEFAULT_MULTIPLIER_HIGH,  # Allow wider B-side uncertainty than A-side.
        "offline_multiplier_diagnostics": _copy_offline_multiplier_diagnostic_defaults(enabled=False),
        "release_protected_advisory_caps": _copy_release_protected_advisory_cap_defaults(enabled=False),
        "mpc_acceptance_fallback": _copy_mpc_acceptance_fallback_defaults(enabled=False),
        "block_group_count": 3,  # Positive integer. Used only when block_groups is None.
        "block_groups": None,  # Optional explicit 0-based physical-state partition, e.g. [[0, 1], [2, 3], [4, 5, 6]].
        "band_offsets": [0, 1, 2],  # Non-negative offsets used in band mode. Must include 0.
        "log_spectral_radius": True,  # Options: False | True. True logs the physical-model spectral radius each step.
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "td3_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["td3_agent"]),
    "sac_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["sac_agent"]),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_REIDENTIFICATION_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "disturb",
    "state_mode": "mismatch",
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        ("td3", "nominal"): {
            "result_prefix": "td3_reidentification_nominal",
            "compare_prefix": "nominal_compare_td3_reidentification",
            "compare_mode": "nominal",
            "plot_start_episode": 2,
            "compare_start_episode": 2,
        },
        ("td3", "disturb"): {
            "result_prefix": "td3_reidentification_disturb",
            "compare_prefix": "disturb_compare_td3_reidentification",
            "compare_mode": "disturb",
            "plot_start_episode": 2,
            "compare_start_episode": 2,
        },
        ("sac", "nominal"): {
            "result_prefix": "sac_reidentification_nominal",
            "compare_prefix": "nominal_compare_sac_reidentification",
            "compare_mode": "nominal",
            "plot_start_episode": 2,
            "compare_start_episode": 2,
        },
        ("sac", "disturb"): {
            "result_prefix": "sac_reidentification_disturb",
            "compare_prefix": "disturb_compare_sac_reidentification",
            "compare_mode": "disturb",
            "plot_start_episode": 2,
            "compare_start_episode": 2,
        },
    },
    "episode_defaults": deepcopy(POLYMER_MATRIX_DEFAULTS["episode_defaults"]),
    "post_warm_start_action_freeze_subepisodes": 5,
    "post_warm_start_actor_freeze_subepisodes": 5,
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "reidentification": {
        "basis_family": "lowrank_polymer",
        "id_component_mode": "AB",
        "observer_update_alignment": "legacy_previous_measurement",
        "candidate_guard_mode": "fro_only",
        "normalize_blend_extras": True,
        "blend_extra_clip": 3.0,
        "blend_residual_scale": 1.0,
        "log_theta_clipping": True,
        "id_solver": "ridge_closed_form",
        "rank_A": 6,
        "rank_B": 2,
        "offline_window": 80,
        "offline_stride": 80,
        "lambda_A_off": 1e-4,
        "lambda_B_off": 1e-3,
        "id_window": 80,
        "id_update_period": 5,
        "lambda_prev_A": 1e-2,
        "lambda_prev_B": 1e-1,
        "lambda_0_A": 1e-4,
        "lambda_0_B": 1e-3,
        "theta_low_A": -0.15,
        "theta_high_A": 0.15,
        "theta_low_B": -0.08,
        "theta_high_B": 0.08,
        "delta_A_max": 0.10,
        "delta_B_max": 0.10,
        "eta_tau_A": 0.1,
        "eta_tau_B": 0.1,
        "observer_refresh_enabled": False,
        "observer_refresh_every_episodes": 10,
        "rho_obs": 0.25,
        "force_eta_constant": None,
        "disable_identification": False,
    },
    "td3_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["td3_agent"]),
    "sac_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["sac_agent"]),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_WEIGHT_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "disturb",
    "state_mode": "mismatch",
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        ("td3", "nominal"): {"result_prefix": "td3_weights_nominal", "compare_prefix": "nominal_compare_td3_weights", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("td3", "disturb"): {"result_prefix": "td3_weights_disturb", "compare_prefix": "disturb_compare_td3_weights", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "nominal"): {"result_prefix": "sac_weights_nominal", "compare_prefix": "nominal_compare_sac_weights", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "disturb"): {"result_prefix": "sac_weights_disturb", "compare_prefix": "disturb_compare_sac_weights", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
    },
    "episode_defaults": {"n_tests": 200, "set_points_len": 400, "warm_start": 10, "test_cycle": [False, False, False, False, False]},
    "post_warm_start_action_freeze_subepisodes": 5,
    "post_warm_start_actor_freeze_subepisodes": 5,
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.array([0.75, 0.75, 0.75, 0.75], float),
        "high_coef": np.array([2.0, 2.0, 2.0, 2.0], float),
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "td3_agent": {
        "actor_hidden": [256, 256],
        "critic_hidden": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,
        "multistep_mode": "one_step",
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 2,
        "target_policy_smoothing_noise_std": 0.1,
        "noise_clip": 0.2,
        "max_action": 1.0,
        "tau": 0.005,
        "std_start": 0.2,
        "std_end": 0.02,
        "std_decay_rate": 0.99995,
        "std_decay_mode": "exp",
        "actor_freeze": 0,
        "exploration_mode": "param_noise",
        "loss_type": "huber",
        "param_noise_resample_interval": 4,
    },
    "sac_agent": {
        "actor_hidden": [256, 256],
        "critic_hidden": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.995,
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
        "loss_type": "huber",
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_RESIDUAL_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "disturb",
    "state_mode": "mismatch",  # Options: "standard" | "mismatch". The latter feeds the authority error to the agent and normalizes it in the same way as the state features.
    **_copy_residual_authority_defaults(),
    "use_rho_authority": True,  # Legacy alias kept for notebook compatibility.
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        ("td3", "nominal"): {"result_prefix": "td3_residual_nominal", "compare_prefix": "nominal_compare_td3_residual", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("td3", "disturb"): {"result_prefix": "td3_residual_disturb", "compare_prefix": "disturb_compare_td3_residual", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "nominal"): {"result_prefix": "sac_residual_nominal", "compare_prefix": "nominal_compare_sac_residual", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "disturb"): {"result_prefix": "sac_residual_disturb", "compare_prefix": "disturb_compare_sac_residual", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
    },
    "episode_defaults": {"n_tests": 200, "set_points_len": 400, "warm_start": 10, "test_cycle": [False, False, False, False, False]},
    "post_warm_start_action_freeze_subepisodes": 5,
    "post_warm_start_actor_freeze_subepisodes": 5,
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.array([-0.25, -0.25], float),
        "high_coef": np.array([0.25, 0.25], float),
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "td3_agent": {
        "actor_hidden": [256, 256],
        "critic_hidden": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,
        "multistep_mode": "one_step",
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 2,
        "target_policy_smoothing_noise_std": 0.1,
        "noise_clip": 0.2,
        "max_action": 1.0,
        "tau": 0.005,
        "std_start": 0.2,
        "std_end": 0.02,
        "std_decay_rate": 0.99995,
        "std_decay_mode": "exp",
        "actor_freeze": 0,
        "exploration_mode": "param_noise",
        "loss_type": "huber",
        "param_noise_resample_interval": 4,
    },
    "sac_agent": {
        "actor_hidden": [256, 256],
        "critic_hidden": [256, 256],
        **_copy_replay_defaults(),
        "gamma": 0.995,
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
        "loss_type": "huber",
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_COMBINED_DEFAULTS = {
    "run_mode": "disturb",
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    # Enable/disable flags:
    #   True -> instantiate that agent block
    #   False -> leave it out of the combined supervisor
    "enable_horizon": True,
    "horizon_agent_kind": "dqn",
    "horizon_state_mode": "mismatch",
    "enable_matrix": True,
    "matrix_agent_kind": "td3",
    "matrix_state_mode": "mismatch",
    "enable_weights": True,
    "weights_agent_kind": "td3",
    "weights_state_mode": "mismatch",
    "enable_residual": True,
    "residual_agent_kind": "td3",
    "residual_state_mode": "mismatch",
    **_copy_residual_authority_defaults(),
    "use_rho_authority": True,  # Legacy alias kept for notebook compatibility.
    "run_profiles": {
        "nominal": {"result_prefix_template": "combined_nominal_{suffix}", "compare_prefix_template": "nominal_compare_combined_{suffix}", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        "disturb": {"result_prefix_template": "combined_disturb_{suffix}", "compare_prefix_template": "disturb_compare_combined_{suffix}", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
    },
    "episode_defaults": deepcopy(POLYMER_HORIZON_STANDARD_DEFAULTS["episode_defaults"]),
    "td3_post_warm_start_action_freeze_subepisodes": 5,
    "td3_post_warm_start_actor_freeze_subepisodes": 5,
    "controller": {
        "decision_interval": 4,
        "predict_grid": list(HORIZON_PREDICT_GRID),
        "control_grid": list(HORIZON_CONTROL_GRID),
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "model_low": _polymer_matrix_multiplier_bounds()[0],
        "model_high": _polymer_matrix_multiplier_bounds()[1],
        "weights_low": POLYMER_WEIGHT_DEFAULTS["controller"]["low_coef"].copy(),
        "weights_high": POLYMER_WEIGHT_DEFAULTS["controller"]["high_coef"].copy(),
        "residual_low": POLYMER_RESIDUAL_DEFAULTS["controller"]["low_coef"].copy(),
        "residual_high": POLYMER_RESIDUAL_DEFAULTS["controller"]["high_coef"].copy(),
        "offline_multiplier_diagnostics": deepcopy(POLYMER_MATRIX_DEFAULTS["controller"]["offline_multiplier_diagnostics"]),
        "release_protected_advisory_caps": deepcopy(POLYMER_MATRIX_DEFAULTS["controller"]["release_protected_advisory_caps"]),
        "mpc_acceptance_fallback": deepcopy(POLYMER_MATRIX_DEFAULTS["controller"]["mpc_acceptance_fallback"]),
        **_copy_mismatch_defaults(),
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "horizon_agent": deepcopy(POLYMER_HORIZON_STANDARD_DEFAULTS["agent"]),
    "horizon_dueling_agent": deepcopy(POLYMER_HORIZON_DUELING_DEFAULTS["agent"]),
    "matrix_td3_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["td3_agent"]),
    "matrix_sac_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["sac_agent"]),
    "weights_td3_agent": deepcopy(POLYMER_WEIGHT_DEFAULTS["td3_agent"]),
    "weights_sac_agent": deepcopy(POLYMER_WEIGHT_DEFAULTS["sac_agent"]),
    "residual_td3_agent": deepcopy(POLYMER_RESIDUAL_DEFAULTS["td3_agent"]),
    "residual_sac_agent": deepcopy(POLYMER_RESIDUAL_DEFAULTS["sac_agent"]),
    # Backward-compatible aliases for older notebook cells.
    "td3_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["td3_agent"]),
    "sac_agent": deepcopy(POLYMER_MATRIX_DEFAULTS["sac_agent"]),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
}

POLYMER_POLES_EXPERIMENT_DEFAULTS = {
    # This notebook is not part of the main unified workflow, but it still uses
    # the same polymer system and reward defaults. Keeping it here prevents the
    # replay-buffer or path defaults from drifting.
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    "buffer_capacity": 150_000,
    "system_setup": deepcopy(POLYMER_SYSTEM_SETUP),
    "reward": _copy_reward_defaults(),
}

POLYMER_NOTEBOOK_DEFAULTS = {
    "system_identification": POLYMER_SYSTEM_IDENTIFICATION_DEFAULTS,
    "baseline": POLYMER_BASELINE_DEFAULTS,
    "horizon_standard": POLYMER_HORIZON_STANDARD_DEFAULTS,
    "horizon_dueling": POLYMER_HORIZON_DUELING_DEFAULTS,
    "matrix": POLYMER_MATRIX_DEFAULTS,
    "structured_matrix": POLYMER_STRUCTURED_MATRIX_DEFAULTS,
    "reidentification": POLYMER_REIDENTIFICATION_DEFAULTS,
    "weights": POLYMER_WEIGHT_DEFAULTS,
    "residual": POLYMER_RESIDUAL_DEFAULTS,
    "combined": POLYMER_COMBINED_DEFAULTS,
    "poles_experiment": POLYMER_POLES_EXPERIMENT_DEFAULTS,
}


def get_polymer_notebook_defaults(family: str) -> dict:
    """
    Return a deep-copied parameter dictionary for the requested polymer notebook
    family so notebooks can mutate local values safely.
    """

    key = str(family).strip().lower()
    if key not in POLYMER_NOTEBOOK_DEFAULTS:
        raise KeyError(f"Unknown polymer notebook family: {family}")
    return deepcopy(POLYMER_NOTEBOOK_DEFAULTS[key])


__all__ = [
    "POLYMER_NOTEBOOK_DEFAULTS",
    "get_polymer_notebook_defaults",
]
