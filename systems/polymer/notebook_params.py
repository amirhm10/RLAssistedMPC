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


def _copy_matrix_robust_defaults():
    return {
        # Fixed-estimator / robust-prediction MPC controls for matrix methods:
        # - input_tightening_frac: fraction of the MPC input span removed from
        #   each side before the assisted solve; practical range is [0.0, 0.10]
        # - enable_accept_norm_test: reject assisted models that move too far
        #   from the nominal A/B matrices in relative Frobenius norm
        # - eps_A_norm_frac / eps_B_norm_frac: relative matrix-deviation
        #   thresholds used by the norm acceptance test; practical range is
        #   roughly [0.01, 0.20] depending on how aggressive the multipliers are
        # - enable_accept_prediction_test: reject assisted models whose short
        #   nominal-vs-assisted output rollout deviates too far
        # - prediction_check_horizon: positive integer short probe horizon used
        #   by the acceptance test; 1-5 is the practical range for this repo
        # - eps_y_pred_scaled: max allowed scaled output prediction deviation;
        #   practical range is roughly [0.02, 0.30]
        # - enable_solver_fallback: when True, allow a final nominal/full-bounds
        #   solve if the tightened solve fails
        # - probe_input_mode: currently only "hold_current_input" is supported
        # - recalculate_observer_on_matrix_change: retained only for backward
        #   compatibility; matrix methods now ignore it because estimation is
        #   fixed nominal by design
        "input_tightening_frac": 0.02,
        "enable_accept_norm_test": True,
        "eps_A_norm_frac": 0.05,
        "eps_B_norm_frac": 0.05,
        "enable_accept_prediction_test": True,
        "prediction_check_horizon": 2,
        "eps_y_pred_scaled": 0.10,
        "enable_solver_fallback": True,
        "probe_input_mode": "hold_current_input",
        "recalculate_observer_on_matrix_change": False,
    }


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
    "run_mode": "nominal",  # Options: "nominal" | "disturb"
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
            "qi_change": 0.95,
            "qs_change": 1.05,
            "ha_change": 0.92,
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
            "qi_change": 0.95,
            "qs_change": 1.05,
            "ha_change": 0.92,
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
    "run_mode": "nominal",  # Options: "nominal" | "disturb"
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
        "warm_start": 5,
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
        "mismatch_clip": 3.0,  # Positive float. None disables the post-normalization clip.
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "agent": {
        "hidden_layers": [512, 512, 512, 512, 512],
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
    "run_mode": "nominal",
    "state_mode": "standard",
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
        "set_points_len": 200,
        "warm_start": 5,
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
        "mismatch_clip": 3.0,
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
        "hidden_layers": [512, 512, 512, 512, 512],
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
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.array([0.95, 0.95, 0.95], float),
        "high_coef": np.array([1.05, 1.05, 1.05], float),
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        **_copy_matrix_robust_defaults(),
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "td3_agent": {
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,  # Positive integer. Typical TD3 ablations use 1, 3, or 5.
        "multistep_mode": "one_step",  # Options: "one_step" | "n_step" | "lambda"
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 4,
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
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
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
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        **_copy_matrix_robust_defaults(),
        "update_family": "block",  # Options: "block" | "band". Block-lite is the primary first experiment.
        "range_profile": "tight",  # Options: "tight" | "default" | "wide". Tight is the safe first default.
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

POLYMER_WEIGHT_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "nominal",
    "state_mode": "standard",
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {
        ("td3", "nominal"): {"result_prefix": "td3_weights_nominal", "compare_prefix": "nominal_compare_td3_weights", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("td3", "disturb"): {"result_prefix": "td3_weights_disturb", "compare_prefix": "disturb_compare_td3_weights", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "nominal"): {"result_prefix": "sac_weights_nominal", "compare_prefix": "nominal_compare_sac_weights", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        ("sac", "disturb"): {"result_prefix": "sac_weights_disturb", "compare_prefix": "disturb_compare_sac_weights", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
    },
    "episode_defaults": {"n_tests": 200, "set_points_len": 400, "warm_start": 0, "test_cycle": [False, False, False, False, False]},
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.array([0.5, 0.5, 0.5, 0.5], float),
        "high_coef": np.array([3.0, 3.0, 3.0, 3.0], float),
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "td3_agent": {
        "actor_hidden": [512, 512, 512, 512, 512],
        "critic_hidden": [512, 512, 512, 512, 512],
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
        "actor_hidden": [512, 512, 512, 512, 512],
        "critic_hidden": [512, 512, 512, 512, 512],
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
    "run_mode": "nominal",
    "state_mode": "mismatch",  # Options: "standard" | "mismatch". The latter feeds the authority error to the agent and normalizes it in the same way as the state features.
    "use_rho_authority": True,  # Only meaningful when state_mode == "mismatch"
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
    "controller": {
        "predict_h": 9,
        "cont_h": 3,
        "Q1_penalty": 5.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.array([-0.25, -0.25], float),
        "high_coef": np.array([0.25, 0.25], float),
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "td3_agent": {
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,
        "multistep_mode": "one_step",
        "lambda_value": 0.9,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 4,
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
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
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
    "run_mode": "nominal",
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    **deepcopy(POLYMER_COMMON_OVERRIDE_DEFAULTS),
    # Enable/disable flags:
    #   True -> instantiate that agent block
    #   False -> leave it out of the combined supervisor
    "enable_horizon": True,
    "horizon_state_mode": "standard",
    "enable_matrix": True,
    "matrix_agent_kind": "td3",
    "matrix_state_mode": "standard",
    "enable_weights": True,
    "weights_agent_kind": "td3",
    "weights_state_mode": "standard",
    "enable_residual": True,
    "residual_agent_kind": "td3",
    "residual_state_mode": "standard",
    "use_rho_authority": True,
    "run_profiles": {
        "nominal": {"result_prefix_template": "combined_nominal_{suffix}", "compare_prefix_template": "nominal_compare_combined_{suffix}", "compare_mode": "nominal", "plot_start_episode": 2, "compare_start_episode": 2},
        "disturb": {"result_prefix_template": "combined_disturb_{suffix}", "compare_prefix_template": "disturb_compare_combined_{suffix}", "compare_mode": "disturb", "plot_start_episode": 2, "compare_start_episode": 2},
    },
    "episode_defaults": {"n_tests": 200, "set_points_len": 200, "warm_start": 0, "test_cycle": [False, False, False, False, False]},
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
        "model_low": np.array([0.95, 0.95, 0.95], float),
        "model_high": np.array([1.05, 1.05, 1.05], float),
        "weights_low": np.array([0.9, 0.9, 0.9, 0.9], float),
        "weights_high": np.array([1.1, 1.1, 1.1, 1.1], float),
        "residual_low": np.array([-0.5, -0.5], float),
        "residual_high": np.array([0.5, 0.5], float),
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        **_copy_matrix_robust_defaults(),
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.85,
        "qs_change": 1.3,
        "ha_change": 0.85,
    },
    "horizon_agent": {
        "hidden_layers": [512, 512, 512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.99,
        "n_step": 1,
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
        "loss_type": "huber",
        "eps_start": 0.3,
        "eps_end": 0.01,
        "eps_decay_rate": 0.99999,
        "eps_decay_mode": "exp",
    },
    "td3_agent": {
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "batch_size": 128,
        "policy_delay": 4,
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
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,
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

POLYMER_POLES_EXPERIMENT_DEFAULTS = {
    # This notebook is not part of the main unified workflow, but it still uses
    # the same polymer system and reward defaults. Keeping it here prevents the
    # replay-buffer or path defaults from drifting.
    **deepcopy(POLYMER_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(POLYMER_COMMON_PATH_DEFAULTS),
    "buffer_capacity": 40_000,
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
