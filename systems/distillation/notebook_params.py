from __future__ import annotations

from copy import deepcopy

import numpy as np

from .config import (
    DELTA_T_HOURS,
    DISTILLATION_BASELINE_RUN_PROFILES,
    DISTILLATION_COMBINED_RUN_PROFILES,
    DISTILLATION_COMBINED_SETPOINTS_PHYS,
    DISTILLATION_INPUT_BOUNDS,
    DISTILLATION_MATRIX_RUN_PROFILES,
    DISTILLATION_NOMINAL_CONDITIONS,
    DISTILLATION_OBSERVER_POLES,
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
    "run_mode": "nominal",
    "disturbance_profile": "none",  # "none" | "ramp" | "fluctuation"
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
    "run_mode": "nominal",
    "disturbance_profile": "none",
    "state_mode": "standard",  # "standard" | "mismatch"
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "episode_defaults": {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False]},
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
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "agent": {
        "hidden_layers": [512, 512, 512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.99,
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
        "exploration_mode": "noisy",
        "loss_type": "huber",
        "eps_start": 0.3,
        "eps_end": 0.01,
        "eps_decay_rate": 0.99999,
        "eps_decay_mode": "exp",
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}
DISTILLATION_HORIZON_STANDARD_DEFAULTS["run_profiles"] = {
    key: dict(value) for key, value in {
        ("nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
        ("disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
        ("disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    }.items()
}

DISTILLATION_HORIZON_DUELING_DEFAULTS = {
    "run_mode": "nominal",
    "disturbance_profile": "none",
    "state_mode": "standard",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["run_profiles"]),
    "episode_defaults": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["episode_defaults"]),
    "controller": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["controller"]),
    "agent": {
        "seed": 7,
        "hidden_layers": [512, 512, 512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.99,
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
        "exploration_mode": "noisy",
        "loss_type": "huber",
        "eps_start": 0.30,
        "eps_end": 0.01,
        "eps_decay_rate": 0.99995,
        "eps_decay_mode": "linear",
        "eps_decay_steps": 15_000,
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_MATRIX_DEFAULTS = {
    "agent_kind": "td3",  # "td3" | "sac"
    "run_mode": "nominal",
    "disturbance_profile": "none",
    "state_mode": "standard",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_MATRIX_RUN_PROFILES),
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
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
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "td3_agent": {
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.995,
        "n_step": 1,  # Positive integer. Typical TD3 studies here use 1, 3, or 5.
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
        "exploration_mode": "param_noise",
        "loss_type": "huber",
        "param_noise_resample_interval": 4,
    },
    "sac_agent": {
        "actor_hidden": [512, 512, 512],
        "critic_hidden": [512, 512, 512],
        **_copy_replay_defaults(),
        "gamma": 0.995,
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
        "loss_type": "huber",
    },
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_STRUCTURED_MATRIX_DEFAULTS = {
    "agent_kind": "td3",  # "td3" | "sac"
    "run_mode": "nominal",
    "disturbance_profile": "none",
    "state_mode": "standard",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_MATRIX_RUN_PROFILES),
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        "update_family": "block",  # Options: "block" | "band". Block-lite is the primary first experiment.
        "range_profile": "tight",  # Options: "tight" | "default" | "wide". Tight is the safe first default.
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

DISTILLATION_WEIGHT_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "nominal",
    "disturbance_profile": "none",
    "state_mode": "standard",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_WEIGHT_RUN_PROFILES),
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.asarray(WEIGHT_MULTIPLIER_BOUNDS["low"], float).copy(),
        "high_coef": np.asarray(WEIGHT_MULTIPLIER_BOUNDS["high"], float).copy(),
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
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
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_RESIDUAL_DEFAULTS = {
    "agent_kind": "td3",
    "run_mode": "nominal",
    "disturbance_profile": "none",
    "state_mode": "mismatch",
    "use_rho_authority": True,
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": deepcopy(DISTILLATION_RESIDUAL_RUN_PROFILES),
    "controller": {
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "low_coef": np.asarray(RESIDUAL_BOUNDS["low"], float).copy(),
        "high_coef": np.asarray(RESIDUAL_BOUNDS["high"], float).copy(),
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
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

DISTILLATION_COMBINED_DEFAULTS = {
    "run_mode": "nominal",
    "disturbance_profile": "none",
    **deepcopy(DISTILLATION_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_PATH_DEFAULTS),
    **deepcopy(DISTILLATION_ASPEN_DEFAULTS),
    **deepcopy(DISTILLATION_COMMON_OVERRIDE_DEFAULTS),
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
    "residual_state_mode": "mismatch",
    "use_rho_authority": True,
    "run_profiles": deepcopy(DISTILLATION_COMBINED_RUN_PROFILES),
    "controller": {
        # The archived distillation combined notebook switched every 5 steps.
        "decision_interval": 5,
        "predict_grid": list(HORIZON_PREDICT_GRID),
        "control_grid": list(HORIZON_CONTROL_GRID),
        "predict_h": 6,
        "cont_h": 3,
        "Q1_penalty": 1.0,
        "Q2_penalty": 1.0,
        "R1_penalty": 1.0,
        "R2_penalty": 1.0,
        "model_low_by_agent": {
            key: np.asarray(value["low"], float).copy() for key, value in MATRIX_MULTIPLIER_BOUNDS.items()
        },
        "model_high_by_agent": {
            key: np.asarray(value["high"], float).copy() for key, value in MATRIX_MULTIPLIER_BOUNDS.items()
        },
        "weights_low": np.asarray(WEIGHT_MULTIPLIER_BOUNDS["low"], float).copy(),
        "weights_high": np.asarray(WEIGHT_MULTIPLIER_BOUNDS["high"], float).copy(),
        "residual_low": np.asarray(RESIDUAL_BOUNDS["low"], float).copy(),
        "residual_high": np.asarray(RESIDUAL_BOUNDS["high"], float).copy(),
        "mismatch_clip": 3.0,
        "use_shifted_mpc_warm_start": False,
        "nominal_qi": 0.0,
        "nominal_qs": 0.0,
        "nominal_ha": 0.0,
        "qi_change": 1.0,
        "qs_change": 1.0,
        "ha_change": 1.0,
    },
    "horizon_agent": deepcopy(DISTILLATION_HORIZON_STANDARD_DEFAULTS["agent"]),
    "td3_agent": deepcopy(DISTILLATION_MATRIX_DEFAULTS["td3_agent"]),
    "sac_agent": deepcopy(DISTILLATION_MATRIX_DEFAULTS["sac_agent"]),
    "reward": _copy_reward_defaults(),
    "system_setup": deepcopy(DISTILLATION_SYSTEM_SETUP),
}

DISTILLATION_NOTEBOOK_DEFAULTS = {
    "system_identification": DISTILLATION_SYSTEM_IDENTIFICATION_DEFAULTS,
    "baseline": DISTILLATION_BASELINE_DEFAULTS,
    "horizon_standard": DISTILLATION_HORIZON_STANDARD_DEFAULTS,
    "horizon_dueling": DISTILLATION_HORIZON_DUELING_DEFAULTS,
    "matrix": DISTILLATION_MATRIX_DEFAULTS,
    "structured_matrix": DISTILLATION_STRUCTURED_MATRIX_DEFAULTS,
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
    "DISTILLATION_NOTEBOOK_DEFAULTS",
    "get_distillation_notebook_defaults",
]
