from __future__ import annotations

from copy import deepcopy

import numpy as np

from .config import (
    VANDEVUSSE_BASELINE_CA0_BLOCKS,
    VANDEVUSSE_BASELINE_OBSERVER_UPDATE_MODE_DEFAULT,
    VANDEVUSSE_BASELINE_OBSERVER_POLES_DEFAULT,
    VANDEVUSSE_BASELINE_OBSERVER_POLES_FALLBACK,
    VANDEVUSSE_BASELINE_Q_OUT,
    VANDEVUSSE_BASELINE_R_IN,
    VANDEVUSSE_BASELINE_RUN_PROFILES,
    VANDEVUSSE_BASELINE_SETPOINT_RANGE_PHYS,
    VANDEVUSSE_BASELINE_SETPOINTS_PHYS,
    VANDEVUSSE_BASELINE_THERMAL_STUDY_SETPOINTS_PHYS,
    VANDEVUSSE_BASELINE_TIN_BLOCKS,
    VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K,
    VANDEVUSSE_BENCHMARK_SOURCES,
    VANDEVUSSE_BENCHMARK_STATE_SEED,
    VANDEVUSSE_CA0_RANGE,
    VANDEVUSSE_CB_TARGET_RANGE,
    VANDEVUSSE_DELTA_T_HOURS,
    VANDEVUSSE_DESIGN_PARAMS,
    VANDEVUSSE_INPUT_BOUNDS,
    VANDEVUSSE_LINEARIZATION_DEFAULTS,
    VANDEVUSSE_PINN_CROSSCHECK_DESIGN_PARAMS,
    VANDEVUSSE_SS_INPUTS,
    VANDEVUSSE_SYSTEM_ID_STEP_HOLD_HOURS,
    VANDEVUSSE_SYSTEM_ID_STEP_TESTS,
    VANDEVUSSE_SYSTEM_ID_INITIAL_HOLD_HOURS,
    VANDEVUSSE_SYSTEM_PARAMS,
)


VANDEVUSSE_COMMON_PATH_DEFAULTS = {
    "data_dir_override": None,
    "results_dir_override": None,
    "result_prefix_override": None,
    "baseline_save_path_override": None,
}

VANDEVUSSE_COMMON_DISPLAY_DEFAULTS = {
    "style_profile": "hybrid",
    "save_pdf": False,
}

VANDEVUSSE_COMMON_OVERRIDE_DEFAULTS = {
    "n_tests_override": None,
    "set_points_len_override": None,
    "warm_start_override": None,
    "test_cycle_override": None,
    "plot_start_episode_override": None,
}

VANDEVUSSE_SYSTEM_SETUP = {
    "delta_t_hours": float(VANDEVUSSE_DELTA_T_HOURS),
    "system_params": np.asarray(VANDEVUSSE_SYSTEM_PARAMS, float).copy(),
    "design_params": np.asarray(VANDEVUSSE_DESIGN_PARAMS, float).copy(),
    "legacy_crosscheck_design_params": np.asarray(VANDEVUSSE_PINN_CROSSCHECK_DESIGN_PARAMS, float).copy(),
    "ss_inputs": np.asarray(VANDEVUSSE_SS_INPUTS, float).copy(),
    "benchmark_state_seed": np.asarray(VANDEVUSSE_BENCHMARK_STATE_SEED, float).copy(),
    "benchmark_sources": dict(VANDEVUSSE_BENCHMARK_SOURCES),
    "benchmark_feed_temperature_k": float(VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K),
    "input_bounds": {
        "u_min": np.asarray(VANDEVUSSE_INPUT_BOUNDS["u_min"], float).copy(),
        "u_max": np.asarray(VANDEVUSSE_INPUT_BOUNDS["u_max"], float).copy(),
    },
    "disturbance_bounds": {
        "c_A0": np.asarray(VANDEVUSSE_CA0_RANGE, float).copy(),
    },
    "target_range_c_B": np.asarray(VANDEVUSSE_CB_TARGET_RANGE, float).copy(),
}

VANDEVUSSE_SYSTEM_IDENTIFICATION_DEFAULTS = {
    "run_new_experiments": True,
    "show_linearization_diagnostics": True,
    "show_validation_plots": True,
    "save_metadata_json": True,
    "initial_hold_hours": float(VANDEVUSSE_SYSTEM_ID_INITIAL_HOLD_HOURS),
    "step_hold_hours": float(VANDEVUSSE_SYSTEM_ID_STEP_HOLD_HOURS),
    "step_tests": [
        {
            "name": str(step_cfg["name"]),
            "save_filename": str(step_cfg["save_filename"]),
            "input_index": step_cfg["input_index"],
            "step_delta": np.asarray(step_cfg["step_delta"], float).copy(),
        }
        for step_cfg in VANDEVUSSE_SYSTEM_ID_STEP_TESTS
    ],
    "linearization": {
        "state_eps_rel": float(VANDEVUSSE_LINEARIZATION_DEFAULTS["state_eps_rel"]),
        "state_eps_abs": float(VANDEVUSSE_LINEARIZATION_DEFAULTS["state_eps_abs"]),
        "input_eps_rel": float(VANDEVUSSE_LINEARIZATION_DEFAULTS["input_eps_rel"]),
        "input_eps_abs": float(VANDEVUSSE_LINEARIZATION_DEFAULTS["input_eps_abs"]),
        "discretization_method": str(VANDEVUSSE_LINEARIZATION_DEFAULTS["discretization_method"]),
    },
    **deepcopy(VANDEVUSSE_COMMON_PATH_DEFAULTS),
    "system_setup": deepcopy(VANDEVUSSE_SYSTEM_SETUP),
}

VANDEVUSSE_BASELINE_SYSTEM_SETUP = {
    **deepcopy(VANDEVUSSE_SYSTEM_SETUP),
    "setpoint_range_phys": np.asarray(VANDEVUSSE_BASELINE_SETPOINT_RANGE_PHYS, float).copy(),
    "baseline_setpoints_phys": np.asarray(VANDEVUSSE_BASELINE_SETPOINTS_PHYS, float).copy(),
    "thermal_study_setpoints_phys": np.asarray(VANDEVUSSE_BASELINE_THERMAL_STUDY_SETPOINTS_PHYS, float).copy(),
    "observer_poles_default": np.asarray(VANDEVUSSE_BASELINE_OBSERVER_POLES_DEFAULT, float).copy(),
    "observer_poles_fallback": np.asarray(VANDEVUSSE_BASELINE_OBSERVER_POLES_FALLBACK, float).copy(),
    "disturbance_block_values": {
        "c_A0": np.asarray(VANDEVUSSE_BASELINE_CA0_BLOCKS, float).copy(),
        "T_in": np.asarray(VANDEVUSSE_BASELINE_TIN_BLOCKS, float).copy(),
    },
}

VANDEVUSSE_BASELINE_POLE_SEARCH_DEFAULTS = {
    "run_search": False,
    "n_samples": 50,
    "seed": 42,
    "low": 0.55,
    "high": 0.85,
    "mode": "uniform",
    "top_k": 5,
    "n_tests_override": 1,
    "set_points_len_override": 25,
    "test_cycle_override": [False],
}

VANDEVUSSE_BASELINE_DEFAULTS = {
    "run_mode": "disturb",
    "disturbance_profile": "ca0_blocks",
    "observer_update_mode": str(VANDEVUSSE_BASELINE_OBSERVER_UPDATE_MODE_DEFAULT),
    "use_manual_observer_poles": False,
    "manual_observer_poles": np.asarray(VANDEVUSSE_BASELINE_OBSERVER_POLES_DEFAULT, float).copy(),
    **deepcopy(VANDEVUSSE_COMMON_DISPLAY_DEFAULTS),
    **deepcopy(VANDEVUSSE_COMMON_PATH_DEFAULTS),
    **deepcopy(VANDEVUSSE_COMMON_OVERRIDE_DEFAULTS),
    "run_profiles": {key: dict(value) for key, value in VANDEVUSSE_BASELINE_RUN_PROFILES.items()},
    "pole_search": deepcopy(VANDEVUSSE_BASELINE_POLE_SEARCH_DEFAULTS),
    "controller": {
        "predict_h": 10,
        "cont_h": 3,
        "Q_out": np.asarray(VANDEVUSSE_BASELINE_Q_OUT, float).copy(),
        "R_in": np.asarray(VANDEVUSSE_BASELINE_R_IN, float).copy(),
        "use_shifted_mpc_warm_start": False,
    },
    "system_setup": deepcopy(VANDEVUSSE_BASELINE_SYSTEM_SETUP),
}

VANDEVUSSE_NOTEBOOK_DEFAULTS = {
    "system_identification": VANDEVUSSE_SYSTEM_IDENTIFICATION_DEFAULTS,
    "baseline": VANDEVUSSE_BASELINE_DEFAULTS,
}


def get_vandevusse_notebook_defaults(family: str) -> dict:
    key = str(family).strip().lower()
    if key not in VANDEVUSSE_NOTEBOOK_DEFAULTS:
        raise KeyError(f"Unknown Van de Vusse notebook family: {family}")
    return deepcopy(VANDEVUSSE_NOTEBOOK_DEFAULTS[key])


__all__ = [
    "VANDEVUSSE_NOTEBOOK_DEFAULTS",
    "get_vandevusse_notebook_defaults",
]
