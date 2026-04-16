from __future__ import annotations

from copy import deepcopy

import numpy as np

from .config import (
    VANDEVUSSE_BENCHMARK_STATE_SEED,
    VANDEVUSSE_CA0_RANGE,
    VANDEVUSSE_CB_TARGET_RANGE,
    VANDEVUSSE_DELTA_T_HOURS,
    VANDEVUSSE_DESIGN_PARAMS,
    VANDEVUSSE_INPUT_BOUNDS,
    VANDEVUSSE_LINEARIZATION_DEFAULTS,
    VANDEVUSSE_SS_INPUTS,
    VANDEVUSSE_SYSTEM_ID_STEP_HOLD_HOURS,
    VANDEVUSSE_SYSTEM_ID_STEP_TESTS,
    VANDEVUSSE_SYSTEM_ID_INITIAL_HOLD_HOURS,
    VANDEVUSSE_SYSTEM_PARAMS,
)


VANDEVUSSE_COMMON_PATH_DEFAULTS = {
    "data_dir_override": None,
    "results_dir_override": None,
}

VANDEVUSSE_SYSTEM_SETUP = {
    "delta_t_hours": float(VANDEVUSSE_DELTA_T_HOURS),
    "system_params": np.asarray(VANDEVUSSE_SYSTEM_PARAMS, float).copy(),
    "design_params": np.asarray(VANDEVUSSE_DESIGN_PARAMS, float).copy(),
    "ss_inputs": np.asarray(VANDEVUSSE_SS_INPUTS, float).copy(),
    "benchmark_state_seed": np.asarray(VANDEVUSSE_BENCHMARK_STATE_SEED, float).copy(),
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

VANDEVUSSE_NOTEBOOK_DEFAULTS = {
    "system_identification": VANDEVUSSE_SYSTEM_IDENTIFICATION_DEFAULTS,
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
