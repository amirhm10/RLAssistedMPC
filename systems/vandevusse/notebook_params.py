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
    VANDEVUSSE_SS_INPUTS,
    VANDEVUSSE_SYSTEM_ID_POST_WINDOW_STEPS,
    VANDEVUSSE_SYSTEM_ID_PRE_WINDOW_STEPS,
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
    "show_fopdt_plots": True,
    "show_validation_plots": True,
    "save_metadata_json": True,
    "delay_quantization": "round",
    "initial_hold_hours": float(VANDEVUSSE_SYSTEM_ID_INITIAL_HOLD_HOURS),
    "step_hold_hours": float(VANDEVUSSE_SYSTEM_ID_STEP_HOLD_HOURS),
    "step_tests": [
        {
            "name": str(step_cfg["name"]),
            "save_filename": str(step_cfg["save_filename"]),
            "input_index": step_cfg["input_index"],
            "step_delta": np.asarray(step_cfg["step_delta"], float).copy(),
            "fit_use": bool(step_cfg["fit_use"]),
        }
        for step_cfg in VANDEVUSSE_SYSTEM_ID_STEP_TESTS
    ],
    "fit": {
        "pre_window_steps": int(VANDEVUSSE_SYSTEM_ID_PRE_WINDOW_STEPS),
        "post_window_steps": int(VANDEVUSSE_SYSTEM_ID_POST_WINDOW_STEPS),
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
