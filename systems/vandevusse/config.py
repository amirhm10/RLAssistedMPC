from pathlib import Path

import numpy as np


VANDEVUSSE_ROOT = Path("VanDeVusse")
VANDEVUSSE_DATA_SUBDIR = VANDEVUSSE_ROOT / "Data"
VANDEVUSSE_RESULT_SUBDIR = VANDEVUSSE_ROOT / "Results"

# The Van de Vusse benchmark is faster than the polymer case study, so the
# canonical system-identification sample time is finer.
VANDEVUSSE_DELTA_T_HOURS = 0.01

# PINN-paper parameter table used for the nonlinear plant implementation.
VANDEVUSSE_SYSTEM_PARAMS = np.array(
    [
        1.287e12,
        1.287e12,
        9.043e9,
        -9758.3,
        -9758.3,
        -8560.0,
        4.2,
        -11.0,
        -41.85,
        0.9342,
        3.01,
        2.00,
        4032.0,
        0.215,
        0.01,
        5.0,
    ],
    dtype=float,
)

# Feed concentration and inlet temperature remain explicit plant design
# parameters. Temperatures are stored in Kelvin throughout the code.
VANDEVUSSE_DESIGN_PARAMS = np.array([5.10, 378.1], dtype=float)

# Klatt/Engell benchmark operating point used for local ID around the main
# production region. The nonlinear plant still solves its own steady state from
# these inputs and the PINN parameterization.
VANDEVUSSE_BENCHMARK_STATE_SEED = np.array([1.235, 0.9, 407.29, 402.10], dtype=float)
VANDEVUSSE_SS_INPUTS = np.array([18.83, -4495.7], dtype=float)

VANDEVUSSE_INPUT_BOUNDS = {
    "u_min": np.array([5.0, -8500.0], dtype=float),
    "u_max": np.array([35.0, 0.0], dtype=float),
}
VANDEVUSSE_CA0_RANGE = np.array([4.5, 5.7], dtype=float)
VANDEVUSSE_CB_TARGET_RANGE = np.array([0.7, 0.95], dtype=float)

# The active Van de Vusse identification workflow now uses direct local
# linearization at the benchmark operating point. Step tests remain only for
# scaling and validation of the linearized nominal model.
VANDEVUSSE_SYSTEM_ID_INITIAL_HOLD_HOURS = 0.2
VANDEVUSSE_SYSTEM_ID_STEP_HOLD_HOURS = 1.2
VANDEVUSSE_LINEARIZATION_DEFAULTS = {
    "state_eps_rel": 1e-6,
    "state_eps_abs": 1e-8,
    "input_eps_rel": 1e-6,
    "input_eps_abs": 1e-6,
    "discretization_method": "zoh",
}

VANDEVUSSE_SYSTEM_ID_CSV_COLUMNS = ["F", "Q_K", "c_B", "T"]
VANDEVUSSE_SYSTEM_ID_STEP_TESTS = [
    {
        "name": "F_step",
        "save_filename": "F_step.csv",
        "input_index": 0,
        "step_delta": np.array([2.0, 0.0], dtype=float),
    },
    {
        "name": "QK_step",
        "save_filename": "QK_step.csv",
        "input_index": 1,
        "step_delta": np.array([0.0, -500.0], dtype=float),
    },
    {
        "name": "combined_validation",
        "save_filename": "combined_validation.csv",
        "input_index": None,
        "step_delta": np.array([2.0, -500.0], dtype=float),
    },
]
