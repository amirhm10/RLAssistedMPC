from pathlib import Path

import numpy as np


VANDEVUSSE_ROOT = Path("VanDeVusse")
VANDEVUSSE_DATA_SUBDIR = VANDEVUSSE_ROOT / "Data"
VANDEVUSSE_RESULT_SUBDIR = VANDEVUSSE_ROOT / "Results"

# The Van de Vusse benchmark is faster than the polymer case study, so the
# canonical system-identification sample time is finer.
VANDEVUSSE_DELTA_T_HOURS = 0.01

# Control-benchmark-authoritative Van de Vusse parameterization.
# The canonical defaults are taken from the benchmark-control literature:
# - Chen, Kremling, Allgower (1995) for the benchmark problem framing
# - Klatt and Engell (1998) for the main control-benchmark parameter table
# Later PINN/modeling papers may be used only as secondary cross-checks.
VANDEVUSSE_BENCHMARK_SOURCES = {
    "historical_benchmark": "Chen, Kremling, Allgower (1995)",
    "main_control_benchmark": "Klatt and Engell (1998)",
    "secondary_crosscheck_only": "Later PINN/modeling papers",
}

# The benchmark parameter values below match the control-benchmark literature.
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
# The control benchmark uses c_A0 = 5.1 mol/L and theta_0 = 130.0 C = 403.15 K.
VANDEVUSSE_BENCHMARK_FEED_CONCENTRATION = 5.10
VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K = 403.15

# Retain the later-modeling-paper feed temperature only as a cross-check value;
# it is not the authoritative default for the benchmark implementation.
VANDEVUSSE_PINN_CROSSCHECK_DESIGN_PARAMS = np.array([5.10, 378.10], dtype=float)
VANDEVUSSE_DESIGN_PARAMS = np.array(
    [VANDEVUSSE_BENCHMARK_FEED_CONCENTRATION, VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K],
    dtype=float,
)

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
VANDEVUSSE_BASELINE_SETPOINTS_PHYS = np.array(
    [
        [0.90, 407.29],
        [0.85, 407.29],
        [0.93, 407.29],
        [0.90, 407.29],
    ],
    dtype=float,
)
VANDEVUSSE_BASELINE_THERMAL_STUDY_SETPOINTS_PHYS = np.array(
    [
        [0.90, 406.90],
        [0.90, 407.60],
    ],
    dtype=float,
)
VANDEVUSSE_BASELINE_SETPOINT_RANGE_PHYS = np.array([[0.85, 406.90], [0.93, 407.60]], dtype=float)
VANDEVUSSE_BASELINE_DISTURBANCE_PROFILES = ("none", "ca0_blocks")
VANDEVUSSE_BASELINE_CA0_BLOCKS = np.array([5.10, 4.70, 5.50, 5.10], dtype=float)
VANDEVUSSE_BASELINE_TIN_BLOCKS = np.array(
    [
        VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K,
        VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K,
        VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K,
        VANDEVUSSE_BENCHMARK_FEED_TEMPERATURE_K,
    ],
    dtype=float,
)
VANDEVUSSE_BASELINE_OBSERVER_POLES_DEFAULT = np.array([0.45, 0.50, 0.55, 0.60, 0.70, 0.75], dtype=float)
VANDEVUSSE_BASELINE_OBSERVER_POLES_FALLBACK = np.array([0.55, 0.60, 0.65, 0.70, 0.80, 0.85], dtype=float)
VANDEVUSSE_BASELINE_Q_OUT = np.array([1.0, 1.0], dtype=float)
VANDEVUSSE_BASELINE_R_IN = np.array([1.0, 1.0], dtype=float)
VANDEVUSSE_BASELINE_RUN_PROFILES = {
    ("nominal", "none"): {
        "n_tests": 2,
        "set_points_len": 100,
        "warm_start": 0,
        "test_cycle": [False, False],
        "plot_start_episode": 1,
    },
    ("disturb", "ca0_blocks"): {
        "n_tests": 20,
        "set_points_len": 100,
        "warm_start": 0,
        "test_cycle": [False, False, False, False, False],
        "plot_start_episode": 2,
    },
}

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
