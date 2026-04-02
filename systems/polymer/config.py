from pathlib import Path

import numpy as np


POLYMER_ROOT = Path("Polymer")
POLYMER_DATA_SUBDIR = POLYMER_ROOT / "Data"
POLYMER_RESULT_SUBDIR = POLYMER_ROOT / "Results"

POLYMER_DELTA_T_HOURS = 0.5

POLYMER_SYSTEM_PARAMS = np.array(
    [2.142e17, 14897.0, 3.816e10, 3557.0, 4.50e12, 843.0, 0.6, -6.99e4, 1.05e6, 1506.0, 4043.0, 104.14],
    dtype=float,
)
POLYMER_DESIGN_PARAMS = np.array([0.5888, 8.6981, 108.0, 459.0, 330.0, 295.0, 3000.0, 3312.4], dtype=float)
POLYMER_SS_INPUTS = np.array([471.6, 378.0], dtype=float)

POLYMER_INPUT_BOUNDS = {
    "u_min": np.array([71.6, 78.0], dtype=float),
    "u_max": np.array([870.0, 670.0], dtype=float),
}

POLYMER_SETPOINT_RANGE_PHYS = np.array([[3.2, 321.0], [4.5, 325.0]], dtype=float)
POLYMER_RL_SETPOINTS_PHYS = np.array([[4.5, 324.0], [3.4, 321.0]], dtype=float)

POLYMER_OBSERVER_POLES = np.array(
    [0.44619852, 0.33547649, 0.36380595, 0.70467118, 0.3562966, 0.42900673, 0.4228262, 0.96916776, 0.91230187],
    dtype=float,
)

HORIZON_PREDICT_GRID = list(range(8, 20))
HORIZON_CONTROL_GRID = list(range(3, 10))

RL_REWARD_DEFAULTS = {
    "k_rel": np.array([0.003, 0.0003], dtype=float),
    "band_floor_phys": np.array([0.006, 0.07], dtype=float),
    "Q_diag": np.array([518.0, 90.0], dtype=float),
    "R_diag": np.array([90.0, 90.0], dtype=float),
    "tau_frac": 0.7,
    "gamma_out": 0.5,
    "gamma_in": 0.5,
    "beta": 7.0,
    "gate": "geom",
    "lam_in": 1.0,
    "bonus_kind": "exp",
    "bonus_k": 12.0,
    "bonus_p": 0.6,
    "bonus_c": 20.0,
}
