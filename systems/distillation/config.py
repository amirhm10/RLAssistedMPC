import os
from pathlib import Path

import numpy as np


DELTA_T_HOURS = 1.0 / 6.0

DISTILLATION_NOMINAL_CONDITIONS = np.array(
    [1.50032484e5, -2.10309105e1, 2.08083248e1, 6.30485237e-1, 3.69514734e-1, -2.4e1],
    dtype=float,
)
DISTILLATION_SS_INPUTS = np.array([320000.0, 110.0], dtype=float)
DISTILLATION_SS_INPUTS_SYSTEM_ID = np.array([400000.0, 130.0], dtype=float)
DISTILLATION_INPUT_BOUNDS = {
    "u_min": np.array([300000.0, 100.0], dtype=float),
    "u_max": np.array([460000.0, 150.0], dtype=float),
}
DISTILLATION_SETPOINT_RANGE_PHYS = np.array([[0.002, -26.0], [0.05, -16.0]], dtype=float)
DISTILLATION_RL_SETPOINTS_PHYS = np.array([[0.013, -23.0], [0.028, -21.0]], dtype=float)
DISTILLATION_COMBINED_SETPOINTS_PHYS = np.array([[0.013, -23.0], [0.018, -22.0]], dtype=float)

HORIZON_PREDICT_GRID = list(range(4, 15))
HORIZON_CONTROL_GRID = list(range(2, 14))

MATRIX_MULTIPLIER_BOUNDS = {
    "td3": {
        "low": np.array([0.9, 0.9, 0.9], dtype=float),
        "high": np.array([1.1, 1.1, 1.1], dtype=float),
    },
    "sac": {
        "low": np.array([0.95, 0.95, 0.95], dtype=float),
        "high": np.array([1.05, 1.05, 1.05], dtype=float),
    },
}
WEIGHT_MULTIPLIER_BOUNDS = {
    "low": np.array([0.5, 0.5, 0.5, 0.5], dtype=float),
    "high": np.array([3.0, 3.0, 3.0, 3.0], dtype=float),
}
RESIDUAL_BOUNDS = {
    "low": np.array([-0.5, -0.5], dtype=float),
    "high": np.array([0.5, 0.5], dtype=float),
}

RL_REWARD_DEFAULTS = {
    "k_rel": np.array([0.3, 0.02], dtype=float),
    "band_floor_phys": np.array([0.003, 0.3], dtype=float),
    "Q_diag": np.array([3.7e4, 1.5e3], dtype=float),
    "R_diag": np.array([2.5e3, 2.5e3], dtype=float),
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

ARCHIVED_DISTILLATION_ROOT = Path("DIstillation Column Case") / "RL_assisted_MPC_DL"
DISTILLATION_DATA_SUBDIR = Path("Data") / "distillation"
DISTILLATION_RESULT_SUBDIR = Path("Result") / "distillation"

DEFAULT_ASPEN_ROOT = Path(
    os.environ.get(
        "DISTILLATION_ASPEN_ROOT",
        r"C:\Users\HAMEDI\Desktop\FinalDocuments\FinalDocuments\C2SplitterControlFiles\AspenFiles\dynsim\Plant",
    )
)

_FAMILY_FILE_MAP = {
    "system_id": {"none": "C2S_SS_simulation4.dynf", "ramp": "C2S_SS_simulation1.dynf", "fluctuation": "C2S_SS_simulation2.dynf"},
    "baseline": {"none": "C2S_SS_simulation4.dynf", "ramp": "C2S_SS_simulation1.dynf", "fluctuation": "C2S_SS_simulation2.dynf"},
    "horizon": {"none": "C2S_SS_simulation4.dynf", "ramp": "C2S_SS_simulation5.dynf", "fluctuation": "C2S_SS_simulation6.dynf"},
    "matrix_td3": {"none": "C2S_SS_simulation1.dynf", "ramp": "C2S_SS_simulation3.dynf", "fluctuation": "C2S_SS_simulation2.dynf"},
    "matrix_sac": {"none": "C2S_SS_simulation7.dynf", "ramp": "C2S_SS_simulation8.dynf", "fluctuation": "C2S_SS_simulation9.dynf"},
    "weights": {"none": "C2S_SS_simulation10.dynf", "ramp": "C2S_SS_simulation11.dynf", "fluctuation": "C2S_SS_simulation12.dynf"},
    "residual": {"none": "C2S_SS_simulation10.dynf", "ramp": "C2S_SS_simulation11.dynf", "fluctuation": "C2S_SS_simulation12.dynf"},
    "combined": {"none": "C2S_SS_simulation10.dynf", "ramp": "C2S_SS_simulation11.dynf", "fluctuation": "C2S_SS_simulation12.dynf"},
}


def default_plant_paths(family, disturbance_profile):
    disturbance_profile = str(disturbance_profile).lower()
    family = str(family).lower()
    if family not in _FAMILY_FILE_MAP:
        raise KeyError(f"Unknown distillation notebook family: {family}")
    if disturbance_profile not in _FAMILY_FILE_MAP[family]:
        raise KeyError(f"Unknown disturbance profile '{disturbance_profile}' for {family}.")
    dyn_name = _FAMILY_FILE_MAP[family][disturbance_profile]
    dyn_path = DEFAULT_ASPEN_ROOT / dyn_name
    snaps_path = DEFAULT_ASPEN_ROOT / dyn_name.replace(".dynf", "")
    return dyn_path, snaps_path
