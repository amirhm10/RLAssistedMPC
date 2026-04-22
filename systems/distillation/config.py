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
# Use one shared supervisory setpoint pair across all distillation notebooks so
# baseline and RL studies are directly comparable by default.
DISTILLATION_RL_SETPOINTS_PHYS = np.array([[0.013, -23.0], [0.028, -21.0]], dtype=float)
# Keep the combined study on the same supervisory targets as the other
# distillation RL notebooks so warm-start MPC behavior is directly comparable.
DISTILLATION_COMBINED_SETPOINTS_PHYS = DISTILLATION_RL_SETPOINTS_PHYS.copy()
DISTILLATION_OBSERVER_POLES = np.array([0.032, 0.03501095, 0.04099389, 0.04190188, 0.07477281, 0.01153274, 0.41036367])

HORIZON_PREDICT_GRID = list(range(4, 15))
HORIZON_CONTROL_GRID = list(range(2, 14))

DISTILLATION_MATRIX_ALPHA_UPPER_CAP = 1.1929
DISTILLATION_DEFAULT_MULTIPLIER_LOW = 0.75
DISTILLATION_DEFAULT_MULTIPLIER_HIGH = 1.25

MATRIX_MULTIPLIER_BOUNDS = {
    "td3": {
        "low": np.array([DISTILLATION_DEFAULT_MULTIPLIER_LOW, DISTILLATION_DEFAULT_MULTIPLIER_LOW, DISTILLATION_DEFAULT_MULTIPLIER_LOW], dtype=float),
        "high": np.array([min(DISTILLATION_DEFAULT_MULTIPLIER_HIGH, DISTILLATION_MATRIX_ALPHA_UPPER_CAP), DISTILLATION_DEFAULT_MULTIPLIER_HIGH, DISTILLATION_DEFAULT_MULTIPLIER_HIGH], dtype=float),
    },
    "sac": {
        "low": np.array([DISTILLATION_DEFAULT_MULTIPLIER_LOW, DISTILLATION_DEFAULT_MULTIPLIER_LOW, DISTILLATION_DEFAULT_MULTIPLIER_LOW], dtype=float),
        "high": np.array([min(DISTILLATION_DEFAULT_MULTIPLIER_HIGH, DISTILLATION_MATRIX_ALPHA_UPPER_CAP), DISTILLATION_DEFAULT_MULTIPLIER_HIGH, DISTILLATION_DEFAULT_MULTIPLIER_HIGH], dtype=float),
    },
}
WEIGHT_MULTIPLIER_BOUNDS = {
    "low": np.array([0.75, 0.75, 0.75, 0.75], dtype=float),
    "high": np.array([2.0, 2.0, 2.0, 2.0], dtype=float),
}
RESIDUAL_BOUNDS = {
    "low": np.array([-0.05, -0.05], dtype=float),
    "high": np.array([0.05, 0.05], dtype=float),
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
    "reward_scale": 1.0,
}

DISTILLATION_BASELINE_RUN_PROFILES = {
    ("nominal", "none"): {"n_tests": 2, "set_points_len": 100, "warm_start": 5, "test_cycle": [False, True], "plot_start_episode": 1},
    ("disturb", "ramp"): {"n_tests": 200, "set_points_len": 100, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2},
    ("disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2},
}

DISTILLATION_HORIZON_RUN_PROFILES = {
    ("nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
}

DISTILLATION_MATRIX_RUN_PROFILES = {
    ("td3", "nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "nominal", "none"): {"n_tests": 200, "set_points_len": 100, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 100, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 100, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
}

DISTILLATION_WEIGHT_RUN_PROFILES = {
    ("td3", "nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 5, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
}

DISTILLATION_RESIDUAL_RUN_PROFILES = {
    ("td3", "nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10
    , "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
}

DISTILLATION_REIDENTIFICATION_RUN_PROFILES = {
    ("td3", "nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("td3", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("sac", "disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
}

DISTILLATION_COMBINED_RUN_PROFILES = {
    ("nominal", "none"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("disturb", "ramp"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
    ("disturb", "fluctuation"): {"n_tests": 200, "set_points_len": 200, "warm_start": 10, "test_cycle": [False, False, False, False, False], "plot_start_episode": 2, "compare_start_episode": 2},
}

ARCHIVED_DISTILLATION_ROOT = Path("DIstillation Column Case") / "RL_assisted_MPC_DL"
DISTILLATION_ROOT = Path("Distillation")
DISTILLATION_DATA_SUBDIR = DISTILLATION_ROOT / "Data"
DISTILLATION_RESULT_SUBDIR = DISTILLATION_ROOT / "Results"

DEFAULT_ASPEN_ROOT = Path(
    os.environ.get(
        "DISTILLATION_ASPEN_ROOT",
        r"C:\Users\HAMEDI\Desktop\FinalDocuments\FinalDocuments\C2SplitterControlFiles\AspenFiles\dynsim\Plant",
    )
)

_FAMILY_FILE_MAP = {
    # Distillation Aspen defaults use one simulation number per notebook family.
    # All disturbance profiles for a family resolve to the same file, and new
    # families should take the next unused simulation number.
    "system_id": {"none": "C2S_SS_simulation1.dynf", "ramp": "C2S_SS_simulation1.dynf", "fluctuation": "C2S_SS_simulation1.dynf"},
    "baseline": {"none": "C2S_SS_simulation2.dynf", "ramp": "C2S_SS_simulation2.dynf", "fluctuation": "C2S_SS_simulation2.dynf"},
    "horizon": {"none": "C2S_SS_simulation3.dynf", "ramp": "C2S_SS_simulation3.dynf", "fluctuation": "C2S_SS_simulation3.dynf"},
    "horizon_dueling": {"none": "C2S_SS_simulation4.dynf", "ramp": "C2S_SS_simulation4.dynf", "fluctuation": "C2S_SS_simulation4.dynf"},
    "matrix": {"none": "C2S_SS_simulation5.dynf", "ramp": "C2S_SS_simulation5.dynf", "fluctuation": "C2S_SS_simulation5.dynf"},
    "structured_matrix": {"none": "C2S_SS_simulation6.dynf", "ramp": "C2S_SS_simulation6.dynf", "fluctuation": "C2S_SS_simulation6.dynf"},
    "weights": {"none": "C2S_SS_simulation7.dynf", "ramp": "C2S_SS_simulation7.dynf", "fluctuation": "C2S_SS_simulation7.dynf"},
    "residual": {"none": "C2S_SS_simulation8.dynf", "ramp": "C2S_SS_simulation8.dynf", "fluctuation": "C2S_SS_simulation8.dynf"},
    "combined": {"none": "C2S_SS_simulation9.dynf", "ramp": "C2S_SS_simulation9.dynf", "fluctuation": "C2S_SS_simulation9.dynf"},
    "reidentification": {"none": "C2S_SS_simulation10.dynf", "ramp": "C2S_SS_simulation10.dynf", "fluctuation": "C2S_SS_simulation10.dynf"},
}


def default_plant_paths(family, disturbance_profile):
    disturbance_profile = str(disturbance_profile).lower()
    family = str(family).lower()
    family = {
        "system_identification": "system_id",
        "horizon_standard": "horizon",
        "matrix_td3": "matrix",
        "matrix_sac": "matrix",
        "structured_matrix_td3": "structured_matrix",
        "structured_matrix_sac": "structured_matrix",
    }.get(family, family)
    if family not in _FAMILY_FILE_MAP:
        raise KeyError(f"Unknown distillation notebook family: {family}")
    if disturbance_profile not in _FAMILY_FILE_MAP[family]:
        raise KeyError(f"Unknown disturbance profile '{disturbance_profile}' for {family}.")
    dyn_name = _FAMILY_FILE_MAP[family][disturbance_profile]
    dyn_path = DEFAULT_ASPEN_ROOT / dyn_name
    snaps_path = DEFAULT_ASPEN_ROOT / dyn_name.replace(".dynf", "")
    return dyn_path, snaps_path


def resolve_aspen_paths(
    family,
    disturbance_profile,
    aspen_preset=None,
    dyn_path_override=None,
    snaps_path_override=None,
    aspen_root=None,
):
    root = Path(aspen_root) if aspen_root else DEFAULT_ASPEN_ROOT
    source = "family-default"

    if dyn_path_override:
        dyn_path = Path(dyn_path_override).expanduser()
        source = "manual-path"
    elif aspen_preset not in (None, "", "default", "auto"):
        preset_str = str(aspen_preset).strip()
        if preset_str.isdigit():
            dyn_path = root / f"C2S_SS_simulation{int(preset_str)}.dynf"
            source = f"preset-{int(preset_str)}"
        else:
            raise ValueError("ASPEN_PRESET must be an integer simulation number, 'default', or empty.")
    else:
        dyn_path, _ = default_plant_paths(family, disturbance_profile)

    if snaps_path_override:
        snaps_path = Path(snaps_path_override).expanduser()
    else:
        snaps_path = dyn_path.with_suffix("")

    return dyn_path, snaps_path, source
