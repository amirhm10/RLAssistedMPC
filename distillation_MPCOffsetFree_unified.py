# Generated from distillation_MPCOffsetFree_unified.ipynb

# --- Cell 0 (markdown) ---
# # Distillation Offset-Free MPC Baseline
#
# Canonical disturbed OF-MPC baseline for the two-phase 200+100 schedule.

# --- Cell 1 (markdown) ---
# ## User Config

# --- Cell 2 (code) ---
from pathlib import Path
import os
import pickle

from systems.distillation import (
    canonical_distillation_training_profile,
    default_distillation_profile_episode_count,
    get_distillation_notebook_defaults,
)
from utils.notebook_setup import prepare_distillation_notebook_env, print_grouped_notebook_summary

NB = get_distillation_notebook_defaults("baseline")
RUN_MODE = NB["run_mode"]
DISTURBANCE_PROFILE = NB["disturbance_profile"]
STYLE_PROFILE = NB["style_profile"]
SAVE_PDF = NB["save_pdf"]
ASPEN_PRESET = NB["aspen_preset"]
ASPEN_PATH_OVERRIDE = NB["aspen_path_override"]
SNAPS_PATH_OVERRIDE = NB["snaps_path_override"]
ASPEN_ROOT_OVERRIDE = NB["aspen_root_override"]
DISTILLATION_VISIBLE = NB["distillation_visible"]
DISTILLATION_DATA_DIR_OVERRIDE = NB["data_dir_override"]
DISTILLATION_RESULTS_DIR_OVERRIDE = NB["results_dir_override"]
RESULT_PREFIX_OVERRIDE = NB["result_prefix_override"]
BASELINE_SAVE_PATH_OVERRIDE = NB["baseline_save_path_override"]
N_TESTS_OVERRIDE = NB["n_tests_override"]
SET_POINTS_LEN_OVERRIDE = NB["set_points_len_override"]
TEST_CYCLE_OVERRIDE = NB["test_cycle_override"]
PLOT_START_EPISODE_OVERRIDE = NB["plot_start_episode_override"]
TRAINING_PROFILE_OVERRIDE = NB.get("training_profile_override")

REPO_ROOT, DATA_DIR, RESULT_DIR, DISTURBANCE_PROFILE, DYN_PATH, SNAPS_PATH, ASPEN_SOURCE = prepare_distillation_notebook_env(run_mode=RUN_MODE, disturbance_profile=DISTURBANCE_PROFILE, family="baseline", aspen_preset=ASPEN_PRESET, dyn_path_override=ASPEN_PATH_OVERRIDE, snaps_path_override=SNAPS_PATH_OVERRIDE, aspen_root_override=ASPEN_ROOT_OVERRIDE, data_dir_override=DISTILLATION_DATA_DIR_OVERRIDE, results_dir_override=DISTILLATION_RESULTS_DIR_OVERRIDE)
os.chdir(REPO_ROOT)

# --- Cell 3 (markdown) ---
# ## Imports

# --- Cell 4 (code) ---
import numpy as np

from Simulation.mpc import MpcSolverGeneral
from systems.distillation.data_io import canonical_baseline_path, load_distillation_system_data
from systems.distillation.labels import DISTILLATION_SYSTEM_METADATA
from systems.distillation.plant import build_distillation_system, distillation_system_stepper
from systems.distillation.scenarios import build_distillation_training_profile
from utils.helpers import apply_min_max
from utils.mpc_baseline_runner import run_offsetfree_mpc
from utils.plotting import plot_baseline_mpc_results
from utils.rewards import make_reward_fn_relative_QR

# --- Cell 5 (markdown) ---
# ## System And Data Setup

# --- Cell 6 (code) ---
# Build the plant, load the canonical data bundle, and prepare the supervisory setpoint scenario.
SYS = NB["system_setup"]
nominal_conditions = SYS["nominal_conditions"].copy()
ss_inputs = SYS["ss_inputs"].copy()
u_min = SYS["input_bounds"]["u_min"].copy()
u_max = SYS["input_bounds"]["u_max"].copy()
setpoint_y = SYS["setpoint_range_phys"].copy()
system = build_distillation_system(path=DYN_PATH, ss_inputs=ss_inputs, initialization_point=nominal_conditions, delta_t=SYS["delta_t_hours"], visible=DISTILLATION_VISIBLE)
steady_states = {"ss_inputs": np.asarray(system.ss_inputs, float).copy(), "y_ss": np.asarray(system.y_ss, float).copy()}
system_data = load_distillation_system_data(REPO_ROOT, steady_states, setpoint_y, u_min, u_max, data_override=DISTILLATION_DATA_DIR_OVERRIDE)
A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
inputs_number = int(B_aug.shape[1])
y_sp_scenario_phys = SYS["rl_setpoints_phys"].copy()
y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[inputs_number:], data_max[inputs_number:]) - apply_min_max(steady_states["y_ss"], data_min[inputs_number:], data_max[inputs_number:])

# --- Cell 7 (markdown) ---
# ## Controller And Reward Setup

# --- Cell 8 (code) ---
# Run-profile, controller, reward, and agent setup.
CTRL = NB["controller"]
REWARD_CFG = NB["reward"]
RUN_PROFILE = NB["run_profiles"][(RUN_MODE, DISTURBANCE_PROFILE)]
poles = SYS["observer_poles"].copy()
n_inputs = int(B_aug.shape[1])
predict_h = CTRL["predict_h"]
cont_h = CTRL["cont_h"]
Q1_penalty = CTRL["Q1_penalty"]
Q2_penalty = CTRL["Q2_penalty"]
R1_penalty = CTRL["R1_penalty"]
R2_penalty = CTRL["R2_penalty"]
USE_SHIFTED_MPC_WARM_START = CTRL["use_shifted_mpc_warm_start"]
TRAINING_PROFILE_NAME = canonical_distillation_training_profile(
    TRAINING_PROFILE_OVERRIDE or RUN_PROFILE.get("profile_name", "legacy_200")
)
profile_default_n_tests = (
    default_distillation_profile_episode_count(TRAINING_PROFILE_NAME)
    if TRAINING_PROFILE_OVERRIDE is not None
    else RUN_PROFILE["n_tests"]
)
n_tests = int(profile_default_n_tests if N_TESTS_OVERRIDE is None else N_TESTS_OVERRIDE)
set_points_len = int(RUN_PROFILE["set_points_len"] if SET_POINTS_LEN_OVERRIDE is None else SET_POINTS_LEN_OVERRIDE)
warm_start = int(RUN_PROFILE["warm_start"])
if warm_start >= n_tests:
    warm_start = max(0, n_tests - 1)
TEST_CYCLE = list(RUN_PROFILE["test_cycle"] if TEST_CYCLE_OVERRIDE is None else TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = int(RUN_PROFILE.get("plot_start_episode", 1) if PLOT_START_EPISODE_OVERRIDE is None else PLOT_START_EPISODE_OVERRIDE)
PROFILE_SUFFIX = "" if TRAINING_PROFILE_NAME == "legacy_200" else f"_{TRAINING_PROFILE_NAME}"
RESULT_PREFIX = RESULT_PREFIX_OVERRIDE or f"distillation_baseline_{RUN_MODE}_{DISTURBANCE_PROFILE}{PROFILE_SUFFIX}"
BASELINE_SAVE_PATH = Path(BASELINE_SAVE_PATH_OVERRIDE).expanduser() if BASELINE_SAVE_PATH_OVERRIDE else canonical_baseline_path(
    REPO_ROOT,
    RUN_MODE,
    DISTURBANCE_PROFILE,
    data_override=DISTILLATION_DATA_DIR_OVERRIDE,
    training_profile_name=TRAINING_PROFILE_NAME,
)
if not BASELINE_SAVE_PATH.is_absolute():
    BASELINE_SAVE_PATH = Path(REPO_ROOT) / BASELINE_SAVE_PATH
BASELINE_SAVE_PATH = BASELINE_SAVE_PATH.resolve()
if BASELINE_SAVE_PATH.exists():
    raise FileExistsError(f"Refusing to overwrite existing baseline data: {BASELINE_SAVE_PATH}")

DISTURBANCE_NOMINAL_FEED = float(system.feed.FmR.Value)
EPISODE_BUNDLE = build_distillation_training_profile(
    profile_name=TRAINING_PROFILE_NAME,
    run_mode=RUN_MODE,
    disturbance_profile=DISTURBANCE_PROFILE,
    y_sp_scenario_phys=y_sp_scenario_phys,
    n_tests=n_tests,
    set_points_len=set_points_len,
    warm_start=warm_start,
    test_cycle=TEST_CYCLE,
    steady_outputs=steady_states["y_ss"],
    data_min=data_min,
    data_max=data_max,
    n_inputs=inputs_number,
    nominal_feed=DISTURBANCE_NOMINAL_FEED,
)
disturbance_schedule = EPISODE_BUNDLE["disturbance_schedule"]
# Reward setup.
reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, n_inputs, **REWARD_CFG)
MPC_obj = MpcSolverGeneral(A_aug, B_aug, C_aug, Q_out=np.array([Q1_penalty, Q2_penalty], float), R_in=np.array([R1_penalty, R2_penalty], float), NP=predict_h, NC=cont_h)

# --- Cell 9 (markdown) ---
# ## Resolved Summary

# --- Cell 10 (code) ---
print_grouped_notebook_summary(
    "Distillation baseline run summary",
    {
        "Paths": {"Repo root": REPO_ROOT, "Data dir": DATA_DIR, "Results dir": RESULT_DIR, "Aspen source": ASPEN_SOURCE, "Dyn path": DYN_PATH, "Snaps path": SNAPS_PATH, "Baseline save path": BASELINE_SAVE_PATH},
        "Run setup": {"Run mode": RUN_MODE, "Disturbance profile": DISTURBANCE_PROFILE, "training_profile_name": TRAINING_PROFILE_NAME, "n_tests": n_tests, "set_points_len": set_points_len, "warm_start": warm_start, "test_cycle": TEST_CYCLE, "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START, "phase_boundary_step": EPISODE_BUNDLE.get("phase_switch_step")},
        "System / controller": {"delta_t_hours": SYS["delta_t_hours"], "predict_h": predict_h, "cont_h": cont_h, "observer_poles": poles.tolist(), "phase1_setpoints_phys": EPISODE_BUNDLE.get("phase1_setpoints_phys").tolist(), "phase2_setpoints_phys": None if EPISODE_BUNDLE.get("phase2_setpoints_phys") is None else EPISODE_BUNDLE["phase2_setpoints_phys"].tolist()},
        "Reward": reward_params,
        "Plotting / export": {"style_profile": STYLE_PROFILE, "save_pdf": SAVE_PDF, "result_prefix": RESULT_PREFIX, "plot_start_episode": PLOT_START_EPISODE},
    },
)

# --- Cell 11 (markdown) ---
# ## Run

# --- Cell 12 (code) ---
# Assemble the shared runner configuration and execute the rollout.
mpc_cfg = {
    "run_mode": RUN_MODE,
    "training_profile_name": TRAINING_PROFILE_NAME,
    "n_tests": n_tests,
    "set_points_len": set_points_len,
    "warm_start": warm_start,
    "test_cycle": TEST_CYCLE,
    "predict_h": predict_h,
    "cont_h": cont_h,
    "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
    "Q1_penalty": Q1_penalty,
    "Q2_penalty": Q2_penalty,
    "R1_penalty": R1_penalty,
    "R2_penalty": R2_penalty,
    "nominal_qi": CTRL["nominal_qi"],
    "nominal_qs": CTRL["nominal_qs"],
    "nominal_ha": CTRL["nominal_ha"],
    "qi_change": CTRL["qi_change"],
    "qs_change": CTRL["qs_change"],
    "ha_change": CTRL["ha_change"],
    "b_min": system_data["b_min"],
    "b_max": system_data["b_max"],
}

runtime_ctx = {
    "system": system,
    "MPC_obj": MPC_obj,
    "steady_states": steady_states,
    "data_min": data_min,
    "data_max": data_max,
    "A_aug": A_aug,
    "B_aug": B_aug,
    "C_aug": C_aug,
    "y_sp_scenario": y_sp_scenario,
    "episode_bundle": EPISODE_BUNDLE,
    "reward_fn": reward_fn,
    "system_metadata": DISTILLATION_SYSTEM_METADATA,
    "poles": poles,
    "disturbance_schedule": disturbance_schedule,
    "system_stepper": distillation_system_stepper,
    "disturbance_labels": DISTILLATION_SYSTEM_METADATA["disturbance_labels"],
}

result_bundle = run_offsetfree_mpc(mpc_cfg=mpc_cfg, runtime_ctx=runtime_ctx)
result_bundle["experiment_metadata"] = {
    "experiment_name": TRAINING_PROFILE_NAME,
    "phase1_episodes": 200,
    "phase2_episodes": 100,
    "phase_boundary_episode": EPISODE_BUNDLE.get("phase_switch_episode"),
    "phase_boundary_step": EPISODE_BUNDLE.get("phase_switch_step"),
    "pickle_path": str(BASELINE_SAVE_PATH),
}

# --- Cell 13 (markdown) ---
# ## Plotting And Export

# --- Cell 14 (code) ---
out_dir = plot_baseline_mpc_results(
    result_bundle=result_bundle,
    plot_cfg={
        "directory": RESULT_DIR,
        "prefix_name": RESULT_PREFIX,
        "start_episode": PLOT_START_EPISODE,
        "save_pdf": SAVE_PDF,
        "style_profile": STYLE_PROFILE,
    },
)
with open(BASELINE_SAVE_PATH, "xb") as handle:
    pickle.dump(result_bundle, handle)
print(f"Saved distillation baseline to: {BASELINE_SAVE_PATH}")
print(f"Baseline plot/output directory: {out_dir}")

try:
    system.close(SNAPS_PATH)
except Exception as exc:
    print(f"Could not close Aspen cleanly: {exc}")

# --- Cell 15 (code) ---
