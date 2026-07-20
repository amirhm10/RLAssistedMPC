# Generated from MPCOffsetFree_unified.ipynb

# --- Cell 0 (markdown) ---
# # Polymer Offset-Free MPC Baseline
#
# Canonical polymer baseline MPC notebook. Edit the config cell below for one-off runs, or change `systems/polymer/notebook_params.py` for repo-wide defaults.

# --- Cell 1 (markdown) ---
# ## User Config

# --- Cell 2 (code) ---
from pathlib import Path
import os
import pickle

from systems.polymer import (
    canonical_polymer_training_profile,
    default_polymer_profile_episode_count,
    get_polymer_notebook_defaults,
)
from systems.polymer.data_io import canonical_baseline_path
from utils.notebook_setup import prepare_polymer_notebook_env, print_grouped_notebook_summary

NB = get_polymer_notebook_defaults("baseline")

# Main notebook controls.
# Edit the values below for a one-off run, or edit systems/polymer/notebook_params.py
# if you want the repo-wide defaults to change for every polymer notebook.
RUN_MODE = NB["run_mode"]  # "nominal" | "disturb"
STYLE_PROFILE = NB["style_profile"]  # "hybrid" | "paper" | "debug"
SAVE_PDF = NB["save_pdf"]  # False -> PNG only, True -> PNG + PDF

# Canonical polymer paths. Leave as None to use Polymer/Data and Polymer/Results.
POLYMER_DATA_DIR_OVERRIDE = NB["data_dir_override"]
POLYMER_RESULTS_DIR_OVERRIDE = NB["results_dir_override"]

# Optional naming/path overrides. Leave as None to use the canonical mode-based names.
RESULT_PREFIX_OVERRIDE = NB["result_prefix_override"]
BASELINE_SAVE_PATH_OVERRIDE = NB["baseline_save_path_override"]

# Optional run-size overrides. Leave as None to use the mode defaults from the shared parameter file.
N_TESTS_OVERRIDE = NB["n_tests_override"]
SET_POINTS_LEN_OVERRIDE = NB["set_points_len_override"]
WARM_START_OVERRIDE = NB["warm_start_override"]
TEST_CYCLE_OVERRIDE = NB["test_cycle_override"]
PLOT_START_EPISODE_OVERRIDE = NB["plot_start_episode_override"]
TRAINING_PROFILE_OVERRIDE = NB.get("training_profile_override")

REPO_ROOT, DATA_DIR, RESULT_DIR = prepare_polymer_notebook_env(
    data_dir_override=POLYMER_DATA_DIR_OVERRIDE,
    results_dir_override=POLYMER_RESULTS_DIR_OVERRIDE,
)
os.chdir(REPO_ROOT)

RUN_PROFILE = NB["run_profiles"][RUN_MODE]

# A full grouped summary is printed later once the runtime configuration is fully resolved.

# --- Cell 3 (markdown) ---
# ## Imports

# --- Cell 4 (code) ---
import numpy as np

from Simulation.mpc import MpcSolverGeneral
from Simulation.system_functions import PolymerCSTR
from systems.polymer import (
    POLYMER_DELTA_T_HOURS,
    POLYMER_DESIGN_PARAMS,
    POLYMER_INPUT_BOUNDS,
    POLYMER_OBSERVER_POLES,
    POLYMER_RL_SETPOINTS_PHYS,
    POLYMER_SETPOINT_RANGE_PHYS,
    POLYMER_SS_INPUTS,
    POLYMER_SYSTEM_METADATA,
    POLYMER_SYSTEM_PARAMS,
    RL_REWARD_DEFAULTS,
    load_polymer_system_data,
)
from utils.helpers import apply_min_max
from utils.mpc_baseline_runner import run_offsetfree_mpc
from utils.plotting import plot_baseline_mpc_results
from utils.rewards import make_reward_fn_relative_QR

# --- Cell 5 (markdown) ---
# ## System And Data Setup

# --- Cell 6 (code) ---
# Build the polymer plant and load the canonical polymer model/scaling bundle.
SYS = NB["system_setup"]
system_params = SYS["system_params"].copy()
system_design_params = SYS["design_params"].copy()
system_steady_state_inputs = SYS["ss_inputs"].copy()
delta_t = SYS["delta_t_hours"]

cstr = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t)
steady_states = {"ss_inputs": cstr.ss_inputs, "y_ss": cstr.y_ss}

setpoint_y = SYS["setpoint_range_phys"].copy()
u_min = SYS["input_bounds"]["u_min"].copy()
u_max = SYS["input_bounds"]["u_max"].copy()

system_data = load_polymer_system_data(
    REPO_ROOT,
    steady_states=steady_states,
    setpoint_y=setpoint_y,
    u_min=u_min,
    u_max=u_max,
    n_inputs=2,
    data_override=POLYMER_DATA_DIR_OVERRIDE,
)

A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
min_max_dict = system_data["min_max_dict"]

inputs_number = int(B_aug.shape[1])
y_sp_scenario_phys = SYS["rl_setpoints_phys"].copy()
y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[inputs_number:], data_max[inputs_number:]) - apply_min_max(
    steady_states["y_ss"], data_min[inputs_number:], data_max[inputs_number:]
)

# --- Cell 7 (markdown) ---
# ## Controller And Reward Setup

# --- Cell 8 (code) ---
# Controller, reward, and plotting defaults.
# All of these values come from systems/polymer/notebook_params.py unless you override them here.
CTRL = NB["controller"]
REWARD_CFG = NB["reward"]

TRAINING_PROFILE_NAME = canonical_polymer_training_profile(
    TRAINING_PROFILE_OVERRIDE or RUN_PROFILE.get("profile_name", "legacy_gradual_200")
)
profile_default_n_tests = (
    default_polymer_profile_episode_count(TRAINING_PROFILE_NAME)
    if TRAINING_PROFILE_OVERRIDE is not None
    else RUN_PROFILE["n_tests"]
)
n_tests = profile_default_n_tests if N_TESTS_OVERRIDE is None else int(N_TESTS_OVERRIDE)
set_points_len = RUN_PROFILE["set_points_len"] if SET_POINTS_LEN_OVERRIDE is None else int(SET_POINTS_LEN_OVERRIDE)
warm_start = RUN_PROFILE.get("warm_start", 0) if WARM_START_OVERRIDE is None else int(WARM_START_OVERRIDE)
TEST_CYCLE = list(RUN_PROFILE["test_cycle"]) if TEST_CYCLE_OVERRIDE is None else list(TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = RUN_PROFILE.get("plot_start_episode", 1) if PLOT_START_EPISODE_OVERRIDE is None else int(PLOT_START_EPISODE_OVERRIDE)
RESULT_PREFIX = RESULT_PREFIX_OVERRIDE or RUN_PROFILE["result_prefix"]
BASELINE_SAVE_PATH = Path(BASELINE_SAVE_PATH_OVERRIDE).expanduser() if BASELINE_SAVE_PATH_OVERRIDE else canonical_baseline_path(
    REPO_ROOT,
    RUN_MODE,
    data_override=POLYMER_DATA_DIR_OVERRIDE,
    training_profile_name=TRAINING_PROFILE_NAME,
)

poles = SYS["observer_poles"].copy()

n_inputs = int(B_aug.shape[1])
predict_h = CTRL["predict_h"]
cont_h = CTRL["cont_h"]
Q1_penalty = CTRL["Q1_penalty"]
Q2_penalty = CTRL["Q2_penalty"]
R1_penalty = CTRL["R1_penalty"]
R2_penalty = CTRL["R2_penalty"]
USE_SHIFTED_MPC_WARM_START = CTRL["use_shifted_mpc_warm_start"]

u_ss = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
b_min = apply_min_max(SYS["input_bounds"]["u_min"], data_min[:n_inputs], data_max[:n_inputs]) - u_ss
b_max = apply_min_max(SYS["input_bounds"]["u_max"], data_min[:n_inputs], data_max[:n_inputs]) - u_ss

MPC_obj = MpcSolverGeneral(
    A_aug,
    B_aug,
    C_aug,
    Q_out=np.array([Q1_penalty, Q2_penalty]),
    R_in=np.array([R1_penalty, R2_penalty]),
    NP=predict_h,
    NC=cont_h,
)

# Reward setup.
reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, n_inputs=n_inputs, **REWARD_CFG)

print_grouped_notebook_summary(
    "Resolved baseline parameters",
    {
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "warm_start": warm_start,
        "plot_start_episode": PLOT_START_EPISODE,
        "training_profile_name": TRAINING_PROFILE_NAME,
        "baseline_save_path": BASELINE_SAVE_PATH,
        "Use shifted MPC warm start": USE_SHIFTED_MPC_WARM_START,
    },
)

# --- Cell 9 (markdown) ---
# ## Resolved Summary

# --- Cell 10 (code) ---
print_grouped_notebook_summary(
    "Polymer baseline run summary",
    {
        "Paths": {"Repo root": REPO_ROOT, "Data dir": DATA_DIR, "Results dir": RESULT_DIR, "Baseline save path": BASELINE_SAVE_PATH},
        "Run setup": {"Run mode": RUN_MODE, "training_profile_name": TRAINING_PROFILE_NAME, "n_tests": n_tests, "set_points_len": set_points_len, "warm_start": warm_start, "test_cycle": TEST_CYCLE, "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START},
        "System / controller": {"delta_t_hours": delta_t, "predict_h": predict_h, "cont_h": cont_h, "observer_poles": poles.tolist(), "setpoints_phys": y_sp_scenario_phys.tolist()},
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
    "nominal_qi": RUN_PROFILE["nominal_qi"],
    "nominal_qs": RUN_PROFILE["nominal_qs"],
    "nominal_ha": RUN_PROFILE["nominal_ha"],
    "qi_change": RUN_PROFILE["qi_change"],
    "qs_change": RUN_PROFILE["qs_change"],
    "ha_change": RUN_PROFILE["ha_change"],
    "b_min": b_min,
    "b_max": b_max,
}

runtime_ctx = {
    "system": PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t),
    "MPC_obj": MPC_obj,
    "steady_states": steady_states,
    "min_max_dict": min_max_dict,
    "data_min": data_min,
    "data_max": data_max,
    "A_aug": A_aug,
    "B_aug": B_aug,
    "C_aug": C_aug,
    "poles": poles,
    "y_sp_scenario": y_sp_scenario,
    "reward_fn": reward_fn,
    "system_metadata": POLYMER_SYSTEM_METADATA,
    "reward_params": reward_params,
}

result_bundle = run_offsetfree_mpc(mpc_cfg=mpc_cfg, runtime_ctx=runtime_ctx)
result_bundle["reward_params"] = reward_params
result_bundle["run_profile"] = RUN_PROFILE

print(f"nFE: {result_bundle['nFE']}")
print(f"Avg reward samples: {len(result_bundle['avg_rewards'])}")

# --- Cell 13 (markdown) ---
# ## Plotting And Export

# --- Cell 14 (code) ---
# Generate saved figures for the baseline MPC run.
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

legacy_payload = {
    "u_mpc": result_bundle["u"],
    "y_mpc": result_bundle["y"],
    "avg_rewards": result_bundle["avg_rewards"],
    "avg_rewards_mpc": result_bundle["avg_rewards_mpc"],
    "rewards_step": result_bundle["rewards_step"],
    "rewards_mpc": result_bundle["rewards_mpc"],
    "xhatdhat": result_bundle["xhatdhat"],
    "yhat": result_bundle["yhat"],
    "delta_y_storage": result_bundle["delta_y_storage"],
    "delta_u_storage": result_bundle["delta_u_storage"],
    "delat_u_storage": result_bundle["delta_u_storage"],
    "run_mode": result_bundle["run_mode"],
    "disturbance_profile": result_bundle["disturbance_profile"],
    "steady_states": result_bundle["steady_states"],
    "y_sp": result_bundle["y_sp"],
    "nFE": result_bundle["nFE"],
    "delta_t": result_bundle["delta_t"],
    "time_in_sub_episodes": result_bundle["time_in_sub_episodes"],
    "data_min": result_bundle["data_min"],
    "data_max": result_bundle["data_max"],
}
for key in (
    "training_profile_name",
    "experiment_phase_windows",
    "phase_switch_step",
    "phase_switch_episode",
    "exploration_freeze_step",
    "exploration_freeze_settings",
    "fouling_active",
    "fouled_ha",
    "phase2_learning_enabled",
    "phase2_final_episode_evaluation_only",
    "phase2_episode_status",
    "phase1_setpoints_phys",
    "phase2_setpoints_phys",
    "phase1_disturbance_end",
    "phase2_disturbance_end",
):
    if key in result_bundle:
        legacy_payload[key] = result_bundle[key]

with open(BASELINE_SAVE_PATH, "wb") as file:
    pickle.dump(legacy_payload, file)

print(f"Plots saved to  : {out_dir}")
print(f"Baseline pickle : {BASELINE_SAVE_PATH}")

# --- Cell 15 (code) ---
