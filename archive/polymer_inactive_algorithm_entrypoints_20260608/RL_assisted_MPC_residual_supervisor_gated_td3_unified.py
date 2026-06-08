# Generated from RL_assisted_MPC_residual_unified.ipynb

# --- Cell 0 (markdown) ---
# # Polymer Residual Supervisor-Gated TD3
#
# Dedicated polymer residual entrypoint for the opt-in supervisor-gated TD3 agent.
# The existing TD3/SAC and TD7 residual entrypoints are preserved unchanged.

# --- Cell 1 (markdown) ---
# ## User Config

# --- Cell 2 (code) ---
from pathlib import Path
import os

from systems.polymer import get_polymer_notebook_defaults
from systems.polymer.data_io import canonical_baseline_path
from utils.notebook_setup import prepare_polymer_notebook_env, print_grouped_notebook_summary

NOTEBOOK_SOURCE = globals().get(
    "NOTEBOOK_SOURCE_OVERRIDE",
    "RL_assisted_MPC_residual_supervisor_gated_td3_unified.py",
)
RUN_SUMMARY_TITLE = globals().get(
    "RUN_SUMMARY_TITLE_OVERRIDE",
    "Polymer Residual Supervisor-Gated TD3 run summary",
)
NB_CONFIGURE = globals().get("NB_CONFIGURE")
NB = get_polymer_notebook_defaults("residual")
NB["state_mode"] = "standard"
NB["run_profiles"][("sg_td3", "nominal")]["result_prefix"] = "sg_td3_residual_nominal_standard"
NB["run_profiles"][("sg_td3", "nominal")]["compare_prefix"] = "nominal_compare_sg_td3_residual_standard"
NB["run_profiles"][("sg_td3", "disturb")]["result_prefix"] = "sg_td3_residual_disturb_standard"
NB["run_profiles"][("sg_td3", "disturb")]["compare_prefix"] = "disturb_compare_sg_td3_residual_standard"
if NB_CONFIGURE is not None:
    configured_nb = NB_CONFIGURE(NB)
    if configured_nb is not None:
        NB = configured_nb
NB["agent_kind"] = "sg_td3"
AGENT_KIND = "sg_td3"
RUN_MODE = NB["run_mode"]
STATE_MODE = NB["state_mode"]
RESIDUAL_AUTHORITY_ENABLED = bool(NB.get("residual_authority_enabled", STATE_MODE == "mismatch"))
USE_RHO_AUTHORITY = NB["authority_use_rho"]
APPEND_RHO_TO_STATE = bool(NB["append_rho_to_state"])
AUTHORITY_BETA_RES = NB["authority_beta_res"]
AUTHORITY_DU0_RES = NB["authority_du0_res"]
AUTHORITY_ETA_TOL = NB["authority_eta_tol"]
AUTHORITY_RHO_FLOOR = NB["authority_rho_floor"]
AUTHORITY_RHO_POWER = NB["authority_rho_power"]
RHO_MAPPING_MODE = NB["rho_mapping_mode"]
AUTHORITY_RHO_K = NB["authority_rho_k"]
RESIDUAL_ZERO_DEADBAND_ENABLED = bool(NB["residual_zero_deadband_enabled"])
RESIDUAL_ZERO_TRACKING_RAW_THRESHOLD = NB["residual_zero_tracking_raw_threshold"]
RESIDUAL_ZERO_INNOVATION_RAW_THRESHOLD = NB["residual_zero_innovation_raw_threshold"]
BEHAVIORAL_CLONING_CFG = dict(NB["behavioral_cloning"])
TD3_AUTHORITY_RAMP_CFG = dict(NB.get("td3_authority_ramp", {}))
RESIDUAL_SAFETY_CFG = dict(NB.get("residual_safety", {}))
STYLE_PROFILE = NB["style_profile"]
SAVE_PDF = NB["save_pdf"]
POLYMER_DATA_DIR_OVERRIDE = NB["data_dir_override"]
POLYMER_RESULTS_DIR_OVERRIDE = NB["results_dir_override"]
RESULT_PREFIX_OVERRIDE = NB["result_prefix_override"]
COMPARE_PREFIX_OVERRIDE = NB["compare_prefix_override"]
BASELINE_MPC_PATH_OVERRIDE = NB["baseline_mpc_path_override"]
N_TESTS_OVERRIDE = NB["n_tests_override"]
SET_POINTS_LEN_OVERRIDE = NB["set_points_len_override"]
WARM_START_OVERRIDE = NB["warm_start_override"]
TEST_CYCLE_OVERRIDE = NB["test_cycle_override"]
PLOT_START_EPISODE_OVERRIDE = NB["plot_start_episode_override"]
COMPARE_START_EPISODE_OVERRIDE = NB["compare_start_episode_override"]
REPO_ROOT, DATA_DIR, RESULT_DIR = prepare_polymer_notebook_env(
    data_dir_override=POLYMER_DATA_DIR_OVERRIDE,
    results_dir_override=POLYMER_RESULTS_DIR_OVERRIDE,
)
os.chdir(REPO_ROOT)
RUN_PROFILE = NB["run_profiles"][(AGENT_KIND, RUN_MODE)]

# --- Cell 3 (markdown) ---
# ## Imports

# --- Cell 4 (code) ---
import numpy as np
import torch

from Simulation.mpc import MpcSolverGeneral
from Simulation.system_functions import PolymerCSTR
from TD3Agent.supervisor_gated_agent import SupervisorGatedTD3Agent, SupervisorGateConfig
from systems.polymer import (
    POLYMER_OBSERVER_POLES,
    POLYMER_SYSTEM_METADATA,
    load_polymer_system_data,
)
from utils.helpers import apply_min_max
from utils.plotting import compare_mpc_rl_from_dirs, plot_residual_results
from utils.rewards import make_reward_fn_relative_QR
from utils.residual_runner import run_residual_supervisor
from utils.state_features import get_rl_state_dim

# --- Cell 5 (markdown) ---
# ## System And Data Setup

# --- Cell 6 (code) ---
SYS = NB["system_setup"]
system_params = SYS["system_params"].copy()
system_design_params = SYS["design_params"].copy()
system_steady_state_inputs = SYS["ss_inputs"].copy()
delta_t = SYS["delta_t_hours"]
cstr_ss = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t)
steady_states = {"ss_inputs": cstr_ss.ss_inputs, "y_ss": cstr_ss.y_ss}
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
y_sp_scenario = apply_min_max(
    y_sp_scenario_phys,
    data_min[inputs_number:],
    data_max[inputs_number:],
) - apply_min_max(
    steady_states["y_ss"],
    data_min[inputs_number:],
    data_max[inputs_number:],
)

# --- Cell 7 (markdown) ---
# ## Run / Reward / Agent Setup

# --- Cell 8 (code) ---
EPISODE_CFG = NB["episode_defaults"]
CTRL = NB["controller"]
TD3_CFG = NB["td3_agent"]
GATE_CFG = dict(NB.get("supervisor_gate", {}))
REWARD_CFG = NB["reward"]

n_tests = int(EPISODE_CFG["n_tests"] if N_TESTS_OVERRIDE is None else N_TESTS_OVERRIDE)
set_points_len = int(EPISODE_CFG["set_points_len"] if SET_POINTS_LEN_OVERRIDE is None else SET_POINTS_LEN_OVERRIDE)
warm_start = int(EPISODE_CFG["warm_start"] if WARM_START_OVERRIDE is None else WARM_START_OVERRIDE)
TEST_CYCLE = list(EPISODE_CFG["test_cycle"] if TEST_CYCLE_OVERRIDE is None else TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = int(
    RUN_PROFILE["plot_start_episode"] if PLOT_START_EPISODE_OVERRIDE is None else PLOT_START_EPISODE_OVERRIDE
)
COMPARE_START_EPISODE = int(
    RUN_PROFILE["compare_start_episode"] if COMPARE_START_EPISODE_OVERRIDE is None else COMPARE_START_EPISODE_OVERRIDE
)
RESULT_PREFIX = RESULT_PREFIX_OVERRIDE or RUN_PROFILE["result_prefix"]
COMPARE_PREFIX = COMPARE_PREFIX_OVERRIDE or RUN_PROFILE["compare_prefix"]
BASELINE_MPC_PATH = (
    Path(BASELINE_MPC_PATH_OVERRIDE).expanduser()
    if BASELINE_MPC_PATH_OVERRIDE
    else canonical_baseline_path(REPO_ROOT, RUN_MODE, data_override=POLYMER_DATA_DIR_OVERRIDE)
)
N_INPUTS = int(B_aug.shape[1])
N_OUTPUTS = int(C_aug.shape[0])
STATE_DIM = get_rl_state_dim(
    A_aug.shape[0],
    N_OUTPUTS,
    N_INPUTS,
    STATE_MODE,
    append_rho_to_state=(STATE_MODE == "mismatch" and APPEND_RHO_TO_STATE),
)
MISMATCH_CLIP = CTRL["mismatch_clip"]
INNOVATION_SCALE_MODE = CTRL["innovation_scale_mode"]
INNOVATION_SCALE_REF = CTRL["innovation_scale_ref"]
TRACKING_SCALE_MODE = CTRL["tracking_scale_mode"]
TRACKING_ETA_TOL = CTRL["tracking_eta_tol"]
TRACKING_SCALE_FLOOR = CTRL["tracking_scale_floor"]
TRACKING_SCALE_FLOOR_MODE = CTRL["tracking_scale_floor_mode"]
BASE_STATE_NORM_MODE = CTRL["base_state_norm_mode"]
BASE_STATE_RUNNING_NORM_CLIP = CTRL["base_state_running_norm_clip"]
BASE_STATE_RUNNING_NORM_EPS = CTRL["base_state_running_norm_eps"]
MISMATCH_FEATURE_TRANSFORM_MODE = CTRL["mismatch_feature_transform_mode"]
MISMATCH_TRANSFORM_TANH_SCALE = CTRL["mismatch_transform_tanh_scale"]
MISMATCH_TRANSFORM_POST_CLIP = CTRL["mismatch_transform_post_clip"]
OBSERVER_UPDATE_ALIGNMENT = CTRL["observer_update_alignment"]
ACTION_DIM = N_INPUTS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predict_h = CTRL["predict_h"]
cont_h = CTRL["cont_h"]
Q1_penalty = CTRL["Q1_penalty"]
Q2_penalty = CTRL["Q2_penalty"]
R1_penalty = CTRL["R1_penalty"]
R2_penalty = CTRL["R2_penalty"]
LOW_COEF = CTRL["low_coef"].copy()
HIGH_COEF = CTRL["high_coef"].copy()
TD3_N_STEP = int(TD3_CFG["n_step"])
TD3_MULTISTEP_MODE = TD3_CFG["multistep_mode"]
TD3_LAMBDA_VALUE = float(TD3_CFG["lambda_value"])
TD3_BUFFER_SIZE = int(TD3_CFG["buffer_size"])
TD3_REPLAY_FRAC_PER = float(TD3_CFG["replay_frac_per"])
TD3_REPLAY_FRAC_RECENT = float(TD3_CFG["replay_frac_recent"])
TD3_REPLAY_RECENT_WINDOW_MULT = int(TD3_CFG["replay_recent_window_mult"])
TD3_REPLAY_RECENT_WINDOW = (
    int(TD3_CFG["replay_recent_window"])
    if TD3_CFG["replay_recent_window"] is not None
    else min(TD3_BUFFER_SIZE, TD3_REPLAY_RECENT_WINDOW_MULT * set_points_len)
)
TD3_REPLAY_ALPHA = float(TD3_CFG["replay_alpha"])
TD3_REPLAY_BETA_START = float(TD3_CFG["replay_beta_start"])
TD3_REPLAY_BETA_END = float(TD3_CFG["replay_beta_end"])
TD3_REPLAY_BETA_STEPS = int(TD3_CFG["replay_beta_steps"])
N_STEP = TD3_N_STEP
MULTISTEP_MODE = TD3_MULTISTEP_MODE
LAMBDA_VALUE = TD3_LAMBDA_VALUE
USE_SHIFTED_MPC_WARM_START = CTRL["use_shifted_mpc_warm_start"]
ACTIVE_REPLAY_SETTINGS = {
    "buffer_size": TD3_BUFFER_SIZE,
    "replay_frac_per": TD3_REPLAY_FRAC_PER,
    "replay_frac_recent": TD3_REPLAY_FRAC_RECENT,
    "replay_recent_window_mult": TD3_REPLAY_RECENT_WINDOW_MULT,
    "replay_recent_window": TD3_REPLAY_RECENT_WINDOW,
    "replay_alpha": TD3_REPLAY_ALPHA,
    "replay_beta_start": TD3_REPLAY_BETA_START,
    "replay_beta_end": TD3_REPLAY_BETA_END,
    "replay_beta_steps": TD3_REPLAY_BETA_STEPS,
}
nominal_qs = CTRL["nominal_qs"]
nominal_qi = CTRL["nominal_qi"]
nominal_hA = CTRL["nominal_ha"]
qi_change = CTRL["qi_change"]
qs_change = CTRL["qs_change"]
ha_change = CTRL["ha_change"]
MPC_obj = MpcSolverGeneral(
    A_aug,
    B_aug,
    C_aug,
    Q_out=np.array([Q1_penalty, Q2_penalty], float),
    R_in=np.array([R1_penalty, R2_penalty], float),
    NP=predict_h,
    NC=cont_h,
)
reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, N_INPUTS, **REWARD_CFG)
residual_agent = SupervisorGatedTD3Agent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    actor_hidden=list(TD3_CFG["actor_hidden"]),
    critic_hidden=list(TD3_CFG["critic_hidden"]),
    gamma=TD3_CFG["gamma"],
    actor_lr=TD3_CFG["actor_lr"],
    critic_lr=TD3_CFG["critic_lr"],
    batch_size=TD3_CFG["batch_size"],
    policy_delay=TD3_CFG["policy_delay"],
    target_policy_smoothing_noise_std=TD3_CFG["target_policy_smoothing_noise_std"],
    noise_clip=TD3_CFG["noise_clip"],
    max_action=TD3_CFG["max_action"],
    tau=TD3_CFG["tau"],
    std_start=TD3_CFG["std_start"],
    std_end=TD3_CFG["std_end"],
    std_decay_rate=TD3_CFG["std_decay_rate"],
    std_decay_mode=TD3_CFG["std_decay_mode"],
    buffer_size=TD3_BUFFER_SIZE,
    replay_frac_per=TD3_REPLAY_FRAC_PER,
    replay_frac_recent=TD3_REPLAY_FRAC_RECENT,
    replay_recent_window=TD3_REPLAY_RECENT_WINDOW,
    replay_alpha=TD3_REPLAY_ALPHA,
    replay_beta_start=TD3_REPLAY_BETA_START,
    replay_beta_end=TD3_REPLAY_BETA_END,
    replay_beta_steps=TD3_REPLAY_BETA_STEPS,
    device=DEVICE,
    actor_freeze=TD3_CFG["actor_freeze"],
    exploration_mode=TD3_CFG["exploration_mode"],
    loss_type=TD3_CFG["loss_type"],
    param_noise_resample_interval=TD3_CFG["param_noise_resample_interval"],
    n_step=TD3_N_STEP,
    multistep_mode=TD3_MULTISTEP_MODE,
    lambda_value=TD3_LAMBDA_VALUE,
    supervisor_gate_config=SupervisorGateConfig(**GATE_CFG),
)

REPLAY_SETTINGS = ACTIVE_REPLAY_SETTINGS

# --- Cell 9 (markdown) ---
# ## Resolved Summary

# --- Cell 10 (code) ---
print_grouped_notebook_summary(
    RUN_SUMMARY_TITLE,
    {
        "Paths": {
            "Repo root": REPO_ROOT,
            "Data dir": DATA_DIR,
            "Results dir": RESULT_DIR,
            "Baseline MPC": BASELINE_MPC_PATH,
        },
        "Run setup": {
            "Agent kind": AGENT_KIND,
            "Run mode": RUN_MODE,
            "State mode": STATE_MODE,
            "n_tests": n_tests,
            "set_points_len": set_points_len,
            "warm_start": warm_start,
            "test_cycle": TEST_CYCLE,
            "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
            "use_rho_authority": USE_RHO_AUTHORITY,
        },
        "System / controller": {
            "delta_t_hours": delta_t,
            "predict_h": predict_h,
            "cont_h": cont_h,
            "Q penalties": [Q1_penalty, Q2_penalty],
            "R penalties": [R1_penalty, R2_penalty],
            "observer_poles": POLYMER_OBSERVER_POLES.tolist(),
        },
        "Reward": reward_params,
        "Agent": {
            "supervisor": "zero residual correction gated by TD3 critics",
            "buffer_size": TD3_CFG["buffer_size"],
            "n_step": N_STEP,
            "multistep_mode": MULTISTEP_MODE,
            "lambda_value": LAMBDA_VALUE,
            "exploration_mode": TD3_CFG["exploration_mode"],
            "loss_type": TD3_CFG["loss_type"],
        },
        "Supervisor gate": GATE_CFG,
        "Replay": REPLAY_SETTINGS,
        "Mismatch": {
            "clip": MISMATCH_CLIP,
            "innovation_scale_mode": INNOVATION_SCALE_MODE,
            "tracking_scale_mode": TRACKING_SCALE_MODE,
            "tracking_eta_tol": TRACKING_ETA_TOL,
            "tracking_scale_floor_mode": TRACKING_SCALE_FLOOR_MODE,
        },
        "Residual authority": {
            "enabled": RESIDUAL_AUTHORITY_ENABLED,
            "use_rho": USE_RHO_AUTHORITY,
            "append_rho_to_state": APPEND_RHO_TO_STATE,
            "rho_floor": AUTHORITY_RHO_FLOOR,
            "rho_power": AUTHORITY_RHO_POWER,
        },
        "Behavioral cloning": dict(BEHAVIORAL_CLONING_CFG),
        "TD3 controlled authority": dict(TD3_AUTHORITY_RAMP_CFG),
        "Plotting / export": {
            "style_profile": STYLE_PROFILE,
            "save_pdf": SAVE_PDF,
            "result_prefix": RESULT_PREFIX,
            "compare_prefix": COMPARE_PREFIX,
            "plot_start_episode": PLOT_START_EPISODE,
            "compare_start_episode": COMPARE_START_EPISODE,
        },
    },
)

# --- Cell 11 (markdown) ---
# ## Run

# --- Cell 12 (code) ---
residual_cfg = {
    "agent_kind": AGENT_KIND,
    "run_mode": RUN_MODE,
    "state_mode": STATE_MODE,
    "mismatch_clip": MISMATCH_CLIP,
    "innovation_scale_mode": INNOVATION_SCALE_MODE,
    "innovation_scale_ref": INNOVATION_SCALE_REF,
    "tracking_scale_mode": TRACKING_SCALE_MODE,
    "tracking_eta_tol": TRACKING_ETA_TOL,
    "tracking_scale_floor": TRACKING_SCALE_FLOOR,
    "tracking_scale_floor_mode": TRACKING_SCALE_FLOOR_MODE,
    "base_state_norm_mode": BASE_STATE_NORM_MODE,
    "base_state_running_norm_clip": BASE_STATE_RUNNING_NORM_CLIP,
    "base_state_running_norm_eps": BASE_STATE_RUNNING_NORM_EPS,
    "mismatch_feature_transform_mode": MISMATCH_FEATURE_TRANSFORM_MODE,
    "mismatch_transform_tanh_scale": MISMATCH_TRANSFORM_TANH_SCALE,
    "mismatch_transform_post_clip": MISMATCH_TRANSFORM_POST_CLIP,
    "observer_update_alignment": OBSERVER_UPDATE_ALIGNMENT,
    "notebook_source": NOTEBOOK_SOURCE,
    "supervisor_gate": dict(GATE_CFG),
    "residual_authority_enabled": RESIDUAL_AUTHORITY_ENABLED,
    "authority_use_rho": USE_RHO_AUTHORITY,
    "use_rho_authority": USE_RHO_AUTHORITY,
    "append_rho_to_state": APPEND_RHO_TO_STATE,
    "authority_beta_res": AUTHORITY_BETA_RES,
    "authority_du0_res": AUTHORITY_DU0_RES,
    "authority_eta_tol": AUTHORITY_ETA_TOL,
    "authority_rho_floor": AUTHORITY_RHO_FLOOR,
    "authority_rho_power": AUTHORITY_RHO_POWER,
    "rho_mapping_mode": RHO_MAPPING_MODE,
    "authority_rho_k": AUTHORITY_RHO_K,
    "residual_zero_deadband_enabled": RESIDUAL_ZERO_DEADBAND_ENABLED,
    "residual_zero_tracking_raw_threshold": RESIDUAL_ZERO_TRACKING_RAW_THRESHOLD,
    "residual_zero_innovation_raw_threshold": RESIDUAL_ZERO_INNOVATION_RAW_THRESHOLD,
    "n_tests": n_tests,
    "set_points_len": set_points_len,
    "n_step": N_STEP,
    "multistep_mode": MULTISTEP_MODE,
    "lambda_value": LAMBDA_VALUE,
    "warm_start": warm_start,
    "post_warm_start_action_freeze_subepisodes": int(NB["post_warm_start_action_freeze_subepisodes"]),
    "post_warm_start_actor_freeze_subepisodes": int(NB["post_warm_start_actor_freeze_subepisodes"]),
    "behavioral_cloning": dict(BEHAVIORAL_CLONING_CFG),
    "td3_authority_ramp": dict(TD3_AUTHORITY_RAMP_CFG),
    "residual_safety": dict(RESIDUAL_SAFETY_CFG),
    "test_cycle": TEST_CYCLE,
    "predict_h": predict_h,
    "cont_h": cont_h,
    "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
    "low_coef": LOW_COEF,
    "high_coef": HIGH_COEF,
    "nominal_qi": nominal_qi,
    "nominal_qs": nominal_qs,
    "nominal_ha": nominal_hA,
    "qi_change": qi_change,
    "qs_change": qs_change,
    "ha_change": ha_change,
    "Q1_penalty": Q1_penalty,
    "Q2_penalty": Q2_penalty,
    "R1_penalty": R1_penalty,
    "R2_penalty": R2_penalty,
    "b_min": system_data["b_min"],
    "b_max": system_data["b_max"],
}

cstr = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t)
runtime_ctx = {
    "system": cstr,
    "agent": residual_agent,
    "MPC_obj": MPC_obj,
    "steady_states": steady_states,
    "min_max_dict": min_max_dict,
    "data_min": data_min,
    "data_max": data_max,
    "A_aug": A_aug,
    "B_aug": B_aug,
    "C_aug": C_aug,
    "poles": POLYMER_OBSERVER_POLES.copy(),
    "y_sp_scenario": y_sp_scenario,
    "reward_fn": reward_fn,
    "system_metadata": POLYMER_SYSTEM_METADATA,
    "reward_params": reward_params,
}

result_bundle = run_residual_supervisor(residual_cfg=residual_cfg, runtime_ctx=runtime_ctx)
result_bundle["mpc_path_or_dir"] = BASELINE_MPC_PATH

# --- Cell 13 (markdown) ---
# ## Plotting And Export

# --- Cell 14 (code) ---
out_dir_rl = plot_residual_results(
    result_bundle=result_bundle,
    plot_cfg={
        "directory": RESULT_DIR,
        "prefix_name": RESULT_PREFIX,
        "start_episode": PLOT_START_EPISODE,
        "save_pdf": SAVE_PDF,
    },
)

out_dir_cmp = compare_mpc_rl_from_dirs(
    rl_dir=out_dir_rl,
    mpc_path_or_dir=BASELINE_MPC_PATH,
    reward_fn=reward_fn,
    directory=RESULT_DIR,
    prefix_name=COMPARE_PREFIX,
    compare_mode=RUN_PROFILE["compare_mode"],
    start_episode=COMPARE_START_EPISODE,
)

print("RL result directory:", out_dir_rl)
print("Comparison directory:", out_dir_cmp)

# --- Cell 15 (code) ---
