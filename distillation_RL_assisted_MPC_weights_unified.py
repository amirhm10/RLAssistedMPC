# Generated from distillation_RL_assisted_MPC_weights_unified.ipynb

# --- Cell 0 (markdown) ---
# # Distillation Weight Multiplier Supervisor
#
# Canonical unified distillation notebook. Edit the config cell below for one-off runs, or change `systems/distillation/notebook_params.py` for repo-wide defaults.

# --- Cell 1 (markdown) ---
# ## User Config And Imports

# --- Cell 2 (code) ---
from pathlib import Path
import os

import numpy as np
import torch

from SACAgent.sac_agent import SACAgent
from Simulation.mpc import MpcSolverGeneral
from TD3Agent.agent import TD3Agent
from systems.distillation import DISTILLATION_SYSTEM_METADATA, get_distillation_notebook_defaults
from systems.distillation.data_io import canonical_baseline_path, load_distillation_system_data
from systems.distillation.plant import build_distillation_system, distillation_system_stepper
from systems.distillation.scenarios import build_distillation_disturbance_schedule
from utils.helpers import apply_min_max
from utils.notebook_setup import prepare_distillation_notebook_env, print_grouped_notebook_summary
from utils.plotting import compare_mpc_rl_from_dirs, plot_weight_multiplier_results
from utils.rewards import make_reward_fn_relative_QR
from utils.state_features import get_rl_state_dim
from utils.weights_runner import run_weight_multiplier_supervisor

NB = get_distillation_notebook_defaults("weights")
AGENT_KIND = NB["agent_kind"]
# AGENT_KIND = "sac"
RUN_MODE = NB["run_mode"]
DISTURBANCE_PROFILE = NB["disturbance_profile"]
STATE_MODE = NB["state_mode"]
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
COMPARE_PREFIX_OVERRIDE = NB["compare_prefix_override"]
BASELINE_MPC_PATH_OVERRIDE = NB["baseline_mpc_path_override"]
N_TESTS_OVERRIDE = NB["n_tests_override"]
SET_POINTS_LEN_OVERRIDE = NB["set_points_len_override"]
WARM_START_OVERRIDE = NB["warm_start_override"]
TEST_CYCLE_OVERRIDE = NB["test_cycle_override"]
PLOT_START_EPISODE_OVERRIDE = NB["plot_start_episode_override"]
COMPARE_START_EPISODE_OVERRIDE = NB["compare_start_episode_override"]
BEHAVIORAL_CLONING_CFG = dict(NB["behavioral_cloning"])
REPO_ROOT, DATA_DIR, RESULT_DIR, DISTURBANCE_PROFILE, DYN_PATH, SNAPS_PATH, ASPEN_SOURCE = prepare_distillation_notebook_env(run_mode=RUN_MODE, disturbance_profile=DISTURBANCE_PROFILE, family="weights", aspen_preset=ASPEN_PRESET, dyn_path_override=ASPEN_PATH_OVERRIDE, snaps_path_override=SNAPS_PATH_OVERRIDE, aspen_root_override=ASPEN_ROOT_OVERRIDE, data_dir_override=DISTILLATION_DATA_DIR_OVERRIDE, results_dir_override=DISTILLATION_RESULTS_DIR_OVERRIDE)
os.chdir(REPO_ROOT)

# --- Cell 3 (markdown) ---
# ## System And Data Setup

# --- Cell 4 (code) ---
# Build the plant, load the canonical data bundle, and prepare the supervisory setpoint scenario.
SYS = NB["system_setup"]
RUN_PROFILE = NB["run_profiles"][(AGENT_KIND, RUN_MODE, DISTURBANCE_PROFILE)]
nominal_conditions = SYS["nominal_conditions"].copy()
ss_inputs = SYS["ss_inputs"].copy()
u_min = SYS["input_bounds"]["u_min"].copy()
u_max = SYS["input_bounds"]["u_max"].copy()
setpoint_y = SYS["setpoint_range_phys"].copy()
y_sp_scenario_phys = SYS["rl_setpoints_phys"].copy()
system = build_distillation_system(path=DYN_PATH, ss_inputs=ss_inputs, initialization_point=nominal_conditions, delta_t=SYS["delta_t_hours"], visible=DISTILLATION_VISIBLE)
steady_states = {"ss_inputs": np.asarray(system.ss_inputs, float).copy(), "y_ss": np.asarray(system.y_ss, float).copy()}
system_data = load_distillation_system_data(REPO_ROOT, steady_states, setpoint_y, u_min, u_max, data_override=DISTILLATION_DATA_DIR_OVERRIDE)
A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
min_max_dict = system_data["min_max_dict"]
inputs_number = int(B_aug.shape[1])
y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[inputs_number:], data_max[inputs_number:]) - apply_min_max(steady_states["y_ss"], data_min[inputs_number:], data_max[inputs_number:])
RESULT_PREFIX = RESULT_PREFIX_OVERRIDE or f"distillation_weights_{AGENT_KIND}_{RUN_MODE}_{DISTURBANCE_PROFILE}_{STATE_MODE}_unified"
COMPARE_PREFIX = COMPARE_PREFIX_OVERRIDE or f"distillation_compare_weights_{AGENT_KIND}_{RUN_MODE}_{DISTURBANCE_PROFILE}_{STATE_MODE}"
BASELINE_MPC_PATH = Path(BASELINE_MPC_PATH_OVERRIDE).expanduser() if BASELINE_MPC_PATH_OVERRIDE else canonical_baseline_path(REPO_ROOT, RUN_MODE, DISTURBANCE_PROFILE, data_override=DISTILLATION_DATA_DIR_OVERRIDE)
n_tests = int(RUN_PROFILE["n_tests"] if N_TESTS_OVERRIDE is None else N_TESTS_OVERRIDE)
set_points_len = int(RUN_PROFILE["set_points_len"] if SET_POINTS_LEN_OVERRIDE is None else SET_POINTS_LEN_OVERRIDE)
warm_start = int(RUN_PROFILE["warm_start"] if WARM_START_OVERRIDE is None else WARM_START_OVERRIDE)
TEST_CYCLE = list(RUN_PROFILE["test_cycle"] if TEST_CYCLE_OVERRIDE is None else TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = int(RUN_PROFILE.get("plot_start_episode", 1) if PLOT_START_EPISODE_OVERRIDE is None else PLOT_START_EPISODE_OVERRIDE)
COMPARE_START_EPISODE = int(RUN_PROFILE.get("compare_start_episode", PLOT_START_EPISODE) if COMPARE_START_EPISODE_OVERRIDE is None else COMPARE_START_EPISODE_OVERRIDE)
TOTAL_STEPS = n_tests * set_points_len * len(y_sp_scenario_phys)
DISTURBANCE_NOMINAL_FEED = float(system.feed.FmR.Value)
DISTURBANCE_SCHEDULE = build_distillation_disturbance_schedule(RUN_MODE, DISTURBANCE_PROFILE, TOTAL_STEPS, nominal_feed=DISTURBANCE_NOMINAL_FEED)

# --- Cell 5 (markdown) ---
# ## Run / Reward / Agent Setup

# --- Cell 6 (code) ---
# Run-profile, controller, reward, and agent setup.
CTRL = NB["controller"]
TD3_CFG = NB["td3_agent"]
SAC_CFG = NB["sac_agent"]
REWARD_CFG = NB["reward"]
poles = SYS["observer_poles"].copy()
N_INPUTS = int(B_aug.shape[1])
N_OUTPUTS = int(C_aug.shape[0])
STATE_DIM = get_rl_state_dim(A_aug.shape[0], N_OUTPUTS, N_INPUTS, STATE_MODE)
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
ACTION_DIM = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predict_h = CTRL["predict_h"]
cont_h = CTRL["cont_h"]
Q1_penalty = CTRL["Q1_penalty"]
Q2_penalty = CTRL["Q2_penalty"]
R1_penalty = CTRL["R1_penalty"]
R2_penalty = CTRL["R2_penalty"]
LOW_COEF = CTRL["low_coef"].copy()
HIGH_COEF = CTRL["high_coef"].copy()
TD3_EXPLORATION_MODE = TD3_CFG["exploration_mode"]
TD3_N_STEP = int(TD3_CFG["n_step"])
TD3_MULTISTEP_MODE = TD3_CFG["multistep_mode"]
TD3_LAMBDA_VALUE = float(TD3_CFG["lambda_value"])
TD3_LOSS_TYPE = TD3_CFG["loss_type"]
TD3_PARAM_NOISE_RESAMPLE_INTERVAL = TD3_CFG["param_noise_resample_interval"]
TD3_PARAM_NOISE_STD_START = float(TD3_CFG["param_noise_std_start"])
TD3_PARAM_NOISE_STD_END = float(TD3_CFG["param_noise_std_end"])
TD3_BUFFER_SIZE = int(TD3_CFG["buffer_size"])
TD3_REPLAY_FRAC_PER = float(TD3_CFG["replay_frac_per"])
TD3_REPLAY_FRAC_RECENT = float(TD3_CFG["replay_frac_recent"])
TD3_REPLAY_RECENT_WINDOW_MULT = int(TD3_CFG["replay_recent_window_mult"])
TD3_REPLAY_RECENT_WINDOW = int(TD3_CFG["replay_recent_window"]) if TD3_CFG["replay_recent_window"] is not None else min(TD3_BUFFER_SIZE, TD3_REPLAY_RECENT_WINDOW_MULT * set_points_len)
TD3_REPLAY_ALPHA = float(TD3_CFG["replay_alpha"])
TD3_REPLAY_BETA_START = float(TD3_CFG["replay_beta_start"])
TD3_REPLAY_BETA_END = float(TD3_CFG["replay_beta_end"])
TD3_REPLAY_BETA_STEPS = int(TD3_CFG["replay_beta_steps"])
SAC_BUFFER_SIZE = int(SAC_CFG["buffer_size"])
SAC_REPLAY_FRAC_PER = float(SAC_CFG["replay_frac_per"])
SAC_REPLAY_FRAC_RECENT = float(SAC_CFG["replay_frac_recent"])
SAC_REPLAY_RECENT_WINDOW_MULT = int(SAC_CFG["replay_recent_window_mult"])
SAC_REPLAY_RECENT_WINDOW = int(SAC_CFG["replay_recent_window"]) if SAC_CFG["replay_recent_window"] is not None else min(SAC_BUFFER_SIZE, SAC_REPLAY_RECENT_WINDOW_MULT * set_points_len)
SAC_REPLAY_ALPHA = float(SAC_CFG["replay_alpha"])
SAC_REPLAY_BETA_START = float(SAC_CFG["replay_beta_start"])
SAC_REPLAY_BETA_END = float(SAC_CFG["replay_beta_end"])
SAC_REPLAY_BETA_STEPS = int(SAC_CFG["replay_beta_steps"])
SAC_LOSS_TYPE = SAC_CFG["loss_type"]
SAC_N_STEP = int(SAC_CFG["n_step"])
SAC_MULTISTEP_MODE = SAC_CFG["multistep_mode"]
SAC_LAMBDA_VALUE = float(SAC_CFG["lambda_value"])
N_STEP = TD3_N_STEP if AGENT_KIND == "td3" else SAC_N_STEP
MULTISTEP_MODE = TD3_MULTISTEP_MODE if AGENT_KIND == "td3" else SAC_MULTISTEP_MODE
LAMBDA_VALUE = TD3_LAMBDA_VALUE if AGENT_KIND == "td3" else SAC_LAMBDA_VALUE
USE_SHIFTED_MPC_WARM_START = CTRL["use_shifted_mpc_warm_start"]
ACTIVE_BUFFER_SIZE = TD3_BUFFER_SIZE if AGENT_KIND == "td3" else SAC_BUFFER_SIZE
ACTIVE_REPLAY_SETTINGS = {
    "buffer_size": ACTIVE_BUFFER_SIZE,
    "replay_frac_per": TD3_REPLAY_FRAC_PER if AGENT_KIND == "td3" else SAC_REPLAY_FRAC_PER,
    "replay_frac_recent": TD3_REPLAY_FRAC_RECENT if AGENT_KIND == "td3" else SAC_REPLAY_FRAC_RECENT,
    "replay_recent_window_mult": TD3_REPLAY_RECENT_WINDOW_MULT if AGENT_KIND == "td3" else SAC_REPLAY_RECENT_WINDOW_MULT,
    "replay_recent_window": TD3_REPLAY_RECENT_WINDOW if AGENT_KIND == "td3" else SAC_REPLAY_RECENT_WINDOW,
    "replay_alpha": TD3_REPLAY_ALPHA if AGENT_KIND == "td3" else SAC_REPLAY_ALPHA,
    "replay_beta_start": TD3_REPLAY_BETA_START if AGENT_KIND == "td3" else SAC_REPLAY_BETA_START,
    "replay_beta_end": TD3_REPLAY_BETA_END if AGENT_KIND == "td3" else SAC_REPLAY_BETA_END,
    "replay_beta_steps": TD3_REPLAY_BETA_STEPS if AGENT_KIND == "td3" else SAC_REPLAY_BETA_STEPS,
}
# Reward setup.
reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, N_INPUTS, **REWARD_CFG)
nominal_qi = CTRL["nominal_qi"]
nominal_qs = CTRL["nominal_qs"]
nominal_hA = CTRL["nominal_ha"]
qi_change = CTRL["qi_change"]
qs_change = CTRL["qs_change"]
ha_change = CTRL["ha_change"]
MPC_obj = MpcSolverGeneral(A_aug, B_aug, C_aug, Q_out=np.array([Q1_penalty, Q2_penalty], float), R_in=np.array([R1_penalty, R2_penalty], float), NP=predict_h, NC=cont_h)
# Agent setup.
if AGENT_KIND == "td3":
    weight_agent = TD3Agent(state_dim=STATE_DIM, action_dim=ACTION_DIM, actor_hidden=list(TD3_CFG["actor_hidden"]), critic_hidden=list(TD3_CFG["critic_hidden"]), gamma=TD3_CFG["gamma"], actor_lr=TD3_CFG["actor_lr"], critic_lr=TD3_CFG["critic_lr"], batch_size=TD3_CFG["batch_size"], policy_delay=TD3_CFG["policy_delay"], target_policy_smoothing_noise_std=TD3_CFG["target_policy_smoothing_noise_std"], noise_clip=TD3_CFG["noise_clip"], max_action=TD3_CFG["max_action"], tau=TD3_CFG["tau"], std_start=TD3_CFG["std_start"], std_end=TD3_CFG["std_end"], std_decay_rate=TD3_CFG["std_decay_rate"], std_decay_mode=TD3_CFG["std_decay_mode"], buffer_size=TD3_BUFFER_SIZE, replay_frac_per=TD3_REPLAY_FRAC_PER, replay_frac_recent=TD3_REPLAY_FRAC_RECENT, replay_recent_window=TD3_REPLAY_RECENT_WINDOW, replay_alpha=TD3_REPLAY_ALPHA, replay_beta_start=TD3_REPLAY_BETA_START, replay_beta_end=TD3_REPLAY_BETA_END, replay_beta_steps=TD3_REPLAY_BETA_STEPS, device=DEVICE, actor_freeze=TD3_CFG["actor_freeze"], exploration_mode=TD3_EXPLORATION_MODE, loss_type=TD3_LOSS_TYPE, param_noise_std_start=TD3_PARAM_NOISE_STD_START, param_noise_std_end=TD3_PARAM_NOISE_STD_END, param_noise_resample_interval=TD3_PARAM_NOISE_RESAMPLE_INTERVAL, n_step=TD3_N_STEP, multistep_mode=TD3_MULTISTEP_MODE, lambda_value=TD3_LAMBDA_VALUE)
elif AGENT_KIND == "sac":
    target_entropy = -ACTION_DIM if SAC_CFG["target_entropy"] == "auto_negative_action_dim" else SAC_CFG["target_entropy"]
    weight_agent = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, actor_hidden=list(SAC_CFG["actor_hidden"]), critic_hidden=list(SAC_CFG["critic_hidden"]), gamma=SAC_CFG["gamma"], actor_lr=SAC_CFG["actor_lr"], critic_lr=SAC_CFG["critic_lr"], alpha_lr=SAC_CFG["alpha_lr"], batch_size=SAC_CFG["batch_size"], grad_clip_norm=SAC_CFG["grad_clip_norm"], init_alpha=SAC_CFG["init_alpha"], learn_alpha=SAC_CFG["learn_alpha"], target_entropy=target_entropy, target_update=SAC_CFG["target_update"], tau=SAC_CFG["tau"], hard_update_interval=SAC_CFG["hard_update_interval"], activation=SAC_CFG["activation"], use_layernorm=SAC_CFG["use_layernorm"], dropout=SAC_CFG["dropout"], max_action=SAC_CFG["max_action"], buffer_size=SAC_BUFFER_SIZE, replay_frac_per=SAC_REPLAY_FRAC_PER, replay_frac_recent=SAC_REPLAY_FRAC_RECENT, replay_recent_window=SAC_REPLAY_RECENT_WINDOW, replay_alpha=SAC_REPLAY_ALPHA, replay_beta_start=SAC_REPLAY_BETA_START, replay_beta_end=SAC_REPLAY_BETA_END, replay_beta_steps=SAC_REPLAY_BETA_STEPS, device=DEVICE, use_adamw=SAC_CFG["use_adamw"], actor_freeze=SAC_CFG["actor_freeze"], loss_type=SAC_LOSS_TYPE, n_step=SAC_N_STEP, multistep_mode=SAC_MULTISTEP_MODE, lambda_value=SAC_LAMBDA_VALUE)
else:
    raise ValueError("AGENT_KIND must be 'td3' or 'sac'.")

REPLAY_SETTINGS = ACTIVE_REPLAY_SETTINGS

# --- Cell 7 (markdown) ---
# ## Resolved Summary

# --- Cell 8 (code) ---
print_grouped_notebook_summary(
    "Distillation Weight Multiplier Supervisor run summary",
    {
        "Paths": {"Repo root": REPO_ROOT, "Data dir": DATA_DIR, "Results dir": RESULT_DIR, "Aspen source": ASPEN_SOURCE, "Dyn path": DYN_PATH, "Snaps path": SNAPS_PATH, "Baseline MPC": BASELINE_MPC_PATH},
        "Run setup": {"Agent kind": AGENT_KIND, "Run mode": RUN_MODE, "Disturbance profile": DISTURBANCE_PROFILE, "State mode": STATE_MODE, "n_tests": n_tests, "set_points_len": set_points_len, "warm_start": warm_start, "test_cycle": TEST_CYCLE, "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START},
        "System / controller": {"delta_t_hours": SYS["delta_t_hours"], "predict_h": predict_h, "cont_h": cont_h, "Q penalties": [Q1_penalty, Q2_penalty], "R penalties": [R1_penalty, R2_penalty], "observer_poles": poles.tolist()},
        "Reward": reward_params,
        "Agent": {"supervisor": "weight multiplier", "buffer_size": (TD3_CFG if AGENT_KIND == "td3" else SAC_CFG)["buffer_size"], "n_step": N_STEP, "multistep_mode": MULTISTEP_MODE, "lambda_value": LAMBDA_VALUE, "exploration_mode": TD3_EXPLORATION_MODE if AGENT_KIND == "td3" else "policy_stochastic", "loss_type": TD3_LOSS_TYPE if AGENT_KIND == "td3" else SAC_LOSS_TYPE},
        "Replay": REPLAY_SETTINGS,
        "Behavioral cloning": dict(BEHAVIORAL_CLONING_CFG),
        "Mismatch": {"clip": MISMATCH_CLIP, "innovation_scale_mode": INNOVATION_SCALE_MODE, "tracking_scale_mode": TRACKING_SCALE_MODE, "tracking_eta_tol": TRACKING_ETA_TOL, "tracking_scale_floor_mode": TRACKING_SCALE_FLOOR_MODE},
        "Plotting / export": {"style_profile": STYLE_PROFILE, "save_pdf": SAVE_PDF, "result_prefix": RESULT_PREFIX, "compare_prefix": COMPARE_PREFIX, "plot_start_episode": PLOT_START_EPISODE, "compare_start_episode": COMPARE_START_EPISODE},
    },
)

# --- Cell 9 (markdown) ---
# ## Run

# --- Cell 10 (code) ---
# Assemble the shared runner configuration and execute the rollout.
weight_cfg = {
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
    "notebook_source": "distillation_RL_assisted_MPC_weights_unified.ipynb",
    "n_tests": n_tests,
    "set_points_len": set_points_len,
    "n_step": N_STEP,
    "multistep_mode": MULTISTEP_MODE,
    "lambda_value": LAMBDA_VALUE,
    "warm_start": warm_start,
    "post_warm_start_action_freeze_subepisodes": int(NB["post_warm_start_action_freeze_subepisodes"]),
    "post_warm_start_actor_freeze_subepisodes": int(NB["post_warm_start_actor_freeze_subepisodes"]),
    "behavioral_cloning": dict(BEHAVIORAL_CLONING_CFG),
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

runtime_ctx = {
    "system": system,
    "agent": weight_agent,
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
    "system_metadata": DISTILLATION_SYSTEM_METADATA,
    "reward_params": reward_params,
    "system_stepper": distillation_system_stepper,
    "disturbance_labels": DISTILLATION_SYSTEM_METADATA.get("disturbance_labels"),
    "disturbance_schedule": DISTURBANCE_SCHEDULE,
}

result_bundle = run_weight_multiplier_supervisor(weight_cfg=weight_cfg, runtime_ctx=runtime_ctx)
result_bundle["mpc_path_or_dir"] = BASELINE_MPC_PATH

# --- Cell 11 (markdown) ---
# ## Plotting And Export

# --- Cell 12 (code) ---
# Generate saved figures and any requested MPC comparison plots.
out_dir_rl = plot_weight_multiplier_results(
    result_bundle=result_bundle,
    plot_cfg={
        "directory": RESULT_DIR,
        "prefix_name": RESULT_PREFIX,
        "start_episode": PLOT_START_EPISODE,
        "save_pdf": SAVE_PDF,
        "style_profile": STYLE_PROFILE,
    },
)

out_dir_cmp = compare_mpc_rl_from_dirs(
    rl_dir=out_dir_rl,
    mpc_path_or_dir=BASELINE_MPC_PATH,
    reward_fn=reward_fn,
    directory=RESULT_DIR,
    prefix_name=COMPARE_PREFIX,
    compare_mode=RUN_MODE,
    start_episode=COMPARE_START_EPISODE,
    save_pdf=SAVE_PDF,
    style_profile=STYLE_PROFILE,
)

print(out_dir_rl)
print(out_dir_cmp)

try:
    system.close(SNAPS_PATH)
except Exception:
    pass

# --- Cell 13 (code) ---
