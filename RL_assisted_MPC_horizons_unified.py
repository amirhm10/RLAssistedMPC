# Generated from RL_assisted_MPC_horizons_unified.ipynb

# --- Cell 0 (markdown) ---
# # Polymer Horizon Supervisor
#
# Unified polymer horizon notebook. Edit the config cell below for one-off runs, or change `systems/polymer/notebook_params.py` for repo-wide defaults.

# --- Cell 1 (markdown) ---
# ## User Config

# --- Cell 2 (code) ---
from pathlib import Path
import os

from systems.polymer import (
    canonical_polymer_training_profile,
    default_polymer_profile_episode_count,
    get_polymer_notebook_defaults,
)
from systems.polymer.data_io import canonical_baseline_path
from utils.notebook_setup import prepare_polymer_notebook_env, print_grouped_notebook_summary

NOTEBOOK_SOURCE = globals().get("NOTEBOOK_SOURCE_OVERRIDE", "RL_assisted_MPC_horizons_unified.py")
RUN_SUMMARY_TITLE = globals().get(
    "RUN_SUMMARY_TITLE_OVERRIDE",
    "Polymer Horizon Supervisor run summary",
)
NB_CONFIGURE = globals().get("NB_CONFIGURE")
NB = get_polymer_notebook_defaults("horizon_standard")
if NB_CONFIGURE is not None:
    configured_nb = NB_CONFIGURE(NB)
    if not isinstance(configured_nb, dict):
        raise TypeError("NB_CONFIGURE must return a notebook-default dictionary.")
    NB = configured_nb

# Main notebook controls.
# Edit these values for a one-off run, or edit systems/polymer/notebook_params.py
# if you want the repo-wide defaults to change for every polymer horizon notebook.
AGENT_KIND = str(NB.get("agent_kind", "dqn")).strip().lower()
if AGENT_KIND not in {"dqn", "sg_dqn"}:
    raise ValueError("Polymer horizon runner supports only AGENT_KIND 'dqn' or 'sg_dqn'.")
RUN_MODE = NB["run_mode"]  # "nominal" | "disturb"
STATE_MODE = NB["state_mode"]  # "standard" | "mismatch"
STYLE_PROFILE = NB["style_profile"]  # "hybrid" | "paper" | "debug"
SAVE_PDF = NB["save_pdf"]

# Optional directory overrides. Leave as None to use Polymer/Data and Polymer/Results.
POLYMER_DATA_DIR_OVERRIDE = NB["data_dir_override"]
POLYMER_RESULTS_DIR_OVERRIDE = NB["results_dir_override"]

# Optional naming/path overrides. Leave as None to keep the shared defaults.
RESULT_PREFIX_OVERRIDE = NB["result_prefix_override"]
COMPARE_PREFIX_OVERRIDE = NB["compare_prefix_override"]
BASELINE_MPC_PATH_OVERRIDE = NB["baseline_mpc_path_override"]

# Optional run-size overrides. Leave as None to use the shared notebook defaults.
N_TESTS_OVERRIDE = NB["n_tests_override"]
SET_POINTS_LEN_OVERRIDE = NB["set_points_len_override"]
WARM_START_OVERRIDE = NB["warm_start_override"]
TEST_CYCLE_OVERRIDE = NB["test_cycle_override"]
PLOT_START_EPISODE_OVERRIDE = NB["plot_start_episode_override"]
COMPARE_START_EPISODE_OVERRIDE = NB["compare_start_episode_override"]
TRAINING_PROFILE_OVERRIDE = NB.get("training_profile_override")

REPO_ROOT, DATA_DIR, RESULT_DIR = prepare_polymer_notebook_env(
    data_dir_override=POLYMER_DATA_DIR_OVERRIDE,
    results_dir_override=POLYMER_RESULTS_DIR_OVERRIDE,
)
os.chdir(REPO_ROOT)

RUN_PROFILE = NB["run_profiles"].get((AGENT_KIND, RUN_MODE), NB["run_profiles"].get(RUN_MODE))
if RUN_PROFILE is None:
    raise KeyError(f"No run profile configured for {(AGENT_KIND, RUN_MODE)!r}.")

# A full grouped summary is printed later once the runtime configuration is fully resolved.

# --- Cell 3 (markdown) ---
# ## Imports

# --- Cell 4 (code) ---
import numpy as np
import torch

from DQN.dqn_agent import DQNAgent
from DQN.supervisor_gated_dqn_agent import SupervisorGatedDQNAgent

_DEFAULT_HORIZON_AGENT_CLASS = SupervisorGatedDQNAgent if AGENT_KIND == "sg_dqn" else DQNAgent
HORIZON_AGENT_CLASS = globals().get("HORIZON_AGENT_CLASS_OVERRIDE", _DEFAULT_HORIZON_AGENT_CLASS)
from Simulation.system_functions import PolymerCSTR
from systems.polymer import (
    HORIZON_CONTROL_GRID,
    HORIZON_PREDICT_GRID,
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
from utils.helpers import apply_min_max, build_horizon_recipes
from utils.plotting import (
    compare_mpc_rl_from_dirs,
    plot_horizon_results,
)
from utils.rewards import make_reward_fn_relative_QR
from utils.state_features import get_rl_state_dim

from utils.horizon_runner import run_dqn_mpc_horizon_supervisor
from utils.observer import compute_observer_gain

# --- Cell 5 (markdown) ---
# ## System And Data Setup

# --- Cell 6 (code) ---
# Build the polymer plant and load the canonical polymer data bundle.
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
# ## Run / Reward / Agent Setup

# --- Cell 8 (code) ---
# Controller, reward, and agent defaults.
CTRL = NB["controller"]
AGENT_CFG = NB["agent"]
REWARD_CFG = NB["reward"]
EPISODE_CFG = NB["episode_defaults"]
TRAINING_PROFILE_NAME = canonical_polymer_training_profile(
    TRAINING_PROFILE_OVERRIDE or EPISODE_CFG.get("profile_name", "legacy_gradual_200")
)

profile_default_n_tests = (
    default_polymer_profile_episode_count(TRAINING_PROFILE_NAME)
    if TRAINING_PROFILE_OVERRIDE is not None
    else EPISODE_CFG["n_tests"]
)
n_tests = int(profile_default_n_tests if N_TESTS_OVERRIDE is None else N_TESTS_OVERRIDE)
set_points_len = int(EPISODE_CFG["set_points_len"] if SET_POINTS_LEN_OVERRIDE is None else SET_POINTS_LEN_OVERRIDE)
warm_start = int(EPISODE_CFG["warm_start"] if WARM_START_OVERRIDE is None else WARM_START_OVERRIDE)
TEST_CYCLE = list(EPISODE_CFG["test_cycle"] if TEST_CYCLE_OVERRIDE is None else TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = int(RUN_PROFILE["plot_start_episode"] if PLOT_START_EPISODE_OVERRIDE is None else PLOT_START_EPISODE_OVERRIDE)
COMPARE_START_EPISODE = int(RUN_PROFILE["compare_start_episode"] if COMPARE_START_EPISODE_OVERRIDE is None else COMPARE_START_EPISODE_OVERRIDE)
RESULT_PREFIX = RESULT_PREFIX_OVERRIDE or RUN_PROFILE["result_prefix"]
COMPARE_PREFIX = COMPARE_PREFIX_OVERRIDE or RUN_PROFILE["compare_prefix"]
BASELINE_MPC_PATH = Path(BASELINE_MPC_PATH_OVERRIDE).expanduser() if BASELINE_MPC_PATH_OVERRIDE else canonical_baseline_path(
    REPO_ROOT,
    RUN_MODE,
    data_override=POLYMER_DATA_DIR_OVERRIDE,
    training_profile_name=TRAINING_PROFILE_NAME,
)

poles = SYS["observer_poles"].copy()
PREDICT_GRID = list(CTRL["predict_grid"])
CONTROL_GRID = list(CTRL["control_grid"])
HORIZON_RECIPES = build_horizon_recipes(PREDICT_GRID, CONTROL_GRID)
STATE_DIM = get_rl_state_dim(A_aug.shape[0], C_aug.shape[0], B_aug.shape[1], STATE_MODE)
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_LAYERS = list(AGENT_CFG["hidden_layers"])
BUFFER_SIZE = int(AGENT_CFG["buffer_size"])
REPLAY_FRAC_PER = float(AGENT_CFG["replay_frac_per"])
REPLAY_FRAC_RECENT = float(AGENT_CFG["replay_frac_recent"])
REPLAY_RECENT_WINDOW_MULT = int(AGENT_CFG["replay_recent_window_mult"])
REPLAY_RECENT_WINDOW = int(AGENT_CFG["replay_recent_window"]) if AGENT_CFG["replay_recent_window"] is not None else min(BUFFER_SIZE, REPLAY_RECENT_WINDOW_MULT * set_points_len)
REPLAY_ALPHA = float(AGENT_CFG["replay_alpha"])
REPLAY_BETA_START = float(AGENT_CFG["replay_beta_start"])
REPLAY_BETA_END = float(AGENT_CFG["replay_beta_end"])
REPLAY_BETA_STEPS = int(AGENT_CFG["replay_beta_steps"])
N_STEP = int(AGENT_CFG["n_step"])
MULTISTEP_MODE = AGENT_CFG["multistep_mode"]
LAMBDA_VALUE = float(AGENT_CFG["lambda_value"])
EPS_DECAY_STEPS = int(AGENT_CFG.get("eps_decay_steps", 100_000))
DECISION_INTERVAL = int(CTRL["decision_interval"])
EXPLORATION_MODE = AGENT_CFG["exploration_mode"]
LOSS_TYPE = AGENT_CFG["loss_type"]
USE_SHIFTED_MPC_WARM_START = CTRL["use_shifted_mpc_warm_start"]
POST_WARM_START_ACTION_FREEZE_SUBEPISODES = int(max(0, NB.get("post_warm_start_action_freeze_subepisodes", 0)))
HORIZON_SAFETY_CFG = dict(NB.get("horizon_safety", {}) or {})
REPLAY_SETTINGS = {
    "buffer_size": BUFFER_SIZE,
    "replay_frac_per": REPLAY_FRAC_PER,
    "replay_frac_recent": REPLAY_FRAC_RECENT,
    "replay_recent_window_mult": REPLAY_RECENT_WINDOW_MULT,
    "replay_recent_window": REPLAY_RECENT_WINDOW,
    "replay_alpha": REPLAY_ALPHA,
    "replay_beta_start": REPLAY_BETA_START,
    "replay_beta_end": REPLAY_BETA_END,
    "replay_beta_steps": REPLAY_BETA_STEPS,
}

predict_h = CTRL["predict_h"]
cont_h = CTRL["cont_h"]
Q1_penalty = CTRL["Q1_penalty"]
Q2_penalty = CTRL["Q2_penalty"]
R1_penalty = CTRL["R1_penalty"]
R2_penalty = CTRL["R2_penalty"]
nominal_qi = CTRL["nominal_qi"]
nominal_qs = CTRL["nominal_qs"]
nominal_hA = CTRL["nominal_ha"]
qi_change = CTRL["qi_change"]
qs_change = CTRL["qs_change"]
ha_change = CTRL["ha_change"]

# Reward setup.
reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, n_inputs=2, **REWARD_CFG)

# Agent setup.
agent_extra_kwargs = {}
if AGENT_KIND == "sg_dqn" and AGENT_CFG.get("supervisor_gate") is not None:
    agent_extra_kwargs["supervisor_gate_config"] = AGENT_CFG["supervisor_gate"]
dqn_agent = HORIZON_AGENT_CLASS(
    state_dim=STATE_DIM,
    action_dim=len(HORIZON_RECIPES),
    hidden_dim=HIDDEN_LAYERS,
    gamma=AGENT_CFG["gamma"],
    n_step=N_STEP,
    multistep_mode=MULTISTEP_MODE,
    lambda_value=LAMBDA_VALUE,
    lr=AGENT_CFG["lr"],
    batch_size=AGENT_CFG["batch_size"],
    buffer_size=BUFFER_SIZE,
    replay_frac_per=REPLAY_FRAC_PER,
    replay_frac_recent=REPLAY_FRAC_RECENT,
    replay_recent_window=REPLAY_RECENT_WINDOW,
    replay_alpha=REPLAY_ALPHA,
    replay_beta_start=REPLAY_BETA_START,
    replay_beta_end=REPLAY_BETA_END,
    replay_beta_steps=REPLAY_BETA_STEPS,
    grad_clip_norm=AGENT_CFG["grad_clip_norm"],
    double_dqn=AGENT_CFG["double_dqn"],
    target_update=AGENT_CFG["target_update"],
    tau=AGENT_CFG["tau"],
    hard_update_interval=AGENT_CFG["hard_update_interval"],
    activation=AGENT_CFG["activation"],
    use_layer_norm=AGENT_CFG["use_layer_norm"],
    dropout=AGENT_CFG["dropout"],
    device=DEVICE,
    exploration_mode=EXPLORATION_MODE,
    loss_type=LOSS_TYPE,
    eps_start=AGENT_CFG["eps_start"],
    eps_end=AGENT_CFG["eps_end"],
    eps_decay_rate=AGENT_CFG["eps_decay_rate"],
    eps_decay_steps=EPS_DECAY_STEPS,
    eps_decay_mode=AGENT_CFG["eps_decay_mode"],
    target_combine=AGENT_CFG["target_combine"],
    **agent_extra_kwargs,
)

print_grouped_notebook_summary(
    "Resolved horizon parameters",
    {
        "n_tests": n_tests,
        "training_profile_name": TRAINING_PROFILE_NAME,
        "set_points_len": set_points_len,
        "warm_start": warm_start,
        "plot_start_episode": PLOT_START_EPISODE,
        "compare_start_episode": COMPARE_START_EPISODE,
        "decision_interval": DECISION_INTERVAL,
        "q_warm_release_subepisodes": POST_WARM_START_ACTION_FREEZE_SUBEPISODES,
        "buffer_size": BUFFER_SIZE,
        "n_step": N_STEP,
        "multistep_mode": MULTISTEP_MODE,
        "lambda_value": LAMBDA_VALUE,
        "exploration_mode": EXPLORATION_MODE,
        "loss_type": LOSS_TYPE,
    },
)

# --- Cell 9 (markdown) ---
# ## Resolved Summary

# --- Cell 10 (code) ---
print_grouped_notebook_summary(
    RUN_SUMMARY_TITLE,
    {
        "Paths": {"Repo root": REPO_ROOT, "Data dir": DATA_DIR, "Results dir": RESULT_DIR, "Baseline MPC": BASELINE_MPC_PATH},
        "Run setup": {"Run mode": RUN_MODE, "State mode": STATE_MODE, "n_tests": n_tests, "set_points_len": set_points_len, "warm_start": warm_start, "q_warm_release_subepisodes": POST_WARM_START_ACTION_FREEZE_SUBEPISODES, "test_cycle": TEST_CYCLE, "decision_interval": DECISION_INTERVAL, "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START},
        "System / controller": {"delta_t_hours": delta_t, "predict_h": predict_h, "cont_h": cont_h, "predict_grid": PREDICT_GRID, "control_grid": CONTROL_GRID, "observer_poles": poles.tolist()},
        "Reward": reward_params,
        "Agent": {"algorithm": "sg_dqn" if AGENT_KIND == "sg_dqn" else "ddqn", "agent_kind": AGENT_KIND, "hidden_layers": AGENT_CFG["hidden_layers"], "buffer_size": BUFFER_SIZE, "n_step": N_STEP, "multistep_mode": MULTISTEP_MODE, "lambda_value": LAMBDA_VALUE, "exploration_mode": EXPLORATION_MODE, "loss_type": LOSS_TYPE, "supervisor_gate": AGENT_CFG.get("supervisor_gate")},
        "Replay": REPLAY_SETTINGS,
        "Safety": {"horizon_safety_enabled": bool(HORIZON_SAFETY_CFG.get("enabled", False)), "release_filter_enabled": bool(HORIZON_SAFETY_CFG.get("release_filter", {}).get("enabled", False)), "reward_probation_enabled": bool(HORIZON_SAFETY_CFG.get("reward_probation", {}).get("enabled", False)), "shadow_default_mpc_enabled": bool(HORIZON_SAFETY_CFG.get("shadow_default_mpc", {}).get("enabled", False))},
        "Mismatch": {"clip": MISMATCH_CLIP, "innovation_scale_mode": INNOVATION_SCALE_MODE, "tracking_scale_mode": TRACKING_SCALE_MODE, "tracking_eta_tol": TRACKING_ETA_TOL, "tracking_scale_floor_mode": TRACKING_SCALE_FLOOR_MODE},
        "Plotting / export": {"style_profile": STYLE_PROFILE, "save_pdf": SAVE_PDF, "result_prefix": RESULT_PREFIX, "compare_prefix": COMPARE_PREFIX, "plot_start_episode": PLOT_START_EPISODE, "compare_start_episode": COMPARE_START_EPISODE},
    },
)

# --- Cell 11 (markdown) ---
# ## Run

# --- Cell 12 (code) ---
# Assemble the shared runner configuration and execute the rollout.
horizon_cfg = {
    "mode": RUN_MODE,
    "training_profile_name": TRAINING_PROFILE_NAME,
    "state_mode": STATE_MODE,
    "algorithm": "sg_dqn" if AGENT_KIND == "sg_dqn" else "ddqn",
    "agent_kind": AGENT_KIND,
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
    "horizon_safety": HORIZON_SAFETY_CFG,
    "post_warm_start_action_freeze_subepisodes": POST_WARM_START_ACTION_FREEZE_SUBEPISODES,
    "supervisor_gate": AGENT_CFG.get("supervisor_gate"),
    "notebook_source": NOTEBOOK_SOURCE,
    "predict_h": predict_h,
    "cont_h": cont_h,
    "decision_interval": DECISION_INTERVAL,
    "warm_start": warm_start,
    "test_cycle": TEST_CYCLE,
    "n_tests": n_tests,
    "set_points_len": set_points_len,
    "n_step": N_STEP,
    "multistep_mode": MULTISTEP_MODE,
    "lambda_value": LAMBDA_VALUE,
    "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
    "nominal_qi": nominal_qi,
    "nominal_qs": nominal_qs,
    "nominal_ha": nominal_hA,
    "qi_change": qi_change,
    "qs_change": qs_change,
    "ha_change": ha_change,
    "b_min": system_data["b_min"],
    "b_max": system_data["b_max"],
    "Q1_penalty": Q1_penalty,
    "Q2_penalty": Q2_penalty,
    "R1_penalty": R1_penalty,
    "R2_penalty": R2_penalty,
}

runtime_ctx = {
    "system": PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t),
    "y_sp_scenario": y_sp_scenario,
    "steady_states": steady_states,
    "min_max_dict": min_max_dict,
    "agent": dqn_agent,
    "A_aug": A_aug,
    "B_aug": B_aug,
    "C_aug": C_aug,
    "poles": poles,
    "data_min": data_min,
    "data_max": data_max,
    "horizon_recipes": HORIZON_RECIPES,
    "reward_fn": reward_fn,
    "system_metadata": POLYMER_SYSTEM_METADATA,
    "reward_params": reward_params,
    "system_metadata": POLYMER_SYSTEM_METADATA,
}

result_bundle = run_dqn_mpc_horizon_supervisor(horizon_cfg=horizon_cfg, runtime_ctx=runtime_ctx)
result_bundle["mpc_path_or_dir"] = BASELINE_MPC_PATH
result_bundle["reward_params"] = reward_params
result_bundle["run_profile"] = RUN_PROFILE

print(f"nFE: {result_bundle['nFE']}")
print(f"Avg reward samples: {len(result_bundle['avg_rewards'])}")

# --- Cell 13 (markdown) ---
# ## Plotting And Export

# --- Cell 14 (code) ---
# Generate saved figures and any requested MPC comparison plots.
out_dir_rl = plot_horizon_results(
    result_bundle=result_bundle,
    plot_cfg={
        "directory": RESULT_DIR,
        "prefix_name": RESULT_PREFIX,
        "start_episode": PLOT_START_EPISODE,
        "recipe_counts": True,
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
    allow_missing_baseline=True,
    expected_training_profile_name=TRAINING_PROFILE_NAME,
)

print(f"RL plots saved to      : {out_dir_rl}")
print(f"Comparison plots saved : {out_dir_cmp}")

# --- Cell 15 (code) ---
