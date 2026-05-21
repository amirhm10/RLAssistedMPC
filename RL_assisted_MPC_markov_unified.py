# Generated from RL_assisted_MPC_markov_unified.ipynb

# --- Cell 0 (markdown) ---
# # RL Assisted MPC Markov Unified
#
# Single supported notebook entrypoint for the consolidated polymer Markov TD3 supervisor.

# --- Cell 1 (code) ---
from pathlib import Path
import os

from systems.polymer import get_polymer_notebook_defaults
from systems.polymer.data_io import canonical_baseline_path
from utils.notebook_setup import prepare_polymer_notebook_env, print_grouped_notebook_summary

NB = get_polymer_notebook_defaults("markov")

AGENT_KIND = NB["agent_kind"]
RUN_MODE = NB["run_mode"]
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

# Optional one-off notebook overrides.
MAX_STEPS_OVERRIDE = None
DEBUG_VALIDATE_LIFTED_OVERRIDE = None
DEBUG_RUN_SHADOW_LS_OVERRIDE = None
Z_BOUND_OVERRIDE = None
RL_STORE_EXECUTED_IN_REPLAY_OVERRIDE = None

# Optional one-off release override can be set here if needed.

REPO_ROOT, DATA_DIR, RESULT_DIR = prepare_polymer_notebook_env(
    data_dir_override=POLYMER_DATA_DIR_OVERRIDE,
    results_dir_override=POLYMER_RESULTS_DIR_OVERRIDE,
)
os.chdir(REPO_ROOT)
RUN_PROFILE = NB["run_profiles"][(AGENT_KIND, RUN_MODE)]

# --- Cell 2 (code) ---
import numpy as np

from Simulation.mpc import MpcSolverGeneral
from Simulation.system_functions import PolymerCSTR
from systems.polymer import POLYMER_SYSTEM_METADATA, load_polymer_system_data
from utils.helpers import apply_min_max
from utils.markov_runner import run_markov_correction_supervisor
from utils.plotting import compare_mpc_rl_from_dirs, plot_markov_correction_results
from utils.rewards import make_reward_fn_relative_QR

# --- Cell 3 (code) ---
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

n_inputs = int(B_aug.shape[1])
y_sp_scenario_phys = SYS["rl_setpoints_phys"].copy()
y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[n_inputs:], data_max[n_inputs:]) - apply_min_max(
    steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:]
)

# --- Cell 4 (code) ---
EPISODE_CFG = NB["episode_defaults"]
CTRL = NB["controller"]
TD3_CFG = NB["td3_agent"]
REWARD_CFG = NB["reward"]
BEHAVIORAL_CLONING = dict(NB.get("behavioral_cloning", {}))

n_tests = int(EPISODE_CFG["n_tests"] if N_TESTS_OVERRIDE is None else N_TESTS_OVERRIDE)
set_points_len = int(EPISODE_CFG["set_points_len"] if SET_POINTS_LEN_OVERRIDE is None else SET_POINTS_LEN_OVERRIDE)
warm_start = int(EPISODE_CFG["warm_start"] if WARM_START_OVERRIDE is None else WARM_START_OVERRIDE)
TEST_CYCLE = list(EPISODE_CFG["test_cycle"] if TEST_CYCLE_OVERRIDE is None else TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = int(RUN_PROFILE["plot_start_episode"] if PLOT_START_EPISODE_OVERRIDE is None else PLOT_START_EPISODE_OVERRIDE)
COMPARE_START_EPISODE = int(RUN_PROFILE["compare_start_episode"] if COMPARE_START_EPISODE_OVERRIDE is None else COMPARE_START_EPISODE_OVERRIDE)
RESULT_PREFIX_BASE = RESULT_PREFIX_OVERRIDE or RUN_PROFILE["result_prefix"]
COMPARE_PREFIX_BASE = COMPARE_PREFIX_OVERRIDE or RUN_PROFILE["compare_prefix"]
BASELINE_MPC_PATH = Path(BASELINE_MPC_PATH_OVERRIDE).expanduser() if BASELINE_MPC_PATH_OVERRIDE else canonical_baseline_path(
    REPO_ROOT,
    RUN_MODE,
    data_override=POLYMER_DATA_DIR_OVERRIDE,
)

poles = SYS["observer_poles"].copy()
predict_h = CTRL["predict_h"]
cont_h = CTRL["cont_h"]
decision_interval = int(CTRL["decision_interval"])
Q1_penalty = CTRL["Q1_penalty"]
Q2_penalty = CTRL["Q2_penalty"]
R1_penalty = CTRL["R1_penalty"]
R2_penalty = CTRL["R2_penalty"]
basis_family = CTRL["basis_family"]
z_bound = float(CTRL["z_bound"] if Z_BOUND_OVERRIDE is None else Z_BOUND_OVERRIDE)
z_safety = dict(CTRL.get("z_safety", {}))
prediction_window = int(CTRL["prediction_window"])
lambda_z = float(CTRL["lambda_z"])
s_pred_min = float(CTRL["s_pred_min"])
gain_drift_max = float(CTRL["gain_drift_max"])
nominal_cost_relative_tol = float(CTRL["nominal_cost_relative_tol"])
nominal_cost_absolute_tol = float(CTRL["nominal_cost_absolute_tol"])
nominal_solver_mode = str(CTRL.get("nominal_solver_mode", "state_space_shared"))
td3_seed = TD3_CFG.get("seed")
mismatch_clip = float(CTRL["mismatch_clip"])
base_state_norm_mode = str(CTRL["base_state_norm_mode"])
base_state_running_norm_clip = float(CTRL["base_state_running_norm_clip"])
base_state_running_norm_eps = float(CTRL["base_state_running_norm_eps"])
innovation_scale_mode = str(CTRL["innovation_scale_mode"])
innovation_scale_ref = CTRL["innovation_scale_ref"]
tracking_scale_mode = str(CTRL["tracking_scale_mode"])
tracking_eta_tol = float(CTRL["tracking_eta_tol"])
tracking_scale_floor = CTRL["tracking_scale_floor"]
tracking_scale_floor_mode = str(CTRL["tracking_scale_floor_mode"])
mismatch_feature_transform_mode = str(CTRL["mismatch_feature_transform_mode"])
mismatch_transform_tanh_scale = float(CTRL["mismatch_transform_tanh_scale"])
mismatch_transform_post_clip = CTRL["mismatch_transform_post_clip"]
run_adaptive_ls = bool(CTRL["run_adaptive_ls"])
run_live_corrected_mpc = bool(CTRL["run_live_corrected_mpc"])
run_rl_proposal = bool(CTRL["run_rl_proposal"])
rl_fallback_to_ls = bool(CTRL["rl_fallback_to_ls"])
force_td3_execute = bool(CTRL["force_td3_execute"])
rl_store_executed_action_in_replay = bool(
    CTRL["rl_store_executed_action_in_replay"]
    if RL_STORE_EXECUTED_IN_REPLAY_OVERRIDE is None
    else RL_STORE_EXECUTED_IN_REPLAY_OVERRIDE
)
replay_storage_mode = "executed" if rl_store_executed_action_in_replay else "requested"
REPLAY_PREFIX_SUFFIX = None if RL_STORE_EXECUTED_IN_REPLAY_OVERRIDE is None else ("replay_exec" if rl_store_executed_action_in_replay else "replay_req")
Z_BOUND_PREFIX_SUFFIX = None if Z_BOUND_OVERRIDE is None else f"zbound_{int(round(z_bound * 100)):03d}"
RUN_PREFIX_SUFFIXES = [suffix for suffix in [REPLAY_PREFIX_SUFFIX, Z_BOUND_PREFIX_SUFFIX] if suffix is not None]
RESULT_PREFIX = RESULT_PREFIX_BASE if not RUN_PREFIX_SUFFIXES else f"{RESULT_PREFIX_BASE}_{'_'.join(RUN_PREFIX_SUFFIXES)}"
COMPARE_PREFIX = COMPARE_PREFIX_BASE if not RUN_PREFIX_SUFFIXES else f"{COMPARE_PREFIX_BASE}_{'_'.join(RUN_PREFIX_SUFFIXES)}"
rl_save_agent_checkpoint = bool(CTRL["rl_save_agent_checkpoint"])
debug_validate_lifted = bool(CTRL["debug_validate_lifted"] if DEBUG_VALIDATE_LIFTED_OVERRIDE is None else DEBUG_VALIDATE_LIFTED_OVERRIDE)
debug_run_shadow_ls = bool(CTRL["debug_run_shadow_ls"] if DEBUG_RUN_SHADOW_LS_OVERRIDE is None else DEBUG_RUN_SHADOW_LS_OVERRIDE)
USE_SHIFTED_MPC_WARM_START = bool(CTRL["use_shifted_mpc_warm_start"])
observer_update_alignment = CTRL["observer_update_alignment"]

u_ss = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
b_min = apply_min_max(SYS["input_bounds"]["u_min"], data_min[:n_inputs], data_max[:n_inputs]) - u_ss
b_max = apply_min_max(SYS["input_bounds"]["u_max"], data_min[:n_inputs], data_max[:n_inputs]) - u_ss

MPC_obj = MpcSolverGeneral(
    A_aug,
    B_aug,
    C_aug,
    Q_out=np.array([Q1_penalty, Q2_penalty], float),
    R_in=np.array([R1_penalty, R2_penalty], float),
    NP=predict_h,
    NC=cont_h,
)

reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, n_inputs=n_inputs, **REWARD_CFG)

print_grouped_notebook_summary(
    "Resolved Markov parameters",
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
            "TD3 seed": td3_seed,
            "n_tests": n_tests,
            "set_points_len": set_points_len,
            "warm_start": warm_start,
            "max_steps": MAX_STEPS_OVERRIDE,
            "plot_start_episode": PLOT_START_EPISODE,
            "compare_start_episode": COMPARE_START_EPISODE,
        },
        "Controller": {
            "predict_h": predict_h,
            "cont_h": cont_h,
            "decision_interval": decision_interval,
            "basis_family": basis_family,
            "z_bound": z_bound,
            "z_bound_override": Z_BOUND_OVERRIDE,
            "z_safety": z_safety,
            "prediction_window": prediction_window,
            "gain_drift_max": gain_drift_max,
            "nominal_solver_mode": nominal_solver_mode,
            "force_td3_execute": force_td3_execute,
            "td3_priority_fallback_enabled": bool(CTRL.get("td3_priority_fallback", {}).get("enabled", False)),
            "td3_authority_ramp_enabled": bool(CTRL.get("td3_priority_fallback", {}).get("authority_ramp", {}).get("enabled", False)),
            "td3_reward_probation_enabled": bool(CTRL.get("td3_priority_fallback", {}).get("reward_probation", {}).get("enabled", False)),
            "td3_authority_scales": CTRL.get("td3_priority_fallback", {}).get("authority_ramp", {}),
            "td3_priority_fallback": CTRL.get("td3_priority_fallback", {}),
            "replay_storage_mode": replay_storage_mode,
            "base_state_norm_mode": base_state_norm_mode,
            "mismatch_feature_transform_mode": mismatch_feature_transform_mode,
            "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
        },
        "Behavioral cloning": BEHAVIORAL_CLONING,
        "Reward": reward_params,
        "Debug": {
            "debug_validate_lifted": debug_validate_lifted,
            "debug_run_shadow_ls": debug_run_shadow_ls,
        },
    },
)

# --- Cell 5 (code) ---
markov_cfg = {
    "agent_kind": AGENT_KIND,
    "run_mode": RUN_MODE,
    "n_tests": n_tests,
    "set_points_len": set_points_len,
    "warm_start": warm_start,
    "test_cycle": TEST_CYCLE,
    "predict_h": predict_h,
    "cont_h": cont_h,
    "decision_interval": decision_interval,
    "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
    "observer_update_alignment": observer_update_alignment,
    "basis_family": basis_family,
    "z_bound": z_bound,
    "z_safety": z_safety,
    "prediction_window": prediction_window,
    "lambda_z": lambda_z,
    "s_pred_min": s_pred_min,
    "gain_drift_max": gain_drift_max,
    "nominal_solver_mode": nominal_solver_mode,
    "nominal_cost_relative_tol": nominal_cost_relative_tol,
    "nominal_cost_absolute_tol": nominal_cost_absolute_tol,
    "mismatch_clip": mismatch_clip,
    "base_state_norm_mode": base_state_norm_mode,
    "base_state_running_norm_clip": base_state_running_norm_clip,
    "base_state_running_norm_eps": base_state_running_norm_eps,
    "innovation_scale_mode": innovation_scale_mode,
    "innovation_scale_ref": innovation_scale_ref,
    "tracking_scale_mode": tracking_scale_mode,
    "tracking_eta_tol": tracking_eta_tol,
    "tracking_scale_floor": tracking_scale_floor,
    "tracking_scale_floor_mode": tracking_scale_floor_mode,
    "mismatch_feature_transform_mode": mismatch_feature_transform_mode,
    "mismatch_transform_tanh_scale": mismatch_transform_tanh_scale,
    "mismatch_transform_post_clip": mismatch_transform_post_clip,
    "run_adaptive_ls": run_adaptive_ls,
    "run_live_corrected_mpc": run_live_corrected_mpc,
    "run_rl_proposal": run_rl_proposal,
    "rl_fallback_to_ls": rl_fallback_to_ls,
    "force_td3_execute": force_td3_execute,
    "td3_priority_fallback": CTRL.get("td3_priority_fallback", {}),
    "rl_store_executed_action_in_replay": rl_store_executed_action_in_replay,
    "rl_save_agent_checkpoint": rl_save_agent_checkpoint,
    "debug_validate_lifted": debug_validate_lifted,
    "debug_run_shadow_ls": debug_run_shadow_ls,
    "nominal_qi": CTRL["nominal_qi"],
    "nominal_qs": CTRL["nominal_qs"],
    "nominal_ha": CTRL["nominal_ha"],
    "qi_change": CTRL["qi_change"],
    "qs_change": CTRL["qs_change"],
    "ha_change": CTRL["ha_change"],
    "Q1_penalty": Q1_penalty,
    "Q2_penalty": Q2_penalty,
    "R1_penalty": R1_penalty,
    "R2_penalty": R2_penalty,
    "b_min": b_min,
    "b_max": b_max,
    "behavioral_cloning": BEHAVIORAL_CLONING,
    "td3_agent": TD3_CFG,
    "max_steps": MAX_STEPS_OVERRIDE,
}

runtime_ctx = {
    "system_factory": lambda: PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t),
    "MPC_obj": MPC_obj,
    "steady_states": steady_states,
    "system_data": system_data,
    "data_min": data_min,
    "data_max": data_max,
    "A_aug": A_aug,
    "B_aug": B_aug,
    "C_aug": C_aug,
    "poles": poles,
    "y_sp_scenario": y_sp_scenario,
    "reward_fn": reward_fn,
    "reward_params": reward_params,
    "system_metadata": POLYMER_SYSTEM_METADATA,
    "delta_t": delta_t,
}

result_bundle = run_markov_correction_supervisor(markov_cfg=markov_cfg, runtime_ctx=runtime_ctx)
result_bundle["mpc_path_or_dir"] = BASELINE_MPC_PATH
result_bundle["agent_kind"] = AGENT_KIND
result_bundle["run_mode"] = RUN_MODE

# --- Cell 6 (code) ---
out_dir_rl = plot_markov_correction_results(
    result_bundle=result_bundle,
    plot_cfg={
        "directory": RESULT_DIR,
        "prefix_name": RESULT_PREFIX,
        "start_episode": PLOT_START_EPISODE,
        "save_pdf": SAVE_PDF,
        "style_profile": STYLE_PROFILE,
        "save_agent_checkpoint": rl_save_agent_checkpoint,
        "agent_checkpoint_prefix": "td3_markov_agent",
        "s_pred_min": s_pred_min,
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
    n_inputs=n_inputs,
    save_pdf=SAVE_PDF,
    style_profile=STYLE_PROFILE,
)

print("RL result directory:", out_dir_rl)
print("Comparison directory:", out_dir_cmp)

# --- Cell 7 (code) ---
result_bundle["summary_metrics"]
