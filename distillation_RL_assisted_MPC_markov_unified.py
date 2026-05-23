# Generated from distillation_RL_assisted_MPC_markov_unified.ipynb

# --- Cell 0 (markdown) ---
# # Distillation Markov Correction Supervisor
#
# Canonical unified distillation notebook for the TD3 Markov correction workflow.

# --- Cell 1 (code) ---
from pathlib import Path
import os

import numpy as np

from Simulation.mpc import MpcSolverGeneral
from systems.distillation import DISTILLATION_SYSTEM_METADATA, get_distillation_notebook_defaults
from systems.distillation.data_io import canonical_baseline_path, load_distillation_system_data
from systems.distillation.plant import build_distillation_system, distillation_system_stepper
from systems.distillation.scenarios import build_distillation_disturbance_schedule
from utils.helpers import apply_min_max
from utils.markov_runner import run_markov_correction_supervisor
from utils.notebook_setup import prepare_distillation_notebook_env, print_grouped_notebook_summary
from utils.plotting import compare_mpc_rl_from_dirs, plot_markov_correction_results
from utils.rewards import make_reward_fn_relative_QR

NB = get_distillation_notebook_defaults("markov")
AGENT_KIND = NB["agent_kind"]
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
COMPARE_PREFIX_OVERRIDE = NB["compare_prefix_override"]
BASELINE_MPC_PATH_OVERRIDE = NB["baseline_mpc_path_override"]
N_TESTS_OVERRIDE = NB["n_tests_override"]
SET_POINTS_LEN_OVERRIDE = NB["set_points_len_override"]
WARM_START_OVERRIDE = NB["warm_start_override"]
TEST_CYCLE_OVERRIDE = NB["test_cycle_override"]
PLOT_START_EPISODE_OVERRIDE = NB["plot_start_episode_override"]
COMPARE_START_EPISODE_OVERRIDE = NB["compare_start_episode_override"]

NOTEBOOK_FAMILY = "markov"
MAX_STEPS_OVERRIDE = None
DEBUG_VALIDATE_LIFTED_OVERRIDE = None
DEBUG_RUN_SHADOW_LS_OVERRIDE = None

REPO_ROOT, DATA_DIR, RESULT_DIR, DISTURBANCE_PROFILE, DYN_PATH, SNAPS_PATH, ASPEN_SOURCE = prepare_distillation_notebook_env(
    run_mode=RUN_MODE,
    disturbance_profile=DISTURBANCE_PROFILE,
    family=NOTEBOOK_FAMILY,
    aspen_preset=ASPEN_PRESET,
    dyn_path_override=ASPEN_PATH_OVERRIDE,
    snaps_path_override=SNAPS_PATH_OVERRIDE,
    aspen_root_override=ASPEN_ROOT_OVERRIDE,
    data_dir_override=DISTILLATION_DATA_DIR_OVERRIDE,
    results_dir_override=DISTILLATION_RESULTS_DIR_OVERRIDE,
)
os.chdir(REPO_ROOT)
if AGENT_KIND != "td3":
    raise ValueError("Distillation Markov v1 supports TD3 only.")

# --- Cell 2 (code) ---
SYS = NB["system_setup"]
RUN_PROFILE = NB["run_profiles"][(AGENT_KIND, RUN_MODE, DISTURBANCE_PROFILE)]

nominal_conditions = SYS["nominal_conditions"].copy()
ss_inputs = SYS["ss_inputs"].copy()
u_min = SYS["input_bounds"]["u_min"].copy()
u_max = SYS["input_bounds"]["u_max"].copy()
setpoint_y = SYS["setpoint_range_phys"].copy()
y_sp_scenario_phys = SYS["rl_setpoints_phys"].copy()
delta_t = SYS["delta_t_hours"]

system = build_distillation_system(
    path=DYN_PATH,
    ss_inputs=ss_inputs,
    initialization_point=nominal_conditions,
    delta_t=delta_t,
    visible=DISTILLATION_VISIBLE,
)
steady_states = {
    "ss_inputs": np.asarray(system.ss_inputs, float).copy(),
    "y_ss": np.asarray(system.y_ss, float).copy(),
}
disturbance_nominal_feed = float(system.feed.FmR.Value)

system_data = load_distillation_system_data(
    REPO_ROOT,
    steady_states,
    setpoint_y,
    u_min,
    u_max,
    data_override=DISTILLATION_DATA_DIR_OVERRIDE,
)
A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
inputs_number = int(B_aug.shape[1])
y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[inputs_number:], data_max[inputs_number:]) - apply_min_max(
    steady_states["y_ss"], data_min[inputs_number:], data_max[inputs_number:]
)

RESULT_PREFIX = RESULT_PREFIX_OVERRIDE or f"distillation_markov_{AGENT_KIND}_{RUN_MODE}_{DISTURBANCE_PROFILE}_unified"
COMPARE_PREFIX = COMPARE_PREFIX_OVERRIDE or f"distillation_compare_markov_{AGENT_KIND}_{RUN_MODE}_{DISTURBANCE_PROFILE}"
BASELINE_MPC_PATH = Path(BASELINE_MPC_PATH_OVERRIDE).expanduser() if BASELINE_MPC_PATH_OVERRIDE else canonical_baseline_path(
    REPO_ROOT,
    RUN_MODE,
    DISTURBANCE_PROFILE,
    data_override=DISTILLATION_DATA_DIR_OVERRIDE,
)
def close_markov_system(system):
    try:
        system.close(SNAPS_PATH)
    except Exception:
        pass

# --- Cell 3 (code) ---
EPISODE_CFG = RUN_PROFILE
CTRL = NB["controller"]
TD3_CFG = NB["td3_agent"]
REWARD_CFG = NB["reward"]
BEHAVIORAL_CLONING = dict(NB.get("behavioral_cloning", {}))
TD3_AUTHORITY_RAMP_CFG = dict(CTRL.get("td3_authority_ramp", {}))

n_tests = int(EPISODE_CFG["n_tests"] if N_TESTS_OVERRIDE is None else N_TESTS_OVERRIDE)
set_points_len = int(EPISODE_CFG["set_points_len"] if SET_POINTS_LEN_OVERRIDE is None else SET_POINTS_LEN_OVERRIDE)
warm_start = int(EPISODE_CFG["warm_start"] if WARM_START_OVERRIDE is None else WARM_START_OVERRIDE)
TEST_CYCLE = list(EPISODE_CFG["test_cycle"] if TEST_CYCLE_OVERRIDE is None else TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = int(EPISODE_CFG["plot_start_episode"] if PLOT_START_EPISODE_OVERRIDE is None else PLOT_START_EPISODE_OVERRIDE)
COMPARE_START_EPISODE = int(EPISODE_CFG["compare_start_episode"] if COMPARE_START_EPISODE_OVERRIDE is None else COMPARE_START_EPISODE_OVERRIDE)

TOTAL_STEPS = int(n_tests * set_points_len * len(y_sp_scenario_phys))
DISTURBANCE_SCHEDULE = build_distillation_disturbance_schedule(
    RUN_MODE,
    DISTURBANCE_PROFILE,
    TOTAL_STEPS,
    nominal_feed=disturbance_nominal_feed,
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
z_bound = float(CTRL["z_bound"])
z_safety = CTRL.get("z_safety", {})
prediction_window = int(CTRL["prediction_window"])
lambda_z = float(CTRL["lambda_z"])
s_pred_min = float(CTRL["s_pred_min"])
gain_drift_max = float(CTRL["gain_drift_max"])
nominal_cost_relative_tol = float(CTRL["nominal_cost_relative_tol"])
nominal_cost_absolute_tol = float(CTRL["nominal_cost_absolute_tol"])
nominal_solver_mode = str(CTRL.get("nominal_solver_mode", "state_space_shared"))
td3_seed = TD3_CFG.get("seed")
run_adaptive_ls = bool(CTRL["run_adaptive_ls"])
run_live_corrected_mpc = bool(CTRL["run_live_corrected_mpc"])
run_rl_proposal = bool(CTRL["run_rl_proposal"])
rl_fallback_to_ls = bool(CTRL["rl_fallback_to_ls"])
force_td3_execute = bool(CTRL["force_td3_execute"])
rl_store_executed_action_in_replay = bool(CTRL["rl_store_executed_action_in_replay"])
rl_save_agent_checkpoint = bool(CTRL["rl_save_agent_checkpoint"])
debug_validate_lifted = bool(CTRL["debug_validate_lifted"] if DEBUG_VALIDATE_LIFTED_OVERRIDE is None else DEBUG_VALIDATE_LIFTED_OVERRIDE)
debug_run_shadow_ls = bool(CTRL["debug_run_shadow_ls"] if DEBUG_RUN_SHADOW_LS_OVERRIDE is None else DEBUG_RUN_SHADOW_LS_OVERRIDE)
USE_SHIFTED_MPC_WARM_START = bool(CTRL["use_shifted_mpc_warm_start"])
observer_update_alignment = CTRL["observer_update_alignment"]

MPC_obj = MpcSolverGeneral(
    A_aug,
    B_aug,
    C_aug,
    Q_out=np.array([Q1_penalty, Q2_penalty], float),
    R_in=np.array([R1_penalty, R2_penalty], float),
    NP=predict_h,
    NC=cont_h,
)
reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, inputs_number, **REWARD_CFG)

print_grouped_notebook_summary(
    "Resolved Distillation Markov parameters",
    {
        "Paths": {
            "Repo root": REPO_ROOT,
            "Data dir": DATA_DIR,
            "Results dir": RESULT_DIR,
            "Aspen source": ASPEN_SOURCE,
            "Dyn path": DYN_PATH,
            "Snaps path": SNAPS_PATH,
            "Baseline MPC": BASELINE_MPC_PATH,
        },
        "Run setup": {
            "Agent kind": AGENT_KIND,
            "Run mode": RUN_MODE,
            "Disturbance profile": DISTURBANCE_PROFILE,
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
            "z_safety": z_safety,
            "prediction_window": prediction_window,
            "gain_drift_max": gain_drift_max,
            "nominal_solver_mode": nominal_solver_mode,
            "force_td3_execute": force_td3_execute,
            "td3_controlled_authority": TD3_AUTHORITY_RAMP_CFG,
            "td3_priority_fallback_enabled": bool(CTRL.get("td3_priority_fallback", {}).get("enabled", False)),
            "td3_authority_ramp_enabled": bool(CTRL.get("td3_priority_fallback", {}).get("authority_ramp", {}).get("enabled", False)),
            "td3_reward_probation_enabled": bool(CTRL.get("td3_priority_fallback", {}).get("reward_probation", {}).get("enabled", False)),
            "td3_authority_scales": CTRL.get("td3_priority_fallback", {}).get("authority_ramp", {}),
            "td3_priority_fallback": CTRL.get("td3_priority_fallback", {}),
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

# --- Cell 4 (code) ---
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
    "run_adaptive_ls": run_adaptive_ls,
    "run_live_corrected_mpc": run_live_corrected_mpc,
    "run_rl_proposal": run_rl_proposal,
    "rl_fallback_to_ls": rl_fallback_to_ls,
    "force_td3_execute": force_td3_execute,
    "td3_authority_ramp": dict(TD3_AUTHORITY_RAMP_CFG),
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
    "b_min": system_data["b_min"],
    "b_max": system_data["b_max"],
    "behavioral_cloning": BEHAVIORAL_CLONING,
    "td3_agent": TD3_CFG,
    "max_steps": MAX_STEPS_OVERRIDE,
}

runtime_ctx = {
    "system": system,
    "system_teardown": close_markov_system,
    "system_stepper": distillation_system_stepper,
    "disturbance_schedule": DISTURBANCE_SCHEDULE,
    "delta_t": delta_t,
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
    "system_metadata": DISTILLATION_SYSTEM_METADATA,
}

result_bundle = run_markov_correction_supervisor(markov_cfg=markov_cfg, runtime_ctx=runtime_ctx)
result_bundle["mpc_path_or_dir"] = BASELINE_MPC_PATH
result_bundle["agent_kind"] = AGENT_KIND
result_bundle["run_mode"] = RUN_MODE

# --- Cell 5 (code) ---
out_dir_rl = plot_markov_correction_results(
    result_bundle=result_bundle,
    plot_cfg={
        "directory": RESULT_DIR,
        "prefix_name": RESULT_PREFIX,
        "start_episode": PLOT_START_EPISODE,
        "save_pdf": SAVE_PDF,
        "style_profile": STYLE_PROFILE,
        "save_agent_checkpoint": rl_save_agent_checkpoint,
        "agent_checkpoint_prefix": "distillation_td3_markov_agent",
        "s_pred_min": s_pred_min,
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

# --- Cell 6 (code) ---
print(result_bundle["summary_metrics"])
