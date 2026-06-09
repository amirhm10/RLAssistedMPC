# Active distillation combined supervisor entrypoint.
#
# Two active modes are supported:
#   sg    -> SG-DQN horizon + SG-TD3 Markov/weights/residual
#   plain -> DQN horizon + TD3 Markov/weights/residual

from pathlib import Path
import os

import numpy as np
import torch

from DQN.dqn_agent import DQNAgent
from DQN.supervisor_gated_dqn_agent import SupervisorGatedDQNAgent
from TD3Agent.agent import TD3Agent
from TD3Agent.supervisor_gated_agent import SupervisorGatedTD3Agent, SupervisorGateConfig
from systems.distillation import (
    DISTILLATION_SYSTEM_METADATA,
    build_distillation_disturbance_schedule,
    build_distillation_system,
    distillation_system_stepper,
    get_distillation_notebook_defaults,
    resolve_distillation_combined_agent_kinds,
)
from systems.distillation.data_io import canonical_baseline_path, load_distillation_system_data
from utils.combined_runner import run_combined_supervisor
from utils.helpers import apply_min_max, build_horizon_recipes
from utils.markov_runner import compute_markov_blocks, make_markov_basis
from utils.notebook_setup import prepare_distillation_notebook_env, print_grouped_notebook_summary
from utils.plotting import compare_mpc_rl_from_dirs, plot_combined_results
from utils.rewards import make_reward_fn_relative_QR
from utils.state_features import get_rl_state_dim


NOTEBOOK_SOURCE = globals().get("NOTEBOOK_SOURCE_OVERRIDE", "distillation_RL_assisted_MPC_combined_unified.py")
RUN_SUMMARY_TITLE = globals().get(
    "RUN_SUMMARY_TITLE_OVERRIDE",
    "Distillation combined supervisor run summary",
)
NB_CONFIGURE = globals().get("NB_CONFIGURE")
NB = get_distillation_notebook_defaults("combined")
if NB_CONFIGURE is not None:
    configured_nb = NB_CONFIGURE(NB)
    if configured_nb is not None:
        NB = configured_nb


def _normalize_combined_mode(mode) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"without_sg", "no_sg"}:
        return "plain"
    return normalized


def _replay_params(cfg, set_points_len):
    buffer_size = int(cfg["buffer_size"])
    replay_recent_window_mult = int(cfg["replay_recent_window_mult"])
    replay_recent_window = (
        int(cfg["replay_recent_window"])
        if cfg["replay_recent_window"] is not None
        else min(buffer_size, replay_recent_window_mult * int(set_points_len))
    )
    return {
        "buffer_size": buffer_size,
        "replay_frac_per": float(cfg["replay_frac_per"]),
        "replay_frac_recent": float(cfg["replay_frac_recent"]),
        "replay_recent_window_mult": replay_recent_window_mult,
        "replay_recent_window": replay_recent_window,
        "replay_alpha": float(cfg["replay_alpha"]),
        "replay_beta_start": float(cfg["replay_beta_start"]),
        "replay_beta_end": float(cfg["replay_beta_end"]),
        "replay_beta_steps": int(cfg["replay_beta_steps"]),
    }


def _assert_equal_config(name, actual, expected) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not np.allclose(np.asarray(actual, float), np.asarray(expected, float)):
            raise ValueError(f"Combined default drift detected for {name}.")
        return
    if actual != expected:
        raise ValueError(f"Combined default drift detected for {name}.")


def validate_standalone_parity(nb: dict) -> None:
    horizon_nb = get_distillation_notebook_defaults("horizon_standard")
    markov_nb = get_distillation_notebook_defaults("markov")
    weights_nb = get_distillation_notebook_defaults("weights")
    residual_nb = get_distillation_notebook_defaults("residual")

    _assert_equal_config("horizon_agent", nb["horizon_agent"], horizon_nb["agent"])
    _assert_equal_config("horizon_supervisor_gate", nb["horizon_supervisor_gate"], horizon_nb["supervisor_gate"])
    _assert_equal_config("horizon_safety", nb["horizon_safety"], horizon_nb["horizon_safety"])
    _assert_equal_config("markov_td3_agent", nb["markov_td3_agent"], markov_nb["td3_agent"])
    _assert_equal_config("markov_supervisor_gate", nb["markov_supervisor_gate"], markov_nb["supervisor_gate"])
    _assert_equal_config("weights_td3_agent", nb["weights_td3_agent"], weights_nb["td3_agent"])
    _assert_equal_config("weights_supervisor_gate", nb["weights_supervisor_gate"], weights_nb["supervisor_gate"])
    _assert_equal_config("weight_safety", nb["weight_safety"], weights_nb["weight_safety"])
    _assert_equal_config("residual_td3_agent", nb["residual_td3_agent"], residual_nb["td3_agent"])
    _assert_equal_config("residual_supervisor_gate", nb["residual_supervisor_gate"], residual_nb["supervisor_gate"])
    _assert_equal_config("residual_safety", nb["residual_safety"], residual_nb["residual_safety"])

    ctrl = nb["controller"]
    for key in (
        "decision_interval",
        "predict_grid",
        "control_grid",
        "predict_h",
        "cont_h",
        "Q1_penalty",
        "Q2_penalty",
        "R1_penalty",
        "R2_penalty",
        "mismatch_clip",
        "innovation_scale_mode",
        "innovation_scale_ref",
        "tracking_scale_mode",
        "tracking_eta_tol",
        "tracking_scale_floor",
        "tracking_scale_floor_mode",
        "base_state_norm_mode",
        "base_state_running_norm_clip",
        "base_state_running_norm_eps",
        "mismatch_feature_transform_mode",
        "mismatch_transform_tanh_scale",
        "mismatch_transform_post_clip",
        "observer_update_alignment",
    ):
        _assert_equal_config(f"controller.{key}", ctrl[key], horizon_nb["controller"][key])

    for key in (
        "basis_family",
        "z_bound",
        "z_safety",
        "prediction_window",
        "lambda_z",
        "s_pred_min",
        "gain_drift_max",
        "nominal_cost_relative_tol",
        "nominal_cost_absolute_tol",
        "run_adaptive_ls",
        "run_live_corrected_mpc",
        "run_rl_proposal",
        "rl_fallback_to_ls",
        "force_td3_execute",
        "force_td3_respects_warm_start",
        "rl_store_executed_action_in_replay",
        "td3_priority_fallback",
        "markov_shadow_safety",
    ):
        _assert_equal_config(f"controller.{key}", ctrl[key], markov_nb["controller"].get(key))

    _assert_equal_config("controller.weights_low", ctrl["weights_low"], weights_nb["controller"]["low_coef"])
    _assert_equal_config("controller.weights_high", ctrl["weights_high"], weights_nb["controller"]["high_coef"])
    _assert_equal_config("controller.residual_low", ctrl["residual_low"], residual_nb["controller"]["low_coef"])
    _assert_equal_config("controller.residual_high", ctrl["residual_high"], residual_nb["controller"]["high_coef"])

    for key in (
        "residual_authority_enabled",
        "append_rho_to_state",
        "authority_use_rho",
        "use_rho_authority",
        "residual_zero_deadband_enabled",
    ):
        _assert_equal_config(key, nb[key], residual_nb[key])


def make_continuous_agent(agent_kind, state_dim, action_dim, td3_cfg, device, set_points_len, supervisor_gate=None):
    if agent_kind not in {"td3", "sg_td3"}:
        raise ValueError("Continuous combined agent kind must be 'td3' or 'sg_td3'.")
    replay = _replay_params(td3_cfg, set_points_len)
    agent_cls = SupervisorGatedTD3Agent if agent_kind == "sg_td3" else TD3Agent
    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden=list(td3_cfg["actor_hidden"]),
        critic_hidden=list(td3_cfg["critic_hidden"]),
        gamma=td3_cfg["gamma"],
        actor_lr=td3_cfg["actor_lr"],
        critic_lr=td3_cfg["critic_lr"],
        batch_size=td3_cfg["batch_size"],
        n_step=td3_cfg["n_step"],
        multistep_mode=td3_cfg["multistep_mode"],
        lambda_value=td3_cfg["lambda_value"],
        grad_clip_norm=td3_cfg.get("grad_clip_norm", 10.0),
        policy_delay=td3_cfg["policy_delay"],
        target_policy_smoothing_noise_std=td3_cfg["target_policy_smoothing_noise_std"],
        noise_clip=td3_cfg["noise_clip"],
        max_action=td3_cfg["max_action"],
        tau=td3_cfg["tau"],
        std_start=td3_cfg["std_start"],
        std_end=td3_cfg["std_end"],
        std_decay_rate=td3_cfg["std_decay_rate"],
        std_decay_mode=td3_cfg["std_decay_mode"],
        buffer_size=replay["buffer_size"],
        replay_frac_per=replay["replay_frac_per"],
        replay_frac_recent=replay["replay_frac_recent"],
        replay_recent_window=replay["replay_recent_window"],
        replay_alpha=replay["replay_alpha"],
        replay_beta_start=replay["replay_beta_start"],
        replay_beta_end=replay["replay_beta_end"],
        replay_beta_steps=replay["replay_beta_steps"],
        device=device,
        actor_freeze=td3_cfg["actor_freeze"],
        exploration_mode=td3_cfg["exploration_mode"],
        loss_type=td3_cfg["loss_type"],
        param_noise_std_start=td3_cfg.get("param_noise_std_start", td3_cfg["std_start"]),
        param_noise_std_end=td3_cfg.get("param_noise_std_end", td3_cfg["std_end"]),
        param_noise_resample_interval=td3_cfg["param_noise_resample_interval"],
    )
    if agent_kind == "sg_td3":
        kwargs["supervisor_gate_config"] = SupervisorGateConfig(**dict(supervisor_gate or {}))
    return agent_cls(**kwargs)


COMBINED_AGENT_MODE = _normalize_combined_mode(NB.get("combined_agent_mode", "sg"))
RESOLVED_AGENT_KINDS = resolve_distillation_combined_agent_kinds(COMBINED_AGENT_MODE)
NB["combined_agent_mode"] = COMBINED_AGENT_MODE

RUN_MODE = NB["run_mode"]
DISTURBANCE_PROFILE = NB["disturbance_profile"]
STYLE_PROFILE = NB["style_profile"]
SAVE_PDF = NB["save_pdf"]

ENABLE_HORIZON = bool(NB["enable_horizon"])
HORIZON_AGENT_KIND = RESOLVED_AGENT_KINDS["horizon_agent_kind"]
HORIZON_STATE_MODE = NB["horizon_state_mode"]

ENABLE_MARKOV = bool(NB.get("enable_markov", True))
MARKOV_AGENT_KIND = RESOLVED_AGENT_KINDS["markov_agent_kind"]
MARKOV_STATE_MODE = NB.get("markov_state_mode", "mismatch")

ENABLE_MATRIX = bool(NB.get("enable_matrix", False))
MATRIX_AGENT_KIND = str(NB.get("matrix_agent_kind", "td3")).lower()
MATRIX_STATE_MODE = NB.get("matrix_state_mode", "mismatch")

ENABLE_WEIGHTS = bool(NB["enable_weights"])
WEIGHTS_AGENT_KIND = RESOLVED_AGENT_KINDS["weights_agent_kind"]
WEIGHTS_STATE_MODE = NB["weights_state_mode"]

ENABLE_RESIDUAL = bool(NB["enable_residual"])
RESIDUAL_AGENT_KIND = RESOLVED_AGENT_KINDS["residual_agent_kind"]
RESIDUAL_STATE_MODE = NB["residual_state_mode"]

if ENABLE_MATRIX:
    raise ValueError("The active distillation combined runner keeps the legacy matrix branch disabled.")
if COMBINED_AGENT_MODE not in {"sg", "plain"}:
    raise ValueError("combined_agent_mode must be 'sg' or 'plain'.")
if HORIZON_AGENT_KIND not in {"dqn", "sg_dqn"}:
    raise ValueError("Combined horizon agent kind must be 'dqn' or 'sg_dqn'.")
for agent_kind in (MARKOV_AGENT_KIND, WEIGHTS_AGENT_KIND, RESIDUAL_AGENT_KIND):
    if agent_kind not in {"td3", "sg_td3"}:
        raise ValueError("Combined continuous agent kinds must be 'td3' or 'sg_td3'.")

USE_RHO_AUTHORITY = bool(NB["authority_use_rho"])
RESIDUAL_AUTHORITY_ENABLED = bool(NB["residual_authority_enabled"])
APPEND_RHO_TO_STATE = bool(NB["append_rho_to_state"])
if any((USE_RHO_AUTHORITY, RESIDUAL_AUTHORITY_ENABLED, APPEND_RHO_TO_STATE)):
    raise ValueError("The active distillation combined runner keeps residual rho authority disabled.")

AUTHORITY_BETA_RES = NB["authority_beta_res"]
AUTHORITY_DU0_RES = NB["authority_du0_res"]
AUTHORITY_ETA_TOL = NB["authority_eta_tol"]
AUTHORITY_RHO_FLOOR = NB["authority_rho_floor"]
AUTHORITY_RHO_POWER = NB["authority_rho_power"]
RHO_MAPPING_MODE = NB["rho_mapping_mode"]
AUTHORITY_RHO_K = NB["authority_rho_k"]
RESIDUAL_ZERO_DEADBAND_ENABLED = bool(NB["residual_zero_deadband_enabled"])
if RESIDUAL_ZERO_DEADBAND_ENABLED:
    raise ValueError("The active distillation combined runner keeps residual deadband disabled.")
RESIDUAL_ZERO_TRACKING_RAW_THRESHOLD = NB["residual_zero_tracking_raw_threshold"]
RESIDUAL_ZERO_INNOVATION_RAW_THRESHOLD = NB["residual_zero_innovation_raw_threshold"]

validate_standalone_parity(NB)

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

REPO_ROOT, DATA_DIR, RESULT_DIR, DISTURBANCE_PROFILE, DYN_PATH, SNAPS_PATH, ASPEN_SOURCE = prepare_distillation_notebook_env(
    run_mode=RUN_MODE,
    disturbance_profile=DISTURBANCE_PROFILE,
    family="combined",
    aspen_preset=ASPEN_PRESET,
    dyn_path_override=ASPEN_PATH_OVERRIDE,
    snaps_path_override=SNAPS_PATH_OVERRIDE,
    aspen_root_override=ASPEN_ROOT_OVERRIDE,
    data_dir_override=DISTILLATION_DATA_DIR_OVERRIDE,
    results_dir_override=DISTILLATION_RESULTS_DIR_OVERRIDE,
)
os.chdir(REPO_ROOT)
RUN_PROFILE = NB["run_profiles"][(RUN_MODE, DISTURBANCE_PROFILE)]

SYS = NB["system_setup"]
nominal_conditions = SYS["nominal_conditions"].copy()
ss_inputs = SYS["ss_inputs"].copy()
u_min = SYS["input_bounds"]["u_min"].copy()
u_max = SYS["input_bounds"]["u_max"].copy()
setpoint_y = SYS["setpoint_range_phys"].copy()
y_sp_scenario_phys = SYS.get("combined_setpoints_phys", SYS["rl_setpoints_phys"]).copy()
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
    steady_states=steady_states,
    setpoint_y=setpoint_y,
    u_min=u_min,
    u_max=u_max,
    data_override=DISTILLATION_DATA_DIR_OVERRIDE,
)
A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
min_max_dict = system_data["min_max_dict"]

N_INPUTS = int(B_aug.shape[1])
N_OUTPUTS = int(C_aug.shape[0])
y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[N_INPUTS:], data_max[N_INPUTS:]) - apply_min_max(
    steady_states["y_ss"],
    data_min[N_INPUTS:],
    data_max[N_INPUTS:],
)

EPISODE_CFG = NB["episode_defaults"]
CTRL = NB["controller"]
REWARD_CFG = NB["reward"]
HORIZON_CFG = NB["horizon_agent"]
MARKOV_TD3_CFG = NB["markov_td3_agent"]
WEIGHTS_TD3_CFG = NB["weights_td3_agent"]
RESIDUAL_TD3_CFG = NB["residual_td3_agent"]
HORIZON_GATE_CFG = dict(NB.get("horizon_supervisor_gate", HORIZON_CFG.get("supervisor_gate", {})))
MARKOV_GATE_CFG = dict(NB.get("markov_supervisor_gate", {}))
WEIGHTS_GATE_CFG = dict(NB.get("weights_supervisor_gate", {}))
RESIDUAL_GATE_CFG = dict(NB.get("residual_supervisor_gate", {}))

n_tests = int(RUN_PROFILE.get("n_tests", EPISODE_CFG["n_tests"]) if N_TESTS_OVERRIDE is None else N_TESTS_OVERRIDE)
set_points_len = int(
    RUN_PROFILE.get("set_points_len", EPISODE_CFG["set_points_len"])
    if SET_POINTS_LEN_OVERRIDE is None
    else SET_POINTS_LEN_OVERRIDE
)
warm_start = int(
    RUN_PROFILE.get("warm_start", EPISODE_CFG["warm_start"])
    if WARM_START_OVERRIDE is None
    else WARM_START_OVERRIDE
)
TEST_CYCLE = list(RUN_PROFILE.get("test_cycle", EPISODE_CFG["test_cycle"]) if TEST_CYCLE_OVERRIDE is None else TEST_CYCLE_OVERRIDE)
PLOT_START_EPISODE = int(RUN_PROFILE.get("plot_start_episode", 1) if PLOT_START_EPISODE_OVERRIDE is None else PLOT_START_EPISODE_OVERRIDE)
COMPARE_START_EPISODE = int(
    RUN_PROFILE.get("compare_start_episode", PLOT_START_EPISODE)
    if COMPARE_START_EPISODE_OVERRIDE is None
    else COMPARE_START_EPISODE_OVERRIDE
)
RESULT_PREFIX = RESULT_PREFIX_OVERRIDE or RUN_PROFILE["result_prefix_template"].format(mode=COMBINED_AGENT_MODE)
COMPARE_PREFIX = COMPARE_PREFIX_OVERRIDE or RUN_PROFILE["compare_prefix_template"].format(mode=COMBINED_AGENT_MODE)
BASELINE_MPC_PATH = (
    Path(BASELINE_MPC_PATH_OVERRIDE).expanduser()
    if BASELINE_MPC_PATH_OVERRIDE
    else canonical_baseline_path(
        REPO_ROOT,
        RUN_MODE,
        DISTURBANCE_PROFILE,
        data_override=DISTILLATION_DATA_DIR_OVERRIDE,
    )
)

TOTAL_STEPS = n_tests * set_points_len * len(y_sp_scenario_phys)
DISTURBANCE_SCHEDULE = build_distillation_disturbance_schedule(
    RUN_MODE,
    DISTURBANCE_PROFILE,
    TOTAL_STEPS,
    nominal_feed=disturbance_nominal_feed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICT_GRID = list(CTRL["predict_grid"])
CONTROL_GRID = list(CTRL["control_grid"])
HORIZON_RECIPES = build_horizon_recipes(PREDICT_GRID, CONTROL_GRID)

predict_h = int(CTRL["predict_h"])
cont_h = int(CTRL["cont_h"])
DECISION_INTERVAL = int(CTRL["decision_interval"])
Q1_penalty = float(CTRL["Q1_penalty"])
Q2_penalty = float(CTRL["Q2_penalty"])
R1_penalty = float(CTRL["R1_penalty"])
R2_penalty = float(CTRL["R2_penalty"])
USE_SHIFTED_MPC_WARM_START = bool(CTRL["use_shifted_mpc_warm_start"])

MISMATCH_COMMON = {
    "mismatch_clip": CTRL["mismatch_clip"],
    "innovation_scale_mode": CTRL["innovation_scale_mode"],
    "innovation_scale_ref": CTRL["innovation_scale_ref"],
    "tracking_scale_mode": CTRL["tracking_scale_mode"],
    "tracking_eta_tol": CTRL["tracking_eta_tol"],
    "tracking_scale_floor": CTRL["tracking_scale_floor"],
    "tracking_scale_floor_mode": CTRL["tracking_scale_floor_mode"],
    "base_state_norm_mode": CTRL["base_state_norm_mode"],
    "base_state_running_norm_clip": CTRL["base_state_running_norm_clip"],
    "base_state_running_norm_eps": CTRL["base_state_running_norm_eps"],
    "mismatch_feature_transform_mode": CTRL["mismatch_feature_transform_mode"],
    "mismatch_transform_tanh_scale": CTRL["mismatch_transform_tanh_scale"],
    "mismatch_transform_post_clip": CTRL["mismatch_transform_post_clip"],
    "observer_update_alignment": CTRL["observer_update_alignment"],
}

MODEL_LOW = CTRL["model_low"].copy()
MODEL_HIGH = CTRL["model_high"].copy()
WEIGHTS_LOW = CTRL["weights_low"].copy()
WEIGHTS_HIGH = CTRL["weights_high"].copy()
RESIDUAL_LOW = CTRL["residual_low"].copy()
RESIDUAL_HIGH = CTRL["residual_high"].copy()

nominal_qs = CTRL["nominal_qs"]
nominal_qi = CTRL["nominal_qi"]
nominal_hA = CTRL["nominal_ha"]
qi_change = CTRL["qi_change"]
qs_change = CTRL["qs_change"]
ha_change = CTRL["ha_change"]

reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, N_INPUTS, **REWARD_CFG)

markov_blocks_preview = compute_markov_blocks(A_aug, B_aug, C_aug, predict_h)
markov_basis_preview, MARKOV_BASIS_LABELS = make_markov_basis(markov_blocks_preview, CTRL["basis_family"])
MARKOV_ACTION_DIM = int(markov_basis_preview.shape[0])

horizon_state_dim = get_rl_state_dim(A_aug.shape[0], N_OUTPUTS, N_INPUTS, HORIZON_STATE_MODE)
markov_base_state_dim = get_rl_state_dim(A_aug.shape[0], N_OUTPUTS, N_INPUTS, MARKOV_STATE_MODE)
markov_state_dim = markov_base_state_dim + 2 * MARKOV_ACTION_DIM + 2 + 2 + 4
matrix_state_dim = get_rl_state_dim(A_aug.shape[0], N_OUTPUTS, N_INPUTS, MATRIX_STATE_MODE)
weights_state_dim = get_rl_state_dim(A_aug.shape[0], N_OUTPUTS, N_INPUTS, WEIGHTS_STATE_MODE)
residual_state_dim = get_rl_state_dim(
    A_aug.shape[0],
    N_OUTPUTS,
    N_INPUTS,
    RESIDUAL_STATE_MODE,
    append_rho_to_state=(RESIDUAL_STATE_MODE == "mismatch" and APPEND_RHO_TO_STATE),
)

horizon_agent = None
if ENABLE_HORIZON:
    horizon_replay = _replay_params(HORIZON_CFG, set_points_len)
    horizon_agent_kwargs = dict(
        state_dim=horizon_state_dim,
        action_dim=len(HORIZON_RECIPES),
        hidden_dim=list(HORIZON_CFG["hidden_layers"]),
        gamma=HORIZON_CFG["gamma"],
        lr=HORIZON_CFG["lr"],
        batch_size=HORIZON_CFG["batch_size"],
        buffer_size=horizon_replay["buffer_size"],
        replay_frac_per=horizon_replay["replay_frac_per"],
        replay_frac_recent=horizon_replay["replay_frac_recent"],
        replay_recent_window=horizon_replay["replay_recent_window"],
        replay_alpha=horizon_replay["replay_alpha"],
        replay_beta_start=horizon_replay["replay_beta_start"],
        replay_beta_end=horizon_replay["replay_beta_end"],
        replay_beta_steps=horizon_replay["replay_beta_steps"],
        n_step=HORIZON_CFG["n_step"],
        multistep_mode=HORIZON_CFG["multistep_mode"],
        lambda_value=HORIZON_CFG["lambda_value"],
        grad_clip_norm=HORIZON_CFG["grad_clip_norm"],
        double_dqn=HORIZON_CFG["double_dqn"],
        target_update=HORIZON_CFG["target_update"],
        tau=HORIZON_CFG["tau"],
        hard_update_interval=HORIZON_CFG["hard_update_interval"],
        activation=HORIZON_CFG["activation"],
        use_layer_norm=HORIZON_CFG["use_layer_norm"],
        dropout=HORIZON_CFG["dropout"],
        device=DEVICE,
        exploration_mode=HORIZON_CFG["exploration_mode"],
        loss_type=HORIZON_CFG["loss_type"],
        eps_start=HORIZON_CFG["eps_start"],
        eps_end=HORIZON_CFG["eps_end"],
        eps_decay_rate=HORIZON_CFG["eps_decay_rate"],
        eps_decay_steps=HORIZON_CFG.get("eps_decay_steps", 100_000),
        eps_decay_mode=HORIZON_CFG["eps_decay_mode"],
    )
    if HORIZON_AGENT_KIND == "dqn":
        horizon_agent = DQNAgent(**horizon_agent_kwargs, target_combine=HORIZON_CFG.get("target_combine"))
    elif HORIZON_AGENT_KIND == "sg_dqn":
        horizon_agent = SupervisorGatedDQNAgent(
            **horizon_agent_kwargs,
            target_combine=HORIZON_CFG.get("target_combine"),
            supervisor_gate_config=HORIZON_GATE_CFG,
        )
    else:
        raise ValueError("HORIZON_AGENT_KIND must be 'dqn' or 'sg_dqn'.")

markov_agent = None
if ENABLE_MARKOV:
    markov_agent = make_continuous_agent(
        MARKOV_AGENT_KIND,
        markov_state_dim,
        MARKOV_ACTION_DIM,
        MARKOV_TD3_CFG,
        DEVICE,
        set_points_len,
        supervisor_gate=MARKOV_GATE_CFG,
    )

matrix_agent = None
weights_agent = None
if ENABLE_WEIGHTS:
    weights_agent = make_continuous_agent(
        WEIGHTS_AGENT_KIND,
        weights_state_dim,
        4,
        WEIGHTS_TD3_CFG,
        DEVICE,
        set_points_len,
        supervisor_gate=WEIGHTS_GATE_CFG,
    )

residual_agent = None
if ENABLE_RESIDUAL:
    residual_agent = make_continuous_agent(
        RESIDUAL_AGENT_KIND,
        residual_state_dim,
        N_INPUTS,
        RESIDUAL_TD3_CFG,
        DEVICE,
        set_points_len,
        supervisor_gate=RESIDUAL_GATE_CFG,
    )

REPLAY_SETTINGS = {
    "horizon": _replay_params(HORIZON_CFG, set_points_len),
    "markov_td3": _replay_params(MARKOV_TD3_CFG, set_points_len),
    "weights_td3": _replay_params(WEIGHTS_TD3_CFG, set_points_len),
    "residual_td3": _replay_params(RESIDUAL_TD3_CFG, set_points_len),
}

print_grouped_notebook_summary(
    RUN_SUMMARY_TITLE,
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
            "Combined agent mode": COMBINED_AGENT_MODE,
            "Run mode": RUN_MODE,
            "Disturbance profile": DISTURBANCE_PROFILE,
            "n_tests": n_tests,
            "set_points_len": set_points_len,
            "warm_start": warm_start,
            "test_cycle": TEST_CYCLE,
            "decision_interval": DECISION_INTERVAL,
            "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
        },
        "Agents": {
            "horizon": f"enabled={ENABLE_HORIZON}, kind={HORIZON_AGENT_KIND}, state={HORIZON_STATE_MODE}",
            "markov_dynamic_matrix": f"enabled={ENABLE_MARKOV}, kind={MARKOV_AGENT_KIND}, state={MARKOV_STATE_MODE}",
            "legacy_matrix": f"enabled={ENABLE_MATRIX}, kind={MATRIX_AGENT_KIND}, state={MATRIX_STATE_MODE}",
            "weights": f"enabled={ENABLE_WEIGHTS}, kind={WEIGHTS_AGENT_KIND}, state={WEIGHTS_STATE_MODE}",
            "residual": f"enabled={ENABLE_RESIDUAL}, kind={RESIDUAL_AGENT_KIND}, state={RESIDUAL_STATE_MODE}",
        },
        "Supervisor gates": {
            "horizon": HORIZON_GATE_CFG if COMBINED_AGENT_MODE == "sg" else None,
            "markov": MARKOV_GATE_CFG if COMBINED_AGENT_MODE == "sg" else None,
            "weights": WEIGHTS_GATE_CFG if COMBINED_AGENT_MODE == "sg" else None,
            "residual": RESIDUAL_GATE_CFG if COMBINED_AGENT_MODE == "sg" else None,
        },
        "Residual authority": {
            "enabled": RESIDUAL_AUTHORITY_ENABLED,
            "use_rho": USE_RHO_AUTHORITY,
            "append_rho_to_state": APPEND_RHO_TO_STATE,
        },
        "Markov safety": {
            "basis_family": CTRL["basis_family"],
            "z_bound": CTRL["z_bound"],
            "z_safety": CTRL.get("z_safety", {}),
            "action_dim": MARKOV_ACTION_DIM,
            "basis_labels": MARKOV_BASIS_LABELS,
        },
        "Reward": reward_params,
        "Replay": REPLAY_SETTINGS,
        "Plotting": {
            "style_profile": STYLE_PROFILE,
            "save_pdf": SAVE_PDF,
            "result_prefix": RESULT_PREFIX,
            "compare_prefix": COMPARE_PREFIX,
        },
    },
)

combined_cfg = {
    "run_mode": RUN_MODE,
    "combined_agent_mode": COMBINED_AGENT_MODE,
    "horizon_agent_kind": HORIZON_AGENT_KIND,
    "notebook_source": NOTEBOOK_SOURCE,
    "decision_interval": DECISION_INTERVAL,
    "n_tests": n_tests,
    "set_points_len": set_points_len,
    "warm_start": warm_start,
    "horizon_post_warm_start_action_freeze_subepisodes": int(
        NB.get("horizon_post_warm_start_action_freeze_subepisodes", 0)
    ),
    "td3_post_warm_start_action_freeze_subepisodes": int(NB["td3_post_warm_start_action_freeze_subepisodes"]),
    "td3_post_warm_start_actor_freeze_subepisodes": int(NB["td3_post_warm_start_actor_freeze_subepisodes"]),
    "test_cycle": TEST_CYCLE,
    "predict_h": predict_h,
    "cont_h": cont_h,
    "use_shifted_mpc_warm_start": USE_SHIFTED_MPC_WARM_START,
    "Q1_penalty": Q1_penalty,
    "Q2_penalty": Q2_penalty,
    "R1_penalty": R1_penalty,
    "R2_penalty": R2_penalty,
    "nominal_qi": nominal_qi,
    "nominal_qs": nominal_qs,
    "nominal_ha": nominal_hA,
    "qi_change": qi_change,
    "qs_change": qs_change,
    "ha_change": ha_change,
    "b_min": system_data["b_min"],
    "b_max": system_data["b_max"],
    "horizon_cfg": {
        "enabled": ENABLE_HORIZON,
        "agent_kind": HORIZON_AGENT_KIND,
        "state_mode": HORIZON_STATE_MODE,
        **MISMATCH_COMMON,
        "horizon_recipes": HORIZON_RECIPES,
        "default_horizons": (predict_h, cont_h),
        "post_warm_start_action_freeze_subepisodes": int(
            NB.get("horizon_post_warm_start_action_freeze_subepisodes", 0)
        ),
        "supervisor_gate": HORIZON_GATE_CFG,
    },
    "markov_cfg": {
        "enabled": ENABLE_MARKOV,
        "agent_kind": MARKOV_AGENT_KIND,
        "state_mode": MARKOV_STATE_MODE,
        **MISMATCH_COMMON,
        "decision_interval": DECISION_INTERVAL,
        "basis_family": CTRL["basis_family"],
        "z_bound": float(CTRL["z_bound"]),
        "z_safety": dict(CTRL.get("z_safety", {})),
        "prediction_window": int(CTRL["prediction_window"]),
        "lambda_z": float(CTRL["lambda_z"]),
        "s_pred_min": float(CTRL["s_pred_min"]),
        "gain_drift_max": float(CTRL["gain_drift_max"]),
        "nominal_cost_relative_tol": float(CTRL["nominal_cost_relative_tol"]),
        "nominal_cost_absolute_tol": float(CTRL["nominal_cost_absolute_tol"]),
        "run_adaptive_ls": bool(CTRL["run_adaptive_ls"]),
        "run_live_corrected_mpc": bool(CTRL["run_live_corrected_mpc"]),
        "run_rl_proposal": bool(CTRL["run_rl_proposal"]),
        "rl_fallback_to_ls": bool(CTRL["rl_fallback_to_ls"]),
        "force_td3_execute": bool(CTRL["force_td3_execute"]),
        "force_td3_respects_warm_start": bool(CTRL.get("force_td3_respects_warm_start", False)),
        "rl_store_executed_action_in_replay": bool(CTRL["rl_store_executed_action_in_replay"]),
        "td3_priority_fallback": dict(CTRL.get("td3_priority_fallback", {})),
        "markov_shadow_safety": dict(CTRL.get("markov_shadow_safety", {})),
        "supervisor_gate": MARKOV_GATE_CFG,
    },
    "matrix_cfg": {
        "enabled": ENABLE_MATRIX,
        "agent_kind": MATRIX_AGENT_KIND,
        "state_mode": MATRIX_STATE_MODE,
        **MISMATCH_COMMON,
        "low_coef": MODEL_LOW,
        "high_coef": MODEL_HIGH,
        "release_protected_advisory_caps": {"enabled": False},
    },
    "weight_cfg": {
        "enabled": ENABLE_WEIGHTS,
        "agent_kind": WEIGHTS_AGENT_KIND,
        "state_mode": WEIGHTS_STATE_MODE,
        **MISMATCH_COMMON,
        "low_coef": WEIGHTS_LOW,
        "high_coef": WEIGHTS_HIGH,
        "weight_safety": dict(NB.get("weight_safety", {})),
        "supervisor_gate": WEIGHTS_GATE_CFG,
    },
    "residual_cfg": {
        "enabled": ENABLE_RESIDUAL,
        "agent_kind": RESIDUAL_AGENT_KIND,
        "state_mode": RESIDUAL_STATE_MODE,
        **MISMATCH_COMMON,
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
        "low_coef": RESIDUAL_LOW,
        "high_coef": RESIDUAL_HIGH,
        "residual_safety": dict(NB.get("residual_safety", {})),
        "supervisor_gate": RESIDUAL_GATE_CFG,
    },
}

agents = {}
if ENABLE_HORIZON:
    agents["horizon"] = horizon_agent
if ENABLE_MARKOV:
    agents["markov"] = markov_agent
if ENABLE_MATRIX:
    agents["matrix"] = matrix_agent
if ENABLE_WEIGHTS:
    agents["weights"] = weights_agent
if ENABLE_RESIDUAL:
    agents["residual"] = residual_agent

runtime_ctx = {
    "system": system,
    "agents": agents,
    "steady_states": steady_states,
    "min_max_dict": min_max_dict,
    "data_min": data_min,
    "data_max": data_max,
    "A_aug": A_aug,
    "B_aug": B_aug,
    "C_aug": C_aug,
    "poles": SYS["observer_poles"].copy(),
    "y_sp_scenario": y_sp_scenario,
    "reward_fn": reward_fn,
    "system_metadata": DISTILLATION_SYSTEM_METADATA,
    "reward_params": reward_params,
    "system_stepper": distillation_system_stepper,
    "disturbance_labels": DISTILLATION_SYSTEM_METADATA.get("disturbance_labels"),
    "disturbance_schedule": DISTURBANCE_SCHEDULE,
}

try:
    result_bundle = run_combined_supervisor(combined_cfg=combined_cfg, runtime_ctx=runtime_ctx)
    result_bundle["mpc_path_or_dir"] = BASELINE_MPC_PATH
    result_bundle["run_profile"] = RUN_PROFILE
    result_bundle["disturbance_profile"] = DISTURBANCE_PROFILE

    out_dir_rl = plot_combined_results(
        result_bundle=result_bundle,
        plot_cfg={
            "directory": RESULT_DIR,
            "prefix_name": RESULT_PREFIX,
            "start_episode": PLOT_START_EPISODE,
            "save_pdf": SAVE_PDF,
            "style_profile": STYLE_PROFILE,
            "system_metadata": DISTILLATION_SYSTEM_METADATA,
            "include_baseline_compare": False,
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
        save_pdf=SAVE_PDF,
        style_profile=STYLE_PROFILE,
    )

    print("Combined result directory:", out_dir_rl)
    print("Comparison result directory:", out_dir_cmp)
finally:
    try:
        system.close(SNAPS_PATH)
    except Exception:
        pass
