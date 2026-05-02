from __future__ import annotations

import csv
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from DuelingDQN.dueling_dqn_agent import DuelingDQNAgent
from Simulation.mpc import MpcSolverGeneral
from Simulation.system_functions import PolymerCSTR
from TD3Agent.agent import TD3Agent
from systems.polymer import POLYMER_OBSERVER_POLES, POLYMER_SYSTEM_METADATA, get_polymer_notebook_defaults, load_polymer_system_data
from systems.polymer.data_io import canonical_baseline_path
from utils.combined_runner import run_combined_supervisor
from utils.helpers import apply_min_max, build_horizon_recipes
from utils.horizon_runner_dueling import run_dueling_dqn_mpc_horizon_supervisor
from utils.matrix_runner import run_matrix_multiplier_supervisor
from utils.notebook_setup import prepare_polymer_notebook_env
from utils.plotting import compare_mpc_rl_from_dirs, plot_combined_results, plot_horizon_results, plot_matrix_multiplier_results, plot_residual_results, plot_weight_multiplier_results
from utils.plotting_core import (
    _save_fig,
    _set_plot_style,
    create_output_dir,
    normalize_external_bundle,
    normalize_result_bundle,
    resolve_system_metadata,
    ysp_scaled_dev_to_phys,
)
from utils.residual_runner import run_residual_supervisor
from utils.rewards import make_reward_fn_relative_QR
from utils.state_features import get_rl_state_dim
from utils.weights_runner import run_weight_multiplier_supervisor


DEFAULT_SEEDS = [7, 11, 23, 37, 101]
DEFAULT_METHODS = ["horizon_dueling", "matrix", "weights", "residual", "combined"]
METHOD_LABELS = {
    "horizon_dueling": "Dueling DQN Horizon",
    "matrix": "TD3 Matrix",
    "weights": "TD3 Weights",
    "residual": "TD3 Residual",
    "combined": "Combined",
}
NOTEBOOK_SOURCES = {
    "horizon_dueling": "RL_assisted_MPC_horizons_dueling_unified.ipynb",
    "matrix": "RL_assisted_MPC_matrices_unified.ipynb",
    "weights": "RL_assisted_MPC_weights_unified.ipynb",
    "residual": "RL_assisted_MPC_residual_unified.ipynb",
    "combined": "RL_assisted_MPC_combined_unified.ipynb",
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    family: str
    label: str
    kind: str


METHOD_SPECS = {
    "horizon_dueling": MethodSpec("horizon_dueling", "horizon_dueling", METHOD_LABELS["horizon_dueling"], "discrete"),
    "matrix": MethodSpec("matrix", "matrix", METHOD_LABELS["matrix"], "continuous"),
    "weights": MethodSpec("weights", "weights", METHOD_LABELS["weights"], "continuous"),
    "residual": MethodSpec("residual", "residual", METHOD_LABELS["residual"], "continuous"),
    "combined": MethodSpec("combined", "combined", METHOD_LABELS["combined"], "combined"),
}


def default_study_config() -> dict[str, Any]:
    return {
        "methods": list(DEFAULT_METHODS),
        "seeds": list(DEFAULT_SEEDS),
        "run_mode": "disturb",
        "state_mode": "mismatch",
        "save_pdf": False,
        "style_profile": "paper",
        "figure_prefix": "polymer_five_seed_core_study",
        "results_dir_override": None,
        "data_dir_override": None,
        "reward_tail_episodes": 20,
    }


def set_global_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _seed_suffix(seed: int) -> str:
    return f"_seed{int(seed):03d}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def _last_episode_slices(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    n_steps = int(bundle["nFE"])
    episode_steps = int(min(max(1, bundle["time_in_sub_episodes"]), n_steps))
    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        bundle["n_inputs"],
    )
    y_last = np.asarray(bundle["y_line_full"][-(episode_steps + 1) :, :], float)
    u_last = np.asarray(bundle["u_step_full"][-episode_steps:, :], float)
    sp_last = np.asarray(y_sp_phys_full[-episode_steps:, :], float)
    delta_t = float(bundle["delta_t"])
    t_line = np.linspace(0.0, episode_steps * delta_t, episode_steps + 1)
    t_step = t_line[:-1]
    return {
        "episode_steps": episode_steps,
        "y_last": y_last,
        "u_last": u_last,
        "sp_last": sp_last,
        "t_line_last": t_line,
        "t_step_last": t_step,
    }


def _compute_metrics(bundle: dict[str, Any], reward_tail_episodes: int = 20) -> dict[str, float]:
    n_inputs = int(bundle["n_inputs"])
    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs,
    )
    y_phys = np.asarray(bundle["y_line_full"][1 : bundle["nFE"] + 1, :], float)
    u_phys = np.asarray(bundle["u_step_full"][: bundle["nFE"], :], float)
    y_sp_phys = np.asarray(y_sp_phys_full[: bundle["nFE"], :], float)
    err = y_phys - y_sp_phys
    abs_err = np.abs(err)
    rmse = np.sqrt(np.mean(np.square(err), axis=0))
    iae = np.sum(abs_err, axis=0) * float(bundle["delta_t"])
    max_abs_err = np.max(abs_err, axis=0)

    episode_steps = int(min(max(1, bundle["time_in_sub_episodes"]), bundle["nFE"]))
    tail_slice = slice(max(0, bundle["nFE"] - episode_steps), bundle["nFE"])
    tail_offset = np.mean(abs_err[tail_slice, :], axis=0)

    delta_u = np.zeros_like(u_phys)
    if u_phys.shape[0] > 1:
        delta_u[1:, :] = np.diff(u_phys, axis=0)
    mean_abs_du_per_input = np.mean(np.abs(delta_u), axis=0)
    mean_abs_du = float(np.mean(mean_abs_du_per_input))

    action_saturation = bundle.get("action_saturation_trace")
    if action_saturation is None:
        sat_fraction = float("nan")
    else:
        sat_fraction = float(np.mean(np.asarray(action_saturation, float)))

    avg_rewards = np.asarray(bundle.get("avg_rewards", []), float).reshape(-1)
    reward_tail = avg_rewards[-int(max(1, reward_tail_episodes)) :] if avg_rewards.size else np.asarray([], float)

    metrics = {
        "episodes": float(avg_rewards.size),
        "final_avg_reward": float(avg_rewards[-1]) if avg_rewards.size else float("nan"),
        "best_avg_reward": float(np.max(avg_rewards)) if avg_rewards.size else float("nan"),
        "tail_avg_reward": float(np.mean(reward_tail)) if reward_tail.size else float("nan"),
        "rmse_out1": float(rmse[0]),
        "rmse_out2": float(rmse[1]),
        "rmse_mean": float(np.mean(rmse)),
        "iae_out1": float(iae[0]),
        "iae_out2": float(iae[1]),
        "iae_mean": float(np.mean(iae)),
        "max_abs_err_out1": float(max_abs_err[0]),
        "max_abs_err_out2": float(max_abs_err[1]),
        "tail_offset_out1": float(tail_offset[0]),
        "tail_offset_out2": float(tail_offset[1]),
        "tail_offset_mean": float(np.mean(tail_offset)),
        "mean_abs_du_in1": float(mean_abs_du_per_input[0]),
        "mean_abs_du_in2": float(mean_abs_du_per_input[1]),
        "mean_abs_du": mean_abs_du,
        "action_saturation_mean": sat_fraction,
    }
    return metrics


def _extract_diagnostics(method_key: str, bundle: dict[str, Any]) -> dict[str, Any]:
    last = _last_episode_slices(bundle)
    diagnostics: dict[str, Any] = {
        "last_episode": last,
        "avg_rewards": np.asarray(bundle.get("avg_rewards", []), float).reshape(-1),
    }
    if method_key == "horizon_dueling":
        action_trace = bundle.get("horizon_action_trace")
        if action_trace is None:
            action_trace = bundle.get("action_trace")
        diagnostics["horizon_action_trace"] = None if action_trace is None else np.asarray(action_trace, int).reshape(-1)
    elif method_key == "matrix":
        diagnostics["alpha_log"] = None if bundle.get("alpha_log") is None else np.asarray(bundle["alpha_log"], float).reshape(-1)
        diagnostics["delta_log"] = None if bundle.get("delta_log") is None else np.asarray(bundle["delta_log"], float)
    elif method_key == "weights":
        diagnostics["weight_log"] = None if bundle.get("weight_log") is None else np.asarray(bundle["weight_log"], float)
    elif method_key == "residual":
        diagnostics["rho_log"] = None if bundle.get("rho_log") is None else np.asarray(bundle["rho_log"], float).reshape(-1)
        diagnostics["residual_exec_log"] = None if bundle.get("residual_exec_log") is None else np.asarray(bundle["residual_exec_log"], float)
    elif method_key == "combined":
        diagnostics["horizon_action_trace"] = None if bundle.get("horizon_action_trace") is None else np.asarray(bundle["horizon_action_trace"], int).reshape(-1)
        diagnostics["matrix_alpha_log"] = None if bundle.get("matrix_alpha_log") is None else np.asarray(bundle["matrix_alpha_log"], float).reshape(-1)
        diagnostics["weight_log"] = None if bundle.get("weight_log") is None else np.asarray(bundle["weight_log"], float)
        diagnostics["rho_log"] = None if bundle.get("rho_log") is None else np.asarray(bundle["rho_log"], float).reshape(-1)
    return diagnostics


def _load_baseline_bundle(baseline_path: Path, reference_bundle: dict[str, Any]) -> dict[str, Any]:
    with baseline_path.open("rb") as handle:
        baseline_raw = pickle.load(handle)
    return normalize_external_bundle(baseline_raw, reference_bundle)


def _build_common_system(nb: dict[str, Any], repo_root: Path, data_dir_override: str | None) -> dict[str, Any]:
    sys_cfg = nb["system_setup"]
    system_params = sys_cfg["system_params"].copy()
    system_design_params = sys_cfg["design_params"].copy()
    system_steady_state_inputs = sys_cfg["ss_inputs"].copy()
    delta_t = sys_cfg["delta_t_hours"]
    cstr_ss = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t)
    steady_states = {"ss_inputs": cstr_ss.ss_inputs, "y_ss": cstr_ss.y_ss}
    setpoint_y = sys_cfg["setpoint_range_phys"].copy()
    u_min = sys_cfg["input_bounds"]["u_min"].copy()
    u_max = sys_cfg["input_bounds"]["u_max"].copy()
    system_data = load_polymer_system_data(
        repo_root,
        steady_states=steady_states,
        setpoint_y=setpoint_y,
        u_min=u_min,
        u_max=u_max,
        n_inputs=2,
        data_override=data_dir_override,
    )
    a_aug = system_data["A_aug"]
    b_aug = system_data["B_aug"]
    c_aug = system_data["C_aug"]
    data_min = system_data["data_min"]
    data_max = system_data["data_max"]
    min_max_dict = system_data["min_max_dict"]
    n_inputs = int(b_aug.shape[1])
    y_sp_scenario_phys = sys_cfg["rl_setpoints_phys"].copy()
    y_sp_scenario = (
        apply_min_max(y_sp_scenario_phys, data_min[n_inputs:], data_max[n_inputs:])
        - apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    )
    reward_params, reward_fn = make_reward_fn_relative_QR(data_min, data_max, n_inputs, **nb["reward"])
    return {
        "system_cfg": sys_cfg,
        "system_params": system_params,
        "system_design_params": system_design_params,
        "system_steady_state_inputs": system_steady_state_inputs,
        "delta_t": delta_t,
        "steady_states": steady_states,
        "system_data": system_data,
        "A_aug": a_aug,
        "B_aug": b_aug,
        "C_aug": c_aug,
        "data_min": data_min,
        "data_max": data_max,
        "min_max_dict": min_max_dict,
        "n_inputs": n_inputs,
        "n_outputs": int(c_aug.shape[0]),
        "y_sp_scenario": y_sp_scenario,
        "reward_fn": reward_fn,
        "reward_params": reward_params,
    }


def _build_td3_agent(cfg: dict[str, Any], state_dim: int, action_dim: int, set_points_len: int, device: torch.device) -> TD3Agent:
    replay_recent_window = (
        int(cfg["replay_recent_window"])
        if cfg["replay_recent_window"] is not None
        else min(int(cfg["buffer_size"]), int(cfg["replay_recent_window_mult"]) * int(set_points_len))
    )
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "actor_hidden": list(cfg["actor_hidden"]),
        "critic_hidden": list(cfg["critic_hidden"]),
        "gamma": cfg["gamma"],
        "actor_lr": cfg["actor_lr"],
        "critic_lr": cfg["critic_lr"],
        "batch_size": cfg["batch_size"],
        "policy_delay": cfg["policy_delay"],
        "target_policy_smoothing_noise_std": cfg["target_policy_smoothing_noise_std"],
        "noise_clip": cfg["noise_clip"],
        "max_action": cfg["max_action"],
        "tau": cfg["tau"],
        "std_start": cfg["std_start"],
        "std_end": cfg["std_end"],
        "std_decay_rate": cfg["std_decay_rate"],
        "std_decay_mode": cfg["std_decay_mode"],
        "buffer_size": int(cfg["buffer_size"]),
        "replay_frac_per": float(cfg["replay_frac_per"]),
        "replay_frac_recent": float(cfg["replay_frac_recent"]),
        "replay_recent_window": replay_recent_window,
        "replay_alpha": float(cfg["replay_alpha"]),
        "replay_beta_start": float(cfg["replay_beta_start"]),
        "replay_beta_end": float(cfg["replay_beta_end"]),
        "replay_beta_steps": int(cfg["replay_beta_steps"]),
        "device": device,
        "actor_freeze": int(cfg["actor_freeze"]),
        "exploration_mode": cfg["exploration_mode"],
        "loss_type": cfg["loss_type"],
        "param_noise_resample_interval": int(cfg["param_noise_resample_interval"]),
        "n_step": int(cfg["n_step"]),
        "multistep_mode": cfg["multistep_mode"],
        "lambda_value": float(cfg["lambda_value"]),
    }
    for key in (
        "grad_clip_norm",
        "param_noise_std_start",
        "param_noise_std_end",
        "param_noise_decay_rate",
        "param_noise_decay_steps",
        "param_noise_decay_mode",
    ):
        if key in cfg:
            kwargs[key] = cfg[key]
    return TD3Agent(**kwargs)


def _build_dueling_agent(cfg: dict[str, Any], state_dim: int, action_dim: int, set_points_len: int, device: torch.device) -> DuelingDQNAgent:
    replay_recent_window = (
        int(cfg["replay_recent_window"])
        if cfg["replay_recent_window"] is not None
        else min(int(cfg["buffer_size"]), int(cfg["replay_recent_window_mult"]) * int(set_points_len))
    )
    return DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=list(cfg["hidden_layers"]),
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        buffer_size=int(cfg["buffer_size"]),
        replay_frac_per=float(cfg["replay_frac_per"]),
        replay_frac_recent=float(cfg["replay_frac_recent"]),
        replay_recent_window=replay_recent_window,
        replay_alpha=float(cfg["replay_alpha"]),
        replay_beta_start=float(cfg["replay_beta_start"]),
        replay_beta_end=float(cfg["replay_beta_end"]),
        replay_beta_steps=int(cfg["replay_beta_steps"]),
        n_step=int(cfg["n_step"]),
        multistep_mode=cfg["multistep_mode"],
        lambda_value=float(cfg["lambda_value"]),
        grad_clip_norm=cfg["grad_clip_norm"],
        double_dqn=cfg["double_dqn"],
        target_update=cfg["target_update"],
        tau=cfg["tau"],
        hard_update_interval=cfg["hard_update_interval"],
        activation=cfg["activation"],
        use_layer_norm=cfg["use_layer_norm"],
        dropout=cfg["dropout"],
        device=device,
        exploration_mode=cfg["exploration_mode"],
        loss_type=cfg["loss_type"],
        eps_start=cfg["eps_start"],
        eps_end=cfg["eps_end"],
        eps_decay_rate=cfg["eps_decay_rate"],
        eps_decay_steps=cfg.get("eps_decay_steps", 100_000),
        eps_decay_mode=cfg["eps_decay_mode"],
    )


def _build_combined_suffix(horizon_kind: str, use_rho: bool) -> str:
    rho_suffix = "rho" if use_rho else "no_rho"
    return "__".join(
        [
            f"h_{horizon_kind}_mismatch",
            "m_td3_mismatch",
            "w_td3_mismatch",
            f"r_td3_mismatch_{rho_suffix}",
        ]
    )


def _prepare_horizon_dueling_run(
    seed: int,
    repo_root: Path,
    result_dir: Path,
    data_dir_override: str | None,
    *,
    style_profile: str,
    save_pdf: bool,
) -> dict[str, Any]:
    nb = get_polymer_notebook_defaults("horizon_dueling")
    common = _build_common_system(nb, repo_root, data_dir_override)
    episode_cfg = nb["episode_defaults"]
    ctrl = nb["controller"]
    agent_cfg = deepcopy(nb["agent"])
    agent_cfg["seed"] = int(seed)
    run_mode = "disturb"
    state_mode = "mismatch"
    run_profile = nb["run_profiles"][run_mode]
    result_prefix = f"{run_profile['result_prefix']}{_seed_suffix(seed)}"
    compare_prefix = f"{run_profile['compare_prefix']}{_seed_suffix(seed)}"
    baseline_path = canonical_baseline_path(repo_root, run_mode, data_override=data_dir_override)
    n_tests = int(episode_cfg["n_tests"])
    set_points_len = int(episode_cfg["set_points_len"])
    warm_start = int(episode_cfg["warm_start"])
    test_cycle = list(episode_cfg["test_cycle"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = get_rl_state_dim(common["A_aug"].shape[0], common["n_outputs"], common["n_inputs"], state_mode)
    horizon_recipes = build_horizon_recipes(list(ctrl["predict_grid"]), list(ctrl["control_grid"]))
    agent = _build_dueling_agent(agent_cfg, state_dim, len(horizon_recipes), set_points_len, device)

    dueling_cfg = {
        "mode": run_mode,
        "state_mode": state_mode,
        "algorithm": "dueling_ddqn",
        "mismatch_clip": ctrl["mismatch_clip"],
        "innovation_scale_mode": ctrl["innovation_scale_mode"],
        "innovation_scale_ref": ctrl["innovation_scale_ref"],
        "tracking_scale_mode": ctrl["tracking_scale_mode"],
        "tracking_eta_tol": ctrl["tracking_eta_tol"],
        "tracking_scale_floor": ctrl["tracking_scale_floor"],
        "tracking_scale_floor_mode": ctrl["tracking_scale_floor_mode"],
        "base_state_norm_mode": ctrl["base_state_norm_mode"],
        "base_state_running_norm_clip": ctrl["base_state_running_norm_clip"],
        "base_state_running_norm_eps": ctrl["base_state_running_norm_eps"],
        "mismatch_feature_transform_mode": ctrl["mismatch_feature_transform_mode"],
        "mismatch_transform_tanh_scale": ctrl["mismatch_transform_tanh_scale"],
        "mismatch_transform_post_clip": ctrl["mismatch_transform_post_clip"],
        "observer_update_alignment": ctrl["observer_update_alignment"],
        "notebook_source": NOTEBOOK_SOURCES["horizon_dueling"],
        "seed": int(seed),
        "predict_h": int(ctrl["predict_h"]),
        "cont_h": int(ctrl["cont_h"]),
        "decision_interval": int(ctrl["decision_interval"]),
        "warm_start": warm_start,
        "test_cycle": test_cycle,
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "n_step": int(agent_cfg["n_step"]),
        "multistep_mode": agent_cfg["multistep_mode"],
        "lambda_value": float(agent_cfg["lambda_value"]),
        "use_shifted_mpc_warm_start": bool(ctrl["use_shifted_mpc_warm_start"]),
        "nominal_qi": float(ctrl["nominal_qi"]),
        "nominal_qs": float(ctrl["nominal_qs"]),
        "nominal_ha": float(ctrl["nominal_ha"]),
        "qi_change": float(ctrl["qi_change"]),
        "qs_change": float(ctrl["qs_change"]),
        "ha_change": float(ctrl["ha_change"]),
        "b_min": common["system_data"]["b_min"],
        "b_max": common["system_data"]["b_max"],
        "Q1_penalty": float(ctrl["Q1_penalty"]),
        "Q2_penalty": float(ctrl["Q2_penalty"]),
        "R1_penalty": float(ctrl["R1_penalty"]),
        "R2_penalty": float(ctrl["R2_penalty"]),
    }
    runtime_ctx = {
        "system": PolymerCSTR(common["system_params"], common["system_design_params"], common["system_steady_state_inputs"], common["delta_t"]),
        "y_sp_scenario": common["y_sp_scenario"],
        "steady_states": common["steady_states"],
        "min_max_dict": common["min_max_dict"],
        "agent": agent,
        "A_aug": common["A_aug"],
        "B_aug": common["B_aug"],
        "C_aug": common["C_aug"],
        "poles": POLYMER_OBSERVER_POLES.copy(),
        "data_min": common["data_min"],
        "data_max": common["data_max"],
        "horizon_recipes": horizon_recipes,
        "reward_fn": common["reward_fn"],
        "system_metadata": POLYMER_SYSTEM_METADATA,
        "reward_params": common["reward_params"],
    }
    result_bundle = run_dueling_dqn_mpc_horizon_supervisor(dueling_cfg=dueling_cfg, runtime_ctx=runtime_ctx)
    result_bundle["mpc_path_or_dir"] = baseline_path
    result_bundle["reward_params"] = common["reward_params"]
    result_bundle["run_profile"] = run_profile
    out_dir_rl = plot_horizon_results(
        result_bundle=result_bundle,
        plot_cfg={
            "directory": result_dir,
            "prefix_name": result_prefix,
            "start_episode": int(run_profile["plot_start_episode"]),
            "recipe_counts": True,
            "save_pdf": bool(save_pdf),
            "style_profile": style_profile,
        },
    )
    out_dir_cmp = compare_mpc_rl_from_dirs(
        rl_dir=out_dir_rl,
        mpc_path_or_dir=baseline_path,
        reward_fn=common["reward_fn"],
        directory=result_dir,
        prefix_name=compare_prefix,
        compare_mode=run_profile["compare_mode"],
        start_episode=int(run_profile["compare_start_episode"]),
        save_pdf=bool(save_pdf),
        style_profile=style_profile,
    )
    return {
        "method_key": "horizon_dueling",
        "method_label": METHOD_LABELS["horizon_dueling"],
        "seed": int(seed),
        "bundle": normalize_result_bundle(result_bundle),
        "bundle_path": str(Path(out_dir_rl) / "input_data.pkl"),
        "result_dir": str(out_dir_rl),
        "compare_dir": str(out_dir_cmp),
        "baseline_path": str(baseline_path),
        "result_prefix": result_prefix,
        "compare_prefix": compare_prefix,
        "config_snapshot": _jsonable(result_bundle.get("config_snapshot")),
        "style_profile": style_profile,
        "save_pdf": bool(save_pdf),
    }


def _prepare_continuous_run(
    method_key: str,
    seed: int,
    repo_root: Path,
    result_dir: Path,
    data_dir_override: str | None,
    *,
    style_profile: str,
    save_pdf: bool,
) -> dict[str, Any]:
    nb = get_polymer_notebook_defaults(method_key)
    common = _build_common_system(nb, repo_root, data_dir_override)
    episode_cfg = nb["episode_defaults"]
    ctrl = nb["controller"]
    td3_cfg = deepcopy(nb["td3_agent"])
    run_mode = "disturb"
    state_mode = "mismatch"
    run_profile = nb["run_profiles"][("td3", run_mode)]
    result_prefix = f"{run_profile['result_prefix']}{_seed_suffix(seed)}"
    compare_prefix = f"{run_profile['compare_prefix']}{_seed_suffix(seed)}"
    baseline_path = canonical_baseline_path(repo_root, run_mode, data_override=data_dir_override)
    n_tests = int(episode_cfg["n_tests"])
    set_points_len = int(episode_cfg["set_points_len"])
    warm_start = int(episode_cfg["warm_start"])
    test_cycle = list(episode_cfg["test_cycle"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method_key == "matrix":
        action_dim = 1 + common["n_inputs"]
    elif method_key == "weights":
        action_dim = 4
    elif method_key == "residual":
        action_dim = common["n_inputs"]
    else:
        raise ValueError(f"Unsupported continuous method: {method_key}")
    state_dim = get_rl_state_dim(
        common["A_aug"].shape[0],
        common["n_outputs"],
        common["n_inputs"],
        state_mode,
        append_rho_to_state=(method_key == "residual" and state_mode == "mismatch" and bool(nb.get("append_rho_to_state", False))),
    )
    agent = _build_td3_agent(td3_cfg, state_dim, action_dim, set_points_len, device)
    mpc_obj = MpcSolverGeneral(
        common["A_aug"],
        common["B_aug"],
        common["C_aug"],
        Q_out=np.array([ctrl["Q1_penalty"], ctrl["Q2_penalty"]], float),
        R_in=np.array([ctrl["R1_penalty"], ctrl["R2_penalty"]], float),
        NP=int(ctrl["predict_h"]),
        NC=int(ctrl["cont_h"]),
    )
    base_cfg = {
        "algorithm": "td3",
        "agent_kind": "td3",
        "run_mode": run_mode,
        "state_mode": state_mode,
        "notebook_source": NOTEBOOK_SOURCES[method_key],
        "seed": int(seed),
        "mismatch_clip": ctrl["mismatch_clip"],
        "innovation_scale_mode": ctrl["innovation_scale_mode"],
        "innovation_scale_ref": ctrl["innovation_scale_ref"],
        "tracking_scale_mode": ctrl["tracking_scale_mode"],
        "tracking_eta_tol": ctrl["tracking_eta_tol"],
        "tracking_scale_floor": ctrl["tracking_scale_floor"],
        "tracking_scale_floor_mode": ctrl["tracking_scale_floor_mode"],
        "base_state_norm_mode": ctrl["base_state_norm_mode"],
        "base_state_running_norm_clip": ctrl["base_state_running_norm_clip"],
        "base_state_running_norm_eps": ctrl["base_state_running_norm_eps"],
        "mismatch_feature_transform_mode": ctrl["mismatch_feature_transform_mode"],
        "mismatch_transform_tanh_scale": ctrl["mismatch_transform_tanh_scale"],
        "mismatch_transform_post_clip": ctrl["mismatch_transform_post_clip"],
        "observer_update_alignment": ctrl["observer_update_alignment"],
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "n_step": int(td3_cfg["n_step"]),
        "multistep_mode": td3_cfg["multistep_mode"],
        "lambda_value": float(td3_cfg["lambda_value"]),
        "warm_start": warm_start,
        "post_warm_start_action_freeze_subepisodes": int(nb["post_warm_start_action_freeze_subepisodes"]),
        "post_warm_start_actor_freeze_subepisodes": int(nb["post_warm_start_actor_freeze_subepisodes"]),
        "test_cycle": test_cycle,
        "predict_h": int(ctrl["predict_h"]),
        "cont_h": int(ctrl["cont_h"]),
        "use_shifted_mpc_warm_start": bool(ctrl["use_shifted_mpc_warm_start"]),
        "nominal_qi": float(ctrl["nominal_qi"]),
        "nominal_qs": float(ctrl["nominal_qs"]),
        "nominal_ha": float(ctrl["nominal_ha"]),
        "qi_change": float(ctrl["qi_change"]),
        "qs_change": float(ctrl["qs_change"]),
        "ha_change": float(ctrl["ha_change"]),
        "Q1_penalty": float(ctrl["Q1_penalty"]),
        "Q2_penalty": float(ctrl["Q2_penalty"]),
        "R1_penalty": float(ctrl["R1_penalty"]),
        "R2_penalty": float(ctrl["R2_penalty"]),
        "b_min": common["system_data"]["b_min"],
        "b_max": common["system_data"]["b_max"],
    }
    if method_key == "matrix":
        run_cfg = dict(base_cfg)
        run_cfg["low_coef"] = ctrl["low_coef"].copy()
        run_cfg["high_coef"] = ctrl["high_coef"].copy()
        run_cfg["release_protected_advisory_caps"] = dict(ctrl["release_protected_advisory_caps"])
        run_cfg["behavioral_cloning"] = dict(nb["behavioral_cloning"])
        run_cfg["mpc_acceptance_fallback"] = dict(ctrl["mpc_acceptance_fallback"])
        run_cfg["mpc_dual_cost_shadow"] = dict(ctrl["mpc_dual_cost_shadow"])
        run_cfg["mpc_usefulness_gate"] = dict(ctrl["mpc_usefulness_gate"])
        runtime_ctx = {
            "system": PolymerCSTR(common["system_params"], common["system_design_params"], common["system_steady_state_inputs"], common["delta_t"]),
            "agent": agent,
            "MPC_obj": mpc_obj,
            "steady_states": common["steady_states"],
            "min_max_dict": common["min_max_dict"],
            "data_min": common["data_min"],
            "data_max": common["data_max"],
            "A_aug": common["A_aug"],
            "B_aug": common["B_aug"],
            "C_aug": common["C_aug"],
            "poles": POLYMER_OBSERVER_POLES.copy(),
            "y_sp_scenario": common["y_sp_scenario"],
            "reward_fn": common["reward_fn"],
            "system_metadata": POLYMER_SYSTEM_METADATA,
            "reward_params": common["reward_params"],
        }
        result_bundle = run_matrix_multiplier_supervisor(matrix_cfg=run_cfg, runtime_ctx=runtime_ctx)
        result_bundle["mpc_path_or_dir"] = baseline_path
        out_dir_rl = plot_matrix_multiplier_results(
            result_bundle=result_bundle,
            plot_cfg={"directory": result_dir, "prefix_name": result_prefix, "start_episode": int(run_profile["plot_start_episode"]), "save_pdf": bool(save_pdf), "style_profile": style_profile},
        )
    elif method_key == "weights":
        run_cfg = dict(base_cfg)
        run_cfg["low_coef"] = np.asarray(ctrl["low_coef"], float).copy()
        run_cfg["high_coef"] = np.asarray(ctrl["high_coef"], float).copy()
        runtime_ctx = {
            "system": PolymerCSTR(common["system_params"], common["system_design_params"], common["system_steady_state_inputs"], common["delta_t"]),
            "agent": agent,
            "MPC_obj": mpc_obj,
            "steady_states": common["steady_states"],
            "min_max_dict": common["min_max_dict"],
            "data_min": common["data_min"],
            "data_max": common["data_max"],
            "A_aug": common["A_aug"],
            "B_aug": common["B_aug"],
            "C_aug": common["C_aug"],
            "poles": POLYMER_OBSERVER_POLES.copy(),
            "y_sp_scenario": common["y_sp_scenario"],
            "reward_fn": common["reward_fn"],
            "system_metadata": POLYMER_SYSTEM_METADATA,
            "reward_params": common["reward_params"],
        }
        result_bundle = run_weight_multiplier_supervisor(weight_cfg=run_cfg, runtime_ctx=runtime_ctx)
        result_bundle["mpc_path_or_dir"] = baseline_path
        out_dir_rl = plot_weight_multiplier_results(
            result_bundle=result_bundle,
            plot_cfg={"directory": result_dir, "prefix_name": result_prefix, "start_episode": int(run_profile["plot_start_episode"]), "save_pdf": bool(save_pdf), "style_profile": style_profile},
        )
    else:
        run_cfg = dict(base_cfg)
        run_cfg.update(
            {
                "authority_use_rho": bool(nb["authority_use_rho"]),
                "use_rho_authority": bool(nb["use_rho_authority"]),
                "append_rho_to_state": bool(nb["append_rho_to_state"]),
                "authority_beta_res": np.asarray(nb["authority_beta_res"], float).copy(),
                "authority_du0_res": np.asarray(nb["authority_du0_res"], float).copy(),
                "authority_eta_tol": float(nb["authority_eta_tol"]),
                "authority_rho_floor": float(nb["authority_rho_floor"]),
                "authority_rho_power": float(nb["authority_rho_power"]),
                "rho_mapping_mode": nb["rho_mapping_mode"],
                "authority_rho_k": float(nb["authority_rho_k"]),
                "residual_zero_deadband_enabled": bool(nb["residual_zero_deadband_enabled"]),
                "residual_zero_tracking_raw_threshold": float(nb["residual_zero_tracking_raw_threshold"]),
                "residual_zero_innovation_raw_threshold": float(nb["residual_zero_innovation_raw_threshold"]),
                "behavioral_cloning": dict(nb["behavioral_cloning"]),
                "low_coef": np.asarray(ctrl["low_coef"], float).copy(),
                "high_coef": np.asarray(ctrl["high_coef"], float).copy(),
            }
        )
        runtime_ctx = {
            "system": PolymerCSTR(common["system_params"], common["system_design_params"], common["system_steady_state_inputs"], common["delta_t"]),
            "agent": agent,
            "MPC_obj": mpc_obj,
            "steady_states": common["steady_states"],
            "min_max_dict": common["min_max_dict"],
            "data_min": common["data_min"],
            "data_max": common["data_max"],
            "A_aug": common["A_aug"],
            "B_aug": common["B_aug"],
            "C_aug": common["C_aug"],
            "poles": POLYMER_OBSERVER_POLES.copy(),
            "y_sp_scenario": common["y_sp_scenario"],
            "reward_fn": common["reward_fn"],
            "system_metadata": POLYMER_SYSTEM_METADATA,
            "reward_params": common["reward_params"],
        }
        result_bundle = run_residual_supervisor(residual_cfg=run_cfg, runtime_ctx=runtime_ctx)
        result_bundle["mpc_path_or_dir"] = baseline_path
        out_dir_rl = plot_residual_results(
            result_bundle=result_bundle,
            plot_cfg={"directory": result_dir, "prefix_name": result_prefix, "start_episode": int(run_profile["plot_start_episode"]), "save_pdf": bool(save_pdf), "style_profile": style_profile},
        )
    out_dir_cmp = compare_mpc_rl_from_dirs(
        rl_dir=out_dir_rl,
        mpc_path_or_dir=baseline_path,
        reward_fn=common["reward_fn"],
        directory=result_dir,
        prefix_name=compare_prefix,
        compare_mode=run_profile["compare_mode"],
        start_episode=int(run_profile["compare_start_episode"]),
        save_pdf=bool(save_pdf),
        style_profile=style_profile,
    )
    return {
        "method_key": method_key,
        "method_label": METHOD_LABELS[method_key],
        "seed": int(seed),
        "bundle": normalize_result_bundle(result_bundle),
        "bundle_path": str(Path(out_dir_rl) / "input_data.pkl"),
        "result_dir": str(out_dir_rl),
        "compare_dir": str(out_dir_cmp),
        "baseline_path": str(baseline_path),
        "result_prefix": result_prefix,
        "compare_prefix": compare_prefix,
        "config_snapshot": _jsonable(result_bundle.get("config_snapshot")),
        "style_profile": style_profile,
        "save_pdf": bool(save_pdf),
    }


def _prepare_combined_run(
    seed: int,
    repo_root: Path,
    result_dir: Path,
    data_dir_override: str | None,
    *,
    style_profile: str,
    save_pdf: bool,
) -> dict[str, Any]:
    nb = get_polymer_notebook_defaults("combined")
    nb["horizon_agent_kind"] = "dueling_dqn"
    common = _build_common_system(nb, repo_root, data_dir_override)
    episode_cfg = nb["episode_defaults"]
    ctrl = nb["controller"]
    horizon_cfg = deepcopy(nb["horizon_dueling_agent"])
    matrix_cfg_td3 = deepcopy(nb["matrix_td3_agent"])
    weights_cfg_td3 = deepcopy(nb["weights_td3_agent"])
    residual_cfg_td3 = deepcopy(nb["residual_td3_agent"])
    run_mode = "disturb"
    run_profile = nb["run_profiles"][run_mode]
    active_suffix = _build_combined_suffix("dueling_dqn", bool(nb["use_rho_authority"]))
    result_prefix = f"{run_profile['result_prefix_template'].format(suffix=active_suffix)}{_seed_suffix(seed)}"
    compare_prefix = f"{run_profile['compare_prefix_template'].format(suffix=active_suffix)}{_seed_suffix(seed)}"
    baseline_path = canonical_baseline_path(repo_root, run_mode, data_override=data_dir_override)
    n_tests = int(episode_cfg["n_tests"])
    set_points_len = int(episode_cfg["set_points_len"])
    warm_start = int(episode_cfg["warm_start"])
    test_cycle = list(episode_cfg["test_cycle"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon_state_mode = str(nb["horizon_state_mode"]).lower()
    matrix_state_mode = str(nb["matrix_state_mode"]).lower()
    weights_state_mode = str(nb["weights_state_mode"]).lower()
    residual_state_mode = str(nb["residual_state_mode"]).lower()
    horizon_state_dim = get_rl_state_dim(common["A_aug"].shape[0], common["n_outputs"], common["n_inputs"], horizon_state_mode)
    matrix_state_dim = get_rl_state_dim(common["A_aug"].shape[0], common["n_outputs"], common["n_inputs"], matrix_state_mode)
    weights_state_dim = get_rl_state_dim(common["A_aug"].shape[0], common["n_outputs"], common["n_inputs"], weights_state_mode)
    residual_state_dim = get_rl_state_dim(
        common["A_aug"].shape[0],
        common["n_outputs"],
        common["n_inputs"],
        residual_state_mode,
        append_rho_to_state=(residual_state_mode == "mismatch" and bool(nb["append_rho_to_state"])),
    )
    horizon_recipes = build_horizon_recipes(list(ctrl["predict_grid"]), list(ctrl["control_grid"]))
    horizon_agent = _build_dueling_agent(horizon_cfg, horizon_state_dim, len(horizon_recipes), set_points_len, device)
    matrix_agent = _build_td3_agent(matrix_cfg_td3, matrix_state_dim, 1 + common["n_inputs"], set_points_len, device)
    weights_agent = _build_td3_agent(weights_cfg_td3, weights_state_dim, 4, set_points_len, device)
    residual_agent = _build_td3_agent(residual_cfg_td3, residual_state_dim, common["n_inputs"], set_points_len, device)
    combined_cfg = {
        "run_mode": run_mode,
        "horizon_agent_kind": "dueling_dqn",
        "notebook_source": NOTEBOOK_SOURCES["combined"],
        "seed": int(seed),
        "decision_interval": int(ctrl["decision_interval"]),
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "warm_start": warm_start,
        "td3_post_warm_start_action_freeze_subepisodes": int(nb["td3_post_warm_start_action_freeze_subepisodes"]),
        "td3_post_warm_start_actor_freeze_subepisodes": int(nb["td3_post_warm_start_actor_freeze_subepisodes"]),
        "test_cycle": test_cycle,
        "predict_h": int(ctrl["predict_h"]),
        "cont_h": int(ctrl["cont_h"]),
        "use_shifted_mpc_warm_start": bool(ctrl["use_shifted_mpc_warm_start"]),
        "Q1_penalty": float(ctrl["Q1_penalty"]),
        "Q2_penalty": float(ctrl["Q2_penalty"]),
        "R1_penalty": float(ctrl["R1_penalty"]),
        "R2_penalty": float(ctrl["R2_penalty"]),
        "nominal_qi": float(ctrl["nominal_qi"]),
        "nominal_qs": float(ctrl["nominal_qs"]),
        "nominal_ha": float(ctrl["nominal_ha"]),
        "qi_change": float(ctrl["qi_change"]),
        "qs_change": float(ctrl["qs_change"]),
        "ha_change": float(ctrl["ha_change"]),
        "b_min": common["system_data"]["b_min"],
        "b_max": common["system_data"]["b_max"],
        "horizon_cfg": {
            "enabled": True,
            "agent_kind": "dueling_dqn",
            "seed": int(seed),
            "state_mode": horizon_state_mode,
            "mismatch_clip": ctrl["mismatch_clip"],
            "innovation_scale_mode": ctrl["innovation_scale_mode"],
            "innovation_scale_ref": ctrl["innovation_scale_ref"],
            "tracking_scale_mode": ctrl["tracking_scale_mode"],
            "tracking_eta_tol": ctrl["tracking_eta_tol"],
            "tracking_scale_floor": ctrl["tracking_scale_floor"],
            "tracking_scale_floor_mode": ctrl["tracking_scale_floor_mode"],
            "base_state_norm_mode": ctrl["base_state_norm_mode"],
            "base_state_running_norm_clip": ctrl["base_state_running_norm_clip"],
            "base_state_running_norm_eps": ctrl["base_state_running_norm_eps"],
            "mismatch_feature_transform_mode": ctrl["mismatch_feature_transform_mode"],
            "mismatch_transform_tanh_scale": ctrl["mismatch_transform_tanh_scale"],
            "mismatch_transform_post_clip": ctrl["mismatch_transform_post_clip"],
            "observer_update_alignment": ctrl["observer_update_alignment"],
            "horizon_recipes": horizon_recipes,
            "default_horizons": (int(ctrl["predict_h"]), int(ctrl["cont_h"])),
        },
        "matrix_cfg": {
            "enabled": True,
            "agent_kind": "td3",
            "seed": int(seed),
            "state_mode": matrix_state_mode,
            "mismatch_clip": ctrl["mismatch_clip"],
            "innovation_scale_mode": ctrl["innovation_scale_mode"],
            "innovation_scale_ref": ctrl["innovation_scale_ref"],
            "tracking_scale_mode": ctrl["tracking_scale_mode"],
            "tracking_eta_tol": ctrl["tracking_eta_tol"],
            "tracking_scale_floor": ctrl["tracking_scale_floor"],
            "tracking_scale_floor_mode": ctrl["tracking_scale_floor_mode"],
            "base_state_norm_mode": ctrl["base_state_norm_mode"],
            "base_state_running_norm_clip": ctrl["base_state_running_norm_clip"],
            "base_state_running_norm_eps": ctrl["base_state_running_norm_eps"],
            "mismatch_feature_transform_mode": ctrl["mismatch_feature_transform_mode"],
            "mismatch_transform_tanh_scale": ctrl["mismatch_transform_tanh_scale"],
            "mismatch_transform_post_clip": ctrl["mismatch_transform_post_clip"],
            "observer_update_alignment": ctrl["observer_update_alignment"],
            "low_coef": np.asarray(ctrl["model_low"], float).copy(),
            "high_coef": np.asarray(ctrl["model_high"], float).copy(),
            "release_protected_advisory_caps": dict(ctrl["release_protected_advisory_caps"]),
        },
        "weight_cfg": {
            "enabled": True,
            "agent_kind": "td3",
            "seed": int(seed),
            "state_mode": weights_state_mode,
            "mismatch_clip": ctrl["mismatch_clip"],
            "innovation_scale_mode": ctrl["innovation_scale_mode"],
            "innovation_scale_ref": ctrl["innovation_scale_ref"],
            "tracking_scale_mode": ctrl["tracking_scale_mode"],
            "tracking_eta_tol": ctrl["tracking_eta_tol"],
            "tracking_scale_floor": ctrl["tracking_scale_floor"],
            "tracking_scale_floor_mode": ctrl["tracking_scale_floor_mode"],
            "base_state_norm_mode": ctrl["base_state_norm_mode"],
            "base_state_running_norm_clip": ctrl["base_state_running_norm_clip"],
            "base_state_running_norm_eps": ctrl["base_state_running_norm_eps"],
            "mismatch_feature_transform_mode": ctrl["mismatch_feature_transform_mode"],
            "mismatch_transform_tanh_scale": ctrl["mismatch_transform_tanh_scale"],
            "mismatch_transform_post_clip": ctrl["mismatch_transform_post_clip"],
            "observer_update_alignment": ctrl["observer_update_alignment"],
            "low_coef": np.asarray(ctrl["weights_low"], float).copy(),
            "high_coef": np.asarray(ctrl["weights_high"], float).copy(),
        },
        "residual_cfg": {
            "enabled": True,
            "agent_kind": "td3",
            "seed": int(seed),
            "state_mode": residual_state_mode,
            "mismatch_clip": ctrl["mismatch_clip"],
            "innovation_scale_mode": ctrl["innovation_scale_mode"],
            "innovation_scale_ref": ctrl["innovation_scale_ref"],
            "tracking_scale_mode": ctrl["tracking_scale_mode"],
            "tracking_eta_tol": ctrl["tracking_eta_tol"],
            "tracking_scale_floor": ctrl["tracking_scale_floor"],
            "tracking_scale_floor_mode": ctrl["tracking_scale_floor_mode"],
            "base_state_norm_mode": ctrl["base_state_norm_mode"],
            "base_state_running_norm_clip": ctrl["base_state_running_norm_clip"],
            "base_state_running_norm_eps": ctrl["base_state_running_norm_eps"],
            "mismatch_feature_transform_mode": ctrl["mismatch_feature_transform_mode"],
            "mismatch_transform_tanh_scale": ctrl["mismatch_transform_tanh_scale"],
            "mismatch_transform_post_clip": ctrl["mismatch_transform_post_clip"],
            "observer_update_alignment": ctrl["observer_update_alignment"],
            "authority_use_rho": bool(nb["use_rho_authority"]),
            "use_rho_authority": bool(nb["use_rho_authority"]),
            "append_rho_to_state": bool(nb["append_rho_to_state"]),
            "authority_beta_res": np.asarray(nb["authority_beta_res"], float).copy(),
            "authority_du0_res": np.asarray(nb["authority_du0_res"], float).copy(),
            "authority_eta_tol": float(nb["authority_eta_tol"]),
            "authority_rho_floor": float(nb["authority_rho_floor"]),
            "authority_rho_power": float(nb["authority_rho_power"]),
            "rho_mapping_mode": nb["rho_mapping_mode"],
            "authority_rho_k": float(nb["authority_rho_k"]),
            "residual_zero_deadband_enabled": bool(nb["residual_zero_deadband_enabled"]),
            "residual_zero_tracking_raw_threshold": float(nb["residual_zero_tracking_raw_threshold"]),
            "residual_zero_innovation_raw_threshold": float(nb["residual_zero_innovation_raw_threshold"]),
            "low_coef": np.asarray(ctrl["residual_low"], float).copy(),
            "high_coef": np.asarray(ctrl["residual_high"], float).copy(),
        },
    }
    runtime_ctx = {
        "system": PolymerCSTR(common["system_params"], common["system_design_params"], common["system_steady_state_inputs"], common["delta_t"]),
        "agents": {
            "horizon": horizon_agent,
            "matrix": matrix_agent,
            "weights": weights_agent,
            "residual": residual_agent,
        },
        "steady_states": common["steady_states"],
        "min_max_dict": common["min_max_dict"],
        "data_min": common["data_min"],
        "data_max": common["data_max"],
        "A_aug": common["A_aug"],
        "B_aug": common["B_aug"],
        "C_aug": common["C_aug"],
        "poles": POLYMER_OBSERVER_POLES.copy(),
        "y_sp_scenario": common["y_sp_scenario"],
        "reward_fn": common["reward_fn"],
        "system_metadata": POLYMER_SYSTEM_METADATA,
        "reward_params": common["reward_params"],
    }
    result_bundle = run_combined_supervisor(combined_cfg=combined_cfg, runtime_ctx=runtime_ctx)
    result_bundle["mpc_path_or_dir"] = baseline_path
    out_dir_rl = plot_combined_results(
        result_bundle=result_bundle,
        plot_cfg={
            "directory": result_dir,
            "prefix_name": result_prefix,
            "start_episode": int(run_profile["plot_start_episode"]),
            "save_pdf": bool(save_pdf),
            "style_profile": style_profile,
            "system_metadata": POLYMER_SYSTEM_METADATA,
            "include_baseline_compare": False,
        },
    )
    out_dir_cmp = compare_mpc_rl_from_dirs(
        rl_dir=out_dir_rl,
        mpc_path_or_dir=baseline_path,
        reward_fn=common["reward_fn"],
        directory=result_dir,
        prefix_name=compare_prefix,
        compare_mode=run_profile["compare_mode"],
        start_episode=int(run_profile["compare_start_episode"]),
        save_pdf=bool(save_pdf),
        style_profile=style_profile,
    )
    return {
        "method_key": "combined",
        "method_label": METHOD_LABELS["combined"],
        "seed": int(seed),
        "bundle": normalize_result_bundle(result_bundle),
        "bundle_path": str(Path(out_dir_rl) / "input_data.pkl"),
        "result_dir": str(out_dir_rl),
        "compare_dir": str(out_dir_cmp),
        "baseline_path": str(baseline_path),
        "result_prefix": result_prefix,
        "compare_prefix": compare_prefix,
        "config_snapshot": _jsonable(result_bundle.get("config_snapshot")),
        "style_profile": style_profile,
        "save_pdf": bool(save_pdf),
    }


def _run_method(
    method_key: str,
    seed: int,
    repo_root: Path,
    result_dir: Path,
    data_dir_override: str | None,
    *,
    style_profile: str,
    save_pdf: bool,
) -> dict[str, Any]:
    set_global_seeds(seed)
    if method_key == "horizon_dueling":
        return _prepare_horizon_dueling_run(seed, repo_root, result_dir, data_dir_override, style_profile=style_profile, save_pdf=save_pdf)
    if method_key in {"matrix", "weights", "residual"}:
        return _prepare_continuous_run(method_key, seed, repo_root, result_dir, data_dir_override, style_profile=style_profile, save_pdf=save_pdf)
    if method_key == "combined":
        return _prepare_combined_run(seed, repo_root, result_dir, data_dir_override, style_profile=style_profile, save_pdf=save_pdf)
    raise KeyError(f"Unsupported method: {method_key}")


def _build_run_record(raw_run: dict[str, Any], reward_tail_episodes: int) -> dict[str, Any]:
    bundle = raw_run["bundle"]
    record = {k: v for k, v in raw_run.items() if k != "bundle"}
    record["algorithm"] = bundle.get("algorithm")
    record["method_family"] = bundle.get("method_family")
    record["run_mode"] = bundle.get("run_mode")
    record["state_mode"] = bundle.get("state_mode")
    record["metrics"] = _compute_metrics(bundle, reward_tail_episodes=reward_tail_episodes)
    record["diagnostics"] = _extract_diagnostics(raw_run["method_key"], bundle)
    record["nFE"] = int(bundle["nFE"])
    record["time_in_sub_episodes"] = int(bundle["time_in_sub_episodes"])
    return record


def _append_baseline_deltas(run_records: list[dict[str, Any]], baseline_metrics: dict[str, float]) -> None:
    for record in run_records:
        metrics = record["metrics"]
        metrics["delta_rmse_mean_vs_mpc"] = metrics["rmse_mean"] - baseline_metrics["rmse_mean"]
        metrics["delta_iae_mean_vs_mpc"] = metrics["iae_mean"] - baseline_metrics["iae_mean"]
        metrics["delta_mean_abs_du_vs_mpc"] = metrics["mean_abs_du"] - baseline_metrics["mean_abs_du"]
        metrics["delta_tail_offset_mean_vs_mpc"] = metrics["tail_offset_mean"] - baseline_metrics["tail_offset_mean"]


def _aggregate_method_records(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[record["method_key"]].append(record)
    summary_rows: list[dict[str, Any]] = []
    for method_key in DEFAULT_METHODS:
        rows = grouped.get(method_key, [])
        if not rows:
            continue
        metrics_keys = sorted(rows[0]["metrics"].keys())
        row: dict[str, Any] = {
            "row_type": "method_summary",
            "method_key": method_key,
            "method_label": METHOD_LABELS[method_key],
            "n_runs": len(rows),
        }
        for key in metrics_keys:
            mean_val, std_val = _mean_std([float(r["metrics"].get(key, float("nan"))) for r in rows])
            row[f"{key}_mean"] = mean_val
            row[f"{key}_std"] = std_val
        row["bundle_paths"] = ";".join(str(r["bundle_path"]) for r in rows)
        summary_rows.append(row)
    return summary_rows


def _make_flat_rows(run_records: list[dict[str, Any]], method_summary_rows: list[dict[str, Any]], baseline_metrics: dict[str, float], baseline_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in run_records:
        row = {
            "row_type": "run",
            "method_key": record["method_key"],
            "method_label": record["method_label"],
            "seed": record["seed"],
            "bundle_path": record["bundle_path"],
            "result_dir": record["result_dir"],
            "compare_dir": record["compare_dir"],
            "baseline_path": str(baseline_path),
        }
        row.update(record["metrics"])
        rows.append(row)
    rows.extend(method_summary_rows)
    rows.append({"row_type": "baseline", "method_key": "mpc", "method_label": "Baseline MPC", "bundle_path": str(baseline_path), **baseline_metrics})
    return rows


def _representative_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    finals = np.asarray([r["metrics"]["final_avg_reward"] for r in records], float)
    target = float(np.nanmedian(finals))
    idx = int(np.nanargmin(np.abs(finals - target)))
    return records[idx]


def _build_method_color_map(method_keys: list[str]) -> dict[str, str]:
    palette = ["#0B3954", "#C81D25", "#2D6A4F", "#6A4C93", "#F4A259", "#3A86FF", "#7D8597"]
    return {key: palette[idx % len(palette)] for idx, key in enumerate(method_keys)}


def _plot_study_design_figure(out_dir: Path, methods: list[str], seeds: list[int], baseline_path: Path, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 2.8 + 0.35 * len(methods)))
    ax.axis("off")
    rows = []
    for method_key in methods:
        if method_key == "horizon_dueling":
            detail = "dueling_dqn"
        elif method_key == "combined":
            detail = "dueling_dqn + td3/td3/td3"
        else:
            detail = "td3"
        rows.append([METHOD_LABELS[method_key], detail, "disturb", "mismatch", ", ".join(str(s) for s in seeds)])
    table = ax.table(
        cellText=rows,
        colLabels=["Method", "Agent policy", "Run mode", "State mode", "Seeds"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    ax.set_title(f"Polymer five-seed core study\nBaseline reference: {baseline_path.name}", fontweight="bold")
    _save_fig(fig, os.fspath(out_dir), "fig_study_design_table", save_pdf=save_pdf)


def _plot_reward_summary(run_records: list[dict[str, Any]], out_dir: Path, save_pdf: bool) -> None:
    method_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in run_records:
        method_groups[record["method_key"]].append(np.asarray(record["diagnostics"]["avg_rewards"], float))
    colors = _build_method_color_map(list(method_groups.keys()))
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for method_key, curves in method_groups.items():
        min_len = min(len(curve) for curve in curves if len(curve))
        if min_len <= 0:
            continue
        stacked = np.stack([curve[:min_len] for curve in curves], axis=0)
        x = np.arange(1, min_len + 1)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        color = colors[method_key]
        ax.plot(x, mean, label=METHOD_LABELS[method_key], color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.set_title("Mean +/- std reward curves across five seeds", fontweight="bold")
    ax.legend(loc="best")
    _save_fig(fig, os.fspath(out_dir), "fig_reward_mean_std_by_method", save_pdf=save_pdf)


def _plot_final_reward_boxplot(run_records: list[dict[str, Any]], out_dir: Path, save_pdf: bool) -> None:
    methods = DEFAULT_METHODS
    data = [[record["metrics"]["final_avg_reward"] for record in run_records if record["method_key"] == method] for method in methods]
    labels = [METHOD_LABELS[method] for method in methods]
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Final average reward")
    ax.set_title("Final average reward distribution across seeds", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    _save_fig(fig, os.fspath(out_dir), "fig_final_reward_boxplot", save_pdf=save_pdf)


def _plot_last_episode_outputs(run_records: list[dict[str, Any]], baseline_bundle: dict[str, Any], out_dir: Path, save_pdf: bool) -> None:
    methods = DEFAULT_METHODS
    colors = _build_method_color_map(methods)
    baseline_last = _last_episode_slices(baseline_bundle)
    metadata = resolve_system_metadata(bundle=baseline_bundle, plot_cfg={}, n_outputs=2, n_inputs=2)
    fig, axs = plt.subplots(2, 1, figsize=(10.0, 7.2), sharex=True)
    for output_idx, ax in enumerate(axs):
        ax.plot(baseline_last["t_line_last"], baseline_last["y_last"][:, output_idx], color="black", linestyle="--", linewidth=2.0, label="Baseline MPC" if output_idx == 0 else None)
        ax.step(baseline_last["t_step_last"], baseline_last["sp_last"][:, output_idx], where="post", color="0.3", linestyle=":", linewidth=2.0, label="Setpoint" if output_idx == 0 else None)
        for method_key in methods:
            method_records = [record for record in run_records if record["method_key"] == method_key]
            if not method_records:
                continue
            color = colors[method_key]
            series = []
            for record in method_records:
                last = record["diagnostics"]["last_episode"]
                ax.plot(last["t_line_last"], last["y_last"][:, output_idx], color=color, alpha=0.18, linewidth=1.0)
                series.append(last["y_last"][:, output_idx])
            mean_series = np.mean(np.stack(series, axis=0), axis=0)
            ax.plot(last["t_line_last"], mean_series, color=color, linewidth=2.4, label=METHOD_LABELS[method_key] if output_idx == 0 else None)
        ax.set_ylabel(metadata["output_labels"][output_idx])
    axs[-1].set_xlabel(metadata["time_label"])
    axs[0].legend(loc="best", ncol=2)
    axs[0].set_title("Last-episode output tracking across methods", fontweight="bold")
    _save_fig(fig, os.fspath(out_dir), "fig_last_episode_outputs_by_method", save_pdf=save_pdf)


def _plot_baseline_delta_panel(method_summary_rows: list[dict[str, Any]], out_dir: Path, save_pdf: bool) -> None:
    labels = [row["method_label"] for row in method_summary_rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.bar(x - width, [row["delta_rmse_mean_vs_mpc_mean"] for row in method_summary_rows], width=width, label="Delta RMSE mean")
    ax.bar(x, [row["delta_iae_mean_vs_mpc_mean"] for row in method_summary_rows], width=width, label="Delta IAE mean")
    ax.bar(x + width, [row["delta_mean_abs_du_vs_mpc_mean"] for row in method_summary_rows], width=width, label="Delta mean |Δu|")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("RL - MPC")
    ax.set_title("Baseline-vs-method metric deltas", fontweight="bold")
    ax.legend(loc="best")
    _save_fig(fig, os.fspath(out_dir), "fig_baseline_delta_panel", save_pdf=save_pdf)


def _plot_input_movement_summary(method_summary_rows: list[dict[str, Any]], out_dir: Path, save_pdf: bool) -> None:
    labels = [row["method_label"] for row in method_summary_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.bar(x - width / 2.0, [row["mean_abs_du_mean"] for row in method_summary_rows], width=width, label="Mean |Δu|")
    ax.bar(x + width / 2.0, [row["action_saturation_mean_mean"] for row in method_summary_rows], width=width, label="Action saturation mean")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Value")
    ax.set_title("Input movement and saturation summary", fontweight="bold")
    ax.legend(loc="best")
    _save_fig(fig, os.fspath(out_dir), "fig_input_movement_summary", save_pdf=save_pdf)


def _compute_rankings(method_summary_rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    metric_specs = [
        ("rmse_mean_mean", "asc"),
        ("iae_mean_mean", "asc"),
        ("mean_abs_du_mean", "asc"),
        ("final_avg_reward_mean", "desc"),
    ]
    ranks: dict[str, list[int]] = defaultdict(list)
    for metric_key, direction in metric_specs:
        ordered = sorted(
            method_summary_rows,
            key=lambda row: (float("inf") if not np.isfinite(row[metric_key]) else row[metric_key]),
            reverse=(direction == "desc"),
        )
        for idx, row in enumerate(ordered, start=1):
            ranks[row["method_key"]].append(idx)
    return sorted(((method_key, float(np.mean(values))) for method_key, values in ranks.items()), key=lambda item: item[1])


def _plot_method_rankings(method_summary_rows: list[dict[str, Any]], out_dir: Path, save_pdf: bool) -> None:
    rankings = _compute_rankings(method_summary_rows)
    labels = [METHOD_LABELS[key] for key, _ in rankings]
    values = [rank for _, rank in rankings]
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.bar(labels, values)
    ax.set_ylabel("Average rank (lower is better)")
    ax.set_title("Method ranking using primary slide metrics", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    _save_fig(fig, os.fspath(out_dir), "fig_method_rankings", save_pdf=save_pdf)


def _plot_variability_summary(method_summary_rows: list[dict[str, Any]], out_dir: Path, save_pdf: bool) -> None:
    labels = [row["method_label"] for row in method_summary_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.bar(x - width / 2.0, [row["final_avg_reward_std"] for row in method_summary_rows], width=width, label="Std final avg reward")
    ax.bar(x + width / 2.0, [row["rmse_mean_std"] for row in method_summary_rows], width=width, label="Std RMSE mean")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Std across seeds")
    ax.set_title("Seed-to-seed variability summary", fontweight="bold")
    ax.legend(loc="best")
    _save_fig(fig, os.fspath(out_dir), "fig_variability_summary", save_pdf=save_pdf)


def _plot_method_diagnostic(record: dict[str, Any], out_dir: Path, save_pdf: bool) -> None:
    method_key = record["method_key"]
    label = record["method_label"]
    diag = record["diagnostics"]
    last = diag["last_episode"]
    if method_key == "horizon_dueling":
        trace = diag.get("horizon_action_trace")
        if trace is None or len(trace) == 0:
            return
        counts = Counter(trace.tolist())
        xs = list(sorted(counts.keys()))
        ys = [counts[idx] for idx in xs]
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        ax.bar(xs, ys)
        ax.set_xlabel("Horizon recipe index")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: recipe usage summary", fontweight="bold")
        _save_fig(fig, os.fspath(out_dir), "fig_diag_horizon_dueling_recipe_counts", save_pdf=save_pdf)
        return
    if method_key == "matrix":
        alpha = diag.get("alpha_log")
        if alpha is None or len(alpha) == 0:
            return
        tail = alpha[-last["episode_steps"] :]
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        ax.plot(last["t_step_last"], tail)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Alpha")
        ax.set_title(f"{label}: last-episode alpha trace", fontweight="bold")
        _save_fig(fig, os.fspath(out_dir), "fig_diag_matrix_alpha_trace", save_pdf=save_pdf)
        return
    if method_key == "weights":
        weight_log = diag.get("weight_log")
        if weight_log is None or len(weight_log) == 0:
            return
        tail = weight_log[-last["episode_steps"] :, :]
        fig, axs = plt.subplots(tail.shape[1], 1, figsize=(8.8, 3.2 + 1.6 * tail.shape[1]), sharex=True)
        if tail.shape[1] == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(last["t_step_last"], tail[:, idx])
            ax.set_ylabel(f"w{idx + 1}")
        axs[-1].set_xlabel("Time (h)")
        axs[0].set_title(f"{label}: last-episode weight multipliers", fontweight="bold")
        _save_fig(fig, os.fspath(out_dir), "fig_diag_weights_weight_trace", save_pdf=save_pdf)
        return
    if method_key == "residual":
        rho = diag.get("rho_log")
        residual_exec = diag.get("residual_exec_log")
        if rho is None or residual_exec is None or len(rho) == 0 or len(residual_exec) == 0:
            return
        rho_tail = rho[-last["episode_steps"] :]
        res_tail = residual_exec[-last["episode_steps"] :, :]
        fig, axs = plt.subplots(3, 1, figsize=(8.8, 7.4), sharex=True)
        axs[0].plot(last["t_step_last"], rho_tail)
        axs[0].set_ylabel("rho")
        axs[1].plot(last["t_step_last"], res_tail[:, 0])
        axs[1].set_ylabel("res_1")
        axs[2].plot(last["t_step_last"], res_tail[:, 1])
        axs[2].set_ylabel("res_2")
        axs[2].set_xlabel("Time (h)")
        axs[0].set_title(f"{label}: residual authority activity", fontweight="bold")
        _save_fig(fig, os.fspath(out_dir), "fig_diag_residual_authority", save_pdf=save_pdf)
        return
    if method_key == "combined":
        horizon = diag.get("horizon_action_trace")
        alpha = diag.get("matrix_alpha_log")
        weight_log = diag.get("weight_log")
        rho = diag.get("rho_log")
        fig, axs = plt.subplots(4, 1, figsize=(9.2, 8.8), sharex=False)
        if horizon is not None and len(horizon):
            axs[0].step(np.arange(len(horizon[-last["episode_steps"] :])), horizon[-last["episode_steps"] :], where="post")
        axs[0].set_ylabel("Recipe")
        if alpha is not None and len(alpha):
            axs[1].plot(last["t_step_last"], alpha[-last["episode_steps"] :])
        axs[1].set_ylabel("Alpha")
        if weight_log is not None and len(weight_log):
            tail = weight_log[-last["episode_steps"] :, :]
            axs[2].plot(last["t_step_last"], np.mean(tail, axis=1))
        axs[2].set_ylabel("Mean w")
        if rho is not None and len(rho):
            axs[3].plot(last["t_step_last"], rho[-last["episode_steps"] :])
        axs[3].set_ylabel("rho")
        axs[3].set_xlabel("Time (h)")
        axs[0].set_title(f"{label}: decision timeline and subsystem activity", fontweight="bold")
        _save_fig(fig, os.fspath(out_dir), "fig_diag_combined_activity", save_pdf=save_pdf)


def _write_markdown_outputs(
    out_dir: Path,
    run_records: list[dict[str, Any]],
    method_summary_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, float],
    baseline_path: Path,
) -> None:
    summary_headers = [
        "Method",
        "Final reward",
        "Tail reward",
        "RMSE mean",
        "IAE mean",
        "Tail offset",
        "Mean |Δu|",
        "ΔRMSE vs MPC",
    ]
    summary_rows = []
    for row in method_summary_rows:
        summary_rows.append(
            [
                row["method_label"],
                _format_mean_std(row["final_avg_reward_mean"], row["final_avg_reward_std"]),
                _format_mean_std(row["tail_avg_reward_mean"], row["tail_avg_reward_std"]),
                _format_mean_std(row["rmse_mean_mean"], row["rmse_mean_std"]),
                _format_mean_std(row["iae_mean_mean"], row["iae_mean_std"]),
                _format_mean_std(row["tail_offset_mean_mean"], row["tail_offset_mean_std"]),
                _format_mean_std(row["mean_abs_du_mean"], row["mean_abs_du_std"]),
                _format_mean_std(row["delta_rmse_mean_vs_mpc_mean"], row["delta_rmse_mean_vs_mpc_std"]),
            ]
        )
    summary_md = [
        "# Polymer Five-Seed Core Study",
        "",
        "## Method Summary",
        "",
        _markdown_table(summary_headers, summary_rows),
        "",
        "## Baseline Reference",
        "",
        f"- Baseline path: `{baseline_path}`",
        f"- RMSE mean: {baseline_metrics['rmse_mean']:.4f}",
        f"- IAE mean: {baseline_metrics['iae_mean']:.4f}",
        f"- Tail offset mean: {baseline_metrics['tail_offset_mean']:.4f}",
        f"- Mean |Δu|: {baseline_metrics['mean_abs_du']:.4f}",
    ]
    (out_dir / "polymer_five_seed_core_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    appendix_headers = [
        "Method",
        "Seed",
        "Final reward",
        "RMSE mean",
        "IAE mean",
        "Mean |Δu|",
        "Bundle path",
    ]
    appendix_rows = [
        [
            record["method_label"],
            record["seed"],
            f"{record['metrics']['final_avg_reward']:.4f}",
            f"{record['metrics']['rmse_mean']:.4f}",
            f"{record['metrics']['iae_mean']:.4f}",
            f"{record['metrics']['mean_abs_du']:.4f}",
            record["bundle_path"],
        ]
        for record in run_records
    ]
    appendix_md = [
        "# Polymer Five-Seed Core Study Appendix",
        "",
        _markdown_table(appendix_headers, appendix_rows),
    ]
    (out_dir / "polymer_five_seed_all_runs.md").write_text("\n".join(appendix_md) + "\n", encoding="utf-8")


def run_polymer_multiseed_core_study(
    *,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    data_dir_override: str | None = None,
    results_dir_override: str | None = None,
    save_pdf: bool = False,
    style_profile: str = "paper",
    figure_prefix: str = "polymer_five_seed_core_study",
    reward_tail_episodes: int = 20,
) -> dict[str, Any]:
    methods = list(methods or DEFAULT_METHODS)
    seeds = [int(seed) for seed in (seeds or DEFAULT_SEEDS)]
    repo_root, _, result_dir = prepare_polymer_notebook_env(data_dir_override=data_dir_override, results_dir_override=results_dir_override)
    os.chdir(repo_root)
    figure_root = Path(repo_root) / "report" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(create_output_dir(os.fspath(figure_root), figure_prefix))
    _set_plot_style(style_profile)

    raw_runs = []
    for method_key in methods:
        if method_key not in METHOD_SPECS:
            raise KeyError(f"Unsupported method: {method_key}")
        for seed in seeds:
            raw_runs.append(
                _run_method(
                    method_key,
                    seed,
                    repo_root,
                    result_dir,
                    data_dir_override,
                    style_profile=style_profile,
                    save_pdf=save_pdf,
                )
            )

    if not raw_runs:
        raise ValueError("No runs were executed.")

    reference_bundle = raw_runs[0]["bundle"]
    baseline_path = Path(raw_runs[0]["baseline_path"])
    baseline_bundle = _load_baseline_bundle(baseline_path, reference_bundle)
    baseline_metrics = _compute_metrics(baseline_bundle, reward_tail_episodes=reward_tail_episodes)

    run_records = [_build_run_record(raw_run, reward_tail_episodes) for raw_run in raw_runs]
    _append_baseline_deltas(run_records, baseline_metrics)
    method_summary_rows = _aggregate_method_records(run_records)
    flat_rows = _make_flat_rows(run_records, method_summary_rows, baseline_metrics, baseline_path)

    slide_metric_rows = [
        {
            "method_label": row["method_label"],
            "final_avg_reward_mean": row["final_avg_reward_mean"],
            "final_avg_reward_std": row["final_avg_reward_std"],
            "rmse_mean_mean": row["rmse_mean_mean"],
            "rmse_mean_std": row["rmse_mean_std"],
            "iae_mean_mean": row["iae_mean_mean"],
            "iae_mean_std": row["iae_mean_std"],
            "mean_abs_du_mean": row["mean_abs_du_mean"],
            "mean_abs_du_std": row["mean_abs_du_std"],
            "delta_rmse_mean_vs_mpc_mean": row["delta_rmse_mean_vs_mpc_mean"],
            "delta_rmse_mean_vs_mpc_std": row["delta_rmse_mean_vs_mpc_std"],
        }
        for row in method_summary_rows
    ]

    _write_csv(out_dir / "polymer_five_seed_core_flat_summary.csv", flat_rows)
    _write_csv(out_dir / "polymer_five_seed_core_slide_metrics.csv", slide_metric_rows)
    _write_json(
        out_dir / "polymer_five_seed_manifest.json",
        {
            "methods": methods,
            "seeds": seeds,
            "baseline_path": str(baseline_path),
            "output_dir": str(out_dir),
            "run_records": run_records,
            "method_summary_rows": method_summary_rows,
            "baseline_metrics": baseline_metrics,
        },
    )
    _write_markdown_outputs(out_dir, run_records, method_summary_rows, baseline_metrics, baseline_path)

    _plot_study_design_figure(out_dir, methods, seeds, baseline_path, save_pdf)
    _plot_reward_summary(run_records, out_dir, save_pdf)
    _plot_final_reward_boxplot(run_records, out_dir, save_pdf)
    _plot_last_episode_outputs(run_records, baseline_bundle, out_dir, save_pdf)
    _plot_baseline_delta_panel(method_summary_rows, out_dir, save_pdf)
    _plot_input_movement_summary(method_summary_rows, out_dir, save_pdf)
    _plot_method_rankings(method_summary_rows, out_dir, save_pdf)
    _plot_variability_summary(method_summary_rows, out_dir, save_pdf)

    for method_key in methods:
        representative = _representative_record([record for record in run_records if record["method_key"] == method_key])
        _plot_method_diagnostic(representative, out_dir, save_pdf)

    return {
        "output_dir": str(out_dir),
        "baseline_path": str(baseline_path),
        "run_records": run_records,
        "method_summary_rows": method_summary_rows,
        "baseline_metrics": baseline_metrics,
        "flat_summary_csv": str(out_dir / "polymer_five_seed_core_flat_summary.csv"),
        "slide_metrics_csv": str(out_dir / "polymer_five_seed_core_slide_metrics.csv"),
        "summary_markdown": str(out_dir / "polymer_five_seed_core_summary.md"),
        "manifest_json": str(out_dir / "polymer_five_seed_manifest.json"),
    }


__all__ = ["DEFAULT_METHODS", "DEFAULT_SEEDS", "METHOD_LABELS", "default_study_config", "run_polymer_multiseed_core_study", "set_global_seeds"]
