from __future__ import annotations

import argparse
import json
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Simulation.mpc import MpcSolverGeneral, augment_state_space, compute_observer_gain, exponential_decay_bonus
from Simulation.system_functions import PolymerCSTR
from TD3Agent.agent import TD3Agent
from systems.polymer.data_io import (
    canonical_baseline_path,
    ensure_polymer_directories,
    load_polymer_system_data,
    resolve_polymer_result_dir,
)
from systems.polymer.labels import POLYMER_SYSTEM_METADATA
from systems.polymer.notebook_params import (
    POLYMER_BASELINE_DEFAULTS,
    POLYMER_MARKOV_DEFAULTS,
    POLYMER_MATRIX_DEFAULTS,
)
from utils.agent_step_runtime import replay_train_continuous_agent, select_continuous_action
from utils.helpers import (
    apply_min_max,
    build_polymer_disturbance_schedule,
    disturbance_profile_from_schedule,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
)
from utils.plotting import compare_mpc_rl_from_dirs
from utils.rewards import make_reward_fn_relative_QR


def build_config(overrides=None):
    nb = deepcopy(POLYMER_BASELINE_DEFAULTS)
    matrix_nb = deepcopy(POLYMER_MATRIX_DEFAULTS)
    run_mode = str(nb["run_mode"]).lower()
    run_profile = deepcopy(nb["run_profiles"][run_mode])
    controller = deepcopy(nb["controller"])
    rl_episode = deepcopy(matrix_nb["episode_defaults"])
    rl_controller = deepcopy(matrix_nb["controller"])

    cfg = {
        "agent_kind": "td3",
        "run_mode": run_mode,
        "n_tests": int(rl_episode.get("n_tests", run_profile["n_tests"])),
        "set_points_len": int(rl_episode.get("set_points_len", run_profile["set_points_len"])),
        "warm_start": int(rl_episode.get("warm_start", 10)),
        "test_cycle": list(rl_episode.get("test_cycle", run_profile["test_cycle"])),
        "nominal_qi": float(run_profile["nominal_qi"]),
        "nominal_qs": float(run_profile["nominal_qs"]),
        "nominal_ha": float(run_profile["nominal_ha"]),
        "qi_change": float(run_profile["qi_change"]),
        "qs_change": float(run_profile["qs_change"]),
        "ha_change": float(run_profile["ha_change"]),
        "predict_h": int(controller["predict_h"]),
        "cont_h": int(controller["cont_h"]),
        "decision_interval": int(rl_controller.get("decision_interval", 1)),
        "Q1_penalty": float(controller["Q1_penalty"]),
        "Q2_penalty": float(controller["Q2_penalty"]),
        "R1_penalty": float(controller["R1_penalty"]),
        "R2_penalty": float(controller["R2_penalty"]),
        "use_shifted_mpc_warm_start": bool(controller.get("use_shifted_mpc_warm_start", False)),
        "basis_family": "io_pair_gain",
        "z_bound": 0.05,
        "prediction_window": 20,
        "lambda_z": 1.0e-3,
        "s_pred_min": 1.0e-6,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.10,
        "nominal_cost_absolute_tol": 1.0e-8,
        "run_shadow_only": True,
        "run_adaptive_ls": True,
        "run_live_corrected_mpc": True,
        "run_rl_proposal": True,
        "rl_fallback_to_ls": True,
        "rl_store_executed_action_in_replay": True,
        "rl_save_agent_checkpoint": True,
        "reward": deepcopy(POLYMER_MARKOV_DEFAULTS["reward"]),
        "td3_agent": deepcopy(matrix_nb["td3_agent"]),
        "save_outputs": True,
        "make_plots": True,
        "save_pdf": bool(nb.get("save_pdf", False)),
        "style_profile": str(nb.get("style_profile", "hybrid")),
        "plot_start_episode": int(run_profile.get("plot_start_episode", 1)),
        "seed": 42,
        "max_steps": None,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def compute_markov_blocks(A, B, C, p_max):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    blocks = []
    Apow = np.eye(A.shape[0])
    for _ in range(int(p_max)):
        blocks.append(C @ Apow @ B)
        Apow = Apow @ A
    return np.asarray(blocks, float)


def build_toeplitz_from_markov(m_blocks, P, M, hold_last=True):
    m_blocks = np.asarray(m_blocks, float)
    P = int(P)
    M = int(M)
    ny, nu = m_blocks.shape[1], m_blocks.shape[2]
    G = np.zeros((P * ny, M * nu), dtype=float)
    for h in range(P):
        for l in range(h + 1):
            col = l if (l < M or not hold_last) else M - 1
            if col >= M:
                continue
            G[h * ny : (h + 1) * ny, col * nu : (col + 1) * nu] += m_blocks[h - l]
    return G


def free_response(A, C, x0, P):
    A = np.asarray(A, float)
    C = np.asarray(C, float)
    x = np.asarray(x0, float).reshape(-1)
    ys = []
    for _ in range(int(P)):
        x = A @ x
        ys.append(C @ x)
    return np.asarray(ys, float).reshape(-1)


def lifted_predict(A, C, x0, G, U_seq, P, M):
    del M
    return free_response(A, C, x0, P) + np.asarray(G, float) @ np.asarray(U_seq, float).reshape(-1)


def state_space_predict(A, B, C, x0, U_seq, P, M):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    U = np.asarray(U_seq, float).reshape(int(M), B.shape[1])
    x = np.asarray(x0, float).reshape(-1)
    ys = []
    for h in range(int(P)):
        idx = h if h < int(M) else int(M) - 1
        x = A @ x + B @ U[idx]
        ys.append(C @ x)
    return np.asarray(ys, float).reshape(-1)


def make_markov_basis(m_blocks, basis_family):
    m_blocks = np.asarray(m_blocks, float)
    P, ny, nu = m_blocks.shape
    family = str(basis_family).lower()
    basis = []
    labels = []
    if family == "io_pair_gain":
        for out_idx in range(ny):
            for in_idx in range(nu):
                b = np.zeros_like(m_blocks)
                b[:, out_idx, in_idx] = m_blocks[:, out_idx, in_idx]
                basis.append(b)
                labels.append(f"y{out_idx + 1}_u{in_idx + 1}")
    elif family == "input_channel_gain":
        for in_idx in range(nu):
            b = np.zeros_like(m_blocks)
            b[:, :, in_idx] = m_blocks[:, :, in_idx]
            basis.append(b)
            labels.append(f"u{in_idx + 1}")
    elif family == "delay_shift":
        shifted = np.zeros_like(m_blocks)
        shifted[1:, :, :] = m_blocks[:-1, :, :]
        basis.append(shifted - m_blocks)
        labels.append("delay_shift")
    else:
        raise ValueError("basis_family must be io_pair_gain, input_channel_gain, or delay_shift.")
    return np.asarray(basis, float), labels


def apply_markov_correction(m_blocks, basis_blocks, z):
    corrected = np.asarray(m_blocks, float).copy()
    z = np.asarray(z, float).reshape(-1)
    for idx, z_value in enumerate(z):
        corrected += float(z_value) * basis_blocks[idx]
    return corrected


def gain_drift(Gz, G0, Wy=None, Wu=None, eps=1.0e-12):
    Gz = np.asarray(Gz, float)
    G0 = np.asarray(G0, float)
    left = np.eye(G0.shape[0]) if Wy is None else np.asarray(Wy, float)
    right = np.eye(G0.shape[1]) if Wu is None else np.asarray(Wu, float)
    num = np.linalg.norm(left @ (Gz - G0) @ right, ord="fro")
    den = np.linalg.norm(left @ G0 @ right, ord="fro") + float(eps)
    return float(num / den)


def available_prediction_indices(current_step, P, W):
    last_tau = int(current_step) - int(P)
    if last_tau < 0:
        return []
    first_tau = max(0, last_tau - int(W) + 1)
    return list(range(first_tau, last_tau + 1))


def real_u_window(u_dev_log, tau, M):
    U = np.zeros((int(M), u_dev_log.shape[1]), dtype=float)
    for j in range(int(M)):
        src = min(tau + j, u_dev_log.shape[0] - 1)
        U[j, :] = u_dev_log[src, :]
    return U


def prediction_improvement_score(
    *,
    z,
    history,
    m_blocks,
    basis_blocks,
    G0,
    A,
    C,
    P,
    M,
    Wy,
    lambda_z,
    current_step,
    prediction_window,
):
    indices = available_prediction_indices(current_step, P, prediction_window)
    if not indices:
        return {
            "score": 0.0,
            "nominal_sse": np.nan,
            "corrected_sse": np.nan,
            "n_windows": 0,
            "output_nominal_sse": np.full(C.shape[0], np.nan),
            "output_corrected_sse": np.full(C.shape[0], np.nan),
        }
    mz = apply_markov_correction(m_blocks, basis_blocks, z)
    Gz = build_toeplitz_from_markov(mz, P, M)
    ny = C.shape[0]
    Wy = np.eye(P * ny) if Wy is None else np.asarray(Wy, float)
    nom_sse = 0.0
    cor_sse = 0.0
    out_nom = np.zeros(ny, dtype=float)
    out_cor = np.zeros(ny, dtype=float)
    for tau in indices:
        U_real = real_u_window(history["u_dev_log"], tau, M)
        y_meas = history["y_scaled_dev"][tau + 1 : tau + P + 1, :].reshape(-1)
        y_free = free_response(A, C, history["xhat_before"][tau], P)
        e0 = y_meas - (y_free + G0 @ U_real.reshape(-1))
        ez = y_meas - (y_free + Gz @ U_real.reshape(-1))
        nom_sse += float(np.sum((Wy @ e0) ** 2))
        cor_sse += float(np.sum((Wy @ ez) ** 2))
        e0_mat = e0.reshape(P, ny)
        ez_mat = ez.reshape(P, ny)
        out_nom += np.sum(e0_mat**2, axis=0)
        out_cor += np.sum(ez_mat**2, axis=0)
    penalty = float(lambda_z) * float(np.sum(np.asarray(z, float) ** 2))
    return {
        "score": float(nom_sse - cor_sse - penalty),
        "nominal_sse": float(nom_sse),
        "corrected_sse": float(cor_sse),
        "n_windows": len(indices),
        "output_nominal_sse": out_nom,
        "output_corrected_sse": out_cor,
    }


def fit_markov_ls_correction(z0, bounds, history, m_blocks, basis_blocks, G0, A, C, P, M, Wy, lambda_z, current_step, prediction_window):
    def objective(z):
        score = prediction_improvement_score(
            z=z,
            history=history,
            m_blocks=m_blocks,
            basis_blocks=basis_blocks,
            G0=G0,
            A=A,
            C=C,
            P=P,
            M=M,
            Wy=Wy,
            lambda_z=lambda_z,
            current_step=current_step,
            prediction_window=prediction_window,
        )
        if score["n_windows"] == 0:
            return 0.0
        return float(score["corrected_sse"] + float(lambda_z) * np.sum(np.asarray(z, float) ** 2))

    result = spo.minimize(objective, np.asarray(z0, float), method="L-BFGS-B", bounds=bounds)
    z_star = np.asarray(result.x if result.success else z0, float)
    score = prediction_improvement_score(
        z=z_star,
        history=history,
        m_blocks=m_blocks,
        basis_blocks=basis_blocks,
        G0=G0,
        A=A,
        C=C,
        P=P,
        M=M,
        Wy=Wy,
        lambda_z=lambda_z,
        current_step=current_step,
        prediction_window=prediction_window,
    )
    return z_star, result, score


def z_to_raw_action(z, z_bound):
    z_bound = max(float(z_bound), 1.0e-12)
    return np.clip(np.asarray(z, float).reshape(-1) / z_bound, -1.0, 1.0)


def raw_action_to_z(raw_action, z_bound):
    return np.clip(np.asarray(raw_action, float).reshape(-1), -1.0, 1.0) * float(z_bound)


def markov_rl_state(x_model, tracking_error, innovation, u_prev_dev, z_prev, z_ls, ls_score, ls_gain_drift):
    return np.concatenate(
        [
            np.asarray(x_model, float).reshape(-1),
            np.asarray(tracking_error, float).reshape(-1),
            np.asarray(innovation, float).reshape(-1),
            np.asarray(u_prev_dev, float).reshape(-1),
            np.asarray(z_prev, float).reshape(-1),
            np.asarray(z_ls, float).reshape(-1),
            np.asarray([float(ls_score), float(ls_gain_drift)], float),
        ]
    ).astype(np.float32, copy=False)


def build_test_flags(ctx):
    flags = np.zeros(int(ctx["nFE"]), dtype=bool)
    active = False
    starts = sorted((int(k), bool(v)) for k, v in ctx["test_train_dict"].items())
    start_idx = 0
    for step in range(int(ctx["nFE"])):
        while start_idx < len(starts) and starts[start_idx][0] <= step:
            active = starts[start_idx][1]
            start_idx += 1
        flags[step] = active
    return flags


def make_td3_markov_agent(config, state_dim, action_dim):
    td3_cfg = deepcopy(config["td3_agent"])
    buffer_size = int(td3_cfg.get("buffer_size", 40_000))
    recent_window = td3_cfg.get("replay_recent_window")
    if recent_window is None:
        recent_window = min(
            buffer_size,
            int(td3_cfg.get("replay_recent_window_mult", 5)) * int(config["set_points_len"]),
        )

    return TD3Agent(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        actor_hidden=list(td3_cfg["actor_hidden"]),
        critic_hidden=list(td3_cfg["critic_hidden"]),
        gamma=float(td3_cfg.get("gamma", 0.995)),
        actor_lr=float(td3_cfg.get("actor_lr", 1.0e-4)),
        critic_lr=float(td3_cfg.get("critic_lr", 1.0e-4)),
        batch_size=int(td3_cfg.get("batch_size", 128)),
        n_step=int(td3_cfg.get("n_step", 1)),
        multistep_mode=str(td3_cfg.get("multistep_mode", "one_step")),
        lambda_value=float(td3_cfg.get("lambda_value", 0.9)),
        grad_clip_norm=td3_cfg.get("grad_clip_norm", 10.0),
        policy_delay=int(td3_cfg.get("policy_delay", 2)),
        target_policy_smoothing_noise_std=float(td3_cfg.get("target_policy_smoothing_noise_std", 0.1)),
        noise_clip=float(td3_cfg.get("noise_clip", 0.2)),
        max_action=float(td3_cfg.get("max_action", 1.0)),
        target_update=str(td3_cfg.get("target_update", "soft")),
        tau=float(td3_cfg.get("tau", 0.005)),
        hard_update_interval=int(td3_cfg.get("hard_update_interval", 10_000)),
        target_combine=str(td3_cfg.get("target_combine", "min")),
        activation=str(td3_cfg.get("activation", "relu")),
        use_layernorm=bool(td3_cfg.get("use_layernorm", False)),
        dropout=float(td3_cfg.get("dropout", 0.0)),
        std_start=float(td3_cfg.get("std_start", 0.2)),
        std_end=float(td3_cfg.get("std_end", 0.02)),
        std_decay_rate=float(td3_cfg.get("std_decay_rate", 0.99995)),
        std_decay_mode=str(td3_cfg.get("std_decay_mode", "exp")),
        exploration_mode=str(td3_cfg.get("exploration_mode", "param_noise")),
        param_noise_resample_interval=int(td3_cfg.get("param_noise_resample_interval", 4)),
        loss_type=str(td3_cfg.get("loss_type", "huber")),
        buffer_size=buffer_size,
        replay_frac_per=float(td3_cfg.get("replay_frac_per", 0.5)),
        replay_frac_recent=float(td3_cfg.get("replay_frac_recent", 0.2)),
        replay_recent_window=int(recent_window),
        replay_alpha=float(td3_cfg.get("replay_alpha", 0.6)),
        replay_beta_start=float(td3_cfg.get("replay_beta_start", 0.4)),
        replay_beta_end=float(td3_cfg.get("replay_beta_end", 1.0)),
        replay_beta_steps=int(td3_cfg.get("replay_beta_steps", 50_000)),
        actor_freeze=int(td3_cfg.get("actor_freeze", 0)),
    )


def lifted_mpc_cost(U_flat, y_sp, u_prev_dev, Y_free, G, Q_out, R_in, P, M):
    U = np.asarray(U_flat, float).reshape(int(M), -1)
    ny = len(np.asarray(y_sp).reshape(-1))
    y_pred = (np.asarray(Y_free, float) + np.asarray(G, float) @ U.reshape(-1)).reshape(int(P), ny)
    y_err = y_pred - np.asarray(y_sp, float).reshape(1, ny)
    U_prev = np.vstack([np.asarray(u_prev_dev, float).reshape(1, -1), U[:-1, :]])
    dU = U - U_prev
    return float(np.sum(np.asarray(Q_out, float).reshape(1, ny) * y_err**2) + np.sum(np.asarray(R_in, float).reshape(1, -1) * dU**2))


def solve_lifted_mpc(y_sp, u_prev_dev, x0_model, A, C, G, Q_out, R_in, P, M, bounds, x_init):
    Y_free = free_response(A, C, x0_model, P)
    fun = lambda U_flat: lifted_mpc_cost(U_flat, y_sp, u_prev_dev, Y_free, G, Q_out, R_in, P, M)
    sol = spo.minimize(fun, np.asarray(x_init, float), method="SLSQP", bounds=bounds)
    if not sol.success:
        return np.asarray(x_init, float), float(fun(x_init)), sol
    return np.asarray(sol.x, float), float(sol.fun), sol


def legacy_mpc_reward(delta_y, delta_u, y_sp, Q_out, R_in):
    Q_out = np.asarray(Q_out, float).reshape(-1)
    R_in = np.asarray(R_in, float).reshape(-1)
    reward = -(np.sum(Q_out * np.asarray(delta_y, float) ** 2) + np.sum(R_in * np.asarray(delta_u, float) ** 2))
    error_norm = np.abs(np.linalg.norm((delta_y).reshape(1, -1), axis=0) / (np.asarray(y_sp, float) + 1.0e-15)) * 100.0
    if np.all(error_norm <= 5.0):
        reward += exponential_decay_bonus(float(np.mean(error_norm)))
    return float(reward)


def make_compare_reward_fn(ctx):
    return ctx["reward_fn"]


def avg_by_episode(rewards, sub_episode_changes, time_in_sub_episodes):
    avg = []
    for idx in sorted(sub_episode_changes):
        start = max(0, int(idx) - int(time_in_sub_episodes) + 1)
        stop = int(idx)
        avg.append(float(np.mean(rewards[start:stop])) if stop > start else float("nan"))
    return np.asarray(avg, float)


def build_context(config):
    nb = deepcopy(POLYMER_BASELINE_DEFAULTS)
    sys_cfg = deepcopy(nb["system_setup"])
    run_mode = str(config["run_mode"]).lower()
    ensure_polymer_directories(REPO_ROOT)

    system_params = np.asarray(sys_cfg["system_params"], float).copy()
    design_params = np.asarray(sys_cfg["design_params"], float).copy()
    ss_inputs = np.asarray(sys_cfg["ss_inputs"], float).copy()
    delta_t = float(sys_cfg["delta_t_hours"])
    cstr_ss = PolymerCSTR(system_params, design_params, ss_inputs, delta_t)
    steady_states = {"ss_inputs": cstr_ss.ss_inputs.copy(), "y_ss": cstr_ss.y_ss.copy()}

    y_sp_scenario_phys = np.asarray(sys_cfg["rl_setpoints_phys"], float).copy()
    setpoint_y = np.asarray(sys_cfg["setpoint_range_phys"], float).copy()
    input_bounds = sys_cfg["input_bounds"]
    system_data = load_polymer_system_data(
        REPO_ROOT,
        steady_states=steady_states,
        setpoint_y=setpoint_y,
        u_min=np.asarray(input_bounds["u_min"], float),
        u_max=np.asarray(input_bounds["u_max"], float),
        n_inputs=2,
        data_override=nb.get("polymer_data_dir_override"),
    )
    A_aug = np.asarray(system_data["A_aug"], float)
    B_aug = np.asarray(system_data["B_aug"], float)
    C_aug = np.asarray(system_data["C_aug"], float)
    n_inputs = B_aug.shape[1]
    data_min = np.asarray(system_data["data_min"], float)
    data_max = np.asarray(system_data["data_max"], float)
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scenario = apply_min_max(y_sp_scenario_phys, data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
    reward_cfg = deepcopy(config.get("reward", POLYMER_MARKOV_DEFAULTS["reward"]))
    reward_params, reward_fn = make_reward_fn_relative_QR(
        data_min,
        data_max,
        n_inputs=n_inputs,
        **reward_cfg,
    )

    y_sp, nFE, sub_episode_changes, time_in_sub_episodes, test_train_dict, warm_start_step, qi, qs, ha = (
        generate_setpoints_training_rl_gradually(
            y_sp_scenario,
            int(config["n_tests"]),
            int(config["set_points_len"]),
            int(config["warm_start"]),
            list(config["test_cycle"]),
            float(config["nominal_qi"]),
            float(config["nominal_qs"]),
            float(config["nominal_ha"]),
            float(config["qi_change"]),
            float(config["qs_change"]),
            float(config["ha_change"]),
        )
    )
    max_steps = config.get("max_steps")
    if max_steps is not None:
        nFE = min(int(nFE), int(max_steps))
        y_sp = y_sp[:nFE, :]
        qi = qi[:nFE]
        qs = qs[:nFE]
        ha = ha[:nFE]
        sub_episode_changes = {k: v for k, v in sub_episode_changes.items() if k < nFE}
        test_train_dict = {k: v for k, v in test_train_dict.items() if k < nFE}

    observer_poles = np.asarray(sys_cfg["observer_poles"], float)
    L = compute_observer_gain(A_aug, C_aug, observer_poles)

    Q_out = np.asarray([config["Q1_penalty"], config["Q2_penalty"]], float)
    R_in = np.asarray([config["R1_penalty"], config["R2_penalty"]], float)
    ss_scaled_inputs = np.asarray(system_data["u_ss_scaled"], float)
    y_sp_phys = reverse_min_max(y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
    bounds = [
        (float(system_data["b_min"][inp]), float(system_data["b_max"][inp]))
        for _ in range(int(config["cont_h"]))
        for inp in range(n_inputs)
    ]

    return {
        "nb": nb,
        "sys_cfg": sys_cfg,
        "run_mode": run_mode,
        "system_params": system_params,
        "design_params": design_params,
        "ss_inputs": ss_inputs,
        "steady_states": steady_states,
        "delta_t": delta_t,
        "system_data": system_data,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "L": L,
        "Q_out": Q_out,
        "R_in": R_in,
        "bounds": bounds,
        "ss_scaled_inputs": ss_scaled_inputs,
        "y_ss_scaled": y_ss_scaled,
        "y_sp": y_sp,
        "y_sp_phys": y_sp_phys,
        "reward_params": reward_params,
        "reward_fn": reward_fn,
        "nFE": int(nFE),
        "sub_episode_changes": sub_episode_changes,
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "test_train_dict": test_train_dict,
        "warm_start_step": int(warm_start_step),
        "qi": np.asarray(qi, float),
        "qs": np.asarray(qs, float),
        "ha": np.asarray(ha, float),
        "disturbance_schedule": build_polymer_disturbance_schedule(qi, qs, ha),
        "baseline_path": canonical_baseline_path(REPO_ROOT, run_mode),
        "result_base": resolve_polymer_result_dir(REPO_ROOT),
        "result_root": resolve_polymer_result_dir(REPO_ROOT) / "polymer_markov_corrected_mpc",
    }


MARKOV_ACTION_SOURCE = {
    0: "nominal_no_markov",
    1: "warm_start_ls",
    2: "td3_accepted",
    3: "ls_fallback",
    4: "nominal_fallback",
    5: "ls_no_rl",
}


def initialize_history(nFE, nx, ny, nu, z_dim, rl_state_dim=0):
    return {
        "xhat_before": np.zeros((nFE, nx), dtype=float),
        "xhat_after": np.zeros((nFE + 1, nx), dtype=float),
        "y_phys": np.zeros((nFE + 1, ny), dtype=float),
        "y_scaled_dev": np.zeros((nFE + 1, ny), dtype=float),
        "u_phys_log": np.zeros((nFE, nu), dtype=float),
        "u_scaled_abs_log": np.zeros((nFE, nu), dtype=float),
        "u_dev_log": np.zeros((nFE, nu), dtype=float),
        "du_log": np.zeros((nFE, nu), dtype=float),
        "rewards": np.zeros(nFE, dtype=float),
        "z_log": np.zeros((nFE, z_dim), dtype=float),
        "z_proposed_log": np.zeros((nFE, z_dim), dtype=float),
        "z_executed_log": np.zeros((nFE, z_dim), dtype=float),
        "s_pred_log": np.zeros(nFE, dtype=float),
        "gain_drift_log": np.zeros(nFE, dtype=float),
        "accepted_log": np.zeros(nFE, dtype=int),
        "fallback_log": np.ones(nFE, dtype=int),
        "prediction_error_nominal_log": np.full(nFE, np.nan, dtype=float),
        "prediction_error_markov_log": np.full(nFE, np.nan, dtype=float),
        "candidate_index_log": np.full(nFE, -1, dtype=int),
        "rl_state_log": np.zeros((nFE, int(rl_state_dim)), dtype=float),
        "rl_requested_raw_action_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_executed_raw_action_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_requested_z_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_ls_z_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_action_source_log": np.zeros(nFE, dtype=int),
        "rl_decision_taken_log": np.zeros(nFE, dtype=int),
        "rl_policy_source_log": np.zeros(nFE, dtype=int),
        "rl_replay_pushed_log": np.zeros(nFE, dtype=int),
        "rl_train_called_log": np.zeros(nFE, dtype=int),
        "rl_train_updated_log": np.zeros(nFE, dtype=int),
        "rl_actor_loss_log": np.full(nFE, np.nan, dtype=float),
        "rl_critic_loss_log": np.full(nFE, np.nan, dtype=float),
        "rl_bc_loss_log": np.full(nFE, np.nan, dtype=float),
        "rl_test_step_log": np.zeros(nFE, dtype=int),
    }


def run_closed_loop(config, ctx, m_blocks, basis_blocks, G0, Wy, *, use_markov, print_progress=False):
    P = int(config["predict_h"])
    M = int(config["cont_h"])
    A = ctx["A_aug"]
    B = ctx["B_aug"]
    C = ctx["C_aug"]
    nFE = ctx["nFE"]
    ny, nu = C.shape[0], B.shape[1]
    z_dim = basis_blocks.shape[0]
    z_bounds = [(-float(config["z_bound"]), float(config["z_bound"])) for _ in range(z_dim)]
    rl_state_dim = A.shape[0] + ny + ny + nu + z_dim + z_dim + 2
    history = initialize_history(nFE, A.shape[0], ny, nu, z_dim, rl_state_dim)
    use_rl = bool(use_markov and config.get("run_rl_proposal", False) and str(config.get("agent_kind", "td3")).lower() == "td3")
    rl_agent = make_td3_markov_agent(config, rl_state_dim, z_dim) if use_rl else None
    test_flags = build_test_flags(ctx)

    system = PolymerCSTR(ctx["system_params"], ctx["design_params"], ctx["ss_inputs"], ctx["delta_t"])
    history["y_phys"][0, :] = system.current_output
    history["y_scaled_dev"][0, :] = apply_min_max(system.current_output, ctx["system_data"]["data_min"][nu:], ctx["system_data"]["data_max"][nu:]) - ctx["y_ss_scaled"]
    x_model = np.zeros(A.shape[0], dtype=float)
    x_init = np.zeros(M * nu, dtype=float)
    z_prev = np.zeros(z_dim, dtype=float)
    last_raw_action = None
    last_action_test = None
    pending_transition = None
    avg_rewards = []

    def evaluate_markov_candidate(z_candidate, u_prev_dev, x_model, x_init_candidate, nominal_cost):
        mz_candidate = apply_markov_correction(m_blocks, basis_blocks, z_candidate)
        G_candidate = build_toeplitz_from_markov(mz_candidate, P, M)
        candidate_drift = gain_drift(G_candidate, G0)
        U_candidate, J_candidate, sol_candidate = solve_lifted_mpc(
            ctx["y_sp"][step],
            u_prev_dev,
            x_model,
            A,
            C,
            G_candidate,
            ctx["Q_out"],
            ctx["R_in"],
            P,
            M,
            ctx["bounds"],
            x_init_candidate,
        )
        nominal_cost_of_candidate = lifted_mpc_cost(
            U_candidate,
            ctx["y_sp"][step],
            u_prev_dev,
            free_response(A, C, x_model, P),
            G0,
            ctx["Q_out"],
            ctx["R_in"],
            P,
            M,
        )
        loose_tol = float(config["nominal_cost_absolute_tol"]) + float(config["nominal_cost_relative_tol"]) * abs(float(nominal_cost))
        return {
            "U": U_candidate,
            "J": J_candidate,
            "sol": sol_candidate,
            "drift": candidate_drift,
            "nominal_cost": nominal_cost_of_candidate,
            "cost_guard_pass": nominal_cost_of_candidate <= float(nominal_cost) + loose_tol,
        }

    for step in range(nFE):
        scaled_current_input = apply_min_max(system.current_input, ctx["system_data"]["data_min"][:nu], ctx["system_data"]["data_max"][:nu])
        u_prev_dev = scaled_current_input - ctx["ss_scaled_inputs"]
        history["xhat_before"][step, :] = x_model

        U0, J0, sol0 = solve_lifted_mpc(
            ctx["y_sp"][step],
            u_prev_dev,
            x_model,
            A,
            C,
            G0,
            ctx["Q_out"],
            ctx["R_in"],
            P,
            M,
            ctx["bounds"],
            x_init,
        )
        U_exec = U0.copy()
        z_exec = np.zeros(z_dim, dtype=float)
        accepted = False
        fallback = True
        action_source = 0
        raw_requested = np.zeros(z_dim, dtype=float)
        raw_executed = np.zeros(z_dim, dtype=float)
        score = {
            "score": 0.0,
            "nominal_sse": np.nan,
            "corrected_sse": np.nan,
            "n_windows": 0,
            "output_nominal_sse": np.full(ny, np.nan),
            "output_corrected_sse": np.full(ny, np.nan),
        }
        ls_score = dict(score)
        drift = 0.0
        ls_drift = 0.0
        ls_accepted = False
        z_ls = np.zeros(z_dim, dtype=float)
        U_ls = None

        if use_markov and config.get("run_adaptive_ls", True) and step >= P:
            z_ls, _ls_result, ls_score = fit_markov_ls_correction(
                z_prev,
                z_bounds,
                history,
                m_blocks,
                basis_blocks,
                G0,
                A,
                C,
                P,
                M,
                Wy,
                float(config["lambda_z"]),
                step,
                int(config["prediction_window"]),
            )
            ls_eval = evaluate_markov_candidate(z_ls, u_prev_dev, x_model, U0, J0)
            ls_drift = float(ls_eval["drift"])
            ls_accepted = bool(
                sol0.success
                and ls_eval["sol"].success
                and ls_score["score"] > float(config["s_pred_min"])
                and ls_drift <= float(config["gain_drift_max"])
                and ls_eval["cost_guard_pass"]
            )
            if ls_accepted:
                U_ls = ls_eval["U"]

        z_ls_safe = z_ls if ls_accepted else np.zeros(z_dim, dtype=float)
        yhat = C @ x_model
        innovation = history["y_scaled_dev"][step, :] - yhat
        tracking_error = history["y_scaled_dev"][step, :] - ctx["y_sp"][step, :]
        rl_state = markov_rl_state(
            x_model,
            tracking_error,
            innovation,
            u_prev_dev,
            z_prev,
            z_ls_safe,
            ls_score["score"],
            ls_drift,
        )
        history["rl_state_log"][step, :] = rl_state
        history["rl_ls_z_log"][step, :] = z_ls

        if pending_transition is not None and rl_agent is not None:
            train_info = replay_train_continuous_agent(
                agent=rl_agent,
                state=pending_transition["state"],
                action=pending_transition["action"],
                reward=pending_transition["reward"],
                next_state=rl_state,
                done=0.0,
                step=pending_transition["step"],
                test=pending_transition["test"],
                train_start_step=ctx["warm_start_step"],
            )
            pending_step = int(pending_transition["step"])
            history["rl_replay_pushed_log"][pending_step] = int(train_info["pushed"])
            history["rl_train_called_log"][pending_step] = int(train_info["trained"])
            train_meta = train_info.get("train_meta")
            if train_meta is not None:
                history["rl_train_updated_log"][pending_step] = int(train_meta.get("critic_updated", False))
                if train_meta.get("actor_loss") is not None:
                    history["rl_actor_loss_log"][pending_step] = float(train_meta["actor_loss"])
                if train_meta.get("critic_loss") is not None:
                    history["rl_critic_loss_log"][pending_step] = float(train_meta["critic_loss"])
                if train_meta.get("bc_loss") is not None:
                    history["rl_bc_loss_log"][pending_step] = float(train_meta["bc_loss"])
            pending_transition = None

        if use_markov and config["run_live_corrected_mpc"] and step >= P:
            if rl_agent is not None:
                baseline_raw = z_to_raw_action(z_ls_safe, config["z_bound"])
                test_step = bool(test_flags[step])
                decision = select_continuous_action(
                    agent=rl_agent,
                    state=rl_state,
                    step=step,
                    warm_start_step=ctx["warm_start_step"],
                    decision_interval=int(config.get("decision_interval", 1)),
                    last_action=last_raw_action,
                    last_action_test=last_action_test,
                    test=test_step,
                    baseline_action=baseline_raw,
                    action_dim=z_dim,
                    nonfinite_fallback=True,
                )
                raw_requested = np.asarray(decision.action, float).reshape(-1)
                last_raw_action = decision.last_action
                last_action_test = decision.last_action_test
                history["rl_decision_taken_log"][step] = int(decision.decision_taken)
                history["rl_policy_source_log"][step] = int(decision.source)
                history["rl_test_step_log"][step] = int(test_step)

                z_requested = raw_action_to_z(raw_requested, config["z_bound"])
                history["rl_requested_raw_action_log"][step, :] = raw_requested
                history["rl_requested_z_log"][step, :] = z_requested
                history["z_proposed_log"][step, :] = z_requested

                if step > ctx["warm_start_step"]:
                    rl_score = prediction_improvement_score(
                        z=z_requested,
                        history=history,
                        m_blocks=m_blocks,
                        basis_blocks=basis_blocks,
                        G0=G0,
                        A=A,
                        C=C,
                        P=P,
                        M=M,
                        Wy=Wy,
                        lambda_z=float(config["lambda_z"]),
                        current_step=step,
                        prediction_window=int(config["prediction_window"]),
                    )
                    rl_eval = evaluate_markov_candidate(z_requested, u_prev_dev, x_model, U0, J0)
                    rl_accepted = bool(
                        sol0.success
                        and rl_eval["sol"].success
                        and rl_score["score"] > float(config["s_pred_min"])
                        and rl_eval["drift"] <= float(config["gain_drift_max"])
                        and rl_eval["cost_guard_pass"]
                    )
                    if rl_accepted:
                        U_exec = rl_eval["U"]
                        z_exec = z_requested
                        raw_executed = raw_requested
                        score = rl_score
                        drift = float(rl_eval["drift"])
                        accepted = True
                        fallback = False
                        action_source = 2
                        z_prev = z_exec
                    elif bool(config.get("rl_fallback_to_ls", True)) and ls_accepted and U_ls is not None:
                        U_exec = U_ls
                        z_exec = z_ls
                        raw_executed = z_to_raw_action(z_ls, config["z_bound"])
                        score = ls_score
                        drift = ls_drift
                        accepted = True
                        fallback = True
                        action_source = 3
                        z_prev = z_exec
                    else:
                        raw_executed = np.zeros(z_dim, dtype=float)
                        action_source = 4
                elif ls_accepted and U_ls is not None:
                    U_exec = U_ls
                    z_exec = z_ls
                    raw_executed = baseline_raw
                    score = ls_score
                    drift = ls_drift
                    accepted = True
                    fallback = False
                    action_source = 1
                    z_prev = z_exec
                else:
                    raw_executed = np.zeros(z_dim, dtype=float)
                    action_source = 4
            elif ls_accepted and U_ls is not None:
                U_exec = U_ls
                z_exec = z_ls
                raw_requested = z_to_raw_action(z_ls, config["z_bound"])
                raw_executed = raw_requested
                score = ls_score
                drift = ls_drift
                accepted = True
                fallback = False
                action_source = 5
                z_prev = z_exec

        u_dev = U_exec[:nu]
        u_scaled_abs = u_dev + ctx["ss_scaled_inputs"]
        u_phys = reverse_min_max(u_scaled_abs, ctx["system_data"]["data_min"][:nu], ctx["system_data"]["data_max"][:nu])
        du = u_dev - u_prev_dev

        system.Qi = float(ctx["qi"][step])
        system.Qs = float(ctx["qs"][step])
        system.hA = float(ctx["ha"][step])
        system.current_input = u_phys
        system.step()

        history["u_phys_log"][step, :] = u_phys
        history["u_scaled_abs_log"][step, :] = u_scaled_abs
        history["u_dev_log"][step, :] = u_dev
        history["du_log"][step, :] = du
        history["y_phys"][step + 1, :] = system.current_output
        history["y_scaled_dev"][step + 1, :] = apply_min_max(system.current_output, ctx["system_data"]["data_min"][nu:], ctx["system_data"]["data_max"][nu:]) - ctx["y_ss_scaled"]

        x_model = A @ x_model + B @ u_dev + ctx["L"] @ innovation
        history["xhat_after"][step + 1, :] = x_model

        delta_y = history["y_scaled_dev"][step + 1, :] - ctx["y_sp"][step, :]
        history["rewards"][step] = float(ctx["reward_fn"](delta_y, du, ctx["y_sp_phys"][step, :]))
        history["z_log"][step, :] = z_exec
        history["z_executed_log"][step, :] = z_exec
        history["rl_executed_raw_action_log"][step, :] = raw_executed
        history["rl_action_source_log"][step] = int(action_source)
        history["s_pred_log"][step] = float(score["score"])
        history["gain_drift_log"][step] = float(drift)
        history["accepted_log"][step] = int(accepted)
        history["fallback_log"][step] = int(fallback)
        history["prediction_error_nominal_log"][step] = float(score["nominal_sse"]) if np.isfinite(score["nominal_sse"]) else np.nan
        history["prediction_error_markov_log"][step] = float(score["corrected_sse"]) if np.isfinite(score["corrected_sse"]) else np.nan

        if rl_agent is not None:
            transition_action = raw_executed if bool(config.get("rl_store_executed_action_in_replay", True)) else raw_requested
            pending_transition = {
                "step": step,
                "state": rl_state.copy(),
                "action": np.asarray(transition_action, float).copy(),
                "reward": float(history["rewards"][step]),
                "test": bool(test_flags[step]),
            }

        if step in ctx["sub_episode_changes"]:
            start = max(0, step - int(ctx["time_in_sub_episodes"]) + 1)
            stop = step + 1
            avg_reward = float(np.mean(history["rewards"][start:stop]))
            avg_rewards.append(avg_reward)
            if print_progress:
                accepted_fraction = float(np.mean(history["accepted_log"][start:stop]))
                td3_accepted_fraction = float(np.mean(history["rl_action_source_log"][start:stop] == 2))
                ls_fallback_fraction = float(np.mean(history["rl_action_source_log"][start:stop] == 3))
                nominal_fallback_fraction = float(np.mean(history["rl_action_source_log"][start:stop] == 4))
                mean_z = np.mean(history["z_executed_log"][start:stop, :], axis=0)
                print(
                    "Sub_Episode:",
                    ctx["sub_episode_changes"][step],
                    "| avg. reward:",
                    avg_reward,
                    "| accepted:",
                    accepted_fraction,
                    "| TD3 accepted:",
                    td3_accepted_fraction,
                    "| LS fallback:",
                    ls_fallback_fraction,
                    "| nominal fallback:",
                    nominal_fallback_fraction,
                    "| avg z:",
                    mean_z,
                )

        if config["use_shifted_mpc_warm_start"]:
            U_mat = U_exec.reshape(M, nu)
            x_init = np.vstack([U_mat[1:, :], U_mat[-1:, :]]).reshape(-1)
        else:
            x_init = np.zeros(M * nu, dtype=float)

    if pending_transition is not None and rl_agent is not None:
        train_info = replay_train_continuous_agent(
            agent=rl_agent,
            state=pending_transition["state"],
            action=pending_transition["action"],
            reward=pending_transition["reward"],
            next_state=pending_transition["state"],
            done=1.0,
            step=pending_transition["step"],
            test=pending_transition["test"],
            train_start_step=ctx["warm_start_step"],
        )
        pending_step = int(pending_transition["step"])
        history["rl_replay_pushed_log"][pending_step] = int(train_info["pushed"])
        history["rl_train_called_log"][pending_step] = int(train_info["trained"])
        train_meta = train_info.get("train_meta")
        if train_meta is not None:
            history["rl_train_updated_log"][pending_step] = int(train_meta.get("critic_updated", False))
            if train_meta.get("actor_loss") is not None:
                history["rl_actor_loss_log"][pending_step] = float(train_meta["actor_loss"])
            if train_meta.get("critic_loss") is not None:
                history["rl_critic_loss_log"][pending_step] = float(train_meta["critic_loss"])
            if train_meta.get("bc_loss") is not None:
                history["rl_bc_loss_log"][pending_step] = float(train_meta["bc_loss"])

    history["avg_rewards"] = (
        np.asarray(avg_rewards, float)
        if avg_rewards
        else avg_by_episode(history["rewards"], ctx["sub_episode_changes"], ctx["time_in_sub_episodes"])
    )
    history["_rl_agent"] = rl_agent
    history["rl_state_dim"] = int(rl_state_dim)
    history["rl_action_dim"] = int(z_dim)
    history["rl_action_source_names"] = dict(MARKOV_ACTION_SOURCE)
    history["rl_agent_checkpoint_path"] = None
    return history


def run_shadow_and_ls(config, ctx, nominal_history, m_blocks, basis_blocks, G0, Wy):
    P = int(config["predict_h"])
    M = int(config["cont_h"])
    A = ctx["A_aug"]
    C = ctx["C_aug"]
    nFE = ctx["nFE"]
    z_dim = basis_blocks.shape[0]
    z_bound = float(config["z_bound"])
    z_bounds = [(-z_bound, z_bound) for _ in range(z_dim)]
    candidates = [np.zeros(z_dim, dtype=float)]
    if str(config["basis_family"]).lower() == "delay_shift":
        for value in (0.05, 0.10, 0.20):
            candidates.append(np.asarray([min(z_bound, value)], dtype=float))
    else:
        for value in (0.02, -0.02, z_bound, -z_bound):
            for idx in range(z_dim):
                z = np.zeros(z_dim, dtype=float)
                z[idx] = value
                candidates.append(z)

    shadow = {
        "candidate_z": np.asarray(candidates, float),
        "best_candidate_index": np.full(nFE, -1, dtype=int),
        "best_candidate_z": np.zeros((nFE, z_dim), dtype=float),
        "best_s_pred": np.zeros(nFE, dtype=float),
        "best_gain_drift": np.zeros(nFE, dtype=float),
        "ls_z": np.zeros((nFE, z_dim), dtype=float),
        "ls_s_pred": np.zeros(nFE, dtype=float),
        "ls_gain_drift": np.zeros(nFE, dtype=float),
        "ls_accepted": np.zeros(nFE, dtype=int),
        "nominal_sse": np.full(nFE, np.nan, dtype=float),
        "candidate_sse": np.full(nFE, np.nan, dtype=float),
        "ls_sse": np.full(nFE, np.nan, dtype=float),
        "output_nominal_sse": np.full((nFE, C.shape[0]), np.nan, dtype=float),
        "output_ls_sse": np.full((nFE, C.shape[0]), np.nan, dtype=float),
    }
    z_prev = np.zeros(z_dim, dtype=float)
    for step in range(nFE):
        best_idx = -1
        best_score = -np.inf
        best_payload = None
        best_z = np.zeros(z_dim, dtype=float)
        for cand_idx, z in enumerate(candidates):
            score = prediction_improvement_score(
                z=z,
                history=nominal_history,
                m_blocks=m_blocks,
                basis_blocks=basis_blocks,
                G0=G0,
                A=A,
                C=C,
                P=P,
                M=M,
                Wy=Wy,
                lambda_z=float(config["lambda_z"]),
                current_step=step,
                prediction_window=int(config["prediction_window"]),
            )
            if score["n_windows"] > 0 and score["score"] > best_score:
                best_idx = cand_idx
                best_score = score["score"]
                best_payload = score
                best_z = z
        if best_payload is not None:
            Gbest = build_toeplitz_from_markov(apply_markov_correction(m_blocks, basis_blocks, best_z), P, M)
            shadow["best_candidate_index"][step] = best_idx
            shadow["best_candidate_z"][step, :] = best_z
            shadow["best_s_pred"][step] = float(best_score)
            shadow["best_gain_drift"][step] = gain_drift(Gbest, G0)
            shadow["nominal_sse"][step] = best_payload["nominal_sse"]
            shadow["candidate_sse"][step] = best_payload["corrected_sse"]

        if config["run_adaptive_ls"]:
            z_star, _result, score = fit_markov_ls_correction(
                z_prev,
                z_bounds,
                nominal_history,
                m_blocks,
                basis_blocks,
                G0,
                A,
                C,
                P,
                M,
                Wy,
                float(config["lambda_z"]),
                step,
                int(config["prediction_window"]),
            )
            Gz = build_toeplitz_from_markov(apply_markov_correction(m_blocks, basis_blocks, z_star), P, M)
            drift = gain_drift(Gz, G0)
            accepted = bool(score["n_windows"] > 0 and score["score"] > float(config["s_pred_min"]) and drift <= float(config["gain_drift_max"]))
            shadow["ls_z"][step, :] = z_star
            shadow["ls_s_pred"][step] = float(score["score"])
            shadow["ls_gain_drift"][step] = float(drift)
            shadow["ls_accepted"][step] = int(accepted)
            shadow["ls_sse"][step] = score["corrected_sse"]
            shadow["output_nominal_sse"][step, :] = score["output_nominal_sse"]
            shadow["output_ls_sse"][step, :] = score["output_corrected_sse"]
            if accepted:
                z_prev = z_star
    return shadow


def plot_phase_outputs(config, ctx, nominal, markov, shadow, basis_labels, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    nFE = ctx["nFE"]
    t_step = np.arange(nFE) * ctx["delta_t"]
    t_line = np.arange(nFE + 1) * ctx["delta_t"]
    y_sp_phys = reverse_min_max(
        ctx["y_sp"] + ctx["y_ss_scaled"],
        ctx["system_data"]["data_min"][ctx["B_aug"].shape[1] :],
        ctx["system_data"]["data_max"][ctx["B_aug"].shape[1] :],
    )
    output_labels = POLYMER_SYSTEM_METADATA.get("output_labels", ["eta", "T"])
    input_labels = POLYMER_SYSTEM_METADATA.get("input_labels", ["Qc", "Qm"])

    def save(fig, name):
        path = fig_dir / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures.append(str(path))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(t_step, shadow["best_s_pred"], label="best candidate")
    ax.plot(t_step, shadow["ls_s_pred"], label="adaptive LS", alpha=0.85)
    ax.axhline(float(config["s_pred_min"]), color="k", linestyle="--", linewidth=1.0, label="S_min")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Prediction improvement score")
    ax.legend(loc="best")
    save(fig, "phase2_prediction_score_trace.png")

    valid_idx = shadow["best_candidate_index"][shadow["best_candidate_index"] >= 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if len(valid_idx):
        ax.hist(valid_idx, bins=np.arange(valid_idx.max() + 3) - 0.5)
    ax.set_xlabel("Candidate index")
    ax.set_ylabel("Selection count")
    save(fig, "phase2_candidate_selection_histogram.png")

    fig, axs = plt.subplots(max(1, len(basis_labels)), 1, figsize=(8.0, 2.2 + 1.6 * max(1, len(basis_labels))), sharex=True)
    axs = np.atleast_1d(axs)
    for idx, label in enumerate(basis_labels):
        axs[idx].plot(t_step, shadow["ls_z"][:, idx])
        axs[idx].set_ylabel(label)
    axs[-1].set_xlabel("Time (h)")
    save(fig, "phase3_z_trace.png")

    nom = shadow["nominal_sse"]
    ls = shadow["ls_sse"]
    improvement = np.where(np.isfinite(nom) & np.isfinite(ls), nom - ls, np.nan)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(t_step, improvement)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Nominal SSE - LS SSE")
    save(fig, "phase3_prediction_error_improvement.png")

    fig, axs = plt.subplots(ctx["C_aug"].shape[0], 1, figsize=(8.5, 5.8), sharex=True)
    axs = np.atleast_1d(axs)
    for idx, ax in enumerate(axs):
        ax.plot(t_line, nominal["y_phys"][:, idx], label="Nominal MPC")
        ax.plot(t_line, markov["y_phys"][:, idx], label="Markov corrected", alpha=0.85)
        ax.step(t_step, y_sp_phys[:nFE, idx], where="post", linestyle="--", label="Setpoint")
        ax.set_ylabel(output_labels[idx] if idx < len(output_labels) else f"y{idx + 1}")
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    save(fig, "phase4_outputs_compare.png")

    fig, axs = plt.subplots(ctx["B_aug"].shape[1], 1, figsize=(8.5, 5.4), sharex=True)
    axs = np.atleast_1d(axs)
    for idx, ax in enumerate(axs):
        ax.step(t_step, nominal["u_phys_log"][:, idx], where="post", label="Nominal MPC")
        ax.step(t_step, markov["u_phys_log"][:, idx], where="post", label="Markov corrected", alpha=0.85)
        ax.set_ylabel(input_labels[idx] if idx < len(input_labels) else f"u{idx + 1}")
    axs[-1].set_xlabel("Time (h)")
    axs[0].legend(loc="best")
    save(fig, "phase4_inputs_compare.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(nominal["avg_rewards"], "o-", label="Nominal MPC")
    ax.plot(markov["avg_rewards"], "s-", label="Markov corrected")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.legend(loc="best")
    save(fig, "phase4_reward_compare.png")

    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    axs[0].step(t_step, markov["accepted_log"], where="post")
    axs[0].set_ylabel("Accepted")
    axs[1].plot(t_step, markov["gain_drift_log"])
    axs[1].axhline(float(config["gain_drift_max"]), color="k", linestyle="--", linewidth=1.0)
    axs[1].set_ylabel("Gain drift")
    axs[1].set_xlabel("Time (h)")
    save(fig, "phase4_acceptance_and_gain_drift.png")

    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True)
    axs[0].step(t_step, markov["rl_action_source_log"], where="post")
    axs[0].set_yticks(sorted(MARKOV_ACTION_SOURCE))
    axs[0].set_yticklabels([MARKOV_ACTION_SOURCE[key] for key in sorted(MARKOV_ACTION_SOURCE)], fontsize=8)
    axs[0].set_ylabel("RL source")
    axs[1].plot(t_step, np.linalg.norm(markov["rl_requested_raw_action_log"], axis=1), label="requested")
    axs[1].plot(t_step, np.linalg.norm(markov["rl_executed_raw_action_log"], axis=1), label="executed", alpha=0.85)
    axs[1].set_xlabel("Time (h)")
    axs[1].set_ylabel("Raw action norm")
    axs[1].legend(loc="best")
    save(fig, "phase5_rl_action_source_and_norm.png")

    valid = np.isfinite(improvement)
    reward_delta = markov["rewards"] - nominal["rewards"]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.scatter(improvement[valid], reward_delta[valid], s=10, alpha=0.45)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction SSE improvement")
    ax.set_ylabel("Reward delta")
    save(fig, "phase4_prediction_improvement_vs_reward.png")

    return figures


def phase1_equivalence(config, ctx, G0, fig_dir):
    rng = np.random.default_rng(int(config["seed"]))
    P = int(config["predict_h"])
    M = int(config["cont_h"])
    A = ctx["A_aug"]
    B = ctx["B_aug"]
    C = ctx["C_aug"]
    x0 = rng.normal(0.0, 0.05, size=A.shape[0])
    U = rng.uniform(-0.05, 0.05, size=(M, B.shape[1]))
    y_state = state_space_predict(A, B, C, x0, U, P, M)
    y_lifted = lifted_predict(A, C, x0, G0, U, P, M)
    err = y_state - y_lifted
    metrics = {
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "relative_error": float(np.linalg.norm(err) / (np.linalg.norm(y_state) + 1.0e-12)),
    }
    if config["make_plots"]:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(7.8, 6.0), sharex=True)
        axs[0].plot(y_state, "o-", label="state-space")
        axs[0].plot(y_lifted, "s--", label="lifted")
        axs[0].set_ylabel("Stacked output")
        axs[0].legend(loc="best")
        axs[1].stem(np.arange(err.size), err)
        axs[1].set_xlabel("Stack index")
        axs[1].set_ylabel("Error")
        fig.savefig(Path(fig_dir) / "phase1_lifted_equivalence.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return metrics


def summarize_metrics(ctx, nominal, markov, shadow, phase1_metrics):
    nFE = ctx["nFE"]
    ny = ctx["C_aug"].shape[0]
    y_nom_scaled = nominal["y_scaled_dev"][1 : nFE + 1, :]
    y_mar_scaled = markov["y_scaled_dev"][1 : nFE + 1, :]
    e_nom = y_nom_scaled - ctx["y_sp"][:nFE, :]
    e_mar = y_mar_scaled - ctx["y_sp"][:nFE, :]
    mae_nom = np.mean(np.abs(e_nom), axis=0)
    mae_mar = np.mean(np.abs(e_mar), axis=0)
    move_nom = float(np.mean(np.linalg.norm(nominal["du_log"], axis=1)))
    move_mar = float(np.mean(np.linalg.norm(markov["du_log"], axis=1)))
    reward_delta = float(np.nanmean(markov["rewards"] - nominal["rewards"]))
    positive_shadow = float(np.mean(shadow["best_s_pred"] > 0.0))
    adaptive_accept = float(np.mean(shadow["ls_accepted"]))
    live_accept = float(np.mean(markov["accepted_log"]))
    td3_accept = float(np.mean(markov["rl_action_source_log"] == 2))
    ls_fallback = float(np.mean(markov["rl_action_source_log"] == 3))
    nominal_fallback = float(np.mean(markov["rl_action_source_log"] == 4))
    output_mae_delta = float(np.mean(mae_mar - mae_nom))
    input_move_delta = move_mar - move_nom
    rows = [
        ("Lifted equivalence max error", phase1_metrics["max_abs_error"], phase1_metrics["max_abs_error"] < 1.0e-8),
        ("Any positive shadow S_pred fraction", positive_shadow, positive_shadow > 0.05),
        ("Adaptive LS accepted fraction", adaptive_accept, adaptive_accept > 0.05),
        ("Live corrected accepted fraction", live_accept, live_accept > 0.01),
        ("TD3 accepted action fraction", td3_accept, td3_accept >= 0.0),
        ("Reward delta mean", reward_delta, reward_delta >= -1.0e-6),
        ("Output MAE delta", output_mae_delta, output_mae_delta <= 0.02),
        ("Input movement delta", input_move_delta, input_move_delta <= 0.02),
    ]
    verification = pd.DataFrame(rows, columns=["Check", "Value", "Pass"])
    summary = {
        "nFE": nFE,
        "n_outputs": ny,
        "phase1_max_abs_error": phase1_metrics["max_abs_error"],
        "positive_shadow_fraction": positive_shadow,
        "adaptive_ls_accepted_fraction": adaptive_accept,
        "live_corrected_accepted_fraction": live_accept,
        "td3_accepted_fraction": td3_accept,
        "ls_fallback_fraction": ls_fallback,
        "nominal_fallback_fraction": nominal_fallback,
        "rl_replay_push_count": int(np.sum(markov["rl_replay_pushed_log"])),
        "rl_train_update_count": int(np.sum(markov["rl_train_updated_log"])),
        "reward_delta_mean": reward_delta,
        "output_mae_nominal_mean": float(np.mean(mae_nom)),
        "output_mae_markov_mean": float(np.mean(mae_mar)),
        "output_mae_delta_mean": output_mae_delta,
        "input_movement_nominal": move_nom,
        "input_movement_markov": move_mar,
        "input_movement_delta": input_move_delta,
    }
    return summary, verification


def make_episode_reward_table(ctx, nominal, markov):
    rows = []
    nominal_episode_rewards = np.asarray(nominal["avg_rewards"], float).reshape(-1)
    markov_episode_rewards = np.asarray(markov["avg_rewards"], float).reshape(-1)
    completed = sorted((int(end_step), int(ep)) for end_step, ep in ctx["sub_episode_changes"].items())
    for idx, (end_step, ep) in enumerate(completed):
        if idx >= nominal_episode_rewards.size or idx >= markov_episode_rewards.size:
            continue
        rows.append(
            {
                "episode": ep,
                "complete_episode": True,
                "start_step": max(0, end_step - int(ctx["time_in_sub_episodes"]) + 1),
                "end_step": end_step,
                "nominal_mpc_avg_reward": nominal_episode_rewards[idx],
                "markov_td3_avg_reward": markov_episode_rewards[idx],
                "markov_minus_nominal": markov_episode_rewards[idx] - nominal_episode_rewards[idx],
            }
        )

    last_complete = completed[-1][0] if completed else -1
    if int(ctx["nFE"]) > last_complete + 1:
        start = last_complete + 1
        stop = int(ctx["nFE"])
        nominal_partial = float(np.mean(nominal["rewards"][start:stop])) if stop > start else np.nan
        markov_partial = float(np.mean(markov["rewards"][start:stop])) if stop > start else np.nan
        rows.append(
            {
                "episode": (completed[-1][1] + 1) if completed else 1,
                "complete_episode": False,
                "start_step": start,
                "end_step": stop - 1,
                "nominal_mpc_avg_reward": nominal_partial,
                "markov_td3_avg_reward": markov_partial,
                "markov_minus_nominal": markov_partial - nominal_partial,
            }
        )

    return pd.DataFrame(rows)


def make_bundle(config, ctx, nominal, markov, shadow, phase1_metrics, summary, figures, m_blocks, basis_labels):
    return {
        "method": "prediction_error_validated_markov_correction",
        "config": deepcopy(config),
        "A": ctx["system_data"]["A"],
        "B": ctx["system_data"]["B"],
        "C": ctx["system_data"]["C"],
        "A_aug": ctx["A_aug"],
        "B_aug": ctx["B_aug"],
        "C_aug": ctx["C_aug"],
        "M_blocks_nominal": m_blocks,
        "basis_family": config["basis_family"],
        "basis_labels": basis_labels,
        "reward_params": ctx["reward_params"],
        "phase1_metrics": phase1_metrics,
        "summary_metrics": summary,
        "episode_average_rewards": make_episode_reward_table(ctx, nominal, markov),
        "z_log": markov["z_log"],
        "z_proposed_log": markov["z_proposed_log"],
        "z_executed_log": markov["z_executed_log"],
        "rl_state_dim": markov["rl_state_dim"],
        "rl_action_dim": markov["rl_action_dim"],
        "rl_action_source_names": markov["rl_action_source_names"],
        "rl_state_log": markov["rl_state_log"],
        "rl_requested_raw_action_log": markov["rl_requested_raw_action_log"],
        "rl_executed_raw_action_log": markov["rl_executed_raw_action_log"],
        "rl_requested_z_log": markov["rl_requested_z_log"],
        "rl_ls_z_log": markov["rl_ls_z_log"],
        "rl_action_source_log": markov["rl_action_source_log"],
        "rl_decision_taken_log": markov["rl_decision_taken_log"],
        "rl_policy_source_log": markov["rl_policy_source_log"],
        "rl_replay_pushed_log": markov["rl_replay_pushed_log"],
        "rl_train_called_log": markov["rl_train_called_log"],
        "rl_train_updated_log": markov["rl_train_updated_log"],
        "rl_actor_loss_log": markov["rl_actor_loss_log"],
        "rl_critic_loss_log": markov["rl_critic_loss_log"],
        "rl_bc_loss_log": markov["rl_bc_loss_log"],
        "rl_test_step_log": markov["rl_test_step_log"],
        "rl_agent_checkpoint_path": markov["rl_agent_checkpoint_path"],
        "s_pred_log": markov["s_pred_log"],
        "gain_drift_log": markov["gain_drift_log"],
        "accepted_log": markov["accepted_log"],
        "fallback_log": markov["fallback_log"],
        "y_nominal": nominal["y_phys"],
        "u_nominal": nominal["u_phys_log"],
        "y_markov": markov["y_phys"],
        "u_markov": markov["u_phys_log"],
        "rewards_nominal": nominal["rewards"],
        "rewards_markov": markov["rewards"],
        "avg_rewards_nominal": nominal["avg_rewards"],
        "avg_rewards_markov": markov["avg_rewards"],
        "prediction_error_nominal_log": markov["prediction_error_nominal_log"],
        "prediction_error_markov_log": markov["prediction_error_markov_log"],
        "shadow": shadow,
        "figures": list(figures),
        "y": markov["y_phys"],
        "u": markov["u_phys_log"],
        "y_mpc": nominal["y_phys"],
        "u_mpc": nominal["u_phys_log"],
        "y_sp": ctx["y_sp"],
        "steady_states": ctx["steady_states"],
        "data_min": ctx["system_data"]["data_min"],
        "data_max": ctx["system_data"]["data_max"],
        "nFE": ctx["nFE"],
        "delta_t": ctx["delta_t"],
        "time_in_sub_episodes": ctx["time_in_sub_episodes"],
        "test_train_dict": ctx["test_train_dict"],
        "avg_rewards": markov["avg_rewards"],
        "rewards_step": markov["rewards"],
        "delta_y_storage": markov["y_scaled_dev"][1 : ctx["nFE"] + 1, :] - ctx["y_sp"][: ctx["nFE"], :],
        "delta_u_storage": markov["du_log"],
        "disturbance_profile": disturbance_profile_from_schedule(ctx["disturbance_schedule"]),
    }


def write_report(report_path, config, ctx, phase1_metrics, summary, verification, figures, result_path, comparison_dir=None):
    figure_lines = "\n".join(f"- `{Path(path).as_posix()}`" for path in figures)
    verification_md = dataframe_to_markdown(verification)
    text = f"""# Polymer Markov Correction Progress

## Objective

This note tracks the polymer-only TD3-assisted prototype for prediction-error-validated Markov-parameter correction of offset-free MPC. The prototype keeps the existing nonlinear polymer plant, nominal observer, scaling, disturbance schedule, and standard comparison plotting workflow unchanged. It tests whether finite-horizon input-output Markov corrections can reduce recent plant prediction error before they are allowed to affect the MPC action, while TD3 learns to propose bounded correction coordinates after the warm-start period.

## Files inspected

- `polymer_markov_correction_plan.md`
- `Simulation/mpc.py`
- `Simulation/system_functions.py`
- `systems/polymer/notebook_params.py`
- `systems/polymer/data_io.py`
- `utils/helpers.py`
- `utils/plotting.py`
- `utils/plotting_core.py`
- `utils/rewards.py`

## What the existing method was doing

The existing matrix-supervisor family perturbs the state-space model through low-dimensional multipliers, such as scaled versions of $A$ and $B$. Earlier Step 3D-style logic relied heavily on internal MPC cost comparisons. For polymer, the hard usefulness gate accepted essentially no live candidates in the prior analysis, so the method collapsed toward nominal MPC behavior.

## Why Step 3D is not reused as the main gate

The new prototype uses recent plant prediction error as the primary evidence. The old gate compared quantities such as:

$$ J_0(U_z^\\star) \\quad \\text{{and}} \\quad J_z(U_0^\\star)-J_z(U_z^\\star). $$

The adjusted Markov method first tests:

$$ \\|Y^{{\\mathrm{{meas}}}}-Y^0\\|_2^2 - \\|Y^{{\\mathrm{{meas}}}}-Y^z\\|_2^2. $$

The nominal MPC cost check remains only as a loose guard against catastrophic corrected actions.

## Mathematical formulation

The nominal offset-free model is:

$$ x_{{a,k+1}} = A_a x_{{a,k}} + B_a u_k^{{\\mathrm{{dev}}}}, \\qquad y_k^{{\\mathrm{{dev}}}} = C_a x_{{a,k}}. $$

For prediction horizon $P$ and control horizon $M$, the lifted prediction is:

$$ Y_k = Y_{{\\mathrm{{free}},k}} + G_z(P,M)U_k. $$

The correction changes finite-horizon Markov blocks rather than the state-space matrices:

$$ M_i(z_k)=M_{{i,0}}+\\sum_{{j=1}}^r z_{{j,k}}M_{{i,j}}^{{\\mathrm{{basis}}}}. $$

The prediction-error score is:

$$ S_{{\\mathrm{{pred}}}}(z)=\\sum_\\tau \\left(\\|W_y(Y_\\tau^{{\\mathrm{{meas}}}}-Y_\\tau^0)\\|_2^2-\\|W_y(Y_\\tau^{{\\mathrm{{meas}}}}-Y_\\tau^z)\\|_2^2\\right)-\\lambda_z\\|z\\|_2^2. $$

## Step-by-step algorithm

1. Load the existing polymer unified configuration. The default case is disturbed polymer with `n_tests=200`, `set_points_len=400`, `warm_start=10`, `predict_h=9`, `cont_h=3`, and the existing MPC penalties. The TD3 hyperparameters come from `POLYMER_MATRIX_DEFAULTS["td3_agent"]`.

2. Load the identified polymer model, scaling data, input bounds, steady states, observer poles, and canonical baseline MPC result path through the existing polymer helper layer. The nonlinear plant rollout still uses `PolymerCSTR`.

3. Build the offset-free augmented model and the lifted Markov representation. The code computes nominal Markov blocks $M_{{i,0}}=C_aA_a^{{i-1}}B_a$ and then builds the Toeplitz prediction matrix $G_0(P,M)$ using the same absolute scaled input-deviation convention as `MpcSolverGeneral`.

4. Validate the lifted prediction implementation before any live correction is allowed. A random input sequence is simulated both with the state-space recursion and with the lifted Toeplitz form. Live Markov correction is permitted only if the maximum absolute difference is below `1e-8`.

5. Run a nominal closed-loop rollout. At every time step, nominal MPC solves with $G_0$, the nonlinear polymer plant advances, the nominal observer updates, and the resulting trajectory becomes the history used by shadow scoring and LS diagnostics.

6. Score shadow Markov candidates without applying them. Candidate $z$ values are evaluated only on already observed windows, so future plant measurements are not used. This step estimates whether any Markov basis direction would have improved recent finite-horizon prediction error.

7. Fit the constrained LS teacher. The LS problem searches for a bounded $z_{{\\mathrm{{LS}}}}$ that reduces recent prediction error while paying the regularization penalty $\\lambda_z\\|z\\|_2^2$. The LS candidate is accepted only if its prediction score is positive enough and its lifted gain drift remains below the configured limit.

8. Build the TD3 Markov state for the live rollout. The state vector concatenates the nominal observer state, current tracking error, innovation, previous input deviation, previous executed correction $z$, current accepted LS teacher correction, LS prediction score, and LS gain drift.

9. Select a bounded TD3 correction action. During warm start, the baseline action is the LS teacher action. After `warm_start_step`, TD3 proposes a raw action in $[-1,1]^r$, which is mapped to the Markov correction by $z_{{\\mathrm{{TD3}}}}=z_{{\\mathrm{{bound}}}}a_{{\\mathrm{{TD3}}}}$.

10. Safety-filter the TD3 proposal. The runner solves corrected MPC with the TD3-corrected lifted matrix and accepts it only when the nominal solve succeeds, the corrected solve succeeds, the prediction score exceeds `s_pred_min`, gain drift is below `gain_drift_max`, and the loose nominal-cost guard passes.

11. Fall back in a fixed order if TD3 is not accepted. If TD3 fails the filter, the controller tries the accepted LS correction. If LS is unavailable or fails, the controller applies nominal MPC. The replay action is the executed action, not merely the requested TD3 action.

12. Advance the nonlinear plant and update replay. The selected first input move is applied to `PolymerCSTR`, the nominal observer updates, the shared unified relative-QR reward is computed, and the transition is pushed to TD3 replay on train steps. TD3 training starts only after the configured warm-start boundary.

13. Save artifacts in the polymer result tree. The run writes `input_data.pkl`, summary tables, verification tables, Markov diagnostic figures, RL diagnostic logs, and the TD3 checkpoint under `Polymer/Results/polymer_markov_corrected_mpc/<timestamp>/`. Standard MPC comparison plots are generated with `compare_mpc_rl_from_dirs()` under `Polymer/Results/polymer_markov_compare_disturb/<timestamp>/`.

14. Interpret results conservatively. The method is not considered successful from prediction score alone. A successful full run must pass lifted equivalence, show meaningful prediction-error improvement, avoid material output-MAE degradation, avoid excessive input movement, and produce acceptable reward relative to nominal MPC.

## Phase 1: lifted-prediction equivalence validation

The notebook validates the lifted absolute input-deviation convention against the state-space rollout. The pass threshold is `max_abs_error < 1e-8`.

Observed max absolute error: `{phase1_metrics["max_abs_error"]:.6e}`.

## Phase 2: shadow prediction-error scoring

Candidate Markov corrections are scored on nominal closed-loop history without executing corrected actions. The fraction of steps with positive best shadow score is `{summary["positive_shadow_fraction"]:.4f}`.

## Phase 3: adaptive LS Markov correction

The adaptive constrained LS correction estimates $z_k$ with bounds and regularization, then accepts it only when prediction improvement and gain-drift checks pass. The accepted fraction is `{summary["adaptive_ls_accepted_fraction"]:.4f}`.

## Phase 4: corrected MPC with loose safety guard

The live corrected controller solves both nominal and corrected lifted MPC. It executes the corrected first input only when prediction-error validation, gain-drift, and the loose nominal-cost guard pass. The live accepted fraction is `{summary["live_corrected_accepted_fraction"]:.4f}`.

## Phase 5: TD3 Markov proposal

TD3 is enabled by default through `run_rl_proposal=True` and proposes normalized Markov correction coordinates in $[-1,1]$. The runner maps the raw action to $z_k$, stores the executed action in replay, and uses the shared unified relative-QR reward. Constrained LS remains the warm-start teacher and safety fallback. The TD3 accepted fraction is `{summary["td3_accepted_fraction"]:.4f}`, the LS fallback fraction is `{summary["ls_fallback_fraction"]:.4f}`, and the nominal fallback fraction is `{summary["nominal_fallback_fraction"]:.4f}`. This run pushed `{summary["rl_replay_push_count"]}` replay transitions and recorded `{summary["rl_train_update_count"]}` TD3 critic updates.

## Result summary

{verification_md}

Result bundle: `{Path(result_path).as_posix() if result_path else "not saved"}`

Comparison directory: `{Path(comparison_dir).as_posix() if comparison_dir else "not generated"}`

## Smoke-run interpretation

This run has `nFE={ctx["nFE"]}` and `warm_start_step={ctx["warm_start_step"]}`. If the run is shorter than or equal to the warm-start boundary, TD3 is configured, checkpointed, and populated with replay data, but post-warm-start TD3 action acceptance and gradient updates are not expected. In that case, accepted Markov moves mainly validate the LS teacher and safety-gated execution path rather than TD3 closed-loop superiority.

## Figures

{figure_lines if figure_lines else "- Figures were not generated in this run."}

## Bugs, inconsistencies, or risks found

- Online prediction-error validation only uses windows whose full prediction horizon has already been observed. This avoids future leakage, but delays correction evidence by $P$ steps.
- The notebook keeps the observer nominal. If Markov corrections become large, prediction and observer dynamics can diverge.
- The full default run uses the existing polymer disturbed setting with `n_tests=200` and `set_points_len=400`, so a complete run may be long.

## Limitations

This is now an RL-active polymer Markov prototype: the TD3 training path is enabled by default, constrained LS remains a safety fallback, and shared RL/MPC modules are reused rather than modified. Closed-loop superiority is not claimed unless a saved full run shows improved or neutral output MAE, controlled input movement, and acceptable reward relative to nominal MPC.

## Next experiment

Run the full disturbed polymer default after the smoke path passes, then compare output-wise MAE, input movement, TD3 accepted fraction, LS fallback fraction, and reward delta against nominal MPC. If TD3 is mostly filtered out, inspect the replay losses and test whether the LS teacher action should be behavior-cloning weighted during early training.

## Remaining uncertainty

The method should be considered successful only if the saved metrics show lifted equivalence, meaningful prediction-error improvement, no material output MAE degradation, and controlled input movement.
"""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def dataframe_to_markdown(df):
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def run_polymer_markov_correction(overrides=None):
    config = build_config(overrides)
    ctx = build_context(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = ctx["result_root"] / timestamp
    fig_dir = result_dir
    if config["save_outputs"] or config["make_plots"]:
        result_dir.mkdir(parents=True, exist_ok=True)

    P = int(config["predict_h"])
    M = int(config["cont_h"])
    m_blocks = compute_markov_blocks(ctx["A_aug"], ctx["B_aug"], ctx["C_aug"], P)
    G0 = build_toeplitz_from_markov(m_blocks, P, M)
    basis_blocks, basis_labels = make_markov_basis(m_blocks, config["basis_family"])
    Wy = np.eye(P * ctx["C_aug"].shape[0])

    phase1_metrics = phase1_equivalence(config, ctx, G0, fig_dir)
    nominal = run_closed_loop(config, ctx, m_blocks, basis_blocks, G0, Wy, use_markov=False, print_progress=False)
    shadow = run_shadow_and_ls(config, ctx, nominal, m_blocks, basis_blocks, G0, Wy)
    can_run_live = bool(config["run_live_corrected_mpc"]) and phase1_metrics["max_abs_error"] < 1.0e-8
    markov = run_closed_loop(config, ctx, m_blocks, basis_blocks, G0, Wy, use_markov=can_run_live, print_progress=True)

    figures = [str(fig_dir / "phase1_lifted_equivalence.png")] if config["make_plots"] else []
    if config["make_plots"]:
        figures.extend(plot_phase_outputs(config, ctx, nominal, markov, shadow, basis_labels, fig_dir))

    summary, verification = summarize_metrics(ctx, nominal, markov, shadow, phase1_metrics)
    episode_rewards = make_episode_reward_table(ctx, nominal, markov)
    result_path = None
    comparison_dir = None
    if config["save_outputs"]:
        if config.get("rl_save_agent_checkpoint", True) and markov.get("_rl_agent") is not None:
            markov["rl_agent_checkpoint_path"] = markov["_rl_agent"].save(str(result_dir), prefix="td3_markov_agent")
        pd.DataFrame([summary]).to_csv(result_dir / "summary_metrics.csv", index=False)
        episode_rewards.to_csv(result_dir / "episode_average_rewards.csv", index=False)
        pd.DataFrame(
            {
                "best_s_pred": shadow["best_s_pred"],
                "ls_s_pred": shadow["ls_s_pred"],
                "best_gain_drift": shadow["best_gain_drift"],
                "ls_gain_drift": shadow["ls_gain_drift"],
            }
        ).to_csv(result_dir / "prediction_score_summary.csv", index=False)
        pd.DataFrame(
            {
                "ls_accepted": shadow["ls_accepted"],
                "live_accepted": markov["accepted_log"],
                "live_fallback": markov["fallback_log"],
                "live_gain_drift": markov["gain_drift_log"],
                "action_source": markov["rl_action_source_log"],
                "action_source_name": [
                    markov["rl_action_source_names"].get(int(value), "unknown")
                    for value in markov["rl_action_source_log"]
                ],
            }
        ).to_csv(result_dir / "acceptance_summary.csv", index=False)
        pd.DataFrame(
            {
                "requested_raw_norm": np.linalg.norm(markov["rl_requested_raw_action_log"], axis=1),
                "executed_raw_norm": np.linalg.norm(markov["rl_executed_raw_action_log"], axis=1),
                "requested_z_norm": np.linalg.norm(markov["rl_requested_z_log"], axis=1),
                "executed_z_norm": np.linalg.norm(markov["z_executed_log"], axis=1),
                "ls_z_norm": np.linalg.norm(markov["rl_ls_z_log"], axis=1),
                "action_source": markov["rl_action_source_log"],
                "decision_taken": markov["rl_decision_taken_log"],
                "replay_pushed": markov["rl_replay_pushed_log"],
                "train_called": markov["rl_train_called_log"],
                "train_updated": markov["rl_train_updated_log"],
                "actor_loss": markov["rl_actor_loss_log"],
                "critic_loss": markov["rl_critic_loss_log"],
                "test_step": markov["rl_test_step_log"],
            }
        ).to_csv(result_dir / "rl_diagnostics.csv", index=False)
        verification.to_csv(result_dir / "verification_table.csv", index=False)

        bundle = make_bundle(config, ctx, nominal, markov, shadow, phase1_metrics, summary, figures, m_blocks, basis_labels)
        result_path = result_dir / "input_data.pkl"
        with open(result_path, "wb") as handle:
            pickle.dump(bundle, handle)
        with open(result_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        if config["make_plots"]:
            baseline_path = Path(ctx["baseline_path"])
            if baseline_path.exists():
                try:
                    comparison_dir = Path(
                        compare_mpc_rl_from_dirs(
                            rl_dir=result_dir,
                            mpc_path_or_dir=baseline_path,
                            reward_fn=make_compare_reward_fn(ctx),
                            directory=ctx["result_base"],
                            prefix_name=f"polymer_markov_compare_{ctx['run_mode']}",
                            compare_mode=ctx["run_mode"],
                            start_episode=int(config["plot_start_episode"]),
                            n_inputs=ctx["B_aug"].shape[1],
                            save_pdf=bool(config["save_pdf"]),
                            style_profile=str(config["style_profile"]),
                        )
                    )
                    comparison_figures = [str(path) for path in sorted(comparison_dir.glob("*.png"))]
                    figures.extend(comparison_figures)
                    bundle["figures"] = list(figures)
                    bundle["comparison_dir"] = str(comparison_dir)
                    with open(result_path, "wb") as handle:
                        pickle.dump(bundle, handle)
                except Exception as exc:
                    print(f"Existing MPC comparison plotting helper skipped: {exc}")
            else:
                print(f"Existing MPC comparison plotting helper skipped: baseline not found at {baseline_path}")

    write_report(
        REPO_ROOT / "report" / "polymer_markov_correction_progress.md",
        config,
        ctx,
        phase1_metrics,
        summary,
        verification,
        figures,
        result_path,
        comparison_dir,
    )

    print("RL result directory:", result_dir)
    print("Comparison directory:", comparison_dir)
    print("\nVerification:")
    print(verification.to_string(index=False))
    print("\nEpisode average rewards:")
    print(episode_rewards.to_string(index=False))
    return {
        "config": config,
        "context": ctx,
        "phase1_metrics": phase1_metrics,
        "summary": summary,
        "verification": verification,
        "episode_reward_table": episode_rewards,
        "figure_dir": fig_dir,
        "result_dir": result_dir,
        "comparison_dir": comparison_dir,
        "figures": figures,
        "result_path": result_path,
        "report_path": REPO_ROOT / "report" / "polymer_markov_correction_progress.md",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the polymer Markov correction prototype.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional smoke-test step cap. Defaults to the full polymer config.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-save", action="store_true", help="Do not save result bundles or CSV summaries.")
    parser.add_argument("--basis-family", default=None, choices=["io_pair_gain", "input_channel_gain", "delay_shift"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.no_plots:
        overrides["make_plots"] = False
    if args.no_save:
        overrides["save_outputs"] = False
    if args.basis_family is not None:
        overrides["basis_family"] = args.basis_family
    run_polymer_markov_correction(overrides)

