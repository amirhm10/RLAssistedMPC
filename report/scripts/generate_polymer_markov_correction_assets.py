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
from systems.polymer.data_io import (
    canonical_baseline_path,
    ensure_polymer_directories,
    load_polymer_system_data,
    resolve_polymer_result_dir,
)
from systems.polymer.labels import POLYMER_SYSTEM_METADATA
from systems.polymer.notebook_params import POLYMER_BASELINE_DEFAULTS
from utils.helpers import (
    apply_min_max,
    build_polymer_disturbance_schedule,
    disturbance_profile_from_schedule,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
)
from utils.plotting import plot_baseline_mpc_results


def build_config(overrides=None):
    nb = deepcopy(POLYMER_BASELINE_DEFAULTS)
    run_mode = str(nb["run_mode"]).lower()
    run_profile = deepcopy(nb["run_profiles"][run_mode])
    controller = deepcopy(nb["controller"])

    cfg = {
        "run_mode": run_mode,
        "n_tests": int(run_profile["n_tests"]),
        "set_points_len": int(run_profile["set_points_len"]),
        "warm_start": int(run_profile["warm_start"]),
        "test_cycle": list(run_profile["test_cycle"]),
        "nominal_qi": float(run_profile["nominal_qi"]),
        "nominal_qs": float(run_profile["nominal_qs"]),
        "nominal_ha": float(run_profile["nominal_ha"]),
        "qi_change": float(run_profile["qi_change"]),
        "qs_change": float(run_profile["qs_change"]),
        "ha_change": float(run_profile["ha_change"]),
        "predict_h": int(controller["predict_h"]),
        "cont_h": int(controller["cont_h"]),
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
        "run_rl_proposal": False,
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
        "result_root": resolve_polymer_result_dir(REPO_ROOT) / "polymer_markov_corrected_mpc",
    }


def initialize_history(nFE, nx, ny, nu, z_dim):
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
    }


def run_closed_loop(config, ctx, m_blocks, basis_blocks, G0, Wy, *, use_markov):
    P = int(config["predict_h"])
    M = int(config["cont_h"])
    A = ctx["A_aug"]
    B = ctx["B_aug"]
    C = ctx["C_aug"]
    nFE = ctx["nFE"]
    ny, nu = C.shape[0], B.shape[1]
    z_dim = basis_blocks.shape[0]
    z_bounds = [(-float(config["z_bound"]), float(config["z_bound"])) for _ in range(z_dim)]
    history = initialize_history(nFE, A.shape[0], ny, nu, z_dim)

    system = PolymerCSTR(ctx["system_params"], ctx["design_params"], ctx["ss_inputs"], ctx["delta_t"])
    history["y_phys"][0, :] = system.current_output
    history["y_scaled_dev"][0, :] = apply_min_max(system.current_output, ctx["system_data"]["data_min"][nu:], ctx["system_data"]["data_max"][nu:]) - ctx["y_ss_scaled"]
    x_model = np.zeros(A.shape[0], dtype=float)
    x_init = np.zeros(M * nu, dtype=float)
    z_prev = np.zeros(z_dim, dtype=float)

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
        score = {
            "score": 0.0,
            "nominal_sse": np.nan,
            "corrected_sse": np.nan,
            "n_windows": 0,
        }
        drift = 0.0

        if use_markov and config["run_live_corrected_mpc"] and step >= P:
            z_star, _ls_result, score = fit_markov_ls_correction(
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
            mz = apply_markov_correction(m_blocks, basis_blocks, z_star)
            Gz = build_toeplitz_from_markov(mz, P, M)
            drift = gain_drift(Gz, G0)
            Uz, Jz, solz = solve_lifted_mpc(
                ctx["y_sp"][step],
                u_prev_dev,
                x_model,
                A,
                C,
                Gz,
                ctx["Q_out"],
                ctx["R_in"],
                P,
                M,
                ctx["bounds"],
                U0,
            )
            nominal_cost_of_Uz = lifted_mpc_cost(Uz, ctx["y_sp"][step], u_prev_dev, free_response(A, C, x_model, P), G0, ctx["Q_out"], ctx["R_in"], P, M)
            loose_tol = float(config["nominal_cost_absolute_tol"]) + float(config["nominal_cost_relative_tol"]) * abs(float(J0))
            accepted = bool(
                sol0.success
                and solz.success
                and score["score"] > float(config["s_pred_min"])
                and drift <= float(config["gain_drift_max"])
                and nominal_cost_of_Uz <= float(J0) + loose_tol
            )
            history["z_proposed_log"][step, :] = z_star
            if accepted:
                U_exec = Uz
                z_exec = z_star
                fallback = False
                z_prev = z_star

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

        yhat = C @ x_model
        innovation = history["y_scaled_dev"][step, :] - yhat
        x_model = A @ x_model + B @ u_dev + ctx["L"] @ innovation
        history["xhat_after"][step + 1, :] = x_model

        delta_y = history["y_scaled_dev"][step + 1, :] - ctx["y_sp"][step, :]
        history["rewards"][step] = legacy_mpc_reward(delta_y, du, ctx["y_sp"][step, :], ctx["Q_out"], ctx["R_in"])
        history["z_log"][step, :] = z_exec
        history["z_executed_log"][step, :] = z_exec
        history["s_pred_log"][step] = float(score["score"])
        history["gain_drift_log"][step] = float(drift)
        history["accepted_log"][step] = int(accepted)
        history["fallback_log"][step] = int(fallback)
        history["prediction_error_nominal_log"][step] = float(score["nominal_sse"]) if np.isfinite(score["nominal_sse"]) else np.nan
        history["prediction_error_markov_log"][step] = float(score["corrected_sse"]) if np.isfinite(score["corrected_sse"]) else np.nan

        if config["use_shifted_mpc_warm_start"]:
            U_mat = U_exec.reshape(M, nu)
            x_init = np.vstack([U_mat[1:, :], U_mat[-1:, :]]).reshape(-1)
        else:
            x_init = np.zeros(M * nu, dtype=float)

    history["avg_rewards"] = avg_by_episode(history["rewards"], ctx["sub_episode_changes"], ctx["time_in_sub_episodes"])
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
    output_mae_delta = float(np.mean(mae_mar - mae_nom))
    input_move_delta = move_mar - move_nom
    rows = [
        ("Lifted equivalence max error", phase1_metrics["max_abs_error"], phase1_metrics["max_abs_error"] < 1.0e-8),
        ("Any positive shadow S_pred fraction", positive_shadow, positive_shadow > 0.05),
        ("Adaptive LS accepted fraction", adaptive_accept, adaptive_accept > 0.05),
        ("Live corrected accepted fraction", live_accept, live_accept > 0.01),
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
        "reward_delta_mean": reward_delta,
        "output_mae_nominal_mean": float(np.mean(mae_nom)),
        "output_mae_markov_mean": float(np.mean(mae_mar)),
        "output_mae_delta_mean": output_mae_delta,
        "input_movement_nominal": move_nom,
        "input_movement_markov": move_mar,
        "input_movement_delta": input_move_delta,
    }
    return summary, verification


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
        "phase1_metrics": phase1_metrics,
        "summary_metrics": summary,
        "z_log": markov["z_log"],
        "z_proposed_log": markov["z_proposed_log"],
        "z_executed_log": markov["z_executed_log"],
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


def write_report(report_path, config, ctx, phase1_metrics, summary, verification, figures, result_path):
    figure_lines = "\n".join(f"- `{Path(path).as_posix()}`" for path in figures)
    verification_md = dataframe_to_markdown(verification)
    text = f"""# Polymer Markov Correction Progress

## Objective

This note tracks the polymer-only prototype for prediction-error-validated Markov-parameter correction of offset-free MPC. The prototype keeps the existing observer and nonlinear polymer plant workflow unchanged, and tests whether finite-horizon input-output Markov corrections reduce recent plant prediction error before they are allowed to affect the MPC action.

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

## Phase 1: lifted-prediction equivalence validation

The notebook validates the lifted absolute input-deviation convention against the state-space rollout. The pass threshold is `max_abs_error < 1e-8`.

Observed max absolute error: `{phase1_metrics["max_abs_error"]:.6e}`.

## Phase 2: shadow prediction-error scoring

Candidate Markov corrections are scored on nominal closed-loop history without executing corrected actions. The fraction of steps with positive best shadow score is `{summary["positive_shadow_fraction"]:.4f}`.

## Phase 3: adaptive LS Markov correction

The adaptive constrained LS correction estimates $z_k$ with bounds and regularization, then accepts it only when prediction improvement and gain-drift checks pass. The accepted fraction is `{summary["adaptive_ls_accepted_fraction"]:.4f}`.

## Phase 4: corrected MPC with loose safety guard

The live corrected controller solves both nominal and corrected lifted MPC. It executes the corrected first input only when prediction-error validation, gain-drift, and the loose nominal-cost guard pass. The live accepted fraction is `{summary["live_corrected_accepted_fraction"]:.4f}`.

## Phase 5: RL proposal scaffold

The notebook stores requested and executed $z$, fallback status, prediction score, and gain drift. Full TD3 proposal training remains disabled by default through `run_rl_proposal=False`.

## Result summary

{verification_md}

Result bundle: `{Path(result_path).as_posix() if result_path else "not saved"}`

## Figures

{figure_lines if figure_lines else "- Figures were not generated in this run."}

## Bugs, inconsistencies, or risks found

- Online prediction-error validation only uses windows whose full prediction horizon has already been observed. This avoids future leakage, but delays correction evidence by $P$ steps.
- The notebook keeps the observer nominal. If Markov corrections become large, prediction and observer dynamics can diverge.
- The full default run uses the existing polymer disturbed setting with `n_tests=200` and `set_points_len=400`, so a complete run may be long.

## Limitations

This is a first-pass polymer prototype. It does not modify shared RL/MPC code, does not prove closed-loop superiority, and does not train a TD3 proposal policy by default.

## Next experiment

Run the full disturbed polymer default after the smoke path passes, then compare output-wise MAE, input movement, accepted fraction, and reward delta against nominal MPC. If LS saturates at bounds or live acceptance is near zero, test the input-channel gain basis before widening the `io_pair_gain` bounds.

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
    fig_dir = REPO_ROOT / "report" / "figures" / f"polymer_markov_correction_{datetime.now().strftime('%Y%m%d')}"
    result_dir = ctx["result_root"] / timestamp
    if config["save_outputs"]:
        result_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

    P = int(config["predict_h"])
    M = int(config["cont_h"])
    m_blocks = compute_markov_blocks(ctx["A_aug"], ctx["B_aug"], ctx["C_aug"], P)
    G0 = build_toeplitz_from_markov(m_blocks, P, M)
    basis_blocks, basis_labels = make_markov_basis(m_blocks, config["basis_family"])
    Wy = np.eye(P * ctx["C_aug"].shape[0])

    phase1_metrics = phase1_equivalence(config, ctx, G0, fig_dir)
    nominal = run_closed_loop(config, ctx, m_blocks, basis_blocks, G0, Wy, use_markov=False)
    shadow = run_shadow_and_ls(config, ctx, nominal, m_blocks, basis_blocks, G0, Wy)
    can_run_live = bool(config["run_live_corrected_mpc"]) and phase1_metrics["max_abs_error"] < 1.0e-8
    markov = run_closed_loop(config, ctx, m_blocks, basis_blocks, G0, Wy, use_markov=can_run_live)

    figures = [str(fig_dir / "phase1_lifted_equivalence.png")] if config["make_plots"] else []
    if config["make_plots"]:
        figures.extend(plot_phase_outputs(config, ctx, nominal, markov, shadow, basis_labels, fig_dir))

    summary, verification = summarize_metrics(ctx, nominal, markov, shadow, phase1_metrics)
    result_path = None
    if config["save_outputs"]:
        pd.DataFrame([summary]).to_csv(fig_dir / "summary_metrics.csv", index=False)
        pd.DataFrame(
            {
                "best_s_pred": shadow["best_s_pred"],
                "ls_s_pred": shadow["ls_s_pred"],
                "best_gain_drift": shadow["best_gain_drift"],
                "ls_gain_drift": shadow["ls_gain_drift"],
            }
        ).to_csv(fig_dir / "prediction_score_summary.csv", index=False)
        pd.DataFrame(
            {
                "ls_accepted": shadow["ls_accepted"],
                "live_accepted": markov["accepted_log"],
                "live_fallback": markov["fallback_log"],
                "live_gain_drift": markov["gain_drift_log"],
            }
        ).to_csv(fig_dir / "acceptance_summary.csv", index=False)
        verification.to_csv(fig_dir / "verification_table.csv", index=False)

        bundle = make_bundle(config, ctx, nominal, markov, shadow, phase1_metrics, summary, figures, m_blocks, basis_labels)
        result_path = result_dir / "input_data.pkl"
        with open(result_path, "wb") as handle:
            pickle.dump(bundle, handle)
        with open(result_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        try:
            baseline_bundle = {
                "y": nominal["y_phys"],
                "u": nominal["u_phys_log"],
                "avg_rewards": nominal["avg_rewards"],
                "rewards_step": nominal["rewards"],
                "delta_y_storage": nominal["y_scaled_dev"][1 : ctx["nFE"] + 1, :] - ctx["y_sp"][: ctx["nFE"], :],
                "delta_u_storage": nominal["du_log"],
                "y_sp": ctx["y_sp"],
                "steady_states": ctx["steady_states"],
                "data_min": ctx["system_data"]["data_min"],
                "data_max": ctx["system_data"]["data_max"],
                "nFE": ctx["nFE"],
                "delta_t": ctx["delta_t"],
                "time_in_sub_episodes": ctx["time_in_sub_episodes"],
                "test_train_dict": ctx["test_train_dict"],
            }
            plot_baseline_mpc_results(
                baseline_bundle,
                {
                    "directory": str(fig_dir),
                    "prefix_name": "existing_plotter_nominal_baseline",
                    "start_episode": int(config["plot_start_episode"]),
                    "save_pdf": bool(config["save_pdf"]),
                    "style_profile": str(config["style_profile"]),
                },
            )
        except Exception as exc:
            print(f"Existing baseline plotting helper skipped: {exc}")

    write_report(
        REPO_ROOT / "report" / "polymer_markov_correction_progress.md",
        config,
        ctx,
        phase1_metrics,
        summary,
        verification,
        figures,
        result_path,
    )

    print(verification.to_string(index=False))
    return {
        "config": config,
        "context": ctx,
        "phase1_metrics": phase1_metrics,
        "summary": summary,
        "verification": verification,
        "figure_dir": fig_dir,
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
