from __future__ import annotations

from copy import deepcopy

import numpy as np
import scipy.optimize as spo

from TD3Agent.agent import TD3Agent
from utils.agent_step_runtime import replay_train_continuous_agent, select_continuous_action
from utils.helpers import (
    apply_min_max,
    build_polymer_disturbance_schedule,
    disturbance_profile_from_schedule,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
    shift_control_sequence,
    step_system_with_disturbance,
)
from utils.multiplier_sensitivity import build_markov_matrix
from utils.observer import compute_observer_gain


MARKOV_ACTION_SOURCE = {
    0: "nominal_no_markov",
    1: "warm_start_ls",
    2: "td3_accepted",
    3: "ls_fallback",
    4: "nominal_fallback",
    5: "ls_no_rl",
}


def compute_markov_blocks(A, B, C, predict_h):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    predict_h = int(predict_h)
    ny = int(C.shape[0])
    nu = int(B.shape[1])
    flat = build_markov_matrix(A, B, C, predict_h)
    return np.asarray(flat, float).reshape(predict_h, ny, nu)


def build_toeplitz_from_markov(m_blocks, predict_h, control_horizon, hold_last=True):
    m_blocks = np.asarray(m_blocks, float)
    predict_h = int(predict_h)
    control_horizon = int(control_horizon)
    ny = int(m_blocks.shape[1])
    nu = int(m_blocks.shape[2])
    G = np.zeros((predict_h * ny, control_horizon * nu), dtype=float)
    for row_idx in range(predict_h):
        for lag_idx in range(row_idx + 1):
            col_idx = lag_idx if (lag_idx < control_horizon or not hold_last) else control_horizon - 1
            if col_idx >= control_horizon:
                continue
            row = slice(row_idx * ny, (row_idx + 1) * ny)
            col = slice(col_idx * nu, (col_idx + 1) * nu)
            G[row, col] += m_blocks[row_idx - lag_idx]
    return G


def free_response(A, C, x0, predict_h):
    A = np.asarray(A, float)
    C = np.asarray(C, float)
    x = np.asarray(x0, float).reshape(-1)
    ys = []
    for _ in range(int(predict_h)):
        x = A @ x
        ys.append(C @ x)
    return np.asarray(ys, float).reshape(-1)


def state_space_predict(A, B, C, x0, U_seq, predict_h, control_horizon):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    U = np.asarray(U_seq, float).reshape(int(control_horizon), B.shape[1])
    x = np.asarray(x0, float).reshape(-1)
    ys = []
    for horizon_idx in range(int(predict_h)):
        move_idx = horizon_idx if horizon_idx < int(control_horizon) else int(control_horizon) - 1
        x = A @ x + B @ U[move_idx]
        ys.append(C @ x)
    return np.asarray(ys, float).reshape(-1)


def lifted_predict(A, C, x0, G, U_seq, predict_h):
    return free_response(A, C, x0, predict_h) + np.asarray(G, float) @ np.asarray(U_seq, float).reshape(-1)


def make_markov_basis(m_blocks, basis_family):
    m_blocks = np.asarray(m_blocks, float)
    _, ny, nu = m_blocks.shape
    family = str(basis_family).lower()
    basis = []
    labels = []

    if family == "io_pair_gain":
        for output_idx in range(ny):
            for input_idx in range(nu):
                basis_block = np.zeros_like(m_blocks)
                basis_block[:, output_idx, input_idx] = m_blocks[:, output_idx, input_idx]
                basis.append(basis_block)
                labels.append(f"y{output_idx + 1}_u{input_idx + 1}")
    elif family == "input_channel_gain":
        for input_idx in range(nu):
            basis_block = np.zeros_like(m_blocks)
            basis_block[:, :, input_idx] = m_blocks[:, :, input_idx]
            basis.append(basis_block)
            labels.append(f"u{input_idx + 1}")
    elif family == "delay_shift":
        shifted = np.zeros_like(m_blocks)
        shifted[1:, :, :] = m_blocks[:-1, :, :]
        basis.append(shifted - m_blocks)
        labels.append("delay_shift")
    else:
        raise ValueError("basis_family must be 'io_pair_gain', 'input_channel_gain', or 'delay_shift'.")

    return np.asarray(basis, float), labels


def apply_markov_correction(m_blocks, basis_blocks, z):
    corrected = np.asarray(m_blocks, float).copy()
    basis_blocks = np.asarray(basis_blocks, float)
    z = np.asarray(z, float).reshape(-1)
    for idx, z_value in enumerate(z):
        corrected += float(z_value) * basis_blocks[idx]
    return corrected


def gain_drift(G_candidate, G_nominal, Wy=None, Wu=None, eps=1.0e-12):
    G_candidate = np.asarray(G_candidate, float)
    G_nominal = np.asarray(G_nominal, float)
    left = np.eye(G_nominal.shape[0]) if Wy is None else np.asarray(Wy, float)
    right = np.eye(G_nominal.shape[1]) if Wu is None else np.asarray(Wu, float)
    num = np.linalg.norm(left @ (G_candidate - G_nominal) @ right, ord="fro")
    den = np.linalg.norm(left @ G_nominal @ right, ord="fro") + float(eps)
    return float(num / den)


def available_prediction_indices(current_step, predict_h, prediction_window):
    last_tau = int(current_step) - int(predict_h)
    if last_tau < 0:
        return []
    first_tau = max(0, last_tau - int(prediction_window) + 1)
    return list(range(first_tau, last_tau + 1))


def real_u_window(u_dev_log, tau, control_horizon):
    control_horizon = int(control_horizon)
    u_dev_log = np.asarray(u_dev_log, float)
    U = np.zeros((control_horizon, u_dev_log.shape[1]), dtype=float)
    for idx in range(control_horizon):
        src = min(int(tau) + idx, u_dev_log.shape[0] - 1)
        U[idx, :] = u_dev_log[src, :]
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
    predict_h,
    control_horizon,
    Wy,
    lambda_z,
    current_step,
    prediction_window,
):
    indices = available_prediction_indices(current_step, predict_h, prediction_window)
    ny = int(C.shape[0])
    if not indices:
        return {
            "score": 0.0,
            "nominal_sse": np.nan,
            "corrected_sse": np.nan,
            "n_windows": 0,
            "output_nominal_sse": np.full(ny, np.nan, dtype=float),
            "output_corrected_sse": np.full(ny, np.nan, dtype=float),
        }

    mz = apply_markov_correction(m_blocks, basis_blocks, z)
    Gz = build_toeplitz_from_markov(mz, predict_h, control_horizon)
    Wy = np.eye(predict_h * ny) if Wy is None else np.asarray(Wy, float)
    nominal_sse = 0.0
    corrected_sse = 0.0
    output_nominal_sse = np.zeros(ny, dtype=float)
    output_corrected_sse = np.zeros(ny, dtype=float)

    for tau in indices:
        U_real = real_u_window(history["u_dev_log"], tau, control_horizon)
        y_meas = history["y_scaled_dev"][tau + 1 : tau + predict_h + 1, :].reshape(-1)
        y_free = free_response(A, C, history["xhat_before"][tau], predict_h)
        e0 = y_meas - (y_free + G0 @ U_real.reshape(-1))
        ez = y_meas - (y_free + Gz @ U_real.reshape(-1))
        nominal_sse += float(np.sum((Wy @ e0) ** 2))
        corrected_sse += float(np.sum((Wy @ ez) ** 2))
        output_nominal_sse += np.sum(e0.reshape(predict_h, ny) ** 2, axis=0)
        output_corrected_sse += np.sum(ez.reshape(predict_h, ny) ** 2, axis=0)

    penalty = float(lambda_z) * float(np.sum(np.asarray(z, float) ** 2))
    return {
        "score": float(nominal_sse - corrected_sse - penalty),
        "nominal_sse": float(nominal_sse),
        "corrected_sse": float(corrected_sse),
        "n_windows": len(indices),
        "output_nominal_sse": output_nominal_sse,
        "output_corrected_sse": output_corrected_sse,
    }


def fit_markov_ls_correction(
    z0,
    bounds,
    history,
    m_blocks,
    basis_blocks,
    G0,
    A,
    C,
    predict_h,
    control_horizon,
    Wy,
    lambda_z,
    current_step,
    prediction_window,
):
    def objective(z):
        score = prediction_improvement_score(
            z=z,
            history=history,
            m_blocks=m_blocks,
            basis_blocks=basis_blocks,
            G0=G0,
            A=A,
            C=C,
            predict_h=predict_h,
            control_horizon=control_horizon,
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
        predict_h=predict_h,
        control_horizon=control_horizon,
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


def build_test_flags(nFE, test_train_dict):
    flags = np.zeros(int(nFE), dtype=bool)
    active = False
    starts = sorted((int(k), bool(v)) for k, v in dict(test_train_dict).items())
    start_idx = 0
    for step in range(int(nFE)):
        while start_idx < len(starts) and starts[start_idx][0] <= step:
            active = starts[start_idx][1]
            start_idx += 1
        flags[step] = active
    return flags


def avg_by_episode(rewards, sub_episode_changes, time_in_sub_episodes):
    rewards = np.asarray(rewards, float)
    avg = []
    for idx in sorted(sub_episode_changes):
        start = max(0, int(idx) - int(time_in_sub_episodes) + 1)
        stop = int(idx) + 1
        avg.append(float(np.mean(rewards[start:stop])) if stop > start else float("nan"))
    return np.asarray(avg, float)


def make_td3_markov_agent(config, state_dim, action_dim, *, set_points_len):
    td3_cfg = deepcopy(config["td3_agent"])
    buffer_size = int(td3_cfg.get("buffer_size", 40_000))
    recent_window = td3_cfg.get("replay_recent_window")
    if recent_window is None:
        recent_window = min(
            buffer_size,
            int(td3_cfg.get("replay_recent_window_mult", 5)) * int(set_points_len),
        )

    return TD3Agent(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        seed=td3_cfg.get("seed"),
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


def lifted_mpc_cost(U_flat, y_sp, u_prev_dev, Y_free, G, Q_out, R_in, predict_h, control_horizon):
    U = np.asarray(U_flat, float).reshape(int(control_horizon), -1)
    del predict_h
    ny = len(np.asarray(y_sp, float).reshape(-1))
    y_pred = (np.asarray(Y_free, float) + np.asarray(G, float) @ U.reshape(-1)).reshape(-1, ny)
    y_err = y_pred - np.asarray(y_sp, float).reshape(1, ny)
    U_prev = np.vstack([np.asarray(u_prev_dev, float).reshape(1, -1), U[:-1, :]])
    dU = U - U_prev
    return float(
        np.sum(np.asarray(Q_out, float).reshape(1, ny) * y_err**2)
        + np.sum(np.asarray(R_in, float).reshape(1, -1) * dU**2)
    )


def solve_lifted_mpc(y_sp, u_prev_dev, x0_model, A, C, G, Q_out, R_in, predict_h, control_horizon, bounds, x_init):
    Y_free = free_response(A, C, x0_model, predict_h)
    objective = lambda U_flat: lifted_mpc_cost(
        U_flat,
        y_sp,
        u_prev_dev,
        Y_free,
        G,
        Q_out,
        R_in,
        predict_h,
        control_horizon,
    )
    sol = spo.minimize(objective, np.asarray(x_init, float), method="SLSQP", bounds=bounds)
    if not sol.success:
        return np.asarray(x_init, float), float(objective(x_init)), sol
    return np.asarray(sol.x, float), float(sol.fun), sol


def solve_nominal_mpc_step(mpc_obj, y_sp, u_prev_dev, x0_model, bounds, x_init):
    sol = spo.minimize(
        lambda x: mpc_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
        np.asarray(x_init, float),
        bounds=bounds,
        constraints=[],
    )
    return np.asarray(sol.x, float), float(sol.fun), sol


def solve_nominal_reference_step(
    *,
    nominal_solver_mode,
    mpc_obj,
    y_sp,
    u_prev_dev,
    x0_model,
    A,
    C,
    G0,
    Q_out,
    R_in,
    predict_h,
    control_horizon,
    bounds,
    x_init,
):
    mode = str(nominal_solver_mode).strip().lower()
    if mode == "state_space_shared":
        return solve_nominal_mpc_step(mpc_obj, y_sp, u_prev_dev, x0_model, bounds, x_init)
    if mode == "lifted_g0_prototype":
        return solve_lifted_mpc(
            y_sp,
            u_prev_dev,
            x0_model,
            A,
            C,
            G0,
            Q_out,
            R_in,
            predict_h,
            control_horizon,
            bounds,
            x_init,
        )
    raise ValueError(
        "nominal_solver_mode must be 'state_space_shared' or 'lifted_g0_prototype'."
    )


def initialize_history(nFE, nx, ny, nu, z_dim, control_horizon, rl_state_dim=0):
    return {
        "xhat_before": np.zeros((nFE, nx), dtype=float),
        "xhat_after": np.zeros((nFE + 1, nx), dtype=float),
        "yhat": np.zeros((ny, nFE), dtype=float),
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
        "u_sequence_nominal_log": np.empty((nFE, control_horizon * nu), dtype=float),
        "u_sequence_requested_log": np.full((nFE, control_horizon * nu), np.nan, dtype=float),
        "u_sequence_ls_log": np.full((nFE, control_horizon * nu), np.nan, dtype=float),
        "u_sequence_executed_log": np.empty((nFE, control_horizon * nu), dtype=float),
        "u0_nominal_log": np.empty((nFE, nu), dtype=float),
        "u0_requested_log": np.full((nFE, nu), np.nan, dtype=float),
        "u0_ls_log": np.full((nFE, nu), np.nan, dtype=float),
        "u0_executed_log": np.empty((nFE, nu), dtype=float),
        "u0_requested_minus_nominal_log": np.full((nFE, nu), np.nan, dtype=float),
        "u0_ls_minus_nominal_log": np.full((nFE, nu), np.nan, dtype=float),
        "u0_executed_minus_nominal_log": np.empty((nFE, nu), dtype=float),
        "u0_requested_minus_nominal_norm_log": np.full(nFE, np.nan, dtype=float),
        "u0_ls_minus_nominal_norm_log": np.full(nFE, np.nan, dtype=float),
        "u0_executed_minus_nominal_norm_log": np.full(nFE, np.nan, dtype=float),
        "u_sequence_requested_minus_nominal_norm_log": np.full(nFE, np.nan, dtype=float),
        "u_sequence_ls_minus_nominal_norm_log": np.full(nFE, np.nan, dtype=float),
        "u_sequence_executed_minus_nominal_norm_log": np.full(nFE, np.nan, dtype=float),
        "nominal_cost_log": np.full(nFE, np.nan, dtype=float),
        "requested_candidate_native_cost_log": np.full(nFE, np.nan, dtype=float),
        "requested_candidate_nominal_cost_log": np.full(nFE, np.nan, dtype=float),
        "requested_cost_margin_log": np.full(nFE, np.nan, dtype=float),
        "requested_cost_guard_pass_log": np.full(nFE, -1, dtype=int),
        "requested_gain_drift_log": np.full(nFE, np.nan, dtype=float),
        "requested_prediction_score_log": np.full(nFE, np.nan, dtype=float),
        "ls_candidate_native_cost_log": np.full(nFE, np.nan, dtype=float),
        "ls_candidate_nominal_cost_log": np.full(nFE, np.nan, dtype=float),
        "ls_cost_margin_log": np.full(nFE, np.nan, dtype=float),
        "ls_cost_guard_pass_log": np.full(nFE, -1, dtype=int),
        "ls_gain_drift_log": np.full(nFE, np.nan, dtype=float),
        "ls_prediction_score_log": np.full(nFE, np.nan, dtype=float),
        "executed_candidate_native_cost_log": np.full(nFE, np.nan, dtype=float),
        "executed_candidate_nominal_cost_log": np.full(nFE, np.nan, dtype=float),
        "executed_cost_margin_log": np.full(nFE, np.nan, dtype=float),
        "executed_cost_guard_pass_log": np.full(nFE, -1, dtype=int),
        "executed_gain_drift_log": np.full(nFE, np.nan, dtype=float),
        "executed_prediction_score_log": np.full(nFE, np.nan, dtype=float),
        "avg_rewards": np.asarray([], dtype=float),
        "rl_state_dim": int(rl_state_dim),
        "rl_action_dim": int(z_dim),
        "rl_action_source_names": dict(MARKOV_ACTION_SOURCE),
        "rl_agent_checkpoint_path": None,
    }


def _flatten_control_sequence(U_seq):
    return np.asarray(U_seq, float).reshape(-1)


def _record_nominal_stage(history, step, U_nominal, nominal_cost, nu):
    U_flat = _flatten_control_sequence(U_nominal)
    history["u_sequence_nominal_log"][step, :] = U_flat
    history["u0_nominal_log"][step, :] = U_flat[:nu]
    history["nominal_cost_log"][step] = float(nominal_cost)


def _record_candidate_stage(history, step, prefix, U_candidate, candidate_eval, candidate_score, U_nominal, nominal_cost, nu):
    U_flat = _flatten_control_sequence(U_candidate)
    U_nom_flat = _flatten_control_sequence(U_nominal)
    diff_flat = U_flat - U_nom_flat
    diff_first = diff_flat[:nu]
    history[f"u_sequence_{prefix}_log"][step, :] = U_flat
    history[f"u0_{prefix}_log"][step, :] = U_flat[:nu]
    history[f"u0_{prefix}_minus_nominal_log"][step, :] = diff_first
    history[f"u0_{prefix}_minus_nominal_norm_log"][step] = float(np.linalg.norm(diff_first))
    history[f"u_sequence_{prefix}_minus_nominal_norm_log"][step] = float(np.linalg.norm(diff_flat))
    if candidate_eval is not None:
        history[f"{prefix}_candidate_native_cost_log"][step] = float(candidate_eval["J"])
        history[f"{prefix}_candidate_nominal_cost_log"][step] = float(candidate_eval["nominal_cost"])
        history[f"{prefix}_cost_margin_log"][step] = float(candidate_eval["nominal_cost"] - float(nominal_cost))
        history[f"{prefix}_cost_guard_pass_log"][step] = int(bool(candidate_eval["cost_guard_pass"]))
        history[f"{prefix}_gain_drift_log"][step] = float(candidate_eval["drift"])
    if candidate_score is not None:
        history[f"{prefix}_prediction_score_log"][step] = float(candidate_score["score"])


def phase1_equivalence_metrics(config, ctx, G0):
    predict_h = int(config["predict_h"])
    control_horizon = int(config["cont_h"])
    A = np.asarray(ctx["A_aug"], float)
    B = np.asarray(ctx["B_aug"], float)
    C = np.asarray(ctx["C_aug"], float)
    x0 = np.linspace(0.01, 0.01 * A.shape[0], A.shape[0], dtype=float)
    U = np.linspace(
        0.0,
        min(0.05, float(np.max(np.abs(ctx["b_max"])))),
        control_horizon * B.shape[1],
        dtype=float,
    )
    y_state = state_space_predict(A, B, C, x0, U, predict_h, control_horizon)
    y_lifted = lifted_predict(A, C, x0, G0, U, predict_h)
    diff = y_state - y_lifted
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "rms_error": float(np.sqrt(np.mean(diff**2))),
        "y_state": y_state.reshape(predict_h, C.shape[0]),
        "y_lifted": y_lifted.reshape(predict_h, C.shape[0]),
    }


def run_shadow_and_ls_debug(config, ctx, nominal_history, m_blocks, basis_blocks, G0, Wy):
    predict_h = int(config["predict_h"])
    control_horizon = int(config["cont_h"])
    A = np.asarray(ctx["A_aug"], float)
    C = np.asarray(ctx["C_aug"], float)
    nFE = int(ctx["nFE"])
    z_dim = int(basis_blocks.shape[0])
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
                predict_h=predict_h,
                control_horizon=control_horizon,
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
            G_best = build_toeplitz_from_markov(
                apply_markov_correction(m_blocks, basis_blocks, best_z),
                predict_h,
                control_horizon,
            )
            shadow["best_candidate_index"][step] = best_idx
            shadow["best_candidate_z"][step, :] = best_z
            shadow["best_s_pred"][step] = float(best_score)
            shadow["best_gain_drift"][step] = gain_drift(G_best, G0)
            shadow["nominal_sse"][step] = best_payload["nominal_sse"]
            shadow["candidate_sse"][step] = best_payload["corrected_sse"]

        if bool(config.get("run_adaptive_ls", True)):
            z_star, _ls_result, score = fit_markov_ls_correction(
                z_prev,
                z_bounds,
                nominal_history,
                m_blocks,
                basis_blocks,
                G0,
                A,
                C,
                predict_h,
                control_horizon,
                Wy,
                float(config["lambda_z"]),
                step,
                int(config["prediction_window"]),
            )
            Gz = build_toeplitz_from_markov(
                apply_markov_correction(m_blocks, basis_blocks, z_star),
                predict_h,
                control_horizon,
            )
            drift = gain_drift(Gz, G0)
            accepted = bool(
                score["n_windows"] > 0
                and score["score"] > float(config["s_pred_min"])
                and drift <= float(config["gain_drift_max"])
            )
            shadow["ls_z"][step, :] = z_star
            shadow["ls_s_pred"][step] = float(score["score"])
            shadow["ls_gain_drift"][step] = float(drift)
            shadow["ls_accepted"][step] = int(accepted)
            shadow["ls_sse"][step] = score["corrected_sse"]
            if accepted:
                z_prev = z_star
    return shadow


def build_runtime_context(markov_cfg, runtime_ctx):
    system_data = runtime_ctx["system_data"]
    A_aug = np.asarray(runtime_ctx["A_aug"], float)
    B_aug = np.asarray(runtime_ctx["B_aug"], float)
    C_aug = np.asarray(runtime_ctx["C_aug"], float)
    n_inputs = int(B_aug.shape[1])
    data_min = np.asarray(runtime_ctx["data_min"], float)
    data_max = np.asarray(runtime_ctx["data_max"], float)
    steady_states = runtime_ctx["steady_states"]
    y_sp_scenario = np.asarray(runtime_ctx["y_sp_scenario"], float)
    reward_fn = runtime_ctx["reward_fn"]
    reward_params = runtime_ctx.get("reward_params", {})
    system_metadata = runtime_ctx.get("system_metadata")

    (
        y_sp,
        nFE,
        sub_episode_changes_dict,
        time_in_sub_episodes,
        test_train_dict,
        warm_start_step,
        qi,
        qs,
        ha,
    ) = generate_setpoints_training_rl_gradually(
        y_sp_scenario,
        int(markov_cfg["n_tests"]),
        int(markov_cfg["set_points_len"]),
        int(markov_cfg["warm_start"]),
        list(markov_cfg["test_cycle"]),
        float(markov_cfg["nominal_qi"]),
        float(markov_cfg["nominal_qs"]),
        float(markov_cfg["nominal_ha"]),
        float(markov_cfg["qi_change"]),
        float(markov_cfg["qs_change"]),
        float(markov_cfg["ha_change"]),
    )

    max_steps = markov_cfg.get("max_steps")
    if max_steps is not None:
        nFE = min(int(nFE), int(max_steps))
        y_sp = y_sp[:nFE, :]
        qi = np.asarray(qi, float)[:nFE]
        qs = np.asarray(qs, float)[:nFE]
        ha = np.asarray(ha, float)[:nFE]
        sub_episode_changes_dict = {k: v for k, v in sub_episode_changes_dict.items() if int(k) < int(nFE)}
        test_train_dict = {k: v for k, v in test_train_dict.items() if int(k) < int(nFE)}

    observer_alignment = str(
        markov_cfg.get("observer_update_alignment", "legacy_previous_measurement")
    ).strip().lower()
    if observer_alignment not in {"legacy_previous_measurement", "predictor_corrector_current"}:
        raise ValueError(
            "observer_update_alignment must be 'legacy_previous_measurement' or 'predictor_corrector_current'."
        )

    poles = np.asarray(runtime_ctx["poles"], float)
    bounds = tuple(
        (float(markov_cfg["b_min"][inp]), float(markov_cfg["b_max"][inp]))
        for _ in range(int(markov_cfg["cont_h"]))
        for inp in range(n_inputs)
    )
    ss_scaled_inputs = np.asarray(system_data["u_ss_scaled"], float)
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    disturbance_schedule = None
    if str(markov_cfg["run_mode"]).lower() == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = {
                # Temporary Step-3 parity probe:
                # PolymerCSTR reads plant disturbances from Qi/Qs/hA attributes.
                # The consolidated helper path was emitting lowercase qi/qs/ha,
                # which does not reproduce the legacy live-step semantics.
                "Qi": np.asarray(qi, float),
                "Qs": np.asarray(qs, float),
                "hA": np.asarray(ha, float),
            }

    return {
        "system_factory": runtime_ctx["system_factory"],
        "system_stepper": runtime_ctx.get("system_stepper"),
        "delta_t": float(runtime_ctx.get("delta_t", getattr(runtime_ctx["system_factory"](), "delta_t", 0.5))),
        "system_data": system_data,
        "system_metadata": system_metadata,
        "reward_fn": reward_fn,
        "reward_params": reward_params,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "MPC_obj": runtime_ctx["MPC_obj"],
        "steady_states": steady_states,
        "data_min": data_min,
        "data_max": data_max,
        "n_inputs": n_inputs,
        "n_outputs": int(C_aug.shape[0]),
        "bounds": bounds,
        "b_min": np.asarray(markov_cfg["b_min"], float),
        "b_max": np.asarray(markov_cfg["b_max"], float),
        "ss_scaled_inputs": ss_scaled_inputs,
        "y_ss_scaled": y_ss_scaled,
        "y_sp": np.asarray(y_sp, float),
        "nFE": int(nFE),
        "sub_episode_changes_dict": sub_episode_changes_dict,
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "test_train_dict": test_train_dict,
        "warm_start_step": int(warm_start_step),
        "qi": np.asarray(qi, float),
        "qs": np.asarray(qs, float),
        "ha": np.asarray(ha, float),
        "disturbance_schedule": disturbance_schedule,
        "observer_alignment": observer_alignment,
        "poles": poles,
        "run_mode": str(markov_cfg["run_mode"]).lower(),
    }


def run_single_closed_loop(config, ctx, m_blocks, basis_blocks, G0, Wy, *, use_markov, print_progress=False):
    predict_h = int(config["predict_h"])
    control_horizon = int(config["cont_h"])
    nominal_solver_mode = str(config.get("nominal_solver_mode", "state_space_shared")).strip().lower()
    force_td3_execute = bool(config.get("force_td3_execute", False))
    A = np.asarray(ctx["A_aug"], float)
    B = np.asarray(ctx["B_aug"], float)
    C = np.asarray(ctx["C_aug"], float)
    nFE = int(ctx["nFE"])
    ny = int(C.shape[0])
    nu = int(B.shape[1])
    z_dim = int(basis_blocks.shape[0])
    z_bounds = [(-float(config["z_bound"]), float(config["z_bound"])) for _ in range(z_dim)]
    rl_state_dim = int(A.shape[0] + ny + ny + nu + z_dim + z_dim + 2)
    history = initialize_history(nFE, A.shape[0], ny, nu, z_dim, control_horizon, rl_state_dim)
    test_flags = build_test_flags(nFE, ctx["test_train_dict"])
    use_rl = bool(use_markov and config.get("run_rl_proposal", False))
    action_warm_start_step = -1 if force_td3_execute else int(ctx["warm_start_step"])

    rl_agent = None
    if use_rl:
        rl_agent = ctx.get("agent")
        if rl_agent is None:
            if str(config.get("agent_kind", "td3")).lower() != "td3":
                raise ValueError("The shared Markov runner currently supports TD3 live proposals only.")
            rl_agent = make_td3_markov_agent(
                config,
                rl_state_dim,
                z_dim,
                set_points_len=int(config["set_points_len"]),
            )

    system = ctx["system_factory"]()
    L = compute_observer_gain(A, C, ctx["poles"])
    history["y_phys"][0, :] = np.asarray(system.current_output, float)
    history["y_scaled_dev"][0, :] = (
        apply_min_max(system.current_output, ctx["data_min"][nu:], ctx["data_max"][nu:]) - ctx["y_ss_scaled"]
    )

    x_model = np.zeros(A.shape[0], dtype=float)
    x_init = np.zeros(control_horizon * nu, dtype=float)
    z_prev = np.zeros(z_dim, dtype=float)
    last_raw_action = None
    last_action_test = None
    pending_transition = None
    avg_rewards = []

    def evaluate_markov_candidate(z_candidate, u_prev_dev, x_model_now, nominal_guess, nominal_cost):
        mz_candidate = apply_markov_correction(m_blocks, basis_blocks, z_candidate)
        G_candidate = build_toeplitz_from_markov(mz_candidate, predict_h, control_horizon)
        candidate_drift = gain_drift(G_candidate, G0)
        U_candidate, J_candidate, sol_candidate = solve_lifted_mpc(
            ctx["y_sp"][step],
            u_prev_dev,
            x_model_now,
            A,
            C,
            G_candidate,
            config["Q_out"],
            config["R_in"],
            predict_h,
            control_horizon,
            ctx["bounds"],
            nominal_guess,
        )
        nominal_cost_of_candidate = lifted_mpc_cost(
            U_candidate,
            ctx["y_sp"][step],
            u_prev_dev,
            free_response(A, C, x_model_now, predict_h),
            G0,
            config["Q_out"],
            config["R_in"],
            predict_h,
            control_horizon,
        )
        loose_tol = float(config["nominal_cost_absolute_tol"]) + float(config["nominal_cost_relative_tol"]) * abs(
            float(nominal_cost)
        )
        return {
            "U": U_candidate,
            "J": J_candidate,
            "sol": sol_candidate,
            "drift": candidate_drift,
            "nominal_cost": nominal_cost_of_candidate,
            "cost_guard_pass": nominal_cost_of_candidate <= float(nominal_cost) + loose_tol,
        }

    for step in range(nFE):
        scaled_current_input = apply_min_max(system.current_input, ctx["data_min"][:nu], ctx["data_max"][:nu])
        u_prev_dev = scaled_current_input - ctx["ss_scaled_inputs"]
        history["xhat_before"][step, :] = x_model
        yhat = C @ x_model
        history["yhat"][:, step] = yhat

        nominal_guess = x_init if bool(config.get("use_shifted_mpc_warm_start", False)) else np.zeros(
            control_horizon * nu, dtype=float
        )
        U0, J0, sol0 = solve_nominal_reference_step(
            nominal_solver_mode=nominal_solver_mode,
            mpc_obj=ctx["MPC_obj"],
            y_sp=ctx["y_sp"][step],
            u_prev_dev=u_prev_dev,
            x0_model=x_model,
            A=A,
            C=C,
            G0=G0,
            Q_out=config["Q_out"],
            R_in=config["R_in"],
            predict_h=predict_h,
            control_horizon=control_horizon,
            bounds=ctx["bounds"],
            x_init=nominal_guess,
        )
        _record_nominal_stage(history, step, U0, J0, nu)
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
        }
        ls_score = dict(score)
        drift = 0.0
        ls_drift = 0.0
        ls_accepted = False
        z_ls = np.zeros(z_dim, dtype=float)
        U_ls = None
        ls_eval = None
        rl_eval = None
        rl_score = None
        executed_eval = {
            "U": U0.copy(),
            "J": float(J0),
            "nominal_cost": float(J0),
            "cost_guard_pass": True,
            "drift": 0.0,
        }
        executed_score = score

        if use_markov and bool(config.get("run_adaptive_ls", True)) and step >= predict_h:
            z_ls, _ls_result, ls_score = fit_markov_ls_correction(
                z_prev,
                z_bounds,
                history,
                m_blocks,
                basis_blocks,
                G0,
                A,
                C,
                predict_h,
                control_horizon,
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
            _record_candidate_stage(history, step, "ls", ls_eval["U"], ls_eval, ls_score, U0, J0, nu)

        z_ls_safe = z_ls if ls_accepted else np.zeros(z_dim, dtype=float)
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

        if use_markov and bool(config.get("run_live_corrected_mpc", True)) and step >= predict_h:
            if rl_agent is not None:
                baseline_raw = z_to_raw_action(z_ls_safe, config["z_bound"])
                test_step = bool(test_flags[step])
                decision = select_continuous_action(
                    agent=rl_agent,
                    state=rl_state,
                    step=step,
                    warm_start_step=action_warm_start_step,
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

                if force_td3_execute:
                    rl_score = prediction_improvement_score(
                        z=z_requested,
                        history=history,
                        m_blocks=m_blocks,
                        basis_blocks=basis_blocks,
                        G0=G0,
                        A=A,
                        C=C,
                        predict_h=predict_h,
                        control_horizon=control_horizon,
                        Wy=Wy,
                        lambda_z=float(config["lambda_z"]),
                        current_step=step,
                        prediction_window=int(config["prediction_window"]),
                    )
                    rl_eval = evaluate_markov_candidate(z_requested, u_prev_dev, x_model, U0, J0)
                    _record_candidate_stage(history, step, "requested", rl_eval["U"], rl_eval, rl_score, U0, J0, nu)
                    U_exec = rl_eval["U"]
                    z_exec = z_requested
                    raw_executed = raw_requested
                    score = rl_score
                    drift = float(rl_eval["drift"])
                    accepted = True
                    fallback = False
                    action_source = 2
                    z_prev = z_exec
                    executed_eval = rl_eval
                    executed_score = rl_score
                elif step > ctx["warm_start_step"]:
                    rl_score = prediction_improvement_score(
                        z=z_requested,
                        history=history,
                        m_blocks=m_blocks,
                        basis_blocks=basis_blocks,
                        G0=G0,
                        A=A,
                        C=C,
                        predict_h=predict_h,
                        control_horizon=control_horizon,
                        Wy=Wy,
                        lambda_z=float(config["lambda_z"]),
                        current_step=step,
                        prediction_window=int(config["prediction_window"]),
                    )
                    rl_eval = evaluate_markov_candidate(z_requested, u_prev_dev, x_model, U0, J0)
                    _record_candidate_stage(history, step, "requested", rl_eval["U"], rl_eval, rl_score, U0, J0, nu)
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
                        executed_eval = rl_eval
                        executed_score = rl_score
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
                        executed_eval = ls_eval
                        executed_score = ls_score
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
                    executed_eval = ls_eval
                    executed_score = ls_score
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
                executed_eval = ls_eval
                executed_score = ls_score

        _record_candidate_stage(history, step, "executed", U_exec, executed_eval, executed_score, U0, J0, nu)
        u_dev = U_exec[:nu]
        u_scaled_abs = u_dev + ctx["ss_scaled_inputs"]
        u_phys = reverse_min_max(u_scaled_abs, ctx["data_min"][:nu], ctx["data_max"][:nu])
        du = u_dev - u_prev_dev

        system.current_input = u_phys
        step_system_with_disturbance(
            system,
            idx=step if ctx["run_mode"] == "disturb" else None,
            disturbance_schedule=ctx["disturbance_schedule"],
            system_stepper=ctx["system_stepper"],
        )

        history["u_phys_log"][step, :] = u_phys
        history["u_scaled_abs_log"][step, :] = u_scaled_abs
        history["u_dev_log"][step, :] = u_dev
        history["du_log"][step, :] = du
        history["y_phys"][step + 1, :] = np.asarray(system.current_output, float)
        y_current_scaled = (
            apply_min_max(system.current_output, ctx["data_min"][nu:], ctx["data_max"][nu:]) - ctx["y_ss_scaled"]
        )
        history["y_scaled_dev"][step + 1, :] = y_current_scaled

        if ctx["observer_alignment"] == "predictor_corrector_current":
            x_pred = A @ x_model + B @ u_dev
            y_pred_next = C @ x_pred
            x_model = x_pred + L @ (y_current_scaled - y_pred_next).T
            history["yhat"][:, step] = C @ x_model
        else:
            x_model = A @ x_model + B @ u_dev + L @ innovation
        history["xhat_after"][step + 1, :] = x_model

        delta_y = history["y_scaled_dev"][step + 1, :] - ctx["y_sp"][step, :]
        y_sp_phys = reverse_min_max(
            ctx["y_sp"][step, :] + ctx["y_ss_scaled"],
            ctx["data_min"][nu:],
            ctx["data_max"][nu:],
        )
        history["rewards"][step] = float(ctx["reward_fn"](delta_y, du, y_sp_phys))
        history["z_log"][step, :] = z_exec
        history["z_executed_log"][step, :] = z_exec
        history["rl_executed_raw_action_log"][step, :] = raw_executed
        history["rl_action_source_log"][step] = int(action_source)
        history["s_pred_log"][step] = float(score["score"])
        history["gain_drift_log"][step] = float(drift)
        history["accepted_log"][step] = int(accepted)
        history["fallback_log"][step] = int(fallback)
        history["prediction_error_nominal_log"][step] = (
            float(score["nominal_sse"]) if np.isfinite(score["nominal_sse"]) else np.nan
        )
        history["prediction_error_markov_log"][step] = (
            float(score["corrected_sse"]) if np.isfinite(score["corrected_sse"]) else np.nan
        )

        if rl_agent is not None:
            transition_action = raw_executed if bool(config.get("rl_store_executed_action_in_replay", True)) else raw_requested
            pending_transition = {
                "step": step,
                "state": rl_state.copy(),
                "action": np.asarray(transition_action, float).copy(),
                "reward": float(history["rewards"][step]),
                "test": bool(test_flags[step]),
            }

        if step in ctx["sub_episode_changes_dict"]:
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
                    ctx["sub_episode_changes_dict"][step],
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

        if bool(config.get("use_shifted_mpc_warm_start", False)):
            x_init = shift_control_sequence(U_exec[: nu * control_horizon], nu, control_horizon)
        else:
            x_init = np.zeros(control_horizon * nu, dtype=float)

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
        else avg_by_episode(history["rewards"], ctx["sub_episode_changes_dict"], ctx["time_in_sub_episodes"])
    )
    history["_rl_agent"] = rl_agent
    return history


def summarize_history(config, ctx, history):
    nFE = int(ctx["nFE"])
    accepted_fraction = float(np.mean(history["accepted_log"])) if nFE else 0.0
    td3_accepted_fraction = float(np.mean(history["rl_action_source_log"] == 2)) if nFE else 0.0
    ls_fallback_fraction = float(np.mean(history["rl_action_source_log"] == 3)) if nFE else 0.0
    nominal_fallback_fraction = float(np.mean(history["rl_action_source_log"] == 4)) if nFE else 0.0
    replay_push_count = int(np.sum(history["rl_replay_pushed_log"]))
    train_update_count = int(np.sum(history["rl_train_updated_log"]))
    return {
        "agent_kind": str(config.get("agent_kind", "td3")).lower(),
        "run_mode": str(config["run_mode"]).lower(),
        "nominal_solver_mode": str(config.get("nominal_solver_mode", "state_space_shared")).lower(),
        "force_td3_execute": bool(config.get("force_td3_execute", False)),
        "td3_seed": config.get("td3_agent", {}).get("seed"),
        "accepted_fraction": accepted_fraction,
        "td3_accepted_fraction": td3_accepted_fraction,
        "ls_fallback_fraction": ls_fallback_fraction,
        "nominal_fallback_fraction": nominal_fallback_fraction,
        "reward_mean": float(np.mean(history["rewards"])) if nFE else np.nan,
        "reward_final_episode": float(history["avg_rewards"][-1]) if history["avg_rewards"].size else np.nan,
        "gain_drift_mean": float(np.nanmean(history["gain_drift_log"])) if nFE else np.nan,
        "prediction_score_mean": float(np.nanmean(history["s_pred_log"])) if nFE else np.nan,
        "rl_replay_push_count": replay_push_count,
        "rl_train_update_count": train_update_count,
    }


def run_markov_correction_supervisor(markov_cfg, runtime_ctx):
    config = deepcopy(markov_cfg)
    config["Q_out"] = np.asarray([config["Q1_penalty"], config["Q2_penalty"]], float)
    config["R_in"] = np.asarray([config["R1_penalty"], config["R2_penalty"]], float)
    ctx = build_runtime_context(config, runtime_ctx)

    m_blocks = compute_markov_blocks(ctx["A_aug"], ctx["B_aug"], ctx["C_aug"], int(config["predict_h"]))
    basis_blocks, basis_labels = make_markov_basis(m_blocks, config["basis_family"])
    G0 = build_toeplitz_from_markov(m_blocks, int(config["predict_h"]), int(config["cont_h"]))
    Wy = np.eye(int(config["predict_h"]) * ctx["n_outputs"])

    debug_phase1 = None
    if bool(config.get("debug_validate_lifted", False)):
        debug_phase1 = phase1_equivalence_metrics(config, ctx, G0)

    debug_nominal = None
    debug_shadow = None
    if bool(config.get("debug_run_shadow_ls", False)):
        debug_nominal = run_single_closed_loop(
            config,
            ctx,
            m_blocks,
            basis_blocks,
            G0,
            Wy,
            use_markov=False,
            print_progress=False,
        )
        debug_shadow = run_shadow_and_ls_debug(config, ctx, debug_nominal, m_blocks, basis_blocks, G0, Wy)

    history = run_single_closed_loop(
        config,
        ctx,
        m_blocks,
        basis_blocks,
        G0,
        Wy,
        use_markov=bool(config.get("run_live_corrected_mpc", True)),
        print_progress=True,
    )
    summary_metrics = summarize_history(config, ctx, history)

    return {
        "agent_kind": str(config.get("agent_kind", "td3")).lower(),
        "run_mode": ctx["run_mode"],
        "nominal_solver_mode": str(config.get("nominal_solver_mode", "state_space_shared")).lower(),
        "force_td3_execute": bool(config.get("force_td3_execute", False)),
        "td3_seed": config.get("td3_agent", {}).get("seed"),
        "system_metadata": ctx["system_metadata"],
        "A": None if ctx.get("system_data", {}).get("A") is None else np.asarray(ctx["system_data"]["A"], float),
        "B": None if ctx.get("system_data", {}).get("B") is None else np.asarray(ctx["system_data"]["B"], float),
        "C": None if ctx.get("system_data", {}).get("C") is None else np.asarray(ctx["system_data"]["C"], float),
        "A_aug": np.asarray(ctx["A_aug"], float),
        "B_aug": np.asarray(ctx["B_aug"], float),
        "C_aug": np.asarray(ctx["C_aug"], float),
        "y_sp": ctx["y_sp"],
        "steady_states": ctx["steady_states"],
        "nFE": int(ctx["nFE"]),
        "delta_t": float(ctx["delta_t"]),
        "time_in_sub_episodes": int(ctx["time_in_sub_episodes"]),
        "y": history["y_phys"],
        "u": history["u_phys_log"],
        "avg_rewards": history["avg_rewards"],
        "rewards_step": history["rewards"],
        "delta_y_storage": history["y_scaled_dev"][1 : ctx["nFE"] + 1, :] - ctx["y_sp"][: ctx["nFE"], :],
        "delta_u_storage": history["du_log"],
        "data_min": ctx["data_min"],
        "data_max": ctx["data_max"],
        "yhat": history["yhat"],
        "xhatdhat": history["xhat_after"].T,
        "test_train_dict": ctx["test_train_dict"],
        "sub_episodes_changes_dict": ctx["sub_episode_changes_dict"],
        "disturbance_profile": disturbance_profile_from_schedule(ctx["disturbance_schedule"])
        if ctx["run_mode"] == "disturb"
        else None,
        "warm_start_step": int(ctx["warm_start_step"]),
        "mpc_horizons": (int(config["predict_h"]), int(config["cont_h"])),
        "use_shifted_mpc_warm_start": bool(config.get("use_shifted_mpc_warm_start", False)),
        "observer_update_mode": ctx["observer_alignment"],
        "reward_params": ctx["reward_params"],
        "summary_metrics": summary_metrics,
        "M_blocks_nominal": m_blocks,
        "basis_family": str(config["basis_family"]),
        "basis_labels": list(basis_labels),
        "markov_s_pred_min": float(config["s_pred_min"]),
        "markov_gain_drift_max": float(config["gain_drift_max"]),
        "z_log": history["z_log"],
        "z_proposed_log": history["z_proposed_log"],
        "z_executed_log": history["z_executed_log"],
        "s_pred_log": history["s_pred_log"],
        "gain_drift_log": history["gain_drift_log"],
        "accepted_log": history["accepted_log"],
        "fallback_log": history["fallback_log"],
        "prediction_error_nominal_log": history["prediction_error_nominal_log"],
        "prediction_error_markov_log": history["prediction_error_markov_log"],
        "rl_state_dim": history["rl_state_dim"],
        "rl_action_dim": history["rl_action_dim"],
        "rl_state_log": history["rl_state_log"],
        "rl_requested_raw_action_log": history["rl_requested_raw_action_log"],
        "rl_executed_raw_action_log": history["rl_executed_raw_action_log"],
        "rl_requested_z_log": history["rl_requested_z_log"],
        "rl_ls_z_log": history["rl_ls_z_log"],
        "rl_action_source_log": history["rl_action_source_log"],
        "rl_action_source_names": history["rl_action_source_names"],
        "rl_decision_taken_log": history["rl_decision_taken_log"],
        "rl_policy_source_log": history["rl_policy_source_log"],
        "rl_replay_pushed_log": history["rl_replay_pushed_log"],
        "rl_train_called_log": history["rl_train_called_log"],
        "rl_train_updated_log": history["rl_train_updated_log"],
        "rl_actor_loss_log": history["rl_actor_loss_log"],
        "rl_critic_loss_log": history["rl_critic_loss_log"],
        "rl_bc_loss_log": history["rl_bc_loss_log"],
        "rl_test_step_log": history["rl_test_step_log"],
        "u_sequence_nominal_log": history["u_sequence_nominal_log"],
        "u_sequence_requested_log": history["u_sequence_requested_log"],
        "u_sequence_ls_log": history["u_sequence_ls_log"],
        "u_sequence_executed_log": history["u_sequence_executed_log"],
        "u0_nominal_log": history["u0_nominal_log"],
        "u0_requested_log": history["u0_requested_log"],
        "u0_ls_log": history["u0_ls_log"],
        "u0_executed_log": history["u0_executed_log"],
        "u0_requested_minus_nominal_log": history["u0_requested_minus_nominal_log"],
        "u0_ls_minus_nominal_log": history["u0_ls_minus_nominal_log"],
        "u0_executed_minus_nominal_log": history["u0_executed_minus_nominal_log"],
        "u0_requested_minus_nominal_norm_log": history["u0_requested_minus_nominal_norm_log"],
        "u0_ls_minus_nominal_norm_log": history["u0_ls_minus_nominal_norm_log"],
        "u0_executed_minus_nominal_norm_log": history["u0_executed_minus_nominal_norm_log"],
        "u_sequence_requested_minus_nominal_norm_log": history["u_sequence_requested_minus_nominal_norm_log"],
        "u_sequence_ls_minus_nominal_norm_log": history["u_sequence_ls_minus_nominal_norm_log"],
        "u_sequence_executed_minus_nominal_norm_log": history["u_sequence_executed_minus_nominal_norm_log"],
        "nominal_cost_log": history["nominal_cost_log"],
        "requested_candidate_native_cost_log": history["requested_candidate_native_cost_log"],
        "requested_candidate_nominal_cost_log": history["requested_candidate_nominal_cost_log"],
        "requested_cost_margin_log": history["requested_cost_margin_log"],
        "requested_cost_guard_pass_log": history["requested_cost_guard_pass_log"],
        "requested_gain_drift_log": history["requested_gain_drift_log"],
        "requested_prediction_score_log": history["requested_prediction_score_log"],
        "ls_candidate_native_cost_log": history["ls_candidate_native_cost_log"],
        "ls_candidate_nominal_cost_log": history["ls_candidate_nominal_cost_log"],
        "ls_cost_margin_log": history["ls_cost_margin_log"],
        "ls_cost_guard_pass_log": history["ls_cost_guard_pass_log"],
        "ls_gain_drift_log": history["ls_gain_drift_log"],
        "ls_prediction_score_log": history["ls_prediction_score_log"],
        "executed_candidate_native_cost_log": history["executed_candidate_native_cost_log"],
        "executed_candidate_nominal_cost_log": history["executed_candidate_nominal_cost_log"],
        "executed_cost_margin_log": history["executed_cost_margin_log"],
        "executed_cost_guard_pass_log": history["executed_cost_guard_pass_log"],
        "executed_gain_drift_log": history["executed_gain_drift_log"],
        "executed_prediction_score_log": history["executed_prediction_score_log"],
        "rl_agent_checkpoint_path": history["rl_agent_checkpoint_path"],
        "debug_phase1_metrics": debug_phase1,
        "debug_shadow": debug_shadow,
        "debug_nominal": None
        if debug_nominal is None
        else {
            "y": debug_nominal["y_phys"],
            "u": debug_nominal["u_phys_log"],
            "avg_rewards": debug_nominal["avg_rewards"],
            "rewards": debug_nominal["rewards"],
        },
        "_rl_agent": history.get("_rl_agent"),
    }
