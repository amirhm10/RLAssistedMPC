from __future__ import annotations

from copy import deepcopy

import numpy as np
import scipy.optimize as spo

from TD3Agent.agent import TD3Agent
from TD3Agent.supervisor_gated_agent import SupervisorGatedTD3Agent, SupervisorGateConfig
from TD3Agent.supervisor_replay_buffer import (
    SOURCE_FALLBACK,
    SOURCE_POLICY,
    SOURCE_SUPERVISOR,
    SOURCE_WARM_START,
)
from utils.agent_step_runtime import replay_train_continuous_agent, select_continuous_action
from utils.behavioral_cloning import (
    apply_bc_handoff_action,
    build_behavioral_cloning_bundle_fields,
    build_bc_handoff_bundle_fields,
    build_behavioral_cloning_schedule,
    build_protected_bc_release_gate_bundle_fields,
    init_bc_handoff_logs,
    init_behavioral_cloning_logs,
    init_protected_bc_release_gate,
    record_bc_handoff_step,
    record_behavioral_cloning_step,
    resolve_bc_handoff_authority,
    resolve_behavioral_cloning_context,
    update_protected_bc_release_gate,
)
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
from utils.state_features import (
    build_rl_state,
    compute_tracking_scale_now,
    get_rl_state_dim,
    make_state_conditioner_from_settings,
    resolve_mismatch_settings,
)
from utils.td3_authority_ramp import (
    build_td3_authority_ramp_bundle_fields,
    init_td3_authority_ramp_logs,
    record_td3_authority_ramp_step,
    resolve_td3_authority_ramp,
)


MARKOV_ACTION_SOURCE = {
    0: "nominal_no_markov",
    1: "warm_start_ls",
    2: "td3_accepted",
    3: "ls_fallback",
    4: "nominal_fallback",
    5: "ls_no_rl",
    6: "sg_supervisor_ls",
    7: "sg_supervisor_mpc",
    8: "sg_solver_fallback_supervisor",
}

TD3_PRIORITY_PHASE_CODE = {
    "none": 0,
    "protected": 1,
    "ramp": 2,
    "full": 3,
}

DEFAULT_TD3_GAMMA = 0.99

MARKOV_SUPERVISOR_KIND = {
    0: "none",
    1: "ls",
    2: "mpc",
}

MARKOV_SG_SOLVER_FALLBACK_REASON = {
    0: "none",
    1: "policy_solve_failure",
    2: "nonfinite_policy",
}


def _td3_priority_cfg(config):
    cfg = config.get("td3_priority_fallback", {})
    return cfg if isinstance(cfg, dict) else {}


def _td3_priority_enabled(config):
    return bool(_td3_priority_cfg(config).get("enabled", False))


def _markov_shadow_safety_cfg(config):
    cfg = config.get("markov_shadow_safety", {})
    return cfg if isinstance(cfg, dict) else {}


def _is_supervisor_gated_markov(config) -> bool:
    return str(config.get("agent_kind", "td3")).strip().lower() == "sg_td3"


def resolve_markov_supervisor_action(
    *,
    mode,
    z_ls,
    ls_accepted,
    z_bound,
    z_dim,
):
    """Return the normalized SG supervisor action and source kind."""
    mode = str(mode or "ls_else_mpc").strip().lower()
    if mode not in {"ls_else_mpc", "mpc_only"}:
        raise ValueError("markov_supervisor_mode must be 'ls_else_mpc' or 'mpc_only'.")
    use_ls = bool(mode == "ls_else_mpc" and ls_accepted)
    if use_ls:
        z_supervisor = np.asarray(z_ls, float).reshape(-1)
        if z_supervisor.size != int(z_dim):
            raise ValueError(f"z_ls has size {z_supervisor.size}, expected {int(z_dim)}.")
        return {
            "z": z_supervisor.copy(),
            "raw_action": z_to_raw_action(z_supervisor, z_bound),
            "kind_code": 1,
            "kind_name": MARKOV_SUPERVISOR_KIND[1],
            "action_source": 6,
        }
    z_zero = np.zeros(int(z_dim), dtype=float)
    return {
        "z": z_zero,
        "raw_action": z_zero.copy(),
        "kind_code": 2,
        "kind_name": MARKOV_SUPERVISOR_KIND[2],
        "action_source": 7,
    }


def _count_actor_freeze_train_steps(
    *,
    n_steps,
    test_flags,
    train_start_step,
    actor_freeze_end_step,
    batch_size,
    initial_buffer_size=0,
):
    buffer_size = int(max(0, initial_buffer_size))
    count = 0
    for step in range(int(max(0, n_steps))):
        if bool(test_flags[step]):
            continue
        buffer_size += 1
        if (
            step >= int(train_start_step)
            and step <= int(actor_freeze_end_step)
            and buffer_size >= int(max(1, batch_size))
        ):
            count += 1
    return int(count)


def _shadow_runtime_config(config, shadow_cfg):
    shadow_config = deepcopy(config)
    if isinstance(shadow_cfg.get("z_safety"), dict):
        shadow_config["z_safety"] = deepcopy(shadow_cfg["z_safety"])
    if isinstance(shadow_cfg.get("td3_priority_fallback"), dict):
        shadow_config["td3_priority_fallback"] = deepcopy(shadow_cfg["td3_priority_fallback"])
    return shadow_config


def _td3_priority_phase(config, ctx, step):
    cfg = _td3_priority_cfg(config)
    protected = int(max(0, cfg.get("protected_subepisodes", 0)))
    ramp = int(max(0, cfg.get("ramp_subepisodes", 0)))
    time_in_sub = int(max(1, ctx["time_in_sub_episodes"]))
    warm_start_step = int(ctx["warm_start_step"])
    post_warm_step = max(0, int(step) - warm_start_step - 1)
    post_warm_episode = int(post_warm_step // time_in_sub) + 1
    if post_warm_episode <= protected:
        return "protected"
    if post_warm_episode <= protected + ramp:
        return "ramp"
    return "full"


def _td3_priority_cost_cap(config, phase):
    cfg = _td3_priority_cfg(config)
    caps = cfg.get("cost_caps", {})
    if not isinstance(caps, dict):
        caps = {}
    phase_cap = caps.get(phase, caps.get("full", {}))
    if not isinstance(phase_cap, dict):
        phase_cap = {}
    return {
        "absolute": float(phase_cap.get("absolute", config.get("nominal_cost_absolute_tol", 1.0e-8))),
        "relative": float(phase_cap.get("relative", config.get("nominal_cost_relative_tol", 0.0))),
    }


def _td3_priority_authority_scale(config, ctx, step, probation_active=False):
    cfg = _td3_priority_cfg(config)
    ramp_cfg = cfg.get("authority_ramp", {})
    if not _td3_priority_enabled(config) or not isinstance(ramp_cfg, dict) or not ramp_cfg.get("enabled", False):
        return 1.0

    phase = _td3_priority_phase(config, ctx, step)
    if phase == "protected":
        scale = float(ramp_cfg.get("protected_scale", 1.0))
    elif phase == "ramp":
        protected = int(max(0, cfg.get("protected_subepisodes", 0)))
        ramp = int(max(0, cfg.get("ramp_subepisodes", 0)))
        post_warm_episode = _td3_priority_post_warm_episode(ctx, step)
        ramp_index = max(1, post_warm_episode - protected)
        start = float(ramp_cfg.get("ramp_start_scale", ramp_cfg.get("protected_scale", 1.0)))
        end = float(ramp_cfg.get("ramp_end_scale", ramp_cfg.get("full_scale", 1.0)))
        if ramp <= 1:
            progress = 1.0
        else:
            progress = float(np.clip((ramp_index - 1) / float(ramp - 1), 0.0, 1.0))
        scale = start + progress * (end - start)
    else:
        scale = float(ramp_cfg.get("full_scale", 1.0))

    probation_cfg = cfg.get("reward_probation", {})
    if probation_active and isinstance(probation_cfg, dict):
        scale = min(scale, float(probation_cfg.get("cooldown_scale", scale)))
    return float(np.clip(scale, 0.0, 1.0))


def _td3_priority_post_warm_episode(ctx, step):
    time_in_sub = int(max(1, ctx["time_in_sub_episodes"]))
    warm_start_step = int(ctx["warm_start_step"])
    post_warm_step = max(0, int(step) - warm_start_step - 1)
    return int(post_warm_step // time_in_sub) + 1


def _z_safety_cfg(config):
    cfg = config.get("z_safety", {})
    return cfg if isinstance(cfg, dict) else {}


def _z_safety_enabled(config):
    return bool(_z_safety_cfg(config).get("enabled", False))


def resolve_z_safety_effective_cap(config, ctx, step, probation_active=False):
    """Return the active coordinate-wise z cap for the current Markov step."""
    z_bound = float(config["z_bound"])
    cfg = _z_safety_cfg(config)
    if not _z_safety_enabled(config):
        return z_bound

    phase = _td3_priority_phase(config, ctx, step)
    if phase == "protected":
        cap = float(cfg.get("protected_cap", z_bound))
    elif phase == "ramp":
        priority_cfg = _td3_priority_cfg(config)
        protected = int(max(0, priority_cfg.get("protected_subepisodes", 0)))
        ramp = int(max(1, priority_cfg.get("ramp_subepisodes", 1)))
        post_warm_episode = _td3_priority_post_warm_episode(ctx, step)
        ramp_index = int(np.clip(post_warm_episode - protected, 1, ramp))
        frac = 1.0 if ramp <= 1 else float(ramp_index - 1) / float(ramp - 1)
        start = float(cfg.get("ramp_start_cap", cfg.get("protected_cap", z_bound)))
        end = float(cfg.get("ramp_end_cap", cfg.get("full_cap", z_bound)))
        cap = start + frac * (end - start)
    else:
        cap = float(cfg.get("full_cap", z_bound))

    if probation_active:
        cap = min(cap, float(cfg.get("probation_cap", cap)))
    return float(np.clip(cap, 0.0, z_bound))


def apply_z_safety_projection(z, config, effective_cap=None):
    """Clip z by coordinate and optional vector-norm trust region."""
    z_in = np.asarray(z, float).reshape(-1)
    z_bound = float(config["z_bound"])
    cfg = _z_safety_cfg(config)
    enabled = _z_safety_enabled(config)
    coord_cap = z_bound if effective_cap is None else float(effective_cap)
    if not enabled:
        coord_cap = z_bound
    coord_cap = float(np.clip(coord_cap, 0.0, z_bound))

    norm_before = float(np.linalg.norm(z_in))
    z_coord = np.clip(z_in, -coord_cap, coord_cap)
    coord_clip_active = bool(np.any(np.abs(z_coord - z_in) > 1.0e-12))
    norm_after_coord = float(np.linalg.norm(z_coord))

    vector_cfg = cfg.get("vector_norm_cap", {})
    if not isinstance(vector_cfg, dict):
        vector_cfg = {}
    norm_cap_enabled = bool(enabled and vector_cfg.get("enabled", False))
    max_norm = float(vector_cfg.get("max_norm", np.inf))
    projection_scale = 1.0
    vector_projection_active = False
    z_out = z_coord
    if norm_cap_enabled and np.isfinite(max_norm) and max_norm > 0.0 and norm_after_coord > max_norm:
        projection_scale = float(max_norm / max(norm_after_coord, 1.0e-12))
        z_out = z_coord * projection_scale
        vector_projection_active = True

    norm_after = float(np.linalg.norm(z_out))
    return z_out, {
        "enabled": enabled,
        "effective_cap": coord_cap,
        "norm_before": norm_before,
        "norm_after": norm_after,
        "projection_scale": projection_scale,
        "projection_active": bool(coord_clip_active or vector_projection_active),
        "coord_clip_active": coord_clip_active,
        "vector_projection_active": vector_projection_active,
    }


def _td3_priority_candidate_allowed(config, ctx, step, candidate_eval, candidate_score, nominal_success):
    if candidate_eval is None:
        return False
    sol = candidate_eval.get("sol")
    if not (bool(nominal_success) and sol is not None and bool(getattr(sol, "success", False))):
        return False

    cfg = _td3_priority_cfg(config)
    score_hard_min = cfg.get("score_hard_min", None)
    if score_hard_min is not None:
        if candidate_score is None:
            return False
        score_value = float(candidate_score.get("score", np.nan))
        if (not np.isfinite(score_value)) or score_value < float(score_hard_min):
            return False

    drift_limit = float(cfg.get("gain_drift_max", config.get("gain_drift_max", np.inf)))
    if float(candidate_eval["drift"]) > drift_limit:
        return False

    phase = _td3_priority_phase(config, ctx, step)
    cap = _td3_priority_cost_cap(config, phase)
    nominal_cost = float(candidate_eval.get("reference_nominal_cost", 0.0))
    margin = float(candidate_eval.get("cost_margin", 0.0))
    allowed_margin = float(cap["absolute"]) + float(cap["relative"]) * abs(nominal_cost)
    return margin <= allowed_margin


def _truncate_disturbance_schedule(disturbance_schedule, n_steps):
    if disturbance_schedule is None:
        return None
    n_steps = int(n_steps)
    if isinstance(disturbance_schedule, dict):
        return {
            key: np.asarray(value, float)[:n_steps] if np.asarray(value).ndim > 0 else float(value)
            for key, value in disturbance_schedule.items()
        }
    arr = np.asarray(disturbance_schedule, float)
    if arr.ndim == 0:
        return arr
    return arr[:n_steps]


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

    agent_kind = str(config.get("agent_kind", "td3")).strip().lower()
    if agent_kind not in {"td3", "sg_td3"}:
        raise ValueError("Markov TD3 agent construction supports only 'td3' and 'sg_td3'.")

    agent_cls = SupervisorGatedTD3Agent if agent_kind == "sg_td3" else TD3Agent
    extra_kwargs = {}
    if agent_kind == "sg_td3":
        extra_kwargs["supervisor_gate_config"] = SupervisorGateConfig(**dict(config.get("supervisor_gate", {}) or {}))

    return agent_cls(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        seed=td3_cfg.get("seed"),
        actor_hidden=list(td3_cfg["actor_hidden"]),
        critic_hidden=list(td3_cfg["critic_hidden"]),
        gamma=float(td3_cfg.get("gamma", DEFAULT_TD3_GAMMA)),
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
        param_noise_std_start=float(td3_cfg.get("param_noise_std_start", 0.2)),
        param_noise_std_end=float(td3_cfg.get("param_noise_std_end", 0.02)),
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
        **extra_kwargs,
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
        "rl_actor_raw_action_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_requested_raw_action_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_executed_raw_action_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_requested_z_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_ls_z_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_requested_z_uncapped_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_ls_z_uncapped_log": np.zeros((nFE, z_dim), dtype=float),
        "rl_action_source_log": np.zeros(nFE, dtype=int),
        "rl_decision_taken_log": np.zeros(nFE, dtype=int),
        "rl_policy_source_log": np.zeros(nFE, dtype=int),
        "sg_policy_action_raw_log": np.full((nFE, z_dim), np.nan, dtype=float),
        "sg_supervisor_action_raw_log": np.full((nFE, z_dim), np.nan, dtype=float),
        "sg_executed_action_raw_log": np.full((nFE, z_dim), np.nan, dtype=float),
        "sg_previous_action_raw_log": np.full((nFE, z_dim), np.nan, dtype=float),
        "sg_selected_source_log": np.zeros(nFE, dtype=int),
        "sg_score_policy_log": np.full(nFE, np.nan, dtype=float),
        "sg_score_supervisor_log": np.full(nFE, np.nan, dtype=float),
        "sg_advantage_log": np.full(nFE, np.nan, dtype=float),
        "sg_q1_policy_log": np.full(nFE, np.nan, dtype=float),
        "sg_q2_policy_log": np.full(nFE, np.nan, dtype=float),
        "sg_q1_supervisor_log": np.full(nFE, np.nan, dtype=float),
        "sg_q2_supervisor_log": np.full(nFE, np.nan, dtype=float),
        "sg_q_gap_policy_log": np.full(nFE, np.nan, dtype=float),
        "sg_q_gap_supervisor_log": np.full(nFE, np.nan, dtype=float),
        "sg_supervisor_kind_log": np.zeros(nFE, dtype=int),
        "sg_solver_fallback_reason_log": np.zeros(nFE, dtype=int),
        "rl_replay_pushed_log": np.zeros(nFE, dtype=int),
        "rl_train_called_log": np.zeros(nFE, dtype=int),
        "rl_train_updated_log": np.zeros(nFE, dtype=int),
        "rl_actor_loss_log": np.full(nFE, np.nan, dtype=float),
        "rl_critic_loss_log": np.full(nFE, np.nan, dtype=float),
        "rl_bc_loss_log": np.full(nFE, np.nan, dtype=float),
        "rl_test_step_log": np.zeros(nFE, dtype=int),
        "td3_priority_phase_log": np.zeros(nFE, dtype=int),
        "td3_authority_scale_log": np.ones(nFE, dtype=float),
        "td3_authority_ramp_logs": init_td3_authority_ramp_logs(nFE, z_dim),
        "td3_probation_active_log": np.zeros(nFE, dtype=int),
        "td3_probation_trigger_log": np.zeros(nFE, dtype=int),
        "z_safety_effective_cap_log": np.full(nFE, np.nan, dtype=float),
        "z_safety_requested_norm_before_log": np.full(nFE, np.nan, dtype=float),
        "z_safety_requested_norm_after_log": np.full(nFE, np.nan, dtype=float),
        "z_safety_requested_projection_scale_log": np.full(nFE, np.nan, dtype=float),
        "z_safety_requested_projection_active_log": np.zeros(nFE, dtype=int),
        "z_safety_requested_coord_clip_active_log": np.zeros(nFE, dtype=int),
        "z_safety_requested_vector_projection_active_log": np.zeros(nFE, dtype=int),
        "z_safety_ls_norm_before_log": np.full(nFE, np.nan, dtype=float),
        "z_safety_ls_norm_after_log": np.full(nFE, np.nan, dtype=float),
        "z_safety_ls_projection_scale_log": np.full(nFE, np.nan, dtype=float),
        "z_safety_ls_projection_active_log": np.zeros(nFE, dtype=int),
        "z_safety_ls_coord_clip_active_log": np.zeros(nFE, dtype=int),
        "z_safety_ls_vector_projection_active_log": np.zeros(nFE, dtype=int),
        "shadow_z_safety_effective_cap_log": np.full(nFE, np.nan, dtype=float),
        "shadow_z_safety_requested_norm_before_log": np.full(nFE, np.nan, dtype=float),
        "shadow_z_safety_requested_norm_after_log": np.full(nFE, np.nan, dtype=float),
        "shadow_z_safety_requested_projection_scale_log": np.full(nFE, np.nan, dtype=float),
        "shadow_z_safety_requested_projection_active_log": np.zeros(nFE, dtype=int),
        "shadow_z_safety_requested_coord_clip_active_log": np.zeros(nFE, dtype=int),
        "shadow_z_safety_requested_vector_projection_active_log": np.zeros(nFE, dtype=int),
        "shadow_td3_priority_phase_log": np.zeros(nFE, dtype=int),
        "shadow_td3_priority_authority_scale_log": np.ones(nFE, dtype=float),
        "shadow_td3_priority_allowed_log": np.full(nFE, -1, dtype=int),
        "shadow_ls_priority_allowed_log": np.full(nFE, -1, dtype=int),
        "shadow_nominal_fallback_eligible_log": np.full(nFE, -1, dtype=int),
        "shadow_bc_handoff_authority_log": np.full(nFE, np.nan, dtype=float),
        "shadow_bc_handoff_delta_norm_log": np.full(nFE, np.nan, dtype=float),
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
        "requested_legacy_hard_gate_pass_log": np.full(nFE, -1, dtype=int),
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


def _record_z_safety_stage(history, step, prefix, safety_info):
    history[f"z_safety_{prefix}_norm_before_log"][step] = float(safety_info["norm_before"])
    history[f"z_safety_{prefix}_norm_after_log"][step] = float(safety_info["norm_after"])
    history[f"z_safety_{prefix}_projection_scale_log"][step] = float(safety_info["projection_scale"])
    history[f"z_safety_{prefix}_projection_active_log"][step] = int(bool(safety_info["projection_active"]))
    history[f"z_safety_{prefix}_coord_clip_active_log"][step] = int(bool(safety_info["coord_clip_active"]))
    history[f"z_safety_{prefix}_vector_projection_active_log"][step] = int(
        bool(safety_info["vector_projection_active"])
    )


def _record_shadow_z_safety_stage(history, step, prefix, safety_info):
    base = f"shadow_z_safety_{prefix}"
    history[f"{base}_norm_before_log"][step] = float(safety_info["norm_before"])
    history[f"{base}_norm_after_log"][step] = float(safety_info["norm_after"])
    history[f"{base}_projection_scale_log"][step] = float(safety_info["projection_scale"])
    history[f"{base}_projection_active_log"][step] = int(bool(safety_info["projection_active"]))
    history[f"{base}_coord_clip_active_log"][step] = int(bool(safety_info["coord_clip_active"]))
    history[f"{base}_vector_projection_active_log"][step] = int(bool(safety_info["vector_projection_active"]))


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

    episode_bundle = runtime_ctx.get("episode_bundle")
    if episode_bundle is None:
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
            float(markov_cfg.get("nominal_qi", 0.0)),
            float(markov_cfg.get("nominal_qs", 0.0)),
            float(markov_cfg.get("nominal_ha", 0.0)),
            float(markov_cfg.get("qi_change", 1.0)),
            float(markov_cfg.get("qs_change", 1.0)),
            float(markov_cfg.get("ha_change", 1.0)),
        )
    else:
        y_sp = np.asarray(episode_bundle["y_sp"], float)
        nFE = int(episode_bundle["nFE"])
        sub_episode_changes_dict = dict(episode_bundle["sub_episode_changes_dict"])
        time_in_sub_episodes = int(episode_bundle["time_in_sub_episodes"])
        test_train_dict = dict(episode_bundle["test_train_dict"])
        warm_start_step = int(episode_bundle["warm_start_step"])
        qi = np.asarray(episode_bundle.get("qi", np.zeros(nFE, dtype=float)), float)
        qs = np.asarray(episode_bundle.get("qs", np.zeros(nFE, dtype=float)), float)
        ha = np.asarray(episode_bundle.get("ha", np.zeros(nFE, dtype=float)), float)

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
    disturbance_schedule = runtime_ctx.get("disturbance_schedule")
    if max_steps is not None:
        disturbance_schedule = _truncate_disturbance_schedule(disturbance_schedule, nFE)
    if disturbance_schedule is None and str(markov_cfg["run_mode"]).lower() == "disturb":
        disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    min_max_dict = runtime_ctx.get("min_max_dict")
    if min_max_dict is None:
        min_max_dict = runtime_ctx.get("system_data", {}).get("min_max_dict")
    if min_max_dict is None:
        raise KeyError("runtime_ctx or system_data must provide 'min_max_dict' for Markov state conditioning.")

    mismatch_cfg = resolve_mismatch_settings(
        state_mode="mismatch",
        mismatch_cfg=markov_cfg,
        reward_params=reward_params,
        y_sp_scenario=y_sp,
        steady_states=steady_states,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
    )

    system = runtime_ctx.get("system")
    system_factory = runtime_ctx.get("system_factory")
    if system is None and system_factory is None:
        raise KeyError("runtime_ctx must provide either 'system' or 'system_factory'.")

    delta_t = runtime_ctx.get("delta_t")
    if delta_t is None:
        delta_t = getattr(system, "delta_t", 0.5) if system is not None else 0.5

    return {
        "system": system,
        "system_factory": system_factory,
        "system_stepper": runtime_ctx.get("system_stepper"),
        "system_teardown": runtime_ctx.get("system_teardown"),
        "delta_t": float(delta_t),
        "system_data": system_data,
        "min_max_dict": min_max_dict,
        "mismatch_cfg": mismatch_cfg,
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
    force_td3_respects_warm_start = bool(config.get("force_td3_respects_warm_start", False))
    supervisor_gated_markov = _is_supervisor_gated_markov(config)
    markov_supervisor_mode = str(config.get("markov_supervisor_mode", "ls_else_mpc")).strip().lower()
    A = np.asarray(ctx["A_aug"], float)
    B = np.asarray(ctx["B_aug"], float)
    C = np.asarray(ctx["C_aug"], float)
    nFE = int(ctx["nFE"])
    ny = int(C.shape[0])
    nu = int(B.shape[1])
    z_dim = int(basis_blocks.shape[0])
    z_bounds = [(-float(config["z_bound"]), float(config["z_bound"])) for _ in range(z_dim)]
    rl_state_dim = int(get_rl_state_dim(A.shape[0], ny, nu, "mismatch", append_rho_to_state=False) + z_dim + z_dim + 2)
    history = initialize_history(nFE, A.shape[0], ny, nu, z_dim, control_horizon, rl_state_dim)
    test_flags = build_test_flags(nFE, ctx["test_train_dict"])
    state_conditioner = make_state_conditioner_from_settings(ctx["mismatch_cfg"])
    use_rl = bool(use_markov and config.get("run_rl_proposal", False))
    action_warm_start_step = (
        int(ctx["warm_start_step"])
        if (not force_td3_execute or force_td3_respects_warm_start)
        else -1
    )
    shadow_safety_cfg = _markov_shadow_safety_cfg(config)
    shadow_safety_enabled = bool(shadow_safety_cfg.get("enabled", False))
    shadow_config = _shadow_runtime_config(config, shadow_safety_cfg) if shadow_safety_enabled else config
    bc_schedule = build_behavioral_cloning_schedule(
        config=config.get("behavioral_cloning", {}),
        warm_start_step=ctx["warm_start_step"],
        time_in_sub_episodes=ctx["time_in_sub_episodes"],
        n_steps=nFE,
    )
    bc_logs = init_behavioral_cloning_logs(nFE)
    bc_handoff_logs = init_bc_handoff_logs(nFE, z_dim)
    bc_release_gate = init_protected_bc_release_gate(bc_schedule, nFE)
    bc_handoff_enabled = bool(dict(bc_schedule.get("handoff", {}) or {}).get("enabled", False))
    shadow_bc_handoff_cfg = dict(shadow_safety_cfg.get("bc_handoff", {}) or {})
    shadow_bc_schedule = build_behavioral_cloning_schedule(
        config={"enabled": False, "handoff": shadow_bc_handoff_cfg},
        warm_start_step=ctx["warm_start_step"],
        time_in_sub_episodes=ctx["time_in_sub_episodes"],
        n_steps=nFE,
    )
    if bool(bc_release_gate["state"].get("enabled", False)) and not bool(
        bc_release_gate["state"].get("diagnostic_only", False)
    ):
        action_warm_start_step = int(ctx["warm_start_step"])
    if bc_handoff_enabled:
        action_warm_start_step = -1
    bc_train_start_step = (
        int(bc_schedule.get("start_step", ctx["warm_start_step"]))
        if bool(bc_schedule.get("enabled", False))
        else int(ctx["warm_start_step"])
    )
    td3_authority_ramp_cfg = dict(config.get("td3_authority_ramp", {}) or {})
    sg_action_freeze_subepisodes = int(max(0, config.get("post_warm_start_action_freeze_subepisodes", 0)))
    sg_actor_freeze_subepisodes = int(max(0, config.get("post_warm_start_actor_freeze_subepisodes", 0)))
    sg_action_freeze_end_step = int(
        ctx["warm_start_step"] + sg_action_freeze_subepisodes * max(1, ctx["time_in_sub_episodes"])
    )
    sg_actor_freeze_end_step = int(
        ctx["warm_start_step"] + sg_actor_freeze_subepisodes * max(1, ctx["time_in_sub_episodes"])
    )

    rl_agent = None
    if use_rl:
        rl_agent = ctx.get("agent")
        if rl_agent is None:
            if str(config.get("agent_kind", "td3")).lower() not in {"td3", "sg_td3"}:
                raise ValueError("The shared Markov runner currently supports TD3/SG-TD3 live proposals only.")
            rl_agent = make_td3_markov_agent(
                config,
                rl_state_dim,
                z_dim,
                set_points_len=int(config["set_points_len"]),
            )
        if supervisor_gated_markov:
            freeze_train_steps = _count_actor_freeze_train_steps(
                n_steps=nFE,
                test_flags=test_flags,
                train_start_step=bc_train_start_step,
                actor_freeze_end_step=sg_actor_freeze_end_step,
                batch_size=getattr(rl_agent, "batch_size", 1),
                initial_buffer_size=len(getattr(rl_agent, "buffer", [])),
            )
            rl_agent.actor_freeze = int(max(getattr(rl_agent, "actor_freeze", 0), freeze_train_steps))

    system = ctx.get("system")
    if system is None:
        system_factory = ctx.get("system_factory")
        if not callable(system_factory):
            raise ValueError("No usable system or system_factory was provided to the Markov runner.")
        system = system_factory()
    try:
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
        warm_reference_rewards = []
        warm_release_reference_reward = None
        probation_cooldown_until_episode = 0
        probation_trigger_count = 0
        warm_subepisodes = int(ctx["warm_start_step"] // max(1, ctx["time_in_sub_episodes"]))

        def flush_pending_transition(next_state, done):
            nonlocal pending_transition
            if pending_transition is None or rl_agent is None:
                return

            pending_step = int(pending_transition["step"])
            bc_context = resolve_behavioral_cloning_context(
                bc_schedule,
                step_idx=pending_step,
                target_action=pending_transition["bc_target_action"],
                policy_action=pending_transition["policy_action"],
                tail_meta=pending_transition.get("bc_tail_meta"),
            )
            if supervisor_gated_markov:
                pushed = False
                trained = False
                train_meta = None
                if not bool(pending_transition["test"]):
                    rl_agent.push_supervised(
                        pending_transition["state"],
                        pending_transition["action"],
                        pending_transition["reward"],
                        next_state,
                        done,
                        policy_action=pending_transition.get("policy_action"),
                        supervisor_action=pending_transition.get("supervisor_action"),
                        previous_action=pending_transition.get("previous_action"),
                        selected_source=pending_transition.get("selected_source", SOURCE_SUPERVISOR),
                        score_policy=pending_transition.get("score_policy", np.nan),
                        score_supervisor=pending_transition.get("score_supervisor", np.nan),
                        advantage_policy_supervisor=pending_transition.get("advantage_policy_supervisor", np.nan),
                    )
                    pushed = True
                    if pending_step >= bc_train_start_step:
                        train_meta = rl_agent.train_step(bc_context=bc_context)
                        trained = True
                train_info = {"pushed": pushed, "trained": trained, "train_meta": train_meta}
            else:
                train_info = replay_train_continuous_agent(
                    agent=rl_agent,
                    state=pending_transition["state"],
                    action=pending_transition["action"],
                    reward=pending_transition["reward"],
                    next_state=next_state,
                    done=done,
                    step=pending_step,
                    test=pending_transition["test"],
                    train_start_step=bc_train_start_step,
                    bc_context=bc_context,
                )
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
            record_behavioral_cloning_step(
                bc_logs,
                step_idx=pending_step,
                bc_context=bc_context,
                policy_action=pending_transition["policy_action"],
                target_action=pending_transition["bc_target_action"],
                train_meta=train_meta,
            )
            pending_transition = None

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
            cost_margin = float(nominal_cost_of_candidate) - float(nominal_cost)
            return {
                "U": U_candidate,
                "J": J_candidate,
                "sol": sol_candidate,
                "drift": candidate_drift,
                "nominal_cost": nominal_cost_of_candidate,
                "reference_nominal_cost": float(nominal_cost),
                "cost_margin": cost_margin,
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
            raw_actor_requested = np.zeros(z_dim, dtype=float)
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
            z_shadow_ls = np.zeros(z_dim, dtype=float)
            shadow_ls_eval = None
            shadow_ls_score = None
            shadow_ls_target_available = False
            executed_eval = {
                "U": U0.copy(),
                "J": float(J0),
                "nominal_cost": float(J0),
                "reference_nominal_cost": float(J0),
                "cost_margin": 0.0,
                "cost_guard_pass": True,
                "drift": 0.0,
                "sol": sol0,
            }
            executed_score = score
            current_subepisode = int(step // max(1, ctx["time_in_sub_episodes"])) + 1
            probation_active = bool(
                _td3_priority_enabled(config)
                and not force_td3_execute
                and step > ctx["warm_start_step"]
                and current_subepisode <= int(probation_cooldown_until_episode)
            )
            z_safety_effective_cap = resolve_z_safety_effective_cap(
                config,
                ctx,
                step,
                probation_active=probation_active,
            )
            history["z_safety_effective_cap_log"][step] = float(z_safety_effective_cap)
            shadow_z_safety_effective_cap = np.nan
            if shadow_safety_enabled:
                shadow_z_safety_effective_cap = resolve_z_safety_effective_cap(
                    shadow_config,
                    ctx,
                    step,
                    probation_active=False,
                )
                history["shadow_z_safety_effective_cap_log"][step] = float(shadow_z_safety_effective_cap)

            if use_markov and bool(config.get("run_adaptive_ls", True)) and step >= predict_h:
                z_ls_uncapped, _ls_result, ls_score = fit_markov_ls_correction(
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
                history["rl_ls_z_uncapped_log"][step, :] = z_ls_uncapped
                z_ls, ls_safety_info = apply_z_safety_projection(
                    z_ls_uncapped,
                    config,
                    effective_cap=z_safety_effective_cap,
                )
                _record_z_safety_stage(history, step, "ls", ls_safety_info)
                ls_score = prediction_improvement_score(
                    z=z_ls,
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

            if (
                use_markov
                and shadow_safety_enabled
                and bool(shadow_safety_cfg.get("compute_ls_candidate", False))
                and not bool(config.get("run_adaptive_ls", True))
                and step >= predict_h
            ):
                z_shadow_ls_uncapped, _shadow_ls_result, shadow_ls_score = fit_markov_ls_correction(
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
                z_shadow_ls, _shadow_ls_safety_info = apply_z_safety_projection(
                    z_shadow_ls_uncapped,
                    shadow_config,
                    effective_cap=shadow_z_safety_effective_cap,
                )
                shadow_ls_score = prediction_improvement_score(
                    z=z_shadow_ls,
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
                shadow_ls_eval = evaluate_markov_candidate(z_shadow_ls, u_prev_dev, x_model, U0, J0)
                shadow_ls_target_available = bool(
                    shadow_ls_eval is not None
                    and shadow_ls_eval.get("sol") is not None
                    and bool(getattr(shadow_ls_eval["sol"], "success", False))
                )

            z_ls_safe = z_ls if ls_accepted else np.zeros(z_dim, dtype=float)
            innovation = history["y_scaled_dev"][step, :] - yhat
            tracking_error = history["y_scaled_dev"][step, :] - ctx["y_sp"][step, :]
            y_sp_phys = reverse_min_max(
                ctx["y_sp"][step, :] + ctx["y_ss_scaled"],
                ctx["data_min"][nu:],
                ctx["data_max"][nu:],
            )
            _, tracking_scale_now = compute_tracking_scale_now(
                y_sp_phys=y_sp_phys,
                data_min=ctx["data_min"],
                data_max=ctx["data_max"],
                n_inputs=nu,
                k_rel=ctx["mismatch_cfg"]["k_rel"],
                band_floor_phys=ctx["mismatch_cfg"]["band_floor_phys"],
                tracking_eta_tol=ctx["mismatch_cfg"]["tracking_eta_tol"],
                tracking_scale_floor=ctx["mismatch_cfg"]["tracking_scale_floor"],
            )
            conditioned_state, _state_debug = build_rl_state(
                min_max_dict=ctx["min_max_dict"],
                x_d_states=x_model,
                y_sp=ctx["y_sp"][step, :],
                u=u_prev_dev,
                state_mode="mismatch",
                y_prev_scaled=history["y_scaled_dev"][step, :],
                yhat_pred=yhat,
                innovation_scale_ref=ctx["mismatch_cfg"]["innovation_scale_ref"],
                tracking_scale_now=tracking_scale_now,
                mismatch_clip=ctx["mismatch_cfg"]["mismatch_clip"],
                state_conditioner=state_conditioner,
                update_state_conditioner=True,
                mismatch_feature_transform_mode=ctx["mismatch_cfg"]["mismatch_feature_transform_mode"],
                mismatch_transform_tanh_scale=ctx["mismatch_cfg"]["mismatch_transform_tanh_scale"],
                mismatch_transform_post_clip=ctx["mismatch_cfg"]["mismatch_transform_post_clip"],
            )
            rl_state = np.concatenate(
                [
                    np.asarray(conditioned_state, np.float32).reshape(-1),
                    np.asarray(z_prev, np.float32).reshape(-1),
                    np.asarray(z_ls_safe, np.float32).reshape(-1),
                    np.asarray([float(ls_score["score"]), float(ls_drift)], np.float32),
                ]
            ).astype(np.float32, copy=False)
            if rl_state.size != rl_state_dim:
                raise ValueError(f"Conditioned Markov RL state has size {rl_state.size}, expected {rl_state_dim}.")
            history["rl_state_log"][step, :] = rl_state
            history["rl_ls_z_log"][step, :] = z_ls
            bc_target_raw = np.zeros(z_dim, dtype=float)
            bc_target_is_ls = False

            flush_pending_transition(rl_state, 0.0)

            if use_markov and bool(config.get("run_live_corrected_mpc", True)) and (
                rl_agent is not None or force_td3_execute or step >= predict_h
            ):
                if rl_agent is not None:
                    baseline_z_for_bc = z_ls_safe
                    baseline_is_ls = bool(ls_accepted and U_ls is not None)
                    if not baseline_is_ls and shadow_ls_target_available:
                        baseline_z_for_bc = z_shadow_ls
                        baseline_is_ls = True
                    baseline_raw = z_to_raw_action(baseline_z_for_bc, config["z_bound"])
                    bc_target_raw = baseline_raw.copy()
                    bc_target_is_ls = bool(baseline_is_ls)
                    test_step = bool(test_flags[step])
                    supervisor_payload = resolve_markov_supervisor_action(
                        mode=markov_supervisor_mode,
                        z_ls=z_ls,
                        ls_accepted=bool(ls_accepted and U_ls is not None),
                        z_bound=config["z_bound"],
                        z_dim=z_dim,
                    )
                    supervisor_raw = np.asarray(supervisor_payload["raw_action"], float).reshape(-1)
                    policy_raw_for_gate = np.asarray(rl_agent.act_eval(rl_state), float).reshape(-1)
                    policy_action_nonfinite = bool(
                        policy_raw_for_gate.size != z_dim or not np.all(np.isfinite(policy_raw_for_gate))
                    )
                    if policy_action_nonfinite:
                        policy_raw_for_gate = supervisor_raw.copy() if supervisor_gated_markov else baseline_raw.copy()
                    release_info = update_protected_bc_release_gate(
                        bc_release_gate["state"],
                        bc_release_gate["logs"],
                        step_idx=step,
                        warm_start_step=ctx["warm_start_step"],
                        policy_action=policy_raw_for_gate,
                        target_action=baseline_raw,
                    )
                    ramp_info = resolve_td3_authority_ramp(
                        td3_authority_ramp_cfg,
                        step_idx=step,
                        warm_start_step=ctx["warm_start_step"],
                        time_in_sub_episodes=ctx["time_in_sub_episodes"],
                    )
                    td3_live_released = bool(release_info.get("live_released", release_info.get("released", False)))
                    gate_override = bool(
                        release_info.get("live_blocked", release_info.get("blocked", False))
                        and ramp_info["live_enabled"]
                    )
                    if gate_override:
                        td3_live_released = True
                    force_td3_this_step = bool(
                        force_td3_execute
                        and (
                            not force_td3_respects_warm_start
                            or step > ctx["warm_start_step"]
                        )
                    )
                    if force_td3_this_step and (
                        not bool(release_info.get("enabled", False))
                        or bool(release_info.get("diagnostic_only", False))
                    ):
                        td3_live_released = True
                    if supervisor_gated_markov:
                        previous_action_for_gate = (
                            np.asarray(history["rl_executed_raw_action_log"][step - 1, :], float).reshape(-1)
                            if step > 0
                            else supervisor_raw.copy()
                        )
                        forced_supervisor = bool(step <= sg_action_freeze_end_step)
                        if forced_supervisor:
                            raw_requested = supervisor_raw.copy()
                            raw_actor_requested = policy_raw_for_gate.copy()
                            sg_selected_source = SOURCE_WARM_START if step <= ctx["warm_start_step"] else SOURCE_SUPERVISOR
                            sg_score_policy = np.nan
                            sg_score_supervisor = np.nan
                            sg_advantage = np.nan
                            sg_q1_policy = np.nan
                            sg_q2_policy = np.nan
                            sg_q1_supervisor = np.nan
                            sg_q2_supervisor = np.nan
                            sg_q_gap_policy = np.nan
                            sg_q_gap_supervisor = np.nan
                            decision_taken = 0
                            policy_source = 0 if step <= ctx["warm_start_step"] else 1
                        else:
                            sg_decision = rl_agent.select_action_with_supervisor(
                                rl_state,
                                supervisor_action=supervisor_raw,
                                previous_action=previous_action_for_gate,
                                explore=not test_step,
                                test=test_step,
                            )
                            raw_requested = np.asarray(sg_decision.action, float).reshape(-1)
                            raw_actor_requested = np.asarray(sg_decision.policy_action, float).reshape(-1)
                            sg_selected_source = int(sg_decision.selected_source)
                            sg_score_policy = float(sg_decision.score_policy)
                            sg_score_supervisor = float(sg_decision.score_supervisor)
                            sg_advantage = float(sg_decision.advantage_policy_supervisor)
                            sg_q1_policy = float(sg_decision.q1_policy)
                            sg_q2_policy = float(sg_decision.q2_policy)
                            sg_q1_supervisor = float(sg_decision.q1_supervisor)
                            sg_q2_supervisor = float(sg_decision.q2_supervisor)
                            sg_q_gap_policy = float(sg_decision.q_gap_policy)
                            sg_q_gap_supervisor = float(sg_decision.q_gap_supervisor)
                            decision_taken = 1
                            policy_source = 3 if test_step else 2
                        if policy_action_nonfinite and not forced_supervisor:
                            sg_selected_source = SOURCE_FALLBACK
                            raw_requested = supervisor_raw.copy()
                        authority_scale = 1.0
                        record_td3_authority_ramp_step(
                            history["td3_authority_ramp_logs"],
                            step_idx=step,
                            ramp_info=ramp_info,
                            preclip_action=raw_requested,
                            postclip_action=raw_requested,
                            projection_active=False,
                            delta_norm=0.0,
                            gate_override=False,
                        )
                        handoff_info = resolve_bc_handoff_authority(bc_schedule, step_idx=step)
                        record_bc_handoff_step(
                            bc_handoff_logs,
                            step_idx=step,
                            authority_info=handoff_info,
                            safe_action=supervisor_raw,
                            td3_action=raw_requested,
                            executed_action=raw_requested,
                        )
                        if shadow_safety_enabled and bool(shadow_bc_handoff_cfg.get("enabled", False)):
                            shadow_handoff_info = resolve_bc_handoff_authority(shadow_bc_schedule, step_idx=step)
                            shadow_handoff_action = apply_bc_handoff_action(
                                raw_requested,
                                supervisor_raw,
                                shadow_handoff_info["authority"],
                            )
                            history["shadow_bc_handoff_authority_log"][step] = float(shadow_handoff_info["authority"])
                            history["shadow_bc_handoff_delta_norm_log"][step] = float(
                                np.linalg.norm(shadow_handoff_action - raw_requested)
                            )
                        last_action_test = test_step
                        history["rl_decision_taken_log"][step] = int(decision_taken)
                        history["rl_policy_source_log"][step] = int(policy_source)
                        history["sg_policy_action_raw_log"][step, :] = raw_actor_requested
                        history["sg_supervisor_action_raw_log"][step, :] = supervisor_raw
                        history["sg_previous_action_raw_log"][step, :] = previous_action_for_gate
                        history["sg_selected_source_log"][step] = int(sg_selected_source)
                        history["sg_score_policy_log"][step] = float(sg_score_policy)
                        history["sg_score_supervisor_log"][step] = float(sg_score_supervisor)
                        history["sg_advantage_log"][step] = float(sg_advantage)
                        history["sg_q1_policy_log"][step] = float(sg_q1_policy)
                        history["sg_q2_policy_log"][step] = float(sg_q2_policy)
                        history["sg_q1_supervisor_log"][step] = float(sg_q1_supervisor)
                        history["sg_q2_supervisor_log"][step] = float(sg_q2_supervisor)
                        history["sg_q_gap_policy_log"][step] = float(sg_q_gap_policy)
                        history["sg_q_gap_supervisor_log"][step] = float(sg_q_gap_supervisor)
                        history["sg_supervisor_kind_log"][step] = int(supervisor_payload["kind_code"])
                    else:
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
                        raw_actor_requested = policy_raw_for_gate.copy()
                        raw_requested = np.asarray(decision.action, float).reshape(-1)
                        authority_scale = (
                            _td3_priority_authority_scale(config, ctx, step, probation_active=probation_active)
                            if (
                                td3_live_released
                                and _td3_priority_enabled(config)
                                and not force_td3_execute
                                and step > ctx["warm_start_step"]
                            )
                            else 1.0
                        )
                        if authority_scale < 1.0:
                            raw_requested = np.clip(raw_requested * authority_scale, -1.0, 1.0)
                        record_td3_authority_ramp_step(
                            history["td3_authority_ramp_logs"],
                            step_idx=step,
                            ramp_info=ramp_info,
                            preclip_action=np.asarray(decision.action, float).reshape(-1),
                            postclip_action=raw_requested,
                            projection_active=bool(authority_scale < 1.0),
                            delta_norm=float(np.linalg.norm(raw_requested - np.asarray(decision.action, float).reshape(-1))),
                            gate_override=gate_override,
                        )
                        handoff_info = resolve_bc_handoff_authority(bc_schedule, step_idx=step)
                        handoff_td3_action = raw_requested.copy()
                        raw_requested = apply_bc_handoff_action(
                            handoff_td3_action,
                            baseline_raw,
                            handoff_info["authority"],
                        )
                        record_bc_handoff_step(
                            bc_handoff_logs,
                            step_idx=step,
                            authority_info=handoff_info,
                            safe_action=baseline_raw,
                            td3_action=handoff_td3_action,
                            executed_action=raw_requested,
                        )
                        if shadow_safety_enabled and bool(shadow_bc_handoff_cfg.get("enabled", False)):
                            shadow_handoff_info = resolve_bc_handoff_authority(shadow_bc_schedule, step_idx=step)
                            shadow_handoff_action = apply_bc_handoff_action(
                                handoff_td3_action,
                                baseline_raw,
                                shadow_handoff_info["authority"],
                            )
                            history["shadow_bc_handoff_authority_log"][step] = float(shadow_handoff_info["authority"])
                            history["shadow_bc_handoff_delta_norm_log"][step] = float(
                                np.linalg.norm(shadow_handoff_action - handoff_td3_action)
                            )
                        last_action_test = decision.last_action_test
                        history["rl_decision_taken_log"][step] = int(decision.decision_taken)
                        history["rl_policy_source_log"][step] = int(decision.source)
                    history["rl_test_step_log"][step] = int(test_step)
                    history["rl_actor_raw_action_log"][step, :] = raw_actor_requested
                    history["td3_priority_phase_log"][step] = TD3_PRIORITY_PHASE_CODE.get(
                        _td3_priority_phase(config, ctx, step) if step > ctx["warm_start_step"] else "none",
                        0,
                    )
                    history["td3_authority_scale_log"][step] = float(authority_scale)
                    history["td3_probation_active_log"][step] = int(probation_active)

                    z_requested_uncapped = raw_action_to_z(raw_requested, config["z_bound"])
                    history["rl_requested_z_uncapped_log"][step, :] = z_requested_uncapped
                    if shadow_safety_enabled:
                        _shadow_z_requested, shadow_requested_safety_info = apply_z_safety_projection(
                            z_requested_uncapped,
                            shadow_config,
                            effective_cap=shadow_z_safety_effective_cap,
                        )
                        _record_shadow_z_safety_stage(
                            history,
                            step,
                            "requested",
                            shadow_requested_safety_info,
                        )
                    z_requested, requested_safety_info = apply_z_safety_projection(
                        z_requested_uncapped,
                        config,
                        effective_cap=z_safety_effective_cap,
                    )
                    _record_z_safety_stage(history, step, "requested", requested_safety_info)
                    raw_requested = z_to_raw_action(z_requested, config["z_bound"])
                    last_raw_action = raw_requested.copy()
                    history["rl_requested_raw_action_log"][step, :] = raw_requested
                    history["rl_requested_z_log"][step, :] = z_requested
                    history["z_proposed_log"][step, :] = z_requested

                    if supervisor_gated_markov:
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
                        if int(history["sg_supervisor_kind_log"][step]) == 1 and np.allclose(
                            z_requested,
                            z_ls,
                            atol=1.0e-12,
                        ):
                            rl_eval = ls_eval
                            rl_score = ls_score
                        elif np.allclose(z_requested, np.zeros(z_dim, dtype=float), atol=1.0e-12):
                            rl_eval = executed_eval
                        else:
                            rl_eval = evaluate_markov_candidate(z_requested, u_prev_dev, x_model, U0, J0)
                        _record_candidate_stage(history, step, "requested", rl_eval["U"], rl_eval, rl_score, U0, J0, nu)
                        legacy_rl_hard_gate_pass = bool(
                            sol0.success
                            and rl_eval["sol"].success
                            and rl_score["score"] > float(config["s_pred_min"])
                            and rl_eval["drift"] <= float(config["gain_drift_max"])
                            and rl_eval["cost_guard_pass"]
                        )
                        history["requested_legacy_hard_gate_pass_log"][step] = int(legacy_rl_hard_gate_pass)
                        selected_source = int(history["sg_selected_source_log"][step])
                        supervisor_kind = int(history["sg_supervisor_kind_log"][step])
                        supervisor_eval = ls_eval if supervisor_kind == 1 and ls_eval is not None else {
                            "U": U0.copy(),
                            "J": float(J0),
                            "nominal_cost": float(J0),
                            "reference_nominal_cost": float(J0),
                            "cost_margin": 0.0,
                            "cost_guard_pass": True,
                            "drift": 0.0,
                            "sol": sol0,
                        }
                        supervisor_score = ls_score if supervisor_kind == 1 and ls_eval is not None else score
                        supervisor_z = z_ls if supervisor_kind == 1 and ls_eval is not None else np.zeros(z_dim, dtype=float)
                        supervisor_raw_exec = (
                            z_to_raw_action(supervisor_z, config["z_bound"])
                            if supervisor_kind == 1 and ls_eval is not None
                            else np.zeros(z_dim, dtype=float)
                        )
                        policy_solve_success = bool(
                            rl_eval is not None
                            and rl_eval.get("sol") is not None
                            and bool(getattr(rl_eval["sol"], "success", False))
                        )
                        if selected_source == SOURCE_POLICY and policy_solve_success:
                            U_exec = rl_eval["U"]
                            z_exec = z_requested
                            raw_executed = raw_requested
                            score = rl_score
                            drift = float(rl_eval["drift"])
                            accepted = True
                            fallback = False
                            action_source = 2
                            executed_eval = rl_eval
                            executed_score = rl_score
                        elif selected_source == SOURCE_POLICY:
                            U_exec = supervisor_eval["U"]
                            z_exec = supervisor_z
                            raw_executed = supervisor_raw_exec
                            score = supervisor_score
                            drift = float(supervisor_eval["drift"])
                            accepted = bool(supervisor_kind == 1)
                            fallback = True
                            action_source = 8
                            executed_eval = supervisor_eval
                            executed_score = supervisor_score
                            history["sg_solver_fallback_reason_log"][step] = 1
                        elif selected_source == SOURCE_FALLBACK:
                            U_exec = supervisor_eval["U"]
                            z_exec = supervisor_z
                            raw_executed = supervisor_raw_exec
                            score = supervisor_score
                            drift = float(supervisor_eval["drift"])
                            accepted = bool(supervisor_kind == 1)
                            fallback = True
                            action_source = 8
                            executed_eval = supervisor_eval
                            executed_score = supervisor_score
                            history["sg_solver_fallback_reason_log"][step] = 2
                        else:
                            U_exec = supervisor_eval["U"]
                            z_exec = supervisor_z
                            raw_executed = supervisor_raw_exec
                            score = supervisor_score
                            drift = float(supervisor_eval["drift"])
                            accepted = bool(supervisor_kind == 1)
                            fallback = False
                            action_source = 6 if supervisor_kind == 1 else 7
                            executed_eval = supervisor_eval
                            executed_score = supervisor_score
                        z_prev = z_exec
                    elif force_td3_this_step and td3_live_released:
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
                        legacy_rl_hard_gate_pass = bool(
                            sol0.success
                            and rl_eval["sol"].success
                            and rl_score["score"] > float(config["s_pred_min"])
                            and rl_eval["drift"] <= float(config["gain_drift_max"])
                            and rl_eval["cost_guard_pass"]
                        )
                        history["requested_legacy_hard_gate_pass_log"][step] = int(legacy_rl_hard_gate_pass)
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
                    elif td3_live_released and step > ctx["warm_start_step"]:
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
                        legacy_rl_hard_gate_pass = bool(
                            sol0.success
                            and rl_eval["sol"].success
                            and rl_score["score"] > float(config["s_pred_min"])
                            and rl_eval["drift"] <= float(config["gain_drift_max"])
                            and rl_eval["cost_guard_pass"]
                        )
                        history["requested_legacy_hard_gate_pass_log"][step] = int(legacy_rl_hard_gate_pass)
                        if _td3_priority_enabled(config):
                            rl_accepted = _td3_priority_candidate_allowed(
                                config, ctx, step, rl_eval, rl_score, sol0.success
                            )
                            ls_priority_accepted = bool(
                                config.get("rl_fallback_to_ls", True)
                                and ls_eval is not None
                                and _td3_priority_candidate_allowed(config, ctx, step, ls_eval, ls_score, sol0.success)
                            )
                        else:
                            rl_accepted = legacy_rl_hard_gate_pass
                            ls_priority_accepted = bool(config.get("rl_fallback_to_ls", True)) and ls_accepted
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
                        elif ls_priority_accepted:
                            U_exec = ls_eval["U"]
                            z_exec = z_ls
                            raw_executed = z_to_raw_action(z_ls, config["z_bound"])
                            score = ls_score
                            drift = float(ls_eval["drift"])
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
                    bc_target_raw = raw_executed.copy()
                    bc_target_is_ls = True

            if shadow_safety_enabled:
                shadow_phase = _td3_priority_phase(shadow_config, ctx, step) if step > ctx["warm_start_step"] else "none"
                history["shadow_td3_priority_phase_log"][step] = TD3_PRIORITY_PHASE_CODE.get(shadow_phase, 0)
                history["shadow_td3_priority_authority_scale_log"][step] = float(
                    _td3_priority_authority_scale(shadow_config, ctx, step, probation_active=False)
                )
                if rl_eval is not None and rl_score is not None:
                    shadow_rl_allowed = _td3_priority_candidate_allowed(
                        shadow_config, ctx, step, rl_eval, rl_score, sol0.success
                    )
                    history["shadow_td3_priority_allowed_log"][step] = int(bool(shadow_rl_allowed))
                else:
                    shadow_rl_allowed = False
                ls_eval_for_shadow = ls_eval if ls_eval is not None else shadow_ls_eval
                ls_score_for_shadow = ls_score if ls_eval is not None else shadow_ls_score
                if ls_eval_for_shadow is not None and ls_score_for_shadow is not None:
                    shadow_ls_allowed = _td3_priority_candidate_allowed(
                        shadow_config, ctx, step, ls_eval_for_shadow, ls_score_for_shadow, sol0.success
                    )
                    history["shadow_ls_priority_allowed_log"][step] = int(bool(shadow_ls_allowed))
                else:
                    shadow_ls_allowed = False
                history["shadow_nominal_fallback_eligible_log"][step] = int(
                    bool(sol0.success and not shadow_rl_allowed and not shadow_ls_allowed)
                )

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
            history["rewards"][step] = float(ctx["reward_fn"](delta_y, du, y_sp_phys))
            history["z_log"][step, :] = z_exec
            history["z_executed_log"][step, :] = z_exec
            history["rl_executed_raw_action_log"][step, :] = raw_executed
            if supervisor_gated_markov:
                history["sg_executed_action_raw_log"][step, :] = raw_executed
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
                transition_action = (
                    raw_executed if bool(config.get("rl_store_executed_action_in_replay", True)) else raw_requested
                )
                if not bc_target_is_ls:
                    bc_target_raw = np.asarray(raw_executed, float).copy()
                pending_transition = {
                    "step": step,
                    "state": rl_state.copy(),
                    "action": np.asarray(transition_action, float).copy(),
                    "policy_action": np.asarray(
                        history["sg_policy_action_raw_log"][step, :]
                        if supervisor_gated_markov
                        else raw_requested,
                        float,
                    ).copy(),
                    "supervisor_action": np.asarray(
                        history["sg_supervisor_action_raw_log"][step, :]
                        if supervisor_gated_markov
                        else bc_target_raw,
                        float,
                    ).copy(),
                    "previous_action": np.asarray(
                        history["sg_previous_action_raw_log"][step, :]
                        if supervisor_gated_markov
                        else raw_executed,
                        float,
                    ).copy(),
                    "selected_source": int(history["sg_selected_source_log"][step])
                    if supervisor_gated_markov
                    else SOURCE_POLICY,
                    "score_policy": float(history["sg_score_policy_log"][step])
                    if supervisor_gated_markov
                    else np.nan,
                    "score_supervisor": float(history["sg_score_supervisor_log"][step])
                    if supervisor_gated_markov
                    else np.nan,
                    "advantage_policy_supervisor": float(history["sg_advantage_log"][step])
                    if supervisor_gated_markov
                    else np.nan,
                    "bc_target_action": np.asarray(bc_target_raw, float).copy(),
                    "bc_tail_meta": {
                        "target_is_ls": bool(bc_target_is_ls),
                        "requested_score": None if rl_score is None else float(rl_score["score"]),
                        "ls_score": None if ls_score is None else float(ls_score["score"]),
                    },
                    "reward": float(history["rewards"][step]),
                    "test": bool(test_flags[step]),
                }

            if step in ctx["sub_episode_changes_dict"]:
                start = max(0, step - int(ctx["time_in_sub_episodes"]) + 1)
                stop = step + 1
                avg_reward = float(np.mean(history["rewards"][start:stop]))
                avg_rewards.append(avg_reward)
                subepisode_idx = int(ctx["sub_episode_changes_dict"][step])
                if subepisode_idx <= warm_subepisodes:
                    warm_reference_rewards.append(avg_reward)
                    reward_probation_cfg = _td3_priority_cfg(config).get("reward_probation", {})
                    if not isinstance(reward_probation_cfg, dict):
                        reward_probation_cfg = {}
                    n_ref = int(
                        max(
                            1,
                            reward_probation_cfg.get("reference_warm_episodes", 3),
                        )
                    )
                    warm_release_reference_reward = float(np.mean(warm_reference_rewards[-n_ref:]))
                else:
                    probation_cfg = _td3_priority_cfg(config).get("reward_probation", {})
                    if not isinstance(probation_cfg, dict):
                        probation_cfg = {}
                    probation_enabled = bool(
                        _td3_priority_enabled(config)
                        and isinstance(probation_cfg, dict)
                        and probation_cfg.get("enabled", False)
                        and not force_td3_execute
                        and warm_release_reference_reward is not None
                    )
                    collapse_threshold = float(probation_cfg.get("collapse_threshold", np.inf))
                    if probation_enabled and avg_reward < float(warm_release_reference_reward) - collapse_threshold:
                        cooldown = int(max(0, probation_cfg.get("cooldown_subepisodes", 0)))
                        if cooldown > 0:
                            probation_cooldown_until_episode = max(
                                int(probation_cooldown_until_episode),
                                subepisode_idx + cooldown,
                            )
                            probation_trigger_count += 1
                            history["td3_probation_trigger_log"][step] = 1
                if print_progress:
                    accepted_fraction = float(np.mean(history["accepted_log"][start:stop]))
                    td3_sub_fraction = float(np.mean(history["rl_action_source_log"][start:stop] == 2))
                    ls_sub_fraction = float(np.mean(history["rl_action_source_log"][start:stop] == 3))
                    nominal_sub_fraction = float(np.mean(history["rl_action_source_log"][start:stop] == 4))
                    post_start = min(stop, int(ctx["warm_start_step"]) + 1)
                    if stop > post_start:
                        post_sources = history["rl_action_source_log"][post_start:stop]
                        td3_post_fraction = float(np.mean(post_sources == 2))
                        ls_post_fraction = float(np.mean(post_sources == 3))
                        nominal_post_fraction = float(np.mean(post_sources == 4))
                    else:
                        td3_post_fraction = np.nan
                        ls_post_fraction = np.nan
                        nominal_post_fraction = np.nan
                    mean_z = np.mean(history["z_executed_log"][start:stop, :], axis=0)
                    print(
                        "Sub_Episode:",
                        ctx["sub_episode_changes_dict"][step],
                        "| avg. reward:",
                        avg_reward,
                        "| accepted (subepisode):",
                        accepted_fraction,
                        "| TD3 executed (subepisode):",
                        td3_sub_fraction,
                        "| LS fallback (subepisode):",
                        ls_sub_fraction,
                        "| nominal fallback (subepisode):",
                        nominal_sub_fraction,
                        "| TD3/LS/nominal post-warm cumulative:",
                        (td3_post_fraction, ls_post_fraction, nominal_post_fraction),
                        "| avg z:",
                        mean_z,
                    )

            if bool(config.get("use_shifted_mpc_warm_start", False)):
                x_init = shift_control_sequence(U_exec[: nu * control_horizon], nu, control_horizon)
            else:
                x_init = np.zeros(control_horizon * nu, dtype=float)

        if pending_transition is not None:
            flush_pending_transition(pending_transition["state"], 1.0)

        history["avg_rewards"] = (
            np.asarray(avg_rewards, float)
            if avg_rewards
            else avg_by_episode(history["rewards"], ctx["sub_episode_changes_dict"], ctx["time_in_sub_episodes"])
        )
        history["_markov_state_norm_stats"] = state_conditioner.export_state()
        history["_behavioral_cloning_schedule"] = bc_schedule
        history["_behavioral_cloning_logs"] = bc_logs
        history["_bc_handoff_logs"] = bc_handoff_logs
        history["_protected_bc_release_gate"] = bc_release_gate
        history["_td3_probation_trigger_count"] = int(probation_trigger_count)
        history["_td3_warm_release_reference_reward"] = warm_release_reference_reward
        history["_rl_agent"] = rl_agent
        return history
    finally:
        teardown = ctx.get("system_teardown")
        if callable(teardown):
            teardown(system)
        elif hasattr(system, "close"):
            system.close()


def summarize_history(config, ctx, history):
    nFE = int(ctx["nFE"])
    accepted_fraction = float(np.mean(history["accepted_log"])) if nFE else 0.0
    td3_accepted_fraction = float(np.mean(history["rl_action_source_log"] == 2)) if nFE else 0.0
    ls_fallback_fraction = float(np.mean(history["rl_action_source_log"] == 3)) if nFE else 0.0
    nominal_fallback_fraction = float(np.mean(history["rl_action_source_log"] == 4)) if nFE else 0.0
    ramp_logs = history.get("td3_authority_ramp_logs", {})
    ramp_live_log = np.asarray(ramp_logs.get("td3_authority_ramp_live_log", []), int)
    ramp_override_log = np.asarray(ramp_logs.get("td3_authority_ramp_gate_override_log", []), int)
    post_warm_start_step = min(nFE, int(ctx["warm_start_step"]) + 1)
    post_sources = history["rl_action_source_log"][post_warm_start_step:nFE]
    post_accepted = history["accepted_log"][post_warm_start_step:nFE]
    post_ramp_live = ramp_live_log[post_warm_start_step:nFE] if ramp_live_log.size >= nFE else np.asarray([], int)
    post_ramp_override = (
        ramp_override_log[post_warm_start_step:nFE] if ramp_override_log.size >= nFE else np.asarray([], int)
    )
    has_post_warm = post_sources.size > 0
    accepted_fraction_post_warm = float(np.mean(post_accepted)) if has_post_warm else 0.0
    td3_accepted_fraction_post_warm = float(np.mean(post_sources == 2)) if has_post_warm else 0.0
    ls_fallback_fraction_post_warm = float(np.mean(post_sources == 3)) if has_post_warm else 0.0
    nominal_fallback_fraction_post_warm = float(np.mean(post_sources == 4)) if has_post_warm else 0.0
    replay_push_count = int(np.sum(history["rl_replay_pushed_log"]))
    train_update_count = int(np.sum(history["rl_train_updated_log"]))
    sg_selected = np.asarray(history.get("sg_selected_source_log", []), int)
    sg_kind = np.asarray(history.get("sg_supervisor_kind_log", []), int)
    sg_fallback_reason = np.asarray(history.get("sg_solver_fallback_reason_log", []), int)
    sg_has_logs = bool(_is_supervisor_gated_markov(config) and sg_selected.size)
    shadow_td3 = np.asarray(history.get("shadow_td3_priority_allowed_log", []), int)
    shadow_ls = np.asarray(history.get("shadow_ls_priority_allowed_log", []), int)
    shadow_nominal = np.asarray(history.get("shadow_nominal_fallback_eligible_log", []), int)

    def _valid_mean(values, target=1):
        arr = np.asarray(values, int)
        valid = arr >= 0
        return float(np.mean(arr[valid] == int(target))) if bool(valid.any()) else 0.0

    return {
        "agent_kind": str(config.get("agent_kind", "td3")).lower(),
        "run_mode": str(config["run_mode"]).lower(),
        "nominal_solver_mode": str(config.get("nominal_solver_mode", "state_space_shared")).lower(),
        "td3_priority_fallback_enabled": _td3_priority_enabled(config),
        "force_td3_execute": bool(config.get("force_td3_execute", False)),
        "force_td3_respects_warm_start": bool(config.get("force_td3_respects_warm_start", False)),
        "rl_store_executed_action_in_replay": bool(config.get("rl_store_executed_action_in_replay", True)),
        "td3_seed": config.get("td3_agent", {}).get("seed"),
        "accepted_fraction": accepted_fraction,
        "td3_accepted_fraction": td3_accepted_fraction,
        "ls_fallback_fraction": ls_fallback_fraction,
        "nominal_fallback_fraction": nominal_fallback_fraction,
        "post_warm_start_step": int(post_warm_start_step),
        "accepted_fraction_post_warm": accepted_fraction_post_warm,
        "td3_accepted_fraction_post_warm": td3_accepted_fraction_post_warm,
        "ls_fallback_fraction_post_warm": ls_fallback_fraction_post_warm,
        "nominal_fallback_fraction_post_warm": nominal_fallback_fraction_post_warm,
        "reward_mean": float(np.mean(history["rewards"])) if nFE else np.nan,
        "reward_final_episode": float(history["avg_rewards"][-1]) if history["avg_rewards"].size else np.nan,
        "gain_drift_mean": float(np.nanmean(history["gain_drift_log"])) if nFE else np.nan,
        "prediction_score_mean": float(np.nanmean(history["s_pred_log"])) if nFE else np.nan,
        "td3_authority_scale_mean": float(np.nanmean(history["td3_authority_scale_log"])) if nFE else np.nan,
        "td3_authority_ramp_live_fraction_post_warm": float(np.mean(post_ramp_live)) if post_ramp_live.size else 0.0,
        "td3_authority_ramp_gate_override_fraction_post_warm": float(np.mean(post_ramp_override))
        if post_ramp_override.size
        else 0.0,
        "td3_probation_active_fraction": float(np.mean(history["td3_probation_active_log"])) if nFE else 0.0,
        "td3_probation_trigger_count": int(history.get("_td3_probation_trigger_count", 0)),
        "td3_warm_release_reference_reward": history.get("_td3_warm_release_reference_reward"),
        "rl_replay_push_count": replay_push_count,
        "rl_train_update_count": train_update_count,
        "sg_policy_fraction": float(np.mean(sg_selected == SOURCE_POLICY)) if sg_has_logs else 0.0,
        "sg_supervisor_fraction": float(
            np.mean((sg_selected == SOURCE_SUPERVISOR) | (sg_selected == SOURCE_WARM_START))
        )
        if sg_has_logs
        else 0.0,
        "sg_ls_supervisor_fraction": float(np.mean(sg_kind == 1)) if sg_has_logs else 0.0,
        "sg_mpc_supervisor_fraction": float(np.mean(sg_kind == 2)) if sg_has_logs else 0.0,
        "sg_solver_fallback_fraction": float(np.mean(sg_fallback_reason != 0)) if sg_has_logs else 0.0,
        "shadow_td3_priority_allowed_fraction": _valid_mean(shadow_td3),
        "shadow_ls_priority_allowed_fraction": _valid_mean(shadow_ls),
        "shadow_nominal_fallback_eligible_fraction": _valid_mean(shadow_nominal),
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

    result_bundle = {
        "agent_kind": str(config.get("agent_kind", "td3")).lower(),
        "notebook_source": config.get("notebook_source"),
        "run_mode": ctx["run_mode"],
        "nominal_solver_mode": str(config.get("nominal_solver_mode", "state_space_shared")).lower(),
        "td3_priority_fallback": deepcopy(config.get("td3_priority_fallback", {})),
        "td3_authority_ramp": deepcopy(config.get("td3_authority_ramp", {})),
        "force_td3_execute": bool(config.get("force_td3_execute", False)),
        "rl_store_executed_action_in_replay": bool(config.get("rl_store_executed_action_in_replay", True)),
        "replay_storage_mode": "executed"
        if bool(config.get("rl_store_executed_action_in_replay", True))
        else "requested",
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
        "markov_base_state_norm_stats": history.get("_markov_state_norm_stats"),
        "markov_state_mode": "mismatch_conditioned",
        "markov_mismatch_feature_transform_mode": ctx["mismatch_cfg"]["mismatch_feature_transform_mode"],
        "M_blocks_nominal": m_blocks,
        "basis_family": str(config["basis_family"]),
        "basis_labels": list(basis_labels),
        "markov_z_bound": float(config["z_bound"]),
        "markov_supervisor_mode": str(config.get("markov_supervisor_mode", "ls_else_mpc")),
        "markov_live_safety_mode": str(config.get("markov_live_safety_mode", "default")),
        "supervisor_gate": deepcopy(config.get("supervisor_gate", {})),
        "z_safety": deepcopy(config.get("z_safety", {})),
        "markov_shadow_safety": deepcopy(config.get("markov_shadow_safety", {})),
        "markov_shadow_safety_enabled": bool(_markov_shadow_safety_cfg(config).get("enabled", False)),
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
        "rl_actor_raw_action_log": history["rl_actor_raw_action_log"],
        "rl_requested_raw_action_log": history["rl_requested_raw_action_log"],
        "rl_executed_raw_action_log": history["rl_executed_raw_action_log"],
        "rl_requested_z_log": history["rl_requested_z_log"],
        "rl_ls_z_log": history["rl_ls_z_log"],
        "rl_requested_z_uncapped_log": history["rl_requested_z_uncapped_log"],
        "rl_ls_z_uncapped_log": history["rl_ls_z_uncapped_log"],
        "rl_action_source_log": history["rl_action_source_log"],
        "rl_action_source_names": history["rl_action_source_names"],
        "rl_decision_taken_log": history["rl_decision_taken_log"],
        "rl_policy_source_log": history["rl_policy_source_log"],
        "sg_policy_action_raw_log": history["sg_policy_action_raw_log"],
        "sg_supervisor_action_raw_log": history["sg_supervisor_action_raw_log"],
        "sg_executed_action_raw_log": history["sg_executed_action_raw_log"],
        "sg_previous_action_raw_log": history["sg_previous_action_raw_log"],
        "sg_selected_source_log": history["sg_selected_source_log"],
        "sg_score_policy_log": history["sg_score_policy_log"],
        "sg_score_supervisor_log": history["sg_score_supervisor_log"],
        "sg_advantage_log": history["sg_advantage_log"],
        "sg_q1_policy_log": history["sg_q1_policy_log"],
        "sg_q2_policy_log": history["sg_q2_policy_log"],
        "sg_q1_supervisor_log": history["sg_q1_supervisor_log"],
        "sg_q2_supervisor_log": history["sg_q2_supervisor_log"],
        "sg_q_gap_policy_log": history["sg_q_gap_policy_log"],
        "sg_q_gap_supervisor_log": history["sg_q_gap_supervisor_log"],
        "sg_supervisor_kind_log": history["sg_supervisor_kind_log"],
        "sg_supervisor_kind_names": dict(MARKOV_SUPERVISOR_KIND),
        "sg_solver_fallback_reason_log": history["sg_solver_fallback_reason_log"],
        "sg_solver_fallback_reason_names": dict(MARKOV_SG_SOLVER_FALLBACK_REASON),
        "rl_replay_pushed_log": history["rl_replay_pushed_log"],
        "rl_train_called_log": history["rl_train_called_log"],
        "rl_train_updated_log": history["rl_train_updated_log"],
        "rl_actor_loss_log": history["rl_actor_loss_log"],
        "rl_critic_loss_log": history["rl_critic_loss_log"],
        "rl_bc_loss_log": history["rl_bc_loss_log"],
        "rl_test_step_log": history["rl_test_step_log"],
        "td3_priority_phase_log": history["td3_priority_phase_log"],
        "td3_priority_phase_codes": dict(TD3_PRIORITY_PHASE_CODE),
        "td3_authority_scale_log": history["td3_authority_scale_log"],
        **build_td3_authority_ramp_bundle_fields(
            config.get("td3_authority_ramp", {}),
            history["td3_authority_ramp_logs"],
            prefix="rl_",
        ),
        "td3_probation_active_log": history["td3_probation_active_log"],
        "td3_probation_trigger_log": history["td3_probation_trigger_log"],
        "td3_probation_trigger_count": int(history.get("_td3_probation_trigger_count", 0)),
        "td3_warm_release_reference_reward": history.get("_td3_warm_release_reference_reward"),
        "z_safety_effective_cap_log": history["z_safety_effective_cap_log"],
        "z_safety_requested_norm_before_log": history["z_safety_requested_norm_before_log"],
        "z_safety_requested_norm_after_log": history["z_safety_requested_norm_after_log"],
        "z_safety_requested_projection_scale_log": history["z_safety_requested_projection_scale_log"],
        "z_safety_requested_projection_active_log": history["z_safety_requested_projection_active_log"],
        "z_safety_requested_coord_clip_active_log": history["z_safety_requested_coord_clip_active_log"],
        "z_safety_requested_vector_projection_active_log": history["z_safety_requested_vector_projection_active_log"],
        "z_safety_ls_norm_before_log": history["z_safety_ls_norm_before_log"],
        "z_safety_ls_norm_after_log": history["z_safety_ls_norm_after_log"],
        "z_safety_ls_projection_scale_log": history["z_safety_ls_projection_scale_log"],
        "z_safety_ls_projection_active_log": history["z_safety_ls_projection_active_log"],
        "z_safety_ls_coord_clip_active_log": history["z_safety_ls_coord_clip_active_log"],
        "z_safety_ls_vector_projection_active_log": history["z_safety_ls_vector_projection_active_log"],
        "shadow_z_safety_effective_cap_log": history["shadow_z_safety_effective_cap_log"],
        "shadow_z_safety_requested_norm_before_log": history["shadow_z_safety_requested_norm_before_log"],
        "shadow_z_safety_requested_norm_after_log": history["shadow_z_safety_requested_norm_after_log"],
        "shadow_z_safety_requested_projection_scale_log": history["shadow_z_safety_requested_projection_scale_log"],
        "shadow_z_safety_requested_projection_active_log": history["shadow_z_safety_requested_projection_active_log"],
        "shadow_z_safety_requested_coord_clip_active_log": history["shadow_z_safety_requested_coord_clip_active_log"],
        "shadow_z_safety_requested_vector_projection_active_log": history["shadow_z_safety_requested_vector_projection_active_log"],
        "shadow_td3_priority_phase_log": history["shadow_td3_priority_phase_log"],
        "shadow_td3_priority_authority_scale_log": history["shadow_td3_priority_authority_scale_log"],
        "shadow_td3_priority_allowed_log": history["shadow_td3_priority_allowed_log"],
        "shadow_ls_priority_allowed_log": history["shadow_ls_priority_allowed_log"],
        "shadow_nominal_fallback_eligible_log": history["shadow_nominal_fallback_eligible_log"],
        "shadow_bc_handoff_authority_log": history["shadow_bc_handoff_authority_log"],
        "shadow_bc_handoff_delta_norm_log": history["shadow_bc_handoff_delta_norm_log"],
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
        "requested_legacy_hard_gate_pass_log": history["requested_legacy_hard_gate_pass_log"],
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
    result_bundle.update(
        build_behavioral_cloning_bundle_fields(
            history.get("_behavioral_cloning_schedule", {}),
            history.get("_behavioral_cloning_logs", init_behavioral_cloning_logs(int(ctx["nFE"]))),
        )
    )
    result_bundle.update(
        build_bc_handoff_bundle_fields(
            history.get("_behavioral_cloning_schedule", {}),
            history.get("_bc_handoff_logs", init_bc_handoff_logs(int(ctx["nFE"]), int(np.asarray(basis_blocks).shape[0]))),
        )
    )
    result_bundle.update(
        build_protected_bc_release_gate_bundle_fields(
            history.get("_protected_bc_release_gate", {}),
            prefix="rl_",
        )
    )
    return result_bundle
