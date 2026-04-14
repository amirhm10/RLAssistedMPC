from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.optimize as spo

from Simulation.mpc import augment_state_space
from systems.polymer.data_io import canonical_baseline_path
from utils.helpers import (
    apply_min_max,
    build_polymer_disturbance_schedule,
    disturbance_profile_from_schedule,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
    shift_control_sequence,
    step_system_with_disturbance,
)
from utils.reidentification import (
    REIDENTIFICATION_BASIS_FAMILY,
    REIDENTIFICATION_BLEND_EXTRA_CLIP,
    REIDENTIFICATION_BLEND_RESIDUAL_SCALE,
    REIDENTIFICATION_CANDIDATE_GUARD_MODE,
    REIDENTIFICATION_COMPONENT_MODE,
    REIDENTIFICATION_LOG_THETA_CLIPPING,
    REIDENTIFICATION_NORMALIZE_BLEND_EXTRAS,
    REIDENTIFICATION_OBSERVER_ALIGNMENT,
    RollingIDBuffer,
    attempt_observer_refresh,
    blend_prediction_model,
    build_or_load_polymer_reidentification_basis,
    build_reidentification_policy_state,
    eta_to_raw_action,
    map_action_to_dual_eta,
    normalize_force_eta_constant,
    reconstruct_model_from_theta,
    resolve_reidentification_lambda_vectors,
    resolve_reidentification_theta_bounds,
    select_reidentified_model,
    smooth_eta,
    solve_reidentification_batch,
)
from utils.replay_snapshot import attach_single_agent_replay_snapshot
from utils.state_features import build_rl_state, compute_tracking_scale_now, resolve_mismatch_settings


def _relative_fro(candidate, nominal):
    candidate = np.asarray(candidate, float)
    nominal = np.asarray(nominal, float)
    denom = float(np.linalg.norm(nominal, ord="fro"))
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(candidate - nominal, ord="fro") / denom)


def _solve_prediction_step(mpc_obj, y_sp, u_prev_dev, x0_model, initial_guess, bounds, step_idx):
    try:
        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
            np.asarray(initial_guess, float),
            bounds=bounds,
            constraints=[],
        )
    except Exception as exc:
        raise RuntimeError(f"Online re-identification MPC solve failed at step {step_idx}: {exc}") from exc

    success = bool(
        sol is not None
        and bool(getattr(sol, "success", True))
        and getattr(sol, "x", None) is not None
        and np.all(np.isfinite(np.asarray(sol.x, float)))
        and np.isfinite(float(getattr(sol, "fun", 0.0)))
    )
    if not success:
        message = str(getattr(sol, "message", "unknown solver failure"))
        raise RuntimeError(f"Online re-identification MPC solve failed at step {step_idx}: {message}")
    return sol


def _update_observer_state(*, A_obs_aug, B_obs_aug, L_obs, x_prev, u_dev, y_prev_scaled, yhat_current):
    return A_obs_aug @ x_prev + B_obs_aug @ u_dev + L_obs @ (y_prev_scaled - yhat_current).T


def _resolve_offline_basis_paths(reid_cfg, runtime_ctx, run_mode: str) -> tuple[Path, Path]:
    baseline_override = reid_cfg.get("baseline_mpc_path", runtime_ctx.get("baseline_mpc_path"))
    if baseline_override is not None:
        baseline_path = Path(baseline_override).expanduser().resolve()
    else:
        repo_root = Path(reid_cfg.get("repo_root", runtime_ctx.get("repo_root", Path.cwd()))).expanduser().resolve()
        data_override = reid_cfg.get("data_dir_override", runtime_ctx.get("polymer_data_dir", None))
        baseline_path = canonical_baseline_path(repo_root, run_mode, data_override=data_override).resolve()

    cache_dir = reid_cfg.get("polymer_data_dir", runtime_ctx.get("polymer_data_dir", baseline_path.parent))
    return baseline_path, Path(cache_dir).expanduser().resolve()


def run_reidentification_supervisor(reid_cfg, runtime_ctx):
    """
    Run the polymer low-rank re-identification + dual-RL-blend workflow.

    The observer remains nominal by default. When enabled, observer refresh is
    applied episodically using interval-mean accepted identified models.
    """

    system = runtime_ctx["system"]
    agent = runtime_ctx["agent"]
    mpc_obj = runtime_ctx["MPC_obj"]
    steady_states = runtime_ctx["steady_states"]
    min_max_dict = runtime_ctx["min_max_dict"]
    data_min = np.asarray(runtime_ctx["data_min"], float)
    data_max = np.asarray(runtime_ctx["data_max"], float)
    A_aug_nom = np.asarray(runtime_ctx["A_aug"], float)
    B_aug_nom = np.asarray(runtime_ctx["B_aug"], float)
    C_aug = np.asarray(runtime_ctx["C_aug"], float)
    poles = np.asarray(runtime_ctx["poles"], float)
    y_sp_scenario = np.asarray(runtime_ctx["y_sp_scenario"], float)
    reward_fn = runtime_ctx["reward_fn"]
    system_stepper = runtime_ctx.get("system_stepper")
    system_metadata = runtime_ctx.get("system_metadata")
    disturbance_labels = runtime_ctx.get("disturbance_labels")

    n_inputs = int(B_aug_nom.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug_nom.shape[0])
    n_phys = n_states - n_outputs

    A0_phys = np.asarray(runtime_ctx.get("A", A_aug_nom[:n_phys, :n_phys]), float)
    B0_phys = np.asarray(runtime_ctx.get("B", B_aug_nom[:n_phys, :]), float)
    C_phys = np.asarray(runtime_ctx.get("C", C_aug[:, :n_phys]), float)

    agent_kind = str(reid_cfg["agent_kind"]).lower()
    run_mode = str(reid_cfg["run_mode"]).lower()
    state_mode = str(reid_cfg.get("state_mode", "standard")).lower()
    if agent_kind not in {"td3", "sac"}:
        raise ValueError("reid_cfg['agent_kind'] must be 'td3' or 'sac'.")
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("reid_cfg['run_mode'] must be 'nominal' or 'disturb'.")

    mismatch_cfg = resolve_mismatch_settings(
        state_mode=state_mode,
        mismatch_cfg=reid_cfg,
        reward_params=runtime_ctx.get("reward_params", {}),
        y_sp_scenario=y_sp_scenario,
        steady_states=steady_states,
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
    )
    mismatch_clip = mismatch_cfg["mismatch_clip"]

    use_shifted_mpc_warm_start = bool(reid_cfg.get("use_shifted_mpc_warm_start", False))
    id_window = int(reid_cfg.get("id_window", 80))
    id_update_period = int(reid_cfg.get("id_update_period", 5))
    if id_window <= 0 or id_update_period <= 0:
        raise ValueError("id_window and id_update_period must be positive.")

    rank_A = int(reid_cfg.get("rank_A", 6))
    rank_B = int(reid_cfg.get("rank_B", 2))
    offline_window = int(reid_cfg.get("offline_window", 80))
    offline_stride = int(reid_cfg.get("offline_stride", 80))
    lambda_A_off = float(reid_cfg.get("lambda_A_off", 1e-4))
    lambda_B_off = float(reid_cfg.get("lambda_B_off", 1e-3))
    if rank_A <= 0 or rank_B <= 0:
        raise ValueError("rank_A and rank_B must be positive.")

    delta_A_max = float(reid_cfg.get("delta_A_max", 0.10))
    delta_B_max = float(reid_cfg.get("delta_B_max", 0.10))
    if delta_A_max < 0.0 or delta_B_max < 0.0:
        raise ValueError("delta_A_max and delta_B_max must be non-negative.")

    blend_extra_clip = float(reid_cfg.get("blend_extra_clip", REIDENTIFICATION_BLEND_EXTRA_CLIP))
    blend_residual_scale = float(reid_cfg.get("blend_residual_scale", REIDENTIFICATION_BLEND_RESIDUAL_SCALE))
    log_theta_clipping = bool(reid_cfg.get("log_theta_clipping", REIDENTIFICATION_LOG_THETA_CLIPPING))
    eta_tau = np.asarray(
        [
            float(reid_cfg.get("eta_tau_A", 0.1)),
            float(reid_cfg.get("eta_tau_B", 0.1)),
        ],
        dtype=float,
    )
    force_eta_constant = normalize_force_eta_constant(reid_cfg.get("force_eta_constant"))
    disable_identification = bool(reid_cfg.get("disable_identification", False))
    observer_refresh_enabled = bool(reid_cfg.get("observer_refresh_enabled", False))
    observer_refresh_every_episodes = int(reid_cfg.get("observer_refresh_every_episodes", 10))
    rho_obs = float(reid_cfg.get("rho_obs", 0.25))
    if observer_refresh_every_episodes <= 0:
        raise ValueError("observer_refresh_every_episodes must be positive.")

    baseline_path, cache_dir = _resolve_offline_basis_paths(reid_cfg, runtime_ctx, run_mode)
    basis = build_or_load_polymer_reidentification_basis(
        baseline_path=baseline_path,
        cache_dir=cache_dir,
        A_ref=A0_phys,
        B_ref=B0_phys,
        rank_A=rank_A,
        rank_B=rank_B,
        offline_window=offline_window,
        offline_stride=offline_stride,
        lambda_A_off=lambda_A_off,
        lambda_B_off=lambda_B_off,
        run_mode=run_mode,
    )
    theta_low, theta_high = resolve_reidentification_theta_bounds(basis=basis, cfg=reid_cfg)
    lambda_prev_vec, lambda_0_vec = resolve_reidentification_lambda_vectors(basis=basis, cfg=reid_cfg)
    theta_A_indices = np.asarray(basis["theta_A_indices"], dtype=int)
    theta_B_indices = np.asarray(basis["theta_B_indices"], dtype=int)
    theta_nominal = np.zeros(basis["theta_dim"], dtype=float)

    (
        y_sp,
        nFE,
        sub_episodes_changes_dict,
        time_in_sub_episodes,
        test_train_dict,
        warm_start_step,
        qi,
        qs,
        ha,
    ) = generate_setpoints_training_rl_gradually(
        y_sp_scenario,
        int(reid_cfg["n_tests"]),
        int(reid_cfg["set_points_len"]),
        int(reid_cfg["warm_start"]),
        list(reid_cfg["test_cycle"]),
        float(reid_cfg["nominal_qi"]),
        float(reid_cfg["nominal_qs"]),
        float(reid_cfg["nominal_ha"]),
        float(reid_cfg["qi_change"]),
        float(reid_cfg["qs_change"]),
        float(reid_cfg["ha_change"]),
    )

    disturbance_schedule = None
    if run_mode == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    cont_h = int(reid_cfg.get("cont_h", 1))
    b_min = np.asarray(reid_cfg["b_min"], float).reshape(-1)
    b_max = np.asarray(reid_cfg["b_max"], float).reshape(-1)
    bounds = tuple((float(b_min[j]), float(b_max[j])) for _ in range(cont_h) for j in range(n_inputs))
    ic_opt = np.zeros(n_inputs * cont_h)

    y_system = np.zeros((nFE + 1, n_outputs))
    y_system[0, :] = np.asarray(system.current_output, float)
    u_mpc = np.zeros((nFE, n_inputs))
    rewards = np.zeros(nFE)
    avg_rewards = []
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    delta_y_storage = np.zeros((nFE, n_outputs))
    delta_u_storage = np.zeros((nFE, n_inputs))
    innovation_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_scale_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None

    eta_A_log = np.zeros(nFE)
    eta_B_log = np.zeros(nFE)
    eta_A_raw_log = np.zeros(nFE)
    eta_B_raw_log = np.zeros(nFE)
    action_raw_log = np.zeros((nFE, 2))
    theta_hat_log = np.zeros((nFE, theta_nominal.size))
    theta_active_log = np.zeros((nFE, theta_nominal.size))
    theta_candidate_log = np.zeros((nFE, theta_nominal.size))
    theta_unclipped_log = np.zeros((nFE, theta_nominal.size))
    theta_lower_hit_mask_log = np.zeros((nFE, theta_nominal.size), dtype=int)
    theta_upper_hit_mask_log = np.zeros((nFE, theta_nominal.size), dtype=int)
    theta_clipped_fraction_log = np.zeros(nFE)
    id_residual_norm_log = np.zeros(nFE)
    id_condition_number_log = np.zeros(nFE)
    id_update_event_log = np.zeros(nFE, dtype=int)
    id_update_success_log = np.zeros(nFE, dtype=int)
    id_fallback_log = np.zeros(nFE, dtype=int)
    id_source_code_log = np.zeros(nFE, dtype=int)
    id_candidate_valid_log = np.zeros(nFE, dtype=int)
    active_A_model_delta_ratio_log = np.zeros(nFE)
    active_B_model_delta_ratio_log = np.zeros(nFE)
    candidate_A_model_delta_ratio_log = np.zeros(nFE)
    candidate_B_model_delta_ratio_log = np.zeros(nFE)
    pred_A_model_delta_ratio_log = np.zeros(nFE)
    pred_B_model_delta_ratio_log = np.zeros(nFE)
    observer_A_model_delta_ratio_log = np.zeros(nFE)
    observer_B_model_delta_ratio_log = np.zeros(nFE)
    observer_refresh_event_log = np.zeros(nFE, dtype=int)
    observer_refresh_success_log = np.zeros(nFE, dtype=int)

    A_obs_current_phys = A0_phys.copy()
    B_obs_current_phys = B0_phys.copy()
    A_obs_aug = A_aug_nom.copy()
    B_obs_aug = B_aug_nom.copy()
    L_obs_current = runtime_ctx.get("L_obs_nominal")
    if L_obs_current is None:
        from utils.observer import compute_observer_gain  # local import to keep module interface compact

        L_obs_current = compute_observer_gain(A_obs_aug, C_aug, poles)
    L_obs_current = np.asarray(L_obs_current, float)

    A_id_phys = A0_phys.copy()
    B_id_phys = B0_phys.copy()
    theta_hat = theta_nominal.copy()
    eta_prev = np.zeros(2, dtype=float)
    last_id_residual_norm = 0.0
    last_id_condition_number = 0.0
    last_candidate_A_ratio = 0.0
    last_candidate_B_ratio = 0.0
    last_candidate_valid_flag = 0.0
    last_observer_refresh_success_flag = 0.0
    last_fallback_flag = 0.0
    invalid_id_solve_count = 0
    id_solver_failure_count = 0
    id_update_success_count = 0
    observer_refresh_attempt_count = 0
    observer_refresh_success_count = 0

    id_buffer = RollingIDBuffer(
        maxlen=max(id_window, int(reid_cfg.get("buffer_maxlen", max(4 * id_window, 200)))),
        state_dim=n_phys,
        input_dim=n_inputs,
    )
    accepted_A_interval = []
    accepted_B_interval = []
    episode_counter = 0

    test = False
    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_pred = C_aug @ xhatdhat[:, i]
        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])

        tracking_scale_now = None
        if state_mode == "mismatch":
            _, tracking_scale_now = compute_tracking_scale_now(
                y_sp_phys=y_sp_phys,
                data_min=data_min,
                data_max=data_max,
                n_inputs=n_inputs,
                k_rel=mismatch_cfg["k_rel"],
                band_floor_phys=mismatch_cfg["band_floor_phys"],
                tracking_eta_tol=mismatch_cfg["tracking_eta_tol"],
                tracking_scale_floor=mismatch_cfg["tracking_scale_floor"],
            )

        base_state, state_debug = build_rl_state(
            min_max_dict=min_max_dict,
            x_d_states=xhatdhat[:, i],
            y_sp=y_sp[i, :],
            u=scaled_current_input_dev,
            state_mode=state_mode,
            y_prev_scaled=y_prev_scaled,
            yhat_pred=yhat_pred,
            innovation_scale_ref=mismatch_cfg["innovation_scale_ref"],
            tracking_scale_now=tracking_scale_now,
            mismatch_clip=mismatch_clip,
        )
        current_active_A_ratio = _relative_fro(A_id_phys, A0_phys)
        current_active_B_ratio = _relative_fro(B_id_phys, B0_phys)
        current_rl_state = build_reidentification_policy_state(
            base_state=base_state,
            prev_eta_A=float(eta_prev[0]),
            prev_eta_B=float(eta_prev[1]),
            residual_norm=last_id_residual_norm,
            active_A_ratio=current_active_A_ratio,
            active_B_ratio=current_active_B_ratio,
            candidate_valid_flag=last_candidate_valid_flag,
            observer_refresh_success_flag=last_observer_refresh_success_flag,
            fallback_flag=last_fallback_flag,
            delta_A_max=delta_A_max,
            delta_B_max=delta_B_max,
            blend_extra_clip=blend_extra_clip,
            blend_residual_scale=blend_residual_scale,
        )
        if innovation_log is not None:
            innovation_log[i, :] = state_debug["innovation"]
            tracking_error_log[i, :] = state_debug["tracking_error"]
            tracking_scale_log[i, :] = state_debug["tracking_scale_now"]

        if i <= warm_start_step:
            raw_action = np.array([-1.0, -1.0], dtype=float)
            eta_raw = np.zeros(2, dtype=float)
            eta = np.zeros(2, dtype=float)
        elif force_eta_constant is not None:
            raw_action = eta_to_raw_action(force_eta_constant)
            eta_raw = np.asarray(force_eta_constant, dtype=float).copy()
            eta = smooth_eta(eta_prev, eta_raw, eta_tau)
        else:
            if not test:
                action = np.asarray(agent.take_action(current_rl_state, explore=True), float).reshape(-1)
            else:
                action = np.asarray(agent.act_eval(current_rl_state), float).reshape(-1)
            raw_action, eta_raw = map_action_to_dual_eta(action[:2])
            eta = smooth_eta(eta_prev, eta_raw, eta_tau)

        action_raw_log[i, :] = raw_action
        eta_A_raw_log[i] = float(eta_raw[0])
        eta_B_raw_log[i] = float(eta_raw[1])
        eta_A_log[i] = float(eta[0])
        eta_B_log[i] = float(eta[1])

        A_pred_phys, B_pred_phys = blend_prediction_model(
            A0_phys=A0_phys,
            B0_phys=B0_phys,
            A_id_phys=A_id_phys,
            B_id_phys=B_id_phys,
            eta_A=float(eta[0]),
            eta_B=float(eta[1]),
        )
        A_pred_aug, B_pred_aug, _ = augment_state_space(A_pred_phys, B_pred_phys, C_phys)
        pred_A_model_delta_ratio_log[i] = _relative_fro(A_pred_phys, A0_phys)
        pred_B_model_delta_ratio_log[i] = _relative_fro(B_pred_phys, B0_phys)
        active_A_model_delta_ratio_log[i] = current_active_A_ratio
        active_B_model_delta_ratio_log[i] = current_active_B_ratio
        candidate_A_model_delta_ratio_log[i] = last_candidate_A_ratio
        candidate_B_model_delta_ratio_log[i] = last_candidate_B_ratio
        observer_A_model_delta_ratio_log[i] = _relative_fro(A_obs_current_phys, A0_phys)
        observer_B_model_delta_ratio_log[i] = _relative_fro(B_obs_current_phys, B0_phys)
        theta_hat_log[i, :] = theta_hat
        theta_active_log[i, :] = theta_hat
        theta_candidate_log[i, :] = theta_hat
        theta_unclipped_log[i, :] = theta_hat
        id_residual_norm_log[i] = last_id_residual_norm
        id_condition_number_log[i] = last_id_condition_number
        id_candidate_valid_log[i] = int(last_candidate_valid_flag > 0.5)

        ic_opt_step = ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * cont_h)
        mpc_obj.A = A_pred_aug
        mpc_obj.B = B_pred_aug
        sol = _solve_prediction_step(
            mpc_obj=mpc_obj,
            y_sp=y_sp[i, :],
            u_prev_dev=scaled_current_input_dev,
            x0_model=xhatdhat[:, i],
            initial_guess=ic_opt_step,
            bounds=bounds,
            step_idx=i,
        )

        if use_shifted_mpc_warm_start:
            ic_opt = shift_control_sequence(sol.x[: n_inputs * cont_h], n_inputs, cont_h)
        else:
            ic_opt = np.zeros(n_inputs * cont_h)

        u_mpc[i, :] = sol.x[:n_inputs] + ss_scaled_inputs
        u_plant = reverse_min_max(u_mpc[i, :], data_min[:n_inputs], data_max[:n_inputs])
        delta_u = u_mpc[i, :] - scaled_current_input
        delta_u_storage[i, :] = delta_u

        system.current_input = u_plant
        step_system_with_disturbance(
            system,
            idx=i,
            disturbance_schedule=disturbance_schedule,
            system_stepper=system_stepper,
        )
        y_system[i + 1, :] = np.asarray(system.current_output, float)

        y_current_scaled = apply_min_max(y_system[i + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        delta_y = y_current_scaled - y_sp[i, :]
        delta_y_storage[i, :] = delta_y

        yhat[:, i] = yhat_pred
        xhatdhat[:, i + 1] = _update_observer_state(
            A_obs_aug=A_obs_aug,
            B_obs_aug=B_obs_aug,
            L_obs=L_obs_current,
            x_prev=xhatdhat[:, i],
            u_dev=(u_mpc[i, :] - ss_scaled_inputs),
            y_prev_scaled=y_prev_scaled,
            yhat_current=yhat[:, i],
        )

        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_mpc[i, :] - ss_scaled_inputs
        yhat_next_pred = C_aug @ xhatdhat[:, i + 1]
        next_tracking_scale_now = None
        if state_mode == "mismatch":
            _, next_tracking_scale_now = compute_tracking_scale_now(
                y_sp_phys=y_sp_phys,
                data_min=data_min,
                data_max=data_max,
                n_inputs=n_inputs,
                k_rel=mismatch_cfg["k_rel"],
                band_floor_phys=mismatch_cfg["band_floor_phys"],
                tracking_eta_tol=mismatch_cfg["tracking_eta_tol"],
                tracking_scale_floor=mismatch_cfg["tracking_scale_floor"],
            )

        next_base_state, _ = build_rl_state(
            min_max_dict=min_max_dict,
            x_d_states=xhatdhat[:, i + 1],
            y_sp=y_sp[i, :],
            u=next_u_dev,
            state_mode=state_mode,
            y_prev_scaled=y_current_scaled,
            yhat_pred=yhat_next_pred,
            innovation_scale_ref=mismatch_cfg["innovation_scale_ref"],
            tracking_scale_now=next_tracking_scale_now,
            mismatch_clip=mismatch_clip,
        )

        x_phys_t = xhatdhat[:n_phys, i]
        x_phys_tp1 = xhatdhat[:n_phys, i + 1]
        id_buffer.push(x_phys_t, next_u_dev, x_phys_tp1)

        update_success_flag = 0.0
        fallback_flag = 0.0
        candidate_valid_flag = last_candidate_valid_flag
        observer_refresh_success_state_flag = last_observer_refresh_success_flag

        if not disable_identification and (i + 1) % id_update_period == 0 and len(id_buffer) >= id_window:
            id_update_event_log[i] = 1
            solver_result = solve_reidentification_batch(
                batch=id_buffer.get_recent(id_window),
                A0_phys=A0_phys,
                B0_phys=B0_phys,
                basis=basis,
                theta_prev=theta_hat,
                cfg=reid_cfg,
            )
            last_id_residual_norm = float(solver_result["residual_norm"])
            last_id_condition_number = float(solver_result["condition_number"])

            if solver_result["success"]:
                A_candidate, B_candidate = reconstruct_model_from_theta(
                    A0_phys=A0_phys,
                    B0_phys=B0_phys,
                    basis=basis,
                    theta=solver_result["theta"],
                )
                selection = select_reidentified_model(
                    A_candidate=A_candidate,
                    B_candidate=B_candidate,
                    theta_candidate=solver_result["theta"],
                    theta_unclipped=solver_result.get("theta_unclipped"),
                    solve_success=True,
                    A0_phys=A0_phys,
                    B0_phys=B0_phys,
                    A_prev=A_id_phys,
                    B_prev=B_id_phys,
                    theta_prev=theta_hat,
                    theta_low=theta_low,
                    theta_high=theta_high,
                    delta_A_max=delta_A_max,
                    delta_B_max=delta_B_max,
                )
                theta_candidate = np.asarray(solver_result["theta"], float)
                theta_unclipped = np.asarray(solver_result.get("theta_unclipped", solver_result["theta"]), float)
                candidate_A_ratio = _relative_fro(A_candidate, A0_phys)
                candidate_B_ratio = _relative_fro(B_candidate, B0_phys)
            else:
                selection = select_reidentified_model(
                    A_candidate=None,
                    B_candidate=None,
                    theta_candidate=None,
                    theta_unclipped=None,
                    solve_success=False,
                    A0_phys=A0_phys,
                    B0_phys=B0_phys,
                    A_prev=A_id_phys,
                    B_prev=B_id_phys,
                    theta_prev=theta_hat,
                    theta_low=theta_low,
                    theta_high=theta_high,
                    delta_A_max=delta_A_max,
                    delta_B_max=delta_B_max,
                )
                theta_candidate = theta_hat.copy()
                theta_unclipped = theta_hat.copy()
                candidate_A_ratio = current_active_A_ratio
                candidate_B_ratio = current_active_B_ratio
                id_solver_failure_count += 1

            A_id_phys = selection["A_active"]
            B_id_phys = selection["B_active"]
            theta_hat = selection["theta_active"]
            update_success_flag = float(selection["update_success"])
            fallback_flag = float(selection["fallback_used"])
            candidate_valid_flag = float(selection["candidate_eval"]["valid"])
            id_update_success_log[i] = int(selection["update_success"])
            id_fallback_log[i] = int(selection["fallback_used"])
            id_source_code_log[i] = 1 if selection["source"] == "candidate" else 2
            id_candidate_valid_log[i] = int(selection["candidate_eval"]["valid"])
            last_candidate_A_ratio = float(candidate_A_ratio)
            last_candidate_B_ratio = float(candidate_B_ratio)

            if selection["update_success"]:
                id_update_success_count += 1
                accepted_A_interval.append(np.asarray(A_id_phys, float).copy())
                accepted_B_interval.append(np.asarray(B_id_phys, float).copy())
            else:
                invalid_id_solve_count += 1

            theta_hat_log[i, :] = theta_hat
            theta_active_log[i, :] = theta_hat
            theta_candidate_log[i, :] = theta_candidate
            theta_unclipped_log[i, :] = theta_unclipped
            if log_theta_clipping:
                theta_lower_hit_mask_log[i, :] = selection["theta_eval"]["lower_hit_mask"]
                theta_upper_hit_mask_log[i, :] = selection["theta_eval"]["upper_hit_mask"]
                theta_clipped_fraction_log[i] = float(selection["theta_eval"]["clipped_fraction"])
            active_A_model_delta_ratio_log[i] = _relative_fro(A_id_phys, A0_phys)
            active_B_model_delta_ratio_log[i] = _relative_fro(B_id_phys, B0_phys)
            candidate_A_model_delta_ratio_log[i] = candidate_A_ratio
            candidate_B_model_delta_ratio_log[i] = candidate_B_ratio

        if i in sub_episodes_changes_dict:
            episode_counter += 1
            if observer_refresh_enabled and i > warm_start_step and episode_counter % observer_refresh_every_episodes == 0:
                observer_refresh_event_log[i] = 1
                observer_refresh_attempt_count += 1
                if accepted_A_interval and accepted_B_interval:
                    refresh = attempt_observer_refresh(
                        A_obs_current_phys=A_obs_current_phys,
                        B_obs_current_phys=B_obs_current_phys,
                        A_interval_mean=np.mean(np.stack(accepted_A_interval, axis=0), axis=0),
                        B_interval_mean=np.mean(np.stack(accepted_B_interval, axis=0), axis=0),
                        A0_phys=A0_phys,
                        B0_phys=B0_phys,
                        C_phys=C_phys,
                        C_aug=C_aug,
                        poles=poles,
                        rho_obs=rho_obs,
                        delta_A_max=delta_A_max,
                        delta_B_max=delta_B_max,
                    )
                    if refresh["accepted"]:
                        A_obs_current_phys = refresh["A_obs_phys"]
                        B_obs_current_phys = refresh["B_obs_phys"]
                        A_obs_aug = refresh["A_obs_aug"]
                        B_obs_aug = refresh["B_obs_aug"]
                        L_obs_current = refresh["L_obs"]
                        observer_refresh_success_log[i] = 1
                        observer_refresh_success_state_flag = 1.0
                        observer_refresh_success_count += 1
                    else:
                        observer_refresh_success_state_flag = 0.0
                else:
                    observer_refresh_success_state_flag = 0.0
                accepted_A_interval = []
                accepted_B_interval = []

        next_active_A_ratio = _relative_fro(A_id_phys, A0_phys)
        next_active_B_ratio = _relative_fro(B_id_phys, B0_phys)
        next_rl_state = build_reidentification_policy_state(
            base_state=next_base_state,
            prev_eta_A=float(eta[0]),
            prev_eta_B=float(eta[1]),
            residual_norm=last_id_residual_norm,
            active_A_ratio=next_active_A_ratio,
            active_B_ratio=next_active_B_ratio,
            candidate_valid_flag=candidate_valid_flag,
            observer_refresh_success_flag=observer_refresh_success_state_flag,
            fallback_flag=fallback_flag,
            delta_A_max=delta_A_max,
            delta_B_max=delta_B_max,
            blend_extra_clip=blend_extra_clip,
            blend_residual_scale=blend_residual_scale,
        )

        if not test:
            agent.push(
                current_rl_state,
                np.asarray(raw_action, np.float32),
                reward,
                next_rl_state,
                0.0,
            )
            if i >= warm_start_step:
                agent.train_step()

        eta_prev = eta
        last_candidate_valid_flag = candidate_valid_flag
        last_observer_refresh_success_flag = observer_refresh_success_state_flag
        last_fallback_flag = fallback_flag

        if i in sub_episodes_changes_dict:
            lo = max(0, i - time_in_sub_episodes + 1)
            hi = i + 1
            avg_rewards.append(float(np.mean(rewards[lo:hi])))
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_rewards[-1],
                "| eta_A:",
                float(np.mean(eta_A_log[lo:hi])),
                "| eta_B:",
                float(np.mean(eta_B_log[lo:hi])),
                "| ||dA||/||A0||:",
                float(np.mean(active_A_model_delta_ratio_log[lo:hi])),
                "| ||dB||/||B0||:",
                float(np.mean(active_B_model_delta_ratio_log[lo:hi])),
            )

    mpc_obj.A = A_aug_nom
    mpc_obj.B = B_aug_nom
    if hasattr(agent, "flush_nstep"):
        agent.flush_nstep()

    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if run_mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )
    basis_metadata = dict(basis.get("metadata") or {})

    result_bundle = {
        "agent_kind": agent_kind,
        "run_mode": run_mode,
        "algorithm": agent_kind,
        "state_mode": state_mode,
        "system_metadata": system_metadata,
        "notebook_source": reid_cfg.get("notebook_source"),
        "config_snapshot": dict(reid_cfg),
        "seed": reid_cfg.get("seed"),
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": int(nFE),
        "delta_t": float(system.delta_t),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y": y_system,
        "u": reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs]),
        "avg_rewards": np.asarray(avg_rewards, float),
        "rewards_step": rewards,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "data_min": data_min,
        "data_max": data_max,
        "yhat": yhat,
        "xhatdhat": xhatdhat,
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "warm_start_step": int(warm_start_step),
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
        "innovation_log": innovation_log,
        "tracking_error_log": tracking_error_log,
        "innovation_scale_ref": mismatch_cfg["innovation_scale_ref"],
        "tracking_scale_log": tracking_scale_log,
        "band_ref_scaled": mismatch_cfg["band_ref_scaled"],
        "mismatch_clip": mismatch_clip,
        "mpc_horizons": (
            int(reid_cfg["predict_h"]),
            int(reid_cfg["cont_h"]),
        )
        if "predict_h" in reid_cfg and "cont_h" in reid_cfg
        else None,
        "estimator_mode": "episodic_observer_refresh" if observer_refresh_enabled else "nominal_observer",
        "prediction_model_mode": "online_lowrank_reidentification_dual_blend",
        "basis_family": REIDENTIFICATION_BASIS_FAMILY,
        "id_component_mode": REIDENTIFICATION_COMPONENT_MODE,
        "observer_update_alignment": REIDENTIFICATION_OBSERVER_ALIGNMENT,
        "candidate_guard_mode": REIDENTIFICATION_CANDIDATE_GUARD_MODE,
        "normalize_blend_extras": REIDENTIFICATION_NORMALIZE_BLEND_EXTRAS,
        "blend_extra_clip": blend_extra_clip,
        "blend_residual_scale": blend_residual_scale,
        "log_theta_clipping": log_theta_clipping,
        "id_basis_name": basis["basis_name"],
        "theta_labels": list(basis["theta_labels"]),
        "theta_labels_A": list(basis["theta_labels_A"]),
        "theta_labels_B": list(basis["theta_labels_B"]),
        "alpha_labels": list(basis.get("alpha_labels", [])),
        "beta_labels": list(basis.get("beta_labels", [])),
        "theta_A_indices": theta_A_indices,
        "theta_B_indices": theta_B_indices,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "theta_low_A": np.asarray(theta_low[theta_A_indices], float),
        "theta_high_A": np.asarray(theta_high[theta_A_indices], float),
        "theta_low_B": np.asarray(theta_low[theta_B_indices], float),
        "theta_high_B": np.asarray(theta_high[theta_B_indices], float),
        "lambda_prev_vector": lambda_prev_vec,
        "lambda_0_vector": lambda_0_vec,
        "lambda_prev_A": np.asarray(lambda_prev_vec[theta_A_indices], float),
        "lambda_prev_B": np.asarray(lambda_prev_vec[theta_B_indices], float),
        "lambda_0_A": np.asarray(lambda_0_vec[theta_A_indices], float),
        "lambda_0_B": np.asarray(lambda_0_vec[theta_B_indices], float),
        "theta_hat_log": theta_hat_log,
        "theta_active_log": theta_active_log,
        "theta_candidate_log": theta_candidate_log,
        "theta_unclipped_log": theta_unclipped_log,
        "theta_lower_hit_mask_log": theta_lower_hit_mask_log,
        "theta_upper_hit_mask_log": theta_upper_hit_mask_log,
        "theta_clipped_fraction_log": theta_clipped_fraction_log,
        "id_residual_norm_log": id_residual_norm_log,
        "id_condition_number_log": id_condition_number_log,
        "id_update_event_log": id_update_event_log,
        "id_update_success_log": id_update_success_log,
        "id_fallback_log": id_fallback_log,
        "id_source_code_log": id_source_code_log,
        "id_candidate_valid_log": id_candidate_valid_log,
        "active_A_model_delta_ratio_log": active_A_model_delta_ratio_log,
        "active_B_model_delta_ratio_log": active_B_model_delta_ratio_log,
        "candidate_A_model_delta_ratio_log": candidate_A_model_delta_ratio_log,
        "candidate_B_model_delta_ratio_log": candidate_B_model_delta_ratio_log,
        "pred_A_model_delta_ratio_log": pred_A_model_delta_ratio_log,
        "pred_B_model_delta_ratio_log": pred_B_model_delta_ratio_log,
        "observer_A_model_delta_ratio_log": observer_A_model_delta_ratio_log,
        "observer_B_model_delta_ratio_log": observer_B_model_delta_ratio_log,
        "observer_refresh_event_log": observer_refresh_event_log,
        "observer_refresh_success_log": observer_refresh_success_log,
        "observer_refresh_enabled": observer_refresh_enabled,
        "observer_refresh_every_episodes": int(observer_refresh_every_episodes),
        "observer_refresh_attempt_count": int(observer_refresh_attempt_count),
        "observer_refresh_success_count": int(observer_refresh_success_count),
        "rho_obs": float(rho_obs),
        "rank_A": int(rank_A),
        "rank_B": int(rank_B),
        "offline_window": int(offline_window),
        "offline_stride": int(offline_stride),
        "lambda_A_off": float(lambda_A_off),
        "lambda_B_off": float(lambda_B_off),
        "offline_basis_source_path": basis_metadata.get("source_path"),
        "offline_basis_cache_path": basis_metadata.get("cache_path"),
        "basis_window_count": int(basis_metadata.get("window_count", 0)),
        "basis_singular_values_A": np.asarray(basis.get("singular_values_A", []), float),
        "basis_singular_values_B": np.asarray(basis.get("singular_values_B", []), float),
        "eta_A_log": eta_A_log,
        "eta_B_log": eta_B_log,
        "eta_A_raw_log": eta_A_raw_log,
        "eta_B_raw_log": eta_B_raw_log,
        "action_raw_log": action_raw_log,
        "invalid_id_solve_count": int(invalid_id_solve_count),
        "id_solver_failure_count": int(id_solver_failure_count),
        "id_update_success_count": int(id_update_success_count),
        "n_step": int(getattr(agent, "n_step", 1)),
        "multistep_mode": str(getattr(agent, "multistep_mode", "one_step")),
        "lambda_value": getattr(agent, "lambda_value", None),
        "force_eta_constant": None if force_eta_constant is None else np.asarray(force_eta_constant, float),
        "disable_identification": disable_identification,
        "action_dim": 2,
    }
    for attr in (
        "actor_losses",
        "critic_losses",
        "alpha_losses",
        "alphas",
        "critic_q1_trace",
        "critic_q2_trace",
        "critic_q_gap_trace",
        "exploration_trace",
        "exploration_magnitude_trace",
        "param_noise_scale_trace",
        "action_saturation_trace",
        "entropy_trace",
        "mean_log_prob_trace",
        "reward_n_mean_trace",
        "discount_n_mean_trace",
        "bootstrap_q_mean_trace",
        "n_actual_mean_trace",
        "truncated_fraction_trace",
        "lambda_return_mean_trace",
        "target_logprob_mean_trace",
    ):
        if hasattr(agent, attr):
            result_bundle[attr] = np.asarray(getattr(agent, attr), float)
    attach_single_agent_replay_snapshot(result_bundle, agent)
    return result_bundle


__all__ = ["run_reidentification_supervisor"]
