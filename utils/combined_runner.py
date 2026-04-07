import numpy as np
import scipy.optimize as spo

from Simulation.mpc import MpcSolverGeneral
from utils.helpers import (
    action_to_horizons,
    apply_min_max,
    build_polymer_disturbance_schedule,
    disturbance_profile_from_schedule,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
    shift_control_sequence,
    step_system_with_disturbance,
)
from utils.observer import compute_observer_gain
from utils.state_features import build_rl_state, default_mismatch_scale


def _map_to_bounds(action, low, high):
    action = np.asarray(action, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    return low + ((action + 1.0) / 2.0) * (high - low)


def _map_from_bounds(value, low, high):
    value = np.asarray(value, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    return 2.0 * (value - low) / (high - low) - 1.0


def _compute_band_scaled(y_sp_phys, data_min, data_max, n_inputs, k_rel, band_floor_phys):
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    dy = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_sp_phys = np.asarray(y_sp_phys, float)
    k_rel = np.asarray(k_rel, float)
    band_floor_phys = np.asarray(band_floor_phys, float)
    band_phys = np.maximum(k_rel * np.abs(y_sp_phys), band_floor_phys)
    return band_phys / dy


def _normalize_state_mode(cfg, default="standard"):
    return str(cfg.get("state_mode", default)).lower()


def _maybe_get_agent(agents, name, enabled):
    agent = agents.get(name)
    if enabled and agent is None:
        raise ValueError(f"Active agent '{name}' is missing from runtime_ctx['agents'].")
    return agent


def _extract_losses(agent, prefix):
    payload = {}
    if agent is None:
        return payload
    attr_map = {
        "actor_losses": f"{prefix}_actor_losses",
        "critic_losses": f"{prefix}_critic_losses",
        "alpha_losses": f"{prefix}_alpha_losses",
        "alphas": f"{prefix}_alphas",
        "critic_q1_trace": f"{prefix}_critic_q1_trace",
        "critic_q2_trace": f"{prefix}_critic_q2_trace",
        "critic_q_gap_trace": f"{prefix}_critic_q_gap_trace",
        "exploration_trace": f"{prefix}_exploration_trace",
        "exploration_magnitude_trace": f"{prefix}_exploration_magnitude_trace",
        "param_noise_scale_trace": f"{prefix}_param_noise_scale_trace",
        "action_saturation_trace": f"{prefix}_action_saturation_trace",
        "entropy_trace": f"{prefix}_entropy_trace",
        "mean_log_prob_trace": f"{prefix}_mean_log_prob_trace",
        "loss_history": f"{prefix}_dqn_loss_trace",
        "epsilon_trace": f"{prefix}_epsilon_trace",
        "avg_td_error_trace": f"{prefix}_avg_td_error_trace",
        "avg_max_q_trace": f"{prefix}_avg_max_q_trace",
        "avg_value_trace": f"{prefix}_avg_value_trace",
        "avg_advantage_spread_trace": f"{prefix}_avg_advantage_spread_trace",
        "avg_chosen_q_trace": f"{prefix}_avg_chosen_q_trace",
        "noisy_sigma_trace": f"{prefix}_noisy_sigma_trace",
    }
    for attr, key in attr_map.items():
        if hasattr(agent, attr):
            payload[key] = np.asarray(getattr(agent, attr), float)
    return payload


def run_combined_supervisor(combined_cfg, runtime_ctx):
    """
    Run the unified four-agent combined supervisor and return a normalized result bundle.

    Parameters
    ----------
    combined_cfg : dict
        Decision-complete runtime config assembled in the notebook.
    runtime_ctx : dict
        Prepared objects and shared data assembled in the notebook.
    """

    system = runtime_ctx["system"]
    agents = dict(runtime_ctx.get("agents", {}))
    steady_states = runtime_ctx["steady_states"]
    min_max_dict = runtime_ctx["min_max_dict"]
    data_min = np.asarray(runtime_ctx["data_min"], float)
    data_max = np.asarray(runtime_ctx["data_max"], float)
    A_aug = np.asarray(runtime_ctx["A_aug"], float)
    B_aug = np.asarray(runtime_ctx["B_aug"], float)
    C_aug = np.asarray(runtime_ctx["C_aug"], float)
    poles = np.asarray(runtime_ctx["poles"], float)
    y_sp_scenario = np.asarray(runtime_ctx["y_sp_scenario"], float)
    reward_fn = runtime_ctx["reward_fn"]
    reward_params = runtime_ctx.get("reward_params", {})
    system_stepper = runtime_ctx.get("system_stepper")
    system_metadata = runtime_ctx.get("system_metadata")
    disturbance_labels = runtime_ctx.get("disturbance_labels")

    run_mode = str(combined_cfg["run_mode"]).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("combined_cfg['run_mode'] must be 'nominal' or 'disturb'.")

    horizon_cfg = dict(combined_cfg.get("horizon_cfg", {}))
    matrix_cfg = dict(combined_cfg.get("matrix_cfg", {}))
    weight_cfg = dict(combined_cfg.get("weight_cfg", {}))
    residual_cfg = dict(combined_cfg.get("residual_cfg", {}))

    horizon_enabled = bool(horizon_cfg.get("enabled", False))
    matrix_enabled = bool(matrix_cfg.get("enabled", False))
    weight_enabled = bool(weight_cfg.get("enabled", False))
    residual_enabled = bool(residual_cfg.get("enabled", False))
    if not any((horizon_enabled, matrix_enabled, weight_enabled, residual_enabled)):
        raise ValueError("At least one agent must be enabled in the combined supervisor.")

    horizon_agent = _maybe_get_agent(agents, "horizon", horizon_enabled)
    matrix_agent = _maybe_get_agent(agents, "matrix", matrix_enabled)
    weight_agent = _maybe_get_agent(agents, "weights", weight_enabled)
    residual_agent = _maybe_get_agent(agents, "residual", residual_enabled)

    decision_interval = int(combined_cfg["decision_interval"])
    predict_h = int(combined_cfg["predict_h"])
    cont_h = int(combined_cfg["cont_h"])
    q_base = np.array([combined_cfg["Q1_penalty"], combined_cfg["Q2_penalty"]], float)
    r_base = np.array([combined_cfg["R1_penalty"], combined_cfg["R2_penalty"]], float)

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
        int(combined_cfg["n_tests"]),
        int(combined_cfg["set_points_len"]),
        int(combined_cfg["warm_start"]),
        list(combined_cfg["test_cycle"]),
        float(combined_cfg["nominal_qi"]),
        float(combined_cfg["nominal_qs"]),
        float(combined_cfg["nominal_ha"]),
        float(combined_cfg["qi_change"]),
        float(combined_cfg["qs_change"]),
        float(combined_cfg["ha_change"]),
    )

    disturbance_schedule = None
    if run_mode == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug.shape[0])
    n_phys = n_states - n_outputs

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    u_min_scaled_abs = np.asarray(combined_cfg["b_min"], float) + ss_scaled_inputs
    u_max_scaled_abs = np.asarray(combined_cfg["b_max"], float) + ss_scaled_inputs

    model_low = np.asarray(matrix_cfg.get("low_coef", np.ones(1 + n_inputs)), float).reshape(-1)
    model_high = np.asarray(matrix_cfg.get("high_coef", np.ones(1 + n_inputs)), float).reshape(-1)
    if model_low.size != 1 + n_inputs or model_high.size != 1 + n_inputs:
        raise ValueError("matrix_cfg low/high bounds must have length 1 + n_inputs.")
    model_baseline_raw = _map_from_bounds(np.ones(1 + n_inputs, dtype=float), model_low, model_high)

    weights_low = np.asarray(weight_cfg.get("low_coef", np.ones(4)), float).reshape(-1)
    weights_high = np.asarray(weight_cfg.get("high_coef", np.ones(4)), float).reshape(-1)
    if weights_low.size != 4 or weights_high.size != 4:
        raise ValueError("weight_cfg low/high bounds must have length 4.")
    weight_baseline_raw = _map_from_bounds(np.ones(4, dtype=float), weights_low, weights_high)

    residual_low = np.asarray(residual_cfg.get("low_coef", np.zeros(n_inputs)), float).reshape(-1)
    residual_high = np.asarray(residual_cfg.get("high_coef", np.zeros(n_inputs)), float).reshape(-1)
    if residual_low.size != n_inputs or residual_high.size != n_inputs:
        raise ValueError("residual_cfg low/high bounds must match the number of manipulated inputs.")
    if np.any(residual_low > 0.0) or np.any(residual_high < 0.0):
        raise ValueError("Residual bounds must bracket zero so warm start can apply zero correction.")
    residual_baseline_raw = _map_from_bounds(np.zeros(n_inputs, dtype=float), residual_low, residual_high)

    horizon_recipes = list(horizon_cfg.get("horizon_recipes", []))
    if horizon_enabled and not horizon_recipes:
        raise ValueError("horizon_cfg['horizon_recipes'] must be provided when the horizon agent is enabled.")
    default_horizons = tuple(horizon_cfg.get("default_horizons", (predict_h, cont_h)))
    if horizon_enabled:
        if default_horizons not in horizon_recipes:
            raise ValueError("horizon_cfg['default_horizons'] must be present in horizon_recipes.")
        horizon_baseline_idx = int(horizon_recipes.index(default_horizons))
    else:
        horizon_baseline_idx = 0

    matrix_agent_kind = str(matrix_cfg.get("agent_kind", "td3")).lower()
    weight_agent_kind = str(weight_cfg.get("agent_kind", "td3")).lower()
    residual_agent_kind = str(residual_cfg.get("agent_kind", "td3")).lower()
    matrix_state_mode = _normalize_state_mode(matrix_cfg)
    weight_state_mode = _normalize_state_mode(weight_cfg)
    residual_state_mode = _normalize_state_mode(residual_cfg)
    horizon_state_mode = _normalize_state_mode(horizon_cfg)
    mismatch_scales = {
        "horizon": None if horizon_state_mode != "mismatch" else np.asarray(horizon_cfg.get("mismatch_scale", default_mismatch_scale(min_max_dict)), float),
        "matrix": None if matrix_state_mode != "mismatch" else np.asarray(matrix_cfg.get("mismatch_scale", default_mismatch_scale(min_max_dict)), float),
        "weights": None if weight_state_mode != "mismatch" else np.asarray(weight_cfg.get("mismatch_scale", default_mismatch_scale(min_max_dict)), float),
        "residual": None if residual_state_mode != "mismatch" else np.asarray(residual_cfg.get("mismatch_scale", default_mismatch_scale(min_max_dict)), float),
    }
    mismatch_clips = {
        "horizon": horizon_cfg.get("mismatch_clip", 3.0),
        "matrix": matrix_cfg.get("mismatch_clip", 3.0),
        "weights": weight_cfg.get("mismatch_clip", 3.0),
        "residual": residual_cfg.get("mismatch_clip", 3.0),
    }
    use_rho_authority = bool(residual_cfg.get("use_rho_authority", True))
    use_shifted_mpc_warm_start = bool(combined_cfg.get("use_shifted_mpc_warm_start", False))

    k_rel = np.asarray(reward_params.get("k_rel", np.array([0.003, 0.0003])), float)
    band_floor_phys = np.asarray(reward_params.get("band_floor_phys", np.array([0.006, 0.07])), float)
    beta_res = np.array([0.5, 0.5], dtype=np.float32)
    du0_res = np.array([0.001, 0.001], dtype=np.float32)
    eta_tol = 0.3

    y_system = np.zeros((nFE + 1, n_outputs))
    y_system[0, :] = np.asarray(system.current_output, float)
    u_applied_scaled = np.zeros((nFE, n_inputs))
    u_base_scaled = np.zeros((nFE, n_inputs))
    rewards = np.zeros(nFE)
    avg_rewards = []
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    delta_y_storage = np.zeros((nFE, n_outputs))
    delta_u_storage = np.zeros((nFE, n_inputs))

    horizon_trace = np.zeros((nFE, 2), dtype=int)
    horizon_action_trace = np.zeros(nFE, dtype=int)
    horizon_decision_log = np.zeros(nFE, dtype=int)

    matrix_alpha_log = np.ones(nFE, dtype=float)
    matrix_delta_log = np.ones((nFE, n_inputs), dtype=float)
    matrix_decision_log = np.zeros(nFE, dtype=int)

    weight_log = np.ones((nFE, 4), dtype=float)
    weight_decision_log = np.zeros(nFE, dtype=int)

    residual_raw_log = np.zeros((nFE, n_inputs), dtype=float)
    residual_exec_log = np.zeros((nFE, n_inputs), dtype=float)
    residual_decision_log = np.zeros(nFE, dtype=int)
    rho_log = np.zeros(nFE, dtype=float) if residual_state_mode == "mismatch" else None

    mismatch_logs = {
        "horizon": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if horizon_enabled and horizon_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if horizon_enabled and horizon_state_mode == "mismatch" else None,
        },
        "matrix": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if matrix_enabled and matrix_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if matrix_enabled and matrix_state_mode == "mismatch" else None,
        },
        "weights": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if weight_enabled and weight_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if weight_enabled and weight_state_mode == "mismatch" else None,
        },
        "residual": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if residual_enabled and residual_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if residual_enabled and residual_state_mode == "mismatch" else None,
        },
    }

    A_base = np.asarray(A_aug, float).copy()
    B_base = np.asarray(B_aug, float).copy()
    current_Hp, current_Hc = int(default_horizons[0]), int(default_horizons[1])
    MPC_obj = MpcSolverGeneral(
        A_base.copy(),
        B_base.copy(),
        C_aug,
        Q_out=q_base.copy(),
        R_in=r_base.copy(),
        NP=current_Hp,
        NC=current_Hc,
    )
    L = compute_observer_gain(MPC_obj.A, MPC_obj.C, poles)
    current_ic_opt = np.zeros(n_inputs * current_Hc)

    last_horizon_idx = None
    last_model_raw = None
    last_weight_raw = None
    last_residual_raw = None
    test = False

    current_states = {}
    current_state_debugs = {}

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_pred = MPC_obj.C @ xhatdhat[:, i]

        def build_agent_state(name, state_mode):
            state, debug = build_rl_state(
                min_max_dict=min_max_dict,
                x_d_states=xhatdhat[:, i],
                y_sp=y_sp[i, :],
                u=scaled_current_input_dev,
                state_mode=state_mode,
                y_prev_scaled=y_prev_scaled,
                yhat_pred=yhat_pred,
                mismatch_scale=mismatch_scales[name],
                mismatch_clip=mismatch_clips[name],
            )
            log_pack = mismatch_logs[name]
            if log_pack["innovation"] is not None:
                log_pack["innovation"][i, :] = debug["innovation"]
                log_pack["tracking_error"][i, :] = debug["tracking_error"]
            return state, debug

        if horizon_enabled:
            current_states["horizon"], current_state_debugs["horizon"] = build_agent_state("horizon", horizon_state_mode)
        if matrix_enabled:
            current_states["matrix"], current_state_debugs["matrix"] = build_agent_state("matrix", matrix_state_mode)
        if weight_enabled:
            current_states["weights"], current_state_debugs["weights"] = build_agent_state("weights", weight_state_mode)
        if residual_enabled:
            current_states["residual"], current_state_debugs["residual"] = build_agent_state("residual", residual_state_mode)

        if horizon_enabled:
            if i <= warm_start_step:
                h_idx = horizon_baseline_idx
            elif ((i % decision_interval) == 0) or (last_horizon_idx is None):
                horizon_decision_log[i] = 1
                if not test:
                    h_idx = int(horizon_agent.take_action(current_states["horizon"].astype(np.float32), eval_mode=False))
                else:
                    h_idx = int(horizon_agent.act_eval(current_states["horizon"].astype(np.float32)))
                last_horizon_idx = h_idx
            else:
                h_idx = int(last_horizon_idx)
            Hp, Hc = action_to_horizons(horizon_recipes, h_idx)
        else:
            h_idx = horizon_baseline_idx
            Hp, Hc = current_Hp, current_Hc

        if matrix_enabled:
            if i <= warm_start_step:
                model_raw = model_baseline_raw.copy()
            elif ((i % decision_interval) == 0) or (last_model_raw is None):
                matrix_decision_log[i] = 1
                if not test:
                    model_raw = np.asarray(matrix_agent.take_action(current_states["matrix"], explore=True), float)
                else:
                    model_raw = np.asarray(matrix_agent.act_eval(current_states["matrix"]), float)
                last_model_raw = model_raw.copy()
            else:
                model_raw = last_model_raw.copy()
        else:
            model_raw = model_baseline_raw.copy()
        model_raw = np.asarray(model_raw, float).reshape(-1)
        model_mapped = _map_to_bounds(model_raw, model_low, model_high)
        alpha = float(model_mapped[0])
        delta = np.asarray(model_mapped[1 : 1 + n_inputs], float).reshape(-1)
        matrix_alpha_log[i] = alpha
        matrix_delta_log[i, :] = delta

        A_now = A_base.copy()
        B_now = B_base.copy()
        A_now[:n_phys, :n_phys] *= alpha
        B_now[:n_phys, :] *= delta.reshape(1, -1)

        if weight_enabled:
            if i <= warm_start_step:
                weight_raw = weight_baseline_raw.copy()
            elif ((i % decision_interval) == 0) or (last_weight_raw is None):
                weight_decision_log[i] = 1
                if not test:
                    weight_raw = np.asarray(weight_agent.take_action(current_states["weights"], explore=True), float)
                else:
                    weight_raw = np.asarray(weight_agent.act_eval(current_states["weights"]), float)
                last_weight_raw = weight_raw.copy()
            else:
                weight_raw = last_weight_raw.copy()
        else:
            weight_raw = weight_baseline_raw.copy()
        weight_raw = np.asarray(weight_raw, float).reshape(-1)
        if weight_raw.size != 4:
            raise ValueError("Weights action must contain 4 elements for [Q1, Q2, R1, R2].")
        weight_mult = _map_to_bounds(weight_raw, weights_low, weights_high)
        weight_log[i, :] = weight_mult
        Q_now = np.array([q_base[0] * weight_mult[0], q_base[1] * weight_mult[1]], dtype=float)
        R_now = np.array([r_base[0] * weight_mult[2], r_base[1] * weight_mult[3]], dtype=float)

        if (int(Hp), int(Hc)) != (current_Hp, current_Hc):
            MPC_obj = MpcSolverGeneral(
                A_now.copy(),
                B_now.copy(),
                C_aug,
                Q_out=Q_now.copy(),
                R_in=R_now.copy(),
                NP=int(Hp),
                NC=int(Hc),
            )
            current_Hp, current_Hc = int(Hp), int(Hc)
            current_ic_opt = np.zeros(n_inputs * current_Hc)
        else:
            MPC_obj.A = A_now
            MPC_obj.B = B_now
            MPC_obj.Q_out = Q_now
            MPC_obj.R_in = R_now

        horizon_trace[i, :] = (current_Hp, current_Hc)
        horizon_action_trace[i] = int(h_idx)

        bounds = tuple(
            (float(combined_cfg["b_min"][j]), float(combined_cfg["b_max"][j]))
            for _ in range(current_Hc)
            for j in range(n_inputs)
        )

        ic_opt_step = current_ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * current_Hc)

        sol = spo.minimize(
            lambda x: MPC_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            ic_opt_step,
            bounds=bounds,
            constraints=[],
        )
        if use_shifted_mpc_warm_start:
            current_ic_opt = shift_control_sequence(sol.x[: n_inputs * current_Hc], n_inputs, current_Hc)
        else:
            current_ic_opt = np.zeros(n_inputs * current_Hc)

        u_base = np.asarray(sol.x[:n_inputs], float) + ss_scaled_inputs
        u_base = np.clip(u_base, u_min_scaled_abs, u_max_scaled_abs)
        u_base_scaled[i, :] = u_base

        if residual_enabled:
            if i <= warm_start_step:
                residual_raw_action = residual_baseline_raw.copy()
            elif ((i % decision_interval) == 0) or (last_residual_raw is None):
                residual_decision_log[i] = 1
                if not test:
                    residual_raw_action = np.asarray(
                        residual_agent.take_action(current_states["residual"], explore=True),
                        float,
                    )
                else:
                    residual_raw_action = np.asarray(residual_agent.act_eval(current_states["residual"]), float)
                last_residual_raw = residual_raw_action.copy()
            else:
                residual_raw_action = last_residual_raw.copy()
        else:
            residual_raw_action = residual_baseline_raw.copy()
        residual_raw_action = np.asarray(residual_raw_action, float).reshape(-1)
        residual_raw = _map_to_bounds(residual_raw_action, residual_low, residual_high).reshape(-1)
        residual_raw_log[i, :] = residual_raw

        low_headroom = (u_min_scaled_abs - u_base).astype(np.float32)
        high_headroom = (u_max_scaled_abs - u_base).astype(np.float32)
        if residual_enabled and residual_state_mode == "mismatch":
            y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
            band_scaled = _compute_band_scaled(
                y_sp_phys=y_sp_phys,
                data_min=data_min,
                data_max=data_max,
                n_inputs=n_inputs,
                k_rel=k_rel,
                band_floor_phys=band_floor_phys,
            ).astype(np.float32)
            e_track = current_state_debugs["residual"]["tracking_error"]
            delta_u_mpc = (u_base - scaled_current_input).astype(np.float32)
            eps_i = (eta_tol * band_scaled).astype(np.float32)
            rho = float(np.clip(np.max(np.abs(e_track) / np.maximum(eps_i, 1e-12)), 0.0, 1.0))
            if rho_log is not None:
                rho_log[i] = rho
            authority_scale = rho if use_rho_authority else 1.0
            mag = (authority_scale * beta_res) * (np.abs(delta_u_mpc) + du0_res)
            low_bound = np.maximum(-mag, low_headroom)
            high_bound = np.minimum(mag, high_headroom)
            bad = low_bound > high_bound
            if np.any(bad):
                low_bound[bad] = 0.0
                high_bound[bad] = 0.0
            residual_exec = np.clip(residual_raw, low_bound, high_bound)
        else:
            residual_exec = np.clip(
                residual_raw,
                np.maximum(residual_low, low_headroom),
                np.minimum(residual_high, high_headroom),
            )

        u_applied_scaled_abs = np.clip(u_base + residual_exec, u_min_scaled_abs, u_max_scaled_abs)
        residual_exec = u_applied_scaled_abs - u_base
        residual_exec_log[i, :] = residual_exec
        u_applied_scaled[i, :] = u_applied_scaled_abs

        delta_u = u_applied_scaled_abs - scaled_current_input
        delta_u_storage[i, :] = delta_u

        system.current_input = reverse_min_max(u_applied_scaled_abs, data_min[:n_inputs], data_max[:n_inputs])
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
        xhatdhat[:, i + 1] = (
            MPC_obj.A @ xhatdhat[:, i]
            + MPC_obj.B @ (u_applied_scaled_abs - ss_scaled_inputs)
            + L @ (y_prev_scaled - yhat[:, i]).T
        )

        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_applied_scaled_abs - ss_scaled_inputs
        yhat_next_pred = MPC_obj.C @ xhatdhat[:, i + 1]

        def build_next_state(name, state_mode):
            next_state, _ = build_rl_state(
                min_max_dict=min_max_dict,
                x_d_states=xhatdhat[:, i + 1],
                y_sp=y_sp[i, :],
                u=next_u_dev,
                state_mode=state_mode,
                y_prev_scaled=y_current_scaled,
                yhat_pred=yhat_next_pred,
                mismatch_scale=mismatch_scales[name],
                mismatch_clip=mismatch_clips[name],
            )
            return next_state

        if not test:
            if horizon_enabled and i > warm_start_step:
                next_state = build_next_state("horizon", horizon_state_mode)
                horizon_agent.push(
                    current_states["horizon"].astype(np.float32),
                    int(h_idx),
                    reward,
                    next_state.astype(np.float32),
                    0.0,
                )
                if i >= warm_start_step:
                    horizon_agent.train_step()

            if matrix_enabled and i > warm_start_step:
                next_state = build_next_state("matrix", matrix_state_mode)
                matrix_agent.push(
                    current_states["matrix"].astype(np.float32),
                    np.asarray(model_raw, np.float32),
                    reward,
                    next_state.astype(np.float32),
                    0.0,
                )
                if i >= warm_start_step:
                    matrix_agent.train_step()

            if weight_enabled and i > warm_start_step:
                next_state = build_next_state("weights", weight_state_mode)
                weight_agent.push(
                    current_states["weights"].astype(np.float32),
                    np.asarray(weight_raw, np.float32),
                    reward,
                    next_state.astype(np.float32),
                    0.0,
                )
                if i >= warm_start_step:
                    weight_agent.train_step()

            if residual_enabled and i > warm_start_step:
                next_state = build_next_state("residual", residual_state_mode)
                residual_exec_action = _map_from_bounds(residual_exec, residual_low, residual_high).astype(np.float32)
                residual_exec_action = np.clip(residual_exec_action, -1.0, 1.0)
                residual_agent.push(
                    current_states["residual"].astype(np.float32),
                    residual_exec_action,
                    reward,
                    next_state.astype(np.float32),
                    0.0,
                )
                if i >= warm_start_step:
                    residual_agent.train_step()

        if i in sub_episodes_changes_dict:
            avg_rewards.append(float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1])))
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_rewards[-1],
                "| Hp,Hc:",
                tuple(horizon_trace[i, :]),
                "| alpha:",
                matrix_alpha_log[i],
                "| weights:",
                weight_log[i, :],
                "| residual:",
                residual_exec_log[i, :],
            )

    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if run_mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )

    result_bundle = {
        "run_mode": run_mode,
        "method_family": "combined",
        "algorithm": "multi_agent",
        "system_metadata": system_metadata,
        "notebook_source": combined_cfg.get("notebook_source"),
        "config_snapshot": dict(combined_cfg),
        "seed": combined_cfg.get("seed"),
        "decision_interval": int(decision_interval),
        "active_agents": {
            "horizon": horizon_enabled,
            "matrix": matrix_enabled,
            "weights": weight_enabled,
            "residual": residual_enabled,
        },
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": int(nFE),
        "delta_t": float(system.delta_t),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y": y_system,
        "u": reverse_min_max(u_applied_scaled, data_min[:n_inputs], data_max[:n_inputs]),
        "u_base": reverse_min_max(u_base_scaled, data_min[:n_inputs], data_max[:n_inputs]),
        "avg_rewards": np.asarray(avg_rewards, float),
        "rewards_step": rewards,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "data_min": data_min,
        "data_max": data_max,
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "warm_start_step": int(warm_start_step),
        "yhat": yhat,
        "xhatdhat": xhatdhat,
        "horizon_trace": horizon_trace,
        "horizon_action_trace": horizon_action_trace,
        "horizon_decision_log": horizon_decision_log,
        "horizon_state_mode": horizon_state_mode,
        "horizon_agent_kind": "dqn",
        "horizon_recipes": horizon_recipes if horizon_enabled else None,
        "matrix_alpha_log": matrix_alpha_log,
        "matrix_delta_log": matrix_delta_log,
        "matrix_decision_log": matrix_decision_log,
        "matrix_agent_kind": matrix_agent_kind,
        "matrix_state_mode": matrix_state_mode,
        "matrix_low_coef": model_low,
        "matrix_high_coef": model_high,
        "weight_log": weight_log,
        "weight_decision_log": weight_decision_log,
        "weight_agent_kind": weight_agent_kind,
        "weight_state_mode": weight_state_mode,
        "weight_low_coef": weights_low,
        "weight_high_coef": weights_high,
        "residual_raw_log": residual_raw_log,
        "residual_exec_log": residual_exec_log,
        "residual_decision_log": residual_decision_log,
        "residual_agent_kind": residual_agent_kind,
        "residual_state_mode": residual_state_mode,
        "residual_low_coef": residual_low,
        "residual_high_coef": residual_high,
        "use_rho_authority": use_rho_authority,
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
        "rho_log": rho_log,
        "horizon_innovation_log": mismatch_logs["horizon"]["innovation"],
        "horizon_tracking_error_log": mismatch_logs["horizon"]["tracking_error"],
        "matrix_innovation_log": mismatch_logs["matrix"]["innovation"],
        "matrix_tracking_error_log": mismatch_logs["matrix"]["tracking_error"],
        "weight_innovation_log": mismatch_logs["weights"]["innovation"],
        "weight_tracking_error_log": mismatch_logs["weights"]["tracking_error"],
        "residual_innovation_log": mismatch_logs["residual"]["innovation"],
        "residual_tracking_error_log": mismatch_logs["residual"]["tracking_error"],
        "horizon_mismatch_scale": mismatch_scales["horizon"],
        "matrix_mismatch_scale": mismatch_scales["matrix"],
        "weight_mismatch_scale": mismatch_scales["weights"],
        "residual_mismatch_scale": mismatch_scales["residual"],
        "mpc_horizons": (predict_h, cont_h),
    }

    result_bundle.update(_extract_losses(horizon_agent, "horizon"))
    result_bundle.update(_extract_losses(matrix_agent, "matrix"))
    result_bundle.update(_extract_losses(weight_agent, "weight"))
    result_bundle.update(_extract_losses(residual_agent, "residual"))

    return result_bundle
