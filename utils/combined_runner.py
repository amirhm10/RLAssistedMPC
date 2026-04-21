import warnings

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
from utils.multiplier_mapping import map_centered_action_to_bounds, map_centered_bounds_to_action
from utils.observer import compute_observer_gain
from utils.observation_conditioning import update_observer_state
from utils.replay_snapshot import capture_named_agent_replay_snapshots
from utils.residual_authority import compute_residual_rho, project_residual_action
from utils.state_features import (
    build_rl_state,
    compute_tracking_scale_now,
    make_state_conditioner_from_settings,
    resolve_mismatch_settings,
)


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


def _solve_assisted_prediction_step(mpc_obj, y_sp, u_prev_dev, x0_model, initial_guess, bounds, step_idx):
    try:
        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
            np.asarray(initial_guess, float),
            bounds=bounds,
            constraints=[],
        )
    except Exception as exc:
        raise RuntimeError(f"Combined matrix MPC solve failed at step {step_idx}: {exc}") from exc

    success = bool(
        sol is not None
        and bool(getattr(sol, "success", True))
        and getattr(sol, "x", None) is not None
        and np.all(np.isfinite(np.asarray(sol.x, float)))
        and np.isfinite(float(getattr(sol, "fun", 0.0)))
    )
    if not success:
        message = str(getattr(sol, "message", "unknown solver failure"))
        raise RuntimeError(f"Combined matrix MPC solve failed at step {step_idx}: {message}")
    return sol


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
    model_baseline_raw = map_centered_bounds_to_action(
        np.ones(1 + n_inputs, dtype=float),
        model_low,
        model_high,
        nominal=1.0,
    )

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
    residual_mismatch_seed_cfg = dict(residual_cfg)
    residual_mismatch_seed_cfg.setdefault("tracking_eta_tol", residual_cfg.get("authority_eta_tol", 0.3))
    mismatch_cfgs = {
        "horizon": resolve_mismatch_settings(
            state_mode=horizon_state_mode,
            mismatch_cfg=horizon_cfg,
            reward_params=reward_params,
            y_sp_scenario=y_sp_scenario,
            steady_states=steady_states,
            data_min=data_min,
            data_max=data_max,
            n_inputs=n_inputs,
        ),
        "matrix": resolve_mismatch_settings(
            state_mode=matrix_state_mode,
            mismatch_cfg=matrix_cfg,
            reward_params=reward_params,
            y_sp_scenario=y_sp_scenario,
            steady_states=steady_states,
            data_min=data_min,
            data_max=data_max,
            n_inputs=n_inputs,
        ),
        "weights": resolve_mismatch_settings(
            state_mode=weight_state_mode,
            mismatch_cfg=weight_cfg,
            reward_params=reward_params,
            y_sp_scenario=y_sp_scenario,
            steady_states=steady_states,
            data_min=data_min,
            data_max=data_max,
            n_inputs=n_inputs,
        ),
        "residual": resolve_mismatch_settings(
            state_mode=residual_state_mode,
            mismatch_cfg=residual_mismatch_seed_cfg,
            reward_params=reward_params,
            y_sp_scenario=y_sp_scenario,
            steady_states=steady_states,
            data_min=data_min,
            data_max=data_max,
            n_inputs=n_inputs,
        ),
    }
    authority_use_rho = bool(residual_cfg.get("authority_use_rho", residual_cfg.get("use_rho_authority", True)))
    use_shifted_mpc_warm_start = bool(combined_cfg.get("use_shifted_mpc_warm_start", False))
    recalculate_observer_on_matrix_change_requested = bool(
        combined_cfg.get("recalculate_observer_on_matrix_change", False)
    )

    authority_beta_res = np.asarray(
        residual_cfg.get("authority_beta_res", np.full(n_inputs, 0.5, dtype=float)),
        float,
    ).reshape(-1)
    authority_du0_res = np.asarray(
        residual_cfg.get("authority_du0_res", np.full(n_inputs, 0.001, dtype=float)),
        float,
    ).reshape(-1)
    authority_rho_floor = float(residual_cfg.get("authority_rho_floor", 0.15))
    authority_rho_power = float(residual_cfg.get("authority_rho_power", 1.0))
    rho_mapping_mode = str(residual_cfg.get("rho_mapping_mode", "clipped_linear")).strip().lower()
    authority_rho_k = float(residual_cfg.get("authority_rho_k", 0.55))
    residual_zero_deadband_enabled = bool(residual_cfg.get("residual_zero_deadband_enabled", False))
    residual_zero_tracking_raw_threshold = float(residual_cfg.get("residual_zero_tracking_raw_threshold", 0.1))
    residual_zero_innovation_raw_threshold = float(residual_cfg.get("residual_zero_innovation_raw_threshold", 0.1))
    append_rho_to_state = bool(residual_cfg.get("append_rho_to_state", True))
    if authority_beta_res.size != n_inputs or authority_du0_res.size != n_inputs:
        raise ValueError("authority_beta_res and authority_du0_res must match the number of manipulated inputs.")
    state_conditioners = {
        name: make_state_conditioner_from_settings(cfg)
        for name, cfg in mismatch_cfgs.items()
    }
    active_mismatch_observer_modes = {
        mismatch_cfgs[name]["observer_update_alignment"]
        for name, enabled, state_mode in (
            ("horizon", horizon_enabled, horizon_state_mode),
            ("matrix", matrix_enabled, matrix_state_mode),
            ("weights", weight_enabled, weight_state_mode),
            ("residual", residual_enabled, residual_state_mode),
        )
        if enabled and state_mode == "mismatch"
    }
    if len(active_mismatch_observer_modes) > 1:
        raise ValueError("Combined mismatch-enabled agents must share the same observer_update_alignment.")
    observer_update_alignment = (
        next(iter(active_mismatch_observer_modes))
        if active_mismatch_observer_modes
        else "legacy_previous_measurement"
    )

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

    a_res_raw_log = np.zeros((nFE, n_inputs), dtype=float)
    a_res_exec_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_raw_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_exec_log = np.zeros((nFE, n_inputs), dtype=float)
    residual_decision_log = np.zeros(nFE, dtype=int)
    rho_log = np.zeros(nFE, dtype=float) if residual_state_mode == "mismatch" else None
    rho_raw_log = np.zeros(nFE, dtype=float) if residual_state_mode == "mismatch" else None
    rho_eff_log = np.zeros(nFE, dtype=float) if residual_state_mode == "mismatch" else None
    deadband_active_log = np.zeros(nFE, dtype=int)
    projection_active_log = np.zeros(nFE, dtype=int)
    projection_due_to_deadband_log = np.zeros(nFE, dtype=int)
    projection_due_to_authority_log = np.zeros(nFE, dtype=int)
    projection_due_to_headroom_log = np.zeros(nFE, dtype=int)

    mismatch_logs = {
        "horizon": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if horizon_enabled and horizon_state_mode == "mismatch" else None,
            "innovation_raw": np.zeros((nFE, n_outputs), dtype=float) if horizon_enabled and horizon_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if horizon_enabled and horizon_state_mode == "mismatch" else None,
            "tracking_error_raw": np.zeros((nFE, n_outputs), dtype=float) if horizon_enabled and horizon_state_mode == "mismatch" else None,
            "tracking_scale": np.zeros((nFE, n_outputs), dtype=float) if horizon_enabled and horizon_state_mode == "mismatch" else None,
        },
        "matrix": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if matrix_enabled and matrix_state_mode == "mismatch" else None,
            "innovation_raw": np.zeros((nFE, n_outputs), dtype=float) if matrix_enabled and matrix_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if matrix_enabled and matrix_state_mode == "mismatch" else None,
            "tracking_error_raw": np.zeros((nFE, n_outputs), dtype=float) if matrix_enabled and matrix_state_mode == "mismatch" else None,
            "tracking_scale": np.zeros((nFE, n_outputs), dtype=float) if matrix_enabled and matrix_state_mode == "mismatch" else None,
        },
        "weights": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if weight_enabled and weight_state_mode == "mismatch" else None,
            "innovation_raw": np.zeros((nFE, n_outputs), dtype=float) if weight_enabled and weight_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if weight_enabled and weight_state_mode == "mismatch" else None,
            "tracking_error_raw": np.zeros((nFE, n_outputs), dtype=float) if weight_enabled and weight_state_mode == "mismatch" else None,
            "tracking_scale": np.zeros((nFE, n_outputs), dtype=float) if weight_enabled and weight_state_mode == "mismatch" else None,
        },
        "residual": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if residual_enabled and residual_state_mode == "mismatch" else None,
            "innovation_raw": np.zeros((nFE, n_outputs), dtype=float) if residual_enabled and residual_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if residual_enabled and residual_state_mode == "mismatch" else None,
            "tracking_error_raw": np.zeros((nFE, n_outputs), dtype=float) if residual_enabled and residual_state_mode == "mismatch" else None,
            "tracking_scale": np.zeros((nFE, n_outputs), dtype=float) if residual_enabled and residual_state_mode == "mismatch" else None,
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
    A_est = A_base.copy()
    B_est = B_base.copy()
    L_nom = compute_observer_gain(A_est, C_aug, poles)
    current_ic_opt = np.zeros(n_inputs * current_Hc)

    last_horizon_idx = None
    last_model_raw = None
    last_weight_raw = None
    last_residual_raw = None
    test = False
    nonfinite_matrix_action_count = 0
    matrix_A_model_delta_ratio_log = np.zeros(nFE, dtype=float)
    matrix_B_model_delta_ratio_log = np.zeros(nFE, dtype=float)
    b_min = np.asarray(combined_cfg["b_min"], float).reshape(-1)
    b_max = np.asarray(combined_cfg["b_max"], float).reshape(-1)

    current_states = {}
    current_state_debugs = {}

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_pred = C_aug @ xhatdhat[:, i]
        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])

        def build_agent_state(name, state_mode):
            tracking_scale_now = None
            rho_state = None
            mismatch_cfg = mismatch_cfgs[name]
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
                if name == "residual" and mismatch_cfg["append_rho_to_state"]:
                    rho_state = float(
                        compute_residual_rho(
                            tracking_values=(y_prev_scaled - y_sp[i, :]) / np.maximum(tracking_scale_now, 1e-12),
                            rho_mapping_mode=rho_mapping_mode,
                            authority_rho_k=authority_rho_k,
                        )["rho"]
                    )
            state, debug = build_rl_state(
                min_max_dict=min_max_dict,
                x_d_states=xhatdhat[:, i],
                y_sp=y_sp[i, :],
                u=scaled_current_input_dev,
                state_mode=state_mode,
                y_prev_scaled=y_prev_scaled,
                yhat_pred=yhat_pred,
                innovation_scale_ref=mismatch_cfg["innovation_scale_ref"],
                tracking_scale_now=tracking_scale_now,
                mismatch_clip=mismatch_cfg["mismatch_clip"],
                append_rho_to_state=bool(name == "residual" and mismatch_cfg["append_rho_to_state"]),
                rho_value=rho_state,
                state_conditioner=state_conditioners[name],
                update_state_conditioner=True,
                mismatch_feature_transform_mode=mismatch_cfg["mismatch_feature_transform_mode"],
                mismatch_transform_tanh_scale=mismatch_cfg["mismatch_transform_tanh_scale"],
                mismatch_transform_post_clip=mismatch_cfg["mismatch_transform_post_clip"],
            )
            log_pack = mismatch_logs[name]
            if log_pack["innovation"] is not None:
                log_pack["innovation"][i, :] = debug["innovation"]
                log_pack["innovation_raw"][i, :] = debug["innovation_raw"]
                log_pack["tracking_error"][i, :] = debug["tracking_error"]
                log_pack["tracking_error_raw"][i, :] = debug["tracking_error_raw"]
                log_pack["tracking_scale"][i, :] = debug["tracking_scale_now"]
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
        if not np.all(np.isfinite(model_raw)):
            warnings.warn(
                "Combined matrix agent produced a non-finite action; falling back to the last valid or nominal matrix action."
            )
            if last_model_raw is not None and np.all(np.isfinite(last_model_raw)):
                model_raw = last_model_raw.copy()
            else:
                model_raw = model_baseline_raw.copy()
            nonfinite_matrix_action_count += 1
        model_mapped = map_centered_action_to_bounds(
            model_raw,
            model_low,
            model_high,
            nominal=1.0,
        )
        alpha = float(model_mapped[0])
        delta = np.asarray(model_mapped[1 : 1 + n_inputs], float).reshape(-1)
        matrix_alpha_log[i] = alpha
        matrix_delta_log[i, :] = delta

        A_now = A_base.copy()
        B_now = B_base.copy()
        A_now[:n_phys, :n_phys] *= alpha
        B_now[:n_phys, :] *= delta.reshape(1, -1)
        matrix_A_model_delta_ratio_log[i] = np.linalg.norm(
            A_now[:n_phys, :n_phys] - A_base[:n_phys, :n_phys],
            ord="fro",
        ) / max(np.linalg.norm(A_base[:n_phys, :n_phys], ord="fro"), 1e-12)
        matrix_B_model_delta_ratio_log[i] = np.linalg.norm(
            B_now[:n_phys, :] - B_base[:n_phys, :],
            ord="fro",
        ) / max(np.linalg.norm(B_base[:n_phys, :], ord="fro"), 1e-12)

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
                A_base.copy(),
                B_base.copy(),
                C_aug,
                Q_out=Q_now.copy(),
                R_in=R_now.copy(),
                NP=int(Hp),
                NC=int(Hc),
            )
            current_Hp, current_Hc = int(Hp), int(Hc)
            current_ic_opt = np.zeros(n_inputs * current_Hc)
        else:
            MPC_obj.A = A_base.copy()
            MPC_obj.B = B_base.copy()
            MPC_obj.Q_out = Q_now
            MPC_obj.R_in = R_now

        horizon_trace[i, :] = (current_Hp, current_Hc)
        horizon_action_trace[i] = int(h_idx)

        bounds = tuple(
            (float(b_min[j]), float(b_max[j]))
            for _ in range(current_Hc)
            for j in range(b_min.size)
        )

        if not (np.all(np.isfinite(A_now)) and np.all(np.isfinite(B_now))):
            raise RuntimeError(f"Combined matrix prediction model became non-finite at step {i}.")

        ic_opt_step = current_ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * current_Hc)
        MPC_obj.A = A_now
        MPC_obj.B = B_now
        sol = _solve_assisted_prediction_step(
            mpc_obj=MPC_obj,
            y_sp=y_sp[i, :],
            u_prev_dev=scaled_current_input_dev,
            x0_model=xhatdhat[:, i],
            initial_guess=ic_opt_step,
            bounds=bounds,
            step_idx=i,
        )
        if use_shifted_mpc_warm_start:
            current_ic_opt = shift_control_sequence(
                sol.x[: n_inputs * current_Hc],
                n_inputs,
                current_Hc,
            )
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
        a_res_raw_log[i, :] = residual_raw_action

        residual_projection = project_residual_action(
            action_raw=residual_raw_action,
            low_coef=residual_low,
            high_coef=residual_high,
            u_base=u_base,
            scaled_current_input=scaled_current_input,
            u_min_scaled_abs=u_min_scaled_abs,
            u_max_scaled_abs=u_max_scaled_abs,
            apply_authority=(residual_enabled and residual_state_mode == "mismatch"),
            authority_use_rho=authority_use_rho,
            tracking_error_feat=None if not residual_enabled else current_state_debugs["residual"]["tracking_error"],
            tracking_error_raw=None if not residual_enabled else current_state_debugs["residual"]["tracking_error_raw"],
            innovation_raw=None if not residual_enabled else current_state_debugs["residual"]["innovation_raw"],
            authority_beta_res=authority_beta_res,
            authority_du0_res=authority_du0_res,
            authority_rho_floor=authority_rho_floor,
            authority_rho_power=authority_rho_power,
            rho_mapping_mode=rho_mapping_mode,
            authority_rho_k=authority_rho_k,
            residual_zero_deadband_enabled=residual_zero_deadband_enabled,
            residual_zero_tracking_raw_threshold=residual_zero_tracking_raw_threshold,
            residual_zero_innovation_raw_threshold=residual_zero_innovation_raw_threshold,
        )
        if rho_log is not None:
            rho_log[i] = float(residual_projection["rho"])
            rho_raw_log[i] = float(residual_projection["rho_raw"])
            rho_eff_log[i] = float(residual_projection["rho_eff"])
        deadband_active_log[i] = int(residual_projection["deadband_active"])
        projection_active_log[i] = int(residual_projection["projection_active"])
        projection_due_to_deadband_log[i] = int(residual_projection["projection_due_to_deadband"])
        projection_due_to_authority_log[i] = int(residual_projection["projection_due_to_authority"])
        projection_due_to_headroom_log[i] = int(residual_projection["projection_due_to_headroom"])
        delta_u_res_raw_log[i, :] = residual_projection["delta_u_res_raw"]
        delta_u_res_exec_log[i, :] = residual_projection["delta_u_res_exec"]
        a_res_exec_log[i, :] = residual_projection["a_exec"]
        u_applied_scaled[i, :] = residual_projection["u_applied_scaled_abs"]

        delta_u = u_applied_scaled[i, :] - scaled_current_input
        delta_u_storage[i, :] = delta_u

        system.current_input = reverse_min_max(u_applied_scaled[i, :], data_min[:n_inputs], data_max[:n_inputs])
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

        xhatdhat[:, i + 1], yhat[:, i], observer_update_alignment = update_observer_state(
            A=A_est,
            B=B_est,
            C=C_aug,
            L=L_nom,
            x_prev=xhatdhat[:, i],
            u_dev=(u_applied_scaled[i, :] - ss_scaled_inputs),
            y_prev_scaled=y_prev_scaled,
            y_current_scaled=y_current_scaled,
            observer_update_alignment=observer_update_alignment,
        )

        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_applied_scaled[i, :] - ss_scaled_inputs
        yhat_next_pred = C_aug @ xhatdhat[:, i + 1]

        def build_next_state(name, state_mode):
            mismatch_cfg = mismatch_cfgs[name]
            next_tracking_scale_now = None
            next_rho_state = None
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
                if name == "residual" and mismatch_cfg["append_rho_to_state"]:
                    next_rho_state = float(
                        compute_residual_rho(
                            tracking_values=(y_current_scaled - y_sp[i, :]) / np.maximum(next_tracking_scale_now, 1e-12),
                            rho_mapping_mode=rho_mapping_mode,
                            authority_rho_k=authority_rho_k,
                        )["rho"]
                    )
            next_state, _ = build_rl_state(
                min_max_dict=min_max_dict,
                x_d_states=xhatdhat[:, i + 1],
                y_sp=y_sp[i, :],
                u=next_u_dev,
                state_mode=state_mode,
                y_prev_scaled=y_current_scaled,
                yhat_pred=yhat_next_pred,
                innovation_scale_ref=mismatch_cfg["innovation_scale_ref"],
                tracking_scale_now=next_tracking_scale_now,
                mismatch_clip=mismatch_cfg["mismatch_clip"],
                append_rho_to_state=bool(name == "residual" and mismatch_cfg["append_rho_to_state"]),
                rho_value=next_rho_state,
                state_conditioner=state_conditioners[name],
                update_state_conditioner=False,
                mismatch_feature_transform_mode=mismatch_cfg["mismatch_feature_transform_mode"],
                mismatch_transform_tanh_scale=mismatch_cfg["mismatch_transform_tanh_scale"],
                mismatch_transform_post_clip=mismatch_cfg["mismatch_transform_post_clip"],
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
                residual_agent.push(
                    current_states["residual"].astype(np.float32),
                    residual_projection["a_exec"],
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
                delta_u_res_exec_log[i, :],
            )

    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if run_mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )
    for continuous_agent in (matrix_agent, weight_agent, residual_agent):
        if continuous_agent is not None and hasattr(continuous_agent, "flush_nstep"):
            continuous_agent.flush_nstep()

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
        "a_res_raw_log": a_res_raw_log,
        "a_res_exec_log": a_res_exec_log,
        "delta_u_res_raw_log": delta_u_res_raw_log,
        "delta_u_res_exec_log": delta_u_res_exec_log,
        "residual_raw_log": delta_u_res_raw_log,
        "residual_exec_log": delta_u_res_exec_log,
        "residual_decision_log": residual_decision_log,
        "residual_agent_kind": residual_agent_kind,
        "residual_state_mode": residual_state_mode,
        "residual_low_coef": residual_low,
        "residual_high_coef": residual_high,
        "authority_use_rho": authority_use_rho,
        "use_rho_authority": authority_use_rho,
        "append_rho_to_state": append_rho_to_state,
        "authority_beta_res": authority_beta_res,
        "authority_du0_res": authority_du0_res,
        "authority_rho_floor": authority_rho_floor,
        "authority_rho_power": authority_rho_power,
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
        "recalculate_observer_on_matrix_change": recalculate_observer_on_matrix_change_requested,
        "recalculate_observer_on_matrix_change_ignored": True,
        "nonfinite_matrix_action_count": int(nonfinite_matrix_action_count),
        "estimator_mode": "fixed_nominal",
        "matrix_prediction_model_mode": "rl_assisted",
        "matrix_A_model_delta_ratio_log": matrix_A_model_delta_ratio_log,
        "matrix_B_model_delta_ratio_log": matrix_B_model_delta_ratio_log,
        "rho_log": rho_log,
        "rho_raw_log": rho_raw_log,
        "rho_eff_log": rho_eff_log,
        "deadband_active_log": deadband_active_log,
        "projection_active_log": projection_active_log,
        "projection_due_to_deadband_log": projection_due_to_deadband_log,
        "projection_due_to_authority_log": projection_due_to_authority_log,
        "projection_due_to_headroom_log": projection_due_to_headroom_log,
        "horizon_innovation_log": mismatch_logs["horizon"]["innovation"],
        "horizon_innovation_raw_log": mismatch_logs["horizon"]["innovation_raw"],
        "horizon_tracking_error_log": mismatch_logs["horizon"]["tracking_error"],
        "horizon_tracking_error_raw_log": mismatch_logs["horizon"]["tracking_error_raw"],
        "horizon_tracking_scale_log": mismatch_logs["horizon"]["tracking_scale"],
        "horizon_innovation_scale_ref": mismatch_cfgs["horizon"]["innovation_scale_ref"],
        "horizon_band_ref_scaled": mismatch_cfgs["horizon"]["band_ref_scaled"],
        "horizon_mismatch_clip": mismatch_cfgs["horizon"]["mismatch_clip"],
        "horizon_base_state_norm_mode": mismatch_cfgs["horizon"]["base_state_norm_mode"],
        "horizon_base_state_norm_stats": state_conditioners["horizon"].export_state(),
        "horizon_mismatch_feature_transform_mode": mismatch_cfgs["horizon"]["mismatch_feature_transform_mode"],
        "matrix_innovation_log": mismatch_logs["matrix"]["innovation"],
        "matrix_innovation_raw_log": mismatch_logs["matrix"]["innovation_raw"],
        "matrix_tracking_error_log": mismatch_logs["matrix"]["tracking_error"],
        "matrix_tracking_error_raw_log": mismatch_logs["matrix"]["tracking_error_raw"],
        "matrix_tracking_scale_log": mismatch_logs["matrix"]["tracking_scale"],
        "matrix_innovation_scale_ref": mismatch_cfgs["matrix"]["innovation_scale_ref"],
        "matrix_band_ref_scaled": mismatch_cfgs["matrix"]["band_ref_scaled"],
        "matrix_mismatch_clip": mismatch_cfgs["matrix"]["mismatch_clip"],
        "matrix_base_state_norm_mode": mismatch_cfgs["matrix"]["base_state_norm_mode"],
        "matrix_base_state_norm_stats": state_conditioners["matrix"].export_state(),
        "matrix_mismatch_feature_transform_mode": mismatch_cfgs["matrix"]["mismatch_feature_transform_mode"],
        "weight_innovation_log": mismatch_logs["weights"]["innovation"],
        "weight_innovation_raw_log": mismatch_logs["weights"]["innovation_raw"],
        "weight_tracking_error_log": mismatch_logs["weights"]["tracking_error"],
        "weight_tracking_error_raw_log": mismatch_logs["weights"]["tracking_error_raw"],
        "weight_tracking_scale_log": mismatch_logs["weights"]["tracking_scale"],
        "weight_innovation_scale_ref": mismatch_cfgs["weights"]["innovation_scale_ref"],
        "weight_band_ref_scaled": mismatch_cfgs["weights"]["band_ref_scaled"],
        "weight_mismatch_clip": mismatch_cfgs["weights"]["mismatch_clip"],
        "weight_base_state_norm_mode": mismatch_cfgs["weights"]["base_state_norm_mode"],
        "weight_base_state_norm_stats": state_conditioners["weights"].export_state(),
        "weight_mismatch_feature_transform_mode": mismatch_cfgs["weights"]["mismatch_feature_transform_mode"],
        "residual_innovation_log": mismatch_logs["residual"]["innovation"],
        "residual_innovation_raw_log": mismatch_logs["residual"]["innovation_raw"],
        "residual_tracking_error_log": mismatch_logs["residual"]["tracking_error"],
        "residual_tracking_error_raw_log": mismatch_logs["residual"]["tracking_error_raw"],
        "residual_tracking_scale_log": mismatch_logs["residual"]["tracking_scale"],
        "residual_innovation_scale_ref": mismatch_cfgs["residual"]["innovation_scale_ref"],
        "residual_band_ref_scaled": mismatch_cfgs["residual"]["band_ref_scaled"],
        "residual_mismatch_clip": mismatch_cfgs["residual"]["mismatch_clip"],
        "residual_base_state_norm_mode": mismatch_cfgs["residual"]["base_state_norm_mode"],
        "residual_base_state_norm_stats": state_conditioners["residual"].export_state(),
        "residual_mismatch_feature_transform_mode": mismatch_cfgs["residual"]["mismatch_feature_transform_mode"],
        "rho_mapping_mode": rho_mapping_mode,
        "authority_rho_k": authority_rho_k,
        "residual_zero_deadband_enabled": residual_zero_deadband_enabled,
        "residual_zero_tracking_raw_threshold": residual_zero_tracking_raw_threshold,
        "residual_zero_innovation_raw_threshold": residual_zero_innovation_raw_threshold,
        "observer_update_alignment": observer_update_alignment,
        "mpc_horizons": (predict_h, cont_h),
    }

    result_bundle.update(_extract_losses(horizon_agent, "horizon"))
    result_bundle.update(_extract_losses(matrix_agent, "matrix"))
    result_bundle.update(_extract_losses(weight_agent, "weight"))
    result_bundle.update(_extract_losses(residual_agent, "residual"))
    replay_snapshots = capture_named_agent_replay_snapshots(
        {
            "horizon": horizon_agent,
            "matrix": matrix_agent,
            "weights": weight_agent,
            "residual": residual_agent,
        }
    )
    if replay_snapshots:
        result_bundle["replay_buffer_snapshots"] = replay_snapshots

    return result_bundle
