import warnings

import numpy as np
import scipy.optimize as spo

from Simulation.mpc import MpcSolverGeneral
from TD3Agent.supervisor_replay_buffer import (
    SOURCE_FALLBACK as SG_SOURCE_FALLBACK,
    SOURCE_HELD as SG_SOURCE_HELD,
    SOURCE_POLICY as SG_SOURCE_POLICY,
    SOURCE_SUPERVISOR as SG_SOURCE_SUPERVISOR,
    SOURCE_WARM_START as SG_SOURCE_WARM_START,
)
from utils.agent_step_runtime import (
    replay_train_continuous_agent,
    replay_train_horizon_agent,
    replay_train_supervisor_gated_continuous_agent,
    replay_train_supervisor_gated_horizon_agent,
    select_continuous_action,
    select_horizon_action,
    select_supervisor_gated_continuous_action,
    select_supervisor_gated_horizon_action,
)
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
from utils.markov_runner import (
    TD3_PRIORITY_PHASE_CODE,
    _td3_priority_authority_scale,
    _td3_priority_candidate_allowed,
    _td3_priority_enabled,
    _td3_priority_phase,
    apply_markov_correction,
    apply_z_safety_projection,
    build_toeplitz_from_markov,
    compute_markov_blocks,
    fit_markov_ls_correction,
    free_response,
    gain_drift,
    lifted_mpc_cost,
    make_markov_basis,
    prediction_improvement_score,
    raw_action_to_z,
    resolve_z_safety_effective_cap,
    solve_lifted_mpc,
    z_to_raw_action,
)
from utils.multiplier_mapping import map_centered_action_to_bounds, map_centered_bounds_to_action
from utils.multiplier_release_schedule import (
    build_release_authority_schedule,
    clip_multipliers_to_release_bounds,
    map_effective_multipliers_to_raw_action,
)
from utils.observer import compute_observer_gain
from utils.observation_conditioning import update_observer_state
from utils.phase1_hidden_release import (
    build_phase1_bundle_fields,
    build_phase1_schedule,
    init_phase1_train_traces,
)
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


def _projection_is_finite(projection):
    for key in ("a_exec", "delta_u_res_exec", "u_applied_scaled_abs"):
        value = projection.get(key)
        if value is None or not np.all(np.isfinite(value)):
            return False
    return True


def _sg_source_summary(source_window):
    sources = np.asarray(source_window, int).reshape(-1)
    if sources.size == 0:
        return "none"
    return (
        f"policy={int(np.sum(sources == SG_SOURCE_POLICY))},"
        f"supervisor={int(np.sum(sources == SG_SOURCE_SUPERVISOR))},"
        f"warm={int(np.sum(sources == SG_SOURCE_WARM_START))},"
        f"held={int(np.sum(sources == SG_SOURCE_HELD))},"
        f"fallback={int(np.sum(sources == SG_SOURCE_FALLBACK))}"
    )


def _normalize_state_mode(cfg, default="standard"):
    return str(cfg.get("state_mode", default)).lower()


def _is_sg_td3_agent_kind(agent_kind):
    return str(agent_kind).strip().lower() == "sg_td3"


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


def _default_markov_score(n_outputs):
    return {
        "score": 0.0,
        "nominal_sse": np.nan,
        "corrected_sse": np.nan,
        "n_windows": 0,
        "output_nominal_sse": np.full(int(n_outputs), np.nan, dtype=float),
        "output_corrected_sse": np.full(int(n_outputs), np.nan, dtype=float),
    }


def _record_markov_safety_stage(logs, step, prefix, safety_info):
    logs[f"markov_z_safety_{prefix}_norm_before_log"][step] = float(safety_info["norm_before"])
    logs[f"markov_z_safety_{prefix}_norm_after_log"][step] = float(safety_info["norm_after"])
    logs[f"markov_z_safety_{prefix}_projection_scale_log"][step] = float(safety_info["projection_scale"])
    logs[f"markov_z_safety_{prefix}_projection_active_log"][step] = int(bool(safety_info["projection_active"]))
    logs[f"markov_z_safety_{prefix}_coord_clip_active_log"][step] = int(bool(safety_info["coord_clip_active"]))
    logs[f"markov_z_safety_{prefix}_vector_projection_active_log"][step] = int(
        bool(safety_info["vector_projection_active"])
    )


def _combined_markov_state(base_state, z_prev, z_ls, ls_score, ls_gain_drift, horizons, horizon_scale, weight_mult):
    hp, hc = horizons
    hp_scale, hc_scale = horizon_scale
    horizon_features = np.asarray(
        [
            float(hp) / max(float(hp_scale), 1.0),
            float(hc) / max(float(hc_scale), 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            np.asarray(base_state, np.float32).reshape(-1),
            np.asarray(z_prev, np.float32).reshape(-1),
            np.asarray(z_ls, np.float32).reshape(-1),
            np.asarray([float(ls_score), float(ls_gain_drift)], np.float32),
            horizon_features,
            np.asarray(weight_mult, np.float32).reshape(-1),
        ]
    ).astype(np.float32, copy=False)


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
    markov_cfg = dict(combined_cfg.get("markov_cfg", {}))
    matrix_cfg = dict(combined_cfg.get("matrix_cfg", {}))
    weight_cfg = dict(combined_cfg.get("weight_cfg", {}))
    residual_cfg = dict(combined_cfg.get("residual_cfg", {}))

    horizon_enabled = bool(horizon_cfg.get("enabled", False))
    markov_enabled = bool(markov_cfg.get("enabled", False))
    matrix_enabled = bool(matrix_cfg.get("enabled", False))
    weight_enabled = bool(weight_cfg.get("enabled", False))
    residual_enabled = bool(residual_cfg.get("enabled", False))
    weight_safety_cfg = dict(weight_cfg.get("weight_safety", {}) or {})
    weight_safety_enabled = bool(weight_safety_cfg.get("enabled", False))
    fallback_to_identity_on_nonfinite = bool(
        weight_safety_enabled and weight_safety_cfg.get("fallback_to_identity_on_nonfinite", False)
    )
    residual_safety_cfg = dict(residual_cfg.get("residual_safety", {}) or {})
    residual_safety_enabled = bool(residual_safety_cfg.get("enabled", False))
    fallback_to_zero_on_nonfinite = bool(
        residual_safety_enabled and residual_safety_cfg.get("fallback_to_zero_on_nonfinite", False)
    )
    if markov_enabled and matrix_enabled:
        raise ValueError("Enable either the Markov model supervisor or the matrix supervisor, not both.")
    if not any((horizon_enabled, markov_enabled, matrix_enabled, weight_enabled, residual_enabled)):
        raise ValueError("At least one agent must be enabled in the combined supervisor.")

    horizon_agent = _maybe_get_agent(agents, "horizon", horizon_enabled)
    markov_agent = _maybe_get_agent(agents, "markov", markov_enabled)
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

    markov_basis_labels = []
    markov_z_dim = 0
    markov_z_bound = float(markov_cfg.get("z_bound", 0.0))
    markov_baseline_raw = np.zeros(0, dtype=float)
    if markov_enabled:
        markov_preview_blocks = compute_markov_blocks(A_aug, B_aug, C_aug, predict_h)
        markov_preview_basis, markov_basis_labels = make_markov_basis(
            markov_preview_blocks,
            markov_cfg.get("basis_family", "io_pair_gain"),
        )
        markov_z_dim = int(markov_preview_basis.shape[0])
        markov_baseline_raw = np.zeros(markov_z_dim, dtype=float)
        if markov_z_bound <= 0.0:
            raise ValueError("markov_cfg['z_bound'] must be positive when the Markov agent is enabled.")

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

    horizon_agent_kind = str(
        horizon_cfg.get("agent_kind", combined_cfg.get("horizon_agent_kind", "dqn"))
    ).lower()
    if horizon_agent_kind not in {"dqn", "sg_dqn"}:
        raise ValueError("horizon agent_kind must be 'dqn' or 'sg_dqn'.")
    markov_agent_kind = str(markov_cfg.get("agent_kind", "td3")).lower()
    if markov_enabled and markov_agent_kind not in {"td3", "sg_td3"}:
        raise ValueError("markov agent_kind must be 'td3' or 'sg_td3' in the combined Markov supervisor.")
    matrix_agent_kind = str(matrix_cfg.get("agent_kind", "td3")).lower()
    weight_agent_kind = str(weight_cfg.get("agent_kind", "td3")).lower()
    residual_agent_kind = str(residual_cfg.get("agent_kind", "td3")).lower()
    if weight_enabled and weight_agent_kind not in {"td3", "sg_td3"}:
        raise ValueError("weight agent_kind must be 'td3' or 'sg_td3'.")
    if residual_enabled and residual_agent_kind not in {"td3", "sg_td3"}:
        raise ValueError("residual agent_kind must be 'td3' or 'sg_td3'.")
    horizon_sg_enabled = horizon_agent_kind == "sg_dqn"
    markov_sg_enabled = _is_sg_td3_agent_kind(markov_agent_kind)
    weight_sg_enabled = _is_sg_td3_agent_kind(weight_agent_kind)
    residual_sg_enabled = _is_sg_td3_agent_kind(residual_agent_kind)
    markov_state_mode = _normalize_state_mode(markov_cfg, "mismatch")
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
        "markov": resolve_mismatch_settings(
            state_mode=markov_state_mode,
            mismatch_cfg=markov_cfg,
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
    residual_authority_enabled = bool(residual_cfg.get("residual_authority_enabled", False))
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
            ("markov", markov_enabled, markov_state_mode),
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

    td3_phase1_action_freeze_subepisodes = int(combined_cfg.get("td3_post_warm_start_action_freeze_subepisodes", 0))
    td3_phase1_actor_freeze_subepisodes = int(combined_cfg.get("td3_post_warm_start_actor_freeze_subepisodes", 0))
    horizon_action_freeze_steps = int(
        max(0, int(horizon_cfg.get("post_warm_start_action_freeze_subepisodes", 0)))
        * int(time_in_sub_episodes)
    )

    markov_phase1 = None
    if markov_enabled and markov_agent is not None and markov_agent_kind in {"td3", "sg_td3"}:
        markov_phase1 = build_phase1_schedule(
            agent_kind=markov_agent_kind,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            n_steps=nFE,
            test_train_dict=test_train_dict,
            action_freeze_subepisodes=td3_phase1_action_freeze_subepisodes,
            actor_freeze_subepisodes=td3_phase1_actor_freeze_subepisodes,
            batch_size=getattr(markov_agent, "batch_size", 1),
            initial_buffer_size=len(getattr(markov_agent, "buffer", [])),
            base_actor_freeze=getattr(markov_agent, "actor_freeze", 0),
            push_start_step=0,
            train_start_step=warm_start_step,
        )
        markov_agent.actor_freeze = int(markov_phase1["effective_actor_freeze"])

    matrix_phase1 = None
    if matrix_enabled and matrix_agent is not None and matrix_agent_kind == "td3":
        matrix_phase1 = build_phase1_schedule(
            agent_kind=matrix_agent_kind,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            n_steps=nFE,
            test_train_dict=test_train_dict,
            action_freeze_subepisodes=td3_phase1_action_freeze_subepisodes,
            actor_freeze_subepisodes=td3_phase1_actor_freeze_subepisodes,
            batch_size=getattr(matrix_agent, "batch_size", 1),
            initial_buffer_size=len(getattr(matrix_agent, "buffer", [])),
            base_actor_freeze=getattr(matrix_agent, "actor_freeze", 0),
            push_start_step=0,
            train_start_step=warm_start_step,
        )
        matrix_agent.actor_freeze = int(matrix_phase1["effective_actor_freeze"])

    weight_phase1 = None
    if weight_enabled and weight_agent is not None and weight_agent_kind in {"td3", "sg_td3"}:
        weight_phase1 = build_phase1_schedule(
            agent_kind=weight_agent_kind,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            n_steps=nFE,
            test_train_dict=test_train_dict,
            action_freeze_subepisodes=td3_phase1_action_freeze_subepisodes,
            actor_freeze_subepisodes=td3_phase1_actor_freeze_subepisodes,
            batch_size=getattr(weight_agent, "batch_size", 1),
            initial_buffer_size=len(getattr(weight_agent, "buffer", [])),
            base_actor_freeze=getattr(weight_agent, "actor_freeze", 0),
            push_start_step=0,
            train_start_step=warm_start_step,
        )
        weight_agent.actor_freeze = int(weight_phase1["effective_actor_freeze"])

    residual_phase1 = None
    if residual_enabled and residual_agent is not None and residual_agent_kind in {"td3", "sg_td3"}:
        residual_phase1 = build_phase1_schedule(
            agent_kind=residual_agent_kind,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            n_steps=nFE,
            test_train_dict=test_train_dict,
            action_freeze_subepisodes=td3_phase1_action_freeze_subepisodes,
            actor_freeze_subepisodes=td3_phase1_actor_freeze_subepisodes,
            batch_size=getattr(residual_agent, "batch_size", 1),
            initial_buffer_size=len(getattr(residual_agent, "buffer", [])),
            base_actor_freeze=getattr(residual_agent, "actor_freeze", 0),
            push_start_step=0,
            train_start_step=warm_start_step,
        )
        residual_agent.actor_freeze = int(residual_phase1["effective_actor_freeze"])

    release_cfg = (
        dict(matrix_cfg.get("release_protected_advisory_caps", {}))
        if matrix_enabled
        else {"enabled": False}
    )
    release_labels = ["alpha"] + [f"B_col_{idx + 1}" for idx in range(n_inputs)]
    release_action_freeze_end_step = matrix_phase1["action_freeze_end_step"] if matrix_phase1 is not None else warm_start_step
    matrix_release_schedule = build_release_authority_schedule(
        config=release_cfg,
        labels=release_labels,
        wide_low=model_low,
        wide_high=model_high,
        warm_start_step=warm_start_step,
        action_freeze_end_step=release_action_freeze_end_step,
        time_in_sub_episodes=time_in_sub_episodes,
        n_steps=nFE,
    )
    matrix_release_store_executed_action = bool(
        matrix_release_schedule.get("store_executed_action_in_replay", True)
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
    horizon_sg_policy_action_log = np.full(nFE, -1, dtype=int) if horizon_sg_enabled else None
    horizon_sg_supervisor_action_log = np.full(nFE, -1, dtype=int) if horizon_sg_enabled else None
    horizon_sg_previous_action_log = np.full(nFE, -1, dtype=int) if horizon_sg_enabled else None
    horizon_sg_selected_source_log = np.zeros(nFE, dtype=int) if horizon_sg_enabled else None
    horizon_sg_score_policy_log = np.full(nFE, np.nan, dtype=float) if horizon_sg_enabled else None
    horizon_sg_score_supervisor_log = np.full(nFE, np.nan, dtype=float) if horizon_sg_enabled else None
    horizon_sg_advantage_log = np.full(nFE, np.nan, dtype=float) if horizon_sg_enabled else None
    horizon_sg_q_policy_log = np.full(nFE, np.nan, dtype=float) if horizon_sg_enabled else None
    horizon_sg_q_supervisor_log = np.full(nFE, np.nan, dtype=float) if horizon_sg_enabled else None

    markov_z_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_z_proposed_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_z_executed_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_requested_z_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_requested_z_uncapped_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_ls_z_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_ls_z_uncapped_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_actor_raw_action_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_requested_raw_action_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_executed_raw_action_log = np.zeros((nFE, markov_z_dim), dtype=float)
    markov_decision_log = np.zeros(nFE, dtype=int)
    markov_action_source_log = np.zeros(nFE, dtype=int)
    markov_policy_source_log = np.zeros(nFE, dtype=int)
    markov_replay_pushed_log = np.zeros(nFE, dtype=int)
    markov_train_called_log = np.zeros(nFE, dtype=int)
    markov_train_updated_log = np.zeros(nFE, dtype=int)
    markov_prediction_score_log = np.full(nFE, np.nan, dtype=float)
    markov_ls_prediction_score_log = np.full(nFE, np.nan, dtype=float)
    markov_requested_prediction_score_log = np.full(nFE, np.nan, dtype=float)
    markov_gain_drift_log = np.zeros(nFE, dtype=float)
    markov_ls_gain_drift_log = np.zeros(nFE, dtype=float)
    markov_requested_gain_drift_log = np.zeros(nFE, dtype=float)
    markov_accepted_log = np.zeros(nFE, dtype=int)
    markov_fallback_log = np.zeros(nFE, dtype=int)
    markov_nominal_cost_log = np.full(nFE, np.nan, dtype=float)
    markov_executed_cost_margin_log = np.full(nFE, np.nan, dtype=float)
    markov_td3_priority_phase_log = np.zeros(nFE, dtype=int)
    markov_td3_authority_scale_log = np.ones(nFE, dtype=float)
    markov_td3_probation_active_log = np.zeros(nFE, dtype=int)
    markov_td3_probation_trigger_log = np.zeros(nFE, dtype=int)
    markov_z_safety_logs = {
        "markov_z_safety_effective_cap_log": np.full(nFE, np.nan, dtype=float),
        "markov_z_safety_requested_norm_before_log": np.full(nFE, np.nan, dtype=float),
        "markov_z_safety_requested_norm_after_log": np.full(nFE, np.nan, dtype=float),
        "markov_z_safety_requested_projection_scale_log": np.full(nFE, np.nan, dtype=float),
        "markov_z_safety_requested_projection_active_log": np.zeros(nFE, dtype=int),
        "markov_z_safety_requested_coord_clip_active_log": np.zeros(nFE, dtype=int),
        "markov_z_safety_requested_vector_projection_active_log": np.zeros(nFE, dtype=int),
        "markov_z_safety_ls_norm_before_log": np.full(nFE, np.nan, dtype=float),
        "markov_z_safety_ls_norm_after_log": np.full(nFE, np.nan, dtype=float),
        "markov_z_safety_ls_projection_scale_log": np.full(nFE, np.nan, dtype=float),
        "markov_z_safety_ls_projection_active_log": np.zeros(nFE, dtype=int),
        "markov_z_safety_ls_coord_clip_active_log": np.zeros(nFE, dtype=int),
        "markov_z_safety_ls_vector_projection_active_log": np.zeros(nFE, dtype=int),
    }
    markov_phase1_action_source_log = np.zeros(nFE, dtype=int) if markov_phase1 is not None else None
    markov_policy_action_raw_log = np.zeros((nFE, markov_z_dim), dtype=float) if markov_phase1 is not None else None
    markov_phase1_executed_action_raw_log = (
        np.zeros((nFE, markov_z_dim), dtype=float) if markov_phase1 is not None else None
    )
    markov_phase1_train_traces = init_phase1_train_traces() if markov_phase1 is not None else None
    markov_sg_policy_action_raw_log = (
        np.zeros((nFE, markov_z_dim), dtype=float) if markov_sg_enabled else None
    )
    markov_sg_supervisor_action_raw_log = (
        np.zeros((nFE, markov_z_dim), dtype=float) if markov_sg_enabled else None
    )
    markov_sg_previous_action_raw_log = (
        np.zeros((nFE, markov_z_dim), dtype=float) if markov_sg_enabled else None
    )
    markov_sg_selected_source_log = np.zeros(nFE, dtype=int) if markov_sg_enabled else None
    markov_sg_score_policy_log = np.full(nFE, np.nan, dtype=float) if markov_sg_enabled else None
    markov_sg_score_supervisor_log = np.full(nFE, np.nan, dtype=float) if markov_sg_enabled else None
    markov_sg_advantage_log = np.full(nFE, np.nan, dtype=float) if markov_sg_enabled else None

    matrix_alpha_log = np.ones(nFE, dtype=float)
    matrix_delta_log = np.ones((nFE, n_inputs), dtype=float)
    matrix_decision_log = np.zeros(nFE, dtype=int)
    matrix_policy_multiplier_log = np.ones((nFE, model_baseline_raw.size), dtype=float)
    matrix_candidate_multiplier_log = np.ones((nFE, model_baseline_raw.size), dtype=float)
    matrix_executed_multiplier_log = np.ones((nFE, model_baseline_raw.size), dtype=float)
    matrix_candidate_action_raw_log = np.zeros((nFE, model_baseline_raw.size), dtype=float)
    matrix_final_executed_action_raw_log = np.zeros((nFE, model_baseline_raw.size), dtype=float)
    matrix_release_effective_low_log = np.ones((nFE, model_baseline_raw.size), dtype=float)
    matrix_release_effective_high_log = np.ones((nFE, model_baseline_raw.size), dtype=float)
    matrix_release_phase_log = np.zeros(nFE, dtype=int)
    matrix_release_guard_active_log = np.zeros(nFE, dtype=int)
    matrix_release_clip_fraction_log = np.zeros(nFE, dtype=float)
    matrix_release_ramp_fraction_log = np.zeros(nFE, dtype=float)
    matrix_release_policy_action_raw_log = np.zeros((nFE, model_baseline_raw.size), dtype=float)
    matrix_release_executed_action_raw_log = np.zeros((nFE, model_baseline_raw.size), dtype=float)
    matrix_phase1_action_source_log = np.zeros(nFE, dtype=int) if matrix_phase1 is not None else None
    matrix_policy_action_raw_log = np.zeros((nFE, model_baseline_raw.size), dtype=float) if matrix_phase1 is not None else None
    matrix_executed_action_raw_log = np.zeros((nFE, model_baseline_raw.size), dtype=float) if matrix_phase1 is not None else None
    matrix_phase1_train_traces = init_phase1_train_traces() if matrix_phase1 is not None else None

    weight_log = np.ones((nFE, 4), dtype=float)
    weight_decision_log = np.zeros(nFE, dtype=int)
    weight_phase1_action_source_log = np.zeros(nFE, dtype=int) if weight_phase1 is not None else None
    weight_policy_action_raw_log = np.zeros((nFE, weight_baseline_raw.size), dtype=float) if weight_phase1 is not None else None
    weight_executed_action_raw_log = np.zeros((nFE, weight_baseline_raw.size), dtype=float) if weight_phase1 is not None else None
    weight_phase1_train_traces = init_phase1_train_traces() if weight_phase1 is not None else None
    weight_sg_policy_action_raw_log = (
        np.zeros((nFE, weight_baseline_raw.size), dtype=float) if weight_sg_enabled else None
    )
    weight_sg_supervisor_action_raw_log = (
        np.zeros((nFE, weight_baseline_raw.size), dtype=float) if weight_sg_enabled else None
    )
    weight_sg_previous_action_raw_log = (
        np.zeros((nFE, weight_baseline_raw.size), dtype=float) if weight_sg_enabled else None
    )
    weight_sg_selected_source_log = np.zeros(nFE, dtype=int) if weight_sg_enabled else None
    weight_sg_score_policy_log = np.full(nFE, np.nan, dtype=float) if weight_sg_enabled else None
    weight_sg_score_supervisor_log = np.full(nFE, np.nan, dtype=float) if weight_sg_enabled else None
    weight_sg_advantage_log = np.full(nFE, np.nan, dtype=float) if weight_sg_enabled else None

    a_res_raw_log = np.zeros((nFE, n_inputs), dtype=float)
    a_res_exec_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_raw_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_exec_log = np.zeros((nFE, n_inputs), dtype=float)
    residual_decision_log = np.zeros(nFE, dtype=int)
    residual_phase1_action_source_log = np.zeros(nFE, dtype=int) if residual_phase1 is not None else None
    residual_policy_action_raw_log = np.zeros((nFE, residual_baseline_raw.size), dtype=float) if residual_phase1 is not None else None
    residual_executed_action_raw_log = np.zeros((nFE, residual_baseline_raw.size), dtype=float) if residual_phase1 is not None else None
    residual_phase1_train_traces = init_phase1_train_traces() if residual_phase1 is not None else None
    residual_sg_policy_action_raw_log = (
        np.zeros((nFE, residual_baseline_raw.size), dtype=float) if residual_sg_enabled else None
    )
    residual_sg_supervisor_action_raw_log = (
        np.zeros((nFE, residual_baseline_raw.size), dtype=float) if residual_sg_enabled else None
    )
    residual_sg_previous_action_raw_log = (
        np.zeros((nFE, residual_baseline_raw.size), dtype=float) if residual_sg_enabled else None
    )
    residual_sg_selected_source_log = np.zeros(nFE, dtype=int) if residual_sg_enabled else None
    residual_sg_score_policy_log = np.full(nFE, np.nan, dtype=float) if residual_sg_enabled else None
    residual_sg_score_supervisor_log = np.full(nFE, np.nan, dtype=float) if residual_sg_enabled else None
    residual_sg_advantage_log = np.full(nFE, np.nan, dtype=float) if residual_sg_enabled else None
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
        "markov": {
            "innovation": np.zeros((nFE, n_outputs), dtype=float) if markov_enabled and markov_state_mode == "mismatch" else None,
            "innovation_raw": np.zeros((nFE, n_outputs), dtype=float) if markov_enabled and markov_state_mode == "mismatch" else None,
            "tracking_error": np.zeros((nFE, n_outputs), dtype=float) if markov_enabled and markov_state_mode == "mismatch" else None,
            "tracking_error_raw": np.zeros((nFE, n_outputs), dtype=float) if markov_enabled and markov_state_mode == "mismatch" else None,
            "tracking_scale": np.zeros((nFE, n_outputs), dtype=float) if markov_enabled and markov_state_mode == "mismatch" else None,
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
    test = False
    nonfinite_matrix_action_count = 0
    matrix_A_model_delta_ratio_log = np.zeros(nFE, dtype=float)
    matrix_B_model_delta_ratio_log = np.zeros(nFE, dtype=float)
    b_min = np.asarray(combined_cfg["b_min"], float).reshape(-1)
    b_max = np.asarray(combined_cfg["b_max"], float).reshape(-1)
    markov_ctx = {
        "warm_start_step": int(warm_start_step),
        "time_in_sub_episodes": int(time_in_sub_episodes),
    }
    markov_history = {
        "xhat_before": np.zeros((nFE, n_states), dtype=float),
        "y_scaled_dev": np.zeros((nFE + 1, n_outputs), dtype=float),
        "u_dev_log": np.zeros((nFE, n_inputs), dtype=float),
    }
    markov_history["y_scaled_dev"][0, :] = apply_min_max(
        y_system[0, :],
        data_min[n_inputs:],
        data_max[n_inputs:],
    ) - y_ss_scaled
    z_prev = np.zeros(markov_z_dim, dtype=float)
    last_markov_raw_action = None
    last_markov_action_test = None
    pending_markov_transition = None
    markov_warm_reference_rewards = []
    markov_warm_release_reference_reward = None
    markov_probation_cooldown_until_subepisode = 0
    markov_probation_trigger_count = 0
    markov_warm_subepisodes = int(warm_start_step // max(1, time_in_sub_episodes))

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
        if markov_enabled:
            markov_history["xhat_before"][i, :] = xhatdhat[:, i]

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
            if horizon_sg_enabled:
                horizon_decision = select_supervisor_gated_horizon_action(
                    agent=horizon_agent,
                    state=current_states["horizon"],
                    step=i,
                    warm_start_step=warm_start_step,
                    decision_interval=decision_interval,
                    default_action=horizon_baseline_idx,
                    supervisor_action=horizon_baseline_idx,
                    last_action=last_horizon_idx,
                    test=test,
                    post_warm_action_freeze_steps=horizon_action_freeze_steps,
                )
                horizon_sg_policy_action_log[i] = int(horizon_decision.policy_action)
                horizon_sg_supervisor_action_log[i] = int(horizon_decision.supervisor_action)
                horizon_sg_previous_action_log[i] = int(horizon_decision.previous_action)
                horizon_sg_selected_source_log[i] = int(horizon_decision.selected_source)
                horizon_sg_score_policy_log[i] = float(horizon_decision.score_policy)
                horizon_sg_score_supervisor_log[i] = float(horizon_decision.score_supervisor)
                horizon_sg_advantage_log[i] = float(horizon_decision.advantage_policy_supervisor)
                horizon_sg_q_policy_log[i] = float(horizon_decision.q_policy)
                horizon_sg_q_supervisor_log[i] = float(horizon_decision.q_supervisor)
            else:
                horizon_decision = select_horizon_action(
                    agent=horizon_agent,
                    state=current_states["horizon"],
                    step=i,
                    warm_start_step=warm_start_step,
                    decision_interval=decision_interval,
                    default_action=horizon_baseline_idx,
                    last_action=last_horizon_idx,
                    test=test,
                    post_warm_action_freeze_steps=horizon_action_freeze_steps,
                )
            h_idx = int(horizon_decision.action)
            last_horizon_idx = horizon_decision.last_action
            horizon_decision_log[i] = int(horizon_decision.decision_taken)
            Hp, Hc = action_to_horizons(horizon_recipes, h_idx)
        else:
            h_idx = horizon_baseline_idx
            Hp, Hc = current_Hp, current_Hc

        if matrix_enabled:
            matrix_action_decision = select_continuous_action(
                agent=matrix_agent,
                state=current_states["matrix"],
                step=i,
                warm_start_step=warm_start_step,
                test=test,
                baseline_action=model_baseline_raw,
                phase1=matrix_phase1,
                action_dim=model_baseline_raw.size,
                nonfinite_fallback=True,
            )
            model_raw = matrix_action_decision.action
            matrix_policy_raw = matrix_action_decision.policy_action
            current_model_source = int(matrix_action_decision.source)
            matrix_decision_log[i] = int(i > warm_start_step and current_model_source in {2, 3})
        else:
            model_raw = model_baseline_raw.copy()
            matrix_policy_raw = None
            current_model_source = 0
        model_raw = np.asarray(model_raw, float).reshape(-1)
        if not np.all(np.isfinite(model_raw)):
            warnings.warn(
                "Combined matrix agent produced a non-finite action; falling back to the last valid or nominal matrix action."
            )
            model_raw = model_baseline_raw.copy()
            nonfinite_matrix_action_count += 1
        elif matrix_enabled and matrix_action_decision.nonfinite_fallback_used:
            nonfinite_matrix_action_count += 1
        if matrix_phase1 is not None:
            matrix_policy_action_raw_log[i, :] = np.asarray(
                matrix_policy_raw if matrix_policy_raw is not None else model_baseline_raw,
                float,
            ).reshape(-1)
            matrix_phase1_action_source_log[i] = int(current_model_source)
        policy_mapped = map_centered_action_to_bounds(
            model_raw,
            model_low,
            model_high,
            nominal=1.0,
        )
        release_trace = clip_multipliers_to_release_bounds(policy_mapped, matrix_release_schedule, i)
        model_mapped = np.asarray(release_trace["multipliers"], float).reshape(-1)
        matrix_executed_raw = map_effective_multipliers_to_raw_action(model_mapped, model_low, model_high)
        if matrix_phase1 is not None:
            matrix_executed_action_raw_log[i, :] = np.asarray(matrix_executed_raw, float).reshape(-1)
        matrix_policy_multiplier_log[i, :] = np.asarray(policy_mapped, float).reshape(-1)
        matrix_candidate_multiplier_log[i, :] = np.asarray(model_mapped, float).reshape(-1)
        matrix_executed_multiplier_log[i, :] = np.asarray(model_mapped, float).reshape(-1)
        matrix_candidate_action_raw_log[i, :] = np.asarray(matrix_executed_raw, float).reshape(-1)
        matrix_final_executed_action_raw_log[i, :] = np.asarray(matrix_executed_raw, float).reshape(-1)
        matrix_release_effective_low_log[i, :] = np.asarray(release_trace["low"], float).reshape(-1)
        matrix_release_effective_high_log[i, :] = np.asarray(release_trace["high"], float).reshape(-1)
        matrix_release_phase_log[i] = int(release_trace["phase_code"])
        matrix_release_guard_active_log[i] = int(bool(release_trace["guard_active"]))
        matrix_release_clip_fraction_log[i] = float(release_trace["clip_fraction"])
        matrix_release_ramp_fraction_log[i] = float(release_trace["ramp_fraction"])
        matrix_release_policy_action_raw_log[i, :] = np.asarray(model_raw, float).reshape(-1)
        matrix_release_executed_action_raw_log[i, :] = np.asarray(matrix_executed_raw, float).reshape(-1)
        matrix_replay_action = matrix_executed_raw if matrix_release_store_executed_action else model_raw
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
            if weight_sg_enabled:
                weight_action_decision = select_supervisor_gated_continuous_action(
                    agent=weight_agent,
                    state=current_states["weights"],
                    step=i,
                    warm_start_step=warm_start_step,
                    test=test,
                    baseline_action=weight_baseline_raw,
                    supervisor_action=weight_baseline_raw,
                    phase1=weight_phase1,
                    action_dim=4,
                    nonfinite_fallback=fallback_to_identity_on_nonfinite,
                )
                weight_sg_policy_action_raw_log[i, :] = weight_action_decision.policy_action
                weight_sg_supervisor_action_raw_log[i, :] = weight_action_decision.supervisor_action
                weight_sg_previous_action_raw_log[i, :] = weight_action_decision.previous_action
                weight_sg_selected_source_log[i] = int(weight_action_decision.selected_source)
                weight_sg_score_policy_log[i] = float(weight_action_decision.score_policy)
                weight_sg_score_supervisor_log[i] = float(weight_action_decision.score_supervisor)
                weight_sg_advantage_log[i] = float(weight_action_decision.advantage_policy_supervisor)
            else:
                weight_action_decision = select_continuous_action(
                    agent=weight_agent,
                    state=current_states["weights"],
                    step=i,
                    warm_start_step=warm_start_step,
                    test=test,
                    baseline_action=weight_baseline_raw,
                    phase1=weight_phase1,
                    action_dim=4,
                    nonfinite_fallback=fallback_to_identity_on_nonfinite,
                )
            weight_raw = weight_action_decision.action
            weight_policy_raw = weight_action_decision.policy_action
            current_weight_source = int(weight_action_decision.source)
            weight_decision_log[i] = int(i > warm_start_step and current_weight_source in {2, 3})
        else:
            weight_raw = weight_baseline_raw.copy()
            weight_policy_raw = None
            current_weight_source = 0
        weight_raw = np.asarray(weight_raw, float).reshape(-1)
        if weight_raw.size != 4:
            raise ValueError("Weights action must contain 4 elements for [Q1, Q2, R1, R2].")
        weight_mult = _map_to_bounds(weight_raw, weights_low, weights_high)
        if (
            not np.all(np.isfinite(weight_mult))
            or np.any(weight_mult < weights_low - 1.0e-9)
            or np.any(weight_mult > weights_high + 1.0e-9)
        ):
            weight_raw = weight_baseline_raw.copy()
            weight_mult = np.ones(4, dtype=float)
        weight_log[i, :] = weight_mult
        if weight_phase1 is not None:
            weight_policy_action_raw_log[i, :] = np.asarray(
                weight_policy_raw if weight_policy_raw is not None else weight_baseline_raw,
                float,
            ).reshape(-1)
            weight_executed_action_raw_log[i, :] = np.asarray(weight_raw, float).reshape(-1)
            weight_phase1_action_source_log[i] = int(current_weight_source)
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
        if markov_enabled:
            markov_base_state, markov_state_debug = build_agent_state("markov", markov_state_mode)
            current_state_debugs["markov"] = markov_state_debug
            m_blocks = compute_markov_blocks(A_base, B_base, C_aug, current_Hp)
            basis_blocks, markov_basis_labels_step = make_markov_basis(
                m_blocks,
                markov_cfg.get("basis_family", "io_pair_gain"),
            )
            if not markov_basis_labels:
                markov_basis_labels = list(markov_basis_labels_step)
            G0 = build_toeplitz_from_markov(m_blocks, current_Hp, current_Hc)
            Wy = np.eye(current_Hp * n_outputs, dtype=float)
            z_bounds = [(-markov_z_bound, markov_z_bound) for _ in range(markov_z_dim)]
            U0, J0, sol0 = solve_lifted_mpc(
                y_sp[i, :],
                scaled_current_input_dev,
                xhatdhat[:, i],
                A_base,
                C_aug,
                G0,
                Q_now,
                R_now,
                current_Hp,
                current_Hc,
                bounds,
                ic_opt_step,
            )
            markov_nominal_cost_log[i] = float(J0)

            def evaluate_markov_candidate(z_candidate):
                mz_candidate = apply_markov_correction(m_blocks, basis_blocks, z_candidate)
                G_candidate = build_toeplitz_from_markov(mz_candidate, current_Hp, current_Hc)
                candidate_drift = gain_drift(G_candidate, G0)
                U_candidate, J_candidate, sol_candidate = solve_lifted_mpc(
                    y_sp[i, :],
                    scaled_current_input_dev,
                    xhatdhat[:, i],
                    A_base,
                    C_aug,
                    G_candidate,
                    Q_now,
                    R_now,
                    current_Hp,
                    current_Hc,
                    bounds,
                    U0,
                )
                nominal_cost_of_candidate = lifted_mpc_cost(
                    U_candidate,
                    y_sp[i, :],
                    scaled_current_input_dev,
                    free_response(A_base, C_aug, xhatdhat[:, i], current_Hp),
                    G0,
                    Q_now,
                    R_now,
                    current_Hp,
                    current_Hc,
                )
                loose_tol = float(markov_cfg["nominal_cost_absolute_tol"]) + float(
                    markov_cfg["nominal_cost_relative_tol"]
                ) * abs(float(J0))
                cost_margin = float(nominal_cost_of_candidate) - float(J0)
                return {
                    "U": U_candidate,
                    "J": J_candidate,
                    "sol": sol_candidate,
                    "drift": candidate_drift,
                    "nominal_cost": nominal_cost_of_candidate,
                    "reference_nominal_cost": float(J0),
                    "cost_margin": cost_margin,
                    "cost_guard_pass": nominal_cost_of_candidate <= float(J0) + loose_tol,
                }

            default_score = _default_markov_score(n_outputs)
            ls_score = dict(default_score)
            requested_score = dict(default_score)
            executed_score = dict(default_score)
            ls_eval = None
            requested_eval = None
            executed_eval = {
                "U": U0.copy(),
                "J": float(J0),
                "sol": sol0,
                "drift": 0.0,
                "nominal_cost": float(J0),
                "reference_nominal_cost": float(J0),
                "cost_margin": 0.0,
                "cost_guard_pass": True,
            }
            z_ls = np.zeros(markov_z_dim, dtype=float)
            z_ls_safe = np.zeros(markov_z_dim, dtype=float)
            z_requested = np.zeros(markov_z_dim, dtype=float)
            z_exec = np.zeros(markov_z_dim, dtype=float)
            raw_actor_requested = np.zeros(markov_z_dim, dtype=float)
            raw_requested = np.zeros(markov_z_dim, dtype=float)
            raw_executed = np.zeros(markov_z_dim, dtype=float)
            U_exec = U0.copy()
            markov_source = 0
            markov_fallback = True
            markov_accepted = False
            ls_accepted = False
            ls_drift = 0.0
            requested_drift = 0.0
            executed_drift = 0.0
            current_subepisode = int(i // max(1, time_in_sub_episodes)) + 1
            probation_active = bool(
                _td3_priority_enabled(markov_cfg)
                and not bool(markov_cfg.get("force_td3_execute", False))
                and i > warm_start_step
                and current_subepisode <= int(markov_probation_cooldown_until_subepisode)
            )
            z_safety_effective_cap = resolve_z_safety_effective_cap(
                markov_cfg,
                markov_ctx,
                i,
                probation_active=probation_active,
            )
            markov_z_safety_logs["markov_z_safety_effective_cap_log"][i] = float(z_safety_effective_cap)
            markov_td3_probation_active_log[i] = int(probation_active)

            if bool(markov_cfg.get("run_adaptive_ls", True)) and i >= current_Hp:
                z_ls_uncapped, _ls_result, ls_score = fit_markov_ls_correction(
                    z_prev,
                    z_bounds,
                    markov_history,
                    m_blocks,
                    basis_blocks,
                    G0,
                    A_base,
                    C_aug,
                    current_Hp,
                    current_Hc,
                    Wy,
                    float(markov_cfg["lambda_z"]),
                    i,
                    int(markov_cfg["prediction_window"]),
                )
                markov_ls_z_uncapped_log[i, :] = z_ls_uncapped
                z_ls, ls_safety_info = apply_z_safety_projection(
                    z_ls_uncapped,
                    markov_cfg,
                    effective_cap=z_safety_effective_cap,
                )
                _record_markov_safety_stage(markov_z_safety_logs, i, "ls", ls_safety_info)
                ls_score = prediction_improvement_score(
                    z=z_ls,
                    history=markov_history,
                    m_blocks=m_blocks,
                    basis_blocks=basis_blocks,
                    G0=G0,
                    A=A_base,
                    C=C_aug,
                    predict_h=current_Hp,
                    control_horizon=current_Hc,
                    Wy=Wy,
                    lambda_z=float(markov_cfg["lambda_z"]),
                    current_step=i,
                    prediction_window=int(markov_cfg["prediction_window"]),
                )
                ls_eval = evaluate_markov_candidate(z_ls)
                ls_drift = float(ls_eval["drift"])
                ls_accepted = bool(
                    sol0.success
                    and ls_eval["sol"].success
                    and ls_score["score"] > float(markov_cfg["s_pred_min"])
                    and ls_drift <= float(markov_cfg["gain_drift_max"])
                    and ls_eval["cost_guard_pass"]
                )
                z_ls_safe = z_ls if ls_accepted else np.zeros(markov_z_dim, dtype=float)

            horizon_scale = (
                max([hp for hp, _ in horizon_recipes], default=current_Hp),
                max([hc for _, hc in horizon_recipes], default=current_Hc),
            )
            markov_state = _combined_markov_state(
                markov_base_state,
                z_prev,
                z_ls_safe,
                float(ls_score.get("score", 0.0)),
                ls_drift,
                (current_Hp, current_Hc),
                horizon_scale,
                weight_mult,
            )
            current_states["markov"] = markov_state

            if pending_markov_transition is not None:
                if pending_markov_transition.get("reward") is None:
                    raise RuntimeError(
                        f"Combined Markov transition at step {pending_markov_transition['step']} "
                        "has no reward before replay flush."
                    )
                if markov_sg_enabled and pending_markov_transition.get("decision") is not None:
                    replay_result = replay_train_supervisor_gated_continuous_agent(
                        agent=markov_agent,
                        state=pending_markov_transition["state"],
                        action=pending_markov_transition["action"],
                        reward=pending_markov_transition["reward"],
                        next_state=markov_state,
                        done=0.0,
                        step=pending_markov_transition["step"],
                        test=pending_markov_transition["test"],
                        train_start_step=warm_start_step,
                        decision=pending_markov_transition["decision"],
                        phase1_train_traces=markov_phase1_train_traces if markov_phase1 is not None else None,
                    )
                else:
                    replay_result = replay_train_continuous_agent(
                        agent=markov_agent,
                        state=pending_markov_transition["state"],
                        action=pending_markov_transition["action"],
                        reward=pending_markov_transition["reward"],
                        next_state=markov_state,
                        done=0.0,
                        step=pending_markov_transition["step"],
                        test=pending_markov_transition["test"],
                        train_start_step=warm_start_step,
                        phase1_train_traces=markov_phase1_train_traces if markov_phase1 is not None else None,
                    )
                train_meta = replay_result.get("train_meta")
                markov_replay_pushed_log[pending_markov_transition["step"]] = int(
                    bool(replay_result.get("pushed", False))
                )
                markov_train_called_log[pending_markov_transition["step"]] = int(
                    bool(replay_result.get("trained", False))
                )
                markov_train_updated_log[pending_markov_transition["step"]] = int(
                    bool(train_meta is not None and train_meta.get("critic_updated", False))
                )
                pending_markov_transition = None

            if bool(markov_cfg.get("run_rl_proposal", True)) and markov_agent is not None:
                baseline_raw = z_to_raw_action(z_ls_safe, markov_z_bound)
                if markov_sg_enabled:
                    markov_decision = select_supervisor_gated_continuous_action(
                        agent=markov_agent,
                        state=markov_state,
                        step=i,
                        warm_start_step=warm_start_step,
                        decision_interval=1,
                        last_action=last_markov_raw_action,
                        last_action_test=last_markov_action_test,
                        test=test,
                        baseline_action=baseline_raw,
                        supervisor_action=baseline_raw,
                        phase1=markov_phase1,
                        action_dim=markov_z_dim,
                        nonfinite_fallback=True,
                    )
                    markov_sg_policy_action_raw_log[i, :] = markov_decision.policy_action
                    markov_sg_supervisor_action_raw_log[i, :] = markov_decision.supervisor_action
                    markov_sg_previous_action_raw_log[i, :] = markov_decision.previous_action
                    markov_sg_selected_source_log[i] = int(markov_decision.selected_source)
                    markov_sg_score_policy_log[i] = float(markov_decision.score_policy)
                    markov_sg_score_supervisor_log[i] = float(markov_decision.score_supervisor)
                    markov_sg_advantage_log[i] = float(markov_decision.advantage_policy_supervisor)
                else:
                    markov_decision = select_continuous_action(
                        agent=markov_agent,
                        state=markov_state,
                        step=i,
                        warm_start_step=warm_start_step,
                        decision_interval=int(markov_cfg.get("decision_interval", 1)),
                        last_action=last_markov_raw_action,
                        last_action_test=last_markov_action_test,
                        test=test,
                        baseline_action=baseline_raw,
                        phase1=markov_phase1,
                        action_dim=markov_z_dim,
                        nonfinite_fallback=True,
                    )
                raw_actor_requested = np.asarray(
                    markov_decision.policy_action
                    if markov_sg_enabled and markov_decision.policy_action is not None
                    else markov_decision.action,
                    float,
                ).reshape(-1)
                raw_requested = np.asarray(markov_decision.action, float).reshape(-1)
                markov_policy_source_log[i] = int(markov_decision.source)
                markov_decision_log[i] = int(markov_decision.decision_taken)
                last_markov_action_test = markov_decision.last_action_test
                if markov_phase1 is not None:
                    markov_policy_action_raw_log[i, :] = np.asarray(
                        markov_decision.policy_action if markov_decision.policy_action is not None else baseline_raw,
                        float,
                    ).reshape(-1)
                    markov_phase1_action_source_log[i] = int(markov_decision.source)

                authority_scale = (
                    _td3_priority_authority_scale(markov_cfg, markov_ctx, i, probation_active=probation_active)
                    if _td3_priority_enabled(markov_cfg) and i > warm_start_step
                    else 1.0
                )
                raw_requested = np.clip(raw_requested * authority_scale, -1.0, 1.0)
                markov_td3_authority_scale_log[i] = float(authority_scale)
                markov_td3_priority_phase_log[i] = TD3_PRIORITY_PHASE_CODE.get(
                    _td3_priority_phase(markov_cfg, markov_ctx, i) if i > warm_start_step else "none",
                    0,
                )
                z_requested_uncapped = raw_action_to_z(raw_requested, markov_z_bound)
                markov_requested_z_uncapped_log[i, :] = z_requested_uncapped
                z_requested, requested_safety_info = apply_z_safety_projection(
                    z_requested_uncapped,
                    markov_cfg,
                    effective_cap=z_safety_effective_cap,
                )
                _record_markov_safety_stage(markov_z_safety_logs, i, "requested", requested_safety_info)
                raw_requested = z_to_raw_action(z_requested, markov_z_bound)
                last_markov_raw_action = raw_requested.copy()
                markov_actor_raw_action_log[i, :] = raw_actor_requested
                markov_requested_raw_action_log[i, :] = raw_requested
                markov_requested_z_log[i, :] = z_requested
                markov_z_proposed_log[i, :] = z_requested

                if bool(markov_cfg.get("run_live_corrected_mpc", True)) and i >= current_Hp:
                    requested_score = prediction_improvement_score(
                        z=z_requested,
                        history=markov_history,
                        m_blocks=m_blocks,
                        basis_blocks=basis_blocks,
                        G0=G0,
                        A=A_base,
                        C=C_aug,
                        predict_h=current_Hp,
                        control_horizon=current_Hc,
                        Wy=Wy,
                        lambda_z=float(markov_cfg["lambda_z"]),
                        current_step=i,
                        prediction_window=int(markov_cfg["prediction_window"]),
                    )
                    requested_eval = evaluate_markov_candidate(z_requested)
                    requested_drift = float(requested_eval["drift"])
                    if bool(markov_cfg.get("force_td3_execute", False)):
                        requested_accepted = True
                    elif _td3_priority_enabled(markov_cfg):
                        requested_accepted = _td3_priority_candidate_allowed(
                            markov_cfg,
                            markov_ctx,
                            i,
                            requested_eval,
                            requested_score,
                            sol0.success,
                        )
                    else:
                        requested_accepted = bool(
                            sol0.success
                            and requested_eval["sol"].success
                            and requested_score["score"] > float(markov_cfg["s_pred_min"])
                            and requested_eval["drift"] <= float(markov_cfg["gain_drift_max"])
                            and requested_eval["cost_guard_pass"]
                        )
                    if markov_sg_enabled:
                        selected_source = int(markov_decision.selected_source)
                        supervisor_is_ls = bool(ls_accepted and ls_eval is not None)
                        if supervisor_is_ls:
                            supervisor_U = ls_eval["U"]
                            supervisor_z = z_ls
                            supervisor_raw = z_to_raw_action(z_ls, markov_z_bound)
                            supervisor_score = ls_score
                            supervisor_eval = ls_eval
                            supervisor_drift = ls_drift
                        else:
                            supervisor_U = U0.copy()
                            supervisor_z = np.zeros(markov_z_dim, dtype=float)
                            supervisor_raw = np.zeros(markov_z_dim, dtype=float)
                            supervisor_score = dict(default_score)
                            supervisor_eval = {
                                "U": U0.copy(),
                                "J": float(J0),
                                "sol": sol0,
                                "drift": 0.0,
                                "nominal_cost": float(J0),
                                "reference_nominal_cost": float(J0),
                                "cost_margin": 0.0,
                                "cost_guard_pass": True,
                            }
                            supervisor_drift = 0.0

                        if i <= warm_start_step:
                            if supervisor_is_ls:
                                U_exec = supervisor_U
                                z_exec = supervisor_z
                                raw_executed = supervisor_raw
                                executed_score = supervisor_score
                                executed_eval = supervisor_eval
                                executed_drift = supervisor_drift
                                markov_source = 1
                                markov_fallback = False
                                markov_accepted = True
                            else:
                                markov_source = 0
                        elif selected_source == SG_SOURCE_POLICY and requested_accepted:
                            U_exec = requested_eval["U"]
                            z_exec = z_requested
                            raw_executed = raw_requested
                            executed_score = requested_score
                            executed_eval = requested_eval
                            executed_drift = requested_drift
                            markov_source = 2
                            markov_fallback = False
                            markov_accepted = True
                        elif selected_source in {SG_SOURCE_POLICY, SG_SOURCE_FALLBACK}:
                            U_exec = supervisor_U
                            z_exec = supervisor_z
                            raw_executed = supervisor_raw
                            executed_score = supervisor_score
                            executed_eval = supervisor_eval
                            executed_drift = supervisor_drift
                            markov_source = 8
                            markov_fallback = True
                            markov_accepted = supervisor_is_ls
                        else:
                            U_exec = supervisor_U
                            z_exec = supervisor_z
                            raw_executed = supervisor_raw
                            executed_score = supervisor_score
                            executed_eval = supervisor_eval
                            executed_drift = supervisor_drift
                            markov_source = 6 if supervisor_is_ls else 7
                            markov_fallback = False
                            markov_accepted = supervisor_is_ls
                    elif requested_accepted and i > warm_start_step:
                        U_exec = requested_eval["U"]
                        z_exec = z_requested
                        raw_executed = raw_requested
                        executed_score = requested_score
                        executed_eval = requested_eval
                        executed_drift = requested_drift
                        markov_source = 2
                        markov_fallback = False
                        markov_accepted = True
                    elif bool(markov_cfg.get("rl_fallback_to_ls", True)) and ls_accepted and ls_eval is not None:
                        U_exec = ls_eval["U"]
                        z_exec = z_ls
                        raw_executed = z_to_raw_action(z_ls, markov_z_bound)
                        executed_score = ls_score
                        executed_eval = ls_eval
                        executed_drift = ls_drift
                        markov_source = 3 if i > warm_start_step else 1
                        markov_fallback = True
                        markov_accepted = True
                    else:
                        markov_source = 4 if i > warm_start_step else 0
                elif ls_accepted and ls_eval is not None:
                    U_exec = ls_eval["U"]
                    z_exec = z_ls
                    raw_executed = z_to_raw_action(z_ls, markov_z_bound)
                    executed_score = ls_score
                    executed_eval = ls_eval
                    executed_drift = ls_drift
                    markov_source = 1
                    markov_fallback = False
                    markov_accepted = True
            elif ls_accepted and ls_eval is not None:
                U_exec = ls_eval["U"]
                z_exec = z_ls
                raw_requested = z_to_raw_action(z_ls, markov_z_bound)
                raw_executed = raw_requested
                executed_score = ls_score
                executed_eval = ls_eval
                executed_drift = ls_drift
                markov_source = 5
                markov_fallback = False
                markov_accepted = True

            z_prev = z_exec.copy()
            markov_z_log[i, :] = z_exec
            markov_z_executed_log[i, :] = z_exec
            markov_ls_z_log[i, :] = z_ls
            markov_executed_raw_action_log[i, :] = raw_executed
            markov_action_source_log[i] = int(markov_source)
            markov_accepted_log[i] = int(markov_accepted)
            markov_fallback_log[i] = int(markov_fallback)
            markov_prediction_score_log[i] = float(executed_score.get("score", np.nan))
            markov_ls_prediction_score_log[i] = float(ls_score.get("score", np.nan))
            markov_requested_prediction_score_log[i] = float(requested_score.get("score", np.nan))
            markov_gain_drift_log[i] = float(executed_drift)
            markov_ls_gain_drift_log[i] = float(ls_drift)
            markov_requested_gain_drift_log[i] = float(requested_drift)
            markov_executed_cost_margin_log[i] = float(executed_eval.get("cost_margin", np.nan))
            if markov_phase1 is not None:
                markov_phase1_executed_action_raw_log[i, :] = raw_executed

            markov_replay_action = (
                raw_executed if bool(markov_cfg.get("rl_store_executed_action_in_replay", True)) else raw_requested
            )
            if not test:
                pending_markov_transition = {
                    "state": markov_state.copy(),
                    "action": np.asarray(markov_replay_action, float).copy(),
                    "reward": None,
                    "step": i,
                    "test": False,
                    "decision": markov_decision if markov_sg_enabled else None,
                }
            sol = sol0
            sol.x = np.asarray(U_exec, float)
        else:
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
            if residual_sg_enabled:
                residual_action_decision = select_supervisor_gated_continuous_action(
                    agent=residual_agent,
                    state=current_states["residual"],
                    step=i,
                    warm_start_step=warm_start_step,
                    test=test,
                    baseline_action=residual_baseline_raw,
                    supervisor_action=residual_baseline_raw,
                    phase1=residual_phase1,
                    action_dim=n_inputs,
                    nonfinite_fallback=fallback_to_zero_on_nonfinite,
                )
                residual_sg_policy_action_raw_log[i, :] = residual_action_decision.policy_action
                residual_sg_supervisor_action_raw_log[i, :] = residual_action_decision.supervisor_action
                residual_sg_previous_action_raw_log[i, :] = residual_action_decision.previous_action
                residual_sg_selected_source_log[i] = int(residual_action_decision.selected_source)
                residual_sg_score_policy_log[i] = float(residual_action_decision.score_policy)
                residual_sg_score_supervisor_log[i] = float(residual_action_decision.score_supervisor)
                residual_sg_advantage_log[i] = float(residual_action_decision.advantage_policy_supervisor)
            else:
                residual_action_decision = select_continuous_action(
                    agent=residual_agent,
                    state=current_states["residual"],
                    step=i,
                    warm_start_step=warm_start_step,
                    test=test,
                    baseline_action=residual_baseline_raw,
                    phase1=residual_phase1,
                    action_dim=n_inputs,
                    nonfinite_fallback=fallback_to_zero_on_nonfinite,
                )
            residual_raw_action = residual_action_decision.action
            residual_policy_raw = residual_action_decision.policy_action
            current_residual_source = int(residual_action_decision.source)
            residual_decision_log[i] = int(i > warm_start_step and current_residual_source in {2, 3})
        else:
            residual_raw_action = residual_baseline_raw.copy()
            residual_policy_raw = None
            current_residual_source = 0
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
            apply_authority=bool(
                residual_enabled
                and residual_state_mode == "mismatch"
                and residual_authority_enabled
            ),
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
        if fallback_to_zero_on_nonfinite and not _projection_is_finite(residual_projection):
            residual_raw_action = residual_baseline_raw.copy()
            residual_projection = project_residual_action(
                action_raw=residual_raw_action,
                low_coef=residual_low,
                high_coef=residual_high,
                u_base=u_base,
                scaled_current_input=scaled_current_input,
                u_min_scaled_abs=u_min_scaled_abs,
                u_max_scaled_abs=u_max_scaled_abs,
                apply_authority=False,
                authority_use_rho=False,
                residual_zero_deadband_enabled=False,
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
        if residual_phase1 is not None:
            residual_policy_action_raw_log[i, :] = np.asarray(
                residual_policy_raw if residual_policy_raw is not None else residual_baseline_raw,
                float,
            ).reshape(-1)
            residual_executed_action_raw_log[i, :] = np.asarray(residual_projection["a_exec"], float).reshape(-1)
            residual_phase1_action_source_log[i] = int(current_residual_source)
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
        if markov_enabled:
            markov_history["y_scaled_dev"][i + 1, :] = y_current_scaled
            markov_history["u_dev_log"][i, :] = u_applied_scaled[i, :] - ss_scaled_inputs
            if pending_markov_transition is not None and int(pending_markov_transition["step"]) == int(i):
                pending_markov_transition["reward"] = reward

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
            if horizon_enabled:
                next_state = build_next_state("horizon", horizon_state_mode)
                if horizon_sg_enabled:
                    replay_train_supervisor_gated_horizon_agent(
                        agent=horizon_agent,
                        state=current_states["horizon"],
                        action=h_idx,
                        reward=reward,
                        next_state=next_state,
                        done=0.0,
                        step=i,
                        test=False,
                        replay_start_step=time_in_sub_episodes,
                        train_start_step=warm_start_step,
                        decision=horizon_decision,
                    )
                else:
                    replay_train_horizon_agent(
                        agent=horizon_agent,
                        state=current_states["horizon"],
                        action=h_idx,
                        reward=reward,
                        next_state=next_state,
                        done=0.0,
                        step=i,
                        test=False,
                        replay_start_step=time_in_sub_episodes,
                        train_start_step=warm_start_step,
                    )

            if matrix_enabled:
                next_state = build_next_state("matrix", matrix_state_mode)
                replay_train_continuous_agent(
                    agent=matrix_agent,
                    state=current_states["matrix"],
                    action=matrix_replay_action,
                    reward=reward,
                    next_state=next_state,
                    done=0.0,
                    step=i,
                    test=False,
                    train_start_step=warm_start_step,
                    phase1_train_traces=matrix_phase1_train_traces if matrix_phase1 is not None else None,
                )

            if weight_enabled:
                next_state = build_next_state("weights", weight_state_mode)
                if weight_sg_enabled:
                    replay_train_supervisor_gated_continuous_agent(
                        agent=weight_agent,
                        state=current_states["weights"],
                        action=weight_raw,
                        reward=reward,
                        next_state=next_state,
                        done=0.0,
                        step=i,
                        test=False,
                        train_start_step=warm_start_step,
                        decision=weight_action_decision,
                        phase1_train_traces=weight_phase1_train_traces if weight_phase1 is not None else None,
                    )
                else:
                    replay_train_continuous_agent(
                        agent=weight_agent,
                        state=current_states["weights"],
                        action=weight_raw,
                        reward=reward,
                        next_state=next_state,
                        done=0.0,
                        step=i,
                        test=False,
                        train_start_step=warm_start_step,
                        phase1_train_traces=weight_phase1_train_traces if weight_phase1 is not None else None,
                    )

            if residual_enabled:
                next_state = build_next_state("residual", residual_state_mode)
                if residual_sg_enabled:
                    replay_train_supervisor_gated_continuous_agent(
                        agent=residual_agent,
                        state=current_states["residual"],
                        action=residual_projection["a_exec"],
                        reward=reward,
                        next_state=next_state,
                        done=0.0,
                        step=i,
                        test=False,
                        train_start_step=warm_start_step,
                        decision=residual_action_decision,
                        phase1_train_traces=residual_phase1_train_traces if residual_phase1 is not None else None,
                    )
                else:
                    replay_train_continuous_agent(
                        agent=residual_agent,
                        state=current_states["residual"],
                        action=residual_projection["a_exec"],
                        reward=reward,
                        next_state=next_state,
                        done=0.0,
                        step=i,
                        test=False,
                        train_start_step=warm_start_step,
                        phase1_train_traces=residual_phase1_train_traces if residual_phase1 is not None else None,
                    )

        if i in sub_episodes_changes_dict:
            subepisode_avg_reward = float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1]))
            avg_rewards.append(subepisode_avg_reward)
            subepisode_idx = int(sub_episodes_changes_dict[i])
            window_start = max(0, i - time_in_sub_episodes + 1)
            horizon_sg_source_summary = "off"
            if horizon_sg_enabled and horizon_sg_selected_source_log is not None:
                horizon_sg_source_summary = _sg_source_summary(
                    horizon_sg_selected_source_log[window_start : i + 1]
                )
            avg_markov_z_window = (
                np.mean(markov_z_log[window_start : i + 1, :], axis=0) if markov_enabled else "off"
            )
            markov_sg_source_summary = "off"
            if markov_sg_enabled and markov_sg_selected_source_log is not None:
                markov_sg_source_summary = _sg_source_summary(
                    markov_sg_selected_source_log[window_start : i + 1]
                )
            markov_exec_source_summary = "off"
            if markov_enabled:
                markov_sources = markov_action_source_log[window_start : i + 1]
                markov_exec_source_summary = (
                    f"td3={int(np.sum(markov_sources == 2))},"
                    f"sg_ls={int(np.sum(markov_sources == 6))},"
                    f"sg_mpc={int(np.sum(markov_sources == 7))},"
                    f"solver_fb={int(np.sum(markov_sources == 8))},"
                    f"ls_fb={int(np.sum(markov_sources == 3))},"
                    f"nom_fb={int(np.sum(markov_sources == 4))}"
                )
            weight_window = weight_log[window_start : i + 1, :]
            residual_window = delta_u_res_exec_log[window_start : i + 1, :]
            avg_weight_window = np.mean(weight_window, axis=0) if weight_enabled else "off"
            avg_residual_window = np.mean(residual_window, axis=0) if residual_enabled else "off"
            weight_sg_source_summary = "off"
            if weight_sg_enabled and weight_sg_selected_source_log is not None:
                weight_sg_source_summary = _sg_source_summary(
                    weight_sg_selected_source_log[window_start : i + 1]
                )
            residual_sg_source_summary = "off"
            if residual_sg_enabled and residual_sg_selected_source_log is not None:
                residual_sg_source_summary = _sg_source_summary(
                    residual_sg_selected_source_log[window_start : i + 1]
                )
            if markov_enabled:
                if subepisode_idx <= markov_warm_subepisodes:
                    markov_warm_reference_rewards.append(subepisode_avg_reward)
                    reward_probation_cfg = markov_cfg.get("td3_priority_fallback", {}).get("reward_probation", {})
                    if not isinstance(reward_probation_cfg, dict):
                        reward_probation_cfg = {}
                    n_ref = int(max(1, reward_probation_cfg.get("reference_warm_episodes", 3)))
                    markov_warm_release_reference_reward = float(np.mean(markov_warm_reference_rewards[-n_ref:]))
                else:
                    probation_cfg = markov_cfg.get("td3_priority_fallback", {}).get("reward_probation", {})
                    if not isinstance(probation_cfg, dict):
                        probation_cfg = {}
                    probation_enabled = bool(
                        _td3_priority_enabled(markov_cfg)
                        and probation_cfg.get("enabled", False)
                        and not bool(markov_cfg.get("force_td3_execute", False))
                        and markov_warm_release_reference_reward is not None
                    )
                    collapse_threshold = float(probation_cfg.get("collapse_threshold", np.inf))
                    if (
                        probation_enabled
                        and subepisode_avg_reward < float(markov_warm_release_reference_reward) - collapse_threshold
                    ):
                        cooldown = int(max(0, probation_cfg.get("cooldown_subepisodes", 0)))
                        if cooldown > 0:
                            markov_probation_cooldown_until_subepisode = max(
                                int(markov_probation_cooldown_until_subepisode),
                                subepisode_idx + cooldown,
                            )
                            markov_probation_trigger_count += 1
                            markov_td3_probation_trigger_log[i] = 1
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_rewards[-1],
                "| Hp,Hc:",
                tuple(horizon_trace[i, :]),
                "| h sg src:",
                horizon_sg_source_summary,
                "| avg z:",
                avg_markov_z_window,
                "| m exec src:",
                markov_exec_source_summary,
                "| m sg src:",
                markov_sg_source_summary,
                "| alpha:",
                matrix_alpha_log[i] if matrix_enabled else "off",
                "| avg weights:",
                avg_weight_window,
                "| w sg src:",
                weight_sg_source_summary,
                "| avg residual:",
                avg_residual_window,
                "| r sg src:",
                residual_sg_source_summary,
            )

    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if run_mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )
    for continuous_agent in (markov_agent, matrix_agent, weight_agent, residual_agent):
        if continuous_agent is not None and hasattr(continuous_agent, "flush_nstep"):
            continuous_agent.flush_nstep()
    post_warm_mask = np.arange(nFE) > int(warm_start_step)
    markov_post_warm_count = int(np.sum(post_warm_mask)) if markov_enabled else 0
    markov_source_post_warm = markov_action_source_log[post_warm_mask] if markov_post_warm_count else np.array([])
    markov_source_all_count = int(max(1, nFE))

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
            "markov": markov_enabled,
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
        "horizon_agent_kind": horizon_agent_kind,
        "horizon_sg_policy_action_log": horizon_sg_policy_action_log,
        "horizon_sg_supervisor_action_log": horizon_sg_supervisor_action_log,
        "horizon_sg_previous_action_log": horizon_sg_previous_action_log,
        "horizon_sg_selected_source_log": horizon_sg_selected_source_log,
        "horizon_sg_score_policy_log": horizon_sg_score_policy_log,
        "horizon_sg_score_supervisor_log": horizon_sg_score_supervisor_log,
        "horizon_sg_advantage_log": horizon_sg_advantage_log,
        "horizon_sg_q_policy_log": horizon_sg_q_policy_log,
        "horizon_sg_q_supervisor_log": horizon_sg_q_supervisor_log,
        "horizon_recipes": horizon_recipes if horizon_enabled else None,
        "combined_agent_mode": combined_cfg.get("combined_agent_mode"),
        "markov_basis_family": markov_cfg.get("basis_family", "io_pair_gain"),
        "markov_basis_labels": list(markov_basis_labels),
        "markov_state_dim": int(markov_state.size) if markov_enabled and "markov" in current_states else None,
        "markov_z_bound": float(markov_z_bound) if markov_enabled else None,
        "markov_z_safety": dict(markov_cfg.get("z_safety", {})) if markov_enabled else None,
        "markov_td3_priority_fallback": dict(markov_cfg.get("td3_priority_fallback", {})) if markov_enabled else None,
        "markov_z_log": markov_z_log,
        "markov_z_proposed_log": markov_z_proposed_log,
        "markov_z_executed_log": markov_z_executed_log,
        "markov_requested_z_log": markov_requested_z_log,
        "markov_requested_z_uncapped_log": markov_requested_z_uncapped_log,
        "markov_ls_z_log": markov_ls_z_log,
        "markov_ls_z_uncapped_log": markov_ls_z_uncapped_log,
        "markov_actor_raw_action_log": markov_actor_raw_action_log,
        "markov_requested_raw_action_log": markov_requested_raw_action_log,
        "markov_executed_raw_action_log": markov_executed_raw_action_log,
        "markov_decision_log": markov_decision_log,
        "markov_action_source_log": markov_action_source_log,
        "markov_policy_source_log": markov_policy_source_log,
        "markov_replay_pushed_log": markov_replay_pushed_log,
        "markov_train_called_log": markov_train_called_log,
        "markov_train_updated_log": markov_train_updated_log,
        "markov_prediction_score_log": markov_prediction_score_log,
        "markov_ls_prediction_score_log": markov_ls_prediction_score_log,
        "markov_requested_prediction_score_log": markov_requested_prediction_score_log,
        "markov_gain_drift_log": markov_gain_drift_log,
        "markov_ls_gain_drift_log": markov_ls_gain_drift_log,
        "markov_requested_gain_drift_log": markov_requested_gain_drift_log,
        "markov_accepted_log": markov_accepted_log,
        "markov_fallback_log": markov_fallback_log,
        "markov_nominal_cost_log": markov_nominal_cost_log,
        "markov_executed_cost_margin_log": markov_executed_cost_margin_log,
        "markov_td3_priority_phase_log": markov_td3_priority_phase_log,
        "markov_td3_authority_scale_log": markov_td3_authority_scale_log,
        "markov_td3_probation_active_log": markov_td3_probation_active_log,
        "markov_td3_probation_trigger_log": markov_td3_probation_trigger_log,
        "markov_td3_probation_trigger_count": int(markov_probation_trigger_count),
        "markov_sg_policy_action_raw_log": markov_sg_policy_action_raw_log,
        "markov_sg_supervisor_action_raw_log": markov_sg_supervisor_action_raw_log,
        "markov_sg_previous_action_raw_log": markov_sg_previous_action_raw_log,
        "markov_sg_selected_source_log": markov_sg_selected_source_log,
        "markov_sg_score_policy_log": markov_sg_score_policy_log,
        "markov_sg_score_supervisor_log": markov_sg_score_supervisor_log,
        "markov_sg_advantage_log": markov_sg_advantage_log,
        "markov_td3_source_fraction": float(np.mean(markov_action_source_log == 2)) if markov_enabled else 0.0,
        "markov_ls_fallback_source_fraction": float(np.mean(markov_action_source_log == 3)) if markov_enabled else 0.0,
        "markov_nominal_fallback_source_fraction": float(np.mean(markov_action_source_log == 4)) if markov_enabled else 0.0,
        "markov_sg_ls_supervisor_source_fraction": float(np.mean(markov_action_source_log == 6)) if markov_enabled else 0.0,
        "markov_sg_mpc_supervisor_source_fraction": float(np.mean(markov_action_source_log == 7)) if markov_enabled else 0.0,
        "markov_sg_solver_fallback_source_fraction": float(np.mean(markov_action_source_log == 8)) if markov_enabled else 0.0,
        "markov_td3_source_fraction_post_warm": (
            float(np.mean(markov_source_post_warm == 2)) if markov_post_warm_count else 0.0
        ),
        "markov_ls_fallback_source_fraction_post_warm": (
            float(np.mean(markov_source_post_warm == 3)) if markov_post_warm_count else 0.0
        ),
        "markov_nominal_fallback_source_fraction_post_warm": (
            float(np.mean(markov_source_post_warm == 4)) if markov_post_warm_count else 0.0
        ),
        "markov_ls_source_fraction": float(
            np.sum(np.isin(markov_action_source_log, [1, 3, 5])) / markov_source_all_count
        )
        if markov_enabled
        else 0.0,
        **markov_z_safety_logs,
        "matrix_alpha_log": matrix_alpha_log,
        "matrix_delta_log": matrix_delta_log,
        "matrix_decision_log": matrix_decision_log,
        "matrix_agent_kind": matrix_agent_kind,
        "matrix_state_mode": matrix_state_mode,
        "matrix_low_coef": model_low,
        "matrix_high_coef": model_high,
        "matrix_policy_multiplier_log": matrix_policy_multiplier_log,
        "matrix_candidate_multiplier_log": matrix_candidate_multiplier_log,
        "matrix_executed_multiplier_log": matrix_executed_multiplier_log,
        "matrix_candidate_action_raw_log": matrix_candidate_action_raw_log,
        "matrix_final_executed_action_raw_log": matrix_final_executed_action_raw_log,
        "matrix_release_schedule": matrix_release_schedule,
        "matrix_release_guard_enabled": bool(matrix_release_schedule.get("enabled", False)),
        "matrix_release_phase_log": matrix_release_phase_log,
        "matrix_release_guard_active_log": matrix_release_guard_active_log,
        "matrix_release_clip_fraction_log": matrix_release_clip_fraction_log,
        "matrix_release_ramp_fraction_log": matrix_release_ramp_fraction_log,
        "matrix_release_effective_low_log": matrix_release_effective_low_log,
        "matrix_release_effective_high_log": matrix_release_effective_high_log,
        "matrix_release_policy_action_raw_log": matrix_release_policy_action_raw_log,
        "matrix_release_executed_action_raw_log": matrix_release_executed_action_raw_log,
        "matrix_release_store_executed_action_in_replay": matrix_release_store_executed_action,
        "weight_log": weight_log,
        "weight_decision_log": weight_decision_log,
        "weight_agent_kind": weight_agent_kind,
        "weight_state_mode": weight_state_mode,
        "weight_low_coef": weights_low,
        "weight_high_coef": weights_high,
        "weight_safety": dict(weight_safety_cfg),
        "weight_safety_enabled": bool(weight_safety_enabled),
        "weight_fallback_to_identity_on_nonfinite": bool(fallback_to_identity_on_nonfinite),
        "weight_sg_policy_action_raw_log": weight_sg_policy_action_raw_log,
        "weight_sg_supervisor_action_raw_log": weight_sg_supervisor_action_raw_log,
        "weight_sg_previous_action_raw_log": weight_sg_previous_action_raw_log,
        "weight_sg_selected_source_log": weight_sg_selected_source_log,
        "weight_sg_score_policy_log": weight_sg_score_policy_log,
        "weight_sg_score_supervisor_log": weight_sg_score_supervisor_log,
        "weight_sg_advantage_log": weight_sg_advantage_log,
        "a_res_raw_log": a_res_raw_log,
        "a_res_exec_log": a_res_exec_log,
        "delta_u_res_raw_log": delta_u_res_raw_log,
        "delta_u_res_exec_log": delta_u_res_exec_log,
        "residual_raw_log": delta_u_res_raw_log,
        "residual_exec_log": delta_u_res_exec_log,
        "residual_decision_log": residual_decision_log,
        "residual_agent_kind": residual_agent_kind,
        "residual_state_mode": residual_state_mode,
        "residual_authority_enabled": bool(residual_authority_enabled),
        "residual_low_coef": residual_low,
        "residual_high_coef": residual_high,
        "residual_safety": dict(residual_safety_cfg),
        "residual_safety_enabled": bool(residual_safety_enabled),
        "residual_fallback_to_zero_on_nonfinite": bool(fallback_to_zero_on_nonfinite),
        "residual_sg_policy_action_raw_log": residual_sg_policy_action_raw_log,
        "residual_sg_supervisor_action_raw_log": residual_sg_supervisor_action_raw_log,
        "residual_sg_previous_action_raw_log": residual_sg_previous_action_raw_log,
        "residual_sg_selected_source_log": residual_sg_selected_source_log,
        "residual_sg_score_policy_log": residual_sg_score_policy_log,
        "residual_sg_score_supervisor_log": residual_sg_score_supervisor_log,
        "residual_sg_advantage_log": residual_sg_advantage_log,
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
        "markov_innovation_log": mismatch_logs["markov"]["innovation"],
        "markov_innovation_raw_log": mismatch_logs["markov"]["innovation_raw"],
        "markov_tracking_error_log": mismatch_logs["markov"]["tracking_error"],
        "markov_tracking_error_raw_log": mismatch_logs["markov"]["tracking_error_raw"],
        "markov_tracking_scale_log": mismatch_logs["markov"]["tracking_scale"],
        "markov_innovation_scale_ref": mismatch_cfgs["markov"]["innovation_scale_ref"],
        "markov_band_ref_scaled": mismatch_cfgs["markov"]["band_ref_scaled"],
        "markov_mismatch_clip": mismatch_cfgs["markov"]["mismatch_clip"],
        "markov_base_state_norm_mode": mismatch_cfgs["markov"]["base_state_norm_mode"],
        "markov_base_state_norm_stats": state_conditioners["markov"].export_state(),
        "markov_mismatch_feature_transform_mode": mismatch_cfgs["markov"]["mismatch_feature_transform_mode"],
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
    result_bundle.update(_extract_losses(markov_agent, "markov"))
    result_bundle.update(_extract_losses(matrix_agent, "matrix"))
    result_bundle.update(_extract_losses(weight_agent, "weight"))
    result_bundle.update(_extract_losses(residual_agent, "residual"))
    if markov_phase1 is not None:
        result_bundle.update(
            build_phase1_bundle_fields(
                markov_phase1,
                policy_action_raw_log=markov_policy_action_raw_log,
                executed_action_raw_log=markov_phase1_executed_action_raw_log,
                action_source_log=markov_phase1_action_source_log,
                traces=markov_phase1_train_traces,
                prefix="markov_",
            )
        )
    if matrix_phase1 is not None:
        result_bundle.update(
            build_phase1_bundle_fields(
                matrix_phase1,
                policy_action_raw_log=matrix_policy_action_raw_log,
                executed_action_raw_log=matrix_executed_action_raw_log,
                action_source_log=matrix_phase1_action_source_log,
                traces=matrix_phase1_train_traces,
                prefix="matrix_",
            )
        )
    if weight_phase1 is not None:
        result_bundle.update(
            build_phase1_bundle_fields(
                weight_phase1,
                policy_action_raw_log=weight_policy_action_raw_log,
                executed_action_raw_log=weight_executed_action_raw_log,
                action_source_log=weight_phase1_action_source_log,
                traces=weight_phase1_train_traces,
                prefix="weight_",
            )
        )
    if residual_phase1 is not None:
        result_bundle.update(
            build_phase1_bundle_fields(
                residual_phase1,
                policy_action_raw_log=residual_policy_action_raw_log,
                executed_action_raw_log=residual_executed_action_raw_log,
                action_source_log=residual_phase1_action_source_log,
                traces=residual_phase1_train_traces,
                prefix="residual_",
            )
        )
    replay_snapshots = capture_named_agent_replay_snapshots(
        {
            "horizon": horizon_agent,
            "markov": markov_agent,
            "matrix": matrix_agent,
            "weights": weight_agent,
            "residual": residual_agent,
        }
    )
    if replay_snapshots:
        result_bundle["replay_buffer_snapshots"] = replay_snapshots

    return result_bundle
