import numpy as np
import scipy.optimize as spo

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
from utils.observer import compute_observer_gain
from utils.observation_conditioning import update_observer_state
from utils.phase1_hidden_release import (
    build_phase1_bundle_fields,
    build_phase1_schedule,
    init_phase1_train_traces,
)
from utils.replay_snapshot import attach_single_agent_replay_snapshot
from utils.residual_authority import compute_residual_rho, map_from_bounds, map_to_bounds, project_residual_action
from utils.state_features import (
    build_rl_state,
    compute_tracking_scale_now,
    make_state_conditioner_from_settings,
    resolve_mismatch_settings,
)
from utils.td3_authority_ramp import (
    apply_symmetric_deviation_cap,
    build_td3_authority_ramp_bundle_fields,
    init_td3_authority_ramp_logs,
    record_td3_authority_ramp_step,
    resolve_td3_authority_ramp,
)


def _float_or_nan(value):
    if value is None:
        return float("nan")
    return float(value)


RESIDUAL_ACTION_SOURCE_CODES = {
    "warm_zero": 0,
    "td3_accepted": 1,
    "projected_td3": 2,
    "zero_fallback": 3,
}

RESIDUAL_ZERO_FALLBACK_REASON_CODES = {
    "none": 0,
    "nonfinite_selected_action": 1,
    "nonfinite_projected_action": 2,
}


def _projection_is_finite(projection):
    for key in ("a_exec", "delta_u_res_exec", "u_applied_scaled_abs"):
        value = np.asarray(projection.get(key), float)
        if not np.all(np.isfinite(value)):
            return False
    return True


def run_residual_supervisor(residual_cfg, runtime_ctx):
    """
    Run the TD3/SAC residual-correction supervisor and return a normalized result bundle.

    Parameters
    ----------
    residual_cfg : dict
        Runtime config assembled in the notebook.
    runtime_ctx : dict
        Prepared objects and shared data assembled in the notebook.
    """

    system = runtime_ctx["system"]
    agent = runtime_ctx["agent"]
    mpc_obj = runtime_ctx["MPC_obj"]
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
    system_stepper = runtime_ctx.get("system_stepper")
    system_metadata = runtime_ctx.get("system_metadata")
    disturbance_labels = runtime_ctx.get("disturbance_labels")

    agent_kind = str(residual_cfg["agent_kind"]).lower()
    run_mode = str(residual_cfg["run_mode"]).lower()
    state_mode = str(residual_cfg.get("state_mode", "standard")).lower()
    residual_authority_enabled = bool(residual_cfg.get("residual_authority_enabled", state_mode == "mismatch"))
    authority_use_rho = bool(residual_cfg.get("authority_use_rho", residual_cfg.get("use_rho_authority", True)))
    residual_safety_cfg = dict(residual_cfg.get("residual_safety", {}) or {})
    residual_safety_enabled = bool(residual_safety_cfg.get("enabled", False))
    reward_probation_cfg = dict(residual_safety_cfg.get("reward_probation", {}) or {})
    reward_probation_enabled = bool(residual_safety_enabled and reward_probation_cfg.get("enabled", False))
    probation_reference_warm_episodes = int(max(1, reward_probation_cfg.get("reference_warm_episodes", 3)))
    probation_collapse_threshold = float(reward_probation_cfg.get("collapse_threshold", 5.0))
    probation_cooldown_subepisodes = int(max(0, reward_probation_cfg.get("cooldown_subepisodes", 0)))
    probation_cooldown_residual_cap = float(reward_probation_cfg.get("cooldown_residual_cap", 0.005))
    fallback_to_zero_on_nonfinite = bool(
        residual_safety_enabled and residual_safety_cfg.get("fallback_to_zero_on_nonfinite", False)
    )
    shadow_rho_authority_enabled = bool(
        residual_safety_enabled
        and state_mode == "mismatch"
        and dict(residual_safety_cfg.get("shadow_rho_authority", {}) or {}).get("enabled", False)
    )
    shadow_residual_deadband_enabled = bool(
        residual_safety_enabled
        and dict(residual_safety_cfg.get("shadow_residual_deadband", {}) or {}).get("enabled", False)
    )
    shadow_direction_risk_enabled = bool(
        residual_safety_enabled
        and dict(residual_safety_cfg.get("shadow_direction_risk", {}) or {}).get("enabled", False)
    )
    if agent_kind not in {"td3", "sac"}:
        raise ValueError("residual_cfg['agent_kind'] must be 'td3' or 'sac'.")
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("residual_cfg['run_mode'] must be 'nominal' or 'disturb'.")
    use_shifted_mpc_warm_start = bool(residual_cfg.get("use_shifted_mpc_warm_start", False))
    mismatch_seed_cfg = dict(residual_cfg)
    mismatch_seed_cfg.setdefault("tracking_eta_tol", residual_cfg.get("authority_eta_tol", 0.3))
    mismatch_cfg = resolve_mismatch_settings(
        state_mode=state_mode,
        mismatch_cfg=mismatch_seed_cfg,
        reward_params=runtime_ctx.get("reward_params", {}),
        y_sp_scenario=y_sp_scenario,
        steady_states=steady_states,
        data_min=data_min,
        data_max=data_max,
        n_inputs=B_aug.shape[1],
    )
    mismatch_clip = mismatch_cfg["mismatch_clip"]
    state_conditioner = make_state_conditioner_from_settings(mismatch_cfg)
    observer_update_alignment = (
        mismatch_cfg["observer_update_alignment"] if state_mode == "mismatch" else "legacy_previous_measurement"
    )

    low_coef = np.asarray(residual_cfg["low_coef"], float).reshape(-1)
    high_coef = np.asarray(residual_cfg["high_coef"], float).reshape(-1)
    action_dim = int(B_aug.shape[1])
    if low_coef.size != action_dim or high_coef.size != action_dim:
        raise ValueError("low_coef/high_coef must match the number of manipulated inputs.")
    if np.any(low_coef > 0.0) or np.any(high_coef < 0.0):
        raise ValueError("Residual bounds must bracket zero so warm start can apply zero correction.")

    zero_action = map_from_bounds(np.zeros(action_dim, dtype=float), low_coef, high_coef)

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
        int(residual_cfg["n_tests"]),
        int(residual_cfg["set_points_len"]),
        int(residual_cfg["warm_start"]),
        list(residual_cfg["test_cycle"]),
        float(residual_cfg["nominal_qi"]),
        float(residual_cfg["nominal_qs"]),
        float(residual_cfg["nominal_ha"]),
        float(residual_cfg["qi_change"]),
        float(residual_cfg["qs_change"]),
        float(residual_cfg["ha_change"]),
    )

    disturbance_schedule = None
    if run_mode == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    bc_schedule = build_behavioral_cloning_schedule(
        config=residual_cfg.get("behavioral_cloning", {}),
        warm_start_step=warm_start_step,
        time_in_sub_episodes=time_in_sub_episodes,
        n_steps=nFE,
    )
    bc_logs = init_behavioral_cloning_logs(nFE)
    bc_handoff_logs = init_bc_handoff_logs(nFE, action_dim)
    bc_release_gate = init_protected_bc_release_gate(bc_schedule, nFE)
    bc_handoff_enabled = bool(agent_kind == "td3" and dict(bc_schedule.get("handoff", {}) or {}).get("enabled", False))
    bc_action_gap_tolerance = float(bc_schedule.get("action_gap_tolerance", 0.0))
    bc_target_mode = str(bc_schedule.get("target_mode", "nominal_only")).strip().lower()
    bc_train_start_step = (
        int(bc_schedule.get("start_step", warm_start_step))
        if bool(bc_schedule.get("enabled", False))
        else int(warm_start_step)
    )
    protected_bc_release_enabled = bool(bc_release_gate["state"].get("live_blocking_enabled", False))
    td3_authority_ramp_cfg = dict(residual_cfg.get("td3_authority_ramp", {}) or {}) if agent_kind == "td3" else {}
    td3_authority_ramp_logs = init_td3_authority_ramp_logs(nFE, action_dim)

    phase1 = None
    phase1_action_source_log = None
    phase1_train_traces = None
    if agent_kind == "td3":
        phase1 = build_phase1_schedule(
            agent_kind=agent_kind,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            n_steps=nFE,
            test_train_dict=test_train_dict,
            action_freeze_subepisodes=0
            if (protected_bc_release_enabled or bc_handoff_enabled)
            else residual_cfg.get("post_warm_start_action_freeze_subepisodes", 0),
            actor_freeze_subepisodes=0
            if (protected_bc_release_enabled or bc_handoff_enabled)
            else residual_cfg.get("post_warm_start_actor_freeze_subepisodes", 0),
            batch_size=getattr(agent, "batch_size", 1),
            initial_buffer_size=len(getattr(agent, "buffer", [])),
            base_actor_freeze=getattr(agent, "actor_freeze", 0),
            push_start_step=0,
            train_start_step=bc_train_start_step,
        )
        agent.actor_freeze = int(phase1["effective_actor_freeze"])
        phase1_action_source_log = np.zeros(nFE, dtype=int)
        phase1_train_traces = init_phase1_train_traces()
    policy_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    executed_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    policy_executed_gap_norm_log = np.zeros(nFE, dtype=float)
    residual_raw_executed_norm_ratio_log = np.full(nFE, np.nan, dtype=float)

    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug.shape[0])

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    u_min_scaled_abs = np.asarray(residual_cfg["b_min"], float) + ss_scaled_inputs
    u_max_scaled_abs = np.asarray(residual_cfg["b_max"], float) + ss_scaled_inputs
    L = compute_observer_gain(mpc_obj.A, mpc_obj.C, poles)
    reward_params = runtime_ctx.get("reward_params", {})
    authority_beta_res = np.asarray(
        residual_cfg.get("authority_beta_res", np.full(action_dim, 0.5, dtype=float)),
        float,
    ).reshape(-1)
    authority_du0_res = np.asarray(
        residual_cfg.get("authority_du0_res", np.full(action_dim, 0.001, dtype=float)),
        float,
    ).reshape(-1)
    if authority_beta_res.size != action_dim or authority_du0_res.size != action_dim:
        raise ValueError("authority_beta_res and authority_du0_res must match the number of manipulated inputs.")
    authority_rho_floor = float(residual_cfg.get("authority_rho_floor", 0.15))
    authority_rho_power = float(residual_cfg.get("authority_rho_power", 1.0))
    rho_mapping_mode = str(residual_cfg.get("rho_mapping_mode", "clipped_linear")).strip().lower()
    authority_rho_k = float(residual_cfg.get("authority_rho_k", 0.55))
    residual_zero_deadband_enabled = bool(residual_cfg.get("residual_zero_deadband_enabled", False))
    residual_zero_tracking_raw_threshold = float(residual_cfg.get("residual_zero_tracking_raw_threshold", 0.1))
    residual_zero_innovation_raw_threshold = float(residual_cfg.get("residual_zero_innovation_raw_threshold", 0.1))
    append_rho_to_state = bool(residual_cfg.get("append_rho_to_state", True))

    cont_h = int(residual_cfg.get("cont_h", 1))
    bounds = tuple(
        (float(residual_cfg["b_min"][j]), float(residual_cfg["b_max"][j]))
        for _ in range(cont_h)
        for j in range(n_inputs)
    )
    ic_opt = np.zeros(n_inputs * cont_h)

    y_system = np.zeros((nFE + 1, n_outputs))
    y_system[0, :] = np.asarray(system.current_output, float)
    u_rl_scaled = np.zeros((nFE, n_inputs))
    u_base_scaled = np.zeros((nFE, n_inputs))
    rewards = np.zeros(nFE)
    avg_rewards = []
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    delta_y_storage = np.zeros((nFE, n_outputs))
    delta_u_storage = np.zeros((nFE, n_inputs))
    a_res_raw_log = np.zeros((nFE, n_inputs), dtype=float)
    a_res_exec_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_raw_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_exec_log = np.zeros((nFE, n_inputs), dtype=float)
    rho_log = np.full(nFE, np.nan) if state_mode == "mismatch" else None
    rho_raw_log = np.full(nFE, np.nan) if state_mode == "mismatch" else None
    rho_eff_log = np.full(nFE, np.nan) if state_mode == "mismatch" else None
    innovation_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    innovation_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_scale_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    deadband_active_log = np.zeros(nFE, dtype=int)
    projection_active_log = np.zeros(nFE, dtype=int)
    projection_due_to_deadband_log = np.zeros(nFE, dtype=int)
    projection_due_to_authority_log = np.zeros(nFE, dtype=int)
    projection_due_to_headroom_log = np.zeros(nFE, dtype=int)
    residual_action_source_log = np.zeros(nFE, dtype=int)
    residual_zero_fallback_reason_log = np.zeros(nFE, dtype=int)
    residual_active_cap_log = np.full(nFE, np.nan, dtype=float)
    residual_cap_projection_active_log = np.zeros(nFE, dtype=int)
    residual_cap_projection_norm_log = np.full(nFE, np.nan, dtype=float)
    residual_probation_active_log = np.zeros(nFE, dtype=int)
    residual_probation_trigger_log = np.zeros(nFE, dtype=int)
    residual_probation_reference_reward_log = np.full(nFE, np.nan, dtype=float)
    residual_probation_cooldown_until_subepisode_log = np.zeros(nFE, dtype=int)
    residual_requested_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    residual_post_handoff_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    residual_post_cap_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    delta_u_res_requested_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_post_handoff_log = np.zeros((nFE, n_inputs), dtype=float)
    delta_u_res_post_cap_log = np.zeros((nFE, n_inputs), dtype=float)
    shadow_rho_log = np.full(nFE, np.nan, dtype=float)
    shadow_rho_raw_log = np.full(nFE, np.nan, dtype=float)
    shadow_rho_eff_log = np.full(nFE, np.nan, dtype=float)
    shadow_rho_projection_active_log = np.zeros(nFE, dtype=int)
    shadow_rho_projection_due_to_deadband_log = np.zeros(nFE, dtype=int)
    shadow_rho_projection_due_to_authority_log = np.zeros(nFE, dtype=int)
    shadow_rho_projection_due_to_headroom_log = np.zeros(nFE, dtype=int)
    shadow_rho_deadband_active_log = np.zeros(nFE, dtype=int)
    shadow_rho_exec_diff_norm_log = np.full(nFE, np.nan, dtype=float)
    shadow_rho_delta_u_res_exec_log = np.full((nFE, n_inputs), np.nan, dtype=float)
    shadow_rho_a_res_exec_log = np.full((nFE, action_dim), np.nan, dtype=float)
    residual_predicted_direction_risk_log = np.full(nFE, np.nan, dtype=float)
    residual_predicted_nominal_error_norm_log = np.full(nFE, np.nan, dtype=float)
    residual_predicted_candidate_error_norm_log = np.full(nFE, np.nan, dtype=float)
    warm_reference_rewards = []
    warm_start_subepisodes = int(np.ceil(float(warm_start_step + 1) / float(max(1, time_in_sub_episodes))))
    probation_cooldown_until_subepisode = 0
    test = False

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_pred = mpc_obj.C @ xhatdhat[:, i]
        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        tracking_scale_now = None
        rho_state = None
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
            rho_state = float(
                compute_residual_rho(
                    tracking_values=(y_prev_scaled - y_sp[i, :]) / np.maximum(tracking_scale_now, 1e-12),
                    rho_mapping_mode=rho_mapping_mode,
                    authority_rho_k=authority_rho_k,
                )["rho"]
            )
        current_rl_state, state_debug = build_rl_state(
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
            append_rho_to_state=bool(state_mode == "mismatch" and append_rho_to_state),
            rho_value=rho_state,
            state_conditioner=state_conditioner,
            update_state_conditioner=True,
            mismatch_feature_transform_mode=mismatch_cfg["mismatch_feature_transform_mode"],
            mismatch_transform_tanh_scale=mismatch_cfg["mismatch_transform_tanh_scale"],
            mismatch_transform_post_clip=mismatch_cfg["mismatch_transform_post_clip"],
        )
        if innovation_log is not None:
            innovation_log[i, :] = state_debug["innovation"]
            innovation_raw_log[i, :] = state_debug["innovation_raw"]
            tracking_error_log[i, :] = state_debug["tracking_error"]
            tracking_error_raw_log[i, :] = state_debug["tracking_error_raw"]
            tracking_scale_log[i, :] = state_debug["tracking_scale_now"]

        policy_action_for_log = zero_action.copy()
        if agent_kind == "td3":
            policy_action_for_log = np.asarray(agent.act_eval(current_rl_state), float).reshape(-1)
            if policy_action_for_log.size != action_dim or not np.all(np.isfinite(policy_action_for_log)):
                policy_action_for_log = zero_action.copy()
        release_info = update_protected_bc_release_gate(
            bc_release_gate["state"],
            bc_release_gate["logs"],
            step_idx=i,
            warm_start_step=warm_start_step,
            policy_action=policy_action_for_log,
            target_action=zero_action,
        )

        current_subepisode = int(i // time_in_sub_episodes) + 1
        if warm_reference_rewards:
            reference_window = warm_reference_rewards[-probation_reference_warm_episodes:]
            residual_probation_reference_reward_log[i] = float(np.mean(reference_window))
        probation_active = bool(
            reward_probation_enabled
            and current_subepisode > warm_start_subepisodes
            and current_subepisode <= probation_cooldown_until_subepisode
        )
        residual_probation_active_log[i] = int(probation_active)
        residual_probation_cooldown_until_subepisode_log[i] = int(probation_cooldown_until_subepisode)

        action_decision = select_continuous_action(
            agent=agent,
            state=current_rl_state,
            step=i,
            warm_start_step=-1 if bc_handoff_enabled else warm_start_step,
            test=test,
            baseline_action=zero_action,
            phase1=phase1,
            action_dim=action_dim,
            nonfinite_fallback=fallback_to_zero_on_nonfinite,
        )
        action_requested = np.asarray(action_decision.action, float).reshape(-1)
        residual_requested_action_raw_log[i, :] = action_requested
        delta_u_res_requested_log[i, :] = map_to_bounds(action_requested, low_coef, high_coef).reshape(-1)

        ramp_info = resolve_td3_authority_ramp(
            td3_authority_ramp_cfg,
            step_idx=i,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
        )
        live_gate_blocked = bool(release_info.get("live_blocked", False))
        gate_override = bool(live_gate_blocked and ramp_info["live_enabled"])
        action_for_handoff = zero_action.copy() if (live_gate_blocked and not ramp_info["live_enabled"]) else action_requested

        handoff_info = resolve_bc_handoff_authority(bc_schedule, step_idx=i)
        handoff_td3_action = np.asarray(action_for_handoff, float).reshape(-1)
        action_post_handoff = apply_bc_handoff_action(
            handoff_td3_action,
            zero_action,
            handoff_info["authority"],
        )
        record_bc_handoff_step(
            bc_handoff_logs,
            step_idx=i,
            authority_info=handoff_info,
            safe_action=zero_action,
            td3_action=handoff_td3_action,
            executed_action=action_post_handoff,
        )
        residual_post_handoff_action_raw_log[i, :] = action_post_handoff
        residual_post_handoff = map_to_bounds(action_post_handoff, low_coef, high_coef).reshape(-1)
        delta_u_res_post_handoff_log[i, :] = residual_post_handoff

        active_cap = float(ramp_info["cap"]) if bool(ramp_info["live_enabled"]) else float("nan")
        cap_active = bool(ramp_info["live_enabled"])
        if probation_active:
            probation_cap = max(0.0, probation_cooldown_residual_cap)
            active_cap = float(min(active_cap, probation_cap)) if np.isfinite(active_cap) else float(probation_cap)
            cap_active = True
        residual_active_cap_log[i] = active_cap if cap_active else float("nan")
        residual_post_cap, ramp_clip_info = apply_symmetric_deviation_cap(
            residual_post_handoff,
            center=np.zeros(action_dim, dtype=float),
            cap=active_cap,
            low=low_coef,
            high=high_coef,
            active=cap_active,
        )
        action_post_cap = np.clip(map_from_bounds(residual_post_cap, low_coef, high_coef), -1.0, 1.0)
        residual_cap_projection_active_log[i] = int(ramp_clip_info["projection_active"])
        residual_cap_projection_norm_log[i] = float(ramp_clip_info["delta_norm"])
        residual_post_cap_action_raw_log[i, :] = action_post_cap
        delta_u_res_post_cap_log[i, :] = residual_post_cap
        record_td3_authority_ramp_step(
            td3_authority_ramp_logs,
            step_idx=i,
            ramp_info={
                **dict(ramp_info),
                "cap": active_cap,
                "live_enabled": bool(cap_active),
            },
            preclip_action=action_post_handoff,
            postclip_action=action_post_cap,
            projection_active=ramp_clip_info["projection_active"],
            delta_norm=ramp_clip_info["delta_norm"],
            gate_override=gate_override,
        )

        action = action_post_cap
        a_res_raw_log[i, :] = np.asarray(action, float).reshape(-1)
        policy_action_raw_log[i, :] = policy_action_for_log

        ic_opt_step = ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * cont_h)

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            ic_opt_step,
            bounds=bounds,
            constraints=[],
        )
        if use_shifted_mpc_warm_start:
            ic_opt = shift_control_sequence(sol.x[: n_inputs * cont_h], n_inputs, cont_h)
        else:
            ic_opt = np.zeros(n_inputs * cont_h)

        u_base = np.asarray(sol.x[:n_inputs], float) + ss_scaled_inputs
        u_base = np.clip(u_base, u_min_scaled_abs, u_max_scaled_abs)
        u_base_scaled[i, :] = u_base

        projection = project_residual_action(
            action_raw=action,
            low_coef=low_coef,
            high_coef=high_coef,
            u_base=u_base,
            scaled_current_input=scaled_current_input,
            u_min_scaled_abs=u_min_scaled_abs,
            u_max_scaled_abs=u_max_scaled_abs,
            apply_authority=bool(state_mode == "mismatch" and residual_authority_enabled),
            authority_use_rho=authority_use_rho,
            tracking_error_feat=state_debug["tracking_error"],
            tracking_error_raw=state_debug["tracking_error_raw"],
            innovation_raw=state_debug["innovation_raw"],
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
        fallback_reason_code = RESIDUAL_ZERO_FALLBACK_REASON_CODES["none"]
        if bool(action_decision.nonfinite_fallback_used):
            fallback_reason_code = RESIDUAL_ZERO_FALLBACK_REASON_CODES["nonfinite_selected_action"]
        if fallback_to_zero_on_nonfinite and not _projection_is_finite(projection):
            projection = project_residual_action(
                action_raw=zero_action,
                low_coef=low_coef,
                high_coef=high_coef,
                u_base=u_base,
                scaled_current_input=scaled_current_input,
                u_min_scaled_abs=u_min_scaled_abs,
                u_max_scaled_abs=u_max_scaled_abs,
                apply_authority=False,
                authority_use_rho=False,
            )
            action = zero_action.copy()
            fallback_reason_code = RESIDUAL_ZERO_FALLBACK_REASON_CODES["nonfinite_projected_action"]
        residual_zero_fallback_reason_log[i] = int(fallback_reason_code)

        if shadow_rho_authority_enabled:
            shadow_projection = project_residual_action(
                action_raw=action,
                low_coef=low_coef,
                high_coef=high_coef,
                u_base=u_base,
                scaled_current_input=scaled_current_input,
                u_min_scaled_abs=u_min_scaled_abs,
                u_max_scaled_abs=u_max_scaled_abs,
                apply_authority=True,
                authority_use_rho=True,
                tracking_error_feat=state_debug["tracking_error"],
                tracking_error_raw=state_debug["tracking_error_raw"],
                innovation_raw=state_debug["innovation_raw"],
                authority_beta_res=authority_beta_res,
                authority_du0_res=authority_du0_res,
                authority_rho_floor=authority_rho_floor,
                authority_rho_power=authority_rho_power,
                rho_mapping_mode=rho_mapping_mode,
                authority_rho_k=authority_rho_k,
                residual_zero_deadband_enabled=bool(
                    residual_zero_deadband_enabled and shadow_residual_deadband_enabled
                ),
                residual_zero_tracking_raw_threshold=residual_zero_tracking_raw_threshold,
                residual_zero_innovation_raw_threshold=residual_zero_innovation_raw_threshold,
            )
            shadow_rho_log[i] = _float_or_nan(shadow_projection["rho"])
            shadow_rho_raw_log[i] = _float_or_nan(shadow_projection["rho_raw"])
            shadow_rho_eff_log[i] = _float_or_nan(shadow_projection["rho_eff"])
            shadow_rho_projection_active_log[i] = int(shadow_projection["projection_active"])
            shadow_rho_projection_due_to_deadband_log[i] = int(shadow_projection["projection_due_to_deadband"])
            shadow_rho_projection_due_to_authority_log[i] = int(shadow_projection["projection_due_to_authority"])
            shadow_rho_projection_due_to_headroom_log[i] = int(shadow_projection["projection_due_to_headroom"])
            shadow_rho_deadband_active_log[i] = int(shadow_projection["deadband_active"])
            shadow_rho_delta_u_res_exec_log[i, :] = shadow_projection["delta_u_res_exec"]
            shadow_rho_a_res_exec_log[i, :] = shadow_projection["a_exec"]
            shadow_rho_exec_diff_norm_log[i] = float(
                np.linalg.norm(shadow_projection["delta_u_res_exec"] - projection["delta_u_res_exec"])
            )
        if rho_log is not None:
            rho_log[i] = _float_or_nan(projection["rho"])
            rho_raw_log[i] = _float_or_nan(projection["rho_raw"])
            rho_eff_log[i] = _float_or_nan(projection["rho_eff"])
        deadband_active_log[i] = int(projection["deadband_active"])
        projection_active_log[i] = int(projection["projection_active"])
        projection_due_to_deadband_log[i] = int(projection["projection_due_to_deadband"])
        projection_due_to_authority_log[i] = int(projection["projection_due_to_authority"])
        projection_due_to_headroom_log[i] = int(projection["projection_due_to_headroom"])
        delta_u_res_raw_log[i, :] = projection["delta_u_res_raw"]
        delta_u_res_exec_log[i, :] = projection["delta_u_res_exec"]
        a_res_exec_log[i, :] = projection["a_exec"]
        executed_action_raw_log[i, :] = np.asarray(projection["a_exec"], float).reshape(-1)
        policy_executed_gap_norm_log[i] = float(
            np.linalg.norm(policy_action_raw_log[i, :] - executed_action_raw_log[i, :])
        )
        raw_norm = float(np.linalg.norm(policy_action_raw_log[i, :]))
        exec_norm = float(np.linalg.norm(a_res_exec_log[i, :]))
        residual_raw_executed_norm_ratio_log[i] = exec_norm / max(raw_norm, 1.0e-12)
        if fallback_reason_code != RESIDUAL_ZERO_FALLBACK_REASON_CODES["none"] or (
            live_gate_blocked and not ramp_info["live_enabled"]
        ):
            residual_action_source_log[i] = RESIDUAL_ACTION_SOURCE_CODES["zero_fallback"]
        elif i <= warm_start_step and float(np.linalg.norm(delta_u_res_exec_log[i, :])) <= 1.0e-12:
            residual_action_source_log[i] = RESIDUAL_ACTION_SOURCE_CODES["warm_zero"]
        elif bool(ramp_clip_info["projection_active"]) or bool(projection["projection_active"]):
            residual_action_source_log[i] = RESIDUAL_ACTION_SOURCE_CODES["projected_td3"]
        else:
            residual_action_source_log[i] = RESIDUAL_ACTION_SOURCE_CODES["td3_accepted"]
        if phase1 is not None:
            hard_blocked = bool(live_gate_blocked) and not ramp_info["live_enabled"]
            phase1_action_source_log[i] = 1 if hard_blocked else int(action_decision.source)
        u_rl_scaled[i, :] = projection["u_applied_scaled_abs"]

        delta_u = u_rl_scaled[i, :] - scaled_current_input
        delta_u_storage[i, :] = delta_u
        action_exec = projection["a_exec"]

        u_plant = reverse_min_max(u_rl_scaled[i, :], data_min[:n_inputs], data_max[:n_inputs])
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

        xhatdhat[:, i + 1], yhat[:, i], observer_update_alignment = update_observer_state(
            A=mpc_obj.A,
            B=mpc_obj.B,
            C=mpc_obj.C,
            L=L,
            x_prev=xhatdhat[:, i],
            u_dev=(u_rl_scaled[i, :] - ss_scaled_inputs),
            y_prev_scaled=y_prev_scaled,
            y_current_scaled=y_current_scaled,
            observer_update_alignment=observer_update_alignment,
        )

        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_rl_scaled[i, :] - ss_scaled_inputs
        yhat_next_pred = mpc_obj.C @ xhatdhat[:, i + 1]
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
            next_rho_state = float(
                compute_residual_rho(
                    tracking_values=(y_current_scaled - y_sp[i, :]) / np.maximum(next_tracking_scale_now, 1e-12),
                    rho_mapping_mode=rho_mapping_mode,
                    authority_rho_k=authority_rho_k,
                )["rho"]
            )
        next_rl_state, _ = build_rl_state(
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
            append_rho_to_state=bool(state_mode == "mismatch" and append_rho_to_state),
            rho_value=next_rho_state,
            state_conditioner=state_conditioner,
            update_state_conditioner=False,
            mismatch_feature_transform_mode=mismatch_cfg["mismatch_feature_transform_mode"],
            mismatch_transform_tanh_scale=mismatch_cfg["mismatch_transform_tanh_scale"],
            mismatch_transform_post_clip=mismatch_cfg["mismatch_transform_post_clip"],
        )

        if bc_target_mode == "executed_action":
            bc_target_action = np.asarray(action_exec, float).reshape(-1)
        else:
            bc_target_action = zero_action.copy()
        bc_context = None
        if not test:
            if float(np.max(np.abs(policy_action_raw_log[i, :] - bc_target_action))) > bc_action_gap_tolerance:
                bc_context = resolve_behavioral_cloning_context(
                    bc_schedule,
                    step_idx=i,
                    target_action=bc_target_action,
                )

        train_result = replay_train_continuous_agent(
            agent=agent,
            state=current_rl_state,
            action=action_exec,
            reward=reward,
            next_state=next_rl_state,
            done=0.0,
            step=i,
            test=test,
            train_start_step=bc_train_start_step,
            phase1_train_traces=phase1_train_traces if phase1 is not None else None,
            bc_context=bc_context,
        )
        record_behavioral_cloning_step(
            bc_logs,
            step_idx=i,
            bc_context=bc_context,
            policy_action=policy_action_raw_log[i, :],
            target_action=bc_target_action,
            target_mode=bc_target_mode,
            train_meta=train_result.get("train_meta"),
        )

        if i in sub_episodes_changes_dict:
            avg_rewards.append(float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1])))
            completed_subepisode = int(i // time_in_sub_episodes) + 1
            if completed_subepisode <= warm_start_subepisodes:
                warm_reference_rewards.append(float(avg_rewards[-1]))
            elif reward_probation_enabled and warm_reference_rewards:
                reference_window = warm_reference_rewards[-probation_reference_warm_episodes:]
                reference_reward = float(np.mean(reference_window))
                collapsed = bool(float(avg_rewards[-1]) < reference_reward - probation_collapse_threshold)
                if collapsed and probation_cooldown_subepisodes > 0:
                    probation_cooldown_until_subepisode = max(
                        probation_cooldown_until_subepisode,
                        completed_subepisode + probation_cooldown_subepisodes,
                    )
                    residual_probation_trigger_log[i] = 1
                    residual_probation_reference_reward_log[i] = reference_reward
                    residual_probation_cooldown_until_subepisode_log[i] = int(
                        probation_cooldown_until_subepisode
                    )
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_rewards[-1],
                "| avg residual:",
                np.mean(delta_u_res_exec_log[max(0, i - time_in_sub_episodes + 1) : i + 1, :], axis=0),
            )

    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if run_mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )
    if hasattr(agent, "flush_nstep"):
        agent.flush_nstep()

    result_bundle = {
        "agent_kind": agent_kind,
        "run_mode": run_mode,
        "method_family": "residual",
        "algorithm": agent_kind,
        "state_mode": state_mode,
        "system_metadata": system_metadata,
        "authority_use_rho": authority_use_rho,
        "use_rho_authority": authority_use_rho,
        "residual_authority_enabled": residual_authority_enabled,
        "notebook_source": residual_cfg.get("notebook_source"),
        "config_snapshot": dict(residual_cfg),
        "seed": residual_cfg.get("seed"),
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": int(nFE),
        "delta_t": float(system.delta_t),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y": y_system,
        "u": reverse_min_max(u_rl_scaled, data_min[:n_inputs], data_max[:n_inputs]),
        "u_base": reverse_min_max(u_base_scaled, data_min[:n_inputs], data_max[:n_inputs]),
        "avg_rewards": np.asarray(avg_rewards, float),
        "rewards_step": rewards,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "data_min": data_min,
        "data_max": data_max,
        "yhat": yhat,
        "xhatdhat": xhatdhat,
        "a_res_raw_log": a_res_raw_log,
        "a_res_exec_log": a_res_exec_log,
        "delta_u_res_raw_log": delta_u_res_raw_log,
        "delta_u_res_exec_log": delta_u_res_exec_log,
        "residual_raw_log": delta_u_res_raw_log,
        "residual_exec_log": delta_u_res_exec_log,
        "residual_safety": dict(residual_safety_cfg),
        "residual_safety_enabled": bool(residual_safety_enabled),
        "residual_reward_probation_enabled": bool(reward_probation_enabled),
        "residual_fallback_to_zero_on_nonfinite": bool(fallback_to_zero_on_nonfinite),
        "residual_probation_reference_warm_episodes": int(probation_reference_warm_episodes),
        "residual_probation_collapse_threshold": float(probation_collapse_threshold),
        "residual_probation_cooldown_subepisodes": int(probation_cooldown_subepisodes),
        "residual_probation_cooldown_residual_cap": float(probation_cooldown_residual_cap),
        "residual_action_source_codes": dict(RESIDUAL_ACTION_SOURCE_CODES),
        "residual_action_source_log": residual_action_source_log,
        "residual_zero_fallback_reason_codes": dict(RESIDUAL_ZERO_FALLBACK_REASON_CODES),
        "residual_zero_fallback_reason_log": residual_zero_fallback_reason_log,
        "residual_active_cap_log": residual_active_cap_log,
        "residual_cap_projection_active_log": residual_cap_projection_active_log,
        "residual_cap_projection_norm_log": residual_cap_projection_norm_log,
        "residual_probation_active_log": residual_probation_active_log,
        "residual_probation_trigger_log": residual_probation_trigger_log,
        "residual_probation_reference_reward_log": residual_probation_reference_reward_log,
        "residual_probation_cooldown_until_subepisode_log": residual_probation_cooldown_until_subepisode_log,
        "residual_requested_action_raw_log": residual_requested_action_raw_log,
        "residual_post_handoff_action_raw_log": residual_post_handoff_action_raw_log,
        "residual_post_cap_action_raw_log": residual_post_cap_action_raw_log,
        "delta_u_res_requested_log": delta_u_res_requested_log,
        "delta_u_res_post_handoff_log": delta_u_res_post_handoff_log,
        "delta_u_res_post_cap_log": delta_u_res_post_cap_log,
        "policy_action_raw_log": policy_action_raw_log,
        "executed_action_raw_log": executed_action_raw_log,
        "policy_executed_gap_norm_log": policy_executed_gap_norm_log,
        "residual_raw_executed_norm_ratio_log": residual_raw_executed_norm_ratio_log,
        "residual_bc_target_gap_log": bc_logs["bc_policy_target_distance_log"],
        "rho_log": rho_log,
        "rho_raw_log": rho_raw_log,
        "rho_eff_log": rho_eff_log,
        "deadband_active_log": deadband_active_log,
        "projection_active_log": projection_active_log,
        "projection_due_to_deadband_log": projection_due_to_deadband_log,
        "projection_due_to_authority_log": projection_due_to_authority_log,
        "projection_due_to_headroom_log": projection_due_to_headroom_log,
        "shadow_rho_authority_enabled": bool(shadow_rho_authority_enabled),
        "shadow_residual_deadband_enabled": bool(shadow_residual_deadband_enabled),
        "shadow_rho_log": shadow_rho_log,
        "shadow_rho_raw_log": shadow_rho_raw_log,
        "shadow_rho_eff_log": shadow_rho_eff_log,
        "shadow_rho_projection_active_log": shadow_rho_projection_active_log,
        "shadow_rho_projection_due_to_deadband_log": shadow_rho_projection_due_to_deadband_log,
        "shadow_rho_projection_due_to_authority_log": shadow_rho_projection_due_to_authority_log,
        "shadow_rho_projection_due_to_headroom_log": shadow_rho_projection_due_to_headroom_log,
        "shadow_rho_deadband_active_log": shadow_rho_deadband_active_log,
        "shadow_rho_exec_diff_norm_log": shadow_rho_exec_diff_norm_log,
        "shadow_rho_delta_u_res_exec_log": shadow_rho_delta_u_res_exec_log,
        "shadow_rho_a_res_exec_log": shadow_rho_a_res_exec_log,
        "shadow_direction_risk_enabled": bool(shadow_direction_risk_enabled),
        "residual_predicted_direction_risk_log": residual_predicted_direction_risk_log,
        "residual_predicted_nominal_error_norm_log": residual_predicted_nominal_error_norm_log,
        "residual_predicted_candidate_error_norm_log": residual_predicted_candidate_error_norm_log,
        "low_coef": low_coef,
        "high_coef": high_coef,
        "innovation_log": innovation_log,
        "innovation_raw_log": innovation_raw_log,
        "tracking_error_log": tracking_error_log,
        "tracking_error_raw_log": tracking_error_raw_log,
        "innovation_scale_ref": mismatch_cfg["innovation_scale_ref"],
        "tracking_scale_log": tracking_scale_log,
        "band_ref_scaled": mismatch_cfg["band_ref_scaled"],
        "mismatch_clip": mismatch_clip,
        "base_state_norm_mode": mismatch_cfg["base_state_norm_mode"],
        "base_state_running_norm_clip": mismatch_cfg["base_state_running_norm_clip"],
        "base_state_running_norm_eps": mismatch_cfg["base_state_running_norm_eps"],
        "base_state_norm_stats": state_conditioner.export_state(),
        "mismatch_feature_transform_mode": mismatch_cfg["mismatch_feature_transform_mode"],
        "mismatch_transform_tanh_scale": mismatch_cfg["mismatch_transform_tanh_scale"],
        "mismatch_transform_post_clip": mismatch_cfg["mismatch_transform_post_clip"],
        "append_rho_to_state": append_rho_to_state,
        "authority_beta_res": authority_beta_res,
        "authority_du0_res": authority_du0_res,
        "authority_rho_floor": authority_rho_floor,
        "authority_rho_power": authority_rho_power,
        "rho_mapping_mode": rho_mapping_mode,
        "authority_rho_k": authority_rho_k,
        "rho_raw_source": "tracking_error_raw",
        "residual_zero_deadband_enabled": residual_zero_deadband_enabled,
        "residual_zero_tracking_raw_threshold": residual_zero_tracking_raw_threshold,
        "residual_zero_innovation_raw_threshold": residual_zero_innovation_raw_threshold,
        "observer_update_alignment": observer_update_alignment,
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "warm_start_step": int(warm_start_step),
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
        "n_step": int(getattr(agent, "n_step", 1)),
        "multistep_mode": str(getattr(agent, "multistep_mode", "one_step")),
        "lambda_value": getattr(agent, "lambda_value", None),
        "mpc_horizons": (
            int(residual_cfg["predict_h"]),
            int(residual_cfg["cont_h"]),
        )
        if "predict_h" in residual_cfg and "cont_h" in residual_cfg
        else None,
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
        "bc_active_trace",
        "bc_weight_trace",
        "bc_loss_trace",
        "bc_actor_target_distance_trace",
    ):
        if hasattr(agent, attr):
            result_bundle[attr] = np.asarray(getattr(agent, attr), float)
    result_bundle.update(build_behavioral_cloning_bundle_fields(bc_schedule, bc_logs))
    result_bundle.update(build_bc_handoff_bundle_fields(bc_schedule, bc_handoff_logs))
    result_bundle.update(build_protected_bc_release_gate_bundle_fields(bc_release_gate))
    result_bundle.update(build_td3_authority_ramp_bundle_fields(td3_authority_ramp_cfg, td3_authority_ramp_logs))
    if phase1 is not None:
        result_bundle.update(
            build_phase1_bundle_fields(
                phase1,
                policy_action_raw_log=policy_action_raw_log,
                executed_action_raw_log=executed_action_raw_log,
                action_source_log=phase1_action_source_log,
                traces=phase1_train_traces,
            )
        )

    attach_single_agent_replay_snapshot(result_bundle, agent)
    return result_bundle
