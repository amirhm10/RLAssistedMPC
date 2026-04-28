import numpy as np
import scipy.optimize as spo

from utils.helpers import (
    apply_min_max,
    build_polymer_disturbance_schedule,
    disturbance_profile_from_schedule,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
    shift_control_sequence,
    step_system_with_disturbance,
)
from utils.behavioral_cloning import (
    build_behavioral_cloning_bundle_fields,
    build_behavioral_cloning_schedule,
    init_behavioral_cloning_logs,
    record_behavioral_cloning_step,
    resolve_behavioral_cloning_context,
)
from utils.observer import compute_observer_gain
from utils.observation_conditioning import update_observer_state
from utils.mpc_acceptance_gate import run_mpc_acceptance_gate, run_mpc_dual_cost_shadow
from utils.multiplier_release_schedule import (
    build_release_authority_schedule,
    clip_multipliers_to_release_bounds,
    map_effective_multipliers_to_raw_action,
)
from utils.phase1_hidden_release import (
    build_phase1_bundle_fields,
    build_phase1_schedule,
    init_phase1_train_traces,
    record_phase1_train_step,
    resolve_phase1_action_source,
)
from utils.replay_snapshot import attach_single_agent_replay_snapshot
from utils.state_features import (
    build_rl_state,
    compute_tracking_scale_now,
    make_state_conditioner_from_settings,
    resolve_mismatch_settings,
)
from utils.structured_model_update import (
    build_band_scaled_model,
    build_block_scaled_model,
    build_structured_update_spec,
    map_multipliers_to_normalized_action,
    map_normalized_action_to_multipliers,
)


def _structured_spec_matches_cfg(structured_spec, structured_cfg):
    if structured_spec is None:
        return False

    def _arr_or_none(value):
        if value is None:
            return None
        return np.asarray(value, float).reshape(-1)

    if str(structured_spec.get("update_family", "")).lower() != str(structured_cfg.get("update_family", "")).lower():
        return False
    if str(structured_spec.get("range_profile", "")).lower() != str(structured_cfg.get("range_profile", "tight")).lower():
        return False

    low_a_spec = _arr_or_none(structured_spec.get("low_a"))
    high_a_spec = _arr_or_none(structured_spec.get("high_a"))
    low_b_spec = _arr_or_none(structured_spec.get("low_b"))
    high_b_spec = _arr_or_none(structured_spec.get("high_b"))
    low_a_cfg = _arr_or_none(structured_cfg.get("a_low_override"))
    high_a_cfg = _arr_or_none(structured_cfg.get("a_high_override"))
    low_b_cfg = _arr_or_none(structured_cfg.get("b_low_override"))
    high_b_cfg = _arr_or_none(structured_cfg.get("b_high_override"))

    def _override_matches(spec_arr, cfg_arr):
        if cfg_arr is None:
            return True
        if spec_arr is None:
            return False
        if cfg_arr.size == 1:
            return np.allclose(spec_arr, float(cfg_arr[0]))
        return spec_arr.shape == cfg_arr.shape and np.allclose(spec_arr, cfg_arr)

    if not _override_matches(low_a_spec, low_a_cfg):
        return False
    if not _override_matches(high_a_spec, high_a_cfg):
        return False
    if not _override_matches(low_b_spec, low_b_cfg):
        return False
    if not _override_matches(high_b_spec, high_b_cfg):
        return False

    return True


def _solve_assisted_prediction_step(mpc_obj, y_sp, u_prev_dev, x0_model, initial_guess, bounds, step_idx):
    try:
        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
            np.asarray(initial_guess, float),
            bounds=bounds,
            constraints=[],
        )
    except Exception as exc:
        raise RuntimeError(f"Assisted structured-matrix MPC solve failed at step {step_idx}: {exc}") from exc

    success = bool(
        sol is not None
        and bool(getattr(sol, "success", True))
        and getattr(sol, "x", None) is not None
        and np.all(np.isfinite(np.asarray(sol.x, float)))
        and np.isfinite(float(getattr(sol, "fun", 0.0)))
    )
    if not success:
        message = str(getattr(sol, "message", "unknown solver failure"))
        raise RuntimeError(f"Assisted structured-matrix MPC solve failed at step {step_idx}: {message}")
    return sol


def _build_structured_update_payload(*, update_builder, update_family, A_base, B_base, n_outputs, update_cfg, theta_A, theta_B):
    if update_family == "block":
        return update_builder(
            A_aug=A_base,
            B_aug=B_base,
            n_outputs=n_outputs,
            block_cfg=update_cfg,
            theta_A=theta_A,
            theta_B=theta_B,
        )
    return update_builder(
        A_aug=A_base,
        B_aug=B_base,
        n_outputs=n_outputs,
        band_cfg=update_cfg,
        theta_A=theta_A,
        theta_B=theta_B,
    )


def run_structured_matrix_supervisor(structured_cfg, runtime_ctx):
    """
    Run the TD3/SAC structured matrix-update supervisor and return a normalized result bundle.

    Parameters
    ----------
    structured_cfg : dict
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
    reward_params = runtime_ctx.get("reward_params", {})
    system_stepper = runtime_ctx.get("system_stepper")
    system_metadata = runtime_ctx.get("system_metadata")
    disturbance_labels = runtime_ctx.get("disturbance_labels")

    agent_kind = str(structured_cfg["agent_kind"]).lower()
    run_mode = str(structured_cfg["run_mode"]).lower()
    state_mode = str(structured_cfg.get("state_mode", "standard")).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("structured_cfg['run_mode'] must be 'nominal' or 'disturb'.")
    if agent_kind not in {"td3", "sac"}:
        raise ValueError("structured_cfg['agent_kind'] must be 'td3' or 'sac'.")

    use_shifted_mpc_warm_start = bool(structured_cfg.get("use_shifted_mpc_warm_start", False))
    recalculate_observer_requested = bool(structured_cfg.get("recalculate_observer_on_matrix_change", False))
    log_spectral_radius = bool(structured_cfg.get("log_spectral_radius", True))
    mismatch_cfg = resolve_mismatch_settings(
        state_mode=state_mode,
        mismatch_cfg=structured_cfg,
        reward_params=reward_params,
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

    structured_spec = structured_cfg.get("structured_spec")
    structured_spec_refreshed = False
    if not _structured_spec_matches_cfg(structured_spec, structured_cfg):
        structured_spec = build_structured_update_spec(
            A_aug=A_aug,
            B_aug=B_aug,
            n_outputs=int(C_aug.shape[0]),
            update_family=structured_cfg["update_family"],
            range_profile=structured_cfg.get("range_profile", "tight"),
            block_group_count=structured_cfg.get("block_group_count", 3),
            block_groups=structured_cfg.get("block_groups"),
            band_offsets=structured_cfg.get("band_offsets"),
            a_low_override=structured_cfg.get("a_low_override"),
            a_high_override=structured_cfg.get("a_high_override"),
            b_low_override=structured_cfg.get("b_low_override"),
            b_high_override=structured_cfg.get("b_high_override"),
        )
        structured_spec_refreshed = True

    update_family = str(structured_spec["update_family"]).lower()
    action_dim = int(structured_spec["action_dim"])
    a_dim = int(structured_spec["a_dim"])
    b_dim = int(structured_spec["b_dim"])
    low_bounds = np.asarray(structured_spec["low_bounds"], float)
    high_bounds = np.asarray(structured_spec["high_bounds"], float)
    action_labels = tuple(str(label) for label in structured_spec["action_labels"])
    prediction_fallback_on_solve_failure = bool(structured_cfg.get("prediction_fallback_on_solve_failure", True))
    structured_baseline_raw = map_multipliers_to_normalized_action(
        np.ones(action_dim, dtype=float),
        low_bounds,
        high_bounds,
    )

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
        int(structured_cfg["n_tests"]),
        int(structured_cfg["set_points_len"]),
        int(structured_cfg["warm_start"]),
        list(structured_cfg["test_cycle"]),
        float(structured_cfg["nominal_qi"]),
        float(structured_cfg["nominal_qs"]),
        float(structured_cfg["nominal_ha"]),
        float(structured_cfg["qi_change"]),
        float(structured_cfg["qs_change"]),
        float(structured_cfg["ha_change"]),
    )

    disturbance_schedule = None
    if run_mode == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    phase1 = None
    phase1_action_source_log = None
    policy_action_raw_log = None
    executed_action_raw_log = None
    phase1_train_traces = None
    if agent_kind == "td3":
        phase1 = build_phase1_schedule(
            agent_kind=agent_kind,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            n_steps=nFE,
            test_train_dict=test_train_dict,
            action_freeze_subepisodes=structured_cfg.get("post_warm_start_action_freeze_subepisodes", 0),
            actor_freeze_subepisodes=structured_cfg.get("post_warm_start_actor_freeze_subepisodes", 0),
            batch_size=getattr(agent, "batch_size", 1),
            initial_buffer_size=len(getattr(agent, "buffer", [])),
            base_actor_freeze=getattr(agent, "actor_freeze", 0),
            push_start_step=0,
            train_start_step=warm_start_step,
        )
        agent.actor_freeze = int(phase1["effective_actor_freeze"])
        phase1_action_source_log = np.zeros(nFE, dtype=int)
        policy_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
        executed_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
        phase1_train_traces = init_phase1_train_traces()

    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug.shape[0])

    release_cfg = dict(structured_cfg.get("release_protected_advisory_caps", {}))
    release_action_freeze_end_step = phase1["action_freeze_end_step"] if phase1 is not None else warm_start_step
    release_schedule = build_release_authority_schedule(
        config=release_cfg,
        labels=action_labels,
        wide_low=low_bounds,
        wide_high=high_bounds,
        warm_start_step=warm_start_step,
        action_freeze_end_step=release_action_freeze_end_step,
        time_in_sub_episodes=time_in_sub_episodes,
        n_steps=nFE,
    )
    release_store_executed_action = bool(release_schedule.get("store_executed_action_in_replay", True))
    acceptance_cfg = dict(structured_cfg.get("mpc_acceptance_fallback", {}))
    acceptance_enabled = bool(acceptance_cfg.get("enabled", False))
    acceptance_store_executed_action = bool(
        acceptance_cfg.get("store_executed_action_in_replay", release_store_executed_action)
    )
    dual_cost_shadow_cfg = dict(structured_cfg.get("mpc_dual_cost_shadow", {}))
    dual_cost_shadow_enabled = bool(dual_cost_shadow_cfg.get("enabled", False))
    bc_schedule = build_behavioral_cloning_schedule(
        config=structured_cfg.get("behavioral_cloning", {}),
        warm_start_step=warm_start_step,
        time_in_sub_episodes=time_in_sub_episodes,
        n_steps=nFE,
    )
    bc_logs = init_behavioral_cloning_logs(nFE)

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    y_system = np.zeros((nFE + 1, n_outputs))
    y_system[0, :] = np.asarray(system.current_output, float)
    u_mpc = np.zeros((nFE, n_inputs))
    rewards = np.zeros(nFE)
    avg_rewards = []
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    delta_y_storage = np.zeros((nFE, n_outputs))
    delta_u_storage = np.zeros((nFE, n_inputs))
    raw_action_log = np.zeros((nFE, action_dim))
    candidate_action_log = np.zeros((nFE, action_dim))
    effective_action_log = np.zeros((nFE, action_dim))
    mapped_multiplier_log = np.zeros((nFE, action_dim))
    candidate_multiplier_log = np.zeros((nFE, action_dim))
    effective_multiplier_log = np.zeros((nFE, action_dim))
    release_effective_low_log = np.zeros((nFE, action_dim))
    release_effective_high_log = np.zeros((nFE, action_dim))
    release_phase_log = np.zeros(nFE, dtype=int)
    release_guard_active_log = np.zeros(nFE, dtype=int)
    release_clip_fraction_log = np.zeros(nFE)
    release_ramp_fraction_log = np.zeros(nFE)
    theta_a_log = np.zeros((nFE, a_dim))
    theta_b_log = np.zeros((nFE, b_dim))
    effective_theta_a_log = np.zeros((nFE, a_dim))
    effective_theta_b_log = np.zeros((nFE, b_dim))
    A_fro_ratio_log = np.zeros(nFE)
    B_fro_ratio_log = np.zeros(nFE)
    spectral_radius_log = np.zeros(nFE)
    candidate_A_fro_ratio_log = np.zeros(nFE)
    candidate_B_fro_ratio_log = np.zeros(nFE)
    candidate_spectral_radius_log = np.zeros(nFE)
    action_saturation_fraction_log = np.zeros(nFE)
    near_bound_fraction_log = np.zeros(nFE)
    prediction_fallback_active_log = np.zeros(nFE, dtype=int)
    prediction_fallback_reason_log = np.zeros(nFE, dtype=int)
    acceptance_active_log = np.zeros(nFE, dtype=int)
    acceptance_accepted_log = np.ones(nFE, dtype=int)
    acceptance_fallback_active_log = np.zeros(nFE, dtype=int)
    acceptance_reason_code_log = np.zeros(nFE, dtype=int)
    acceptance_candidate_cost_on_nominal_log = np.full(nFE, np.nan)
    acceptance_candidate_cost_native_log = np.full(nFE, np.nan)
    acceptance_nominal_cost_log = np.full(nFE, np.nan)
    acceptance_cost_margin_log = np.full(nFE, np.nan)
    acceptance_threshold_log = np.full(nFE, np.nan)
    dual_cost_shadow_active_log = np.zeros(nFE, dtype=int)
    dual_cost_shadow_reason_code_log = np.zeros(nFE, dtype=int)
    dual_cost_shadow_candidate_cost_native_log = np.full(nFE, np.nan)
    dual_cost_shadow_nominal_cost_log = np.full(nFE, np.nan)
    dual_cost_shadow_candidate_cost_on_nominal_log = np.full(nFE, np.nan)
    dual_cost_shadow_nominal_cost_on_candidate_log = np.full(nFE, np.nan)
    dual_cost_shadow_nominal_penalty_log = np.full(nFE, np.nan)
    dual_cost_shadow_safe_threshold_log = np.full(nFE, np.nan)
    dual_cost_shadow_candidate_advantage_log = np.full(nFE, np.nan)
    dual_cost_shadow_safe_pass_log = np.zeros(nFE, dtype=int)
    dual_cost_shadow_benefit_pass_log = np.zeros(nFE, dtype=int)
    dual_cost_shadow_dual_pass_log = np.zeros(nFE, dtype=int)
    innovation_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    innovation_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_scale_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None

    update_builder = build_block_scaled_model if update_family == "block" else build_band_scaled_model
    update_cfg = structured_spec["block_cfg"] if update_family == "block" else structured_spec["band_cfg"]
    A_base = np.asarray(mpc_obj.A, float).copy()
    B_base = np.asarray(mpc_obj.B, float).copy()
    nominal_update_payload = _build_structured_update_payload(
        update_builder=update_builder,
        update_family=update_family,
        A_base=A_base,
        B_base=B_base,
        n_outputs=n_outputs,
        update_cfg=update_cfg,
        theta_A=np.ones(a_dim, dtype=float),
        theta_B=np.ones(b_dim, dtype=float),
    )
    A_est = A_base.copy()
    B_est = B_base.copy()
    L_nom = compute_observer_gain(A_est, C_aug, poles)
    test = False
    nonfinite_action_count = 0
    structured_update_fallback_count = 0
    structured_prediction_fallback_count = 0
    saturation_threshold = float(structured_cfg.get("action_saturation_threshold", 0.98))
    near_bound_tolerance = float(structured_cfg.get("near_bound_relative_tolerance", 0.05))

    cont_h = int(structured_cfg.get("cont_h", 1))
    b_min = np.asarray(structured_cfg["b_min"], float).reshape(-1)
    b_max = np.asarray(structured_cfg["b_max"], float).reshape(-1)
    original_bounds = tuple(
        (float(b_min[j]), float(b_max[j]))
        for _ in range(cont_h)
        for j in range(b_min.size)
    )
    ic_opt = np.zeros(n_inputs * cont_h)

    episode_avg_theta_a = []
    episode_avg_theta_b = []
    episode_avg_action_saturation = []
    episode_avg_near_bound = []
    episode_avg_A_ratio = []
    episode_avg_B_ratio = []

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

        phase1_hidden_active = bool(phase1 is not None and phase1["enabled"] and phase1["hidden_window_active_log"][i])
        policy_action = None
        if i > warm_start_step:
            if phase1 is not None:
                policy_action = np.asarray(agent.act_eval(current_rl_state), float).reshape(-1)
                if not np.all(np.isfinite(policy_action)):
                    policy_action = structured_baseline_raw.copy()
            if phase1_hidden_active:
                action = structured_baseline_raw.copy()
            elif not test:
                action = np.asarray(agent.take_action(current_rl_state, explore=True), float).reshape(-1)
            else:
                action = policy_action.copy() if policy_action is not None else np.asarray(agent.act_eval(current_rl_state), float).reshape(-1)
        else:
            action = structured_baseline_raw.copy()
            if phase1 is not None:
                policy_action = structured_baseline_raw.copy()

        if not np.all(np.isfinite(action)):
            action = structured_baseline_raw.copy()
            nonfinite_action_count += 1

        try:
            mapped = map_normalized_action_to_multipliers(action, low_bounds, high_bounds)
            release_trace = clip_multipliers_to_release_bounds(mapped, release_schedule, i)
            effective_mapped_for_update = np.asarray(release_trace["multipliers"], float)
            effective_action_for_update = map_effective_multipliers_to_raw_action(
                effective_mapped_for_update,
                low_bounds,
                high_bounds,
            )
            update_payload = _build_structured_update_payload(
                update_builder=update_builder,
                update_family=update_family,
                A_base=A_base,
                B_base=B_base,
                n_outputs=n_outputs,
                update_cfg=update_cfg,
                theta_A=effective_mapped_for_update[:a_dim],
                theta_B=effective_mapped_for_update[a_dim:],
            )
        except Exception:
            action = structured_baseline_raw.copy()
            mapped = np.ones(action_dim, dtype=float)
            release_trace = clip_multipliers_to_release_bounds(mapped, release_schedule, i)
            effective_mapped_for_update = np.ones(action_dim, dtype=float)
            effective_action_for_update = structured_baseline_raw.copy()
            update_payload = nominal_update_payload
            structured_update_fallback_count += 1

        raw_action_log[i, :] = action
        candidate_action_log[i, :] = effective_action_for_update
        mapped_multiplier_log[i, :] = mapped
        candidate_multiplier_log[i, :] = effective_mapped_for_update
        theta_a_log[i, :] = mapped[:a_dim]
        theta_b_log[i, :] = mapped[a_dim:]
        candidate_A_fro_ratio_log[i] = float(update_payload["A_fro_ratio"])
        candidate_B_fro_ratio_log[i] = float(update_payload["B_fro_ratio"])
        candidate_spectral_radius_log[i] = float(update_payload["spectral_radius"]) if log_spectral_radius else np.nan
        action_saturation_fraction_log[i] = float(np.mean(np.abs(action) >= saturation_threshold))
        bound_span = np.maximum(high_bounds - low_bounds, 1e-12)
        near_low = (mapped - low_bounds) <= near_bound_tolerance * bound_span
        near_high = (high_bounds - mapped) <= near_bound_tolerance * bound_span
        near_bound_fraction_log[i] = float(np.mean(near_low | near_high))
        release_effective_low_log[i, :] = release_trace["low"]
        release_effective_high_log[i, :] = release_trace["high"]
        release_phase_log[i] = int(release_trace["phase_code"])
        release_guard_active_log[i] = int(bool(release_trace["guard_active"]))
        release_clip_fraction_log[i] = float(release_trace["clip_fraction"])
        release_ramp_fraction_log[i] = float(release_trace["ramp_fraction"])

        A_candidate = np.asarray(update_payload["A_aug"], float)
        B_candidate = np.asarray(update_payload["B_aug"], float)
        prediction_payload = update_payload
        effective_action = effective_action_for_update.copy()
        effective_mapped = effective_mapped_for_update.copy()
        prediction_fallback_reason = 0

        if (
            (not acceptance_enabled)
            and (not dual_cost_shadow_enabled)
            and not (np.all(np.isfinite(A_candidate)) and np.all(np.isfinite(B_candidate)))
        ):
            if not prediction_fallback_on_solve_failure:
                raise RuntimeError(f"Assisted structured matrix prediction model became non-finite at step {i}.")
            prediction_payload = nominal_update_payload
            effective_action = structured_baseline_raw.copy()
            effective_mapped = np.ones(action_dim, dtype=float)
            prediction_fallback_active_log[i] = 1
            prediction_fallback_reason = 1
            structured_prediction_fallback_count += 1

        ic_opt_step = ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * cont_h)
        shadow_trace = None
        if dual_cost_shadow_enabled:
            shadow_trace = run_mpc_dual_cost_shadow(
                mpc_obj=mpc_obj,
                solve_fn=_solve_assisted_prediction_step,
                shadow_cfg=dual_cost_shadow_cfg,
                A_candidate=A_candidate,
                B_candidate=B_candidate,
                A_nominal=A_base,
                B_nominal=B_base,
                y_sp=y_sp[i, :],
                u_prev_dev=scaled_current_input_dev,
                x0_model=xhatdhat[:, i],
                initial_guess=ic_opt_step,
                bounds=original_bounds,
                step_idx=i,
            )

        if acceptance_enabled:
            acceptance_trace = run_mpc_acceptance_gate(
                mpc_obj=mpc_obj,
                solve_fn=_solve_assisted_prediction_step,
                acceptance_cfg=acceptance_cfg,
                A_candidate=A_candidate,
                B_candidate=B_candidate,
                A_nominal=A_base,
                B_nominal=B_base,
                y_sp=y_sp[i, :],
                u_prev_dev=scaled_current_input_dev,
                x0_model=xhatdhat[:, i],
                initial_guess=ic_opt_step,
                bounds=original_bounds,
                step_idx=i,
            )
            sol = acceptance_trace["sol"]
            if bool(acceptance_trace["accepted"]):
                prediction_payload = update_payload
                effective_action = effective_action_for_update.copy()
                effective_mapped = effective_mapped_for_update.copy()
            else:
                prediction_payload = nominal_update_payload
                effective_action = structured_baseline_raw.copy()
                effective_mapped = np.ones(action_dim, dtype=float)
        elif dual_cost_shadow_enabled:
            acceptance_trace = {
                "accepted": True,
                "fallback_active": False,
                "reason_code": 0,
                "candidate_cost_on_nominal": np.nan,
                "candidate_cost_native": np.nan,
                "nominal_cost": np.nan,
                "cost_margin": np.nan,
                "threshold": np.nan,
            }
            sol = shadow_trace["sol"]
            if bool(shadow_trace["executed_candidate"]):
                prediction_payload = update_payload
                effective_action = effective_action_for_update.copy()
                effective_mapped = effective_mapped_for_update.copy()
            else:
                prediction_payload = nominal_update_payload
                effective_action = structured_baseline_raw.copy()
                effective_mapped = np.ones(action_dim, dtype=float)
                prediction_fallback_active_log[i] = 1
                prediction_fallback_reason = 3
                structured_prediction_fallback_count += 1
        else:
            acceptance_trace = {
                "accepted": True,
                "fallback_active": False,
                "reason_code": 0,
                "candidate_cost_on_nominal": np.nan,
                "candidate_cost_native": np.nan,
                "nominal_cost": np.nan,
                "cost_margin": np.nan,
                "threshold": np.nan,
            }
            mpc_obj.A = np.asarray(prediction_payload["A_aug"], float)
            mpc_obj.B = np.asarray(prediction_payload["B_aug"], float)
            try:
                sol = _solve_assisted_prediction_step(
                    mpc_obj=mpc_obj,
                    y_sp=y_sp[i, :],
                    u_prev_dev=scaled_current_input_dev,
                    x0_model=xhatdhat[:, i],
                    initial_guess=ic_opt_step,
                    bounds=original_bounds,
                    step_idx=i,
                )
            except RuntimeError as exc:
                if not prediction_fallback_on_solve_failure:
                    raise
                if prediction_fallback_reason != 0:
                    raise RuntimeError(
                        f"Assisted structured-matrix MPC solve failed at step {i} while already on nominal fallback: {exc}"
                    ) from exc
                prediction_payload = nominal_update_payload
                effective_action = structured_baseline_raw.copy()
                effective_mapped = np.ones(action_dim, dtype=float)
                mpc_obj.A = np.asarray(prediction_payload["A_aug"], float)
                mpc_obj.B = np.asarray(prediction_payload["B_aug"], float)
                prediction_fallback_active_log[i] = 1
                prediction_fallback_reason = 2
                structured_prediction_fallback_count += 1
                try:
                    sol = _solve_assisted_prediction_step(
                        mpc_obj=mpc_obj,
                        y_sp=y_sp[i, :],
                        u_prev_dev=scaled_current_input_dev,
                        x0_model=xhatdhat[:, i],
                        initial_guess=ic_opt_step,
                        bounds=original_bounds,
                        step_idx=i,
                    )
                except RuntimeError as fallback_exc:
                    raise RuntimeError(
                        f"Assisted structured-matrix MPC solve failed at step {i}; "
                        f"nominal fallback also failed. Candidate error: {exc}. "
                        f"Fallback error: {fallback_exc}"
                    ) from fallback_exc

        effective_action_log[i, :] = effective_action
        if phase1 is not None:
            policy_action_raw_log[i, :] = np.asarray(
                policy_action if policy_action is not None else structured_baseline_raw,
                float,
            ).reshape(-1)
            executed_action_raw_log[i, :] = np.asarray(effective_action, float).reshape(-1)
            phase1_action_source_log[i] = int(
                resolve_phase1_action_source(
                    i,
                    warm_start_step,
                    phase1_hidden_active,
                    test,
                )
            )
        effective_multiplier_log[i, :] = effective_mapped
        effective_theta_a_log[i, :] = effective_mapped[:a_dim]
        effective_theta_b_log[i, :] = effective_mapped[a_dim:]
        A_fro_ratio_log[i] = float(prediction_payload["A_fro_ratio"])
        B_fro_ratio_log[i] = float(prediction_payload["B_fro_ratio"])
        spectral_radius_log[i] = float(prediction_payload["spectral_radius"]) if log_spectral_radius else np.nan
        prediction_fallback_reason_log[i] = int(prediction_fallback_reason)
        acceptance_active_log[i] = int(acceptance_enabled)
        acceptance_accepted_log[i] = int(bool(acceptance_trace["accepted"]))
        acceptance_fallback_active_log[i] = int(bool(acceptance_trace["fallback_active"]))
        acceptance_reason_code_log[i] = int(acceptance_trace["reason_code"])
        acceptance_candidate_cost_on_nominal_log[i] = float(acceptance_trace["candidate_cost_on_nominal"])
        acceptance_candidate_cost_native_log[i] = float(acceptance_trace["candidate_cost_native"])
        acceptance_nominal_cost_log[i] = float(acceptance_trace["nominal_cost"])
        acceptance_cost_margin_log[i] = float(acceptance_trace["cost_margin"])
        acceptance_threshold_log[i] = float(acceptance_trace["threshold"])
        if shadow_trace is not None:
            dual_cost_shadow_active_log[i] = 1
            dual_cost_shadow_reason_code_log[i] = int(shadow_trace["reason_code"])
            dual_cost_shadow_candidate_cost_native_log[i] = float(shadow_trace["candidate_cost_native"])
            dual_cost_shadow_nominal_cost_log[i] = float(shadow_trace["nominal_cost"])
            dual_cost_shadow_candidate_cost_on_nominal_log[i] = float(shadow_trace["candidate_cost_on_nominal"])
            dual_cost_shadow_nominal_cost_on_candidate_log[i] = float(shadow_trace["nominal_cost_on_candidate"])
            dual_cost_shadow_nominal_penalty_log[i] = float(shadow_trace["nominal_penalty"])
            dual_cost_shadow_safe_threshold_log[i] = float(shadow_trace["safe_threshold"])
            dual_cost_shadow_candidate_advantage_log[i] = float(shadow_trace["candidate_advantage"])
            dual_cost_shadow_safe_pass_log[i] = int(bool(shadow_trace["safe_pass"]))
            dual_cost_shadow_benefit_pass_log[i] = int(bool(shadow_trace["benefit_pass"]))
            dual_cost_shadow_dual_pass_log[i] = int(bool(shadow_trace["dual_pass"]))

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

        xhatdhat[:, i + 1], yhat[:, i], observer_update_alignment = update_observer_state(
            A=A_est,
            B=B_est,
            C=C_aug,
            L=L_nom,
            x_prev=xhatdhat[:, i],
            u_dev=(u_mpc[i, :] - ss_scaled_inputs),
            y_prev_scaled=y_prev_scaled,
            y_current_scaled=y_current_scaled,
            observer_update_alignment=observer_update_alignment,
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
            state_conditioner=state_conditioner,
            update_state_conditioner=False,
            mismatch_feature_transform_mode=mismatch_cfg["mismatch_feature_transform_mode"],
            mismatch_transform_tanh_scale=mismatch_cfg["mismatch_transform_tanh_scale"],
            mismatch_transform_post_clip=mismatch_cfg["mismatch_transform_post_clip"],
        )

        if not test:
            replay_action = effective_action if acceptance_store_executed_action else action
            bc_context = resolve_behavioral_cloning_context(
                bc_schedule,
                step_idx=i,
                nominal_target_action=structured_baseline_raw,
                action_labels=action_labels,
            )
            agent.push(
                current_rl_state,
                np.asarray(replay_action, np.float32),
                reward,
                next_rl_state,
                0.0,
            )
            train_meta = None
            if i >= warm_start_step:
                train_meta = agent.train_step(bc_context=bc_context)
                if phase1 is not None:
                    record_phase1_train_step(phase1_train_traces, i, train_meta)
            record_behavioral_cloning_step(
                bc_logs,
                step_idx=i,
                bc_context=bc_context,
                policy_action=np.asarray(policy_action if policy_action is not None else action, float).reshape(-1),
                nominal_target_action=structured_baseline_raw,
                train_meta=train_meta,
            )
        else:
            record_behavioral_cloning_step(
                bc_logs,
                step_idx=i,
                bc_context=None,
                policy_action=np.asarray(policy_action if policy_action is not None else action, float).reshape(-1),
                nominal_target_action=structured_baseline_raw,
                train_meta=None,
            )

        if i in sub_episodes_changes_dict:
            lo = max(0, i - time_in_sub_episodes + 1)
            hi = i + 1
            avg_rewards.append(float(np.mean(rewards[lo:hi])))
            episode_avg_theta_a.append(np.mean(theta_a_log[lo:hi, :], axis=0))
            episode_avg_theta_b.append(np.mean(theta_b_log[lo:hi, :], axis=0))
            episode_avg_action_saturation.append(float(np.mean(action_saturation_fraction_log[lo:hi])))
            episode_avg_near_bound.append(float(np.mean(near_bound_fraction_log[lo:hi])))
            episode_avg_A_ratio.append(float(np.mean(A_fro_ratio_log[lo:hi])))
            episode_avg_B_ratio.append(float(np.mean(B_fro_ratio_log[lo:hi])))
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_rewards[-1],
                "| theta_A:",
                np.mean(theta_a_log[lo:hi, :], axis=0),
                "| theta_B:",
                np.mean(theta_b_log[lo:hi, :], axis=0),
            )

    mpc_obj.A = A_base
    mpc_obj.B = B_base
    if hasattr(agent, "flush_nstep"):
        agent.flush_nstep()
    u_rl = reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])

    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if run_mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )

    result_bundle = {
        "agent_kind": agent_kind,
        "run_mode": run_mode,
        "method_family": "structured_matrix",
        "algorithm": agent_kind,
        "state_mode": state_mode,
        "system_metadata": system_metadata,
        "notebook_source": structured_cfg.get("notebook_source"),
        "config_snapshot": dict(structured_cfg),
        "seed": structured_cfg.get("seed"),
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": int(nFE),
        "delta_t": float(system.delta_t),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y": y_system,
        "u": u_rl,
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
        "recalculate_observer_on_matrix_change": recalculate_observer_requested,
        "recalculate_observer_on_matrix_change_ignored": True,
        "log_spectral_radius": log_spectral_radius,
        "prediction_fallback_on_solve_failure": prediction_fallback_on_solve_failure,
        "nonfinite_matrix_action_count": int(nonfinite_action_count),
        "structured_update_fallback_count": int(structured_update_fallback_count),
        "structured_prediction_fallback_count": int(structured_prediction_fallback_count),
        "n_step": int(getattr(agent, "n_step", 1)),
        "multistep_mode": str(getattr(agent, "multistep_mode", "one_step")),
        "lambda_value": getattr(agent, "lambda_value", None),
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
        "observer_update_alignment": observer_update_alignment,
        "mpc_horizons": (
            int(structured_cfg["predict_h"]),
            int(structured_cfg["cont_h"]),
        )
        if "predict_h" in structured_cfg and "cont_h" in structured_cfg
        else None,
        "update_family": update_family,
        "range_profile": structured_spec["range_profile"],
        "structured_action_dim": int(action_dim),
        "structured_a_dim": int(a_dim),
        "structured_b_dim": int(b_dim),
        "action_labels": tuple(structured_spec["action_labels"]),
        "theta_a_labels": tuple(structured_spec["theta_a_labels"]),
        "theta_b_labels": tuple(structured_spec["theta_b_labels"]),
        "structured_low_bounds": low_bounds,
        "structured_high_bounds": high_bounds,
        "structured_low_a": np.asarray(structured_spec["low_a"], float),
        "structured_high_a": np.asarray(structured_spec["high_a"], float),
        "structured_low_b": np.asarray(structured_spec["low_b"], float),
        "structured_high_b": np.asarray(structured_spec["high_b"], float),
        "structured_spec_refreshed": bool(structured_spec_refreshed),
        "block_cfg": structured_spec["block_cfg"],
        "band_cfg": structured_spec["band_cfg"],
        "raw_action_log": raw_action_log,
        "candidate_action_log": candidate_action_log,
        "effective_action_log": effective_action_log,
        "mapped_multiplier_log": mapped_multiplier_log,
        "candidate_multiplier_log": candidate_multiplier_log,
        "effective_multiplier_log": effective_multiplier_log,
        "release_schedule": release_schedule,
        "release_guard_enabled": bool(release_schedule.get("enabled", False)),
        "release_phase_log": release_phase_log,
        "release_guard_active_log": release_guard_active_log,
        "release_clip_fraction_log": release_clip_fraction_log,
        "release_ramp_fraction_log": release_ramp_fraction_log,
        "release_effective_low_log": release_effective_low_log,
        "release_effective_high_log": release_effective_high_log,
        "release_store_executed_action_in_replay": release_store_executed_action,
        "mpc_acceptance_fallback": acceptance_cfg,
        "mpc_acceptance_enabled": acceptance_enabled,
        "mpc_acceptance_store_executed_action_in_replay": acceptance_store_executed_action,
        "mpc_acceptance_active_log": acceptance_active_log,
        "mpc_acceptance_accepted_log": acceptance_accepted_log,
        "mpc_acceptance_fallback_active_log": acceptance_fallback_active_log,
        "mpc_acceptance_reason_code_log": acceptance_reason_code_log,
        "mpc_acceptance_candidate_cost_on_nominal_log": acceptance_candidate_cost_on_nominal_log,
        "mpc_acceptance_candidate_cost_native_log": acceptance_candidate_cost_native_log,
        "mpc_acceptance_nominal_cost_log": acceptance_nominal_cost_log,
        "mpc_acceptance_cost_margin_log": acceptance_cost_margin_log,
        "mpc_acceptance_threshold_log": acceptance_threshold_log,
        "mpc_dual_cost_shadow": dual_cost_shadow_cfg,
        "mpc_dual_cost_shadow_enabled": dual_cost_shadow_enabled,
        "mpc_dual_cost_shadow_active_log": dual_cost_shadow_active_log,
        "mpc_dual_cost_shadow_reason_code_log": dual_cost_shadow_reason_code_log,
        "mpc_dual_cost_shadow_candidate_cost_native_log": dual_cost_shadow_candidate_cost_native_log,
        "mpc_dual_cost_shadow_nominal_cost_log": dual_cost_shadow_nominal_cost_log,
        "mpc_dual_cost_shadow_candidate_cost_on_nominal_log": dual_cost_shadow_candidate_cost_on_nominal_log,
        "mpc_dual_cost_shadow_nominal_cost_on_candidate_log": dual_cost_shadow_nominal_cost_on_candidate_log,
        "mpc_dual_cost_shadow_nominal_penalty_log": dual_cost_shadow_nominal_penalty_log,
        "mpc_dual_cost_shadow_safe_threshold_log": dual_cost_shadow_safe_threshold_log,
        "mpc_dual_cost_shadow_candidate_advantage_log": dual_cost_shadow_candidate_advantage_log,
        "mpc_dual_cost_shadow_safe_pass_log": dual_cost_shadow_safe_pass_log,
        "mpc_dual_cost_shadow_benefit_pass_log": dual_cost_shadow_benefit_pass_log,
        "mpc_dual_cost_shadow_dual_pass_log": dual_cost_shadow_dual_pass_log,
        "theta_a_log": theta_a_log,
        "theta_b_log": theta_b_log,
        "effective_theta_a_log": effective_theta_a_log,
        "effective_theta_b_log": effective_theta_b_log,
        "A_model_delta_ratio_log": A_fro_ratio_log,
        "B_model_delta_ratio_log": B_fro_ratio_log,
        "spectral_radius_log": spectral_radius_log,
        "candidate_A_model_delta_ratio_log": candidate_A_fro_ratio_log,
        "candidate_B_model_delta_ratio_log": candidate_B_fro_ratio_log,
        "candidate_spectral_radius_log": candidate_spectral_radius_log,
        "action_saturation_fraction_log": action_saturation_fraction_log,
        "near_bound_fraction_log": near_bound_fraction_log,
        "prediction_fallback_active_log": prediction_fallback_active_log,
        "prediction_fallback_reason_log": prediction_fallback_reason_log,
        "near_bound_relative_tolerance": float(near_bound_tolerance),
        "action_saturation_threshold": float(saturation_threshold),
        "episode_avg_theta_a": np.asarray(episode_avg_theta_a, float),
        "episode_avg_theta_b": np.asarray(episode_avg_theta_b, float),
        "episode_avg_action_saturation": np.asarray(episode_avg_action_saturation, float),
        "episode_avg_near_bound": np.asarray(episode_avg_near_bound, float),
        "episode_avg_A_model_delta_ratio": np.asarray(episode_avg_A_ratio, float),
        "episode_avg_B_model_delta_ratio": np.asarray(episode_avg_B_ratio, float),
        "estimator_mode": "fixed_nominal",
        "prediction_model_mode": "rl_assisted",
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
        "target_q_mean_trace",
        "actor_loss_trace",
        "critic_loss_trace",
        "alpha_loss_trace",
        "offpolicy_rho_mean_trace",
        "offpolicy_c_mean_trace",
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
