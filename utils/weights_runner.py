from types import SimpleNamespace

import numpy as np
import scipy.optimize as spo

from TD3Agent.supervisor_replay_buffer import (
    SOURCE_FALLBACK,
    SOURCE_POLICY,
    SOURCE_SUPERVISOR,
    SOURCE_WARM_START,
)
from systems.polymer.scenarios import build_polymer_training_profile
from utils.agent_step_runtime import replay_train_continuous_agent, select_continuous_action
from utils.episode_profiles import episode_profile_result_fields, resolve_episode_bundle
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
    record_phase1_train_step,
)
from utils.exploration_freeze import (
    effective_agent_exploration_value,
    exploration_freeze_result_fields,
    maybe_freeze_agent_exploration,
)
from utils.replay_snapshot import attach_single_agent_replay_snapshot
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


def _set_penalties(mpc_obj, q_base, r_base, multipliers):
    multipliers = np.asarray(multipliers, float).reshape(-1)
    if multipliers.size != 4:
        raise ValueError("weights runner expects 4 multipliers for [Q1, Q2, R1, R2].")

    mpc_obj.Q_out = np.array(
        [
            float(q_base[0] * multipliers[0]),
            float(q_base[1] * multipliers[1]),
        ],
        dtype=float,
    )
    mpc_obj.R_in = np.array(
        [
            float(r_base[0] * multipliers[2]),
            float(r_base[1] * multipliers[3]),
        ],
        dtype=float,
    )


WEIGHT_ACTION_SOURCE_CODES = {
    "identity_warm": 0,
    "td3_accepted": 1,
    "projected_td3": 2,
    "identity_fallback": 3,
}

WEIGHT_FALLBACK_REASON_CODES = {
    "none": 0,
    "nonfinite_selected_action": 1,
    "invalid_multiplier": 2,
    "mpc_solve_failure": 3,
}


def _solve_successful(sol):
    return bool(getattr(sol, "success", False)) and np.all(np.isfinite(np.asarray(sol.x, float)))


def run_weight_multiplier_supervisor(weight_cfg, runtime_ctx):
    """
    Run the TD3/SAC weight-multiplier supervisor and return a normalized result bundle.

    Parameters
    ----------
    weight_cfg : dict
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

    agent_kind = str(weight_cfg["agent_kind"]).lower()
    supervisor_gated_agent_kind = agent_kind in {"sg_td3", "sg_sac"}
    supervisor_gated_td3_agent_kind = agent_kind == "sg_td3"
    supervisor_gated_sac_agent_kind = agent_kind == "sg_sac"
    td3_like_agent_kind = agent_kind in {"td3", "sg_td3"}
    run_mode = str(weight_cfg["run_mode"]).lower()
    state_mode = str(weight_cfg.get("state_mode", "standard")).lower()
    weight_safety_cfg = dict(weight_cfg.get("weight_safety", {}) or {})
    weight_safety_enabled = bool(weight_safety_cfg.get("enabled", False))
    reward_probation_cfg = dict(weight_safety_cfg.get("reward_probation", {}) or {})
    reward_probation_enabled = bool(weight_safety_enabled and reward_probation_cfg.get("enabled", False))
    probation_reference_warm_episodes = int(max(1, reward_probation_cfg.get("reference_warm_episodes", 3)))
    probation_collapse_threshold = float(reward_probation_cfg.get("collapse_threshold", 5.0))
    probation_cooldown_subepisodes = int(max(0, reward_probation_cfg.get("cooldown_subepisodes", 0)))
    probation_cooldown_multiplier_cap = float(reward_probation_cfg.get("cooldown_multiplier_cap", 0.10))
    fallback_to_identity_on_nonfinite = bool(
        weight_safety_enabled and weight_safety_cfg.get("fallback_to_identity_on_nonfinite", False)
    )
    fallback_to_identity_on_solve_failure = bool(
        weight_safety_enabled and weight_safety_cfg.get("fallback_to_identity_on_solve_failure", False)
    )
    shadow_identity_cfg = dict(weight_safety_cfg.get("shadow_identity_mpc", {}) or {})
    shadow_identity_enabled = bool(weight_safety_enabled and shadow_identity_cfg.get("enabled", False))
    shadow_identity_stride = int(max(1, shadow_identity_cfg.get("diagnostic_stride", 5)))
    if agent_kind not in {"td3", "sac", "sg_td3", "sg_sac"}:
        raise ValueError("weight_cfg['agent_kind'] must be 'td3', 'sac', 'sg_td3', or 'sg_sac'.")
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("weight_cfg['run_mode'] must be 'nominal' or 'disturb'.")
    use_shifted_mpc_warm_start = bool(weight_cfg.get("use_shifted_mpc_warm_start", False))
    mismatch_cfg = resolve_mismatch_settings(
        state_mode=state_mode,
        mismatch_cfg=weight_cfg,
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

    low_coef = np.asarray(weight_cfg["low_coef"], float).reshape(-1)
    high_coef = np.asarray(weight_cfg["high_coef"], float).reshape(-1)
    if low_coef.size != 4 or high_coef.size != 4:
        raise ValueError("low_coef/high_coef must each have length 4 for [Q1, Q2, R1, R2].")

    q_base = np.array([weight_cfg["Q1_penalty"], weight_cfg["Q2_penalty"]], dtype=float)
    r_base = np.array([weight_cfg["R1_penalty"], weight_cfg["R2_penalty"]], dtype=float)
    action_dim = 4
    identity_action = _map_from_bounds(np.ones(4, dtype=float), low_coef, high_coef)

    episode_bundle = resolve_episode_bundle(
        runtime_ctx,
        fallback_builder=lambda: build_polymer_training_profile(
            profile_name=weight_cfg.get("training_profile_name"),
            y_sp_scenario=y_sp_scenario,
            n_tests=int(weight_cfg["n_tests"]),
            set_points_len=int(weight_cfg["set_points_len"]),
            warm_start=int(weight_cfg["warm_start"]),
            test_cycle=list(weight_cfg["test_cycle"]),
            nominal_qi=float(weight_cfg["nominal_qi"]),
            nominal_qs=float(weight_cfg["nominal_qs"]),
            nominal_ha=float(weight_cfg["nominal_ha"]),
            qi_change=float(weight_cfg["qi_change"]),
            qs_change=float(weight_cfg["qs_change"]),
            ha_change=float(weight_cfg["ha_change"]),
            steady_outputs=steady_states["y_ss"],
            data_min=data_min,
            data_max=data_max,
            n_inputs=int(B_aug.shape[1]),
        ),
        expected_n_tests=int(weight_cfg["n_tests"]),
        expected_n_outputs=int(C_aug.shape[0]),
    )
    y_sp = np.asarray(episode_bundle["y_sp"], float)
    nFE = int(episode_bundle["nFE"])
    sub_episodes_changes_dict = dict(episode_bundle["sub_episodes_changes_dict"])
    time_in_sub_episodes = int(episode_bundle["time_in_sub_episodes"])
    test_train_dict = dict(episode_bundle["test_train_dict"])
    warm_start_step = int(episode_bundle["warm_start_step"])
    qi = np.asarray(episode_bundle["qi"], float)
    qs = np.asarray(episode_bundle["qs"], float)
    ha = np.asarray(episode_bundle["ha"], float)

    disturbance_schedule = None
    if run_mode == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = episode_bundle.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    bc_schedule = build_behavioral_cloning_schedule(
        config=weight_cfg.get("behavioral_cloning", {}),
        warm_start_step=warm_start_step,
        time_in_sub_episodes=time_in_sub_episodes,
        n_steps=nFE,
    )
    bc_logs = init_behavioral_cloning_logs(nFE)
    bc_handoff_logs = init_bc_handoff_logs(nFE, action_dim)
    bc_release_gate = init_protected_bc_release_gate(bc_schedule, nFE)
    bc_handoff_enabled = bool(
        td3_like_agent_kind and dict(bc_schedule.get("handoff", {}) or {}).get("enabled", False)
    )
    bc_action_gap_tolerance = float(bc_schedule.get("action_gap_tolerance", 0.0))
    bc_train_start_step = (
        int(bc_schedule.get("start_step", warm_start_step))
        if bool(bc_schedule.get("enabled", False))
        else int(warm_start_step)
    )
    protected_bc_release_enabled = bool(bc_release_gate["state"].get("live_blocking_enabled", False))
    td3_authority_ramp_cfg = dict(weight_cfg.get("td3_authority_ramp", {}) or {}) if td3_like_agent_kind else {}
    td3_authority_ramp_logs = init_td3_authority_ramp_logs(nFE, action_dim)

    phase1 = None
    phase1_action_source_log = None
    policy_action_raw_log = None
    executed_action_raw_log = None
    phase1_train_traces = None
    if td3_like_agent_kind or supervisor_gated_sac_agent_kind:
        phase1 = build_phase1_schedule(
            agent_kind=agent_kind,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            n_steps=nFE,
            test_train_dict=test_train_dict,
            action_freeze_subepisodes=0
            if (protected_bc_release_enabled or bc_handoff_enabled)
            else weight_cfg.get("post_warm_start_action_freeze_subepisodes", 0),
            actor_freeze_subepisodes=0
            if (protected_bc_release_enabled or bc_handoff_enabled)
            else weight_cfg.get("post_warm_start_actor_freeze_subepisodes", 0),
            batch_size=getattr(agent, "batch_size", 1),
            initial_buffer_size=len(getattr(agent, "buffer", [])),
            base_actor_freeze=getattr(agent, "actor_freeze", 0),
            push_start_step=0,
            train_start_step=bc_train_start_step,
        )
        agent.actor_freeze = int(phase1["effective_actor_freeze"])
        if supervisor_gated_sac_agent_kind:
            agent.alpha_freeze = int(max(getattr(agent, "alpha_freeze", 0), agent.actor_freeze))
        phase1_action_source_log = np.zeros(nFE, dtype=int)
        policy_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
        executed_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
        phase1_train_traces = init_phase1_train_traces()

    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug.shape[0])

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    L = compute_observer_gain(mpc_obj.A, mpc_obj.C, poles)

    cont_h = int(weight_cfg.get("cont_h", 1))
    bnds = tuple(
        (float(weight_cfg["b_min"][j]), float(weight_cfg["b_max"][j]))
        for _ in range(cont_h)
        for j in range(n_inputs)
    )
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
    weight_log = np.zeros((nFE, 4))
    weight_requested_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    weight_post_handoff_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    weight_post_cap_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    weight_executed_action_raw_log = np.zeros((nFE, action_dim), dtype=float)
    weight_requested_multiplier_log = np.zeros((nFE, action_dim), dtype=float)
    weight_post_handoff_multiplier_log = np.zeros((nFE, action_dim), dtype=float)
    weight_post_cap_multiplier_log = np.zeros((nFE, action_dim), dtype=float)
    weight_active_cap_log = np.full(nFE, np.nan, dtype=float)
    weight_cap_projection_active_log = np.zeros(nFE, dtype=int)
    weight_cap_projection_norm_log = np.full(nFE, np.nan, dtype=float)
    weight_probation_active_log = np.zeros(nFE, dtype=int)
    weight_probation_trigger_log = np.zeros(nFE, dtype=int)
    weight_probation_reference_reward_log = np.full(nFE, np.nan, dtype=float)
    weight_probation_cooldown_until_subepisode_log = np.zeros(nFE, dtype=int)
    weight_action_source_log = np.zeros(nFE, dtype=int)
    weight_fallback_reason_log = np.zeros(nFE, dtype=int)
    weight_multiplier_saturation_log = np.zeros(nFE, dtype=int)
    weight_multiplier_saturation_count_log = np.zeros(nFE, dtype=int)
    weight_shadow_selected_objective_log = np.full(nFE, np.nan, dtype=float)
    weight_shadow_identity_objective_log = np.full(nFE, np.nan, dtype=float)
    weight_shadow_first_move_delta_norm_log = np.full(nFE, np.nan, dtype=float)
    weight_shadow_selected_first_move_log = np.full((nFE, n_inputs), np.nan, dtype=float)
    weight_shadow_identity_first_move_log = np.full((nFE, n_inputs), np.nan, dtype=float)
    sg_policy_action_raw_log = np.full((nFE, action_dim), np.nan, dtype=float)
    sg_supervisor_action_raw_log = np.full((nFE, action_dim), np.nan, dtype=float)
    sg_executed_action_raw_log = np.full((nFE, action_dim), np.nan, dtype=float)
    sg_previous_action_raw_log = np.full((nFE, action_dim), np.nan, dtype=float)
    sg_selected_source_log = np.zeros(nFE, dtype=int)
    sg_score_policy_log = np.full(nFE, np.nan, dtype=float)
    sg_score_supervisor_log = np.full(nFE, np.nan, dtype=float)
    sg_advantage_log = np.full(nFE, np.nan, dtype=float)
    sg_q1_policy_log = np.full(nFE, np.nan, dtype=float)
    sg_q2_policy_log = np.full(nFE, np.nan, dtype=float)
    sg_q1_supervisor_log = np.full(nFE, np.nan, dtype=float)
    sg_q2_supervisor_log = np.full(nFE, np.nan, dtype=float)
    sg_q_gap_policy_log = np.full(nFE, np.nan, dtype=float)
    sg_q_gap_supervisor_log = np.full(nFE, np.nan, dtype=float)
    warm_reference_rewards = []
    warm_start_subepisodes = int(np.ceil(float(warm_start_step + 1) / float(max(1, time_in_sub_episodes))))
    probation_cooldown_until_subepisode = 0
    innovation_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    innovation_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_scale_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    test = False
    effective_exploration_step_log = np.zeros(nFE, dtype=float)
    exploration_freeze_step = episode_bundle.get("exploration_freeze_step")

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])
        maybe_freeze_agent_exploration(agent, environment_step=i, freeze_step=exploration_freeze_step)
        effective_exploration_step_log[i] = effective_agent_exploration_value(agent, test=test)

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_pred = mpc_obj.C @ xhatdhat[:, i]
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

        policy_action_for_gate = identity_action.copy()
        if td3_like_agent_kind:
            policy_action_for_gate = np.asarray(agent.act_eval(current_rl_state), float).reshape(-1)
            if policy_action_for_gate.size != action_dim or not np.all(np.isfinite(policy_action_for_gate)):
                policy_action_for_gate = identity_action.copy()
        release_info = update_protected_bc_release_gate(
            bc_release_gate["state"],
            bc_release_gate["logs"],
            step_idx=i,
            warm_start_step=warm_start_step,
            policy_action=policy_action_for_gate,
            target_action=identity_action,
        )

        current_subepisode = int(i // time_in_sub_episodes) + 1
        if warm_reference_rewards:
            reference_window = warm_reference_rewards[-probation_reference_warm_episodes:]
            weight_probation_reference_reward_log[i] = float(np.mean(reference_window))
        probation_active = bool(
            reward_probation_enabled
            and current_subepisode > warm_start_subepisodes
            and current_subepisode <= probation_cooldown_until_subepisode
        )
        weight_probation_active_log[i] = int(probation_active)
        weight_probation_cooldown_until_subepisode_log[i] = int(probation_cooldown_until_subepisode)

        sg_previous_action_for_gate = (
            executed_action_raw_log[i - 1, :].copy()
            if supervisor_gated_agent_kind and executed_action_raw_log is not None and i > 0
            else identity_action.copy()
        )
        if supervisor_gated_agent_kind:
            hidden_active = bool(
                phase1 is not None
                and phase1.get("enabled", False)
                and bool(phase1["hidden_window_active_log"][i])
            )
            warm_blocked = bool(i <= (-1 if bc_handoff_enabled else warm_start_step))
            if warm_blocked or hidden_active:
                action_requested = identity_action.copy()
                policy_action_for_log = np.asarray(agent.act_eval(current_rl_state), float).reshape(-1)
                if policy_action_for_log.size != action_dim or not np.all(np.isfinite(policy_action_for_log)):
                    policy_action_for_log = identity_action.copy()
                sg_selected_source = SOURCE_WARM_START if warm_blocked else SOURCE_SUPERVISOR
                nonfinite_selected = False
                source = 0 if warm_blocked else 1
            else:
                sg_decision = agent.select_action_with_supervisor(
                    current_rl_state,
                    supervisor_action=identity_action,
                    previous_action=sg_previous_action_for_gate,
                    explore=not test,
                    test=test,
                )
                action_requested = np.asarray(sg_decision.action, float).reshape(-1)
                policy_action_for_log = np.asarray(sg_decision.policy_action, float).reshape(-1)
                sg_selected_source = int(sg_decision.selected_source)
                nonfinite_selected = bool(sg_selected_source == SOURCE_FALLBACK)
                source = 3 if test else 2
                sg_score_policy_log[i] = float(sg_decision.score_policy)
                sg_score_supervisor_log[i] = float(sg_decision.score_supervisor)
                sg_advantage_log[i] = float(sg_decision.advantage_policy_supervisor)
                sg_q1_policy_log[i] = float(sg_decision.q1_policy)
                sg_q2_policy_log[i] = float(sg_decision.q2_policy)
                sg_q1_supervisor_log[i] = float(sg_decision.q1_supervisor)
                sg_q2_supervisor_log[i] = float(sg_decision.q2_supervisor)
                sg_q_gap_policy_log[i] = float(sg_decision.q_gap_policy)
                sg_q_gap_supervisor_log[i] = float(sg_decision.q_gap_supervisor)
            sg_policy_action_raw_log[i, :] = policy_action_for_log
            sg_supervisor_action_raw_log[i, :] = identity_action
            sg_previous_action_raw_log[i, :] = sg_previous_action_for_gate
            sg_selected_source_log[i] = int(sg_selected_source)
            policy_action_for_gate = policy_action_for_log
            action_decision = SimpleNamespace(
                action=np.asarray(action_requested, float).reshape(-1),
                source=int(source),
                nonfinite_fallback_used=bool(nonfinite_selected),
            )
            policy_action = policy_action_for_log
        else:
            action_decision = select_continuous_action(
                agent=agent,
                state=current_rl_state,
                step=i,
                warm_start_step=-1 if bc_handoff_enabled else warm_start_step,
                test=test,
                baseline_action=identity_action,
                phase1=phase1,
                action_dim=action_dim,
                nonfinite_fallback=fallback_to_identity_on_nonfinite,
            )
            policy_action = action_decision.policy_action
        action_requested = np.asarray(action_decision.action, float).reshape(-1)
        weight_requested_action_raw_log[i, :] = action_requested
        weight_requested_multiplier_log[i, :] = _map_to_bounds(action_requested, low_coef, high_coef).reshape(-1)

        ramp_info = resolve_td3_authority_ramp(
            td3_authority_ramp_cfg,
            step_idx=i,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
        )
        live_gate_blocked = bool(release_info.get("live_blocked", False))
        gate_override = bool(live_gate_blocked and ramp_info["live_enabled"])
        action_for_handoff = identity_action.copy() if (live_gate_blocked and not ramp_info["live_enabled"]) else action_requested
        handoff_info = resolve_bc_handoff_authority(bc_schedule, step_idx=i)
        handoff_td3_action = np.asarray(action_for_handoff, float).reshape(-1)
        action_post_handoff = apply_bc_handoff_action(
            handoff_td3_action,
            identity_action,
            handoff_info["authority"],
        )
        record_bc_handoff_step(
            bc_handoff_logs,
            step_idx=i,
            authority_info=handoff_info,
            safe_action=identity_action,
            td3_action=handoff_td3_action,
            executed_action=action_post_handoff,
        )
        weight_post_handoff_action_raw_log[i, :] = action_post_handoff
        multipliers_post_handoff = _map_to_bounds(action_post_handoff, low_coef, high_coef).reshape(-1)
        weight_post_handoff_multiplier_log[i, :] = multipliers_post_handoff

        active_cap = float(ramp_info["cap"]) if bool(ramp_info["live_enabled"]) else float("nan")
        cap_active = bool(ramp_info["live_enabled"])
        if probation_active:
            cooldown_cap = max(0.0, probation_cooldown_multiplier_cap)
            active_cap = float(min(active_cap, cooldown_cap)) if np.isfinite(active_cap) else float(cooldown_cap)
            cap_active = True
        weight_active_cap_log[i] = active_cap if cap_active else float("nan")
        multipliers_post_cap, ramp_clip_info = apply_symmetric_deviation_cap(
            multipliers_post_handoff,
            center=np.ones(4, dtype=float),
            cap=active_cap,
            low=low_coef,
            high=high_coef,
            active=cap_active,
        )
        action_post_cap = np.clip(_map_from_bounds(multipliers_post_cap, low_coef, high_coef), -1.0, 1.0)
        weight_post_cap_action_raw_log[i, :] = action_post_cap
        weight_post_cap_multiplier_log[i, :] = multipliers_post_cap
        weight_cap_projection_active_log[i] = int(ramp_clip_info["projection_active"])
        weight_cap_projection_norm_log[i] = float(ramp_clip_info["delta_norm"])
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
        multipliers = np.asarray(multipliers_post_cap, float).reshape(-1)
        fallback_reason_code = WEIGHT_FALLBACK_REASON_CODES["none"]
        if bool(action_decision.nonfinite_fallback_used):
            fallback_reason_code = WEIGHT_FALLBACK_REASON_CODES["nonfinite_selected_action"]
        if (
            not np.all(np.isfinite(multipliers))
            or np.any(multipliers < low_coef - 1.0e-9)
            or np.any(multipliers > high_coef + 1.0e-9)
        ):
            action = identity_action.copy()
            multipliers = np.ones(4, dtype=float)
            fallback_reason_code = WEIGHT_FALLBACK_REASON_CODES["invalid_multiplier"]

        if phase1 is not None:
            policy_action_raw_log[i, :] = np.asarray(
                policy_action if policy_action is not None else policy_action_for_gate,
                float,
            ).reshape(-1)
            executed_action_raw_log[i, :] = np.asarray(action, float).reshape(-1)
            hard_blocked = bool(live_gate_blocked) and not ramp_info["live_enabled"]
            phase1_action_source_log[i] = 1 if hard_blocked else int(action_decision.source)

        if fallback_reason_code != WEIGHT_FALLBACK_REASON_CODES["none"] or (
            live_gate_blocked and not ramp_info["live_enabled"]
        ):
            weight_action_source_log[i] = WEIGHT_ACTION_SOURCE_CODES["identity_fallback"]
        elif i <= warm_start_step and float(np.linalg.norm(multipliers - np.ones(4, dtype=float))) <= 1.0e-12:
            weight_action_source_log[i] = WEIGHT_ACTION_SOURCE_CODES["identity_warm"]
        elif supervisor_gated_agent_kind and sg_selected_source_log[i] == SOURCE_SUPERVISOR:
            weight_action_source_log[i] = WEIGHT_ACTION_SOURCE_CODES["identity_fallback"]
        elif bool(ramp_clip_info["projection_active"]):
            weight_action_source_log[i] = WEIGHT_ACTION_SOURCE_CODES["projected_td3"]
        else:
            weight_action_source_log[i] = WEIGHT_ACTION_SOURCE_CODES["td3_accepted"]
        weight_fallback_reason_log[i] = int(fallback_reason_code)
        weight_executed_action_raw_log[i, :] = action
        if supervisor_gated_agent_kind:
            sg_executed_action_raw_log[i, :] = action
        weight_log[i, :] = multipliers
        sat = np.isclose(multipliers, low_coef, atol=1.0e-9) | np.isclose(multipliers, high_coef, atol=1.0e-9)
        weight_multiplier_saturation_count_log[i] = int(np.sum(sat))
        weight_multiplier_saturation_log[i] = int(np.any(sat))
        _set_penalties(mpc_obj, q_base, r_base, multipliers)

        ic_opt_step = ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * cont_h)

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            ic_opt_step,
            bounds=bnds,
            constraints=[],
        )
        selected_candidate_sol = sol
        if fallback_to_identity_on_solve_failure and not _solve_successful(sol):
            action = identity_action.copy()
            multipliers = np.ones(4, dtype=float)
            weight_executed_action_raw_log[i, :] = action
            weight_log[i, :] = multipliers
            weight_multiplier_saturation_count_log[i] = 0
            weight_multiplier_saturation_log[i] = 0
            weight_action_source_log[i] = WEIGHT_ACTION_SOURCE_CODES["identity_fallback"]
            weight_fallback_reason_log[i] = WEIGHT_FALLBACK_REASON_CODES["mpc_solve_failure"]
            if phase1 is not None:
                executed_action_raw_log[i, :] = action
            _set_penalties(mpc_obj, q_base, r_base, multipliers)
            sol = spo.minimize(
                lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
                np.zeros(n_inputs * cont_h),
                bounds=bnds,
                constraints=[],
            )

        if use_shifted_mpc_warm_start:
            ic_opt = shift_control_sequence(sol.x[: n_inputs * cont_h], n_inputs, cont_h)
        else:
            ic_opt = np.zeros(n_inputs * cont_h)

        selected_first_move_scaled_abs = np.asarray(sol.x[:n_inputs], float) + ss_scaled_inputs
        if shadow_identity_enabled and i % shadow_identity_stride == 0:
            weight_shadow_selected_objective_log[i] = (
                float(selected_candidate_sol.fun) if np.isfinite(selected_candidate_sol.fun) else float("nan")
            )
            if np.all(np.isfinite(np.asarray(selected_candidate_sol.x[:n_inputs], float))):
                weight_shadow_selected_first_move_log[i, :] = (
                    np.asarray(selected_candidate_sol.x[:n_inputs], float) + ss_scaled_inputs
                )
            if np.allclose(multipliers, np.ones(4, dtype=float), atol=1.0e-12):
                identity_sol = sol
            else:
                _set_penalties(mpc_obj, q_base, r_base, np.ones(4, dtype=float))
                identity_sol = spo.minimize(
                    lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
                    np.zeros(n_inputs * cont_h),
                    bounds=bnds,
                    constraints=[],
                )
                _set_penalties(mpc_obj, q_base, r_base, multipliers)
            weight_shadow_identity_objective_log[i] = (
                float(identity_sol.fun) if np.isfinite(identity_sol.fun) else float("nan")
            )
            identity_first_move_scaled_abs = np.asarray(identity_sol.x[:n_inputs], float) + ss_scaled_inputs
            weight_shadow_identity_first_move_log[i, :] = identity_first_move_scaled_abs
            if np.all(np.isfinite(weight_shadow_selected_first_move_log[i, :])):
                weight_shadow_first_move_delta_norm_log[i] = float(
                    np.linalg.norm(weight_shadow_selected_first_move_log[i, :] - identity_first_move_scaled_abs)
                )

        u_mpc[i, :] = selected_first_move_scaled_abs
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
            A=mpc_obj.A,
            B=mpc_obj.B,
            C=mpc_obj.C,
            L=L,
            x_prev=xhatdhat[:, i],
            u_dev=(u_mpc[i, :] - ss_scaled_inputs),
            y_prev_scaled=y_prev_scaled,
            y_current_scaled=y_current_scaled,
            observer_update_alignment=observer_update_alignment,
        )

        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_mpc[i, :] - ss_scaled_inputs
        yhat_next_pred = mpc_obj.C @ xhatdhat[:, i + 1]
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

        bc_context = None
        if td3_like_agent_kind and not test:
            if float(np.max(np.abs(policy_action_for_gate - identity_action))) > bc_action_gap_tolerance:
                bc_context = resolve_behavioral_cloning_context(
                    bc_schedule,
                    step_idx=i,
                    target_action=identity_action,
                    policy_action=policy_action_for_gate,
                )

        if supervisor_gated_agent_kind:
            train_result = {"pushed": False, "trained": False, "train_meta": None}
            if not test:
                agent.push_supervised(
                    np.asarray(current_rl_state, np.float32),
                    np.asarray(action, np.float32),
                    float(reward),
                    np.asarray(next_rl_state, np.float32),
                    False,
                    policy_action=sg_policy_action_raw_log[i, :],
                    supervisor_action=identity_action,
                    previous_action=sg_previous_action_for_gate,
                    selected_source=int(sg_selected_source_log[i]),
                    score_policy=sg_score_policy_log[i],
                    score_supervisor=sg_score_supervisor_log[i],
                    advantage_policy_supervisor=sg_advantage_log[i],
                )
                train_result["pushed"] = True
                if i >= bc_train_start_step:
                    train_meta = agent.train_step(bc_context=bc_context)
                    train_result["trained"] = True
                    train_result["train_meta"] = train_meta
                    if phase1_train_traces is not None:
                        record_phase1_train_step(phase1_train_traces, i, train_meta)
        else:
            train_result = replay_train_continuous_agent(
                agent=agent,
                state=current_rl_state,
                action=action,
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
            policy_action=policy_action_for_gate,
            target_action=identity_action,
            target_mode=str(bc_schedule.get("target_mode", "nominal_only")),
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
                    weight_probation_trigger_log[i] = 1
                    weight_probation_reference_reward_log[i] = reference_reward
                    weight_probation_cooldown_until_subepisode_log[i] = int(
                        probation_cooldown_until_subepisode
                    )
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_rewards[-1],
                "| avg multipliers:",
                np.mean(weight_log[max(0, i - time_in_sub_episodes + 1) : i + 1, :], axis=0),
            )

    _set_penalties(mpc_obj, q_base, r_base, np.ones(4, dtype=float))
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
        "method_family": "weights",
        "algorithm": agent_kind,
        "state_mode": state_mode,
        "system_metadata": system_metadata,
        "notebook_source": weight_cfg.get("notebook_source"),
        "config_snapshot": dict(weight_cfg),
        "seed": weight_cfg.get("seed"),
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
        "weight_log": weight_log,
        "weight_safety": dict(weight_safety_cfg),
        "weight_safety_enabled": bool(weight_safety_enabled),
        "weight_action_source_codes": dict(WEIGHT_ACTION_SOURCE_CODES),
        "weight_action_source_log": weight_action_source_log,
        "weight_fallback_reason_codes": dict(WEIGHT_FALLBACK_REASON_CODES),
        "weight_fallback_reason_log": weight_fallback_reason_log,
        "weight_requested_action_raw_log": weight_requested_action_raw_log,
        "weight_post_handoff_action_raw_log": weight_post_handoff_action_raw_log,
        "weight_post_cap_action_raw_log": weight_post_cap_action_raw_log,
        "weight_executed_action_raw_log": weight_executed_action_raw_log,
        "weight_requested_multiplier_log": weight_requested_multiplier_log,
        "weight_post_handoff_multiplier_log": weight_post_handoff_multiplier_log,
        "weight_post_cap_multiplier_log": weight_post_cap_multiplier_log,
        "weight_active_cap_log": weight_active_cap_log,
        "weight_cap_projection_active_log": weight_cap_projection_active_log,
        "weight_cap_projection_norm_log": weight_cap_projection_norm_log,
        "weight_probation_active_log": weight_probation_active_log,
        "weight_probation_trigger_log": weight_probation_trigger_log,
        "weight_probation_reference_reward_log": weight_probation_reference_reward_log,
        "weight_probation_cooldown_until_subepisode_log": weight_probation_cooldown_until_subepisode_log,
        "weight_multiplier_saturation_log": weight_multiplier_saturation_log,
        "weight_multiplier_saturation_count_log": weight_multiplier_saturation_count_log,
        "weight_shadow_identity_mpc_enabled": bool(shadow_identity_enabled),
        "weight_shadow_diagnostic_stride": int(shadow_identity_stride),
        "weight_shadow_selected_objective_log": weight_shadow_selected_objective_log,
        "weight_shadow_identity_objective_log": weight_shadow_identity_objective_log,
        "weight_shadow_first_move_delta_norm_log": weight_shadow_first_move_delta_norm_log,
        "weight_shadow_selected_first_move_log": weight_shadow_selected_first_move_log,
        "weight_shadow_identity_first_move_log": weight_shadow_identity_first_move_log,
        "supervisor_gated_td3_enabled": bool(supervisor_gated_td3_agent_kind),
        "supervisor_gated_sac_enabled": bool(supervisor_gated_sac_agent_kind),
        "supervisor_gated_algorithm": agent_kind if supervisor_gated_agent_kind else None,
        "sg_policy_action_raw_log": sg_policy_action_raw_log if supervisor_gated_agent_kind else None,
        "sg_supervisor_action_raw_log": sg_supervisor_action_raw_log if supervisor_gated_agent_kind else None,
        "sg_executed_action_raw_log": sg_executed_action_raw_log if supervisor_gated_agent_kind else None,
        "sg_previous_action_raw_log": sg_previous_action_raw_log if supervisor_gated_agent_kind else None,
        "sg_selected_source_log": sg_selected_source_log if supervisor_gated_agent_kind else None,
        "sg_score_policy_log": sg_score_policy_log if supervisor_gated_agent_kind else None,
        "sg_score_supervisor_log": sg_score_supervisor_log if supervisor_gated_agent_kind else None,
        "sg_advantage_log": sg_advantage_log if supervisor_gated_agent_kind else None,
        "sg_q1_policy_log": sg_q1_policy_log if supervisor_gated_agent_kind else None,
        "sg_q2_policy_log": sg_q2_policy_log if supervisor_gated_agent_kind else None,
        "sg_q1_supervisor_log": sg_q1_supervisor_log if supervisor_gated_agent_kind else None,
        "sg_q2_supervisor_log": sg_q2_supervisor_log if supervisor_gated_agent_kind else None,
        "sg_q_gap_policy_log": sg_q_gap_policy_log if supervisor_gated_agent_kind else None,
        "sg_q_gap_supervisor_log": sg_q_gap_supervisor_log if supervisor_gated_agent_kind else None,
        "sg_rl_selected_fraction": (
            float(np.mean(sg_selected_source_log == SOURCE_POLICY)) if supervisor_gated_agent_kind and nFE else None
        ),
        "sg_supervisor_selected_fraction": (
            float(np.mean(sg_selected_source_log == SOURCE_SUPERVISOR))
            if supervisor_gated_agent_kind and nFE
            else None
        ),
        "low_coef": low_coef,
        "high_coef": high_coef,
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "warm_start_step": int(warm_start_step),
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
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
            int(weight_cfg["predict_h"]),
            int(weight_cfg["cont_h"]),
        )
        if "predict_h" in weight_cfg and "cont_h" in weight_cfg
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
        "sg_score_policy_trace",
        "sg_score_supervisor_trace",
        "sg_advantage_trace",
        "sg_selected_source_trace",
        "sg_q_policy_trace",
        "sg_q_supervisor_trace",
        "sg_q_gap_policy_trace",
        "sg_q_gap_supervisor_trace",
        "sg_weight_trace",
        "sg_bc_loss_trace",
        "sg_sampled_bc_loss_trace",
        "sg_smooth_loss_trace",
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
    result_bundle.update(episode_profile_result_fields(episode_bundle))
    result_bundle.update(exploration_freeze_result_fields(agent, effective_exploration_step_log))

    attach_single_agent_replay_snapshot(result_bundle, agent)
    return result_bundle
