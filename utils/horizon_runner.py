import numpy as np
import scipy.optimize as spo

from Simulation.mpc import MpcSolverGeneral
from utils.agent_step_runtime import (
    DISCRETE_SUPERVISOR_SOURCE_NAMES,
    replay_train_horizon_agent,
    replay_train_supervisor_gated_horizon_agent,
    select_horizon_action,
    select_supervisor_gated_horizon_action,
)
from DQN.supervisor_gated_dqn_agent import SOURCE_POLICY, SOURCE_SUPERVISOR
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
from utils.horizon_safety import (
    build_horizon_safety_bundle_fields,
    init_horizon_safety_logs,
    record_horizon_safety_step,
    resolve_horizon_safety,
)
from utils.observer import compute_observer_gain
from utils.observation_conditioning import update_observer_state
from utils.phase1_hidden_release import (
    ACTION_SOURCE_PHASE1_HIDDEN_BASELINE,
    ACTION_SOURCE_WARM_START_BASELINE,
)
from utils.replay_snapshot import attach_single_agent_replay_snapshot
from utils.state_features import (
    build_rl_state,
    compute_tracking_scale_now,
    make_state_conditioner_from_settings,
    resolve_mismatch_settings,
)


def run_dqn_mpc_horizon_supervisor(horizon_cfg, runtime_ctx):
    """
    Run the DQN-assisted horizon supervisor and return a normalized result bundle.

    Parameters
    ----------
    horizon_cfg : dict
        Runtime config assembled in the notebook. Required keys:
        mode, predict_h, cont_h, decision_interval, warm_start, test_cycle,
        n_tests, set_points_len, nominal_qi, nominal_qs, nominal_ha,
        qi_change, qs_change, ha_change, b_min, b_max,
        Q1_penalty, Q2_penalty, R1_penalty, R2_penalty.
    runtime_ctx : dict
        Prepared objects and shared data. Required keys:
        system, y_sp_scenario, steady_states, min_max_dict, agent,
        A_aug, B_aug, C_aug, poles/L, data_min, data_max, horizon_recipes, reward_fn.
    """

    system = runtime_ctx["system"]
    y_sp_scenario = np.asarray(runtime_ctx["y_sp_scenario"], float)
    steady_states = runtime_ctx["steady_states"]
    min_max_dict = runtime_ctx["min_max_dict"]
    agent = runtime_ctx["agent"]
    A_aug = np.asarray(runtime_ctx["A_aug"], float)
    B_aug = np.asarray(runtime_ctx["B_aug"], float)
    C_aug = np.asarray(runtime_ctx["C_aug"], float)
    L = runtime_ctx.get("L")
    if L is None:
        poles = np.asarray(runtime_ctx["poles"], float)
        L = compute_observer_gain(A_aug, C_aug, poles)
    else:
        L = np.asarray(L, float)
    data_min = np.asarray(runtime_ctx["data_min"], float)
    data_max = np.asarray(runtime_ctx["data_max"], float)
    h_recipes = list(runtime_ctx["horizon_recipes"])
    reward_fn = runtime_ctx["reward_fn"]
    reward_params = runtime_ctx.get("reward_params", {})
    system_stepper = runtime_ctx.get("system_stepper")
    system_metadata = runtime_ctx.get("system_metadata")
    disturbance_labels = runtime_ctx.get("disturbance_labels")

    mode = horizon_cfg["mode"]
    state_mode = str(horizon_cfg.get("state_mode", "standard")).lower()
    predict_h = int(horizon_cfg["predict_h"])
    cont_h = int(horizon_cfg["cont_h"])
    decision_interval = int(horizon_cfg["decision_interval"])
    use_shifted_mpc_warm_start = bool(horizon_cfg.get("use_shifted_mpc_warm_start", False))
    mismatch_cfg = resolve_mismatch_settings(
        state_mode=state_mode,
        mismatch_cfg=horizon_cfg,
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

    y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, warm_start_step, qi, qs, ha = (
        generate_setpoints_training_rl_gradually(
            y_sp_scenario,
            int(horizon_cfg["n_tests"]),
            int(horizon_cfg["set_points_len"]),
            int(horizon_cfg["warm_start"]),
            list(horizon_cfg["test_cycle"]),
            float(horizon_cfg["nominal_qi"]),
            float(horizon_cfg["nominal_qs"]),
            float(horizon_cfg["nominal_ha"]),
            float(horizon_cfg["qi_change"]),
            float(horizon_cfg["qs_change"]),
            float(horizon_cfg["ha_change"]),
        )
    )

    disturbance_schedule = None
    if mode == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    n_inputs = B_aug.shape[1]
    n_outputs = C_aug.shape[0]
    n_states = A_aug.shape[0]

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    y_system = np.zeros((nFE + 1, n_outputs))
    y_system[0, :] = system.current_output
    u_mpc = np.zeros((nFE, n_inputs))
    rewards = np.zeros(nFE)
    avg_rewards = []
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    horizon_trace = np.zeros((nFE, 2), dtype=int)
    action_trace = np.zeros(nFE, dtype=int)
    delta_y_storage = np.zeros((nFE, n_outputs))
    delta_u_storage = np.zeros((nFE, n_inputs))
    innovation_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    innovation_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_scale_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None

    last_action = None
    current_Hp, current_Hc = predict_h, cont_h
    test = False

    mpc_obj = MpcSolverGeneral(
        A_aug,
        B_aug,
        C_aug,
        Q_out=np.array([horizon_cfg["Q1_penalty"], horizon_cfg["Q2_penalty"]], float),
        R_in=np.array([horizon_cfg["R1_penalty"], horizon_cfg["R2_penalty"]], float),
        NP=predict_h,
        NC=cont_h,
    )

    b1 = (float(horizon_cfg["b_min"][0]), float(horizon_cfg["b_max"][0]))
    b2 = (float(horizon_cfg["b_min"][1]), float(horizon_cfg["b_max"][1]))
    cons = []

    def rebuild_mpc(Hp, Hc):
        return MpcSolverGeneral(
            A_aug,
            B_aug,
            C_aug,
            Q_out=np.array([horizon_cfg["Q1_penalty"], horizon_cfg["Q2_penalty"]], float),
            R_in=np.array([horizon_cfg["R1_penalty"], horizon_cfg["R2_penalty"]], float),
            NP=int(Hp),
            NC=int(Hc),
        )

    default_action = [idx for idx, recipe in enumerate(h_recipes) if recipe == (predict_h, cont_h)]
    if not default_action:
        raise ValueError("Default (predict_h, cont_h) is not present in horizon_recipes.")
    default_action = int(default_action[0])
    agent_kind = str(horizon_cfg.get("agent_kind", horizon_cfg.get("algorithm", "ddqn"))).strip().lower()
    supervisor_gated_horizon = bool(agent_kind in {"sg_dqn", "supervisor_gated_dqn"})
    current_ic_opt = np.zeros(n_inputs * int(current_Hc))
    horizon_safety_cfg = dict(horizon_cfg.get("horizon_safety", {}) or {})
    horizon_safety_logs = init_horizon_safety_logs(nFE)
    post_warm_action_freeze_subepisodes = int(
        max(0, horizon_cfg.get("post_warm_start_action_freeze_subepisodes", 0))
    )
    post_warm_action_freeze_steps = int(post_warm_action_freeze_subepisodes * time_in_sub_episodes)
    post_warm_action_freeze_end_step = int(warm_start_step + post_warm_action_freeze_steps)
    horizon_action_source_log = np.zeros(nFE, dtype=int)
    horizon_decision_log = np.zeros(nFE, dtype=int)
    horizon_q_warm_release_active_log = np.zeros(nFE, dtype=int)
    reward_probation_cfg = dict(horizon_safety_cfg.get("reward_probation", {}) or {})
    reward_probation_enabled = bool(horizon_safety_cfg.get("enabled", False) and reward_probation_cfg.get("enabled", False))
    probation_reference_warm_episodes = int(max(1, reward_probation_cfg.get("reference_warm_episodes", 3)))
    probation_collapse_threshold = float(reward_probation_cfg.get("collapse_threshold", 5.0))
    probation_cooldown_subepisodes = int(max(0, reward_probation_cfg.get("cooldown_subepisodes", 0)))
    warm_reference_rewards = []
    warm_start_subepisodes = int(np.ceil(float(warm_start_step + 1) / float(max(1, time_in_sub_episodes))))
    probation_cooldown_until_subepisode = 0
    horizon_reward_probation_trigger_log = np.zeros(nFE, dtype=int)
    horizon_reward_probation_reference_reward_log = np.full(nFE, np.nan, dtype=float)
    horizon_reward_probation_cooldown_until_subepisode_log = np.zeros(nFE, dtype=int)
    horizon_change_log = np.zeros(nFE, dtype=int)
    horizon_subepisode_switch_count_log = np.zeros(nFE, dtype=int)
    horizon_shadow_cfg = dict(horizon_safety_cfg.get("shadow_default_mpc", {}) or {})
    horizon_shadow_enabled = bool(horizon_safety_cfg.get("enabled", False) and horizon_shadow_cfg.get("enabled", False))
    horizon_shadow_stride = int(max(1, horizon_shadow_cfg.get("diagnostic_stride", 4)))
    horizon_shadow_selected_objective_log = np.full(nFE, np.nan, dtype=float)
    horizon_shadow_default_objective_log = np.full(nFE, np.nan, dtype=float)
    horizon_shadow_first_move_delta_norm_log = np.full(nFE, np.nan, dtype=float)
    horizon_shadow_selected_first_move_log = np.full((nFE, n_inputs), np.nan, dtype=float)
    horizon_shadow_default_first_move_log = np.full((nFE, n_inputs), np.nan, dtype=float)
    sg_policy_action_log = np.full(nFE, -1, dtype=int) if supervisor_gated_horizon else None
    sg_supervisor_action_log = np.full(nFE, -1, dtype=int) if supervisor_gated_horizon else None
    sg_executed_action_log = np.full(nFE, -1, dtype=int) if supervisor_gated_horizon else None
    sg_previous_action_log = np.full(nFE, -1, dtype=int) if supervisor_gated_horizon else None
    sg_selected_source_log = np.full(nFE, -1, dtype=int) if supervisor_gated_horizon else None
    sg_score_policy_log = np.full(nFE, np.nan, dtype=float) if supervisor_gated_horizon else None
    sg_score_supervisor_log = np.full(nFE, np.nan, dtype=float) if supervisor_gated_horizon else None
    sg_advantage_log = np.full(nFE, np.nan, dtype=float) if supervisor_gated_horizon else None
    sg_q_policy_log = np.full(nFE, np.nan, dtype=float) if supervisor_gated_horizon else None
    sg_q_supervisor_log = np.full(nFE, np.nan, dtype=float) if supervisor_gated_horizon else None
    previous_executed_pair = None
    subepisode_switch_count = 0

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_pred = mpc_obj.C @ xhatdhat[:, i]
        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        band_scaled_now = None
        tracking_scale_now = None
        if state_mode == "mismatch":
            band_scaled_now, tracking_scale_now = compute_tracking_scale_now(
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

        if supervisor_gated_horizon:
            horizon_decision = select_supervisor_gated_horizon_action(
                agent=agent,
                state=current_rl_state,
                step=i,
                warm_start_step=warm_start_step,
                decision_interval=decision_interval,
                default_action=default_action,
                supervisor_action=default_action,
                last_action=last_action,
                test=test,
                post_warm_action_freeze_steps=post_warm_action_freeze_steps,
            )
        else:
            horizon_decision = select_horizon_action(
                agent=agent,
                state=current_rl_state,
                step=i,
                warm_start_step=warm_start_step,
                decision_interval=decision_interval,
                default_action=default_action,
                last_action=last_action,
                test=test,
                post_warm_action_freeze_steps=post_warm_action_freeze_steps,
            )
        horizon_action_source_log[i] = int(horizon_decision.source)
        horizon_decision_log[i] = int(horizon_decision.decision_taken)
        if supervisor_gated_horizon:
            sg_policy_action_log[i] = int(horizon_decision.policy_action)
            sg_supervisor_action_log[i] = int(horizon_decision.supervisor_action)
            sg_previous_action_log[i] = int(horizon_decision.previous_action)
            sg_selected_source_log[i] = int(horizon_decision.selected_source)
            sg_score_policy_log[i] = float(horizon_decision.score_policy)
            sg_score_supervisor_log[i] = float(horizon_decision.score_supervisor)
            sg_advantage_log[i] = float(horizon_decision.advantage_policy_supervisor)
            sg_q_policy_log[i] = float(horizon_decision.q_policy)
            sg_q_supervisor_log[i] = float(horizon_decision.q_supervisor)
        if warm_start_step < i <= post_warm_action_freeze_end_step:
            horizon_q_warm_release_active_log[i] = 1
        requested_a_idx = int(horizon_decision.action)
        safety_info = resolve_horizon_safety(
            horizon_safety_cfg,
            step_idx=i,
            warm_start_step=warm_start_step,
            time_in_sub_episodes=time_in_sub_episodes,
            requested_action=requested_a_idx,
            horizon_recipes=h_recipes,
            default_action=default_action,
            cooldown_until_subepisode=probation_cooldown_until_subepisode,
        )
        record_horizon_safety_step(horizon_safety_logs, step_idx=i, safety_info=safety_info)
        horizon_reward_probation_cooldown_until_subepisode_log[i] = int(probation_cooldown_until_subepisode)
        a_idx = int(safety_info["executed_action"])
        if int(horizon_decision.source) in {
            ACTION_SOURCE_WARM_START_BASELINE,
            ACTION_SOURCE_PHASE1_HIDDEN_BASELINE,
        }:
            last_action = None
        else:
            last_action = a_idx
        if supervisor_gated_horizon:
            sg_executed_action_log[i] = int(a_idx)
        Hp, Hc = action_to_horizons(h_recipes, a_idx)
        executed_pair = (int(Hp), int(Hc))
        if previous_executed_pair is not None and executed_pair != previous_executed_pair:
            horizon_change_log[i] = 1
            subepisode_switch_count += 1
        horizon_subepisode_switch_count_log[i] = int(subepisode_switch_count)
        previous_executed_pair = executed_pair
        if (Hp, Hc) != (current_Hp, current_Hc):
            mpc_obj = rebuild_mpc(Hp, Hc)
            current_Hp, current_Hc = Hp, Hc
            current_ic_opt = np.zeros(n_inputs * int(current_Hc))

        action_trace[i] = a_idx
        horizon_trace[i] = (Hp, Hc)

        bnds = (b1, b2) * int(Hc)

        ic_opt = current_ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * int(Hc))

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            ic_opt,
            bounds=bnds,
            constraints=cons,
        )
        if use_shifted_mpc_warm_start:
            current_ic_opt = shift_control_sequence(sol.x[: n_inputs * int(current_Hc)], n_inputs, int(current_Hc))
        else:
            current_ic_opt = np.zeros(n_inputs * int(current_Hc))

        selected_first_move_scaled_abs = np.asarray(sol.x[:n_inputs], float) + ss_scaled_inputs
        if horizon_shadow_enabled and i % horizon_shadow_stride == 0:
            horizon_shadow_selected_objective_log[i] = float(sol.fun) if np.isfinite(sol.fun) else float("nan")
            horizon_shadow_selected_first_move_log[i, :] = selected_first_move_scaled_abs
            if (int(Hp), int(Hc)) == (int(predict_h), int(cont_h)):
                default_first_move_scaled_abs = selected_first_move_scaled_abs.copy()
                horizon_shadow_default_objective_log[i] = horizon_shadow_selected_objective_log[i]
            else:
                default_mpc_obj = rebuild_mpc(predict_h, cont_h)
                default_bnds = (b1, b2) * int(cont_h)
                default_ic = np.zeros(n_inputs * int(cont_h))
                default_sol = spo.minimize(
                    lambda x: default_mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
                    default_ic,
                    bounds=default_bnds,
                    constraints=cons,
                )
                default_first_move_scaled_abs = np.asarray(default_sol.x[:n_inputs], float) + ss_scaled_inputs
                horizon_shadow_default_objective_log[i] = (
                    float(default_sol.fun) if np.isfinite(default_sol.fun) else float("nan")
                )
            horizon_shadow_default_first_move_log[i, :] = default_first_move_scaled_abs
            horizon_shadow_first_move_delta_norm_log[i] = float(
                np.linalg.norm(selected_first_move_scaled_abs - default_first_move_scaled_abs)
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

        y_system[i + 1, :] = system.current_output

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

        reward = reward_fn(delta_y, delta_u, y_sp_phys)
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
        done = 0.0

        if supervisor_gated_horizon:
            replay_train_supervisor_gated_horizon_agent(
                agent=agent,
                state=current_rl_state,
                action=a_idx,
                reward=reward,
                next_state=next_rl_state,
                done=done,
                step=i,
                test=test,
                replay_start_step=time_in_sub_episodes,
                train_start_step=warm_start_step,
                decision=horizon_decision,
            )
        else:
            replay_train_horizon_agent(
                agent=agent,
                state=current_rl_state,
                action=a_idx,
                reward=reward,
                next_state=next_rl_state,
                done=done,
                step=i,
                test=test,
                replay_start_step=time_in_sub_episodes,
                train_start_step=warm_start_step,
            )

        if i in sub_episodes_changes_dict:
            avg_reward = float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1): i + 1]))
            avg_rewards.append(avg_reward)
            completed_subepisode = int(i // time_in_sub_episodes) + 1
            if completed_subepisode <= warm_start_subepisodes:
                warm_reference_rewards.append(float(avg_reward))
            elif reward_probation_enabled and warm_reference_rewards:
                reference_window = warm_reference_rewards[-probation_reference_warm_episodes:]
                reference_reward = float(np.mean(reference_window))
                collapsed = bool(float(avg_reward) < reference_reward - probation_collapse_threshold)
                if collapsed and probation_cooldown_subepisodes > 0:
                    probation_cooldown_until_subepisode = max(
                        probation_cooldown_until_subepisode,
                        completed_subepisode + probation_cooldown_subepisodes,
                    )
                    horizon_reward_probation_trigger_log[i] = 1
                    horizon_reward_probation_reference_reward_log[i] = reference_reward
                    horizon_reward_probation_cooldown_until_subepisode_log[i] = int(
                        probation_cooldown_until_subepisode
                    )
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_reward,
                "| Hp,Hc:",
                (int(Hp), int(Hc)),
            )
            subepisode_switch_count = 0

    u_rl = reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])
    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )

    result_bundle = {
        "mode": mode,
        "run_mode": mode,
        "method_family": "horizon",
        "algorithm": str(horizon_cfg.get("algorithm", "ddqn")).lower(),
        "agent_kind": agent_kind,
        "state_mode": state_mode,
        "system_metadata": system_metadata,
        "notebook_source": horizon_cfg.get("notebook_source"),
        "config_snapshot": dict(horizon_cfg),
        "seed": horizon_cfg.get("seed"),
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
        "horizon_trace": horizon_trace,
        "action_trace": action_trace,
        "horizon_recipes": h_recipes,
        "horizon_action_source_log": horizon_action_source_log,
        "horizon_action_source_codes": {
            "warm_default": 0,
            "q_warm_release_default": 1,
            "policy_train_live": 2,
            "policy_eval_live": 3,
            "held_interval": 4,
        },
        "horizon_decision_log": horizon_decision_log,
        "horizon_q_warm_release_enabled": bool(post_warm_action_freeze_subepisodes > 0),
        "horizon_q_warm_release_subepisodes": int(post_warm_action_freeze_subepisodes),
        "horizon_q_warm_release_steps": int(post_warm_action_freeze_steps),
        "horizon_q_warm_release_end_step": int(post_warm_action_freeze_end_step),
        "horizon_q_warm_release_active_log": horizon_q_warm_release_active_log,
        "horizon_reward_probation_enabled": bool(reward_probation_enabled),
        "horizon_reward_probation_trigger_log": horizon_reward_probation_trigger_log,
        "horizon_reward_probation_reference_reward_log": horizon_reward_probation_reference_reward_log,
        "horizon_reward_probation_cooldown_until_subepisode_log": horizon_reward_probation_cooldown_until_subepisode_log,
        "horizon_change_log": horizon_change_log,
        "horizon_subepisode_switch_count_log": horizon_subepisode_switch_count_log,
        "horizon_shadow_default_mpc_enabled": bool(horizon_shadow_enabled),
        "horizon_shadow_diagnostic_stride": int(horizon_shadow_stride),
        "horizon_shadow_selected_objective_log": horizon_shadow_selected_objective_log,
        "horizon_shadow_default_objective_log": horizon_shadow_default_objective_log,
        "horizon_shadow_first_move_delta_norm_log": horizon_shadow_first_move_delta_norm_log,
        "horizon_shadow_selected_first_move_log": horizon_shadow_selected_first_move_log,
        "horizon_shadow_default_first_move_log": horizon_shadow_default_first_move_log,
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "mpc_horizons": (predict_h, cont_h),
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
        "warm_start_step": int(warm_start_step),
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
    }
    if supervisor_gated_horizon:
        valid_sg = sg_selected_source_log >= 0
        sg_den = int(max(1, int(np.sum(valid_sg))))
        result_bundle.update(
            {
                "supervisor_gated_dqn_enabled": True,
                "horizon_supervisor_action": int(default_action),
                "horizon_supervisor_pair": tuple(int(v) for v in h_recipes[int(default_action)]),
                "sg_source_names": dict(DISCRETE_SUPERVISOR_SOURCE_NAMES),
                "sg_policy_action_log": sg_policy_action_log,
                "sg_supervisor_action_log": sg_supervisor_action_log,
                "sg_executed_action_log": sg_executed_action_log,
                "sg_previous_action_log": sg_previous_action_log,
                "sg_selected_source_log": sg_selected_source_log,
                "sg_score_policy_log": sg_score_policy_log,
                "sg_score_supervisor_log": sg_score_supervisor_log,
                "sg_advantage_log": sg_advantage_log,
                "sg_q_policy_log": sg_q_policy_log,
                "sg_q_supervisor_log": sg_q_supervisor_log,
                "sg_policy_fraction": float(np.sum(sg_selected_source_log[valid_sg] == SOURCE_POLICY) / sg_den),
                "sg_supervisor_fraction": float(
                    np.sum(sg_selected_source_log[valid_sg] == SOURCE_SUPERVISOR) / sg_den
                ),
            }
        )

    diagnostics = {
        "dqn_loss_trace": getattr(agent, "loss_history", None),
        "exploration_trace": getattr(agent, "exploration_trace", None),
        "epsilon_trace": getattr(agent, "epsilon_trace", None),
        "avg_td_error_trace": getattr(agent, "avg_td_error_trace", None),
        "avg_max_q_trace": getattr(agent, "avg_max_q_trace", None),
        "avg_chosen_q_trace": getattr(agent, "avg_chosen_q_trace", None),
        "noisy_sigma_trace": getattr(agent, "noisy_sigma_trace", None),
        "reward_n_mean_trace": getattr(agent, "reward_n_mean_trace", None),
        "discount_n_mean_trace": getattr(agent, "discount_n_mean_trace", None),
        "bootstrap_q_mean_trace": getattr(agent, "bootstrap_q_mean_trace", None),
        "n_actual_mean_trace": getattr(agent, "n_actual_mean_trace", None),
        "truncated_fraction_trace": getattr(agent, "truncated_fraction_trace", None),
        "lambda_return_mean_trace": getattr(agent, "lambda_return_mean_trace", None),
        "offpolicy_rho_mean_trace": getattr(agent, "offpolicy_rho_mean_trace", None),
        "offpolicy_c_mean_trace": getattr(agent, "offpolicy_c_mean_trace", None),
        "behavior_logprob_mean_trace": getattr(agent, "behavior_logprob_mean_trace", None),
        "retrace_c_clip_fraction_trace": getattr(agent, "retrace_c_clip_fraction_trace", None),
    }
    for key, value in diagnostics.items():
        if value is not None:
            result_bundle[key] = np.asarray(value, float).reshape(-1)
    result_bundle.update(build_horizon_safety_bundle_fields(horizon_safety_cfg, horizon_safety_logs))

    attach_single_agent_replay_snapshot(result_bundle, agent)
    return result_bundle
