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
from utils.observer import compute_observer_gain
from utils.observation_conditioning import update_observer_state
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
    run_mode = str(weight_cfg["run_mode"]).lower()
    state_mode = str(weight_cfg.get("state_mode", "standard")).lower()
    if agent_kind not in {"td3", "sac"}:
        raise ValueError("weight_cfg['agent_kind'] must be 'td3' or 'sac'.")
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
        int(weight_cfg["n_tests"]),
        int(weight_cfg["set_points_len"]),
        int(weight_cfg["warm_start"]),
        list(weight_cfg["test_cycle"]),
        float(weight_cfg["nominal_qi"]),
        float(weight_cfg["nominal_qs"]),
        float(weight_cfg["nominal_ha"]),
        float(weight_cfg["qi_change"]),
        float(weight_cfg["qs_change"]),
        float(weight_cfg["ha_change"]),
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
            action_freeze_subepisodes=weight_cfg.get("post_warm_start_action_freeze_subepisodes", 0),
            actor_freeze_subepisodes=weight_cfg.get("post_warm_start_actor_freeze_subepisodes", 0),
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
    innovation_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    innovation_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_raw_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_scale_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
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
                    policy_action = identity_action.copy()
            if phase1_hidden_active:
                action = identity_action.copy()
            elif not test:
                action = np.asarray(agent.take_action(current_rl_state, explore=True), float).reshape(-1)
            else:
                action = policy_action.copy() if policy_action is not None else np.asarray(agent.act_eval(current_rl_state), float).reshape(-1)
        else:
            action = identity_action.copy()
            if phase1 is not None:
                policy_action = identity_action.copy()

        if action.size != action_dim:
            raise ValueError("weights runner expects action_dim == 4.")

        if phase1 is not None:
            policy_action_raw_log[i, :] = np.asarray(
                policy_action if policy_action is not None else identity_action,
                float,
            ).reshape(-1)
            executed_action_raw_log[i, :] = np.asarray(action, float).reshape(-1)
            phase1_action_source_log[i] = int(
                resolve_phase1_action_source(
                    i,
                    warm_start_step,
                    phase1_hidden_active,
                    test,
                )
            )

        multipliers = _map_to_bounds(action, low_coef, high_coef).reshape(-1)
        weight_log[i, :] = multipliers
        _set_penalties(mpc_obj, q_base, r_base, multipliers)

        ic_opt_step = ic_opt if use_shifted_mpc_warm_start else np.zeros(n_inputs * cont_h)

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            ic_opt_step,
            bounds=bnds,
            constraints=[],
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

        if not test:
            agent.push(
                current_rl_state,
                np.asarray(action, np.float32),
                reward,
                next_rl_state,
                0.0,
            )
            if i >= warm_start_step:
                train_meta = agent.train_step()
                if phase1 is not None:
                    record_phase1_train_step(phase1_train_traces, i, train_meta)

        if i in sub_episodes_changes_dict:
            avg_rewards.append(float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1])))
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
    ):
        if hasattr(agent, attr):
            result_bundle[attr] = np.asarray(getattr(agent, attr), float)
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
