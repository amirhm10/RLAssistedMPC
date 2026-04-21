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
from utils.replay_snapshot import attach_single_agent_replay_snapshot
from utils.residual_authority import compute_residual_rho, map_from_bounds, project_residual_action
from utils.state_features import (
    build_rl_state,
    compute_tracking_scale_now,
    make_state_conditioner_from_settings,
    resolve_mismatch_settings,
)


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
    authority_use_rho = bool(residual_cfg.get("authority_use_rho", residual_cfg.get("use_rho_authority", True)))
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
    rho_log = np.zeros(nFE) if state_mode == "mismatch" else None
    rho_raw_log = np.zeros(nFE) if state_mode == "mismatch" else None
    rho_eff_log = np.zeros(nFE) if state_mode == "mismatch" else None
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

        if i > warm_start_step:
            if not test:
                action = np.asarray(agent.take_action(current_rl_state, explore=True), float)
            else:
                action = np.asarray(agent.act_eval(current_rl_state), float)
        else:
            action = zero_action.copy()

        if action.size != action_dim:
            raise ValueError("residual runner expects action_dim == n_inputs.")

        a_res_raw_log[i, :] = np.asarray(action, float).reshape(-1)

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
            apply_authority=(state_mode == "mismatch"),
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
        if rho_log is not None:
            rho_log[i] = float(projection["rho"])
            rho_raw_log[i] = float(projection["rho_raw"])
            rho_eff_log[i] = float(projection["rho_eff"])
        deadband_active_log[i] = int(projection["deadband_active"])
        projection_active_log[i] = int(projection["projection_active"])
        projection_due_to_deadband_log[i] = int(projection["projection_due_to_deadband"])
        projection_due_to_authority_log[i] = int(projection["projection_due_to_authority"])
        projection_due_to_headroom_log[i] = int(projection["projection_due_to_headroom"])
        delta_u_res_raw_log[i, :] = projection["delta_u_res_raw"]
        delta_u_res_exec_log[i, :] = projection["delta_u_res_exec"]
        a_res_exec_log[i, :] = projection["a_exec"]
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

        if not test:
            agent.push(
                current_rl_state,
                action_exec,
                reward,
                next_rl_state,
                0.0,
            )
            if i >= warm_start_step:
                agent.train_step()

        if i in sub_episodes_changes_dict:
            avg_rewards.append(float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1])))
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
        "rho_log": rho_log,
        "rho_raw_log": rho_raw_log,
        "rho_eff_log": rho_eff_log,
        "deadband_active_log": deadband_active_log,
        "projection_active_log": projection_active_log,
        "projection_due_to_deadband_log": projection_due_to_deadband_log,
        "projection_due_to_authority_log": projection_due_to_authority_log,
        "projection_due_to_headroom_log": projection_due_to_headroom_log,
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
    ):
        if hasattr(agent, attr):
            result_bundle[attr] = np.asarray(getattr(agent, attr), float)

    attach_single_agent_replay_snapshot(result_bundle, agent)
    return result_bundle
