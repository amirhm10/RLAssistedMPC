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


def run_dueling_dqn_mpc_horizon_supervisor(dueling_cfg, runtime_ctx):
    """
    Run the dueling-DDQN-assisted horizon supervisor and return a normalized result bundle.

    The runtime behavior intentionally mirrors the current unified horizon flow:
    same setpoint scheduling, same reward path, same decision interval semantics,
    same mismatch-state support, and the same optional shifted MPC warm start.
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
    system_stepper = runtime_ctx.get("system_stepper")
    system_metadata = runtime_ctx.get("system_metadata")
    disturbance_labels = runtime_ctx.get("disturbance_labels")

    mode = str(dueling_cfg["mode"]).lower()
    state_mode = str(dueling_cfg.get("state_mode", "standard")).lower()
    predict_h = int(dueling_cfg["predict_h"])
    cont_h = int(dueling_cfg["cont_h"])
    decision_interval = int(dueling_cfg["decision_interval"])
    use_shifted_mpc_warm_start = bool(dueling_cfg.get("use_shifted_mpc_warm_start", False))
    mismatch_scale = None
    mismatch_clip = dueling_cfg.get("mismatch_clip", 3.0)
    if state_mode == "mismatch":
        mismatch_scale = np.asarray(
            dueling_cfg.get("mismatch_scale", default_mismatch_scale(min_max_dict)),
            float,
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
        int(dueling_cfg["n_tests"]),
        int(dueling_cfg["set_points_len"]),
        int(dueling_cfg["warm_start"]),
        list(dueling_cfg["test_cycle"]),
        float(dueling_cfg["nominal_qi"]),
        float(dueling_cfg["nominal_qs"]),
        float(dueling_cfg["nominal_ha"]),
        float(dueling_cfg["qi_change"]),
        float(dueling_cfg["qs_change"]),
        float(dueling_cfg["ha_change"]),
    )

    disturbance_schedule = None
    if mode == "disturb":
        disturbance_schedule = runtime_ctx.get("disturbance_schedule")
        if disturbance_schedule is None:
            disturbance_schedule = build_polymer_disturbance_schedule(qi=qi, qs=qs, ha=ha)

    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug.shape[0])

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    y_system = np.zeros((nFE + 1, n_outputs))
    y_system[0, :] = np.asarray(system.current_output, float)
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
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None

    last_action = None
    current_Hp, current_Hc = predict_h, cont_h
    test = False

    mpc_obj = MpcSolverGeneral(
        A_aug,
        B_aug,
        C_aug,
        Q_out=np.array([dueling_cfg["Q1_penalty"], dueling_cfg["Q2_penalty"]], float),
        R_in=np.array([dueling_cfg["R1_penalty"], dueling_cfg["R2_penalty"]], float),
        NP=predict_h,
        NC=cont_h,
    )

    b1 = (float(dueling_cfg["b_min"][0]), float(dueling_cfg["b_max"][0]))
    b2 = (float(dueling_cfg["b_min"][1]), float(dueling_cfg["b_max"][1]))
    cons = []

    def rebuild_mpc(Hp, Hc):
        return MpcSolverGeneral(
            A_aug,
            B_aug,
            C_aug,
            Q_out=np.array([dueling_cfg["Q1_penalty"], dueling_cfg["Q2_penalty"]], float),
            R_in=np.array([dueling_cfg["R1_penalty"], dueling_cfg["R2_penalty"]], float),
            NP=int(Hp),
            NC=int(Hc),
        )

    default_action = [idx for idx, recipe in enumerate(h_recipes) if recipe == (predict_h, cont_h)]
    if not default_action:
        raise ValueError("Default (predict_h, cont_h) is not present in horizon_recipes.")
    default_action = int(default_action[0])
    current_ic_opt = np.zeros(n_inputs * int(current_Hc))

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_pred = mpc_obj.C @ xhatdhat[:, i]
        current_rl_state, state_debug = build_rl_state(
            min_max_dict=min_max_dict,
            x_d_states=xhatdhat[:, i],
            y_sp=y_sp[i, :],
            u=scaled_current_input_dev,
            state_mode=state_mode,
            y_prev_scaled=y_prev_scaled,
            yhat_pred=yhat_pred,
            mismatch_scale=mismatch_scale,
            mismatch_clip=mismatch_clip,
        )
        if innovation_log is not None:
            innovation_log[i, :] = state_debug["innovation"]
            tracking_error_log[i, :] = state_debug["tracking_error"]

        if i > warm_start_step:
            if (i % decision_interval == 0) or (last_action is None):
                if not test:
                    a_idx = int(agent.take_action(current_rl_state.astype(np.float32), eval_mode=False))
                else:
                    a_idx = int(agent.act_eval(current_rl_state.astype(np.float32)))
                Hp, Hc = action_to_horizons(h_recipes, a_idx)
                if (Hp, Hc) != (current_Hp, current_Hc):
                    mpc_obj = rebuild_mpc(Hp, Hc)
                    current_Hp, current_Hc = Hp, Hc
                    current_ic_opt = np.zeros(n_inputs * int(current_Hc))
                last_action = a_idx
            else:
                a_idx = int(last_action)
                Hp, Hc = current_Hp, current_Hc
        else:
            a_idx = default_action
            Hp, Hc = action_to_horizons(h_recipes, a_idx)

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
        xhatdhat[:, i + 1] = (
            mpc_obj.A @ xhatdhat[:, i]
            + mpc_obj.B @ (u_mpc[i, :] - ss_scaled_inputs)
            + L @ (y_prev_scaled - yhat[:, i]).T
        )

        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_mpc[i, :] - ss_scaled_inputs
        yhat_next_pred = mpc_obj.C @ xhatdhat[:, i + 1]
        next_rl_state, _ = build_rl_state(
            min_max_dict=min_max_dict,
            x_d_states=xhatdhat[:, i + 1],
            y_sp=y_sp[i, :],
            u=next_u_dev,
            state_mode=state_mode,
            y_prev_scaled=y_current_scaled,
            yhat_pred=yhat_next_pred,
            mismatch_scale=mismatch_scale,
            mismatch_clip=mismatch_clip,
        )
        done = 0.0

        if not test:
            if i > time_in_sub_episodes:
                agent.push(
                    current_rl_state.astype(np.float32),
                    int(a_idx),
                    float(reward),
                    next_rl_state.astype(np.float32),
                    float(done),
                )
            if i >= warm_start_step:
                agent.train_step()

        if i in sub_episodes_changes_dict:
            avg_reward = float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1]))
            avg_rewards.append(avg_reward)
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_reward,
                "| Hp,Hc:",
                (int(Hp), int(Hc)),
            )

    u_rl = reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])
    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )

    result_bundle = {
        "mode": mode,
        "state_mode": state_mode,
        "algorithm": "dueling_ddqn",
        "system_metadata": system_metadata,
        "notebook_source": dueling_cfg.get("notebook_source"),
        "config_snapshot": dict(dueling_cfg),
        "seed": dueling_cfg.get("seed"),
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
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "mpc_horizons": (predict_h, cont_h),
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
        "warm_start_step": int(warm_start_step),
        "innovation_log": innovation_log,
        "tracking_error_log": tracking_error_log,
        "mismatch_scale": mismatch_scale,
        "mismatch_clip": mismatch_clip,
    }

    diagnostics = {
        "dqn_loss_trace": getattr(agent, "loss_history", None),
        "exploration_trace": getattr(agent, "exploration_trace", None),
        "epsilon_trace": getattr(agent, "epsilon_trace", None),
        "avg_td_error_trace": getattr(agent, "avg_td_error_trace", None),
        "avg_max_q_trace": getattr(agent, "avg_max_q_trace", None),
        "avg_chosen_q_trace": getattr(agent, "avg_chosen_q_trace", None),
        "avg_value_trace": getattr(agent, "avg_value_trace", None),
        "avg_advantage_spread_trace": getattr(agent, "avg_advantage_spread_trace", None),
        "noisy_sigma_trace": getattr(agent, "noisy_sigma_trace", None),
    }
    for key, value in diagnostics.items():
        if value is not None:
            result_bundle[key] = np.asarray(value, float).reshape(-1)

    return result_bundle
