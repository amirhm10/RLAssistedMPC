import numpy as np
import scipy.optimize as spo

from Simulation.mpc import MpcSolverGeneral
from utils.helpers import (
    action_to_horizons,
    apply_min_max,
    apply_rl_scaled,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
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
        A_aug, B_aug, C_aug, L, data_min, data_max, horizon_recipes, reward_fn.
    """

    system = runtime_ctx["system"]
    y_sp_scenario = np.asarray(runtime_ctx["y_sp_scenario"], float)
    steady_states = runtime_ctx["steady_states"]
    min_max_dict = runtime_ctx["min_max_dict"]
    agent = runtime_ctx["agent"]
    A_aug = np.asarray(runtime_ctx["A_aug"], float)
    B_aug = np.asarray(runtime_ctx["B_aug"], float)
    C_aug = np.asarray(runtime_ctx["C_aug"], float)
    L = np.asarray(runtime_ctx["L"], float)
    data_min = np.asarray(runtime_ctx["data_min"], float)
    data_max = np.asarray(runtime_ctx["data_max"], float)
    h_recipes = list(runtime_ctx["horizon_recipes"])
    reward_fn = runtime_ctx["reward_fn"]

    mode = horizon_cfg["mode"]
    predict_h = int(horizon_cfg["predict_h"])
    cont_h = int(horizon_cfg["cont_h"])
    decision_interval = int(horizon_cfg["decision_interval"])
    reward_scale = float(horizon_cfg.get("reward_scale", 0.01))

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

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        current_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i], y_sp[i, :], scaled_current_input_dev)

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
        IC_opt = np.zeros(n_inputs * int(Hc))

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            IC_opt,
            bounds=bnds,
            constraints=cons,
        )

        u_mpc[i, :] = sol.x[:n_inputs] + ss_scaled_inputs
        u_plant = reverse_min_max(u_mpc[i, :], data_min[:n_inputs], data_max[:n_inputs])
        delta_u = u_mpc[i, :] - scaled_current_input
        delta_u_storage[i, :] = delta_u

        system.current_input = u_plant
        system.step()
        if mode == "disturb":
            system.hA = ha[i]
            system.Qs = qs[i]
            system.Qi = qi[i]

        y_system[i + 1, :] = system.current_output

        y_current_scaled = apply_min_max(y_system[i + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        y_prev_scaled = apply_min_max(y_system[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        delta_y = y_current_scaled - y_sp[i, :]
        delta_y_storage[i, :] = delta_y

        yhat[:, i] = mpc_obj.C @ xhatdhat[:, i]
        xhatdhat[:, i + 1] = (
            mpc_obj.A @ xhatdhat[:, i]
            + mpc_obj.B @ (u_mpc[i, :] - ss_scaled_inputs)
            + L @ (y_prev_scaled - yhat[:, i]).T
        )

        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        reward = reward_fn(delta_y, delta_u, y_sp_phys) * reward_scale
        rewards[i] = reward

        next_u_dev = u_mpc[i, :] - ss_scaled_inputs
        next_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i + 1], y_sp[i, :], next_u_dev)
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
            avg_reward = float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1): i + 1]))
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
    disturbance_profile = None
    if mode == "disturb":
        disturbance_profile = {
            "qi": np.asarray(qi, float),
            "qs": np.asarray(qs, float),
            "ha": np.asarray(ha, float),
        }

    return {
        "mode": mode,
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
        "warm_start_step": int(warm_start_step),
        "reward_scale": reward_scale,
    }
