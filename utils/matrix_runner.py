import numpy as np
import scipy.optimize as spo

from utils.helpers import (
    apply_min_max,
    apply_rl_scaled,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
)
from utils.observer import compute_observer_gain


def _map_to_bounds(action, low, high):
    action = np.asarray(action, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    return low + ((action + 1.0) / 2.0) * (high - low)


def run_matrix_multiplier_supervisor(matrix_cfg, runtime_ctx):
    """
    Run the TD3/SAC matrix-multiplier supervisor and return a normalized result bundle.

    Parameters
    ----------
    matrix_cfg : dict
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

    agent_kind = str(matrix_cfg["agent_kind"]).lower()
    run_mode = str(matrix_cfg["run_mode"]).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("matrix_cfg['run_mode'] must be 'nominal' or 'disturb'.")
    if agent_kind not in {"td3", "sac"}:
        raise ValueError("matrix_cfg['agent_kind'] must be 'td3' or 'sac'.")

    low_coef = np.asarray(matrix_cfg["low_coef"], float)
    high_coef = np.asarray(matrix_cfg["high_coef"], float)
    action_dim = int(low_coef.size)

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
        int(matrix_cfg["n_tests"]),
        int(matrix_cfg["set_points_len"]),
        int(matrix_cfg["warm_start"]),
        list(matrix_cfg["test_cycle"]),
        float(matrix_cfg["nominal_qi"]),
        float(matrix_cfg["nominal_qs"]),
        float(matrix_cfg["nominal_ha"]),
        float(matrix_cfg["qi_change"]),
        float(matrix_cfg["qs_change"]),
        float(matrix_cfg["ha_change"]),
    )

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
    delta_y_storage = np.zeros((nFE, n_outputs))
    delta_u_storage = np.zeros((nFE, n_inputs))
    alpha_log = np.zeros(nFE)
    delta_log = np.zeros((nFE, n_inputs))

    A_base = np.asarray(mpc_obj.A, float).copy()
    B_base = np.asarray(mpc_obj.B, float).copy()
    L = compute_observer_gain(mpc_obj.A, mpc_obj.C, poles)
    n_phys = n_states - n_outputs
    test = False

    cont_h = int(matrix_cfg.get("cont_h", 1))
    bnds = tuple(
        (float(matrix_cfg["b_min"][j]), float(matrix_cfg["b_max"][j]))
        for _ in range(cont_h)
        for j in range(n_inputs)
    )
    IC_opt = np.zeros(n_inputs * cont_h)

    for i in range(nFE):
        if i in test_train_dict:
            test = bool(test_train_dict[i])

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs
        current_rl_state = apply_rl_scaled(
            min_max_dict,
            xhatdhat[:, i],
            y_sp[i, :],
            scaled_current_input_dev,
        ).astype(np.float32)

        if i > warm_start_step:
            if not test:
                action = np.asarray(agent.take_action(current_rl_state, explore=True), float)
            else:
                action = np.asarray(agent.act_eval(current_rl_state), float)
        else:
            action = np.zeros(action_dim, dtype=float)

        action_mapped = _map_to_bounds(action, low_coef, high_coef)
        alpha = float(np.ravel(action_mapped[:1])[0])
        delta = np.asarray(action_mapped[-n_inputs:], float)
        alpha_log[i] = alpha
        delta_log[i, :] = delta

        A_change = A_base.copy()
        B_change = B_base.copy()
        A_change[:n_phys, :n_phys] *= alpha
        B_change[:n_phys, :] *= delta.reshape(1, -1)
        mpc_obj.A = A_change
        mpc_obj.B = B_change

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            IC_opt,
            bounds=bnds,
            constraints=[],
        )

        u_mpc[i, :] = sol.x[:n_inputs] + ss_scaled_inputs
        u_plant = reverse_min_max(u_mpc[i, :], data_min[:n_inputs], data_max[:n_inputs])
        delta_u = u_mpc[i, :] - scaled_current_input
        delta_u_storage[i, :] = delta_u

        system.current_input = u_plant
        system.step()
        if run_mode == "disturb":
            system.hA = ha[i]
            system.Qs = qs[i]
            system.Qi = qi[i]

        y_system[i + 1, :] = np.asarray(system.current_output, float)

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
        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_mpc[i, :] - ss_scaled_inputs
        next_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i + 1], y_sp[i, :], next_u_dev).astype(np.float32)

        if not test:
            agent.push(
                current_rl_state,
                np.asarray(action, np.float32),
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
                "| alpha:",
                float(np.mean(alpha_log[max(0, i - time_in_sub_episodes + 1) : i + 1])),
                "| delta:",
                np.mean(delta_log[max(0, i - time_in_sub_episodes + 1) : i + 1, :], axis=0),
            )

    mpc_obj.A = A_base
    mpc_obj.B = B_base
    u_rl = reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])

    disturbance_profile = None
    if run_mode == "disturb":
        disturbance_profile = {
            "qi": np.asarray(qi, float),
            "qs": np.asarray(qs, float),
            "ha": np.asarray(ha, float),
        }

    result_bundle = {
        "agent_kind": agent_kind,
        "run_mode": run_mode,
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
        "alpha_log": alpha_log,
        "delta_log": delta_log,
        "low_coef": low_coef,
        "high_coef": high_coef,
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "warm_start_step": int(warm_start_step),
        "mpc_horizons": (
            int(matrix_cfg["predict_h"]),
            int(matrix_cfg["cont_h"]),
        )
        if "predict_h" in matrix_cfg and "cont_h" in matrix_cfg
        else None,
    }

    for attr in ("actor_losses", "critic_losses", "alpha_losses", "alphas"):
        if hasattr(agent, attr):
            result_bundle[attr] = np.asarray(getattr(agent, attr), float)

    return result_bundle
