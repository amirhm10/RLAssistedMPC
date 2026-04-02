import numpy as np
import scipy.optimize as spo

from utils.helpers import (
    apply_min_max,
    generate_setpoints_training_rl_gradually,
    reverse_min_max,
)


def run_offsetfree_mpc(mpc_cfg, runtime_ctx):
    """
    Run the baseline offset-free MPC controller and return a normalized result bundle.

    Parameters
    ----------
    mpc_cfg : dict
        Runtime config assembled in the notebook.
    runtime_ctx : dict
        Prepared objects and shared data assembled in the notebook.
    """

    system = runtime_ctx["system"]
    mpc_obj = runtime_ctx["MPC_obj"]
    steady_states = runtime_ctx["steady_states"]
    data_min = np.asarray(runtime_ctx["data_min"], float)
    data_max = np.asarray(runtime_ctx["data_max"], float)
    A_aug = np.asarray(runtime_ctx["A_aug"], float)
    B_aug = np.asarray(runtime_ctx["B_aug"], float)
    C_aug = np.asarray(runtime_ctx["C_aug"], float)
    y_sp_scenario = np.asarray(runtime_ctx["y_sp_scenario"], float)
    reward_fn = runtime_ctx["reward_fn"]
    L = np.asarray(runtime_ctx["L"], float)

    run_mode = str(mpc_cfg["run_mode"]).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("mpc_cfg['run_mode'] must be 'nominal' or 'disturb'.")

    reward_scale = float(mpc_cfg.get("reward_scale", 1.0))

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
        int(mpc_cfg["n_tests"]),
        int(mpc_cfg["set_points_len"]),
        int(mpc_cfg["warm_start"]),
        list(mpc_cfg["test_cycle"]),
        float(mpc_cfg["nominal_qi"]),
        float(mpc_cfg["nominal_qs"]),
        float(mpc_cfg["nominal_ha"]),
        float(mpc_cfg["qi_change"]),
        float(mpc_cfg["qs_change"]),
        float(mpc_cfg["ha_change"]),
    )

    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug.shape[0])

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    cont_h = int(mpc_cfg["cont_h"])
    bounds = tuple(
        (float(mpc_cfg["b_min"][j]), float(mpc_cfg["b_max"][j]))
        for _ in range(cont_h)
        for j in range(n_inputs)
    )
    ic_opt = np.zeros(n_inputs * cont_h)

    y_mpc = np.zeros((nFE + 1, n_outputs))
    y_mpc[0, :] = np.asarray(system.current_output, float)
    u_mpc = np.zeros((nFE, n_inputs))
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    rewards = np.zeros(nFE)
    rewards_mpc = np.zeros(nFE)
    avg_rewards = []
    avg_rewards_mpc = []
    delta_y_storage = np.zeros((nFE, n_outputs))
    delta_u_storage = np.zeros((nFE, n_inputs))

    for i in range(nFE):
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            ic_opt,
            bounds=bounds,
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

        y_mpc[i + 1, :] = np.asarray(system.current_output, float)

        y_current_scaled = apply_min_max(y_mpc[i + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        y_prev_scaled = apply_min_max(y_mpc[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
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
        rewards[i] = reward * reward_scale
        rewards_mpc[i] = -(
            float(mpc_cfg["Q1_penalty"]) * delta_y[0] ** 2
            + float(mpc_cfg["Q2_penalty"]) * delta_y[1] ** 2
            + float(mpc_cfg["R1_penalty"]) * delta_u[0] ** 2
            + float(mpc_cfg["R2_penalty"]) * delta_u[1] ** 2
        )

        if i in sub_episodes_changes_dict:
            avg_rewards.append(float(np.mean(rewards[max(0, i - time_in_sub_episodes + 1) : i + 1])))
            avg_rewards_mpc.append(float(np.mean(rewards_mpc[max(0, i - time_in_sub_episodes + 1) : i + 1])))
            print(
                "Sub_Episode:",
                sub_episodes_changes_dict[i],
                "| avg. reward:",
                avg_rewards[-1],
                "| avg. reward MPC:",
                avg_rewards_mpc[-1],
            )

    disturbance_profile = None
    if run_mode == "disturb":
        disturbance_profile = {
            "qi": np.asarray(qi, float),
            "qs": np.asarray(qs, float),
            "ha": np.asarray(ha, float),
        }

    return {
        "run_mode": run_mode,
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": int(nFE),
        "delta_t": float(system.delta_t),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y": y_mpc,
        "u": reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs]),
        "avg_rewards": np.asarray(avg_rewards, float),
        "avg_rewards_mpc": np.asarray(avg_rewards_mpc, float),
        "rewards_step": rewards,
        "rewards_mpc": rewards_mpc,
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
        "mpc_horizons": (int(mpc_cfg["predict_h"]), int(mpc_cfg["cont_h"])),
    }
