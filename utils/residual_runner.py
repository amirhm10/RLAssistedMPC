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


def _map_from_bounds(value, low, high):
    value = np.asarray(value, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    return 2.0 * (value - low) / (high - low) - 1.0


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

    agent_kind = str(residual_cfg["agent_kind"]).lower()
    run_mode = str(residual_cfg["run_mode"]).lower()
    if agent_kind not in {"td3", "sac"}:
        raise ValueError("residual_cfg['agent_kind'] must be 'td3' or 'sac'.")
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("residual_cfg['run_mode'] must be 'nominal' or 'disturb'.")

    low_coef = np.asarray(residual_cfg["low_coef"], float).reshape(-1)
    high_coef = np.asarray(residual_cfg["high_coef"], float).reshape(-1)
    action_dim = int(B_aug.shape[1])
    if low_coef.size != action_dim or high_coef.size != action_dim:
        raise ValueError("low_coef/high_coef must match the number of manipulated inputs.")
    if np.any(low_coef > 0.0) or np.any(high_coef < 0.0):
        raise ValueError("Residual bounds must bracket zero so warm start can apply zero correction.")

    zero_action = _map_from_bounds(np.zeros(action_dim, dtype=float), low_coef, high_coef)

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

    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    n_states = int(A_aug.shape[0])

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    u_min_scaled_abs = np.asarray(residual_cfg["b_min"], float) + ss_scaled_inputs
    u_max_scaled_abs = np.asarray(residual_cfg["b_max"], float) + ss_scaled_inputs
    L = compute_observer_gain(mpc_obj.A, mpc_obj.C, poles)

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
    residual_raw_log = np.zeros((nFE, n_inputs))
    residual_exec_log = np.zeros((nFE, n_inputs))
    test = False

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
            action = zero_action.copy()

        if action.size != action_dim:
            raise ValueError("residual runner expects action_dim == n_inputs.")

        residual_raw = _map_to_bounds(action, low_coef, high_coef).reshape(-1)
        residual_raw_log[i, :] = residual_raw

        sol = spo.minimize(
            lambda x: mpc_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev, xhatdhat[:, i]),
            ic_opt,
            bounds=bounds,
            constraints=[],
        )

        u_base = np.asarray(sol.x[:n_inputs], float) + ss_scaled_inputs
        u_base = np.clip(u_base, u_min_scaled_abs, u_max_scaled_abs)
        u_base_scaled[i, :] = u_base

        low_headroom = u_min_scaled_abs - u_base
        high_headroom = u_max_scaled_abs - u_base
        residual_exec = np.clip(residual_raw, np.maximum(low_coef, low_headroom), np.minimum(high_coef, high_headroom))
        u_applied_scaled_abs = np.clip(u_base + residual_exec, u_min_scaled_abs, u_max_scaled_abs)
        residual_exec = u_applied_scaled_abs - u_base
        residual_exec_log[i, :] = residual_exec
        u_rl_scaled[i, :] = u_applied_scaled_abs

        delta_u = u_applied_scaled_abs - scaled_current_input
        delta_u_storage[i, :] = delta_u
        action_exec = _map_from_bounds(residual_exec, low_coef, high_coef).astype(np.float32)
        action_exec = np.clip(action_exec, -1.0, 1.0)

        u_plant = reverse_min_max(u_applied_scaled_abs, data_min[:n_inputs], data_max[:n_inputs])
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
            + mpc_obj.B @ (u_applied_scaled_abs - ss_scaled_inputs)
            + L @ (y_prev_scaled - yhat[:, i]).T
        )

        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
        rewards[i] = reward

        next_u_dev = u_applied_scaled_abs - ss_scaled_inputs
        next_rl_state = apply_rl_scaled(min_max_dict, xhatdhat[:, i + 1], y_sp[i, :], next_u_dev).astype(np.float32)

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
                np.mean(residual_exec_log[max(0, i - time_in_sub_episodes + 1) : i + 1, :], axis=0),
            )

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
        "residual_raw_log": residual_raw_log,
        "residual_exec_log": residual_exec_log,
        "low_coef": low_coef,
        "high_coef": high_coef,
        "test_train_dict": test_train_dict,
        "sub_episodes_changes_dict": sub_episodes_changes_dict,
        "disturbance_profile": disturbance_profile,
        "warm_start_step": int(warm_start_step),
        "mpc_horizons": (
            int(residual_cfg["predict_h"]),
            int(residual_cfg["cont_h"]),
        )
        if "predict_h" in residual_cfg and "cont_h" in residual_cfg
        else None,
    }

    for attr in ("actor_losses", "critic_losses", "alpha_losses", "alphas"):
        if hasattr(agent, attr):
            result_bundle[attr] = np.asarray(getattr(agent, attr), float)

    return result_bundle
