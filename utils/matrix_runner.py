import warnings

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
from utils.state_features import build_rl_state, default_mismatch_scale


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
    system_stepper = runtime_ctx.get("system_stepper")
    system_metadata = runtime_ctx.get("system_metadata")
    disturbance_labels = runtime_ctx.get("disturbance_labels")

    agent_kind = str(matrix_cfg["agent_kind"]).lower()
    run_mode = str(matrix_cfg["run_mode"]).lower()
    state_mode = str(matrix_cfg.get("state_mode", "standard")).lower()
    if run_mode not in {"nominal", "disturb"}:
        raise ValueError("matrix_cfg['run_mode'] must be 'nominal' or 'disturb'.")
    if agent_kind not in {"td3", "sac"}:
        raise ValueError("matrix_cfg['agent_kind'] must be 'td3' or 'sac'.")
    use_shifted_mpc_warm_start = bool(matrix_cfg.get("use_shifted_mpc_warm_start", False))
    recalculate_observer_on_matrix_change = bool(
        matrix_cfg.get("recalculate_observer_on_matrix_change", False)
    )
    mismatch_scale = None
    mismatch_clip = matrix_cfg.get("mismatch_clip", 3.0)
    if state_mode == "mismatch":
        mismatch_scale = np.asarray(
            matrix_cfg.get("mismatch_scale", default_mismatch_scale(min_max_dict)),
            float,
        )

    low_coef = np.asarray(matrix_cfg["low_coef"], float)
    high_coef = np.asarray(matrix_cfg["high_coef"], float)
    action_dim = int(low_coef.size)
    matrix_baseline_raw = _map_from_bounds(np.ones(action_dim, dtype=float), low_coef, high_coef)

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
    innovation_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None
    tracking_error_log = np.zeros((nFE, n_outputs)) if state_mode == "mismatch" else None

    A_base = np.asarray(mpc_obj.A, float).copy()
    B_base = np.asarray(mpc_obj.B, float).copy()
    L = compute_observer_gain(mpc_obj.A, mpc_obj.C, poles)
    n_phys = n_states - n_outputs
    test = False
    nonfinite_matrix_action_count = 0
    observer_recalc_fallback_count = 0

    cont_h = int(matrix_cfg.get("cont_h", 1))
    bnds = tuple(
        (float(matrix_cfg["b_min"][j]), float(matrix_cfg["b_max"][j]))
        for _ in range(cont_h)
        for j in range(n_inputs)
    )
    ic_opt = np.zeros(n_inputs * cont_h)

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
            if not test:
                action = np.asarray(agent.take_action(current_rl_state, explore=True), float)
            else:
                action = np.asarray(agent.act_eval(current_rl_state), float)
        else:
            action = np.zeros(action_dim, dtype=float)

        if not np.all(np.isfinite(action)):
            warnings.warn(
                "Matrix agent produced a non-finite action; falling back to the nominal matrix action for this step."
            )
            action = matrix_baseline_raw.copy()
            nonfinite_matrix_action_count += 1

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
        if recalculate_observer_on_matrix_change:
            if np.all(np.isfinite(mpc_obj.A)) and np.all(np.isfinite(mpc_obj.C)):
                try:
                    L = compute_observer_gain(mpc_obj.A, mpc_obj.C, poles)
                except Exception:
                    warnings.warn(
                        "Observer gain recalculation failed for the updated matrix model; keeping the previous observer gain."
                    )
                    observer_recalc_fallback_count += 1
            else:
                warnings.warn(
                    "Updated matrix model contains non-finite values; keeping the previous observer gain."
                )
                observer_recalc_fallback_count += 1

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

    disturbance_profile = disturbance_profile_from_schedule(
        disturbance_schedule if run_mode == "disturb" else None,
        disturbance_labels=disturbance_labels,
    )

    result_bundle = {
        "agent_kind": agent_kind,
        "run_mode": run_mode,
        "method_family": "matrix",
        "algorithm": agent_kind,
        "state_mode": state_mode,
        "system_metadata": system_metadata,
        "notebook_source": matrix_cfg.get("notebook_source"),
        "config_snapshot": dict(matrix_cfg),
        "seed": matrix_cfg.get("seed"),
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
        "use_shifted_mpc_warm_start": use_shifted_mpc_warm_start,
        "recalculate_observer_on_matrix_change": recalculate_observer_on_matrix_change,
        "nonfinite_matrix_action_count": int(nonfinite_matrix_action_count),
        "observer_recalc_fallback_count": int(observer_recalc_fallback_count),
        "n_step": int(getattr(agent, "n_step", 1)),
        "multistep_mode": str(getattr(agent, "multistep_mode", "one_step")),
        "lambda_value": getattr(agent, "lambda_value", None),
        "innovation_log": innovation_log,
        "tracking_error_log": tracking_error_log,
        "mismatch_scale": mismatch_scale,
        "mismatch_clip": mismatch_clip,
        "mpc_horizons": (
            int(matrix_cfg["predict_h"]),
            int(matrix_cfg["cont_h"]),
        )
        if "predict_h" in matrix_cfg and "cont_h" in matrix_cfg
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

    return result_bundle
