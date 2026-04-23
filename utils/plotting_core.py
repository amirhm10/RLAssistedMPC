import collections
import gc
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from utils.helpers import apply_min_max, reverse_min_max


def _set_plot_style(style_profile="hybrid"):
    style_profile = str(style_profile).lower()
    if style_profile not in {"hybrid", "paper", "debug"}:
        style_profile = "hybrid"

    profile_map = {
        "hybrid": {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "figure.dpi": 120,
        },
        "paper": {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "figure.dpi": 140,
        },
        "debug": {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.3,
            "lines.markersize": 5,
            "figure.dpi": 120,
        },
    }
    palette = ["#0B3954", "#C81D25", "#2D6A4F", "#6A4C93", "#F4A259", "#7D8597", "#3A86FF", "#8338EC"]
    rc = {
        "axes.labelweight": "bold",
        "axes.titlesize": profile_map[style_profile]["axes.labelsize"],
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.35,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": plt.cycler(color=palette),
    }
    rc.update(profile_map[style_profile])
    plt.rcParams.update(rc)


def _make_axes_bold(ax):
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def _save_fig(fig, out_dir, fname_base, save_pdf=False):
    png_path = os.path.join(out_dir, fname_base + ".png")
    pdf_path = os.path.join(out_dir, fname_base + ".pdf")

    def _is_render_memory_error(exc):
        if isinstance(exc, MemoryError):
            return True
        message = str(exc).lower()
        return "image size of" in message or "bad allocation" in message or "array is too big" in message

    try:
        try:
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
        except Exception as exc:
            if not _is_render_memory_error(exc):
                raise
            fig.savefig(png_path, dpi=180)

        if save_pdf:
            try:
                fig.savefig(pdf_path, bbox_inches="tight")
            except Exception as exc:
                if not _is_render_memory_error(exc):
                    raise
                fig.savefig(pdf_path)
    finally:
        plt.close(fig)
        gc.collect()


def _default_system_metadata(n_outputs, n_inputs):
    return {
        "system_name": "polymer",
        "output_labels": [r"$\eta$ (L/g)", r"$T$ (K)"] if n_outputs == 2 else [f"y{idx + 1}" for idx in range(n_outputs)],
        "input_labels": [r"$Q_c$ (L/h)", r"$Q_m$ (L/h)"] if n_inputs == 2 else [f"u{idx + 1}" for idx in range(n_inputs)],
        "time_label": "Time (h)",
        "disturbance_labels": None,
    }


def resolve_system_metadata(bundle=None, plot_cfg=None, n_outputs=None, n_inputs=None):
    bundle = bundle or {}
    plot_cfg = plot_cfg or {}
    metadata = _default_system_metadata(int(n_outputs), int(n_inputs))
    metadata.update(dict(bundle.get("system_metadata") or {}))
    metadata.update(dict(plot_cfg.get("system_metadata") or {}))

    output_labels = list(metadata.get("output_labels") or metadata["output_labels"])
    input_labels = list(metadata.get("input_labels") or metadata["input_labels"])
    time_label = str(metadata.get("time_label") or "Time (h)")
    disturbance_labels = metadata.get("disturbance_labels")
    if disturbance_labels is not None:
        disturbance_labels = list(disturbance_labels)
    metadata["output_labels"] = output_labels
    metadata["input_labels"] = input_labels
    metadata["time_label"] = time_label
    metadata["disturbance_labels"] = disturbance_labels
    return metadata


def disturbance_plot_items(disturbance_profile, disturbance_labels=None):
    if not disturbance_profile:
        return []
    label_lookup = {}
    if disturbance_labels:
        keys = list(disturbance_profile.keys())
        label_lookup = {key: disturbance_labels[idx] for idx, key in enumerate(keys[: len(disturbance_labels)])}
    return [(key, label_lookup.get(key, key), np.asarray(value, float)) for key, value in disturbance_profile.items()]


def normalize_result_bundle(result_bundle):
    bundle = dict(result_bundle)

    y = bundle.get("y", bundle.get("y_rl", bundle.get("y_mpc")))
    u = bundle.get("u", bundle.get("u_rl", bundle.get("u_mpc")))
    if y is None or u is None:
        raise KeyError("result_bundle must contain y/u or legacy y_rl/u_rl or y_mpc/u_mpc keys.")

    bundle["y"] = np.asarray(y, float)
    bundle["u"] = np.asarray(u, float)
    bundle["avg_rewards"] = np.asarray(bundle.get("avg_rewards", []), float)
    bundle["avg_rewards_mpc"] = np.asarray(bundle.get("avg_rewards_mpc", []), float)
    bundle["data_min"] = np.asarray(bundle["data_min"], float)
    bundle["data_max"] = np.asarray(bundle["data_max"], float)
    bundle["nFE"] = int(bundle["nFE"])
    bundle["delta_t"] = float(bundle["delta_t"])
    bundle["time_in_sub_episodes"] = int(bundle["time_in_sub_episodes"])
    bundle["y_sp"] = np.asarray(bundle["y_sp"], float)
    bundle["rewards_step"] = None if bundle.get("rewards_step") is None else np.asarray(bundle["rewards_step"], float)
    bundle["rewards_mpc"] = None if bundle.get("rewards_mpc") is None else np.asarray(bundle["rewards_mpc"], float)
    bundle["delta_y_storage"] = None if bundle.get("delta_y_storage") is None else np.asarray(bundle["delta_y_storage"], float)
    bundle["delta_u_storage"] = None if bundle.get("delta_u_storage") is None else np.asarray(bundle["delta_u_storage"], float)
    bundle["horizon_trace"] = None if bundle.get("horizon_trace") is None else np.asarray(bundle["horizon_trace"], float)
    bundle["action_trace"] = None if bundle.get("action_trace") is None else np.asarray(bundle["action_trace"], int)
    bundle["horizon_action_trace"] = (
        None if bundle.get("horizon_action_trace") is None else np.asarray(bundle["horizon_action_trace"], int).reshape(-1)
    )
    bundle["horizon_decision_log"] = (
        None if bundle.get("horizon_decision_log") is None else np.asarray(bundle["horizon_decision_log"], int).reshape(-1)
    )
    bundle["yhat"] = None if bundle.get("yhat") is None else np.asarray(bundle["yhat"], float)
    bundle["xhatdhat"] = None if bundle.get("xhatdhat") is None else np.asarray(bundle["xhatdhat"], float)
    bundle["alpha_log"] = None if bundle.get("alpha_log") is None else np.asarray(bundle["alpha_log"], float).reshape(-1)
    bundle["delta_log"] = None if bundle.get("delta_log") is None else np.asarray(bundle["delta_log"], float)
    bundle["matrix_alpha_log"] = (
        None if bundle.get("matrix_alpha_log") is None else np.asarray(bundle["matrix_alpha_log"], float).reshape(-1)
    )
    bundle["matrix_delta_log"] = (
        None if bundle.get("matrix_delta_log") is None else np.asarray(bundle["matrix_delta_log"], float)
    )
    bundle["matrix_decision_log"] = (
        None if bundle.get("matrix_decision_log") is None else np.asarray(bundle["matrix_decision_log"], int).reshape(-1)
    )
    bundle["weight_log"] = None if bundle.get("weight_log") is None else np.asarray(bundle["weight_log"], float)
    bundle["weight_decision_log"] = (
        None if bundle.get("weight_decision_log") is None else np.asarray(bundle["weight_decision_log"], int).reshape(-1)
    )
    if bundle.get("residual_raw_log") is None and bundle.get("delta_u_res_raw_log") is not None:
        bundle["residual_raw_log"] = bundle.get("delta_u_res_raw_log")
    if bundle.get("residual_exec_log") is None and bundle.get("delta_u_res_exec_log") is not None:
        bundle["residual_exec_log"] = bundle.get("delta_u_res_exec_log")
    bundle["a_res_raw_log"] = None if bundle.get("a_res_raw_log") is None else np.asarray(bundle["a_res_raw_log"], float)
    bundle["a_res_exec_log"] = None if bundle.get("a_res_exec_log") is None else np.asarray(bundle["a_res_exec_log"], float)
    bundle["delta_u_res_raw_log"] = (
        None if bundle.get("delta_u_res_raw_log") is None else np.asarray(bundle["delta_u_res_raw_log"], float)
    )
    bundle["delta_u_res_exec_log"] = (
        None if bundle.get("delta_u_res_exec_log") is None else np.asarray(bundle["delta_u_res_exec_log"], float)
    )
    bundle["residual_exec_log"] = (
        None if bundle.get("residual_exec_log") is None else np.asarray(bundle["residual_exec_log"], float)
    )
    bundle["residual_raw_log"] = (
        None if bundle.get("residual_raw_log") is None else np.asarray(bundle["residual_raw_log"], float)
    )
    bundle["residual_decision_log"] = (
        None if bundle.get("residual_decision_log") is None else np.asarray(bundle["residual_decision_log"], int).reshape(-1)
    )
    bundle["rho_log"] = None if bundle.get("rho_log") is None else np.asarray(bundle["rho_log"], float).reshape(-1)
    bundle["rho_raw_log"] = None if bundle.get("rho_raw_log") is None else np.asarray(bundle["rho_raw_log"], float).reshape(-1)
    bundle["rho_eff_log"] = None if bundle.get("rho_eff_log") is None else np.asarray(bundle["rho_eff_log"], float).reshape(-1)
    for key in (
        "deadband_active_log",
        "projection_active_log",
        "projection_due_to_deadband_log",
        "projection_due_to_authority_log",
        "projection_due_to_headroom_log",
    ):
        bundle[key] = None if bundle.get(key) is None else np.asarray(bundle[key], int).reshape(-1)
    bundle["innovation_log"] = None if bundle.get("innovation_log") is None else np.asarray(bundle["innovation_log"], float)
    bundle["innovation_raw_log"] = (
        None if bundle.get("innovation_raw_log") is None else np.asarray(bundle["innovation_raw_log"], float)
    )
    bundle["tracking_error_log"] = (
        None if bundle.get("tracking_error_log") is None else np.asarray(bundle["tracking_error_log"], float)
    )
    bundle["tracking_error_raw_log"] = (
        None if bundle.get("tracking_error_raw_log") is None else np.asarray(bundle["tracking_error_raw_log"], float)
    )
    bundle["tracking_scale_log"] = (
        None if bundle.get("tracking_scale_log") is None else np.asarray(bundle["tracking_scale_log"], float)
    )
    for key in (
        "horizon_innovation_log",
        "horizon_innovation_raw_log",
        "horizon_tracking_error_log",
        "horizon_tracking_error_raw_log",
        "horizon_tracking_scale_log",
        "matrix_innovation_log",
        "matrix_innovation_raw_log",
        "matrix_tracking_error_log",
        "matrix_tracking_error_raw_log",
        "matrix_tracking_scale_log",
        "weight_innovation_log",
        "weight_innovation_raw_log",
        "weight_tracking_error_log",
        "weight_tracking_error_raw_log",
        "weight_tracking_scale_log",
        "residual_innovation_log",
        "residual_innovation_raw_log",
        "residual_tracking_error_log",
        "residual_tracking_error_raw_log",
        "residual_tracking_scale_log",
    ):
        bundle[key] = None if bundle.get(key) is None else np.asarray(bundle[key], float)
    bundle["u_base"] = None if bundle.get("u_base") is None else np.asarray(bundle["u_base"], float)
    bundle["actor_losses"] = None if bundle.get("actor_losses") is None else np.asarray(bundle["actor_losses"], float).reshape(-1)
    bundle["critic_losses"] = None if bundle.get("critic_losses") is None else np.asarray(bundle["critic_losses"], float).reshape(-1)
    bundle["alpha_losses"] = None if bundle.get("alpha_losses") is None else np.asarray(bundle["alpha_losses"], float).reshape(-1)
    bundle["alphas"] = None if bundle.get("alphas") is None else np.asarray(bundle["alphas"], float).reshape(-1)
    bundle["dqn_loss_trace"] = None if bundle.get("dqn_loss_trace") is None else np.asarray(bundle["dqn_loss_trace"], float).reshape(-1)
    bundle["exploration_trace"] = (
        None if bundle.get("exploration_trace") is None else np.asarray(bundle["exploration_trace"], float).reshape(-1)
    )
    bundle["epsilon_trace"] = None if bundle.get("epsilon_trace") is None else np.asarray(bundle["epsilon_trace"], float).reshape(-1)
    bundle["avg_td_error_trace"] = (
        None if bundle.get("avg_td_error_trace") is None else np.asarray(bundle["avg_td_error_trace"], float).reshape(-1)
    )
    bundle["avg_max_q_trace"] = (
        None if bundle.get("avg_max_q_trace") is None else np.asarray(bundle["avg_max_q_trace"], float).reshape(-1)
    )
    bundle["avg_chosen_q_trace"] = (
        None if bundle.get("avg_chosen_q_trace") is None else np.asarray(bundle["avg_chosen_q_trace"], float).reshape(-1)
    )
    bundle["avg_value_trace"] = (
        None if bundle.get("avg_value_trace") is None else np.asarray(bundle["avg_value_trace"], float).reshape(-1)
    )
    bundle["avg_advantage_spread_trace"] = (
        None
        if bundle.get("avg_advantage_spread_trace") is None
        else np.asarray(bundle["avg_advantage_spread_trace"], float).reshape(-1)
    )
    bundle["noisy_sigma_trace"] = (
        None if bundle.get("noisy_sigma_trace") is None else np.asarray(bundle["noisy_sigma_trace"], float).reshape(-1)
    )
    bundle["reward_n_mean_trace"] = (
        None if bundle.get("reward_n_mean_trace") is None else np.asarray(bundle["reward_n_mean_trace"], float).reshape(-1)
    )
    bundle["discount_n_mean_trace"] = (
        None if bundle.get("discount_n_mean_trace") is None else np.asarray(bundle["discount_n_mean_trace"], float).reshape(-1)
    )
    bundle["bootstrap_q_mean_trace"] = (
        None if bundle.get("bootstrap_q_mean_trace") is None else np.asarray(bundle["bootstrap_q_mean_trace"], float).reshape(-1)
    )
    bundle["n_actual_mean_trace"] = (
        None if bundle.get("n_actual_mean_trace") is None else np.asarray(bundle["n_actual_mean_trace"], float).reshape(-1)
    )
    bundle["truncated_fraction_trace"] = (
        None
        if bundle.get("truncated_fraction_trace") is None
        else np.asarray(bundle["truncated_fraction_trace"], float).reshape(-1)
    )
    bundle["lambda_return_mean_trace"] = (
        None
        if bundle.get("lambda_return_mean_trace") is None
        else np.asarray(bundle["lambda_return_mean_trace"], float).reshape(-1)
    )
    bundle["offpolicy_rho_mean_trace"] = (
        None
        if bundle.get("offpolicy_rho_mean_trace") is None
        else np.asarray(bundle["offpolicy_rho_mean_trace"], float).reshape(-1)
    )
    bundle["offpolicy_c_mean_trace"] = (
        None
        if bundle.get("offpolicy_c_mean_trace") is None
        else np.asarray(bundle["offpolicy_c_mean_trace"], float).reshape(-1)
    )
    bundle["behavior_logprob_mean_trace"] = (
        None
        if bundle.get("behavior_logprob_mean_trace") is None
        else np.asarray(bundle["behavior_logprob_mean_trace"], float).reshape(-1)
    )
    bundle["target_logprob_mean_trace"] = (
        None
        if bundle.get("target_logprob_mean_trace") is None
        else np.asarray(bundle["target_logprob_mean_trace"], float).reshape(-1)
    )
    bundle["eta_A_log"] = None if bundle.get("eta_A_log") is None else np.asarray(bundle["eta_A_log"], float).reshape(-1)
    bundle["eta_B_log"] = None if bundle.get("eta_B_log") is None else np.asarray(bundle["eta_B_log"], float).reshape(-1)
    bundle["eta_A_raw_log"] = None if bundle.get("eta_A_raw_log") is None else np.asarray(bundle["eta_A_raw_log"], float).reshape(-1)
    bundle["eta_B_raw_log"] = None if bundle.get("eta_B_raw_log") is None else np.asarray(bundle["eta_B_raw_log"], float).reshape(-1)
    bundle["eta_log"] = None if bundle.get("eta_log") is None else np.asarray(bundle["eta_log"], float).reshape(-1)
    bundle["eta_raw_log"] = None if bundle.get("eta_raw_log") is None else np.asarray(bundle["eta_raw_log"], float).reshape(-1)
    bundle["action_raw_log"] = None if bundle.get("action_raw_log") is None else np.asarray(bundle["action_raw_log"], float)
    bundle["theta_hat_log"] = None if bundle.get("theta_hat_log") is None else np.asarray(bundle["theta_hat_log"], float)
    bundle["theta_active_log"] = None if bundle.get("theta_active_log") is None else np.asarray(bundle["theta_active_log"], float)
    bundle["theta_candidate_log"] = None if bundle.get("theta_candidate_log") is None else np.asarray(bundle["theta_candidate_log"], float)
    bundle["theta_unclipped_log"] = None if bundle.get("theta_unclipped_log") is None else np.asarray(bundle["theta_unclipped_log"], float)
    bundle["theta_lower_hit_mask_log"] = (
        None if bundle.get("theta_lower_hit_mask_log") is None else np.asarray(bundle["theta_lower_hit_mask_log"], float)
    )
    bundle["theta_upper_hit_mask_log"] = (
        None if bundle.get("theta_upper_hit_mask_log") is None else np.asarray(bundle["theta_upper_hit_mask_log"], float)
    )
    bundle["theta_clipped_fraction_log"] = (
        None
        if bundle.get("theta_clipped_fraction_log") is None
        else np.asarray(bundle["theta_clipped_fraction_log"], float).reshape(-1)
    )
    bundle["id_residual_norm_log"] = (
        None if bundle.get("id_residual_norm_log") is None else np.asarray(bundle["id_residual_norm_log"], float).reshape(-1)
    )
    bundle["id_condition_number_log"] = (
        None
        if bundle.get("id_condition_number_log") is None
        else np.asarray(bundle["id_condition_number_log"], float).reshape(-1)
    )
    bundle["id_update_event_log"] = (
        None if bundle.get("id_update_event_log") is None else np.asarray(bundle["id_update_event_log"], int).reshape(-1)
    )
    bundle["id_update_success_log"] = (
        None if bundle.get("id_update_success_log") is None else np.asarray(bundle["id_update_success_log"], int).reshape(-1)
    )
    bundle["id_fallback_log"] = (
        None if bundle.get("id_fallback_log") is None else np.asarray(bundle["id_fallback_log"], int).reshape(-1)
    )
    bundle["id_valid_flag_log"] = (
        None if bundle.get("id_valid_flag_log") is None else np.asarray(bundle["id_valid_flag_log"], int).reshape(-1)
    )
    bundle["id_source_code_log"] = (
        None if bundle.get("id_source_code_log") is None else np.asarray(bundle["id_source_code_log"], int).reshape(-1)
    )
    bundle["id_candidate_valid_log"] = (
        None if bundle.get("id_candidate_valid_log") is None else np.asarray(bundle["id_candidate_valid_log"], int).reshape(-1)
    )
    bundle["id_A_model_delta_ratio_log"] = (
        None
        if bundle.get("id_A_model_delta_ratio_log") is None
        else np.asarray(bundle["id_A_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["id_B_model_delta_ratio_log"] = (
        None
        if bundle.get("id_B_model_delta_ratio_log") is None
        else np.asarray(bundle["id_B_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["pred_A_model_delta_ratio_log"] = (
        None
        if bundle.get("pred_A_model_delta_ratio_log") is None
        else np.asarray(bundle["pred_A_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["pred_B_model_delta_ratio_log"] = (
        None
        if bundle.get("pred_B_model_delta_ratio_log") is None
        else np.asarray(bundle["pred_B_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["observer_A_model_delta_ratio_log"] = (
        None
        if bundle.get("observer_A_model_delta_ratio_log") is None
        else np.asarray(bundle["observer_A_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["observer_B_model_delta_ratio_log"] = (
        None
        if bundle.get("observer_B_model_delta_ratio_log") is None
        else np.asarray(bundle["observer_B_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["observer_refresh_event_log"] = (
        None
        if bundle.get("observer_refresh_event_log") is None
        else np.asarray(bundle["observer_refresh_event_log"], int).reshape(-1)
    )
    bundle["observer_refresh_success_log"] = (
        None
        if bundle.get("observer_refresh_success_log") is None
        else np.asarray(bundle["observer_refresh_success_log"], int).reshape(-1)
    )
    bundle["basis_singular_values_A"] = (
        None if bundle.get("basis_singular_values_A") is None else np.asarray(bundle["basis_singular_values_A"], float).reshape(-1)
    )
    bundle["basis_singular_values_B"] = (
        None if bundle.get("basis_singular_values_B") is None else np.asarray(bundle["basis_singular_values_B"], float).reshape(-1)
    )
    bundle["candidate_A_model_delta_ratio_log"] = (
        None
        if bundle.get("candidate_A_model_delta_ratio_log") is None
        else np.asarray(bundle["candidate_A_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["candidate_B_model_delta_ratio_log"] = (
        None
        if bundle.get("candidate_B_model_delta_ratio_log") is None
        else np.asarray(bundle["candidate_B_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["active_A_model_delta_ratio_log"] = (
        None
        if bundle.get("active_A_model_delta_ratio_log") is None
        else np.asarray(bundle["active_A_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["active_B_model_delta_ratio_log"] = (
        None
        if bundle.get("active_B_model_delta_ratio_log") is None
        else np.asarray(bundle["active_B_model_delta_ratio_log"], float).reshape(-1)
    )
    bundle["retrace_c_clip_fraction_trace"] = (
        None
        if bundle.get("retrace_c_clip_fraction_trace") is None
        else np.asarray(bundle["retrace_c_clip_fraction_trace"], float).reshape(-1)
    )
    bundle["critic_q1_trace"] = (
        None if bundle.get("critic_q1_trace") is None else np.asarray(bundle["critic_q1_trace"], float).reshape(-1)
    )
    bundle["critic_q2_trace"] = (
        None if bundle.get("critic_q2_trace") is None else np.asarray(bundle["critic_q2_trace"], float).reshape(-1)
    )
    bundle["critic_q_gap_trace"] = (
        None if bundle.get("critic_q_gap_trace") is None else np.asarray(bundle["critic_q_gap_trace"], float).reshape(-1)
    )
    bundle["exploration_magnitude_trace"] = (
        None
        if bundle.get("exploration_magnitude_trace") is None
        else np.asarray(bundle["exploration_magnitude_trace"], float).reshape(-1)
    )
    bundle["param_noise_scale_trace"] = (
        None
        if bundle.get("param_noise_scale_trace") is None
        else np.asarray(bundle["param_noise_scale_trace"], float).reshape(-1)
    )
    bundle["action_saturation_trace"] = (
        None
        if bundle.get("action_saturation_trace") is None
        else np.asarray(bundle["action_saturation_trace"], float).reshape(-1)
    )
    bundle["entropy_trace"] = (
        None if bundle.get("entropy_trace") is None else np.asarray(bundle["entropy_trace"], float).reshape(-1)
    )
    bundle["mean_log_prob_trace"] = (
        None if bundle.get("mean_log_prob_trace") is None else np.asarray(bundle["mean_log_prob_trace"], float).reshape(-1)
    )
    for key in (
        "matrix_actor_losses",
        "matrix_critic_losses",
        "matrix_alpha_losses",
        "matrix_alphas",
        "matrix_critic_q1_trace",
        "matrix_critic_q2_trace",
        "matrix_critic_q_gap_trace",
        "matrix_exploration_trace",
        "matrix_exploration_magnitude_trace",
        "matrix_param_noise_scale_trace",
        "matrix_action_saturation_trace",
        "matrix_entropy_trace",
        "matrix_mean_log_prob_trace",
        "weight_actor_losses",
        "weight_critic_losses",
        "weight_alpha_losses",
        "weight_alphas",
        "weight_critic_q1_trace",
        "weight_critic_q2_trace",
        "weight_critic_q_gap_trace",
        "weight_exploration_trace",
        "weight_exploration_magnitude_trace",
        "weight_param_noise_scale_trace",
        "weight_action_saturation_trace",
        "weight_entropy_trace",
        "weight_mean_log_prob_trace",
        "residual_actor_losses",
        "residual_critic_losses",
        "residual_alpha_losses",
        "residual_alphas",
        "residual_critic_q1_trace",
        "residual_critic_q2_trace",
        "residual_critic_q_gap_trace",
        "residual_exploration_trace",
        "residual_exploration_magnitude_trace",
        "residual_param_noise_scale_trace",
        "residual_action_saturation_trace",
        "residual_entropy_trace",
        "residual_mean_log_prob_trace",
        "horizon_dqn_loss_trace",
        "horizon_exploration_trace",
        "horizon_epsilon_trace",
        "horizon_avg_td_error_trace",
        "horizon_avg_max_q_trace",
        "horizon_avg_value_trace",
        "horizon_avg_advantage_spread_trace",
        "horizon_avg_chosen_q_trace",
        "horizon_noisy_sigma_trace",
        "horizon_reward_n_mean_trace",
        "horizon_discount_n_mean_trace",
        "horizon_bootstrap_q_mean_trace",
        "horizon_n_actual_mean_trace",
        "horizon_truncated_fraction_trace",
        "horizon_lambda_return_mean_trace",
        "horizon_offpolicy_rho_mean_trace",
        "horizon_offpolicy_c_mean_trace",
        "horizon_behavior_logprob_mean_trace",
        "horizon_retrace_c_clip_fraction_trace",
        "matrix_reward_n_mean_trace",
        "matrix_discount_n_mean_trace",
        "matrix_bootstrap_q_mean_trace",
        "matrix_n_actual_mean_trace",
        "matrix_truncated_fraction_trace",
        "matrix_lambda_return_mean_trace",
        "matrix_target_logprob_mean_trace",
        "weight_reward_n_mean_trace",
        "weight_discount_n_mean_trace",
        "weight_bootstrap_q_mean_trace",
        "weight_n_actual_mean_trace",
        "weight_truncated_fraction_trace",
        "weight_lambda_return_mean_trace",
        "weight_target_logprob_mean_trace",
        "residual_reward_n_mean_trace",
        "residual_discount_n_mean_trace",
        "residual_bootstrap_q_mean_trace",
        "residual_n_actual_mean_trace",
        "residual_truncated_fraction_trace",
        "residual_lambda_return_mean_trace",
        "residual_target_logprob_mean_trace",
    ):
        bundle[key] = None if bundle.get(key) is None else np.asarray(bundle[key], float).reshape(-1)
    bundle["low_coef"] = None if bundle.get("low_coef") is None else np.asarray(bundle["low_coef"], float).reshape(-1)
    bundle["high_coef"] = None if bundle.get("high_coef") is None else np.asarray(bundle["high_coef"], float).reshape(-1)
    bundle["matrix_low_coef"] = (
        None if bundle.get("matrix_low_coef") is None else np.asarray(bundle["matrix_low_coef"], float).reshape(-1)
    )
    bundle["matrix_high_coef"] = (
        None if bundle.get("matrix_high_coef") is None else np.asarray(bundle["matrix_high_coef"], float).reshape(-1)
    )
    bundle["weight_low_coef"] = (
        None if bundle.get("weight_low_coef") is None else np.asarray(bundle["weight_low_coef"], float).reshape(-1)
    )
    bundle["weight_high_coef"] = (
        None if bundle.get("weight_high_coef") is None else np.asarray(bundle["weight_high_coef"], float).reshape(-1)
    )
    bundle["residual_low_coef"] = (
        None if bundle.get("residual_low_coef") is None else np.asarray(bundle["residual_low_coef"], float).reshape(-1)
    )
    bundle["residual_high_coef"] = (
        None if bundle.get("residual_high_coef") is None else np.asarray(bundle["residual_high_coef"], float).reshape(-1)
    )
    for prefix in ("", "matrix_", "weight_", "residual_"):
        enabled_key = f"{prefix}phase1_enabled"
        if enabled_key in bundle:
            bundle[enabled_key] = bool(bundle.get(enabled_key, False))
        for key in (
            "phase1_action_freeze_subepisodes",
            "phase1_actor_freeze_subepisodes",
            "phase1_action_freeze_start_step",
            "phase1_action_freeze_end_step",
            "phase1_first_live_action_step",
            "phase1_actor_freeze_train_steps",
            "phase1_effective_actor_freeze",
        ):
            full_key = f"{prefix}{key}"
            if full_key in bundle:
                bundle[full_key] = int(bundle.get(full_key, 0) or 0)
        for key in (
            "phase1_hidden_window_active_log",
            "phase1_action_source_log",
            "critic_update_env_step_trace",
            "actor_update_slot_env_step_trace",
            "actor_update_applied_env_step_trace",
            "actor_update_blocked_env_step_trace",
        ):
            full_key = f"{prefix}{key}"
            bundle[full_key] = None if bundle.get(full_key) is None else np.asarray(bundle[full_key], int).reshape(-1)
        for key in ("policy_action_raw_log", "executed_action_raw_log"):
            full_key = f"{prefix}{key}"
            bundle[full_key] = None if bundle.get(full_key) is None else np.asarray(bundle[full_key], float)
    bundle["n_step"] = int(bundle.get("n_step", 1) or 1)
    bundle["multistep_mode"] = str(bundle.get("multistep_mode", "one_step"))
    bundle["lambda_value"] = bundle.get("lambda_value", None)
    bundle["state_mode"] = str(bundle.get("state_mode", "standard")).lower()
    bundle["system_metadata"] = dict(bundle.get("system_metadata") or {})

    if bundle["y"].shape[0] == bundle["nFE"]:
        bundle["y_line_full"] = np.vstack([bundle["y"], bundle["y"][-1:, :]])
    else:
        bundle["y_line_full"] = bundle["y"][: bundle["nFE"] + 1, :]

    bundle["u_step_full"] = bundle["u"][: bundle["nFE"], :]
    bundle["n_inputs"] = bundle["u_step_full"].shape[1]
    bundle["n_outputs"] = bundle["y_line_full"].shape[1]

    if bundle["delta_log"] is not None:
        if bundle["delta_log"].ndim == 1:
            bundle["delta_log"] = bundle["delta_log"][:, None]
        if bundle["delta_log"].shape[0] > bundle["nFE"]:
            bundle["delta_log"] = bundle["delta_log"][: bundle["nFE"], :]

    if bundle["alpha_log"] is not None and bundle["alpha_log"].shape[0] > bundle["nFE"]:
        bundle["alpha_log"] = bundle["alpha_log"][: bundle["nFE"]]
    if bundle["matrix_alpha_log"] is not None and bundle["matrix_alpha_log"].shape[0] > bundle["nFE"]:
        bundle["matrix_alpha_log"] = bundle["matrix_alpha_log"][: bundle["nFE"]]
    for key in (
        "eta_A_log",
        "eta_B_log",
        "eta_A_raw_log",
        "eta_B_raw_log",
        "id_residual_norm_log",
        "id_condition_number_log",
        "active_A_model_delta_ratio_log",
        "active_B_model_delta_ratio_log",
        "candidate_A_model_delta_ratio_log",
        "candidate_B_model_delta_ratio_log",
        "pred_A_model_delta_ratio_log",
        "pred_B_model_delta_ratio_log",
        "observer_A_model_delta_ratio_log",
        "observer_B_model_delta_ratio_log",
        "observer_refresh_event_log",
        "observer_refresh_success_log",
    ):
        if bundle.get(key) is not None and bundle[key].shape[0] > bundle["nFE"]:
            bundle[key] = bundle[key][: bundle["nFE"]]

    if bundle["weight_log"] is not None:
        if bundle["weight_log"].ndim == 1:
            bundle["weight_log"] = bundle["weight_log"][:, None]
        if bundle["weight_log"].shape[0] > bundle["nFE"]:
            bundle["weight_log"] = bundle["weight_log"][: bundle["nFE"], :]
    if bundle["matrix_delta_log"] is not None:
        if bundle["matrix_delta_log"].ndim == 1:
            bundle["matrix_delta_log"] = bundle["matrix_delta_log"][:, None]
        if bundle["matrix_delta_log"].shape[0] > bundle["nFE"]:
            bundle["matrix_delta_log"] = bundle["matrix_delta_log"][: bundle["nFE"], :]

    if bundle["residual_exec_log"] is not None:
        if bundle["residual_exec_log"].ndim == 1:
            bundle["residual_exec_log"] = bundle["residual_exec_log"][:, None]
        if bundle["residual_exec_log"].shape[0] > bundle["nFE"]:
            bundle["residual_exec_log"] = bundle["residual_exec_log"][: bundle["nFE"], :]

    if bundle["residual_raw_log"] is not None:
        if bundle["residual_raw_log"].ndim == 1:
            bundle["residual_raw_log"] = bundle["residual_raw_log"][:, None]
        if bundle["residual_raw_log"].shape[0] > bundle["nFE"]:
            bundle["residual_raw_log"] = bundle["residual_raw_log"][: bundle["nFE"], :]

    for key in ("a_res_raw_log", "a_res_exec_log", "delta_u_res_raw_log", "delta_u_res_exec_log"):
        if bundle.get(key) is not None:
            if bundle[key].ndim == 1:
                bundle[key] = bundle[key][:, None]
            if bundle[key].shape[0] > bundle["nFE"]:
                bundle[key] = bundle[key][: bundle["nFE"], :]

    for prefix in ("", "matrix_", "weight_", "residual_"):
        for key in ("phase1_hidden_window_active_log", "phase1_action_source_log"):
            full_key = f"{prefix}{key}"
            if bundle.get(full_key) is not None and bundle[full_key].shape[0] > bundle["nFE"]:
                bundle[full_key] = bundle[full_key][: bundle["nFE"]]
        for key in ("policy_action_raw_log", "executed_action_raw_log"):
            full_key = f"{prefix}{key}"
            if bundle.get(full_key) is not None:
                if bundle[full_key].ndim == 1:
                    bundle[full_key] = bundle[full_key][:, None]
                if bundle[full_key].shape[0] > bundle["nFE"]:
                    bundle[full_key] = bundle[full_key][: bundle["nFE"], :]

    if bundle["u_base"] is not None:
        if bundle["u_base"].shape[0] > bundle["nFE"]:
            bundle["u_base"] = bundle["u_base"][: bundle["nFE"], :]

    for key in ("innovation_log", "tracking_error_log", "tracking_scale_log"):
        if bundle[key] is not None:
            if bundle[key].ndim == 1:
                bundle[key] = bundle[key][:, None]
            if bundle[key].shape[0] > bundle["nFE"]:
                bundle[key] = bundle[key][: bundle["nFE"], :]
    for key in (
        "horizon_innovation_log",
        "horizon_innovation_raw_log",
        "horizon_tracking_error_log",
        "horizon_tracking_error_raw_log",
        "horizon_tracking_scale_log",
        "matrix_innovation_log",
        "matrix_innovation_raw_log",
        "matrix_tracking_error_log",
        "matrix_tracking_error_raw_log",
        "matrix_tracking_scale_log",
        "weight_innovation_log",
        "weight_innovation_raw_log",
        "weight_tracking_error_log",
        "weight_tracking_error_raw_log",
        "weight_tracking_scale_log",
        "residual_innovation_log",
        "residual_innovation_raw_log",
        "residual_tracking_error_log",
        "residual_tracking_error_raw_log",
        "residual_tracking_scale_log",
    ):
        if bundle[key] is not None:
            if bundle[key].ndim == 1:
                bundle[key] = bundle[key][:, None]
            if bundle[key].shape[0] > bundle["nFE"]:
                bundle[key] = bundle[key][: bundle["nFE"], :]

    return bundle


def normalize_external_bundle(result_bundle, reference_bundle):
    merged = dict(result_bundle)
    for key in ("y_sp", "steady_states", "nFE", "delta_t", "time_in_sub_episodes", "data_min", "data_max"):
        if key not in merged:
            merged[key] = reference_bundle[key]
    return normalize_result_bundle(merged)


def build_storage_bundle(bundle, start_episode):
    stored = dict(bundle)
    stored["y_rl"] = bundle["y_line_full"]
    stored["u_rl"] = bundle["u_step_full"]
    stored["y_mpc"] = bundle["y_line_full"]
    stored["u_mpc"] = bundle["u_step_full"]
    stored["y"] = bundle["y_line_full"]
    stored["u"] = bundle["u_step_full"]
    stored["start_episode_plotted"] = int(start_episode)
    return stored


def create_output_dir(directory, prefix_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_bundle_pickle(out_dir, stored_bundle):
    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as handle:
        pickle.dump(stored_bundle, handle)


def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs):
    y_ss_phys = np.asarray(steady_states["y_ss"], float)
    y_ss_scaled = apply_min_max(y_ss_phys, data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled = np.asarray(y_sp_scaled_dev, float) + y_ss_scaled
    return reverse_min_max(y_sp_scaled, data_min[n_inputs:], data_max[n_inputs:])


def slice_avg_rewards(avg_rewards, n_ep_total, start_episode):
    avg = np.asarray(avg_rewards, float)
    if len(avg) == n_ep_total + 1:
        avg = avg[1:]
    start_episode = int(max(1, start_episode))
    start_idx = max(0, start_episode - 1)
    avg = avg[start_idx:]
    x = np.arange(start_episode, start_episode + len(avg))
    return x, avg


def episode_spans(test_train_dict, nFE):
    if not test_train_dict:
        return []
    starts = sorted(int(k) for k in test_train_dict.keys())
    spans = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else int(nFE)
        spans.append((start, end, bool(test_train_dict[start])))
    return spans


def shade_test_regions(ax, spans, delta_t):
    for start, end, is_test in spans:
        if is_test:
            ax.axvspan(start * delta_t, end * delta_t, color="orange", alpha=0.12, linewidth=0)


def load_pickle(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "input_data.pkl")
        if os.path.exists(candidate):
            with open(candidate, "rb") as handle:
                return pickle.load(handle)
        for name in sorted(os.listdir(path)):
            if name.endswith((".pickle", ".pkl")):
                with open(os.path.join(path, name), "rb") as handle:
                    return pickle.load(handle)
        raise FileNotFoundError(f"No pickle files found in directory: {path}")

    with open(path, "rb") as handle:
        return pickle.load(handle)


def recompute_step_rewards(y_line_full, u_step, y_sp, steady_states, data_min, data_max, reward_fn):
    n_inputs = u_step.shape[1]
    nFE = min(len(y_sp), len(u_step))
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])

    y_scaled_abs = apply_min_max(y_line_full[: nFE + 1, :], data_min[n_inputs:], data_max[n_inputs:])
    y_scaled_dev = y_scaled_abs - y_ss_scaled
    u_scaled_abs = apply_min_max(u_step[:nFE, :], data_min[:n_inputs], data_max[:n_inputs])

    delta_y = y_scaled_dev[1:, :] - y_sp[:nFE, :]
    delta_u = np.zeros((nFE, n_inputs))
    delta_u[0, :] = u_scaled_abs[0, :] - ss_scaled_inputs
    if nFE > 1:
        delta_u[1:, :] = u_scaled_abs[1:, :] - u_scaled_abs[:-1, :]

    y_sp_phys = ysp_scaled_dev_to_phys(y_sp[:nFE, :], steady_states, data_min, data_max, n_inputs)
    rewards = np.zeros(nFE)
    for idx in range(nFE):
        rewards[idx] = reward_fn(delta_y[idx], delta_u[idx], y_sp_phys=y_sp_phys[idx])
    return rewards, delta_y, delta_u


def _plot_mismatch_diagnostics(bundle, out_dir, prefix, t_step, t_step_blk, start_step, W, s_last, last_steps, spans, delta_t, save_pdf):
    innovation_log = bundle.get("innovation_log")
    tracking_error_log = bundle.get("tracking_error_log")
    if innovation_log is None and tracking_error_log is None:
        return

    n_outputs = bundle["n_outputs"]

    def _plot_series(series, ylabel_stem, title_stem, fname_base, t_axis):
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.3 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_axis, series[:, idx])
            if len(t_axis) == len(t_step):
                shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"{ylabel_stem}{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, f"{prefix}_{fname_base}", save_pdf=save_pdf)

    if innovation_log is not None:
        innov_seg = innovation_log[start_step : start_step + W, :]
        _plot_series(innov_seg, "innov", "Innovation", "innovation_full", t_step)
        _plot_series(innov_seg[s_last : s_last + last_steps, :], "innov", "Innovation", "innovation_last_block", t_step_blk)

    if tracking_error_log is not None:
        terr_seg = tracking_error_log[start_step : start_step + W, :]
        _plot_series(terr_seg, "etrack", "Tracking Error", "tracking_state_full", t_step)
        _plot_series(
            terr_seg[s_last : s_last + last_steps, :],
            "etrack",
            "Tracking Error",
            "tracking_state_last_block",
            t_step_blk,
        )


def _plot_nstep_diagnostics(out_dir, fname_base, bundle, save_pdf):
    traces = [
        ("reward_n", bundle.get("reward_n_mean_trace")),
        ("discount_n", bundle.get("discount_n_mean_trace")),
        ("bootstrap", bundle.get("bootstrap_q_mean_trace")),
        ("n_actual", bundle.get("n_actual_mean_trace")),
        ("truncated", bundle.get("truncated_fraction_trace")),
        ("lambda_return", bundle.get("lambda_return_mean_trace")),
        ("rho", bundle.get("offpolicy_rho_mean_trace")),
        ("c", bundle.get("offpolicy_c_mean_trace")),
        ("behavior_logp", bundle.get("behavior_logprob_mean_trace")),
        ("target_logp", bundle.get("target_logprob_mean_trace")),
        ("retrace_clip", bundle.get("retrace_c_clip_fraction_trace")),
    ]
    active = [(label, np.asarray(values, float).reshape(-1)) for label, values in traces if values is not None and len(values) > 0]
    if not active:
        return

    fig, axs = plt.subplots(len(active), 1, figsize=(8.4, 3.0 + 2.0 * max(1, len(active) - 1)), sharex=True)
    if len(active) == 1:
        axs = [axs]
    for ax, (label, values) in zip(axs, active):
        ax.plot(np.arange(1, len(values) + 1), values)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel("Update #")
    _save_fig(fig, out_dir, fname_base, save_pdf=save_pdf)


def _phase1_window_info(bundle, prefix=""):
    if not bool(bundle.get(f"{prefix}phase1_enabled", False)):
        return None
    warm_start_step = int(bundle.get("warm_start_step", 0))
    hidden_start = int(bundle.get(f"{prefix}phase1_action_freeze_start_step", warm_start_step + 1))
    hidden_end = int(bundle.get(f"{prefix}phase1_action_freeze_end_step", warm_start_step))
    time_in_sub_episodes = int(max(1, bundle.get("time_in_sub_episodes", 1)))
    nFE = int(bundle["nFE"])
    window_start = max(0, warm_start_step - time_in_sub_episodes)
    window_end = min(nFE - 1, hidden_end + time_in_sub_episodes)
    return {
        "warm_start_step": warm_start_step,
        "hidden_start": hidden_start,
        "hidden_end": hidden_end,
        "window_start": window_start,
        "window_end": window_end,
    }


def _shade_phase1_regions(ax, info, delta_t):
    ax.axvspan(
        info["window_start"] * delta_t,
        (info["warm_start_step"] + 1) * delta_t,
        color="0.75",
        alpha=0.12,
        linewidth=0,
    )
    if info["hidden_end"] >= info["hidden_start"]:
        ax.axvspan(
            info["hidden_start"] * delta_t,
            (info["hidden_end"] + 1) * delta_t,
            color="#C81D25",
            alpha=0.10,
            linewidth=0,
        )
    ax.axvline((info["warm_start_step"] + 1) * delta_t, color="0.35", linestyle="--", linewidth=1.2)


def _plot_phase1_release_window_single_agent(bundle, out_dir, time_label, save_pdf):
    info = _phase1_window_info(bundle)
    if info is None:
        return
    policy = bundle.get("policy_action_raw_log")
    executed = bundle.get("executed_action_raw_log")
    if policy is None or executed is None:
        return

    window_start = info["window_start"]
    window_end = info["window_end"]
    delta_t = float(bundle["delta_t"])
    spans = episode_spans(bundle.get("test_train_dict"), bundle["nFE"])
    t_window = np.arange(window_start, window_end + 1) * delta_t
    policy_seg = policy[window_start : window_end + 1, :]
    executed_seg = executed[window_start : window_end + 1, :]
    n_dims = int(policy_seg.shape[1])

    fig, axs = plt.subplots(n_dims, 1, figsize=(8.8, 2.8 + 2.0 * max(1, n_dims - 1)), sharex=True)
    if n_dims == 1:
        axs = [axs]
    for dim_idx, ax in enumerate(axs):
        ax.step(t_window, policy_seg[:, dim_idx], where="post", label="Policy Raw")
        ax.step(t_window, executed_seg[:, dim_idx], where="post", linestyle="--", label="Executed Raw")
        shade_test_regions(ax, spans, delta_t)
        _shade_phase1_regions(ax, info, delta_t)
        ax.set_ylabel(f"a[{dim_idx + 1}]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_phase1_release_window_actions", save_pdf=save_pdf)

    critic_steps = np.asarray(bundle.get("critic_update_env_step_trace"), int).reshape(-1)
    actor_slot_steps = np.asarray(bundle.get("actor_update_slot_env_step_trace"), int).reshape(-1)
    actor_applied_steps = np.asarray(bundle.get("actor_update_applied_env_step_trace"), int).reshape(-1)
    actor_blocked_steps = np.asarray(bundle.get("actor_update_blocked_env_step_trace"), int).reshape(-1)

    def _window_trace(trace):
        return trace[(trace >= window_start) & (trace <= window_end)]

    fig, axs = plt.subplots(2, 1, figsize=(8.8, 6.0), sharex=True)
    critic_window = _window_trace(critic_steps)
    if critic_window.size > 0:
        axs[0].scatter(critic_window * delta_t, np.ones_like(critic_window, dtype=float), marker="|", s=200, label="Critic")
    shade_test_regions(axs[0], spans, delta_t)
    _shade_phase1_regions(axs[0], info, delta_t)
    axs[0].set_ylabel("Critic")
    axs[0].set_yticks([1.0])
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    _make_axes_bold(axs[0])

    actor_slot_window = _window_trace(actor_slot_steps)
    actor_applied_window = _window_trace(actor_applied_steps)
    actor_blocked_window = _window_trace(actor_blocked_steps)
    if actor_slot_window.size > 0:
        axs[1].scatter(actor_slot_window * delta_t, np.full(actor_slot_window.size, 1.0), marker="|", s=180, label="Slot")
    if actor_blocked_window.size > 0:
        axs[1].scatter(actor_blocked_window * delta_t, np.full(actor_blocked_window.size, 0.7), marker="x", s=60, label="Blocked")
    if actor_applied_window.size > 0:
        axs[1].scatter(actor_applied_window * delta_t, np.full(actor_applied_window.size, 0.4), marker="o", s=35, label="Applied")
    shade_test_regions(axs[1], spans, delta_t)
    _shade_phase1_regions(axs[1], info, delta_t)
    axs[1].set_ylabel("Actor")
    axs[1].set_yticks([1.0, 0.7, 0.4])
    axs[1].set_yticklabels(["slot", "blocked", "applied"])
    axs[1].set_xlabel(time_label)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].legend(loc="best")
    _make_axes_bold(axs[1])
    _save_fig(fig, out_dir, "fig_phase1_release_window_updates", save_pdf=save_pdf)


def _plot_phase1_release_window_combined(bundle, out_dir, time_label, save_pdf):
    families = [
        ("matrix_", "Matrix"),
        ("weight_", "Weights"),
        ("residual_", "Residual"),
    ]
    active = [(prefix, label, _phase1_window_info(bundle, prefix)) for prefix, label in families]
    active = [(prefix, label, info) for prefix, label, info in active if info is not None]
    if not active:
        return

    delta_t = float(bundle["delta_t"])
    spans = episode_spans(bundle.get("test_train_dict"), bundle["nFE"])
    window_start = min(info["window_start"] for _, _, info in active)
    window_end = max(info["window_end"] for _, _, info in active)
    t_window = np.arange(window_start, window_end + 1) * delta_t

    fig, axs = plt.subplots(len(active), 1, figsize=(9.4, 3.0 + 2.4 * max(1, len(active) - 1)), sharex=True)
    if len(active) == 1:
        axs = [axs]
    for ax, (prefix, label, info) in zip(axs, active):
        policy = bundle.get(f"{prefix}policy_action_raw_log")
        executed = bundle.get(f"{prefix}executed_action_raw_log")
        if policy is None or executed is None:
            continue
        policy_seg = policy[window_start : window_end + 1, :]
        executed_seg = executed[window_start : window_end + 1, :]
        for dim_idx in range(policy_seg.shape[1]):
            ax.step(t_window, policy_seg[:, dim_idx], where="post", linewidth=1.8, alpha=0.9, label=f"p{dim_idx + 1}")
            ax.step(t_window, executed_seg[:, dim_idx], where="post", linestyle="--", linewidth=1.4, alpha=0.9, label=f"e{dim_idx + 1}")
        shade_test_regions(ax, spans, delta_t)
        _shade_phase1_regions(ax, info, delta_t)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best", ncol=2)
    _save_fig(fig, out_dir, "fig_combined_phase1_release_window_td3_actions", save_pdf=save_pdf)

    fig, axs = plt.subplots(len(active), 1, figsize=(9.4, 3.0 + 2.2 * max(1, len(active) - 1)), sharex=True)
    if len(active) == 1:
        axs = [axs]
    for ax, (prefix, label, info) in zip(axs, active):
        def _window_trace(key):
            trace = np.asarray(bundle.get(f"{prefix}{key}"), int).reshape(-1)
            return trace[(trace >= window_start) & (trace <= window_end)]

        critic_steps = _window_trace("critic_update_env_step_trace")
        slot_steps = _window_trace("actor_update_slot_env_step_trace")
        blocked_steps = _window_trace("actor_update_blocked_env_step_trace")
        applied_steps = _window_trace("actor_update_applied_env_step_trace")
        if critic_steps.size > 0:
            ax.scatter(critic_steps * delta_t, np.full(critic_steps.size, 1.0), marker="|", s=180, label="critic")
        if slot_steps.size > 0:
            ax.scatter(slot_steps * delta_t, np.full(slot_steps.size, 0.75), marker="|", s=160, label="slot")
        if blocked_steps.size > 0:
            ax.scatter(blocked_steps * delta_t, np.full(blocked_steps.size, 0.5), marker="x", s=55, label="blocked")
        if applied_steps.size > 0:
            ax.scatter(applied_steps * delta_t, np.full(applied_steps.size, 0.25), marker="o", s=32, label="applied")
        shade_test_regions(ax, spans, delta_t)
        _shade_phase1_regions(ax, info, delta_t)
        ax.set_ylabel(label)
        ax.set_yticks([1.0, 0.75, 0.5, 0.25])
        ax.set_yticklabels(["c", "s", "b", "a"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best", ncol=2)
    _save_fig(fig, out_dir, "fig_combined_phase1_release_window_td3_updates", save_pdf=save_pdf)


def plot_baseline_mpc_results_core(result_bundle, plot_cfg):
    style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
    _set_plot_style(style_profile=style_profile)
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "mpc_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]
    metadata = resolve_system_metadata(bundle=bundle, plot_cfg=plot_cfg, n_outputs=n_outputs, n_inputs=n_inputs)
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]
    time_label = metadata["time_label"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="MPC")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_mpc_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="MPC")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_mpc_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_mpc_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_mpc_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    x_ep_mpc, y_ep_mpc = slice_avg_rewards(bundle.get("avg_rewards_mpc", []), n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-", label="Shared reward")
    if len(y_ep_mpc) > 0:
        ax.plot(x_ep_mpc, y_ep_mpc, "s--", label="Quadratic MPC reward")
    ax.set_ylabel("Average Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_mpc_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.plot(t_step, rewards_seg, label="Shared reward")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_mpc_step_rewards", save_pdf=save_pdf)

    rewards_mpc = bundle.get("rewards_mpc")
    if rewards_mpc is not None:
        rewards_mpc_seg = rewards_mpc[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.plot(t_step, rewards_mpc_seg, label="Quadratic MPC reward")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_mpc_step_rewards_quadratic", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        dy_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.2, 4.0 + 2.0 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, dy_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        axs[-1].plot(t_step, np.linalg.norm(dy_seg, axis=1))
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel(r"$||e||_2$")
        axs[-1].set_xlabel(time_label)
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_mpc_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(rf"$\Delta u_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_mpc_delta_u", save_pdf=save_pdf)

    if bundle.get("yhat") is not None:
        yhat_seg = bundle["yhat"][:, start_step : start_step + W].T
        y_meas_seg = y_line[1 : 1 + W, :]
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.2, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_step, y_meas_seg[:, idx], label="Measured")
            ax.plot(t_step, yhat_seg[:, idx], linestyle="--", label="Observer")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_mpc_observer_overlay", save_pdf=save_pdf)

    if bundle.get("disturbance_profile") is not None:
        disturbance_items = disturbance_plot_items(bundle["disturbance_profile"], metadata.get("disturbance_labels"))
        if disturbance_items:
            fig, axs = plt.subplots(len(disturbance_items), 1, figsize=(8.2, 3.0 + 2.0 * max(1, len(disturbance_items) - 1)), sharex=True)
            if len(disturbance_items) == 1:
                axs = [axs]
            for ax, (_, label, series) in zip(axs, disturbance_items):
                ax.plot(t_step, series[start_step : start_step + W])
                shade_test_regions(ax, spans, delta_t)
                ax.set_ylabel(label)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
            axs[-1].set_xlabel(time_label)
            _save_fig(fig, out_dir, "fig_mpc_disturbance_profile", save_pdf=save_pdf)
    stored_bundle = build_storage_bundle(bundle, start_episode)
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_horizon_results_core(result_bundle, plot_cfg):
    _set_plot_style(style_profile=plot_cfg.get("style_profile", "hybrid"))
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "horizon_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    recipe_counts = bool(plot_cfg.get("recipe_counts", True))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]
    metadata = resolve_system_metadata(bundle=bundle, plot_cfg=plot_cfg, n_outputs=n_outputs, n_inputs=n_inputs)
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]
    time_label = metadata["time_label"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_horizon_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_horizon_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_horizon_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_horizon_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_horizon_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_horizon_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel(time_label)
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_horizon_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(f"du{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_horizon_delta_u", save_pdf=save_pdf)

    horizon_trace = bundle.get("horizon_trace")
    mpc_horizons = bundle.get("mpc_horizons")
    if horizon_trace is not None:
        ht_full = np.asarray(horizon_trace, float)
        if ht_full.shape[0] > nFE:
            ht_full = ht_full[:nFE, :]
        ht = ht_full[start_step : start_step + W, :]
        Hp = ht[:, 0].astype(int)
        Hc = ht[:, 1].astype(int)
        Hp0 = None if not mpc_horizons else int(mpc_horizons[0])
        Hc0 = None if not mpc_horizons else int(mpc_horizons[1])

        fig, axs = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
        axs[0].step(t_step, Hp, where="post")
        axs[1].step(t_step, Hc, where="post")
        for ax, val, label in [(axs[0], Hp0, "Hp"), (axs[1], Hc0, "Hc")]:
            if val is not None:
                ax.axhline(float(val), color="red", linestyle=":", linewidth=2.0)
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_horizon_horizons_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
        axs[0].step(t_step_blk, Hp[s_last : s_last + last_steps], where="post")
        axs[1].step(t_step_blk, Hc[s_last : s_last + last_steps], where="post")
        for ax, val, label in [(axs[0], Hp0, "Hp"), (axs[1], Hc0, "Hc")]:
            if val is not None:
                ax.axhline(float(val), color="red", linestyle=":", linewidth=2.0)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        _save_fig(fig, out_dir, "fig_horizon_horizons_last_block", save_pdf=save_pdf)

        if recipe_counts:
            counts = collections.Counter(list(zip(Hp.tolist(), Hc.tolist())))
            items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            top = items[:20]
            key0 = None if Hp0 is None or Hc0 is None else (Hp0, Hc0)
            if key0 is not None and key0 not in dict(top):
                top.append((key0, counts.get(key0, 0)))
            labels = [f"Hp={k[0]},Hc={k[1]}" for k, _ in top]
            freqs = [v for _, v in top]
            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            if freqs:
                bars = ax.bar(np.arange(len(freqs)), freqs)
                if key0 is not None:
                    for idx, (k, _) in enumerate(top):
                        if k == key0:
                            bars[idx].set_color("red")
                            ax.text(idx, freqs[idx], "MPC", ha="center", va="bottom", fontweight="bold")
                            break
                ax.set_xticks(np.arange(len(freqs)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_horizon_recipe_counts", save_pdf=save_pdf)

    action_trace = bundle.get("action_trace")
    if action_trace is not None:
        action_seg = np.asarray(action_trace, int)[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.step(t_step, action_seg, where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Action")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_horizon_action_trace", save_pdf=save_pdf)

    loss_trace = bundle.get("dqn_loss_trace")
    exploration_trace = bundle.get("exploration_trace")
    epsilon_trace = bundle.get("epsilon_trace")
    td_trace = bundle.get("avg_td_error_trace")
    max_q_trace = bundle.get("avg_max_q_trace")
    chosen_q_trace = bundle.get("avg_chosen_q_trace")
    value_trace = bundle.get("avg_value_trace")
    adv_spread_trace = bundle.get("avg_advantage_spread_trace")
    noisy_sigma_trace = bundle.get("noisy_sigma_trace")
    reward_n_trace = bundle.get("reward_n_mean_trace")
    discount_n_trace = bundle.get("discount_n_mean_trace")
    bootstrap_q_trace = bundle.get("bootstrap_q_mean_trace")
    n_actual_trace = bundle.get("n_actual_mean_trace")
    truncated_fraction_trace = bundle.get("truncated_fraction_trace")
    lambda_return_trace = bundle.get("lambda_return_mean_trace")
    rho_trace = bundle.get("offpolicy_rho_mean_trace")
    c_trace = bundle.get("offpolicy_c_mean_trace")
    behavior_logprob_trace = bundle.get("behavior_logprob_mean_trace")
    retrace_clip_trace = bundle.get("retrace_c_clip_fraction_trace")
    if any(trace is not None and len(trace) > 0 for trace in (loss_trace, exploration_trace, td_trace, max_q_trace, chosen_q_trace, value_trace, adv_spread_trace, noisy_sigma_trace, lambda_return_trace, rho_trace, c_trace, behavior_logprob_trace, retrace_clip_trace)):
        traces = [
            ("Loss", loss_trace),
            ("Exploration", exploration_trace),
            ("Avg |TD|", td_trace),
            ("Avg max Q", max_q_trace),
            ("Avg chosen Q", chosen_q_trace),
            ("Avg V(s)", value_trace),
            ("Avg adv spread", adv_spread_trace),
            ("Noisy sigma", noisy_sigma_trace),
            ("Reward_n", reward_n_trace),
            ("Discount_n", discount_n_trace),
            ("Bootstrap Q", bootstrap_q_trace),
            ("n_actual", n_actual_trace),
            ("Truncated", truncated_fraction_trace),
            ("Lambda return", lambda_return_trace),
            ("Retrace rho", rho_trace),
            ("Retrace c", c_trace),
            ("Behavior logp", behavior_logprob_trace),
            ("Retrace clip", retrace_clip_trace),
        ]
        active_traces = [(label, np.asarray(trace, float).reshape(-1)) for label, trace in traces if trace is not None and len(trace) > 0]
        fig, axs = plt.subplots(len(active_traces), 1, figsize=(8.6, 3.0 + 2.0 * max(1, len(active_traces) - 1)), sharex=True)
        if len(active_traces) == 1:
            axs = [axs]
        for ax, (label, trace) in zip(axs, active_traces):
            ax.plot(np.arange(1, len(trace) + 1), trace)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Update #")
        _save_fig(fig, out_dir, "fig_horizon_dqn_training_diagnostics", save_pdf=save_pdf)

    if epsilon_trace is not None and len(epsilon_trace) > 0:
        eps = np.asarray(epsilon_trace, float).reshape(-1)
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        ax.plot(np.arange(1, len(eps) + 1), eps)
        ax.set_ylabel("Epsilon")
        ax.set_xlabel("Update #")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_horizon_epsilon_trace", save_pdf=save_pdf)

    _plot_nstep_diagnostics(out_dir, "fig_horizon_nstep_decomposition", bundle, save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_horizon_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        disturbance_items = disturbance_plot_items(disturbance_profile, metadata.get("disturbance_labels"))
        if disturbance_items:
            fig, axs = plt.subplots(len(disturbance_items), 1, figsize=(8.6, 3.0 + 2.0 * max(1, len(disturbance_items) - 1)), sharex=True)
            if len(disturbance_items) == 1:
                axs = [axs]
            for idx, (_, label, series) in enumerate(disturbance_items):
                axs[idx].plot(t_step, series[start_step : start_step + W])
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(label)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel(time_label)
            _save_fig(fig, out_dir, "fig_horizon_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_horizon",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    stored_bundle = build_storage_bundle(bundle, start_episode)
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_matrix_multiplier_results_core(result_bundle, plot_cfg):
    style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
    _set_plot_style(style_profile=style_profile)
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "matrix_multiplier_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]
    metadata = resolve_system_metadata(bundle=bundle, plot_cfg=plot_cfg, n_outputs=n_outputs, n_inputs=n_inputs)
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]
    time_label = metadata["time_label"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_matrix_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_matrix_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_matrix_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_matrix_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_matrix_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_matrix_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e[{idx + 1}]")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel(time_label)
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_matrix_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(rf"$\Delta u_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_matrix_delta_u", save_pdf=save_pdf)

    alpha_log = bundle.get("alpha_log")
    delta_log = bundle.get("delta_log")
    low_coef = bundle.get("low_coef")
    high_coef = bundle.get("high_coef")
    if alpha_log is not None:
        alpha_seg = alpha_log[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        if low_coef is not None and high_coef is not None and low_coef.size > 0:
            ax.fill_between(t_step, float(low_coef[0]), float(high_coef[0]), color="tab:blue", alpha=0.12, step="post")
            ax.axhline(float(low_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
            ax.axhline(float(high_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
        ax.step(t_step, alpha_seg, where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("alpha")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_matrix_alpha_full", save_pdf=save_pdf)

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        if low_coef is not None and high_coef is not None and low_coef.size > 0:
            ax.fill_between(t_step_blk, float(low_coef[0]), float(high_coef[0]), color="tab:blue", alpha=0.12, step="post")
            ax.axhline(float(low_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
            ax.axhline(float(high_coef[0]), color="tab:blue", linestyle="--", linewidth=1.2)
        ax.step(t_step_blk, alpha_seg[s_last : s_last + last_steps], where="post")
        ax.set_ylabel("alpha")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_matrix_alpha_last_block", save_pdf=save_pdf)

    if delta_log is not None:
        delta_seg = delta_log[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and high_coef.size > idx + 1:
                ax.fill_between(
                    t_step,
                    float(low_coef[idx + 1]),
                    float(high_coef[idx + 1]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
            ax.step(t_step, delta_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(rf"$\delta_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_matrix_delta_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and high_coef.size > idx + 1:
                ax.fill_between(
                    t_step_blk,
                    float(low_coef[idx + 1]),
                    float(high_coef[idx + 1]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx + 1]), color="tab:green", linestyle="--", linewidth=1.2)
            ax.step(t_step_blk, delta_seg[s_last : s_last + last_steps, idx], where="post")
            ax.set_ylabel(rf"$\delta_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_matrix_delta_last_block", save_pdf=save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_matrix_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        disturbance_items = disturbance_plot_items(disturbance_profile, metadata.get("disturbance_labels"))
        if disturbance_items:
            fig, axs = plt.subplots(
                len(disturbance_items),
                1,
                figsize=(8.6, 3.0 + 2.0 * max(1, len(disturbance_items) - 1)),
                sharex=True,
            )
            if len(disturbance_items) == 1:
                axs = [axs]
            for idx, (_, label, series) in enumerate(disturbance_items):
                axs[idx].plot(t_step, np.asarray(series, float)[start_step : start_step + W])
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(label)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel(time_label)
            _save_fig(fig, out_dir, "fig_matrix_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_matrix",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    diag_series = []
    if bundle.get("actor_losses") is not None and len(bundle["actor_losses"]) > 0:
        diag_series.append(("actor_loss", bundle["actor_losses"]))
    if bundle.get("critic_losses") is not None and len(bundle["critic_losses"]) > 0:
        diag_series.append(("critic_loss", bundle["critic_losses"]))
    if bundle.get("alphas") is not None and len(bundle["alphas"]) > 0:
        diag_series.append(("sac_alpha", bundle["alphas"]))
    if bundle.get("alpha_losses") is not None and len(bundle["alpha_losses"]) > 0:
        diag_series.append(("sac_alpha_loss", bundle["alpha_losses"]))
    if bundle.get("critic_q1_trace") is not None and len(bundle["critic_q1_trace"]) > 0:
        diag_series.append(("critic_q1", bundle["critic_q1_trace"]))
    if bundle.get("critic_q2_trace") is not None and len(bundle["critic_q2_trace"]) > 0:
        diag_series.append(("critic_q2", bundle["critic_q2_trace"]))
    if bundle.get("critic_q_gap_trace") is not None and len(bundle["critic_q_gap_trace"]) > 0:
        diag_series.append(("critic_q_gap", bundle["critic_q_gap_trace"]))
    if bundle.get("exploration_magnitude_trace") is not None and len(bundle["exploration_magnitude_trace"]) > 0:
        diag_series.append(("exploration_mag", bundle["exploration_magnitude_trace"]))
    if bundle.get("param_noise_scale_trace") is not None and len(bundle["param_noise_scale_trace"]) > 0:
        diag_series.append(("param_noise_scale", bundle["param_noise_scale_trace"]))
    if bundle.get("action_saturation_trace") is not None and len(bundle["action_saturation_trace"]) > 0:
        diag_series.append(("action_saturation", bundle["action_saturation_trace"]))
    if bundle.get("entropy_trace") is not None and len(bundle["entropy_trace"]) > 0:
        diag_series.append(("entropy", bundle["entropy_trace"]))
    if bundle.get("mean_log_prob_trace") is not None and len(bundle["mean_log_prob_trace"]) > 0:
        diag_series.append(("mean_log_prob", bundle["mean_log_prob_trace"]))
    if diag_series:
        fig, axs = plt.subplots(len(diag_series), 1, figsize=(8.4, 3.0 + 2.2 * max(1, len(diag_series) - 1)), sharex=False)
        if len(diag_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, diag_series):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Update #")
        _save_fig(fig, out_dir, "fig_matrix_training_diagnostics", save_pdf=save_pdf)

    _plot_nstep_diagnostics(out_dir, "fig_matrix_nstep_decomposition", bundle, save_pdf)
    _plot_phase1_release_window_single_agent(bundle, out_dir, time_label, save_pdf)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    stored_bundle.update(
        {
            "alpha_log": bundle.get("alpha_log"),
            "delta_log": bundle.get("delta_log"),
            "low_coef": bundle.get("low_coef"),
            "high_coef": bundle.get("high_coef"),
            "agent_kind": bundle.get("agent_kind"),
            "run_mode": bundle.get("run_mode"),
            "test_train_dict": bundle.get("test_train_dict"),
            "disturbance_profile": bundle.get("disturbance_profile"),
            "actor_losses": bundle.get("actor_losses"),
            "critic_losses": bundle.get("critic_losses"),
            "alpha_losses": bundle.get("alpha_losses"),
            "alphas": bundle.get("alphas"),
            "mpc_path_or_dir": bundle.get("mpc_path_or_dir"),
            "estimator_mode": bundle.get("estimator_mode"),
            "prediction_model_mode": bundle.get("prediction_model_mode"),
        }
    )
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_structured_matrix_results_core(result_bundle, plot_cfg):
    out_dir = plot_matrix_multiplier_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)
    bundle = normalize_result_bundle(result_bundle)
    save_pdf = bool(plot_cfg.get("save_pdf", False))

    action_labels = list(bundle.get("action_labels") or [])
    theta_a_labels = list(bundle.get("theta_a_labels") or [])
    theta_b_labels = list(bundle.get("theta_b_labels") or [])
    low_bounds = bundle.get("structured_low_bounds")
    high_bounds = bundle.get("structured_high_bounds")
    mapped_multiplier_log = bundle.get("mapped_multiplier_log")
    A_ratio = bundle.get("A_model_delta_ratio_log")
    B_ratio = bundle.get("B_model_delta_ratio_log")
    spectral_radius = bundle.get("spectral_radius_log")
    action_saturation = bundle.get("action_saturation_fraction_log")
    near_bound = bundle.get("near_bound_fraction_log")
    episode_avg_theta_a = bundle.get("episode_avg_theta_a")
    episode_avg_theta_b = bundle.get("episode_avg_theta_b")

    if mapped_multiplier_log is not None and low_bounds is not None and high_bounds is not None and action_labels:
        mapped_multiplier_log = np.asarray(mapped_multiplier_log, float)
        low_bounds = np.asarray(low_bounds, float)
        high_bounds = np.asarray(high_bounds, float)
        t_idx = np.arange(1, mapped_multiplier_log.shape[0] + 1)
        fig, axs = plt.subplots(
            mapped_multiplier_log.shape[1],
            1,
            figsize=(9.0, 3.0 + 1.9 * max(1, mapped_multiplier_log.shape[1] - 1)),
            sharex=True,
        )
        if mapped_multiplier_log.shape[1] == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.fill_between(
                t_idx,
                float(low_bounds[idx]),
                float(high_bounds[idx]),
                color="tab:blue",
                alpha=0.12,
                step="post",
            )
            ax.axhline(float(low_bounds[idx]), color="tab:blue", linestyle="--", linewidth=1.0)
            ax.axhline(float(high_bounds[idx]), color="tab:blue", linestyle="--", linewidth=1.0)
            ax.step(t_idx, mapped_multiplier_log[:, idx], where="post")
            ax.set_ylabel(action_labels[idx] if idx < len(action_labels) else f"theta_{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Step")
        _save_fig(fig, out_dir, "fig_structured_matrix_multipliers_full", save_pdf=save_pdf)

    if A_ratio is not None or B_ratio is not None or spectral_radius is not None:
        metrics = []
        if A_ratio is not None:
            metrics.append(("A Fro ratio", np.asarray(A_ratio, float)))
        if B_ratio is not None:
            metrics.append(("B Fro ratio", np.asarray(B_ratio, float)))
        if spectral_radius is not None:
            metrics.append(("Spectral radius", np.asarray(spectral_radius, float)))
        fig, axs = plt.subplots(len(metrics), 1, figsize=(8.6, 3.0 + 2.2 * max(1, len(metrics) - 1)), sharex=True)
        if len(metrics) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, metrics):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Step")
        _save_fig(fig, out_dir, "fig_structured_matrix_model_delta_metrics", save_pdf=save_pdf)

    if action_saturation is not None or near_bound is not None:
        metrics = []
        if action_saturation is not None:
            metrics.append(("Action saturation", np.asarray(action_saturation, float)))
        if near_bound is not None:
            metrics.append(("Near-bound frac", np.asarray(near_bound, float)))
        fig, axs = plt.subplots(len(metrics), 1, figsize=(8.4, 3.0 + 2.1 * max(1, len(metrics) - 1)), sharex=True)
        if len(metrics) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, metrics):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Step")
        _save_fig(fig, out_dir, "fig_structured_matrix_action_health", save_pdf=save_pdf)

    episode_labels = np.arange(1, len(np.asarray(bundle.get("avg_rewards", []), float)) + 1)
    if episode_labels.size > 0 and (
        (episode_avg_theta_a is not None and np.asarray(episode_avg_theta_a).size > 0)
        or (episode_avg_theta_b is not None and np.asarray(episode_avg_theta_b).size > 0)
    ):
        series_specs = []
        if episode_avg_theta_a is not None and np.asarray(episode_avg_theta_a).size > 0:
            arr = np.asarray(episode_avg_theta_a, float)
            for idx in range(arr.shape[1]):
                label = theta_a_labels[idx] if idx < len(theta_a_labels) else f"theta_A_{idx + 1}"
                series_specs.append((label, arr[:, idx]))
        if episode_avg_theta_b is not None and np.asarray(episode_avg_theta_b).size > 0:
            arr = np.asarray(episode_avg_theta_b, float)
            for idx in range(arr.shape[1]):
                label = theta_b_labels[idx] if idx < len(theta_b_labels) else f"theta_B_{idx + 1}"
                series_specs.append((label, arr[:, idx]))
        fig, axs = plt.subplots(len(series_specs), 1, figsize=(8.8, 3.0 + 2.0 * max(1, len(series_specs) - 1)), sharex=True)
        if len(series_specs) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, series_specs):
            ax.plot(episode_labels[: len(series)], series, marker="o")
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Sub-episode")
        _save_fig(fig, out_dir, "fig_structured_matrix_episode_average_multipliers", save_pdf=save_pdf)

    stored_path = os.path.join(out_dir, "input_data.pkl")
    if os.path.exists(stored_path):
        with open(stored_path, "rb") as handle:
            stored_bundle = pickle.load(handle)
    else:
        stored_bundle = {}
    for key in (
        "update_family",
        "range_profile",
        "structured_action_dim",
        "structured_a_dim",
        "structured_b_dim",
        "action_labels",
        "theta_a_labels",
        "theta_b_labels",
        "structured_low_bounds",
        "structured_high_bounds",
        "structured_low_a",
        "structured_high_a",
        "structured_low_b",
        "structured_high_b",
        "block_cfg",
        "band_cfg",
        "raw_action_log",
        "mapped_multiplier_log",
        "theta_a_log",
        "theta_b_log",
        "A_model_delta_ratio_log",
        "B_model_delta_ratio_log",
        "spectral_radius_log",
        "action_saturation_fraction_log",
        "near_bound_fraction_log",
        "near_bound_relative_tolerance",
        "action_saturation_threshold",
        "episode_avg_theta_a",
        "episode_avg_theta_b",
        "episode_avg_action_saturation",
        "episode_avg_near_bound",
        "episode_avg_A_model_delta_ratio",
        "episode_avg_B_model_delta_ratio",
        "structured_update_fallback_count",
    ):
        if key in bundle:
            stored_bundle[key] = bundle[key]
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_reidentification_results_core(result_bundle, plot_cfg):
    include_base_matrix_plots = bool(plot_cfg.get("include_base_matrix_plots", False))
    if include_base_matrix_plots:
        out_dir = plot_matrix_multiplier_results_core(result_bundle=result_bundle, plot_cfg=plot_cfg)
    else:
        style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
        _set_plot_style(style_profile=style_profile)
        bundle_for_storage = normalize_result_bundle(result_bundle)
        directory = os.fspath(plot_cfg["directory"])
        prefix_name = plot_cfg.get("prefix_name", "reidentification_result")
        start_episode = int(plot_cfg.get("start_episode", 1))
        out_dir = create_output_dir(directory, prefix_name)
        stored_bundle = build_storage_bundle(bundle_for_storage, start_episode)
        stored_bundle.pop("eta_log", None)
        stored_bundle.pop("eta_raw_log", None)
        save_bundle_pickle(out_dir, stored_bundle)
    bundle = normalize_result_bundle(result_bundle)
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    debug_id_plots = bool(plot_cfg.get("debug_id_plots", False))

    eta_A_log = bundle.get("eta_A_log")
    eta_B_log = bundle.get("eta_B_log")
    eta_A_raw_log = bundle.get("eta_A_raw_log")
    eta_B_raw_log = bundle.get("eta_B_raw_log")
    theta_hat_log = bundle.get("theta_hat_log")
    theta_active_log = bundle.get("theta_active_log")
    theta_candidate_log = bundle.get("theta_candidate_log")
    theta_unclipped_log = bundle.get("theta_unclipped_log")
    theta_labels = list(bundle.get("theta_labels") or [])
    id_residual_norm_log = bundle.get("id_residual_norm_log")
    id_condition_number_log = bundle.get("id_condition_number_log")
    id_update_event_log = bundle.get("id_update_event_log")
    id_update_success_log = bundle.get("id_update_success_log")
    id_fallback_log = bundle.get("id_fallback_log")
    id_candidate_valid_log = bundle.get("id_candidate_valid_log")
    theta_lower_hit_mask_log = bundle.get("theta_lower_hit_mask_log")
    theta_upper_hit_mask_log = bundle.get("theta_upper_hit_mask_log")
    theta_clipped_fraction_log = bundle.get("theta_clipped_fraction_log")
    active_A_ratio = bundle.get("active_A_model_delta_ratio_log")
    active_B_ratio = bundle.get("active_B_model_delta_ratio_log")
    candidate_A_ratio = bundle.get("candidate_A_model_delta_ratio_log")
    candidate_B_ratio = bundle.get("candidate_B_model_delta_ratio_log")
    pred_A_ratio = bundle.get("pred_A_model_delta_ratio_log")
    pred_B_ratio = bundle.get("pred_B_model_delta_ratio_log")
    observer_A_ratio = bundle.get("observer_A_model_delta_ratio_log")
    observer_B_ratio = bundle.get("observer_B_model_delta_ratio_log")
    observer_refresh_event_log = bundle.get("observer_refresh_event_log")
    observer_refresh_success_log = bundle.get("observer_refresh_success_log")
    basis_singular_values_A = bundle.get("basis_singular_values_A")
    basis_singular_values_B = bundle.get("basis_singular_values_B")
    rewards_step = bundle.get("rewards_step")

    if not include_base_matrix_plots:
        y_line_full = bundle["y_line_full"]
        u_step_full = bundle["u_step_full"]
        nFE = bundle["nFE"]
        delta_t = bundle["delta_t"]
        time_in_sub_episodes = bundle["time_in_sub_episodes"]
        n_inputs = bundle["n_inputs"]
        n_outputs = bundle["n_outputs"]
        metadata = resolve_system_metadata(bundle=bundle, plot_cfg=plot_cfg, n_outputs=n_outputs, n_inputs=n_inputs)
        output_labels = metadata["output_labels"]
        input_labels = metadata["input_labels"]
        time_label = metadata["time_label"]
        y_sp_phys_full = ysp_scaled_dev_to_phys(
            bundle["y_sp"],
            bundle["steady_states"],
            bundle["data_min"],
            bundle["data_max"],
            n_inputs=n_inputs,
        )

        start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
        W = int(len(y_sp_phys_full[start_step:, :]))
        y_line = y_line_full[start_step:, :]
        y_sp_phys = y_sp_phys_full[start_step:, :]
        u_line = u_step_full[start_step:, :]
        t_line = np.linspace(0.0, W * delta_t, W + 1)
        t_step = t_line[:-1]
        last_steps = int(min(max(20, time_in_sub_episodes), W))
        s_last = max(0, W - last_steps)
        t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
        t_step_blk = t_line_blk[:-1]
        spans = episode_spans(bundle.get("test_train_dict"), nFE)

        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_line, y_line[:, idx], label="RL")
            ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_reidentification_outputs_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
            ax.step(t_step_blk, y_sp_phys[s_last : s_last + last_steps, idx], where="post", linestyle="--", label="Setpoint")
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_reidentification_outputs_last_block", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.3 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, u_line[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(input_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_reidentification_inputs_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.2, 3.0 + 2.3 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
            ax.set_ylabel(input_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_reidentification_inputs_last_block", save_pdf=save_pdf)

        avg_rewards = bundle.get("avg_rewards")
        if avg_rewards is not None and len(avg_rewards) > 0:
            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            x_ep, y_ep = slice_avg_rewards(avg_rewards, len(avg_rewards), start_episode)
            ax.plot(x_ep, y_ep, marker="o")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average reward")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_reidentification_avg_rewards", save_pdf=save_pdf)

        if rewards_step is not None:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            ax.plot(np.arange(1, len(rewards_step) + 1), rewards_step)
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_reidentification_step_rewards", save_pdf=save_pdf)

    if eta_A_log is not None and eta_B_log is not None and len(eta_A_log) > 0 and len(eta_B_log) > 0:
        t_idx = np.arange(1, min(len(eta_A_log), len(eta_B_log)) + 1)
        fig, axs = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
        axs[0].step(t_idx, eta_A_log[: len(t_idx)], where="post", label=r"$\eta_A$")
        if eta_A_raw_log is not None and len(eta_A_raw_log) >= len(t_idx):
            axs[0].step(t_idx, eta_A_raw_log[: len(t_idx)], where="post", linestyle="--", alpha=0.7, label=r"$\eta_{A,raw}$")
        axs[1].step(t_idx, eta_B_log[: len(t_idx)], where="post", label=r"$\eta_B$")
        if eta_B_raw_log is not None and len(eta_B_raw_log) >= len(t_idx):
            axs[1].step(t_idx, eta_B_raw_log[: len(t_idx)], where="post", linestyle="--", alpha=0.7, label=r"$\eta_{B,raw}$")
        axs[0].set_ylabel(r"$\eta_A$")
        axs[1].set_ylabel(r"$\eta_B$")
        axs[1].set_xlabel("Step")
        for ax in axs:
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="best")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_reidentification_dual_eta", save_pdf=save_pdf)

    theta_active_for_plot = theta_active_log if theta_active_log is not None else theta_hat_log
    if debug_id_plots and theta_active_for_plot is not None and theta_active_for_plot.size > 0:
        theta_active_for_plot = np.asarray(theta_active_for_plot, float)
        fig, axs = plt.subplots(
            theta_active_for_plot.shape[1],
            1,
            figsize=(8.8, 3.0 + 1.9 * max(1, theta_active_for_plot.shape[1] - 1)),
            sharex=True,
        )
        if theta_active_for_plot.shape[1] == 1:
            axs = [axs]
        step_idx = np.arange(1, theta_active_for_plot.shape[0] + 1)
        candidate_theta = None if theta_candidate_log is None else np.asarray(theta_candidate_log, float)
        unclipped_theta = None if theta_unclipped_log is None else np.asarray(theta_unclipped_log, float)
        for idx, ax in enumerate(axs):
            if candidate_theta is not None and candidate_theta.shape == theta_active_for_plot.shape:
                ax.plot(step_idx, candidate_theta[:, idx], linestyle="--", alpha=0.8, label="Candidate")
            if unclipped_theta is not None and unclipped_theta.shape == theta_active_for_plot.shape:
                ax.plot(step_idx, unclipped_theta[:, idx], linestyle=":", alpha=0.8, label="Unclipped")
            ax.plot(step_idx, theta_active_for_plot[:, idx], label="Active")
            ax.set_ylabel(theta_labels[idx] if idx < len(theta_labels) else f"theta_{idx + 1}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Step")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_reidentification_theta_history", save_pdf=save_pdf)

    diag_series = []
    if id_residual_norm_log is not None:
        diag_series.append(("ID residual", np.asarray(id_residual_norm_log, float)))
    if id_condition_number_log is not None:
        diag_series.append(("ID cond.", np.asarray(id_condition_number_log, float)))
    if theta_clipped_fraction_log is not None:
        diag_series.append(("Theta clipped frac", np.asarray(theta_clipped_fraction_log, float)))
    if debug_id_plots and diag_series:
        fig, axs = plt.subplots(len(diag_series), 1, figsize=(8.6, 3.0 + 2.2 * max(1, len(diag_series) - 1)), sharex=True)
        if len(diag_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, diag_series):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Step")
        _save_fig(fig, out_dir, "fig_reidentification_id_diagnostics", save_pdf=save_pdf)

    if active_A_ratio is not None and active_B_ratio is not None:
        fig, axs = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True)
        axs[0].plot(np.arange(1, len(active_A_ratio) + 1), active_A_ratio)
        axs[0].set_ylabel("Active A Fro ratio")
        axs[1].plot(np.arange(1, len(active_B_ratio) + 1), active_B_ratio)
        axs[1].set_ylabel("Active B Fro ratio")
        axs[1].set_xlabel("Step")
        for ax in axs:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_reidentification_active_model_delta", save_pdf=save_pdf)

    event_series = []
    if observer_refresh_event_log is not None:
        event_series.append(("Refresh event", np.asarray(observer_refresh_event_log, float)))
    if observer_refresh_success_log is not None:
        event_series.append(("Refresh success", np.asarray(observer_refresh_success_log, float)))
    if event_series:
        fig, axs = plt.subplots(len(event_series), 1, figsize=(8.6, 3.0 + 1.8 * max(1, len(event_series) - 1)), sharex=True)
        if len(event_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, event_series):
            ax.step(np.arange(1, len(series) + 1), series, where="post")
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Step")
        _save_fig(fig, out_dir, "fig_reidentification_observer_refresh", save_pdf=save_pdf)

    if debug_id_plots and (theta_lower_hit_mask_log is not None or theta_upper_hit_mask_log is not None):
        lower_mask = None if theta_lower_hit_mask_log is None else np.asarray(theta_lower_hit_mask_log, float)
        upper_mask = None if theta_upper_hit_mask_log is None else np.asarray(theta_upper_hit_mask_log, float)
        n_theta = 0
        if lower_mask is not None:
            n_theta = lower_mask.shape[1]
        elif upper_mask is not None:
            n_theta = upper_mask.shape[1]
        if n_theta > 0:
            fig, axs = plt.subplots(n_theta, 1, figsize=(8.8, 3.0 + 1.8 * max(1, n_theta - 1)), sharex=True)
            if n_theta == 1:
                axs = [axs]
            t_idx = np.arange(1, (lower_mask.shape[0] if lower_mask is not None else upper_mask.shape[0]) + 1)
            for idx, ax in enumerate(axs):
                if lower_mask is not None:
                    ax.step(t_idx, lower_mask[:, idx], where="post", label="Lower hit")
                if upper_mask is not None:
                    ax.step(t_idx, upper_mask[:, idx], where="post", linestyle="--", label="Upper hit")
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(theta_labels[idx] if idx < len(theta_labels) else f"theta_{idx + 1}")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
            axs[-1].set_xlabel("Step")
            axs[0].legend(loc="best")
            _save_fig(fig, out_dir, "fig_reidentification_theta_bound_hits", save_pdf=save_pdf)

    if debug_id_plots and (candidate_A_ratio is not None or pred_A_ratio is not None or observer_A_ratio is not None):
        metrics = []
        if active_A_ratio is not None:
            metrics.append(("Active A Fro ratio", np.asarray(active_A_ratio, float)))
        if active_B_ratio is not None:
            metrics.append(("Active B Fro ratio", np.asarray(active_B_ratio, float)))
        if candidate_A_ratio is not None:
            metrics.append(("Candidate A Fro ratio", np.asarray(candidate_A_ratio, float)))
        if candidate_B_ratio is not None:
            metrics.append(("Candidate B Fro ratio", np.asarray(candidate_B_ratio, float)))
        if pred_A_ratio is not None:
            metrics.append(("Pred A Fro ratio", np.asarray(pred_A_ratio, float)))
        if pred_B_ratio is not None:
            metrics.append(("Pred B Fro ratio", np.asarray(pred_B_ratio, float)))
        if observer_A_ratio is not None:
            metrics.append(("Observer A Fro ratio", np.asarray(observer_A_ratio, float)))
        if observer_B_ratio is not None:
            metrics.append(("Observer B Fro ratio", np.asarray(observer_B_ratio, float)))
        fig, axs = plt.subplots(len(metrics), 1, figsize=(8.8, 3.2 + 2.1 * max(1, len(metrics) - 1)), sharex=True)
        if len(metrics) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, metrics):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Step")
        _save_fig(fig, out_dir, "fig_reidentification_model_delta_debug", save_pdf=save_pdf)

    if debug_id_plots and (basis_singular_values_A is not None or basis_singular_values_B is not None):
        sv_series = []
        if basis_singular_values_A is not None and len(basis_singular_values_A) > 0:
            sv_series.append(("A singular values", np.asarray(basis_singular_values_A, float)))
        if basis_singular_values_B is not None and len(basis_singular_values_B) > 0:
            sv_series.append(("B singular values", np.asarray(basis_singular_values_B, float)))
        if sv_series:
            fig, axs = plt.subplots(len(sv_series), 1, figsize=(7.8, 3.0 + 1.6 * max(1, len(sv_series) - 1)), sharex=False)
            if len(sv_series) == 1:
                axs = [axs]
            for ax, (label, series) in zip(axs, sv_series):
                ax.plot(np.arange(1, len(series) + 1), series, marker="o")
                ax.set_ylabel(label)
                ax.set_xlabel("Mode")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_reidentification_basis_singular_values", save_pdf=save_pdf)

    stored_path = os.path.join(out_dir, "input_data.pkl")
    if os.path.exists(stored_path):
        with open(stored_path, "rb") as handle:
            stored_bundle = pickle.load(handle)
    else:
        stored_bundle = {}

    stored_bundle.pop("eta_log", None)
    stored_bundle.pop("eta_raw_log", None)

    for key in (
        "prediction_model_mode",
        "eta_A_log",
        "eta_B_log",
        "eta_A_raw_log",
        "eta_B_raw_log",
        "action_raw_log",
        "theta_labels",
        "theta_labels_A",
        "theta_labels_B",
        "alpha_labels",
        "beta_labels",
        "theta_A_indices",
        "theta_B_indices",
        "theta_low",
        "theta_high",
        "theta_low_A",
        "theta_high_A",
        "theta_low_B",
        "theta_high_B",
        "lambda_prev_vector",
        "lambda_0_vector",
        "lambda_prev_A",
        "lambda_prev_B",
        "lambda_0_A",
        "lambda_0_B",
        "theta_hat_log",
        "theta_active_log",
        "theta_candidate_log",
        "theta_unclipped_log",
        "theta_lower_hit_mask_log",
        "theta_upper_hit_mask_log",
        "theta_clipped_fraction_log",
        "id_basis_name",
        "basis_family",
        "id_residual_norm_log",
        "id_condition_number_log",
        "id_update_event_log",
        "id_update_success_log",
        "id_fallback_log",
        "id_valid_flag_log",
        "id_source_code_log",
        "id_candidate_valid_log",
        "id_A_model_delta_ratio_log",
        "id_B_model_delta_ratio_log",
        "active_A_model_delta_ratio_log",
        "active_B_model_delta_ratio_log",
        "candidate_A_model_delta_ratio_log",
        "candidate_B_model_delta_ratio_log",
        "pred_A_model_delta_ratio_log",
        "pred_B_model_delta_ratio_log",
        "observer_A_model_delta_ratio_log",
        "observer_B_model_delta_ratio_log",
        "observer_refresh_event_log",
        "observer_refresh_success_log",
        "observer_refresh_enabled",
        "observer_refresh_every_episodes",
        "observer_refresh_attempt_count",
        "observer_refresh_success_count",
        "rho_obs",
        "rank_A",
        "rank_B",
        "offline_window",
        "offline_stride",
        "lambda_A_off",
        "lambda_B_off",
        "offline_basis_source_path",
        "offline_basis_cache_path",
        "basis_window_count",
        "basis_singular_values_A",
        "basis_singular_values_B",
        "invalid_id_solve_count",
        "id_solver_failure_count",
        "id_update_success_count",
        "force_eta_constant",
        "disable_identification",
        "action_dim",
        "candidate_guard_mode",
        "observer_update_alignment",
        "normalize_blend_extras",
        "blend_extra_clip",
        "blend_residual_scale",
        "log_theta_clipping",
        "id_component_mode",
    ):
        if key in bundle:
            stored_bundle[key] = bundle[key]
    stored_bundle["debug_id_plots"] = debug_id_plots
    _plot_phase1_release_window_single_agent(bundle, out_dir, time_label, save_pdf)
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_weight_multiplier_results_core(result_bundle, plot_cfg):
    style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
    _set_plot_style(style_profile=style_profile)
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "weight_multiplier_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]
    metadata = resolve_system_metadata(bundle=bundle, plot_cfg=plot_cfg, n_outputs=n_outputs, n_inputs=n_inputs)
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]
    time_label = metadata["time_label"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_weights_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_weights_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_weights_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_weights_inputs_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_weights_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_weights_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e[{idx + 1}]")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel(time_label)
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_weights_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(rf"$\Delta u_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_weights_delta_u", save_pdf=save_pdf)

    weight_log = bundle.get("weight_log")
    low_coef = bundle.get("low_coef")
    high_coef = bundle.get("high_coef")
    if weight_log is not None:
        w_seg = weight_log[start_step : start_step + W, :]
        names = ["Q1", "Q2", "R1", "R2"]
        fig, axs = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == 4 and high_coef.size == 4:
                ax.fill_between(
                    t_step,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:purple",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
            ax.step(t_step, w_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(names[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_weights_multipliers_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == 4 and high_coef.size == 4:
                ax.fill_between(
                    t_step_blk,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:purple",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:purple", linestyle="--", linewidth=1.2)
            ax.step(t_step_blk, w_seg[s_last : s_last + last_steps, idx], where="post")
            ax.set_ylabel(names[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_weights_multipliers_last_block", save_pdf=save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_weights_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        disturbance_items = disturbance_plot_items(disturbance_profile, metadata.get("disturbance_labels"))
        if disturbance_items:
            fig, axs = plt.subplots(
                len(disturbance_items),
                1,
                figsize=(8.6, 3.0 + 2.0 * max(1, len(disturbance_items) - 1)),
                sharex=True,
            )
            if len(disturbance_items) == 1:
                axs = [axs]
            for idx, (_, label, series) in enumerate(disturbance_items):
                axs[idx].plot(t_step, np.asarray(series, float)[start_step : start_step + W])
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(label)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel(time_label)
            _save_fig(fig, out_dir, "fig_weights_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_weights",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    diag_series = []
    if bundle.get("actor_losses") is not None and len(bundle["actor_losses"]) > 0:
        diag_series.append(("actor_loss", bundle["actor_losses"]))
    if bundle.get("critic_losses") is not None and len(bundle["critic_losses"]) > 0:
        diag_series.append(("critic_loss", bundle["critic_losses"]))
    if bundle.get("alphas") is not None and len(bundle["alphas"]) > 0:
        diag_series.append(("sac_alpha", bundle["alphas"]))
    if bundle.get("alpha_losses") is not None and len(bundle["alpha_losses"]) > 0:
        diag_series.append(("sac_alpha_loss", bundle["alpha_losses"]))
    if bundle.get("critic_q1_trace") is not None and len(bundle["critic_q1_trace"]) > 0:
        diag_series.append(("critic_q1", bundle["critic_q1_trace"]))
    if bundle.get("critic_q2_trace") is not None and len(bundle["critic_q2_trace"]) > 0:
        diag_series.append(("critic_q2", bundle["critic_q2_trace"]))
    if bundle.get("critic_q_gap_trace") is not None and len(bundle["critic_q_gap_trace"]) > 0:
        diag_series.append(("critic_q_gap", bundle["critic_q_gap_trace"]))
    if bundle.get("exploration_magnitude_trace") is not None and len(bundle["exploration_magnitude_trace"]) > 0:
        diag_series.append(("exploration_mag", bundle["exploration_magnitude_trace"]))
    if bundle.get("param_noise_scale_trace") is not None and len(bundle["param_noise_scale_trace"]) > 0:
        diag_series.append(("param_noise_scale", bundle["param_noise_scale_trace"]))
    if bundle.get("action_saturation_trace") is not None and len(bundle["action_saturation_trace"]) > 0:
        diag_series.append(("action_saturation", bundle["action_saturation_trace"]))
    if bundle.get("entropy_trace") is not None and len(bundle["entropy_trace"]) > 0:
        diag_series.append(("entropy", bundle["entropy_trace"]))
    if bundle.get("mean_log_prob_trace") is not None and len(bundle["mean_log_prob_trace"]) > 0:
        diag_series.append(("mean_log_prob", bundle["mean_log_prob_trace"]))
    if diag_series:
        fig, axs = plt.subplots(len(diag_series), 1, figsize=(8.4, 3.0 + 2.2 * max(1, len(diag_series) - 1)), sharex=False)
        if len(diag_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, diag_series):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Update #")
        _save_fig(fig, out_dir, "fig_weights_training_diagnostics", save_pdf=save_pdf)

    _plot_nstep_diagnostics(out_dir, "fig_weights_nstep_decomposition", bundle, save_pdf)
    _plot_phase1_release_window_single_agent(bundle, out_dir, time_label, save_pdf)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    stored_bundle.update(
        {
            "weight_log": bundle.get("weight_log"),
            "low_coef": bundle.get("low_coef"),
            "high_coef": bundle.get("high_coef"),
            "agent_kind": bundle.get("agent_kind"),
            "run_mode": bundle.get("run_mode"),
            "test_train_dict": bundle.get("test_train_dict"),
            "disturbance_profile": bundle.get("disturbance_profile"),
            "actor_losses": bundle.get("actor_losses"),
            "critic_losses": bundle.get("critic_losses"),
            "alpha_losses": bundle.get("alpha_losses"),
            "alphas": bundle.get("alphas"),
            "mpc_path_or_dir": bundle.get("mpc_path_or_dir"),
        }
    )
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_residual_results_core(result_bundle, plot_cfg):
    style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
    _set_plot_style(style_profile=style_profile)
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "residual_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    out_dir = create_output_dir(directory, prefix_name)

    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]
    metadata = resolve_system_metadata(bundle=bundle, plot_cfg=plot_cfg, n_outputs=n_outputs, n_inputs=n_inputs)
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]
    time_label = metadata["time_label"]

    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_line[:, idx], label="RL")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle="--", label="Setpoint")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_residual_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
    if n_outputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_blk, y_line[s_last : s_last + last_steps + 1, idx], label="RL")
        ax.step(
            t_step_blk,
            y_sp_phys[s_last : s_last + last_steps, idx],
            where="post",
            linestyle="--",
            label="Setpoint",
        )
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "fig_residual_outputs_last_block", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step, u_line[:, idx], where="post")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_residual_inputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post")
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    _save_fig(fig, out_dir, "fig_residual_inputs_last_block", save_pdf=save_pdf)

    u_base = bundle.get("u_base")
    if u_base is not None:
        u_base_seg = u_base[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.2 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, u_line[:, idx], where="post", label="Applied")
            ax.step(t_step, u_base_seg[:, idx], where="post", linestyle="--", label="MPC baseline")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(input_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_inputs_overlay_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.2 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step_blk, u_line[s_last : s_last + last_steps, idx], where="post", label="Applied")
            ax.step(
                t_step_blk,
                u_base_seg[s_last : s_last + last_steps, idx],
                where="post",
                linestyle="--",
                label="MPC baseline",
            )
            ax.set_ylabel(input_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_inputs_overlay_last_block", save_pdf=save_pdf)

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_residual_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t_step, rewards_seg)
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Reward")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_residual_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        err_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.6, 3.2 + 2.3 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, err_seg[:, idx])
            shade_test_regions(axs[idx], spans, delta_t)
            axs[idx].set_ylabel(f"e[{idx + 1}]")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        err_norm = np.linalg.norm(err_seg, axis=1)
        axs[-1].plot(t_step, err_norm)
        shade_test_regions(axs[-1], spans, delta_t)
        axs[-1].set_ylabel("||e||")
        axs[-1].set_xlabel(time_label)
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_residual_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(rf"$\Delta u_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_residual_delta_u", save_pdf=save_pdf)

    residual_exec_log = bundle.get("residual_exec_log")
    residual_raw_log = bundle.get("residual_raw_log")
    low_coef = bundle.get("low_coef")
    high_coef = bundle.get("high_coef")
    if residual_exec_log is not None:
        exec_seg = residual_exec_log[start_step : start_step + W, :]
        raw_seg = None if residual_raw_log is None else residual_raw_log[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == n_inputs and high_coef.size == n_inputs:
                ax.fill_between(
                    t_step,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
            if raw_seg is not None:
                ax.step(t_step, raw_seg[:, idx], where="post", linestyle=":", label="Raw")
            ax.step(t_step, exec_seg[:, idx], where="post", label="Executed")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(rf"$u^r_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_correction_full", save_pdf=save_pdf)

        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            if low_coef is not None and high_coef is not None and low_coef.size == n_inputs and high_coef.size == n_inputs:
                ax.fill_between(
                    t_step_blk,
                    float(low_coef[idx]),
                    float(high_coef[idx]),
                    color="tab:green",
                    alpha=0.12,
                    step="post",
                )
                ax.axhline(float(low_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
                ax.axhline(float(high_coef[idx]), color="tab:green", linestyle="--", linewidth=1.2)
            if raw_seg is not None:
                ax.step(
                    t_step_blk,
                    raw_seg[s_last : s_last + last_steps, idx],
                    where="post",
                    linestyle=":",
                    label="Raw",
                )
            ax.step(
                t_step_blk,
                exec_seg[s_last : s_last + last_steps, idx],
                where="post",
                label="Executed",
            )
            ax.set_ylabel(rf"$u^r_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_correction_last_block", save_pdf=save_pdf)

    if bundle.get("rho_log") is not None:
        rho_seg = bundle["rho_log"][start_step : start_step + W]
        rho_eff_seg = None if bundle.get("rho_eff_log") is None else bundle["rho_eff_log"][start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(t_step, rho_seg, label=r"$\rho$")
        if rho_eff_seg is not None:
            ax.plot(t_step, rho_eff_seg, linestyle="--", label=r"$\rho_{\mathrm{eff}}$")
        shade_test_regions(ax, spans, delta_t)
        ax.set_ylabel("Authority")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        ax.legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_rho_trace", save_pdf=save_pdf)

    projection_keys = (
        ("deadband_active_log", "Deadband active"),
        ("projection_active_log", "Projected"),
        ("projection_due_to_deadband_log", "Deadband limited"),
        ("projection_due_to_authority_log", "Authority limited"),
        ("projection_due_to_headroom_log", "Headroom limited"),
    )
    if any(bundle.get(key) is not None for key, _ in projection_keys):
        labels = []
        values = []
        for key, label in projection_keys:
            series = bundle.get(key)
            if series is None:
                continue
            labels.append(label)
            values.append(float(np.mean(np.asarray(series[start_step : start_step + W], float))))
        if values:
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            ax.bar(np.arange(len(values)), values, color=["#2D6A4F", "#355070", "#8E5572", "#6D597A", "#B56576"][: len(values)])
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels(labels, rotation=10)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Fraction of steps")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_residual_projection_fractions", save_pdf=save_pdf)

    yhat = bundle.get("yhat")
    if yhat is not None:
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.6, 3.0 + 2.5 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            meas_scaled = apply_min_max(
                y_line_full[1:, idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            ) - apply_min_max(
                bundle["steady_states"]["y_ss"][idx],
                bundle["data_min"][n_inputs + idx],
                bundle["data_max"][n_inputs + idx],
            )
            ax.plot(t_step, np.asarray(yhat[idx, start_step : start_step + W], float), label="Observer")
            ax.plot(t_step, meas_scaled[start_step : start_step + W], linestyle="--", label="Measurement")
            shade_test_regions(ax, spans, delta_t)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_residual_observer_overlay", save_pdf=save_pdf)

    disturbance_profile = bundle.get("disturbance_profile")
    if disturbance_profile:
        disturbance_items = disturbance_plot_items(disturbance_profile, metadata.get("disturbance_labels"))
        if disturbance_items:
            fig, axs = plt.subplots(
                len(disturbance_items),
                1,
                figsize=(8.6, 3.0 + 2.0 * max(1, len(disturbance_items) - 1)),
                sharex=True,
            )
            if len(disturbance_items) == 1:
                axs = [axs]
            for idx, (_, label, series) in enumerate(disturbance_items):
                axs[idx].plot(t_step, np.asarray(series, float)[start_step : start_step + W])
                shade_test_regions(axs[idx], spans, delta_t)
                axs[idx].set_ylabel(label)
                axs[idx].spines["top"].set_visible(False)
                axs[idx].spines["right"].set_visible(False)
                _make_axes_bold(axs[idx])
            axs[-1].set_xlabel(time_label)
            _save_fig(fig, out_dir, "fig_residual_disturbance_profile", save_pdf=save_pdf)

    _plot_mismatch_diagnostics(
        bundle=bundle,
        out_dir=out_dir,
        prefix="fig_residual",
        t_step=t_step,
        t_step_blk=t_step_blk,
        start_step=start_step,
        W=W,
        s_last=s_last,
        last_steps=last_steps,
        spans=spans,
        delta_t=delta_t,
        save_pdf=save_pdf,
    )

    diag_series = []
    if bundle.get("actor_losses") is not None and len(bundle["actor_losses"]) > 0:
        diag_series.append(("actor_loss", bundle["actor_losses"]))
    if bundle.get("critic_losses") is not None and len(bundle["critic_losses"]) > 0:
        diag_series.append(("critic_loss", bundle["critic_losses"]))
    if bundle.get("alphas") is not None and len(bundle["alphas"]) > 0:
        diag_series.append(("sac_alpha", bundle["alphas"]))
    if bundle.get("alpha_losses") is not None and len(bundle["alpha_losses"]) > 0:
        diag_series.append(("sac_alpha_loss", bundle["alpha_losses"]))
    if bundle.get("critic_q1_trace") is not None and len(bundle["critic_q1_trace"]) > 0:
        diag_series.append(("critic_q1", bundle["critic_q1_trace"]))
    if bundle.get("critic_q2_trace") is not None and len(bundle["critic_q2_trace"]) > 0:
        diag_series.append(("critic_q2", bundle["critic_q2_trace"]))
    if bundle.get("critic_q_gap_trace") is not None and len(bundle["critic_q_gap_trace"]) > 0:
        diag_series.append(("critic_q_gap", bundle["critic_q_gap_trace"]))
    if bundle.get("exploration_magnitude_trace") is not None and len(bundle["exploration_magnitude_trace"]) > 0:
        diag_series.append(("exploration_mag", bundle["exploration_magnitude_trace"]))
    if bundle.get("param_noise_scale_trace") is not None and len(bundle["param_noise_scale_trace"]) > 0:
        diag_series.append(("param_noise_scale", bundle["param_noise_scale_trace"]))
    if bundle.get("action_saturation_trace") is not None and len(bundle["action_saturation_trace"]) > 0:
        diag_series.append(("action_saturation", bundle["action_saturation_trace"]))
    if bundle.get("entropy_trace") is not None and len(bundle["entropy_trace"]) > 0:
        diag_series.append(("entropy", bundle["entropy_trace"]))
    if bundle.get("mean_log_prob_trace") is not None and len(bundle["mean_log_prob_trace"]) > 0:
        diag_series.append(("mean_log_prob", bundle["mean_log_prob_trace"]))
    if diag_series:
        fig, axs = plt.subplots(
            len(diag_series),
            1,
            figsize=(8.4, 3.0 + 2.2 * max(1, len(diag_series) - 1)),
            sharex=False,
        )
        if len(diag_series) == 1:
            axs = [axs]
        for ax, (label, series) in zip(axs, diag_series):
            ax.plot(np.arange(1, len(series) + 1), series)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Update #")
        _save_fig(fig, out_dir, "fig_residual_training_diagnostics", save_pdf=save_pdf)

    _plot_nstep_diagnostics(out_dir, "fig_residual_nstep_decomposition", bundle, save_pdf)
    _plot_phase1_release_window_single_agent(bundle, out_dir, time_label, save_pdf)

    stored_bundle = build_storage_bundle(bundle, start_episode)
    stored_bundle.update(
        {
            "residual_exec_log": bundle.get("residual_exec_log"),
            "residual_raw_log": bundle.get("residual_raw_log"),
            "u_base": bundle.get("u_base"),
            "low_coef": bundle.get("low_coef"),
            "high_coef": bundle.get("high_coef"),
            "agent_kind": bundle.get("agent_kind"),
            "run_mode": bundle.get("run_mode"),
            "test_train_dict": bundle.get("test_train_dict"),
            "disturbance_profile": bundle.get("disturbance_profile"),
            "actor_losses": bundle.get("actor_losses"),
            "critic_losses": bundle.get("critic_losses"),
            "alpha_losses": bundle.get("alpha_losses"),
            "alphas": bundle.get("alphas"),
            "mpc_path_or_dir": bundle.get("mpc_path_or_dir"),
        }
    )
    save_bundle_pickle(out_dir, stored_bundle)
    return out_dir


def plot_combined_results_core(result_bundle, plot_cfg):
    style_profile = str(plot_cfg.get("style_profile", "hybrid")).lower()
    _set_plot_style(style_profile=style_profile)
    bundle = normalize_result_bundle(result_bundle)

    directory = os.fspath(plot_cfg["directory"])
    prefix_name = plot_cfg.get("prefix_name", "combined_result")
    start_episode = int(plot_cfg.get("start_episode", 1))
    compare_start_episode = int(plot_cfg.get("compare_start_episode", start_episode))
    save_pdf = bool(plot_cfg.get("save_pdf", False))
    reward_fn = plot_cfg.get("reward_fn")
    include_baseline_compare = bool(plot_cfg.get("include_baseline_compare", True))
    compare_mode = str(plot_cfg.get("compare_mode", bundle.get("run_mode", "nominal"))).lower()
    compare_prefix = plot_cfg.get("compare_prefix", "baseline_compare")
    out_dir = create_output_dir(directory, prefix_name)

    active_agents = dict(bundle.get("active_agents", {}))
    y_line_full = bundle["y_line_full"]
    u_step_full = bundle["u_step_full"]
    nFE = bundle["nFE"]
    delta_t = bundle["delta_t"]
    time_in_sub_episodes = bundle["time_in_sub_episodes"]
    n_inputs = bundle["n_inputs"]
    n_outputs = bundle["n_outputs"]
    y_sp_phys_full = ysp_scaled_dev_to_phys(
        bundle["y_sp"],
        bundle["steady_states"],
        bundle["data_min"],
        bundle["data_max"],
        n_inputs=n_inputs,
    )

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE - 1)))
    W = int(len(y_sp_phys_full[start_step:, :]))
    y_line = y_line_full[start_step:, :]
    y_sp_phys = y_sp_phys_full[start_step:, :]
    u_line = u_step_full[start_step:, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    last_steps = int(min(max(20, time_in_sub_episodes), W))
    s_last = max(0, W - last_steps)
    t_line_blk = np.linspace(0.0, last_steps * delta_t, last_steps + 1)
    t_step_blk = t_line_blk[:-1]
    spans = episode_spans(bundle.get("test_train_dict"), nFE)
    debug_mode = style_profile == "debug"
    metadata = resolve_system_metadata(bundle=bundle, plot_cfg=plot_cfg, n_outputs=n_outputs, n_inputs=n_inputs)
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]
    time_label = metadata["time_label"]

    def shade_segment(ax, segment_start, segment_len):
        for span_start, span_end, is_test in spans:
            if not is_test:
                continue
            clipped_start = max(segment_start, span_start)
            clipped_end = min(segment_start + segment_len, span_end)
            if clipped_end > clipped_start:
                ax.axvspan(
                    (clipped_start - segment_start) * delta_t,
                    (clipped_end - segment_start) * delta_t,
                    color="orange",
                    alpha=0.12,
                    linewidth=0,
                )

    def plot_outputs(prefix, y_seg, ysp_seg, t_line_local, t_step_local, segment_start, segment_len):
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.4, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_line_local, y_seg[:, idx], label="Combined RL")
            ax.step(t_step_local, ysp_seg[:, idx], where="post", linestyle="--", label="Setpoint")
            shade_segment(ax, segment_start, segment_len)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, prefix, save_pdf=save_pdf)

    def plot_inputs(prefix, u_seg, t_step_local, segment_start, segment_len):
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step_local, u_seg[:, idx], where="post")
            shade_segment(ax, segment_start, segment_len)
            ax.set_ylabel(input_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, prefix, save_pdf=save_pdf)

    plot_outputs("fig_combined_outputs_full", y_line, y_sp_phys, t_line, t_step, start_step, W)
    plot_outputs(
        "fig_combined_outputs_last_block",
        y_line[s_last : s_last + last_steps + 1, :],
        y_sp_phys[s_last : s_last + last_steps, :],
        t_line_blk,
        t_step_blk,
        start_step + s_last,
        last_steps,
    )
    plot_inputs("fig_combined_inputs_full", u_line, t_step, start_step, W)
    plot_inputs(
        "fig_combined_inputs_last_block",
        u_line[s_last : s_last + last_steps, :],
        t_step_blk,
        start_step + s_last,
        last_steps,
    )

    n_ep_total = int(nFE // time_in_sub_episodes) if time_in_sub_episodes > 0 else 0
    x_ep, y_ep = slice_avg_rewards(bundle["avg_rewards"], n_ep_total, start_episode)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    if len(y_ep) > 0:
        ax.plot(x_ep, y_ep, "o-")
    ax.set_ylabel("Average Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8, integer=True))
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "fig_combined_avg_rewards", save_pdf=save_pdf)

    rewards_step = bundle.get("rewards_step")
    if rewards_step is not None:
        rewards_seg = rewards_step[start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.plot(t_step, rewards_seg)
        shade_segment(ax, start_step, W)
        ax.set_ylabel("Reward")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        _save_fig(fig, out_dir, "fig_combined_step_rewards", save_pdf=save_pdf)

    delta_y_storage = bundle.get("delta_y_storage")
    if delta_y_storage is not None:
        dy_seg = delta_y_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_outputs + 1, 1, figsize=(8.4, 4.2 + 2.1 * n_outputs), sharex=True)
        for idx in range(n_outputs):
            axs[idx].plot(t_step, dy_seg[:, idx])
            shade_segment(axs[idx], start_step, W)
            axs[idx].set_ylabel(f"e{idx + 1}")
            axs[idx].spines["top"].set_visible(False)
            axs[idx].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx])
        axs[-1].plot(t_step, np.linalg.norm(dy_seg, axis=1))
        shade_segment(axs[-1], start_step, W)
        axs[-1].set_ylabel(r"$||e||_2$")
        axs[-1].set_xlabel(time_label)
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        _make_axes_bold(axs[-1])
        _save_fig(fig, out_dir, "fig_combined_tracking_error", save_pdf=save_pdf)

    delta_u_storage = bundle.get("delta_u_storage")
    if delta_u_storage is not None:
        du_seg = delta_u_storage[start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.step(t_step, du_seg[:, idx], where="post")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(rf"$\Delta u_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_combined_delta_u", save_pdf=save_pdf)

    if bundle.get("yhat") is not None:
        yhat_seg = bundle["yhat"][:, start_step : start_step + W].T
        y_meas_seg = y_line[1 : 1 + W, :]
        fig, axs = plt.subplots(n_outputs, 1, figsize=(8.4, 3.0 + 2.4 * max(1, n_outputs - 1)), sharex=True)
        if n_outputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_step, y_meas_seg[:, idx], label="Measured")
            ax.plot(t_step, yhat_seg[:, idx], linestyle="--", label="Observer")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(output_labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel("Time (h)")
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_combined_observer_overlay", save_pdf=save_pdf)

    if bundle.get("disturbance_profile") is not None:
        disturbance_items = disturbance_plot_items(bundle["disturbance_profile"], metadata.get("disturbance_labels"))
        if disturbance_items:
            fig, axs = plt.subplots(
                len(disturbance_items),
                1,
                figsize=(8.3, 3.2 + 2.0 * max(1, len(disturbance_items) - 1)),
                sharex=True,
            )
            if len(disturbance_items) == 1:
                axs = [axs]
            for ax, (_, label, series) in zip(axs, disturbance_items):
                ax.plot(t_step, np.asarray(series, float)[start_step : start_step + W])
                shade_segment(ax, start_step, W)
                ax.set_ylabel(label)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
            axs[-1].set_xlabel(time_label)
            _save_fig(fig, out_dir, "fig_combined_disturbance_profile", save_pdf=save_pdf)

    if active_agents.get("horizon") and bundle.get("horizon_trace") is not None:
        horizon_trace = np.asarray(bundle["horizon_trace"], float)
        ht_seg = horizon_trace[start_step : start_step + W, :]
        fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
        axs[0].step(t_step, ht_seg[:, 0], where="post")
        shade_segment(axs[0], start_step, W)
        axs[0].set_ylabel("Hp")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])
        axs[1].step(t_step, ht_seg[:, 1], where="post")
        shade_segment(axs[1], start_step, W)
        axs[1].set_ylabel("Hc")
        axs[1].set_xlabel(time_label)
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        _make_axes_bold(axs[1])
        _save_fig(fig, out_dir, "fig_combined_horizons", save_pdf=save_pdf)

        if bundle.get("horizon_action_trace") is not None:
            a_seg = bundle["horizon_action_trace"][start_step : start_step + W]
            fig, ax = plt.subplots(figsize=(8.2, 4.7))
            ax.step(t_step, a_seg, where="post")
            shade_segment(ax, start_step, W)
            ax.set_ylabel("Action Index")
            ax.set_xlabel(time_label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_combined_horizon_action_trace", save_pdf=save_pdf)

        if debug_mode and bundle.get("horizon_action_trace") is not None:
            counts = collections.Counter(bundle["horizon_action_trace"][start_step : start_step + W].tolist())
            labels = [str(k) for k, _ in sorted(counts.items())]
            values = [v for _, v in sorted(counts.items())]
            fig, ax = plt.subplots(figsize=(7.8, 4.5))
            ax.bar(np.arange(len(values)), values)
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Count")
            ax.set_xlabel("Horizon Action")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_combined_horizon_action_hist", save_pdf=save_pdf)

    if active_agents.get("matrix") and bundle.get("matrix_alpha_log") is not None:
        alpha_seg = bundle["matrix_alpha_log"][start_step : start_step + W]
        delta_seg = bundle["matrix_delta_log"][start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs + 1, 1, figsize=(8.4, 4.0 + 2.0 * n_inputs), sharex=True)
        axs[0].plot(t_step, alpha_seg)
        if bundle.get("matrix_low_coef") is not None:
            axs[0].axhline(bundle["matrix_low_coef"][0], linestyle=":", color="#7D8597")
            axs[0].axhline(bundle["matrix_high_coef"][0], linestyle=":", color="#7D8597")
        shade_segment(axs[0], start_step, W)
        axs[0].set_ylabel(r"$\alpha$")
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        _make_axes_bold(axs[0])
        for idx in range(n_inputs):
            axs[idx + 1].plot(t_step, delta_seg[:, idx])
            if bundle.get("matrix_low_coef") is not None:
                axs[idx + 1].axhline(bundle["matrix_low_coef"][idx + 1], linestyle=":", color="#7D8597")
                axs[idx + 1].axhline(bundle["matrix_high_coef"][idx + 1], linestyle=":", color="#7D8597")
            shade_segment(axs[idx + 1], start_step, W)
            axs[idx + 1].set_ylabel(rf"$\delta_{{{idx + 1}}}$")
            axs[idx + 1].spines["top"].set_visible(False)
            axs[idx + 1].spines["right"].set_visible(False)
            _make_axes_bold(axs[idx + 1])
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_combined_matrix_multipliers", save_pdf=save_pdf)

    if active_agents.get("weights") and bundle.get("weight_log") is not None:
        weight_seg = bundle["weight_log"][start_step : start_step + W, :]
        labels = ["Q1", "Q2", "R1", "R2"]
        fig, axs = plt.subplots(4, 1, figsize=(8.4, 9.0), sharex=True)
        for idx, ax in enumerate(axs):
            ax.plot(t_step, weight_seg[:, idx])
            if bundle.get("weight_low_coef") is not None:
                ax.axhline(bundle["weight_low_coef"][idx], linestyle=":", color="#7D8597")
                ax.axhline(bundle["weight_high_coef"][idx], linestyle=":", color="#7D8597")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(labels[idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_combined_weight_multipliers", save_pdf=save_pdf)

    if active_agents.get("residual") and bundle.get("residual_exec_log") is not None:
        raw_seg = bundle["residual_raw_log"][start_step : start_step + W, :]
        exec_seg = bundle["residual_exec_log"][start_step : start_step + W, :]
        fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
        if n_inputs == 1:
            axs = [axs]
        for idx, ax in enumerate(axs):
            ax.plot(t_step, raw_seg[:, idx], label="Raw")
            ax.plot(t_step, exec_seg[:, idx], linestyle="--", label="Executed")
            if bundle.get("residual_low_coef") is not None:
                ax.axhline(bundle["residual_low_coef"][idx], linestyle=":", color="#7D8597")
                ax.axhline(bundle["residual_high_coef"][idx], linestyle=":", color="#7D8597")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(rf"$u^r_{{{idx + 1}}}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        axs[0].legend(loc="best")
        _save_fig(fig, out_dir, "fig_combined_residual_traces", save_pdf=save_pdf)

        if bundle.get("u_base") is not None:
            u_base_seg = bundle["u_base"][start_step : start_step + W, :]
            fig, axs = plt.subplots(n_inputs, 1, figsize=(8.4, 3.0 + 2.1 * max(1, n_inputs - 1)), sharex=True)
            if n_inputs == 1:
                axs = [axs]
            for idx, ax in enumerate(axs):
                ax.step(t_step, u_line[:, idx], where="post", label="Applied")
                ax.step(t_step, u_base_seg[:, idx], where="post", linestyle="--", label="MPC baseline")
                shade_segment(ax, start_step, W)
                ax.set_ylabel(input_labels[idx])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
            axs[-1].set_xlabel(time_label)
            axs[0].legend(loc="best")
            _save_fig(fig, out_dir, "fig_combined_residual_input_overlay", save_pdf=save_pdf)

    decision_series = []
    decision_labels = []
    for name, key in (
        ("Horizon", "horizon_decision_log"),
        ("Matrix", "matrix_decision_log"),
        ("Weights", "weight_decision_log"),
        ("Residual", "residual_decision_log"),
    ):
        values = bundle.get(key)
        enabled = active_agents.get(name.lower(), False) if name != "Weights" else active_agents.get("weights", False)
        if values is not None and enabled:
            decision_series.append(np.asarray(values, int)[start_step : start_step + W])
            decision_labels.append(name)
    if decision_series:
        fig, axs = plt.subplots(len(decision_series), 1, figsize=(8.2, 2.2 + 1.5 * len(decision_series)), sharex=True)
        if len(decision_series) == 1:
            axs = [axs]
        for ax, label, series in zip(axs, decision_labels, decision_series):
            ax.step(t_step, series, where="post")
            shade_segment(ax, start_step, W)
            ax.set_ylabel(label)
            ax.set_ylim(-0.05, 1.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
        axs[-1].set_xlabel(time_label)
        _save_fig(fig, out_dir, "fig_combined_decision_timeline", save_pdf=save_pdf)

    if debug_mode and bundle.get("rho_log") is not None:
        rho_seg = bundle["rho_log"][start_step : start_step + W]
        rho_eff_seg = None if bundle.get("rho_eff_log") is None else bundle["rho_eff_log"][start_step : start_step + W]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.plot(t_step, rho_seg, label=r"$\rho$")
        if rho_eff_seg is not None:
            ax.plot(t_step, rho_eff_seg, linestyle="--", label=r"$\rho_{eff}$")
        shade_segment(ax, start_step, W)
        ax.set_ylabel(r"$\rho$")
        ax.set_xlabel(time_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
        if rho_eff_seg is not None:
            ax.legend(loc="best")
        _save_fig(fig, out_dir, "fig_combined_rho_trace", save_pdf=save_pdf)

    if debug_mode and any(bundle.get(key) is not None for key in ("deadband_active_log", "projection_active_log", "projection_due_to_deadband_log", "projection_due_to_authority_log", "projection_due_to_headroom_log")):
        labels = []
        values = []
        colors = []
        for key, label, color in (
            ("deadband_active_log", "Deadband active", "#2D6A4F"),
            ("projection_active_log", "Projected", "#355070"),
            ("projection_due_to_deadband_log", "Deadband limited", "#8E5572"),
            ("projection_due_to_authority_log", "Authority limited", "#6D597A"),
            ("projection_due_to_headroom_log", "Headroom limited", "#B56576"),
        ):
            series = bundle.get(key)
            if series is None:
                continue
            labels.append(label)
            values.append(float(np.mean(np.asarray(series[start_step : start_step + W], float))))
            colors.append(color)
        if values:
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.bar(np.arange(len(values)), values, color=colors)
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels(labels, rotation=10)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Fraction of steps")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _make_axes_bold(ax)
            _save_fig(fig, out_dir, "fig_combined_residual_projection_fractions", save_pdf=save_pdf)

    if debug_mode:
        def plot_named_mismatch(agent_key, title_key):
            innov = bundle.get(f"{agent_key}_innovation_log")
            terr = bundle.get(f"{agent_key}_tracking_error_log")
            if innov is None and terr is None:
                return
            series_items = []
            if innov is not None:
                series_items.append((innov[start_step : start_step + W, :], f"{title_key} innovation", f"{agent_key}_innovation"))
            if terr is not None:
                series_items.append((terr[start_step : start_step + W, :], f"{title_key} tracking", f"{agent_key}_tracking"))
            for series, label_stem, fname in series_items:
                fig, axs = plt.subplots(n_outputs, 1, figsize=(8.4, 3.0 + 2.2 * max(1, n_outputs - 1)), sharex=True)
                if n_outputs == 1:
                    axs = [axs]
                for idx, ax in enumerate(axs):
                    ax.plot(t_step, series[:, idx])
                    shade_segment(ax, start_step, W)
                    ax.set_ylabel(f"{label_stem} {idx + 1}")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    _make_axes_bold(ax)
                axs[-1].set_xlabel(time_label)
                _save_fig(fig, out_dir, f"fig_combined_{fname}", save_pdf=save_pdf)

        for key in ("horizon", "matrix", "weight", "residual"):
            plot_named_mismatch(key, key.capitalize())

        def plot_losses(prefix, label):
            plots = [
                (bundle.get(f"{prefix}_actor_losses"), f"{label} actor loss", f"fig_combined_{prefix}_actor_loss"),
                (bundle.get(f"{prefix}_critic_losses"), f"{label} critic loss", f"fig_combined_{prefix}_critic_loss"),
                (bundle.get(f"{prefix}_alpha_losses"), f"{label} alpha loss", f"fig_combined_{prefix}_alpha_loss"),
                (bundle.get(f"{prefix}_alphas"), f"{label} alpha", f"fig_combined_{prefix}_alpha_trace"),
                (bundle.get(f"{prefix}_critic_q1_trace"), f"{label} critic q1", f"fig_combined_{prefix}_critic_q1"),
                (bundle.get(f"{prefix}_critic_q2_trace"), f"{label} critic q2", f"fig_combined_{prefix}_critic_q2"),
                (bundle.get(f"{prefix}_critic_q_gap_trace"), f"{label} critic q gap", f"fig_combined_{prefix}_critic_q_gap"),
                (bundle.get(f"{prefix}_exploration_magnitude_trace"), f"{label} exploration", f"fig_combined_{prefix}_exploration"),
                (bundle.get(f"{prefix}_param_noise_scale_trace"), f"{label} param-noise scale", f"fig_combined_{prefix}_param_noise"),
                (bundle.get(f"{prefix}_action_saturation_trace"), f"{label} action saturation", f"fig_combined_{prefix}_action_saturation"),
                (bundle.get(f"{prefix}_entropy_trace"), f"{label} entropy", f"fig_combined_{prefix}_entropy"),
                (bundle.get(f"{prefix}_mean_log_prob_trace"), f"{label} mean log-prob", f"fig_combined_{prefix}_mean_log_prob"),
            ]
            for values, ylabel, fname in plots:
                if values is None or len(values) == 0:
                    continue
                fig, ax = plt.subplots(figsize=(7.8, 4.6))
                ax.plot(np.arange(1, len(values) + 1), values)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Training step")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                _make_axes_bold(ax)
                _save_fig(fig, out_dir, fname, save_pdf=save_pdf)

        for prefix, label in (("matrix", "Matrix"), ("weight", "Weights"), ("residual", "Residual")):
            plot_losses(prefix, label)

        plot_losses("horizon", "Horizon")

    _plot_phase1_release_window_combined(bundle, out_dir, time_label, save_pdf)
    stored_bundle = build_storage_bundle(bundle, start_episode)
    save_bundle_pickle(out_dir, stored_bundle)

    mpc_path_or_dir = plot_cfg.get("mpc_path_or_dir", bundle.get("mpc_path_or_dir"))
    if include_baseline_compare and reward_fn is not None and mpc_path_or_dir is not None:
        compare_directory = os.fspath(plot_cfg.get("compare_directory", directory))
        compare_mpc_rl_from_dirs_core(
            rl_dir=out_dir,
            mpc_path_or_dir=mpc_path_or_dir,
            reward_fn=reward_fn,
            directory=compare_directory,
            prefix_name=compare_prefix,
            compare_mode=compare_mode,
            start_episode=compare_start_episode,
            n_inputs=n_inputs,
            save_pdf=save_pdf,
            style_profile=style_profile,
        )

    return out_dir


def compare_mpc_rl_from_dirs_core(
    rl_dir,
    mpc_path_or_dir,
    reward_fn,
    directory,
    prefix_name,
    compare_mode="nominal",
    start_episode=1,
    n_inputs=2,
    save_pdf=False,
    style_profile="hybrid",
):
    _set_plot_style(style_profile=style_profile)
    rl_bundle = normalize_result_bundle(load_pickle(rl_dir))
    mpc_data = load_pickle(mpc_path_or_dir)
    mpc_bundle = normalize_external_bundle(mpc_data, rl_bundle)

    out_dir = create_output_dir(os.fspath(directory), prefix_name)

    rl_y = rl_bundle["y_line_full"]
    rl_u = rl_bundle["u_step_full"]
    mpc_y = mpc_bundle["y_line_full"]
    mpc_u = mpc_bundle["u_step_full"]
    metadata = resolve_system_metadata(bundle=rl_bundle, plot_cfg={}, n_outputs=rl_bundle["n_outputs"], n_inputs=rl_bundle["n_inputs"])
    output_labels = metadata["output_labels"]
    input_labels = metadata["input_labels"]
    time_label = metadata["time_label"]

    delta_t = rl_bundle["delta_t"]
    time_in_sub_episodes = rl_bundle["time_in_sub_episodes"]
    y_sp_phys_full = ysp_scaled_dev_to_phys(
        rl_bundle["y_sp"],
        rl_bundle["steady_states"],
        rl_bundle["data_min"],
        rl_bundle["data_max"],
        rl_u.shape[1],
    )
    nFE_sp = int(len(y_sp_phys_full))
    steps_rl = int(max(1, rl_y.shape[0] - 1))
    steps_mpc = int(max(1, mpc_y.shape[0] - 1))

    start_step = int(min(max(0, (start_episode - 1) * time_in_sub_episodes), max(0, nFE_sp - 1)))
    max_start = int(max(0, min(nFE_sp, steps_rl, steps_mpc) - 1))
    start_step = int(min(start_step, max_start))

    W = int(min(nFE_sp - start_step, max(1, steps_rl - start_step), max(1, steps_mpc - start_step)))
    W = int(max(1, W))

    rl_y_seg = rl_y[start_step : start_step + W + 1, :]
    mpc_y_seg = mpc_y[start_step : start_step + W + 1, :]
    rl_u_seg = rl_u[start_step : start_step + W, :]
    mpc_u_seg = mpc_u[start_step : start_step + W, :]
    y_sp_phys = y_sp_phys_full[start_step : start_step + W, :]

    t_line = np.linspace(0.0, W * delta_t, W + 1)
    t_step = t_line[:-1]
    tail = int(min(max(20, time_in_sub_episodes), W))
    tail_start = max(0, W - tail)
    t_line_blk = np.linspace(0.0, tail * delta_t, tail + 1)
    t_step_blk = t_line_blk[:-1]

    # The "last episode" comparison must always mean:
    # final episode of the RL run vs final episode of the MPC run,
    # not the tail of the currently selected overlap window.
    last_episode_steps = int(min(max(1, time_in_sub_episodes), steps_rl, steps_mpc, nFE_sp))
    last_episode_steps = int(max(1, last_episode_steps))
    t_line_last = np.linspace(0.0, last_episode_steps * delta_t, last_episode_steps + 1)
    t_step_last = t_line_last[:-1]
    rl_y_last = rl_y[-(last_episode_steps + 1) :, :]
    mpc_y_last = mpc_y[-(last_episode_steps + 1) :, :]
    rl_u_last = rl_u[-last_episode_steps:, :]
    mpc_u_last = mpc_u[-last_episode_steps:, :]
    sp_last = y_sp_phys_full[-last_episode_steps:, :]

    fig, axs = plt.subplots(rl_y_seg.shape[1], 1, figsize=(8.6, 3.0 + 2.5 * max(1, rl_y_seg.shape[1] - 1)), sharex=True)
    if rl_y_seg.shape[1] == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, rl_y_seg[:, idx], label="RL")
        ax.plot(t_line, mpc_y_seg[:, idx], linestyle="--", label="MPC")
        ax.step(t_step, y_sp_phys[:, idx], where="post", linestyle=":", label="Setpoint")
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_outputs_full", save_pdf=save_pdf)

    fig, axs = plt.subplots(rl_y_seg.shape[1], 1, figsize=(8.6, 3.0 + 2.5 * max(1, rl_y_seg.shape[1] - 1)), sharex=True)
    if rl_y_seg.shape[1] == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line_last, rl_y_last[:, idx], label="RL")
        ax.plot(t_line_last, mpc_y_last[:, idx], linestyle="--", label="MPC")
        ax.step(t_step_last, sp_last[:, idx], where="post", linestyle=":", label="Setpoint")
        ax.set_ylabel(output_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_outputs_last_episode", save_pdf=save_pdf)

    fig, axs = plt.subplots(n_inputs, 1, figsize=(8.6, 3.0 + 2.2 * max(1, n_inputs - 1)), sharex=True)
    if n_inputs == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.step(t_step_last, rl_u_last[:, idx], where="post", label="RL")
        ax.step(t_step_last, mpc_u_last[:, idx], where="post", linestyle="--", label="MPC")
        ax.set_ylabel(input_labels[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _make_axes_bold(ax)
    axs[-1].set_xlabel(time_label)
    axs[0].legend(loc="best")
    _save_fig(fig, out_dir, "compare_inputs_last_episode", save_pdf=save_pdf)

    rl_rewards, _, _ = recompute_step_rewards(
        rl_bundle["y_line_full"],
        rl_bundle["u_step_full"],
        rl_bundle["y_sp"],
        rl_bundle["steady_states"],
        rl_bundle["data_min"],
        rl_bundle["data_max"],
        reward_fn,
    )
    mpc_rewards, _, _ = recompute_step_rewards(
        mpc_bundle["y_line_full"],
        mpc_bundle["u_step_full"],
        rl_bundle["y_sp"],
        rl_bundle["steady_states"],
        rl_bundle["data_min"],
        rl_bundle["data_max"],
        reward_fn,
    )

    n_ep_rl = len(rl_rewards) // time_in_sub_episodes
    n_ep_mpc = len(mpc_rewards) // time_in_sub_episodes
    avg_rl = rl_rewards[: n_ep_rl * time_in_sub_episodes].reshape(n_ep_rl, time_in_sub_episodes).mean(axis=1) if n_ep_rl else np.asarray([])
    avg_mpc = mpc_rewards[: n_ep_mpc * time_in_sub_episodes].reshape(n_ep_mpc, time_in_sub_episodes).mean(axis=1) if n_ep_mpc else np.asarray([])

    x_rl, y_rl = slice_avg_rewards(avg_rl, n_ep_rl, start_episode)
    x_mpc, y_mpc = slice_avg_rewards(avg_mpc, n_ep_mpc, start_episode)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if len(y_rl) > 0:
        ax.plot(x_rl, y_rl, "o-", label="RL")
    if compare_mode == "nominal" and len(y_mpc) > 0 and len(x_rl) > 0:
        ax.plot(x_rl, np.full(x_rl.shape, float(y_mpc[-1])), "s--", label="MPC")
    elif len(y_mpc) > 0:
        ax.plot(x_mpc, y_mpc, "s--", label="MPC")
    ax.set_ylabel("Avg. Reward")
    ax.set_xlabel("Episode #")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")
    _make_axes_bold(ax)
    _save_fig(fig, out_dir, "compare_rewards", save_pdf=save_pdf)

    save_bundle_pickle(
        out_dir,
        {
            "rl_dir": rl_dir,
            "mpc_path_or_dir": mpc_path_or_dir,
            "compare_mode": compare_mode,
            "avg_rewards_rl": avg_rl,
            "avg_rewards_mpc": avg_mpc,
        },
    )
    return out_dir
