from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.rewards import make_reward_fn_prototype_legacy


OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_prototype_nominal_solver_followup_20260509"

LATEST_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260509_184140" / "input_data.pkl"
PREV_UNIFIED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260509_155540" / "input_data.pkl"
OLD_PROTOTYPE_RUN = REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260508_123902" / "input_data.pkl"
CANONICAL_BASELINE = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"

WINDOW = 20
N_EPISODES = 200
STEPS_PER_EPISODE = 800
Z_BOUND = 0.05


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def phys_to_scaled(values: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return (values - data_min) / np.maximum(data_max - data_min, 1.0e-12)


def make_prototype_reward_recipe(bundle: dict):
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    y_ss_scaled = phys_to_scaled(np.asarray(bundle["steady_states"]["y_ss"], float), data_min[2:], data_max[2:])
    params, _reward_fn = make_reward_fn_prototype_legacy(
        data_min=data_min,
        data_max=data_max,
        n_inputs=2,
        y_ss_scaled=y_ss_scaled,
        Q_diag=np.asarray(bundle["reward_params"]["Q_diag"], float),
        R_diag=np.asarray(bundle["reward_params"]["R_diag"], float),
        bonus_A=float(bundle["reward_params"]["bonus_A"]),
        bonus_B=float(bundle["reward_params"]["bonus_B"]),
    )
    return data_min, data_max, params


def calc_prototype_reward_components(
    y_phys: np.ndarray,
    u_phys: np.ndarray,
    y_sp_scaled: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
    params: dict,
) -> dict:
    y_phys = np.asarray(y_phys, float)
    u_phys = np.asarray(u_phys, float)
    y_sp_scaled = np.asarray(y_sp_scaled, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    q_diag = np.asarray(params["Q_diag"], float).reshape(1, -1)
    r_diag = np.asarray(params["R_diag"], float).reshape(1, -1)

    ss_u_scaled = phys_to_scaled(np.asarray(steady_states["ss_inputs"], float), data_min[:2], data_max[:2])
    ss_y_scaled = phys_to_scaled(np.asarray(steady_states["y_ss"], float), data_min[2:], data_max[2:])

    u_scaled = phys_to_scaled(u_phys, data_min[:2], data_max[:2])
    u_dev = u_scaled - ss_u_scaled.reshape(1, -1)
    du = np.vstack([np.zeros((1, 2), dtype=float), np.diff(u_dev, axis=0)])

    y_scaled = phys_to_scaled(y_phys[1:], data_min[2:], data_max[2:])
    y_dev = y_scaled - ss_y_scaled.reshape(1, -1)
    e_scaled = y_dev - y_sp_scaled

    track_cost = np.sum(q_diag * (e_scaled**2), axis=1)
    move_cost = np.sum(r_diag * (du**2), axis=1)
    percentage_error = np.abs(e_scaled / (y_sp_scaled + 1.0e-15)) * 100.0
    mean_percentage_error = np.mean(percentage_error, axis=1)
    inside_gate = np.all(percentage_error <= 5.0, axis=1)
    bonus = np.where(inside_gate, float(params["bonus_A"]) * np.exp(-float(params["bonus_B"]) * mean_percentage_error), 0.0)
    reward = -(track_cost + move_cost) + bonus

    return {
        "reward": reward,
        "track_cost": track_cost,
        "move_cost": move_cost,
        "bonus": bonus,
        "mean_percentage_error": mean_percentage_error,
        "inside_gate": inside_gate.astype(float),
        "e_scaled": e_scaled,
        "du_scaled": du,
    }


def episode_mean(series: np.ndarray) -> np.ndarray:
    return np.asarray(series, float).reshape(N_EPISODES, STEPS_PER_EPISODE).mean(axis=1)


def window_mean(series: np.ndarray, width: int = WINDOW) -> np.ndarray:
    series = np.asarray(series, float)
    out = []
    for start in range(0, len(series), width):
        stop = min(start + width, len(series))
        out.append(float(series[start:stop].mean()))
    return np.asarray(out, float)


def window_labels(width: int = WINDOW) -> list[str]:
    labels = []
    for start in range(0, N_EPISODES, width):
        stop = min(start + width, N_EPISODES)
        labels.append(f"{start + 1}-{stop}")
    return labels


def episode_proto_delta(a_components: dict, b_components: dict) -> dict:
    out = {}
    for key in ["reward", "track_cost", "move_cost", "bonus", "mean_percentage_error", "inside_gate"]:
        out[key] = episode_mean(a_components[key]) - episode_mean(b_components[key])
    return out


def summarize_delta(label: str, episode_delta_dict: dict) -> dict:
    reward_delta = np.asarray(episode_delta_dict["reward"], float)
    return {
        "label": label,
        "reward_delta_mean": float(reward_delta.mean()),
        "reward_delta_last20": float(reward_delta[-20:].mean()),
        "reward_better_frac": float((reward_delta > 0.0).mean()),
        "reward_better_last20_frac": float((reward_delta[-20:] > 0.0).mean()),
        "bonus_delta_mean": float(np.asarray(episode_delta_dict["bonus"], float).mean()),
        "bonus_delta_last20": float(np.asarray(episode_delta_dict["bonus"], float)[-20:].mean()),
        "track_cost_delta_mean": float(np.asarray(episode_delta_dict["track_cost"], float).mean()),
        "track_cost_delta_last20": float(np.asarray(episode_delta_dict["track_cost"], float)[-20:].mean()),
        "move_cost_delta_mean": float(np.asarray(episode_delta_dict["move_cost"], float).mean()),
        "move_cost_delta_last20": float(np.asarray(episode_delta_dict["move_cost"], float)[-20:].mean()),
        "inside_gate_delta_mean": float(np.asarray(episode_delta_dict["inside_gate"], float).mean()),
        "inside_gate_delta_last20": float(np.asarray(episode_delta_dict["inside_gate"], float)[-20:].mean()),
        "mean_percentage_error_delta_mean": float(np.asarray(episode_delta_dict["mean_percentage_error"], float).mean()),
        "mean_percentage_error_delta_last20": float(
            np.asarray(episode_delta_dict["mean_percentage_error"], float)[-20:].mean()
        ),
    }


def source_summary(bundle: dict, label: str) -> dict:
    source = np.asarray(bundle["rl_action_source_log"], int)
    z = np.abs(np.asarray(bundle["z_executed_log"], float))
    return {
        "label": label,
        "nominal_solver_mode": bundle.get("nominal_solver_mode"),
        "td3_fraction": float((source == 2).mean()),
        "ls_fraction": float((source == 3).mean()),
        "nominal_fraction": float((source == 4).mean()),
        "warm_fraction": float((source == 1).mean()),
        "z_sat_any98_fraction": float((z.max(axis=1) >= 0.98 * Z_BOUND).mean()),
        "z_norm_mean": float(np.linalg.norm(np.asarray(bundle["z_executed_log"], float), axis=1).mean()),
        "prediction_score_mean": float(np.asarray(bundle["s_pred_log"], float).mean()),
        "gain_drift_mean": float(np.asarray(bundle["gain_drift_log"], float).mean()),
    }


def scaled_error_episodes(y_phys: np.ndarray, y_sp_scaled: np.ndarray, steady_states: dict, data_min: np.ndarray, data_max: np.ndarray):
    y_scaled = phys_to_scaled(np.asarray(y_phys, float)[1:], data_min[2:], data_max[2:])
    y_ss_scaled = phys_to_scaled(np.asarray(steady_states["y_ss"], float), data_min[2:], data_max[2:])
    y_dev = y_scaled - y_ss_scaled.reshape(1, -1)
    e_scaled = y_dev - np.asarray(y_sp_scaled, float)
    return e_scaled.reshape(N_EPISODES, STEPS_PER_EPISODE, e_scaled.shape[1])


def episode_move_metric(u_phys: np.ndarray) -> np.ndarray:
    u_ep = np.asarray(u_phys, float).reshape(N_EPISODES, STEPS_PER_EPISODE, -1)
    du = np.diff(u_ep, axis=1, prepend=u_ep[:, :1, :])
    return np.mean(np.linalg.norm(du, axis=2), axis=1)


def trajectory_summary(
    label: str,
    a_bundle: dict,
    b_bundle: dict,
    a_y: np.ndarray,
    b_y: np.ndarray,
    a_u: np.ndarray,
    b_u: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
) -> dict:
    e_a = scaled_error_episodes(a_y, a_bundle["y_sp"], steady_states, data_min, data_max)
    e_b = scaled_error_episodes(b_y, b_bundle["y_sp"], steady_states, data_min, data_max)
    move_a = episode_move_metric(a_u)
    move_b = episode_move_metric(b_u)
    y_diff = np.asarray(a_y, float) - np.asarray(b_y, float)
    u_diff = np.asarray(a_u, float) - np.asarray(b_u, float)
    return {
        "label": label,
        "output_mae_delta_full_1": float(np.mean(np.abs(e_a), axis=(0, 1))[0] - np.mean(np.abs(e_b), axis=(0, 1))[0]),
        "output_mae_delta_full_2": float(np.mean(np.abs(e_a), axis=(0, 1))[1] - np.mean(np.abs(e_b), axis=(0, 1))[1]),
        "output_mae_delta_last20_1": float(np.mean(np.abs(e_a[-20:]), axis=(0, 1))[0] - np.mean(np.abs(e_b[-20:]), axis=(0, 1))[0]),
        "output_mae_delta_last20_2": float(np.mean(np.abs(e_a[-20:]), axis=(0, 1))[1] - np.mean(np.abs(e_b[-20:]), axis=(0, 1))[1]),
        "input_move_delta_full": float(move_a.mean() - move_b.mean()),
        "input_move_delta_last20": float(move_a[-20:].mean() - move_b[-20:].mean()),
        "output_rmse_diff_1": float(np.sqrt(np.mean(y_diff[:, 0] ** 2))),
        "output_rmse_diff_2": float(np.sqrt(np.mean(y_diff[:, 1] ** 2))),
        "input_rmse_diff_1": float(np.sqrt(np.mean(u_diff[:, 0] ** 2))),
        "input_rmse_diff_2": float(np.sqrt(np.mean(u_diff[:, 1] ** 2))),
        "output_max_abs_diff": float(np.max(np.abs(y_diff))),
        "input_max_abs_diff": float(np.max(np.abs(u_diff))),
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_proto_reward_window_compare(deltas: dict[str, np.ndarray], out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    xs = np.arange(len(window_labels()))
    for label, values, color in [
        ("Latest lifted-G0 run vs canonical MPC", deltas["latest_vs_baseline"], "#1f77b4"),
        ("Previous unified proto-reward run vs canonical MPC", deltas["prev_vs_baseline"], "#2ca02c"),
        ("Old prototype vs old nominal", deltas["old_vs_old_nominal"], "#ff7f0e"),
    ]:
        ax.plot(xs, window_mean(values), marker="o", linewidth=2.2, label=label, color=color)
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(window_labels(), rotation=30)
    ax.set_ylabel("Prototype reward delta")
    ax.set_xlabel("Episode window")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_latest_component_delta(delta_dict: dict, out_path: Path):
    xs = np.arange(len(window_labels()))
    reward_delta = window_mean(delta_dict["reward"])
    bonus_contrib = window_mean(delta_dict["bonus"])
    track_contrib = window_mean(-delta_dict["track_cost"])
    move_contrib = window_mean(-delta_dict["move_cost"])

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(xs, reward_delta, marker="o", linewidth=2.4, label="Total reward delta", color="#1f77b4")
    ax.plot(xs, bonus_contrib, marker="s", linewidth=2.0, label="Bonus contribution", color="#d62728")
    ax.plot(xs, track_contrib, marker="^", linewidth=1.8, label="Tracking contribution", color="#9467bd")
    ax.plot(xs, move_contrib, marker="d", linewidth=1.8, label="Move contribution", color="#8c564b")
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(window_labels(), rotation=30)
    ax.set_ylabel("Prototype reward contribution")
    ax.set_xlabel("Episode window")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_action_mix_summary(rows: list[dict], out_path: Path):
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    base = np.zeros(len(rows), dtype=float)
    for key, color, legend in [
        ("warm_fraction", "#8c564b", "warm-start LS"),
        ("td3_fraction", "#1f77b4", "TD3 accepted"),
        ("ls_fraction", "#ff7f0e", "LS fallback"),
        ("nominal_fraction", "#7f7f7f", "nominal fallback"),
    ]:
        vals = np.asarray([row[key] for row in rows], float)
        axes[0].bar(x, vals, width=width, bottom=base, color=color, label=legend)
        base += vals
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fraction of steps")
    axes[0].set_title("Action-source mix")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(x, [row["z_sat_any98_fraction"] for row in rows], marker="o", linewidth=2.2, label="Any z saturation")
    axes[1].plot(x, [row["z_norm_mean"] for row in rows], marker="s", linewidth=2.0, label="Mean ||z||")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_title("Correction aggressiveness")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_solver_mode_switch_effect(delta_dict: dict, out_path: Path):
    xs = np.arange(len(window_labels()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].plot(xs, window_mean(delta_dict["reward"]), marker="o", linewidth=2.2, label="Reward delta")
    axes[0].plot(xs, window_mean(delta_dict["bonus"]), marker="s", linewidth=2.0, label="Bonus delta")
    axes[0].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(window_labels(), rotation=30)
    axes[0].set_title("Latest lifted-G0 run minus previous unified run")
    axes[0].set_ylabel("Episode-average delta")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(xs, window_mean(delta_dict["inside_gate"]), marker="^", linewidth=2.0, label="Inside-5% gate delta")
    axes[1].plot(xs, window_mean(delta_dict["mean_percentage_error"]), marker="d", linewidth=2.0, label="Mean % error delta")
    axes[1].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(window_labels(), rotation=30)
    axes[1].set_title("Threshold metrics")
    axes[1].set_ylabel("Episode-average delta")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_last_episode_differences(latest: dict, baseline: dict, prev_unified: dict, old_proto: dict, out_path: Path):
    t = np.arange(STEPS_PER_EPISODE) * float(latest["delta_t"])

    latest_y_diff = np.asarray(latest["y"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        baseline["y_mpc"], float
    )[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]
    prev_y_diff = np.asarray(prev_unified["y"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        baseline["y_mpc"], float
    )[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]
    old_y_diff = np.asarray(old_proto["y_markov"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        old_proto["y_nominal"], float
    )[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]

    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True, constrained_layout=True)
    for row, data, title in [
        (0, latest_y_diff, "Latest lifted-G0 run minus canonical MPC"),
        (1, prev_y_diff, "Previous unified proto-reward run minus canonical MPC"),
        (2, old_y_diff, "Old prototype minus old nominal"),
    ]:
        axes[row, 0].plot(t, data[:, 0], color="#1f77b4", linewidth=2.0)
        axes[row, 0].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        axes[row, 0].set_title(title)
        axes[row, 0].set_ylabel("Viscosity diff")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(t, data[:, 1], color="#ff7f0e", linewidth=2.0)
        axes[row, 1].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        axes[row, 1].set_title(title)
        axes[row, 1].set_ylabel("Temperature diff")
        axes[row, 1].grid(alpha=0.25)

    axes[2, 0].set_xlabel("Time in final episode [h]")
    axes[2, 1].set_xlabel("Time in final episode [h]")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    latest_u_diff = np.asarray(latest["u"], float).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        baseline["u_mpc"], float
    ).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]
    prev_u_diff = np.asarray(prev_unified["u"], float).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        baseline["u_mpc"], float
    ).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]
    old_u_diff = np.asarray(old_proto["u_markov"], float).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        old_proto["u_nominal"], float
    ).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, constrained_layout=True)
    for idx, title in enumerate(["Qc difference", "Qm difference"]):
        axes[idx].plot(t, latest_u_diff[:, idx], color="#1f77b4", linewidth=1.8, label="Latest lifted-G0 vs canonical")
        axes[idx].plot(t, prev_u_diff[:, idx], color="#2ca02c", linewidth=1.8, label="Previous unified vs canonical")
        axes[idx].plot(t, old_u_diff[:, idx], color="#ff7f0e", linewidth=1.8, label="Old prototype vs old nominal")
        axes[idx].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        axes[idx].set_title(title)
        axes[idx].set_ylabel("Input difference")
        axes[idx].grid(alpha=0.25)
        axes[idx].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("Time in final episode [h]")
    fig.savefig(out_path.with_name("last_episode_input_differences.png"), dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    latest = load_pickle(LATEST_RUN)
    prev_unified = load_pickle(PREV_UNIFIED_RUN)
    old_proto = load_pickle(OLD_PROTOTYPE_RUN)
    baseline = load_pickle(CANONICAL_BASELINE)

    data_min, data_max, proto_params = make_prototype_reward_recipe(latest)

    latest_comp = calc_prototype_reward_components(
        latest["y"], latest["u"], latest["y_sp"], latest["steady_states"], data_min, data_max, proto_params
    )
    prev_unified_comp = calc_prototype_reward_components(
        prev_unified["y"], prev_unified["u"], prev_unified["y_sp"], prev_unified["steady_states"], data_min, data_max, proto_params
    )
    old_proto_markov_comp = calc_prototype_reward_components(
        old_proto["y_markov"], old_proto["u_markov"], old_proto["y_sp"], old_proto["steady_states"], data_min, data_max, proto_params
    )
    old_proto_nominal_comp = calc_prototype_reward_components(
        old_proto["y_nominal"], old_proto["u_nominal"], old_proto["y_sp"], old_proto["steady_states"], data_min, data_max, proto_params
    )
    baseline_comp = calc_prototype_reward_components(
        baseline["y_mpc"], baseline["u_mpc"], baseline["y_sp"], latest["steady_states"], data_min, data_max, proto_params
    )

    latest_vs_baseline = episode_proto_delta(latest_comp, baseline_comp)
    prev_vs_baseline = episode_proto_delta(prev_unified_comp, baseline_comp)
    old_vs_old_nominal = episode_proto_delta(old_proto_markov_comp, old_proto_nominal_comp)
    latest_vs_prev = episode_proto_delta(latest_comp, prev_unified_comp)

    summary_rows = [
        summarize_delta("latest_lifted_g0_vs_canonical_baseline", latest_vs_baseline),
        summarize_delta("prev_unified_proto_reward_vs_canonical_baseline", prev_vs_baseline),
        summarize_delta("latest_lifted_g0_vs_prev_unified_proto_reward", latest_vs_prev),
        summarize_delta("old_prototype_vs_old_nominal", old_vs_old_nominal),
    ]
    write_csv(OUT_DIR / "prototype_nominal_solver_followup_summary.csv", summary_rows)

    source_rows = [
        source_summary(latest, "latest_lifted_g0_run"),
        source_summary(prev_unified, "prev_unified_proto_reward_run"),
        source_summary(old_proto, "old_prototype"),
    ]
    write_csv(OUT_DIR / "action_mix_summary.csv", source_rows)

    trajectory_rows = [
        trajectory_summary(
            "latest_lifted_g0_vs_canonical_baseline",
            latest,
            {"y_sp": baseline["y_sp"]},
            latest["y"],
            baseline["y_mpc"],
            latest["u"],
            baseline["u_mpc"],
            latest["steady_states"],
            data_min,
            data_max,
        ),
        trajectory_summary(
            "prev_unified_proto_reward_vs_canonical_baseline",
            prev_unified,
            {"y_sp": baseline["y_sp"]},
            prev_unified["y"],
            baseline["y_mpc"],
            prev_unified["u"],
            baseline["u_mpc"],
            prev_unified["steady_states"],
            data_min,
            data_max,
        ),
        trajectory_summary(
            "latest_lifted_g0_vs_prev_unified_proto_reward",
            latest,
            prev_unified,
            latest["y"],
            prev_unified["y"],
            latest["u"],
            prev_unified["u"],
            latest["steady_states"],
            data_min,
            data_max,
        ),
        trajectory_summary(
            "old_prototype_vs_old_nominal",
            old_proto,
            old_proto,
            old_proto["y_markov"],
            old_proto["y_nominal"],
            old_proto["u_markov"],
            old_proto["u_nominal"],
            old_proto["steady_states"],
            data_min,
            data_max,
        ),
    ]
    write_csv(OUT_DIR / "trajectory_distance_summary.csv", trajectory_rows)

    window_rows = []
    for label, delta_dict in [
        ("latest_lifted_g0_vs_canonical_baseline", latest_vs_baseline),
        ("prev_unified_proto_reward_vs_canonical_baseline", prev_vs_baseline),
        ("latest_lifted_g0_vs_prev_unified_proto_reward", latest_vs_prev),
    ]:
        for idx, start in enumerate(range(0, N_EPISODES, WINDOW)):
            stop = min(start + WINDOW, N_EPISODES)
            window_rows.append(
                {
                    "label": label,
                    "episode_window": f"{start + 1}-{stop}",
                    "reward_delta": float(window_mean(delta_dict["reward"])[idx]),
                    "bonus_delta": float(window_mean(delta_dict["bonus"])[idx]),
                    "track_cost_delta": float(window_mean(delta_dict["track_cost"])[idx]),
                    "move_cost_delta": float(window_mean(delta_dict["move_cost"])[idx]),
                    "inside_gate_delta": float(window_mean(delta_dict["inside_gate"])[idx]),
                    "mean_percentage_error_delta": float(window_mean(delta_dict["mean_percentage_error"])[idx]),
                }
            )
    write_csv(OUT_DIR / "solver_mode_switch_window_summary.csv", window_rows)

    plot_proto_reward_window_compare(
        {
            "latest_vs_baseline": latest_vs_baseline["reward"],
            "prev_vs_baseline": prev_vs_baseline["reward"],
            "old_vs_old_nominal": old_vs_old_nominal["reward"],
        },
        OUT_DIR / "reward_window_compare.png",
    )
    plot_latest_component_delta(latest_vs_baseline, OUT_DIR / "latest_components_vs_baseline.png")
    plot_action_mix_summary(source_rows, OUT_DIR / "action_mix_compare.png")
    plot_solver_mode_switch_effect(latest_vs_prev, OUT_DIR / "solver_mode_switch_effect.png")
    plot_last_episode_differences(latest, baseline, prev_unified, old_proto, OUT_DIR / "last_episode_output_differences.png")

    print(OUT_DIR)


if __name__ == "__main__":
    main()
