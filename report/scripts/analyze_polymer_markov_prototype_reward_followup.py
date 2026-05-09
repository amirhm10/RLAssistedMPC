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


OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_prototype_reward_followup_20260509"

LATEST_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260509_155540" / "input_data.pkl"
EARLIER_UNIFIED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260509_023119" / "input_data.pkl"
PREVIOUS_PROTOTYPE_RUN = (
    REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260508_123902" / "input_data.pkl"
)
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


def make_prototype_reward(bundle: dict, steady_states: dict):
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    y_ss_scaled = phys_to_scaled(np.asarray(steady_states["y_ss"], float), data_min[2:], data_max[2:])
    params, reward_fn = make_reward_fn_prototype_legacy(
        data_min=data_min,
        data_max=data_max,
        n_inputs=2,
        y_ss_scaled=y_ss_scaled,
        Q_diag=np.asarray(bundle["reward_params"]["Q_diag"], float),
        R_diag=np.asarray(bundle["reward_params"]["R_diag"], float),
        bonus_A=float(bundle["reward_params"]["bonus_A"]),
        bonus_B=float(bundle["reward_params"]["bonus_B"]),
    )
    return params, reward_fn


def calc_prototype_reward_components(
    y_phys: np.ndarray,
    u_phys: np.ndarray,
    y_sp_scaled: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
    bonus_A: float,
    bonus_B: float,
) -> dict:
    y_phys = np.asarray(y_phys, float)
    u_phys = np.asarray(u_phys, float)
    y_sp_scaled = np.asarray(y_sp_scaled, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    q_diag = np.asarray(q_diag, float).reshape(1, -1)
    r_diag = np.asarray(r_diag, float).reshape(1, -1)

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
    bonus = np.where(inside_gate, float(bonus_A) * np.exp(-float(bonus_B) * mean_percentage_error), 0.0)
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
        "td3_fraction": float((source == 2).mean()),
        "ls_fraction": float((source == 3).mean()),
        "nominal_fraction": float((source == 4).mean()),
        "warm_fraction": float((source == 1).mean()),
        "z_sat_any98_fraction": float((z.max(axis=1) >= 0.98 * Z_BOUND).mean()),
        "z_norm_mean": float(np.linalg.norm(np.asarray(bundle["z_executed_log"], float), axis=1).mean()),
        "z_abs_mean_1": float(z[:, 0].mean()),
        "z_abs_mean_2": float(z[:, 1].mean()),
        "z_abs_mean_3": float(z[:, 2].mean()),
        "z_abs_mean_4": float(z[:, 3].mean()),
        "prediction_score_mean": float(np.asarray(bundle["s_pred_log"], float).mean()),
        "gain_drift_mean": float(np.asarray(bundle["gain_drift_log"], float).mean()),
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
        ("Latest unified run vs canonical MPC", deltas["latest_vs_baseline"], "#1f77b4"),
        ("Earlier unified run vs canonical MPC", deltas["earlier_vs_baseline"], "#2ca02c"),
        ("Previous prototype vs previous nominal", deltas["previous_vs_previous_nominal"], "#ff7f0e"),
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


def plot_last_episode_differences(latest: dict, baseline: dict, previous: dict, out_path: Path):
    t = np.arange(STEPS_PER_EPISODE) * float(latest["delta_t"])
    latest_y_diff = np.asarray(latest["y"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        baseline["y_mpc"], float
    )[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]
    latest_u_diff = np.asarray(latest["u"], float).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        baseline["u_mpc"], float
    ).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]

    previous_y_diff = np.asarray(previous["y_markov"], float)[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        previous["y_nominal"], float
    )[1:].reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]
    previous_u_diff = np.asarray(previous["u_markov"], float).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1] - np.asarray(
        previous["u_nominal"], float
    ).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    for row, data, title in [
        (0, latest_y_diff, "Latest unified minus canonical MPC"),
        (1, previous_y_diff, "Previous prototype minus previous nominal"),
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
    axes[1, 0].set_xlabel("Time in final episode [h]")
    axes[1, 1].set_xlabel("Time in final episode [h]")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, constrained_layout=True)
    axes[0].plot(t, latest_u_diff[:, 0], color="#1f77b4", linewidth=1.8, label="Latest vs canonical")
    axes[0].plot(t, previous_u_diff[:, 0], color="#ff7f0e", linewidth=1.8, label="Previous vs previous nominal")
    axes[0].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_title("Qc difference")
    axes[0].set_ylabel("Input difference")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(t, latest_u_diff[:, 1], color="#1f77b4", linewidth=1.8, label="Latest vs canonical")
    axes[1].plot(t, previous_u_diff[:, 1], color="#ff7f0e", linewidth=1.8, label="Previous vs previous nominal")
    axes[1].axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[1].set_title("Qm difference")
    axes[1].set_ylabel("Input difference")
    axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("Time in final episode [h]")
    fig.savefig(out_path.with_name("last_episode_input_differences.png"), dpi=220)
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


def plot_comparator_effect(summary_rows: list[dict], out_path: Path):
    labels = [row["label"] for row in summary_rows]
    full = [row["reward_delta_mean"] for row in summary_rows]
    tail = [row["reward_delta_last20"] for row in summary_rows]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, vals, title in zip(axes, [full, tail], ["Full-run prototype reward delta", "Last-20 prototype reward delta"]):
        bars = ax.bar(labels, vals, color=colors[: len(vals)])
        ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("Markov minus comparator reward")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        span = max(max(abs(v) for v in vals), 1.0)
        for bar, value in zip(bars, vals):
            offset = 0.03 * span if value >= 0 else -0.05 * span
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                rotation=90,
            )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    latest = load_pickle(LATEST_RUN)
    earlier = load_pickle(EARLIER_UNIFIED_RUN)
    previous = load_pickle(PREVIOUS_PROTOTYPE_RUN)
    baseline = load_pickle(CANONICAL_BASELINE)

    proto_params, _reward_fn = make_prototype_reward(latest, latest["steady_states"])
    q_diag = np.asarray(proto_params["Q_diag"], float)
    r_diag = np.asarray(proto_params["R_diag"], float)
    bonus_A = float(proto_params["bonus_A"])
    bonus_B = float(proto_params["bonus_B"])
    data_min = np.asarray(latest["data_min"], float)
    data_max = np.asarray(latest["data_max"], float)

    latest_comp = calc_prototype_reward_components(
        latest["y"], latest["u"], latest["y_sp"], latest["steady_states"], data_min, data_max, q_diag, r_diag, bonus_A, bonus_B
    )
    earlier_comp = calc_prototype_reward_components(
        earlier["y"], earlier["u"], earlier["y_sp"], earlier["steady_states"], data_min, data_max, q_diag, r_diag, bonus_A, bonus_B
    )
    previous_markov_comp = calc_prototype_reward_components(
        previous["y_markov"],
        previous["u_markov"],
        previous["y_sp"],
        previous["steady_states"],
        data_min,
        data_max,
        q_diag,
        r_diag,
        bonus_A,
        bonus_B,
    )
    previous_nominal_comp = calc_prototype_reward_components(
        previous["y_nominal"],
        previous["u_nominal"],
        previous["y_sp"],
        previous["steady_states"],
        data_min,
        data_max,
        q_diag,
        r_diag,
        bonus_A,
        bonus_B,
    )
    baseline_comp = calc_prototype_reward_components(
        baseline["y_mpc"], baseline["u_mpc"], baseline["y_sp"], previous["steady_states"], data_min, data_max, q_diag, r_diag, bonus_A, bonus_B
    )

    latest_vs_baseline = episode_proto_delta(latest_comp, baseline_comp)
    earlier_vs_baseline = episode_proto_delta(earlier_comp, baseline_comp)
    previous_vs_previous_nominal = episode_proto_delta(previous_markov_comp, previous_nominal_comp)
    previous_nominal_vs_baseline = episode_proto_delta(previous_nominal_comp, baseline_comp)

    summary_rows = [
        summarize_delta("latest_unified_vs_canonical_baseline", latest_vs_baseline),
        summarize_delta("earlier_unified_vs_canonical_baseline", earlier_vs_baseline),
        summarize_delta("previous_prototype_vs_previous_nominal", previous_vs_previous_nominal),
        summarize_delta("previous_nominal_vs_canonical_baseline", previous_nominal_vs_baseline),
    ]
    write_csv(OUT_DIR / "prototype_reward_followup_summary.csv", summary_rows)

    source_rows = [
        source_summary(latest, "latest_unified_proto_reward"),
        source_summary(earlier, "earlier_unified_shared_reward_run"),
        source_summary(previous, "previous_prototype"),
    ]
    write_csv(OUT_DIR / "action_mix_summary.csv", source_rows)

    window_rows = []
    for label, delta_dict in [
        ("latest_unified_vs_canonical_baseline", latest_vs_baseline),
        ("earlier_unified_vs_canonical_baseline", earlier_vs_baseline),
        ("previous_prototype_vs_previous_nominal", previous_vs_previous_nominal),
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
    write_csv(OUT_DIR / "windowed_prototype_reward_deltas.csv", window_rows)

    plot_proto_reward_window_compare(
        {
            "latest_vs_baseline": latest_vs_baseline["reward"],
            "earlier_vs_baseline": earlier_vs_baseline["reward"],
            "previous_vs_previous_nominal": previous_vs_previous_nominal["reward"],
        },
        OUT_DIR / "prototype_reward_window_compare.png",
    )
    plot_latest_component_delta(latest_vs_baseline, OUT_DIR / "latest_prototype_reward_components.png")
    plot_last_episode_differences(latest, baseline, previous, OUT_DIR / "last_episode_output_differences.png")
    plot_action_mix_summary(source_rows, OUT_DIR / "action_mix_three_run_compare.png")
    plot_comparator_effect(summary_rows, OUT_DIR / "prototype_reward_comparator_effect.png")

    print(OUT_DIR)


if __name__ == "__main__":
    main()
