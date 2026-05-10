from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_latest_run_20260510"
BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"

RUN_SPECS = [
    {
        "run_id": "20260509_023119",
        "compare_id": "20260509_023133",
        "label": "2026-05-09 shared reward, guarded TD3",
        "short_label": "05-09 guarded shared",
    },
    {
        "run_id": "20260509_155540",
        "compare_id": "20260509_155552",
        "label": "2026-05-09 prototype reward rerun",
        "short_label": "05-09 proto rerun",
    },
    {
        "run_id": "20260509_184140",
        "compare_id": "20260509_184152",
        "label": "2026-05-09 prototype reward + prototype solver",
        "short_label": "05-09 proto solver",
    },
    {
        "run_id": "20260509_221004",
        "compare_id": "20260509_221016",
        "label": "2026-05-09 prototype reward + shared solver",
        "short_label": "05-09 proto shared",
    },
    {
        "run_id": "20260510_134814",
        "compare_id": "20260510_134825",
        "label": "2026-05-10 prototype reward, shared solver",
        "short_label": "05-10 proto shared",
    },
    {
        "run_id": "20260510_193643",
        "compare_id": "20260510_193656",
        "label": "2026-05-10 shared reward, forced TD3 execute",
        "short_label": "05-10 forced shared",
    },
]

N_EPISODES = 200
STEPS_PER_EPISODE = 800
WINDOW = 20


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def physical_to_scaled_abs(values: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return (values - data_min) / np.maximum(data_max - data_min, 1.0e-12)


def compute_episode_mae(
    y_phys: np.ndarray,
    y_sp_scaled: np.ndarray,
    steady_states: dict,
    data_min: np.ndarray,
    data_max: np.ndarray,
) -> np.ndarray:
    ss_y_scaled = physical_to_scaled_abs(np.asarray(steady_states["y_ss"], float), data_min[2:], data_max[2:])
    y_scaled = physical_to_scaled_abs(np.asarray(y_phys, float)[1:], data_min[2:], data_max[2:])
    y_dev = y_scaled - ss_y_scaled.reshape(1, -1)
    e = (y_dev - np.asarray(y_sp_scaled, float)).reshape(N_EPISODES, STEPS_PER_EPISODE, 2)
    return np.mean(np.abs(e), axis=1)


def compute_episode_input_movement(u_phys: np.ndarray) -> np.ndarray:
    u_ep = np.asarray(u_phys, float).reshape(N_EPISODES, STEPS_PER_EPISODE, -1)
    du = np.diff(u_ep, axis=1, prepend=u_ep[:, :1, :])
    return np.mean(np.linalg.norm(du, axis=2), axis=1)


def window_mean(values: np.ndarray, width: int = WINDOW) -> np.ndarray:
    values = np.asarray(values, float)
    out = []
    for start in range(0, len(values), width):
        stop = min(start + width, len(values))
        out.append(float(values[start:stop].mean()))
    return np.asarray(out, float)


def window_labels(width: int = WINDOW) -> list[str]:
    labels = []
    for start in range(0, N_EPISODES, width):
        stop = min(start + width, N_EPISODES)
        labels.append(f"{start + 1}-{stop}")
    return labels


def summarize_run(run_spec: dict, run: dict, compare_bundle: dict, baseline: dict) -> tuple[dict, np.ndarray]:
    reward_delta = np.asarray(compare_bundle["avg_rewards_rl"], float) - np.asarray(compare_bundle["avg_rewards_mpc"], float)
    y_sp_scaled = np.asarray(run["y_sp"], float)
    data_min = np.asarray(run["data_min"], float)
    data_max = np.asarray(run["data_max"], float)
    steady_states = run["steady_states"]

    mae_run = compute_episode_mae(np.asarray(run["y"], float), y_sp_scaled, steady_states, data_min, data_max)
    mae_base = compute_episode_mae(np.asarray(baseline["y_mpc"], float), y_sp_scaled, steady_states, data_min, data_max)
    move_run = compute_episode_input_movement(np.asarray(run["u"], float))
    move_base = compute_episode_input_movement(np.asarray(baseline["u_mpc"], float))

    action_source = np.asarray(run["rl_action_source_log"], int)
    reward_params = run.get("reward_params", {})
    summary_metrics = run.get("summary_metrics", {})

    row = {
        "run_id": run_spec["run_id"],
        "label": run_spec["label"],
        "reward_mode": reward_params.get("mode", "shared_relative"),
        "nominal_solver_mode": run.get("nominal_solver_mode", "legacy_prototype"),
        "force_td3_execute": bool(run.get("force_td3_execute", False)),
        "reward_delta_mean": float(reward_delta.mean()),
        "reward_delta_last20": float(reward_delta[-20:].mean()),
        "reward_better_frac": float((reward_delta > 0.0).mean()),
        "reward_better_last20_frac": float((reward_delta[-20:] > 0.0).mean()),
        "output1_mae_delta_mean": float((mae_run[:, 0] - mae_base[:, 0]).mean()),
        "output2_mae_delta_mean": float((mae_run[:, 1] - mae_base[:, 1]).mean()),
        "output1_mae_delta_last20": float((mae_run[-20:, 0] - mae_base[-20:, 0]).mean()),
        "output2_mae_delta_last20": float((mae_run[-20:, 1] - mae_base[-20:, 1]).mean()),
        "input_move_delta_mean": float((move_run - move_base).mean()),
        "input_move_delta_last20": float((move_run[-20:] - move_base[-20:]).mean()),
        "td3_fraction": float((action_source == 2).mean()),
        "ls_fraction": float((action_source == 3).mean()),
        "nominal_fraction": float((action_source == 4).mean()),
        "warm_fraction": float((action_source == 1).mean()),
        "prediction_score_mean": float(np.mean(np.asarray(run["s_pred_log"], float))),
        "gain_drift_mean": float(np.mean(np.asarray(run["gain_drift_log"], float))),
        "accepted_fraction": float(summary_metrics.get("accepted_fraction", np.nan)),
    }
    return row, reward_delta


def plot_reward_mode_history(rows: list[dict], out_path: Path):
    shared_rows = [row for row in rows if row["reward_mode"] == "shared_relative"]
    proto_rows = [row for row in rows if row["reward_mode"] != "shared_relative"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    panels = [
        (axes[0], shared_rows, "Shared Reward Runs"),
        (axes[1], proto_rows, "Prototype Reward Runs"),
    ]
    for ax, panel_rows, title in panels:
        xs = np.arange(len(panel_rows))
        labels = [row["short_label"] if "short_label" in row else row["label"] for row in panel_rows]
        mean_vals = np.asarray([row["reward_delta_mean"] for row in panel_rows], float)
        tail_vals = np.asarray([row["reward_delta_last20"] for row in panel_rows], float)
        colors = ["#2c7fb8" if not row["force_td3_execute"] else "#d95f0e" for row in panel_rows]
        ax.bar(xs, mean_vals, color=colors, alpha=0.85, label="full-run mean")
        ax.plot(xs, tail_vals, color="#111111", marker="o", linewidth=1.8, label="last-20 mean")
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Reward delta vs canonical MPC")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Polymer Markov Reward History by Reward Geometry", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_shared_window_compare(shared_window_data: list[tuple[str, np.ndarray]], out_path: Path):
    xs = np.arange(len(window_labels()))
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    colors = ["#3182bd", "#e6550d"]
    for (label, reward_delta), color in zip(shared_window_data, colors):
        ax.plot(xs, window_mean(reward_delta), marker="o", linewidth=2.3, color=color, label=label)
    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(window_labels(), rotation=30)
    ax.set_ylabel("20-episode reward delta")
    ax.set_xlabel("Episode window")
    ax.set_title("Shared-Reward Polymer Markov Runs Stay Below Canonical MPC")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_pickle(BASELINE_PATH)

    rows: list[dict] = []
    shared_window_data: list[tuple[str, np.ndarray]] = []

    for run_spec in RUN_SPECS:
        run_path = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / run_spec["run_id"] / "input_data.pkl"
        compare_path = (
            REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_markov" / run_spec["compare_id"] / "input_data.pkl"
        )
        run = load_pickle(run_path)
        compare_bundle = load_pickle(compare_path)
        row, reward_delta = summarize_run(run_spec, run, compare_bundle, baseline)
        row["short_label"] = run_spec["short_label"]
        rows.append(row)
        if row["reward_mode"] == "shared_relative":
            shared_window_data.append((run_spec["short_label"], reward_delta))

    write_csv(OUT_DIR / "recent_run_summary.csv", rows)

    latest_row = next(row for row in rows if row["run_id"] == "20260510_193643")
    shared_rows = [row for row in rows if row["reward_mode"] == "shared_relative"]
    summary = {
        "latest_run_id": latest_row["run_id"],
        "latest_compare_id": "20260510_193656",
        "latest_reward_delta_mean": latest_row["reward_delta_mean"],
        "latest_reward_delta_last20": latest_row["reward_delta_last20"],
        "latest_reward_better_frac": latest_row["reward_better_frac"],
        "latest_output1_mae_delta_mean": latest_row["output1_mae_delta_mean"],
        "latest_output2_mae_delta_mean": latest_row["output2_mae_delta_mean"],
        "latest_input_move_delta_mean": latest_row["input_move_delta_mean"],
        "latest_td3_fraction": latest_row["td3_fraction"],
        "latest_ls_fraction": latest_row["ls_fraction"],
        "latest_nominal_fraction": latest_row["nominal_fraction"],
        "latest_prediction_score_mean": latest_row["prediction_score_mean"],
        "latest_gain_drift_mean": latest_row["gain_drift_mean"],
        "all_shared_reward_runs_negative": bool(all(row["reward_delta_mean"] < 0.0 for row in shared_rows)),
        "shared_reward_run_ids": [row["run_id"] for row in shared_rows],
        "canonical_baseline_path": str(BASELINE_PATH),
        "note": "Unified run bundles duplicate y_mpc and u_mpc from the RL trajectory, so all baseline-sensitive metrics here use the canonical baseline pickle and compare bundles.",
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_reward_mode_history(rows, OUT_DIR / "reward_mode_history.png")
    plot_shared_window_compare(shared_window_data, OUT_DIR / "shared_reward_window_compare.png")


if __name__ == "__main__":
    main()
