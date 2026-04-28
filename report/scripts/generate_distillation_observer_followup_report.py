from __future__ import annotations

import ast
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SUMMARY_PATH = REPO_ROOT / "Distillation" / "Results" / "observer_pole_sweep_temp" / "20260428_010057" / "observer_pole_sweep_summary.csv"
FOLLOWUP_SUMMARY_PATH = REPO_ROOT / "Distillation" / "Results" / "observer_pole_sweep_temp" / "20260428_141210" / "observer_pole_sweep_summary.csv"
BASELINE_DATA_DIR = REPO_ROOT / "Distillation" / "Data" / "observer_pole_sweep_temp" / "20260428_010057"
FOLLOWUP_DATA_DIR = REPO_ROOT / "Distillation" / "Data" / "observer_pole_sweep_temp" / "20260428_141210"
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_observer_followup_20260428"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_LABEL = "p00_old_aggressive_reference"
FOLLOWUP_KEY_ORDER = [
    "q01_uniform_fast_minus_04",
    "q02_uniform_fast_minus_02",
    "q03_uniform_fast_center",
    "q04_uniform_fast_plus_02",
    "q05_uniform_fast_plus_04",
    "q06_tail_faster",
    "q07_tail_mid",
    "q08_tail_slower",
    "q09_tail_slowest",
    "q10_front_faster_a",
    "q11_front_faster_b",
]
SELECTED_LABELS = [
    REFERENCE_LABEL,
    "q01_uniform_fast_minus_04",
    "q11_front_faster_b",
    "q03_uniform_fast_center",
]


def reverse_min_max(scaled, data_min, data_max):
    scaled = np.asarray(scaled, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return scaled * (data_max - data_min) + data_min


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["poles_arr"] = df["poles"].apply(ast.literal_eval).apply(lambda x: np.asarray(x, float))
    df["pole_mean"] = df["poles_arr"].apply(np.mean)
    df["pole_max"] = df["poles_arr"].apply(np.max)
    df["pole_min"] = df["poles_arr"].apply(np.min)
    return df


def load_bundle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_selected_bundles():
    bundles = {
        REFERENCE_LABEL: load_bundle(BASELINE_DATA_DIR / f"{REFERENCE_LABEL}.pickle"),
    }
    for label in FOLLOWUP_KEY_ORDER:
        bundles[label] = load_bundle(FOLLOWUP_DATA_DIR / f"{label}.pickle")
    return bundles


def short_label(label: str) -> str:
    mapping = {
        REFERENCE_LABEL: "p00 ref",
        "q01_uniform_fast_minus_04": "q01",
        "q02_uniform_fast_minus_02": "q02",
        "q03_uniform_fast_center": "q03",
        "q04_uniform_fast_plus_02": "q04",
        "q05_uniform_fast_plus_04": "q05",
        "q06_tail_faster": "q06",
        "q07_tail_mid": "q07",
        "q08_tail_slower": "q08",
        "q09_tail_slowest": "q09",
        "q10_front_faster_a": "q10",
        "q11_front_faster_b": "q11",
    }
    return mapping.get(label, label)


def plot_reward_ranking(followup_df: pd.DataFrame, baseline_df: pd.DataFrame, output_path: Path) -> None:
    p00_reward = float(
        baseline_df.loc[baseline_df["label"] == REFERENCE_LABEL, "reward_last_episode_mean"].iloc[0]
    )
    plot_df = followup_df.sort_values("reward_last_episode_mean", ascending=True).reset_index(drop=True)
    colors = ["tab:green" if value > 0.0 else "tab:red" for value in plot_df["reward_last_episode_mean"]]
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.barh([short_label(label) for label in plot_df["label"]], plot_df["reward_last_episode_mean"], color=colors)
    ax.axvline(0.0, color="0.3", linewidth=1.0)
    ax.axvline(p00_reward, color="black", linewidth=1.4, linestyle="--", label="p00 reference")
    ax.set_title("Focused observer follow-up around p19: episode-2 reward ranking")
    ax.set_xlabel("Mean step reward over episode 2")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_trends(followup_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    uniform_order = [
        "q01_uniform_fast_minus_04",
        "q02_uniform_fast_minus_02",
        "q03_uniform_fast_center",
        "q04_uniform_fast_plus_02",
        "q05_uniform_fast_plus_04",
    ]
    tail_order = [
        "q06_tail_faster",
        "q03_uniform_fast_center",
        "q07_tail_mid",
        "q08_tail_slower",
        "q09_tail_slowest",
    ]
    front_order = [
        "q03_uniform_fast_center",
        "q10_front_faster_a",
        "q11_front_faster_b",
    ]
    panel_specs = [
        (axes[0], uniform_order, "Uniform neighborhood around q03"),
        (axes[1], tail_order, "Fixed front end, varied last two poles"),
        (axes[2], front_order, "Faster front-end variants"),
    ]
    for ax, order, title in panel_specs:
        subset = followup_df.set_index("label").loc[order].reset_index()
        ax.plot(
            np.arange(len(order)),
            subset["reward_last_episode_mean"],
            marker="o",
            linewidth=2.0,
            color="tab:blue",
        )
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([short_label(label) for label in order])
        ax.set_title(title)
        ax.set_ylabel("Episode-2 reward")
        ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_outputs_inputs(bundles: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    colors = {
        REFERENCE_LABEL: "black",
        "q01_uniform_fast_minus_04": "tab:green",
        "q11_front_faster_b": "tab:orange",
        "q03_uniform_fast_center": "tab:blue",
    }
    for label in SELECTED_LABELS:
        bundle = bundles[label]
        time_in_sub = int(bundle["time_in_sub_episodes"])
        start = time_in_sub
        end = 2 * time_in_sub
        y = np.asarray(bundle["y"], float)
        u = np.asarray(bundle["u"], float)
        data_min = np.asarray(bundle["data_min"], float)
        data_max = np.asarray(bundle["data_max"], float)
        steady_y = np.asarray(bundle["steady_states"]["y_ss"], float)
        n_inputs = int(u.shape[1])
        y_ss_scaled = (steady_y - data_min[n_inputs:]) / (data_max[n_inputs:] - data_min[n_inputs:])
        y_sp_abs_scaled = np.asarray(bundle["y_sp"], float) + y_ss_scaled
        y_sp_phys = reverse_min_max(y_sp_abs_scaled, data_min[n_inputs:], data_max[n_inputs:])
        t_y = np.arange(start, end + 1)
        t_sp = np.arange(start, end)
        t_u = np.arange(start, end)
        display = short_label(label)
        axes[0, 0].plot(t_y, y[start : end + 1, 0], color=colors[label], linewidth=2.0, label=display)
        axes[0, 1].plot(t_y, y[start : end + 1, 1], color=colors[label], linewidth=2.0, label=display)
        axes[1, 0].plot(t_u, u[start:end, 0], color=colors[label], linewidth=2.0, label=display)
        axes[1, 1].plot(t_u, u[start:end, 1], color=colors[label], linewidth=2.0, label=display)
        if label == REFERENCE_LABEL:
            axes[0, 0].plot(t_sp, y_sp_phys[start:end, 0], color="0.5", linestyle="--", linewidth=1.4, label="y1 setpoint")
            axes[0, 1].plot(t_sp, y_sp_phys[start:end, 1], color="0.5", linestyle="--", linewidth=1.4, label="y2 setpoint")

    titles = [
        "Tray-24 ethane composition",
        "Tray-85 temperature",
        "Reflux flow",
        "Reboiler duty",
    ]
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Step")
    axes[0, 0].set_ylabel("Output")
    axes[1, 0].set_ylabel("Input")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_selected_table(followup_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p00_row = baseline_df.loc[baseline_df["label"] == REFERENCE_LABEL].iloc[0]
    rows.append(
        {
            "label": REFERENCE_LABEL,
            "episode2_reward": float(p00_row["reward_last_episode_mean"]),
            "output1_mae": float(p00_row["output1_mae_last_episode"]),
            "output2_mae": float(p00_row["output2_mae_last_episode"]),
            "mean_pole": float(p00_row["pole_mean"]),
            "max_pole": float(p00_row["pole_max"]),
        }
    )
    for label in [
        "q01_uniform_fast_minus_04",
        "q11_front_faster_b",
        "q10_front_faster_a",
        "q06_tail_faster",
        "q03_uniform_fast_center",
        "q08_tail_slower",
        "q09_tail_slowest",
    ]:
        row = followup_df.loc[followup_df["label"] == label].iloc[0]
        rows.append(
            {
                "label": label,
                "episode2_reward": float(row["reward_last_episode_mean"]),
                "output1_mae": float(row["output1_mae_last_episode"]),
                "output2_mae": float(row["output2_mae_last_episode"]),
                "mean_pole": float(row["pole_mean"]),
                "max_pole": float(row["pole_max"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    baseline_df = load_summary(BASELINE_SUMMARY_PATH)
    followup_df = load_summary(FOLLOWUP_SUMMARY_PATH)
    bundles = load_selected_bundles()

    followup_df.sort_values("reward_last_episode_mean", ascending=False).to_csv(
        OUTPUT_DIR / "observer_followup_ranked_summary.csv", index=False
    )
    build_selected_table(followup_df, baseline_df).to_csv(
        OUTPUT_DIR / "observer_followup_selected_candidates.csv", index=False
    )

    plot_reward_ranking(followup_df, baseline_df, OUTPUT_DIR / "observer_followup_reward_ranking.png")
    plot_family_trends(followup_df, OUTPUT_DIR / "observer_followup_family_reward_trends.png")
    plot_selected_outputs_inputs(bundles, OUTPUT_DIR / "observer_followup_selected_outputs_inputs.png")


if __name__ == "__main__":
    main()
