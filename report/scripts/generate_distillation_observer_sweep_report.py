from __future__ import annotations

import ast
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_RESULT_DIR = REPO_ROOT / "Distillation" / "Results" / "observer_pole_sweep_temp" / "20260428_010057"
SWEEP_DATA_DIR = REPO_ROOT / "Distillation" / "Data" / "observer_pole_sweep_temp" / "20260428_010057"
SUMMARY_PATH = SWEEP_RESULT_DIR / "observer_pole_sweep_summary.csv"
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_observer_pole_sweep_20260428"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_LABELS = [
    "p00_old_aggressive_reference",
    "p19_uniform_fast",
    "p20_uniform_mid",
    "p01_mid_current_test",
    "p29_very_conservative_mixed",
    "p27_two_slow_offsets_wide_gap",
]
ANNOTATE_LABELS = {
    "p00_old_aggressive_reference",
    "p19_uniform_fast",
    "p20_uniform_mid",
    "p01_mid_current_test",
    "p29_very_conservative_mixed",
}


def reverse_min_max(scaled, data_min, data_max):
    scaled = np.asarray(scaled, float)
    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    return scaled * (data_max - data_min) + data_min


def load_bundle(label: str):
    with open(SWEEP_DATA_DIR / f"{label}.pickle", "rb") as handle:
        return pickle.load(handle)


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH)
    df["poles_arr"] = df["poles"].apply(ast.literal_eval).apply(lambda x: np.asarray(x, float))
    df["pole_mean"] = df["poles_arr"].apply(np.mean)
    df["pole_max"] = df["poles_arr"].apply(np.max)
    df["pole_min"] = df["poles_arr"].apply(np.min)
    df["pole_std"] = df["poles_arr"].apply(np.std)
    df["u_sat_total"] = (
        df["u1_lower_sat_frac"] + df["u1_upper_sat_frac"] + df["u2_lower_sat_frac"] + df["u2_upper_sat_frac"]
    )
    return df


def plot_reward_ranking(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("reward_last_episode_mean", ascending=True).reset_index(drop=True)
    colors = ["tab:green" if value > 0.0 else "tab:red" for value in plot_df["reward_last_episode_mean"]]
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    ax.barh(plot_df["label"], plot_df["reward_last_episode_mean"], color=colors)
    ax.axvline(0.0, color="0.3", linewidth=1.0)
    ax.set_title("Distillation observer sweep: last-episode reward ranking")
    ax.set_xlabel("Mean step reward over episode 2")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reward_vs_poles(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    scatter_cfg = [
        ("pole_mean", "Mean observer pole"),
        ("pole_max", "Largest observer pole"),
    ]
    for ax, (column, title) in zip(axes, scatter_cfg, strict=True):
        sc = ax.scatter(
            df[column],
            df["reward_last_episode_mean"],
            c=df["gain_fro_norm"],
            cmap="viridis",
            s=60,
            edgecolors="black",
            linewidths=0.3,
        )
        for _, row in df.iterrows():
            if row["label"] in ANNOTATE_LABELS:
                ax.annotate(
                    row["label"].replace("p00_", "").replace("p19_", "").replace("p20_", "").replace("p01_", "").replace("p29_", ""),
                    (row[column], row["reward_last_episode_mean"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        ax.axhline(0.0, color="0.3", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Mean step reward over episode 2")
        ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.9)
    cbar.set_label("Observer gain Frobenius norm")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_outputs_inputs(selected: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    color_cycle = ["black", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    for color, (label, bundle) in zip(color_cycle, selected.items(), strict=True):
        time_in_sub = int(bundle["time_in_sub_episodes"])
        start = time_in_sub
        end = 2 * time_in_sub
        y = np.asarray(bundle["y"], float)
        u = np.asarray(bundle["u"], float)
        data_min = np.asarray(bundle["data_min"], float)
        data_max = np.asarray(bundle["data_max"], float)
        steady_y = np.asarray(bundle["steady_states"]["y_ss"], float)
        n_inputs = int(u.shape[1])
        y_sp_phys = reverse_min_max(np.asarray(bundle["y_sp"], float) + reverse_min_max(np.zeros_like(steady_y), data_min[n_inputs:], data_max[n_inputs:]) * 0 + 0, data_min[n_inputs:], data_max[n_inputs:])
        # y_sp is stored as a deviation in scaled coordinates, so rebuild the absolute scaled target first.
        y_ss_scaled = (steady_y - data_min[n_inputs:]) / (data_max[n_inputs:] - data_min[n_inputs:])
        y_sp_abs_scaled = np.asarray(bundle["y_sp"], float) + y_ss_scaled
        y_sp_phys = reverse_min_max(y_sp_abs_scaled, data_min[n_inputs:], data_max[n_inputs:])
        t_y = np.arange(start, end + 1)
        t_u = np.arange(start, end)
        label_short = label.replace("p00_", "").replace("p19_", "").replace("p20_", "").replace("p01_", "").replace("p29_", "").replace("p27_", "")
        axes[0, 0].plot(t_y, y[start : end + 1, 0], color=color, linewidth=2.0, label=label_short)
        axes[0, 1].plot(t_y, y[start : end + 1, 1], color=color, linewidth=2.0, label=label_short)
        axes[1, 0].plot(t_u, u[start:end, 0], color=color, linewidth=2.0, label=label_short)
        axes[1, 1].plot(t_u, u[start:end, 1], color=color, linewidth=2.0, label=label_short)
        if color == color_cycle[0]:
            axes[0, 0].plot(t_u, y_sp_phys[start:end, 0], color="0.4", linestyle="--", linewidth=1.5, label="setpoint")
            axes[0, 1].plot(t_u, y_sp_phys[start:end, 1], color="0.4", linestyle="--", linewidth=1.5, label="setpoint")

    axes[0, 0].set_title("Episode 2 output: tray-24 ethane composition")
    axes[0, 1].set_title("Episode 2 output: tray-85 temperature")
    axes[1, 0].set_title("Episode 2 input: reflux")
    axes[1, 1].set_title("Episode 2 input: reboiler duty")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Step")
    axes[0, 0].set_ylabel("Composition (-)")
    axes[0, 1].set_ylabel("Temperature (K)")
    axes[1, 0].set_ylabel("kg/h")
    axes[1, 1].set_ylabel("GJ/h")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_summary()
    df.sort_values("reward_last_episode_mean", ascending=False).to_csv(
        OUTPUT_DIR / "observer_sweep_ranked_summary.csv", index=False
    )
    df.sort_values("reward_last_episode_mean", ascending=False).head(10).to_csv(
        OUTPUT_DIR / "observer_sweep_top10.csv", index=False
    )
    df.sort_values("reward_last_episode_mean", ascending=True).head(10).to_csv(
        OUTPUT_DIR / "observer_sweep_bottom10.csv", index=False
    )
    selected_df = df[df["label"].isin(SELECTED_LABELS)].copy()
    selected_df.to_csv(OUTPUT_DIR / "observer_sweep_selected_candidates.csv", index=False)

    plot_reward_ranking(df, OUTPUT_DIR / "observer_sweep_reward_ranking.png")
    plot_reward_vs_poles(df, OUTPUT_DIR / "observer_sweep_reward_vs_poles.png")
    bundles = {label: load_bundle(label) for label in SELECTED_LABELS}
    plot_selected_outputs_inputs(bundles, OUTPUT_DIR / "observer_sweep_selected_outputs_inputs.png")


if __name__ == "__main__":
    main()
