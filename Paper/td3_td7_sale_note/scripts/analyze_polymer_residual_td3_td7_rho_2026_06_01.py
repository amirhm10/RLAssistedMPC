from __future__ import annotations

import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = (
    ROOT
    / "Paper"
    / "td3_td7_sale_note"
    / "figures"
    / "polymer_residual_td3_td7_rho_2026_06_01"
)

RUNS = [
    {
        "label": "MPC baseline",
        "short": "mpc",
        "path": ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle",
        "color": "#202124",
        "linestyle": "--",
    },
    {
        "label": "TD3 rho on",
        "short": "td3_rho_on",
        "path": ROOT
        / "Polymer"
        / "Results"
        / "td3_residual_disturb"
        / "20260520_214325"
        / "input_data.pkl",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "label": "TD3 rho off",
        "short": "td3_rho_off",
        "path": ROOT
        / "Polymer"
        / "Results"
        / "td3_residual_disturb"
        / "20260601_000722"
        / "input_data.pkl",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    {
        "label": "TD7 rho on",
        "short": "td7_rho_on",
        "path": ROOT
        / "Polymer"
        / "Results"
        / "td7_residual_disturb"
        / "20260531_225718"
        / "input_data.pkl",
        "color": "#2ca02c",
        "linestyle": "-",
    },
    {
        "label": "TD7 rho off",
        "short": "td7_rho_off",
        "path": ROOT
        / "Polymer"
        / "Results"
        / "td7_residual_disturb"
        / "20260601_002221"
        / "input_data.pkl",
        "color": "#d62728",
        "linestyle": "-",
    },
]


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def as_float_array(value, shape=None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if shape is not None and arr.shape != shape:
        return None
    return arr


def ysp_scaled_dev_to_phys(bundle: dict) -> np.ndarray:
    n_inputs = int(bundle.get("n_inputs", 2))
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], dtype=float)
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / np.maximum(
        data_max[n_inputs:] - data_min[n_inputs:], 1.0e-12
    )
    y_sp = np.asarray(bundle["y_sp"], dtype=float)
    y_sp_scaled = y_sp + y_ss_scaled
    return y_sp_scaled * (data_max[n_inputs:] - data_min[n_inputs:]) + data_min[n_inputs:]


def output_line(bundle: dict) -> np.ndarray:
    key = "y" if "y" in bundle else "y_mpc"
    return np.asarray(bundle[key], dtype=float)


def input_line(bundle: dict) -> np.ndarray:
    key = "u" if "u" in bundle else "u_mpc"
    return np.asarray(bundle[key], dtype=float)


def finite_mean(arr: np.ndarray | None, tail: int | None = None) -> float:
    if arr is None:
        return np.nan
    use = arr[-tail:] if tail is not None and arr.ndim > 0 else arr
    if not np.isfinite(use).any():
        return np.nan
    return float(np.nanmean(use))


def norm_trace(bundle: dict, key: str) -> np.ndarray | None:
    arr = as_float_array(bundle.get(key))
    if arr is None or arr.size == 0 or not np.isfinite(arr).any():
        return None
    if arr.ndim == 1:
        return arr
    return np.linalg.norm(arr, axis=1)


def compute_metrics(bundle: dict, label: str, short: str, source_path: Path) -> dict:
    n_fe = int(bundle["nFE"])
    delta_t = float(bundle.get("delta_t", 0.5))
    tail_window = int(bundle.get("time_in_sub_episodes", 800))
    y = output_line(bundle)[: n_fe + 1]
    u = input_line(bundle)[:n_fe]
    y_sp_phys = ysp_scaled_dev_to_phys(bundle)[:n_fe]
    e_phys = y[1 : n_fe + 1] - y_sp_phys
    dy_scaled = as_float_array(bundle.get("delta_y_storage"))
    rewards = as_float_array(bundle.get("avg_rewards"))

    row = {
        "label": label,
        "short": short,
        "path": str(source_path.relative_to(ROOT)),
        "agent": str(bundle.get("agent_kind", "mpc")),
        "algorithm": str(bundle.get("algorithm", "mpc")),
        "use_rho_authority": str(bundle.get("use_rho_authority", "baseline")),
        "residual_authority_enabled": str(bundle.get("residual_authority_enabled", "baseline")),
        "append_rho_to_state": str(bundle.get("append_rho_to_state", "baseline")),
        "notebook_source": str(bundle.get("notebook_source", "MPCOffsetFree_unified")),
        "n_fe": n_fe,
        "tail_window_steps": tail_window,
        "avg_reward_mean": finite_mean(rewards),
        "avg_reward_tail20": finite_mean(rewards, tail=20),
        "avg_reward_last": float(rewards[-1]) if rewards is not None and rewards.size else np.nan,
        "mae_eta": float(np.nanmean(np.abs(e_phys[:, 0]))),
        "mae_T": float(np.nanmean(np.abs(e_phys[:, 1]))),
        "rmse_eta": float(np.sqrt(np.nanmean(e_phys[:, 0] ** 2))),
        "rmse_T": float(np.sqrt(np.nanmean(e_phys[:, 1] ** 2))),
        "tail_mae_eta": float(np.nanmean(np.abs(e_phys[-tail_window:, 0]))),
        "tail_mae_T": float(np.nanmean(np.abs(e_phys[-tail_window:, 1]))),
        "tail_rmse_eta": float(np.sqrt(np.nanmean(e_phys[-tail_window:, 0] ** 2))),
        "tail_rmse_T": float(np.sqrt(np.nanmean(e_phys[-tail_window:, 1] ** 2))),
        "iae_eta": float(np.nansum(np.abs(e_phys[:, 0])) * delta_t),
        "iae_T": float(np.nansum(np.abs(e_phys[:, 1])) * delta_t),
    }

    if dy_scaled is not None and dy_scaled.size:
        scaled_l2 = np.linalg.norm(dy_scaled, axis=1)
        row.update(
            {
                "scaled_l2_mean": float(np.nanmean(scaled_l2)),
                "scaled_l2_tail": float(np.nanmean(scaled_l2[-tail_window:])),
                "scaled_l2_q95": float(np.nanpercentile(scaled_l2, 95)),
            }
        )

    input_move = np.linalg.norm(np.diff(u, axis=0), axis=1)
    row.update(
        {
            "input_move_mean": float(np.nanmean(input_move)),
            "input_move_q95": float(np.nanpercentile(input_move, 95)),
        }
    )

    for key in (
        "a_res_exec_log",
        "delta_u_res_exec_log",
        "rho_log",
        "rho_eff_log",
        "projection_active_log",
        "projection_due_to_authority_log",
        "residual_guard_triggered_log",
        "shadow_rho_exec_diff_norm_log",
        "shadow_rho_projection_due_to_authority_log",
        "shadow_rho_eff_log",
    ):
        trace = norm_trace(bundle, key)
        if trace is None:
            continue
        row[f"{key}_mean"] = float(np.nanmean(trace))
        row[f"{key}_tail"] = float(np.nanmean(trace[-tail_window:]))
        row[f"{key}_max"] = float(np.nanmax(trace))

    return row


def write_metrics_csv(rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    priority = [
        "label",
        "short",
        "agent",
        "algorithm",
        "use_rho_authority",
        "residual_authority_enabled",
        "append_rho_to_state",
        "avg_reward_mean",
        "avg_reward_tail20",
        "avg_reward_last",
        "scaled_l2_mean",
        "scaled_l2_tail",
        "mae_eta",
        "mae_T",
        "tail_mae_eta",
        "tail_mae_T",
        "a_res_exec_log_mean",
        "a_res_exec_log_tail",
        "delta_u_res_exec_log_mean",
        "delta_u_res_exec_log_tail",
        "rho_log_mean",
        "rho_log_tail",
        "rho_eff_log_mean",
        "rho_eff_log_tail",
        "shadow_rho_exec_diff_norm_log_mean",
        "shadow_rho_projection_due_to_authority_log_mean",
        "path",
        "notebook_source",
    ]
    ordered = [f for f in priority if f in fields] + [f for f in fields if f not in priority]
    out_path = OUT_DIR / "polymer_residual_td3_td7_rho_metrics.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def plot_reward(bundle_by_short: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for run in RUNS:
        bundle = bundle_by_short[run["short"]]
        rewards = as_float_array(bundle.get("avg_rewards"))
        if rewards is None:
            continue
        ax.plot(
            np.arange(1, rewards.size + 1),
            rewards,
            label=run["label"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=1.8,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward per episode")
    ax.set_title("Polymer residual supervisor learning trace")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_reward_episode_comparison.png", dpi=220)
    fig.savefig(OUT_DIR / "fig_reward_episode_comparison.pdf")
    plt.close(fig)


def plot_last_cycle_tracking(bundle_by_short: dict[str, dict]) -> None:
    ref_bundle = bundle_by_short["mpc"]
    n_fe = int(ref_bundle["nFE"])
    tail_window = int(ref_bundle.get("time_in_sub_episodes", 800))
    delta_t = float(ref_bundle.get("delta_t", 0.5))
    start = n_fe - tail_window
    t_y = np.arange(tail_window + 1) * delta_t / 60.0
    t_sp = np.arange(tail_window) * delta_t / 60.0
    output_labels = ["viscosity eta", "reactor temperature T"]
    units = ["", "K"]

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), sharex=True)
    for idx, ax in enumerate(axes):
        y_sp = ysp_scaled_dev_to_phys(ref_bundle)[start:n_fe, idx]
        ax.step(
            t_sp,
            y_sp,
            where="post",
            color="#000000",
            linestyle=":",
            linewidth=2.0,
            label="setpoint" if idx == 0 else None,
        )
        for run in RUNS:
            bundle = bundle_by_short[run["short"]]
            y = output_line(bundle)
            ax.plot(
                t_y,
                y[start : n_fe + 1, idx],
                color=run["color"],
                linestyle=run["linestyle"],
                linewidth=1.6,
                label=run["label"] if idx == 0 else None,
            )
        ylabel = output_labels[idx] if not units[idx] else f"{output_labels[idx]} ({units[idx]})"
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Last subepisode time (h)")
    axes[0].set_title("Last subepisode output tracking")
    axes[0].legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_last_cycle_tracking.png", dpi=220)
    fig.savefig(OUT_DIR / "fig_last_cycle_tracking.pdf")
    plt.close(fig)


def plot_residual_and_rho(bundle_by_short: dict[str, dict]) -> None:
    ref_bundle = bundle_by_short["mpc"]
    n_fe = int(ref_bundle["nFE"])
    tail_window = int(ref_bundle.get("time_in_sub_episodes", 800))
    delta_t = float(ref_bundle.get("delta_t", 0.5))
    start = n_fe - tail_window
    t = np.arange(tail_window) * delta_t / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    for run in RUNS[1:]:
        bundle = bundle_by_short[run["short"]]
        residual_norm = norm_trace(bundle, "delta_u_res_exec_log")
        if residual_norm is not None:
            axes[0].plot(
                t,
                residual_norm[start:n_fe],
                color=run["color"],
                linestyle=run["linestyle"],
                linewidth=1.5,
                label=run["label"],
            )
        rho_eff = norm_trace(bundle, "rho_eff_log")
        if rho_eff is not None and np.isfinite(rho_eff).any():
            axes[1].plot(
                t,
                rho_eff[start:n_fe],
                color=run["color"],
                linewidth=1.5,
                label=f"{run['label']} applied",
            )
        shadow_eff = norm_trace(bundle, "shadow_rho_eff_log")
        if shadow_eff is not None and np.isfinite(shadow_eff).any():
            axes[1].plot(
                t,
                shadow_eff[start:n_fe],
                color=run["color"],
                linestyle=":",
                linewidth=1.5,
                label=f"{run['label']} shadow",
            )

    axes[0].set_ylabel("Executed residual delta u norm")
    axes[1].set_ylabel("rho effective")
    axes[1].set_xlabel("Last subepisode time (h)")
    axes[0].set_title("Residual authority and rho attenuation in the last subepisode")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_authority_rho_tail.png", dpi=220)
    fig.savefig(OUT_DIR / "fig_residual_authority_rho_tail.pdf")
    plt.close(fig)


def plot_metric_bars(rows: list[dict]) -> None:
    baseline_tail = rows[0]["avg_reward_tail20"]
    method_rows = rows[1:]
    labels = [row["label"] for row in method_rows]
    x = np.arange(len(method_rows))
    colors = [run["color"] for run in RUNS[1:]]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    tail_gain = [row["avg_reward_tail20"] - baseline_tail for row in method_rows]
    scaled_l2 = [row["scaled_l2_tail"] for row in method_rows]
    residual_norm = [row.get("delta_u_res_exec_log_tail", np.nan) for row in method_rows]

    axes[0].bar(x, tail_gain, color=colors)
    axes[0].axhline(0.0, color="#202124", linewidth=1.0)
    axes[0].set_ylabel("Tail reward gain vs MPC")
    axes[0].set_title("Reward improvement")

    axes[1].bar(x, scaled_l2, color=colors)
    axes[1].axhline(rows[0]["scaled_l2_tail"], color="#202124", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Tail scaled tracking L2")
    axes[1].set_title("Tracking error")

    axes[2].bar(x, residual_norm, color=colors)
    axes[2].set_ylabel("Tail executed residual delta u norm")
    axes[2].set_title("Residual authority")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_metric_bars.png", dpi=220)
    fig.savefig(OUT_DIR / "fig_metric_bars.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_by_short = {run["short"]: load_pickle(run["path"]) for run in RUNS}
    rows = [
        compute_metrics(bundle_by_short[run["short"]], run["label"], run["short"], run["path"])
        for run in RUNS
    ]
    baseline_tail = rows[0]["avg_reward_tail20"]
    for row in rows:
        row["tail_reward_gain_vs_mpc"] = row["avg_reward_tail20"] - baseline_tail
        row["tail_reward_gain_vs_mpc_pct"] = (
            row["tail_reward_gain_vs_mpc"] / abs(baseline_tail) * 100.0
        )

    write_metrics_csv(rows)
    plot_reward(bundle_by_short)
    plot_last_cycle_tracking(bundle_by_short)
    plot_residual_and_rho(bundle_by_short)
    plot_metric_bars(rows)

    print(f"Wrote metrics and figures to {OUT_DIR.relative_to(ROOT)}")
    for row in rows:
        print(
            f"{row['label']}: tail reward {row['avg_reward_tail20']:.6f}, "
            f"tail gain {row['tail_reward_gain_vs_mpc']:.6f}, "
            f"tail scaled L2 {row.get('scaled_l2_tail', np.nan):.6f}"
        )


if __name__ == "__main__":
    main()
