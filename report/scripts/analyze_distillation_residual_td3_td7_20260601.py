"""Analyze the 2026-06-01 distillation residual TD3 and TD7 runs.

The script reads saved bundles only. It compares the latest residual TD3 and
TD7/SALE runs against the current OF-MPC baseline trajectory and the latest
compare-bundle reward recalculation.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "report" / "figures" / "distillation_residual_td3_td7_20260601"
BASE_SCRIPT = ROOT / "report" / "scripts" / "analyze_distillation_wider_safety_20260530.py"

RUNS = {
    "baseline": {
        "label": "OF-MPC",
        "method": "baseline",
        "path": ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
    },
    "td3": {
        "label": "TD3 residual",
        "method": "residual",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260601_170240"
        / "input_data.pkl",
    },
    "td7": {
        "label": "TD7 residual",
        "method": "residual",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td7_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260601_172611"
        / "input_data.pkl",
    },
}

COMPARE_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_compare_residual_td3_disturb_fluctuation"
    / "20260601_170255"
    / "input_data.pkl"
)

WINDOWS = {
    "warm_1_10": (1, 10),
    "handoff_guard_11_20": (11, 20),
    "guard_no_handoff_21_30": (21, 30),
    "unguarded_cap_31_40": (31, 40),
    "recovery_41_60": (41, 60),
    "tail_181_200": (181, 200),
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("distillation_wider_safety_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def finite_mean(values) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_quantile(values, q: float) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def episode_len(bundle: dict) -> int:
    return int(bundle.get("time_in_sub_episodes", 400))


def episode_slice(ep_start: int, ep_end: int, ep_len: int) -> slice:
    return slice((ep_start - 1) * ep_len, ep_end * ep_len)


def episode_rate(values: np.ndarray, ep_len: int) -> np.ndarray:
    arr = np.asarray(values, float)
    n_ep = arr.size // ep_len
    if n_ep <= 0:
        return np.asarray([], float)
    return arr[: n_ep * ep_len].reshape(n_ep, ep_len).mean(axis=1)


def episode_quantile(values: np.ndarray, ep_len: int, q: float) -> np.ndarray:
    arr = np.asarray(values, float)
    n_ep = arr.size // ep_len
    if n_ep <= 0:
        return np.asarray([], float)
    return np.quantile(arr[: n_ep * ep_len].reshape(n_ep, ep_len), q, axis=1)


def find_first_episode(avg: np.ndarray, baseline: np.ndarray, start_idx: int, mode: str) -> int | None:
    """Return one-based first recovery episode after start_idx."""

    if mode == "single":
        for idx in range(start_idx + 1, min(len(avg), len(baseline))):
            if avg[idx] >= baseline[idx]:
                return int(idx + 1)
    if mode == "mean5":
        for idx in range(start_idx + 1, min(len(avg), len(baseline)) - 4):
            if float(np.mean(avg[idx : idx + 5])) >= float(np.mean(baseline[idx : idx + 5])):
                return int(idx + 1)
    if mode == "all10":
        for idx in range(start_idx + 1, min(len(avg), len(baseline)) - 9):
            if bool(np.all(avg[idx : idx + 10] >= baseline[idx : idx + 10])):
                return int(idx + 1)
    return None


def baseline_bundle_with_current_reward() -> dict:
    baseline = dict(load_pickle(RUNS["baseline"]["path"]))
    compare = load_pickle(COMPARE_PATH)
    baseline["avg_rewards"] = np.asarray(compare["avg_rewards_mpc"], float)
    return baseline


def summary_rows(base, bundles: dict[str, dict]) -> pd.DataFrame:
    baseline_avg = np.asarray(bundles["baseline"]["avg_rewards"], float)
    rows = []
    for key, meta in RUNS.items():
        bundle = bundles[key]
        row = base.row_for(key, meta, bundle, "2026-06-01 residual TD3/TD7")
        avg = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)
        warm_start_step = int(bundle.get("warm_start_step", 0))
        ep_len = episode_len(bundle)
        warm_episodes = int(warm_start_step // max(1, ep_len))
        postwarm = avg[warm_episodes:]
        postwarm_min_local = int(np.nanargmin(postwarm)) if postwarm.size else 0
        postwarm_min_idx = warm_episodes + postwarm_min_local
        row["warm_episodes"] = warm_episodes
        row["postwarm_min_reward"] = float(avg[postwarm_min_idx]) if avg.size else float("nan")
        row["postwarm_min_episode"] = int(postwarm_min_idx + 1) if avg.size else -1
        row["recovery_first_episode_ge_baseline"] = find_first_episode(avg, baseline_avg, postwarm_min_idx, "single")
        row["recovery_first_5ep_mean_ge_baseline"] = find_first_episode(avg, baseline_avg, postwarm_min_idx, "mean5")
        row["recovery_first_10ep_all_ge_baseline"] = find_first_episode(avg, baseline_avg, postwarm_min_idx, "all10")
        if key != "baseline":
            guard_rate = episode_rate(bundle["residual_guard_active_log"], ep_len)
            cap_mean = episode_rate(np.isfinite(bundle["residual_active_cap_log"]), ep_len)
            cap_value = episode_rate(np.nan_to_num(bundle["residual_active_cap_log"], nan=0.0), ep_len)
            row["last_guard_episode"] = int(np.where(guard_rate > 0.5)[0][-1] + 1) if np.any(guard_rate > 0.5) else -1
            handoff = dict(bundle.get("bc_handoff", {}) or {})
            if bool(handoff.get("enabled", False)) and int(handoff.get("active_steps", 0)) > 0:
                handoff_end_step = int(handoff.get("start_step", warm_start_step + 1)) + int(
                    handoff.get("active_steps", 0)
                ) - 1
                row["last_handoff_episode"] = int(np.ceil(handoff_end_step / max(1, ep_len)))
            else:
                row["last_handoff_episode"] = -1
            full_cap = float(bundle.get("td3_authority_ramp", {}).get("end_cap", 0.02))
            full_cap_idx = np.where(cap_value >= full_cap - 1.0e-8)[0]
            row["first_full_cap_episode"] = int(full_cap_idx[0] + 1) if full_cap_idx.size else -1
            row["cap_active_episode_fraction"] = float(np.mean(cap_mean > 0.5))
            row["release_gate_release_step"] = int(bundle.get("protected_bc_release_gate_release_step", -1))
            row["release_gate_pass_fraction"] = finite_mean(bundle["release_gate_pass_log"])
            row["release_gate_live_blocking_enabled"] = bool(bundle.get("protected_bc_release_gate_live_blocking_enabled"))
            row["residual_reward_probation_enabled"] = bool(bundle.get("residual_reward_probation_enabled"))
            row["use_rho_authority"] = bool(bundle.get("use_rho_authority"))
            row["residual_authority_enabled"] = bool(bundle.get("residual_authority_enabled"))
        else:
            row["last_guard_episode"] = -1
            row["last_handoff_episode"] = -1
            row["first_full_cap_episode"] = -1
            row["cap_active_episode_fraction"] = float("nan")
            row["release_gate_release_step"] = -1
            row["release_gate_pass_fraction"] = float("nan")
            row["release_gate_live_blocking_enabled"] = False
            row["residual_reward_probation_enabled"] = False
            row["use_rho_authority"] = False
            row["residual_authority_enabled"] = False
        row["tail20_reward_delta_vs_baseline"] = float(row["reward_tail20"] - baseline_avg[-20:].mean())
        row["final_reward_delta_vs_baseline"] = float(row["reward_final"] - baseline_avg[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def window_rows(base, bundles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, meta in RUNS.items():
        bundle = bundles[key]
        avg = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)
        err, _y, ysp = base.step_error_phys(bundle)
        ep_len = episode_len(bundle)
        for window, (ep_start, ep_end) in WINDOWS.items():
            step_sl = episode_slice(ep_start, ep_end, ep_len)
            ep_sl = slice(ep_start - 1, ep_end)
            e = err[step_sl, :]
            ysp_w = ysp[step_sl, :]
            band = np.maximum(base.K_REL * np.abs(ysp_w), base.BAND_FLOOR_PHYS)
            abs_e = np.abs(e)
            row = {
                "key": key,
                "label": meta["label"],
                "window": window,
                "episode_start": ep_start,
                "episode_end": ep_end,
                "reward_mean": float(np.nanmean(avg[ep_sl])),
                "reward_min": float(np.nanmin(avg[ep_sl])),
                "reward_min_episode": int(ep_start + np.nanargmin(avg[ep_sl])),
                "comp_mae": float(np.nanmean(abs_e[:, 0])),
                "temp_mae": float(np.nanmean(abs_e[:, 1])),
                "band_norm_mae": float(np.nanmean(abs_e / np.maximum(band, 1.0e-12))),
                "outside_band_frac": float(np.nanmean(abs_e > band)),
            }
            if key != "baseline":
                requested = np.linalg.norm(bundle["delta_u_res_requested_log"], axis=1)
                post_cap = np.linalg.norm(bundle["delta_u_res_post_cap_log"], axis=1)
                post_guard = np.linalg.norm(bundle["delta_u_res_post_guard_log"], axis=1)
                executed = np.linalg.norm(bundle["delta_u_res_exec_log"], axis=1)
                row.update(
                    {
                        "handoff_authority_mean": finite_mean(bundle["bc_handoff_authority_log"][step_sl]),
                        "cap_mean": finite_mean(bundle["residual_active_cap_log"][step_sl]),
                        "cap_projection_frac": finite_mean(bundle["residual_cap_projection_active_log"][step_sl]),
                        "guard_active_frac": finite_mean(bundle["residual_guard_active_log"][step_sl]),
                        "guard_trigger_frac": finite_mean(bundle["residual_guard_triggered_log"][step_sl]),
                        "guard_selected_scale_mean": finite_mean(bundle["residual_guard_selected_scale_log"][step_sl]),
                        "requested_residual_norm_mean": finite_mean(requested[step_sl]),
                        "post_cap_residual_norm_mean": finite_mean(post_cap[step_sl]),
                        "post_guard_residual_norm_mean": finite_mean(post_guard[step_sl]),
                        "executed_residual_norm_mean": finite_mean(executed[step_sl]),
                        "executed_residual_norm_q95": finite_quantile(executed[step_sl], 0.95),
                        "release_gate_pass_frac": finite_mean(bundle["release_gate_pass_log"][step_sl]),
                        "release_gate_would_block_frac": finite_mean(bundle["release_gate_blocked_log"][step_sl]),
                        "release_gate_rolling_mean_gap": finite_mean(
                            bundle["release_gate_rolling_mean_gap_log"][step_sl]
                        ),
                        "release_gate_rolling_max_coord_gap": finite_mean(
                            bundle["release_gate_rolling_max_coordinate_gap_log"][step_sl]
                        ),
                        "policy_executed_gap_norm_mean": finite_mean(bundle["policy_executed_gap_norm_log"][step_sl]),
                        "shadow_rho_projection_frac": finite_mean(bundle["shadow_rho_projection_active_log"][step_sl]),
                        "shadow_rho_eff_mean": finite_mean(bundle["shadow_rho_eff_log"][step_sl]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def episode_rows(base, bundles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, meta in RUNS.items():
        bundle = bundles[key]
        avg = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)
        err, _y, ysp = base.step_error_phys(bundle)
        ep_len = episode_len(bundle)
        n_ep = min(len(avg), err.shape[0] // ep_len)
        band = np.maximum(base.K_REL * np.abs(ysp[: n_ep * ep_len]), base.BAND_FLOOR_PHYS)
        abs_e = np.abs(err[: n_ep * ep_len])
        comp_mae = abs_e[:, 0].reshape(n_ep, ep_len).mean(axis=1)
        temp_mae = abs_e[:, 1].reshape(n_ep, ep_len).mean(axis=1)
        band_norm = (abs_e / np.maximum(band, 1.0e-12)).reshape(n_ep, ep_len, 2).mean(axis=(1, 2))
        outside = (abs_e > band).reshape(n_ep, ep_len, 2).mean(axis=(1, 2))
        for idx in range(n_ep):
            row = {
                "key": key,
                "label": meta["label"],
                "episode": idx + 1,
                "reward": float(avg[idx]),
                "comp_mae": float(comp_mae[idx]),
                "temp_mae": float(temp_mae[idx]),
                "band_norm_mae": float(band_norm[idx]),
                "outside_band_frac": float(outside[idx]),
            }
            if key != "baseline":
                sl = episode_slice(idx + 1, idx + 1, ep_len)
                executed = np.linalg.norm(bundle["delta_u_res_exec_log"], axis=1)
                row.update(
                    {
                        "guard_active_frac": finite_mean(bundle["residual_guard_active_log"][sl]),
                        "guard_trigger_frac": finite_mean(bundle["residual_guard_triggered_log"][sl]),
                        "cap_projection_frac": finite_mean(bundle["residual_cap_projection_active_log"][sl]),
                        "cap_mean": finite_mean(bundle["residual_active_cap_log"][sl]),
                        "executed_residual_norm_mean": finite_mean(executed[sl]),
                        "executed_residual_norm_q95": finite_quantile(executed[sl], 0.95),
                        "release_gate_rolling_mean_gap": finite_mean(
                            bundle["release_gate_rolling_mean_gap_log"][sl]
                        ),
                        "release_gate_rolling_max_coord_gap": finite_mean(
                            bundle["release_gate_rolling_max_coordinate_gap_log"][sl]
                        ),
                        "release_gate_pass_frac": finite_mean(bundle["release_gate_pass_log"][sl]),
                        "policy_executed_gap_norm_mean": finite_mean(bundle["policy_executed_gap_norm_log"][sl]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def add_phase_lines(ax) -> None:
    phases = [
        (10, "warm end"),
        (20, "handoff end"),
        (30, "guard end"),
        (40, "full cap"),
    ]
    for episode, label in phases:
        ax.axvline(episode, color="#777777", linestyle="--", linewidth=0.9)
        ax.text(episode + 0.6, ax.get_ylim()[1] * 0.88, label, fontsize=8, color="#555555", rotation=90)


def plot_reward_collapse(episodes: pd.DataFrame) -> None:
    colors = {"baseline": "black", "td3": "#4c78a8", "td7": "#e45756"}
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=False)

    for key in ["baseline", "td3", "td7"]:
        df = episodes[episodes["key"] == key]
        axes[0].plot(df["episode"], df["reward"], label=df["label"].iloc[0], color=colors[key], linewidth=1.8)
    axes[0].axvspan(31, 40, color="#f58518", alpha=0.12, label="episode 31-40 collapse window")
    axes[0].set_xlim(1, 100)
    axes[0].set_ylabel("Average reward")
    axes[0].set_title("Residual reward collapse begins after the early guard expires")
    axes[0].grid(True, alpha=0.25)
    add_phase_lines(axes[0])
    axes[0].legend(fontsize=8, loc="lower right")

    for key in ["baseline", "td3", "td7"]:
        df = episodes[episodes["key"] == key]
        axes[1].plot(df["episode"], df["reward"], label=df["label"].iloc[0], color=colors[key], linewidth=1.5)
    axes[1].set_xlim(1, 200)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average reward")
    axes[1].set_title("Both residual agents recover and outperform OF-MPC late")
    axes[1].grid(True, alpha=0.25)
    add_phase_lines(axes[1])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_reward_collapse_recovery.png", dpi=180)
    plt.close(fig)


def plot_window_tracking(windows: pd.DataFrame) -> None:
    order = list(WINDOWS.keys())
    labels = ["warm", "handoff", "guard", "unguarded", "recover", "tail"]
    colors = {"baseline": "black", "td3": "#4c78a8", "td7": "#e45756"}
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)
    x = np.arange(len(order))
    width = 0.25
    offsets = {"baseline": -width, "td3": 0.0, "td7": width}

    for key in ["baseline", "td3", "td7"]:
        df = windows[windows["key"] == key].set_index("window").loc[order]
        axes[0].bar(x + offsets[key], df["reward_mean"], width, label=df["label"].iloc[0], color=colors[key])
        axes[1].bar(x + offsets[key], df["temp_mae"], width, label=df["label"].iloc[0], color=colors[key])
        axes[2].bar(x + offsets[key], df["band_norm_mae"], width, label=df["label"].iloc[0], color=colors[key])

    axes[0].set_ylabel("Reward mean")
    axes[1].set_ylabel("Temp MAE")
    axes[2].set_ylabel("Band-norm MAE")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Tracking degradation is concentrated in the unguarded cap-ramp window")
    axes[0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_window_tracking_errors.png", dpi=180)
    plt.close(fig)


def plot_safety_windows(windows: pd.DataFrame) -> None:
    order = list(WINDOWS.keys())[1:]
    labels = ["handoff", "guard", "unguarded", "recover", "tail"]
    colors = {"td3": "#4c78a8", "td7": "#e45756"}
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.6), sharex=True)
    x = np.arange(len(order))
    width = 0.36
    offsets = {"td3": -width / 2, "td7": width / 2}

    for key in ["td3", "td7"]:
        df = windows[windows["key"] == key].set_index("window").loc[order]
        axes[0].bar(x + offsets[key], df["guard_active_frac"], width, label=RUNS[key]["label"], color=colors[key])
        axes[1].bar(x + offsets[key], df["cap_projection_frac"], width, label=RUNS[key]["label"], color=colors[key])
        axes[2].bar(
            x + offsets[key],
            df["executed_residual_norm_mean"],
            width,
            label=RUNS[key]["label"],
            color=colors[key],
        )

    axes[0].set_ylabel("Guard active")
    axes[1].set_ylabel("Cap projection")
    axes[2].set_ylabel("Mean executed residual norm")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    for ax in axes:
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("The collapse appears when guard activity drops to zero while residual authority remains")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_safety_window_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_tail_tracking(summary: pd.DataFrame) -> None:
    df = summary.set_index("key").loc[["baseline", "td3", "td7"]]
    labels = ["OF-MPC", "TD3", "TD7"]
    colors = ["black", "#4c78a8", "#e45756"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    axes[0].bar(labels, df["tail20_comp_mae"], color=colors)
    axes[0].set_ylabel("Composition MAE")
    axes[1].bar(labels, df["tail20_temp_mae"], color=colors)
    axes[1].set_ylabel("Temperature MAE")
    axes[2].bar(labels, df["tail20_outside_band_frac"], color=colors)
    axes[2].set_ylabel("Outside-band fraction")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Tail-20 tracking comparison")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail_tracking_comparison.png", dpi=180)
    plt.close(fig)


def plot_release_gate(episodes: pd.DataFrame) -> None:
    colors = {"td3": "#4c78a8", "td7": "#e45756"}
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True)
    for key in ["td3", "td7"]:
        df = episodes[episodes["key"] == key]
        axes[0].plot(
            df["episode"],
            df["release_gate_rolling_mean_gap"],
            color=colors[key],
            label=RUNS[key]["label"],
            linewidth=1.6,
        )
        axes[1].plot(
            df["episode"],
            df["release_gate_rolling_max_coord_gap"],
            color=colors[key],
            label=RUNS[key]["label"],
            linewidth=1.6,
        )
    axes[0].axhline(0.25, color="#777777", linestyle="--", linewidth=1.0, label="mean gap threshold")
    axes[1].axhline(0.20, color="#777777", linestyle="--", linewidth=1.0, label="max coord threshold")
    axes[0].set_xlim(1, 100)
    axes[0].set_ylabel("Rolling mean action gap")
    axes[1].set_ylabel("Rolling max coordinate gap")
    axes[1].set_xlabel("Episode")
    axes[0].set_title("The diagnostic release gate never passes under current thresholds")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        add_phase_lines(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_release_gate_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    base = load_base_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bundles = {
        "baseline": baseline_bundle_with_current_reward(),
        "td3": load_pickle(RUNS["td3"]["path"]),
        "td7": load_pickle(RUNS["td7"]["path"]),
    }

    summary = summary_rows(base, bundles)
    windows = window_rows(base, bundles)
    episodes = episode_rows(base, bundles)

    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    windows.to_csv(OUT_DIR / "window_metrics.csv", index=False)
    episodes.to_csv(OUT_DIR / "episode_metrics.csv", index=False)

    plot_reward_collapse(episodes)
    plot_window_tracking(windows)
    plot_safety_windows(windows)
    plot_tail_tracking(summary)
    plot_release_gate(episodes)

    key_cols = [
        "label",
        "reward_tail20",
        "reward_final",
        "postwarm_min_reward",
        "postwarm_min_episode",
        "recovery_first_5ep_mean_ge_baseline",
        "tail20_comp_mae",
        "tail20_temp_mae",
        "tail20_band_norm_mae",
        "tail20_outside_band_frac",
        "tail20_reward_delta_vs_baseline",
        "final_reward_delta_vs_baseline",
        "last_guard_episode",
        "first_full_cap_episode",
        "release_gate_release_step",
    ]
    summary_json = {
        "runs": {key: str(meta["path"].relative_to(ROOT)) for key, meta in RUNS.items()},
        "compare_path": str(COMPARE_PATH.relative_to(ROOT)),
        "key_metrics": summary[key_cols].to_dict(orient="records"),
        "collapse_window": windows[windows["window"] == "unguarded_cap_31_40"].to_dict(orient="records"),
        "artifacts": {
            "figure_dir": str(OUT_DIR.relative_to(ROOT)),
            "csv": sorted(path.name for path in OUT_DIR.glob("*.csv")),
            "figures": sorted(path.name for path in OUT_DIR.glob("fig_*.png")),
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
