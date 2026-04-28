from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "matrix_multiplier_bc_strengthened_20260428"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "scalar_old": {
        "method": "Scalar matrix",
        "variant": "BC-only, 2026-04-27",
        "style": "old",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260427_185932" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260427_185946" / "input_data.pkl",
    },
    "scalar_new": {
        "method": "Scalar matrix",
        "variant": "Stronger BC, 2026-04-27",
        "style": "new",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260427_234906" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260427_234922" / "input_data.pkl",
    },
    "structured_old": {
        "method": "Structured matrix",
        "variant": "BC-only, 2026-04-27",
        "style": "old",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260427_190030" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260427_190047" / "input_data.pkl",
    },
    "structured_new": {
        "method": "Structured matrix",
        "variant": "Stronger BC, 2026-04-27",
        "style": "new",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260427_234944" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260427_235002" / "input_data.pkl",
    },
}

WINDOWS = {
    "11-20": (10, 20),
    "21-40": (20, 40),
    "41-100": (40, 100),
    "101-200": (100, 200),
    "1-200": (0, 200),
}
DIAG_WINDOWS = {
    "11-20": (10, 20),
    "21-40": (20, 40),
    "101-200": (100, 200),
}


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def get_exec_multiplier_log(bundle: dict) -> np.ndarray:
    if "executed_multiplier_log" in bundle:
        return np.asarray(bundle["executed_multiplier_log"], float)
    if "effective_multiplier_log" in bundle:
        return np.asarray(bundle["effective_multiplier_log"], float)
    return np.asarray(bundle["candidate_multiplier_log"], float)


def get_action_saturation_log(bundle: dict) -> np.ndarray:
    if "action_saturation_fraction_log" in bundle:
        return np.asarray(bundle["action_saturation_fraction_log"], float)
    return np.asarray(bundle["action_saturation_trace"], float)


def safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def summarize_reward_windows(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        reward_delta = np.asarray(spec["compare"]["avg_rewards_rl"], float) - np.asarray(spec["compare"]["avg_rewards_mpc"], float)
        for window_name, (start, end) in WINDOWS.items():
            rows.append(
                {
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "style": spec["style"],
                    "window": window_name,
                    "reward_delta_mean": float(np.mean(reward_delta[start:end])),
                    "win_rate": float(np.mean(reward_delta[start:end] > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_diagnostics(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        time_in_sub = int(bundle["time_in_sub_episodes"])
        bc_weight = np.asarray(bundle["bc_weight_log"], float)
        bc_policy_nominal = np.asarray(bundle["bc_policy_nominal_distance_log"], float)
        bc_actor_target = np.asarray(bundle["bc_actor_target_distance_log"], float)
        fallback_active = np.asarray(bundle["mpc_acceptance_fallback_active_log"], float)
        sat = get_action_saturation_log(bundle)
        exec_mult = get_exec_multiplier_log(bundle)
        mult_distance = np.linalg.norm(exec_mult - 1.0, axis=1)
        for window_name, (start_ep, end_ep) in DIAG_WINDOWS.items():
            start_idx = start_ep * time_in_sub
            end_idx = end_ep * time_in_sub
            sat_end = min(end_idx, sat.shape[0])
            actor_slice = bc_actor_target[start_idx:end_idx]
            actor_slice = actor_slice[np.isfinite(actor_slice) & (actor_slice > 0.0)]
            rows.append(
                {
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "style": spec["style"],
                    "window": window_name,
                    "bc_weight_mean": safe_mean(bc_weight[start_idx:end_idx]),
                    "policy_nominal_distance_mean": safe_mean(bc_policy_nominal[start_idx:end_idx]),
                    "actor_target_distance_mean": safe_mean(actor_slice),
                    "multiplier_distance_mean": safe_mean(mult_distance[start_idx:end_idx]),
                    "action_saturation_mean": safe_mean(sat[start_idx:sat_end]),
                    "fallback_steps": int(np.sum(fallback_active[start_idx:end_idx] > 0.5)),
                }
            )
    return pd.DataFrame(rows)


def summarize_isolation(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        cfg = bundle.get("config_snapshot", {})
        candidate = np.asarray(bundle["candidate_multiplier_log"], float)
        executed = get_exec_multiplier_log(bundle)
        release_clip = np.asarray(bundle.get("release_clip_fraction_log", np.zeros(candidate.shape[0])), float)
        fallback = np.asarray(bundle["mpc_acceptance_fallback_active_log"], float)
        rows.append(
            {
                "method": spec["method"],
                "variant": spec["variant"],
                "bc_enabled": bool(bundle["behavioral_cloning"]["enabled"]),
                "release_guard_enabled": bool(bundle.get("release_guard_enabled", False)),
                "acceptance_fallback_enabled": bool(bundle["mpc_acceptance_fallback"]["enabled"]),
                "release_clip_steps": int(np.sum(release_clip > 0.0)),
                "fallback_steps": int(np.sum(fallback > 0.5)),
                "candidate_executed_max_diff": float(np.max(np.abs(candidate - executed))),
                "post_warm_start_action_freeze_subepisodes": int(cfg.get("post_warm_start_action_freeze_subepisodes", 0)),
                "post_warm_start_actor_freeze_subepisodes": int(cfg.get("post_warm_start_actor_freeze_subepisodes", 0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_multiplier_means(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        labels = bundle.get("action_labels")
        if labels is None:
            labels = ["alpha", "B_col_1", "B_col_2"]
        exec_mult = get_exec_multiplier_log(bundle)
        time_in_sub = int(bundle["time_in_sub_episodes"])
        for window_name, (start_ep, end_ep) in DIAG_WINDOWS.items():
            sl = slice(start_ep * time_in_sub, end_ep * time_in_sub)
            means = np.mean(exec_mult[sl], axis=0)
            row = {
                "method": spec["method"],
                "variant": spec["variant"],
                "style": spec["style"],
                "window": window_name,
            }
            for idx, label in enumerate(labels):
                row[str(label)] = float(means[idx])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_reward_delta_traces(run_data: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    method_order = ["Scalar matrix", "Structured matrix"]
    style_map = {"old": ("BC-only, 0.1/10", "tab:blue"), "new": ("Stronger BC", "tab:orange")}
    for row_idx, method in enumerate(method_order):
        specs = [spec for spec in run_data.values() if spec["method"] == method]
        for col_idx, x_limit in enumerate([60, 200]):
            ax = axes[row_idx, col_idx]
            for spec in specs:
                label, color = style_map[spec["style"]]
                delta = np.asarray(spec["compare"]["avg_rewards_rl"], float) - np.asarray(spec["compare"]["avg_rewards_mpc"], float)
                x = np.arange(1, delta.shape[0] + 1)
                ax.plot(x[:x_limit], delta[:x_limit], label=label, color=color, linewidth=2.0)
            for boundary in [10, 20, 40, 100]:
                if boundary <= x_limit:
                    ax.axvline(boundary, color="0.75", linewidth=1.0, linestyle="--")
            ax.axhline(0.0, color="0.4", linewidth=1.0)
            ax.set_title(f"{method} reward delta, episodes 1-{x_limit}")
            ax.set_xlabel("Episode")
            ax.set_ylabel("RL - MPC avg reward")
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reward_windows(reward_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    window_order = ["11-20", "21-40", "41-100", "101-200", "1-200"]
    style_colors = {"old": "tab:blue", "new": "tab:orange"}
    for ax, method in zip(axes, ["Scalar matrix", "Structured matrix"], strict=True):
        subset = reward_df[reward_df["method"] == method]
        x = np.arange(len(window_order))
        width = 0.36
        for idx, style in enumerate(["old", "new"]):
            vals = [
                float(subset[(subset["style"] == style) & (subset["window"] == w)]["reward_delta_mean"].iloc[0])
                for w in window_order
            ]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, color=style_colors[style], label="BC-only, 0.1/10" if style == "old" else "Stronger BC")
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(window_order)
        ax.set_title(method)
        ax.set_ylabel("Mean reward delta vs MPC")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(diag_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    window_order = ["11-20", "21-40", "101-200"]
    metrics = [
        ("bc_weight_mean", "Mean BC weight"),
        ("policy_nominal_distance_mean", "Policy-nominal distance"),
        ("multiplier_distance_mean", "Multiplier distance"),
        ("action_saturation_mean", "Action saturation"),
    ]
    style_colors = {"old": "tab:blue", "new": "tab:orange"}
    for row_idx, method in enumerate(["Scalar matrix", "Structured matrix"]):
        subset = diag_df[diag_df["method"] == method]
        for col_idx, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(window_order))
            width = 0.36
            for idx, style in enumerate(["old", "new"]):
                vals = [
                    float(subset[(subset["style"] == style) & (subset["window"] == w)][metric_key].iloc[0])
                    for w in window_order
                ]
                ax.bar(
                    x + (idx - 0.5) * width,
                    vals,
                    width=width,
                    color=style_colors[style],
                    label="BC-only, 0.1/10" if style == "old" else "Stronger BC",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(window_order)
            if row_idx == 0:
                ax.set_title(metric_title)
            if col_idx == 0:
                ax.set_ylabel(method)
            ax.grid(True, axis="y", alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_data: dict[str, dict] = {}
    for key, spec in RUNS.items():
        run_data[key] = {
            **spec,
            "rl": load_pickle(spec["rl_path"]),
            "compare": load_pickle(spec["compare_path"]),
        }

    reward_df = summarize_reward_windows(run_data)
    diag_df = summarize_diagnostics(run_data)
    isolation_df = summarize_isolation(run_data)
    multiplier_df = summarize_multiplier_means(run_data)

    reward_df.to_csv(OUTPUT_DIR / "polymer_step4_strengthened_reward_windows.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "polymer_step4_strengthened_handoff_diagnostics.csv", index=False)
    isolation_df.to_csv(OUTPUT_DIR / "polymer_step4_strengthened_isolation_checks.csv", index=False)
    multiplier_df.to_csv(OUTPUT_DIR / "polymer_step4_strengthened_multiplier_means.csv", index=False)

    plot_reward_delta_traces(run_data, OUTPUT_DIR / "polymer_step4_strengthened_reward_delta_traces.png")
    plot_reward_windows(reward_df, OUTPUT_DIR / "polymer_step4_strengthened_reward_windows.png")
    plot_diagnostics(diag_df, OUTPUT_DIR / "polymer_step4_strengthened_handoff_diagnostics.png")


if __name__ == "__main__":
    main()
