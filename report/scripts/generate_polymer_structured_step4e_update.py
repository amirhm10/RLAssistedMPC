from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "matrix_multiplier_structured_step4e_20260428"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "bc_only": {
        "variant": "BC-only, 0.1/10",
        "style": "bc_only",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260427_190030" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260427_190047" / "input_data.pkl",
    },
    "strong_uniform": {
        "variant": "Stronger uniform BC, 0.6/25",
        "style": "strong_uniform",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260427_234944" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260427_235002" / "input_data.pkl",
    },
    "weighted": {
        "variant": "Weighted BC, Step 4E",
        "style": "weighted",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260428_025906" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260428_025922" / "input_data.pkl",
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
STYLE_COLORS = {
    "bc_only": "tab:blue",
    "strong_uniform": "tab:orange",
    "weighted": "tab:green",
}


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def summarize_reward_windows(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        delta = np.asarray(spec["compare"]["avg_rewards_rl"], float) - np.asarray(spec["compare"]["avg_rewards_mpc"], float)
        for window_name, (start, end) in WINDOWS.items():
            rows.append(
                {
                    "variant": spec["variant"],
                    "style": spec["style"],
                    "window": window_name,
                    "reward_delta_mean": float(np.mean(delta[start:end])),
                    "win_rate": float(np.mean(delta[start:end] > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_diagnostics(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        time_in_sub = int(bundle["time_in_sub_episodes"])
        bc_weight = np.asarray(bundle["bc_weight_log"], float)
        policy_nominal = np.asarray(bundle["bc_policy_nominal_distance_log"], float)
        actor_target = np.asarray(bundle["bc_actor_target_distance_log"], float)
        effective_multiplier = np.asarray(bundle["effective_multiplier_log"], float)
        sat = np.asarray(bundle["action_saturation_fraction_log"], float)
        fallback = np.asarray(bundle["mpc_acceptance_fallback_active_log"], float)
        mult_distance = np.linalg.norm(effective_multiplier - 1.0, axis=1)
        for window_name, (start_ep, end_ep) in DIAG_WINDOWS.items():
            start_idx = start_ep * time_in_sub
            end_idx = end_ep * time_in_sub
            actor_slice = actor_target[start_idx:end_idx]
            actor_slice = actor_slice[np.isfinite(actor_slice) & (actor_slice > 0.0)]
            row = {
                "variant": spec["variant"],
                "style": spec["style"],
                "window": window_name,
                "bc_weight_mean": safe_mean(bc_weight[start_idx:end_idx]),
                "policy_nominal_distance_mean": safe_mean(policy_nominal[start_idx:end_idx]),
                "actor_target_distance_mean": safe_mean(actor_slice),
                "multiplier_distance_mean": safe_mean(mult_distance[start_idx:end_idx]),
                "action_saturation_mean": safe_mean(sat[start_idx:end_idx]),
                "fallback_steps": int(np.sum(fallback[start_idx:end_idx] > 0.5)),
            }
            means = np.mean(effective_multiplier[start_idx:end_idx], axis=0)
            labels = [str(label) for label in bundle["action_labels"]]
            for idx, label in enumerate(labels):
                row[label] = float(means[idx])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_isolation(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        candidate = np.asarray(bundle["candidate_multiplier_log"], float)
        executed = np.asarray(bundle["effective_multiplier_log"], float)
        release_clip = np.asarray(bundle["release_clip_fraction_log"], float)
        fallback = np.asarray(bundle["mpc_acceptance_fallback_active_log"], float)
        cfg = dict(bundle.get("config_snapshot", {}))
        rows.append(
            {
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


def plot_reward_traces(run_data: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, x_limit in zip(axes, [60, 200], strict=True):
        for spec in run_data.values():
            delta = np.asarray(spec["compare"]["avg_rewards_rl"], float) - np.asarray(spec["compare"]["avg_rewards_mpc"], float)
            x = np.arange(1, delta.shape[0] + 1)
            ax.plot(
                x[:x_limit],
                delta[:x_limit],
                label=spec["variant"],
                color=STYLE_COLORS[spec["style"]],
                linewidth=2.0,
            )
        for boundary in [10, 20, 40, 100]:
            if boundary <= x_limit:
                ax.axvline(boundary, color="0.75", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_title(f"Structured reward delta, episodes 1-{x_limit}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("RL - MPC avg reward")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reward_windows(reward_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    window_order = ["11-20", "21-40", "41-100", "101-200", "1-200"]
    x = np.arange(len(window_order))
    width = 0.24
    style_order = ["bc_only", "strong_uniform", "weighted"]
    for idx, style in enumerate(style_order):
        subset = reward_df[reward_df["style"] == style]
        vals = [float(subset[subset["window"] == w]["reward_delta_mean"].iloc[0]) for w in window_order]
        label = subset["variant"].iloc[0]
        ax.bar(x + (idx - 1) * width, vals, width=width, color=STYLE_COLORS[style], label=label)
    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(window_order)
    ax.set_ylabel("Mean reward delta vs MPC")
    ax.set_title("Structured Step 4 variants by reward window")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(diag_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    window_order = ["11-20", "21-40", "101-200"]
    metrics = [
        ("bc_weight_mean", "Mean BC weight"),
        ("policy_nominal_distance_mean", "Policy-nominal distance"),
        ("multiplier_distance_mean", "Multiplier distance"),
        ("action_saturation_mean", "Action saturation"),
    ]
    style_order = ["bc_only", "strong_uniform", "weighted"]
    x = np.arange(len(window_order))
    width = 0.24
    for ax, (metric_key, metric_title) in zip(axes, metrics, strict=True):
        for idx, style in enumerate(style_order):
            subset = diag_df[diag_df["style"] == style]
            vals = [float(subset[subset["window"] == w][metric_key].iloc[0]) for w in window_order]
            ax.bar(x + (idx - 1) * width, vals, width=width, color=STYLE_COLORS[style], label=subset["variant"].iloc[0])
        ax.set_xticks(x)
        ax.set_xticklabels(window_order)
        ax.set_title(metric_title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
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

    reward_df.to_csv(OUTPUT_DIR / "polymer_structured_step4e_reward_windows.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "polymer_structured_step4e_handoff_diagnostics.csv", index=False)
    isolation_df.to_csv(OUTPUT_DIR / "polymer_structured_step4e_isolation_checks.csv", index=False)

    plot_reward_traces(run_data, OUTPUT_DIR / "polymer_structured_step4e_reward_delta_traces.png")
    plot_reward_windows(reward_df, OUTPUT_DIR / "polymer_structured_step4e_reward_windows.png")
    plot_diagnostics(diag_df, OUTPUT_DIR / "polymer_structured_step4e_handoff_diagnostics.png")


if __name__ == "__main__":
    main()
