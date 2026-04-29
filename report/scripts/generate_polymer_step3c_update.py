from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "matrix_multiplier_step3c_20260428"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "scalar_step3c": {
        "method": "Scalar matrix",
        "variant": "Step 3C shadow study",
        "style": "step3c",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb_step3c_shadow" / "20260428_191043" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers_step3c_shadow" / "20260428_191057" / "input_data.pkl",
    },
    "scalar_step4g": {
        "method": "Scalar matrix",
        "variant": "Step 4G default",
        "style": "step4g",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260428_162645" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260428_162700" / "input_data.pkl",
    },
    "structured_step3c": {
        "method": "Structured matrix",
        "variant": "Step 3C shadow study",
        "style": "step3c",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb_step3c_shadow" / "20260428_191143" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices_step3c_shadow" / "20260428_191201" / "input_data.pkl",
    },
    "structured_step4g": {
        "method": "Structured matrix",
        "variant": "Step 4G default",
        "style": "step4g",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260428_162948" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260428_163005" / "input_data.pkl",
    },
}

WINDOWS = {
    "11-20": (10, 20),
    "21-40": (20, 40),
    "41-100": (40, 100),
    "101-200": (100, 200),
    "1-200": (0, 200),
}

SHADOW_WINDOWS = {
    "11-20": (10, 20),
    "21-40": (20, 40),
    "41-100": (40, 100),
    "101-200": (100, 200),
}

STYLE_COLORS = {
    "step3c": "tab:orange",
    "step4g": "tab:green",
}


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def compare_reward_delta(compare_bundle: dict) -> np.ndarray:
    return np.asarray(compare_bundle["avg_rewards_rl"], float) - np.asarray(compare_bundle["avg_rewards_mpc"], float)


def summarize_reward_windows(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        delta = compare_reward_delta(spec["compare"])
        for window_name, (start, end) in WINDOWS.items():
            rows.append(
                {
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "style": spec["style"],
                    "window": window_name,
                    "reward_delta_mean": float(np.mean(delta[start:end])),
                    "win_rate": float(np.mean(delta[start:end] > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_config_checks(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        cfg = dict(bundle.get("config_snapshot", {}))
        rows.append(
            {
                "method": spec["method"],
                "variant": spec["variant"],
                "behavioral_cloning_enabled": bool(bundle.get("behavioral_cloning_enabled", False)),
                "release_guard_enabled": bool(bundle.get("release_guard_enabled", False)),
                "mpc_acceptance_enabled": bool(bundle.get("mpc_acceptance_enabled", False)),
                "mpc_dual_cost_shadow_enabled": bool(bundle.get("mpc_dual_cost_shadow_enabled", False)),
                "action_freeze_subepisodes": int(cfg.get("post_warm_start_action_freeze_subepisodes", 0)),
                "actor_freeze_subepisodes": int(cfg.get("post_warm_start_actor_freeze_subepisodes", 0)),
                "protected_live_subepisodes": int(cfg.get("release_protected_advisory_caps", {}).get("protected_live_subepisodes", 0)),
                "authority_ramp_subepisodes": int(cfg.get("release_protected_advisory_caps", {}).get("authority_ramp_subepisodes", 0)),
                "fallback_steps": int(np.sum(np.asarray(bundle.get("mpc_acceptance_fallback_active_log", []), float) > 0.5)),
                "candidate_solve_failure_steps": int(np.sum(np.asarray(bundle.get("mpc_dual_cost_shadow_reason_code_log", []), int) == 2)),
            }
        )
    return pd.DataFrame(rows)


def summarize_shadow_windows(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        if spec["style"] != "step3c":
            continue
        bundle = spec["rl"]
        steps_per_episode = int(bundle["time_in_sub_episodes"])
        nominal_penalty = np.asarray(bundle["mpc_dual_cost_shadow_nominal_penalty_log"], float)
        safe_threshold = np.asarray(bundle["mpc_dual_cost_shadow_safe_threshold_log"], float)
        candidate_advantage = np.asarray(bundle["mpc_dual_cost_shadow_candidate_advantage_log"], float)
        safe_pass = np.asarray(bundle["mpc_dual_cost_shadow_safe_pass_log"], float)
        benefit_pass = np.asarray(bundle["mpc_dual_cost_shadow_benefit_pass_log"], float)
        dual_pass = np.asarray(bundle["mpc_dual_cost_shadow_dual_pass_log"], float)
        clip_fraction = np.asarray(bundle["release_clip_fraction_log"], float)
        reward_delta = compare_reward_delta(spec["compare"])
        for window_name, (start_ep, end_ep) in SHADOW_WINDOWS.items():
            start_idx = start_ep * steps_per_episode
            end_idx = end_ep * steps_per_episode
            reward_slice = reward_delta[start_ep:end_ep]
            rows.append(
                {
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "window": window_name,
                    "reward_delta_mean": float(np.mean(reward_slice)),
                    "nominal_penalty_mean": safe_mean(nominal_penalty[start_idx:end_idx]),
                    "safe_threshold_mean": safe_mean(safe_threshold[start_idx:end_idx]),
                    "candidate_advantage_mean": safe_mean(candidate_advantage[start_idx:end_idx]),
                    "safe_pass_rate": safe_mean(safe_pass[start_idx:end_idx]),
                    "benefit_pass_rate": safe_mean(benefit_pass[start_idx:end_idx]),
                    "dual_pass_rate": safe_mean(dual_pass[start_idx:end_idx]),
                    "clip_fraction_mean": safe_mean(clip_fraction[start_idx:end_idx]),
                }
            )
    return pd.DataFrame(rows)


def summarize_episode_correlations(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        if spec["style"] != "step3c":
            continue
        bundle = spec["rl"]
        steps_per_episode = int(bundle["time_in_sub_episodes"])
        reward_delta = compare_reward_delta(spec["compare"])
        safe_pass = np.asarray(bundle["mpc_dual_cost_shadow_safe_pass_log"], float)
        benefit_pass = np.asarray(bundle["mpc_dual_cost_shadow_benefit_pass_log"], float)
        dual_pass = np.asarray(bundle["mpc_dual_cost_shadow_dual_pass_log"], float)
        nominal_penalty = np.asarray(bundle["mpc_dual_cost_shadow_nominal_penalty_log"], float)
        candidate_advantage = np.asarray(bundle["mpc_dual_cost_shadow_candidate_advantage_log"], float)

        ep_safe = np.array([safe_mean(safe_pass[i * steps_per_episode : (i + 1) * steps_per_episode]) for i in range(reward_delta.size)])
        ep_benefit = np.array([safe_mean(benefit_pass[i * steps_per_episode : (i + 1) * steps_per_episode]) for i in range(reward_delta.size)])
        ep_dual = np.array([safe_mean(dual_pass[i * steps_per_episode : (i + 1) * steps_per_episode]) for i in range(reward_delta.size)])
        ep_penalty = np.array([safe_mean(nominal_penalty[i * steps_per_episode : (i + 1) * steps_per_episode]) for i in range(reward_delta.size)])
        ep_advantage = np.array([safe_mean(candidate_advantage[i * steps_per_episode : (i + 1) * steps_per_episode]) for i in range(reward_delta.size)])

        rows.append(
            {
                "method": spec["method"],
                "variant": spec["variant"],
                "reward_vs_safe_pass_corr": float(np.corrcoef(reward_delta, ep_safe)[0, 1]),
                "reward_vs_benefit_pass_corr": float(np.corrcoef(reward_delta, ep_benefit)[0, 1]),
                "reward_vs_dual_pass_corr": float(np.corrcoef(reward_delta, ep_dual)[0, 1]),
                "reward_vs_nominal_penalty_corr": float(np.corrcoef(reward_delta, ep_penalty)[0, 1]),
                "reward_vs_candidate_advantage_corr": float(np.corrcoef(reward_delta, ep_advantage)[0, 1]),
                "safe_pass_full_mean": float(np.mean(ep_safe)),
                "benefit_pass_full_mean": float(np.mean(ep_benefit)),
                "dual_pass_full_mean": float(np.mean(ep_dual)),
            }
        )
    return pd.DataFrame(rows)


def plot_reward_traces(run_data: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for row_idx, method in enumerate(["Scalar matrix", "Structured matrix"]):
        specs = [spec for spec in run_data.values() if spec["method"] == method]
        for col_idx, x_limit in enumerate([60, 200]):
            ax = axes[row_idx, col_idx]
            for spec in specs:
                delta = compare_reward_delta(spec["compare"])
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
    for ax, method in zip(axes, ["Scalar matrix", "Structured matrix"], strict=True):
        subset = reward_df[reward_df["method"] == method]
        x = np.arange(len(window_order))
        width = 0.36
        for idx, style in enumerate(["step3c", "step4g"]):
            vals = [
                float(subset[(subset["style"] == style) & (subset["window"] == window)]["reward_delta_mean"].iloc[0])
                for window in window_order
            ]
            label = subset[subset["style"] == style]["variant"].iloc[0]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, color=STYLE_COLORS[style], label=label)
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(window_order)
        ax.set_title(method)
        ax.set_ylabel("Mean reward delta vs MPC")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shadow_windows(shadow_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    window_order = ["11-20", "21-40", "41-100", "101-200"]

    for ax, method in zip(axes[:, 0], ["Scalar matrix", "Structured matrix"], strict=True):
        subset = shadow_df[shadow_df["method"] == method]
        x = np.arange(len(window_order))
        width = 0.25
        metric_map = [
            ("safe_pass_rate", "Safe pass", "tab:blue"),
            ("benefit_pass_rate", "Benefit pass", "tab:orange"),
            ("dual_pass_rate", "Dual pass", "tab:green"),
        ]
        for idx, (metric_key, label, color) in enumerate(metric_map):
            vals = [float(subset[subset["window"] == window][metric_key].iloc[0]) for window in window_order]
            ax.bar(x + (idx - 1) * width, vals, width=width, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(window_order)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{method}: shadow pass rates")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    for ax, method in zip(axes[:, 1], ["Scalar matrix", "Structured matrix"], strict=True):
        subset = shadow_df[shadow_df["method"] == method]
        x = np.arange(len(window_order))
        penalty = [float(subset[subset["window"] == window]["nominal_penalty_mean"].iloc[0]) for window in window_order]
        threshold = [float(subset[subset["window"] == window]["safe_threshold_mean"].iloc[0]) for window in window_order]
        advantage = [float(subset[subset["window"] == window]["candidate_advantage_mean"].iloc[0]) for window in window_order]
        ax.plot(x, penalty, marker="o", linewidth=2.0, color="tab:red", label="Nominal penalty")
        ax.plot(x, threshold, marker="s", linewidth=2.0, color="tab:purple", label="Safe threshold")
        ax.plot(x, advantage, marker="^", linewidth=2.0, color="tab:cyan", label="Candidate advantage")
        ax.set_xticks(x)
        ax.set_xticklabels(window_order)
        ax.set_title(f"{method}: shadow cost terms")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reward_vs_safe_corr(run_data: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, method in zip(axes, ["Scalar matrix", "Structured matrix"], strict=True):
        spec = next(spec for spec in run_data.values() if spec["method"] == method and spec["style"] == "step3c")
        bundle = spec["rl"]
        reward_delta = compare_reward_delta(spec["compare"])
        steps_per_episode = int(bundle["time_in_sub_episodes"])
        safe_pass = np.asarray(bundle["mpc_dual_cost_shadow_safe_pass_log"], float)
        ep_safe = np.array([safe_mean(safe_pass[i * steps_per_episode : (i + 1) * steps_per_episode]) for i in range(reward_delta.size)])
        coeffs = np.polyfit(ep_safe, reward_delta, deg=1)
        xfit = np.linspace(ep_safe.min(), ep_safe.max(), 100)
        yfit = coeffs[0] * xfit + coeffs[1]
        corr = float(np.corrcoef(reward_delta, ep_safe)[0, 1])
        ax.scatter(ep_safe, reward_delta, s=22, alpha=0.65, color=STYLE_COLORS["step3c"])
        ax.plot(xfit, yfit, color="0.2", linewidth=2.0, label=f"corr = {corr:+.3f}")
        ax.axhline(0.0, color="0.5", linewidth=1.0)
        ax.set_title(f"{method}: reward delta vs safe-pass fraction")
        ax.set_xlabel("Episode mean safe-pass fraction")
        ax.set_ylabel("Episode reward delta vs MPC")
        ax.grid(True, alpha=0.25)
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
    config_df = summarize_config_checks(run_data)
    shadow_df = summarize_shadow_windows(run_data)
    corr_df = summarize_episode_correlations(run_data)

    reward_df.to_csv(OUTPUT_DIR / "polymer_step3c_reward_windows.csv", index=False)
    config_df.to_csv(OUTPUT_DIR / "polymer_step3c_config_summary.csv", index=False)
    shadow_df.to_csv(OUTPUT_DIR / "polymer_step3c_shadow_window_diagnostics.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "polymer_step3c_episode_correlations.csv", index=False)

    plot_reward_traces(run_data, OUTPUT_DIR / "polymer_step3c_reward_delta_traces.png")
    plot_reward_windows(reward_df, OUTPUT_DIR / "polymer_step3c_reward_windows.png")
    plot_shadow_windows(shadow_df, OUTPUT_DIR / "polymer_step3c_shadow_window_diagnostics.png")
    plot_reward_vs_safe_corr(run_data, OUTPUT_DIR / "polymer_step3c_reward_vs_safe_pass.png")


if __name__ == "__main__":
    main()
