from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "matrix_multiplier_step4g_20260428"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "scalar_baseline": {
        "method": "Scalar matrix",
        "variant": "Stronger BC, Step 4B-4C",
        "style": "baseline",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260427_234906" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260427_234922" / "input_data.pkl",
    },
    "scalar_step4g": {
        "method": "Scalar matrix",
        "variant": "Step 4G",
        "style": "step4g",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260428_162645" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_multipliers" / "20260428_162700" / "input_data.pkl",
    },
    "structured_baseline": {
        "method": "Structured matrix",
        "variant": "Weighted BC, Step 4E",
        "style": "baseline",
        "rl_path": REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260428_025906" / "input_data.pkl",
        "compare_path": REPO_ROOT / "Polymer" / "Results" / "disturb_compare_td3_structured_matrices" / "20260428_025922" / "input_data.pkl",
    },
    "structured_step4g": {
        "method": "Structured matrix",
        "variant": "Step 4G",
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

DIAG_WINDOWS = {
    "11-20": (10, 20),
    "21-40": (20, 40),
    "101-200": (100, 200),
}

STYLE_COLORS = {
    "baseline": "tab:blue",
    "step4g": "tab:green",
}

PHASE_LABELS = {
    0: "disabled",
    1: "nominal",
    2: "protected",
    3: "ramp",
    4: "full",
}


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def get_executed_multiplier_log(bundle: dict) -> np.ndarray:
    if "executed_multiplier_log" in bundle:
        return np.asarray(bundle["executed_multiplier_log"], float)
    if "effective_multiplier_log" in bundle:
        return np.asarray(bundle["effective_multiplier_log"], float)
    return np.asarray(bundle["candidate_multiplier_log"], float)


def get_action_saturation_log(bundle: dict) -> np.ndarray:
    if "action_saturation_fraction_log" in bundle:
        return np.asarray(bundle["action_saturation_fraction_log"], float)
    return np.asarray(bundle["action_saturation_trace"], float)


def summarize_reward_windows(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        delta = np.asarray(spec["compare"]["avg_rewards_rl"], float) - np.asarray(spec["compare"]["avg_rewards_mpc"], float)
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


def summarize_diagnostics(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        time_in_sub = int(bundle["time_in_sub_episodes"])
        bc_weight = np.asarray(bundle["bc_weight_log"], float)
        policy_nominal = np.asarray(bundle["bc_policy_nominal_distance_log"], float)
        actor_target = np.asarray(bundle["bc_actor_target_distance_log"], float)
        sat = get_action_saturation_log(bundle)
        executed_multiplier = get_executed_multiplier_log(bundle)
        mult_distance = np.linalg.norm(executed_multiplier - 1.0, axis=1)
        for window_name, (start_ep, end_ep) in DIAG_WINDOWS.items():
            start_idx = start_ep * time_in_sub
            end_idx = end_ep * time_in_sub
            actor_slice = actor_target[start_idx:end_idx]
            actor_slice = actor_slice[np.isfinite(actor_slice) & (actor_slice > 0.0)]
            rows.append(
                {
                    "method": spec["method"],
                    "variant": spec["variant"],
                    "style": spec["style"],
                    "window": window_name,
                    "bc_weight_mean": safe_mean(bc_weight[start_idx:end_idx]),
                    "policy_nominal_distance_mean": safe_mean(policy_nominal[start_idx:end_idx]),
                    "actor_target_distance_mean": safe_mean(actor_slice),
                    "multiplier_distance_mean": safe_mean(mult_distance[start_idx:end_idx]),
                    "action_saturation_mean": safe_mean(sat[start_idx:min(end_idx, sat.shape[0])]),
                }
            )
    return pd.DataFrame(rows)


def summarize_release_guard(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        time_in_sub = int(bundle["time_in_sub_episodes"])
        clip = np.asarray(bundle.get("release_clip_fraction_log", np.zeros(0)), float)
        active = np.asarray(bundle.get("release_guard_active_log", np.zeros_like(clip)), int)
        phase = np.asarray(bundle.get("release_phase_log", np.zeros_like(active)), int)
        for window_name, (start_ep, end_ep) in DIAG_WINDOWS.items():
            start_idx = start_ep * time_in_sub
            end_idx = end_ep * time_in_sub
            phase_slice = phase[start_idx:end_idx]
            row = {
                "method": spec["method"],
                "variant": spec["variant"],
                "style": spec["style"],
                "window": window_name,
                "guard_active_fraction": safe_mean((active[start_idx:end_idx] > 0).astype(float)),
                "clip_fraction_mean": safe_mean(clip[start_idx:end_idx]),
                "clip_steps": int(np.sum(clip[start_idx:end_idx] > 0.0)),
            }
            for code, label in PHASE_LABELS.items():
                row[f"phase_{label}_fraction"] = safe_mean((phase_slice == code).astype(float))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_configs(run_data: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_data.values():
        bundle = spec["rl"]
        cfg = dict(bundle.get("config_snapshot", {}))
        release = dict(bundle.get("release_schedule", {}))
        rows.append(
            {
                "method": spec["method"],
                "variant": spec["variant"],
                "bc_enabled": bool(bundle.get("behavioral_cloning_enabled", False)),
                "release_guard_enabled": bool(bundle.get("release_guard_enabled", False)),
                "acceptance_fallback_enabled": bool(bundle.get("mpc_acceptance_fallback", {}).get("enabled", False)),
                "post_warm_start_action_freeze_subepisodes": int(cfg.get("post_warm_start_action_freeze_subepisodes", 0)),
                "post_warm_start_actor_freeze_subepisodes": int(cfg.get("post_warm_start_actor_freeze_subepisodes", 0)),
                "protected_live_subepisodes": int(
                    cfg.get("release_protected_advisory_caps", {}).get("protected_live_subepisodes", 0)
                ),
                "authority_ramp_subepisodes": int(
                    cfg.get("release_protected_advisory_caps", {}).get("authority_ramp_subepisodes", 0)
                ),
                "release_clip_steps": int(np.sum(np.asarray(bundle.get("release_clip_fraction_log", []), float) > 0.0)),
                "fallback_steps": int(np.sum(np.asarray(bundle.get("mpc_acceptance_fallback_active_log", []), float) > 0.5)),
                "release_schedule_enabled": bool(release.get("enabled", False)),
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
        for idx, style in enumerate(["baseline", "step4g"]):
            vals = [
                float(subset[(subset["style"] == style) & (subset["window"] == w)]["reward_delta_mean"].iloc[0])
                for w in window_order
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


def plot_diagnostics(diag_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    window_order = ["11-20", "21-40", "101-200"]
    metrics = [
        ("bc_weight_mean", "Mean BC weight"),
        ("policy_nominal_distance_mean", "Policy-nominal distance"),
        ("multiplier_distance_mean", "Multiplier distance"),
        ("action_saturation_mean", "Action saturation"),
    ]
    for row_idx, method in enumerate(["Scalar matrix", "Structured matrix"]):
        subset = diag_df[diag_df["method"] == method]
        for col_idx, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(window_order))
            width = 0.36
            for idx, style in enumerate(["baseline", "step4g"]):
                vals = [
                    float(subset[(subset["style"] == style) & (subset["window"] == w)][metric_key].iloc[0])
                    for w in window_order
                ]
                ax.bar(
                    x + (idx - 0.5) * width,
                    vals,
                    width=width,
                    color=STYLE_COLORS[style],
                    label=subset[subset["style"] == style]["variant"].iloc[0],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(window_order)
            ax.set_title(f"{method}: {metric_title}")
            ax.grid(True, axis="y", alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_release_guard(release_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    window_order = ["11-20", "21-40", "101-200"]
    metric_defs = [
        ("guard_active_fraction", "Guard active fraction"),
        ("clip_fraction_mean", "Mean clip fraction"),
        ("phase_ramp_fraction", "Ramp-phase fraction"),
    ]
    for row_idx, method in enumerate(["Scalar matrix", "Structured matrix"]):
        subset = release_df[release_df["method"] == method]
        for col_idx, (metric_key, metric_title) in enumerate(metric_defs):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(window_order))
            width = 0.36
            for idx, style in enumerate(["baseline", "step4g"]):
                vals = [
                    float(subset[(subset["style"] == style) & (subset["window"] == w)][metric_key].iloc[0])
                    for w in window_order
                ]
                ax.bar(
                    x + (idx - 0.5) * width,
                    vals,
                    width=width,
                    color=STYLE_COLORS[style],
                    label=subset[subset["style"] == style]["variant"].iloc[0],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(window_order)
            ax.set_ylim(0.0, 1.05)
            ax.set_title(f"{method}: {metric_title}")
            ax.grid(True, axis="y", alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
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
    release_df = summarize_release_guard(run_data)
    config_df = summarize_configs(run_data)

    reward_df.to_csv(OUTPUT_DIR / "polymer_step4g_reward_windows.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "polymer_step4g_handoff_diagnostics.csv", index=False)
    release_df.to_csv(OUTPUT_DIR / "polymer_step4g_release_guard_diagnostics.csv", index=False)
    config_df.to_csv(OUTPUT_DIR / "polymer_step4g_config_summary.csv", index=False)

    plot_reward_traces(run_data, OUTPUT_DIR / "polymer_step4g_reward_delta_traces.png")
    plot_reward_windows(reward_df, OUTPUT_DIR / "polymer_step4g_reward_windows.png")
    plot_diagnostics(diag_df, OUTPUT_DIR / "polymer_step4g_handoff_diagnostics.png")
    plot_release_guard(release_df, OUTPUT_DIR / "polymer_step4g_release_guard_diagnostics.png")


if __name__ == "__main__":
    main()
