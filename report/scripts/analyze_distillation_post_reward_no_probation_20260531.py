"""Analyze the May 30 distillation no-probation, high-temperature-reward batch.

This script reads saved bundles only. It reuses the metric definitions from the
previous five-runner analysis so comparisons stay consistent across batches.
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
OUT_DIR = ROOT / "report" / "figures" / "distillation_post_reward_no_probation_20260531"
BASE_SCRIPT = ROOT / "report" / "scripts" / "analyze_distillation_wider_safety_20260530.py"


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


def latest_runs():
    return {
        "baseline": {
            "label": "OF-MPC",
            "method": "baseline",
            "path": ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
        },
        "weights": {
            "label": "TD3 weights",
            "method": "weights",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
            / "20260530_220604"
            / "input_data.pkl",
        },
        "horizon": {
            "label": "Horizon DDQN",
            "method": "horizon",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_horizon_disturb_fluctuation_mismatch_unified"
            / "20260530_225843"
            / "input_data.pkl",
        },
        "dueling": {
            "label": "Dueling horizon",
            "method": "dueling_horizon",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
            / "20260530_223346"
            / "input_data.pkl",
        },
        "residual": {
            "label": "TD3 residual",
            "method": "residual",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
            / "20260530_231105"
            / "input_data.pkl",
        },
        "markov": {
            "label": "TD3 Markov",
            "method": "markov",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_markov_td3_disturb_fluctuation_unified"
            / "20260530_230956"
            / "input_data.pkl",
        },
    }


def previous_runs():
    return {
        "weights": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260529_211201"
        / "input_data.pkl",
        "horizon": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260529_213627"
        / "input_data.pkl",
        "dueling": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260529_213847"
        / "input_data.pkl",
        "residual": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260529_213534"
        / "input_data.pkl",
        "markov": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260529_220507"
        / "input_data.pkl",
    }


def baseline_compare_path():
    return (
        ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_residual_td3_disturb_fluctuation"
        / "20260530_231118"
        / "input_data.pkl"
    )


def markov_reference_runs():
    return {
        "20260518 TD3-only success": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified"
        / "20260518_091937"
        / "input_data.pkl",
        "20260518 priority success": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260518_184548"
        / "input_data.pkl",
        "20260529 restored safety": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260529_220507"
        / "input_data.pkl",
    }


def _baseline_delta_rows(current_summary: pd.DataFrame) -> pd.DataFrame:
    current_idx = current_summary.set_index("key")
    baseline = current_idx.loc["baseline"]
    rows = []
    for key in ["weights", "horizon", "dueling", "residual", "markov"]:
        row = current_idx.loc[key]
        rows.append(
            {
                "key": key,
                "label": row["label"],
                "tail20_reward_delta_vs_baseline": float(row["reward_tail20"] - baseline["reward_tail20"]),
                "final_reward_delta_vs_baseline": float(row["reward_final"] - baseline["reward_final"]),
                "first20_min_delta_vs_baseline": float(row["reward_first20_min"] - baseline["reward_first20_min"]),
                "band_norm_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_band_norm_mae"] / baseline["tail20_band_norm_mae"] - 1.0)
                ),
                "outside_band_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_outside_band_frac"] / baseline["tail20_outside_band_frac"] - 1.0)
                ),
                "composition_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_comp_mae"] / baseline["tail20_comp_mae"] - 1.0)
                ),
                "temperature_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_temp_mae"] / baseline["tail20_temp_mae"] - 1.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _previous_delta_rows(current_summary: pd.DataFrame, previous_summary: pd.DataFrame) -> pd.DataFrame:
    current_idx = current_summary.set_index("key")
    previous_idx = previous_summary.set_index("key")
    rows = []
    for key in ["weights", "horizon", "dueling", "residual", "markov"]:
        cur = current_idx.loc[key]
        prev = previous_idx.loc[key]
        rows.append(
            {
                "key": key,
                "label": cur["label"],
                "current_tail20": float(cur["reward_tail20"]),
                "previous_tail20": float(prev["reward_tail20"]),
                "tail20_delta": float(cur["reward_tail20"] - prev["reward_tail20"]),
                "current_final": float(cur["reward_final"]),
                "previous_final": float(prev["reward_final"]),
                "final_delta": float(cur["reward_final"] - prev["reward_final"]),
                "current_first20_min": float(cur["reward_first20_min"]),
                "previous_first20_min": float(prev["reward_first20_min"]),
                "first20_min_delta": float(cur["reward_first20_min"] - prev["reward_first20_min"]),
                "band_norm_mae_delta": float(cur["tail20_band_norm_mae"] - prev["tail20_band_norm_mae"]),
                "outside_band_frac_delta": float(cur["tail20_outside_band_frac"] - prev["tail20_outside_band_frac"]),
                "composition_mae_delta": float(cur["tail20_comp_mae"] - prev["tail20_comp_mae"]),
                "temperature_mae_delta": float(cur["tail20_temp_mae"] - prev["tail20_temp_mae"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_current_vs_previous(compare: pd.DataFrame) -> None:
    methods = ["weights", "horizon", "dueling", "residual", "markov"]
    labels = ["Weights", "Horizon", "Dueling", "Residual", "Markov"]
    df = compare.set_index("key").loc[methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    width = 0.34
    ax.bar(x - width / 2, df["previous_tail20"], width, label="May 29 previous settings", color="#bab0ac")
    ax.bar(x + width / 2, df["current_tail20"], width, label="May 30 no-probation high-temp reward", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tail-20 average reward")
    ax.set_title("Effect of removing reward probation and increasing temperature weight")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_current_vs_previous_tail20.png", dpi=180)
    plt.close(fig)


def _subepisode_rates(values: np.ndarray, episode_len: int) -> np.ndarray:
    values = np.asarray(values, float)
    n = values.size // episode_len
    if n <= 0:
        return np.asarray([], float)
    return values[: n * episode_len].reshape(n, episode_len).mean(axis=1)


def _plot_residual_release_zoom(current: dict[str, dict]) -> None:
    residual = current["residual"]
    baseline = current["baseline"]
    ep_len = int(residual.get("time_in_sub_episodes", 400))
    warm_steps = int(residual.get("warm_start_step", 4000))
    warm_episodes = max(1, warm_steps // ep_len)
    n_show = 60

    res_rewards = np.asarray(residual["avg_rewards"], float)
    base_rewards = np.asarray(baseline["avg_rewards"], float)
    cap = _subepisode_rates(np.asarray(residual.get("residual_cap_projection_active_log", []), float), ep_len)
    source = np.asarray(residual.get("residual_action_source_log", []), int)
    codes = residual.get("residual_action_source_codes", {})
    zero_code = int(codes.get("warm_zero", -999))
    fallback_code = int(codes.get("zero_fallback", -998))
    warm_zero = _subepisode_rates(source == zero_code, ep_len)
    zero_fallback = _subepisode_rates(source == fallback_code, ep_len)

    x = np.arange(1, min(n_show, len(res_rewards), len(base_rewards)) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    axes[0].plot(x, base_rewards[: len(x)], label="OF-MPC", color="black", linewidth=1.6)
    axes[0].plot(x, res_rewards[: len(x)], label="TD3 residual", color="#e45756", linewidth=1.8)
    axes[0].axvline(warm_episodes, color="#777777", linestyle="--", linewidth=1.0, label="warm-start end")
    axes[0].set_ylabel("Average reward")
    axes[0].set_title("Residual early release: reward zoom")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    x_safety = np.arange(1, min(n_show, len(cap)) + 1)
    axes[1].plot(x_safety, cap[: len(x_safety)], label="Residual cap projection", color="#4c78a8")
    axes[1].plot(x_safety, warm_zero[: len(x_safety)], label="Warm zero source", color="#f58518")
    axes[1].plot(x_safety, zero_fallback[: len(x_safety)], label="Zero fallback source", color="#54a24b")
    axes[1].axvline(warm_episodes, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("Subepisode")
    axes[1].set_ylabel("Fraction of steps")
    axes[1].set_title("Residual safety activity during release")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_release_zoom.png", dpi=180)
    plt.close(fig)


def _tail_steps(bundle: dict, episodes: int = 20) -> slice:
    ep_len = int(bundle.get("time_in_sub_episodes", 400))
    nfe = int(bundle.get("nFE", len(bundle.get("y_sp", []))))
    return slice(max(0, nfe - episodes * ep_len), nfe)


def _plot_markov_z_mechanism(current: dict[str, dict]) -> None:
    current_markov = current["markov"]
    reference = {name: load_pickle(path) for name, path in markov_reference_runs().items()}
    runs = {
        "20260518 TD3-only success": reference["20260518 TD3-only success"],
        "20260518 priority success": reference["20260518 priority success"],
        "20260529 restored safety": reference["20260529 restored safety"],
        "20260530 current": current_markov,
    }

    rows = []
    for name, bundle in runs.items():
        z = np.asarray(bundle.get("z_executed_log", []), float)
        sl = _tail_steps(bundle, 20)
        if z.ndim != 2 or z.size == 0:
            continue
        z_tail = z[sl, :]
        rows.append(
            {
                "run": name,
                "tail20_reward": float(np.mean(np.asarray(bundle["avg_rewards"], float)[-20:])),
                "z_norm_mean": float(np.mean(np.linalg.norm(z_tail, axis=1))),
                "abs_z_q95": float(np.quantile(np.abs(z_tail).reshape(-1), 0.95)),
                "z_direction_std": float(np.mean(np.std(z_tail, axis=0))),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "markov_reference_z_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    axes = axes.reshape(-1)
    colors = ["#4c78a8", "#54a24b", "#f58518", "#b279a2"]
    labels = ["z1", "z2", "z3", "z4"]
    for ax, (name, bundle), color in zip(axes[:3], list(runs.items())[:3], colors[:3]):
        z = np.asarray(bundle["z_executed_log"], float)
        sl = _tail_steps(bundle, 20)
        z_tail = z[sl, :]
        n = min(600, z_tail.shape[0])
        for j in range(z_tail.shape[1]):
            ax.plot(z_tail[:n, j], linewidth=1.0, label=labels[j])
        ax.axhline(0.04, color="#999999", linestyle=":", linewidth=0.8)
        ax.axhline(-0.04, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_title(f"{name}\nTail reward {np.mean(np.asarray(bundle['avg_rewards'], float)[-20:]):.2f}")
        ax.set_ylabel("Executed z")
        ax.grid(True, alpha=0.25)
    z_current = np.asarray(current_markov["z_executed_log"], float)
    sl = _tail_steps(current_markov, 20)
    z_tail = z_current[sl, :]
    n = min(600, z_tail.shape[0])
    for j in range(z_tail.shape[1]):
        axes[3].plot(z_tail[:n, j], linewidth=1.0, label=labels[j])
    before = np.asarray(current_markov.get("z_safety_requested_norm_before_log", []), float)[sl]
    after = np.asarray(current_markov.get("z_safety_requested_norm_after_log", []), float)[sl]
    axes[3].plot(before[:n], color="black", linewidth=1.3, linestyle="--", label="requested norm")
    axes[3].plot(after[:n], color="#e45756", linewidth=1.3, linestyle="--", label="executed norm")
    axes[3].axhline(0.06, color="#e45756", linestyle=":", linewidth=0.8)
    axes[3].axhline(0.04, color="#999999", linestyle=":", linewidth=0.8)
    axes[3].axhline(-0.04, color="#999999", linestyle=":", linewidth=0.8)
    axes[3].set_title(
        f"20260530 current\nTail reward {np.mean(np.asarray(current_markov['avg_rewards'], float)[-20:]):.2f}"
    )
    axes[3].set_ylabel("Executed z / norm")
    axes[3].grid(True, alpha=0.25)
    for ax in axes:
        ax.set_xlabel("Tail step index")
        ax.legend(ncol=2, fontsize=7)
    fig.suptitle("Markov z mechanism: successful runs used varied directions, current run saturates one corner")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_markov_z_mechanism.png", dpi=180)
    plt.close(fig)


def main() -> None:
    base = load_base_module()
    base.OUT_DIR = OUT_DIR
    base.CURRENT_RUNS = latest_runs()
    base.PREVIOUS_RUNS = previous_runs()
    base.BASELINE_COMPARE_PATH = baseline_compare_path()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current = {key: load_pickle(meta["path"]) for key, meta in base.CURRENT_RUNS.items()}
    baseline_compare = load_pickle(base.BASELINE_COMPARE_PATH)
    current["baseline"] = dict(current["baseline"])
    current["baseline"]["avg_rewards"] = np.asarray(baseline_compare["avg_rewards_mpc"], float)
    previous = {key: load_pickle(path) for key, path in base.PREVIOUS_RUNS.items()}

    current_rows = [
        base.row_for(key, meta, current[key], "2026-05-30 no-probation high-temperature reward")
        for key, meta in base.CURRENT_RUNS.items()
    ]
    current_summary = pd.DataFrame(current_rows)
    current_summary.to_csv(OUT_DIR / "current_summary_metrics.csv", index=False)

    previous_rows = []
    for key, path in base.PREVIOUS_RUNS.items():
        meta = dict(base.CURRENT_RUNS[key])
        meta["path"] = path
        previous_rows.append(base.row_for(key, meta, previous[key], "2026-05-29 no-probation baseline"))
    previous_summary = pd.DataFrame(previous_rows)
    previous_summary.to_csv(OUT_DIR / "previous_summary_metrics.csv", index=False)

    baseline_compare_df = _baseline_delta_rows(current_summary)
    baseline_compare_df.to_csv(OUT_DIR / "current_vs_baseline_metrics.csv", index=False)

    current_vs_previous = _previous_delta_rows(current_summary, previous_summary)
    current_vs_previous.to_csv(OUT_DIR / "current_vs_previous_metrics.csv", index=False)

    base.plot_reward(current, current_summary)
    _plot_current_vs_previous(current_vs_previous)
    base.plot_tracking(current_summary)
    base.plot_safety(current_summary)
    horizon_counts = base.plot_horizon_usage(current)
    base.plot_weight_residual_markov(current_summary, current)
    _plot_residual_release_zoom(current)
    _plot_markov_z_mechanism(current)

    summary = {
        "current_rank_tail20": current_summary.sort_values("reward_tail20", ascending=False)[
            [
                "label",
                "reward_tail20",
                "reward_tail10",
                "reward_final",
                "reward_first20_min",
                "tail20_band_norm_mae",
                "tail20_outside_band_frac",
            ]
        ].to_dict(orient="records"),
        "current_vs_baseline": baseline_compare_df.to_dict(orient="records"),
        "current_vs_previous": current_vs_previous.to_dict(orient="records"),
        "safety_highlights": current_summary[
            [
                "key",
                "label",
                "tail20_weight_cap_projection_frac",
                "tail20_horizon_projection_frac",
                "tail20_residual_cap_projection_frac",
                "tail20_fallback_frac",
                "weight_probation_trigger_count",
                "horizon_probation_trigger_count",
                "residual_probation_trigger_count",
                "markov_probation_trigger_count",
            ]
        ].to_dict(orient="records"),
        "top_horizon_pairs": horizon_counts.groupby("runner").head(5).to_dict(orient="records"),
        "artifacts": {
            "figure_dir": str(OUT_DIR.relative_to(ROOT)),
            "csvs": sorted(p.name for p in OUT_DIR.glob("*.csv")),
            "figures": sorted(p.name for p in OUT_DIR.glob("fig_*.png")),
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
