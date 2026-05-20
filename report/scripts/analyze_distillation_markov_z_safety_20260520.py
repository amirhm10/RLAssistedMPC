from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation.notebook_params import get_distillation_notebook_defaults


RESULTS_ROOT = REPO_ROOT / "Distillation" / "Results"
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_markov_z_safety_20260520"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "Guarded Markov latest": RESULTS_ROOT
    / "distillation_markov_td3_disturb_fluctuation_unified/20260519_210736/input_data.pkl",
    "Guarded Markov best recent": RESULTS_ROOT
    / "distillation_markov_td3_disturb_fluctuation_unified/20260518_184548/input_data.pkl",
    "TD3-only no-safeguard best": RESULTS_ROOT
    / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified/20260518_091937/input_data.pkl",
}


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict in {path}, found {type(bundle).__name__}")
    return bundle


def arr(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def finite_tail(values: Any, n: int = 20) -> float:
    values_arr = arr(values).reshape(-1)
    values_arr = values_arr[np.isfinite(values_arr)]
    if values_arr.size == 0:
        return float("nan")
    return float(np.mean(values_arr[-min(n, values_arr.size) :]))


def summarize_run(label: str, path: Path) -> dict[str, Any]:
    bundle = load_pickle(path)
    z = arr(bundle.get("z_executed_log"))
    rewards = arr(bundle.get("rewards_step"))
    avg_rewards = arr(bundle.get("avg_rewards"))
    accepted = arr(bundle.get("accepted_log"))
    fallback = arr(bundle.get("fallback_log"))
    action_source = arr(bundle.get("rl_action_source_log"))
    gain_drift = arr(bundle.get("executed_gain_drift_log"))
    cost_margin = arr(bundle.get("executed_cost_margin_log"))
    z_bound = float(bundle.get("markov_z_bound", get_distillation_notebook_defaults("markov")["controller"]["z_bound"]))
    warm_start = int(bundle.get("warm_start_step", 0))
    post = slice(min(warm_start, len(z)), None)
    z_post = z[post]
    rewards_post = rewards[post] if rewards.size else np.asarray([], dtype=float)
    dz = np.diff(z_post, axis=0) if z_post.ndim == 2 and z_post.shape[0] > 1 else np.asarray([], dtype=float)
    labels = bundle.get("basis_labels") or [f"z{i + 1}" for i in range(z.shape[1] if z.ndim == 2 else 0)]
    z_abs = np.abs(z_post)
    z_norm = np.linalg.norm(z_post, axis=1) if z_post.ndim == 2 and z_post.size else np.asarray([], dtype=float)
    z_inf = np.max(z_abs, axis=1) if z_abs.ndim == 2 and z_abs.size else np.asarray([], dtype=float)

    row: dict[str, Any] = {
        "label": label,
        "run_dir": path.parent.name,
        "path": path.as_posix(),
        "z_bound": z_bound,
        "basis_labels": labels,
        "tail20_avg_reward": finite_tail(avg_rewards),
        "final_avg_reward": float(avg_rewards[-1]) if avg_rewards.size else float("nan"),
        "mean_reward": float(np.nanmean(rewards)) if rewards.size else float("nan"),
        "negative_avg_episode_fraction": float(np.mean(avg_rewards < 0.0)) if avg_rewards.size else float("nan"),
        "negative_step_fraction_post_warm": float(np.mean(rewards_post < 0.0)) if rewards_post.size else float("nan"),
        "accepted_fraction": float(np.nanmean(accepted)) if accepted.size else float("nan"),
        "fallback_fraction": float(np.nanmean(fallback)) if fallback.size else float("nan"),
        "td3_accepted_fraction": float(np.nanmean(action_source == 2)) if action_source.size else float("nan"),
        "nominal_fallback_fraction": float(np.nanmean(action_source == 4)) if action_source.size else float("nan"),
        "mean_gain_drift": float(np.nanmean(gain_drift)) if gain_drift.size else float("nan"),
        "max_gain_drift": float(np.nanmax(gain_drift)) if gain_drift.size else float("nan"),
        "mean_cost_margin": float(np.nanmean(cost_margin)) if cost_margin.size else float("nan"),
        "max_cost_margin": float(np.nanmax(cost_margin)) if cost_margin.size else float("nan"),
        "z_abs_q50": float(np.nanquantile(z_abs.reshape(-1), 0.50)) if z_abs.size else float("nan"),
        "z_abs_q75": float(np.nanquantile(z_abs.reshape(-1), 0.75)) if z_abs.size else float("nan"),
        "z_abs_q90": float(np.nanquantile(z_abs.reshape(-1), 0.90)) if z_abs.size else float("nan"),
        "z_abs_q95": float(np.nanquantile(z_abs.reshape(-1), 0.95)) if z_abs.size else float("nan"),
        "z_abs_q99": float(np.nanquantile(z_abs.reshape(-1), 0.99)) if z_abs.size else float("nan"),
        "z_inf_q95": float(np.nanquantile(z_inf, 0.95)) if z_inf.size else float("nan"),
        "z_l2_q95": float(np.nanquantile(z_norm, 0.95)) if z_norm.size else float("nan"),
        "dz_abs_q95": float(np.nanquantile(np.abs(dz).reshape(-1), 0.95)) if dz.size else float("nan"),
        "dz_abs_q99": float(np.nanquantile(np.abs(dz).reshape(-1), 0.99)) if dz.size else float("nan"),
    }
    for threshold in [0.02, 0.03, 0.035, 0.04, 0.0475]:
        row[f"step_fraction_any_coord_abs_gt_{threshold:g}"] = (
            float(np.mean(np.any(z_abs > threshold, axis=1))) if z_abs.ndim == 2 and z_abs.size else float("nan")
        )
        row[f"coord_fraction_abs_gt_{threshold:g}"] = float(np.mean(z_abs > threshold)) if z_abs.size else float("nan")
    for threshold in [0.03, 0.05, 0.07, 0.09]:
        mask = z_norm >= threshold if z_norm.size else np.asarray([], dtype=bool)
        row[f"reward_mean_when_l2_ge_{threshold:g}"] = (
            float(np.mean(rewards_post[mask])) if rewards_post.size and np.any(mask) else float("nan")
        )
        row[f"negative_step_fraction_when_l2_ge_{threshold:g}"] = (
            float(np.mean(rewards_post[mask] < 0.0)) if rewards_post.size and np.any(mask) else float("nan")
        )
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_coordinates(bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run_label, bundle in bundles.items():
        z = arr(bundle.get("z_executed_log"))
        if z.ndim != 2 or not z.size:
            continue
        warm = int(bundle.get("warm_start_step", 0))
        z_post = z[min(warm, len(z)) :]
        labels = bundle.get("basis_labels") or [f"z{i + 1}" for i in range(z.shape[1])]
        for idx, coord_label in enumerate(labels):
            coord = z_post[:, idx]
            rows.append(
                {
                    "run_label": run_label,
                    "coordinate": coord_label,
                    "min": float(np.nanmin(coord)),
                    "q05": float(np.nanquantile(coord, 0.05)),
                    "q50": float(np.nanquantile(coord, 0.50)),
                    "q95": float(np.nanquantile(coord, 0.95)),
                    "max": float(np.nanmax(coord)),
                    "abs_q90": float(np.nanquantile(np.abs(coord), 0.90)),
                    "abs_q95": float(np.nanquantile(np.abs(coord), 0.95)),
                    "abs_q99": float(np.nanquantile(np.abs(coord), 0.99)),
                    "fraction_abs_gt_0.04": float(np.mean(np.abs(coord) > 0.04)),
                    "fraction_abs_gt_0.0475": float(np.mean(np.abs(coord) > 0.0475)),
                }
            )
    return rows


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_runs() -> dict[str, dict[str, Any]]:
    return {label: load_pickle(path) for label, path in RUNS.items()}


def make_reward_trace_figure(bundles: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    colors = {
        "Guarded Markov latest": "#f58518",
        "Guarded Markov best recent": "#54a24b",
        "TD3-only no-safeguard best": "#4c78a8",
    }
    for label, bundle in bundles.items():
        avg = arr(bundle.get("avg_rewards"))
        ax.plot(avg, linewidth=2.2, color=colors.get(label), label=f"{label} ({bundle.get('summary_metrics', {}).get('reward_final_episode', np.nan):.2f} final)")
        neg_idx = np.where(avg < 0.0)[0]
        if neg_idx.size:
            ax.scatter(neg_idx, avg[neg_idx], s=18, color=colors.get(label), edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward")
    ax.set_title("Markov reward traces and negative-reward episodes")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_markov_reward_traces.png", dpi=190)
    plt.close(fig)


def make_z_quantile_figure(rows: list[dict[str, Any]]) -> None:
    labels = ["Guarded\nlatest", "Guarded\nbest recent", "TD3-only\nno safeguard"]
    x = np.arange(len(rows))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    for offset, key, name, color in [
        (-1.5 * width, "z_abs_q75", "q75 |z_i|", "#72b7b2"),
        (-0.5 * width, "z_abs_q90", "q90 |z_i|", "#f58518"),
        (0.5 * width, "z_abs_q95", "q95 |z_i|", "#e45756"),
        (1.5 * width, "z_abs_q99", "q99 |z_i|", "#4c78a8"),
    ]:
        ax.bar(x + offset, [row[key] for row in rows], width, label=name, color=color)
    z_bound = float(rows[0]["z_bound"])
    ax.axhline(z_bound, color="black", linestyle="--", linewidth=1.2, label=f"current bound = {z_bound:g}")
    ax.axhline(0.04, color="#8c6d31", linestyle=":", linewidth=1.4, label="candidate cap = 0.04")
    ax.axhline(0.035, color="#9467bd", linestyle=":", linewidth=1.4, label="candidate cap = 0.035")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute z coordinate")
    ax.set_title("How much of the z range is actually used?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.savefig(OUT_DIR / "fig_z_abs_quantiles.png", dpi=190)
    plt.close(fig)


def make_coordinate_range_figure(coord_rows: list[dict[str, Any]]) -> None:
    run_order = list(RUNS)
    coord_order = ["y1_u1", "y1_u2", "y2_u1", "y2_u2"]
    colors = {
        "Guarded Markov latest": "#f58518",
        "Guarded Markov best recent": "#54a24b",
        "TD3-only no-safeguard best": "#4c78a8",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True, constrained_layout=True)
    by_key = {(row["run_label"], row["coordinate"]): row for row in coord_rows}
    for ax, run_label in zip(axes, run_order):
        y = np.arange(len(coord_order))
        for idx, coord in enumerate(coord_order):
            row = by_key.get((run_label, coord))
            if row is None:
                continue
            ax.hlines(idx, row["min"], row["max"], color=colors[run_label], linewidth=2.0, alpha=0.35)
            ax.hlines(idx, row["q05"], row["q95"], color=colors[run_label], linewidth=6.0, alpha=0.9)
            ax.plot(row["q50"], idx, marker="o", markersize=5, color="black")
        ax.axvline(-0.05, color="black", linestyle="--", linewidth=1.1)
        ax.axvline(0.05, color="black", linestyle="--", linewidth=1.1)
        ax.axvline(-0.04, color="#8c6d31", linestyle=":", linewidth=1.1)
        ax.axvline(0.04, color="#8c6d31", linestyle=":", linewidth=1.1)
        ax.set_yticks(y, coord_order)
        ax.set_xlim(-0.055, 0.055)
        ax.set_xlabel("Executed z")
        ax.set_title(run_label.replace(" Markov ", "\nMarkov\n").replace(" no-safeguard ", "\nno-safeguard\n"))
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Markov basis coordinate")
    fig.suptitle("Executed z ranges by output-input Markov gain coordinate", fontsize=14)
    fig.savefig(OUT_DIR / "fig_z_coordinate_ranges.png", dpi=190)
    plt.close(fig)


def make_cap_pressure_figure(rows: list[dict[str, Any]]) -> None:
    caps = [0.02, 0.03, 0.035, 0.04, 0.0475]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    colors = ["#f58518", "#54a24b", "#4c78a8"]
    for row, color in zip(rows, colors):
        values = [100.0 * row[f"step_fraction_any_coord_abs_gt_{cap:g}"] for cap in caps]
        ax.plot(caps, values, marker="o", linewidth=2.2, color=color, label=row["label"])
    ax.set_xlabel("Hypothetical per-coordinate z cap")
    ax.set_ylabel("Steps that would be clipped (%)")
    ax.set_title("Performance cost proxy for shrinking z_bound")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_z_cap_clipping_pressure.png", dpi=190)
    plt.close(fig)


def make_z_reward_risk_figure(bundles: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    bins = np.linspace(0.0, 0.12, 13)
    for ax, (label, bundle) in zip(axes, bundles.items()):
        z = arr(bundle.get("z_executed_log"))
        rewards = arr(bundle.get("rewards_step"))
        warm = int(bundle.get("warm_start_step", 0))
        z_norm = np.linalg.norm(z[warm:], axis=1)
        rewards_post = rewards[warm:]
        centers = 0.5 * (bins[:-1] + bins[1:])
        mean_reward = []
        neg_fraction = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (z_norm >= lo) & (z_norm < hi)
            mean_reward.append(float(np.mean(rewards_post[mask])) if np.any(mask) else np.nan)
            neg_fraction.append(float(np.mean(rewards_post[mask] < 0.0)) if np.any(mask) else np.nan)
        ax2 = ax.twinx()
        ax.bar(centers, mean_reward, width=0.008, color="#72b7b2", alpha=0.75, label="mean reward")
        ax2.plot(centers, 100.0 * np.asarray(neg_fraction), color="#e45756", marker="o", linewidth=1.8, label="negative step %")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("||z||2")
        ax.grid(axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("Mean step reward")
        ax2.set_ylim(0, 105)
        if ax is axes[-1]:
            ax2.set_ylabel("Negative reward steps (%)")
    fig.suptitle("Large z norms are associated with negative step rewards", fontsize=14)
    fig.savefig(OUT_DIR / "fig_z_norm_reward_risk.png", dpi=190)
    plt.close(fig)


def make_fallback_figure(rows: list[dict[str, Any]]) -> None:
    labels = [row["label"].replace(" ", "\n") for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    accepted = [100.0 * row["accepted_fraction"] for row in rows]
    fallback = [100.0 * row["fallback_fraction"] for row in rows]
    negative = [100.0 * row["negative_avg_episode_fraction"] for row in rows]
    width = 0.25
    ax.bar(x - width, accepted, width, label="accepted %", color="#54a24b")
    ax.bar(x, fallback, width, label="fallback %", color="#f58518")
    ax.bar(x + width, negative, width, label="negative avg episodes %", color="#e45756")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent")
    ax.set_title("Guard activity and negative-reward episodes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_guard_activity_summary.png", dpi=190)
    plt.close(fig)


def main() -> None:
    rows = [summarize_run(label, path) for label, path in RUNS.items()]
    bundles = load_runs()
    coord_rows = summarize_coordinates(bundles)
    write_csv(rows, OUT_DIR / "markov_z_safety_summary.csv")
    write_csv(coord_rows, OUT_DIR / "markov_z_coordinate_summary.csv")
    make_reward_trace_figure(bundles)
    make_z_quantile_figure(rows)
    make_coordinate_range_figure(coord_rows)
    make_cap_pressure_figure(rows)
    make_z_reward_risk_figure(bundles)
    make_fallback_figure(rows)
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize(
                {
                    "current_default_z_bound": get_distillation_notebook_defaults("markov")["controller"]["z_bound"],
                    "runs": rows,
                    "coordinate_rows": coord_rows,
                    "run_paths": {label: path.as_posix() for label, path in RUNS.items()},
                }
            ),
            handle,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
