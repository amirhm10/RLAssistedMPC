from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "Distillation" / "Results" / "distillation_reidentification_td3_disturb_fluctuation_mismatch_unified"
COMPARE_ROOT = ROOT / "Distillation" / "Results" / "distillation_compare_reidentification_td3_disturb_fluctuation_mismatch"
OUT_DIR = ROOT / "report" / "figures" / "distillation_reidentification_2026_04_20"

RUN_SPECS = [
    {
        "run_id": "20260416_113030",
        "compare_id": "20260416_113035",
        "label": "Apr 16: 80/5, open blend",
        "short_label": "Apr 16",
        "color": "#1f77b4",
    },
    {
        "run_id": "20260417_174602",
        "compare_id": "20260417_174605",
        "label": "Apr 17: 160/20, fade + guard",
        "short_label": "Apr 17",
        "color": "#ff7f0e",
    },
    {
        "run_id": "20260418_115020",
        "compare_id": "20260418_115023",
        "label": "Apr 18: 160/20, no fade, no freeze",
        "short_label": "Apr 18",
        "color": "#d62728",
    },
    {
        "run_id": "20260420_171837",
        "compare_id": "20260420_171840",
        "label": "Apr 20: 160/20, freeze, fixed $\\eta=0.05$",
        "short_label": "Apr 20",
        "color": "#2ca02c",
    },
]

WINDOWS = [(1, 10), (11, 30), (31, 60), (61, 100), (101, 150), (151, 200)]


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _episode_mean(arr: np.ndarray, n_eps: int, steps_per_ep: int) -> np.ndarray:
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr.reshape(n_eps, steps_per_ep).mean(axis=1)


def _episode_sum(arr: np.ndarray, n_eps: int, steps_per_ep: int) -> np.ndarray:
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr.reshape(n_eps, steps_per_ep).sum(axis=1)


def _episode_event_mean(values: np.ndarray, mask: np.ndarray, n_eps: int, steps_per_ep: int) -> np.ndarray:
    if values.size == 0 or mask.size == 0:
        return np.asarray([], dtype=float)
    values_2d = values.reshape(n_eps, steps_per_ep)
    mask_2d = mask.reshape(n_eps, steps_per_ep) > 0.5
    out = np.full(n_eps, np.nan, dtype=float)
    for i in range(n_eps):
        if mask_2d[i].any():
            out[i] = float(values_2d[i][mask_2d[i]].mean())
    return out


def _cfg_value(bundle: dict, key: str):
    value = bundle.get(key, bundle.get("config_snapshot", {}).get(key))
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return [float(x) for x in value.reshape(-1)]
    return value


def load_run(spec: dict) -> dict:
    run_bundle = _load_pickle(RUN_ROOT / spec["run_id"] / "input_data.pkl")
    compare_bundle = _load_pickle(COMPARE_ROOT / spec["compare_id"] / "input_data.pkl")

    rl = np.asarray(compare_bundle["avg_rewards_rl"], dtype=float)
    mpc = np.asarray(compare_bundle["avg_rewards_mpc"], dtype=float)
    gap = rl - mpc
    n_eps = len(rl)

    eta_a = np.asarray(run_bundle.get("eta_A_log", []), dtype=float)
    steps_per_ep = int(len(eta_a) // n_eps) if n_eps and eta_a.size else 0
    if n_eps and eta_a.size and len(eta_a) % n_eps != 0:
        raise ValueError(f"Run {spec['run_id']} has non-integer steps per episode.")

    eta_b = np.asarray(run_bundle.get("eta_B_log", []), dtype=float)
    eta_a_raw = np.asarray(run_bundle.get("eta_A_raw_log", []), dtype=float)
    eta_b_raw = np.asarray(run_bundle.get("eta_B_raw_log", []), dtype=float)
    id_event = np.asarray(run_bundle.get("id_update_event_log", []), dtype=float)
    id_success = np.asarray(run_bundle.get("id_update_success_log", []), dtype=float)
    id_valid = np.asarray(run_bundle.get("id_candidate_valid_log", []), dtype=float)
    resid = np.asarray(run_bundle.get("id_residual_norm_log", []), dtype=float)
    cond = np.asarray(run_bundle.get("id_condition_number_log", []), dtype=float)
    active_a = np.asarray(run_bundle.get("active_A_model_delta_ratio_log", []), dtype=float)
    active_b = np.asarray(run_bundle.get("active_B_model_delta_ratio_log", []), dtype=float)

    eta = np.maximum(eta_a, eta_b) if eta_a.size and eta_b.size else np.asarray([], dtype=float)
    eta_raw = np.maximum(eta_a_raw, eta_b_raw) if eta_a_raw.size and eta_b_raw.size else np.asarray([], dtype=float)
    eff_a = eta_a * active_a if eta_a.size and active_a.size else np.asarray([], dtype=float)
    eff_b = eta_b * active_b if eta_b.size and active_b.size else np.asarray([], dtype=float)

    episode_gap = gap
    episode_eta = _episode_mean(eta, n_eps, steps_per_ep)
    episode_eta_raw = _episode_mean(eta_raw, n_eps, steps_per_ep)
    episode_success = _episode_sum(id_success, n_eps, steps_per_ep)
    episode_events = _episode_sum(id_event, n_eps, steps_per_ep)
    episode_valid = _episode_event_mean(id_valid, id_event, n_eps, steps_per_ep)
    episode_resid = _episode_mean(resid, n_eps, steps_per_ep)
    episode_cond = _episode_mean(cond, n_eps, steps_per_ep)
    episode_active_a = _episode_mean(active_a, n_eps, steps_per_ep)
    episode_active_b = _episode_mean(active_b, n_eps, steps_per_ep)
    episode_eff_a = _episode_mean(eff_a, n_eps, steps_per_ep)
    episode_eff_b = _episode_mean(eff_b, n_eps, steps_per_ep)

    windows = {}
    for start, end in WINDOWS:
        sl = slice(start - 1, end)
        windows[f"{start:03d}-{end:03d}"] = {
            "gap_mean": float(np.mean(episode_gap[sl])),
            "gap_last": float(episode_gap[sl][-1]),
            "eta_mean": float(np.mean(episode_eta[sl])) if episode_eta.size else None,
            "eta_raw_mean": float(np.mean(episode_eta_raw[sl])) if episode_eta_raw.size else None,
            "success_sum": int(np.sum(episode_success[sl])) if episode_success.size else 0,
            "event_sum": int(np.sum(episode_events[sl])) if episode_events.size else 0,
            "valid_rate": float(np.nanmean(episode_valid[sl])) if episode_valid.size and np.isfinite(episode_valid[sl]).any() else None,
            "residual_mean": float(np.mean(episode_resid[sl])) if episode_resid.size else None,
            "condition_mean": float(np.mean(episode_cond[sl])) if episode_cond.size else None,
            "active_A_mean": float(np.mean(episode_active_a[sl])) if episode_active_a.size else None,
            "active_B_mean": float(np.mean(episode_active_b[sl])) if episode_active_b.size else None,
            "effective_A_mean": float(np.mean(episode_eff_a[sl])) if episode_eff_a.size else None,
            "effective_B_mean": float(np.mean(episode_eff_b[sl])) if episode_eff_b.size else None,
        }

    summary = {
        "run_id": spec["run_id"],
        "compare_id": spec["compare_id"],
        "label": spec["label"],
        "short_label": spec["short_label"],
        "metrics": {
            "final_gap": float(episode_gap[-1]),
            "avg_gap": float(np.mean(episode_gap)),
            "last50_gap": float(np.mean(episode_gap[-50:])),
            "better_episodes": int(np.sum(episode_gap > 0.0)),
            "final_rl": float(rl[-1]),
            "final_mpc": float(mpc[-1]),
            "eta_mean": float(np.mean(episode_eta)) if episode_eta.size else None,
            "eta_last50_mean": float(np.mean(episode_eta[-50:])) if episode_eta.size else None,
            "success_total": int(np.sum(episode_success)) if episode_success.size else 0,
            "events_total": int(np.sum(episode_events)) if episode_events.size else 0,
            "valid_rate_total": float(np.nanmean(episode_valid)) if episode_valid.size and np.isfinite(episode_valid).any() else None,
            "active_A_mean": float(np.mean(episode_active_a)) if episode_active_a.size else None,
            "active_B_mean": float(np.mean(episode_active_b)) if episode_active_b.size else None,
            "effective_A_mean": float(np.mean(episode_eff_a)) if episode_eff_a.size else None,
            "effective_B_mean": float(np.mean(episode_eff_b)) if episode_eff_b.size else None,
        },
        "setup": {
            "id_window": _cfg_value(run_bundle, "id_window"),
            "id_update_period": _cfg_value(run_bundle, "id_update_period"),
            "candidate_guard_mode": _cfg_value(run_bundle, "candidate_guard_mode"),
            "eta_tau_A": _cfg_value(run_bundle, "eta_tau_A"),
            "eta_tau_B": _cfg_value(run_bundle, "eta_tau_B"),
            "freeze_identification_during_warm_start": _cfg_value(run_bundle, "freeze_identification_during_warm_start"),
            "blend_validity_mode": _cfg_value(run_bundle, "blend_validity_mode"),
            "force_eta_constant": _cfg_value(run_bundle, "force_eta_constant"),
            "lambda_prev_A": _cfg_value(run_bundle, "lambda_prev_A"),
            "theta_low_A": _cfg_value(run_bundle, "theta_low_A"),
            "theta_high_A": _cfg_value(run_bundle, "theta_high_A"),
            "delta_A_max": _cfg_value(run_bundle, "delta_A_max"),
        },
        "windows": windows,
        "series": {
            "reward_gap": episode_gap.tolist(),
            "reward_rl": rl.tolist(),
            "reward_mpc": mpc.tolist(),
            "eta_mean": episode_eta.tolist(),
            "eta_raw_mean": episode_eta_raw.tolist(),
            "id_success_sum": episode_success.tolist(),
            "id_events_sum": episode_events.tolist(),
            "id_valid_rate": episode_valid.tolist(),
            "residual_mean": episode_resid.tolist(),
            "condition_mean": episode_cond.tolist(),
            "active_A_mean": episode_active_a.tolist(),
            "active_B_mean": episode_active_b.tolist(),
            "effective_A_mean": episode_eff_a.tolist(),
            "effective_B_mean": episode_eff_b.tolist(),
        },
    }
    return summary


def _prep_axes(ax, ylabel: str, title: str | None = None):
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.axvspan(0.5, 10.5, color="#f1f3f5", zorder=0)
    ax.axvline(10.5, color="#adb5bd", ls="--", lw=1.0)
    ax.set_xlim(1, 200)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_reward_gap(runs: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(1, 201)
    for run in runs:
        ax.plot(x, run["series"]["reward_gap"], lw=2.0, label=run["label"], color=run["color"])
    _prep_axes(ax, r"$\bar{J}_{\mathrm{RL}} - \bar{J}_{\mathrm{MPC}}$", "Episode Reward Gap")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_gap_overlay.png", dpi=220)
    plt.close(fig)


def plot_eta(runs: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(1, 201)
    for run in runs:
        ax.plot(x, run["series"]["eta_mean"], lw=2.0, label=run["label"], color=run["color"])
    _prep_axes(ax, r"Episode mean $\bar{\eta}$", "Mean Blend Applied to the Identified Model")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "eta_overlay.png", dpi=220)
    plt.close(fig)


def plot_id_activity(runs: list[dict], out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    x = np.arange(1, 201)
    for run in runs:
        axes[0].plot(x, run["series"]["id_success_sum"], lw=2.0, label=run["label"], color=run["color"])
        axes[1].plot(x, run["series"]["id_valid_rate"], lw=2.0, label=run["label"], color=run["color"])
    _prep_axes(axes[0], "Accepted updates / episode", "Identification Activity")
    _prep_axes(axes[1], "Valid-candidate rate", None)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-0.02, 0.45)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "id_activity_overlay.png", dpi=220)
    plt.close(fig)


def plot_effective_deviation(runs: list[dict], out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    x = np.arange(1, 201)
    for run in runs:
        axes[0].plot(x, run["series"]["effective_A_mean"], lw=2.0, label=run["label"], color=run["color"])
        axes[1].plot(x, run["series"]["effective_B_mean"], lw=2.0, label=run["label"], color=run["color"])
    _prep_axes(axes[0], r"Episode mean $\eta_A \|\Delta A\|/\|A_0\|$", "Effective Blended Model Deviation")
    _prep_axes(axes[1], r"Episode mean $\eta_B \|\Delta B\|/\|B_0\|$", None)
    axes[1].set_xlabel("Episode")
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "effective_model_deviation_overlay.png", dpi=220)
    plt.close(fig)


def plot_latest_reward_compare(latest: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(1, 201)
    ax.plot(x, latest["series"]["reward_rl"], lw=2.0, label="RL / fixed small blend", color=latest["color"])
    ax.plot(x, latest["series"]["reward_mpc"], lw=2.0, ls="--", label="Baseline MPC", color="#495057")
    _prep_axes(ax, "Average reward", "Latest Run: Reward Comparison vs MPC")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "latest_reward_compare.png", dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_runs = []
    for spec in RUN_SPECS:
        run = load_run(spec)
        run["color"] = spec["color"]
        loaded_runs.append(run)

    plot_reward_gap(loaded_runs, OUT_DIR)
    plot_eta(loaded_runs, OUT_DIR)
    plot_id_activity(loaded_runs, OUT_DIR)
    plot_effective_deviation(loaded_runs, OUT_DIR)
    plot_latest_reward_compare(loaded_runs[-1], OUT_DIR)

    payload = {
        "episode_length": 400,
        "reward_note": (
            "Absolute reward values changed after April 16, 2026 due to reward-parameter updates. "
            "Cross-run comparisons should therefore rely on RL-minus-MPC gaps, not raw reward levels."
        ),
        "runs": loaded_runs,
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    for run in loaded_runs:
        metrics = run["metrics"]
        print(
            run["run_id"],
            {
                "final_gap": round(metrics["final_gap"], 6),
                "avg_gap": round(metrics["avg_gap"], 6),
                "last50_gap": round(metrics["last50_gap"], 6),
                "better_episodes": metrics["better_episodes"],
                "eta_mean": round(metrics["eta_mean"], 6) if metrics["eta_mean"] is not None else None,
                "success_total": metrics["success_total"],
                "events_total": metrics["events_total"],
            },
        )


if __name__ == "__main__":
    main()
