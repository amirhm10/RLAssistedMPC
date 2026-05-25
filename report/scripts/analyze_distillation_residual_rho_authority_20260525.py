from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation.config import RL_REWARD_DEFAULTS
from utils.helpers import apply_min_max, reverse_min_max
from utils.rewards import make_reward_fn_relative_QR


OUT_DIR = ROOT / "report" / "figures" / "distillation_residual_rho_authority_20260525"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_GLOB = ROOT / "Distillation" / "Results"
CANONICAL_PATTERN = "distillation_residual_*_unified"
TAIL_EPISODES = 20
NEAR_TRACKING_THRESHOLD = 0.1


def _as_array(bundle: dict, *keys: str, dtype=float):
    for key in keys:
        if key in bundle and bundle[key] is not None:
            arr = np.asarray(bundle[key], dtype=dtype)
            if arr.size:
                return arr
    return None


def _safe_mean(x):
    arr = np.asarray(x, float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_frac(x):
    arr = np.asarray(x, float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr > 0.5)) if arr.size else float("nan")


def _ratio_of_means(num, den):
    num_mean = _safe_mean(num)
    den_mean = _safe_mean(den)
    if not np.isfinite(num_mean) or not np.isfinite(den_mean) or abs(den_mean) <= 1.0e-12:
        return float("nan")
    return float(num_mean / den_mean)


def _tail_slice(n_steps: int, set_points_len: int | None):
    if n_steps <= 0:
        return slice(0, 0)
    steps_per_episode = int(set_points_len or 400)
    return slice(max(0, n_steps - TAIL_EPISODES * steps_per_episode), n_steps)


def _recompute_current_rewards(bundle: dict):
    dy = _as_array(bundle, "delta_y_storage")
    du = _as_array(bundle, "delta_u_storage")
    y_sp = _as_array(bundle, "y_sp")
    data_min = _as_array(bundle, "data_min")
    data_max = _as_array(bundle, "data_max")
    steady = bundle.get("steady_states", {}) or {}
    n_inputs = int(bundle.get("n_inputs", 2))
    if dy is None or du is None or y_sp is None or data_min is None or data_max is None:
        return _as_array(bundle, "rewards_step")
    if not isinstance(steady, dict) or "y_ss" not in steady:
        return _as_array(bundle, "rewards_step")

    y_ss_scaled = apply_min_max(np.asarray(steady["y_ss"], float), data_min[n_inputs:], data_max[n_inputs:])
    _, reward_fn = make_reward_fn_relative_QR(
        data_min,
        data_max,
        n_inputs,
        **RL_REWARD_DEFAULTS,
    )
    n = min(len(dy), len(du), len(y_sp))
    rewards = np.zeros(n, dtype=float)
    for i in range(n):
        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        rewards[i] = float(reward_fn(dy[i, :], du[i, :], y_sp_phys=y_sp_phys))
    return rewards


def _run_record(path: Path):
    with path.open("rb") as f:
        bundle = pickle.load(f)

    family = path.parents[1].name
    timestamp = path.parent.name
    agent = "sac" if "_sac_" in family else "td3" if "_td3_" in family else str(bundle.get("agent_kind", "unknown"))
    run_mode = "nominal" if "_nominal_" in family else "disturb" if "_disturb_" in family else str(bundle.get("run_mode", "unknown"))
    profile = "fluctuation" if "fluctuation" in family else "none" if "nominal" in family else "unknown"

    rewards = _recompute_current_rewards(bundle)
    stored_avg = _as_array(bundle, "avg_rewards")
    n_steps = int(len(rewards)) if rewards is not None else 0
    set_points_len = int(bundle.get("set_points_len", 400) or 400)
    tail = _tail_slice(n_steps, set_points_len)
    avg_tail = stored_avg[-TAIL_EPISODES:] if stored_avg is not None and len(stored_avg) else np.asarray([])

    raw = _as_array(bundle, "delta_u_res_raw_log", "residual_raw_log")
    executed = _as_array(bundle, "delta_u_res_exec_log", "residual_exec_log")
    raw_action = _as_array(bundle, "a_res_raw_log", "policy_action_raw_log")
    exec_action = _as_array(bundle, "a_res_exec_log", "executed_action_raw_log")
    tracking = _as_array(bundle, "tracking_error_raw_log")
    innovation = _as_array(bundle, "innovation_raw_log")
    rho = _as_array(bundle, "rho_log")
    rho_eff = _as_array(bundle, "rho_eff_log")
    projection = _as_array(bundle, "projection_active_log")
    proj_auth = _as_array(bundle, "projection_due_to_authority_log")
    proj_head = _as_array(bundle, "projection_due_to_headroom_log")
    proj_deadband = _as_array(bundle, "projection_due_to_deadband_log")
    deadband = _as_array(bundle, "deadband_active_log")

    raw_norm = np.linalg.norm(raw, axis=1) if raw is not None and raw.ndim == 2 else None
    exec_norm = np.linalg.norm(executed, axis=1) if executed is not None and executed.ndim == 2 else None
    raw_action_sat = (
        np.any(np.abs(raw_action) >= 0.99, axis=1)
        if raw_action is not None and raw_action.ndim == 2
        else np.asarray([])
    )
    max_tracking = (
        np.max(np.abs(tracking), axis=1)
        if tracking is not None and tracking.ndim == 2
        else np.asarray([])
    )
    max_innovation = (
        np.max(np.abs(innovation), axis=1)
        if innovation is not None and innovation.ndim == 2
        else np.asarray([])
    )
    near_mask = (
        (max_tracking <= NEAR_TRACKING_THRESHOLD)
        & (max_innovation <= NEAR_TRACKING_THRESHOLD)
        if max_tracking.size and max_innovation.size
        else np.asarray([], dtype=bool)
    )
    far_mask = max_tracking > 1.0 if max_tracking.size else np.asarray([], dtype=bool)

    def sl(arr):
        if arr is None or len(arr) == 0:
            return np.asarray([])
        return np.asarray(arr)[tail]

    def masked(arr, mask):
        if arr is None or len(arr) == 0 or mask.size == 0:
            return np.asarray([])
        n = min(len(arr), len(mask))
        return np.asarray(arr)[:n][mask[:n]]

    tail_raw = sl(raw_norm)
    tail_exec = sl(exec_norm)
    near_exec = masked(exec_norm, near_mask)
    near_raw = masked(raw_norm, near_mask)
    far_exec = masked(exec_norm, far_mask)
    far_raw = masked(raw_norm, far_mask)

    record = {
        "label": f"{agent.upper()} {timestamp}",
        "agent": agent,
        "family": family,
        "timestamp": timestamp,
        "run_mode": run_mode,
        "profile": profile,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "stored_tail_reward": _safe_mean(avg_tail),
        "current_tail_step_reward": _safe_mean(sl(rewards)),
        "current_final_episode_reward": float(stored_avg[-1]) if stored_avg is not None and len(stored_avg) else float("nan"),
        "rho_mean_tail": _safe_mean(sl(rho)),
        "rho_eff_mean_tail": _safe_mean(sl(rho_eff)),
        "rho_eff_q10_tail": float(np.nanpercentile(sl(rho_eff), 10)) if sl(rho_eff).size else float("nan"),
        "projection_frac_tail": _safe_frac(sl(projection)),
        "projection_authority_frac_tail": _safe_frac(sl(proj_auth)),
        "projection_headroom_frac_tail": _safe_frac(sl(proj_head)),
        "projection_deadband_frac_tail": _safe_frac(sl(proj_deadband)),
        "deadband_frac_tail": _safe_frac(sl(deadband)),
        "raw_norm_tail": _safe_mean(tail_raw),
        "exec_norm_tail": _safe_mean(tail_exec),
        "exec_raw_ratio_tail": _ratio_of_means(tail_exec, tail_raw),
        "raw_action_saturation_tail": _safe_frac(sl(raw_action_sat)),
        "near_setpoint_fraction": float(np.mean(near_mask)) if near_mask.size else float("nan"),
        "near_raw_norm": _safe_mean(near_raw),
        "near_exec_norm": _safe_mean(near_exec),
        "near_exec_raw_ratio": _ratio_of_means(near_exec, near_raw),
        "far_raw_norm": _safe_mean(far_raw),
        "far_exec_norm": _safe_mean(far_exec),
        "far_exec_raw_ratio": _ratio_of_means(far_exec, far_raw),
        "authority_use_rho": bundle.get("authority_use_rho", bundle.get("use_rho_authority")),
        "append_rho_to_state": bundle.get("append_rho_to_state"),
        "residual_zero_deadband_enabled": bundle.get("residual_zero_deadband_enabled"),
    }

    traces = {
        "rho_eff": rho_eff,
        "raw_norm": raw_norm,
        "exec_norm": exec_norm,
        "projection": projection,
        "projection_authority": proj_auth,
        "deadband": deadband,
        "max_tracking": max_tracking,
        "rewards": rewards,
    }
    return record, traces


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(rows: list[dict]):
    disturbed = [r for r in rows if r["run_mode"] == "disturb" and r["profile"] == "fluctuation"]
    disturbed.sort(key=lambda r: r["timestamp"])
    x = np.arange(len(disturbed))
    colors = ["#1f77b4" if r["agent"] == "td3" else "#ff7f0e" for r in disturbed]
    labels = [r["timestamp"][4:] for r in disturbed]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].bar(x, [r["current_tail_step_reward"] for r in disturbed], color=colors)
    axes[0].set_ylabel("tail step reward")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Distillation residual history: reward versus rho authority behavior")

    axes[1].bar(x, [r["projection_authority_frac_tail"] for r in disturbed], color=colors)
    axes[1].set_ylabel("authority projection frac")
    axes[1].set_ylim(0, 1.05)

    axes[2].bar(x, [r["exec_raw_ratio_tail"] for r in disturbed], color=colors)
    axes[2].set_ylabel("exec/raw residual norm")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[2].set_xlabel("run timestamp")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_rho_history_metrics.png", dpi=180)
    plt.close(fig)


def _plot_near_far(rows: list[dict]):
    disturbed_td3 = [
        r for r in rows if r["run_mode"] == "disturb" and r["profile"] == "fluctuation" and r["agent"] == "td3"
    ]
    disturbed_td3.sort(key=lambda r: r["timestamp"])
    if len(disturbed_td3) > 12:
        disturbed_td3 = disturbed_td3[-12:]
    x = np.arange(len(disturbed_td3))
    width = 0.35
    labels = [r["timestamp"][4:] for r in disturbed_td3]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(x - width / 2, [r["near_raw_norm"] for r in disturbed_td3], width, label="raw near setpoint")
    axes[0].bar(x + width / 2, [r["near_exec_norm"] for r in disturbed_td3], width, label="executed near setpoint")
    axes[0].set_ylabel("residual norm")
    axes[0].set_title("Rho authority suppresses residual corrections near setpoint")
    axes[0].legend()

    axes[1].bar(x - width / 2, [r["far_raw_norm"] for r in disturbed_td3], width, label="raw far from setpoint")
    axes[1].bar(x + width / 2, [r["far_exec_norm"] for r in disturbed_td3], width, label="executed far from setpoint")
    axes[1].set_ylabel("residual norm")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_near_far_suppression.png", dpi=180)
    plt.close(fig)


def _plot_mechanism(rows: list[dict], traces_by_label: dict[str, dict]):
    candidates = [r for r in rows if r["run_mode"] == "disturb" and r["profile"] == "fluctuation"]
    candidates.sort(key=lambda r: (np.nan_to_num(r["current_tail_step_reward"], nan=-1e9), r["timestamp"]))
    selected = []
    if candidates:
        selected.append(candidates[-1])
    for r in reversed(candidates):
        if r["timestamp"] in {"20260523_203649", "20260521_152021", "20260522_180223"} and r not in selected:
            selected.append(r)
    selected = selected[:4]

    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 2.8 * max(1, len(selected))), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, selected):
        traces = traces_by_label[row["label"]]
        rho_eff = traces["rho_eff"]
        raw = traces["raw_norm"]
        exe = traces["exec_norm"]
        if rho_eff is None or raw is None or exe is None:
            ax.text(0.5, 0.5, f"{row['label']}: missing traces", ha="center", va="center")
            continue
        n = min(len(rho_eff), len(raw), len(exe))
        start = max(0, n - 4000)
        t = np.arange(n - start)
        ax.plot(t, rho_eff[start:n], label="rho_eff", color="#4c78a8", linewidth=1.0)
        ax.plot(t, raw[start:n], label="raw residual norm", color="#f58518", linewidth=0.8, alpha=0.8)
        ax.plot(t, exe[start:n], label="executed residual norm", color="#54a24b", linewidth=0.8, alpha=0.9)
        ax.set_title(f"{row['label']} tail trace")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_rho_tail_mechanism.png", dpi=180)
    plt.close(fig)


def main():
    paths = sorted((RUN_GLOB).glob(f"{CANONICAL_PATTERN}/*/input_data.pkl"))
    rows = []
    traces = {}
    for path in paths:
        record, trace = _run_record(path)
        rows.append(record)
        traces[record["label"]] = trace

    rows.sort(key=lambda r: (r["run_mode"], r["profile"], r["agent"], r["timestamp"]))
    _write_csv(OUT_DIR / "residual_rho_authority_summary.csv", rows)

    disturbed = [r for r in rows if r["run_mode"] == "disturb" and r["profile"] == "fluctuation"]
    td3 = [r for r in disturbed if r["agent"] == "td3"]
    summary = {
        "n_canonical_residual_runs": len(rows),
        "n_disturb_fluctuation_runs": len(disturbed),
        "n_disturb_td3_runs": len(td3),
        "best_disturb_tail_reward": max((r["current_tail_step_reward"] for r in disturbed), default=float("nan")),
        "best_disturb_label": max(disturbed, key=lambda r: r["current_tail_step_reward"])["label"] if disturbed else None,
        "latest_disturb_td3": td3[-1] if td3 else None,
        "near_setpoint_threshold": NEAR_TRACKING_THRESHOLD,
    }
    (OUT_DIR / "residual_rho_authority_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_history(rows)
    _plot_near_far(rows)
    _plot_mechanism(rows, traces)

    print(f"Wrote {len(rows)} residual records to {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
