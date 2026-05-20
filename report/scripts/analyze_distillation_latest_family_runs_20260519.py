from __future__ import annotations

import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation import get_distillation_notebook_defaults

RESULTS_ROOT = REPO_ROOT / "Distillation" / "Results"
BASELINE_PATH = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_latest_family_runs_20260519"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FamilySpec:
    key: str
    label: str
    latest_path: Path
    history_globs: tuple[str, ...]
    default_family: str
    agent_key: str


FAMILIES = [
    FamilySpec(
        key="residual",
        label="Residual TD3",
        latest_path=RESULTS_ROOT
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260519_200250/input_data.pkl",
        history_globs=("distillation_residual_*_disturb_fluctuation*_unified/*/input_data.pkl",),
        default_family="residual",
        agent_key="td3_agent",
    ),
    FamilySpec(
        key="weights",
        label="Weights TD3",
        latest_path=RESULTS_ROOT
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified/20260519_201951/input_data.pkl",
        history_globs=("distillation_weights_*_disturb_fluctuation*_unified/*/input_data.pkl",),
        default_family="weights",
        agent_key="td3_agent",
    ),
    FamilySpec(
        key="horizon",
        label="Horizon DDQN",
        latest_path=RESULTS_ROOT
        / "distillation_horizon_disturb_fluctuation_mismatch_unified/20260519_202111/input_data.pkl",
        history_globs=("distillation_horizon_disturb_fluctuation*_unified/*/input_data.pkl",),
        default_family="horizon_standard",
        agent_key="agent",
    ),
    FamilySpec(
        key="dueling",
        label="Dueling DDQN",
        latest_path=RESULTS_ROOT
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260519_204534/input_data.pkl",
        history_globs=("distillation_dueling_horizon_disturb_fluctuation*_unified/*/input_data.pkl",),
        default_family="horizon_dueling",
        agent_key="agent",
    ),
    FamilySpec(
        key="markov",
        label="Markov TD3",
        latest_path=RESULTS_ROOT
        / "distillation_markov_td3_disturb_fluctuation_unified/20260519_210736/input_data.pkl",
        history_globs=("distillation_markov_td3_disturb_fluctuation*_unified/*/input_data.pkl",),
        default_family="markov",
        agent_key="td3_agent",
    ),
]


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, found {type(obj).__name__}")
    return obj


def as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def finite_tail(values: Any, n: int = 20) -> float:
    arr = as_float_array(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr[-min(n, arr.size) :]))


def final_value(values: Any) -> float:
    arr = as_float_array(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr[-1])


def safe_max_abs(lhs: Any, rhs: Any) -> float:
    a = as_float_array(lhs)
    b = as_float_array(rhs)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return float("nan")
    return float(np.nanmax(np.abs(a - b)))


def tracking_metrics(bundle: dict[str, Any], prefix: str = "rl") -> dict[str, float]:
    y = as_float_array(bundle.get("y_rl", bundle.get("y")))
    y_sp = as_float_array(bundle.get("y_sp"))
    u = as_float_array(bundle.get("u_rl", bundle.get("u")))
    if y.ndim != 2 or y_sp.ndim != 2 or y.shape[0] < 2 or y_sp.shape[0] == 0:
        return {
            f"{prefix}_iae_y1": float("nan"),
            f"{prefix}_iae_y2": float("nan"),
            f"{prefix}_rmse_y1": float("nan"),
            f"{prefix}_rmse_y2": float("nan"),
            f"{prefix}_tail_rmse_y1": float("nan"),
            f"{prefix}_tail_rmse_y2": float("nan"),
            f"{prefix}_input_tv": float("nan"),
        }
    err = y[:-1, :] - y_sp
    tail_len = min(4000, err.shape[0])
    tail_err = err[-tail_len:, :]
    input_tv = float(np.nansum(np.abs(np.diff(u, axis=0)))) if u.ndim == 2 and u.shape[0] > 1 else float("nan")
    return {
        f"{prefix}_iae_y1": float(np.nansum(np.abs(err[:, 0]))),
        f"{prefix}_iae_y2": float(np.nansum(np.abs(err[:, 1]))),
        f"{prefix}_rmse_y1": float(np.sqrt(np.nanmean(err[:, 0] ** 2))),
        f"{prefix}_rmse_y2": float(np.sqrt(np.nanmean(err[:, 1] ** 2))),
        f"{prefix}_tail_rmse_y1": float(np.sqrt(np.nanmean(tail_err[:, 0] ** 2))),
        f"{prefix}_tail_rmse_y2": float(np.sqrt(np.nanmean(tail_err[:, 1] ** 2))),
        f"{prefix}_input_tv": input_tv,
    }


def reward_signature(bundle: dict[str, Any]) -> dict[str, Any]:
    params = bundle.get("reward_params")
    if not isinstance(params, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ["k_rel", "band_floor_phys", "Q_diag", "R_diag", "beta", "reward_scale"]:
        value = params.get(key)
        if isinstance(value, np.ndarray):
            out[key] = value.astype(float).tolist()
        elif value is not None:
            out[key] = value
    return out


def reward_regime(signature: dict[str, Any]) -> str:
    q_diag = signature.get("Q_diag")
    band = signature.get("band_floor_phys")
    k_rel = signature.get("k_rel")
    if q_diag == [37000.0, 5000.0] and band == [0.003, 0.2] and k_rel == [0.3, 0.01]:
        return "current_tight"
    if q_diag == [37000.0, 1500.0] and band == [0.003, 0.3] and k_rel == [0.3, 0.02]:
        return "previous_wide"
    if not signature:
        return "not_logged"
    return "other"


def summarize_history(spec: FamilySpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in spec.history_globs:
        for path in RESULTS_ROOT.glob(pattern):
            try:
                bundle = load_pickle(path)
            except Exception:
                continue
            cfg = bundle.get("config_snapshot", {})
            rows.append(
                {
                    "family": spec.key,
                    "run_dir": path.parent.name,
                    "path": path.as_posix(),
                    "is_latest": path == spec.latest_path,
                    "state_mode": cfg.get("state_mode"),
                    "agent_kind": cfg.get("agent_kind", bundle.get("agent_kind")),
                    "algorithm": cfg.get("algorithm", bundle.get("algorithm")),
                    "tail_reward": finite_tail(bundle.get("avg_rewards")),
                    "final_reward": final_value(bundle.get("avg_rewards")),
                    "reward_signature": reward_signature(bundle),
                    "behavioral_cloning_enabled": bool(bundle.get("behavioral_cloning_enabled", False)),
                    "post_warm_start_action_freeze_subepisodes": cfg.get(
                        "post_warm_start_action_freeze_subepisodes",
                        cfg.get("td3_post_warm_start_action_freeze_subepisodes"),
                    ),
                    "post_warm_start_actor_freeze_subepisodes": cfg.get(
                        "post_warm_start_actor_freeze_subepisodes",
                        cfg.get("td3_post_warm_start_actor_freeze_subepisodes"),
                    ),
                }
            )
    rows.sort(key=lambda row: row["run_dir"])
    return rows


def default_summary(spec: FamilySpec) -> dict[str, Any]:
    nb = get_distillation_notebook_defaults(spec.default_family)
    agent_cfg = dict(nb.get(spec.agent_key, {}))
    reward_cfg = dict(nb.get("reward", {}))
    return {
        "agent_kind_default": nb.get("agent_kind", "dqn" if spec.key == "horizon" else "dueling_dqn" if spec.key == "dueling" else None),
        "state_mode_default": nb.get("state_mode"),
        "hidden_layers": agent_cfg.get("hidden_layers"),
        "actor_hidden": agent_cfg.get("actor_hidden"),
        "critic_hidden": agent_cfg.get("critic_hidden"),
        "gamma": agent_cfg.get("gamma"),
        "n_step": agent_cfg.get("n_step"),
        "multistep_mode": agent_cfg.get("multistep_mode"),
        "target_update": agent_cfg.get("target_update"),
        "policy_delay": agent_cfg.get("policy_delay"),
        "exploration_mode": agent_cfg.get("exploration_mode"),
        "eps_start": agent_cfg.get("eps_start"),
        "std_start": agent_cfg.get("std_start"),
        "param_noise_std_start": agent_cfg.get("param_noise_std_start"),
        "reward": {
            key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value
            for key, value in reward_cfg.items()
            if key in {"k_rel", "band_floor_phys", "Q_diag", "R_diag", "beta", "reward_scale"}
        },
    }


def summarize_latest(spec: FamilySpec, baseline: dict[str, Any], history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = load_pickle(spec.latest_path)
    cfg = bundle.get("config_snapshot", {})
    latest_agent_kind = cfg.get("agent_kind", bundle.get("agent_kind"))
    latest_algorithm = cfg.get("algorithm", bundle.get("algorithm"))
    previous = [row for row in history_rows if not row["is_latest"] and np.isfinite(row["tail_reward"])]
    best_prev = max(previous, key=lambda row: row["tail_reward"]) if previous else None
    previous_same_agent = [
        row
        for row in previous
        if (latest_agent_kind is None or row.get("agent_kind") == latest_agent_kind)
        and (latest_algorithm is None or row.get("algorithm") == latest_algorithm)
    ]
    best_prev_same_agent = max(previous_same_agent, key=lambda row: row["tail_reward"]) if previous_same_agent else None
    baseline_tail = finite_tail(baseline.get("avg_rewards"))
    latest_tail = finite_tail(bundle.get("avg_rewards"))
    latest_final = final_value(bundle.get("avg_rewards"))
    mpc_tail_in_bundle = finite_tail(bundle.get("avg_rewards_mpc"))

    row = {
        "family": spec.key,
        "label": spec.label,
        "latest_run": spec.latest_path.parent.name,
        "latest_path": spec.latest_path.as_posix(),
        "state_mode": cfg.get("state_mode"),
        "agent_kind": latest_agent_kind,
        "algorithm": latest_algorithm,
        "tail_reward": latest_tail,
        "final_reward": latest_final,
        "baseline_tail_reward": baseline_tail,
        "tail_reward_minus_baseline": latest_tail - baseline_tail,
        "mpc_tail_reward_in_bundle": mpc_tail_in_bundle,
        "best_previous_run": None if best_prev is None else best_prev["run_dir"],
        "best_previous_tail_reward": float("nan") if best_prev is None else best_prev["tail_reward"],
        "tail_reward_minus_best_previous": float("nan") if best_prev is None else latest_tail - best_prev["tail_reward"],
        "best_previous_same_agent_run": None if best_prev_same_agent is None else best_prev_same_agent["run_dir"],
        "best_previous_same_agent_tail_reward": float("nan") if best_prev_same_agent is None else best_prev_same_agent["tail_reward"],
        "tail_reward_minus_best_previous_same_agent": float("nan")
        if best_prev_same_agent is None
        else latest_tail - best_prev_same_agent["tail_reward"],
        "max_abs_y_rl_minus_bundle_mpc": safe_max_abs(bundle.get("y_rl"), bundle.get("y_mpc")),
        "max_abs_u_rl_minus_bundle_mpc": safe_max_abs(bundle.get("u_rl"), bundle.get("u_mpc")),
        "max_abs_y_rl_minus_canonical_mpc": safe_max_abs(bundle.get("y_rl"), baseline.get("y")),
        "max_abs_u_rl_minus_canonical_mpc": safe_max_abs(bundle.get("u_rl"), baseline.get("u")),
        "warm_start_step": bundle.get("warm_start_step", cfg.get("warm_start")),
        "behavioral_cloning_enabled": bool(bundle.get("behavioral_cloning_enabled", False)),
        "bc_active_fraction": float(np.nanmean(as_float_array(bundle.get("bc_active_log")) > 0))
        if as_float_array(bundle.get("bc_active_log")).size
        else 0.0,
        "post_warm_start_action_freeze_subepisodes": cfg.get(
            "post_warm_start_action_freeze_subepisodes",
            cfg.get("td3_post_warm_start_action_freeze_subepisodes"),
        ),
        "post_warm_start_actor_freeze_subepisodes": cfg.get(
            "post_warm_start_actor_freeze_subepisodes",
            cfg.get("td3_post_warm_start_actor_freeze_subepisodes"),
        ),
        "reward_signature": reward_signature(bundle),
    }
    row.update(tracking_metrics(bundle, prefix="latest"))

    for log_key in ["phase1_action_source_log", "action_source_log", "fallback_log", "accepted_log"]:
        arr = as_float_array(bundle.get(log_key))
        if arr.size:
            row[f"{log_key}_finite_mean"] = float(np.nanmean(arr))
            row[f"{log_key}_positive_fraction"] = float(np.nanmean(arr > 0))
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe_leaf(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_figures(latest_rows: list[dict[str, Any]], history_rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in latest_rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    latest = [row["tail_reward"] for row in latest_rows]
    baseline = [row["baseline_tail_reward"] for row in latest_rows]
    best_prev = [row["best_previous_tail_reward"] for row in latest_rows]
    width = 0.25
    ax.bar(x - width, best_prev, width, label="Best previous", color="#4c78a8")
    ax.bar(x, latest, width, label="Latest", color="#f58518")
    ax.bar(x + width, baseline, width, label="Baseline MPC", color="#54a24b")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Tail average reward")
    ax.set_title("Latest distillation family runs versus previous best and MPC")
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_latest_tail_reward_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    delta_best = [row["tail_reward_minus_best_previous"] for row in latest_rows]
    delta_mpc = [row["tail_reward_minus_baseline"] for row in latest_rows]
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(x - width / 2.0, delta_best, width, label="Latest - previous best", color="#e45756")
    ax.bar(x + width / 2.0, delta_mpc, width, label="Latest - MPC", color="#72b7b2")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Tail reward difference")
    ax.set_title("Where the latest runs improve or regress")
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_latest_reward_deltas.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    y_diff = [row["max_abs_y_rl_minus_canonical_mpc"] for row in latest_rows]
    u_diff = [row["max_abs_u_rl_minus_canonical_mpc"] for row in latest_rows]
    ax.bar(x - width / 2.0, y_diff, width, label="max |y_rl - y_mpc_file|", color="#b279a2")
    ax.bar(x + width / 2.0, u_diff, width, label="max |u_rl - u_mpc_file|", color="#ff9da6")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Absolute difference")
    ax.set_title("Latest assisted rollouts differ from canonical MPC")
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_latest_rollout_difference_from_mpc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(FAMILIES), 1, figsize=(11, 12), sharex=False, constrained_layout=True)
    for ax, spec in zip(axes, FAMILIES):
        rows = [row for row in history_rows if row["family"] == spec.key and np.isfinite(row["tail_reward"])]
        rows.sort(key=lambda row: row["run_dir"])
        colors = ["#f58518" if row["is_latest"] else "#4c78a8" for row in rows]
        ax.bar([row["run_dir"] for row in rows], [row["tail_reward"] for row in rows], color=colors)
        ax.set_title(spec.label)
        ax.set_ylabel("Tail reward")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Historical tail reward by distillation family", fontsize=14)
    fig.savefig(OUT_DIR / "fig_history_tail_rewards_by_family.png", dpi=180)
    plt.close(fig)

    make_reward_trace_figure(latest_rows)
    make_latest_tracking_figure(latest_rows)
    make_metric_dashboard(latest_rows)
    make_reward_regime_figure(history_rows)
    make_root_cause_matrix_figure()


def make_reward_trace_figure(latest_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(len(FAMILIES), 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for ax, spec, latest_row in zip(axes, FAMILIES, latest_rows):
        latest_bundle = load_pickle(spec.latest_path)
        best_path = Path(str(latest_row["best_previous_run"] or ""))
        best_bundle = None
        if latest_row["best_previous_run"]:
            for pattern in spec.history_globs:
                candidates = list(RESULTS_ROOT.glob(pattern))
                for candidate in candidates:
                    if candidate.parent.name == latest_row["best_previous_run"]:
                        best_bundle = load_pickle(candidate)
                        break
                if best_bundle is not None:
                    break

        ax.plot(as_float_array(latest_bundle.get("avg_rewards")), label=f"latest {spec.latest_path.parent.name}", color="#f58518", linewidth=2.0)
        if best_bundle is not None:
            ax.plot(as_float_array(best_bundle.get("avg_rewards")), label=f"best previous {best_path}", color="#4c78a8", linewidth=2.0)
        ax.axhline(float(latest_row["baseline_tail_reward"]), color="#54a24b", linewidth=1.2, linestyle="--", label="MPC tail reward")
        ax.axvline(int(latest_row.get("warm_start_step", 4000)) / 400.0, color="black", linewidth=0.8, linestyle=":", label="warm-start boundary" if spec.key == "residual" else None)
        ax.set_title(spec.label)
        ax.set_ylabel("Avg reward")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("Subepisode")
    fig.suptitle("Reward traces: latest run versus historical best", fontsize=14)
    fig.savefig(OUT_DIR / "fig_latest_vs_best_reward_traces.png", dpi=180)
    plt.close(fig)


def make_latest_tracking_figure(latest_rows: list[dict[str, Any]]) -> None:
    baseline = load_pickle(BASELINE_PATH)
    y_mpc = as_float_array(baseline.get("y"))
    window = 4000
    fig, axes = plt.subplots(len(FAMILIES), 2, figsize=(13, 13), sharex=True, constrained_layout=True)
    output_labels = ["Tray-24 C2 composition", "Tray-85 temperature"]
    for row_idx, (spec, latest_row) in enumerate(zip(FAMILIES, latest_rows)):
        bundle = load_pickle(spec.latest_path)
        y = as_float_array(bundle.get("y_rl"))
        y_sp = as_float_array(bundle.get("y_sp"))
        n = min(window, y_sp.shape[0], y.shape[0] - 1)
        t = np.arange(n)
        for output_idx in range(2):
            ax = axes[row_idx, output_idx]
            ax.plot(t, y[-n - 1 : -1, output_idx], color="#f58518", linewidth=1.2, label="latest RL")
            if y_mpc.ndim == 2 and y_mpc.shape[0] >= n + 1:
                ax.plot(t, y_mpc[-n - 1 : -1, output_idx], color="#54a24b", linewidth=1.0, alpha=0.75, label="canonical MPC")
            ax.plot(t, y_sp[-n:, output_idx], color="black", linewidth=1.0, linestyle="--", label="setpoint")
            ax.set_title(f"{latest_row['label']} - {output_labels[output_idx]}")
            ax.grid(alpha=0.2)
            if output_idx == 0:
                ax.set_ylabel("Physical output")
            if row_idx == 0 and output_idx == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")
    axes[-1, 0].set_xlabel("Tail sample index")
    axes[-1, 1].set_xlabel("Tail sample index")
    fig.suptitle("Latest final-tail tracking against setpoint and canonical MPC", fontsize=14)
    fig.savefig(OUT_DIR / "fig_latest_final_tail_tracking.png", dpi=180)
    plt.close(fig)


def make_metric_dashboard(latest_rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in latest_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [
        ("latest_tail_rmse_y1", "Tail RMSE: tray-24 C2 composition", "#4c78a8"),
        ("latest_tail_rmse_y2", "Tail RMSE: tray-85 temperature", "#f58518"),
        ("latest_input_tv", "Total input movement", "#b279a2"),
        ("tail_reward_minus_best_previous", "Tail reward gap to previous best", "#e45756"),
    ]
    for ax, (key, title, color) in zip(axes.reshape(-1), panels):
        values = [float(row.get(key, np.nan)) for row in latest_rows]
        ax.axhline(0.0, color="black", linewidth=0.8) if "gap" in title.lower() else None
        ax.bar(x, values, color=color)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Latest-run tracking and reward diagnostics", fontsize=14)
    fig.savefig(OUT_DIR / "fig_latest_tracking_metric_dashboard.png", dpi=180)
    plt.close(fig)


def make_reward_regime_figure(history_rows: list[dict[str, Any]]) -> None:
    selected = [row for row in history_rows if row["family"] in {"horizon", "dueling", "markov"} and np.isfinite(row["tail_reward"])]
    regimes = ["previous_wide", "current_tight", "other"]
    labels = {"previous_wide": "Previous wider reward", "current_tight": "Current tighter reward", "other": "Other reward"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    for ax, family in zip(axes, ["horizon", "dueling", "markov"]):
        rows = [row for row in selected if row["family"] == family]
        means = []
        counts = []
        for regime in regimes:
            vals = [row["tail_reward"] for row in rows if reward_regime(row.get("reward_signature", {})) == regime]
            means.append(float(np.mean(vals)) if vals else np.nan)
            counts.append(len(vals))
        bars = ax.bar(np.arange(len(regimes)), means, color=["#4c78a8", "#f58518", "#bab0ac"])
        for bar, count in zip(bars, counts):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"n={count}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(regimes)), [labels[r] for r in regimes], rotation=25, ha="right")
        ax.set_title(family)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean tail reward")
    fig.suptitle("Reward-regime evidence for horizon, dueling, and Markov families", fontsize=14)
    fig.savefig(OUT_DIR / "fig_reward_regime_tail_rewards.png", dpi=180)
    plt.close(fig)


def make_root_cause_matrix_figure() -> None:
    families = ["Residual", "Weights", "Horizon", "Dueling", "Markov"]
    causes = ["Reward shaping", "Algorithm mismatch", "Guard/execution mode", "Unlogged variability", "Network/gamma"]
    values = np.asarray(
        [
            [0.25, 0.00, 0.25, 0.75, 0.00],
            [0.00, 1.00, 0.00, 0.25, 0.00],
            [1.00, 0.00, 0.00, 0.25, 0.00],
            [1.00, 0.00, 0.00, 0.25, 0.00],
            [0.00, 0.00, 1.00, 0.25, 0.00],
        ],
        dtype=float,
    )
    annotations = [
        ["uncertain", "", "possible", "likely", "no evidence"],
        ["", "strong", "", "minor", "no evidence"],
        ["strong", "", "", "minor", "no evidence"],
        ["strong", "", "", "minor", "no evidence"],
        ["", "", "strong", "minor", "no evidence"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    im = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(causes)), causes, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(families)), families)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=8)
    ax.set_title("Qualitative root-cause evidence matrix")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Evidence strength")
    fig.savefig(OUT_DIR / "fig_root_cause_evidence_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    baseline = load_pickle(BASELINE_PATH)
    all_history: list[dict[str, Any]] = []
    latest_rows: list[dict[str, Any]] = []
    defaults: dict[str, Any] = {}
    for spec in FAMILIES:
        history = summarize_history(spec)
        all_history.extend(history)
        latest_rows.append(summarize_latest(spec, baseline, history))
        defaults[spec.key] = default_summary(spec)

    write_csv(latest_rows, OUT_DIR / "latest_family_summary.csv")
    write_csv(all_history, OUT_DIR / "history_tail_reward_summary.csv")
    make_figures(latest_rows, all_history)

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_json(
                {
                "baseline_path": BASELINE_PATH.as_posix(),
                "latest_family_summary": latest_rows,
                "defaults": defaults,
                "history_count": len(all_history),
                }
            ),
            handle,
            indent=2,
            allow_nan=False,
            default=json_safe_leaf,
        )


if __name__ == "__main__":
    main()
