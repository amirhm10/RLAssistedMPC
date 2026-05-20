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


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


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
            {
                "baseline_path": BASELINE_PATH.as_posix(),
                "latest_family_summary": latest_rows,
                "defaults": defaults,
                "history_count": len(all_history),
            },
            handle,
            indent=2,
            default=json_safe,
        )


if __name__ == "__main__":
    main()
