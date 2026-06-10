from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "final_combined_agent_attribution_20260610"

POLICY_SOURCE_CODE = 2
TAIL_EPISODES = 20

RUN_SPECS = [
    {
        "plant": "polymer",
        "family": "OF-MPC",
        "role": "baseline",
        "search_dirs": [REPO_ROOT / "Polymer" / "Results" / "mpc_offsetfree_disturb_unified"],
        "fallback_path": REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle",
    },
    {
        "plant": "polymer",
        "family": "horizon",
        "role": "standalone",
        "search_dirs": [
            REPO_ROOT
            / "Polymer"
            / "Results"
            / "horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch"
        ],
    },
    {
        "plant": "polymer",
        "family": "Markov",
        "role": "standalone",
        "search_dirs": [
            REPO_ROOT / "Polymer" / "Results" / "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch"
        ],
    },
    {
        "plant": "polymer",
        "family": "weights",
        "role": "standalone",
        "search_dirs": [REPO_ROOT / "Polymer" / "Results" / "sg_td3_weights_critic_warm3_conservative_disturb_mismatch"],
    },
    {
        "plant": "polymer",
        "family": "residual",
        "role": "standalone",
        "search_dirs": [REPO_ROOT / "Polymer" / "Results" / "sg_td3_residual_critic_warm3_conservative_disturb_mismatch"],
    },
    {
        "plant": "polymer",
        "family": "combined",
        "role": "combined",
        "search_dirs": [
            REPO_ROOT
            / "Polymer"
            / "Results"
            / "combined_disturb_sg__h_sg_dqn_mismatch__markov_sg_td3_mismatch__w_sg_td3_mismatch__r_sg_td3_mismatch_no_rho"
        ],
    },
    {
        "plant": "distillation",
        "family": "OF-MPC",
        "role": "baseline",
        "fallback_path": REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
    },
    {
        "plant": "distillation",
        "family": "horizon",
        "role": "standalone",
        "search_dirs": [REPO_ROOT / "Distillation" / "Results" / "distillation_horizon_sg_disturb_fluctuation"],
    },
    {
        "plant": "distillation",
        "family": "Markov",
        "role": "standalone",
        "search_dirs": [REPO_ROOT / "Distillation" / "Results" / "distillation_markov_sg_disturb_fluctuation"],
    },
    {
        "plant": "distillation",
        "family": "weights",
        "role": "standalone",
        "search_dirs": [REPO_ROOT / "Distillation" / "Results" / "distillation_weights_sg_disturb_fluctuation"],
    },
    {
        "plant": "distillation",
        "family": "residual",
        "role": "standalone",
        "search_dirs": [REPO_ROOT / "Distillation" / "Results" / "distillation_residual_sg_disturb_fluctuation"],
    },
    {
        "plant": "distillation",
        "family": "combined",
        "role": "combined",
        "search_dirs": [REPO_ROOT / "Distillation" / "Results" / "distillation_combined_sg_disturb_fluctuation"],
    },
]

COMBINED_AGENT_SPECS = {
    "horizon": {
        "source_keys": ["horizon_sg_selected_source_log"],
        "advantage_keys": ["horizon_sg_advantage_log"],
        "score_policy_keys": ["horizon_sg_score_policy_log"],
        "score_supervisor_keys": ["horizon_sg_score_supervisor_log"],
        "action_keys": ["horizon_action_trace", "action_trace"],
        "projection_keys": ["horizon_projection_active_log"],
        "loss_prefix": "horizon",
    },
    "Markov": {
        "source_keys": ["markov_sg_selected_source_log", "sg_selected_source_log"],
        "exec_source_keys": ["markov_action_source_log", "rl_action_source_log"],
        "advantage_keys": ["markov_sg_advantage_log", "sg_advantage_log"],
        "score_policy_keys": ["markov_sg_score_policy_log", "sg_score_policy_log"],
        "score_supervisor_keys": ["markov_sg_score_supervisor_log", "sg_score_supervisor_log"],
        "action_keys": ["markov_z_executed_log", "markov_z_log", "z_executed_log", "z_log"],
        "projection_keys": [
            "markov_z_safety_requested_projection_active_log",
            "markov_z_safety_requested_coord_clip_active_log",
            "markov_z_safety_requested_vector_projection_active_log",
            "z_safety_requested_projection_active_log",
            "z_safety_requested_coord_clip_active_log",
            "z_safety_requested_vector_projection_active_log",
        ],
        "loss_prefix": "markov",
    },
    "weights": {
        "source_keys": ["weight_sg_selected_source_log", "sg_selected_source_log"],
        "advantage_keys": ["weight_sg_advantage_log", "sg_advantage_log"],
        "score_policy_keys": ["weight_sg_score_policy_log", "sg_score_policy_log"],
        "score_supervisor_keys": ["weight_sg_score_supervisor_log", "sg_score_supervisor_log"],
        "action_keys": ["weight_log"],
        "projection_keys": [],
        "loss_prefix": "weight",
    },
    "residual": {
        "source_keys": ["residual_sg_selected_source_log", "sg_selected_source_log"],
        "exec_source_keys": ["residual_action_source_log"],
        "advantage_keys": ["residual_sg_advantage_log", "sg_advantage_log"],
        "score_policy_keys": ["residual_sg_score_policy_log", "sg_score_policy_log"],
        "score_supervisor_keys": ["residual_sg_score_supervisor_log", "sg_score_supervisor_log"],
        "action_keys": ["residual_exec_log", "residual_executed_action_raw_log"],
        "projection_keys": [
            "projection_active_log",
            "residual_cap_projection_active_log",
            "residual_guard_active_log",
            "deadband_active_log",
        ],
        "loss_prefix": "residual",
    },
}


def _repo_rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _latest_input_data(search_dirs: list[Path] | None, fallback_path: Path | None = None) -> tuple[Path | None, str]:
    candidates: list[Path] = []
    for directory in search_dirs or []:
        if directory.exists():
            candidates.extend(directory.rglob("input_data.pkl"))
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_mtime), ""
    if fallback_path is not None and fallback_path.exists():
        return fallback_path, ""
    searched = ", ".join(_repo_rel(path) for path in (search_dirs or []))
    fallback = _repo_rel(fallback_path) if fallback_path else ""
    return None, f"missing input_data.pkl; searched={searched}; fallback={fallback}"


def _load_bundle(path: Path) -> dict:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict bundle at {path}, got {type(obj)!r}")
    return obj


def _as_float_array(value, *, flatten: bool = False) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if flatten:
        arr = arr.reshape(-1)
    return arr


def _first_array(bundle: dict, keys: list[str], *, flatten: bool = False) -> tuple[str, np.ndarray | None]:
    for key in keys:
        if key in bundle and bundle[key] is not None:
            return key, _as_float_array(bundle[key], flatten=flatten)
    return "", None


def _finite_mean(values: np.ndarray | None) -> float:
    if values is None or values.size == 0:
        return float("nan")
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_median(values: np.ndarray | None) -> float:
    if values is None or values.size == 0:
        return float("nan")
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _finite_fraction(values: np.ndarray | None) -> float:
    if values is None or values.size == 0:
        return float("nan")
    return float(np.mean(np.isfinite(values)))


def _episode_info(bundle: dict) -> tuple[int, int, int]:
    avg_rewards = _as_float_array(bundle.get("avg_rewards"), flatten=True)
    n_episodes = int(avg_rewards.size) if avg_rewards is not None else 0
    time_in_ep = int(bundle.get("time_in_sub_episodes", 0) or 0)
    warm_step = int(bundle.get("warm_start_step", 0) or 0)
    if warm_step > 0 and time_in_ep > 0:
        warm_episodes = int(math.ceil(warm_step / time_in_ep))
    else:
        warm_episodes = int(bundle.get("warm_start", 10) or 10)
    return n_episodes, time_in_ep, warm_episodes


def _step_masks(bundle: dict, n_steps: int) -> dict[str, np.ndarray]:
    n_episodes, time_in_ep, warm_episodes = _episode_info(bundle)
    if time_in_ep <= 0:
        time_in_ep = max(1, n_steps // max(1, n_episodes))
    steps = np.arange(n_steps)
    tail_start = max(0, n_steps - TAIL_EPISODES * time_in_ep)
    post_warm_start = max(0, warm_episodes * time_in_ep)
    return {
        "all": np.ones(n_steps, dtype=bool),
        "post_warm": steps >= post_warm_start,
        "tail": steps >= tail_start,
    }


def _episode_masks(bundle: dict) -> dict[str, np.ndarray]:
    rewards = _as_float_array(bundle.get("avg_rewards"), flatten=True)
    n = 0 if rewards is None else rewards.size
    _, _, warm_episodes = _episode_info(bundle)
    idx = np.arange(n)
    return {
        "all": np.ones(n, dtype=bool),
        "post_warm": idx >= warm_episodes,
        "tail": idx >= max(0, n - TAIL_EPISODES),
    }


def _tracking_array(bundle: dict) -> np.ndarray | None:
    for key in ("delta_y_storage", "tracking_error_raw_log", "tracking_error_log"):
        arr = _as_float_array(bundle.get(key))
        if arr is not None and arr.ndim == 2:
            return arr
    return None


def _input_move_array(bundle: dict) -> np.ndarray | None:
    for key in ("delta_u_storage", "delat_u_storage"):
        arr = _as_float_array(bundle.get(key))
        if arr is not None and arr.ndim == 2:
            return arr
    u = _as_float_array(bundle.get("u"))
    if u is None:
        u = _as_float_array(bundle.get("u_step_full"))
    if u is not None and u.ndim == 2 and u.shape[0] > 1:
        return np.diff(u, axis=0)
    return None


def _run_metrics(spec: dict, path: Path | None, bundle: dict | None, missing_reason: str) -> dict:
    row = {
        "plant": spec["plant"],
        "family": spec["family"],
        "role": spec["role"],
        "status": "missing" if bundle is None else "ok",
        "bundle_path": _repo_rel(path),
        "missing_reason": missing_reason,
        "reward_metric_source": "saved avg_rewards from bundle",
    }
    if bundle is None:
        return row

    rewards = _as_float_array(bundle.get("avg_rewards"), flatten=True)
    ep_masks = _episode_masks(bundle)
    row.update(
        {
            "n_episodes": int(rewards.size) if rewards is not None else 0,
            "n_steps": int(bundle.get("nFE", 0) or (len(bundle.get("rewards_step", [])) if bundle.get("rewards_step") is not None else 0)),
            "time_in_sub_episodes": int(bundle.get("time_in_sub_episodes", 0) or 0),
            "tail_reward": _finite_mean(None if rewards is None else rewards[ep_masks["tail"]]),
            "final_reward": float(rewards[-1]) if rewards is not None and rewards.size else float("nan"),
            "worst_post_warm_reward": _finite_mean(None),
            "negative_post_warm_episodes": "",
        }
    )
    if rewards is not None and rewards.size:
        post = rewards[ep_masks["post_warm"]]
        row["worst_post_warm_reward"] = float(np.nanmin(post)) if post.size else float("nan")
        row["negative_post_warm_episodes"] = int(np.sum(post < 0.0)) if post.size else 0

    tracking = _tracking_array(bundle)
    if tracking is not None and tracking.size:
        masks = _step_masks(bundle, tracking.shape[0])
        tail_tracking = tracking[masks["tail"], :]
        for idx in range(min(2, tracking.shape[1])):
            channel = tail_tracking[:, idx]
            row[f"tail_tracking_mae_y{idx + 1}"] = float(np.nanmean(np.abs(channel)))
            row[f"tail_tracking_rmse_y{idx + 1}"] = float(np.sqrt(np.nanmean(channel**2)))
            row[f"tail_tracking_max_abs_y{idx + 1}"] = float(np.nanmax(np.abs(channel)))
    moves = _input_move_array(bundle)
    if moves is not None and moves.size:
        masks = _step_masks(bundle, moves.shape[0])
        tail_moves = moves[masks["tail"], :]
        row["tail_mean_abs_delta_u"] = float(np.nanmean(np.abs(tail_moves)))
        row["tail_total_variation_u"] = float(np.nansum(np.abs(tail_moves)))

    snapshots = bundle.get("replay_buffer_snapshots") or {}
    if not snapshots and bundle.get("replay_buffer_snapshot") is not None:
        snapshots = {str(spec["family"]).lower(): bundle["replay_buffer_snapshot"]}
    row["replay_snapshot_agents"] = ",".join(sorted(snapshots)) if isinstance(snapshots, dict) else ""
    row["has_replay_snapshot"] = bool(snapshots)
    row["has_input_data"] = True
    return row


def _policy_fraction(source: np.ndarray | None, mask: np.ndarray | None = None) -> float:
    if source is None or source.size == 0:
        return float("nan")
    values = source.reshape(-1)
    if mask is not None:
        values = values[: mask.size][mask[: values.size]]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values == POLICY_SOURCE_CODE))


def _source_entropy(values: np.ndarray | None, mask: np.ndarray | None = None) -> float:
    if values is None or values.size == 0:
        return float("nan")
    arr = values.reshape(-1)
    if mask is not None:
        arr = arr[: mask.size][mask[: arr.size]]
    if arr.size == 0:
        return float("nan")
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / max(1, counts.sum())
    return float(-np.sum(probs * np.log(probs + 1.0e-12)))


def _any_fraction(bundle: dict, keys: list[str], masks: dict[str, np.ndarray]) -> float:
    series = []
    for key in keys:
        arr = _as_float_array(bundle.get(key), flatten=True)
        if arr is not None and arr.size:
            series.append(arr)
    if not series:
        return float("nan")
    n = min(len(item) for item in series)
    stacked = np.vstack([item[:n] for item in series])
    active = np.any(stacked > 0.5, axis=0)
    mask = masks["tail"][:n]
    return float(np.mean(active[mask])) if np.any(mask) else float("nan")


def _action_authority(bundle: dict, agent: str, spec: dict, masks: dict[str, np.ndarray]) -> tuple[float, str]:
    _, action = _first_array(bundle, spec["action_keys"])
    if action is None or action.size == 0:
        return float("nan"), "missing action log"
    if action.ndim == 1:
        action = action.reshape(-1, 1)
    n = min(action.shape[0], masks["tail"].size)
    tail = action[:n, :][masks["tail"][:n], :]
    if tail.size == 0:
        return float("nan"), "empty tail"

    if agent == "horizon":
        entropy = _source_entropy(tail.reshape(-1))
        return entropy, "tail action entropy"
    if agent == "Markov":
        z_bound = float(bundle.get("markov_z_bound", bundle.get("z_bound", np.nan)))
        denom = z_bound if np.isfinite(z_bound) and z_bound > 0 else 1.0
        return float(np.nanmean(np.linalg.norm(tail, axis=1)) / denom), "tail mean z-norm / z_bound"
    if agent == "weights":
        low = _as_float_array(bundle.get("weight_low_coef"), flatten=True)
        high = _as_float_array(bundle.get("weight_high_coef"), flatten=True)
        denom = 1.0
        if low is not None and high is not None and low.size and high.size:
            denom = float(np.nanmax(np.abs(np.concatenate([low - 1.0, high - 1.0]))))
            denom = denom if denom > 0 else 1.0
        return float(np.nanmean(np.abs(tail - 1.0)) / denom), "tail mean abs multiplier distance / range"
    if agent == "residual":
        low = _as_float_array(bundle.get("residual_low_coef"), flatten=True)
        high = _as_float_array(bundle.get("residual_high_coef"), flatten=True)
        denom = 1.0
        if low is not None and high is not None and low.size and high.size:
            denom = float(np.nanmax(np.abs(np.concatenate([low, high]))))
            denom = denom if denom > 0 else 1.0
        return float(np.nanmean(np.linalg.norm(tail, axis=1)) / denom), "tail mean residual norm / bound"
    return float("nan"), "unknown agent"


def _loss_stats(bundle: dict, prefix: str) -> dict:
    actor = _as_float_array(bundle.get(f"{prefix}_actor_losses"), flatten=True)
    critic = _as_float_array(bundle.get(f"{prefix}_critic_losses"), flatten=True)
    if actor is None:
        actor = _as_float_array(bundle.get("actor_losses"), flatten=True)
    if critic is None:
        critic = _as_float_array(bundle.get("critic_losses"), flatten=True)
    tail_slice = slice(max(0, 0 if critic is None else critic.size - 500), None)
    return {
        "actor_loss_finite_frac": _finite_fraction(actor),
        "critic_loss_finite_frac": _finite_fraction(critic),
        "tail_median_critic_loss": _finite_median(None if critic is None else critic[tail_slice]),
    }


def _replay_rows(bundle: dict, agent_key: str) -> int | str:
    snapshots = bundle.get("replay_buffer_snapshots") or {}
    if isinstance(snapshots, dict):
        normalized = {str(key).lower(): value for key, value in snapshots.items()}
        snapshot = normalized.get(agent_key.lower())
        if snapshot is not None and isinstance(snapshot, dict) and snapshot.get("states") is not None:
            return int(np.asarray(snapshot["states"]).shape[0])
    snapshot = bundle.get("replay_buffer_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("states") is not None:
        return int(np.asarray(snapshot["states"]).shape[0])
    return ""


def _combined_agent_rows(plant: str, path: Path | None, bundle: dict | None, missing_reason: str) -> list[dict]:
    rows = []
    if bundle is None:
        for agent in COMBINED_AGENT_SPECS:
            rows.append(
                {
                    "plant": plant,
                    "combined_bundle_path": _repo_rel(path),
                    "agent": agent,
                    "status": "missing",
                    "missing_reason": missing_reason,
                }
            )
        return rows

    n_steps = int(bundle.get("nFE", 0) or len(bundle.get("rewards_step", [])) or 0)
    masks = _step_masks(bundle, n_steps)
    for agent, spec in COMBINED_AGENT_SPECS.items():
        source_key, source = _first_array(bundle, spec.get("source_keys", []), flatten=True)
        exec_key, exec_source = _first_array(bundle, spec.get("exec_source_keys", []), flatten=True)
        advantage_key, advantage = _first_array(bundle, spec["advantage_keys"], flatten=True)
        score_policy_key, score_policy = _first_array(bundle, spec["score_policy_keys"], flatten=True)
        score_supervisor_key, score_supervisor = _first_array(bundle, spec["score_supervisor_keys"], flatten=True)
        authority, authority_label = _action_authority(bundle, agent, spec, masks)
        row = {
            "plant": plant,
            "combined_bundle_path": _repo_rel(path),
            "agent": agent,
            "status": "ok" if source is not None or advantage is not None else "missing_logs",
            "missing_reason": "" if source is not None or advantage is not None else "missing SG source/advantage logs",
            "source_log": source_key,
            "exec_source_log": exec_key,
            "advantage_log": advantage_key,
            "policy_frac_all": _policy_fraction(source, masks["all"]),
            "policy_frac_post_warm": _policy_fraction(source, masks["post_warm"]),
            "policy_frac_tail": _policy_fraction(source, masks["tail"]),
            "exec_policy_frac_tail": _policy_fraction(exec_source, masks["tail"]),
            "tail_median_gate_advantage": _finite_median(
                None if advantage is None else advantage[: masks["tail"].size][masks["tail"][: advantage.size]]
            ),
            "tail_mean_gate_advantage": _finite_mean(
                None if advantage is None else advantage[: masks["tail"].size][masks["tail"][: advantage.size]]
            ),
            "tail_mean_policy_score": _finite_mean(
                None if score_policy is None else score_policy[: masks["tail"].size][masks["tail"][: score_policy.size]]
            ),
            "tail_mean_supervisor_score": _finite_mean(
                None
                if score_supervisor is None
                else score_supervisor[: masks["tail"].size][masks["tail"][: score_supervisor.size]]
            ),
            "tail_action_authority": authority,
            "tail_action_authority_label": authority_label,
            "tail_safety_burden_frac": _any_fraction(bundle, spec.get("projection_keys", []), masks),
            "replay_rows": _replay_rows(bundle, "markov" if agent == "Markov" else agent),
        }
        row.update(_loss_stats(bundle, spec["loss_prefix"]))
        rows.append(row)
    return rows


def _logging_gap_rows(run_rows: list[dict], bundles_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    for row in run_rows:
        bundle = bundles_by_path.get(row.get("bundle_path", ""))
        required = [
            ("input_data.pkl", lambda row, bundle: bool(row.get("bundle_path"))),
            ("avg_rewards", lambda row, bundle: bundle is not None and bundle.get("avg_rewards") is not None),
            ("tracking errors", lambda row, bundle: bundle is not None and _tracking_array(bundle) is not None),
        ]
        if row["role"] != "baseline":
            required.extend(
                [
                    ("replay snapshot", lambda row, bundle: bool(row.get("has_replay_snapshot"))),
                    (
                        "SG source logs",
                        lambda row, bundle: bundle is not None and any("sg_selected_source_log" in key for key in bundle),
                    ),
                ]
            )
        if row["role"] == "combined":
            required.append(("config snapshot", lambda row, bundle: bundle is not None and bundle.get("config_snapshot") is not None))
        for item, predicate in required:
            ok = bool(predicate(row, bundle))
            if ok:
                continue
            rows.append(
                {
                    "plant": row["plant"],
                    "family": row["family"],
                    "role": row["role"],
                    "missing_item": item,
                    "bundle_path": row.get("bundle_path", ""),
                    "reason": row.get("missing_reason", "") or f"{item} not found in saved bundle",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_source_fractions(rows: list[dict]) -> Path:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("policy_frac_tail") not in ("", None)]
    labels = [f"{row['plant']}\n{row['agent']}" for row in ok_rows]
    values = [float(row["policy_frac_tail"]) for row in ok_rows]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    if values:
        ax.bar(np.arange(len(values)), values, color="#4E79A7")
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("tail policy execution fraction")
    ax.set_title("Combined-run agent policy source fractions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "fig_final_combined_source_fractions.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_gate_advantages(rows: list[dict]) -> Path:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("tail_median_gate_advantage") not in ("", None)
    ]
    labels = [f"{row['plant']}\n{row['agent']}" for row in ok_rows]
    values = [float(row["tail_median_gate_advantage"]) for row in ok_rows]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    if values:
        ax.bar(np.arange(len(values)), values, color="#59A14F")
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_ylabel("median tail gate advantage")
    ax.set_title("Policy minus supervisor score by agent")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "fig_final_combined_gate_advantages.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_action_authority(rows: list[dict]) -> Path:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("tail_action_authority") not in ("", None)
    ]
    labels = [f"{row['plant']}\n{row['agent']}" for row in ok_rows]
    values = [float(row["tail_action_authority"]) for row in ok_rows]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    if values:
        ax.bar(np.arange(len(values)), values, color="#F28E2B")
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("tail authority diagnostic")
    ax.set_title("Agent action-authority utilization")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / "fig_final_combined_action_authority.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_metric_heatmap(rows: list[dict]) -> Path:
    families = ["OF-MPC", "horizon", "Markov", "weights", "residual", "combined"]
    plants = ["polymer", "distillation"]
    baseline = {
        row["plant"]: float(row["tail_reward"])
        for row in rows
        if row.get("family") == "OF-MPC" and row.get("status") == "ok" and row.get("tail_reward") not in ("", None)
    }
    values = np.full((len(plants), len(families)), np.nan)
    for row in rows:
        if row.get("status") != "ok" or row.get("tail_reward") in ("", None):
            continue
        if row["plant"] not in plants or row["family"] not in families:
            continue
        b = baseline.get(row["plant"], np.nan)
        values[plants.index(row["plant"]), families.index(row["family"])] = float(row["tail_reward"]) - b

    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    finite = values[np.isfinite(values)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(plants)))
    ax.set_yticklabels(plants)
    for i in range(len(plants)):
        for j in range(len(families)):
            text = "NA" if not np.isfinite(values[i, j]) else f"{values[i, j]:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    ax.set_title("Saved tail reward delta vs OF-MPC")
    fig.colorbar(image, ax=ax, label="saved tail reward delta")
    fig.tight_layout()
    out = OUT_DIR / "fig_final_tail_reward_delta_heatmap.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    attribution_rows: list[dict] = []
    bundles_by_path: dict[str, dict] = {}
    selected_paths: dict[str, str] = {}

    for spec in RUN_SPECS:
        path, missing = _latest_input_data(spec.get("search_dirs"), spec.get("fallback_path"))
        bundle = None
        if path is not None:
            try:
                bundle = _load_bundle(path)
                bundles_by_path[_repo_rel(path)] = bundle
            except Exception as exc:  # noqa: BLE001 - report analysis should keep going.
                missing = f"failed to load bundle: {exc}"
                bundle = None
        selected_paths[f"{spec['plant']}::{spec['family']}"] = _repo_rel(path)
        run_rows.append(_run_metrics(spec, path, bundle, missing))
        if spec["role"] == "combined":
            attribution_rows.extend(_combined_agent_rows(spec["plant"], path, bundle, missing))

    baseline_tail = {
        row["plant"]: float(row["tail_reward"])
        for row in run_rows
        if row.get("family") == "OF-MPC" and row.get("status") == "ok" and row.get("tail_reward") not in ("", None)
    }
    for row in run_rows:
        if row.get("status") == "ok" and row.get("tail_reward") not in ("", None):
            row["tail_reward_delta_vs_ofmpc"] = float(row["tail_reward"]) - baseline_tail.get(row["plant"], float("nan"))

    gap_rows = _logging_gap_rows(run_rows, bundles_by_path)
    if not gap_rows:
        gap_rows.append({"plant": "all", "family": "all", "role": "all", "missing_item": "none", "reason": "no gaps found"})

    run_csv = OUT_DIR / "final_run_journal_table.csv"
    attribution_csv = OUT_DIR / "final_agent_attribution_summary.csv"
    gap_csv = OUT_DIR / "final_logging_gap_table.csv"
    _write_csv(run_csv, run_rows)
    _write_csv(attribution_csv, attribution_rows)
    _write_csv(gap_csv, gap_rows)

    figures = [
        _plot_source_fractions(attribution_rows),
        _plot_gate_advantages(attribution_rows),
        _plot_action_authority(attribution_rows),
        _plot_metric_heatmap(run_rows),
    ]
    summary_json = OUT_DIR / "final_combined_agent_attribution_summary.json"
    with summary_json.open("w") as handle:
        json.dump(
            {
                "selected_paths": selected_paths,
                "run_table": _repo_rel(run_csv),
                "attribution_table": _repo_rel(attribution_csv),
                "logging_gap_table": _repo_rel(gap_csv),
                "figures": [_repo_rel(path) for path in figures],
                "notes": [
                    "Attribution rows are diagnostic, not causal, unless leave-one-agent-out reruns exist.",
                    "Missing rows are retained explicitly to avoid inferring unavailable bundle contents.",
                    "Run-table reward values use saved avg_rewards from each bundle; journal-final comparisons should use a common rescoring pass.",
                ],
            },
            handle,
            indent=2,
        )

    print(f"Wrote {_repo_rel(run_csv)}")
    print(f"Wrote {_repo_rel(attribution_csv)}")
    print(f"Wrote {_repo_rel(gap_csv)}")
    print(f"Wrote {_repo_rel(summary_json)}")
    for figure in figures:
        print(f"Wrote {_repo_rel(figure)}")


if __name__ == "__main__":
    main()
