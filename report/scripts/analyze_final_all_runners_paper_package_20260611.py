from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "final_all_runners_paper_package_20260611"
REPORT_PATH = REPO_ROOT / "report" / "final_all_runners_paper_package_2026_06_11.md"


RUNS = [
    {
        "plant": "polymer",
        "family": "OF-MPC",
        "role": "baseline",
        "label": "OF-MPC",
        "path": REPO_ROOT / "Polymer" / "Results" / "mpc_offsetfree_disturb_unified" / "20260608_143324" / "input_data.pkl",
    },
    {
        "plant": "polymer",
        "family": "horizon",
        "role": "standalone",
        "label": "Horizon SG-DQN",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch"
        / "20260608_151338"
        / "input_data.pkl",
    },
    {
        "plant": "polymer",
        "family": "weights",
        "role": "standalone",
        "label": "Weights SG-TD3",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_weights_critic_warm3_conservative_disturb_mismatch"
        / "20260608_152025"
        / "input_data.pkl",
    },
    {
        "plant": "polymer",
        "family": "residual",
        "role": "standalone",
        "label": "Residual SG-TD3",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_residual_critic_warm3_conservative_disturb_mismatch"
        / "20260610_131706"
        / "input_data.pkl",
    },
    {
        "plant": "polymer",
        "family": "Markov",
        "role": "standalone",
        "label": "Markov SG-TD3",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch"
        / "20260610_170822"
        / "input_data.pkl",
    },
    {
        "plant": "polymer",
        "family": "combined",
        "role": "combined",
        "label": "Combined SG",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "combined_disturb_sg__h_sg_dqn_mismatch__markov_sg_td3_mismatch__w_sg_td3_mismatch__r_sg_td3_mismatch_no_rho"
        / "20260611_045747"
        / "input_data.pkl",
    },
    {
        "plant": "distillation",
        "family": "OF-MPC",
        "role": "baseline",
        "label": "OF-MPC",
        "path": REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
    },
    {
        "plant": "distillation",
        "family": "horizon",
        "role": "standalone",
        "label": "Horizon SG-DQN",
        "path": REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_sg_disturb_fluctuation"
        / "20260609_195213"
        / "input_data.pkl",
    },
    {
        "plant": "distillation",
        "family": "weights",
        "role": "standalone",
        "label": "Weights SG-TD3",
        "path": REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_disturb_fluctuation"
        / "20260609_203050"
        / "input_data.pkl",
    },
    {
        "plant": "distillation",
        "family": "residual",
        "role": "standalone",
        "label": "Residual SG-TD3",
        "path": REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sg_disturb_fluctuation"
        / "20260609_203500"
        / "input_data.pkl",
    },
    {
        "plant": "distillation",
        "family": "Markov",
        "role": "standalone",
        "label": "Markov SG-TD3",
        "path": REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_sg_disturb_fluctuation"
        / "20260611_132048"
        / "input_data.pkl",
    },
    {
        "plant": "distillation",
        "family": "combined",
        "role": "combined",
        "label": "Combined SG",
        "path": REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_combined_sg_disturb_fluctuation"
        / "20260611_095216"
        / "input_data.pkl",
    },
]


OUTPUT_LABELS = {
    "polymer": ("eta", "T"),
    "distillation": ("x24", "T85"),
}

INPUT_LABELS = {
    "polymer": ("Qc", "Qm"),
    "distillation": ("reflux", "reboiler"),
}


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def arr(bundle: dict[str, Any], key: str, *, dtype=float) -> np.ndarray:
    value = bundle.get(key)
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def finite_min(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmin(values))


def finite_max(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmax(values))


def finite_fraction(values: np.ndarray, code: int) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values == code))


def bool_fraction(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values.astype(bool)))


def norm_mean(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return float(np.nanmean(np.linalg.norm(values, axis=1)))


def norm_q95(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return float(np.nanquantile(np.linalg.norm(values, axis=1), 0.95))


def as_physical_setpoint(bundle: dict[str, Any]) -> np.ndarray:
    y_sp = arr(bundle, "y_sp")
    if y_sp.size == 0:
        return y_sp
    if y_sp.ndim == 1:
        y_sp = y_sp.reshape(-1, 1)
    y_ref = arr(bundle, "y")
    if y_ref.size == 0:
        y_ref = arr(bundle, "y_rl")
    # Saved RL runners use scaled-deviation setpoints. Physical outputs are much
    # larger, so convert setpoints back when the magnitude clearly differs.
    if y_ref.size and np.nanmax(np.abs(y_sp)) < 0.25 * max(np.nanmax(np.abs(y_ref)), 1.0):
        data_min = arr(bundle, "data_min").reshape(-1)
        data_max = arr(bundle, "data_max").reshape(-1)
        steady = bundle.get("steady_states", {}) or {}
        y_ss = np.asarray(steady.get("y_ss", []), float).reshape(-1)
        n_outputs = y_sp.shape[1]
        if data_min.size >= 2 * n_outputs and y_ss.size >= n_outputs:
            out_min = data_min[-n_outputs:]
            out_max = data_max[-n_outputs:]
            y_ss_scaled = (y_ss - out_min) / (out_max - out_min)
            return (y_sp + y_ss_scaled) * (out_max - out_min) + out_min
    return y_sp


def primary_y(bundle: dict[str, Any], *, baseline: bool) -> np.ndarray:
    if baseline:
        for key in ("y", "y_mpc", "y_rl"):
            y = arr(bundle, key)
            if y.size:
                return y
    for key in ("y_rl", "y", "y_mpc"):
        y = arr(bundle, key)
        if y.size:
            return y
    return np.asarray([], float)


def primary_u(bundle: dict[str, Any], *, baseline: bool) -> np.ndarray:
    if baseline:
        for key in ("u", "u_mpc", "u_rl"):
            u = arr(bundle, key)
            if u.size:
                return u
    for key in ("u_rl", "u", "u_mpc"):
        u = arr(bundle, key)
        if u.size:
            return u
    return np.asarray([], float)


def window_slices(bundle: dict[str, Any], n_episodes: int, n_steps: int) -> dict[str, slice]:
    steps_per_episode = int(bundle.get("time_in_sub_episodes") or (n_steps // max(n_episodes, 1)))
    warm_start_step = int(bundle.get("warm_start_step") or 10 * steps_per_episode)
    warm_episode = min(n_episodes, int(math.ceil(warm_start_step / max(steps_per_episode, 1))))
    first_live_episode = min(n_episodes, warm_episode + 3)
    return {
        "tail_ep": slice(max(0, n_episodes - 20), n_episodes),
        "post_ep": slice(warm_episode, n_episodes),
        "first_live_ep": slice(first_live_episode, min(n_episodes, first_live_episode + 20)),
        "tail_step": slice(max(0, n_steps - 20 * steps_per_episode), n_steps),
        "post_step": slice(warm_start_step, n_steps),
        "first_live_step": slice(
            first_live_episode * steps_per_episode,
            min(n_steps, (first_live_episode + 20) * steps_per_episode),
        ),
    }


def reward_vector(bundle: dict[str, Any]) -> np.ndarray:
    values = arr(bundle, "avg_rewards").reshape(-1)
    if values.size == 0:
        values = arr(bundle, "avg_rewards_mpc").reshape(-1)
    return values


def tracking_and_input_metrics(bundle: dict[str, Any], sl: dict[str, slice], *, baseline: bool) -> dict[str, float]:
    y = primary_y(bundle, baseline=baseline)
    y_sp = as_physical_setpoint(bundle)
    u = primary_u(bundle, baseline=baseline)
    out: dict[str, float] = {}
    if y.size and y_sp.size:
        y_aligned = y[1 : y_sp.shape[0] + 1, :]
        err_tail = np.asarray(y_aligned[sl["tail_step"], :] - y_sp[sl["tail_step"], :], float)
        err_full = np.asarray(y_aligned[sl["post_step"], :] - y_sp[sl["post_step"], :], float)
        for idx, suffix in enumerate(("y1", "y2")):
            out[f"tail_mae_{suffix}"] = float(np.nanmean(np.abs(err_tail[:, idx])))
            out[f"tail_rmse_{suffix}"] = float(np.sqrt(np.nanmean(err_tail[:, idx] ** 2)))
            out[f"tail_max_abs_{suffix}"] = float(np.nanmax(np.abs(err_tail[:, idx])))
            out[f"postwarm_iae_{suffix}"] = float(np.nansum(np.abs(err_full[:, idx])))
    else:
        for suffix in ("y1", "y2"):
            out[f"tail_mae_{suffix}"] = float("nan")
            out[f"tail_rmse_{suffix}"] = float("nan")
            out[f"tail_max_abs_{suffix}"] = float("nan")
            out[f"postwarm_iae_{suffix}"] = float("nan")
    if u.size:
        u_tail = np.asarray(u[sl["tail_step"], :], float)
        if u_tail.shape[0] > 1:
            du_tail = np.diff(u_tail, axis=0)
            out["tail_mean_abs_du_u1"] = float(np.nanmean(np.abs(du_tail[:, 0])))
            out["tail_mean_abs_du_u2"] = float(np.nanmean(np.abs(du_tail[:, 1])))
            out["tail_total_variation_u"] = float(np.nansum(np.abs(du_tail)))
        else:
            out["tail_mean_abs_du_u1"] = float("nan")
            out["tail_mean_abs_du_u2"] = float("nan")
            out["tail_total_variation_u"] = float("nan")
    else:
        out["tail_mean_abs_du_u1"] = float("nan")
        out["tail_mean_abs_du_u2"] = float("nan")
        out["tail_total_variation_u"] = float("nan")
    return out


def source_metrics(bundle: dict[str, Any], family: str, role: str, sl: dict[str, slice]) -> dict[str, float]:
    family_key = family.lower()
    out = {
        "tail_policy_fraction": float("nan"),
        "first_live_policy_fraction": float("nan"),
        "tail_supervisor_fraction": float("nan"),
        "tail_fallback_fraction": float("nan"),
        "tail_gate_advantage": float("nan"),
        "tail_action_authority": float("nan"),
        "tail_action_authority_q95": float("nan"),
        "tail_projection_fraction": float("nan"),
    }
    if family_key == "of-mpc":
        return out
    prefix = ""
    if role == "combined" and family_key == "combined":
        return out
    if family_key in {"horizon", "weights", "residual"}:
        prefix = "" if role == "standalone" else f"{family_key}_"
        sg = arr(bundle, f"{prefix}sg_selected_source_log", dtype=int).reshape(-1)
        if sg.size == 0 and role == "standalone":
            sg = arr(bundle, "sg_selected_source_log", dtype=int).reshape(-1)
        out["tail_policy_fraction"] = finite_fraction(sg[sl["tail_step"]], 2)
        out["first_live_policy_fraction"] = finite_fraction(sg[sl["first_live_step"]], 2)
        out["tail_supervisor_fraction"] = finite_fraction(sg[sl["tail_step"]], 1)
        adv = arr(bundle, f"{prefix}sg_advantage_log").reshape(-1)
        if adv.size == 0 and role == "standalone":
            adv = arr(bundle, "sg_advantage_log").reshape(-1)
        out["tail_gate_advantage"] = finite_mean(adv[sl["tail_step"]]) if adv.size else float("nan")
        if family_key == "horizon":
            action = arr(bundle, "horizon_action_trace" if role == "standalone" else "horizon_trace")
            out["tail_action_authority"] = float("nan") if action.size == 0 else finite_mean(action[sl["tail_step"]])
        elif family_key == "weights":
            w = arr(bundle, "weight_log" if role == "standalone" else "weight_log")
            if w.size:
                out["tail_action_authority"] = norm_mean(w[sl["tail_step"]] - 1.0)
                out["tail_action_authority_q95"] = norm_q95(w[sl["tail_step"]] - 1.0)
        elif family_key == "residual":
            r = arr(bundle, "delta_u_res_exec_log" if role == "standalone" else "residual_exec_log")
            if r.size == 0:
                r = arr(bundle, "residual_exec_log")
            if r.size:
                out["tail_action_authority"] = norm_mean(r[sl["tail_step"]])
                out["tail_action_authority_q95"] = norm_q95(r[sl["tail_step"]])
            proj = arr(bundle, "projection_due_to_authority_log", dtype=int).reshape(-1)
            out["tail_projection_fraction"] = bool_fraction(proj[sl["tail_step"]]) if proj.size else float("nan")
        return out
    if family_key == "markov":
        action_src_key = "rl_action_source_log" if role == "standalone" else "markov_action_source_log"
        action_src = arr(bundle, action_src_key, dtype=int).reshape(-1)
        sg_key = "sg_selected_source_log" if role == "standalone" else "markov_sg_selected_source_log"
        sg = arr(bundle, sg_key, dtype=int).reshape(-1)
        out["tail_policy_fraction"] = finite_fraction(action_src[sl["tail_step"]], 2)
        out["first_live_policy_fraction"] = finite_fraction(action_src[sl["first_live_step"]], 2)
        out["tail_supervisor_fraction"] = (
            finite_fraction(action_src[sl["tail_step"]], 6) + finite_fraction(action_src[sl["tail_step"]], 7)
        )
        out["tail_fallback_fraction"] = finite_fraction(action_src[sl["tail_step"]], 8)
        adv = arr(bundle, "sg_advantage_log" if role == "standalone" else "markov_sg_advantage_log").reshape(-1)
        out["tail_gate_advantage"] = finite_mean(adv[sl["tail_step"]]) if adv.size else float("nan")
        z = arr(bundle, "z_executed_log" if role == "standalone" else "markov_z_executed_log")
        if z.size:
            out["tail_action_authority"] = norm_mean(z[sl["tail_step"]])
            out["tail_action_authority_q95"] = norm_q95(z[sl["tail_step"]])
        proj = arr(
            bundle,
            "z_safety_requested_projection_active_log"
            if role == "standalone"
            else "markov_z_safety_requested_projection_active_log",
            dtype=int,
        ).reshape(-1)
        out["tail_projection_fraction"] = bool_fraction(proj[sl["tail_step"]]) if proj.size else float("nan")
        if sg.size:
            out["tail_sg_policy_fraction"] = finite_fraction(sg[sl["tail_step"]], 2)
        return out
    return out


def replay_metrics(bundle: dict[str, Any], family: str, role: str) -> dict[str, Any]:
    out = {
        "replay_size": "",
        "replay_capacity": "",
        "replay_available": False,
        "actor_loss_finite_frac": float("nan"),
        "critic_loss_finite_frac": float("nan"),
    }
    if role == "combined":
        return out
    snap = bundle.get("replay_buffer_snapshot")
    if isinstance(snap, dict):
        out["replay_size"] = snap.get("size", "")
        out["replay_capacity"] = snap.get("capacity", "")
        out["replay_available"] = True
    actor = arr(bundle, "actor_losses").reshape(-1)
    critic = arr(bundle, "critic_losses").reshape(-1)
    if actor.size:
        out["actor_loss_finite_frac"] = float(np.mean(np.isfinite(actor)))
    if critic.size:
        out["critic_loss_finite_frac"] = float(np.mean(np.isfinite(critic)))
    return out


def summarize_run(meta: dict[str, Any], baselines: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(meta["path"])
    if not path.exists():
        row = {**meta, "status": "missing", "missing_reason": "input_data.pkl not found", "path": rel(path)}
        return row, {}
    bundle = load_pickle(path)
    rewards = reward_vector(bundle)
    n_steps = int(bundle.get("nFE") or arr(bundle, "y_sp").shape[0])
    sl = window_slices(bundle, rewards.size, n_steps)
    baseline = bool(meta["role"] == "baseline")
    tail_reward = finite_mean(rewards[sl["tail_ep"]])
    row: dict[str, Any] = {
        "plant": meta["plant"],
        "family": meta["family"],
        "role": meta["role"],
        "label": meta["label"],
        "status": "ok",
        "timestamp": path.parent.name,
        "path": rel(path),
        "episodes": int(rewards.size),
        "steps": int(n_steps),
        "tail_reward": tail_reward,
        "tail_reward_delta_vs_ofmpc": tail_reward - baselines.get(meta["plant"], tail_reward),
        "final_reward": float(rewards[-1]) if rewards.size else float("nan"),
        "worst_postwarm_reward": finite_min(rewards[sl["post_ep"]]),
        "first_live_reward": finite_mean(rewards[sl["first_live_ep"]]),
        "negative_postwarm_episodes": int(np.sum(rewards[sl["post_ep"]] < 0.0)) if rewards.size else 0,
        "notebook_source": bundle.get("notebook_source", ""),
        "run_mode": bundle.get("run_mode", ""),
        "disturbance_profile": list((bundle.get("disturbance_profile") or {}).keys())
        if isinstance(bundle.get("disturbance_profile"), dict)
        else bundle.get("disturbance_profile", ""),
    }
    row.update(tracking_and_input_metrics(bundle, sl, baseline=baseline))
    row.update(source_metrics(bundle, str(meta["family"]), str(meta["role"]), sl))
    row.update(replay_metrics(bundle, str(meta["family"]), str(meta["role"])))
    cfg = bundle.get("config_snapshot")
    row["has_config_snapshot"] = isinstance(cfg, dict)
    row["has_agent_config_snapshot"] = bool(isinstance(cfg, dict) and isinstance(cfg.get("agent_config_snapshot"), dict))
    return row, bundle


def summarize_combined_agents(plant: str, bundle: dict[str, Any], baseline_tail: float) -> list[dict[str, Any]]:
    rewards = reward_vector(bundle)
    n_steps = int(bundle.get("nFE") or arr(bundle, "y_sp").shape[0])
    sl = window_slices(bundle, rewards.size, n_steps)
    rows = []
    for agent in ("horizon", "markov", "weights", "residual"):
        family = "Markov" if agent == "markov" else agent
        metrics = source_metrics(bundle, family, "combined", sl)
        if agent == "horizon":
            sg = arr(bundle, "horizon_sg_selected_source_log", dtype=int).reshape(-1)
            metrics["tail_policy_fraction"] = finite_fraction(sg[sl["tail_step"]], 2)
            metrics["tail_supervisor_fraction"] = finite_fraction(sg[sl["tail_step"]], 1)
            metrics["tail_fallback_fraction"] = finite_fraction(sg[sl["tail_step"]], 3)
            adv = arr(bundle, "horizon_sg_advantage_log").reshape(-1)
            metrics["tail_gate_advantage"] = finite_mean(adv[sl["tail_step"]]) if adv.size else float("nan")
        if agent == "markov":
            metrics["tail_sg_policy_fraction"] = metrics.get("tail_sg_policy_fraction", float("nan"))
        elif agent == "weights":
            sg = arr(bundle, "weight_sg_selected_source_log", dtype=int).reshape(-1)
            metrics["tail_policy_fraction"] = finite_fraction(sg[sl["tail_step"]], 2)
            metrics["tail_supervisor_fraction"] = finite_fraction(sg[sl["tail_step"]], 1)
            adv = arr(bundle, "weight_sg_advantage_log").reshape(-1)
            metrics["tail_gate_advantage"] = finite_mean(adv[sl["tail_step"]]) if adv.size else float("nan")
            w = arr(bundle, "weight_log")
            metrics["tail_action_authority"] = norm_mean(w[sl["tail_step"]] - 1.0) if w.size else float("nan")
        elif agent == "residual":
            sg = arr(bundle, "residual_sg_selected_source_log", dtype=int).reshape(-1)
            metrics["tail_policy_fraction"] = finite_fraction(sg[sl["tail_step"]], 2)
            metrics["tail_supervisor_fraction"] = finite_fraction(sg[sl["tail_step"]], 1)
            adv = arr(bundle, "residual_sg_advantage_log").reshape(-1)
            metrics["tail_gate_advantage"] = finite_mean(adv[sl["tail_step"]]) if adv.size else float("nan")
            r = arr(bundle, "residual_exec_log")
            metrics["tail_action_authority"] = norm_mean(r[sl["tail_step"]]) if r.size else float("nan")
        snap = (bundle.get("replay_buffer_snapshots") or {}).get(agent, {})
        actor = arr(bundle, f"{agent}_actor_losses").reshape(-1)
        critic = arr(bundle, f"{agent}_critic_losses").reshape(-1)
        rows.append(
            {
                "plant": plant,
                "agent": agent,
                "tail_reward_of_combined": finite_mean(rewards[sl["tail_ep"]]),
                "combined_delta_vs_ofmpc": finite_mean(rewards[sl["tail_ep"]]) - baseline_tail,
                "replay_size": snap.get("size", "") if isinstance(snap, dict) else "",
                "replay_capacity": snap.get("capacity", "") if isinstance(snap, dict) else "",
                "actor_loss_finite_frac": float(np.mean(np.isfinite(actor))) if actor.size else float("nan"),
                "critic_loss_finite_frac": float(np.mean(np.isfinite(critic))) if critic.size else float("nan"),
                **metrics,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "plant",
        "family",
        "role",
        "label",
        "timestamp",
        "status",
        "tail_reward",
        "tail_reward_delta_vs_ofmpc",
        "final_reward",
        "worst_postwarm_reward",
        "first_live_reward",
        "tail_mae_y1",
        "tail_mae_y2",
        "tail_total_variation_u",
        "tail_policy_fraction",
        "tail_supervisor_fraction",
        "tail_fallback_fraction",
        "tail_gate_advantage",
        "tail_action_authority",
        "replay_size",
        "replay_capacity",
        "has_config_snapshot",
        "has_agent_config_snapshot",
        "path",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_logging_gap_rows(rows: list[dict[str, Any]], attr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gap_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            gap_rows.append(
                {
                    "plant": row.get("plant", ""),
                    "runner_or_agent": row.get("label", row.get("family", "")),
                    "priority": "P0",
                    "gap": "missing final bundle",
                    "current_evidence": row.get("missing_reason", "bundle unavailable"),
                    "paper_risk": "cannot include this runner in reproducible main table",
                    "recommended_fix": "rerun or restore the exact final input_data bundle",
                }
            )
            continue

        bundle_path = REPO_ROOT / str(row["path"])
        episode_csv = bundle_path.parent / "episode_metrics.csv"
        if not episode_csv.exists():
            gap_rows.append(
                {
                    "plant": row["plant"],
                    "runner_or_agent": row["label"],
                    "priority": "P1",
                    "gap": "episode_metrics.csv not saved",
                    "current_evidence": "metrics recoverable from pickle only",
                    "paper_risk": "tables depend on post-hoc parser rather than run-native audit trail",
                    "recommended_fix": "save per-episode reward components, IAE, RMSE, max error, input TV, source fractions, and safety fractions",
                }
            )

        if row["role"] == "combined" and not row.get("has_agent_config_snapshot"):
            gap_rows.append(
                {
                    "plant": row["plant"],
                    "runner_or_agent": row["label"],
                    "priority": "P0",
                    "gap": "combined agent_config_snapshot missing",
                    "current_evidence": "input_data.pkl exists but does not expose final per-agent exploration/config provenance",
                    "paper_risk": "harder to prove Markov/residual param-noise settings in the archived run",
                    "recommended_fix": "rerun this combined case once after the config snapshot guard and archive the resulting bundle",
                }
            )

        if row["role"] == "standalone" and row["family"] in {"Markov", "weights", "residual"}:
            if not row.get("replay_available"):
                gap_rows.append(
                    {
                        "plant": row["plant"],
                        "runner_or_agent": row["label"],
                        "priority": "P1",
                        "gap": "standalone replay snapshot not saved",
                        "current_evidence": "learning losses may exist, but compact replay state/source coverage is unavailable",
                        "paper_risk": "weak replay-coverage argument for why RL saw informative states",
                        "recommended_fix": "save compact replay audit rows for state range, tail-steady range, source mix, and reward component range",
                    }
                )

    for row in attr_rows:
        if row.get("agent") == "markov" and float(row.get("tail_fallback_fraction", 0.0) or 0.0) > 0.9:
            gap_rows.append(
                {
                    "plant": row["plant"],
                    "runner_or_agent": "combined Markov",
                    "priority": "P0",
                    "gap": "tail Markov branch mostly fallback",
                    "current_evidence": f"tail fallback fraction {fmt(row.get('tail_fallback_fraction'))}",
                    "paper_risk": "combined success should not be interpreted as causal Markov success without ablation",
                    "recommended_fix": "inspect solver/source codes and run no-Markov combined ablation under the same disturbance and seed",
                }
            )
    return gap_rows


def build_next_experiment_rows() -> list[dict[str, str]]:
    return [
        {
            "priority": "P0",
            "experiment_or_log": "leave-one-agent-out combined ablations",
            "plant_scope": "polymer and distillation",
            "why_it_matters": "turns diagnostic attribution into causal contribution",
            "minimum_output": "no-horizon, no-Markov, no-weights, no-residual combined bundles with the same seed and disturbance",
            "paper_use": "report C_i = J(all agents) - J(all except i) beside the diagnostic source-fraction table",
        },
        {
            "priority": "P0",
            "experiment_or_log": "no-SG standalone ablations",
            "plant_scope": "horizon, Markov, weights, and residual where practical",
            "why_it_matters": "isolates the safety-gate effect from the RL policy architecture",
            "minimum_output": "keep hard MPC/physical clipping, remove only the SG accept/reject decision, and store identical metrics",
            "paper_use": "main safety claim: SG reduces bad handoff, fallback burden, and worst post-warm episodes",
        },
        {
            "priority": "P0",
            "experiment_or_log": "distillation combined provenance rerun",
            "plant_scope": "distillation combined",
            "why_it_matters": "current final bundle is strong numerically but lacks explicit per-agent config snapshot",
            "minimum_output": "one rerun after the config-snapshot guard with param-noise settings visible in saved JSON/pickle",
            "paper_use": "avoids a provenance footnote in the final combined distillation table",
        },
        {
            "priority": "P1",
            "experiment_or_log": "common rescoring freeze",
            "plant_scope": "all final bundles",
            "why_it_matters": "ensures every table and figure uses the same physical-unit metrics and tail window",
            "minimum_output": "single script that emits final summary, attribution, safety, replay, and logging-gap CSVs",
            "paper_use": "reproducible table-generation method for the journal supplement",
        },
        {
            "priority": "P1",
            "experiment_or_log": "seed replication",
            "plant_scope": "headline runners if compute allows",
            "why_it_matters": "RL results are sensitive to seed and handoff timing",
            "minimum_output": "two extra seeds for OF-MPC, best standalone, combined, and key ablations",
            "paper_use": "median and interquartile intervals rather than single-run claims",
        },
        {
            "priority": "P1",
            "experiment_or_log": "run-native replay and reward audits",
            "plant_scope": "every active RL runner",
            "why_it_matters": "connects final performance to state coverage, source mix, and reward decomposition",
            "minimum_output": "compact replay audit CSV and reward-component episode CSV",
            "paper_use": "appendix tables explaining whether the replay buffer contained informative off-steady-state samples",
        },
    ]


def row_label(row: dict[str, Any]) -> str:
    return f"{row['plant']} {row['family']}"


def make_figures(rows: list[dict[str, Any]], bundles: dict[tuple[str, str], dict[str, Any]], attr_rows: list[dict[str, Any]]) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    families = ["OF-MPC", "horizon", "weights", "residual", "Markov", "combined"]
    colors = {
        "OF-MPC": "#555555",
        "horizon": "#4c78a8",
        "weights": "#f58518",
        "residual": "#54a24b",
        "Markov": "#b279a2",
        "combined": "#e45756",
    }

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for ax, plant in zip(axs, ("polymer", "distillation")):
        plant_rows = {row["family"]: row for row in ok_rows if row["plant"] == plant}
        vals = [plant_rows.get(f, {}).get("tail_reward_delta_vs_ofmpc", np.nan) for f in families]
        ax.bar(range(len(families)), vals, color=[colors[f] for f in families])
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(f"{plant.title()} tail reward gain")
        ax.set_xticks(range(len(families)), families, rotation=35, ha="right")
        ax.set_ylabel("Tail reward delta vs OF-MPC")
        ax.grid(axis="y", alpha=0.25)
    path = OUT_DIR / "fig_final_tail_reward_delta_by_family.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for ax, plant in zip(axs, ("polymer", "distillation")):
        for family in families:
            bundle = bundles.get((plant, family))
            if not bundle:
                continue
            rewards = reward_vector(bundle)
            if rewards.size:
                ax.plot(np.arange(1, rewards.size + 1), rewards, lw=1.2, label=family, color=colors[family])
        ax.axvline(10, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvspan(10, 13, color="0.8", alpha=0.25)
        ax.set_title(f"{plant.title()} reward histories")
        ax.set_xlabel("Subepisode")
        ax.set_ylabel("Average reward")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    path = OUT_DIR / "fig_final_reward_histories_all_runners.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    metric_names = ["tail_mae_y1", "tail_mae_y2", "tail_total_variation_u", "tail_policy_fraction"]
    fig, axs = plt.subplots(1, len(metric_names), figsize=(16, 4.2), constrained_layout=True)
    for ax, metric in zip(axs, metric_names):
        labels = []
        values = []
        bar_colors = []
        for row in ok_rows:
            if row["family"] == "OF-MPC" and metric == "tail_policy_fraction":
                continue
            labels.append(row_label(row))
            values.append(row.get(metric, np.nan))
            bar_colors.append(colors[row["family"]])
        ax.bar(range(len(labels)), values, color=bar_colors)
        ax.set_title(metric.replace("_", " "))
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    path = OUT_DIR / "fig_final_metric_panels_all_runners.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    for plant in ("polymer", "distillation"):
        fig, axs = plt.subplots(2, 2, figsize=(13, 6.5), constrained_layout=True)
        output_labels = OUTPUT_LABELS[plant]
        selected = ["OF-MPC", "residual", "Markov", "combined"]
        for family in selected:
            bundle = bundles.get((plant, family))
            if not bundle:
                continue
            rewards = reward_vector(bundle)
            y = primary_y(bundle, baseline=(family == "OF-MPC"))
            y_sp = as_physical_setpoint(bundle)
            u = primary_u(bundle, baseline=(family == "OF-MPC"))
            if y.size == 0 or y_sp.size == 0:
                continue
            n_steps = y_sp.shape[0]
            steps_per_episode = int(bundle.get("time_in_sub_episodes") or (n_steps // max(rewards.size, 1)))
            start = max(0, n_steps - 20 * steps_per_episode)
            t = np.arange(start, n_steps)
            y_aligned = y[1 : n_steps + 1]
            for idx in (0, 1):
                axs[0, idx].plot(t, y_aligned[start:n_steps, idx], lw=1.1, label=family, color=colors[family])
            if u.size:
                for idx in (0, 1):
                    axs[1, idx].plot(t, u[start:n_steps, idx], lw=1.1, label=family, color=colors[family])
        bundle0 = bundles.get((plant, "OF-MPC"))
        if bundle0:
            y_sp = as_physical_setpoint(bundle0)
            n_steps = y_sp.shape[0]
            rewards = reward_vector(bundle0)
            steps_per_episode = int(bundle0.get("time_in_sub_episodes") or (n_steps // max(rewards.size, 1)))
            start = max(0, n_steps - 20 * steps_per_episode)
            t = np.arange(start, n_steps)
            for idx in (0, 1):
                axs[0, idx].plot(t, y_sp[start:n_steps, idx], lw=1.0, ls="--", color="black", label="setpoint")
        for idx, label in enumerate(output_labels):
            axs[0, idx].set_title(f"{plant.title()} {label} tail tracking")
            axs[0, idx].grid(alpha=0.25)
        for idx, label in enumerate(INPUT_LABELS[plant]):
            axs[1, idx].set_title(f"{plant.title()} {label} tail input")
            axs[1, idx].grid(alpha=0.25)
        axs[0, 0].legend(fontsize=7)
        path = OUT_DIR / f"fig_final_tail_tracking_inputs_{plant}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figure_paths.append(path)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for ax, plant in zip(axs, ("polymer", "distillation")):
        plant_attr = [row for row in attr_rows if row["plant"] == plant]
        agents = [row["agent"] for row in plant_attr]
        policy = [row.get("tail_policy_fraction", np.nan) for row in plant_attr]
        fallback = [row.get("tail_fallback_fraction", np.nan) for row in plant_attr]
        x = np.arange(len(agents))
        ax.bar(x - 0.18, policy, width=0.36, label="policy")
        ax.bar(x + 0.18, fallback, width=0.36, label="fallback")
        ax.set_title(f"{plant.title()} combined source fractions")
        ax.set_xticks(x, agents)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    path = OUT_DIR / "fig_final_combined_source_fractions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    return figure_paths


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(
    rows: list[dict[str, Any]],
    attr_rows: list[dict[str, Any]],
    figures: list[Path],
    logging_gap_rows: list[dict[str, Any]],
    next_step_rows: list[dict[str, str]],
    table_paths: list[Path],
) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    by_plant = {
        plant: {row["family"]: row for row in ok_rows if row["plant"] == plant}
        for plant in ("polymer", "distillation")
    }
    lines: list[str] = [
        "# Final All-Runners Paper Package",
        "",
        "Date: 2026-06-11",
        "",
        "## Scope",
        "",
        "This report includes the finalized standalone and combined runners for both case studies. The main-text set is OF-MPC, horizon SG-DQN, weights SG-TD3, residual SG-TD3, Markov SG-TD3, and the four-agent combined SG runner for polymer and distillation. All metrics below are computed from saved `input_data.pkl` or baseline pickle bundles; no polymer or Aspen simulations are launched.",
        "",
        "## Method Frame",
        "",
        "The common closed-loop structure is offset-free MPC with an RL supervisor. For a supervisor-gated agent, the executed action is selected by comparing a policy score and a supervisor score:",
        "",
        "$$ a_{\\mathrm{exec}} = a_{\\pi}\\ \\mathrm{if}\\ S(a_{\\pi}) - S(a_{\\mathrm{sup}}) > m,\\ \\mathrm{else}\\ a_{\\mathrm{sup}}. $$",
        "",
        "Tracking metrics are computed in physical output units by converting saved scaled-deviation setpoints back to physical coordinates using each bundle's `data_min`, `data_max`, and `steady_states` fields.",
        "",
        "## Final Performance Table",
        "",
        "| Plant | Runner | Role | Tail reward | Delta vs OF-MPC | Worst post-warm | First live | Tail y1 MAE | Tail y2 MAE | Tail input TV | Tail policy frac |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for plant in ("polymer", "distillation"):
        for family in ("OF-MPC", "horizon", "weights", "residual", "Markov", "combined"):
            row = by_plant.get(plant, {}).get(family)
            if not row:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        plant,
                        str(row["label"]),
                        str(row["role"]),
                        fmt(row.get("tail_reward")),
                        fmt(row.get("tail_reward_delta_vs_ofmpc")),
                        fmt(row.get("worst_postwarm_reward")),
                        fmt(row.get("first_live_reward")),
                        fmt(row.get("tail_mae_y1")),
                        fmt(row.get("tail_mae_y2")),
                        fmt(row.get("tail_total_variation_u")),
                        fmt(row.get("tail_policy_fraction")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
        ]
    )
    for plant in ("polymer", "distillation"):
        plant_rows = by_plant[plant]
        best = max(
            [row for fam, row in plant_rows.items() if fam != "OF-MPC"],
            key=lambda row: float(row.get("tail_reward_delta_vs_ofmpc", float("-inf"))),
        )
        combined = plant_rows.get("combined")
        residual = plant_rows.get("residual")
        markov = plant_rows.get("Markov")
        lines.append(
            f"- {plant.title()}: best tail reward among final runners is `{best['label']}` with delta `{fmt(best.get('tail_reward_delta_vs_ofmpc'))}` versus OF-MPC."
        )
        if combined:
            lines.append(
                f"- {plant.title()} combined: tail delta is `{fmt(combined.get('tail_reward_delta_vs_ofmpc'))}` and worst post-warm reward is `{fmt(combined.get('worst_postwarm_reward'))}`."
            )
        if residual and markov:
            lines.append(
                f"- {plant.title()} residual vs Markov: residual tail delta `{fmt(residual.get('tail_reward_delta_vs_ofmpc'))}`, Markov tail delta `{fmt(markov.get('tail_reward_delta_vs_ofmpc'))}`."
            )

    lines.extend(
        [
            "",
            "## Combined-Agent Attribution",
            "",
            "The table below is diagnostic attribution, not causal attribution. Causal attribution still requires leave-one-agent-out combined reruns under the same disturbance and seed.",
            "",
            "| Plant | Agent | Combined tail delta | Tail policy frac | Tail supervisor frac | Tail fallback frac | Tail gate advantage | Tail authority | Replay size | Critic finite frac |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in attr_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["plant"]),
                    str(row["agent"]),
                    fmt(row.get("combined_delta_vs_ofmpc")),
                    fmt(row.get("tail_policy_fraction")),
                    fmt(row.get("tail_supervisor_fraction")),
                    fmt(row.get("tail_fallback_fraction")),
                    fmt(row.get("tail_gate_advantage")),
                    fmt(row.get("tail_action_authority")),
                    str(row.get("replay_size", "NA") or "NA"),
                    fmt(row.get("critic_loss_finite_frac")),
                ]
            )
            + " |"
        )

    dist_markov_attr = next(
        (row for row in attr_rows if row.get("plant") == "distillation" and row.get("agent") == "markov"),
        None,
    )
    poly_markov_attr = next(
        (row for row in attr_rows if row.get("plant") == "polymer" and row.get("agent") == "markov"),
        None,
    )
    lines.extend(
        [
            "",
            "## Diagnostic Caveats",
            "",
            "- Standalone success is strong evidence that each agent family is viable by itself under the final disturbance scenario. It is not, by itself, proof that the same agent caused the combined-run improvement.",
            "- The combined attribution table should therefore be read as a mechanism diagnostic: policy execution, fallback, action authority, replay health, and gate advantage. The causal table still needs the leave-one-agent-out reruns.",
        ]
    )
    if poly_markov_attr:
        lines.append(
            f"- Polymer combined: Markov tail policy execution is `{fmt(poly_markov_attr.get('tail_policy_fraction'))}` with fallback `{fmt(poly_markov_attr.get('tail_fallback_fraction'))}`, so the combined gain is real but Markov's direct causal share is still unresolved."
        )
    if dist_markov_attr:
        lines.append(
            f"- Distillation combined: Markov tail fallback is `{fmt(dist_markov_attr.get('tail_fallback_fraction'))}` while weights and residual have much larger policy fractions. Treat the current combined win as a combined-supervisor result, not as a standalone Markov credit claim."
        )

    lines.extend(
        [
            "",
            "## Next Experiments For The Paper",
            "",
            "| Priority | Experiment or log | Scope | Minimum output | Paper use |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in next_step_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["priority"],
                    row["experiment_or_log"],
                    row["plant_scope"],
                    row["minimum_output"],
                    row["paper_use"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Logging Gap Extract",
            "",
            "The full machine-readable gap list is saved as `final_logging_gap_table.csv`. The most paper-relevant gaps are:",
            "",
            "| Priority | Plant | Runner or agent | Gap | Recommended fix |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    priority_gaps = [row for row in logging_gap_rows if row.get("priority") == "P0"]
    if not priority_gaps:
        priority_gaps = logging_gap_rows[:4]
    for row in priority_gaps[:8]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("priority", "")),
                    str(row.get("plant", "")),
                    str(row.get("runner_or_agent", "")),
                    str(row.get("gap", "")),
                    str(row.get("recommended_fix", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paper Readiness",
            "",
            "What is ready:",
            "",
            "- Both plants now have saved standalone and combined bundles with reward histories, trajectories, source logs, losses, and replay snapshots for combined runs.",
            "- Distillation combined now has `input_data.pkl`, closing the earlier attribution gap.",
            "- Polymer Markov and residual parameter-noise standalone runs are now available and should be used as final standalone rows.",
            "",
            "What still needs one more paper-safe pass:",
            "",
            "- Run leave-one-agent-out combined ablations: no-horizon, no-Markov, no-weights, and no-residual for both plants. This is the clean causal attribution experiment.",
            "- Run no-SG standalone ablations for TD3 families and DQN horizon where practical. Keep hard MPC and physical clipping; remove only the supervisor gate. This supports the safety-gate claim.",
            "- Re-run distillation combined once after the `agent_config_snapshot` guard commit so the final combined bundle explicitly records Markov/residual parameter-noise config provenance.",
            "- Freeze a single common rescoring script for reward, IAE, RMSE, max error, input total variation, source fractions, and safety burden. Then regenerate all main tables from that script.",
            "- Add at least two more seeds for the paper headline rows if compute time allows. If not, report this as single-seed final evidence and avoid statistical claims.",
            "",
            "Recommended new logs for every future final run:",
            "",
            "- `episode_metrics.csv` with reward, reward components, physical IAE/RMSE/max error, input total variation, saturation/projection/fallback fractions, and source fractions.",
            "- `agent_attribution_summary.csv` for combined runs with one row per agent.",
            "- `config_snapshot.json` and `agent_config_snapshot.json` saved as standalone files in addition to pickle storage.",
            "- Compact replay audit CSV per active agent: replay size, capacity, state dimension, feature ranges, tail-steady feature ranges, and source mix.",
            "- Seed, git commit, baseline bundle path, disturbance profile, setpoint schedule identifier, plant/model identifiers, and exact runner script name.",
            "",
            "## Figures And Tables Generated",
            "",
        ]
    )
    for table_path in table_paths:
        lines.append(f"- `{rel(table_path)}`")
    for fig in figures:
        lines.append(f"- `{rel(fig)}`")
    lines.extend(
        [
            "",
            "## Data Files Included",
            "",
        ]
    )
    for run in RUNS:
        lines.append(f"- `{rel(Path(run['path']))}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baselines: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str], dict[str, Any]] = {}

    # First load baselines so every row can compute delta against the same plant baseline.
    for run in RUNS:
        if run["role"] != "baseline":
            continue
        bundle = load_pickle(Path(run["path"]))
        rewards = reward_vector(bundle)
        n_steps = int(bundle.get("nFE") or arr(bundle, "y_sp").shape[0])
        sl = window_slices(bundle, rewards.size, n_steps)
        baselines[run["plant"]] = finite_mean(rewards[sl["tail_ep"]])

    for run in RUNS:
        row, bundle = summarize_run(run, baselines)
        rows.append(row)
        if bundle:
            bundles[(run["plant"], run["family"])] = bundle

    attr_rows: list[dict[str, Any]] = []
    for plant in ("polymer", "distillation"):
        bundle = bundles.get((plant, "combined"))
        if bundle:
            attr_rows.extend(summarize_combined_agents(plant, bundle, baselines[plant]))

    logging_gap_rows = build_logging_gap_rows(rows, attr_rows)
    next_step_rows = build_next_experiment_rows()

    summary_csv = OUT_DIR / "final_all_runners_summary.csv"
    attribution_csv = OUT_DIR / "final_combined_agent_attribution.csv"
    logging_gap_csv = OUT_DIR / "final_logging_gap_table.csv"
    next_steps_csv = OUT_DIR / "final_paper_next_experiments.csv"
    write_csv(summary_csv, rows)
    write_csv(attribution_csv, attr_rows)
    write_csv(logging_gap_csv, logging_gap_rows)
    write_csv(next_steps_csv, next_step_rows)
    (OUT_DIR / "final_all_runners_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT_DIR / "final_combined_agent_attribution.json").write_text(
        json.dumps(attr_rows, indent=2), encoding="utf-8"
    )
    figures = make_figures(rows, bundles, attr_rows)
    table_paths = [summary_csv, attribution_csv, logging_gap_csv, next_steps_csv]
    write_report(rows, attr_rows, figures, logging_gap_rows, next_step_rows, table_paths)

    print(f"Wrote {rel(summary_csv)}")
    print(f"Wrote {rel(attribution_csv)}")
    print(f"Wrote {rel(logging_gap_csv)}")
    print(f"Wrote {rel(next_steps_csv)}")
    for fig in figures:
        print(f"Wrote {rel(fig)}")
    print(f"Wrote {rel(REPORT_PATH)}")


if __name__ == "__main__":
    main()
