from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "report" / "figures" / "polymer_sg_td3_latest_20260601"

OUTPUT_LABELS = ["eta", "T"]
INPUT_LABELS = ["Qc", "Qm"]

RUN_SPECS = {
    "OF-MPC": {
        "path": REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle",
        "kind": "baseline",
        "family": "baseline",
        "color": "#3B3B3B",
    },
    "TD3 Weights": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td3_weights_disturb"
        / "20260520_214118"
        / "input_data.pkl",
        "kind": "weights",
        "family": "weights",
        "color": "#4C78A8",
    },
    "SG-TD3 Weights CW3": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_weights_critic_warm3_conservative_disturb"
        / "20260601_184718"
        / "input_data.pkl",
        "kind": "weights",
        "family": "weights",
        "color": "#F58518",
    },
    "TD3 Residual": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td3_residual_disturb"
        / "20260601_021504"
        / "input_data.pkl",
        "kind": "residual",
        "family": "residual",
        "color": "#4C78A8",
    },
    "SG-TD3 Residual": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_residual_disturb"
        / "20260601_022723"
        / "input_data.pkl",
        "kind": "residual",
        "family": "residual",
        "color": "#54A24B",
    },
    "SG-TD3 Residual CW-old": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_residual_critic_warm_disturb"
        / "20260601_140709"
        / "input_data.pkl",
        "kind": "residual",
        "family": "residual",
        "color": "#B279A2",
    },
    "SG-TD3 Residual CW3": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_residual_critic_warm3_conservative_disturb"
        / "20260601_182754"
        / "input_data.pkl",
        "kind": "residual",
        "family": "residual",
        "color": "#F58518",
    },
    "TD7 Residual": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td7_residual_disturb"
        / "20260601_022931"
        / "input_data.pkl",
        "kind": "residual",
        "family": "residual",
        "color": "#9E765F",
    },
}


@dataclass
class MethodRun:
    label: str
    spec: dict[str, Any]
    bundle: dict[str, Any]
    nfe: int
    time_in_sub: int
    warm_step: int
    y: np.ndarray
    u: np.ndarray
    y_sp_phys: np.ndarray
    err_phys: np.ndarray
    rewards_step: np.ndarray
    avg_rewards: np.ndarray
    delta_u_scaled: np.ndarray

    @property
    def warm_episode(self) -> int:
        return int(self.warm_step // max(1, self.time_in_sub))

    @property
    def episode_count(self) -> int:
        return int(self.avg_rewards.size)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def nanmean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def nanq(values: Any, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{100.0 * val:.{digits}f}%"


def fmt_range(lo: Any, hi: Any, digits: int = 4) -> str:
    try:
        lo_val = float(lo)
        hi_val = float(hi)
    except (TypeError, ValueError):
        return "NA"
    if not (np.isfinite(lo_val) and np.isfinite(hi_val)):
        return "NA"
    return f"{lo_val:.{digits}f} to {hi_val:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(col, "NA")) for col in columns) + " |\n")


def reverse_minmax(x_scaled: Any, lo: Any, hi: Any) -> np.ndarray:
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    return np.asarray(x_scaled, dtype=float) * np.maximum(hi_arr - lo_arr, 1.0e-12) + lo_arr


def ysp_scaled_dev_to_phys(bundle: dict[str, Any]) -> np.ndarray:
    n_inputs = int(bundle.get("n_inputs", 2))
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    y_min = data_min[n_inputs:]
    y_max = data_max[n_inputs:]
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], dtype=float)
    y_ss_scaled = (y_ss - y_min) / np.maximum(y_max - y_min, 1.0e-12)
    return reverse_minmax(np.asarray(bundle["y_sp"], dtype=float) + y_ss_scaled, y_min, y_max)


def load_method(label: str, spec: dict[str, Any]) -> MethodRun:
    bundle = load_pickle(spec["path"])
    nfe = int(bundle["nFE"])
    time_in_sub = int(bundle["time_in_sub_episodes"])
    warm_step = int(bundle.get("warm_start_step", time_in_sub * 10))
    y_sp_phys = ysp_scaled_dev_to_phys(bundle)[:nfe]

    if spec["kind"] == "baseline":
        y = np.asarray(bundle["y_mpc"], dtype=float)
        u = np.asarray(bundle["u_mpc"], dtype=float)
        rewards_step = np.asarray(bundle.get("rewards_step", bundle.get("rewards_mpc", [])), dtype=float)
        avg_rewards = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), dtype=float)
    else:
        y = np.asarray(bundle.get("y", bundle.get("y_rl")), dtype=float)
        u = np.asarray(bundle.get("u", bundle.get("u_rl")), dtype=float)
        rewards_step = np.asarray(bundle.get("rewards_step", []), dtype=float)
        avg_rewards = np.asarray(bundle.get("avg_rewards", []), dtype=float)

    y_eval = y[1 : nfe + 1]
    err_phys = y_eval - y_sp_phys
    delta_u_scaled = np.asarray(bundle.get("delta_u_storage", np.zeros((nfe, 2))), dtype=float)[:nfe]
    return MethodRun(
        label=label,
        spec=spec,
        bundle=bundle,
        nfe=nfe,
        time_in_sub=time_in_sub,
        warm_step=warm_step,
        y=y,
        u=u,
        y_sp_phys=y_sp_phys,
        err_phys=err_phys,
        rewards_step=rewards_step[:nfe],
        avg_rewards=avg_rewards,
        delta_u_scaled=delta_u_scaled,
    )


def window_masks(run: MethodRun) -> dict[str, np.ndarray]:
    idx = np.arange(run.nfe)
    return {
        "postwarm": idx > run.warm_step,
        "tail20": idx >= max(0, run.nfe - 20 * run.time_in_sub),
        "last": idx >= max(0, run.nfe - run.time_in_sub),
    }


def final_episode_steady_mask(run: MethodRun, window_len: int = 100) -> tuple[np.ndarray, str]:
    start = max(0, run.nfe - run.time_in_sub)
    stop = run.nfe
    sp = run.y_sp_phys[start:stop]
    if sp.size == 0:
        return np.zeros(run.nfe, dtype=bool), "NA"

    changes = np.where(np.linalg.norm(np.diff(sp, axis=0), axis=1) > 1.0e-9)[0] + 1
    boundaries = np.r_[0, changes, stop - start]
    mask = np.zeros(run.nfe, dtype=bool)
    windows: list[str] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        width = min(window_len, int(right - left))
        mask[start + right - width : start + right] = True
        windows.append(f"{right - width}-{right - 1}")
    return mask, "; ".join(windows)


def first_live_episode(run: MethodRun) -> int:
    step = int(run.bundle.get("phase1_first_live_action_step", run.warm_step + 1))
    return int(np.clip(step // max(1, run.time_in_sub), 0, max(0, run.episode_count - 1)))


def performance_metrics(run: MethodRun) -> dict[str, Any]:
    masks = window_masks(run)
    tail = masks["tail20"]
    last = masks["last"]
    avg = run.avg_rewards
    post_episode = avg[run.warm_episode :] if avg.size > run.warm_episode else np.array([], dtype=float)
    first_live_idx = first_live_episode(run)
    e_tail = run.err_phys[tail]
    e_last = run.err_phys[last]
    du_tail = run.delta_u_scaled[tail]
    tail_episode_count = min(20, avg.size)
    return {
        "Method": run.label,
        "Bundle": run.spec["path"].relative_to(REPO_ROOT).as_posix(),
        "Mean reward": nanmean(avg),
        "Worst post-warm reward": float(np.nanmin(post_episode)) if post_episode.size else float("nan"),
        "Worst post-warm episode": int(np.nanargmin(post_episode) + run.warm_episode + 1)
        if post_episode.size
        else None,
        "First live episode": int(first_live_idx + 1),
        "First live reward": float(avg[first_live_idx]) if avg.size else float("nan"),
        "Tail-20 reward": nanmean(avg[-tail_episode_count:]),
        "Final reward": float(avg[-1]) if avg.size else float("nan"),
        "Tail eta RMSE": float(np.sqrt(np.mean(e_tail[:, 0] ** 2))),
        "Tail T RMSE": float(np.sqrt(np.mean(e_tail[:, 1] ** 2))),
        "Tail eta MAE": float(np.mean(np.abs(e_tail[:, 0]))),
        "Tail T MAE": float(np.mean(np.abs(e_tail[:, 1]))),
        "Last eta RMSE": float(np.sqrt(np.mean(e_last[:, 0] ** 2))),
        "Last T RMSE": float(np.sqrt(np.mean(e_last[:, 1] ** 2))),
        "Tail mean abs du scaled": float(np.mean(np.abs(du_tail))),
    }


def steady_metrics(run: MethodRun) -> dict[str, Any]:
    mask, windows = final_episode_steady_mask(run)
    e = run.err_phys[mask]
    if e.size == 0:
        e = np.full((1, 2), np.nan)
    return {
        "Method": run.label,
        "Windows": windows,
        "Steps": int(mask.sum()),
        "Eta MAE": float(np.mean(np.abs(e[:, 0]))),
        "T MAE": float(np.mean(np.abs(e[:, 1]))),
        "Eta RMSE": float(np.sqrt(np.mean(e[:, 0] ** 2))),
        "T RMSE": float(np.sqrt(np.mean(e[:, 1] ** 2))),
        "Eta mean signed": float(np.mean(e[:, 0])),
        "T mean signed": float(np.mean(e[:, 1])),
    }


def arr_or_default(bundle: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = bundle.get(key)
    if value is None:
        return np.zeros(shape, dtype=float)
    return np.asarray(value)


def first_array(bundle: dict[str, Any], keys: list[str]) -> np.ndarray | None:
    for key in keys:
        value = bundle.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim > 0:
            return arr
    return None


def weight_diagnostics(run: MethodRun) -> dict[str, Any]:
    nfe = run.nfe
    masks = window_masks(run)
    tail = masks["tail20"]
    post = masks["postwarm"]
    weights = np.asarray(run.bundle["weight_log"], dtype=float)[:nfe]
    low = np.asarray(run.bundle.get("low_coef", [0.75, 0.75, 0.75, 0.75]), dtype=float)
    high = np.asarray(run.bundle.get("high_coef", [2.0, 2.0, 2.0, 2.0]), dtype=float)
    tail_weights = weights[tail]

    source = run.bundle.get("sg_selected_source_log")
    policy_frac = None
    supervisor_frac = None
    if source is not None:
        source_arr = np.asarray(source, dtype=int)[:nfe]
        policy_frac = float(np.mean(source_arr[tail] == 2))
        supervisor_frac = float(np.mean(source_arr[tail] == 1))

    action_source = run.bundle.get("weight_action_source_log")
    identity_source_frac = None
    if action_source is not None:
        action_source_arr = np.asarray(action_source, dtype=int)[:nfe]
        identity_code = int(run.bundle.get("weight_action_source_codes", {}).get("identity_fallback", 3))
        identity_source_frac = float(np.mean(action_source_arr[post] == identity_code))

    action = first_array(run.bundle, ["weight_executed_action_raw_log", "executed_action_raw_log", "action_raw_log"])
    if action is not None and action.ndim == 2:
        tail_action_abs = np.nanmean(np.abs(action[:nfe][tail]), axis=0)
    else:
        tail_action_abs = np.full(4, np.nan)

    common_scale_std = float(np.nanmean(np.std(np.log(np.maximum(tail_weights, 1.0e-12)), axis=1)))
    boundary = np.isclose(tail_weights, low, atol=1.0e-6) | np.isclose(tail_weights, high, atol=1.0e-6)
    return {
        "Method": run.label,
        "Tail policy selected": policy_frac,
        "Tail supervisor selected": supervisor_frac,
        "Tail multipliers mean": "[" + ", ".join(fmt(v, 3) for v in np.nanmean(tail_weights, axis=0)) + "]",
        "Tail multiplier ranges": "; ".join(
            f"m{i + 1} {fmt_range(np.nanmin(tail_weights[:, i]), np.nanmax(tail_weights[:, i]), 3)}"
            for i in range(tail_weights.shape[1])
        ),
        "Tail common-scale std": common_scale_std,
        "Tail boundary fraction": float(np.mean(np.any(boundary, axis=1))),
        "Post identity source fraction": identity_source_frac,
        "Tail cap projection": float(
            np.mean(arr_or_default(run.bundle, "weight_cap_projection_active_log", (nfe,)).astype(int)[:nfe][tail] != 0)
        ),
        "Tail fallback reason": float(
            np.mean(arr_or_default(run.bundle, "weight_fallback_reason_log", (nfe,)).astype(int)[:nfe][tail] != 0)
        ),
        "Tail abs raw action mean": "[" + ", ".join(fmt(v, 3) for v in tail_action_abs) + "]",
    }


def residual_array(run: MethodRun) -> np.ndarray:
    return np.asarray(
        run.bundle.get("delta_u_res_exec_log", run.bundle.get("residual_exec_log", np.zeros((run.nfe, 2)))),
        dtype=float,
    )[: run.nfe]


def residual_diagnostics(run: MethodRun) -> dict[str, Any]:
    nfe = run.nfe
    masks = window_masks(run)
    tail = masks["tail20"]
    residual = residual_array(run)
    tail_res = residual[tail]

    source = run.bundle.get("sg_selected_source_log")
    policy_frac = None
    supervisor_frac = None
    if source is not None:
        source_arr = np.asarray(source, dtype=int)[:nfe]
        policy_frac = float(np.mean(source_arr[tail] == 2))
        supervisor_frac = float(np.mean(source_arr[tail] == 1))

    steady_mask, _ = final_episode_steady_mask(run)
    steady_res = residual[steady_mask]
    if steady_res.size == 0:
        steady_res = np.full((1, 2), np.nan)
    return {
        "Method": run.label,
        "Tail policy selected": policy_frac,
        "Tail supervisor selected": supervisor_frac,
        "Tail residual mean abs": "[" + ", ".join(fmt(v, 4) for v in np.mean(np.abs(tail_res), axis=0)) + "]",
        "Tail residual q95 abs": "[" + ", ".join(fmt(v, 4) for v in np.quantile(np.abs(tail_res), 0.95, axis=0)) + "]",
        "Tail full-bound fraction": float(np.mean(np.any(np.isclose(np.abs(tail_res), 0.25, atol=1.0e-6), axis=1))),
        "Tail cap projection": float(
            np.mean(arr_or_default(run.bundle, "residual_cap_projection_active_log", (nfe,)).astype(int)[:nfe][tail] != 0)
        ),
        "Tail guard trigger": float(
            np.mean(arr_or_default(run.bundle, "residual_guard_triggered_log", (nfe,)).astype(int)[:nfe][tail] != 0)
        ),
        "Tail headroom projection": float(
            np.mean(arr_or_default(run.bundle, "projection_due_to_headroom_log", (nfe,)).astype(int)[:nfe][tail] != 0)
        ),
        "Tail zero fallback": float(
            np.mean(arr_or_default(run.bundle, "residual_zero_fallback_reason_log", (nfe,)).astype(int)[:nfe][tail] != 0)
        ),
        "Steady residual ranges": "; ".join(
            f"{label} {fmt_range(np.nanmin(steady_res[:, i]), np.nanmax(steady_res[:, i]), 4)}"
            for i, label in enumerate(INPUT_LABELS)
        ),
    }


def gate_diagnostics(run: MethodRun) -> dict[str, Any]:
    nfe = run.nfe
    masks = window_masks(run)
    post = masks["postwarm"]
    tail = masks["tail20"]
    source = run.bundle.get("sg_selected_source_log")
    adv = run.bundle.get("sg_advantage_log")
    if source is None or adv is None:
        return {}
    source_arr = np.asarray(source, dtype=int)[:nfe]
    adv_arr = np.asarray(adv, dtype=float)[:nfe]
    finite_tail = tail & np.isfinite(adv_arr)
    q_gap_policy = np.asarray(run.bundle.get("sg_q_gap_policy_log", np.full(nfe, np.nan)), dtype=float)[:nfe]
    q_gap_supervisor = np.asarray(run.bundle.get("sg_q_gap_supervisor_log", np.full(nfe, np.nan)), dtype=float)[:nfe]
    score_policy = np.asarray(run.bundle.get("sg_score_policy_log", np.full(nfe, np.nan)), dtype=float)[:nfe]
    score_supervisor = np.asarray(run.bundle.get("sg_score_supervisor_log", np.full(nfe, np.nan)), dtype=float)[:nfe]
    return {
        "Method": run.label,
        "Post policy selected": float(np.mean(source_arr[post] == 2)),
        "Post supervisor selected": float(np.mean(source_arr[post] == 1)),
        "Tail policy selected": float(np.mean(source_arr[tail] == 2)),
        "Tail supervisor selected": float(np.mean(source_arr[tail] == 1)),
        "Tail advantage mean": nanmean(adv_arr[finite_tail]),
        "Tail advantage q95": nanq(adv_arr[finite_tail], 0.95),
        "Tail policy q-gap mean": nanmean(q_gap_policy[finite_tail]),
        "Tail supervisor q-gap mean": nanmean(q_gap_supervisor[finite_tail]),
        "Tail score policy mean": nanmean(score_policy[finite_tail]),
        "Tail score supervisor mean": nanmean(score_supervisor[finite_tail]),
    }


def recovery_metrics(run: MethodRun, baseline: MethodRun) -> dict[str, Any]:
    n_ep = min(run.avg_rewards.size, baseline.avg_rewards.size)
    avg = run.avg_rewards[:n_ep]
    base = baseline.avg_rewards[:n_ep]
    warm_ep = min(run.warm_episode, n_ep)
    first_better = None
    first_5_better = None
    for idx in range(warm_ep, n_ep):
        if avg[idx] > base[idx]:
            first_better = idx + 1
            break
    for idx in range(warm_ep, n_ep - 4):
        if np.all(avg[idx : idx + 5] > base[idx : idx + 5]):
            first_5_better = idx + 1
            break
    post = avg[warm_ep:]
    return {
        "Method": run.label,
        "Warm mean reward": nanmean(avg[:warm_ep]),
        "Worst post-warm reward": float(np.nanmin(post)) if post.size else float("nan"),
        "Worst post-warm episode": int(np.nanargmin(post) + warm_ep + 1) if post.size else None,
        "First episode above OF-MPC": first_better,
        "First 5 episodes above OF-MPC": first_5_better,
        "Tail20 reward delta vs OF-MPC": nanmean(avg[-20:]) - nanmean(base[-20:]),
        "Final reward delta vs OF-MPC": float(avg[-1] - base[-1]) if avg.size and base.size else float("nan"),
    }


def rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_rewards(runs: list[MethodRun]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    groups = [
        ("Weights", ["OF-MPC", "TD3 Weights", "SG-TD3 Weights CW3"]),
        (
            "Residuals",
            ["OF-MPC", "TD3 Residual", "SG-TD3 Residual", "SG-TD3 Residual CW-old", "SG-TD3 Residual CW3", "TD7 Residual"],
        ),
    ]
    lookup = {run.label: run for run in runs}
    for ax, (title, labels) in zip(axes, groups):
        for label in labels:
            run = lookup[label]
            y = rolling_mean(run.avg_rewards, 5)
            x = np.arange(y.size) + 5
            ax.plot(x, y, label=label, color=run.spec["color"], linewidth=1.8)
        ax.axvline(10, color="#8A8A8A", linestyle="--", linewidth=1.0)
        ax.axvline(13, color="#8A8A8A", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("5-episode mean reward")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reward_curves.png", dpi=180)
    plt.close(fig)


def plot_tail_performance(runs: list[MethodRun], perf_rows: list[dict[str, Any]]) -> None:
    labels = [row["Method"] for row in perf_rows]
    colors = [next(run.spec["color"] for run in runs if run.label == label) for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, [row["Tail-20 reward"] for row in perf_rows], color=colors)
    axes[0].set_ylabel("Tail-20 reward")
    axes[1].bar(x, [row["Tail eta RMSE"] for row in perf_rows], color=colors)
    axes[1].set_ylabel("eta RMSE")
    axes[2].bar(x, [row["Tail T RMSE"] for row in perf_rows], color=colors)
    axes[2].set_ylabel("T RMSE")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tail_performance_bars.png", dpi=180)
    plt.close(fig)


def plot_gate_and_actions(
    gate_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [row["Method"] for row in gate_rows]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [row["Tail policy selected"] for row in gate_rows], width=0.36, label="policy")
    axes[0].bar(x + 0.18, [row["Tail supervisor selected"] for row in gate_rows], width=0.36, label="supervisor")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Tail fraction")
    axes[0].set_title("SG-TD3 gate source")
    axes[0].legend(fontsize=8)

    weight_labels = [row["Method"] for row in weight_rows]
    axes[1].bar(np.arange(len(weight_labels)), [row["Tail common-scale std"] for row in weight_rows], color="#E45756")
    axes[1].set_xticks(np.arange(len(weight_labels)))
    axes[1].set_xticklabels(weight_labels, rotation=30, ha="right")
    axes[1].set_ylabel("mean std(log multipliers)")
    axes[1].set_title("Weight nonuniformity")

    res_labels = [row["Method"] for row in residual_rows]
    res_mean = []
    for row in residual_rows:
        vals = row["Tail residual mean abs"].strip("[]").split(",")
        res_mean.append(float(vals[0]) + float(vals[1]))
    axes[2].bar(np.arange(len(res_labels)), res_mean, color="#72B7B2")
    axes[2].set_xticks(np.arange(len(res_labels)))
    axes[2].set_xticklabels(res_labels, rotation=30, ha="right")
    axes[2].set_ylabel("sum mean abs residual")
    axes[2].set_title("Residual authority use")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gate_and_action_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_last_episode_tracking(runs: list[MethodRun]) -> None:
    labels = ["OF-MPC", "TD3 Weights", "SG-TD3 Weights CW3", "SG-TD3 Residual CW3", "SG-TD3 Residual CW-old"]
    lookup = {run.label: run for run in runs}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for label in labels:
        run = lookup[label]
        start = run.nfe - run.time_in_sub
        x = np.arange(run.time_in_sub)
        y_eval = run.y[1 : run.nfe + 1][start : run.nfe]
        for j, ax in enumerate(axes):
            ax.plot(x, y_eval[:, j], color=run.spec["color"], linewidth=1.5, label=label if j == 0 else None)
    ref = lookup["OF-MPC"]
    start = ref.nfe - ref.time_in_sub
    x = np.arange(ref.time_in_sub)
    for j, ax in enumerate(axes):
        ax.plot(x, ref.y_sp_phys[start : ref.nfe, j], color="black", linestyle="--", linewidth=1.2, label="setpoint" if j == 0 else None)
        ax.set_ylabel(OUTPUT_LABELS[j])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Last episode step")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "last_episode_tracking_overlay.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = [load_method(label, spec) for label, spec in RUN_SPECS.items()]
    baseline = next(run for run in runs if run.label == "OF-MPC")

    perf_rows = [performance_metrics(run) for run in runs]
    steady_rows = [steady_metrics(run) for run in runs]
    weight_rows = [weight_diagnostics(run) for run in runs if run.spec["kind"] == "weights"]
    residual_rows = [residual_diagnostics(run) for run in runs if run.spec["kind"] == "residual"]
    gate_rows = [row for row in (gate_diagnostics(run) for run in runs) if row]
    recovery_rows = [recovery_metrics(run, baseline) for run in runs if run.label != "OF-MPC"]

    write_csv(FIG_DIR / "performance_summary.csv", perf_rows)
    write_csv(FIG_DIR / "steady_state_summary.csv", steady_rows)
    write_csv(FIG_DIR / "weight_diagnostics.csv", weight_rows)
    write_csv(FIG_DIR / "residual_diagnostics.csv", residual_rows)
    write_csv(FIG_DIR / "gate_diagnostics.csv", gate_rows)
    write_csv(FIG_DIR / "recovery_summary.csv", recovery_rows)

    perf_md = [
        {
            "Method": row["Method"],
            "Mean reward": fmt(row["Mean reward"], 3),
            "Worst post-warm": fmt(row["Worst post-warm reward"], 3),
            "Tail-20 reward": fmt(row["Tail-20 reward"], 3),
            "Final reward": fmt(row["Final reward"], 3),
            "Tail eta RMSE": fmt(row["Tail eta RMSE"], 4),
            "Tail T RMSE": fmt(row["Tail T RMSE"], 4),
            "Tail eta MAE": fmt(row["Tail eta MAE"], 4),
            "Tail T MAE": fmt(row["Tail T MAE"], 4),
            "Tail mean abs du": fmt(row["Tail mean abs du scaled"], 4),
        }
        for row in perf_rows
    ]
    steady_md = [
        {
            "Method": row["Method"],
            "Windows": row["Windows"],
            "Steps": row["Steps"],
            "Eta MAE": fmt(row["Eta MAE"], 6),
            "T MAE": fmt(row["T MAE"], 6),
            "Eta RMSE": fmt(row["Eta RMSE"], 6),
            "T RMSE": fmt(row["T RMSE"], 6),
            "Eta mean signed": fmt(row["Eta mean signed"], 6),
            "T mean signed": fmt(row["T mean signed"], 6),
        }
        for row in steady_rows
    ]
    weight_md = [
        {
            "Method": row["Method"],
            "Tail policy": fmt_pct(row["Tail policy selected"]),
            "Tail supervisor": fmt_pct(row["Tail supervisor selected"]),
            "Tail multipliers mean": row["Tail multipliers mean"],
            "Tail common std": fmt(row["Tail common-scale std"], 4),
            "Tail boundary": fmt_pct(row["Tail boundary fraction"]),
            "Post identity source": fmt_pct(row["Post identity source fraction"]),
            "Tail cap projection": fmt_pct(row["Tail cap projection"]),
            "Tail fallback": fmt_pct(row["Tail fallback reason"]),
        }
        for row in weight_rows
    ]
    residual_md = [
        {
            "Method": row["Method"],
            "Tail policy": fmt_pct(row["Tail policy selected"]),
            "Tail supervisor": fmt_pct(row["Tail supervisor selected"]),
            "Tail residual mean abs": row["Tail residual mean abs"],
            "Tail residual q95 abs": row["Tail residual q95 abs"],
            "Tail full-bound": fmt_pct(row["Tail full-bound fraction"]),
            "Tail cap projection": fmt_pct(row["Tail cap projection"]),
            "Tail guard": fmt_pct(row["Tail guard trigger"]),
            "Tail headroom": fmt_pct(row["Tail headroom projection"]),
            "Tail zero fallback": fmt_pct(row["Tail zero fallback"]),
        }
        for row in residual_rows
    ]
    gate_md = [
        {
            "Method": row["Method"],
            "Post policy": fmt_pct(row["Post policy selected"]),
            "Post supervisor": fmt_pct(row["Post supervisor selected"]),
            "Tail policy": fmt_pct(row["Tail policy selected"]),
            "Tail supervisor": fmt_pct(row["Tail supervisor selected"]),
            "Tail adv mean": fmt(row["Tail advantage mean"], 3),
            "Tail adv q95": fmt(row["Tail advantage q95"], 3),
            "Policy q-gap": fmt(row["Tail policy q-gap mean"], 3),
            "Supervisor q-gap": fmt(row["Tail supervisor q-gap mean"], 3),
        }
        for row in gate_rows
    ]
    recovery_md = [
        {
            "Method": row["Method"],
            "Warm mean": fmt(row["Warm mean reward"], 3),
            "Worst post-warm": fmt(row["Worst post-warm reward"], 3),
            "Worst episode": row["Worst post-warm episode"],
            "First ep above OF-MPC": row["First episode above OF-MPC"] or "NA",
            "First 5 above OF-MPC": row["First 5 episodes above OF-MPC"] or "NA",
            "Tail20 delta": fmt(row["Tail20 reward delta vs OF-MPC"], 3),
            "Final delta": fmt(row["Final reward delta vs OF-MPC"], 3),
        }
        for row in recovery_rows
    ]

    write_markdown_table(
        FIG_DIR / "performance_summary.md",
        perf_md,
        [
            "Method",
            "Mean reward",
            "Worst post-warm",
            "Tail-20 reward",
            "Final reward",
            "Tail eta RMSE",
            "Tail T RMSE",
            "Tail eta MAE",
            "Tail T MAE",
            "Tail mean abs du",
        ],
    )
    write_markdown_table(
        FIG_DIR / "steady_state_summary.md",
        steady_md,
        ["Method", "Windows", "Steps", "Eta MAE", "T MAE", "Eta RMSE", "T RMSE", "Eta mean signed", "T mean signed"],
    )
    write_markdown_table(
        FIG_DIR / "weight_diagnostics.md",
        weight_md,
        [
            "Method",
            "Tail policy",
            "Tail supervisor",
            "Tail multipliers mean",
            "Tail common std",
            "Tail boundary",
            "Post identity source",
            "Tail cap projection",
            "Tail fallback",
        ],
    )
    write_markdown_table(
        FIG_DIR / "residual_diagnostics.md",
        residual_md,
        [
            "Method",
            "Tail policy",
            "Tail supervisor",
            "Tail residual mean abs",
            "Tail residual q95 abs",
            "Tail full-bound",
            "Tail cap projection",
            "Tail guard",
            "Tail headroom",
            "Tail zero fallback",
        ],
    )
    write_markdown_table(
        FIG_DIR / "gate_diagnostics.md",
        gate_md,
        [
            "Method",
            "Post policy",
            "Post supervisor",
            "Tail policy",
            "Tail supervisor",
            "Tail adv mean",
            "Tail adv q95",
            "Policy q-gap",
            "Supervisor q-gap",
        ],
    )
    write_markdown_table(
        FIG_DIR / "recovery_summary.md",
        recovery_md,
        [
            "Method",
            "Warm mean",
            "Worst post-warm",
            "Worst episode",
            "First ep above OF-MPC",
            "First 5 above OF-MPC",
            "Tail20 delta",
            "Final delta",
        ],
    )

    analysis_summary = {
        "run_specs": {label: spec["path"].relative_to(REPO_ROOT).as_posix() for label, spec in RUN_SPECS.items()},
        "performance_summary": perf_rows,
        "steady_state_summary": steady_rows,
        "weight_diagnostics": weight_rows,
        "residual_diagnostics": residual_rows,
        "gate_diagnostics": gate_rows,
        "recovery_summary": recovery_rows,
    }
    with (FIG_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis_summary, handle, indent=2)

    plot_rewards(runs)
    plot_tail_performance(runs, perf_rows)
    plot_gate_and_actions(gate_rows, weight_rows, residual_rows)
    plot_last_episode_tracking(runs)

    print(f"Wrote analysis artifacts to {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
