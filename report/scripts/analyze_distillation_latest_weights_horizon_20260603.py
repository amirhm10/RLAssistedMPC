"""Analyze the latest distillation weights and horizon reruns.

This script reads saved result bundles only. It does not launch Aspen or rerun
any controller. Outputs are report-specific CSV summaries, a manifest, and PNG
figures under report/figures/distillation_latest_weights_horizon_20260603/.
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "report" / "figures" / "distillation_latest_weights_horizon_20260603"

OFMPC_PATH = ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"

RUNS = {
    "weights_gaussian": {
        "label": "SG-TD3 weights gaussian",
        "short_label": "Weights SG-TD3",
        "family": "weights",
        "variant": "latest_gaussian_sg_td3",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch"
        / "20260603_124214"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation"
        / "20260603_124224"
        / "input_data.pkl",
        "previous_key": "weights_previous",
    },
    "horizon_epsilon": {
        "label": "DDQN horizon epsilon",
        "short_label": "Horizon DDQN",
        "family": "horizon",
        "variant": "latest_epsilon_ddqn",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260603_130635"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_horizon_disturb_fluctuation_mismatch"
        / "20260603_130644"
        / "input_data.pkl",
        "previous_key": "horizon_previous",
    },
    "dueling_horizon_epsilon": {
        "label": "Dueling DDQN horizon epsilon",
        "short_label": "Dueling horizon",
        "family": "horizon_dueling",
        "variant": "latest_epsilon_dueling_ddqn",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260603_130106"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_dueling_horizon_disturb_fluctuation_mismatch"
        / "20260603_130117"
        / "input_data.pkl",
        "previous_key": "dueling_previous",
    },
    "weights_previous": {
        "label": "SG-TD3 weights previous",
        "short_label": "Weights previous",
        "family": "weights",
        "variant": "previous_sg_td3_param_noise",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch"
        / "20260602_140102"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_weights_sg_td3_critic_warm3_manual_off_disturb_fluctuation"
        / "20260602_140116"
        / "input_data.pkl",
    },
    "horizon_previous": {
        "label": "DDQN horizon previous",
        "short_label": "Horizon previous",
        "family": "horizon",
        "variant": "previous_noisy_ddqn",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260602_144031"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_horizon_disturb_fluctuation_mismatch"
        / "20260602_144045"
        / "input_data.pkl",
    },
    "dueling_previous": {
        "label": "Dueling DDQN horizon previous",
        "short_label": "Dueling previous",
        "family": "horizon_dueling",
        "variant": "previous_noisy_dueling_ddqn",
        "path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260602_144134"
        / "input_data.pkl",
        "compare_path": ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_dueling_horizon_disturb_fluctuation_mismatch"
        / "20260602_144150"
        / "input_data.pkl",
    },
}

LATEST_KEYS = ["weights_gaussian", "horizon_epsilon", "dueling_horizon_epsilon"]
PREVIOUS_KEYS = ["weights_previous", "horizon_previous", "dueling_previous"]
OUTPUT_NAMES = ["x24 ethane", "T85"]
DEFAULT_HORIZON = (6, 3)
REWARD_WARM_EP = 10
EPISODE_LEN = 400

COLORS = {
    "OF-MPC": "#4d4d4d",
    "Weights SG-TD3": "#1b9e77",
    "Horizon DDQN": "#377eb8",
    "Dueling horizon": "#984ea3",
    "Weights previous": "#99d8c9",
    "Horizon previous": "#a6bddb",
    "Dueling previous": "#c2a5cf",
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(values) -> np.ndarray:
    arr = np.asarray(values, float)
    return arr[np.isfinite(arr)]


def finite_mean(values) -> float:
    arr = finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_std(values) -> float:
    arr = finite(values)
    return float(np.std(arr)) if arr.size else float("nan")


def finite_quantile(values, q: float) -> float:
    arr = finite(values)
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def apply_min_max(data, min_val, max_val):
    data = np.asarray(data, float)
    min_val = np.asarray(min_val, float)
    max_val = np.asarray(max_val, float)
    return (data - min_val) / np.maximum(max_val - min_val, 1.0e-12)


def reverse_min_max(data, min_val, max_val):
    data = np.asarray(data, float)
    min_val = np.asarray(min_val, float)
    max_val = np.asarray(max_val, float)
    return data * (max_val - min_val) + min_val


def pick_array(bundle: dict, *keys: str) -> np.ndarray:
    for key in keys:
        value = bundle.get(key)
        if value is not None:
            return np.asarray(value, float)
    raise KeyError(f"No usable array among keys {keys}")


def n_steps(bundle: dict) -> int:
    if bundle.get("nFE") is not None:
        return int(bundle["nFE"])
    return int(np.asarray(bundle["y_sp"]).shape[0])


def episode_len(bundle: dict) -> int:
    return int(bundle.get("time_in_sub_episodes", EPISODE_LEN))


def y_line(bundle: dict, n: int | None = None) -> np.ndarray:
    n = n_steps(bundle) if n is None else int(n)
    y = pick_array(bundle, "y_line_full", "y_rl", "y_mpc", "y")
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] >= n + 1:
        return y[: n + 1, :]
    if y.shape[0] == n:
        return np.vstack([y, y[-1:, :]])
    pad = np.repeat(y[-1:, :], n + 1 - y.shape[0], axis=0)
    return np.vstack([y, pad])


def u_step(bundle: dict, n: int | None = None) -> np.ndarray:
    n = n_steps(bundle) if n is None else int(n)
    u = pick_array(bundle, "u_mpc", "u_rl", "u")
    if u.ndim == 1:
        u = u[:, None]
    if u.shape[0] >= n:
        return u[:n, :]
    pad = np.repeat(u[-1:, :], n - u.shape[0], axis=0)
    return np.vstack([u, pad])


def y_sp_phys(bundle: dict, n: int | None = None) -> np.ndarray:
    n = n_steps(bundle) if n is None else int(n)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = apply_min_max(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    ysp = np.asarray(bundle["y_sp"], float)[:n, :]
    return reverse_min_max(ysp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def delta_u_scaled(bundle: dict, start: int, end: int) -> np.ndarray:
    if bundle.get("delta_u_storage") is not None:
        return np.asarray(bundle["delta_u_storage"], float)[start:end, :]
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", 2))
    u = u_step(bundle)
    u_scaled = apply_min_max(u, data_min[:n_inputs], data_max[:n_inputs])
    ss_inputs = np.asarray(bundle["steady_states"]["ss_inputs"], float)
    ss_scaled = apply_min_max(ss_inputs, data_min[:n_inputs], data_max[:n_inputs])
    du = np.zeros_like(u_scaled)
    du[0, :] = u_scaled[0, :] - ss_scaled
    du[1:, :] = u_scaled[1:, :] - u_scaled[:-1, :]
    return du[start:end, :]


def reward_summary_row(
    method: str,
    family: str,
    variant: str,
    avg_rewards: np.ndarray,
    baseline_avg: np.ndarray | None = None,
) -> dict:
    avg = np.asarray(avg_rewards, float)
    row = {
        "method": method,
        "family": family,
        "variant": variant,
        "mean_reward": finite_mean(avg),
        "worst_post_warm_reward": float(np.nanmin(avg[REWARD_WARM_EP:])),
        "worst_first20_post_warm_reward": float(np.nanmin(avg[REWARD_WARM_EP : REWARD_WARM_EP + 20])),
        "tail20_reward": finite_mean(avg[-20:]),
        "final_reward": float(avg[-1]),
    }
    if baseline_avg is not None:
        base = np.asarray(baseline_avg, float)
        row["tail20_delta_vs_ofmpc"] = row["tail20_reward"] - finite_mean(base[-20:])
        row["final_delta_vs_ofmpc"] = row["final_reward"] - float(base[-1])
        row["worst_first20_delta_vs_ofmpc"] = row["worst_first20_post_warm_reward"] - float(
            np.nanmin(base[REWARD_WARM_EP : REWARD_WARM_EP + 20])
        )
    return row


def tracking_rows(method: str, family: str, bundle: dict, window: str, start: int, end: int) -> list[dict]:
    n = n_steps(bundle)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    y = y_line(bundle, n)
    ysp = y_sp_phys(bundle, n)
    err = y[start + 1 : end + 1, :] - ysp[start:end, :]
    du = delta_u_scaled(bundle, start, end)
    move = float(np.mean(np.abs(du))) if du.size else float("nan")
    rows = []
    for idx, output in enumerate(OUTPUT_NAMES):
        e = err[:, idx]
        rows.append(
            {
                "method": method,
                "family": family,
                "window": window,
                "output": output,
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "max_abs": float(np.max(np.abs(e))),
                "mean_signed": float(np.mean(e)),
                "final_abs": float(abs(e[-1])),
                "mean_abs_du_scaled": move,
            }
        )
    return rows


def pair_key(pair) -> str:
    return f"({int(pair[0])}, {int(pair[1])})"


def horizon_summary_row(method: str, family: str, bundle: dict, start: int, end: int, window: str) -> dict:
    trace = np.asarray(bundle["horizon_executed_trace_log"], int)[start:end, :]
    pairs = [tuple(map(int, row)) for row in trace]
    counts = Counter(pairs)
    top_pair, top_count = counts.most_common(1)[0]
    default_count = counts.get(DEFAULT_HORIZON, 0)
    source_codes = bundle.get("horizon_action_source_codes", {})
    source_log = np.asarray(bundle.get("horizon_action_source_log"), int)[start:end]
    code_to_name = {int(v): str(k) for k, v in source_codes.items()}
    source_parts = {
        f"source_{code_to_name.get(int(code), str(code))}_frac": float(np.mean(source_log == int(code)))
        for code in sorted(np.unique(source_log))
    }
    change = np.asarray(bundle.get("horizon_change_log"), float)[start:end]
    eps = bundle.get("epsilon_trace")
    explore = bundle.get("exploration_trace")
    loss = bundle.get("dqn_loss_trace")
    row = {
        "method": method,
        "family": family,
        "window": window,
        "unique_pairs": int(len(counts)),
        "top_pair": pair_key(top_pair),
        "top_pair_frac": float(top_count / max(1, len(pairs))),
        "default_pair_frac": float(default_count / max(1, len(pairs))),
        "switch_frac": finite_mean(change),
        "mean_predict_horizon": float(np.mean(trace[:, 0])),
        "mean_control_horizon": float(np.mean(trace[:, 1])),
        "trace_tail_epsilon_mean": finite_mean(np.asarray(eps, float)[-1000:]) if eps is not None else float("nan"),
        "trace_final_epsilon": float(np.asarray(eps, float)[-1]) if eps is not None and len(eps) else float("nan"),
        "trace_tail_exploration_mean": finite_mean(np.asarray(explore, float)[-1000:])
        if explore is not None
        else float("nan"),
        "trace_tail_loss_mean": finite_mean(np.asarray(loss, float)[-1000:]) if loss is not None else float("nan"),
        "trace_tail_loss_q95": finite_quantile(np.asarray(loss, float)[-1000:], 0.95) if loss is not None else float("nan"),
    }
    row.update(source_parts)
    return row


def horizon_top_pairs_rows(method: str, bundle: dict, start: int, end: int, top_n: int = 10) -> list[dict]:
    trace = np.asarray(bundle["horizon_executed_trace_log"], int)[start:end, :]
    pairs = [tuple(map(int, row)) for row in trace]
    counts = Counter(pairs)
    total = max(1, len(pairs))
    return [
        {"method": method, "rank": rank, "pair": pair_key(pair), "fraction": float(count / total)}
        for rank, (pair, count) in enumerate(counts.most_common(top_n), start=1)
    ]


def raw_to_multiplier(bundle: dict, raw: np.ndarray) -> np.ndarray:
    low = np.asarray(bundle.get("weight_low_coef", bundle.get("low_coef")), float)
    high = np.asarray(bundle.get("weight_high_coef", bundle.get("high_coef")), float)
    return low + ((np.asarray(raw, float) + 1.0) / 2.0) * (high - low)


def multiplier_diagnostics(mult: np.ndarray, prefix: str = "") -> dict[str, float]:
    arr = np.asarray(mult, float)
    common = np.nanmean(arr, axis=1)
    centered = arr - common[:, None]
    log_disp = np.nanstd(np.log(np.maximum(arr, 1.0e-12)), axis=1)
    dist_identity = np.linalg.norm(arr - 1.0, axis=1)
    return {
        f"{prefix}multiplier_mean": finite_mean(arr),
        f"{prefix}common_multiplier_mean": finite_mean(common),
        f"{prefix}common_multiplier_std_time": finite_std(common),
        f"{prefix}coord_log_dispersion_mean": finite_mean(log_disp),
        f"{prefix}coord_log_dispersion_q95": finite_quantile(log_disp, 0.95),
        f"{prefix}max_relative_coord_abs_mean": finite_mean(np.max(np.abs(centered), axis=1)),
        f"{prefix}dist_identity_mean": finite_mean(dist_identity),
        f"{prefix}dist_identity_q95": finite_quantile(dist_identity, 0.95),
    }


def weights_sg_summary_row(method: str, bundle: dict, start: int, end: int, window: str) -> dict:
    sg_codes = {0: "warm_start", 1: "supervisor", 2: "policy", 3: "held", 4: "fallback"}
    action_codes = {v: k for k, v in bundle.get("weight_action_source_codes", {}).items()}
    sg_source = np.asarray(bundle["sg_selected_source_log"], int)[start:end]
    action_source = np.asarray(bundle["weight_action_source_log"], int)[start:end]
    adv = np.asarray(bundle["sg_advantage_log"], float)[start:end]
    score_policy = np.asarray(bundle["sg_score_policy_log"], float)[start:end]
    score_sup = np.asarray(bundle["sg_score_supervisor_log"], float)[start:end]
    executed = np.asarray(bundle["weight_log"], float)[start:end, :]
    requested = np.asarray(bundle["weight_requested_multiplier_log"], float)[start:end, :]
    policy_raw = np.asarray(bundle["sg_policy_action_raw_log"], float)[start:end, :]
    executed_raw = np.asarray(bundle["sg_executed_action_raw_log"], float)[start:end, :]
    row = {
        "method": method,
        "family": "weights",
        "window": window,
        "sg_policy_frac": float(np.mean(sg_source == 2)),
        "sg_supervisor_frac": float(np.mean(sg_source == 1)),
        "sg_warm_frac": float(np.mean(sg_source == 0)),
        "sg_fallback_frac": float(np.mean(sg_source == 4)),
        "score_policy_gt_supervisor_frac": finite_mean(score_policy > score_sup),
        "advantage_mean": finite_mean(adv),
        "advantage_median": finite_quantile(adv, 0.50),
        "advantage_q05": finite_quantile(adv, 0.05),
        "advantage_q95": finite_quantile(adv, 0.95),
        "policy_raw_norm_mean": finite_mean(np.linalg.norm(policy_raw, axis=1)),
        "executed_raw_norm_mean": finite_mean(np.linalg.norm(executed_raw, axis=1)),
        "policy_executed_raw_gap_mean": finite_mean(np.linalg.norm(policy_raw - executed_raw, axis=1)),
    }
    for code in sorted(np.unique(action_source)):
        row[f"weight_source_{action_codes.get(int(code), str(code))}_frac"] = float(np.mean(action_source == int(code)))
    row.update(multiplier_diagnostics(executed, "executed_"))
    row.update(multiplier_diagnostics(requested, "requested_"))
    return row


def load_all() -> tuple[dict[str, dict], dict[str, dict], dict]:
    bundles = {key: load_pickle(meta["path"]) for key, meta in RUNS.items()}
    compares = {key: load_pickle(meta["compare_path"]) for key, meta in RUNS.items()}
    ofmpc = load_pickle(OFMPC_PATH)
    return bundles, compares, ofmpc


def build_summaries(bundles: dict[str, dict], compares: dict[str, dict], ofmpc: dict):
    latest_compare = compares["weights_gaussian"]
    baseline_avg = np.asarray(latest_compare["avg_rewards_mpc"], float)

    reward_rows = [
        reward_summary_row("OF-MPC", "baseline", "current_reward_baseline", baseline_avg)
    ]
    for key, meta in RUNS.items():
        avg = np.asarray(compares[key]["avg_rewards_rl"], float)
        row = reward_summary_row(meta["short_label"], meta["family"], meta["variant"], avg, baseline_avg)
        previous_key = meta.get("previous_key")
        if previous_key:
            prev_avg = np.asarray(compares[previous_key]["avg_rewards_rl"], float)
            row["tail20_delta_vs_previous"] = row["tail20_reward"] - finite_mean(prev_avg[-20:])
            row["final_delta_vs_previous"] = row["final_reward"] - float(prev_avg[-1])
            row["worst_first20_delta_vs_previous"] = row["worst_first20_post_warm_reward"] - float(
                np.nanmin(prev_avg[REWARD_WARM_EP : REWARD_WARM_EP + 20])
            )
        reward_rows.append(row)

    windows = {
        "early_first20_post_warm": (REWARD_WARM_EP * EPISODE_LEN, (REWARD_WARM_EP + 20) * EPISODE_LEN),
        "tail20": (180 * EPISODE_LEN, 200 * EPISODE_LEN),
        "final_episode": (199 * EPISODE_LEN, 200 * EPISODE_LEN),
    }

    tracking = []
    for window, (start, end) in windows.items():
        tracking.extend(tracking_rows("OF-MPC", "baseline", ofmpc, window, start, end))
        for key in LATEST_KEYS + PREVIOUS_KEYS:
            meta = RUNS[key]
            tracking.extend(tracking_rows(meta["short_label"], meta["family"], bundles[key], window, start, end))

    horizon_rows = []
    top_pair_rows = []
    for key in ["horizon_epsilon", "dueling_horizon_epsilon", "horizon_previous", "dueling_previous"]:
        meta = RUNS[key]
        for window, (start, end) in {
            "early_first20_post_warm": windows["early_first20_post_warm"],
            "tail20": windows["tail20"],
        }.items():
            horizon_rows.append(horizon_summary_row(meta["short_label"], meta["family"], bundles[key], start, end, window))
        top_pair_rows.extend(
            horizon_top_pairs_rows(
                meta["short_label"],
                bundles[key],
                windows["tail20"][0],
                windows["tail20"][1],
            )
        )

    weights_rows = []
    for key in ["weights_gaussian", "weights_previous"]:
        meta = RUNS[key]
        for window, (start, end) in {
            "early_first20_post_warm": windows["early_first20_post_warm"],
            "middle_41_120": (40 * EPISODE_LEN, 120 * EPISODE_LEN),
            "tail20": windows["tail20"],
        }.items():
            weights_rows.append(weights_sg_summary_row(meta["short_label"], bundles[key], start, end, window))

    return reward_rows, tracking, horizon_rows, top_pair_rows, weights_rows


def value_from_rows(rows: list[dict], method: str, key: str, **filters):
    for row in rows:
        if row.get("method") != method:
            continue
        if all(row.get(k) == v for k, v in filters.items()):
            return row.get(key)
    return float("nan")


def plot_reward_curves(reward_rows: list[dict], compares: dict[str, dict]) -> Path:
    path = OUT_DIR / "fig_reward_curves_latest_three_vs_previous.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(1, 201)
    baseline = np.asarray(compares["weights_gaussian"]["avg_rewards_mpc"], float)
    ax.plot(x, baseline, color=COLORS["OF-MPC"], lw=2.0, label="OF-MPC")
    for key in LATEST_KEYS:
        meta = RUNS[key]
        avg = np.asarray(compares[key]["avg_rewards_rl"], float)
        ax.plot(x, avg, lw=2.0, color=COLORS[meta["short_label"]], label=meta["short_label"])
    for key in PREVIOUS_KEYS:
        meta = RUNS[key]
        avg = np.asarray(compares[key]["avg_rewards_rl"], float)
        ax.plot(x, avg, lw=1.2, ls="--", color=COLORS[meta["short_label"]], label=meta["short_label"])
    ax.axvline(REWARD_WARM_EP, color="#777777", lw=1.0, ls=":", label="post-warm release")
    ax.set_title("Episode reward: latest reruns versus June 2 references")
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_reward_summary(reward_rows: list[dict]) -> Path:
    path = OUT_DIR / "fig_reward_summary_latest_vs_previous.png"
    methods = ["OF-MPC", "Weights SG-TD3", "Weights previous", "Horizon DDQN", "Horizon previous", "Dueling horizon", "Dueling previous"]
    metrics = [
        ("tail20_reward", "Tail-20"),
        ("final_reward", "Final"),
        ("worst_first20_post_warm_reward", "Worst first 20 post-warm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True)
    for ax, (field, title) in zip(axes, metrics):
        vals = [value_from_rows(reward_rows, method, field) for method in methods]
        colors = [COLORS.get(method, "#777777") for method in methods]
        ax.bar(np.arange(len(methods)), vals, color=colors)
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=55, ha="right")
    axes[0].set_ylabel("Reward")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_tail_tracking_mae(tracking_rows_data: list[dict]) -> Path:
    path = OUT_DIR / "fig_tail_tracking_mae_latest_vs_previous.png"
    methods = ["OF-MPC", "Weights SG-TD3", "Weights previous", "Horizon DDQN", "Horizon previous", "Dueling horizon", "Dueling previous"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, output in zip(axes, OUTPUT_NAMES):
        vals = [
            value_from_rows(tracking_rows_data, method, "mae", window="tail20", output=output)
            for method in methods
        ]
        ax.bar(np.arange(len(methods)), vals, color=[COLORS.get(method, "#777777") for method in methods])
        ax.set_title(f"Tail MAE: {output}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=55, ha="right")
        ax.set_ylabel("Physical MAE")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_tail_tracking_overlay(bundles: dict[str, dict], ofmpc: dict) -> Path:
    path = OUT_DIR / "fig_tail_tracking_overlay_latest_three.png"
    start, end = 180 * EPISODE_LEN, 200 * EPISODE_LEN
    x = np.arange(start, end) / EPISODE_LEN + 1.0
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    plot_items = [("OF-MPC", ofmpc)] + [(RUNS[key]["short_label"], bundles[key]) for key in LATEST_KEYS]
    for ax_idx, ax in enumerate(axes):
        sp = y_sp_phys(ofmpc)[start:end, ax_idx]
        ax.plot(x, sp, color="#111111", lw=1.1, ls=":", label="setpoint")
        for label, bundle in plot_items:
            yy = y_line(bundle)[start + 1 : end + 1, ax_idx]
            ax.plot(x, yy, lw=1.4, color=COLORS.get(label, "#777777"), label=label)
        ax.set_ylabel(OUTPUT_NAMES[ax_idx])
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Tail tracking overlay for latest three runners")
    axes[1].set_xlabel("Subepisode")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_horizon_usage(horizon_rows: list[dict], top_pair_rows: list[dict]) -> Path:
    path = OUT_DIR / "fig_horizon_usage_and_stability.png"
    methods = ["Horizon DDQN", "Horizon previous", "Dueling horizon", "Dueling previous"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    unique_vals = [value_from_rows(horizon_rows, method, "unique_pairs", window="tail20") for method in methods]
    switch_vals = [value_from_rows(horizon_rows, method, "switch_frac", window="tail20") for method in methods]
    x = np.arange(len(methods))
    width = 0.36
    axes[0].bar(x - width / 2, unique_vals, width, label="unique pairs", color="#377eb8")
    axes[0].set_ylabel("Unique tail recipes")
    ax2 = axes[0].twinx()
    ax2.bar(x + width / 2, switch_vals, width, label="switch fraction", color="#e41a1c")
    ax2.set_ylabel("Tail switch fraction")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=35, ha="right")
    axes[0].set_title("Horizon stability")
    axes[0].grid(True, axis="y", alpha=0.25)

    latest_top = [r for r in top_pair_rows if r["method"] in {"Horizon DDQN", "Dueling horizon"} and r["rank"] <= 6]
    labels = [f"{r['method'].replace(' horizon', '')}\n{r['pair']}" for r in latest_top]
    vals = [r["fraction"] for r in latest_top]
    axes[1].bar(np.arange(len(vals)), vals, color=["#377eb8" if r["method"] == "Horizon DDQN" else "#984ea3" for r in latest_top])
    axes[1].set_xticks(np.arange(len(vals)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("Tail fraction")
    axes[1].set_title("Top tail recipes in latest horizon runs")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_epsilon_reward(compares: dict[str, dict], bundles: dict[str, dict]) -> Path:
    path = OUT_DIR / "fig_horizon_epsilon_and_reward.png"
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=False)
    x_ep = np.arange(1, 201)
    for key in ["horizon_epsilon", "dueling_horizon_epsilon"]:
        meta = RUNS[key]
        axes[0].plot(x_ep, compares[key]["avg_rewards_rl"], color=COLORS[meta["short_label"]], lw=2.0, label=meta["short_label"])
    axes[0].plot(x_ep, compares["horizon_epsilon"]["avg_rewards_mpc"], color=COLORS["OF-MPC"], lw=1.6, label="OF-MPC")
    axes[0].axvline(REWARD_WARM_EP, color="#777777", lw=1.0, ls=":")
    axes[0].set_title("Reward response after epsilon-greedy rerun")
    axes[0].set_ylabel("Average reward")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for key in ["horizon_epsilon", "dueling_horizon_epsilon"]:
        meta = RUNS[key]
        eps = np.asarray(bundles[key]["epsilon_trace"], float)
        x = np.linspace(REWARD_WARM_EP + 1, 200, eps.size)
        axes[1].plot(x, eps, color=COLORS[meta["short_label"]], lw=2.0, label=meta["short_label"])
    axes[1].set_title("Saved epsilon traces")
    axes[1].set_xlabel("Approximate subepisode")
    axes[1].set_ylabel("Epsilon")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_weights_gate(weights_rows: list[dict], bundles: dict[str, dict]) -> Path:
    path = OUT_DIR / "fig_weights_sg_gate_gaussian_vs_previous.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    windows = ["early_first20_post_warm", "middle_41_120", "tail20"]
    labels = ["early", "middle", "tail"]
    x = np.arange(len(labels))
    for method, color in [("Weights SG-TD3", COLORS["Weights SG-TD3"]), ("Weights previous", COLORS["Weights previous"])]:
        vals = [value_from_rows(weights_rows, method, "sg_policy_frac", window=w) for w in windows]
        axes[0, 0].plot(x, vals, marker="o", lw=2.0, color=color, label=method)
        adv = [value_from_rows(weights_rows, method, "advantage_median", window=w) for w in windows]
        axes[0, 1].plot(x, adv, marker="o", lw=2.0, color=color, label=method)
        disp = [value_from_rows(weights_rows, method, "executed_coord_log_dispersion_mean", window=w) for w in windows]
        axes[1, 0].plot(x, disp, marker="o", lw=2.0, color=color, label=method)
        gap = [value_from_rows(weights_rows, method, "policy_executed_raw_gap_mean", window=w) for w in windows]
        axes[1, 1].plot(x, gap, marker="o", lw=2.0, color=color, label=method)
    titles = [
        "SG policy selection",
        "Median policy-supervisor score advantage",
        "Executed multiplier log-dispersion",
        "Policy/executed raw-action gap",
    ]
    ylabels = ["fraction", "score", "std(log multipliers)", "raw norm"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_next_experiment_scorecard(reward_rows: list[dict], tracking_rows_data: list[dict], horizon_rows: list[dict], weights_rows: list[dict]) -> Path:
    path = OUT_DIR / "fig_next_experiment_scorecard.png"
    rows = [
        ("Weights SG-TD3", "tail reward gain", value_from_rows(reward_rows, "Weights SG-TD3", "tail20_delta_vs_ofmpc")),
        ("Weights SG-TD3", "worst early gain", value_from_rows(reward_rows, "Weights SG-TD3", "worst_first20_delta_vs_ofmpc")),
        ("Weights SG-TD3", "tail policy frac", value_from_rows(weights_rows, "Weights SG-TD3", "sg_policy_frac", window="tail20")),
        ("Horizon DDQN", "tail reward gain", value_from_rows(reward_rows, "Horizon DDQN", "tail20_delta_vs_ofmpc")),
        ("Horizon DDQN", "worst early gain", value_from_rows(reward_rows, "Horizon DDQN", "worst_first20_delta_vs_ofmpc")),
        ("Horizon DDQN", "tail switch frac", value_from_rows(horizon_rows, "Horizon DDQN", "switch_frac", window="tail20")),
        ("Dueling horizon", "tail reward gain", value_from_rows(reward_rows, "Dueling horizon", "tail20_delta_vs_ofmpc")),
        ("Dueling horizon", "worst early gain", value_from_rows(reward_rows, "Dueling horizon", "worst_first20_delta_vs_ofmpc")),
        ("Dueling horizon", "tail switch frac", value_from_rows(horizon_rows, "Dueling horizon", "switch_frac", window="tail20")),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    labels = [f"{method}\n{metric}" for method, metric, _ in rows]
    vals = [val for _, _, val in rows]
    colors = [COLORS.get(method, "#777777") for method, _, _ in rows]
    ax.bar(np.arange(len(rows)), vals, color=colors)
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Scorecard for choosing the next experiment")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_figures(
    reward_rows: list[dict],
    tracking_rows_data: list[dict],
    horizon_rows: list[dict],
    top_pair_rows: list[dict],
    weights_rows: list[dict],
    bundles: dict[str, dict],
    compares: dict[str, dict],
    ofmpc: dict,
) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_reward_curves(reward_rows, compares),
        plot_reward_summary(reward_rows),
        plot_tail_tracking_mae(tracking_rows_data),
        plot_tail_tracking_overlay(bundles, ofmpc),
        plot_horizon_usage(horizon_rows, top_pair_rows),
        plot_epsilon_reward(compares, bundles),
        plot_weights_gate(weights_rows, bundles),
        plot_next_experiment_scorecard(reward_rows, tracking_rows_data, horizon_rows, weights_rows),
    ]
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles, compares, ofmpc = load_all()
    reward_rows, tracking, horizon_rows, top_pair_rows, weights_rows = build_summaries(bundles, compares, ofmpc)

    write_csv(OUT_DIR / "reward_summary.csv", reward_rows)
    write_csv(OUT_DIR / "tracking_summary.csv", tracking)
    write_csv(OUT_DIR / "horizon_summary.csv", horizon_rows)
    write_csv(OUT_DIR / "horizon_top_pairs_tail.csv", top_pair_rows)
    write_csv(OUT_DIR / "weights_sg_summary.csv", weights_rows)

    figure_paths = make_figures(reward_rows, tracking, horizon_rows, top_pair_rows, weights_rows, bundles, compares, ofmpc)

    manifest = {
        "created_by": rel(Path(__file__)),
        "artifact_dir": rel(OUT_DIR),
        "source_paths": {
            "ofmpc": rel(OFMPC_PATH),
            **{key: rel(meta["path"]) for key, meta in RUNS.items()},
            **{f"{key}_compare": rel(meta["compare_path"]) for key, meta in RUNS.items()},
        },
        "csv_outputs": [
            rel(OUT_DIR / "reward_summary.csv"),
            rel(OUT_DIR / "tracking_summary.csv"),
            rel(OUT_DIR / "horizon_summary.csv"),
            rel(OUT_DIR / "horizon_top_pairs_tail.csv"),
            rel(OUT_DIR / "weights_sg_summary.csv"),
        ],
        "figure_outputs": [rel(path) for path in figure_paths],
        "windows": {
            "early_first20_post_warm_steps": [REWARD_WARM_EP * EPISODE_LEN, (REWARD_WARM_EP + 20) * EPISODE_LEN],
            "tail20_steps": [180 * EPISODE_LEN, 200 * EPISODE_LEN],
            "final_episode_steps": [199 * EPISODE_LEN, 200 * EPISODE_LEN],
        },
        "notes": [
            "Reward rows use the corresponding compare bundles for RL rewards and the latest compare bundle for the OF-MPC current-reward baseline.",
            "Tracking rows are computed directly from saved y_line_full or y, y_sp, and scaling artifacts.",
            "Horizon action-source fractions include held-interval steps, so policy-live fractions are lower than decision-step-only acceptance.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote analysis artifacts to {rel(OUT_DIR)}")
    for path in figure_paths:
        print(f"  {rel(path)}")


if __name__ == "__main__":
    main()
