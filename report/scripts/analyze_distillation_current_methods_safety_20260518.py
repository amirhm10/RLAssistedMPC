from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_current_methods_safety_20260518"
BASELINE_BUNDLE = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
TAIL_EPISODES = 20
ROLLING_WINDOW = 10
COLLAPSE_THRESHOLD = 5.0


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    family: str
    run_dir: Path
    compare_dir: Path | None
    color: str
    diagnostics_csv: bool = False


RUN_SPECS = [
    RunSpec(
        key="horizon",
        label="Horizon DQN",
        family="horizon",
        run_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_horizon_disturb_fluctuation_mismatch_unified" / "20260518_141636",
        compare_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_horizon_disturb_fluctuation_mismatch" / "20260518_141646",
        color="#1F77B4",
    ),
    RunSpec(
        key="dueling",
        label="Dueling horizon",
        family="dueling_horizon",
        run_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified" / "20260518_140746",
        compare_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_dueling_horizon_disturb_fluctuation_mismatch" / "20260518_140757",
        color="#2563EB",
    ),
    RunSpec(
        key="weights_sac",
        label="Weights SAC",
        family="weights",
        run_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_weights_sac_disturb_fluctuation_mismatch_unified" / "20260518_142138",
        compare_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_weights_sac_disturb_fluctuation_mismatch" / "20260518_142147",
        color="#D97706",
    ),
    RunSpec(
        key="residual_td3",
        label="Residual TD3",
        family="residual",
        run_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified" / "20260518_135423",
        compare_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_residual_td3_disturb_fluctuation" / "20260518_135436",
        color="#7C3AED",
    ),
    RunSpec(
        key="markov_soft",
        label="Markov TD3 soft handoff",
        family="markov",
        run_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_unified" / "20260518_184548",
        compare_dir=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_markov_td3_disturb_fluctuation" / "20260518_184601",
        color="#0B6E4F",
        diagnostics_csv=True,
    ),
    RunSpec(
        key="markov_td3_only",
        label="Markov TD3-only",
        family="markov_reference",
        run_dir=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified"
        / "20260518_091937",
        compare_dir=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_markov_td3_disturb_fluctuation_td3_only_no_safeguard"
        / "20260518_091949",
        color="#DC2626",
        diagnostics_csv=True,
    ),
]

RESIDUAL_REFERENCE_SPECS = {
    "distillation_latest": REPO_ROOT / "Distillation" / "Results" / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified" / "20260518_135423",
    "distillation_prior": REPO_ROOT / "Distillation" / "Results" / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified" / "20260507_212833",
    "polymer_reference": REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260501_000607",
}


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def moving_average(values: np.ndarray, width: int = ROLLING_WINDOW) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < width:
        return arr.copy()
    left = width // 2
    right = width - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def rewards(bundle: dict) -> np.ndarray:
    return np.asarray(bundle["avg_rewards"], dtype=float).reshape(-1)


def compare_mpc_rewards(compare_bundle: dict | None) -> np.ndarray:
    if not compare_bundle:
        return np.asarray([], dtype=float)
    for key in ("avg_rewards_mpc", "avg_rewards"):
        if key in compare_bundle:
            arr = np.asarray(compare_bundle[key], dtype=float).reshape(-1)
            if arr.size:
                return arr
    return np.asarray([], dtype=float)


def steps_per_episode(bundle: dict) -> int:
    return int(bundle.get("time_in_sub_episodes", 400))


def n_inputs(bundle: dict) -> int:
    return int(len(np.asarray(bundle["steady_states"]["ss_inputs"], dtype=float)))


def y_sp_phys(bundle: dict) -> np.ndarray:
    n_in = n_inputs(bundle)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], dtype=float)
    y_range = data_max[n_in:] - data_min[n_in:]
    return y_ss.reshape(1, -1) + np.asarray(bundle["y_sp"], dtype=float) * y_range.reshape(1, -1)


def y_u_arrays(bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rewards(bundle)
    steps = steps_per_episode(bundle)
    n_ep = int(r.size)
    n_out = int(len(np.asarray(bundle["steady_states"]["y_ss"], dtype=float)))
    n_in = n_inputs(bundle)
    y = np.asarray(bundle["y"][1:], dtype=float).reshape(n_ep, steps, n_out)
    u = np.asarray(bundle["u"], dtype=float).reshape(n_ep, steps, n_in)
    sp = y_sp_phys(bundle).reshape(n_ep, steps, n_out)
    return y, u, sp


def baseline_y_u(reference_bundle: dict, baseline_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    r = rewards(reference_bundle)
    steps = steps_per_episode(reference_bundle)
    n_ep = int(r.size)
    n_out = int(len(np.asarray(reference_bundle["steady_states"]["y_ss"], dtype=float)))
    n_in = n_inputs(reference_bundle)
    y = np.asarray(baseline_bundle["y"][1:], dtype=float).reshape(n_ep, steps, n_out)
    u = np.asarray(baseline_bundle["u"], dtype=float).reshape(n_ep, steps, n_in)
    return y, u


def phase_slices(n_ep: int) -> dict[str, slice]:
    return {
        "E1-20": slice(0, min(20, n_ep)),
        "E21-100": slice(min(20, n_ep), min(100, n_ep)),
        "E101-180": slice(min(100, n_ep), min(180, n_ep)),
        "Tail-20": slice(max(0, n_ep - 20), n_ep),
    }


def warm_reference(values: np.ndarray) -> float:
    if values.size >= 10:
        return float(np.mean(values[7:10]))
    return float(np.mean(values[: min(3, values.size)]))


def episode_move_norm(bundle: dict) -> np.ndarray:
    _y, u, _sp = y_u_arrays(bundle)
    n_in = n_inputs(bundle)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    du_span = data_max[:n_in] - data_min[:n_in]
    du = np.diff(np.concatenate([u[:, 0:1, :], u], axis=1), axis=1)
    du_scaled = du / du_span.reshape(1, 1, n_in)
    return np.mean(np.linalg.norm(du_scaled, axis=2), axis=1)


def move_proxy_flags(bundle: dict) -> np.ndarray:
    move = episode_move_norm(bundle)
    med = float(np.median(move))
    mad = float(np.median(np.abs(move - med)))
    threshold = med + 6.0 * max(mad, 1.0e-12)
    return move > threshold


def collapse_flags(values: np.ndarray) -> np.ndarray:
    ref = warm_reference(values)
    return np.asarray(values, dtype=float) < (ref - COLLAPSE_THRESHOLD)


def episode_mean_log(bundle: dict, key: str, default: float = 0.0) -> np.ndarray:
    r = rewards(bundle)
    if key not in bundle or np.asarray(bundle[key]).size == 0:
        return np.full(r.size, default, dtype=float)
    arr = np.asarray(bundle[key], dtype=float)
    steps = steps_per_episode(bundle)
    return arr.reshape(r.size, steps, *arr.shape[1:]).reshape(r.size, steps, -1).mean(axis=(1, 2))


def residual_norm_episode(bundle: dict, key: str) -> np.ndarray:
    r = rewards(bundle)
    if key not in bundle or np.asarray(bundle[key]).size == 0:
        return np.zeros(r.size, dtype=float)
    arr = np.asarray(bundle[key], dtype=float)
    steps = steps_per_episode(bundle)
    return np.linalg.norm(arr.reshape(r.size, steps, -1), axis=2).mean(axis=1)


def method_proxy_flags(spec: RunSpec, bundle: dict) -> dict[str, np.ndarray]:
    r = rewards(bundle)
    collapse = collapse_flags(r)
    move_proxy = move_proxy_flags(bundle)
    flags = {"reward_collapse": collapse, "move_proxy": move_proxy}
    if spec.key == "markov_soft":
        scale = episode_mean_log(bundle, "td3_authority_scale_log", default=1.0)
        probation = episode_mean_log(bundle, "td3_probation_active_log", default=0.0)
        source = np.asarray(bundle["rl_action_source_log"], dtype=int).reshape(r.size, steps_per_episode(bundle))
        non_td3 = np.mean(source != 2, axis=1)
        flags["existing_soft_handoff"] = (scale < 0.999) | (probation > 0.0) | (non_td3 > 0.0)
    if spec.key == "residual_td3":
        projection = episode_mean_log(bundle, "projection_due_to_authority_log", default=0.0)
        gap = episode_mean_log(bundle, "policy_executed_gap_norm_log", default=0.0)
        residual_exec = residual_norm_episode(bundle, "delta_u_res_exec_log")
        flags["residual_projection_risk"] = (projection > 0.5) & (gap > 0.5) & (residual_exec > 0.001)
    combined = np.zeros(r.size, dtype=bool)
    for value in flags.values():
        combined |= value
    flags["combined_proxy"] = combined
    return flags


def tracking_metrics(bundle: dict, baseline_bundle: dict) -> dict:
    y, _u, sp = y_u_arrays(bundle)
    y_mpc, _u_mpc = baseline_y_u(bundle, baseline_bundle)
    err = np.abs(y - sp)
    err_mpc = np.abs(y_mpc - sp)
    n_ep, steps, _ = y.shape
    block_slices = {"SP1": slice(0, steps // 2), "SP2": slice(steps // 2, steps)}
    periods = {"final": slice(n_ep - 1, n_ep), "tail20": slice(max(0, n_ep - TAIL_EPISODES), n_ep)}
    out: dict[str, dict] = {}
    for period_name, ep_slice in periods.items():
        out[period_name] = {}
        for block_name, block_slice in block_slices.items():
            out[period_name][block_name] = {
                "comp_mae": float(np.mean(err[ep_slice, block_slice, 0])),
                "temp_mae": float(np.mean(err[ep_slice, block_slice, 1])),
                "comp_mae_mpc": float(np.mean(err_mpc[ep_slice, block_slice, 0])),
                "temp_mae_mpc": float(np.mean(err_mpc[ep_slice, block_slice, 1])),
            }
    return out


def summarize_markov_diagnostics(spec: RunSpec, bundle: dict) -> dict:
    out: dict[str, object] = {}
    if "rl_action_source_log" in bundle:
        r = rewards(bundle)
        steps = steps_per_episode(bundle)
        src = np.asarray(bundle["rl_action_source_log"], dtype=int).reshape(r.size, steps)
        tail = src[-TAIL_EPISODES:].reshape(-1)
        all_src = src.reshape(-1)
        out["td3_fraction_all"] = float(np.mean(all_src == 2))
        out["td3_fraction_tail20"] = float(np.mean(tail == 2))
        out["nominal_fraction_all"] = float(np.mean((all_src == 0) | (all_src == 4)))
        out["ls_fraction_all"] = float(np.mean((all_src == 3) | (all_src == 5)))
    if spec.key == "markov_soft":
        scale = episode_mean_log(bundle, "td3_authority_scale_log", default=1.0)
        probation = episode_mean_log(bundle, "td3_probation_active_log", default=0.0)
        out["authority_scale_mean_all"] = float(np.mean(scale))
        out["authority_scale_tail20"] = float(np.mean(scale[-TAIL_EPISODES:]))
        out["probation_fraction_all"] = float(np.mean(probation > 0.0))
        out["probation_trigger_count"] = int(bundle.get("td3_probation_trigger_count", 0))
        out["probation_active_episodes"] = [int(i + 1) for i, val in enumerate(probation) if val > 0.0]
    csv_path = spec.run_dir / "markov_stage_diagnostics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        steps = steps_per_episode(bundle)
        tail_df = df.iloc[-TAIL_EPISODES * steps :]
        out["tail20_prediction_score_mean"] = float(np.nanmean(tail_df["executed_prediction_score"]))
        out["tail20_gain_drift_mean"] = float(np.nanmean(tail_df["executed_gain_drift"]))
        out["tail20_cost_margin_mean"] = float(np.nanmean(tail_df["executed_cost_margin"]))
        out["tail20_cost_margin_p95"] = float(np.nanpercentile(tail_df["executed_cost_margin"], 95.0))
    return out


def summarize_residual_bundle(bundle: dict) -> dict:
    r = rewards(bundle)
    steps = steps_per_episode(bundle)
    out = {
        "mean_reward": float(np.mean(r)),
        "tail20_reward": float(np.mean(r[-TAIL_EPISODES:])),
        "final_reward": float(r[-1]),
        "first20_min_reward": float(np.min(r[:20])),
    }
    for key in ("rho_eff_log", "projection_due_to_authority_log", "projection_due_to_deadband_log", "policy_executed_gap_norm_log"):
        if key in bundle and np.asarray(bundle[key]).size:
            arr = np.asarray(bundle[key], dtype=float).reshape(-1)
            out[f"{key}_tail20_mean"] = float(np.mean(arr[-TAIL_EPISODES * steps :]))
            out[f"{key}_mean"] = float(np.mean(arr))
    for key in ("delta_u_res_raw_log", "delta_u_res_exec_log"):
        if key in bundle and np.asarray(bundle[key]).size:
            arr = np.asarray(bundle[key], dtype=float).reshape(r.size, steps, -1)
            norm = np.linalg.norm(arr, axis=2)
            out[f"{key}_tail20_norm_mean"] = float(np.mean(norm[-TAIL_EPISODES:]))
            out[f"{key}_norm_mean"] = float(np.mean(norm))
    for key in ("authority_beta_res", "authority_du0_res", "authority_rho_floor", "rho_mapping_mode", "append_rho_to_state", "authority_use_rho"):
        if key in bundle:
            value = bundle[key]
            if isinstance(value, np.ndarray):
                value = value.astype(float).tolist()
            out[key] = value
    return out


def summarize_runs() -> tuple[dict[str, dict], dict[str, dict], dict]:
    baseline = load_pickle(BASELINE_BUNDLE)
    run_summaries: dict[str, dict] = {}
    flag_by_method: dict[str, dict[str, np.ndarray]] = {}
    for spec in RUN_SPECS:
        bundle = load_pickle(spec.run_dir / "input_data.pkl")
        compare_bundle = load_pickle(spec.compare_dir / "input_data.pkl") if spec.compare_dir else None
        r = rewards(bundle)
        mpc_r = compare_mpc_rewards(compare_bundle)
        tail = slice(max(0, r.size - TAIL_EPISODES), r.size)
        flags = method_proxy_flags(spec, bundle)
        flag_by_method[spec.key] = flags
        run_summaries[spec.key] = {
            "label": spec.label,
            "family": spec.family,
            "run_dir": rel(spec.run_dir),
            "compare_dir": rel(spec.compare_dir) if spec.compare_dir else None,
            "mean_reward": float(np.mean(r)),
            "tail20_reward": float(np.mean(r[tail])),
            "final_reward": float(r[-1]),
            "best_reward": float(np.max(r)),
            "best_episode": int(np.argmax(r) + 1),
            "first20_min_reward": float(np.min(r[:20])),
            "first20_min_episode": int(np.argmin(r[:20]) + 1),
            "warm_reference_reward": warm_reference(r),
            "collapse_episode_count": int(np.sum(flags["reward_collapse"])),
            "move_proxy_episode_count": int(np.sum(flags["move_proxy"])),
            "combined_safety_proxy_episode_count": int(np.sum(flags["combined_proxy"])),
            "tracking": tracking_metrics(bundle, baseline),
            "markov_diagnostics": summarize_markov_diagnostics(spec, bundle),
        }
        if mpc_r.size == r.size:
            run_summaries[spec.key]["own_mpc_tail20_reward"] = float(np.mean(mpc_r[tail]))
            run_summaries[spec.key]["own_mpc_final_reward"] = float(mpc_r[-1])
            run_summaries[spec.key]["tail20_delta_vs_own_mpc"] = float(np.mean(r[tail] - mpc_r[tail]))
        reward_params = bundle.get("reward_params")
        if isinstance(reward_params, dict):
            run_summaries[spec.key]["reward_params"] = {
                key: np.asarray(reward_params[key], dtype=float).tolist()
                for key in ("Q_diag", "R_diag", "k_rel", "band_floor_phys")
                if key in reward_params
            }
    residual_refs: dict[str, dict] = {}
    for key, run_dir in RESIDUAL_REFERENCE_SPECS.items():
        residual_refs[key] = {
            "run_dir": rel(run_dir),
            **summarize_residual_bundle(load_pickle(run_dir / "input_data.pkl")),
        }
    return run_summaries, flag_by_method, residual_refs


def plot_reward_trends(summaries: dict[str, dict]) -> Path:
    fig, ax = plt.subplots(figsize=(13.4, 6.4))
    episodes = None
    for spec in RUN_SPECS:
        bundle = load_pickle(spec.run_dir / "input_data.pkl")
        r = rewards(bundle)
        if episodes is None:
            episodes = np.arange(1, r.size + 1)
        ax.plot(episodes, moving_average(r), color=spec.color, linewidth=2.4 if "Markov" in spec.label else 1.9, label=spec.label)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.axvline(10.5, color="0.55", linewidth=1.0, linestyle=":")
    ax.set_title("Current distillation disturbance-fluctuation runs: learning curves")
    ax.set_xlabel("Sub-episode")
    ax.set_ylabel("Average reward")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8.5, ncol=2)
    out = OUT_DIR / "fig_cross_method_reward_trends.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reward_risk(summaries: dict[str, dict]) -> Path:
    labels = [summaries[spec.key]["label"] for spec in RUN_SPECS]
    x = np.arange(len(labels))
    tail = [summaries[spec.key]["tail20_reward"] for spec in RUN_SPECS]
    first_min = [summaries[spec.key]["first20_min_reward"] for spec in RUN_SPECS]
    colors = [spec.color for spec in RUN_SPECS]

    fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.8))
    axs[0].bar(x, tail, color=colors)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=15, ha="right")
    axs[0].set_ylabel("Tail-20 reward")
    axs[0].set_title("Late performance ranking")
    axs[0].grid(alpha=0.25, axis="y")

    axs[1].scatter(first_min, tail, s=130, color=colors, edgecolor="#111111", linewidth=0.7)
    offsets = {
        "horizon": (-34, -16),
        "dueling": (-18, 12),
        "weights_sac": (-4, -2),
        "residual_td3": (6, 6),
        "markov_soft": (6, 6),
        "markov_td3_only": (6, 6),
    }
    for spec, xv, yv in zip(RUN_SPECS, first_min, tail):
        axs[1].annotate(
            spec.label,
            (xv, yv),
            textcoords="offset points",
            xytext=offsets.get(spec.key, (5, 5)),
            fontsize=8.5,
        )
    axs[1].axvline(0.0, color="0.4", linewidth=0.8)
    axs[1].set_xlabel("Worst first-20 reward")
    axs[1].set_ylabel("Tail-20 reward")
    axs[1].set_title("Safety/performance tradeoff")
    axs[1].grid(alpha=0.25)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = OUT_DIR / "fig_reward_risk_ranking.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tracking(summaries: dict[str, dict]) -> Path:
    labels = [summaries[spec.key]["label"] for spec in RUN_SPECS]
    colors = [spec.color for spec in RUN_SPECS]
    x = np.arange(len(labels))
    metrics = [
        ("SP1", "temp_mae", "SP1 temperature MAE", "K"),
        ("SP2", "temp_mae", "SP2 temperature MAE", "K"),
        ("SP1", "comp_mae", "SP1 composition MAE", "fraction"),
        ("SP2", "comp_mae", "SP2 composition MAE", "fraction"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14.2, 8.8))
    axs = axs.ravel()
    for ax, (block, metric, title, ylabel) in zip(axs, metrics):
        vals = [summaries[spec.key]["tracking"]["tail20"][block][metric] for spec in RUN_SPECS]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"Tail-20 {title}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Current distillation methods: tail-20 tracking errors", y=1.02)
    out = OUT_DIR / "fig_tail20_tracking_metrics.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_markov_soft_handoff() -> Path:
    soft_spec = next(spec for spec in RUN_SPECS if spec.key == "markov_soft")
    td3_spec = next(spec for spec in RUN_SPECS if spec.key == "markov_td3_only")
    soft = load_pickle(soft_spec.run_dir / "input_data.pkl")
    td3 = load_pickle(td3_spec.run_dir / "input_data.pkl")
    r_soft = rewards(soft)
    r_td3 = rewards(td3)
    n_ep = r_soft.size
    steps = steps_per_episode(soft)
    episodes = np.arange(1, n_ep + 1)
    source = np.asarray(soft["rl_action_source_log"], dtype=int).reshape(n_ep, steps)
    td3_fraction = np.mean(source == 2, axis=1)
    nominal_fraction = np.mean((source == 0) | (source == 4), axis=1)
    scale = episode_mean_log(soft, "td3_authority_scale_log", default=1.0)
    probation = episode_mean_log(soft, "td3_probation_active_log", default=0.0)

    fig, axs = plt.subplots(3, 1, figsize=(13.0, 9.6), sharex=True)
    axs[0].plot(episodes, r_td3, color=td3_spec.color, linewidth=1.6, alpha=0.75, label="TD3-only no safeguard")
    axs[0].plot(episodes, r_soft, color=soft_spec.color, linewidth=2.0, label="Soft handoff")
    axs[0].axhline(0.0, color="0.35", linewidth=0.8)
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Markov soft handoff reduces release damage while keeping high late reward")
    axs[0].legend(loc="best", fontsize=9)

    axs[1].plot(episodes, td3_fraction, color="#0B6E4F", linewidth=2.0, label="TD3 fraction")
    axs[1].plot(episodes, nominal_fraction, color="#9CA3AF", linewidth=1.7, label="Nominal fallback fraction")
    axs[1].set_ylim(-0.02, 1.02)
    axs[1].set_ylabel("Fraction")
    axs[1].set_title("Soft handoff still gives TD3 the decision most of the time")
    axs[1].legend(loc="best", fontsize=9)

    axs[2].plot(episodes, scale, color="#D97706", linewidth=2.0, label="Authority scale")
    axs[2].fill_between(episodes, 0.0, probation, color="#7C3AED", alpha=0.22, label="Probation active")
    axs[2].set_ylim(-0.02, 1.05)
    axs[2].set_xlabel("Sub-episode")
    axs[2].set_ylabel("Scale / flag")
    axs[2].set_title("Authority ramp and reward-collapse probation")
    axs[2].legend(loc="best", fontsize=9)

    for ax in axs:
        ax.axvline(10.5, color="0.55", linewidth=1.0, linestyle=":")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = OUT_DIR / "fig_markov_soft_handoff_vs_td3_only.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_residual_distillation_diagnostics() -> Path:
    latest = load_pickle(RESIDUAL_REFERENCE_SPECS["distillation_latest"] / "input_data.pkl")
    prior = load_pickle(RESIDUAL_REFERENCE_SPECS["distillation_prior"] / "input_data.pkl")
    r_latest = rewards(latest)
    r_prior = rewards(prior)
    episodes = np.arange(1, r_latest.size + 1)
    steps = steps_per_episode(latest)
    rho_eff = np.asarray(latest["rho_eff_log"], dtype=float).reshape(r_latest.size, steps).mean(axis=1)
    projection = np.asarray(latest["projection_due_to_authority_log"], dtype=float).reshape(r_latest.size, steps).mean(axis=1)
    exec_norm = residual_norm_episode(latest, "delta_u_res_exec_log")
    raw_norm = residual_norm_episode(latest, "delta_u_res_raw_log")

    fig, axs = plt.subplots(3, 1, figsize=(13.2, 9.4), sharex=True)
    axs[0].plot(episodes, r_prior, color="#9CA3AF", linewidth=1.7, linestyle="--", label="Prior distillation residual, 20260507")
    axs[0].plot(episodes, r_latest, color="#7C3AED", linewidth=2.0, label="Latest distillation residual, 20260518")
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Distillation residual: latest run has a clear release shock")
    axs[0].legend(loc="best", fontsize=9)

    axs[1].plot(episodes, rho_eff, color="#2563EB", linewidth=2.0, label="rho_eff")
    axs[1].plot(episodes, projection, color="#D97706", linewidth=1.8, label="authority projection fraction")
    axs[1].set_ylabel("Fraction / rho")
    axs[1].set_ylim(-0.02, 1.05)
    axs[1].set_title("Rho and projection are active, but they are magnitude checks")
    axs[1].legend(loc="best", fontsize=9)

    axs[2].plot(episodes, raw_norm, color="#DC2626", linewidth=1.7, label="raw residual norm")
    axs[2].plot(episodes, exec_norm, color="#0B6E4F", linewidth=2.0, label="executed residual norm")
    axs[2].set_xlabel("Sub-episode")
    axs[2].set_ylabel("Scaled residual norm")
    axs[2].set_title("Small executed residuals can still be directionally harmful")
    axs[2].legend(loc="best", fontsize=9)

    for ax in axs:
        ax.axvline(15.5, color="0.55", linewidth=1.0, linestyle=":")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = OUT_DIR / "fig_residual_distillation_release_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_residual_cross_case(residual_refs: dict[str, dict]) -> Path:
    labels = ["Dist latest", "Dist prior", "Polymer ref"]
    keys = ["distillation_latest", "distillation_prior", "polymer_reference"]
    colors = ["#7C3AED", "#9CA3AF", "#0B6E4F"]
    metrics = [
        ("rho_eff_log_tail20_mean", "tail-20 rho_eff"),
        ("projection_due_to_authority_log_tail20_mean", "tail-20 authority projection"),
        ("delta_u_res_raw_log_tail20_norm_mean", "tail-20 raw residual norm"),
        ("delta_u_res_exec_log_tail20_norm_mean", "tail-20 executed residual norm"),
        ("policy_executed_gap_norm_log_tail20_mean", "tail-20 raw/executed gap"),
    ]
    fig, axs = plt.subplots(1, len(metrics), figsize=(16.0, 4.8))
    x = np.arange(len(labels))
    for ax, (metric, title) in zip(axs, metrics):
        vals = [float(residual_refs[key].get(metric, np.nan)) for key in keys]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Residual authority is shared, but polymer and distillation operate in different numeric regimes", y=1.02)
    out = OUT_DIR / "fig_residual_polymer_vs_distillation_authority.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_safety_heatmap(summaries: dict[str, dict], flag_by_method: dict[str, dict[str, np.ndarray]]) -> Path:
    phase_names = ["E1-20", "E21-100", "E101-180", "Tail-20"]
    data = []
    labels = []
    for spec in RUN_SPECS:
        flags = flag_by_method[spec.key]["combined_proxy"]
        phases = phase_slices(flags.size)
        data.append([float(np.mean(flags[phases[name]])) if flags[phases[name]].size else np.nan for name in phase_names])
        labels.append(summaries[spec.key]["label"])
    arr = np.asarray(data, dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    im = ax.imshow(arr, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(phase_names)))
    ax.set_xticklabels(phase_names)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="#111111", fontsize=9)
    ax.set_title("Safety-layer activation audit: exact where logged, proxy where candidate diagnostics are missing")
    fig.colorbar(im, ax=ax, label="Fraction of episodes flagged")
    out = OUT_DIR / "fig_safety_activation_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, flag_by_method, residual_refs = summarize_runs()
    figures = {
        "reward_trends": rel(plot_reward_trends(summaries)),
        "reward_risk": rel(plot_reward_risk(summaries)),
        "tracking": rel(plot_tracking(summaries)),
        "markov_soft_handoff": rel(plot_markov_soft_handoff()),
        "residual_distillation": rel(plot_residual_distillation_diagnostics()),
        "residual_cross_case": rel(plot_residual_cross_case(residual_refs)),
        "safety_heatmap": rel(plot_safety_heatmap(summaries, flag_by_method)),
    }
    serializable_flags = {}
    for key, flag_dict in flag_by_method.items():
        serializable_flags[key] = {
            flag_key: {
                "count": int(np.sum(value)),
                "fraction": float(np.mean(value)),
                "episodes": [int(i + 1) for i, active in enumerate(value) if bool(active)][:50],
            }
            for flag_key, value in flag_dict.items()
        }
    summary = {
        "generated_at": "2026-05-18",
        "baseline_bundle": rel(BASELINE_BUNDLE),
        "tail_episodes": TAIL_EPISODES,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "runs": summaries,
        "safety_flags": serializable_flags,
        "residual_references": residual_refs,
        "figures": figures,
        "excluded": {
            "matrix": "excluded by user request",
            "structured_matrix": "excluded by user request",
            "reidentification": "excluded by user request",
            "combined_nominal": "not ranked because no comparable May 18 disturbance-fluctuation combined run was found",
        },
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"figures": figures, "runs": {k: v["tail20_reward"] for k, v in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
