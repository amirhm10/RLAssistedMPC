from __future__ import annotations

import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.helpers import apply_min_max, reverse_min_max


LOOSE_RUN_DIR = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_040_looser_accept" / "20260514_013414"
REF_RUN_DIR = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008" / "20260511_230422"
MPC_BASELINE = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_looser_acceptance_20260514"
KNOWN_NOMINAL_COST_RELATIVE_TOL = {
    LOOSE_RUN_DIR.resolve(): 0.30,
    REF_RUN_DIR.resolve(): 0.10,
}


@dataclass
class RunBundle:
    label: str
    run_dir: Path
    bundle: dict
    stage_df: pd.DataFrame

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle.get("avg_rewards", []), float)

    @property
    def n_episodes(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def steps_per_episode(self) -> int:
        if self.n_episodes <= 0:
            return 0
        return int(len(self.stage_df) // self.n_episodes)

    @property
    def warm_start_step(self) -> int:
        return int(self.bundle.get("warm_start_step", 0))

    @property
    def warm_end_episode(self) -> int | None:
        steps = self.steps_per_episode
        if steps <= 0:
            return None
        return int(self.warm_start_step // steps)

    @property
    def z_bound(self) -> float:
        return float(self.bundle.get("markov_z_bound", np.nan))

    @property
    def s_pred_min(self) -> float:
        return float(self.bundle.get("markov_s_pred_min", np.nan))

    @property
    def gain_drift_max(self) -> float:
        return float(self.bundle.get("markov_gain_drift_max", np.nan))

    @property
    def nominal_cost_relative_tol(self) -> float:
        value = self.bundle.get("nominal_cost_relative_tol")
        if value is not None:
            return float(value)
        return float(KNOWN_NOMINAL_COST_RELATIVE_TOL.get(self.run_dir.resolve(), np.nan))


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_run(label: str, run_dir: Path) -> RunBundle:
    stage_df = pd.read_csv(run_dir / "markov_stage_diagnostics.csv")
    return RunBundle(
        label=label,
        run_dir=run_dir,
        bundle=_load_pickle(run_dir / "input_data.pkl"),
        stage_df=stage_df,
    )


def _episode_mean(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes]
    if arr.ndim == 1:
        arr = arr.reshape(n_episodes, steps)
        out = np.empty(n_episodes, dtype=float)
        for idx, row in enumerate(arr):
            finite = row[np.isfinite(row)]
            out[idx] = float(np.mean(finite)) if finite.size else np.nan
        return out
    arr = arr.reshape(n_episodes, steps, -1)
    flat = arr.reshape(n_episodes, steps, -1)
    out = np.empty(n_episodes, dtype=float)
    for idx, block in enumerate(flat):
        finite = block[np.isfinite(block)]
        out[idx] = float(np.mean(finite)) if finite.size else np.nan
    return out


def _episode_fraction(values: np.ndarray, n_episodes: int, target: int) -> np.ndarray:
    arr = np.asarray(values, int)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.size // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps)
    return np.mean(arr == int(target), axis=1)


def _episode_fraction_bool(mask: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(mask, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.size // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx, row in enumerate(arr):
        finite = row[np.isfinite(row)]
        out[idx] = float(np.mean(finite)) if finite.size else np.nan
    return out


def _episode_masked_fraction(mask: np.ndarray, valid: np.ndarray, n_episodes: int) -> np.ndarray:
    mask = np.asarray(mask, bool)
    valid = np.asarray(valid, bool)
    if n_episodes <= 0 or mask.size == 0:
        return np.asarray([], float)
    steps = mask.size // n_episodes
    mask = mask[: steps * n_episodes].reshape(n_episodes, steps)
    valid = valid[: steps * n_episodes].reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx in range(n_episodes):
        use = valid[idx]
        out[idx] = float(np.mean(mask[idx, use])) if np.any(use) else np.nan
    return out


def _episode_masked_mean(values: np.ndarray, valid: np.ndarray, n_episodes: int) -> np.ndarray:
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    if n_episodes <= 0 or values.size == 0:
        return np.asarray([], float)
    steps = values.shape[0] // n_episodes
    values = values[: steps * n_episodes]
    valid = valid[: steps * n_episodes]
    if values.ndim == 1:
        values = values.reshape(n_episodes, steps)
        valid = valid.reshape(n_episodes, steps)
        out = np.empty(n_episodes, dtype=float)
        for idx in range(n_episodes):
            row = values[idx, valid[idx]]
            row = row[np.isfinite(row)]
            out[idx] = float(np.mean(row)) if row.size else np.nan
        return out
    values = values.reshape(n_episodes, steps, -1)
    valid = valid.reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx in range(n_episodes):
        row = values[idx, valid[idx], :]
        row = row[np.isfinite(row)]
        out[idx] = float(np.mean(row)) if row.size else np.nan
    return out


def _episode_masked_vec_norm(values: np.ndarray, valid: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps, -1)
    valid = valid[: steps * n_episodes].reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx in range(n_episodes):
        block = arr[idx, valid[idx], :]
        if block.size == 0:
            out[idx] = np.nan
            continue
        out[idx] = float(np.mean(np.linalg.norm(block, axis=1)))
    return out


def _episode_masked_raw_saturation(values: np.ndarray, valid: np.ndarray, n_episodes: int, threshold: float = 0.95) -> np.ndarray:
    arr = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps, -1)
    valid = valid[: steps * n_episodes].reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx in range(n_episodes):
        block = arr[idx, valid[idx], :]
        if block.size == 0:
            out[idx] = np.nan
            continue
        out[idx] = float(np.mean(np.max(np.abs(block), axis=1) >= float(threshold)))
    return out


def _window_mask(run: RunBundle, *, tail_episodes: int | None = None) -> np.ndarray:
    steps = run.steps_per_episode
    live = run.stage_df["step"].to_numpy(dtype=int) > run.warm_start_step
    if tail_episodes is None:
        return live
    start_step = max(0, run.n_episodes - int(tail_episodes)) * steps
    tail = run.stage_df["step"].to_numpy(dtype=int) >= start_step
    return live & tail


def _requested_valid_mask(run: RunBundle, base_mask: np.ndarray) -> np.ndarray:
    req_score = run.stage_df["requested_prediction_score"].to_numpy(dtype=float)
    return np.asarray(base_mask, bool) & np.isfinite(req_score)


def _ls_valid_mask(run: RunBundle, base_mask: np.ndarray) -> np.ndarray:
    ls_score = run.stage_df["ls_prediction_score"].to_numpy(dtype=float)
    return np.asarray(base_mask, bool) & np.isfinite(ls_score)


def _gate_arrays(run: RunBundle) -> dict[str, np.ndarray]:
    df = run.stage_df
    score_pass = df["requested_prediction_score"].to_numpy(dtype=float) > run.s_pred_min
    drift_pass = df["requested_gain_drift"].to_numpy(dtype=float) <= run.gain_drift_max
    cost_pass = df["requested_cost_guard_pass"].to_numpy(dtype=int) == 1
    all_pass = score_pass & drift_pass & cost_pass
    return {
        "score_pass": score_pass,
        "drift_pass": drift_pass,
        "cost_pass": cost_pass,
        "all_pass": all_pass,
    }


def _window_gate_metrics(run: RunBundle, mask: np.ndarray, *, kind: str = "requested") -> dict[str, float]:
    df = run.stage_df
    if kind == "requested":
        score = df["requested_prediction_score"].to_numpy(dtype=float)
        drift = df["requested_gain_drift"].to_numpy(dtype=float)
        cost_pass = df["requested_cost_guard_pass"].to_numpy(dtype=int) == 1
        z = np.asarray(run.bundle["rl_requested_z_log"], float)
        raw = np.asarray(run.bundle["rl_requested_raw_action_log"], float)
    elif kind == "ls":
        score = df["ls_prediction_score"].to_numpy(dtype=float)
        drift = df["ls_gain_drift"].to_numpy(dtype=float)
        cost_pass = df["ls_cost_guard_pass"].to_numpy(dtype=int) == 1
        z = np.asarray(run.bundle["rl_ls_z_log"], float)
        raw = z / max(run.z_bound, 1.0e-12)
    else:
        raise ValueError("kind must be 'requested' or 'ls'.")

    use = np.asarray(mask, bool) & np.isfinite(score)
    score_pass = score > run.s_pred_min
    drift_pass = drift <= run.gain_drift_max
    all_pass = score_pass & drift_pass & cost_pass
    z_use = z[use]
    raw_use = raw[use]

    out = {
        "n_steps": int(np.sum(use)),
        "score_pass_fraction": float(np.mean(score_pass[use])) if np.any(use) else np.nan,
        "drift_pass_fraction": float(np.mean(drift_pass[use])) if np.any(use) else np.nan,
        "cost_pass_fraction": float(np.mean(cost_pass[use])) if np.any(use) else np.nan,
        "all_pass_fraction": float(np.mean(all_pass[use])) if np.any(use) else np.nan,
        "score_mean": float(np.nanmean(score[use])) if np.any(use) else np.nan,
        "drift_mean": float(np.nanmean(drift[use])) if np.any(use) else np.nan,
        "z_norm_mean": float(np.mean(np.linalg.norm(z_use, axis=1))) if z_use.size else np.nan,
        "raw_norm_mean": float(np.mean(np.linalg.norm(raw_use, axis=1))) if raw_use.size else np.nan,
        "raw_saturation_fraction": float(np.mean(np.max(np.abs(raw_use), axis=1) >= 0.95)) if raw_use.size else np.nan,
    }

    if kind == "requested" and np.any(use):
        score_only = (~score_pass) & drift_pass & cost_pass
        drift_only = score_pass & (~drift_pass) & cost_pass
        score_and_drift = (~score_pass) & (~drift_pass) & cost_pass
        cost_involved = ~cost_pass
        out.update(
            {
                "score_only_fail_fraction": float(np.mean(score_only[use])),
                "drift_only_fail_fraction": float(np.mean(drift_only[use])),
                "score_and_drift_fail_fraction": float(np.mean(score_and_drift[use])),
                "cost_involved_fail_fraction": float(np.mean(cost_involved[use] & ~all_pass[use])),
            }
        )
    return out


def _final_test_episode_metrics(run: RunBundle, mpc_bundle: dict) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    time_in_sub_episodes = int(run.bundle["time_in_sub_episodes"])
    rl_y = np.asarray(run.bundle["y_line_full"], float)[-(time_in_sub_episodes + 1) :, :]
    rl_u = np.asarray(run.bundle["u_step_full"], float)[-time_in_sub_episodes:, :]
    mpc_y = np.asarray(mpc_bundle["y_mpc"], float)[-(time_in_sub_episodes + 1) :, :]
    mpc_u = np.asarray(mpc_bundle["u_mpc"], float)[-time_in_sub_episodes:, :]

    data_min = np.asarray(run.bundle["data_min"], float)
    data_max = np.asarray(run.bundle["data_max"], float)
    n_inputs = int(rl_u.shape[1])
    y_ss = np.asarray(run.bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = apply_min_max(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled_dev = np.asarray(run.bundle["y_sp"], float)[-time_in_sub_episodes:, :]
    y_sp_phys = reverse_min_max(y_sp_scaled_dev + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])

    rl_err = rl_y[1:, :] - y_sp_phys
    mpc_err = mpc_y[1:, :] - y_sp_phys
    rl_du = np.diff(np.vstack([rl_u[0:1, :], rl_u]), axis=0)
    mpc_du = np.diff(np.vstack([mpc_u[0:1, :], mpc_u]), axis=0)

    metrics = {
        "final_reward_rl": float(run.avg_rewards[-1]),
        "final_reward_mpc": float(np.asarray(mpc_bundle["avg_rewards"], float)[-1]),
        "eta_rmse_rl": float(np.sqrt(np.mean(np.square(rl_err[:, 0])))),
        "eta_rmse_mpc": float(np.sqrt(np.mean(np.square(mpc_err[:, 0])))),
        "temp_rmse_rl": float(np.sqrt(np.mean(np.square(rl_err[:, 1])))),
        "temp_rmse_mpc": float(np.sqrt(np.mean(np.square(mpc_err[:, 1])))),
        "eta_iae_rl": float(np.sum(np.abs(rl_err[:, 0]))),
        "eta_iae_mpc": float(np.sum(np.abs(mpc_err[:, 0]))),
        "temp_iae_rl": float(np.sum(np.abs(rl_err[:, 1]))),
        "temp_iae_mpc": float(np.sum(np.abs(mpc_err[:, 1]))),
        "input_move_norm_rl": float(np.mean(np.linalg.norm(rl_du, axis=1))),
        "input_move_norm_mpc": float(np.mean(np.linalg.norm(mpc_du, axis=1))),
    }
    payload = {
        "rl_y": rl_y,
        "rl_u": rl_u,
        "mpc_y": mpc_y,
        "mpc_u": mpc_u,
        "y_sp_phys": y_sp_phys,
    }
    return metrics, payload


def _collect_metrics(run: RunBundle, mpc_bundle: dict) -> dict[str, float]:
    live_mask = _window_mask(run)
    tail_mask = _window_mask(run, tail_episodes=50)
    req_live = _requested_valid_mask(run, live_mask)
    req_tail = _requested_valid_mask(run, tail_mask)
    ls_live = _ls_valid_mask(run, live_mask)
    ls_tail = _ls_valid_mask(run, tail_mask)
    final_metrics, _payload = _final_test_episode_metrics(run, mpc_bundle)

    src = np.asarray(run.bundle["rl_action_source_log"], int)
    td3_ep = _episode_fraction(src, run.n_episodes, 2)
    ls_ep = _episode_fraction(src, run.n_episodes, 3)
    nom_ep = _episode_fraction(src, run.n_episodes, 4)

    req_live_metrics = _window_gate_metrics(run, req_live, kind="requested")
    req_tail_metrics = _window_gate_metrics(run, req_tail, kind="requested")
    ls_live_metrics = _window_gate_metrics(run, ls_live, kind="ls")
    ls_tail_metrics = _window_gate_metrics(run, ls_tail, kind="ls")

    metrics: dict[str, float] = {
        "z_bound": run.z_bound,
        "s_pred_min": run.s_pred_min,
        "gain_drift_max": run.gain_drift_max,
        "nominal_cost_relative_tol": run.nominal_cost_relative_tol,
        "reward_mean": float(np.mean(run.avg_rewards)),
        "reward_tail50": float(np.mean(run.avg_rewards[-50:])),
        "reward_final": float(run.avg_rewards[-1]),
        "td3_frac_ep11_50": float(np.mean(td3_ep[10:50])),
        "td3_frac_tail50": float(np.mean(td3_ep[-50:])),
        "ls_frac_tail50": float(np.mean(ls_ep[-50:])),
        "nominal_frac_tail50": float(np.mean(nom_ep[-50:])),
        "bc_active_tail50": float(np.nanmean(np.asarray(run.bundle["bc_active_log"], float).reshape(run.n_episodes, run.steps_per_episode)[-50:, :])),
        "bc_weight_tail50": float(np.nanmean(np.asarray(run.bundle["bc_weight_log"], float).reshape(run.n_episodes, run.steps_per_episode)[-50:, :])),
    }

    for prefix, values in (
        ("requested_live", req_live_metrics),
        ("requested_tail", req_tail_metrics),
        ("ls_live", ls_live_metrics),
        ("ls_tail", ls_tail_metrics),
    ):
        for key, value in values.items():
            metrics[f"{prefix}_{key}"] = value

    metrics.update(final_metrics)
    return metrics


def _window_bar_values(metrics: dict[str, float], prefix: str) -> list[float]:
    return [
        float(metrics[f"{prefix}_score_pass_fraction"]),
        float(metrics[f"{prefix}_drift_pass_fraction"]),
        float(metrics[f"{prefix}_cost_pass_fraction"]),
        float(metrics[f"{prefix}_all_pass_fraction"]),
    ]


def _plot_reward_and_mix(loose: RunBundle, ref: RunBundle, mpc_bundle: dict, out_dir: Path) -> Path:
    x = np.arange(1, loose.n_episodes + 1)
    loose_src = np.asarray(loose.bundle["rl_action_source_log"], int)
    ref_src = np.asarray(ref.bundle["rl_action_source_log"], int)
    loose_td3 = _episode_fraction(loose_src, loose.n_episodes, 2)
    loose_ls = _episode_fraction(loose_src, loose.n_episodes, 3)
    loose_nom = _episode_fraction(loose_src, loose.n_episodes, 4)
    ref_td3 = _episode_fraction(ref_src, ref.n_episodes, 2)

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 8.8), sharex=True)
    axs[0].plot(x, loose.avg_rewards, color="#C84C09", linewidth=2.0, label="z = 0.40 looser acceptance")
    axs[0].plot(x, ref.avg_rewards, color="#0B6E4F", linewidth=1.8, linestyle="--", label="z = 0.08 tail-anchor reference")
    axs[0].plot(x, np.asarray(mpc_bundle["avg_rewards"], float), color="#4C4C4C", linewidth=1.6, linestyle=":", label="Nominal MPC")
    axs[0].set_ylabel("Avg. reward")
    axs[0].set_title("Polymer Markov: reward stays competitive while TD3 acceptance collapses")
    axs[0].legend(loc="best")

    axs[1].plot(x, loose_td3, color="#7A1FA2", linewidth=2.0, label="TD3 accepted, z = 0.40")
    axs[1].plot(x, loose_ls, color="#1f77b4", linewidth=1.8, label="LS fallback, z = 0.40")
    axs[1].plot(x, loose_nom, color="#D55E00", linewidth=1.8, label="Nominal fallback, z = 0.40")
    axs[1].plot(x, ref_td3, color="#0B6E4F", linewidth=1.5, linestyle="--", label="TD3 accepted, z = 0.08 ref")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Fraction of steps")
    axs[1].legend(loc="best", ncol=2, fontsize=9)

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if loose.warm_end_episode is not None:
            ax.axvline(loose.warm_end_episode, color="0.35", linestyle="--", linewidth=1.1)

    fig.tight_layout()
    out = out_dir / "fig_reward_and_action_source.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_gate_breakdown(loose: RunBundle, ref: RunBundle, loose_metrics: dict[str, float], ref_metrics: dict[str, float], out_dir: Path) -> Path:
    x = np.arange(1, loose.n_episodes + 1)
    loose_live = _requested_valid_mask(loose, _window_mask(loose))
    ref_live = _requested_valid_mask(ref, _window_mask(ref))
    loose_gates = _gate_arrays(loose)
    ref_gates = _gate_arrays(ref)

    loose_score_ep = _episode_masked_fraction(loose_gates["score_pass"], loose_live, loose.n_episodes)
    loose_drift_ep = _episode_masked_fraction(loose_gates["drift_pass"], loose_live, loose.n_episodes)
    loose_all_ep = _episode_masked_fraction(loose_gates["all_pass"], loose_live, loose.n_episodes)
    ref_all_ep = _episode_masked_fraction(ref_gates["all_pass"], ref_live, ref.n_episodes)

    labels = ["Score", "Drift", "Cost", "All pass"]
    loose_live_bar = _window_bar_values(loose_metrics, "requested_live")
    loose_tail_bar = _window_bar_values(loose_metrics, "requested_tail")
    ref_live_bar = _window_bar_values(ref_metrics, "requested_live")
    ref_tail_bar = _window_bar_values(ref_metrics, "requested_tail")

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 9.8), sharex=False)
    axs[0].plot(x, loose_score_ep, color="#1f77b4", linewidth=1.9, label="Score pass, z = 0.40")
    axs[0].plot(x, loose_drift_ep, color="#C84C09", linewidth=1.9, label="Drift pass, z = 0.40")
    axs[0].plot(x, loose_all_ep, color="#111111", linewidth=2.0, label="All pass, z = 0.40")
    axs[0].plot(x, ref_all_ep, color="#0B6E4F", linewidth=1.7, linestyle="--", label="All pass, z = 0.08 ref")
    axs[0].set_ylabel("Requested-candidate pass fraction")
    axs[0].set_title("The looser score and cost gates are not the binding constraint")
    axs[0].legend(loc="best")
    axs[0].grid(alpha=0.25)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    if loose.warm_end_episode is not None:
        axs[0].axvline(loose.warm_end_episode, color="0.35", linestyle="--", linewidth=1.1)

    bar_x = np.arange(len(labels))
    width = 0.18
    axs[1].bar(bar_x - 1.5 * width, loose_live_bar, width, color="#C84C09", label="z = 0.40 live")
    axs[1].bar(bar_x - 0.5 * width, loose_tail_bar, width, color="#E8A47A", label="z = 0.40 tail")
    axs[1].bar(bar_x + 0.5 * width, ref_live_bar, width, color="#0B6E4F", label="z = 0.08 live")
    axs[1].bar(bar_x + 1.5 * width, ref_tail_bar, width, color="#7FC8A9", label="z = 0.08 tail")
    axs[1].set_xticks(bar_x)
    axs[1].set_xticklabels(labels)
    axs[1].set_ylabel("Fraction of requested TD3 steps")
    axs[1].set_ylim(0.0, 1.05)
    axs[1].legend(loc="upper center", ncol=4, fontsize=9)
    axs[1].grid(alpha=0.25, axis="y")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    fig.tight_layout()
    out = out_dir / "fig_gate_breakdown.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_geometry_and_alignment(loose: RunBundle, ref: RunBundle, out_dir: Path) -> Path:
    x = np.arange(1, loose.n_episodes + 1)
    loose_live = _requested_valid_mask(loose, _window_mask(loose))
    ref_live = _requested_valid_mask(ref, _window_mask(ref))

    loose_sat = _episode_masked_raw_saturation(np.asarray(loose.bundle["rl_requested_raw_action_log"], float), loose_live, loose.n_episodes)
    ref_sat = _episode_masked_raw_saturation(np.asarray(ref.bundle["rl_requested_raw_action_log"], float), ref_live, ref.n_episodes)
    loose_bc = _episode_mean(np.asarray(loose.bundle["bc_active_log"], float), loose.n_episodes)
    ref_bc = _episode_mean(np.asarray(ref.bundle["bc_active_log"], float), ref.n_episodes)

    loose_req_z = _episode_masked_vec_norm(np.asarray(loose.bundle["rl_requested_z_log"], float), loose_live, loose.n_episodes)
    loose_ls_z = _episode_masked_vec_norm(np.asarray(loose.bundle["rl_ls_z_log"], float), loose_live, loose.n_episodes)
    ref_req_z = _episode_masked_vec_norm(np.asarray(ref.bundle["rl_requested_z_log"], float), ref_live, ref.n_episodes)
    ref_ls_z = _episode_masked_vec_norm(np.asarray(ref.bundle["rl_ls_z_log"], float), ref_live, ref.n_episodes)

    loose_req_drift = _episode_masked_mean(loose.stage_df["requested_gain_drift"].to_numpy(dtype=float), loose_live, loose.n_episodes)
    loose_ls_drift = _episode_masked_mean(loose.stage_df["ls_gain_drift"].to_numpy(dtype=float), loose_live, loose.n_episodes)
    ref_req_drift = _episode_masked_mean(ref.stage_df["requested_gain_drift"].to_numpy(dtype=float), ref_live, ref.n_episodes)
    ref_ls_drift = _episode_masked_mean(ref.stage_df["ls_gain_drift"].to_numpy(dtype=float), ref_live, ref.n_episodes)

    fig, axs = plt.subplots(3, 1, figsize=(12.8, 12.0), sharex=True)
    axs[0].plot(x, loose_sat, color="#C84C09", linewidth=1.9, label="Raw-action saturation, z = 0.40")
    axs[0].plot(x, ref_sat, color="#0B6E4F", linewidth=1.6, linestyle="--", label="Raw-action saturation, z = 0.08")
    axs[0].plot(x, loose_bc, color="#7A1FA2", linewidth=1.5, label="BC active, z = 0.40")
    axs[0].plot(x, ref_bc, color="#4C4C4C", linewidth=1.3, linestyle=":", label="BC active, z = 0.08")
    axs[0].set_ylabel("Fraction")
    axs[0].set_title("The wider run saturates the raw policy while teacher guidance becomes less available")
    axs[0].legend(loc="best", ncol=2, fontsize=9)

    axs[1].plot(x, loose_req_z, color="#C84C09", linewidth=1.9, label="||z_TD3||, z = 0.40")
    axs[1].plot(x, loose_ls_z, color="#D55E00", linewidth=1.5, linestyle="--", label="||z_LS||, z = 0.40")
    axs[1].plot(x, ref_req_z, color="#0B6E4F", linewidth=1.7, label="||z_TD3||, z = 0.08")
    axs[1].plot(x, ref_ls_z, color="#2E8B57", linewidth=1.4, linestyle="--", label="||z_LS||, z = 0.08")
    axs[1].set_ylabel("Mean vector norm")
    axs[1].legend(loc="best", ncol=2, fontsize=9)

    axs[2].plot(x, loose_req_drift, color="#C84C09", linewidth=1.9, label="Requested drift, z = 0.40")
    axs[2].plot(x, loose_ls_drift, color="#D55E00", linewidth=1.5, linestyle="--", label="LS drift, z = 0.40")
    axs[2].plot(x, ref_req_drift, color="#0B6E4F", linewidth=1.7, label="Requested drift, z = 0.08")
    axs[2].plot(x, ref_ls_drift, color="#2E8B57", linewidth=1.4, linestyle="--", label="LS drift, z = 0.08")
    axs[2].axhline(loose.gain_drift_max, color="0.2", linewidth=1.1, linestyle=":", label="Drift limit")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Gain drift")
    axs[2].legend(loc="best", ncol=2, fontsize=9)

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if loose.warm_end_episode is not None:
            ax.axvline(loose.warm_end_episode, color="0.35", linestyle="--", linewidth=1.1)

    fig.tight_layout()
    out = out_dir / "fig_geometry_and_alignment.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_final_episode(loose: RunBundle, ref: RunBundle, mpc_bundle: dict, out_dir: Path) -> Path:
    loose_metrics, loose_payload = _final_test_episode_metrics(loose, mpc_bundle)
    _ref_metrics, ref_payload = _final_test_episode_metrics(ref, mpc_bundle)
    delta_t = float(loose.bundle["delta_t"])
    meta = dict(loose.bundle.get("system_metadata", {}))
    output_labels = list(meta.get("output_labels", ["Output 1", "Output 2"]))
    input_labels = list(meta.get("input_labels", ["Input 1", "Input 2"]))
    time_label = str(meta.get("time_label", "Time"))

    loose_y = np.asarray(loose_payload["rl_y"], float)
    loose_u = np.asarray(loose_payload["rl_u"], float)
    ref_y = np.asarray(ref_payload["rl_y"], float)
    ref_u = np.asarray(ref_payload["rl_u"], float)
    mpc_y = np.asarray(loose_payload["mpc_y"], float)
    mpc_u = np.asarray(loose_payload["mpc_u"], float)
    y_sp = np.asarray(loose_payload["y_sp_phys"], float)

    t_line = np.linspace(0.0, loose_u.shape[0] * delta_t, loose_u.shape[0] + 1)
    t_step = t_line[:-1]

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex="col")
    for idx in range(2):
        ax = axs[idx, 0]
        ax.plot(t_line, loose_y[:, idx], color="#C84C09", linewidth=2.0, label="z = 0.40")
        ax.plot(t_line, ref_y[:, idx], color="#0B6E4F", linewidth=1.7, linestyle="--", label="z = 0.08 ref")
        ax.plot(t_line, mpc_y[:, idx], color="#4C4C4C", linewidth=1.5, linestyle=":", label="Nominal MPC")
        ax.step(t_step, y_sp[:, idx], where="post", color="0.2", linewidth=1.2, linestyle="-.", label="Setpoint")
        ax.set_ylabel(output_labels[idx])
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

        ax_u = axs[idx, 1]
        ax_u.step(t_step, loose_u[:, idx], where="post", color="#C84C09", linewidth=2.0, label="z = 0.40")
        ax_u.step(t_step, ref_u[:, idx], where="post", color="#0B6E4F", linewidth=1.7, linestyle="--", label="z = 0.08 ref")
        ax_u.step(t_step, mpc_u[:, idx], where="post", color="#4C4C4C", linewidth=1.5, linestyle=":", label="Nominal MPC")
        ax_u.set_ylabel(input_labels[idx])
        ax_u.grid(alpha=0.25)
        ax_u.spines["top"].set_visible(False)
        ax_u.spines["right"].set_visible(False)
        if idx == 0:
            ax_u.legend(loc="best", fontsize=9)

    axs[0, 0].set_title("Final test episode outputs")
    axs[0, 1].set_title("Final test episode inputs")
    axs[1, 0].set_xlabel(time_label)
    axs[1, 1].set_xlabel(time_label)
    fig.tight_layout()
    out = out_dir / "fig_final_episode_vs_reference_and_mpc.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    del loose_metrics
    return out


def _write_summary(loose_metrics: dict[str, float], ref_metrics: dict[str, float], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_metrics.csv"
    json_path = out_dir / "summary.json"

    rows = []
    for method, metrics in [("z040_looser_accept", loose_metrics), ("z008_tail_anchor_ref", ref_metrics)]:
        row = {"method": method}
        row.update(metrics)
        rows.append(row)

    fieldnames = ["method"] + sorted({key for row in rows for key in row.keys() if key != "method"})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "z040_looser_accept": loose_metrics,
        "z008_tail_anchor_ref": ref_metrics,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return csv_path, json_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loose = _load_run("z = 0.40 looser acceptance", LOOSE_RUN_DIR)
    ref = _load_run("z = 0.08 tail-anchor reference", REF_RUN_DIR)
    mpc_bundle = _load_pickle(MPC_BASELINE)

    loose_metrics = _collect_metrics(loose, mpc_bundle)
    ref_metrics = _collect_metrics(ref, mpc_bundle)

    generated = [
        _plot_reward_and_mix(loose, ref, mpc_bundle, OUT_DIR),
        _plot_gate_breakdown(loose, ref, loose_metrics, ref_metrics, OUT_DIR),
        _plot_geometry_and_alignment(loose, ref, OUT_DIR),
        _plot_final_episode(loose, ref, mpc_bundle, OUT_DIR),
    ]
    csv_path, json_path = _write_summary(loose_metrics, ref_metrics, OUT_DIR)
    generated.extend([csv_path, json_path])

    print("Generated:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
