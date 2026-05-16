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


NOTEBOOK_TD3_ONLY = REPO_ROOT / "RL_assisted_MPC_markov_zbound_008_looser_gate_only_td3_only_unified.ipynb"
NOTEBOOK_GUARDED = REPO_ROOT / "RL_assisted_MPC_markov_zbound_008_looser_gate_only_unified.ipynb"
TD3_ONLY_ROOT = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008_looser_gate_only_td3_only"
GUARDED_ROOT = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008_looser_gate_only"
MPC_BASELINE = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_td3_only_without_safeguard_20260516"
TAIL_EPISODES = 20

ACTION_SOURCE_TD3 = 2
ACTION_SOURCE_LS = 3
ACTION_SOURCE_NOMINAL = 4


@dataclass
class RunBundle:
    label: str
    run_dir: Path
    bundle: dict
    stage_df: pd.DataFrame

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle.get("avg_rewards", []), dtype=float)

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
        return float(value) if value is not None else np.nan


def _latest_timestamp_dir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}.")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_run(label: str, run_root: Path) -> RunBundle:
    run_dir = _latest_timestamp_dir(run_root)
    return RunBundle(
        label=label,
        run_dir=run_dir,
        bundle=_load_pickle(run_dir / "input_data.pkl"),
        stage_df=pd.read_csv(run_dir / "markov_stage_diagnostics.csv"),
    )


def _episode_mean(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], dtype=float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes]
    if arr.ndim == 1:
        arr = arr.reshape(n_episodes, steps)
        return np.nanmean(arr, axis=1)
    arr = arr.reshape(n_episodes, steps, -1)
    return np.nanmean(arr.reshape(n_episodes, -1), axis=1)


def _episode_fraction(values: np.ndarray, n_episodes: int, target: int) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], dtype=float)
    steps = arr.size // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps)
    return np.mean(arr == int(target), axis=1)


def _episode_masked_mean(values: np.ndarray, valid: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], dtype=float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes]
    valid = valid[: steps * n_episodes]
    if arr.ndim == 1:
        arr = arr.reshape(n_episodes, steps)
        valid = valid.reshape(n_episodes, steps)
        out = np.empty(n_episodes, dtype=float)
        for idx in range(n_episodes):
            row = arr[idx, valid[idx]]
            row = row[np.isfinite(row)]
            out[idx] = float(np.mean(row)) if row.size else np.nan
        return out
    arr = arr.reshape(n_episodes, steps, -1)
    valid = valid.reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx in range(n_episodes):
        row = arr[idx, valid[idx], :]
        row = row[np.isfinite(row)]
        out[idx] = float(np.mean(row)) if row.size else np.nan
    return out


def _episode_masked_vec_norm(values: np.ndarray, valid: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], dtype=float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps, -1)
    valid = valid[: steps * n_episodes].reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx in range(n_episodes):
        block = arr[idx, valid[idx], :]
        out[idx] = float(np.mean(np.linalg.norm(block, axis=1))) if block.size else np.nan
    return out


def _episode_masked_raw_saturation(
    values: np.ndarray,
    valid: np.ndarray,
    n_episodes: int,
    threshold: float = 0.95,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], dtype=float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps, -1)
    valid = valid[: steps * n_episodes].reshape(n_episodes, steps)
    out = np.empty(n_episodes, dtype=float)
    for idx in range(n_episodes):
        block = arr[idx, valid[idx], :]
        out[idx] = float(np.mean(np.max(np.abs(block), axis=1) >= threshold)) if block.size else np.nan
    return out


def _window_mask(run: RunBundle, tail_episodes: int | None = None) -> np.ndarray:
    steps = run.steps_per_episode
    live = run.stage_df["step"].to_numpy(dtype=int) > run.warm_start_step
    if tail_episodes is None:
        return live
    start_step = max(0, run.n_episodes - int(tail_episodes)) * steps
    tail = run.stage_df["step"].to_numpy(dtype=int) >= start_step
    return live & tail


def _requested_valid_mask(run: RunBundle, mask: np.ndarray) -> np.ndarray:
    requested_score = run.stage_df["requested_prediction_score"].to_numpy(dtype=float)
    return np.asarray(mask, dtype=bool) & np.isfinite(requested_score)


def _mean_over_mask(values: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        use = arr[mask]
        use = use[np.isfinite(use)]
        return float(np.mean(use)) if use.size else np.nan
    use = arr[mask, :]
    return float(np.mean(np.linalg.norm(use, axis=1))) if use.size else np.nan


def _final_test_episode_metrics(run: RunBundle, mpc_bundle: dict) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    time_in_sub_episodes = int(run.bundle["time_in_sub_episodes"])
    rl_y = np.asarray(run.bundle["y_line_full"], dtype=float)[-(time_in_sub_episodes + 1) :, :]
    rl_u = np.asarray(run.bundle["u_step_full"], dtype=float)[-time_in_sub_episodes:, :]
    mpc_y = np.asarray(mpc_bundle["y_mpc"], dtype=float)[-(time_in_sub_episodes + 1) :, :]
    mpc_u = np.asarray(mpc_bundle["u_mpc"], dtype=float)[-time_in_sub_episodes:, :]

    data_min = np.asarray(run.bundle["data_min"], dtype=float)
    data_max = np.asarray(run.bundle["data_max"], dtype=float)
    n_inputs = int(rl_u.shape[1])
    y_ss = np.asarray(run.bundle["steady_states"]["y_ss"], dtype=float)
    y_ss_scaled = apply_min_max(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled_dev = np.asarray(run.bundle["y_sp"], dtype=float)[-time_in_sub_episodes:, :]
    y_sp_phys = reverse_min_max(y_sp_scaled_dev + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])

    rl_err = rl_y[1:, :] - y_sp_phys
    mpc_err = mpc_y[1:, :] - y_sp_phys
    rl_du = np.diff(np.vstack([rl_u[0:1, :], rl_u]), axis=0)
    mpc_du = np.diff(np.vstack([mpc_u[0:1, :], mpc_u]), axis=0)

    metrics = {
        "final_reward_rl": float(run.avg_rewards[-1]),
        "final_reward_mpc": float(np.asarray(mpc_bundle["avg_rewards"], dtype=float)[-1]),
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
    tail_mask = _window_mask(run, tail_episodes=TAIL_EPISODES)
    req_live = _requested_valid_mask(run, live_mask)
    req_tail = _requested_valid_mask(run, tail_mask)

    src = np.asarray(run.bundle["rl_action_source_log"], dtype=int)
    td3_ep = _episode_fraction(src, run.n_episodes, ACTION_SOURCE_TD3)
    ls_ep = _episode_fraction(src, run.n_episodes, ACTION_SOURCE_LS)
    nominal_ep = _episode_fraction(src, run.n_episodes, ACTION_SOURCE_NOMINAL)

    requested_score = run.stage_df["requested_prediction_score"].to_numpy(dtype=float)
    requested_drift = run.stage_df["requested_gain_drift"].to_numpy(dtype=float)
    requested_z = np.asarray(run.bundle["rl_requested_z_log"], dtype=float)
    requested_raw = np.asarray(run.bundle["rl_requested_raw_action_log"], dtype=float)
    executed_score = run.stage_df["executed_prediction_score"].to_numpy(dtype=float)
    executed_drift = run.stage_df["executed_gain_drift"].to_numpy(dtype=float)
    executed_z = run.stage_df["executed_z_norm"].to_numpy(dtype=float)
    summary_metrics = dict(run.bundle.get("summary_metrics", {}))

    final_metrics, _payload = _final_test_episode_metrics(run, mpc_bundle)

    metrics = {
        "reward_mean": float(np.mean(run.avg_rewards)),
        "reward_tail20": float(np.mean(run.avg_rewards[-TAIL_EPISODES:])),
        "reward_final": float(run.avg_rewards[-1]),
        "td3_fraction_live": float(np.mean(src[live_mask] == ACTION_SOURCE_TD3)),
        "td3_fraction_tail20": float(np.mean(td3_ep[-TAIL_EPISODES:])),
        "ls_fraction_tail20": float(np.mean(ls_ep[-TAIL_EPISODES:])),
        "nominal_fraction_tail20": float(np.mean(nominal_ep[-TAIL_EPISODES:])),
        "prediction_score_mean_live": _mean_over_mask(requested_score, req_live),
        "prediction_score_mean_tail20": _mean_over_mask(requested_score, req_tail),
        "gain_drift_mean_live": _mean_over_mask(requested_drift, req_live),
        "gain_drift_mean_tail20": _mean_over_mask(requested_drift, req_tail),
        "z_norm_mean_live": _mean_over_mask(requested_z, req_live),
        "z_norm_mean_tail20": _mean_over_mask(requested_z, req_tail),
        "executed_prediction_score_mean_live": _mean_over_mask(executed_score, live_mask),
        "executed_prediction_score_mean_tail20": _mean_over_mask(executed_score, tail_mask),
        "executed_gain_drift_mean_live": _mean_over_mask(executed_drift, live_mask),
        "executed_gain_drift_mean_tail20": _mean_over_mask(executed_drift, tail_mask),
        "executed_z_norm_mean_live": _mean_over_mask(executed_z, live_mask),
        "executed_z_norm_mean_tail20": _mean_over_mask(executed_z, tail_mask),
        "raw_saturation_fraction_live": float(np.mean(np.max(np.abs(requested_raw[req_live]), axis=1) >= 0.95)),
        "raw_saturation_fraction_tail20": float(np.mean(np.max(np.abs(requested_raw[req_tail]), axis=1) >= 0.95)),
        "force_td3_execute": bool(run.bundle.get("force_td3_execute", False)),
        "z_bound": run.z_bound,
        "s_pred_min": run.s_pred_min,
        "gain_drift_max": run.gain_drift_max,
        "nominal_cost_relative_tol": run.nominal_cost_relative_tol,
        "accepted_fraction": float(summary_metrics.get("accepted_fraction", np.nan)),
        "td3_accepted_fraction": float(summary_metrics.get("td3_accepted_fraction", np.nan)),
        "ls_fallback_fraction": float(summary_metrics.get("ls_fallback_fraction", np.nan)),
        "nominal_fallback_fraction": float(summary_metrics.get("nominal_fallback_fraction", np.nan)),
    }
    metrics.update(final_metrics)
    return metrics


def _plot_reward_and_mix(td3_only: RunBundle, guarded: RunBundle, mpc_bundle: dict, out_dir: Path) -> Path:
    x = np.arange(1, td3_only.n_episodes + 1)
    td3_src = np.asarray(td3_only.bundle["rl_action_source_log"], dtype=int)
    guarded_src = np.asarray(guarded.bundle["rl_action_source_log"], dtype=int)

    td3_only_frac = _episode_fraction(td3_src, td3_only.n_episodes, ACTION_SOURCE_TD3)
    guarded_td3 = _episode_fraction(guarded_src, guarded.n_episodes, ACTION_SOURCE_TD3)
    guarded_ls = _episode_fraction(guarded_src, guarded.n_episodes, ACTION_SOURCE_LS)
    guarded_nominal = _episode_fraction(guarded_src, guarded.n_episodes, ACTION_SOURCE_NOMINAL)

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 8.8), sharex=True)
    axs[0].plot(x, td3_only.avg_rewards, color="#7A1FA2", linewidth=2.0, label="TD3-only, no safeguard")
    axs[0].plot(x, guarded.avg_rewards, color="#0B6E4F", linewidth=1.8, linestyle="--", label="Guarded TD3 + LS/nominal fallback")
    axs[0].plot(x, np.asarray(mpc_bundle["avg_rewards"], dtype=float), color="#4C4C4C", linewidth=1.6, linestyle=":", label="Nominal MPC")
    axs[0].set_ylabel("Avg. reward")
    axs[0].set_title("Latest polymer Markov TD3-only run improves reward while removing all fallback paths")
    axs[0].legend(loc="best")

    axs[1].plot(x, td3_only_frac, color="#7A1FA2", linewidth=2.0, label="TD3 executed, no safeguard")
    axs[1].plot(x, guarded_td3, color="#0B6E4F", linewidth=1.8, label="TD3 executed, guarded")
    axs[1].plot(x, guarded_ls, color="#1f77b4", linewidth=1.6, linestyle="--", label="LS fallback, guarded")
    axs[1].plot(x, guarded_nominal, color="#D55E00", linewidth=1.6, linestyle=":", label="Nominal fallback, guarded")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Fraction of steps")
    axs[1].legend(loc="best", ncol=2, fontsize=9)

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if td3_only.warm_end_episode is not None:
            ax.axvline(td3_only.warm_end_episode, color="0.35", linestyle="--", linewidth=1.1)

    fig.tight_layout()
    out = out_dir / "fig_reward_and_action_source.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_mechanism_diagnostics(td3_only: RunBundle, guarded: RunBundle, out_dir: Path) -> Path:
    x = np.arange(1, td3_only.n_episodes + 1)
    td3_live = _window_mask(td3_only)
    guarded_live = _window_mask(guarded)
    td3_req_live = _requested_valid_mask(td3_only, td3_live)
    guarded_req_live = _requested_valid_mask(guarded, guarded_live)

    td3_score = _episode_masked_mean(td3_only.stage_df["executed_prediction_score"].to_numpy(dtype=float), td3_live, td3_only.n_episodes)
    guarded_score = _episode_masked_mean(guarded.stage_df["executed_prediction_score"].to_numpy(dtype=float), guarded_live, guarded.n_episodes)
    td3_drift = _episode_masked_mean(td3_only.stage_df["executed_gain_drift"].to_numpy(dtype=float), td3_live, td3_only.n_episodes)
    guarded_drift = _episode_masked_mean(guarded.stage_df["executed_gain_drift"].to_numpy(dtype=float), guarded_live, guarded.n_episodes)
    td3_z = _episode_masked_mean(td3_only.stage_df["executed_z_norm"].to_numpy(dtype=float), td3_live, td3_only.n_episodes)
    guarded_z = _episode_masked_mean(guarded.stage_df["executed_z_norm"].to_numpy(dtype=float), guarded_live, guarded.n_episodes)
    td3_sat = _episode_masked_raw_saturation(np.asarray(td3_only.bundle["rl_requested_raw_action_log"], dtype=float), td3_req_live, td3_only.n_episodes)
    guarded_sat = _episode_masked_raw_saturation(np.asarray(guarded.bundle["rl_requested_raw_action_log"], dtype=float), guarded_req_live, guarded.n_episodes)

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.8), sharex=True)
    axs = axs.ravel()

    axs[0].plot(x, td3_score, color="#7A1FA2", linewidth=1.9, label="TD3-only")
    axs[0].plot(x, guarded_score, color="#0B6E4F", linewidth=1.7, linestyle="--", label="Guarded")
    axs[0].axhline(0.0, color="0.2", linewidth=1.0, linestyle=":")
    axs[0].set_ylabel("Prediction score")
    axs[0].set_title("Executed prediction-improvement score")
    axs[0].legend(loc="best")

    axs[1].plot(x, td3_drift, color="#7A1FA2", linewidth=1.9, label="TD3-only")
    axs[1].plot(x, guarded_drift, color="#0B6E4F", linewidth=1.7, linestyle="--", label="Guarded")
    axs[1].axhline(td3_only.gain_drift_max, color="0.2", linewidth=1.0, linestyle=":", label="Drift limit")
    axs[1].set_ylabel("Gain drift")
    axs[1].set_title("Executed gain drift")
    axs[1].legend(loc="best")

    axs[2].plot(x, td3_z, color="#7A1FA2", linewidth=1.9, label="TD3-only")
    axs[2].plot(x, guarded_z, color="#0B6E4F", linewidth=1.7, linestyle="--", label="Guarded")
    axs[2].set_ylabel("Mean ||z||")
    axs[2].set_xlabel("Episode")
    axs[2].set_title("Executed Markov correction magnitude")

    axs[3].plot(x, td3_sat, color="#7A1FA2", linewidth=1.9, label="TD3-only")
    axs[3].plot(x, guarded_sat, color="#0B6E4F", linewidth=1.7, linestyle="--", label="Guarded")
    axs[3].set_ylabel("Saturation fraction")
    axs[3].set_xlabel("Episode")
    axs[3].set_title("Requested raw-action saturation")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if td3_only.warm_end_episode is not None:
            ax.axvline(td3_only.warm_end_episode, color="0.35", linestyle="--", linewidth=1.1)

    fig.suptitle("Executed diagnostics: the safeguarded controller trades authority for better alignment", y=1.01)
    fig.tight_layout()
    out = out_dir / "fig_mechanism_diagnostics.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_tail_summary(td3_metrics: dict[str, float], guarded_metrics: dict[str, float], out_dir: Path) -> Path:
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.2))
    axs = axs.ravel()

    reward_labels = ["Mean", "Tail-20", "Final"]
    td3_reward = [td3_metrics["reward_mean"], td3_metrics["reward_tail20"], td3_metrics["reward_final"]]
    guarded_reward = [guarded_metrics["reward_mean"], guarded_metrics["reward_tail20"], guarded_metrics["reward_final"]]

    mix_labels = ["TD3", "LS", "Nominal"]
    td3_mix = [td3_metrics["td3_fraction_tail20"], td3_metrics["ls_fraction_tail20"], td3_metrics["nominal_fraction_tail20"]]
    guarded_mix = [guarded_metrics["td3_fraction_tail20"], guarded_metrics["ls_fraction_tail20"], guarded_metrics["nominal_fraction_tail20"]]

    diag_labels = ["Pred. score", "Drift", "||z||", "Raw sat."]
    td3_diag = [
        td3_metrics["prediction_score_mean_tail20"],
        td3_metrics["gain_drift_mean_tail20"],
        td3_metrics["z_norm_mean_tail20"],
        td3_metrics["raw_saturation_fraction_tail20"],
    ]
    guarded_diag = [
        guarded_metrics["prediction_score_mean_tail20"],
        guarded_metrics["gain_drift_mean_tail20"],
        guarded_metrics["z_norm_mean_tail20"],
        guarded_metrics["raw_saturation_fraction_tail20"],
    ]

    final_labels = ["Eta RMSE", "Temp RMSE", "Mean ||du||"]
    td3_final = [td3_metrics["eta_rmse_rl"], td3_metrics["temp_rmse_rl"], td3_metrics["input_move_norm_rl"]]
    guarded_final = [guarded_metrics["eta_rmse_rl"], guarded_metrics["temp_rmse_rl"], guarded_metrics["input_move_norm_rl"]]

    for ax, labels, td3_vals, guarded_vals, title in (
        (axs[0], reward_labels, td3_reward, guarded_reward, "Reward metrics"),
        (axs[1], mix_labels, td3_mix, guarded_mix, "Tail-20 action-source mix"),
        (axs[2], diag_labels, td3_diag, guarded_diag, "Tail-20 mechanism diagnostics"),
        (axs[3], final_labels, td3_final, guarded_final, "Final episode tracking and move size"),
    ):
        x = np.arange(len(labels))
        width = 0.34
        ax.bar(x - width / 2.0, td3_vals, width, color="#7A1FA2", label="TD3-only")
        ax.bar(x + width / 2.0, guarded_vals, width, color="#0B6E4F", label="Guarded")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axs[0].legend(loc="best")
    fig.suptitle("TD3-only outperforms the guarded variant on reward, but with weaker consistency diagnostics", y=1.01)
    fig.tight_layout()
    out = out_dir / "fig_tail_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_final_episode(td3_only: RunBundle, guarded: RunBundle, mpc_bundle: dict, out_dir: Path) -> Path:
    _td3_metrics, td3_payload = _final_test_episode_metrics(td3_only, mpc_bundle)
    _guarded_metrics, guarded_payload = _final_test_episode_metrics(guarded, mpc_bundle)
    delta_t = float(td3_only.bundle["delta_t"])
    meta = dict(td3_only.bundle.get("system_metadata", {}))
    output_labels = list(meta.get("output_labels", ["Output 1", "Output 2"]))
    input_labels = list(meta.get("input_labels", ["Input 1", "Input 2"]))
    time_label = str(meta.get("time_label", "Time"))

    td3_y = np.asarray(td3_payload["rl_y"], dtype=float)
    td3_u = np.asarray(td3_payload["rl_u"], dtype=float)
    guarded_y = np.asarray(guarded_payload["rl_y"], dtype=float)
    guarded_u = np.asarray(guarded_payload["rl_u"], dtype=float)
    mpc_y = np.asarray(td3_payload["mpc_y"], dtype=float)
    mpc_u = np.asarray(td3_payload["mpc_u"], dtype=float)
    y_sp = np.asarray(td3_payload["y_sp_phys"], dtype=float)

    t_line = np.linspace(0.0, td3_u.shape[0] * delta_t, td3_u.shape[0] + 1)
    t_step = t_line[:-1]

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex="col")
    for idx in range(2):
        ax_y = axs[idx, 0]
        ax_y.plot(t_line, td3_y[:, idx], color="#7A1FA2", linewidth=2.0, label="TD3-only")
        ax_y.plot(t_line, guarded_y[:, idx], color="#0B6E4F", linewidth=1.7, linestyle="--", label="Guarded")
        ax_y.plot(t_line, mpc_y[:, idx], color="#4C4C4C", linewidth=1.5, linestyle=":", label="Nominal MPC")
        ax_y.step(t_step, y_sp[:, idx], where="post", color="0.2", linewidth=1.2, linestyle="-.", label="Setpoint")
        ax_y.set_ylabel(output_labels[idx])
        ax_y.grid(alpha=0.25)
        ax_y.spines["top"].set_visible(False)
        ax_y.spines["right"].set_visible(False)
        if idx == 0:
            ax_y.legend(loc="best", fontsize=9)

        ax_u = axs[idx, 1]
        ax_u.step(t_step, td3_u[:, idx], where="post", color="#7A1FA2", linewidth=2.0, label="TD3-only")
        ax_u.step(t_step, guarded_u[:, idx], where="post", color="#0B6E4F", linewidth=1.7, linestyle="--", label="Guarded")
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
    out = out_dir / "fig_final_episode_vs_guarded_and_mpc.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_summary(td3_only: RunBundle, guarded: RunBundle, td3_metrics: dict[str, float], guarded_metrics: dict[str, float], out_dir: Path) -> list[Path]:
    rows = []
    for label, run, metrics in (
        ("td3_only_no_safeguard", td3_only, td3_metrics),
        ("guarded_td3_with_fallback", guarded, guarded_metrics),
    ):
        row = {"method": label, "run_dir": str(run.run_dir.relative_to(REPO_ROOT))}
        row.update(metrics)
        rows.append(row)

    csv_path = out_dir / "summary_metrics.csv"
    fieldnames = ["method", "run_dir"] + sorted({key for row in rows for key in row.keys() if key not in {"method", "run_dir"}})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "notebooks": {
                    "td3_only": str(NOTEBOOK_TD3_ONLY.relative_to(REPO_ROOT)),
                    "guarded": str(NOTEBOOK_GUARDED.relative_to(REPO_ROOT)),
                },
                "runs": {
                    "td3_only": str(td3_only.run_dir.relative_to(REPO_ROOT)),
                    "guarded": str(guarded.run_dir.relative_to(REPO_ROOT)),
                    "mpc_baseline": str(MPC_BASELINE.relative_to(REPO_ROOT)),
                },
                "td3_only_no_safeguard": td3_metrics,
                "guarded_td3_with_fallback": guarded_metrics,
            },
            handle,
            indent=2,
        )

    return [csv_path, json_path]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    td3_only = _load_run("TD3-only without safeguard", TD3_ONLY_ROOT)
    guarded = _load_run("Guarded TD3 with fallback", GUARDED_ROOT)
    mpc_bundle = _load_pickle(MPC_BASELINE)

    td3_metrics = _collect_metrics(td3_only, mpc_bundle)
    guarded_metrics = _collect_metrics(guarded, mpc_bundle)

    generated = [
        _plot_reward_and_mix(td3_only, guarded, mpc_bundle, OUT_DIR),
        _plot_mechanism_diagnostics(td3_only, guarded, OUT_DIR),
        _plot_tail_summary(td3_metrics, guarded_metrics, OUT_DIR),
        _plot_final_episode(td3_only, guarded, mpc_bundle, OUT_DIR),
    ]
    generated.extend(_write_summary(td3_only, guarded, td3_metrics, guarded_metrics, OUT_DIR))

    print("Generated:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
