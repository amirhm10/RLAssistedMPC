from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LATEST_CONDITIONED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260511_171749"
PREVIOUS_EXECUTED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260511_140736"
LEGACY_RUN = REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260510_212956"
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_conditioned_followup_20260511"


@dataclass
class RunBundle:
    label: str
    run_dir: Path
    bundle: dict

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle.get("avg_rewards", []), float)

    @property
    def action_source(self) -> np.ndarray:
        return np.asarray(self.bundle.get("rl_action_source_log", []), int)

    @property
    def n_episodes(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def step_per_episode(self) -> int:
        if self.n_episodes <= 0:
            return 0
        return int(self.action_source.size // self.n_episodes)


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_run(label: str, run_dir: Path) -> RunBundle:
    return RunBundle(label=label, run_dir=run_dir, bundle=_load_pickle(run_dir / "input_data.pkl"))


def _episode_mean(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    arr = arr[: step_per_episode * n_episodes].reshape(n_episodes, step_per_episode)
    out = np.empty(n_episodes, dtype=float)
    for idx, row in enumerate(arr):
        finite = row[np.isfinite(row)]
        out[idx] = float(np.mean(finite)) if finite.size else np.nan
    return out


def _episode_fraction(values: np.ndarray, n_episodes: int, target: int) -> np.ndarray:
    arr = np.asarray(values, int)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    arr = arr[: step_per_episode * n_episodes].reshape(n_episodes, step_per_episode)
    return np.mean(arr == int(target), axis=1)


def _episode_vec_norm(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    arr = arr[: step_per_episode * n_episodes].reshape(n_episodes, step_per_episode, -1)
    return np.nanmean(np.linalg.norm(arr, axis=2), axis=1)


def _episode_fraction_raw_saturation(values: np.ndarray, n_episodes: int, threshold: float = 0.95) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    arr = arr[: step_per_episode * n_episodes].reshape(n_episodes, step_per_episode, -1)
    return np.mean(np.max(np.abs(arr), axis=2) >= float(threshold), axis=1)


def _episode_mean_max_abs(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    arr = arr[: step_per_episode * n_episodes].reshape(n_episodes, step_per_episode, -1)
    return np.nanmean(np.max(np.abs(arr), axis=2), axis=1)


def _warm_windows(run: RunBundle) -> dict[str, int | None]:
    step_per_episode = run.step_per_episode
    warm_start_step = run.bundle.get("warm_start_step")
    warm_end = None if warm_start_step is None or step_per_episode <= 0 else int(warm_start_step // step_per_episode)
    bc = dict(run.bundle.get("behavioral_cloning", {}))
    bc_start = bc_end = None
    if step_per_episode > 0 and bc.get("start_step") is not None and bc.get("end_step") is not None:
        bc_start = int(bc["start_step"] // step_per_episode) + 1
        bc_end = int(bc["end_step"] // step_per_episode) + 1
    return {"warm_end": warm_end, "bc_start": bc_start, "bc_end": bc_end}


def _plot_conditioned_reward_and_mix(latest: RunBundle, previous: RunBundle, legacy: RunBundle, out_dir: Path) -> Path:
    x = np.arange(1, latest.n_episodes + 1)
    latest_td3 = _episode_fraction(latest.bundle["rl_action_source_log"], latest.n_episodes, 2)
    prev_td3 = _episode_fraction(previous.bundle["rl_action_source_log"], previous.n_episodes, 2)
    windows = _warm_windows(latest)

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 8.8), sharex=True)
    axs[0].plot(x, latest.avg_rewards, color="#C84C09", linewidth=2.0, label="Latest conditioned-state run")
    axs[0].plot(x, previous.avg_rewards, color="#1f77b4", linewidth=1.8, linestyle="--", label="Previous executed replay run")
    axs[0].plot(np.arange(1, legacy.n_episodes + 1), legacy.avg_rewards, color="#0B6E4F", linewidth=1.6, linestyle="-.", label="Legacy reference")
    axs[0].set_ylabel("Avg. reward")
    axs[0].set_title("Polymer Markov: conditioned-state follow-up")
    axs[0].legend(loc="best")

    axs[1].plot(x, latest_td3, color="#C84C09", linewidth=2.0, label="Latest conditioned-state TD3 fraction")
    axs[1].plot(x, prev_td3, color="#1f77b4", linewidth=1.8, linestyle="--", label="Previous executed replay TD3 fraction")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("TD3 accepted fraction")
    axs[1].legend(loc="best")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if windows["warm_end"] is not None:
            ax.axvline(windows["warm_end"], color="0.35", linestyle="--", linewidth=1.1)
        if windows["bc_start"] is not None and windows["bc_end"] is not None:
            ax.axvspan(windows["bc_start"], windows["bc_end"], color="#F6C85F", alpha=0.18)

    fig.tight_layout()
    out = out_dir / "conditioned_run_reward_and_td3_mix.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_conditioned_diagnostics(latest: RunBundle, previous: RunBundle, out_dir: Path) -> Path:
    n_ep = latest.n_episodes
    x = np.arange(1, n_ep + 1)
    windows = _warm_windows(latest)

    req_score = _episode_mean(latest.bundle["requested_prediction_score_log"], n_ep)
    ls_score = _episode_mean(latest.bundle["ls_prediction_score_log"], n_ep)
    td3_frac = _episode_fraction(latest.bundle["rl_action_source_log"], n_ep, 2)
    sat_latest = _episode_fraction_raw_saturation(latest.bundle["rl_requested_raw_action_log"], n_ep)
    sat_prev = _episode_fraction_raw_saturation(previous.bundle["rl_requested_raw_action_log"], previous.n_episodes)
    req_zmax = _episode_mean_max_abs(latest.bundle["rl_requested_z_log"], n_ep)
    ls_zmax = _episode_mean_max_abs(latest.bundle["rl_ls_z_log"], n_ep)
    z_gap = _episode_vec_norm(
        np.asarray(latest.bundle["rl_requested_z_log"], float) - np.asarray(latest.bundle["rl_ls_z_log"], float),
        n_ep,
    )
    prev_std = np.nanstd(np.asarray(previous.bundle["rl_state_log"], float), axis=0)
    latest_std = np.nanstd(np.asarray(latest.bundle["rl_state_log"], float), axis=0)
    dims_prev = np.arange(1, prev_std.size + 1)
    dims_latest = np.arange(1, latest_std.size + 1)

    fig, axs = plt.subplots(3, 1, figsize=(12.8, 12.4), sharex=False)
    axs[0].plot(x, req_score, color="#1f77b4", linewidth=1.9, label="Requested TD3 score")
    axs[0].plot(x, ls_score, color="#0B6E4F", linewidth=1.9, label="LS score")
    ax0r = axs[0].twinx()
    ax0r.plot(x, td3_frac, color="#7A1FA2", linewidth=1.4, linestyle="--", label="TD3 accepted fraction")
    axs[0].axhline(0.0, color="0.45", linewidth=1.0, linestyle=":")
    axs[0].set_ylabel("Prediction score")
    ax0r.set_ylabel("TD3 frac")
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0r.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    axs[1].plot(x, sat_latest, color="#C84C09", linewidth=2.0, label="Raw-action saturation, latest run")
    axs[1].plot(x, sat_prev, color="#7A7A7A", linewidth=1.8, linestyle="--", label="Raw-action saturation, previous run")
    axs[1].plot(x, req_zmax, color="#1f77b4", linewidth=1.5, label="Mean max |z_TD3|")
    axs[1].plot(x, ls_zmax, color="#0B6E4F", linewidth=1.5, linestyle="--", label="Mean max |z_LS|")
    axs[1].plot(x, z_gap, color="#D55E00", linewidth=1.4, linestyle="-.", label="||z_TD3 - z_LS||")
    axs[1].axhline(0.05, color="0.35", linewidth=1.0, linestyle=":", label="z_bound")
    axs[1].set_ylabel("Saturation / norm")
    axs[1].legend(loc="best", ncol=2, fontsize=9)

    axs[2].plot(dims_prev, prev_std, color="#7A7A7A", linewidth=1.8, marker="o", markersize=3.0, label="Previous raw-state run")
    axs[2].plot(dims_latest, latest_std, color="#C84C09", linewidth=1.8, marker="s", markersize=3.0, label="Latest conditioned-state run")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("RL state dimension")
    axs[2].set_ylabel("Std. dev. (log scale)")
    axs[2].legend(loc="best")

    for ax in axs[:2]:
        if windows["warm_end"] is not None:
            ax.axvline(windows["warm_end"], color="0.35", linestyle="--", linewidth=1.1)
        if windows["bc_start"] is not None and windows["bc_end"] is not None:
            ax.axvspan(windows["bc_start"], windows["bc_end"], color="#F6C85F", alpha=0.18)
    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax0r.spines["top"].set_visible(False)

    fig.tight_layout()
    out = out_dir / "conditioned_run_geometry_and_scores.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _run_metrics(run: RunBundle) -> dict:
    n_ep = run.n_episodes
    td3 = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 2)
    ls = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 3)
    nom = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 4)
    req = _episode_mean(run.bundle["requested_prediction_score_log"], n_ep)
    lss = _episode_mean(run.bundle["ls_prediction_score_log"], n_ep)
    exe = _episode_mean(run.bundle["executed_prediction_score_log"], n_ep)
    sat = _episode_fraction_raw_saturation(run.bundle["rl_requested_raw_action_log"], n_ep)
    req_zmax = _episode_mean_max_abs(run.bundle["rl_requested_z_log"], n_ep)
    ls_zmax = _episode_mean_max_abs(run.bundle["rl_ls_z_log"], n_ep)
    z_gap = _episode_vec_norm(
        np.asarray(run.bundle["rl_requested_z_log"], float) - np.asarray(run.bundle["rl_ls_z_log"], float),
        n_ep,
    )
    req_drift = _episode_mean(run.bundle["requested_gain_drift_log"], n_ep)
    req_cost_pass = _episode_mean(run.bundle["requested_cost_guard_pass_log"], n_ep)
    state_std = np.nanstd(np.asarray(run.bundle["rl_state_log"], float), axis=0)
    p10 = float(np.percentile(state_std, 10))
    p90 = float(np.percentile(state_std, 90))
    return {
        "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
        "markov_state_mode": run.bundle.get("markov_state_mode"),
        "state_dim": int(np.asarray(run.bundle["rl_state_log"]).shape[1]),
        "reward_mean": float(np.nanmean(run.avg_rewards)),
        "reward_last10": float(np.nanmean(run.avg_rewards[-10:])),
        "td3_fraction_11_50": float(np.nanmean(td3[10:50])),
        "td3_fraction_last50": float(np.nanmean(td3[-50:])),
        "ls_fraction_last50": float(np.nanmean(ls[-50:])),
        "nominal_fraction_last50": float(np.nanmean(nom[-50:])),
        "requested_score_11_50": float(np.nanmean(req[10:50])),
        "requested_score_last50": float(np.nanmean(req[-50:])),
        "ls_score_last50": float(np.nanmean(lss[-50:])),
        "executed_score_last50": float(np.nanmean(exe[-50:])),
        "requested_drift_last50": float(np.nanmean(req_drift[-50:])),
        "requested_cost_pass_last50": float(np.nanmean(req_cost_pass[-50:])),
        "raw_saturation_all": float(np.nanmean(sat)),
        "raw_saturation_last50": float(np.nanmean(sat[-50:])),
        "requested_zmax_last50": float(np.nanmean(req_zmax[-50:])),
        "ls_zmax_last50": float(np.nanmean(ls_zmax[-50:])),
        "z_gap_last50": float(np.nanmean(z_gap[-50:])),
        "state_std_p90_over_p10": float(p90 / max(p10, 1e-12)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = _load_run("Latest conditioned-state run", LATEST_CONDITIONED_RUN)
    previous = _load_run("Previous executed replay run", PREVIOUS_EXECUTED_RUN)
    legacy = _load_run("Legacy reference", LEGACY_RUN)

    fig1 = _plot_conditioned_reward_and_mix(latest, previous, legacy, OUT_DIR)
    fig2 = _plot_conditioned_diagnostics(latest, previous, OUT_DIR)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_conditioned_run": _run_metrics(latest),
        "previous_executed_run": _run_metrics(previous),
        "figure_paths": [
            str(fig1.relative_to(REPO_ROOT)),
            str(fig2.relative_to(REPO_ROOT)),
        ],
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
