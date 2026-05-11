from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LATEST_UNIFIED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260511_120210"
REGRESSION_UNIFIED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260510_215724"
LEGACY_RUN = REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260510_212956"
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_unified_followup_20260511"


ACTION_SOURCE_LABELS = {
    0: "No Markov",
    1: "Warm-start LS",
    2: "TD3 accepted",
    3: "LS fallback",
    4: "Nominal fallback",
    5: "LS no-RL",
}


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

    @property
    def warm_start_episode(self) -> int | None:
        warm_start_step = self.bundle.get("warm_start_step")
        if warm_start_step is None or self.step_per_episode <= 0:
            return None
        return int(warm_start_step // self.step_per_episode)


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_run(label: str, run_dir: Path) -> RunBundle:
    bundle = _load_pickle(run_dir / "input_data.pkl")
    return RunBundle(label=label, run_dir=run_dir, bundle=bundle)


def _episode_mean(values: np.ndarray, n_episodes: int) -> np.ndarray:
    values = np.asarray(values, float)
    if n_episodes <= 0 or values.size == 0:
        return np.asarray([], float)
    step_per_episode = values.shape[0] // n_episodes
    trimmed = values[: step_per_episode * n_episodes]
    rows = trimmed.reshape(n_episodes, step_per_episode)
    means = np.empty(n_episodes, dtype=float)
    for idx, row in enumerate(rows):
        finite = row[np.isfinite(row)]
        means[idx] = float(np.mean(finite)) if finite.size else np.nan
    return means


def _episode_fraction(values: np.ndarray, n_episodes: int, target: int) -> np.ndarray:
    values = np.asarray(values, int)
    if n_episodes <= 0 or values.size == 0:
        return np.asarray([], float)
    step_per_episode = values.shape[0] // n_episodes
    trimmed = values[: step_per_episode * n_episodes]
    return np.mean(trimmed.reshape(n_episodes, step_per_episode) == int(target), axis=1)


def _episode_vec_norm(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    trimmed = arr[: step_per_episode * n_episodes]
    reshaped = trimmed.reshape(n_episodes, step_per_episode, -1)
    return np.nanmean(np.linalg.norm(reshaped, axis=2), axis=1)


def _episode_window_mean(values: np.ndarray, start_episode: int, end_episode: int) -> float:
    segment = np.asarray(values, float)[start_episode - 1 : end_episode]
    return float(np.nanmean(segment))


def _warm_start_and_bc_windows(run: RunBundle) -> dict[str, int | None]:
    warm_ep = run.warm_start_episode
    bc = dict(run.bundle.get("behavioral_cloning", {}))
    start_step = bc.get("start_step")
    end_step = bc.get("end_step")
    if run.step_per_episode <= 0 or start_step is None or end_step is None:
        return {"warm_start_end_episode": warm_ep, "bc_start_episode": None, "bc_end_episode": None}
    bc_start_episode = int(start_step // run.step_per_episode) + 1
    bc_end_episode = int(end_step // run.step_per_episode) + 1
    return {
        "warm_start_end_episode": warm_ep,
        "bc_start_episode": bc_start_episode,
        "bc_end_episode": bc_end_episode,
    }


def _plot_reward_recovery(latest: RunBundle, regression: RunBundle, legacy: RunBundle, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for run, color, style in [
        (latest, "#C84C09", "-"),
        (regression, "#7A7A7A", "--"),
        (legacy, "#0B6E4F", "-."),
    ]:
        episodes = np.arange(1, run.n_episodes + 1)
        ax.plot(episodes, run.avg_rewards, linestyle=style, linewidth=2.0, color=color, label=run.label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.set_title("Polymer Markov reward recovery in the latest unified run")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = out_dir / "latest_run_recovery_and_reward.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_latest_action_mix(latest: RunBundle, out_dir: Path) -> Path:
    episodes = np.arange(1, latest.n_episodes + 1)
    td3_frac = _episode_fraction(latest.action_source, latest.n_episodes, 2)
    ls_frac = _episode_fraction(latest.action_source, latest.n_episodes, 3)
    nominal_frac = _episode_fraction(latest.action_source, latest.n_episodes, 4)
    windows = _warm_start_and_bc_windows(latest)

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.6), sharex=True)

    axs[0].plot(episodes, latest.avg_rewards, color="#C84C09", linewidth=2.0)
    axs[0].set_ylabel("Average reward")
    axs[0].set_title("Latest unified run: reward and action-source mix by episode")
    axs[0].grid(alpha=0.25)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    axs[1].plot(episodes, td3_frac, label="TD3 accepted", color="#1f77b4", linewidth=2.0)
    axs[1].plot(episodes, ls_frac, label="LS fallback", color="#0B6E4F", linewidth=2.0)
    axs[1].plot(episodes, nominal_frac, label="Nominal fallback", color="#D55E00", linewidth=1.8, linestyle="--")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Fraction of steps")
    axs[1].grid(alpha=0.25)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].legend(loc="best")

    warm_ep = windows["warm_start_end_episode"]
    if warm_ep is not None:
        for ax in axs:
            ax.axvline(warm_ep, color="0.35", linestyle="--", linewidth=1.2)
    bc_start = windows["bc_start_episode"]
    bc_end = windows["bc_end_episode"]
    if bc_start is not None and bc_end is not None:
        for ax in axs:
            ax.axvspan(bc_start, bc_end, color="#F6C85F", alpha=0.18)

    fig.tight_layout()
    out_path = out_dir / "latest_run_action_mix.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_td3_decline_diagnostics(latest: RunBundle, out_dir: Path) -> Path:
    n_ep = latest.n_episodes
    episodes = np.arange(1, n_ep + 1)
    bundle = latest.bundle

    td3_frac = _episode_fraction(bundle["rl_action_source_log"], n_ep, 2)
    req_score = _episode_mean(bundle["requested_prediction_score_log"], n_ep)
    ls_score = _episode_mean(bundle["ls_prediction_score_log"], n_ep)
    exe_score = _episode_mean(bundle["executed_prediction_score_log"], n_ep)
    req_gain = _episode_mean(bundle["requested_gain_drift_log"], n_ep)
    ls_gain = _episode_mean(bundle["ls_gain_drift_log"], n_ep)
    req_cost_pass = _episode_mean(bundle["requested_cost_guard_pass_log"], n_ep)
    req_norm = _episode_vec_norm(bundle["rl_requested_z_log"], n_ep)
    ls_norm = _episode_vec_norm(bundle["rl_ls_z_log"], n_ep)
    diff_norm = _episode_vec_norm(
        np.asarray(bundle["rl_requested_z_log"], float) - np.asarray(bundle["rl_ls_z_log"], float),
        n_ep,
    )
    windows = _warm_start_and_bc_windows(latest)

    fig, axs = plt.subplots(3, 1, figsize=(12.5, 11.8), sharex=True)

    axs[0].plot(episodes, req_score, label="Requested TD3 score", color="#1f77b4", linewidth=1.9)
    axs[0].plot(episodes, ls_score, label="LS score", color="#0B6E4F", linewidth=1.9)
    axs[0].plot(episodes, exe_score, label="Executed score", color="#C84C09", linewidth=1.9)
    ax0r = axs[0].twinx()
    ax0r.plot(episodes, td3_frac, label="TD3 accepted fraction", color="#7A1FA2", linewidth=1.4, linestyle="--")
    axs[0].axhline(0.0, color="0.45", linewidth=1.0, linestyle=":")
    axs[0].set_ylabel("Prediction score")
    ax0r.set_ylabel("TD3 fraction")
    axs[0].set_title("TD3 decline mechanism in the latest unified run")
    axs[0].grid(alpha=0.25)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    ax0r.spines["top"].set_visible(False)
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0r.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    axs[1].plot(episodes, req_gain, label="Requested TD3 gain drift", color="#1f77b4", linewidth=1.9)
    axs[1].plot(episodes, ls_gain, label="LS gain drift", color="#0B6E4F", linewidth=1.9)
    axs[1].axhline(float(bundle.get("markov_gain_drift_max", 0.10)), color="#D55E00", linestyle="--", linewidth=1.2, label="Drift limit")
    ax1r = axs[1].twinx()
    ax1r.plot(episodes, req_cost_pass, label="Requested cost-guard pass", color="#7A1FA2", linewidth=1.4, linestyle=":")
    axs[1].set_ylabel("Gain drift")
    ax1r.set_ylabel("Cost pass fraction")
    axs[1].grid(alpha=0.25)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    ax1r.spines["top"].set_visible(False)
    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    axs[2].plot(episodes, req_norm, label="Requested TD3 ||z||", color="#1f77b4", linewidth=1.9)
    axs[2].plot(episodes, ls_norm, label="LS ||z||", color="#0B6E4F", linewidth=1.9)
    axs[2].plot(episodes, diff_norm, label="||z_TD3 - z_LS||", color="#C84C09", linewidth=1.9)
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Correction norm")
    axs[2].grid(alpha=0.25)
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[2].legend(loc="best")

    warm_ep = windows["warm_start_end_episode"]
    if warm_ep is not None:
        for ax in axs:
            ax.axvline(warm_ep, color="0.35", linestyle="--", linewidth=1.2)
    bc_start = windows["bc_start_episode"]
    bc_end = windows["bc_end_episode"]
    if bc_start is not None and bc_end is not None:
        for ax in axs:
            ax.axvspan(bc_start, bc_end, color="#F6C85F", alpha=0.18)

    fig.tight_layout()
    out_path = out_dir / "latest_td3_decline_diagnostics.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _summary(run: RunBundle) -> dict:
    n_ep = run.n_episodes
    src = run.action_source
    td3_frac = _episode_fraction(src, n_ep, 2)
    ls_frac = _episode_fraction(src, n_ep, 3)
    nom_frac = _episode_fraction(src, n_ep, 4)
    req_score = _episode_mean(run.bundle["requested_prediction_score_log"], n_ep)
    ls_score = _episode_mean(run.bundle["ls_prediction_score_log"], n_ep)
    exe_score = _episode_mean(run.bundle["executed_prediction_score_log"], n_ep)
    req_gain = _episode_mean(run.bundle["requested_gain_drift_log"], n_ep)
    req_cost_pass = _episode_mean(run.bundle["requested_cost_guard_pass_log"], n_ep)
    diff_norm = _episode_vec_norm(
        np.asarray(run.bundle["rl_requested_z_log"], float) - np.asarray(run.bundle["rl_ls_z_log"], float),
        n_ep,
    )
    windows = _warm_start_and_bc_windows(run)
    return {
        "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
        "n_episodes": int(n_ep),
        "step_per_episode": int(run.step_per_episode),
        "warm_start_end_episode": windows["warm_start_end_episode"],
        "bc_start_episode": windows["bc_start_episode"],
        "bc_end_episode": windows["bc_end_episode"],
        "avg_reward_mean": float(np.nanmean(run.avg_rewards)),
        "avg_reward_first10": float(np.nanmean(run.avg_rewards[:10])),
        "avg_reward_last10": float(np.nanmean(run.avg_rewards[-10:])),
        "reward_final_episode": float(run.avg_rewards[-1]),
        "overall_td3_fraction": float(np.mean(src == 2)),
        "overall_ls_fraction": float(np.mean(src == 3)),
        "overall_nominal_fraction": float(np.mean(src == 4)),
        "td3_fraction_ep_11_50": _episode_window_mean(td3_frac, 11, min(50, n_ep)),
        "td3_fraction_last50_episodes": _episode_window_mean(td3_frac, max(1, n_ep - 49), n_ep),
        "ls_fraction_last50_episodes": _episode_window_mean(ls_frac, max(1, n_ep - 49), n_ep),
        "nominal_fraction_last50_episodes": _episode_window_mean(nom_frac, max(1, n_ep - 49), n_ep),
        "requested_score_ep_11_50": _episode_window_mean(req_score, 11, min(50, n_ep)),
        "requested_score_last50_episodes": _episode_window_mean(req_score, max(1, n_ep - 49), n_ep),
        "ls_score_ep_11_50": _episode_window_mean(ls_score, 11, min(50, n_ep)),
        "ls_score_last50_episodes": _episode_window_mean(ls_score, max(1, n_ep - 49), n_ep),
        "executed_score_last50_episodes": _episode_window_mean(exe_score, max(1, n_ep - 49), n_ep),
        "requested_gain_last50_episodes": _episode_window_mean(req_gain, max(1, n_ep - 49), n_ep),
        "requested_cost_pass_last50_episodes": _episode_window_mean(req_cost_pass, max(1, n_ep - 49), n_ep),
        "requested_ls_gap_norm_last50_episodes": _episode_window_mean(diff_norm, max(1, n_ep - 49), n_ep),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    latest = _load_run("Latest unified", LATEST_UNIFIED_RUN)
    regression = _load_run("Pre-fix unified", REGRESSION_UNIFIED_RUN)
    legacy = _load_run("Legacy reference", LEGACY_RUN)

    figure_paths = [
        _plot_reward_recovery(latest, regression, legacy, OUT_DIR),
        _plot_latest_action_mix(latest, OUT_DIR),
        _plot_td3_decline_diagnostics(latest, OUT_DIR),
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_unified": _summary(latest),
        "pre_fix_unified": _summary(regression),
        "legacy_reference": _summary(legacy),
        "figure_paths": [str(path.relative_to(REPO_ROOT)) for path in figure_paths],
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
