from __future__ import annotations

import argparse
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


@dataclass
class RunBundle:
    label: str
    run_dir: Path
    bundle: dict

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle.get("avg_rewards", []), float)

    @property
    def n_episodes(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def action_source(self) -> np.ndarray:
        return np.asarray(self.bundle.get("rl_action_source_log", []), int)

    @property
    def step_per_episode(self) -> int:
        if self.n_episodes <= 0:
            return 0
        return int(self.action_source.size // self.n_episodes)

    @property
    def z_bound(self) -> float:
        return float(self.bundle.get("markov_z_bound", np.nan))


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resolve_run_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Run directory does not exist: {p}")
    if p.is_file():
        p = p.parent
    bundle_path = p / "input_data.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Expected input_data.pkl under: {p}")
    return p


def _load_run(label: str, path_str: str) -> RunBundle:
    run_dir = _resolve_run_dir(path_str)
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


def _warm_start_and_bc(run: RunBundle) -> dict[str, int | None]:
    warm_start_step = run.bundle.get("warm_start_step")
    warm_end = None
    if warm_start_step is not None and run.step_per_episode > 0:
        warm_end = int(warm_start_step // run.step_per_episode)
    bc = dict(run.bundle.get("behavioral_cloning", {}))
    start_step = bc.get("start_step")
    end_step = bc.get("end_step")
    bc_start = None
    bc_end = None
    if start_step is not None and end_step is not None and run.step_per_episode > 0:
        bc_start = int(start_step // run.step_per_episode) + 1
        bc_end = int(end_step // run.step_per_episode) + 1
    return {"warm_end": warm_end, "bc_start": bc_start, "bc_end": bc_end}


def _validate_fixed_conditions(low: RunBundle, high: RunBundle) -> dict:
    checks = {
        "markov_state_mode": (low.bundle.get("markov_state_mode"), high.bundle.get("markov_state_mode")),
        "basis_family": (low.bundle.get("basis_family"), high.bundle.get("basis_family")),
        "replay_storage_mode": (low.bundle.get("replay_storage_mode"), high.bundle.get("replay_storage_mode")),
        "warm_start_step": (low.bundle.get("warm_start_step"), high.bundle.get("warm_start_step")),
        "markov_s_pred_min": (low.bundle.get("markov_s_pred_min"), high.bundle.get("markov_s_pred_min")),
        "markov_gain_drift_max": (low.bundle.get("markov_gain_drift_max"), high.bundle.get("markov_gain_drift_max")),
        "behavioral_cloning": (
            {k: v for k, v in dict(low.bundle.get("behavioral_cloning", {})).items() if k not in {"active_log"}},
            {k: v for k, v in dict(high.bundle.get("behavioral_cloning", {})).items() if k not in {"active_log"}},
        ),
    }
    mismatches = {
        name: {"low_z": a, "high_z": b}
        for name, (a, b) in checks.items()
        if a != b
    }
    return {"all_match": len(mismatches) == 0, "mismatches": mismatches}


def _plot_zbound_ab(low: RunBundle, high: RunBundle, out_dir: Path) -> list[Path]:
    n_ep = min(low.n_episodes, high.n_episodes)
    x = np.arange(1, n_ep + 1)
    warm = _warm_start_and_bc(low)

    low_td3 = _episode_fraction(low.bundle["rl_action_source_log"], low.n_episodes, 2)[:n_ep]
    high_td3 = _episode_fraction(high.bundle["rl_action_source_log"], high.n_episodes, 2)[:n_ep]
    low_req = _episode_mean(low.bundle["requested_prediction_score_log"], low.n_episodes)[:n_ep]
    high_req = _episode_mean(high.bundle["requested_prediction_score_log"], high.n_episodes)[:n_ep]
    low_ls = _episode_mean(low.bundle["ls_prediction_score_log"], low.n_episodes)[:n_ep]
    high_ls = _episode_mean(high.bundle["ls_prediction_score_log"], high.n_episodes)[:n_ep]
    low_sat = _episode_fraction_raw_saturation(low.bundle["rl_requested_raw_action_log"], low.n_episodes)[:n_ep]
    high_sat = _episode_fraction_raw_saturation(high.bundle["rl_requested_raw_action_log"], high.n_episodes)[:n_ep]
    low_req_z = _episode_mean_max_abs(low.bundle["rl_requested_z_log"], low.n_episodes)[:n_ep]
    high_req_z = _episode_mean_max_abs(high.bundle["rl_requested_z_log"], high.n_episodes)[:n_ep]
    low_ls_z = _episode_mean_max_abs(low.bundle["rl_ls_z_log"], low.n_episodes)[:n_ep]
    high_ls_z = _episode_mean_max_abs(high.bundle["rl_ls_z_log"], high.n_episodes)[:n_ep]
    low_gap = _episode_vec_norm(np.asarray(low.bundle["rl_requested_z_log"], float) - np.asarray(low.bundle["rl_ls_z_log"], float), low.n_episodes)[:n_ep]
    high_gap = _episode_vec_norm(np.asarray(high.bundle["rl_requested_z_log"], float) - np.asarray(high.bundle["rl_ls_z_log"], float), high.n_episodes)[:n_ep]

    fig, axs = plt.subplots(4, 1, figsize=(12.8, 15.2), sharex=True)
    axs[0].plot(x, low.avg_rewards[:n_ep], color="#1f77b4", linewidth=2.0, label=f"z_bound = {low.z_bound:.2f}")
    axs[0].plot(x, high.avg_rewards[:n_ep], color="#C84C09", linewidth=2.0, label=f"z_bound = {high.z_bound:.2f}")
    axs[0].set_ylabel("Avg. reward")
    axs[0].set_title("Polymer Markov z-bound A/B")
    axs[0].legend(loc="best")

    axs[1].plot(x, low_td3, color="#1f77b4", linewidth=2.0, label=f"TD3 frac, z = {low.z_bound:.2f}")
    axs[1].plot(x, high_td3, color="#C84C09", linewidth=2.0, label=f"TD3 frac, z = {high.z_bound:.2f}")
    axs[1].set_ylabel("TD3 frac")
    axs[1].legend(loc="best")

    axs[2].plot(x, low_req, color="#1f77b4", linewidth=1.8, label=f"Req. TD3 score, z = {low.z_bound:.2f}")
    axs[2].plot(x, high_req, color="#C84C09", linewidth=1.8, label=f"Req. TD3 score, z = {high.z_bound:.2f}")
    axs[2].plot(x, low_ls, color="#1f77b4", linewidth=1.2, linestyle="--", label=f"LS score, z = {low.z_bound:.2f}")
    axs[2].plot(x, high_ls, color="#C84C09", linewidth=1.2, linestyle="--", label=f"LS score, z = {high.z_bound:.2f}")
    axs[2].axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
    axs[2].set_ylabel("Score")
    axs[2].legend(loc="best", ncol=2, fontsize=9)

    axs[3].plot(x, low_sat, color="#1f77b4", linewidth=1.8, label=f"Raw sat, z = {low.z_bound:.2f}")
    axs[3].plot(x, high_sat, color="#C84C09", linewidth=1.8, label=f"Raw sat, z = {high.z_bound:.2f}")
    axs[3].plot(x, low_req_z, color="#1f77b4", linewidth=1.2, linestyle="--", label=f"max|z_TD3|, z = {low.z_bound:.2f}")
    axs[3].plot(x, high_req_z, color="#C84C09", linewidth=1.2, linestyle="--", label=f"max|z_TD3|, z = {high.z_bound:.2f}")
    axs[3].set_ylabel("Saturation / max |z|")
    axs[3].set_xlabel("Episode")
    axs[3].legend(loc="best", ncol=2, fontsize=9)

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if warm["warm_end"] is not None:
            ax.axvline(warm["warm_end"], color="0.35", linestyle="--", linewidth=1.1)
        if warm["bc_start"] is not None and warm["bc_end"] is not None:
            ax.axvspan(warm["bc_start"], warm["bc_end"], color="#F6C85F", alpha=0.18)

    fig.tight_layout()
    out1 = out_dir / "zbound_ab_metrics.png"
    fig.savefig(out1, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 8.8), sharex=True)
    axs[0].plot(x, low_gap, color="#1f77b4", linewidth=2.0, label=f"||z_TD3 - z_LS||, z = {low.z_bound:.2f}")
    axs[0].plot(x, high_gap, color="#C84C09", linewidth=2.0, label=f"||z_TD3 - z_LS||, z = {high.z_bound:.2f}")
    axs[0].plot(x, low_ls_z, color="#1f77b4", linewidth=1.3, linestyle="--", label=f"max|z_LS|, z = {low.z_bound:.2f}")
    axs[0].plot(x, high_ls_z, color="#C84C09", linewidth=1.3, linestyle="--", label=f"max|z_LS|, z = {high.z_bound:.2f}")
    axs[0].set_ylabel("Norm / max |z|")
    axs[0].legend(loc="best", ncol=2, fontsize=9)

    axs[1].plot(x, low.avg_rewards[:n_ep], color="#1f77b4", linewidth=2.0, label=f"z_bound = {low.z_bound:.2f}")
    axs[1].plot(x, high.avg_rewards[:n_ep], color="#C84C09", linewidth=2.0, label=f"z_bound = {high.z_bound:.2f}")
    axs[1].set_ylabel("Avg. reward")
    axs[1].set_xlabel("Episode")
    axs[1].legend(loc="best")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if warm["warm_end"] is not None:
            ax.axvline(warm["warm_end"], color="0.35", linestyle="--", linewidth=1.1)
        if warm["bc_start"] is not None and warm["bc_end"] is not None:
            ax.axvspan(warm["bc_start"], warm["bc_end"], color="#F6C85F", alpha=0.18)

    fig.tight_layout()
    out2 = out_dir / "zbound_ab_geometry.png"
    fig.savefig(out2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out1, out2]


def _summary(low: RunBundle, high: RunBundle) -> dict:
    def run_metrics(run: RunBundle) -> dict:
        n_ep = run.n_episodes
        td3_frac = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 2)
        req_score = _episode_mean(run.bundle["requested_prediction_score_log"], n_ep)
        ls_score = _episode_mean(run.bundle["ls_prediction_score_log"], n_ep)
        raw_sat = _episode_fraction_raw_saturation(run.bundle["rl_requested_raw_action_log"], n_ep)
        req_zmax = _episode_mean_max_abs(run.bundle["rl_requested_z_log"], n_ep)
        ls_zmax = _episode_mean_max_abs(run.bundle["rl_ls_z_log"], n_ep)
        gap = _episode_vec_norm(
            np.asarray(run.bundle["rl_requested_z_log"], float) - np.asarray(run.bundle["rl_ls_z_log"], float),
            n_ep,
        )
        return {
            "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
            "z_bound": float(run.z_bound),
            "markov_state_mode": run.bundle.get("markov_state_mode"),
            "replay_storage_mode": run.bundle.get("replay_storage_mode"),
            "avg_reward_mean": float(np.nanmean(run.avg_rewards)),
            "avg_reward_last10": float(np.nanmean(run.avg_rewards[-10:])),
            "td3_fraction_11_50": float(np.nanmean(td3_frac[10:min(50, n_ep)])),
            "td3_fraction_last50": float(np.nanmean(td3_frac[max(0, n_ep - 50):])),
            "requested_score_11_50": float(np.nanmean(req_score[10:min(50, n_ep)])),
            "requested_score_last50": float(np.nanmean(req_score[max(0, n_ep - 50):])),
            "ls_score_last50": float(np.nanmean(ls_score[max(0, n_ep - 50):])),
            "raw_saturation_all": float(np.nanmean(raw_sat)),
            "raw_saturation_last50": float(np.nanmean(raw_sat[max(0, n_ep - 50):])),
            "requested_zmax_last50": float(np.nanmean(req_zmax[max(0, n_ep - 50):])),
            "ls_zmax_last50": float(np.nanmean(ls_zmax[max(0, n_ep - 50):])),
            "z_gap_last50": float(np.nanmean(gap[max(0, n_ep - 50):])),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_condition_check": _validate_fixed_conditions(low, high),
        "low_z_run": run_metrics(low),
        "high_z_run": run_metrics(high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare polymer Markov z-bound A/B runs.")
    parser.add_argument("--low-z-run", required=True, help="Run directory for the lower z_bound variant, e.g. 0.05.")
    parser.add_argument("--high-z-run", required=True, help="Run directory for the higher z_bound variant, e.g. 0.08.")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "report" / "figures" / "polymer_markov_zbound_ab_20260511"),
        help="Output directory for figures and summary JSON.",
    )
    args = parser.parse_args()

    low = _load_run("Lower z-bound run", args.low_z_run)
    high = _load_run("Higher z-bound run", args.high_z_run)
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = _plot_zbound_ab(low, high, out_dir)
    summary = _summary(low, high)
    summary["figure_paths"] = [str(path.relative_to(REPO_ROOT)) for path in figure_paths]

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
