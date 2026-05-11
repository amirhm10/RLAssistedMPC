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
    def replay_storage_mode(self) -> str:
        return str(self.bundle.get("replay_storage_mode", "unknown"))


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
    arr = arr[: step_per_episode * n_episodes]
    rows = arr.reshape(n_episodes, step_per_episode)
    out = np.empty(n_episodes, dtype=float)
    for idx, row in enumerate(rows):
        finite = row[np.isfinite(row)]
        out[idx] = float(np.mean(finite)) if finite.size else np.nan
    return out


def _episode_fraction(values: np.ndarray, n_episodes: int, target: int) -> np.ndarray:
    arr = np.asarray(values, int)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    arr = arr[: step_per_episode * n_episodes]
    return np.mean(arr.reshape(n_episodes, step_per_episode) == int(target), axis=1)


def _episode_vec_norm(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    step_per_episode = arr.shape[0] // n_episodes
    arr = arr[: step_per_episode * n_episodes]
    reshaped = arr.reshape(n_episodes, step_per_episode, -1)
    return np.nanmean(np.linalg.norm(reshaped, axis=2), axis=1)


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


def _validate_fixed_conditions(executed: RunBundle, requested: RunBundle) -> dict:
    checks = {
        "basis_family": (
            executed.bundle.get("basis_family"),
            requested.bundle.get("basis_family"),
        ),
        "warm_start_step": (
            executed.bundle.get("warm_start_step"),
            requested.bundle.get("warm_start_step"),
        ),
        "markov_s_pred_min": (
            executed.bundle.get("markov_s_pred_min"),
            requested.bundle.get("markov_s_pred_min"),
        ),
        "markov_gain_drift_max": (
            executed.bundle.get("markov_gain_drift_max"),
            requested.bundle.get("markov_gain_drift_max"),
        ),
        "behavioral_cloning": (
            {
                k: v
                for k, v in dict(executed.bundle.get("behavioral_cloning", {})).items()
                if k not in {"active_log"}
            },
            {
                k: v
                for k, v in dict(requested.bundle.get("behavioral_cloning", {})).items()
                if k not in {"active_log"}
            },
        ),
    }
    mismatches = {
        name: {"executed": a, "requested": b}
        for name, (a, b) in checks.items()
        if a != b
    }
    return {"all_match": len(mismatches) == 0, "mismatches": mismatches}


def _plot_ab_metrics(executed: RunBundle, requested: RunBundle, out_dir: Path) -> list[Path]:
    n_ep = min(executed.n_episodes, requested.n_episodes)
    x = np.arange(1, n_ep + 1)
    warm = _warm_start_and_bc(executed)

    ex_td3 = _episode_fraction(executed.bundle["rl_action_source_log"], executed.n_episodes, 2)[:n_ep]
    rq_td3 = _episode_fraction(requested.bundle["rl_action_source_log"], requested.n_episodes, 2)[:n_ep]
    ex_req_score = _episode_mean(executed.bundle["requested_prediction_score_log"], executed.n_episodes)[:n_ep]
    rq_req_score = _episode_mean(requested.bundle["requested_prediction_score_log"], requested.n_episodes)[:n_ep]
    ex_ls_score = _episode_mean(executed.bundle["ls_prediction_score_log"], executed.n_episodes)[:n_ep]
    rq_ls_score = _episode_mean(requested.bundle["ls_prediction_score_log"], requested.n_episodes)[:n_ep]
    ex_gap = _episode_vec_norm(
        np.asarray(executed.bundle["rl_requested_z_log"], float) - np.asarray(executed.bundle["rl_ls_z_log"], float),
        executed.n_episodes,
    )[:n_ep]
    rq_gap = _episode_vec_norm(
        np.asarray(requested.bundle["rl_requested_z_log"], float) - np.asarray(requested.bundle["rl_ls_z_log"], float),
        requested.n_episodes,
    )[:n_ep]

    fig, axs = plt.subplots(4, 1, figsize=(12.8, 14.5), sharex=True)
    axs[0].plot(x, executed.avg_rewards[:n_ep], color="#C84C09", linewidth=2.0, label="Executed action replay")
    axs[0].plot(x, requested.avg_rewards[:n_ep], color="#1f77b4", linewidth=2.0, label="Requested action replay")
    axs[0].set_ylabel("Avg. reward")
    axs[0].set_title("Polymer Markov replay-storage A/B")
    axs[0].legend(loc="best")

    axs[1].plot(x, ex_td3, color="#C84C09", linewidth=2.0, label="Executed replay")
    axs[1].plot(x, rq_td3, color="#1f77b4", linewidth=2.0, label="Requested replay")
    axs[1].set_ylabel("TD3 frac")
    axs[1].legend(loc="best")

    axs[2].plot(x, ex_req_score, color="#C84C09", linewidth=1.8, label="Requested TD3 score, executed replay")
    axs[2].plot(x, rq_req_score, color="#1f77b4", linewidth=1.8, label="Requested TD3 score, requested replay")
    axs[2].plot(x, ex_ls_score, color="#C84C09", linewidth=1.4, linestyle="--", label="LS score, executed replay")
    axs[2].plot(x, rq_ls_score, color="#1f77b4", linewidth=1.4, linestyle="--", label="LS score, requested replay")
    axs[2].axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
    axs[2].set_ylabel("Score")
    axs[2].legend(loc="best", ncol=2, fontsize=9)

    axs[3].plot(x, ex_gap, color="#C84C09", linewidth=2.0, label="||z_TD3 - z_LS||, executed replay")
    axs[3].plot(x, rq_gap, color="#1f77b4", linewidth=2.0, label="||z_TD3 - z_LS||, requested replay")
    axs[3].set_ylabel("Norm")
    axs[3].set_xlabel("Episode")
    axs[3].legend(loc="best")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if warm["warm_end"] is not None:
            ax.axvline(warm["warm_end"], color="0.35", linestyle="--", linewidth=1.1)
        if warm["bc_start"] is not None and warm["bc_end"] is not None:
            ax.axvspan(warm["bc_start"], warm["bc_end"], color="#F6C85F", alpha=0.18)

    fig.tight_layout()
    out1 = out_dir / "replay_storage_ab_metrics.png"
    fig.savefig(out1, dpi=220, bbox_inches="tight")
    plt.close(fig)

    ex_ls_frac = _episode_fraction(executed.bundle["rl_action_source_log"], executed.n_episodes, 3)[:n_ep]
    rq_ls_frac = _episode_fraction(requested.bundle["rl_action_source_log"], requested.n_episodes, 3)[:n_ep]
    ex_nom_frac = _episode_fraction(executed.bundle["rl_action_source_log"], executed.n_episodes, 4)[:n_ep]
    rq_nom_frac = _episode_fraction(requested.bundle["rl_action_source_log"], requested.n_episodes, 4)[:n_ep]

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 8.6), sharex=True)
    axs[0].plot(x, ex_td3, color="#C84C09", linewidth=2.0, label="TD3, executed replay")
    axs[0].plot(x, rq_td3, color="#1f77b4", linewidth=2.0, label="TD3, requested replay")
    axs[0].plot(x, ex_ls_frac, color="#C84C09", linewidth=1.5, linestyle="--", label="LS, executed replay")
    axs[0].plot(x, rq_ls_frac, color="#1f77b4", linewidth=1.5, linestyle="--", label="LS, requested replay")
    axs[0].plot(x, ex_nom_frac, color="#C84C09", linewidth=1.2, linestyle=":", label="Nominal, executed replay")
    axs[0].plot(x, rq_nom_frac, color="#1f77b4", linewidth=1.2, linestyle=":", label="Nominal, requested replay")
    axs[0].set_ylabel("Action-source fraction")
    axs[0].legend(loc="best", ncol=2, fontsize=9)

    axs[1].plot(x, executed.avg_rewards[:n_ep], color="#C84C09", linewidth=2.0, label="Executed action replay")
    axs[1].plot(x, requested.avg_rewards[:n_ep], color="#1f77b4", linewidth=2.0, label="Requested action replay")
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
    out2 = out_dir / "replay_storage_ab_action_mix.png"
    fig.savefig(out2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out1, out2]


def _summary(executed: RunBundle, requested: RunBundle) -> dict:
    def run_metrics(run: RunBundle) -> dict:
        n_ep = run.n_episodes
        td3_frac = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 2)
        ls_score = _episode_mean(run.bundle["ls_prediction_score_log"], n_ep)
        req_score = _episode_mean(run.bundle["requested_prediction_score_log"], n_ep)
        gap = _episode_vec_norm(
            np.asarray(run.bundle["rl_requested_z_log"], float) - np.asarray(run.bundle["rl_ls_z_log"], float),
            n_ep,
        )
        return {
            "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
            "replay_storage_mode": run.replay_storage_mode,
            "n_episodes": int(n_ep),
            "avg_reward_mean": float(np.nanmean(run.avg_rewards)),
            "avg_reward_last10": float(np.nanmean(run.avg_rewards[-10:])),
            "td3_fraction_ep_11_50": float(np.nanmean(td3_frac[10:min(50, n_ep)])),
            "td3_fraction_last50": float(np.nanmean(td3_frac[max(0, n_ep - 50):])),
            "requested_score_ep_11_50": float(np.nanmean(req_score[10:min(50, n_ep)])),
            "requested_score_last50": float(np.nanmean(req_score[max(0, n_ep - 50):])),
            "ls_score_last50": float(np.nanmean(ls_score[max(0, n_ep - 50):])),
            "z_gap_last50": float(np.nanmean(gap[max(0, n_ep - 50):])),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_condition_check": _validate_fixed_conditions(executed, requested),
        "executed_replay_run": run_metrics(executed),
        "requested_replay_run": run_metrics(requested),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare polymer Markov replay-storage A/B runs.")
    parser.add_argument("--executed-run", required=True, help="Run directory for rl_store_executed_action_in_replay=True")
    parser.add_argument("--requested-run", required=True, help="Run directory for rl_store_executed_action_in_replay=False")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "report" / "figures" / "polymer_markov_replay_storage_ab_20260511"),
        help="Output directory for figures and summary JSON.",
    )
    args = parser.parse_args()

    executed = _load_run("Executed action replay", args.executed_run)
    requested = _load_run("Requested action replay", args.requested_run)
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = _plot_ab_metrics(executed, requested, out_dir)
    summary = _summary(executed, requested)
    summary["figure_paths"] = [str(path.relative_to(REPO_ROOT)) for path in figure_paths]

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
