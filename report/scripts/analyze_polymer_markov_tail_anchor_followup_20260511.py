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

from utils.helpers import apply_min_max, reverse_min_max


LATEST_TAIL_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008" / "20260511_230422"
PREV_Z008_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008" / "20260511_213647"
LOW_Z_CONDITIONED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260511_171749"
MPC_BASELINE = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_tail_anchor_followup_20260511"


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


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_run(label: str, run_dir: Path) -> RunBundle:
    return RunBundle(label=label, run_dir=run_dir, bundle=_load_pickle(run_dir / "input_data.pkl"))


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
    raise ValueError("_episode_mean expects a 1D array.")


def _episode_fraction(values: np.ndarray, n_episodes: int, target: int) -> np.ndarray:
    arr = np.asarray(values, int)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.size // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps)
    return np.mean(arr == int(target), axis=1)


def _episode_fraction_raw_saturation(values: np.ndarray, n_episodes: int, threshold: float = 0.95) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps, -1)
    return np.mean(np.max(np.abs(arr), axis=2) >= float(threshold), axis=1)


def _episode_mean_max_abs(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps, -1)
    return np.nanmean(np.max(np.abs(arr), axis=2), axis=1)


def _episode_vec_norm(values: np.ndarray, n_episodes: int) -> np.ndarray:
    arr = np.asarray(values, float)
    if n_episodes <= 0 or arr.size == 0:
        return np.asarray([], float)
    steps = arr.shape[0] // n_episodes
    arr = arr[: steps * n_episodes].reshape(n_episodes, steps, -1)
    return np.nanmean(np.linalg.norm(arr, axis=2), axis=1)


def _warm_windows(run: RunBundle) -> dict[str, int | None]:
    warm_start_step = run.bundle.get("warm_start_step")
    warm_end = None if warm_start_step is None or run.step_per_episode <= 0 else int(warm_start_step // run.step_per_episode)
    bc = dict(run.bundle.get("behavioral_cloning", {}))
    bc_start = bc_end = None
    if run.step_per_episode > 0 and bc.get("start_step") is not None and bc.get("end_step") is not None:
        bc_start = int(bc["start_step"] // run.step_per_episode) + 1
        bc_end = int(bc["end_step"] // run.step_per_episode) + 1
    return {"warm_end": warm_end, "bc_start": bc_start, "bc_end": bc_end}


def _plot_reward_and_td3_mix(latest: RunBundle, prev_z008: RunBundle, low_z: RunBundle, out_dir: Path) -> Path:
    n_ep = latest.n_episodes
    x = np.arange(1, n_ep + 1)
    latest_td3 = _episode_fraction(latest.bundle["rl_action_source_log"], n_ep, 2)
    prev_td3 = _episode_fraction(prev_z008.bundle["rl_action_source_log"], prev_z008.n_episodes, 2)
    low_td3 = _episode_fraction(low_z.bundle["rl_action_source_log"], low_z.n_episodes, 2)
    windows = _warm_windows(latest)

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 8.8), sharex=True)
    axs[0].plot(x, latest.avg_rewards, color="#C84C09", linewidth=2.0, label="Latest z = 0.08 + tail anchor")
    axs[0].plot(x, prev_z008.avg_rewards, color="#7A7A7A", linewidth=1.8, linestyle="--", label="Previous z = 0.08")
    axs[0].plot(x, low_z.avg_rewards, color="#0B6E4F", linewidth=1.8, linestyle="-.", label="Conditioned z = 0.05")
    axs[0].set_ylabel("Avg. reward")
    axs[0].set_title("Polymer Markov: LS tail-anchor follow-up")
    axs[0].legend(loc="best")

    axs[1].plot(x, latest_td3, color="#C84C09", linewidth=2.0, label="Latest z = 0.08 + tail anchor")
    axs[1].plot(x, prev_td3, color="#7A7A7A", linewidth=1.8, linestyle="--", label="Previous z = 0.08")
    axs[1].plot(x, low_td3, color="#0B6E4F", linewidth=1.8, linestyle="-.", label="Conditioned z = 0.05")
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
    out = out_dir / "tail_anchor_reward_and_td3_mix.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_tail_anchor_diagnostics(latest: RunBundle, prev_z008: RunBundle, out_dir: Path) -> Path:
    n_ep = latest.n_episodes
    x = np.arange(1, n_ep + 1)
    windows = _warm_windows(latest)

    req_score = _episode_mean(latest.bundle["requested_prediction_score_log"], n_ep)
    ls_score = _episode_mean(latest.bundle["ls_prediction_score_log"], n_ep)
    exe_score = _episode_mean(latest.bundle["executed_prediction_score_log"], n_ep)
    td3_frac = _episode_fraction(latest.bundle["rl_action_source_log"], n_ep, 2)
    bc_active = _episode_mean(latest.bundle["bc_active_log"], n_ep)
    bc_weight = _episode_mean(latest.bundle["bc_weight_log"], n_ep)
    sat_latest = _episode_fraction_raw_saturation(latest.bundle["rl_requested_raw_action_log"], n_ep)
    sat_prev = _episode_fraction_raw_saturation(prev_z008.bundle["rl_requested_raw_action_log"], prev_z008.n_episodes)
    req_zmax = _episode_mean_max_abs(latest.bundle["rl_requested_z_log"], n_ep)
    ls_zmax = _episode_mean_max_abs(latest.bundle["rl_ls_z_log"], n_ep)
    z_gap = _episode_vec_norm(
        np.asarray(latest.bundle["rl_requested_z_log"], float) - np.asarray(latest.bundle["rl_ls_z_log"], float),
        n_ep,
    )

    fig, axs = plt.subplots(3, 1, figsize=(12.8, 12.2), sharex=True)

    axs[0].plot(x, req_score, color="#1f77b4", linewidth=1.9, label="Requested TD3 score")
    axs[0].plot(x, ls_score, color="#0B6E4F", linewidth=1.9, label="LS score")
    axs[0].plot(x, exe_score, color="#C84C09", linewidth=1.9, label="Executed score")
    ax0r = axs[0].twinx()
    ax0r.plot(x, td3_frac, color="#7A1FA2", linewidth=1.3, linestyle="--", label="TD3 accepted fraction")
    axs[0].axhline(0.0, color="0.45", linewidth=1.0, linestyle=":")
    axs[0].set_ylabel("Prediction score")
    ax0r.set_ylabel("TD3 frac")
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0r.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    axs[1].plot(x, bc_active, color="#C84C09", linewidth=2.0, label="BC/tail-anchor active fraction")
    axs[1].plot(x, bc_weight, color="#D55E00", linewidth=1.5, linestyle="--", label="Mean BC weight")
    axs[1].plot(x, sat_latest, color="#1f77b4", linewidth=1.4, label="Raw-action saturation, latest")
    axs[1].plot(x, sat_prev, color="#7A7A7A", linewidth=1.4, linestyle=":", label="Raw-action saturation, prev z = 0.08")
    axs[1].set_ylabel("Activation / saturation")
    axs[1].legend(loc="best", ncol=2, fontsize=9)

    axs[2].plot(x, req_zmax, color="#1f77b4", linewidth=1.7, label="Mean max |z_TD3|")
    axs[2].plot(x, ls_zmax, color="#0B6E4F", linewidth=1.7, linestyle="--", label="Mean max |z_LS|")
    axs[2].plot(x, z_gap, color="#C84C09", linewidth=1.7, linestyle="-.", label="||z_TD3 - z_LS||")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Norm / max |z|")
    axs[2].legend(loc="best")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if windows["warm_end"] is not None:
            ax.axvline(windows["warm_end"], color="0.35", linestyle="--", linewidth=1.1)
        if windows["bc_start"] is not None and windows["bc_end"] is not None:
            ax.axvspan(windows["bc_start"], windows["bc_end"], color="#F6C85F", alpha=0.18)
    ax0r.spines["top"].set_visible(False)

    fig.tight_layout()
    out = out_dir / "tail_anchor_diagnostics.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _final_test_episode_metrics(latest: RunBundle, mpc_bundle: dict) -> tuple[dict, dict]:
    time_in_sub_episodes = int(latest.bundle["time_in_sub_episodes"])
    rl_y = np.asarray(latest.bundle["y_line_full"], float)[-(time_in_sub_episodes + 1) :, :]
    rl_u = np.asarray(latest.bundle["u_step_full"], float)[-time_in_sub_episodes:, :]
    mpc_y = np.asarray(mpc_bundle["y_mpc"], float)[-(time_in_sub_episodes + 1) :, :]
    mpc_u = np.asarray(mpc_bundle["u_mpc"], float)[-time_in_sub_episodes:, :]

    data_min = np.asarray(latest.bundle["data_min"], float)
    data_max = np.asarray(latest.bundle["data_max"], float)
    n_inputs = int(rl_u.shape[1])
    y_ss = np.asarray(latest.bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = apply_min_max(y_ss, data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled_dev = np.asarray(latest.bundle["y_sp"], float)[-time_in_sub_episodes:, :]
    y_sp_phys = reverse_min_max(
        y_sp_scaled_dev + y_ss_scaled,
        data_min[n_inputs:],
        data_max[n_inputs:],
    )

    rl_err = rl_y[1:, :] - y_sp_phys
    mpc_err = mpc_y[1:, :] - y_sp_phys
    rl_du = np.diff(np.vstack([rl_u[0:1, :], rl_u]), axis=0)
    mpc_du = np.diff(np.vstack([mpc_u[0:1, :], mpc_u]), axis=0)

    metrics = {
        "avg_reward_rl_last_test_episode": float(latest.avg_rewards[-1]),
        "avg_reward_mpc_last_test_episode": float(np.asarray(mpc_bundle["avg_rewards"], float)[-1]),
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


def _plot_final_test_episode(latest: RunBundle, payload: dict, out_dir: Path) -> Path:
    delta_t = float(latest.bundle["delta_t"])
    meta = dict(latest.bundle.get("system_metadata", {}))
    output_labels = list(meta.get("output_labels", ["Output 1", "Output 2"]))
    input_labels = list(meta.get("input_labels", ["Input 1", "Input 2"]))
    time_label = str(meta.get("time_label", "Time"))

    rl_y = np.asarray(payload["rl_y"], float)
    rl_u = np.asarray(payload["rl_u"], float)
    mpc_y = np.asarray(payload["mpc_y"], float)
    mpc_u = np.asarray(payload["mpc_u"], float)
    y_sp_phys = np.asarray(payload["y_sp_phys"], float)

    time_in_sub_episodes = rl_u.shape[0]
    t_line = np.linspace(0.0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)
    t_step = t_line[:-1]

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex="col")
    for idx in range(2):
        ax = axs[idx, 0]
        ax.plot(t_line, rl_y[:, idx], color="#C84C09", linewidth=2.0, label="Markov RL")
        ax.plot(t_line, mpc_y[:, idx], color="#0B6E4F", linewidth=1.8, linestyle="--", label="Nominal MPC")
        ax.step(t_step, y_sp_phys[:, idx], where="post", color="0.35", linewidth=1.5, linestyle=":", label="Setpoint")
        ax.set_ylabel(output_labels[idx])
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

        ax_u = axs[idx, 1]
        ax_u.step(t_step, rl_u[:, idx], where="post", color="#C84C09", linewidth=2.0, label="Markov RL")
        ax_u.step(t_step, mpc_u[:, idx], where="post", color="#0B6E4F", linewidth=1.8, linestyle="--", label="Nominal MPC")
        ax_u.set_ylabel(input_labels[idx])
        ax_u.grid(alpha=0.25)
        ax_u.spines["top"].set_visible(False)
        ax_u.spines["right"].set_visible(False)
        if idx == 0:
            ax_u.legend(loc="best", fontsize=9)

    axs[1, 0].set_xlabel(time_label)
    axs[1, 1].set_xlabel(time_label)
    axs[0, 0].set_title("Final test episode outputs")
    axs[0, 1].set_title("Final test episode inputs")
    fig.tight_layout()
    out = out_dir / "tail_anchor_final_test_episode_vs_nominal_mpc.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _summary(latest: RunBundle, prev_z008: RunBundle, low_z: RunBundle, final_metrics: dict) -> dict:
    def metrics(run: RunBundle) -> dict:
        n_ep = run.n_episodes
        td3 = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 2)
        ls = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 3)
        nom = _episode_fraction(run.bundle["rl_action_source_log"], n_ep, 4)
        req = _episode_mean(run.bundle["requested_prediction_score_log"], n_ep)
        lss = _episode_mean(run.bundle["ls_prediction_score_log"], n_ep)
        exe = _episode_mean(run.bundle["executed_prediction_score_log"], n_ep)
        sat = _episode_fraction_raw_saturation(run.bundle["rl_requested_raw_action_log"], n_ep)
        req_z = _episode_mean_max_abs(run.bundle["rl_requested_z_log"], n_ep)
        ls_z = _episode_mean_max_abs(run.bundle["rl_ls_z_log"], n_ep)
        gap = _episode_vec_norm(
            np.asarray(run.bundle["rl_requested_z_log"], float) - np.asarray(run.bundle["rl_ls_z_log"], float),
            n_ep,
        )
        bc_active = _episode_mean(run.bundle["bc_active_log"], n_ep)
        bc_weight = _episode_mean(run.bundle["bc_weight_log"], n_ep)
        return {
            "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
            "z_bound": run.bundle.get("markov_z_bound"),
            "tail_anchor": dict(run.bundle.get("behavioral_cloning", {})).get("tail_anchor"),
            "avg_reward_mean": float(np.nanmean(run.avg_rewards)),
            "avg_reward_last10": float(np.nanmean(run.avg_rewards[-10:])),
            "avg_reward_final": float(run.avg_rewards[-1]),
            "td3_fraction_11_50": float(np.nanmean(td3[10:50])),
            "td3_fraction_last50": float(np.nanmean(td3[-50:])),
            "ls_fraction_last50": float(np.nanmean(ls[-50:])),
            "nominal_fraction_last50": float(np.nanmean(nom[-50:])),
            "requested_score_last50": float(np.nanmean(req[-50:])),
            "ls_score_last50": float(np.nanmean(lss[-50:])),
            "executed_score_last50": float(np.nanmean(exe[-50:])),
            "raw_saturation_last50": float(np.nanmean(sat[-50:])),
            "requested_zmax_last50": float(np.nanmean(req_z[-50:])),
            "ls_zmax_last50": float(np.nanmean(ls_z[-50:])),
            "z_gap_last50": float(np.nanmean(gap[-50:])),
            "bc_active_all": float(np.nanmean(np.asarray(run.bundle["bc_active_log"], float))),
            "bc_active_last50": float(np.nanmean(bc_active[-50:])),
            "bc_weight_last50": float(np.nanmean(bc_weight[-50:])),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_tail_run": metrics(latest),
        "previous_z008_run": metrics(prev_z008),
        "conditioned_low_z_run": metrics(low_z),
        "final_test_episode_vs_mpc": final_metrics,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = _load_run("Latest z = 0.08 + tail anchor", LATEST_TAIL_RUN)
    prev_z008 = _load_run("Previous z = 0.08", PREV_Z008_RUN)
    low_z = _load_run("Conditioned z = 0.05", LOW_Z_CONDITIONED_RUN)
    mpc_bundle = _load_pickle(MPC_BASELINE)

    fig1 = _plot_reward_and_td3_mix(latest, prev_z008, low_z, OUT_DIR)
    fig2 = _plot_tail_anchor_diagnostics(latest, prev_z008, OUT_DIR)
    final_metrics, final_payload = _final_test_episode_metrics(latest, mpc_bundle)
    fig3 = _plot_final_test_episode(latest, final_payload, OUT_DIR)

    summary = _summary(latest, prev_z008, low_z, final_metrics)
    summary["figure_paths"] = [
        str(fig1.relative_to(REPO_ROOT)),
        str(fig2.relative_to(REPO_ROOT)),
        str(fig3.relative_to(REPO_ROOT)),
    ]
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
