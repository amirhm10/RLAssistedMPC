from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUN_PATH = REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_td3_disturb_fluctuation_mismatch_unified" / "20260511_183650" / "input_data.pkl"
BASELINE_PATH = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_matrix_latest_followup_20260511"


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _scaled_setpoints_to_phys(bundle: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    n_inputs = int(np.asarray(bundle["u_step_full"], float).shape[1])
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_sp_scaled_dev = np.asarray(bundle["y_sp"], float)
    return (y_sp_scaled_dev + y_ss_scaled) * (data_max[n_inputs:] - data_min[n_inputs:]) + data_min[n_inputs:]


def _build_rows(run: dict, baseline: dict) -> list[dict]:
    T = int(run["time_in_sub_episodes"])
    S = T // 2
    N = int(np.asarray(run["avg_rewards"], float).size)
    y_sp_phys = _scaled_setpoints_to_phys(run)
    y_rl = np.asarray(run["y_line_full"], float)
    y_mpc = np.asarray(baseline["y"], float)
    u_rl = np.asarray(run["u_step_full"], float)
    u_mpc = np.asarray(baseline["u"], float)
    alpha = np.asarray(run["alpha_log"], float)

    rows: list[dict] = []
    for ep in range(N):
        base_idx = ep * T
        for half, name in [(0, "sp1"), (1, "sp2")]:
            a = base_idx + half * S
            b = a + S
            yr = y_rl[a + 1 : b + 1]
            ym = y_mpc[a + 1 : b + 1]
            sp = y_sp_phys[a:b]
            ur = u_rl[a:b]
            um = u_mpc[a:b]
            al = alpha[a:b]

            err_rl = yr - sp
            err_mpc = ym - sp
            rmse_rl = np.sqrt(np.mean(err_rl**2, axis=0))
            rmse_mpc = np.sqrt(np.mean(err_mpc**2, axis=0))
            iae_rl = np.mean(np.abs(err_rl), axis=0)
            iae_mpc = np.mean(np.abs(err_mpc), axis=0)
            jitter_rl = np.std(np.diff(yr, axis=0), axis=0)
            jitter_mpc = np.std(np.diff(ym, axis=0), axis=0)
            du_rl = np.diff(np.vstack([ur[:1], ur]), axis=0)
            du_mpc = np.diff(np.vstack([um[:1], um]), axis=0)
            move_rl = float(np.mean(np.linalg.norm(du_rl, axis=1)))
            move_mpc = float(np.mean(np.linalg.norm(du_mpc, axis=1)))
            alpha_tv = float(np.mean(np.abs(np.diff(al)))) if al.size > 1 else np.nan
            alpha_std = float(np.std(al))

            rows.append(
                {
                    "episode": ep + 1,
                    "setpoint_block": name,
                    "rmse_rl": rmse_rl,
                    "rmse_mpc": rmse_mpc,
                    "iae_rl": iae_rl,
                    "iae_mpc": iae_mpc,
                    "jitter_rl": jitter_rl,
                    "jitter_mpc": jitter_mpc,
                    "move_rl": move_rl,
                    "move_mpc": move_mpc,
                    "alpha_tv": alpha_tv,
                    "alpha_std": alpha_std,
                }
            )
    return rows


def _plot_reward_and_block_deltas(run: dict, baseline: dict, rows: list[dict]) -> Path:
    avg_rl = np.asarray(run["avg_rewards"], float)
    avg_mpc = np.asarray(baseline["avg_rewards_mpc"], float)
    N = avg_rl.size
    x = np.arange(1, N + 1)

    sp1 = [r for r in rows if r["setpoint_block"] == "sp1"]
    sp2 = [r for r in rows if r["setpoint_block"] == "sp2"]
    sp1_t_rmse_delta = np.asarray([r["rmse_rl"][1] - r["rmse_mpc"][1] for r in sp1], float)
    sp2_t_rmse_delta = np.asarray([r["rmse_rl"][1] - r["rmse_mpc"][1] for r in sp2], float)

    fig, axs = plt.subplots(2, 1, figsize=(12.8, 8.8), sharex=True)
    axs[0].plot(x, avg_rl, color="#C84C09", linewidth=2.0, label="TD3 matrix RL")
    axs[0].plot(x, avg_mpc, color="#1f77b4", linewidth=2.0, linestyle="--", label="Disturbance MPC")
    axs[0].set_ylabel("Avg. reward")
    axs[0].set_title("Latest distillation scalar-matrix run versus disturbance MPC")
    axs[0].legend(loc="best")

    axs[1].plot(x, sp1_t_rmse_delta, color="#D55E00", linewidth=1.8, label="Setpoint block 1: T85 RMSE delta")
    axs[1].plot(x, sp2_t_rmse_delta, color="#0B6E4F", linewidth=1.8, label="Setpoint block 2: T85 RMSE delta")
    axs[1].axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("RL - MPC")
    axs[1].legend(loc="best")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "distillation_matrix_latest_reward_and_block_deltas.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_last_episode_dashboard(run: dict, baseline: dict) -> Path:
    T = int(run["time_in_sub_episodes"])
    S = T // 2
    y_sp_phys = _scaled_setpoints_to_phys(run)
    y_rl = np.asarray(run["y_line_full"], float)
    y_mpc = np.asarray(baseline["y"], float)
    u_rl = np.asarray(run["u_step_full"], float)
    u_mpc = np.asarray(baseline["u"], float)
    alpha = np.asarray(run["alpha_log"], float)
    base_idx = (int(np.asarray(run["avg_rewards"]).size) - 1) * T

    x_y = np.arange(T + 1)
    x_u = np.arange(T)

    fig, axs = plt.subplots(3, 1, figsize=(12.8, 10.8), sharex=False)
    labels_y = ["x24 composition", "T85 temperature"]
    colors = ["#1f77b4", "#C84C09"]
    for idx in range(2):
        axs[0].plot(x_y, y_rl[base_idx : base_idx + T + 1, idx], color=colors[idx], linewidth=2.0, label=f"RL {labels_y[idx]}")
        axs[0].plot(x_y, y_mpc[base_idx : base_idx + T + 1, idx], color=colors[idx], linewidth=1.4, linestyle="--", label=f"MPC {labels_y[idx]}")
        axs[0].plot(x_u, y_sp_phys[base_idx : base_idx + T, idx], color=colors[idx], linewidth=1.0, linestyle=":", alpha=0.85)
    axs[0].axvline(S, color="0.35", linestyle="--", linewidth=1.1)
    axs[0].set_ylabel("Outputs")
    axs[0].set_title("Latest final test episode: output comparison by setpoint block")
    axs[0].legend(loc="best", ncol=2, fontsize=9)

    axs[1].plot(x_u, np.linalg.norm(np.diff(np.vstack([u_rl[base_idx : base_idx + 1], u_rl[base_idx : base_idx + T]]), axis=0), axis=1), color="#C84C09", linewidth=1.8, label="RL ||Δu||")
    axs[1].plot(x_u, np.linalg.norm(np.diff(np.vstack([u_mpc[base_idx : base_idx + 1], u_mpc[base_idx : base_idx + T]]), axis=0), axis=1), color="#1f77b4", linewidth=1.8, linestyle="--", label="MPC ||Δu||")
    axs[1].axvline(S, color="0.35", linestyle="--", linewidth=1.1)
    axs[1].set_ylabel("Move norm")
    axs[1].legend(loc="best")

    axs[2].plot(x_u, alpha[base_idx : base_idx + T], color="#7A1FA2", linewidth=1.8, label="Scalar alpha")
    axs[2].axvline(S, color="0.35", linestyle="--", linewidth=1.1)
    axs[2].set_ylabel("Alpha")
    axs[2].set_xlabel("Step within final episode")
    axs[2].legend(loc="best")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "distillation_matrix_latest_last_episode_dashboard.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_tail_block_summary(rows: list[dict]) -> Path:
    tail_sp1 = [r for r in rows if r["setpoint_block"] == "sp1" and r["episode"] > 180]
    tail_sp2 = [r for r in rows if r["setpoint_block"] == "sp2" and r["episode"] > 180]

    def summarize(block_rows: list[dict]) -> dict:
        return {
            "t85_rmse_delta": float(np.mean([r["rmse_rl"][1] - r["rmse_mpc"][1] for r in block_rows])),
            "x24_rmse_delta": float(np.mean([r["rmse_rl"][0] - r["rmse_mpc"][0] for r in block_rows])),
            "t85_jitter_delta": float(np.mean([r["jitter_rl"][1] - r["jitter_mpc"][1] for r in block_rows])),
            "move_delta": float(np.mean([r["move_rl"] - r["move_mpc"] for r in block_rows])),
            "alpha_tv": float(np.mean([r["alpha_tv"] for r in block_rows])),
            "alpha_std": float(np.mean([r["alpha_std"] for r in block_rows])),
        }

    s1 = summarize(tail_sp1)
    s2 = summarize(tail_sp2)
    labels = ["T85 RMSE Δ", "x24 RMSE Δ", "T85 jitter Δ", "Move Δ", "alpha TV", "alpha std"]
    vals1 = [s1["t85_rmse_delta"], s1["x24_rmse_delta"], s1["t85_jitter_delta"], s1["move_delta"], s1["alpha_tv"], s1["alpha_std"]]
    vals2 = [s2["t85_rmse_delta"], s2["x24_rmse_delta"], s2["t85_jitter_delta"], s2["move_delta"], s2["alpha_tv"], s2["alpha_std"]]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    ax.bar(x - w / 2, vals1, width=w, color="#D55E00", label="Setpoint block 1")
    ax.bar(x + w / 2, vals2, width=w, color="#0B6E4F", label="Setpoint block 2")
    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean tail metric")
    ax.set_title("Tail-20 block summary versus disturbance MPC")
    ax.legend(loc="best")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "distillation_matrix_latest_tail_block_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run = _load_pickle(RUN_PATH)
    baseline = _load_pickle(BASELINE_PATH)
    rows = _build_rows(run, baseline)

    fig1 = _plot_reward_and_block_deltas(run, baseline, rows)
    fig2 = _plot_last_episode_dashboard(run, baseline)
    fig3 = _plot_tail_block_summary(rows)

    tail_sp1 = [r for r in rows if r["setpoint_block"] == "sp1" and r["episode"] > 180]
    tail_sp2 = [r for r in rows if r["setpoint_block"] == "sp2" and r["episode"] > 180]
    last_sp1 = rows[-2]
    last_sp2 = rows[-1]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(RUN_PATH.parent.relative_to(REPO_ROOT)),
        "baseline_path": str(BASELINE_PATH.relative_to(REPO_ROOT)),
        "test_episode_index": int(rows[-1]["episode"]),
        "bundle_mpc_is_rl_duplicate": bool(
            np.array_equal(np.asarray(run["y_line_full"]), np.asarray(run["y_mpc"]))
            and np.array_equal(np.asarray(run["u_step_full"]), np.asarray(run["u_mpc"]))
        ),
        "reward_summary": {
            "mean_reward_rl": float(np.mean(np.asarray(run["avg_rewards"], float))),
            "mean_reward_mpc": float(np.mean(np.asarray(baseline["avg_rewards_mpc"], float))),
            "last10_reward_rl": float(np.mean(np.asarray(run["avg_rewards"], float)[-10:])),
            "last10_reward_mpc": float(np.mean(np.asarray(baseline["avg_rewards_mpc"], float)[-10:])),
            "last_episode_reward_rl": float(np.asarray(run["avg_rewards"], float)[-1]),
            "last_episode_reward_mpc": float(np.asarray(baseline["avg_rewards_mpc"], float)[-1]),
        },
        "last_episode_sp1": {
            "rmse_rl": np.asarray(last_sp1["rmse_rl"], float).tolist(),
            "rmse_mpc": np.asarray(last_sp1["rmse_mpc"], float).tolist(),
            "iae_rl": np.asarray(last_sp1["iae_rl"], float).tolist(),
            "iae_mpc": np.asarray(last_sp1["iae_mpc"], float).tolist(),
            "jitter_rl": np.asarray(last_sp1["jitter_rl"], float).tolist(),
            "jitter_mpc": np.asarray(last_sp1["jitter_mpc"], float).tolist(),
            "move_rl": float(last_sp1["move_rl"]),
            "move_mpc": float(last_sp1["move_mpc"]),
            "alpha_tv": float(last_sp1["alpha_tv"]),
            "alpha_std": float(last_sp1["alpha_std"]),
        },
        "last_episode_sp2": {
            "rmse_rl": np.asarray(last_sp2["rmse_rl"], float).tolist(),
            "rmse_mpc": np.asarray(last_sp2["rmse_mpc"], float).tolist(),
            "iae_rl": np.asarray(last_sp2["iae_rl"], float).tolist(),
            "iae_mpc": np.asarray(last_sp2["iae_mpc"], float).tolist(),
            "jitter_rl": np.asarray(last_sp2["jitter_rl"], float).tolist(),
            "jitter_mpc": np.asarray(last_sp2["jitter_mpc"], float).tolist(),
            "move_rl": float(last_sp2["move_rl"]),
            "move_mpc": float(last_sp2["move_mpc"]),
            "alpha_tv": float(last_sp2["alpha_tv"]),
            "alpha_std": float(last_sp2["alpha_std"]),
        },
        "tail20_block_means": {
            "sp1_t85_rmse_delta": float(np.mean([r["rmse_rl"][1] - r["rmse_mpc"][1] for r in tail_sp1])),
            "sp2_t85_rmse_delta": float(np.mean([r["rmse_rl"][1] - r["rmse_mpc"][1] for r in tail_sp2])),
            "sp1_move_delta": float(np.mean([r["move_rl"] - r["move_mpc"] for r in tail_sp1])),
            "sp2_move_delta": float(np.mean([r["move_rl"] - r["move_mpc"] for r in tail_sp2])),
            "sp1_alpha_tv": float(np.mean([r["alpha_tv"] for r in tail_sp1])),
            "sp2_alpha_tv": float(np.mean([r["alpha_tv"] for r in tail_sp2])),
        },
        "figure_paths": [
            str(fig1.relative_to(REPO_ROOT)),
            str(fig2.relative_to(REPO_ROOT)),
            str(fig3.relative_to(REPO_ROOT)),
        ],
    }

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
