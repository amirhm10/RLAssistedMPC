from __future__ import annotations

import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "Polymer" / "Results"
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_zbound_20260609"
RL_FAMILY = "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch"
COMPARE_FAMILY = "disturb_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow_mismatch"


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def latest_rl_runs_by_bound() -> dict[float, Path]:
    runs: dict[float, Path] = {}
    for path in sorted((RESULTS_ROOT / RL_FAMILY).glob("*/input_data.pkl")):
        data = load_pickle(path)
        z_bound = data.get("markov_z_bound")
        if z_bound is None:
            continue
        z_bound = round(float(z_bound), 6)
        current = runs.get(z_bound)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            runs[z_bound] = path
    return dict(sorted(runs.items()))


def compare_runs_by_rl_dir() -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    for path in sorted((RESULTS_ROOT / COMPARE_FAMILY).glob("*/input_data.pkl")):
        data = load_pickle(path)
        rl_dir = data.get("rl_dir")
        if not rl_dir:
            continue
        mapping[Path(rl_dir).resolve()] = path
    return mapping


def fraction_equal(values: np.ndarray, code: int) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values == code))


def binary_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values.astype(bool)))


def window_stats(values: np.ndarray, sl: slice) -> tuple[float, float, float]:
    selected = values[sl]
    if selected.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(selected)), float(np.nanquantile(selected, 0.95)), float(np.nanmax(selected))


def summarize_run(z_bound: float, rl_path: Path, compare_path: Path | None) -> dict[str, object]:
    data = load_pickle(rl_path)
    compare = load_pickle(compare_path) if compare_path else {}

    rewards_rl = np.asarray(compare.get("avg_rewards_rl", data.get("avg_rewards", [])), float).reshape(-1)
    rewards_mpc = np.asarray(compare.get("avg_rewards_mpc", []), float).reshape(-1)
    z = np.asarray(data.get("z_executed_log", []), float)
    steps_per_episode = int(z.shape[0] // rewards_rl.size) if rewards_rl.size and z.size else 0
    warm_start_step = int(data.get("warm_start_step", steps_per_episode * 10))
    warm_episode = int(np.ceil(warm_start_step / steps_per_episode)) if steps_per_episode else 10

    episode_tail = slice(max(0, rewards_rl.size - 20), rewards_rl.size)
    episode_post = slice(warm_episode, rewards_rl.size)
    episode_first20 = slice(warm_episode, min(rewards_rl.size, warm_episode + 20))
    step_post = slice(warm_start_step, None)
    step_tail = slice(max(0, z.shape[0] - 20 * steps_per_episode), z.shape[0]) if steps_per_episode else slice(0, 0)

    z_norm = np.linalg.norm(z, axis=1) if z.size else np.asarray([], float)
    z_coord_cap = np.any(np.abs(z) >= 0.95 * z_bound, axis=1) if z.size else np.asarray([], bool)
    sg_sources = np.asarray(data.get("sg_selected_source_log", []), int).reshape(-1)
    action_sources = np.asarray(data.get("rl_action_source_log", []), int).reshape(-1)
    shadow_project = np.asarray(data.get("shadow_z_safety_requested_projection_active_log", []), int).reshape(-1)
    shadow_coord_clip = np.asarray(data.get("shadow_z_safety_requested_coord_clip_active_log", []), int).reshape(-1)
    sg_advantage = np.asarray(data.get("sg_advantage_log", []), float).reshape(-1)

    post_mean, post_q95, post_max = window_stats(z_norm, step_post)
    tail_mean, tail_q95, tail_max = window_stats(z_norm, step_tail)

    row: dict[str, object] = {
        "z_bound": z_bound,
        "rl_run": str(rl_path.parent.relative_to(REPO_ROOT)),
        "compare_run": "" if compare_path is None else str(compare_path.parent.relative_to(REPO_ROOT)),
        "episodes": int(rewards_rl.size),
        "steps_per_episode": steps_per_episode,
        "warm_episode": warm_episode,
        "reward_mean_rl": float(np.nanmean(rewards_rl)),
        "reward_tail20_rl": float(np.nanmean(rewards_rl[episode_tail])),
        "reward_final_rl": float(rewards_rl[-1]),
        "reward_worst_postwarm_rl": float(np.nanmin(rewards_rl[episode_post])),
        "reward_worst_first20_postwarm_rl": float(np.nanmin(rewards_rl[episode_first20])),
        "z_norm_post_mean": post_mean,
        "z_norm_post_q95": post_q95,
        "z_norm_post_max": post_max,
        "z_norm_tail_mean": tail_mean,
        "z_norm_tail_q95": tail_q95,
        "z_norm_tail_max": tail_max,
        "z_coord_cap_post_frac": binary_fraction(z_coord_cap[step_post]),
        "z_coord_cap_tail_frac": binary_fraction(z_coord_cap[step_tail]),
        "sg_policy_post_frac": fraction_equal(sg_sources[step_post], 2),
        "sg_policy_tail_frac": fraction_equal(sg_sources[step_tail], 2),
        "sg_supervisor_post_frac": fraction_equal(sg_sources[step_post], 1),
        "sg_supervisor_tail_frac": fraction_equal(sg_sources[step_tail], 1),
        "source_td3_post_frac": fraction_equal(action_sources[step_post], 2),
        "source_td3_tail_frac": fraction_equal(action_sources[step_tail], 2),
        "source_sg_ls_post_frac": fraction_equal(action_sources[step_post], 6),
        "source_sg_ls_tail_frac": fraction_equal(action_sources[step_tail], 6),
        "source_sg_mpc_post_frac": fraction_equal(action_sources[step_post], 7),
        "source_sg_mpc_tail_frac": fraction_equal(action_sources[step_tail], 7),
        "shadow_projection_post_frac": binary_fraction(shadow_project[step_post]),
        "shadow_projection_tail_frac": binary_fraction(shadow_project[step_tail]),
        "shadow_coord_clip_post_frac": binary_fraction(shadow_coord_clip[step_post]),
        "shadow_coord_clip_tail_frac": binary_fraction(shadow_coord_clip[step_tail]),
        "sg_advantage_post_mean": float(np.nanmean(sg_advantage[step_post])) if sg_advantage.size else float("nan"),
        "sg_advantage_tail_mean": float(np.nanmean(sg_advantage[step_tail])) if sg_advantage.size else float("nan"),
    }

    if rewards_mpc.size:
        row.update(
            {
                "reward_mean_mpc": float(np.nanmean(rewards_mpc)),
                "reward_tail20_mpc": float(np.nanmean(rewards_mpc[episode_tail])),
                "reward_final_mpc": float(rewards_mpc[-1]),
                "reward_worst_postwarm_mpc": float(np.nanmin(rewards_mpc[episode_post])),
                "reward_tail20_delta": float(np.nanmean(rewards_rl[episode_tail]) - np.nanmean(rewards_mpc[episode_tail])),
                "reward_final_delta": float(rewards_rl[-1] - rewards_mpc[-1]),
            }
        )
    else:
        row.update(
            {
                "reward_mean_mpc": float("nan"),
                "reward_tail20_mpc": float("nan"),
                "reward_final_mpc": float("nan"),
                "reward_worst_postwarm_mpc": float("nan"),
                "reward_tail20_delta": float("nan"),
                "reward_final_delta": float("nan"),
            }
        )
    return row


def write_csv(rows: list[dict[str, object]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "polymer_markov_zbound_summary.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_summary(rows: list[dict[str, object]]) -> list[Path]:
    z = np.asarray([row["z_bound"] for row in rows], float)
    tail_delta = np.asarray([row["reward_tail20_delta"] for row in rows], float)
    worst_post = np.asarray([row["reward_worst_postwarm_rl"] for row in rows], float)
    tail_q95 = np.asarray([row["z_norm_tail_q95"] for row in rows], float)
    tail_max = np.asarray([row["z_norm_tail_max"] for row in rows], float)
    vector_cap = 2.0 * z
    policy_tail = np.asarray([row["sg_policy_tail_frac"] for row in rows], float)
    cap_tail = np.asarray([row["z_coord_cap_tail_frac"] for row in rows], float)
    shadow_tail = np.asarray([row["shadow_projection_tail_frac"] for row in rows], float)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    axs[0].plot(z, tail_delta, marker="o", linewidth=2, color="#2166ac")
    axs[0].axhline(0.0, color="black", linewidth=0.8)
    axs[0].set_title("Tail reward gain")
    axs[0].set_xlabel("z_bound")
    axs[0].set_ylabel("RL - OF-MPC tail reward")
    axs[0].grid(alpha=0.25)

    axs[1].plot(z, worst_post, marker="o", linewidth=2, color="#b2182b")
    axs[1].set_title("Post-warm downside")
    axs[1].set_xlabel("z_bound")
    axs[1].set_ylabel("Worst RL episode reward")
    axs[1].grid(alpha=0.25)

    axs[2].plot(z, tail_q95, marker="o", linewidth=2, label="tail q95 norm", color="#4d9221")
    axs[2].plot(z, tail_max, marker="s", linewidth=1.5, label="tail max norm", color="#7f3b08")
    axs[2].plot(z, vector_cap, linestyle="--", linewidth=1.3, label="vector cap", color="#666666")
    axs[2].set_title("Executed z usage")
    axs[2].set_xlabel("z_bound")
    axs[2].set_ylabel("z 2-norm")
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.25)

    path1 = OUT_DIR / "polymer_markov_zbound_reward_and_usage.png"
    fig.savefig(path1, dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    axs[0].plot(z, policy_tail, marker="o", linewidth=2, color="#2166ac")
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_title("Tail policy authority")
    axs[0].set_xlabel("z_bound")
    axs[0].set_ylabel("SG policy fraction")
    axs[0].grid(alpha=0.25)

    axs[1].plot(z, cap_tail, marker="o", linewidth=2, color="#7f3b08")
    axs[1].set_ylim(0.0, 1.0)
    axs[1].set_title("Tail cap contact")
    axs[1].set_xlabel("z_bound")
    axs[1].set_ylabel("fraction near coord cap")
    axs[1].grid(alpha=0.25)

    axs[2].plot(z, shadow_tail, marker="o", linewidth=2, color="#b2182b")
    axs[2].set_ylim(0.0, 1.0)
    axs[2].set_title("Shadow safety pressure")
    axs[2].set_xlabel("z_bound")
    axs[2].set_ylabel("tail projection fraction")
    axs[2].grid(alpha=0.25)

    path2 = OUT_DIR / "polymer_markov_zbound_source_and_safety.png"
    fig.savefig(path2, dpi=180)
    plt.close(fig)
    return [path1, path2]


def main() -> None:
    rl_runs = latest_rl_runs_by_bound()
    compare_map = compare_runs_by_rl_dir()
    rows = []
    for z_bound, rl_path in rl_runs.items():
        rows.append(summarize_run(z_bound, rl_path, compare_map.get(rl_path.parent.resolve())))

    csv_path = write_csv(rows)
    figure_paths = plot_summary(rows)

    print("Wrote", csv_path.relative_to(REPO_ROOT))
    for path in figure_paths:
        print("Wrote", path.relative_to(REPO_ROOT))
    print()
    for row in rows:
        print(
            f"z={row['z_bound']:.2f} tail_delta={row['reward_tail20_delta']:.4f} "
            f"tail_rl={row['reward_tail20_rl']:.4f} worst_post={row['reward_worst_postwarm_rl']:.4f} "
            f"tail_z_q95={row['z_norm_tail_q95']:.4f} policy_tail={row['sg_policy_tail_frac']:.4f} "
            f"shadow_tail={row['shadow_projection_tail_frac']:.4f}"
        )


if __name__ == "__main__":
    main()
