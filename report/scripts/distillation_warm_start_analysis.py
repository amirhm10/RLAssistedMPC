from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation.notebook_params import get_distillation_notebook_defaults  # noqa: E402
from utils.helpers import generate_setpoints_training_rl_gradually  # noqa: E402


REPORT_DIR = REPO_ROOT / "report"
FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, float)
    if values.size == 0:
        return values.copy()
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def build_schedule(family: str) -> dict:
    defaults = get_distillation_notebook_defaults(family)
    agent_kind = str(defaults["agent_kind"]).lower()
    run_mode = str(defaults["run_mode"]).lower()
    disturbance_profile = str(defaults["disturbance_profile"]).lower()
    run_profile = defaults["run_profiles"][(agent_kind, run_mode, disturbance_profile)]
    y_sp_scenario = np.asarray(defaults["system_setup"]["rl_setpoints_phys"], float)

    (
        _y_sp,
        nFE,
        _sub_episodes_changes_dict,
        time_in_sub_episodes,
        test_train_dict,
        warm_start_step,
        _qi,
        _qs,
        _ha,
    ) = generate_setpoints_training_rl_gradually(
        y_sp_scenario,
        int(run_profile["n_tests"]),
        int(run_profile["set_points_len"]),
        int(run_profile["warm_start"]),
        list(run_profile["test_cycle"]),
        float(defaults["controller"]["nominal_qi"]),
        float(defaults["controller"]["nominal_qs"]),
        float(defaults["controller"]["nominal_ha"]),
        float(defaults["controller"]["qi_change"]),
        float(defaults["controller"]["qs_change"]),
        float(defaults["controller"]["ha_change"]),
    )

    td3_cfg = defaults["td3_agent"]
    policy_delay = int(td3_cfg["policy_delay"])
    actor_freeze = int(td3_cfg["actor_freeze"])

    warm_start_fraction = []
    train_fraction = []
    eval_fraction = []
    warm_start_share = []
    critic_updates = []
    actor_updates = []
    episode_ids = np.arange(1, int(run_profile["n_tests"]) + 1)

    total_pushed = 0
    warm_start_pushed = 0
    cumulative_train_steps = 0
    cumulative_actor_updates = 0
    test = False

    for episode_idx in range(int(run_profile["n_tests"])):
        start = episode_idx * time_in_sub_episodes
        end = start + time_in_sub_episodes
        warm_steps = 0
        train_steps = 0
        eval_steps = 0
        pushed_this_episode = 0
        warm_pushed_this_episode = 0

        for i in range(start, end):
            if i in test_train_dict:
                test = bool(test_train_dict[i])

            if i > warm_start_step:
                if test:
                    eval_steps += 1
            else:
                warm_steps += 1

            if (not test) and i >= warm_start_step:
                train_steps += 1

            if not test:
                pushed_this_episode += 1
                if i <= warm_start_step:
                    warm_pushed_this_episode += 1

        total_pushed += pushed_this_episode
        warm_start_pushed += warm_pushed_this_episode
        total_prev = cumulative_train_steps
        cumulative_train_steps += train_steps

        for total_it in range(total_prev, cumulative_train_steps):
            if total_it % policy_delay == 0 and total_it >= actor_freeze:
                cumulative_actor_updates += 1

        warm_start_fraction.append(warm_steps / time_in_sub_episodes)
        train_fraction.append(train_steps / time_in_sub_episodes)
        eval_fraction.append(eval_steps / time_in_sub_episodes)
        warm_start_share.append(warm_start_pushed / max(total_pushed, 1))
        critic_updates.append(cumulative_train_steps)
        actor_updates.append(cumulative_actor_updates)

    return {
        "family": family,
        "episode_ids": episode_ids,
        "nFE": int(nFE),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "warm_start_step": int(warm_start_step),
        "warm_start_fraction": np.asarray(warm_start_fraction, float),
        "train_fraction": np.asarray(train_fraction, float),
        "eval_fraction": np.asarray(eval_fraction, float),
        "warm_start_share": np.asarray(warm_start_share, float),
        "critic_updates": np.asarray(critic_updates, int),
        "actor_updates": np.asarray(actor_updates, int),
        "policy_delay": policy_delay,
        "actor_freeze": actor_freeze,
        "batch_size": int(td3_cfg["batch_size"]),
        "first_train_step": int(warm_start_step),
        "first_learned_action_step": int(warm_start_step + 1),
        "final_eval_step": int((int(run_profile["n_tests"]) - 1) * time_in_sub_episodes),
    }


def load_latest_residual_bundle() -> tuple[Path, dict]:
    result_root = REPO_ROOT / "Distillation" / "Results" / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
    run_dirs = sorted([path for path in result_root.iterdir() if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No saved residual runs found under {result_root}")
    latest = run_dirs[-1]
    bundle_path = latest / "input_data.pkl"
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    return bundle_path, bundle


def make_schedule_figure(matrix_schedule: dict, residual_schedule: dict) -> Path:
    episodes = matrix_schedule["episode_ids"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.step(episodes, matrix_schedule["warm_start_fraction"], where="mid", label="Warm-start action fraction", color="#1f77b4", linewidth=2.0)
    ax.step(episodes, 1.0 - matrix_schedule["warm_start_fraction"] - matrix_schedule["eval_fraction"], where="mid", label="RL exploration action fraction", color="#ff7f0e", linewidth=2.0)
    ax.step(episodes, matrix_schedule["eval_fraction"], where="mid", label="Evaluation-only fraction", color="#2ca02c", linewidth=2.0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Step Fraction")
    ax.set_title("Distillation Matrix/Residual Warm-Start Schedule Derived From Code")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.axvline(10.5, color="black", linestyle="--", linewidth=1.2)
    ax.text(
        11.5,
        0.20,
        "Episode 11 is the boundary episode:\ntrain starts at step 4000,\nfirst learned action at step 4001.",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    ax = axes[1]
    ax.step(episodes, matrix_schedule["train_fraction"], where="mid", color="#d62728", linewidth=2.0, label="train_step fraction")
    ax_twin = ax.twinx()
    ax_twin.plot(episodes, matrix_schedule["critic_updates"], color="#9467bd", linewidth=1.8, label="Cumulative critic updates")
    ax_twin.plot(episodes, matrix_schedule["actor_updates"], color="#8c564b", linewidth=1.8, label="Cumulative actor updates")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("train_step Fraction")
    ax_twin.set_ylabel("Cumulative Updates")
    ax.grid(alpha=0.25)
    ax.axvline(10.5, color="black", linestyle="--", linewidth=1.2)
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_twin.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", ncol=3, fontsize=9)

    ax = axes[2]
    ax.plot(episodes, matrix_schedule["warm_start_share"], color="#17becf", linewidth=2.0)
    ax.axvline(10.5, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Warm-Start Share")
    ax.set_xlabel("Sub-Episode")
    ax.grid(alpha=0.25)
    ax.set_title("Share of Warm-Start Transitions in the Accumulated Replay Dataset")

    figure_path = FIGURE_DIR / "distillation_warm_start_schedule.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def make_residual_bundle_figure(bundle_path: Path, bundle: dict) -> Path:
    avg_rewards = np.asarray(bundle["avg_rewards"], float)
    steps_per_episode = int(bundle["time_in_sub_episodes"])
    delta_u_res_exec_log = np.asarray(bundle["delta_u_res_exec_log"], float)
    residual_norm_per_episode = np.linalg.norm(delta_u_res_exec_log, axis=1).reshape(-1, steps_per_episode).mean(axis=1)
    episodes = np.arange(1, avg_rewards.size + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(episodes, avg_rewards, color="#1f77b4", alpha=0.35, linewidth=1.2, label="Episode reward")
    ax.plot(episodes, moving_average(avg_rewards, 5), color="#1f77b4", linewidth=2.2, label="5-episode moving average")
    ax.axvspan(0.5, 10.5, color="#d9eaf7", alpha=0.7, label="Warm-start episodes")
    ax.axvspan(avg_rewards.size - 0.5, avg_rewards.size + 0.5, color="#d8f0d2", alpha=0.7, label="Forced test-only final episode")
    ax.axvline(10.5, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Average Reward")
    ax.set_title(f"Saved Distillation Residual TD3 Run: {bundle_path.parent.name}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=9)

    ax = axes[1]
    ax.plot(episodes, residual_norm_per_episode, color="#d62728", alpha=0.35, linewidth=1.2, label="Mean executed residual norm")
    ax.plot(episodes, moving_average(residual_norm_per_episode, 5), color="#d62728", linewidth=2.2, label="5-episode moving average")
    ax.axvspan(0.5, 10.5, color="#d9eaf7", alpha=0.7)
    ax.axvspan(avg_rewards.size - 0.5, avg_rewards.size + 0.5, color="#d8f0d2", alpha=0.7)
    ax.axvline(10.5, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Mean ||Residual||")
    ax.set_xlabel("Sub-Episode")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    figure_path = FIGURE_DIR / "distillation_residual_warm_start_effects.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def make_literature_support_figure() -> Path:
    papers = [
        "Vecerik17\nDDPGfD",
        "Nair18\nDemo+HER",
        "AWAC21",
        "TD3+BC21",
        "CQL20",
        "Cetin24",
    ]
    mechanisms = [
        "Replay seed /\ndemos",
        "Offline→online\nhandoff",
        "Actor\nregularization",
        "Conservative\ncritic",
        "Direct support for\nwarm-start idea",
    ]
    # 0 = none, 1 = partial, 2 = strong
    data = np.asarray(
        [
            [2, 0, 1, 0, 2],  # Vecerik17
            [2, 1, 0, 0, 2],  # Nair18
            [1, 2, 2, 0, 2],  # AWAC
            [0, 1, 2, 0, 1],  # TD3+BC
            [0, 1, 0, 2, 1],  # CQL
            [0, 2, 1, 0, 1],  # Cetin24
        ],
        float,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=2.0, aspect="auto")
    ax.set_xticks(np.arange(len(mechanisms)))
    ax.set_xticklabels(mechanisms, fontsize=10)
    ax.set_yticks(np.arange(len(papers)))
    ax.set_yticklabels(papers, fontsize=10)
    ax.set_title("Literature Support Map For Warm-Start Training Variants")

    labels = {0.0: "none", 1.0: "partial", 2.0: "strong"}
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            ax.text(col, row, labels[value], ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_ticks([0.0, 1.0, 2.0])
    cbar.set_ticklabels(["none", "partial", "strong"])

    figure_path = FIGURE_DIR / "distillation_warm_start_literature_support.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def make_method_tradeoff_figure() -> Path:
    methods = [
        "Current\nTD3",
        "Critic-only\nrelease",
        "TD3+BC\nwarm start",
        "AWAC-style\nhandoff",
        "CQL-style\ncritic",
    ]
    degradation_resistance = np.asarray([1.0, 2.0, 3.2, 3.8, 4.2])
    implementation_cost = np.asarray([0.8, 1.2, 2.0, 2.8, 3.4])
    policy_freedom = np.asarray([4.5, 3.8, 3.0, 2.8, 2.2])

    x = np.arange(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.bar(x - width, degradation_resistance, width=width, label="Expected degradation resistance", color="#1f77b4")
    ax.bar(x, implementation_cost, width=width, label="Implementation cost", color="#ff7f0e")
    ax.bar(x + width, policy_freedom, width=width, label="Policy freedom during handoff", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Relative score (higher = more)")
    ax.set_title("Next-Step Method Tradeoffs For This Repo")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncol=3, fontsize=9)

    figure_path = FIGURE_DIR / "distillation_warm_start_method_tradeoffs.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def make_phase1_variant_figure(matrix_schedule: dict, freeze_subepisodes: int = 5) -> Path:
    episodes = matrix_schedule["episode_ids"]
    current_live_action = 1.0 - matrix_schedule["warm_start_fraction"] - matrix_schedule["eval_fraction"]
    current_critic_updates = matrix_schedule["critic_updates"].astype(float)
    current_actor_updates = matrix_schedule["actor_updates"].astype(float)

    warm_boundary_episode = int(np.flatnonzero(matrix_schedule["train_fraction"] > 0.0)[0]) + 1
    freeze_start = warm_boundary_episode
    freeze_end = warm_boundary_episode + int(freeze_subepisodes) - 1
    hidden_steps = int(freeze_subepisodes) * int(matrix_schedule["time_in_sub_episodes"])

    hidden_live_action = current_live_action.copy()
    hidden_live_action[freeze_start - 1 : freeze_end] = 0.0

    critic_only_actor_updates = np.zeros_like(current_actor_updates)
    for idx, critic_count in enumerate(current_critic_updates):
        if critic_count <= hidden_steps:
            critic_only_actor_updates[idx] = 0.0
        else:
            effective_train_steps = critic_count - hidden_steps
            critic_only_actor_updates[idx] = 1.0 + np.floor(
                (effective_train_steps - 1.0) / float(matrix_schedule["policy_delay"])
            )

    actor_hidden_actor_updates = current_actor_updates.copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.step(episodes, current_live_action, where="mid", linewidth=2.0, color="#1f77b4", label="Current code")
    ax.step(episodes, hidden_live_action, where="mid", linewidth=2.0, color="#ff7f0e", label="5-episode hidden-action variants")
    ax.axvspan(freeze_start - 0.5, freeze_end + 0.5, color="#f8e3c5", alpha=0.65, label="Hidden-action window")
    ax.set_ylabel("Executed RL Action Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Phase 1 Schedule Comparison For A 5-Episode Hidden-Action Window")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=9)

    ax = axes[1]
    ax.plot(episodes, current_critic_updates, linewidth=2.0, color="#9467bd", label="Cumulative critic updates")
    ax.plot(episodes, actor_hidden_actor_updates, linewidth=2.0, color="#d62728", label="If actor trains during hidden window")
    ax.plot(
        episodes,
        critic_only_actor_updates,
        linewidth=2.0,
        color="#2ca02c",
        label="If actor remains frozen during hidden window",
    )
    ax.axvspan(freeze_start - 0.5, freeze_end + 0.5, color="#f8e3c5", alpha=0.65)
    ax.set_ylabel("Cumulative Updates")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    ax = axes[2]
    actor_hidden_gap = actor_hidden_actor_updates - critic_only_actor_updates
    ax.bar(episodes, actor_hidden_gap, color="#8c564b", width=0.75)
    ax.axvspan(freeze_start - 0.5, freeze_end + 0.5, color="#f8e3c5", alpha=0.65)
    ax.set_ylabel("Extra Hidden Actor Updates")
    ax.set_xlabel("Sub-Episode")
    ax.grid(alpha=0.25)
    ax.set_title("Actor drift risk before release if actions are frozen but actor optimization still runs")
    ax.text(
        freeze_end + 0.7,
        max(float(actor_hidden_gap.max()) * 0.55, 1.0),
        "These updates happen before any learned\naction is deployed to the plant.",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    figure_path = FIGURE_DIR / "distillation_phase1_variant_schedule.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> None:
    matrix_schedule = build_schedule("matrix")
    residual_schedule = build_schedule("residual")
    bundle_path, bundle = load_latest_residual_bundle()

    schedule_figure = make_schedule_figure(matrix_schedule, residual_schedule)
    residual_figure = make_residual_bundle_figure(bundle_path, bundle)
    literature_figure = make_literature_support_figure()
    tradeoff_figure = make_method_tradeoff_figure()
    phase1_variant_figure = make_phase1_variant_figure(matrix_schedule, freeze_subepisodes=5)

    avg_rewards = np.asarray(bundle["avg_rewards"], float)
    steps_per_episode = int(bundle["time_in_sub_episodes"])
    residual_norm_per_episode = np.linalg.norm(np.asarray(bundle["delta_u_res_exec_log"], float), axis=1).reshape(-1, steps_per_episode).mean(axis=1)

    print("Generated figures:")
    print(schedule_figure)
    print(residual_figure)
    print(literature_figure)
    print(tradeoff_figure)
    print(phase1_variant_figure)
    print()
    print("Matrix/residual schedule summary:")
    print(f"  warm_start_step={matrix_schedule['warm_start_step']}")
    print(f"  first_train_step={matrix_schedule['first_train_step']}")
    print(f"  first_learned_action_step={matrix_schedule['first_learned_action_step']}")
    print(f"  final_eval_step={matrix_schedule['final_eval_step']}")
    print(f"  expected_critic_updates={matrix_schedule['critic_updates'][-1]}")
    print(f"  expected_actor_updates={matrix_schedule['actor_updates'][-1]}")
    print()
    print("Residual saved bundle summary:")
    print(f"  bundle_path={bundle_path}")
    print(f"  critic_losses_len={len(bundle['critic_losses'])}")
    print(f"  actor_losses_len={len(bundle['actor_losses'])}")
    print(f"  avg_reward_episodes_1_10={avg_rewards[:10].mean():.6f}")
    print(f"  avg_reward_episodes_11_20={avg_rewards[10:20].mean():.6f}")
    print(f"  avg_reward_episodes_101_199={avg_rewards[100:199].mean():.6f}")
    print(f"  residual_norm_episodes_1_10={residual_norm_per_episode[:10].mean():.9f}")
    print(f"  residual_norm_episodes_11_20={residual_norm_per_episode[10:20].mean():.9f}")


if __name__ == "__main__":
    main()
