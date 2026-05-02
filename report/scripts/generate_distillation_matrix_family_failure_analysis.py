from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_matrix_family_failure_analysis_20260501"
SUMMARY_CSV = FIG_DIR / "distillation_matrix_run_summary.csv"

N_EPISODES = 200
DISTURB_BASELINE_PATH = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    short_label: str
    family: str
    structure: str
    agent: str
    run_mode: str
    state_mode: str
    rl_path: Path
    compare_path: Path
    baseline_path: Path | None = None


RUNS = [
    RunSpec(
        key="td3_b_only",
        label="TD3 scalar B-only disturb mismatch",
        short_label="TD3 B-only",
        family="matrix",
        structure="scalar",
        agent="td3",
        run_mode="disturb",
        state_mode="mismatch",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_td3_disturb_fluctuation_mismatch_unified" / "20260425_082831" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_matrix_td3_disturb_fluctuation_mismatch" / "20260425_082842" / "input_data.pkl",
        baseline_path=DISTURB_BASELINE_PATH,
    ),
    RunSpec(
        key="td3_a_only",
        label="TD3 scalar A-only disturb mismatch",
        short_label="TD3 A-only",
        family="matrix",
        structure="scalar",
        agent="td3",
        run_mode="disturb",
        state_mode="mismatch",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_td3_disturb_fluctuation_mismatch_unified" / "20260429_033606" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_matrix_td3_disturb_fluctuation_mismatch" / "20260429_033621" / "input_data.pkl",
        baseline_path=DISTURB_BASELINE_PATH,
    ),
    RunSpec(
        key="sac_scalar_disturb",
        label="SAC scalar disturb standard",
        short_label="SAC scalar",
        family="matrix",
        structure="scalar",
        agent="sac",
        run_mode="disturb",
        state_mode="standard",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_sac_disturb_fluctuation_standard_unified" / "20260415_104840" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_matrix_sac_disturb_fluctuation_standard" / "20260415_104846" / "input_data.pkl",
    ),
    RunSpec(
        key="sac_structured_disturb",
        label="SAC structured disturb standard",
        short_label="SAC structured",
        family="structured_matrix",
        structure="structured",
        agent="sac",
        run_mode="disturb",
        state_mode="standard",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_structured_matrix_sac_disturb_fluctuation_standard_unified" / "20260415_120923" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_structured_matrix_sac_disturb_fluctuation_standard" / "20260415_120930" / "input_data.pkl",
    ),
    RunSpec(
        key="td3_scalar_nominal",
        label="TD3 scalar nominal standard",
        short_label="TD3 scalar nominal",
        family="matrix",
        structure="scalar",
        agent="td3",
        run_mode="nominal",
        state_mode="standard",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_td3_nominal_none_standard_unified" / "20260412_134300" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_matrix_td3_nominal_none_standard" / "20260412_134310" / "input_data.pkl",
    ),
    RunSpec(
        key="td3_structured_nominal",
        label="TD3 structured nominal standard",
        short_label="TD3 structured nominal",
        family="structured_matrix",
        structure="structured",
        agent="td3",
        run_mode="nominal",
        state_mode="standard",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_structured_matrix_td3_nominal_none_standard_unified" / "20260412_134447" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_structured_matrix_td3_nominal_none_standard" / "20260412_134459" / "input_data.pkl",
    ),
]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def as_python_bool(value):
    if value is None:
        return None
    return bool(value)


def get_flag(bundle: dict, config: dict, name: str):
    if name in bundle:
        return as_python_bool(bundle.get(name))
    if name in config:
        return as_python_bool(config.get(name))
    return None


def reward_delta(compare_bundle: dict) -> np.ndarray:
    rl = np.asarray(compare_bundle["avg_rewards_rl"], float).reshape(-1)
    mpc = np.asarray(compare_bundle["avg_rewards_mpc"], float).reshape(-1)
    if mpc.size == 0:
        raise ValueError("compare bundle is missing avg_rewards_mpc.")
    if mpc.size != rl.size:
        mpc = np.full(rl.shape, float(mpc[-1]), dtype=float)
    return rl - mpc


def per_episode_abs_physical(bundle: dict) -> dict[str, np.ndarray | float]:
    delta_y = np.asarray(bundle["delta_y_storage"], float)
    delta_u = np.asarray(bundle["delta_u_storage"], float)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(delta_u.shape[1])
    output_ranges = data_max[n_inputs:] - data_min[n_inputs:]
    input_ranges = data_max[:n_inputs] - data_min[:n_inputs]
    episode_len = int(delta_y.shape[0] // N_EPISODES)

    abs_y_phys = np.abs(delta_y).reshape(N_EPISODES, episode_len, -1) * output_ranges.reshape(1, 1, -1)
    abs_u_phys = np.abs(delta_u).reshape(N_EPISODES, episode_len, -1) * input_ranges.reshape(1, 1, -1)

    mae_by_output = abs_y_phys.mean(axis=1)
    move_by_input = abs_u_phys.mean(axis=1)

    return {
        "episode_len": episode_len,
        "mae_by_output": mae_by_output,
        "mae_mean": mae_by_output.mean(axis=1),
        "move_by_input": move_by_input,
        "move_mean": move_by_input.mean(axis=1),
    }


def summarize_run(spec: RunSpec) -> tuple[dict, dict]:
    rl_bundle = load_pickle(spec.rl_path)
    compare_bundle = load_pickle(spec.compare_path)
    cfg = dict(rl_bundle.get("config_snapshot", {}))

    rewards = reward_delta(compare_bundle)
    metrics = per_episode_abs_physical(rl_bundle)

    row = {
        "key": spec.key,
        "label": spec.label,
        "short_label": spec.short_label,
        "family": spec.family,
        "structure": spec.structure,
        "agent": spec.agent,
        "run_mode": spec.run_mode,
        "state_mode": spec.state_mode,
        "set_points_len": int(cfg.get("set_points_len", 0)),
        "warm_start": int(cfg.get("warm_start", 0)),
        "steps_per_episode": int(metrics["episode_len"]),
        "release_guard_enabled": get_flag(rl_bundle, cfg, "release_guard_enabled"),
        "mpc_acceptance_enabled": get_flag(rl_bundle, cfg, "mpc_acceptance_enabled"),
        "mpc_usefulness_gate_enabled": get_flag(rl_bundle, cfg, "mpc_usefulness_gate_enabled"),
        "reward_delta_full_mean": float(rewards.mean()),
        "reward_delta_last20_mean": float(rewards[-20:].mean()),
        "reward_delta_last10_mean": float(rewards[-10:].mean()),
        "reward_delta_last_episode": float(rewards[-1]),
        "reward_delta_best_episode": float(rewards.max()),
        "reward_delta_best_episode_index": int(rewards.argmax() + 1),
        "positive_episode_count": int(np.sum(rewards > 0.0)),
        "positive_episode_count_post15": int(np.sum(rewards[15:] > 0.0)),
        "first_positive_episode_post15": (
            int(np.flatnonzero(rewards[15:] > 0.0)[0] + 16) if np.any(rewards[15:] > 0.0) else None
        ),
        "final_mae_phys_mean": float(metrics["mae_mean"][-1]),
        "final_mae_phys_out1": float(metrics["mae_by_output"][-1, 0]),
        "final_mae_phys_out2": float(metrics["mae_by_output"][-1, 1]),
        "tail20_mae_phys_mean": float(metrics["mae_mean"][-20:].mean()),
        "tail20_mae_phys_out1": float(metrics["mae_by_output"][-20:, 0].mean()),
        "tail20_mae_phys_out2": float(metrics["mae_by_output"][-20:, 1].mean()),
        "final_input_move_phys_mean": float(metrics["move_mean"][-1]),
        "tail20_input_move_phys_mean": float(metrics["move_mean"][-20:].mean()),
        "rl_path": str(spec.rl_path.relative_to(REPO_ROOT)),
        "compare_path": str(spec.compare_path.relative_to(REPO_ROOT)),
        "baseline_path": str(spec.baseline_path.relative_to(REPO_ROOT)) if spec.baseline_path is not None else "",
        "direct_mpc_tracking_compare": False,
        "mpc_final_mae_phys_mean": None,
        "mpc_final_mae_phys_out2": None,
        "mpc_tail20_mae_phys_mean": None,
        "mpc_tail20_mae_phys_out2": None,
        "mpc_final_input_move_phys_mean": None,
        "mpc_tail20_input_move_phys_mean": None,
    }

    mpc_metrics = None
    if spec.baseline_path is not None and spec.baseline_path.exists():
        baseline_bundle = load_pickle(spec.baseline_path)
        candidate_mpc_metrics = per_episode_abs_physical(baseline_bundle)
        if int(candidate_mpc_metrics["episode_len"]) == int(metrics["episode_len"]):
            mpc_metrics = candidate_mpc_metrics
            row["direct_mpc_tracking_compare"] = True
            row["mpc_final_mae_phys_mean"] = float(mpc_metrics["mae_mean"][-1])
            row["mpc_final_mae_phys_out2"] = float(mpc_metrics["mae_by_output"][-1, 1])
            row["mpc_tail20_mae_phys_mean"] = float(mpc_metrics["mae_mean"][-20:].mean())
            row["mpc_tail20_mae_phys_out2"] = float(mpc_metrics["mae_by_output"][-20:, 1].mean())
            row["mpc_final_input_move_phys_mean"] = float(mpc_metrics["move_mean"][-1])
            row["mpc_tail20_input_move_phys_mean"] = float(mpc_metrics["move_mean"][-20:].mean())

    return row, {
        "spec": spec,
        "config": cfg,
        "rl_bundle": rl_bundle,
        "compare_bundle": compare_bundle,
        "reward_delta": rewards,
        "metrics": metrics,
        "mpc_metrics": mpc_metrics,
    }


def write_summary_csv(rows: list[dict]):
    fieldnames = [
        "key",
        "label",
        "short_label",
        "family",
        "structure",
        "agent",
        "run_mode",
        "state_mode",
        "set_points_len",
        "warm_start",
        "steps_per_episode",
        "release_guard_enabled",
        "mpc_acceptance_enabled",
        "mpc_usefulness_gate_enabled",
        "reward_delta_full_mean",
        "reward_delta_last20_mean",
        "reward_delta_last10_mean",
        "reward_delta_last_episode",
        "reward_delta_best_episode",
        "reward_delta_best_episode_index",
        "positive_episode_count",
        "positive_episode_count_post15",
        "first_positive_episode_post15",
        "final_mae_phys_mean",
        "final_mae_phys_out1",
        "final_mae_phys_out2",
        "tail20_mae_phys_mean",
        "tail20_mae_phys_out1",
        "tail20_mae_phys_out2",
        "final_input_move_phys_mean",
        "tail20_input_move_phys_mean",
        "direct_mpc_tracking_compare",
        "mpc_final_mae_phys_mean",
        "mpc_final_mae_phys_out2",
        "mpc_tail20_mae_phys_mean",
        "mpc_tail20_mae_phys_out2",
        "mpc_final_input_move_phys_mean",
        "mpc_tail20_input_move_phys_mean",
        "rl_path",
        "compare_path",
        "baseline_path",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_reward_delta_overview(run_data: dict[str, dict]):
    order = [
        "td3_b_only",
        "td3_a_only",
        "sac_scalar_disturb",
        "sac_structured_disturb",
    ]
    colors = {
        "td3_b_only": "#d62728",
        "td3_a_only": "#ff7f0e",
        "sac_scalar_disturb": "#1f77b4",
        "sac_structured_disturb": "#2ca02c",
    }
    styles = {
        "td3_b_only": "-",
        "td3_a_only": "-",
        "sac_scalar_disturb": "--",
        "sac_structured_disturb": "--",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    episodes = np.arange(1, N_EPISODES + 1)
    for key in order:
        series = run_data[key]["reward_delta"]
        label = run_data[key]["spec"].short_label
        axes[0].plot(episodes, series, styles[key], lw=2.2, color=colors[key], label=label)
    axes[0].axhline(0.0, color="0.3", lw=1.0, linestyle=":")
    axes[0].set_title("Distillation Disturbance Reward Delta vs MPC")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward delta (RL - MPC)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    start = 141
    for key in ("td3_b_only", "td3_a_only"):
        series = run_data[key]["reward_delta"][start - 1 :]
        label = run_data[key]["spec"].short_label
        axes[1].plot(np.arange(start, N_EPISODES + 1), series, lw=2.4, color=colors[key], label=label)
    axes[1].axhline(0.0, color="0.3", lw=1.0, linestyle=":")
    axes[1].set_title("Late-Episode TD3 Disturbance Window")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward delta (RL - MPC)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out_path = FIG_DIR / "distillation_matrix_reward_delta_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_td3_directionality_tradeoff(run_data: dict[str, dict]):
    a_row = run_data["td3_a_only"]["row"]
    b_row = run_data["td3_b_only"]["row"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    reward_windows = ["Last 20", "Last 10", "Last 1"]
    reward_a = [
        a_row["reward_delta_last20_mean"],
        a_row["reward_delta_last10_mean"],
        a_row["reward_delta_last_episode"],
    ]
    reward_b = [
        b_row["reward_delta_last20_mean"],
        b_row["reward_delta_last10_mean"],
        b_row["reward_delta_last_episode"],
    ]
    x = np.arange(len(reward_windows))
    width = 0.34
    axes[0].bar(x - width / 2, reward_a, width, color="#ff7f0e", label="A-only")
    axes[0].bar(x + width / 2, reward_b, width, color="#d62728", label="B-only")
    axes[0].axhline(0.0, color="0.3", lw=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reward_windows)
    axes[0].set_title("TD3 Reward Delta Windows")
    axes[0].set_ylabel("Reward delta (RL - MPC)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    metric_labels = ["Final", "Tail 20"]
    out2_a = [a_row["final_mae_phys_out2"], a_row["tail20_mae_phys_out2"]]
    out2_b = [b_row["final_mae_phys_out2"], b_row["tail20_mae_phys_out2"]]
    out2_mpc = [a_row["mpc_final_mae_phys_out2"], a_row["mpc_tail20_mae_phys_out2"]]
    x2 = np.arange(len(metric_labels))
    axes[1].bar(x2 - width, out2_a, width, color="#ff7f0e", label="A-only")
    axes[1].bar(x2, out2_b, width, color="#d62728", label="B-only")
    axes[1].bar(x2 + width, out2_mpc, width, color="0.55", label="MPC")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metric_labels)
    axes[1].set_title("Output-2 Physical MAE")
    axes[1].set_ylabel("MAE")
    axes[1].grid(axis="y", alpha=0.25)

    move_a = [a_row["final_input_move_phys_mean"], a_row["tail20_input_move_phys_mean"]]
    move_b = [b_row["final_input_move_phys_mean"], b_row["tail20_input_move_phys_mean"]]
    move_mpc = [a_row["mpc_final_input_move_phys_mean"], a_row["mpc_tail20_input_move_phys_mean"]]
    axes[2].bar(x2 - width, move_a, width, color="#ff7f0e", label="A-only")
    axes[2].bar(x2, move_b, width, color="#d62728", label="B-only")
    axes[2].bar(x2 + width, move_mpc, width, color="0.55", label="MPC")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(metric_labels)
    axes[2].set_title("Mean Physical Input Movement")
    axes[2].set_ylabel("Mean |delta u|")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path = FIG_DIR / "distillation_td3_directionality_tradeoff.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_reward_vs_tail_tradeoff(run_data: dict[str, dict]):
    selected = [
        "td3_b_only",
        "td3_a_only",
        "sac_scalar_disturb",
        "sac_structured_disturb",
    ]
    colors = {
        "td3_b_only": "#d62728",
        "td3_a_only": "#ff7f0e",
        "sac_scalar_disturb": "#1f77b4",
        "sac_structured_disturb": "#2ca02c",
    }
    markers = {
        "td3_b_only": "o",
        "td3_a_only": "o",
        "sac_scalar_disturb": "s",
        "sac_structured_disturb": "D",
    }

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for key in selected:
        row = run_data[key]["row"]
        x = row["tail20_mae_phys_out2"]
        y = row["reward_delta_last20_mean"]
        size = 50.0 + 0.9 * row["tail20_input_move_phys_mean"]
        ax.scatter(x, y, s=size, marker=markers[key], color=colors[key], alpha=0.85, edgecolors="black", linewidths=0.6)
        ax.annotate(run_data[key]["spec"].short_label, (x, y), xytext=(6, 6), textcoords="offset points")

    ax.axhline(0.0, color="0.3", lw=1.0, linestyle=":")
    ax.set_title("Reward vs Tail Output-2 Tradeoff")
    ax.set_xlabel("Tail-20 physical MAE on output 2")
    ax.set_ylabel("Last-20 reward delta (RL - MPC)")
    ax.grid(alpha=0.25)

    note = "Marker size proportional to tail-20 mean physical input movement."
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    fig.tight_layout()
    out_path = FIG_DIR / "distillation_matrix_reward_vs_tail_tradeoff.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    run_data: dict[str, dict] = {}
    for spec in RUNS:
        row, payload = summarize_run(spec)
        payload["row"] = row
        rows.append(row)
        run_data[spec.key] = payload

    write_summary_csv(rows)
    plot_reward_delta_overview(run_data)
    plot_td3_directionality_tradeoff(run_data)
    plot_reward_vs_tail_tradeoff(run_data)

    print(f"Wrote summary CSV: {SUMMARY_CSV}")
    for name in (
        "distillation_matrix_reward_delta_overview.png",
        "distillation_td3_directionality_tradeoff.png",
        "distillation_matrix_reward_vs_tail_tradeoff.png",
    ):
        print(f"Wrote figure: {FIG_DIR / name}")


if __name__ == "__main__":
    main()
