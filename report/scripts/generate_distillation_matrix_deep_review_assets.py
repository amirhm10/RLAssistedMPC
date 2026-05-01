from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_matrix_deep_review_20260501"
SUMMARY_CSV = FIG_DIR / "distillation_matrix_deep_review_summary.csv"
WINDOW_CSV = FIG_DIR / "distillation_matrix_window_summary.csv"

N_EPISODES = 200
DISTURB_BASELINE_PATH = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
WINDOWS = [
    ("1-10", 1, 10),
    ("11-15", 11, 15),
    ("16-30", 16, 30),
    ("31-100", 31, 100),
    ("101-200", 101, 200),
    ("181-200", 181, 200),
]


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    short_label: str
    family: str
    structure: str
    agent: str
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
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_td3_disturb_fluctuation_mismatch_unified" / "20260429_033606" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_matrix_td3_disturb_fluctuation_mismatch" / "20260429_033621" / "input_data.pkl",
        baseline_path=DISTURB_BASELINE_PATH,
    ),
    RunSpec(
        key="sac_scalar",
        label="SAC scalar disturb standard",
        short_label="SAC scalar",
        family="matrix",
        structure="scalar",
        agent="sac",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_sac_disturb_fluctuation_standard_unified" / "20260415_104840" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_matrix_sac_disturb_fluctuation_standard" / "20260415_104846" / "input_data.pkl",
        baseline_path=DISTURB_BASELINE_PATH,
    ),
    RunSpec(
        key="sac_structured",
        label="SAC structured disturb standard",
        short_label="SAC structured",
        family="structured_matrix",
        structure="structured",
        agent="sac",
        rl_path=REPO_ROOT / "Distillation" / "Results" / "distillation_structured_matrix_sac_disturb_fluctuation_standard_unified" / "20260415_120923" / "input_data.pkl",
        compare_path=REPO_ROOT / "Distillation" / "Results" / "distillation_compare_structured_matrix_sac_disturb_fluctuation_standard" / "20260415_120930" / "input_data.pkl",
        baseline_path=DISTURB_BASELINE_PATH,
    ),
]


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def get_flag(bundle: dict, config: dict, name: str):
    if name in bundle:
        return bool(bundle[name])
    if name in config:
        return bool(config[name])
    return None


def reward_delta(compare_bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rl = np.asarray(compare_bundle["avg_rewards_rl"], float).reshape(-1)
    mpc = np.asarray(compare_bundle["avg_rewards_mpc"], float).reshape(-1)
    if mpc.size != rl.size:
        mpc = np.full(rl.shape, float(mpc[-1]), dtype=float)
    return rl - mpc, rl, mpc


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

    return {
        "episode_len": episode_len,
        "mae_by_output": abs_y_phys.mean(axis=1),
        "mae_mean": abs_y_phys.mean(axis=(1, 2)),
        "move_by_input": abs_u_phys.mean(axis=1),
        "move_mean": abs_u_phys.mean(axis=(1, 2)),
    }


def schedule_info(bundle: dict) -> dict[str, float | int | None]:
    cfg = dict(bundle.get("config_snapshot", {}))
    metrics = per_episode_abs_physical(bundle)
    warm_start = cfg.get("warm_start")
    if warm_start is None and "warm_start_step" in bundle and "time_in_sub_episodes" in bundle:
        denom = float(bundle["time_in_sub_episodes"])
        warm_start = None if denom <= 0.0 else float(bundle["warm_start_step"]) / denom

    set_points_len = cfg.get("set_points_len")
    if set_points_len is None and "y_sp" in bundle:
        y_sp = np.asarray(bundle["y_sp"], float)
        if y_sp.ndim == 2 and y_sp.shape[0] >= 2:
            diffs = np.any(np.abs(np.diff(y_sp, axis=0)) > 1e-12, axis=1)
            change_idx = np.flatnonzero(diffs)
            if change_idx.size > 0:
                set_points_len = int(change_idx[0] + 1)

    return {
        "steps_per_episode": int(metrics["episode_len"]),
        "set_points_len": set_points_len,
        "warm_start": warm_start,
        "avg_rewards_len": int(np.asarray(bundle.get("avg_rewards", []), float).size),
    }


def summarize_run(spec: RunSpec, baseline_bundle: dict, baseline_metrics: dict) -> tuple[dict, dict, list[dict]]:
    rl_bundle = load_pickle(spec.rl_path)
    compare_bundle = load_pickle(spec.compare_path)
    cfg = dict(rl_bundle.get("config_snapshot", {}))
    rewards, _, _ = reward_delta(compare_bundle)
    metrics = per_episode_abs_physical(rl_bundle)
    sched = schedule_info(rl_bundle)
    baseline_sched = schedule_info(baseline_bundle)

    row = {
        "key": spec.key,
        "label": spec.label,
        "short_label": spec.short_label,
        "family": spec.family,
        "structure": spec.structure,
        "agent": spec.agent,
        "run_mode": cfg.get("run_mode"),
        "state_mode": cfg.get("state_mode"),
        "set_points_len": sched["set_points_len"],
        "warm_start": sched["warm_start"],
        "steps_per_episode": sched["steps_per_episode"],
        "baseline_steps_per_episode": baseline_sched["steps_per_episode"],
        "baseline_set_points_len": baseline_sched["set_points_len"],
        "baseline_warm_start": baseline_sched["warm_start"],
        "schedule_matches_baseline_steps": bool(sched["steps_per_episode"] == baseline_sched["steps_per_episode"]),
        "release_guard_enabled": get_flag(rl_bundle, cfg, "release_guard_enabled")
        if "release_guard_enabled" in rl_bundle
        else get_flag(cfg, cfg.get("release_protected_advisory_caps", {}), "enabled"),
        "behavioral_cloning_enabled": get_flag(rl_bundle, cfg, "behavioral_cloning_enabled")
        if "behavioral_cloning_enabled" in rl_bundle
        else get_flag(cfg, cfg.get("behavioral_cloning", {}), "enabled"),
        "dual_cost_shadow_enabled": get_flag(rl_bundle, cfg, "dual_cost_shadow_enabled")
        if "dual_cost_shadow_enabled" in rl_bundle
        else get_flag(cfg, cfg.get("mpc_dual_cost_shadow", {}), "enabled"),
        "usefulness_gate_enabled": get_flag(rl_bundle, cfg, "mpc_usefulness_gate_enabled")
        if "mpc_usefulness_gate_enabled" in rl_bundle
        else get_flag(cfg, cfg.get("mpc_usefulness_gate", {}), "enabled"),
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
        "tail20_mae_phys_mean": float(metrics["mae_mean"][-20:].mean()),
        "tail20_mae_phys_out1": float(metrics["mae_by_output"][-20:, 0].mean()),
        "tail20_mae_phys_out2": float(metrics["mae_by_output"][-20:, 1].mean()),
        "tail20_input_move_phys_mean": float(metrics["move_mean"][-20:].mean()),
        "direct_mpc_tracking_compare": bool(sched["steps_per_episode"] == baseline_sched["steps_per_episode"]),
        "mpc_tail20_mae_phys_mean": None,
        "mpc_tail20_mae_phys_out1": None,
        "mpc_tail20_mae_phys_out2": None,
        "mpc_tail20_input_move_phys_mean": None,
        "rl_path": str(spec.rl_path.relative_to(REPO_ROOT)),
        "compare_path": str(spec.compare_path.relative_to(REPO_ROOT)),
        "baseline_path": str(spec.baseline_path.relative_to(REPO_ROOT)) if spec.baseline_path else "",
    }

    if row["direct_mpc_tracking_compare"]:
        row["mpc_tail20_mae_phys_mean"] = float(baseline_metrics["mae_mean"][-20:].mean())
        row["mpc_tail20_mae_phys_out1"] = float(baseline_metrics["mae_by_output"][-20:, 0].mean())
        row["mpc_tail20_mae_phys_out2"] = float(baseline_metrics["mae_by_output"][-20:, 1].mean())
        row["mpc_tail20_input_move_phys_mean"] = float(baseline_metrics["move_mean"][-20:].mean())

    window_rows: list[dict] = []
    for window_name, start, end in WINDOWS:
        sl = slice(start - 1, end)
        reward_window = rewards[sl]
        window_row = {
            "run_key": spec.key,
            "short_label": spec.short_label,
            "window": window_name,
            "start_episode": start,
            "end_episode": end,
            "reward_delta_mean": float(reward_window.mean()),
            "positive_count": int(np.sum(reward_window > 0.0)),
            "episodes_in_window": int(reward_window.size),
            "mae_mean": float(metrics["mae_mean"][sl].mean()),
            "mae_out1": float(metrics["mae_by_output"][sl, 0].mean()),
            "mae_out2": float(metrics["mae_by_output"][sl, 1].mean()),
            "move_mean": float(metrics["move_mean"][sl].mean()),
        }
        window_rows.append(window_row)

    payload = {
        "spec": spec,
        "row": row,
        "config": cfg,
        "rl_bundle": rl_bundle,
        "compare_bundle": compare_bundle,
        "reward_delta": rewards,
        "metrics": metrics,
        "schedule": sched,
    }
    return row, payload, window_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_schedule_alignment(run_data: dict[str, dict], baseline_bundle: dict):
    baseline_sched = schedule_info(baseline_bundle)
    order = ["baseline", "td3_b_only", "td3_a_only", "sac_scalar", "sac_structured"]
    labels = {
        "baseline": "Disturb MPC\nbaseline",
        "td3_b_only": "TD3\nB-only",
        "td3_a_only": "TD3\nA-only",
        "sac_scalar": "SAC\nscalar",
        "sac_structured": "SAC\nstructured",
    }
    colors = {
        "baseline": "#7f7f7f",
        "td3_b_only": "#d62728",
        "td3_a_only": "#ff7f0e",
        "sac_scalar": "#1f77b4",
        "sac_structured": "#2ca02c",
    }

    steps = [baseline_sched["steps_per_episode"]]
    set_points = [baseline_sched["set_points_len"]]
    warm_start = [baseline_sched["warm_start"]]
    for key in order[1:]:
        steps.append(run_data[key]["schedule"]["steps_per_episode"])
        set_points.append(run_data[key]["schedule"]["set_points_len"])
        warm_start.append(run_data[key]["schedule"]["warm_start"])

    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    for ax, values, title, ylabel in (
        (axes[0], steps, "Episode Length", "steps per episode"),
        (axes[1], set_points, "Setpoint Block Length", "set_points_len"),
        (axes[2], warm_start, "Warm-Start Episodes", "warm_start"),
    ):
        ax.bar(x, values, color=[colors[k] for k in order], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in order])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].text(
        0.02,
        0.98,
        "TD3 mismatch runs align on episode length.\nSAC disturbance runs do not.",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    out_path = FIG_DIR / "distillation_matrix_schedule_alignment.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_family_reward_windows(window_rows: list[dict]):
    order = ["td3_b_only", "td3_a_only", "sac_scalar", "sac_structured"]
    labels = {
        "td3_b_only": "TD3 B-only",
        "td3_a_only": "TD3 A-only",
        "sac_scalar": "SAC scalar",
        "sac_structured": "SAC structured",
    }
    colors = {
        "td3_b_only": "#d62728",
        "td3_a_only": "#ff7f0e",
        "sac_scalar": "#1f77b4",
        "sac_structured": "#2ca02c",
    }
    window_names = [name for name, _, _ in WINDOWS]
    width = 0.18
    x = np.arange(len(window_names))

    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    for idx, key in enumerate(order):
        vals = [
            next(row["reward_delta_mean"] for row in window_rows if row["run_key"] == key and row["window"] == name)
            for name in window_names
        ]
        ax.bar(x + (idx - 1.5) * width, vals, width, color=colors[key], label=labels[key])

    ax.axhline(0.0, color="0.3", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(window_names)
    ax.set_ylabel("Reward delta (RL - MPC)")
    ax.set_title("Distillation Matrix Reward Delta by Episode Window")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    out_path = FIG_DIR / "distillation_matrix_family_reward_windows.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_td3_tail_physical_tradeoff(run_data: dict[str, dict]):
    b_row = run_data["td3_b_only"]["row"]
    a_row = run_data["td3_a_only"]["row"]

    metric_names = [
        ("Last-20 reward delta", "reward_delta_last20_mean"),
        ("Tail-20 x24 MAE", "tail20_mae_phys_out1"),
        ("Tail-20 T85 MAE", "tail20_mae_phys_out2"),
        ("Tail-20 mean |delta u|", "tail20_input_move_phys_mean"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.6))
    width = 0.26
    x = np.arange(1)

    for ax, (title, key) in zip(axes, metric_names):
        a_val = a_row[key]
        b_val = b_row[key]
        ax.bar(x - width, [a_val], width, color="#ff7f0e", label="A-only")
        ax.bar(x, [b_val], width, color="#d62728", label="B-only")
        if key.startswith("tail20_"):
            mpc_key = "mpc_" + key
            ax.bar(x + width, [b_row[mpc_key]], width, color="0.55", label="MPC")
        ax.set_xticks([])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if key == "reward_delta_last20_mean":
            ax.axhline(0.0, color="0.3", lw=1.0)

    axes[0].legend(frameon=False, loc="upper center")
    fig.suptitle("Matched Distillation TD3 Tail Tradeoff", y=1.02)
    fig.tight_layout()
    out_path = FIG_DIR / "distillation_td3_tail_physical_tradeoff.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    baseline_bundle = load_pickle(DISTURB_BASELINE_PATH)
    baseline_metrics = per_episode_abs_physical(baseline_bundle)

    summary_rows: list[dict] = []
    window_rows: list[dict] = []
    run_data: dict[str, dict] = {}

    for spec in RUNS:
        row, payload, run_windows = summarize_run(spec, baseline_bundle, baseline_metrics)
        summary_rows.append(row)
        window_rows.extend(run_windows)
        run_data[spec.key] = payload

    write_csv(
        SUMMARY_CSV,
        summary_rows,
        [
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
            "baseline_steps_per_episode",
            "baseline_set_points_len",
            "baseline_warm_start",
            "schedule_matches_baseline_steps",
            "release_guard_enabled",
            "behavioral_cloning_enabled",
            "dual_cost_shadow_enabled",
            "usefulness_gate_enabled",
            "reward_delta_full_mean",
            "reward_delta_last20_mean",
            "reward_delta_last10_mean",
            "reward_delta_last_episode",
            "reward_delta_best_episode",
            "reward_delta_best_episode_index",
            "positive_episode_count",
            "positive_episode_count_post15",
            "first_positive_episode_post15",
            "tail20_mae_phys_mean",
            "tail20_mae_phys_out1",
            "tail20_mae_phys_out2",
            "tail20_input_move_phys_mean",
            "direct_mpc_tracking_compare",
            "mpc_tail20_mae_phys_mean",
            "mpc_tail20_mae_phys_out1",
            "mpc_tail20_mae_phys_out2",
            "mpc_tail20_input_move_phys_mean",
            "rl_path",
            "compare_path",
            "baseline_path",
        ],
    )
    write_csv(
        WINDOW_CSV,
        window_rows,
        [
            "run_key",
            "short_label",
            "window",
            "start_episode",
            "end_episode",
            "reward_delta_mean",
            "positive_count",
            "episodes_in_window",
            "mae_mean",
            "mae_out1",
            "mae_out2",
            "move_mean",
        ],
    )

    plot_schedule_alignment(run_data, baseline_bundle)
    plot_family_reward_windows(window_rows)
    plot_td3_tail_physical_tradeoff(run_data)

    print(f"Wrote summary CSV: {SUMMARY_CSV}")
    print(f"Wrote window CSV: {WINDOW_CSV}")
    print(f"Wrote figure: {FIG_DIR / 'distillation_matrix_schedule_alignment.png'}")
    print(f"Wrote figure: {FIG_DIR / 'distillation_matrix_family_reward_windows.png'}")
    print(f"Wrote figure: {FIG_DIR / 'distillation_td3_tail_physical_tradeoff.png'}")


if __name__ == "__main__":
    main()
