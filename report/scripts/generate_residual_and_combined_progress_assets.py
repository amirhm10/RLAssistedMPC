from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "report" / "figures" / "residual_and_combined_progress_20260429"
SUMMARY_CSV = OUTPUT_DIR / "residual_and_combined_progress_summary.csv"

POLYMER_BASELINE_DIST = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
POLYMER_BASELINE_NOMINAL = REPO_ROOT / "Polymer" / "Data" / "mpc_results_nominal.pickle"
DISTILLATION_BASELINE_FLUCT = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"


@dataclass(frozen=True)
class RunSpec:
    slug: str
    label: str
    short_label: str
    system: str
    family: str
    run_mode: str
    path: Path
    baseline_path: Path


RUN_SPECS = [
    RunSpec(
        slug="polymer_nominal_20260402",
        label="Polymer residual nominal 2026-04-02",
        short_label="P nom 0402",
        system="polymer",
        family="residual_nominal",
        run_mode="nominal",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_nominal" / "20260402_174941" / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_NOMINAL,
    ),
    RunSpec(
        slug="polymer_nominal_20260408a",
        label="Polymer residual nominal 2026-04-08 run A",
        short_label="P nom 0408A",
        system="polymer",
        family="residual_nominal",
        run_mode="nominal",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_nominal" / "20260408_121130" / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_NOMINAL,
    ),
    RunSpec(
        slug="polymer_nominal_20260408b",
        label="Polymer residual nominal 2026-04-08 run B",
        short_label="P nom 0408B",
        system="polymer",
        family="residual_nominal",
        run_mode="nominal",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_nominal" / "20260408_131932" / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_NOMINAL,
    ),
    RunSpec(
        slug="polymer_disturb_20260413",
        label="Polymer residual disturb 2026-04-13",
        short_label="P res 0413",
        system="polymer",
        family="residual_disturb",
        run_mode="disturb",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260413_004620" / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_DIST,
    ),
    RunSpec(
        slug="polymer_disturb_20260420",
        label="Polymer residual disturb 2026-04-20",
        short_label="P res 0420",
        system="polymer",
        family="residual_disturb",
        run_mode="disturb",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260420_225631" / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_DIST,
    ),
    RunSpec(
        slug="polymer_disturb_20260422",
        label="Polymer residual disturb 2026-04-22",
        short_label="P res 0422",
        system="polymer",
        family="residual_disturb",
        run_mode="disturb",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260422_181610" / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_DIST,
    ),
    RunSpec(
        slug="polymer_disturb_20260423",
        label="Polymer residual disturb 2026-04-23",
        short_label="P res 0423",
        system="polymer",
        family="residual_disturb",
        run_mode="disturb",
        path=REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260423_025802" / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_DIST,
    ),
    RunSpec(
        slug="polymer_combined_20260426a",
        label="Polymer combined disturb 2026-04-26 run A",
        short_label="P comb A",
        system="polymer",
        family="combined_disturb",
        run_mode="disturb",
        path=REPO_ROOT
        / "Polymer"
        / "Results"
        / "combined_disturb_h_dqn_mismatch__m_td3_mismatch__w_td3_mismatch__r_td3_mismatch_rho"
        / "20260426_042026"
        / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_DIST,
    ),
    RunSpec(
        slug="polymer_combined_20260426b",
        label="Polymer combined disturb 2026-04-26 run B",
        short_label="P comb B",
        system="polymer",
        family="combined_disturb",
        run_mode="disturb",
        path=REPO_ROOT
        / "Polymer"
        / "Results"
        / "combined_disturb_h_dqn_mismatch__m_td3_mismatch__w_td3_mismatch__r_td3_mismatch_rho"
        / "20260426_193310"
        / "input_data.pkl",
        baseline_path=POLYMER_BASELINE_DIST,
    ),
    RunSpec(
        slug="distill_td3_20260414",
        label="Distillation residual TD3 2026-04-14",
        short_label="D td3 0414",
        system="distillation",
        family="distillation_td3",
        run_mode="disturb",
        path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260414_063405"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_FLUCT,
    ),
    RunSpec(
        slug="distill_sac_20260415",
        label="Distillation residual SAC 2026-04-15",
        short_label="D sac 0415",
        system="distillation",
        family="distillation_sac",
        run_mode="disturb",
        path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified"
        / "20260415_191909"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_FLUCT,
    ),
    RunSpec(
        slug="distill_td3_20260417",
        label="Distillation residual TD3 2026-04-17",
        short_label="D td3 0417",
        system="distillation",
        family="distillation_td3",
        run_mode="disturb",
        path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260417_181426"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_FLUCT,
    ),
    RunSpec(
        slug="distill_td3_20260420",
        label="Distillation residual TD3 2026-04-20",
        short_label="D td3 0420",
        system="distillation",
        family="distillation_td3",
        run_mode="disturb",
        path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260420_175629"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_FLUCT,
    ),
    RunSpec(
        slug="distill_td3_20260425",
        label="Distillation residual TD3 2026-04-25",
        short_label="D td3 0425",
        system="distillation",
        family="distillation_td3",
        run_mode="disturb",
        path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified"
        / "20260425_082259"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_FLUCT,
    ),
    RunSpec(
        slug="distill_sac_20260427",
        label="Distillation residual SAC 2026-04-27",
        short_label="D sac 0427",
        system="distillation",
        family="distillation_sac",
        run_mode="disturb",
        path=REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified"
        / "20260427_194850"
        / "input_data.pkl",
        baseline_path=DISTILLATION_BASELINE_FLUCT,
    ),
]


FAMILY_COLORS = {
    "residual_nominal": "#4C78A8",
    "residual_disturb": "#1F77B4",
    "combined_disturb": "#F28E2B",
    "distillation_td3": "#59A14F",
    "distillation_sac": "#E15759",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def output_array(bundle: dict, key: str = "y") -> np.ndarray:
    arr = bundle.get(key)
    if arr is None:
        arr = bundle.get("y_mpc")
    arr = np.asarray(arr, float)
    return arr[:-1]


def setpoint_phys(bundle: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(bundle.get("n_inputs", len(np.asarray(bundle["steady_states"]["ss_inputs"], float))))
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_sp = np.asarray(bundle["y_sp"], float)
    return data_min[n_inputs:] + (y_sp + y_ss_scaled) * (data_max[n_inputs:] - data_min[n_inputs:])


def late_segment_mask(y_sp: np.ndarray) -> np.ndarray:
    change_idx = np.flatnonzero(np.any(np.abs(np.diff(y_sp, axis=0)) > 1e-12, axis=1)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(y_sp)]))
    mask = np.zeros(len(y_sp), dtype=bool)
    for start, end in zip(starts, ends):
        segment_len = int(end - start)
        if segment_len <= 5:
            continue
        window = max(10, min(30, segment_len // 4))
        mask[max(int(start), int(end) - window) : int(end)] = True
    if not np.any(mask):
        mask[max(0, len(y_sp) - min(40, len(y_sp))) :] = True
    return mask


def reward_series(bundle: dict) -> np.ndarray:
    avg_rewards = bundle.get("avg_rewards")
    if avg_rewards is not None:
        return np.asarray(avg_rewards, float)
    rewards_step = bundle.get("rewards_step")
    steps_per_episode = int(bundle.get("time_in_sub_episodes", 0) or 0)
    if rewards_step is None or steps_per_episode <= 0:
        return np.zeros(0, dtype=float)
    rewards_step = np.asarray(rewards_step, float)
    return np.asarray(
        [np.mean(rewards_step[i : i + steps_per_episode]) for i in range(0, len(rewards_step) - steps_per_episode + 1, steps_per_episode)],
        float,
    )


def reward_metrics(rewards: np.ndarray) -> dict[str, float]:
    if rewards.size == 0:
        return {
            "reward_first10": float("nan"),
            "reward_mid10": float("nan"),
            "reward_last10": float("nan"),
            "reward_last20": float("nan"),
            "reward_best10": float("nan"),
            "reward_drop": float("nan"),
            "episode_count": 0,
        }
    best10 = max(np.mean(rewards[i : i + 10]) for i in range(0, max(1, rewards.size - 9)))
    mid_start = rewards.size // 2
    return {
        "reward_first10": float(np.mean(rewards[: min(10, rewards.size)])),
        "reward_mid10": float(np.mean(rewards[mid_start : min(rewards.size, mid_start + 10)])),
        "reward_last10": float(np.mean(rewards[-min(10, rewards.size) :])),
        "reward_last20": float(np.mean(rewards[-min(20, rewards.size) :])),
        "reward_best10": float(best10),
        "reward_drop": float(best10 - np.mean(rewards[-min(10, rewards.size) :])),
        "episode_count": int(rewards.size),
    }


def tail_metrics(bundle: dict, key: str = "y") -> dict[str, np.ndarray | float]:
    y = output_array(bundle, key=key)
    y_sp = setpoint_phys(bundle)[: len(y)]
    err = y - y_sp
    mask = late_segment_mask(y_sp)
    tail_err = err[mask]
    return {
        "tail_mae_mean": float(np.mean(np.abs(tail_err))),
        "tail_mae_output1": float(np.mean(np.abs(tail_err[:, 0]))),
        "tail_mae_output2": float(np.mean(np.abs(tail_err[:, 1]))),
        "tail_offset_output1": float(np.mean(tail_err[:, 0])),
        "tail_offset_output2": float(np.mean(tail_err[:, 1])),
        "rmse_output1": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_output2": float(np.sqrt(np.mean(err[:, 1] ** 2))),
    }


def last_eval_window(bundle: dict) -> tuple[int, int]:
    test_train = bundle.get("test_train_dict")
    n_steps = int(bundle["nFE"])
    if isinstance(test_train, dict) and test_train:
        eval_starts = sorted(int(k) for k, v in test_train.items() if bool(v))
        if eval_starts:
            start = eval_starts[-1]
            all_starts = sorted(int(k) for k in test_train.keys())
            later = [step for step in all_starts if step > start]
            end = later[0] if later else n_steps
            return start, end
    return max(0, n_steps - 400), n_steps


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def build_summary() -> tuple[list[dict], dict[str, dict]]:
    baseline_cache: dict[Path, dict] = {}
    rows: list[dict] = []
    bundles: dict[str, dict] = {}

    for spec in RUN_SPECS:
        bundle = load_pickle(spec.path)
        bundles[spec.slug] = bundle
        if spec.baseline_path not in baseline_cache:
            baseline_cache[spec.baseline_path] = load_pickle(spec.baseline_path)
        baseline = baseline_cache[spec.baseline_path]

        run_rewards = reward_series(bundle)
        base_rewards = reward_series(baseline)
        run_tail = tail_metrics(bundle, key="y")
        base_tail = tail_metrics(baseline, key="y" if baseline.get("y") is not None else "y_mpc")
        reward_info = reward_metrics(run_rewards)
        base_reward_info = reward_metrics(base_rewards)

        residual_exec = bundle.get("residual_exec_log")
        if residual_exec is None:
            residual_exec = bundle.get("delta_u_res_exec_log")
        residual_exec = None if residual_exec is None else np.asarray(residual_exec, float)

        residual_raw = bundle.get("residual_raw_log")
        if residual_raw is None:
            residual_raw = bundle.get("delta_u_res_raw_log")
        residual_raw = None if residual_raw is None else np.asarray(residual_raw, float)

        rho = bundle.get("rho_eff_log")
        rho = None if rho is None else np.asarray(rho, float)
        projection = bundle.get("projection_active_log")
        projection = None if projection is None else np.asarray(projection, float)

        row = {
            "slug": spec.slug,
            "label": spec.label,
            "short_label": spec.short_label,
            "system": spec.system,
            "family": spec.family,
            "run_mode": spec.run_mode,
            "path": repo_rel(spec.path),
            "baseline_path": repo_rel(spec.baseline_path),
            "tail_mae_mean": run_tail["tail_mae_mean"],
            "tail_mae_output1": run_tail["tail_mae_output1"],
            "tail_mae_output2": run_tail["tail_mae_output2"],
            "tail_offset_output1": run_tail["tail_offset_output1"],
            "tail_offset_output2": run_tail["tail_offset_output2"],
            "rmse_output1": run_tail["rmse_output1"],
            "rmse_output2": run_tail["rmse_output2"],
            "baseline_tail_mae_mean": base_tail["tail_mae_mean"],
            "baseline_tail_mae_output1": base_tail["tail_mae_output1"],
            "baseline_tail_mae_output2": base_tail["tail_mae_output2"],
            "baseline_tail_offset_output1": base_tail["tail_offset_output1"],
            "baseline_tail_offset_output2": base_tail["tail_offset_output2"],
            "tail_mae_delta_vs_baseline": run_tail["tail_mae_mean"] - base_tail["tail_mae_mean"],
            "reward_first10": reward_info["reward_first10"],
            "reward_mid10": reward_info["reward_mid10"],
            "reward_last10": reward_info["reward_last10"],
            "reward_last20": reward_info["reward_last20"],
            "reward_best10": reward_info["reward_best10"],
            "reward_drop": reward_info["reward_drop"],
            "baseline_reward_last20": base_reward_info["reward_last20"],
            "reward_delta_last20_vs_baseline": reward_info["reward_last20"] - base_reward_info["reward_last20"],
            "raw_exec_gap_mean": float(np.mean(np.abs(residual_raw - residual_exec))) if residual_raw is not None and residual_exec is not None else float("nan"),
            "residual_exec_abs_mean": float(np.mean(np.abs(residual_exec))) if residual_exec is not None else float("nan"),
            "residual_raw_abs_mean": float(np.mean(np.abs(residual_raw))) if residual_raw is not None else float("nan"),
            "tail_rho_mean": float(np.mean(rho[-max(1, len(rho) // 10) :])) if rho is not None else float("nan"),
            "projection_rate": float(np.mean(projection)) if projection is not None else float("nan"),
        }
        rows.append(row)
    return rows, bundles


def save_summary_csv(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_polymer_overview(rows: list[dict]) -> Path:
    out_path = OUTPUT_DIR / "polymer_residual_combined_overview.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    disturb_rows = [row for row in rows if row["system"] == "polymer" and row["run_mode"] == "disturb"]
    nominal_rows = [row for row in rows if row["system"] == "polymer" and row["run_mode"] == "nominal"]

    ax = axes[0]
    for row in disturb_rows:
        marker = "s" if row["family"] == "combined_disturb" else "o"
        ax.scatter(
            row["tail_mae_delta_vs_baseline"],
            row["reward_delta_last20_vs_baseline"],
            color=FAMILY_COLORS[row["family"]],
            marker=marker,
            s=85,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.annotate(row["short_label"], (row["tail_mae_delta_vs_baseline"], row["reward_delta_last20_vs_baseline"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.axvline(0.0, color="0.4", linewidth=1.0)
    ax.set_title("Polymer disturbance runs: reward gain vs tail-error cost")
    ax.set_xlabel("Tail MAE delta vs disturbance MPC")
    ax.set_ylabel("Final-20 reward delta vs disturbance MPC")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    x = np.arange(len(nominal_rows))
    values = [row["tail_mae_mean"] for row in nominal_rows]
    labels = [row["short_label"] for row in nominal_rows]
    ax.bar(x, values, color=FAMILY_COLORS["residual_nominal"], edgecolor="black", linewidth=0.4)
    ax.axhline(nominal_rows[0]["baseline_tail_mae_mean"], color="tab:red", linestyle="--", linewidth=1.2, label="Nominal MPC tail MAE")
    ax.set_title("Polymer nominal residual runs: tail MAE")
    ax.set_ylabel("Tail physical MAE mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_polymer_tail_traces(bundles: dict[str, dict]) -> Path:
    out_path = OUTPUT_DIR / "polymer_combined_tail_traces.png"
    combined = bundles["polymer_combined_20260426b"]
    residual = bundles["polymer_disturb_20260423"]

    start_c, end_c = last_eval_window(combined)
    start_r, end_r = last_eval_window(residual)
    start = max(start_c, start_r)
    end = min(end_c, end_r)

    y_sp = setpoint_phys(combined)[start:end]
    y_comb = output_array(combined, "y")[start:end]
    y_res = output_array(residual, "y")[start:end]
    y_mpc = output_array(combined, "y_mpc")[start:end]
    dt = float(combined.get("delta_t", 1.0))
    time = np.arange(end - start) * dt
    output_labels = combined.get("system_metadata", {}).get("output_labels", ["y1", "y2"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.plot(time, y_sp[:, idx], color="black", linestyle="--", linewidth=1.2, label="Setpoint")
        ax.plot(time, y_mpc[:, idx], color="0.35", linewidth=1.5, label="MPC baseline")
        ax.plot(time, y_res[:, idx], color=FAMILY_COLORS["residual_disturb"], linewidth=1.4, label="Residual only")
        ax.plot(time, y_comb[:, idx], color=FAMILY_COLORS["combined_disturb"], linewidth=1.4, label="Combined")
        ax.set_ylabel(output_labels[idx])
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Polymer last evaluation window: combined run shows a small late bias")
    axes[-1].set_xlabel(combined.get("system_metadata", {}).get("time_label", "Time"))
    axes[0].legend(loc="best", ncol=4, frameon=False)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distillation_degradation(rows: list[dict], bundles: dict[str, dict]) -> Path:
    out_path = OUTPUT_DIR / "distillation_reward_degradation.png"
    focus = [
        ("distill_td3_20260417", "TD3 2026-04-17"),
        ("distill_td3_20260425", "TD3 2026-04-25"),
        ("distill_sac_20260427", "SAC 2026-04-27"),
    ]
    base_rewards = reward_series(load_pickle(DISTILLATION_BASELINE_FLUCT))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    episodes = np.arange(1, len(base_rewards) + 1)
    ax.plot(episodes, base_rewards, color="0.4", linestyle="--", linewidth=1.2, label="Fluctuation MPC")
    for slug, label in focus:
        rewards = reward_series(bundles[slug])
        row = next(row for row in rows if row["slug"] == slug)
        color = FAMILY_COLORS[row["family"]]
        ax.plot(np.arange(1, len(rewards) + 1), rewards, linewidth=1.6, color=color, label=label)
    ax.set_title("Distillation reward trajectories")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average episode reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    ax = axes[1]
    focus_rows = [next(row for row in rows if row["slug"] == slug) for slug, _ in focus]
    x = np.arange(len(focus_rows))
    abs_t85_offset = [abs(row["tail_offset_output2"]) for row in focus_rows]
    bars = ax.bar(
        x,
        abs_t85_offset,
        color=[FAMILY_COLORS[row["family"]] for row in focus_rows],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axhline(
        abs(next(row for row in focus_rows)["baseline_tail_offset_output2"]),
        color="0.4",
        linestyle=":",
        linewidth=1.1,
        label=r"Baseline $|$T85 tail offset$|$",
    )
    ax.set_title("Late offset severity and reward drop")
    ax.set_ylabel(r"$|$Tail offset on $T_{85}$$|$")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in focus], rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    for bar, row in zip(bars, focus_rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"drop={row['reward_drop']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_projection_gap(rows: list[dict]) -> Path:
    out_path = OUTPUT_DIR / "residual_projection_gap_summary.png"
    focus_slugs = [
        "polymer_disturb_20260422",
        "polymer_disturb_20260423",
        "polymer_combined_20260426b",
        "distill_td3_20260425",
        "distill_sac_20260427",
    ]
    focus_rows = [next(row for row in rows if row["slug"] == slug) for slug in focus_slugs]

    fig, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(focus_rows))
    gap = [row["raw_exec_gap_mean"] for row in focus_rows]
    bars = ax1.bar(x, gap, color="#76B7B2", edgecolor="black", linewidth=0.4)
    ax1.set_title("Residual raw-to-executed action gap remains large")
    ax1.set_ylabel("Mean |raw residual - executed residual|")
    ax1.set_xticks(x)
    ax1.set_xticklabels([row["short_label"] for row in focus_rows], rotation=0)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    rho = [row["tail_rho_mean"] for row in focus_rows]
    ax2.plot(x, rho, color="#E15759", marker="o", linewidth=1.6)
    ax2.set_ylabel("Mean tail rho")
    ax2.set_ylim(0.0, 1.05)

    for bar, row in zip(bars, focus_rows):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"proj={row['projection_rate']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ensure_dirs()
    rows, bundles = build_summary()
    save_summary_csv(rows)
    outputs = [
        plot_polymer_overview(rows),
        plot_polymer_tail_traces(bundles),
        plot_distillation_degradation(rows, bundles),
        plot_projection_gap(rows),
    ]
    print("Saved assets:")
    print(SUMMARY_CSV)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
