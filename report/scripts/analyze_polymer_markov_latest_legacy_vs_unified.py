from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc"
UNIFIED_ROOT = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb"
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_legacy_vs_unified_20260510"


ACTION_SOURCE_LABELS = {
    0: "No Markov",
    1: "Warm-start LS",
    2: "TD3 accepted",
    3: "LS fallback",
    4: "Nominal fallback",
    5: "LS no-RL",
}


@dataclass
class RunData:
    label: str
    run_dir: Path
    stage_csv: Path
    arrays: dict[str, np.ndarray]


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _latest_run(root: Path) -> Path:
    candidates = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "markov_stage_diagnostics.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No markov_stage_diagnostics.csv runs found under {root}")
    return candidates[-1]


def _load_stage_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    arrays: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        if key == "action_source_name":
            arrays[key] = np.asarray([row[key] for row in rows], dtype=object)
        else:
            arrays[key] = np.asarray([_parse_float(row[key]) for row in rows], dtype=float)
    return arrays


def _load_episode_count(legacy_run_dir: Path, n_steps: int) -> int:
    episode_csv = legacy_run_dir / "episode_average_rewards.csv"
    if episode_csv.exists():
        with episode_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            count = sum(1 for _ in reader)
        if count > 0:
            return int(count)
    return 50 if n_steps % 50 == 0 else 1


def _episode_mean(values: np.ndarray, episode_len: int) -> np.ndarray:
    n_episodes = len(values) // episode_len
    return np.asarray(
        [np.nanmean(values[i * episode_len : (i + 1) * episode_len]) for i in range(n_episodes)],
        dtype=float,
    )


def _episode_fraction(values: np.ndarray, target: int, episode_len: int) -> np.ndarray:
    n_episodes = len(values) // episode_len
    return np.asarray(
        [np.mean(values[i * episode_len : (i + 1) * episode_len] == target) for i in range(n_episodes)],
        dtype=float,
    )


def _first_step(values: np.ndarray, target: int) -> int | None:
    idx = np.where(values == target)[0]
    return None if idx.size == 0 else int(idx[0])


def _summary_dict(run: RunData, episode_len: int) -> dict:
    src = run.arrays["action_source"].astype(int)
    executed_score = run.arrays["executed_prediction_score"]
    requested_score = run.arrays["requested_prediction_score"]
    ls_score = run.arrays["ls_prediction_score"]

    return {
        "label": run.label,
        "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
        "n_steps": int(len(src)),
        "episode_len": int(episode_len),
        "n_episodes": int(len(src) // episode_len),
        "first_td3_step": _first_step(src, 2),
        "first_ls_step": _first_step(src, 1),
        "first_ls_fallback_step": _first_step(src, 3),
        "first_nominal_fallback_step": _first_step(src, 4),
        "mean_executed_z_norm": float(np.nanmean(run.arrays["executed_z_norm"])),
        "mean_executed_gain_drift": float(np.nanmean(run.arrays["executed_gain_drift"])),
        "mean_executed_first_move_diff_norm": float(np.nanmean(run.arrays["executed_first_move_diff_norm"])),
        "mean_executed_full_sequence_diff_norm": float(np.nanmean(run.arrays["executed_full_sequence_diff_norm"])),
        "mean_executed_cost_margin": float(np.nanmean(run.arrays["executed_cost_margin"])),
        "mean_executed_prediction_score": float(np.nanmean(executed_score)),
        "mean_requested_prediction_score": float(np.nanmean(requested_score)),
        "mean_ls_prediction_score": float(np.nanmean(ls_score)),
        "overall_td3_fraction": float(np.mean(src == 2)),
        "overall_ls_fallback_fraction": float(np.mean(src == 3)),
        "overall_nominal_fallback_fraction": float(np.mean(src == 4)),
        "overall_warm_start_ls_fraction": float(np.mean(src == 1)),
        "tail_td3_fraction_last_5000": float(np.mean(src[-5000:] == 2)),
        "tail_ls_fallback_fraction_last_5000": float(np.mean(src[-5000:] == 3)),
        "tail_nominal_fallback_fraction_last_5000": float(np.mean(src[-5000:] == 4)),
    }


def _plot_metric_grid(legacy: RunData, unified: RunData, episode_len: int, out_dir: Path) -> list[Path]:
    episodes = np.arange(1, len(legacy.arrays["step"]) // episode_len + 1)
    fig, axs = plt.subplots(3, 2, figsize=(13.5, 11.5), sharex=True)
    axs = axs.reshape(-1)

    metric_specs = [
        ("executed_z_norm", "Executed z-norm"),
        ("executed_gain_drift", "Executed gain drift"),
        ("executed_first_move_diff_norm", "Executed first-move diff norm"),
        ("executed_full_sequence_diff_norm", "Executed full-sequence diff norm"),
        ("executed_cost_margin", "Executed nominal-cost margin"),
    ]

    for ax, (key, title) in zip(axs[:5], metric_specs):
        leg = _episode_mean(legacy.arrays[key], episode_len)
        uni = _episode_mean(unified.arrays[key], episode_len)
        ax.plot(episodes, leg, label="Legacy", linewidth=2.1, color="#0B6E4F")
        ax.plot(episodes, uni, label="Unified", linewidth=2.1, color="#C84C09")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axs[5]
    for label, source, color, style in [
        ("Legacy TD3", 2, "#0B6E4F", "-"),
        ("Legacy LS", 3, "#0B6E4F", "--"),
        ("Legacy Nominal", 4, "#0B6E4F", ":"),
        ("Unified TD3", 2, "#C84C09", "-"),
        ("Unified LS", 3, "#C84C09", "--"),
        ("Unified Nominal", 4, "#C84C09", ":"),
    ]:
        run = legacy if label.startswith("Legacy") else unified
        ax.plot(
            episodes,
            _episode_fraction(run.arrays["action_source"].astype(int), source, episode_len),
            label=label,
            linewidth=1.8,
            linestyle=style,
            color=color,
        )
    ax.set_title("Action-source fractions")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ax in axs:
        ax.set_xlabel("Episode")

    axs[0].legend(loc="best", fontsize=9)
    axs[5].legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = out_dir / "legacy_vs_unified_metric_grid.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def _plot_prediction_diagnostics(legacy: RunData, unified: RunData, episode_len: int, out_dir: Path) -> list[Path]:
    episodes = np.arange(1, len(legacy.arrays["step"]) // episode_len + 1)
    fig, axs = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)

    for ax, run, color in [(axs[0], legacy, "#0B6E4F"), (axs[1], unified, "#C84C09")]:
        exec_ep = _episode_mean(run.arrays["executed_prediction_score"], episode_len)
        ls_ep = _episode_mean(run.arrays["ls_prediction_score"], episode_len)
        req_ep = _episode_mean(run.arrays["requested_prediction_score"], episode_len)
        td3_ep = _episode_fraction(run.arrays["action_source"].astype(int), 2, episode_len)
        nom_ep = _episode_fraction(run.arrays["action_source"].astype(int), 4, episode_len)

        ax.plot(episodes, exec_ep, label=f"{run.label} executed score", linewidth=2.0, color=color)
        ax.plot(episodes, ls_ep, label=f"{run.label} LS score", linewidth=1.8, color=color, linestyle="--")
        ax.plot(episodes, req_ep, label=f"{run.label} requested score", linewidth=1.5, color=color, linestyle=":")
        ax2 = ax.twinx()
        ax2.plot(episodes, td3_ep, color="#1f77b4", alpha=0.55, linewidth=1.4, label="TD3 frac")
        ax2.plot(episodes, nom_ep, color="#d62728", alpha=0.55, linewidth=1.4, linestyle="--", label="Nominal frac")
        ax.set_ylabel("Prediction score")
        ax2.set_ylabel("Fraction")
        ax.set_title(f"{run.label}: prediction score and action mix")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    axs[1].set_xlabel("Episode")
    fig.tight_layout()
    out_path = out_dir / "legacy_vs_unified_prediction_diagnostics.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    legacy_dir = _latest_run(LEGACY_ROOT)
    unified_dir = _latest_run(UNIFIED_ROOT)
    legacy = RunData(
        label="Legacy",
        run_dir=legacy_dir,
        stage_csv=legacy_dir / "markov_stage_diagnostics.csv",
        arrays=_load_stage_csv(legacy_dir / "markov_stage_diagnostics.csv"),
    )
    unified = RunData(
        label="Unified",
        run_dir=unified_dir,
        stage_csv=unified_dir / "markov_stage_diagnostics.csv",
        arrays=_load_stage_csv(unified_dir / "markov_stage_diagnostics.csv"),
    )

    episode_count = _load_episode_count(legacy_dir, len(legacy.arrays["step"]))
    episode_len = len(legacy.arrays["step"]) // episode_count

    figure_paths: list[Path] = []
    figure_paths.extend(_plot_metric_grid(legacy, unified, episode_len, OUT_DIR))
    figure_paths.extend(_plot_prediction_diagnostics(legacy, unified, episode_len, OUT_DIR))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "legacy": _summary_dict(legacy, episode_len),
        "unified": _summary_dict(unified, episode_len),
        "figure_paths": [str(path.relative_to(REPO_ROOT)) for path in figure_paths],
    }
    with (OUT_DIR / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
