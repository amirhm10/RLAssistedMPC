from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_run_history_audit_20260519"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunSpec:
    method: str
    variant: str
    label: str
    path: Path


REPRESENTATIVE_RUNS = [
    RunSpec(
        method="horizon",
        variant="older",
        label="Horizon old",
        path=REPO_ROOT
        / "Distillation/Results/distillation_horizon_disturb_fluctuation_standard_unified/20260416_192434/input_data.pkl",
    ),
    RunSpec(
        method="horizon",
        variant="latest",
        label="Horizon latest",
        path=REPO_ROOT
        / "Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260518_141636/input_data.pkl",
    ),
    RunSpec(
        method="dueling",
        variant="older",
        label="Dueling old",
        path=REPO_ROOT
        / "Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_standard_unified/20260420_173113/input_data.pkl",
    ),
    RunSpec(
        method="dueling",
        variant="latest",
        label="Dueling latest",
        path=REPO_ROOT
        / "Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260518_140746/input_data.pkl",
    ),
    RunSpec(
        method="weights",
        variant="older",
        label="Weights old",
        path=REPO_ROOT
        / "Distillation/Results/distillation_weights_sac_disturb_fluctuation_standard_unified/20260416_192555/input_data.pkl",
    ),
    RunSpec(
        method="weights",
        variant="latest",
        label="Weights latest",
        path=REPO_ROOT
        / "Distillation/Results/distillation_weights_sac_disturb_fluctuation_mismatch_unified/20260518_142138/input_data.pkl",
    ),
    RunSpec(
        method="residual",
        variant="older",
        label="Residual old",
        path=REPO_ROOT
        / "Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260417_181426/input_data.pkl",
    ),
    RunSpec(
        method="residual",
        variant="latest",
        label="Residual latest",
        path=REPO_ROOT
        / "Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260518_135423/input_data.pkl",
    ),
]


SCAN_ROOTS = [
    REPO_ROOT / "Distillation/Results/distillation_horizon_disturb_fluctuation_standard_unified",
    REPO_ROOT / "Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified",
    REPO_ROOT / "Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_standard_unified",
    REPO_ROOT / "Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified",
    REPO_ROOT / "Distillation/Results/distillation_weights_td3_disturb_fluctuation_standard_unified",
    REPO_ROOT / "Distillation/Results/distillation_weights_sac_disturb_fluctuation_standard_unified",
    REPO_ROOT / "Distillation/Results/distillation_weights_sac_disturb_fluctuation_mismatch_unified",
    REPO_ROOT / "Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified",
    REPO_ROOT / "Distillation/Results/distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified",
]


def load_bundle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def reward_signature(bundle: dict) -> str:
    reward_params = bundle.get("reward_params")
    if not isinstance(reward_params, dict) or not reward_params:
        return "not stored in bundle"
    k_rel = np.asarray(reward_params.get("k_rel", []), float).tolist()
    band = np.asarray(reward_params.get("band_floor_phys", []), float).tolist()
    beta = reward_params.get("beta")
    scale = reward_params.get("reward_scale")
    return f"k_rel={k_rel}, band_floor={band}, beta={beta}, scale={scale}"


def bc_enabled(bundle: dict, config_snapshot: dict) -> bool:
    if bool(bundle.get("behavioral_cloning_enabled", False)):
        return True
    bc_cfg = config_snapshot.get("behavioral_cloning", {})
    return bool(isinstance(bc_cfg, dict) and bc_cfg.get("enabled", False))


def summarize_run(spec: RunSpec) -> dict:
    bundle = load_bundle(spec.path)
    cfg = bundle.get("config_snapshot", {})
    y_rl = np.asarray(bundle["y_rl"], float)
    y_mpc = np.asarray(bundle["y_mpc"], float)
    u_rl = np.asarray(bundle["u_rl"], float)
    u_mpc = np.asarray(bundle["u_mpc"], float)
    avg_rewards = np.asarray(bundle.get("avg_rewards", []), float).reshape(-1)
    tail = avg_rewards[-20:] if avg_rewards.size >= 20 else avg_rewards
    return {
        "method": spec.method,
        "variant": spec.variant,
        "label": spec.label,
        "run_dir": spec.path.parent.name,
        "source_path": spec.path.as_posix(),
        "state_mode": cfg.get("state_mode"),
        "agent_kind": cfg.get("agent_kind", bundle.get("agent_kind")),
        "algorithm": cfg.get("algorithm", bundle.get("algorithm")),
        "tail_reward": float(np.mean(tail)) if tail.size else float("nan"),
        "final_reward": float(avg_rewards[-1]) if avg_rewards.size else float("nan"),
        "max_abs_y_rl_minus_mpc": float(np.max(np.abs(y_rl - y_mpc))),
        "max_abs_u_rl_minus_mpc": float(np.max(np.abs(u_rl - u_mpc))),
        "observer_update_alignment": cfg.get("observer_update_alignment"),
        "use_shifted_mpc_warm_start": bool(cfg.get("use_shifted_mpc_warm_start", False)),
        "append_rho_to_state": cfg.get("append_rho_to_state"),
        "authority_use_rho": cfg.get("authority_use_rho", cfg.get("use_rho_authority")),
        "post_warm_start_action_freeze_subepisodes": cfg.get("post_warm_start_action_freeze_subepisodes"),
        "post_warm_start_actor_freeze_subepisodes": cfg.get("post_warm_start_actor_freeze_subepisodes"),
        "behavioral_cloning_enabled": bc_enabled(bundle, cfg),
        "reward_signature": reward_signature(bundle),
    }


def scan_all_runs() -> dict:
    run_count = 0
    nonzero_runs: list[dict] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            pkl_path = run_dir / "input_data.pkl"
            if not pkl_path.exists():
                continue
            bundle = load_bundle(pkl_path)
            y_diff = float(np.max(np.abs(np.asarray(bundle["y_rl"], float) - np.asarray(bundle["y_mpc"], float))))
            u_diff = float(np.max(np.abs(np.asarray(bundle["u_rl"], float) - np.asarray(bundle["u_mpc"], float))))
            run_count += 1
            if y_diff != 0.0 or u_diff != 0.0:
                nonzero_runs.append(
                    {
                        "source_path": pkl_path.as_posix(),
                        "max_abs_y_rl_minus_mpc": y_diff,
                        "max_abs_u_rl_minus_mpc": u_diff,
                    }
                )
    return {"scanned_run_count": run_count, "nonzero_rl_vs_mpc_run_count": len(nonzero_runs), "nonzero_runs": nonzero_runs}


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], overall: dict, path: Path) -> None:
    labels = [row["label"] for row in rows]
    tail_rewards = [row["tail_reward"] for row in rows]
    y_diffs = [row["max_abs_y_rl_minus_mpc"] for row in rows]
    u_diffs = [row["max_abs_u_rl_minus_mpc"] for row in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    color_map = {"older": "#2a6f97", "latest": "#c44536"}
    colors = [color_map[row["variant"]] for row in rows]
    axes[0].bar(x, tail_rewards, color=colors)
    axes[0].set_ylabel("Tail avg reward")
    axes[0].set_title("Representative older vs latest distillation run rewards")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    for idx, row in enumerate(rows):
        axes[0].text(idx, tail_rewards[idx], f"{tail_rewards[idx]:.2f}", ha="center", va="bottom", fontsize=8)

    width = 0.36
    axes[1].bar(x - width / 2.0, y_diffs, width=width, label="max |y_rl - y_mpc|", color="#577590")
    axes[1].bar(x + width / 2.0, u_diffs, width=width, label="max |u_rl - u_mpc|", color="#f3722c")
    axes[1].set_ylabel("Absolute difference")
    axes[1].set_title(
        f"Stored RL-vs-MPC differences in representative runs (all scanned runs with nonzero diffs: {overall['nonzero_rl_vs_mpc_run_count']}/{overall['scanned_run_count']})"
    )
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0.0, 1.0)
    for idx, (y_diff, u_diff) in enumerate(zip(y_diffs, u_diffs)):
        axes[1].text(idx - width / 2.0, y_diff + 0.02, f"{y_diff:.1f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(idx + width / 2.0, u_diff + 0.02, f"{u_diff:.1f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    representative_rows = [summarize_run(spec) for spec in REPRESENTATIVE_RUNS]
    overall = scan_all_runs()

    write_csv(representative_rows, OUT_DIR / "representative_run_summary.csv")
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"representative_runs": representative_rows, "overall_scan": overall}, handle, indent=2)
    make_figure(representative_rows, overall, OUT_DIR / "fig_reward_and_identity_audit.png")


if __name__ == "__main__":
    main()
