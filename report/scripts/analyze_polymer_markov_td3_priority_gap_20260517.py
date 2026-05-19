from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_td3_priority_gap_20260517"
LATEST_ROOT = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008"
TD3_ONLY_ROOT = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008_looser_gate_only_td3_only"
GUARDED_ROOT = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008_looser_gate_only"
TAIL_EPISODES = 20


@dataclass
class RunData:
    label: str
    run_dir: Path
    bundle: dict
    diagnostics: pd.DataFrame | None

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle.get("avg_rewards", []), dtype=float)

    @property
    def n_episodes(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def steps_per_episode(self) -> int:
        if self.diagnostics is None or self.n_episodes <= 0:
            return int(self.bundle.get("time_in_sub_episodes", 0))
        return int(len(self.diagnostics) // self.n_episodes)


def latest_dir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped result folders under {root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def load_run(label: str, root: Path, with_diagnostics: bool = True) -> RunData:
    run_dir = latest_dir(root)
    with (run_dir / "input_data.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    diagnostics = None
    diag_path = run_dir / "markov_stage_diagnostics.csv"
    if with_diagnostics and diag_path.exists():
        diagnostics = pd.read_csv(diag_path)
    return RunData(label=label, run_dir=run_dir, bundle=bundle, diagnostics=diagnostics)


def action_source_fraction(run: RunData, tail_episodes: int | None = None) -> dict[str, float]:
    df = run.diagnostics
    if df is None:
        source = np.asarray(run.bundle.get("rl_action_source_log", []), dtype=int)
        names = run.bundle.get("rl_action_source_names", {})
        if tail_episodes is not None and run.steps_per_episode > 0:
            source = source[-tail_episodes * run.steps_per_episode :]
        return {
            str(names.get(int(code), code)): float(np.mean(source == code))
            for code in sorted(np.unique(source))
        }
    use = df.copy()
    if tail_episodes is not None and run.steps_per_episode > 0:
        start = max(0, run.n_episodes - tail_episodes) * run.steps_per_episode
        use = use[use["step"] >= start]
    return use["action_source_name"].value_counts(normalize=True).to_dict()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = load_run("Latest unified run", LATEST_ROOT, with_diagnostics=True)
    td3_only = load_run("TD3-only no safeguard", TD3_ONLY_ROOT, with_diagnostics=False)
    guarded = load_run("Previous guarded sibling", GUARDED_ROOT, with_diagnostics=False)

    df = latest.diagnostics
    assert df is not None
    tail_start = max(0, latest.n_episodes - TAIL_EPISODES) * latest.steps_per_episode
    tail = df[df["step"] >= tail_start].copy()

    old_score_pass = tail["requested_prediction_score"].to_numpy(dtype=float) > 0.0
    old_drift_pass = tail["requested_gain_drift"].to_numpy(dtype=float) <= float(
        latest.bundle.get("markov_gain_drift_max", 0.10)
    )
    old_cost_pass = tail["requested_cost_guard_pass"].to_numpy(dtype=float) > 0.5
    priority_abs_cap_pass = tail["requested_cost_margin"].to_numpy(dtype=float) <= 0.10
    finite_td3_candidate = np.isfinite(tail["requested_prediction_score"].to_numpy(dtype=float))

    source_names = [
        "td3_accepted",
        "ls_fallback",
        "nominal_fallback",
        "warm_start_ls",
        "nominal_no_markov",
    ]
    all_frac = action_source_fraction(latest, tail_episodes=None)
    tail_frac = action_source_fraction(latest, tail_episodes=TAIL_EPISODES)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    x = np.arange(len(source_names))
    axes[0].bar(x - 0.18, [all_frac.get(name, 0.0) for name in source_names], width=0.36, label="All steps")
    axes[0].bar(x + 0.18, [tail_frac.get(name, 0.0) for name in source_names], width=0.36, label="Tail 20")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["TD3", "LS", "Nominal", "Warm LS", "No Markov"], rotation=25, ha="right"
    )
    axes[0].set_ylabel("Fraction of executed steps")
    axes[0].set_title("Latest run still executed mostly LS")
    axes[0].legend(frameon=False)

    gate_labels = ["Score > 0", "Drift <= 0.10", "Old cost guard", "Priority abs cap", "Finite TD3"]
    gate_values = [
        float(np.mean(old_score_pass)),
        float(np.mean(old_drift_pass)),
        float(np.mean(old_cost_pass)),
        float(np.mean(priority_abs_cap_pass)),
        float(np.mean(finite_td3_candidate)),
    ]
    axes[1].bar(np.arange(len(gate_labels)), gate_values, color=["#b84a62", "#4c78a8", "#4c78a8", "#54a24b", "#54a24b"])
    axes[1].set_xticks(np.arange(len(gate_labels)))
    axes[1].set_xticklabels(gate_labels, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Tail-20 pass fraction")
    axes[1].set_title("Old score veto, not cost or drift, blocked TD3")
    fig.savefig(OUT_DIR / "fig_latest_priority_gap_action_sources.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    for run, color in [
        (latest, "#b84a62"),
        (td3_only, "#54a24b"),
        (guarded, "#4c78a8"),
    ]:
        rewards = run.avg_rewards
        ax.plot(np.arange(1, rewards.size + 1), rewards, label=run.label, linewidth=1.8, color=color)
        if rewards.size >= TAIL_EPISODES:
            ax.hlines(
                float(np.mean(rewards[-TAIL_EPISODES:])),
                rewards.size - TAIL_EPISODES + 1,
                rewards.size,
                color=color,
                linestyle="--",
                linewidth=1.2,
            )
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward")
    ax.set_title("Latest unified run did not reproduce TD3-only authority")
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_reward_latest_vs_td3_only.png", dpi=180)
    plt.close(fig)

    summary = {
        "latest_run": str(latest.run_dir.relative_to(REPO_ROOT)),
        "td3_only_run": str(td3_only.run_dir.relative_to(REPO_ROOT)),
        "guarded_run": str(guarded.run_dir.relative_to(REPO_ROOT)),
        "latest_td3_priority_fallback": latest.bundle.get("td3_priority_fallback", {}),
        "latest_summary_priority_enabled": latest.bundle.get("summary_metrics", {}).get(
            "td3_priority_fallback_enabled"
        ),
        "latest_behavioral_cloning_enabled": bool(latest.bundle.get("behavioral_cloning_enabled", False)),
        "latest_tail20_action_source_fraction": {k: float(v) for k, v in tail_frac.items()},
        "latest_all_action_source_fraction": {k: float(v) for k, v in all_frac.items()},
        "latest_tail20_score_positive_fraction": gate_values[0],
        "latest_tail20_drift_pass_fraction": gate_values[1],
        "latest_tail20_old_cost_guard_pass_fraction": gate_values[2],
        "latest_tail20_priority_abs_cap_pass_fraction": gate_values[3],
        "latest_tail20_requested_cost_margin_max": float(
            np.nanmax(tail["requested_cost_margin"].to_numpy(dtype=float))
        ),
        "latest_tail20_requested_gain_drift_max": float(
            np.nanmax(tail["requested_gain_drift"].to_numpy(dtype=float))
        ),
        "latest_tail20_reward_mean": float(np.nanmean(latest.avg_rewards[-TAIL_EPISODES:])),
        "td3_only_tail20_reward_mean": float(np.nanmean(td3_only.avg_rewards[-TAIL_EPISODES:])),
        "guarded_tail20_reward_mean": float(np.nanmean(guarded.avg_rewards[-TAIL_EPISODES:])),
        "latest_final_reward": float(latest.avg_rewards[-1]),
        "td3_only_final_reward": float(td3_only.avg_rewards[-1]),
        "guarded_final_reward": float(guarded.avg_rewards[-1]),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
