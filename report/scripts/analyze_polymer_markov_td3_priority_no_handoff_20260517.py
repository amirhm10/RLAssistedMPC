from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_markov_td3_priority_no_handoff_20260517"
TAIL_EPISODES = 20

RUNS = {
    "priority": REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008" / "20260517_235822",
    "no_pass": REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb_zbound_008" / "20260517_210211",
    "td3_only": REPO_ROOT
    / "Polymer"
    / "Results"
    / "td3_markov_disturb_zbound_008_looser_gate_only_td3_only"
    / "20260516_000338",
}


@dataclass
class RunData:
    key: str
    label: str
    run_dir: Path
    bundle: dict
    diagnostics: pd.DataFrame

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle.get("avg_rewards", []), dtype=float)

    @property
    def n_episodes(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def steps_per_episode(self) -> int:
        if self.n_episodes <= 0:
            return int(self.bundle.get("time_in_sub_episodes", 0))
        return int(len(self.diagnostics) // self.n_episodes)


def load_run(key: str, label: str, run_dir: Path) -> RunData:
    with (run_dir / "input_data.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    return RunData(
        key=key,
        label=label,
        run_dir=run_dir,
        bundle=bundle,
        diagnostics=pd.read_csv(run_dir / "markov_stage_diagnostics.csv"),
    )


def episode_fraction(run: RunData, source_name: str) -> np.ndarray:
    names = run.diagnostics["action_source_name"].to_numpy(dtype=str)
    steps = run.steps_per_episode
    names = names[: run.n_episodes * steps].reshape(run.n_episodes, steps)
    return np.mean(names == source_name, axis=1)


def action_source_fraction(run: RunData, tail_episodes: int | None = None) -> dict[str, float]:
    use = run.diagnostics
    if tail_episodes is not None:
        start = max(0, run.n_episodes - tail_episodes) * run.steps_per_episode
        use = use[use["step"] >= start]
    return {str(k): float(v) for k, v in use["action_source_name"].value_counts(normalize=True).items()}


def mean_tail(values: np.ndarray, tail: int = TAIL_EPISODES) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values[-tail:]))


def diagnostics_summary(run: RunData, tail_episodes: int | None = None) -> dict[str, float]:
    use = run.diagnostics
    if tail_episodes is not None:
        start = max(0, run.n_episodes - tail_episodes) * run.steps_per_episode
        use = use[use["step"] >= start]
    out = {
        "requested_score_mean": float(np.nanmean(use["requested_prediction_score"])),
        "requested_score_positive_fraction": float(np.mean(use["requested_prediction_score"] > 0.0)),
        "requested_gain_drift_mean": float(np.nanmean(use["requested_gain_drift"])),
        "requested_gain_drift_max": float(np.nanmax(use["requested_gain_drift"])),
        "requested_cost_margin_mean": float(np.nanmean(use["requested_cost_margin"])),
        "requested_cost_margin_max": float(np.nanmax(use["requested_cost_margin"])),
        "requested_cost_guard_pass_fraction": float(np.mean(use["requested_cost_guard_pass"] == 1)),
        "executed_gain_drift_mean": float(np.nanmean(use["executed_gain_drift"])),
        "executed_cost_margin_mean": float(np.nanmean(use["executed_cost_margin"])),
    }
    return out


def plot_rewards_and_authority(runs: list[RunData]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True, constrained_layout=True)
    colors = {"priority": "#1f9d75", "no_pass": "#4c78a8", "td3_only": "#b84a62"}
    for run in runs:
        x = np.arange(1, run.n_episodes + 1)
        axes[0].plot(x, run.avg_rewards, label=run.label, color=colors[run.key], linewidth=1.8)
        axes[0].hlines(
            mean_tail(run.avg_rewards),
            max(1, run.n_episodes - TAIL_EPISODES + 1),
            run.n_episodes,
            colors=colors[run.key],
            linestyles="--",
            linewidth=1.2,
        )
        axes[1].plot(x, episode_fraction(run, "td3_accepted"), label=run.label, color=colors[run.key], linewidth=1.8)
    axes[0].set_ylabel("Average reward")
    axes[0].set_title("TD3-priority recovers TD3 authority before soft handoff")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("TD3 accepted fraction")
    axes[1].set_xlabel("Subepisode")
    axes[1].set_ylim(-0.03, 1.03)
    fig.savefig(OUT_DIR / "fig_reward_and_td3_authority.png", dpi=180)
    plt.close(fig)


def plot_priority_mechanism(priority: RunData, no_pass: RunData) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    source_order = ["td3_accepted", "ls_fallback", "nominal_fallback", "warm_start_ls"]
    x = np.arange(len(source_order))
    pri_tail = action_source_fraction(priority, TAIL_EPISODES)
    old_tail = action_source_fraction(no_pass, TAIL_EPISODES)
    axes[0].bar(x - 0.18, [pri_tail.get(name, 0.0) for name in source_order], 0.36, label="Priority run")
    axes[0].bar(x + 0.18, [old_tail.get(name, 0.0) for name in source_order], 0.36, label="No pass-through run")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["TD3", "LS", "Nominal", "Warm LS"], rotation=20, ha="right")
    axes[0].set_ylabel("Tail-20 fraction")
    axes[0].set_title("Priority mode removes LS domination")
    axes[0].legend(frameon=False)

    tail = priority.diagnostics.tail(TAIL_EPISODES * priority.steps_per_episode)
    gate_labels = ["Score > 0", "Drift <= 0.10", "Cost guard", "Full cap"]
    gate_values = [
        float(np.mean(tail["requested_prediction_score"] > 0.0)),
        float(np.mean(tail["requested_gain_drift"] <= 0.10)),
        float(np.mean(tail["requested_cost_guard_pass"] == 1)),
        float(np.mean(tail["requested_cost_margin"] <= 0.10)),
    ]
    axes[1].bar(np.arange(len(gate_labels)), gate_values, color=["#b84a62", "#54a24b", "#54a24b", "#54a24b"])
    axes[1].set_xticks(np.arange(len(gate_labels)))
    axes[1].set_xticklabels(gate_labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Tail-20 pass fraction")
    axes[1].set_title("Old score veto would still reject most TD3")
    fig.savefig(OUT_DIR / "fig_priority_mechanism.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [
        load_run("priority", "TD3-priority, no soft handoff", RUNS["priority"]),
        load_run("no_pass", "No pass-through", RUNS["no_pass"]),
        load_run("td3_only", "TD3-only no safeguard", RUNS["td3_only"]),
    ]
    plot_rewards_and_authority(runs)
    plot_priority_mechanism(runs[0], runs[1])

    summary = {}
    for run in runs:
        summary[run.key] = {
            "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
            "td3_priority_fallback": run.bundle.get("td3_priority_fallback"),
            "summary_metrics": run.bundle.get("summary_metrics", {}),
            "behavioral_cloning_enabled": bool(run.bundle.get("behavioral_cloning_enabled", False)),
            "has_soft_handoff_logs": bool("td3_authority_scale_log" in run.bundle),
            "reward_mean": float(np.nanmean(run.avg_rewards)),
            "tail20_reward_mean": mean_tail(run.avg_rewards),
            "final_reward": float(run.avg_rewards[-1]),
            "action_source_all": action_source_fraction(run),
            "action_source_tail20": action_source_fraction(run, TAIL_EPISODES),
            "diagnostics_all": diagnostics_summary(run),
            "diagnostics_tail20": diagnostics_summary(run, TAIL_EPISODES),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
