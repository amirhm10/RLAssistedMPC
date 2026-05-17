from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_markov_td3_decision_authority_20260517"
WINDOW = 9
TAIL_EPISODES = 20

RUN_SPECS = [
    {
        "key": "guarded",
        "label": "Guarded",
        "notebook": "distillation_RL_assisted_MPC_markov_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_unified",
        "color": "#0B6E4F",
        "s_pred_min": 1.0e-6,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.10,
        "nominal_cost_absolute_tol": 1.0e-8,
    },
    {
        "key": "relaxed",
        "label": "Relaxed",
        "notebook": "distillation_RL_assisted_MPC_markov_relaxed_acceptance_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_relaxed_acceptance_unified",
        "color": "#1f77b4",
        "s_pred_min": 5.0e-7,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.15,
        "nominal_cost_absolute_tol": 1.0e-8,
    },
    {
        "key": "ls_only",
        "label": "LS only",
        "notebook": "distillation_RL_assisted_MPC_markov_ls_only_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_ls_only_unified",
        "color": "#F59E0B",
        "s_pred_min": 1.0e-6,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.10,
        "nominal_cost_absolute_tol": 1.0e-8,
    },
    {
        "key": "td3_without_ls",
        "label": "TD3 no LS",
        "notebook": "distillation_RL_assisted_MPC_markov_td3_without_ls_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_td3_without_ls_unified",
        "color": "#C2410C",
        "s_pred_min": 1.0e-6,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.10,
        "nominal_cost_absolute_tol": 1.0e-8,
    },
    {
        "key": "td3_only",
        "label": "TD3 only",
        "notebook": "distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified",
        "color": "#7A1FA2",
        "s_pred_min": 1.0e-6,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.10,
        "nominal_cost_absolute_tol": 1.0e-8,
    },
]

GATE_COUNTERFACTUALS = [
    {
        "key": "guarded_gate",
        "label": "Guarded gate",
        "s_pred_min": 1.0e-6,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.10,
        "nominal_cost_absolute_tol": 1.0e-8,
    },
    {
        "key": "relaxed_gate",
        "label": "Relaxed gate",
        "s_pred_min": 5.0e-7,
        "gain_drift_max": 0.10,
        "nominal_cost_relative_tol": 0.15,
        "nominal_cost_absolute_tol": 1.0e-8,
    },
]


@dataclass
class RunData:
    key: str
    label: str
    color: str
    notebook: str
    run_dir: Path
    bundle: dict
    stage_df: pd.DataFrame
    s_pred_min: float
    gain_drift_max: float
    nominal_cost_relative_tol: float
    nominal_cost_absolute_tol: float

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle["avg_rewards"], dtype=float)

    @property
    def n_episodes(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def steps_per_episode(self) -> int:
        return int(self.bundle["time_in_sub_episodes"])

    @property
    def warm_start_step(self) -> int:
        return int(self.bundle.get("warm_start_step", 0))

    @property
    def warm_start_episodes(self) -> int:
        return int(self.warm_start_step // self.steps_per_episode)

    @property
    def action_source(self) -> np.ndarray:
        return np.asarray(self.bundle["rl_action_source_log"], dtype=int).reshape(self.n_episodes, self.steps_per_episode)

    @property
    def td3_fraction_per_episode(self) -> np.ndarray:
        return np.mean(self.action_source == 2, axis=1)

    @property
    def ls_fraction_per_episode(self) -> np.ndarray:
        return np.mean((self.action_source == 3) | (self.action_source == 5), axis=1)

    @property
    def nominal_fraction_per_episode(self) -> np.ndarray:
        return np.mean((self.action_source == 0) | (self.action_source == 4), axis=1)

    @property
    def bc_config(self) -> dict:
        return dict(self.bundle.get("behavioral_cloning", {}))


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def latest_run_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return candidates[-1]


def moving_average(values: np.ndarray, width: int = WINDOW) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < width:
        return arr.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(arr, kernel, mode="same")


def build_run_data() -> dict[str, RunData]:
    runs: dict[str, RunData] = {}
    for spec in RUN_SPECS:
        run_dir = latest_run_dir(spec["root"])
        runs[spec["key"]] = RunData(
            key=spec["key"],
            label=spec["label"],
            color=spec["color"],
            notebook=spec["notebook"],
            run_dir=run_dir,
            bundle=load_pickle(run_dir / "input_data.pkl"),
            stage_df=pd.read_csv(run_dir / "markov_stage_diagnostics.csv"),
            s_pred_min=float(spec["s_pred_min"]),
            gain_drift_max=float(spec["gain_drift_max"]),
            nominal_cost_relative_tol=float(spec["nominal_cost_relative_tol"]),
            nominal_cost_absolute_tol=float(spec["nominal_cost_absolute_tol"]),
        )
    return runs


def requested_stage_df(run: RunData) -> pd.DataFrame:
    df = run.stage_df.copy()
    df = df.loc[df.index > run.warm_start_step].copy()
    return df[np.isfinite(df["requested_prediction_score"].to_numpy(dtype=float))].copy()


def gate_masks(
    df: pd.DataFrame,
    *,
    s_pred_min: float,
    gain_drift_max: float,
    nominal_cost_relative_tol: float,
    nominal_cost_absolute_tol: float,
) -> dict[str, np.ndarray]:
    score = df["requested_prediction_score"].to_numpy(dtype=float)
    drift = df["requested_gain_drift"].to_numpy(dtype=float)
    cost_margin = df["requested_cost_margin"].to_numpy(dtype=float)
    nominal_cost = df["nominal_cost"].to_numpy(dtype=float)

    score_pass = score > float(s_pred_min)
    drift_pass = drift <= float(gain_drift_max)
    cost_guard = cost_margin <= (
        float(nominal_cost_absolute_tol) + float(nominal_cost_relative_tol) * np.abs(nominal_cost)
    )
    pass_all = score_pass & drift_pass & cost_guard
    return {
        "score_pass": score_pass,
        "drift_pass": drift_pass,
        "cost_pass": cost_guard,
        "pass_all": pass_all,
    }


def gate_breakdown(masks: dict[str, np.ndarray]) -> dict[str, float]:
    score_pass = masks["score_pass"]
    drift_pass = masks["drift_pass"]
    cost_pass = masks["cost_pass"]
    pass_all = masks["pass_all"]

    fail_score_only = (~score_pass) & drift_pass & cost_pass
    fail_cost_only = score_pass & drift_pass & (~cost_pass)
    fail_score_and_cost = (~score_pass) & drift_pass & (~cost_pass)
    fail_other = ~(pass_all | fail_score_only | fail_cost_only | fail_score_and_cost)

    def frac(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if mask.size else np.nan

    return {
        "pass_all": frac(pass_all),
        "fail_score_only": frac(fail_score_only),
        "fail_cost_only": frac(fail_cost_only),
        "fail_score_and_cost": frac(fail_score_and_cost),
        "fail_other": frac(fail_other),
    }


def per_episode_pass_fraction(
    run: RunData,
    *,
    s_pred_min: float,
    gain_drift_max: float,
    nominal_cost_relative_tol: float,
    nominal_cost_absolute_tol: float,
) -> np.ndarray:
    requested = requested_stage_df(run)
    masks = gate_masks(
        requested,
        s_pred_min=s_pred_min,
        gain_drift_max=gain_drift_max,
        nominal_cost_relative_tol=nominal_cost_relative_tol,
        nominal_cost_absolute_tol=nominal_cost_absolute_tol,
    )
    episode_idx = (requested.index.to_numpy(dtype=int) // run.steps_per_episode).astype(int)
    values = np.full(run.n_episodes, np.nan, dtype=float)
    for ep in range(run.n_episodes):
        sel = episode_idx == ep
        if np.any(sel):
            values[ep] = float(np.mean(masks["pass_all"][sel]))
    return values


def summarize_run(run: RunData) -> dict:
    requested = requested_stage_df(run)
    masks = gate_masks(
        requested,
        s_pred_min=run.s_pred_min,
        gain_drift_max=run.gain_drift_max,
        nominal_cost_relative_tol=run.nominal_cost_relative_tol,
        nominal_cost_absolute_tol=run.nominal_cost_absolute_tol,
    )
    breakdown = gate_breakdown(masks)

    tail = slice(max(0, run.n_episodes - TAIL_EPISODES), run.n_episodes)
    tail_requested = requested.iloc[-TAIL_EPISODES * run.steps_per_episode :]
    tail_masks = gate_masks(
        tail_requested,
        s_pred_min=run.s_pred_min,
        gain_drift_max=run.gain_drift_max,
        nominal_cost_relative_tol=run.nominal_cost_relative_tol,
        nominal_cost_absolute_tol=run.nominal_cost_absolute_tol,
    )

    bc_cfg = run.bc_config
    bc_active = np.asarray(bc_cfg.get("active_log", []), dtype=float)
    summary = {
        "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
        "notebook": run.notebook,
        "n_episodes": run.n_episodes,
        "steps_per_episode": run.steps_per_episode,
        "warm_start_step": run.warm_start_step,
        "warm_start_episodes": run.warm_start_episodes,
        "reward_final": float(run.avg_rewards[-1]),
        "reward_tail20": float(np.mean(run.avg_rewards[tail])),
        "reward_best": float(np.max(run.avg_rewards)),
        "reward_best_episode": int(np.argmax(run.avg_rewards) + 1),
        "td3_fraction_tail20": float(np.mean(run.action_source[tail] == 2)),
        "ls_fraction_tail20": float(np.mean((run.action_source[tail] == 3) | (run.action_source[tail] == 5))),
        "nominal_fraction_tail20": float(
            np.mean((run.action_source[tail] == 0) | (run.action_source[tail] == 4))
        ),
        "postwarm_requested_steps": int(requested.shape[0]),
        "postwarm_gate_pass_fraction": float(np.mean(masks["pass_all"])) if requested.shape[0] else np.nan,
        "postwarm_score_pass_fraction": float(np.mean(masks["score_pass"])) if requested.shape[0] else np.nan,
        "postwarm_drift_pass_fraction": float(np.mean(masks["drift_pass"])) if requested.shape[0] else np.nan,
        "postwarm_cost_pass_fraction": float(np.mean(masks["cost_pass"])) if requested.shape[0] else np.nan,
        "postwarm_action_source_td3_fraction": float(np.mean(requested["action_source"].to_numpy(dtype=int) == 2))
        if requested.shape[0]
        else np.nan,
        "tail20_gate_pass_fraction": float(np.mean(tail_masks["pass_all"])) if tail_requested.shape[0] else np.nan,
        "tail20_score_pass_fraction": float(np.mean(tail_masks["score_pass"])) if tail_requested.shape[0] else np.nan,
        "tail20_drift_pass_fraction": float(np.mean(tail_masks["drift_pass"])) if tail_requested.shape[0] else np.nan,
        "tail20_cost_pass_fraction": float(np.mean(tail_masks["cost_pass"])) if tail_requested.shape[0] else np.nan,
        "bc_enabled": bool(run.bundle.get("behavioral_cloning_enabled", False)),
        "bc_target_mode": bc_cfg.get("target_mode"),
        "bc_start_step": int(bc_cfg.get("start_step", -1)) if bc_cfg else -1,
        "bc_end_step": int(bc_cfg.get("end_step", -1)) if bc_cfg else -1,
        "bc_active_fraction": float(np.mean(bc_active)) if bc_active.size else 0.0,
        "replay_uses_executed_action": bool(run.bundle.get("rl_store_executed_action_in_replay", True)),
        **breakdown,
    }
    return summary


def td3_only_counterfactual_summary(run: RunData) -> dict[str, dict]:
    requested = requested_stage_df(run)
    tail_requested = requested.iloc[-TAIL_EPISODES * run.steps_per_episode :]
    out: dict[str, dict] = {}
    for gate in GATE_COUNTERFACTUALS:
        masks_all = gate_masks(
            requested,
            s_pred_min=float(gate["s_pred_min"]),
            gain_drift_max=float(gate["gain_drift_max"]),
            nominal_cost_relative_tol=float(gate["nominal_cost_relative_tol"]),
            nominal_cost_absolute_tol=float(gate["nominal_cost_absolute_tol"]),
        )
        masks_tail = gate_masks(
            tail_requested,
            s_pred_min=float(gate["s_pred_min"]),
            gain_drift_max=float(gate["gain_drift_max"]),
            nominal_cost_relative_tol=float(gate["nominal_cost_relative_tol"]),
            nominal_cost_absolute_tol=float(gate["nominal_cost_absolute_tol"]),
        )
        out[gate["key"]] = {
            "label": gate["label"],
            "postwarm_pass_fraction": float(np.mean(masks_all["pass_all"])),
            "tail20_pass_fraction": float(np.mean(masks_tail["pass_all"])),
            "tail20_score_pass_fraction": float(np.mean(masks_tail["score_pass"])),
            "tail20_cost_pass_fraction": float(np.mean(masks_tail["cost_pass"])),
            "tail20_drift_pass_fraction": float(np.mean(masks_tail["drift_pass"])),
        }
    return out


def plot_reward_vs_authority(summaries: dict[str, dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 6.1))
    for spec in RUN_SPECS:
        summary = summaries[spec["key"]]
        ax.scatter(
            summary["td3_fraction_tail20"],
            summary["reward_tail20"],
            s=150,
            color=spec["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            summary["td3_fraction_tail20"] + 0.015,
            summary["reward_tail20"] + 0.015,
            spec["label"],
            fontsize=9,
        )
    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel("Tail-20 TD3 execution fraction")
    ax.set_ylabel("Tail-20 reward")
    ax.set_title("The best late reward appears only when TD3 actually owns the live decision")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUT_DIR / "fig_reward_vs_td3_authority.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_decision_authority_by_episode(runs: dict[str, RunData]) -> Path:
    reference = runs["guarded"]
    episodes = np.arange(1, reference.n_episodes + 1)
    bc_cfg = reference.bc_config
    bc_start = int(bc_cfg.get("start_step", reference.warm_start_step + 1))
    bc_end = int(bc_cfg.get("end_step", reference.warm_start_step))
    bc_start_ep = bc_start // reference.steps_per_episode + 1
    bc_end_ep = int(np.ceil(bc_end / float(reference.steps_per_episode)))

    fig, axs = plt.subplots(3, 1, figsize=(13.2, 9.0), sharex=True)
    panels = [
        ("td3_fraction_per_episode", "TD3 fraction", "TD3 only becomes a real controller only in the no-safeguard variant"),
        ("ls_fraction_per_episode", "LS fraction", "Relaxing the old gate mostly increases LS execution, not TD3 execution"),
        ("nominal_fraction_per_episode", "Nominal fraction", "Most guarded variants still live close to nominal MPC online"),
    ]

    for ax, (attr, ylabel, title) in zip(axs, panels):
        for spec in RUN_SPECS:
            run = runs[spec["key"]]
            values = moving_average(getattr(run, attr), WINDOW)
            ax.plot(
                episodes,
                values,
                color=run.color,
                linewidth=2.6 if run.key == "td3_only" else 1.9,
                label=run.label,
            )
        ax.axvspan(0.5, reference.warm_start_episodes + 0.5, color="#E5E7EB", alpha=0.7)
        ax.axvspan(bc_start_ep - 0.5, bc_end_ep + 0.5, color="#FDE68A", alpha=0.35)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].legend(loc="upper center", ncol=5, fontsize=9, frameon=False)
    axs[-1].set_xlabel("Sub-episode")
    fig.suptitle(
        "Decision authority over training: warm-start LS and early LS-target BC shape the pre-TD3 regime",
        y=1.01,
    )

    out = OUT_DIR / "fig_decision_authority_by_episode.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gate_breakdown(runs: dict[str, RunData]) -> Path:
    keys = ["guarded", "relaxed", "td3_without_ls", "td3_only"]
    labels = [runs[key].label for key in keys]
    x = np.arange(len(keys))

    categories = [
        ("pass_all", "Would pass all gates", "#7A1FA2"),
        ("fail_score_only", "Fail score only", "#F59E0B"),
        ("fail_cost_only", "Fail cost only", "#1f77b4"),
        ("fail_score_and_cost", "Fail score and cost", "#EF4444"),
        ("fail_other", "Fail other", "#9CA3AF"),
    ]

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    bottoms = np.zeros(len(keys), dtype=float)
    for category_key, category_label, color in categories:
        vals = []
        for key in keys:
            requested = requested_stage_df(runs[key])
            masks = gate_masks(
                requested,
                s_pred_min=runs[key].s_pred_min,
                gain_drift_max=runs[key].gain_drift_max,
                nominal_cost_relative_tol=runs[key].nominal_cost_relative_tol,
                nominal_cost_absolute_tol=runs[key].nominal_cost_absolute_tol,
            )
            vals.append(gate_breakdown(masks)[category_key])
        ax.bar(x, vals, bottom=bottoms, color=color, label=category_label)
        bottoms += np.asarray(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Fraction of post-warm-start requested steps")
    ax.set_title("The old decision logic is dominated by score and nominal-cost rejections")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)

    out = OUT_DIR / "fig_gate_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_td3_only_gate_misalignment(run: RunData) -> Path:
    episodes = np.arange(1, run.n_episodes + 1)
    pass_guarded = per_episode_pass_fraction(
        run,
        s_pred_min=GATE_COUNTERFACTUALS[0]["s_pred_min"],
        gain_drift_max=GATE_COUNTERFACTUALS[0]["gain_drift_max"],
        nominal_cost_relative_tol=GATE_COUNTERFACTUALS[0]["nominal_cost_relative_tol"],
        nominal_cost_absolute_tol=GATE_COUNTERFACTUALS[0]["nominal_cost_absolute_tol"],
    )
    pass_relaxed = per_episode_pass_fraction(
        run,
        s_pred_min=GATE_COUNTERFACTUALS[1]["s_pred_min"],
        gain_drift_max=GATE_COUNTERFACTUALS[1]["gain_drift_max"],
        nominal_cost_relative_tol=GATE_COUNTERFACTUALS[1]["nominal_cost_relative_tol"],
        nominal_cost_absolute_tol=GATE_COUNTERFACTUALS[1]["nominal_cost_absolute_tol"],
    )

    bc_cfg = run.bc_config
    bc_start = int(bc_cfg.get("start_step", run.warm_start_step + 1))
    bc_end = int(bc_cfg.get("end_step", run.warm_start_step))
    bc_start_ep = bc_start // run.steps_per_episode + 1
    bc_end_ep = int(np.ceil(bc_end / float(run.steps_per_episode)))

    fig, axs = plt.subplots(2, 1, figsize=(12.6, 7.4), sharex=True)

    axs[0].plot(episodes, moving_average(run.avg_rewards, WINDOW), color=run.color, linewidth=2.6, label="TD3-only reward")
    axs[0].axvspan(0.5, run.warm_start_episodes + 0.5, color="#E5E7EB", alpha=0.7, label="Warm-start LS")
    axs[0].axvspan(bc_start_ep - 0.5, bc_end_ep + 0.5, color="#FDE68A", alpha=0.35, label="LS-target BC window")
    axs[0].set_ylabel("Average reward")
    axs[0].set_title("TD3-only reward improves even though the old gates would still reject most requested steps")
    axs[0].grid(alpha=0.25)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].legend(loc="best", fontsize=9, frameon=False)

    axs[1].plot(episodes, moving_average(pass_guarded, WINDOW), color="#0B6E4F", linewidth=2.2, label="Would pass guarded gate")
    axs[1].plot(episodes, moving_average(pass_relaxed, WINDOW), color="#1f77b4", linewidth=2.2, label="Would pass relaxed gate")
    axs[1].axvspan(0.5, run.warm_start_episodes + 0.5, color="#E5E7EB", alpha=0.7)
    axs[1].axvspan(bc_start_ep - 0.5, bc_end_ep + 0.5, color="#FDE68A", alpha=0.35)
    axs[1].set_ylim(-0.02, 1.02)
    axs[1].set_ylabel("Gate pass fraction")
    axs[1].set_xlabel("Sub-episode")
    axs[1].grid(alpha=0.25)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].legend(loc="best", fontsize=9, frameon=False)

    out = OUT_DIR / "fig_td3_only_gate_misalignment.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_csv(path: Path, summaries: dict[str, dict]) -> None:
    fieldnames = [
        "key",
        "run_dir",
        "reward_final",
        "reward_tail20",
        "td3_fraction_tail20",
        "ls_fraction_tail20",
        "nominal_fraction_tail20",
        "postwarm_requested_steps",
        "postwarm_gate_pass_fraction",
        "postwarm_score_pass_fraction",
        "postwarm_drift_pass_fraction",
        "postwarm_cost_pass_fraction",
        "tail20_gate_pass_fraction",
        "tail20_score_pass_fraction",
        "tail20_drift_pass_fraction",
        "tail20_cost_pass_fraction",
        "pass_all",
        "fail_score_only",
        "fail_cost_only",
        "fail_score_and_cost",
        "fail_other",
        "bc_enabled",
        "bc_target_mode",
        "bc_start_step",
        "bc_end_step",
        "bc_active_fraction",
        "replay_uses_executed_action",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, summary in summaries.items():
            row = {"key": key}
            row.update({name: summary.get(name) for name in fieldnames if name != "key"})
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = build_run_data()
    summaries = {key: summarize_run(run) for key, run in runs.items()}
    td3_only_counterfactual = td3_only_counterfactual_summary(runs["td3_only"])

    figures = {
        "reward_vs_td3_authority": str(plot_reward_vs_authority(summaries).relative_to(REPO_ROOT)),
        "decision_authority_by_episode": str(plot_decision_authority_by_episode(runs).relative_to(REPO_ROOT)),
        "gate_breakdown": str(plot_gate_breakdown(runs).relative_to(REPO_ROOT)),
        "td3_only_gate_misalignment": str(plot_td3_only_gate_misalignment(runs["td3_only"]).relative_to(REPO_ROOT)),
    }

    summary = {
        "runs": summaries,
        "td3_only_counterfactual": td3_only_counterfactual,
        "common_training_structure": {
            "warm_start_episodes": runs["guarded"].warm_start_episodes,
            "behavioral_cloning_enabled": bool(runs["guarded"].bundle.get("behavioral_cloning_enabled", False)),
            "behavioral_cloning_target_mode": runs["guarded"].bc_config.get("target_mode"),
            "behavioral_cloning_start_step": int(runs["guarded"].bc_config.get("start_step", -1)),
            "behavioral_cloning_end_step": int(runs["guarded"].bc_config.get("end_step", -1)),
            "replay_uses_executed_action": bool(runs["guarded"].bundle.get("rl_store_executed_action_in_replay", True)),
        },
        "figures": figures,
    }

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_csv(OUT_DIR / "summary_metrics.csv", summaries)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
