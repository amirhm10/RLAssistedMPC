"""Analyze the June 7 distillation SG warm-3 replay-40k runs.

This script reads saved result bundles only. It compares the latest five
distillation supervisor-gated runners against the closest previous mismatch
references, generates figures/CSV/JSON artifacts, and appends a June 7 section
to the existing limited-horizon report.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report.scripts import analyze_distillation_sg_limited_horizons_20260606 as base  # noqa: E402
from report.scripts.analyze_distillation_dueling_horizon_history_20260604 import (  # noqa: E402
    CURRENT_REWARD,
    OFMPC_PATH,
    TAIL_EPISODES,
    episode_average,
    finite_mean,
    load_pickle,
    rel,
    reward_step_series,
    step_slice,
    tail_slice,
)


OUT_DIR = ROOT / "report" / "figures" / "distillation_sg_replay40k_warm3_20260607"
REPORT_PATH = ROOT / "report" / "distillation_sg_td3_sg_dqn_limited_horizon_results_2026_06_06.md"
SECTION_PATH = OUT_DIR / "distillation_sg_replay40k_warm3_section.md"
SUMMARY_CSV = OUT_DIR / "distillation_sg_replay40k_warm3_summary.csv"
SUMMARY_JSON = OUT_DIR / "distillation_sg_replay40k_warm3_summary.json"
EPISODE_CSV = OUT_DIR / "distillation_sg_replay40k_warm3_episode_diagnostics.csv"
HORIZON_PAIR_CSV = OUT_DIR / "distillation_sg_replay40k_warm3_horizon_pairs.csv"

MARKER = "## June 7 Replay-40k Warm-3 Setup"
WARM_START_SUBEPISODES = base.WARM_START_SUBEPISODES
HANDOVER_SUBEPISODES = base.HANDOVER_SUBEPISODES
LIVE_RELEASE_SUBEPISODE = base.LIVE_RELEASE_SUBEPISODE


def _result(folder: str, timestamp: str | None = None) -> Path:
    root = ROOT / "Distillation" / "Results" / folder
    if timestamp is not None:
        return root / timestamp / "input_data.pkl"
    candidates = sorted(root.glob("*/input_data.pkl"), key=lambda path: path.parent.name)
    if not candidates:
        raise FileNotFoundError(f"No input_data.pkl files found under {root}")
    return candidates[-1]


FOLDERS = {
    "weights_current": "distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch",
    "residual_current": "distillation_residual_sg_td3_critic_warm3_margin05_paramnoise_manual_off_disturb_fluctuation_mismatch_no_rho",
    "markov_current": "distillation_markov_sg_td3_critic_warm3_margin05_softparamnoise_ls_else_mpc_shadow_disturb_fluctuation_mismatch",
    "horizon_current": "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11",
    "dueling_current": "distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11",
    "residual_previous": "distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho",
    "markov_previous": "distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_mismatch_paramnoise",
    "dueling_wide": "distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch",
}


RUNS = [
    {
        "key": "ofmpc",
        "label": "OF-MPC",
        "family": "baseline",
        "role": "baseline",
        "path": OFMPC_PATH,
    },
    {
        "key": "weights_current",
        "label": "Weights SG-TD3 replay40k",
        "family": "weights",
        "role": "current",
        "reference_key": "weights_previous",
        "path": _result(FOLDERS["weights_current"]),
    },
    {
        "key": "weights_previous",
        "label": "Weights SG-TD3 June 6",
        "family": "weights",
        "role": "reference",
        "path": _result(FOLDERS["weights_current"], "20260606_075201"),
    },
    {
        "key": "residual_current",
        "label": "Residual SG-TD3 param-noise replay40k",
        "family": "residual",
        "role": "current",
        "reference_key": "residual_previous",
        "path": _result(FOLDERS["residual_current"]),
    },
    {
        "key": "residual_previous",
        "label": "Residual SG-TD3 June 6 no-param-noise",
        "family": "residual",
        "role": "reference",
        "path": _result(FOLDERS["residual_previous"], "20260606_074932"),
    },
    {
        "key": "markov_current",
        "label": "Markov SG-TD3 margin0.5 replay40k",
        "family": "markov",
        "role": "current",
        "reference_key": "markov_previous",
        "path": _result(FOLDERS["markov_current"]),
    },
    {
        "key": "markov_previous",
        "label": "Markov SG-TD3 June 6 margin0",
        "family": "markov",
        "role": "reference",
        "path": _result(FOLDERS["markov_previous"], "20260606_105404"),
    },
    {
        "key": "horizon_current",
        "label": "Horizon SG-DQN replay40k",
        "family": "horizon",
        "role": "current",
        "reference_key": "horizon_previous",
        "path": _result(FOLDERS["horizon_current"]),
    },
    {
        "key": "horizon_previous",
        "label": "Horizon SG-DQN June 6",
        "family": "horizon",
        "role": "reference",
        "path": _result(FOLDERS["horizon_current"], "20260606_071451"),
    },
    {
        "key": "dueling_current",
        "label": "Dueling SG-DQN replay40k",
        "family": "dueling horizon",
        "role": "current",
        "reference_key": "dueling_wide",
        "path": _result(FOLDERS["dueling_current"]),
    },
    {
        "key": "dueling_wide",
        "label": "Dueling SG-DQN June 4 wide",
        "family": "dueling horizon",
        "role": "reference",
        "path": _result(FOLDERS["dueling_wide"], "20260604_105140"),
    },
]


EXPECTED_CONFIG = {
    "weights_current": {
        "state_mode": "mismatch",
        "action_freeze": 3,
        "actor_freeze": 3,
        "margin": 0.0,
        "replay_capacity": "40000",
        "noise": "gaussian 0.15 to 0.03",
    },
    "residual_current": {
        "state_mode": "mismatch",
        "action_freeze": 3,
        "actor_freeze": 3,
        "margin": 0.5,
        "replay_capacity": "40000",
        "noise": "param-noise 0.10 to 0.02",
    },
    "markov_current": {
        "state_mode": "mismatch with Markov features",
        "action_freeze": 3,
        "actor_freeze": 3,
        "margin": 0.5,
        "replay_capacity": "40000 default",
        "noise": "param-noise 0.10 to 0.02, trace not saved",
    },
    "horizon_current": {
        "state_mode": "mismatch",
        "action_freeze": 3,
        "actor_freeze": "n/a",
        "margin": 0.0,
        "replay_capacity": "40000",
        "noise": "epsilon",
    },
    "dueling_current": {
        "state_mode": "mismatch",
        "action_freeze": 3,
        "actor_freeze": "n/a",
        "margin": 0.0,
        "replay_capacity": "40000",
        "noise": "epsilon",
    },
}


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 3) -> str:
    return base._fmt(value, digits)


def _safe_float(value: object) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return val


def _row_extra(row: dict, bundle: dict) -> None:
    snapshot = bundle.get("config_snapshot") or {}
    gate = snapshot.get("supervisor_gate") or bundle.get("supervisor_gate") or {}
    replay = bundle.get("replay_buffer_snapshot") or {}
    expected = EXPECTED_CONFIG.get(row.get("key"), {})
    param_trace = bundle.get("param_noise_scale_trace")
    residual_param_trace = bundle.get("residual_param_noise_scale_trace")
    weight_param_trace = bundle.get("weight_param_noise_scale_trace")
    matrix_param_trace = bundle.get("matrix_param_noise_scale_trace")
    state_mode = (
        snapshot.get("state_mode")
        or row.get("markov_state_mode")
        or row.get("state_mode")
        or expected.get("state_mode", "")
    )
    if row.get("key") == "markov_current":
        state_mode = expected["state_mode"]
    trace_len = max(
        len(param_trace) if param_trace is not None else 0,
        len(residual_param_trace) if residual_param_trace is not None else 0,
        len(weight_param_trace) if weight_param_trace is not None else 0,
        len(matrix_param_trace) if matrix_param_trace is not None else 0,
    )
    row.update(
        {
            "config_state_mode": state_mode,
            "config_action_freeze": snapshot.get(
                "post_warm_start_action_freeze_subepisodes", expected.get("action_freeze", "")
            ),
            "config_actor_freeze": snapshot.get(
                "post_warm_start_actor_freeze_subepisodes", expected.get("actor_freeze", "")
            ),
            "config_advantage_margin": gate.get("advantage_margin", expected.get("margin", "")),
            "config_uncertainty_weight": gate.get("score_uncertainty_weight", ""),
            "config_supervisor_action_weight": gate.get("score_supervisor_action_weight", ""),
            "replay_capacity": replay.get("capacity", expected.get("replay_capacity", "")),
            "replay_size": replay.get("size", ""),
            "param_noise_trace_len": trace_len,
            "exploration_note": expected.get("noise", ""),
        }
    )


def _load_rows() -> tuple[list[dict], list[dict], list[dict], dict[str, dict], list[dict]]:
    base.RUNS = RUNS
    first_bundle = load_pickle(RUNS[0]["path"])
    ofmpc_tail = finite_mean(
        episode_average(reward_step_series(first_bundle, CURRENT_REWARD), first_bundle)[-TAIL_EPISODES:]
    )
    summary_rows: list[dict] = []
    episode_rows: list[dict] = []
    horizon_pair_rows: list[dict] = []
    bundles: dict[str, dict] = {}

    for cfg in RUNS:
        if not cfg["path"].exists():
            print(f"missing: {cfg['path']}")
            continue
        row, bundle, rewards = base._row_for_run(cfg, ofmpc_tail)
        _row_extra(row, bundle)
        summary_rows.append(row)
        bundles[cfg["key"]] = bundle
        episode_rows.extend(base._episode_rows_for_run(cfg, bundle, rewards))
        if bundle.get("horizon_recipes") is not None:
            horizon_pair_rows.extend(
                base._horizon_pair_rows(cfg["key"], cfg["label"], bundle, tail_slice(bundle, TAIL_EPISODES), "tail")
            )
            horizon_pair_rows.extend(
                base._horizon_pair_rows(
                    cfg["key"],
                    cfg["label"],
                    bundle,
                    step_slice(
                        bundle,
                        WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES,
                        WARM_START_SUBEPISODES + HANDOVER_SUBEPISODES + 20,
                    ),
                    "first_live",
                )
            )
    comparison_rows = base._comparison_rows(summary_rows)
    return summary_rows, comparison_rows, episode_rows, bundles, horizon_pair_rows


def _current_rows(summary_rows: list[dict]) -> list[dict]:
    return [row for row in summary_rows if row["role"] == "current"]


def _plot_tail_summary(summary_rows: list[dict]) -> Path:
    current = _current_rows(summary_rows)
    baseline = next(row for row in summary_rows if row["key"] == "ofmpc")
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#E15759"]
    x = np.arange(len(current))
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax1.bar(x, [row["tail_reward"] for row in current], color=colors, alpha=0.88)
    ax1.axhline(baseline["tail_reward"], color="#333333", linestyle="--", linewidth=1.0, label="OF-MPC reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels([row["family"] for row in current], rotation=15, ha="right")
    ax1.set_ylabel("tail-20 reward")
    ax1.grid(axis="y", alpha=0.24)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["negative_postwarm_episodes"] for row in current], color="#222222", marker="o")
    ax2.set_ylabel("negative post-warm episodes")
    ax1.set_title("June 7 replay-40k warm-3 distillation SG runs")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "fig_june7_tail_reward_negative_episodes.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_current_vs_reference(comparison_rows: list[dict]) -> Path:
    labels = [row["family"] for row in comparison_rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(10.6, 4.9))
    ax1.bar(x, [row["tail_reward_delta"] for row in comparison_rows], color="#4E79A7", alpha=0.86)
    ax1.axhline(0.0, color="#333333", linewidth=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("tail reward change")
    ax1.grid(axis="y", alpha=0.24)
    ax2 = ax1.twinx()
    ax2.plot(x, [row["tail_temp_mae_delta"] for row in comparison_rows], color="#E15759", marker="o")
    ax2.axhline(0.0, color="#E15759", linestyle=":", linewidth=0.9)
    ax2.set_ylabel("tail T85 MAE change")
    ax1.set_title("June 7 change relative to closest mismatch reference")
    fig.tight_layout()
    out = OUT_DIR / "fig_june7_current_vs_reference_delta.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_episode_rewards(episode_rows: list[dict]) -> Path:
    keys = ["ofmpc", "weights_current", "residual_current", "markov_current", "horizon_current", "dueling_current"]
    colors = {
        "ofmpc": "#333333",
        "weights_current": "#4C78A8",
        "residual_current": "#59A14F",
        "markov_current": "#F28E2B",
        "horizon_current": "#B07AA1",
        "dueling_current": "#E15759",
    }
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    for key in keys:
        series = [row for row in episode_rows if row["key"] == key]
        if not series:
            continue
        ax.plot(
            [row["subepisode"] for row in series],
            [row["reward"] for row in series],
            label=series[0]["label"],
            color=colors[key],
            linewidth=1.15 if key != "ofmpc" else 1.0,
            alpha=0.95 if key != "ofmpc" else 0.65,
        )
    ax.axvspan(1, WARM_START_SUBEPISODES, color="#dddddd", alpha=0.25, label="warm start")
    ax.axvspan(WARM_START_SUBEPISODES + 1, LIVE_RELEASE_SUBEPISODE - 1, color="#f3c567", alpha=0.20, label="critic-only")
    ax.axvline(LIVE_RELEASE_SUBEPISODE, color="#8E3B46", linestyle=":", linewidth=1.0, label="live release")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.7)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("current reward")
    ax.set_title("June 7 episode reward histories")
    ax.grid(alpha=0.22)
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "fig_june7_episode_rewards.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_gate_fractions(episode_rows: list[dict]) -> Path:
    keys = ["weights_current", "residual_current", "markov_current", "horizon_current", "dueling_current"]
    colors = {
        "weights_current": "#4C78A8",
        "residual_current": "#59A14F",
        "markov_current": "#F28E2B",
        "horizon_current": "#B07AA1",
        "dueling_current": "#E15759",
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    for key in keys:
        rows = [row for row in episode_rows if row["key"] == key]
        if not rows:
            continue
        x = np.asarray([row["subepisode"] for row in rows], float)
        y = np.asarray([row["sg_policy_step_frac"] for row in rows], float)
        if y.size >= 5:
            y = np.convolve(np.nan_to_num(y, nan=0.0), np.ones(5) / 5.0, mode="same")
        ax.plot(x, y, label=rows[0]["label"], color=colors[key], linewidth=1.25)
    ax.axvspan(1, WARM_START_SUBEPISODES, color="#dddddd", alpha=0.25)
    ax.axvspan(WARM_START_SUBEPISODES + 1, LIVE_RELEASE_SUBEPISODE - 1, color="#f3c567", alpha=0.20)
    ax.axvline(LIVE_RELEASE_SUBEPISODE, color="#8E3B46", linestyle=":", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("policy-selected step fraction, 5-episode moving average")
    ax.set_title("June 7 supervisor-gate release behavior")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    out = OUT_DIR / "fig_june7_gate_policy_fraction.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_horizon_heatmaps(bundles: dict[str, dict]) -> Path:
    keys = ["horizon_current", "horizon_previous", "dueling_current", "dueling_wide"]
    titles = ["DQN June 7", "DQN June 6", "Dueling June 7", "Dueling June 4 wide"]
    fig, axes = plt.subplots(1, len(keys), figsize=(15.2, 4.4), constrained_layout=True)
    data = []
    vmax = 0.0
    for key in keys:
        heat, nps, ncs = base._heatmap_data(bundles[key], tail_slice(bundles[key], TAIL_EPISODES))
        data.append((heat, nps, ncs))
        if heat.size:
            vmax = max(vmax, float(np.max(heat)))
    im = None
    for ax, title, (heat, nps, ncs) in zip(axes, titles, data):
        if not heat.size:
            ax.set_axis_off()
            continue
        im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=max(vmax, 1.0e-9))
        ax.set_xticks(np.arange(len(nps)))
        ax.set_xticklabels(nps)
        ax.set_yticks(np.arange(len(ncs)))
        ax.set_yticklabels(ncs)
        ax.set_xlabel("Np")
        ax.set_ylabel("Nc")
        ax.set_title(title)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.82, label="tail frequency")
    out = OUT_DIR / "fig_june7_horizon_tail_heatmaps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _worst_current_episode_rows(episode_rows: list[dict]) -> list[dict]:
    out = []
    for key in ["weights_current", "residual_current", "markov_current", "horizon_current", "dueling_current"]:
        candidates = [
            row
            for row in episode_rows
            if row["key"] == key and int(row["subepisode"]) > WARM_START_SUBEPISODES
        ]
        if candidates:
            out.append(min(candidates, key=lambda row: _safe_float(row["reward"])))
    return out


def _top_pair_strings(horizon_pair_rows: list[dict], key: str, window: str = "tail", limit: int = 5) -> str:
    rows = [
        row
        for row in horizon_pair_rows
        if row["key"] == key and row["window"] == window and int(row["rank"]) <= limit
    ]
    if not rows:
        return ""
    return "; ".join(
        f"({int(row['Np'])}, {int(row['Nc'])}) at {_fmt(row['fraction'] * 100.0, 1)} percent"
        for row in rows
    )


def _build_section(
    summary_rows: list[dict],
    comparison_rows: list[dict],
    episode_rows: list[dict],
    horizon_pair_rows: list[dict],
    figures: list[Path],
) -> str:
    by_key = {row["key"]: row for row in summary_rows}
    current = _current_rows(summary_rows)
    ofmpc = by_key["ofmpc"]
    best_tail = max(current, key=lambda row: row["tail_reward"])
    safest = min(current, key=lambda row: (row["negative_postwarm_episodes"], -row["tail_reward"]))
    residual = by_key["residual_current"]
    residual_ref = by_key["residual_previous"]
    markov = by_key["markov_current"]
    markov_ref = by_key["markov_previous"]
    weights = by_key["weights_current"]
    weights_ref = by_key["weights_previous"]
    horizon = by_key["horizon_current"]
    horizon_ref = by_key["horizon_previous"]
    dueling = by_key["dueling_current"]
    dueling_ref = by_key["dueling_wide"]

    lines: list[str] = []
    lines.append(MARKER)
    lines.append("")
    lines.append("Generated on 2026-06-07 from saved result bundles only; Aspen was not relaunched.")
    lines.append("")
    lines.append("### Setup Being Tested")
    lines.append("")
    lines.append(
        "This pass keeps the mismatch-state distillation SG runners, shortens the critic-only handoff to warm-3, "
        "and uses the distillation replay default of 40,000 transitions. Residual and Markov SG-TD3 now use "
        "parameter noise starting at 0.10 and ending at 0.02, with `advantage_margin = 0.5`; weights SG-TD3 keeps "
        "Gaussian exploration and `advantage_margin = 0.0`. Both horizon runners use the limited triangular grid "
        "`Np = 6..11`, `Nc = 3..Np`."
    )
    lines.append("")
    lines.append(
        "The continuous SG-TD3 gate can be summarized as accepting the actor only when the conservative policy score beats "
        "the supervisor score by the configured margin; otherwise the supervisor action is executed. For residual, the "
        "supervisor action is zero residual; for weights it is the identity multiplier; for Markov it is the LS-or-MPC "
        "Markov correction. SG-DQN performs the same handoff idea over discrete horizon recipes using one learned Q score."
    )
    lines.append("")
    lines.append("### Executive Takeaways")
    lines.append("")
    lines.append(
        f"- Best tail performance in this setup is **{best_tail['label']}** with tail reward {_fmt(best_tail['tail_reward'])}, "
        f"{_fmt(best_tail['delta_vs_ofmpc'])} above OF-MPC."
    )
    lines.append(
        f"- The cleanest post-warm stability is **{safest['label']}**, with "
        f"{safest['negative_postwarm_episodes']} negative post-warm episodes and tail reward {_fmt(safest['tail_reward'])}."
    )
    lines.append(
        f"- Residual parameter noise plus margin 0.5 changed the character of the run: tail reward moved from "
        f"{_fmt(residual_ref['tail_reward'])} to {_fmt(residual['tail_reward'])}, while negative post-warm episodes moved from "
        f"{residual_ref['negative_postwarm_episodes']} to {residual['negative_postwarm_episodes']}."
    )
    lines.append(
        f"- Markov margin 0.5 and softer parameter noise changed tail reward from {_fmt(markov_ref['tail_reward'])} to "
        f"{_fmt(markov['tail_reward'])}; the key diagnostic is whether its poorer episodes are from conservative under-correction "
        "or from the actor being accepted too often."
    )
    lines.append(
        f"- Dueling SG-DQN now has a saved limited-grid mismatch run. Compared with the old wide-grid dueling reference, "
        f"tail reward changed by {_fmt(dueling['tail_reward'] - dueling_ref['tail_reward'])} and negative post-warm episodes "
        f"changed by {dueling['negative_postwarm_episodes'] - dueling_ref['negative_postwarm_episodes']}."
    )
    lines.append("")
    lines.append("### June 7 Current Run Summary")
    lines.append("")
    lines.append("| Run | Timestamp | Tail reward | Delta vs OF-MPC | Final reward | Worst post-warm | Neg post-warm | T85 MAE | x24 MAE | Tail policy frac | First-live policy frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in [ofmpc, *current]:
        lines.append(
            f"| {row['label']} | {row['timestamp']} | {_fmt(row['tail_reward'])} | {_fmt(row['delta_vs_ofmpc'])} | "
            f"{_fmt(row['final_reward'])} | {_fmt(row['worst_postwarm_reward'])} | {row['negative_postwarm_episodes']} | "
            f"{_fmt(row['tail_temp_mae'])} | {_fmt(row['tail_comp_mae'], 5)} | "
            f"{_fmt(row.get('tail_sg_policy_step_frac', float('nan')))} | "
            f"{_fmt(row.get('first_live_sg_policy_step_frac', float('nan')))} |"
        )
    lines.append("")
    lines.append("### Change Relative To Closest Mismatch Reference")
    lines.append("")
    lines.append("| Family | Reference | Tail reward change | T85 MAE change | x24 MAE change | Neg post-warm change | Tail policy frac change | First-live policy frac change |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    by_comparison_ref = {row["reference_key"]: row for row in comparison_rows}
    for row in comparison_rows:
        ref_label = by_key[row["reference_key"]]["label"]
        lines.append(
            f"| {row['family']} | {ref_label} | {_fmt(row['tail_reward_delta'])} | "
            f"{_fmt(row['tail_temp_mae_delta'])} | {_fmt(row['tail_comp_mae_delta'], 5)} | "
            f"{row['negative_postwarm_delta']} | {_fmt(row['policy_step_frac_delta'])} | "
            f"{_fmt(row['first_live_policy_step_frac_delta'])} |"
        )
    if not by_comparison_ref:
        lines.append("|  |  |  |  |  |  |  |  |")
    lines.append("")
    lines.append("Positive reward change is good; negative tracking-MAE change is good.")
    lines.append("")
    lines.append("### Configuration Audit")
    lines.append("")
    lines.append("| Run | State mode | Action freeze | Actor freeze | Margin | Replay cap | Replay size | Exploration note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in current:
        replay_size = row.get("replay_size", "")
        if replay_size == "" and row["key"] == "markov_current":
            replay_size = "not saved"
        exploration_note = row.get("exploration_note", "")
        trace_len = int(row.get("param_noise_trace_len") or 0)
        if trace_len:
            exploration_note = f"{exploration_note}; trace {trace_len}"
        lines.append(
            f"| {row['label']} | {row.get('config_state_mode', '')} | {row.get('config_action_freeze', '')} | "
            f"{row.get('config_actor_freeze', '')} | {_fmt(row.get('config_advantage_margin', ''), 2)} | "
            f"{row.get('replay_capacity', '')} | {replay_size} | {exploration_note} |"
        )
    lines.append("")
    lines.append("### Family Interpretation")
    lines.append("")
    lines.append(
        f"**Weights SG-TD3.** Tail reward changed from {_fmt(weights_ref['tail_reward'])} to "
        f"{_fmt(weights['tail_reward'])}. This is still the most defensible continuous supervisor if the goal is "
        "stable improvement: it does not directly add input residuals, and the current run keeps a low first-live policy fraction "
        f"({_fmt(weights['first_live_sg_policy_step_frac'])})."
    )
    lines.append("")
    lines.append(
        f"**Residual SG-TD3.** The residual policy is the highest-leverage actor because accepted actions directly perturb the MPC input. "
        f"In this setup its tail reward is {_fmt(residual['tail_reward'])}, with worst post-warm reward "
        f"{_fmt(residual['worst_postwarm_reward'])}. Relative to June 6, the new settings reduced the number of negative post-warm "
        "episodes, but they did not remove the handoff shock. The margin and parameter noise should be judged by whether they improve "
        "the first 20 live episodes, not only the final tail, because a residual run can recover later after an early disturbance of the plant."
    )
    lines.append("")
    lines.append(
        f"**Markov SG-TD3.** Markov's tail reward is {_fmt(markov['tail_reward'])}. Its first-live policy fraction is "
        f"{_fmt(markov['first_live_sg_policy_step_frac'])}, but negative post-warm episodes increased to "
        f"{markov['negative_postwarm_episodes']}. The worst episode has low policy acceptance, so at least part of the weakness looks like "
        "conservative under-correction or a supervisor/candidate scoring mismatch around the disturbed column state, not simply too much actor authority."
    )
    lines.append("")
    lines.append(
        f"**Horizon SG-DQN.** The single-Q limited-grid DQN changed tail reward from {_fmt(horizon_ref['tail_reward'])} to "
        f"{_fmt(horizon['tail_reward'])}. Because the action is a horizon recipe, the practical question is whether the learned policy "
        "concentrates around a small subset of stable recipes or continues to diffuse across the 39-action grid."
    )
    lines.append("")
    lines.append(
        f"**Dueling SG-DQN.** The dueling network now has a like-for-like limited-grid mismatch result. Against the older wide-grid "
        f"reference, tail reward is {_fmt(dueling['tail_reward'])} versus {_fmt(dueling_ref['tail_reward'])}. If this run still "
        "underperforms, the issue is probably not only the wide action set; the reward/gate signal for horizon selection is still weak."
    )
    lines.append("")
    lines.append("### Worst Post-Warm Episodes")
    lines.append("")
    lines.append("| Run | Worst subepisode | Reward | T85 MAE | x24 MAE | Policy frac | Median gate advantage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in _worst_current_episode_rows(episode_rows):
        lines.append(
            f"| {row['label']} | {int(row['subepisode'])} | {_fmt(row['reward'])} | {_fmt(row['temp_mae'])} | "
            f"{_fmt(row['comp_mae'], 5)} | {_fmt(row.get('sg_policy_step_frac', float('nan')))} | "
            f"{_fmt(row.get('sg_adv_median', float('nan')))} |"
        )
    lines.append("")
    lines.append("### Horizon Candidate Diagnostics")
    lines.append("")
    lines.append("| Run | Tail top pairs |")
    lines.append("|---|---|")
    for key in ["horizon_current", "horizon_previous", "dueling_current", "dueling_wide"]:
        row = by_key[key]
        lines.append(f"| {row['label']} | {_top_pair_strings(horizon_pair_rows, key)} |")
    lines.append("")
    lines.append("### Figures")
    lines.append("")
    for fig_path in figures:
        lines.append(f"- [{fig_path.name}]({rel(fig_path)})")
    lines.append("")
    lines.append("### Next Experiment")
    lines.append("")
    lines.append(
        "Use weights SG-TD3 as the defensible continuous baseline unless residual clearly beats it without creating early negative episodes. "
        "For the next residual run, keep only one major change at a time: either keep the new parameter-noise setting and compare margins "
        "`0.0`, `0.25`, and `0.5`, or keep margin 0.5 and compare Gaussian versus parameter noise. The current bundle cannot fully separate "
        "replay-size effects from exploration and margin effects because they changed together."
    )
    lines.append("")
    lines.append(
        "For Markov, first inspect whether bad episodes have low policy acceptance. If yes, tune the Markov supervisor/gate scoring rather than "
        "making the actor more conservative; if no, reduce the margin/noise pressure. For horizons, keep `Np = 6..11`, `Nc = 3..Np` for one "
        "more paired DQN/dueling run, then narrow only if the tail heatmaps consistently concentrate below `Nc = 7`."
    )
    lines.append("")
    lines.append("### Provenance Added For This Section")
    lines.append("")
    lines.append("Files inspected:")
    lines.append("")
    inspected = [
        "report/distillation_sg_td3_sg_dqn_limited_horizon_results_2026_06_06.md",
        "report/scripts/analyze_distillation_sg_limited_horizons_20260606.py",
        "report/scripts/analyze_distillation_dueling_horizon_history_20260604.py",
        "distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py",
        "distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py",
        "systems/distillation/notebook_params.py",
    ]
    for item in inspected:
        lines.append(f"- `{item}`")
    for row in summary_rows:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    lines.append("Generated outputs:")
    lines.append("")
    for item in [SUMMARY_CSV, SUMMARY_JSON, EPISODE_CSV, HORIZON_PAIR_CSV, SECTION_PATH, *figures]:
        lines.append(f"- `{rel(item)}`")
    return "\n".join(lines) + "\n"


def _append_or_replace_section(section: str) -> None:
    if REPORT_PATH.exists():
        report = REPORT_PATH.read_text(encoding="utf-8")
    else:
        report = "# Distillation SG-TD3 and Limited-Horizon SG-DQN Results\n"
    head = report.split(MARKER, 1)[0].rstrip()
    REPORT_PATH.write_text(f"{head}\n\n{section}", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, comparison_rows, episode_rows, bundles, horizon_pair_rows = _load_rows()
    figures = [
        _plot_tail_summary(summary_rows),
        _plot_current_vs_reference(comparison_rows),
        _plot_episode_rewards(episode_rows),
        _plot_gate_fractions(episode_rows),
        _plot_horizon_heatmaps(bundles),
    ]

    _write_csv(SUMMARY_CSV, summary_rows)
    _write_csv(EPISODE_CSV, episode_rows)
    _write_csv(HORIZON_PAIR_CSV, horizon_pair_rows)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "comparisons": comparison_rows,
                "figures": [rel(path) for path in figures],
                "tail_episodes": TAIL_EPISODES,
                "warm_start_subepisodes": WARM_START_SUBEPISODES,
                "handover_subepisodes": HANDOVER_SUBEPISODES,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    section = _build_section(summary_rows, comparison_rows, episode_rows, horizon_pair_rows, figures)
    SECTION_PATH.write_text(section, encoding="utf-8")
    _append_or_replace_section(section)
    print(f"wrote {rel(REPORT_PATH)}")
    print(f"wrote {rel(SECTION_PATH)}")
    print(f"wrote {rel(SUMMARY_CSV)}")
    print(f"wrote {len(figures)} figures under {rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
