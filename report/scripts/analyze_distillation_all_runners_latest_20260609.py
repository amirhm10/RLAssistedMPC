"""Create a standalone report for the latest distillation runner batch.

The script reads saved bundles only. It does not launch Aspen, and it leaves
raw result directories untouched. Metrics are recomputed with the current
distillation reward helper used by the existing June 2026 analysis scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report.scripts import analyze_distillation_sg_replay40k_warm3_20260607 as src  # noqa: E402


REPORT_DATE = "2026-06-09"
REPORT_PATH = ROOT / "report" / "distillation_all_runners_latest_analysis_2026_06_09.md"
OUT_DIR = ROOT / "report" / "figures" / "distillation_all_runners_latest_20260609"
SUMMARY_CSV = OUT_DIR / "distillation_all_runners_latest_summary.csv"
SUMMARY_JSON = OUT_DIR / "distillation_all_runners_latest_summary.json"
EPISODE_CSV = OUT_DIR / "distillation_all_runners_latest_episode_diagnostics.csv"
HORIZON_PAIR_CSV = OUT_DIR / "distillation_all_runners_latest_horizon_pairs.csv"

WARM_START_SUBEPISODES = src.WARM_START_SUBEPISODES
HANDOVER_SUBEPISODES = src.HANDOVER_SUBEPISODES
LIVE_RELEASE_SUBEPISODE = src.LIVE_RELEASE_SUBEPISODE
TAIL_EPISODES = src.TAIL_EPISODES


def _fmt(value: object, digits: int = 3) -> str:
    return src._fmt(value, digits)


def _report_rel(path: Path) -> str:
    return path.relative_to(REPORT_PATH.parent).as_posix()


def _repo_rel(path: Path | str) -> str:
    path = Path(path)
    if path.is_absolute():
        return path.relative_to(ROOT).as_posix()
    return path.as_posix().replace("\\", "/")


def _current_and_baseline(summary_rows: list[dict]) -> list[dict]:
    return [row for row in summary_rows if row["role"] in {"baseline", "current"}]


def _current(summary_rows: list[dict]) -> list[dict]:
    return [row for row in summary_rows if row["role"] == "current"]


def _by_key(rows: list[dict]) -> dict[str, dict]:
    return {row["key"]: row for row in rows}


def _plot_tail_summary(summary_rows: list[dict]) -> Path:
    current = _current(summary_rows)
    ofmpc = _by_key(summary_rows)["ofmpc"]
    labels = [row["family"] for row in current]
    rewards = [row["tail_reward"] for row in current]
    failures = [row["negative_postwarm_episodes"] for row in current]

    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#B07AA1", "#E15759"]
    x = np.arange(len(labels))
    ax1.bar(x, rewards, color=colors[: len(labels)], alpha=0.88)
    ax1.axhline(ofmpc["tail_reward"], color="#333333", linestyle="--", linewidth=1.2, label="OF-MPC reward")
    ax1.set_ylabel("tail-20 reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=16, ha="right")
    ax1.grid(axis="y", alpha=0.28)

    ax2 = ax1.twinx()
    ax2.plot(x, failures, color="#222222", marker="o", linewidth=1.8)
    ax2.set_ylabel("negative post-warm episodes")
    ax2.set_ylim(bottom=0)

    ax1.set_title("Latest replay-40k warm-3 distillation SG runs")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "fig_latest_tail_reward_negative_episodes.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_current_vs_reference(comparison_rows: list[dict]) -> Path:
    labels = [row["family"] for row in comparison_rows]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    ax1.bar(x, [row["tail_reward_delta"] for row in comparison_rows], color="#4E79A7", alpha=0.86)
    ax1.axhline(0.0, color="#222222", linewidth=0.8)
    ax1.set_ylabel("tail reward change")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=16, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, [row["tail_temp_mae_delta"] for row in comparison_rows], color="#E15759", marker="o")
    ax2.set_ylabel("T85 MAE change")

    ax1.set_title("Latest change relative to closest mismatch reference")
    fig.tight_layout()
    out = OUT_DIR / "fig_latest_current_vs_reference_delta.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_episode_rewards(episode_rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    colors = {
        "ofmpc": "#444444",
        "weights_current": "#4E79A7",
        "residual_current": "#59A14F",
        "markov_current": "#F28E2B",
        "horizon_current": "#B07AA1",
        "dueling_current": "#E15759",
    }
    labels = {
        "ofmpc": "OF-MPC",
        "weights_current": "weights",
        "residual_current": "residual",
        "markov_current": "markov",
        "horizon_current": "horizon",
        "dueling_current": "dueling",
    }
    for key, label in labels.items():
        rows = [row for row in episode_rows if row["key"] == key]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: int(row["subepisode"]))
        ax.plot(
            [int(row["subepisode"]) for row in rows],
            [float(row["reward"]) for row in rows],
            color=colors[key],
            linewidth=1.3 if key != "ofmpc" else 1.1,
            alpha=0.92 if key != "ofmpc" else 0.55,
            label=label,
        )

    ax.axvspan(1, WARM_START_SUBEPISODES, color="#dddddd", alpha=0.24, label="warm start")
    ax.axvspan(
        WARM_START_SUBEPISODES + 1,
        LIVE_RELEASE_SUBEPISODE - 1,
        color="#f3c567",
        alpha=0.20,
        label="critic-only",
    )
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("rescored episode reward")
    ax.set_title("Latest episode reward histories")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "fig_latest_episode_rewards.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_gate_fractions(episode_rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = {
        "weights_current": "#4E79A7",
        "residual_current": "#59A14F",
        "markov_current": "#F28E2B",
        "horizon_current": "#B07AA1",
        "dueling_current": "#E15759",
    }
    for key, color in colors.items():
        rows = [row for row in episode_rows if row["key"] == key]
        values = [
            (int(row["subepisode"]), float(row["sg_policy_step_frac"]))
            for row in rows
            if str(row.get("sg_policy_step_frac", "nan")).lower() != "nan"
        ]
        if not values:
            continue
        values.sort()
        ax.plot([item[0] for item in values], [item[1] for item in values], label=key.replace("_current", ""), color=color)

    ax.axvspan(1, WARM_START_SUBEPISODES, color="#dddddd", alpha=0.24)
    ax.axvspan(WARM_START_SUBEPISODES + 1, LIVE_RELEASE_SUBEPISODE - 1, color="#f3c567", alpha=0.20)
    ax.set_xlabel("subepisode")
    ax.set_ylabel("policy-executed step fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Latest supervisor-gate release behavior")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "fig_latest_gate_policy_fraction.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_horizon_heatmaps(bundles: dict[str, dict]) -> Path:
    specs = [
        ("horizon_current", "DQN latest"),
        ("horizon_previous", "DQN June 6"),
        ("dueling_current", "Dueling latest"),
        ("dueling_wide", "Dueling June 4 wide"),
    ]
    heatmaps = []
    for key, title in specs:
        if key not in bundles:
            continue
        heat, nps, ncs = src.base._heatmap_data(bundles[key], src.tail_slice(bundles[key], TAIL_EPISODES))
        heatmaps.append((title, heat, nps, ncs))
    if not heatmaps:
        raise RuntimeError("No horizon bundles available for heatmap plotting.")

    vmax = max(float(np.nanmax(item[1])) for item in heatmaps)
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(4.1 * len(heatmaps), 4.1), squeeze=False)
    image = None
    for ax, (title, heat, nps, ncs) in zip(axes[0], heatmaps):
        image = ax.imshow(
            heat,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=vmax,
            extent=[min(nps) - 0.5, max(nps) + 0.5, min(ncs) - 0.5, max(ncs) + 0.5],
        )
        ax.set_title(title)
        ax.set_xlabel("Np")
        ax.set_ylabel("Nc")
    fig.subplots_adjust(left=0.05, right=0.89, bottom=0.16, top=0.86, wspace=0.36)
    if image is not None:
        cax = fig.add_axes([0.92, 0.22, 0.012, 0.58])
        fig.colorbar(image, cax=cax, label="tail frequency")
    out = OUT_DIR / "fig_latest_horizon_tail_heatmaps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _markdown_current_summary(summary_rows: list[dict]) -> list[str]:
    lines = [
        "| Run | Timestamp | Tail reward | Delta vs OF-MPC | Final reward | Worst post-warm | Neg post-warm | T85 MAE | x24 MAE | Tail policy frac | First-live policy frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _current_and_baseline(summary_rows):
        lines.append(
            f"| {row['label']} | {row['timestamp']} | {_fmt(row['tail_reward'])} | {_fmt(row['delta_vs_ofmpc'])} | "
            f"{_fmt(row['final_reward'])} | {_fmt(row['worst_postwarm_reward'])} | {row['negative_postwarm_episodes']} | "
            f"{_fmt(row['tail_temp_mae'])} | {_fmt(row['tail_comp_mae'], 5)} | "
            f"{_fmt(row.get('tail_sg_policy_step_frac', float('nan')))} | "
            f"{_fmt(row.get('first_live_sg_policy_step_frac', float('nan')))} |"
        )
    return lines


def _markdown_change_summary(summary_rows: list[dict], comparison_rows: list[dict]) -> list[str]:
    by_key = _by_key(summary_rows)
    lines = [
        "| Family | Reference | Tail reward change | T85 MAE change | x24 MAE change | Neg post-warm change | Tail policy frac change | First-live policy frac change |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        ref = by_key.get(row["reference_key"], {})
        lines.append(
            f"| {row['family']} | {ref.get('label', row['reference_key'])} | {_fmt(row['tail_reward_delta'])} | "
            f"{_fmt(row['tail_temp_mae_delta'])} | {_fmt(row['tail_comp_mae_delta'], 5)} | "
            f"{row['negative_postwarm_delta']} | {_fmt(row['policy_step_frac_delta'])} | "
            f"{_fmt(row['first_live_policy_step_frac_delta'])} |"
        )
    return lines


def _markdown_config(summary_rows: list[dict]) -> list[str]:
    lines = [
        "| Run | State mode | Action freeze | Actor freeze | Margin | Replay cap | Replay size | Exploration |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in _current(summary_rows):
        replay_size = row.get("replay_size", "")
        if replay_size == "" and row["key"] == "markov_current":
            replay_size = "not saved"
        trace_len = int(row.get("param_noise_trace_len") or 0)
        exploration_note = row.get("exploration_note", "")
        if trace_len:
            exploration_note = f"{exploration_note}, trace {trace_len}"
        lines.append(
            f"| {row['label']} | {row.get('config_state_mode', '')} | {row.get('config_action_freeze', '')} | "
            f"{row.get('config_actor_freeze', '')} | {_fmt(row.get('config_advantage_margin', ''), 2)} | "
            f"{row.get('replay_capacity', '')} | {replay_size} | {exploration_note} |"
        )
    return lines


def _markdown_worst(episode_rows: list[dict]) -> list[str]:
    lines = [
        "| Run | Worst subepisode | Reward | T85 MAE | x24 MAE | Policy frac | Median gate advantage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in src._worst_current_episode_rows(episode_rows):
        lines.append(
            f"| {row['label']} | {int(row['subepisode'])} | {_fmt(row['reward'])} | {_fmt(row['temp_mae'])} | "
            f"{_fmt(row['comp_mae'], 5)} | {_fmt(row.get('sg_policy_step_frac', float('nan')))} | "
            f"{_fmt(row.get('sg_adv_median', float('nan')))} |"
        )
    return lines


def _markdown_horizons(summary_rows: list[dict], horizon_pair_rows: list[dict]) -> list[str]:
    by_key = _by_key(summary_rows)
    lines = ["| Run | Tail top pairs |", "|---|---|"]
    for key in ["horizon_current", "horizon_previous", "dueling_current", "dueling_wide"]:
        if key in by_key:
            lines.append(f"| {by_key[key]['label']} | {src._top_pair_strings(horizon_pair_rows, key)} |")
    return lines


def _build_report(
    summary_rows: list[dict],
    comparison_rows: list[dict],
    episode_rows: list[dict],
    horizon_pair_rows: list[dict],
    figures: list[Path],
) -> str:
    current = _current(summary_rows)
    by_key = _by_key(summary_rows)
    best_tail = max(current, key=lambda row: float(row["tail_reward"]))
    no_negative = [row for row in current if int(row["negative_postwarm_episodes"]) == 0]
    best_no_negative = max(no_negative, key=lambda row: float(row["tail_reward"])) if no_negative else None
    weights = by_key["weights_current"]
    residual = by_key["residual_current"]
    markov = by_key["markov_current"]
    horizon = by_key["horizon_current"]
    dueling = by_key["dueling_current"]

    lines: list[str] = []
    lines.append("# Distillation Column Latest Runner Analysis")
    lines.append("")
    lines.append(f"Generated on {REPORT_DATE} from saved result bundles only. Aspen was not relaunched.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report analyzes the latest saved disturbance/fluctuation mismatch batch for the active distillation column runners: "
        "weights SG-TD3, residual SG-TD3, Markov SG-TD3, horizon SG-DQN, and dueling horizon SG-DQN. The disturbed OF-MPC "
        "trajectory in `Distillation/Data/mpc_results_disturb_fluctuation.pickle` is used as the baseline. Older standard, nominal, "
        "matrix, structured-matrix, residual-TD7, and combined artifacts remain in the repository, but they are not mixed into the "
        "headline comparison because they were produced under different runner families or older configurations."
    )
    lines.append("")
    lines.append("## Executive Result")
    lines.append("")
    lines.append(
        f"The best tail reward in the latest active batch is **{best_tail['label']}** with tail reward "
        f"{_fmt(best_tail['tail_reward'])}, which is {_fmt(best_tail['delta_vs_ofmpc'])} above OF-MPC. "
        f"The best no-negative-post-warm run is **{best_no_negative['label']}** with tail reward "
        f"{_fmt(best_no_negative['tail_reward'])}." if best_no_negative else "No current run avoided negative post-warm episodes."
    )
    lines.append("")
    lines.append(
        "The main scientific tradeoff is still authority versus reliability, but the latest residual run is the strongest current result: "
        "it has the best tail reward and avoids negative post-warm episodes. Markov remains almost as strong in the tail but has the visible "
        "negative post-warm episode. Weights is the lower-authority continuous reference because its fallback is the identity multiplier. "
        "The horizon agents are stable in this batch but still weaker than the continuous SG-TD3 families."
    )
    lines.append("")
    lines.extend(_markdown_current_summary(summary_rows))
    lines.append("")
    lines.append("![Tail reward and negative post-warm episodes](%s)" % _report_rel(figures[0]))
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append(
        "The distillation plant outputs are tray-24 ethane composition and tray-85 temperature, and the manipulated inputs are reflux "
        "flow and reboiler duty. The baseline controller is an offset-free linear MPC built from the column identification artifacts. "
        "All active RL runners keep MPC as the move generator or fallback rather than replacing it with direct black-box control."
    )
    lines.append("")
    lines.append(
        "$$ \\min_{\\Delta U} \\sum_{i=1}^{N_p} (y_{k+i}-y_{\\mathrm{sp},k+i})^\\top Q (y_{k+i}-y_{\\mathrm{sp},k+i}) + \\sum_{i=0}^{N_c-1} \\Delta u_{k+i}^\\top R \\Delta u_{k+i}. $$"
    )
    lines.append("")
    lines.append(
        "The comparison rescored every saved trajectory with the current distillation reward helper. In compact form, the reward is a "
        "tracking-and-move penalty plus a smooth in-band bonus:"
    )
    lines.append("")
    lines.append(
        "$$ r_t = -e_t^\\top Q e_t - \\Delta u_t^\\top R \\Delta u_t - \\ell_{\\mathrm{band}}(e_t,y_{\\mathrm{sp},t}) + b_{\\mathrm{inside}}(e_t,y_{\\mathrm{sp},t}). $$"
    )
    lines.append("")
    lines.append(
        "The exact rescoring code computes scaled output errors, scaled input moves, a physical tolerance band using "
        "`k_rel = [0.3, 0.01]`, `band_floor_phys = [0.003, 0.2]`, `Q_diag = [3.7e4, 2.0e4]`, and `R_diag = [2.5e3, 2.5e3]`, "
        "then adds a geometric in-band bonus. This avoids comparing runs on incompatible logged reward revisions."
    )
    lines.append("")
    lines.append("The SG-TD3 runners use a critic-based execution gate:")
    lines.append("")
    lines.append(
        "$$ a_{\\mathrm{exec}} = a_{\\mathrm{rl}}\\ \\mathrm{if}\\ S(a_{\\mathrm{rl}})-S(a_{\\mathrm{sup}})>m,\\ \\mathrm{else}\\ a_{\\mathrm{sup}}. $$"
    )
    lines.append("")
    lines.append(
        "For weights, `a_sup` is the identity multiplier. For residual, it is zero residual. For Markov, it is the LS-or-MPC Markov "
        "correction. The horizon agents use the same idea over the discrete triangular action set "
        "`Np = 6..11`, `Nc = 3..Np`, with `(6, 3)` as the OF-MPC supervisor recipe."
    )
    lines.append("")
    lines.extend(_markdown_config(summary_rows))
    lines.append("")
    lines.append("## Changes From Closest References")
    lines.append("")
    lines.append(
        "Positive reward change is good. Negative T85 or x24 MAE change is good. The reference rows are the closest saved mismatch "
        "runs from the preceding analysis batches, not a full hyperparameter sweep."
    )
    lines.append("")
    lines.extend(_markdown_change_summary(summary_rows, comparison_rows))
    lines.append("")
    lines.append("![Change relative to closest mismatch reference](%s)" % _report_rel(figures[1]))
    lines.append("")
    lines.append("## Learning And Release Evidence")
    lines.append("")
    lines.append(
        "The episode traces separate warm start, critic-only handoff, and live policy release. The worst-post-warm table is used because "
        "a high final tail reward can hide a damaging early live episode."
    )
    lines.append("")
    lines.append("![Episode reward histories](%s)" % _report_rel(figures[2]))
    lines.append("")
    lines.append("![Supervisor-gate policy fraction](%s)" % _report_rel(figures[3]))
    lines.append("")
    lines.extend(_markdown_worst(episode_rows))
    lines.append("")
    lines.append("## Horizon Diagnostics")
    lines.append("")
    lines.append(
        "Both horizon agents now have limited-grid mismatch runs. The heatmaps show whether the learned policy is concentrating on a "
        "small stable recipe subset or simply exploring the 39 valid recipes."
    )
    lines.append("")
    lines.extend(_markdown_horizons(summary_rows, horizon_pair_rows))
    lines.append("")
    lines.append("![Horizon tail heatmaps](%s)" % _report_rel(figures[4]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"**Weights SG-TD3.** Tail reward is {_fmt(weights['tail_reward'])}, with "
        f"{weights['negative_postwarm_episodes']} negative post-warm episodes. This is the conservative continuous reference: it changes "
        "the MPC tradeoff rather than directly adding input corrections, but it does not match the latest residual tail reward."
    )
    lines.append("")
    lines.append(
        f"**Residual SG-TD3.** Tail reward is {_fmt(residual['tail_reward'])}, and the worst post-warm reward is "
        f"{_fmt(residual['worst_postwarm_reward'])}. This is the best latest run, not only the highest-reward run. Because accepted "
        "residuals are plant-facing input corrections, it still needs seed replication before it should be treated as robust."
    )
    lines.append("")
    lines.append(
        f"**Markov SG-TD3.** Tail reward is {_fmt(markov['tail_reward'])}, with "
        f"{markov['negative_postwarm_episodes']} negative post-warm episode. The mechanism is model correction rather than direct input "
        "correction, so weak episodes should be diagnosed through supervisor/candidate scoring and LS-or-MPC correction quality."
    )
    lines.append("")
    lines.append(
        f"**Horizon SG-DQN.** Tail reward is {_fmt(horizon['tail_reward'])}. The action is a horizon recipe, so the method is low-authority "
        "and stable here, but still far behind the best continuous supervisors."
    )
    lines.append("")
    lines.append(
        f"**Dueling SG-DQN.** Tail reward is {_fmt(dueling['tail_reward'])}. Dueling helped create a saved limited-grid mismatch run, but the "
        "performance gap to residual and Markov shows that value decomposition alone is not enough for this column scenario."
    )
    lines.append("")
    lines.append("## Bugs, Inconsistencies, And Risks")
    lines.append("")
    lines.append(
        "- The June 7 analysis folder was stale after the June 8 reruns. This report uses the latest saved timestamps and writes a new dated output folder."
    )
    lines.append(
        "- Markov replay size is not saved in the same way as the other current bundles, so the report records it as `not saved` instead of inferring it."
    )
    lines.append(
        "- The comparison is single-seed per latest runner. Tail rankings should be treated as batch evidence, not a statistical conclusion."
    )
    lines.append(
        "- Some older result families are not included because their saved configs are not like-for-like with the latest SG warm-3 mismatch batch."
    )
    lines.append("")
    lines.append("## Literature Connections")
    lines.append("")
    lines.append(
        "No local BibTeX file was found during this pass, so no new formal citations were added. The interpretation follows the local "
        "`StatsControl2026/rl_assisted_mpc_algorithm_slides_2026_06_02.tex` framing: RL changes an MPC design knob, while MPC and the "
        "supervisor gate define the executable controller envelope. The residual-risk interpretation also matches the local method notes "
        "that residual actions are direct input corrections, whereas weights and horizon changes act through MPC."
    )
    lines.append("")
    lines.append("## Recommended Next Experiments")
    lines.append("")
    lines.append(
        "1. Treat residual SG-TD3 as the current candidate winner and weights SG-TD3 as the conservative continuous baseline. Run two more "
        "seeds for both with the same warm-3 and replay-40k settings, and use tail reward, T85 MAE, x24 MAE, and negative post-warm episodes "
        "as the acceptance metrics."
    )
    lines.append(
        "2. For residual SG-TD3, isolate margin from exploration. Keep parameter noise fixed and sweep `advantage_margin` over `0.0`, `0.25`, "
        "and `0.5`, or keep margin fixed and compare Gaussian versus parameter noise."
    )
    lines.append(
        "3. For Markov SG-TD3, inspect low-reward episodes by gate advantage and policy fraction. If bad episodes have low policy acceptance, tune "
        "the LS-or-MPC supervisor or scoring terms before making the actor more conservative."
    )
    lines.append(
        "4. For horizon and dueling horizon, keep the triangular limited grid for one more paired run. Narrow only if the tail heatmaps again "
        "concentrate around low control horizons."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("Files inspected:")
    lines.append("")
    inspected = [
        "report/scripts/analyze_distillation_sg_replay40k_warm3_20260607.py",
        "report/scripts/analyze_distillation_sg_limited_horizons_20260606.py",
        "report/scripts/analyze_distillation_dueling_horizon_history_20260604.py",
        "distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py",
        "distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py",
        "distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py",
        "systems/distillation/config.py",
        "systems/distillation/notebook_params.py",
        "systems/distillation/labels.py",
        "StatsControl2026/rl_assisted_mpc_algorithm_slides_2026_06_02.tex",
    ]
    for item in inspected:
        lines.append(f"- `{item}`")
    for row in summary_rows:
        lines.append(f"- `{_repo_rel(row['path'])}`")
    lines.append("")
    lines.append("Generated outputs:")
    lines.append("")
    outputs = [REPORT_PATH, SUMMARY_CSV, SUMMARY_JSON, EPISODE_CSV, HORIZON_PAIR_CSV, *figures]
    for item in outputs:
        lines.append(f"- `{_repo_rel(item)}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, comparison_rows, episode_rows, bundles, horizon_pair_rows = src._load_rows()
    figures = [
        _plot_tail_summary(summary_rows),
        _plot_current_vs_reference(comparison_rows),
        _plot_episode_rewards(episode_rows),
        _plot_gate_fractions(episode_rows),
        _plot_horizon_heatmaps(bundles),
    ]

    src._write_csv(SUMMARY_CSV, summary_rows)
    src._write_csv(EPISODE_CSV, episode_rows)
    src._write_csv(HORIZON_PAIR_CSV, horizon_pair_rows)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "report": _repo_rel(REPORT_PATH),
                "summary": summary_rows,
                "comparisons": comparison_rows,
                "figures": [_repo_rel(path) for path in figures],
                "tail_episodes": TAIL_EPISODES,
                "warm_start_subepisodes": WARM_START_SUBEPISODES,
                "handover_subepisodes": HANDOVER_SUBEPISODES,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        _build_report(summary_rows, comparison_rows, episode_rows, horizon_pair_rows, figures),
        encoding="utf-8",
    )
    print(f"wrote {_repo_rel(REPORT_PATH)}")
    print(f"wrote {_repo_rel(SUMMARY_CSV)}")
    print(f"wrote {len(figures)} figures under {_repo_rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
