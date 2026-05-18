from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "safety_layer_and_online_poles_20260518"

MARKOV_TD3_LATEST = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified"
    / "20260518_091937"
    / "input_data.pkl"
)

# Live distillation residual values pasted by the user on 2026-05-18.
DISTILLATION_RESIDUAL_LIVE_REWARDS = np.asarray(
    [
        8.175754939675388,
        14.588388925422613,
        14.61353885430912,
        14.641646436212728,
        14.670011463625908,
        14.699633829182869,
        14.732365458787708,
        14.773747246446526,
        14.81523864726871,
        14.859037702498833,
        14.90098039943715,
        14.94778471467249,
        15.007134840731737,
        15.056602336902323,
        15.432758530276454,
        2.6813131678698774,
        2.2857440296616436,
        2.306153042285366,
        2.328659003351555,
        2.3528708055908525,
        3.130475815218773,
        4.590732257032838,
        4.948252680749001,
        6.2804485832348975,
        6.263260879700279,
        6.297185040307515,
        6.3152060122183755,
        6.696430206661721,
        5.7710193509594605,
        3.52300321129007,
        3.385912856716576,
        3.5339957754628784,
        3.0924338648702876,
        3.082474735079218,
        3.0128329995774146,
        2.749097569107341,
        3.1241998911662416,
        5.069530045374313,
        4.38154477060934,
        4.321827999198339,
        4.252870644236306,
        4.278195845934716,
        2.8902358530522854,
        4.573943786345139,
        4.092096394491689,
        4.185102472176542,
        6.430108587303119,
        7.020220760478534,
        3.9904181114067567,
        4.205611107620112,
    ],
    dtype=float,
)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def plot_bad_episode_motivation(markov_rewards: np.ndarray) -> Path:
    episodes_markov = np.arange(1, min(50, markov_rewards.size) + 1)
    episodes_residual = np.arange(1, DISTILLATION_RESIDUAL_LIVE_REWARDS.size + 1)

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    ax.plot(
        episodes_markov,
        markov_rewards[: episodes_markov.size],
        color="#0B6E4F",
        linewidth=2.2,
        marker="o",
        markersize=3.0,
        label="Distillation Markov TD3-only, no safeguard",
    )
    ax.plot(
        episodes_residual,
        DISTILLATION_RESIDUAL_LIVE_REWARDS,
        color="#7C3AED",
        linewidth=2.0,
        marker="s",
        markersize=3.0,
        label="Distillation residual live run",
    )
    ax.axhline(0.0, color="0.35", linewidth=0.9)
    ax.axvspan(10.5, 16.5, color="#F59E0B", alpha=0.12, label="release/shock region")
    ax.set_title("Motivation for a generic safety layer: high-potential methods still have bad-release episodes")
    ax.set_xlabel("Sub-episode")
    ax.set_ylabel("Average reward")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9)
    out = OUT_DIR / "fig_bad_episode_motivation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def add_box(ax, xy: tuple[float, float], width: float, height: float, text: str, face: str, edge: str = "#1F2937") -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2.0, xy[1] + height / 2.0, text, ha="center", va="center", fontsize=10)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#111827"},
    )


def plot_safety_layer_flow() -> Path:
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    add_box(ax, (0.03, 0.58), 0.18, 0.18, "Plant state\nand setpoint", "#E5E7EB")
    add_box(ax, (0.28, 0.70), 0.18, 0.16, "Nominal MPC\nsafe reference", "#DBEAFE")
    add_box(ax, (0.28, 0.44), 0.18, 0.16, "RL method\nMarkov / residual /\nweights / poles", "#EDE9FE")
    add_box(ax, (0.55, 0.55), 0.22, 0.24, "Candidate safety evaluator\n\ncost margin\npredicted error\ninput headroom\ninnovation risk", "#FEF3C7")
    add_box(ax, (0.84, 0.70), 0.13, 0.14, "execute\ncandidate", "#DCFCE7")
    add_box(ax, (0.84, 0.47), 0.13, 0.14, "attenuate\nor blend", "#FDE68A")
    add_box(ax, (0.84, 0.24), 0.13, 0.14, "fallback to\nnominal MPC", "#FEE2E2")
    add_box(ax, (0.55, 0.18), 0.22, 0.15, "Replay stores\nexecuted action\nand safety reason", "#E0F2FE")

    add_arrow(ax, (0.21, 0.67), (0.28, 0.78))
    add_arrow(ax, (0.21, 0.64), (0.28, 0.52))
    add_arrow(ax, (0.46, 0.78), (0.55, 0.70))
    add_arrow(ax, (0.46, 0.52), (0.55, 0.62))
    add_arrow(ax, (0.77, 0.70), (0.84, 0.77))
    add_arrow(ax, (0.77, 0.62), (0.84, 0.54))
    add_arrow(ax, (0.77, 0.55), (0.84, 0.31))
    add_arrow(ax, (0.90, 0.47), (0.70, 0.33))
    add_arrow(ax, (0.90, 0.24), (0.70, 0.33))

    ax.text(
        0.66,
        0.91,
        "The gate should be method-independent: TD3 stays useful, but catastrophic candidates are screened before plant actuation.",
        ha="center",
        va="center",
        fontsize=11,
        color="#111827",
    )

    out = OUT_DIR / "fig_generic_safety_layer_flow.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markov = load_pickle(MARKOV_TD3_LATEST)
    markov_rewards = np.asarray(markov["avg_rewards"], dtype=float)
    figures = {
        "bad_episode_motivation": str(plot_bad_episode_motivation(markov_rewards).relative_to(REPO_ROOT)),
        "safety_layer_flow": str(plot_safety_layer_flow().relative_to(REPO_ROOT)),
    }
    summary = {
        "markov_td3_latest_path": str(MARKOV_TD3_LATEST.relative_to(REPO_ROOT)),
        "markov_td3_min_first50": float(np.min(markov_rewards[:50])),
        "markov_td3_min_first50_episode": int(np.argmin(markov_rewards[:50]) + 1),
        "residual_live_min_first50": float(np.min(DISTILLATION_RESIDUAL_LIVE_REWARDS)),
        "residual_live_min_first50_episode": int(np.argmin(DISTILLATION_RESIDUAL_LIVE_REWARDS) + 1),
        "figures": figures,
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
