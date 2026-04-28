from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("project1_results_summary.png")


def main():
    # Data transcribed from the manuscript steady-state absolute tracking-error tables.
    # Values are percent reductions in final |e| of the proposed controller relative
    # to the earlier MPC-pretrained RL baseline.
    polymer_reductions = np.array([98, 90, 92, 76, 92, 84, 84, 90], dtype=float)
    c2_reductions = np.array(
        [
            99.16051,
            99.62571,
            99.16372,
            89.69083,
            94.89069,
            91.05442,
            81.30891,
            90.90287,
            91.61054,
            60.87472,
            35.77701,
            97.76745,
        ],
        dtype=float,
    )

    cases = ["Polymer", "C$_2$ splitter"]
    reductions = np.array([polymer_reductions.mean(), c2_reductions.mean()])
    low = np.array([polymer_reductions.min(), c2_reductions.min()])
    high = np.array([polymer_reductions.max(), c2_reductions.max()])

    previous_error = np.array([100.0, 100.0])
    our_error = 100.0 - reductions
    our_low = 100.0 - high
    our_high = 100.0 - low

    maroon = "#7a003c"
    blue = "#205285"
    gray = "#8a8f98"
    light_gray = "#d8dbe0"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, ax = plt.subplots(figsize=(3.25, 1.55), dpi=300)

    y = np.arange(len(cases))[::-1]

    ax.barh(y, previous_error, height=0.42, color=light_gray, edgecolor=gray, linewidth=0.7)
    ax.barh(y, our_error, height=0.42, color=blue, edgecolor=blue, linewidth=0.7)

    err_lower = our_error - our_low
    err_upper = our_high - our_error
    ax.errorbar(
        our_error,
        y,
        xerr=[err_lower, err_upper],
        fmt="none",
        ecolor=maroon,
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        zorder=4,
    )

    for idx, yy in enumerate(y):
        ax.text(
            our_error[idx] + 4,
            yy,
            f"{reductions[idx]:.0f}% less",
            ha="left",
            va="center",
            color=blue,
            fontweight="bold",
            fontsize=8,
        )

    ax.text(100, y[0] + 0.33, "Previous work = 100%", ha="right", va="center", color="#4a5058", fontsize=6.5)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.65, 1.65)
    ax.set_yticks(y)
    ax.set_yticklabels(cases, fontweight="bold")
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("")
    ax.set_title("Our approach reduces final offset", color=maroon, fontweight="bold", pad=2)
    ax.grid(axis="x", color="#e7e8ec", linewidth=0.7)
    ax.set_axisbelow(True)

    ax.text(
        0.5,
        0.02,
        "MPC baseline: slower settling",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#3f444c",
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b8bdc6")
    ax.spines["bottom"].set_color("#b8bdc6")

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
