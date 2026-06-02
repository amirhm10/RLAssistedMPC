"""Follow-up analysis for distillation weights reward provenance and SG gate.

The script reads saved result bundles only. It extends the June 1 weight
rescoring by adding the June 2 SG-TD3 weights run, so historical TD3/SAC
weights and the current SG-TD3 result can be compared under the same active
distillation reward defaults.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    }
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.distillation.config import RL_REWARD_DEFAULTS


OUT_DIR = ROOT / "report" / "figures" / "distillation_weights_reward_gate_followup_20260602"
LATEST_SCRIPT = ROOT / "report" / "scripts" / "analyze_distillation_weights_latest_20260601.py"

TD3_ROOT = ROOT / "Distillation" / "Results" / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
SAC_ROOT = ROOT / "Distillation" / "Results" / "distillation_weights_sac_disturb_fluctuation_mismatch_unified"
BASELINE_PATH = ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
SG_TD3_PATH = (
    ROOT
    / "Distillation"
    / "Results"
    / "distillation_weights_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch"
    / "20260602_140102"
    / "input_data.pkl"
)

SOURCE_NAMES = {
    0: "warm_start",
    1: "supervisor",
    2: "policy",
    3: "held",
    4: "fallback",
}


def load_latest_module():
    spec = importlib.util.spec_from_file_location("weights_latest_20260601", LATEST_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {LATEST_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_mean(values) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def add_sg_gate_metrics(row: dict, bundle: dict, latest) -> None:
    source = bundle.get("sg_selected_source_log")
    if source is None:
        return
    sl = latest.step_slice(bundle, 20)
    arr = np.asarray(source)[sl]
    arr = arr[np.isfinite(arr)].astype(int)
    if arr.size == 0:
        return
    for code, name in SOURCE_NAMES.items():
        row[f"tail20_sg_{name}_frac"] = float(np.mean(arr == code))
    row["tail20_sg_policy_plus_supervisor_frac"] = float(np.mean((arr == 1) | (arr == 2)))


def add_reward_provenance(row: dict) -> None:
    stored = float(row.get("stored_tail20_reward", np.nan))
    current = float(row.get("current_tail20_reward", np.nan))
    row["tail20_logged_minus_current_reward"] = stored - current
    row["tail20_current_minus_baseline_reward"] = np.nan


def summarize_all_runs(latest) -> pd.DataFrame:
    rows: list[dict] = []

    baseline = latest.load_pickle(BASELINE_PATH)
    row = latest.summarize_bundle("baseline", "OF-MPC", BASELINE_PATH, baseline)
    row["label"] = "OF-MPC"
    row["agent_kind"] = "of_mpc"
    row["notebook_source"] = "baseline"
    add_reward_provenance(row)
    rows.append(row)

    for kind, root in (("td3", TD3_ROOT), ("sac", SAC_ROOT)):
        for path in sorted(root.glob("*/input_data.pkl")):
            run = path.parent.name
            bundle = latest.load_pickle(path)
            row = latest.summarize_bundle(kind, run, path, bundle)
            row["label"] = f"{kind.upper()} weights {run}"
            row["agent_kind"] = str(bundle.get("agent_kind", kind))
            row["notebook_source"] = str(bundle.get("notebook_source", "unknown"))
            add_reward_provenance(row)
            rows.append(row)

    sg_bundle = latest.load_pickle(SG_TD3_PATH)
    row = latest.summarize_bundle("sg_td3", "20260602_140102", SG_TD3_PATH, sg_bundle)
    row["label"] = "SG-TD3 weights 20260602_140102"
    row["agent_kind"] = str(sg_bundle.get("agent_kind", "sg_td3"))
    row["notebook_source"] = str(sg_bundle.get("notebook_source", "unknown"))
    add_sg_gate_metrics(row, sg_bundle, latest)
    add_reward_provenance(row)
    rows.append(row)

    df = pd.DataFrame(rows)
    baseline_reward = float(
        df.loc[df["kind"].eq("baseline"), "current_tail20_reward"].iloc[0]
    )
    df["tail20_current_minus_baseline_reward"] = df["current_tail20_reward"] - baseline_reward
    df["rank_current_tail20_reward"] = df["current_tail20_reward"].rank(
        ascending=False,
        method="min",
    )
    return df.sort_values("current_tail20_reward", ascending=False).reset_index(drop=True)


def compact_list(value, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return ""
    return "[" + ", ".join(f"{x:.{digits}f}" for x in arr.ravel()) + "]"


def write_tables(df: pd.DataFrame) -> None:
    core_cols = [
        "kind",
        "run",
        "agent_kind",
        "current_tail20_reward",
        "stored_tail20_reward",
        "tail20_logged_minus_current_reward",
        "tail20_current_minus_baseline_reward",
        "current_final_reward",
        "tail20_comp_mae",
        "tail20_temp_mae",
        "tail20_band_norm_mae",
        "tail20_weight_mean",
        "tail20_sg_policy_frac",
        "tail20_sg_supervisor_frac",
    ]
    existing = [col for col in core_cols if col in df.columns]
    table = df[existing].copy()
    for col in table.columns:
        if col.endswith("_weight_mean"):
            table[col] = table[col].map(compact_list)
    table.to_csv(OUT_DIR / "reward_gate_followup_summary.csv", index=False)

    selected = df[
        df["run"].isin(
            [
                "OF-MPC",
                "20260521_150600",
                "20260528_194904",
                "20260530_220604",
                "20260601_155305",
                "20260602_140102",
                "20260518_142138",
            ]
        )
    ].copy()
    selected.to_csv(OUT_DIR / "reward_gate_selected_runs.csv", index=False)


def plot_tail_ranking(df: pd.DataFrame) -> None:
    selected_runs = [
        "20260528_194904",
        "20260521_150600",
        "20260602_140102",
        "20260518_142138",
        "20260530_220604",
        "OF-MPC",
        "20260601_155305",
    ]
    plot_df = df[df["run"].isin(selected_runs)].copy()
    plot_df["plot_label"] = plot_df.apply(
        lambda row: "OF-MPC"
        if row["kind"] == "baseline"
        else f"{row['kind'].upper()}\n{row['run']}",
        axis=1,
    )
    plot_df = plot_df.sort_values("current_tail20_reward", ascending=True)

    colors = [
        "#5B8DEF" if kind == "td3" else "#4DB6AC" if kind == "sac" else "#C77DFF" if kind == "sg_td3" else "#737373"
        for kind in plot_df["kind"]
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.barh(plot_df["plot_label"], plot_df["current_tail20_reward"], color=colors)
    baseline = float(df.loc[df["kind"].eq("baseline"), "current_tail20_reward"].iloc[0])
    ax.axvline(baseline, color="#404040", linestyle="--", linewidth=1.1, label="OF-MPC tail-20")
    ax.set_xlabel("Tail-20 reward rescored with active 2026-06-02 reward")
    ax.set_title("Distillation Weights Under The Same Active Reward")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail20_current_reward_with_sg.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reward_drift(df: pd.DataFrame) -> None:
    plot_df = df[~df["kind"].eq("baseline")].copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    color_map = {"td3": "#5B8DEF", "sac": "#4DB6AC", "sg_td3": "#C77DFF"}
    for kind, group in plot_df.groupby("kind"):
        ax.scatter(
            group["stored_tail20_reward"],
            group["current_tail20_reward"],
            s=58,
            alpha=0.85,
            color=color_map.get(kind, "#737373"),
            label=kind.upper(),
        )
    lo = float(np.nanmin([plot_df["stored_tail20_reward"].min(), plot_df["current_tail20_reward"].min()]))
    hi = float(np.nanmax([plot_df["stored_tail20_reward"].max(), plot_df["current_tail20_reward"].max()]))
    pad = 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#505050", linestyle="--", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Logged tail-20 reward from saved run")
    ax.set_ylabel("Tail-20 reward rescored with active 2026-06-02 reward")
    ax.set_title("Logged Reward Versus Same-Reward Rescore")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_logged_vs_current_reward_drift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tail_tracking_tradeoff(df: pd.DataFrame) -> None:
    selected = df[
        df["run"].isin(
            [
                "OF-MPC",
                "20260521_150600",
                "20260528_194904",
                "20260530_220604",
                "20260601_155305",
                "20260602_140102",
                "20260518_142138",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    colors = {"td3": "#5B8DEF", "sac": "#4DB6AC", "sg_td3": "#C77DFF", "baseline": "#737373"}
    for _, row in selected.iterrows():
        ax.scatter(
            row["tail20_comp_mae"],
            row["tail20_temp_mae"],
            s=110,
            color=colors.get(row["kind"], "#737373"),
        )
        label = "OF-MPC" if row["kind"] == "baseline" else row["run"][-6:]
        label_offsets = {
            "OF-MPC": (4, -12),
            "155305": (4, 6),
            "194904": (4, 3),
            "150600": (4, 3),
            "142138": (4, 3),
            "220604": (4, -10),
            "140102": (4, 3),
        }
        ax.annotate(
            label,
            (row["tail20_comp_mae"], row["tail20_temp_mae"]),
            xytext=label_offsets.get(label, (4, 3)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Tail-20 composition MAE")
    ax.set_ylabel("Tail-20 temperature MAE")
    ax.set_title("Tail Tracking Tradeoff For Selected Weight Runs")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_tail_tracking_tradeoff_selected.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    best = df.iloc[0].to_dict()
    baseline = df[df["kind"].eq("baseline")].iloc[0].to_dict()
    sg = df[df["kind"].eq("sg_td3")].iloc[0].to_dict()
    latest_td3 = df[df["run"].eq("20260601_155305")].iloc[0].to_dict()

    summary = {
        "reward_defaults": {
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in RL_REWARD_DEFAULTS.items()
        },
        "best_current_tail20": {
            "kind": best["kind"],
            "run": best["run"],
            "current_tail20_reward": best["current_tail20_reward"],
            "stored_tail20_reward": best["stored_tail20_reward"],
            "tail20_comp_mae": best["tail20_comp_mae"],
            "tail20_temp_mae": best["tail20_temp_mae"],
            "tail20_weight_mean": best.get("tail20_weight_mean"),
        },
        "baseline": {
            "current_tail20_reward": baseline["current_tail20_reward"],
            "tail20_comp_mae": baseline["tail20_comp_mae"],
            "tail20_temp_mae": baseline["tail20_temp_mae"],
        },
        "sg_td3_weights": {
            "current_tail20_reward": sg["current_tail20_reward"],
            "stored_tail20_reward": sg["stored_tail20_reward"],
            "tail20_current_minus_baseline_reward": sg["tail20_current_minus_baseline_reward"],
            "tail20_comp_mae": sg["tail20_comp_mae"],
            "tail20_temp_mae": sg["tail20_temp_mae"],
            "tail20_band_norm_mae": sg["tail20_band_norm_mae"],
            "tail20_sg_policy_frac": sg.get("tail20_sg_policy_frac"),
            "tail20_sg_supervisor_frac": sg.get("tail20_sg_supervisor_frac"),
        },
        "latest_plain_td3": {
            "run": latest_td3["run"],
            "current_tail20_reward": latest_td3["current_tail20_reward"],
            "stored_tail20_reward": latest_td3["stored_tail20_reward"],
            "tail20_current_minus_baseline_reward": latest_td3["tail20_current_minus_baseline_reward"],
        },
        "artifacts": {
            "figure_dir": str(OUT_DIR.relative_to(ROOT)),
            "csvs": [
                "reward_gate_followup_summary.csv",
                "reward_gate_selected_runs.csv",
            ],
            "figures": [
                "fig_tail20_current_reward_with_sg.png",
                "fig_logged_vs_current_reward_drift.png",
                "fig_tail_tracking_tradeoff_selected.png",
            ],
        },
    }
    with (OUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = load_latest_module()
    df = summarize_all_runs(latest)
    write_tables(df)
    plot_tail_ranking(df)
    plot_reward_drift(df)
    plot_tail_tracking_tradeoff(df)
    write_summary(df)
    print(f"Wrote follow-up analysis to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
