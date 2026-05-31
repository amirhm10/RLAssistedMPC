"""Analyze the May 30 distillation no-probation, high-temperature-reward batch.

This script reads saved bundles only. It reuses the metric definitions from the
previous five-runner analysis so comparisons stay consistent across batches.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "report" / "figures" / "distillation_post_reward_no_probation_20260531"
BASE_SCRIPT = ROOT / "report" / "scripts" / "analyze_distillation_wider_safety_20260530.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("distillation_wider_safety_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def latest_runs():
    return {
        "baseline": {
            "label": "OF-MPC",
            "method": "baseline",
            "path": ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
        },
        "weights": {
            "label": "TD3 weights",
            "method": "weights",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
            / "20260530_220604"
            / "input_data.pkl",
        },
        "horizon": {
            "label": "Horizon DDQN",
            "method": "horizon",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_horizon_disturb_fluctuation_mismatch_unified"
            / "20260530_225843"
            / "input_data.pkl",
        },
        "dueling": {
            "label": "Dueling horizon",
            "method": "dueling_horizon",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
            / "20260530_223346"
            / "input_data.pkl",
        },
        "residual": {
            "label": "TD3 residual",
            "method": "residual",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
            / "20260530_231105"
            / "input_data.pkl",
        },
        "markov": {
            "label": "TD3 Markov",
            "method": "markov",
            "path": ROOT
            / "Distillation"
            / "Results"
            / "distillation_markov_td3_disturb_fluctuation_unified"
            / "20260530_230956"
            / "input_data.pkl",
        },
    }


def previous_runs():
    return {
        "weights": ROOT
        / "Distillation"
        / "Results"
        / "distillation_weights_td3_disturb_fluctuation_mismatch_unified"
        / "20260529_211201"
        / "input_data.pkl",
        "horizon": ROOT
        / "Distillation"
        / "Results"
        / "distillation_horizon_disturb_fluctuation_mismatch_unified"
        / "20260529_213627"
        / "input_data.pkl",
        "dueling": ROOT
        / "Distillation"
        / "Results"
        / "distillation_dueling_horizon_disturb_fluctuation_mismatch_unified"
        / "20260529_213847"
        / "input_data.pkl",
        "residual": ROOT
        / "Distillation"
        / "Results"
        / "distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified"
        / "20260529_213534"
        / "input_data.pkl",
        "markov": ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_unified"
        / "20260529_220507"
        / "input_data.pkl",
    }


def baseline_compare_path():
    return (
        ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_residual_td3_disturb_fluctuation"
        / "20260530_231118"
        / "input_data.pkl"
    )


def _baseline_delta_rows(current_summary: pd.DataFrame) -> pd.DataFrame:
    current_idx = current_summary.set_index("key")
    baseline = current_idx.loc["baseline"]
    rows = []
    for key in ["weights", "horizon", "dueling", "residual", "markov"]:
        row = current_idx.loc[key]
        rows.append(
            {
                "key": key,
                "label": row["label"],
                "tail20_reward_delta_vs_baseline": float(row["reward_tail20"] - baseline["reward_tail20"]),
                "final_reward_delta_vs_baseline": float(row["reward_final"] - baseline["reward_final"]),
                "first20_min_delta_vs_baseline": float(row["reward_first20_min"] - baseline["reward_first20_min"]),
                "band_norm_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_band_norm_mae"] / baseline["tail20_band_norm_mae"] - 1.0)
                ),
                "outside_band_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_outside_band_frac"] / baseline["tail20_outside_band_frac"] - 1.0)
                ),
                "composition_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_comp_mae"] / baseline["tail20_comp_mae"] - 1.0)
                ),
                "temperature_mae_delta_pct_vs_baseline": float(
                    100.0 * (row["tail20_temp_mae"] / baseline["tail20_temp_mae"] - 1.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _previous_delta_rows(current_summary: pd.DataFrame, previous_summary: pd.DataFrame) -> pd.DataFrame:
    current_idx = current_summary.set_index("key")
    previous_idx = previous_summary.set_index("key")
    rows = []
    for key in ["weights", "horizon", "dueling", "residual", "markov"]:
        cur = current_idx.loc[key]
        prev = previous_idx.loc[key]
        rows.append(
            {
                "key": key,
                "label": cur["label"],
                "current_tail20": float(cur["reward_tail20"]),
                "previous_tail20": float(prev["reward_tail20"]),
                "tail20_delta": float(cur["reward_tail20"] - prev["reward_tail20"]),
                "current_final": float(cur["reward_final"]),
                "previous_final": float(prev["reward_final"]),
                "final_delta": float(cur["reward_final"] - prev["reward_final"]),
                "current_first20_min": float(cur["reward_first20_min"]),
                "previous_first20_min": float(prev["reward_first20_min"]),
                "first20_min_delta": float(cur["reward_first20_min"] - prev["reward_first20_min"]),
                "band_norm_mae_delta": float(cur["tail20_band_norm_mae"] - prev["tail20_band_norm_mae"]),
                "outside_band_frac_delta": float(cur["tail20_outside_band_frac"] - prev["tail20_outside_band_frac"]),
                "composition_mae_delta": float(cur["tail20_comp_mae"] - prev["tail20_comp_mae"]),
                "temperature_mae_delta": float(cur["tail20_temp_mae"] - prev["tail20_temp_mae"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    base = load_base_module()
    base.OUT_DIR = OUT_DIR
    base.CURRENT_RUNS = latest_runs()
    base.PREVIOUS_RUNS = previous_runs()
    base.BASELINE_COMPARE_PATH = baseline_compare_path()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current = {key: load_pickle(meta["path"]) for key, meta in base.CURRENT_RUNS.items()}
    baseline_compare = load_pickle(base.BASELINE_COMPARE_PATH)
    current["baseline"] = dict(current["baseline"])
    current["baseline"]["avg_rewards"] = np.asarray(baseline_compare["avg_rewards_mpc"], float)
    previous = {key: load_pickle(path) for key, path in base.PREVIOUS_RUNS.items()}

    current_rows = [
        base.row_for(key, meta, current[key], "2026-05-30 no-probation high-temperature reward")
        for key, meta in base.CURRENT_RUNS.items()
    ]
    current_summary = pd.DataFrame(current_rows)
    current_summary.to_csv(OUT_DIR / "current_summary_metrics.csv", index=False)

    previous_rows = []
    for key, path in base.PREVIOUS_RUNS.items():
        meta = dict(base.CURRENT_RUNS[key])
        meta["path"] = path
        previous_rows.append(base.row_for(key, meta, previous[key], "2026-05-29 no-probation baseline"))
    previous_summary = pd.DataFrame(previous_rows)
    previous_summary.to_csv(OUT_DIR / "previous_summary_metrics.csv", index=False)

    baseline_compare_df = _baseline_delta_rows(current_summary)
    baseline_compare_df.to_csv(OUT_DIR / "current_vs_baseline_metrics.csv", index=False)

    current_vs_previous = _previous_delta_rows(current_summary, previous_summary)
    current_vs_previous.to_csv(OUT_DIR / "current_vs_previous_metrics.csv", index=False)

    base.plot_reward(current, current_summary)
    base.plot_current_vs_previous(current_vs_previous)
    base.plot_tracking(current_summary)
    base.plot_safety(current_summary)
    horizon_counts = base.plot_horizon_usage(current)
    base.plot_weight_residual_markov(current_summary, current)

    summary = {
        "current_rank_tail20": current_summary.sort_values("reward_tail20", ascending=False)[
            [
                "label",
                "reward_tail20",
                "reward_tail10",
                "reward_final",
                "reward_first20_min",
                "tail20_band_norm_mae",
                "tail20_outside_band_frac",
            ]
        ].to_dict(orient="records"),
        "current_vs_baseline": baseline_compare_df.to_dict(orient="records"),
        "current_vs_previous": current_vs_previous.to_dict(orient="records"),
        "safety_highlights": current_summary[
            [
                "key",
                "label",
                "tail20_weight_cap_projection_frac",
                "tail20_horizon_projection_frac",
                "tail20_residual_cap_projection_frac",
                "tail20_fallback_frac",
                "weight_probation_trigger_count",
                "horizon_probation_trigger_count",
                "residual_probation_trigger_count",
                "markov_probation_trigger_count",
            ]
        ].to_dict(orient="records"),
        "top_horizon_pairs": horizon_counts.groupby("runner").head(5).to_dict(orient="records"),
        "artifacts": {
            "figure_dir": str(OUT_DIR.relative_to(ROOT)),
            "csvs": sorted(p.name for p in OUT_DIR.glob("*.csv")),
            "figures": sorted(p.name for p in OUT_DIR.glob("fig_*.png")),
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
