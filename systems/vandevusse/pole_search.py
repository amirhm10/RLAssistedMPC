from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from utils.helpers import apply_min_max, reverse_min_max

from .baseline_mpc import prepare_vandevusse_offset_free_mpc_runtime, run_vandevusse_offset_free_mpc


def sample_vandevusse_observer_poles(
    rng,
    low=0.55,
    high=0.92,
    mode="uniform",
    n_states=6,
):
    rng = np.random.default_rng(rng)
    low = float(low)
    high = float(high)
    if not (0.0 < low < high < 1.0):
        raise ValueError("Observer pole search bounds must satisfy 0 < low < high < 1.")

    mode = str(mode).strip().lower()
    if mode == "uniform":
        poles = rng.uniform(low, high, size=int(n_states))
    elif mode == "mixed":
        n_states = int(n_states)
        n_low = n_states // 2
        n_high = n_states - n_low
        split = min(max(0.75, low), high)
        low_band = rng.uniform(low, split, size=n_low)
        high_band = rng.uniform(split, high, size=n_high)
        poles = np.concatenate((low_band, high_band), axis=0)
    else:
        raise ValueError("mode must be 'uniform' or 'mixed'.")
    return np.sort(np.asarray(poles, dtype=float))


def _bundle_setpoint_phys(bundle):
    bundle_y_sp = np.asarray(bundle["y_sp"], dtype=float)
    data_min = np.asarray(bundle["data_min"], dtype=float)
    data_max = np.asarray(bundle["data_max"], dtype=float)
    steady_y = np.asarray(bundle["steady_states"]["y_ss"], dtype=float)
    n_inputs = int(np.asarray(bundle["u"], dtype=float).shape[1])
    y_ss_scaled = apply_min_max(steady_y, data_min[n_inputs:], data_max[n_inputs:])
    return reverse_min_max(bundle_y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def _compute_pole_search_metrics(bundle):
    y_meas = np.asarray(bundle["y"], dtype=float)[1:, :]
    u_phys = np.asarray(bundle["u"], dtype=float)
    y_target = _bundle_setpoint_phys(bundle)

    mean_abs_cb_error = float(np.mean(np.abs(y_meas[:, 0] - y_target[:, 0])))
    mean_abs_T_error = float(np.mean(np.abs(y_meas[:, 1] - y_target[:, 1])))
    mean_abs_diff_F = float(np.mean(np.abs(np.diff(u_phys[:, 0])))) if u_phys.shape[0] > 1 else 0.0
    mean_abs_diff_QK = float(np.mean(np.abs(np.diff(u_phys[:, 1])))) if u_phys.shape[0] > 1 else 0.0
    mean_abs_diff_cb = float(np.mean(np.abs(np.diff(y_meas[:, 0])))) if y_meas.shape[0] > 1 else 0.0
    mean_abs_diff_T = float(np.mean(np.abs(np.diff(y_meas[:, 1])))) if y_meas.shape[0] > 1 else 0.0

    score = (
        10.0 * mean_abs_cb_error
        + 5.0 * mean_abs_T_error
        + 1.0 * mean_abs_diff_F
        + 1.0 * mean_abs_diff_QK
        + 2.0 * mean_abs_diff_cb
        + 2.0 * mean_abs_diff_T
    )

    return {
        "score": float(score),
        "mean_abs_cb_error": mean_abs_cb_error,
        "mean_abs_T_error": mean_abs_T_error,
        "mean_abs_diff_F": mean_abs_diff_F,
        "mean_abs_diff_QK": mean_abs_diff_QK,
        "mean_abs_diff_cb": mean_abs_diff_cb,
        "mean_abs_diff_T": mean_abs_diff_T,
    }


def score_vandevusse_observer_poles(
    repo_root,
    baseline_cfg,
    poles,
    *,
    run_mode="nominal",
    disturbance_profile="none",
    n_tests_override=None,
    set_points_len_override=None,
    test_cycle_override=None,
):
    cfg = deepcopy(baseline_cfg)
    cfg["run_mode"] = str(run_mode).lower()
    cfg["disturbance_profile"] = str(disturbance_profile).lower()
    cfg["observer_poles_override"] = np.asarray(poles, dtype=float).copy()
    if n_tests_override is not None:
        cfg["n_tests_override"] = int(n_tests_override)
    if set_points_len_override is not None:
        cfg["set_points_len_override"] = int(set_points_len_override)
    if test_cycle_override is not None:
        cfg["test_cycle_override"] = list(test_cycle_override)

    try:
        prepared = prepare_vandevusse_offset_free_mpc_runtime(repo_root=Path(repo_root), baseline_cfg=cfg)
        bundle = run_vandevusse_offset_free_mpc(prepared)
        metrics = _compute_pole_search_metrics(bundle)
        return {
            "valid": True,
            "score": float(metrics["score"]),
            "sampled_poles": np.asarray(poles, dtype=float).copy(),
            "used_poles": np.asarray(bundle["observer_poles_used"], dtype=float).copy(),
            "used_fallback": bool(bundle["observer_used_fallback"]),
            "observer_spectral_radius": float(bundle["observer_error_spectral_radius"]),
            "observer_update_mode": str(bundle["observer_update_mode"]),
            **metrics,
            "error": None,
            "result_bundle": bundle,
        }
    except Exception as exc:
        return {
            "valid": False,
            "score": float("inf"),
            "sampled_poles": np.asarray(poles, dtype=float).copy(),
            "used_poles": None,
            "used_fallback": None,
            "observer_spectral_radius": None,
            "observer_update_mode": str(cfg.get("observer_update_mode", "")),
            "mean_abs_cb_error": None,
            "mean_abs_T_error": None,
            "mean_abs_diff_F": None,
            "mean_abs_diff_QK": None,
            "mean_abs_diff_cb": None,
            "mean_abs_diff_T": None,
            "error": str(exc),
            "result_bundle": None,
        }


def run_vandevusse_observer_pole_search(
    repo_root,
    baseline_cfg,
    *,
    n_samples=50,
    seed=42,
    low=0.55,
    high=0.92,
    mode="uniform",
    top_k=5,
    n_tests_override=None,
    set_points_len_override=None,
    test_cycle_override=None,
):
    rng = np.random.default_rng(int(seed))
    results = []
    for idx in range(int(n_samples)):
        sampled_poles = sample_vandevusse_observer_poles(
            rng=rng,
            low=low,
            high=high,
            mode=mode,
            n_states=6,
        )
        scored = score_vandevusse_observer_poles(
            repo_root=repo_root,
            baseline_cfg=baseline_cfg,
            poles=sampled_poles,
            run_mode="nominal",
            disturbance_profile="none",
            n_tests_override=n_tests_override,
            set_points_len_override=set_points_len_override,
            test_cycle_override=test_cycle_override,
        )
        scored["sample_index"] = idx
        results.append(scored)

    rows = []
    for row in results:
        rows.append(
            {
                "rank": None,
                "sample_index": row["sample_index"],
                "valid": bool(row["valid"]),
                "score": row["score"],
                "sampled_poles": None if row["sampled_poles"] is None else np.asarray(row["sampled_poles"], float).round(6).tolist(),
                "used_poles": None if row["used_poles"] is None else np.asarray(row["used_poles"], float).round(6).tolist(),
                "used_fallback": row["used_fallback"],
                "observer_spectral_radius": row["observer_spectral_radius"],
                "mean_abs_cb_error": row["mean_abs_cb_error"],
                "mean_abs_T_error": row["mean_abs_T_error"],
                "mean_abs_diff_F": row["mean_abs_diff_F"],
                "mean_abs_diff_QK": row["mean_abs_diff_QK"],
                "mean_abs_diff_cb": row["mean_abs_diff_cb"],
                "mean_abs_diff_T": row["mean_abs_diff_T"],
                "error": row["error"],
            }
        )

    results_df = pd.DataFrame(rows)
    valid_df = results_df[results_df["valid"]].sort_values("score", ascending=True).reset_index(drop=True)
    if not valid_df.empty:
        valid_df["rank"] = np.arange(1, len(valid_df) + 1)
    top_df = valid_df.head(int(top_k)).copy()

    best_row = None
    if not top_df.empty:
        best_index = int(top_df.iloc[0]["sample_index"])
        best_row = results[best_index]

    return {
        "results": results,
        "results_df": results_df,
        "top_df": top_df,
        "best_row": best_row,
    }


__all__ = [
    "run_vandevusse_observer_pole_search",
    "sample_vandevusse_observer_poles",
    "score_vandevusse_observer_poles",
]
