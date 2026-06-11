from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "polymer_param_noise_markov_residual_20260610"

BASELINE_PATH = (
    REPO_ROOT
    / "Polymer"
    / "Results"
    / "mpc_offsetfree_disturb_unified"
    / "20260608_143324"
    / "input_data.pkl"
)

RUNS = [
    {
        "key": "markov_gaussian_z07",
        "family": "Markov",
        "variant": "Gaussian reference",
        "exploration": "gaussian 0.20 -> 0.02",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch"
        / "20260609_220217"
        / "input_data.pkl",
        "notes": "Same Markov mismatch family and z_bound 0.7 before the polymer parameter-noise default switch.",
    },
    {
        "key": "markov_param_z07",
        "family": "Markov",
        "variant": "Parameter-noise latest",
        "exploration": "param-noise 0.10 -> 0.02",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch"
        / "20260610_170822"
        / "input_data.pkl",
        "notes": "Latest user-run Markov param-noise bundle.",
    },
    {
        "key": "residual_gaussian",
        "family": "Residual",
        "variant": "Gaussian reference",
        "exploration": "gaussian 0.20 -> 0.02",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_residual_critic_warm3_conservative_disturb_mismatch"
        / "20260608_152126"
        / "input_data.pkl",
        "notes": "Latest residual mismatch reference before the polymer residual parameter-noise default switch.",
    },
    {
        "key": "residual_param",
        "family": "Residual",
        "variant": "Parameter-noise latest",
        "exploration": "param-noise 0.10 -> 0.02",
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_residual_critic_warm3_conservative_disturb_mismatch"
        / "20260610_131706"
        / "input_data.pkl",
        "notes": "Latest user-run residual param-noise bundle.",
    },
]


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def as_array(bundle: dict[str, Any], key: str, *, dtype=float) -> np.ndarray:
    value = bundle.get(key)
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def finite_min(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmin(values))


def fraction(values: np.ndarray, code: int) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values == code))


def bool_fraction(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values.astype(bool)))


def mean_norm(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return float(np.nanmean(np.linalg.norm(values, axis=1)))


def q95_norm(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return float(np.nanquantile(np.linalg.norm(values, axis=1), 0.95))


def episode_slices(bundle: dict[str, Any], n_episodes: int, n_steps: int) -> dict[str, slice]:
    steps_per_episode = int(bundle.get("time_in_sub_episodes") or (n_steps // max(n_episodes, 1)))
    warm_start_step = int(bundle.get("warm_start_step") or (10 * steps_per_episode))
    warm_episode = int(np.ceil(warm_start_step / max(steps_per_episode, 1)))
    # Active polymer defaults use three post-warm actor-freeze subepisodes.
    live_episode = min(n_episodes, warm_episode + 3)
    return {
        "tail_ep": slice(max(0, n_episodes - 20), n_episodes),
        "post_ep": slice(warm_episode, n_episodes),
        "first_live_ep": slice(live_episode, min(n_episodes, live_episode + 20)),
        "tail_step": slice(max(0, n_steps - 20 * steps_per_episode), n_steps),
        "post_step": slice(warm_start_step, n_steps),
        "first_live_step": slice(live_episode * steps_per_episode, min(n_steps, (live_episode + 20) * steps_per_episode)),
    }


def tracking_metrics(bundle: dict[str, Any], step_slice: slice, *, baseline: bool = False) -> dict[str, float]:
    y_key = "y_mpc" if baseline else "y_rl"
    y = as_array(bundle, y_key)
    y_sp = as_array(bundle, "y_sp")
    if y.size == 0 or y_sp.size == 0:
        return {
            "tracking_mae_y1": float("nan"),
            "tracking_mae_y2": float("nan"),
            "tracking_rmse_y1": float("nan"),
            "tracking_rmse_y2": float("nan"),
            "tracking_max_abs_y1": float("nan"),
            "tracking_max_abs_y2": float("nan"),
        }
    y_aligned = y[1 : y_sp.shape[0] + 1, :]
    err = np.asarray(y_aligned[step_slice, :] - y_sp[step_slice, :], float)
    if err.size == 0:
        return {
            "tracking_mae_y1": float("nan"),
            "tracking_mae_y2": float("nan"),
            "tracking_rmse_y1": float("nan"),
            "tracking_rmse_y2": float("nan"),
            "tracking_max_abs_y1": float("nan"),
            "tracking_max_abs_y2": float("nan"),
        }
    return {
        "tracking_mae_y1": float(np.nanmean(np.abs(err[:, 0]))),
        "tracking_mae_y2": float(np.nanmean(np.abs(err[:, 1]))),
        "tracking_rmse_y1": float(np.sqrt(np.nanmean(err[:, 0] ** 2))),
        "tracking_rmse_y2": float(np.sqrt(np.nanmean(err[:, 1] ** 2))),
        "tracking_max_abs_y1": float(np.nanmax(np.abs(err[:, 0]))),
        "tracking_max_abs_y2": float(np.nanmax(np.abs(err[:, 1]))),
    }


def input_metrics(bundle: dict[str, Any], step_slice: slice, *, baseline: bool = False) -> dict[str, float]:
    u_key = "u_mpc" if baseline else "u_rl"
    u = as_array(bundle, u_key)
    if u.size == 0:
        return {
            "mean_abs_du_u1": float("nan"),
            "mean_abs_du_u2": float("nan"),
            "total_variation_u": float("nan"),
        }
    u_window = np.asarray(u[step_slice, :], float)
    if u_window.shape[0] < 2:
        return {
            "mean_abs_du_u1": float("nan"),
            "mean_abs_du_u2": float("nan"),
            "total_variation_u": float("nan"),
        }
    du = np.diff(u_window, axis=0)
    return {
        "mean_abs_du_u1": float(np.nanmean(np.abs(du[:, 0]))),
        "mean_abs_du_u2": float(np.nanmean(np.abs(du[:, 1]))),
        "total_variation_u": float(np.nansum(np.abs(du))),
    }


def summarize_baseline(bundle: dict[str, Any]) -> dict[str, Any]:
    avg = as_array(bundle, "avg_rewards").reshape(-1)
    n_steps = int(bundle.get("nFE") or as_array(bundle, "y_sp").shape[0])
    sl = episode_slices(bundle, avg.size, n_steps)
    row: dict[str, Any] = {
        "key": "of_mpc",
        "family": "OF-MPC",
        "variant": "disturbed baseline",
        "exploration": "none",
        "path": str(BASELINE_PATH.relative_to(REPO_ROOT)),
        "timestamp": BASELINE_PATH.parent.name,
        "status": "ok",
        "tail_reward": finite_mean(avg[sl["tail_ep"]]),
        "final_reward": float(avg[-1]) if avg.size else float("nan"),
        "worst_postwarm_reward": finite_min(avg[sl["post_ep"]]),
        "first_live_reward": finite_mean(avg[sl["first_live_ep"]]),
        "negative_postwarm_episodes": int(np.sum(avg[sl["post_ep"]] < 0.0)) if avg.size else 0,
        "tail_policy_fraction": float("nan"),
        "first_live_policy_fraction": float("nan"),
        "tail_supervisor_fraction": float("nan"),
        "tail_mpc_supervisor_fraction": float("nan"),
        "tail_ls_supervisor_fraction": float("nan"),
        "tail_action_authority_norm": float("nan"),
        "tail_action_authority_q95": float("nan"),
        "tail_projection_fraction": float("nan"),
        "tail_raw_executed_gap": float("nan"),
        "tail_gate_advantage": float("nan"),
        "param_noise_start": float("nan"),
        "param_noise_mid": float("nan"),
        "param_noise_end": float("nan"),
        "exploration_magnitude_start": float("nan"),
        "exploration_magnitude_mid": float("nan"),
        "exploration_magnitude_end": float("nan"),
        "notes": "Disturbed OF-MPC baseline.",
    }
    row.update({f"tail_{k}": v for k, v in tracking_metrics(bundle, sl["tail_step"], baseline=True).items()})
    row.update({f"tail_{k}": v for k, v in input_metrics(bundle, sl["tail_step"], baseline=True).items()})
    return row


def trace_triplet(bundle: dict[str, Any], key: str) -> tuple[float, float, float]:
    values = as_array(bundle, key).reshape(-1)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(values[0]), float(values[values.size // 2]), float(values[-1])


def summarize_run(meta: dict[str, Any], baseline_tail_reward: float) -> dict[str, Any]:
    path = Path(meta["path"])
    if not path.exists():
        return {
            "key": meta["key"],
            "family": meta["family"],
            "variant": meta["variant"],
            "exploration": meta["exploration"],
            "path": str(path.relative_to(REPO_ROOT)),
            "timestamp": path.parent.name,
            "status": "missing",
            "missing_reason": "input_data.pkl not found",
        }
    bundle = load_pickle(path)
    avg = as_array(bundle, "avg_rewards").reshape(-1)
    n_steps = int(bundle.get("nFE") or as_array(bundle, "y_sp").shape[0])
    sl = episode_slices(bundle, avg.size, n_steps)
    family = str(meta["family"]).lower()

    sg_sources = as_array(bundle, "sg_selected_source_log", dtype=int).reshape(-1)
    sg_adv = as_array(bundle, "sg_advantage_log").reshape(-1)

    if family == "markov":
        action_sources = as_array(bundle, "rl_action_source_log", dtype=int).reshape(-1)
        action_authority = as_array(bundle, "z_executed_log")
        tail_policy_fraction = fraction(action_sources[sl["tail_step"]], 2)
        first_live_policy_fraction = fraction(action_sources[sl["first_live_step"]], 2)
        tail_ls_fraction = fraction(action_sources[sl["tail_step"]], 6)
        tail_mpc_fraction = fraction(action_sources[sl["tail_step"]], 7)
        tail_supervisor_fraction = tail_ls_fraction + tail_mpc_fraction
        projection = as_array(bundle, "z_safety_requested_projection_active_log", dtype=int)
        raw = as_array(bundle, "rl_requested_raw_action_log")
        executed = as_array(bundle, "rl_executed_raw_action_log")
    else:
        action_authority = as_array(bundle, "delta_u_res_exec_log")
        tail_policy_fraction = fraction(sg_sources[sl["tail_step"]], 2)
        first_live_policy_fraction = fraction(sg_sources[sl["first_live_step"]], 2)
        tail_supervisor_fraction = fraction(sg_sources[sl["tail_step"]], 1)
        tail_ls_fraction = float("nan")
        tail_mpc_fraction = float("nan")
        projection = as_array(bundle, "projection_due_to_authority_log", dtype=int)
        raw = as_array(bundle, "delta_u_res_raw_log")
        executed = as_array(bundle, "delta_u_res_exec_log")

    p0, pmid, pend = trace_triplet(bundle, "param_noise_scale_trace")
    e0, emid, eend = trace_triplet(bundle, "exploration_magnitude_trace")

    if raw.size and executed.size:
        gap = np.asarray(raw - executed, float)
    else:
        gap = np.asarray([], float)

    row: dict[str, Any] = {
        "key": meta["key"],
        "family": meta["family"],
        "variant": meta["variant"],
        "exploration": meta["exploration"],
        "path": str(path.relative_to(REPO_ROOT)),
        "timestamp": path.parent.name,
        "status": "ok",
        "markov_z_bound": bundle.get("markov_z_bound", float("nan")),
        "tail_reward": finite_mean(avg[sl["tail_ep"]]),
        "tail_reward_delta_vs_ofmpc": finite_mean(avg[sl["tail_ep"]]) - baseline_tail_reward,
        "final_reward": float(avg[-1]) if avg.size else float("nan"),
        "worst_postwarm_reward": finite_min(avg[sl["post_ep"]]),
        "first_live_reward": finite_mean(avg[sl["first_live_ep"]]),
        "negative_postwarm_episodes": int(np.sum(avg[sl["post_ep"]] < 0.0)) if avg.size else 0,
        "tail_policy_fraction": tail_policy_fraction,
        "first_live_policy_fraction": first_live_policy_fraction,
        "tail_supervisor_fraction": tail_supervisor_fraction,
        "tail_mpc_supervisor_fraction": tail_mpc_fraction,
        "tail_ls_supervisor_fraction": tail_ls_fraction,
        "tail_action_authority_norm": mean_norm(action_authority[sl["tail_step"]]),
        "tail_action_authority_q95": q95_norm(action_authority[sl["tail_step"]]),
        "tail_projection_fraction": bool_fraction(projection[sl["tail_step"]]) if projection.size else float("nan"),
        "tail_raw_executed_gap": mean_norm(gap[sl["tail_step"]]) if gap.size else float("nan"),
        "tail_gate_advantage": finite_mean(sg_adv[sl["tail_step"]]) if sg_adv.size else float("nan"),
        "param_noise_start": p0,
        "param_noise_mid": pmid,
        "param_noise_end": pend,
        "exploration_magnitude_start": e0,
        "exploration_magnitude_mid": emid,
        "exploration_magnitude_end": eend,
        "notes": meta["notes"],
    }
    row.update({f"tail_{k}": v for k, v in tracking_metrics(bundle, sl["tail_step"], baseline=False).items()})
    row.update({f"tail_{k}": v for k, v in input_metrics(bundle, sl["tail_step"], baseline=False).items()})
    return row


def write_csv(rows: list[dict[str, Any]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "polymer_param_noise_markov_residual_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "key",
        "family",
        "variant",
        "exploration",
        "timestamp",
        "status",
        "tail_reward",
        "tail_reward_delta_vs_ofmpc",
        "final_reward",
        "worst_postwarm_reward",
        "first_live_reward",
        "tail_policy_fraction",
        "first_live_policy_fraction",
        "tail_supervisor_fraction",
        "tail_mpc_supervisor_fraction",
        "tail_ls_supervisor_fraction",
        "tail_action_authority_norm",
        "tail_action_authority_q95",
        "tail_projection_fraction",
        "tail_raw_executed_gap",
        "tail_gate_advantage",
        "tail_tracking_mae_y1",
        "tail_tracking_mae_y2",
        "tail_mean_abs_du_u1",
        "tail_mean_abs_du_u2",
        "param_noise_start",
        "param_noise_mid",
        "param_noise_end",
        "exploration_magnitude_start",
        "exploration_magnitude_mid",
        "exploration_magnitude_end",
        "markov_z_bound",
        "path",
        "notes",
    ]
    fieldnames = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_figures(rows: list[dict[str, Any]], bundles: dict[str, dict[str, Any]]) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.0), constrained_layout=True)
    colors = {
        "markov_gaussian_z07": "#7f3b08",
        "markov_param_z07": "#2166ac",
        "residual_gaussian": "#b35806",
        "residual_param": "#1b7837",
        "of_mpc": "#555555",
    }
    labels = {
        "markov_gaussian_z07": "Markov Gaussian z=0.7",
        "markov_param_z07": "Markov param-noise",
        "residual_gaussian": "Residual Gaussian",
        "residual_param": "Residual param-noise",
        "of_mpc": "OF-MPC",
    }
    for key in ["of_mpc", "markov_gaussian_z07", "markov_param_z07"]:
        avg = as_array(bundles[key], "avg_rewards").reshape(-1)
        axs[0].plot(np.arange(1, avg.size + 1), avg, lw=1.5, color=colors[key], label=labels[key])
    axs[0].axvline(10, color="black", lw=0.8, ls="--", alpha=0.5)
    axs[0].axvspan(10, 13, color="0.8", alpha=0.25)
    axs[0].set_title("Markov reward history")
    axs[0].set_xlabel("Subepisode")
    axs[0].set_ylabel("Average reward")
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=8)

    for key in ["of_mpc", "residual_gaussian", "residual_param"]:
        avg = as_array(bundles[key], "avg_rewards").reshape(-1)
        axs[1].plot(np.arange(1, avg.size + 1), avg, lw=1.5, color=colors[key], label=labels[key])
    axs[1].axvline(10, color="black", lw=0.8, ls="--", alpha=0.5)
    axs[1].axvspan(10, 13, color="0.8", alpha=0.25)
    axs[1].set_title("Residual reward history")
    axs[1].set_xlabel("Subepisode")
    axs[1].set_ylabel("Average reward")
    axs[1].grid(alpha=0.25)
    axs[1].legend(fontsize=8)
    path = OUT_DIR / "fig_polymer_param_noise_reward_histories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    metric_rows = [row for row in rows if row["key"] != "of_mpc"]
    x = np.arange(len(metric_rows))
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    bar_colors = [colors[row["key"]] for row in metric_rows]
    xticks = [row["key"].replace("_", "\n") for row in metric_rows]
    axs[0].bar(x, [row.get("tail_reward_delta_vs_ofmpc", np.nan) for row in metric_rows], color=bar_colors)
    axs[0].axhline(0.0, color="black", lw=0.8)
    axs[0].set_title("Tail reward gain vs OF-MPC")
    axs[0].set_ylabel("reward delta")
    axs[0].set_xticks(x, xticks, fontsize=7)
    axs[0].grid(axis="y", alpha=0.25)

    axs[1].bar(x, [row.get("worst_postwarm_reward", np.nan) for row in metric_rows], color=bar_colors)
    axs[1].set_title("Worst post-warm episode")
    axs[1].set_ylabel("reward")
    axs[1].set_xticks(x, xticks, fontsize=7)
    axs[1].grid(axis="y", alpha=0.25)

    axs[2].bar(x, [row.get("tail_policy_fraction", np.nan) for row in metric_rows], color=bar_colors)
    axs[2].set_ylim(0.0, 1.0)
    axs[2].set_title("Tail policy execution")
    axs[2].set_ylabel("fraction")
    axs[2].set_xticks(x, xticks, fontsize=7)
    axs[2].grid(axis="y", alpha=0.25)

    path = OUT_DIR / "fig_polymer_param_noise_metric_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def write_report(rows: list[dict[str, Any]], figure_paths: list[Path], csv_path: Path) -> Path:
    report_path = REPO_ROOT / "report" / "polymer_param_noise_markov_residual_comparison_2026_06_10.md"
    row_by_key = {row["key"]: row for row in rows}
    mg = row_by_key["markov_gaussian_z07"]
    mp = row_by_key["markov_param_z07"]
    rg = row_by_key["residual_gaussian"]
    rp = row_by_key["residual_param"]

    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "NA"
        try:
            value = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"

    lines = [
        "# Polymer Markov/Residual Parameter-Noise Check",
        "",
        "Date: 2026-06-10",
        "",
        "## Scope",
        "",
        "This note compares the latest polymer standalone Markov and residual SG-TD3 parameter-noise runs against the previous Gaussian-action-noise references. It reads saved `input_data.pkl` bundles only and does not launch polymer simulations.",
        "",
        "## Main Conclusion",
        "",
        "The polymer parameter-noise switch looks acceptable. Markov improved clearly relative to the same `z_bound = 0.7` Gaussian reference, and residual improved mildly relative to the June 8 Gaussian reference. The remaining caution is that all polymer reward values are still negative under the saved reward convention, so the decision should be based on relative reward, handoff behavior, tracking, and source diagnostics rather than the sign of reward.",
        "",
        "## Summary Table",
        "",
        "| Family | Variant | Tail reward | Delta vs OF-MPC | Worst post-warm | First-live reward | Tail policy frac | Tail authority | Tail gate adv | Tail y1 MAE | Tail y2 MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ["of_mpc", "markov_gaussian_z07", "markov_param_z07", "residual_gaussian", "residual_param"]:
        row = row_by_key[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["family"]),
                    str(row["variant"]),
                    fmt(row.get("tail_reward")),
                    fmt(row.get("tail_reward_delta_vs_ofmpc")),
                    fmt(row.get("worst_postwarm_reward")),
                    fmt(row.get("first_live_reward")),
                    fmt(row.get("tail_policy_fraction")),
                    fmt(row.get("tail_action_authority_norm")),
                    fmt(row.get("tail_gate_advantage")),
                    fmt(row.get("tail_tracking_mae_y1")),
                    fmt(row.get("tail_tracking_mae_y2")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Markov param-noise tail reward is `{fmt(mp['tail_reward'])}`, compared with `{fmt(mg['tail_reward'])}` for the Gaussian `z_bound = 0.7` reference and `{fmt(row_by_key['of_mpc']['tail_reward'])}` for OF-MPC.",
            f"- Markov worst post-warm reward improves from `{fmt(mg['worst_postwarm_reward'])}` to `{fmt(mp['worst_postwarm_reward'])}`. This is the strongest safety signal in favor of the parameter-noise rerun.",
            f"- Markov tail policy execution increases from `{fmt(mg['tail_policy_fraction'])}` to `{fmt(mp['tail_policy_fraction'])}`, while tail reward and worst-postwarm reward both improve. That suggests the parameter-noise actor was more usable for the SG gate, not merely more aggressive.",
            f"- Residual param-noise tail reward is `{fmt(rp['tail_reward'])}`, compared with `{fmt(rg['tail_reward'])}` for the Gaussian reference. The improvement is modest but positive.",
            f"- Residual worst post-warm reward improves from `{fmt(rg['worst_postwarm_reward'])}` to `{fmt(rp['worst_postwarm_reward'])}`, but the latest residual run is still not as clean as the best older non-mismatch residual family. Treat it as acceptable, not a dramatic win.",
            "",
            "## Recommendation",
            "",
            "- Keep polymer Markov and residual parameter noise `0.10 -> 0.02` as the default for the next combined/standalone checks.",
            "- For Markov, the result supports using parameter noise in the final standalone table because it improves tail reward and reduces the worst handoff/post-warm downside versus the previous Gaussian `z_bound = 0.7` run.",
            "- For residual, parameter noise is acceptable and slightly better than the immediate Gaussian reference, but the claim should be weaker: it is a polish improvement, not a new best mechanism by itself.",
            "- In the paper, report both the standalone Gaussian reference and parameter-noise final rows, because this is a useful example where temporally coherent exploration improves the high-authority Markov block without removing the SG envelope.",
            "",
            "## Artifacts",
            "",
            f"- `{csv_path.relative_to(REPO_ROOT).as_posix()}`",
        ]
    )
    for figure in figure_paths:
        lines.append(f"- `{figure.relative_to(REPO_ROOT).as_posix()}`")
    lines.extend(
        [
            "",
            "## Files Inspected",
            "",
            f"- `{BASELINE_PATH.relative_to(REPO_ROOT).as_posix()}`",
        ]
    )
    for meta in RUNS:
        lines.append(f"- `{Path(meta['path']).relative_to(REPO_ROOT).as_posix()}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_pickle(BASELINE_PATH)
    rows = [summarize_baseline(baseline)]
    baseline_tail_reward = float(rows[0]["tail_reward"])
    bundles = {"of_mpc": baseline}
    for meta in RUNS:
        rows.append(summarize_run(meta, baseline_tail_reward))
        path = Path(meta["path"])
        if path.exists():
            bundles[meta["key"]] = load_pickle(path)

    csv_path = write_csv(rows)
    json_path = OUT_DIR / "polymer_param_noise_markov_residual_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    figure_paths = make_figures(rows, bundles)
    report_path = write_report(rows, figure_paths, csv_path)

    print(f"Wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    for figure in figure_paths:
        print(f"Wrote {figure.relative_to(REPO_ROOT)}")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
