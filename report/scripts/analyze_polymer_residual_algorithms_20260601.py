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


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "report" / "figures" / "polymer_residual_algorithm_comparison_20260601"
TABLE_DIR = FIG_DIR

RUN_SPECS = {
    "OF-MPC": {
        "path": REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle",
        "kind": "baseline",
        "color": "#3B3B3B",
    },
    "TD3 Residual": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td3_residual_disturb"
        / "20260601_021504"
        / "input_data.pkl",
        "kind": "residual",
        "color": "#4C78A8",
    },
    "SG-TD3 Residual": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_residual_disturb"
        / "20260601_022723"
        / "input_data.pkl",
        "kind": "residual",
        "color": "#54A24B",
    },
    "TD7 Residual": {
        "path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "td7_residual_disturb"
        / "20260601_022931"
        / "input_data.pkl",
        "kind": "residual",
        "color": "#B279A2",
    },
}

OUTPUT_LABELS = ["eta", "T"]
INPUT_LABELS = ["Qc", "Qm"]


@dataclass
class MethodRun:
    label: str
    spec: dict
    bundle: dict
    nfe: int
    time_in_sub: int
    warm_step: int
    dt: float
    y: np.ndarray
    u: np.ndarray
    y_sp_phys: np.ndarray
    err_phys: np.ndarray
    rewards_step: np.ndarray
    avg_rewards: np.ndarray
    delta_u_scaled: np.ndarray

    @property
    def episode_count(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def warm_episode(self) -> int:
        return int(self.warm_step // max(1, self.time_in_sub))


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _nanmean(values) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanq(values, q: float) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _fraction(values) -> float:
    arr = np.asarray(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _fmt(value, digits=3) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _fmt_pct(value, digits=1) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{100.0 * val:.{digits}f}%"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict], order: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(order) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(order)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(col, "NA")) for col in order) + " |\n")


def _reverse_minmax(x_scaled, lo, hi):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return np.asarray(x_scaled, float) * np.maximum(hi - lo, 1.0e-12) + lo


def ysp_scaled_dev_to_phys(bundle: dict) -> np.ndarray:
    n_inputs = int(bundle.get("n_inputs", 2))
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    y_min = data_min[n_inputs:]
    y_max = data_max[n_inputs:]
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = (y_ss - y_min) / np.maximum(y_max - y_min, 1.0e-12)
    return _reverse_minmax(np.asarray(bundle["y_sp"], float) + y_ss_scaled, y_min, y_max)


def load_method(label: str, spec: dict) -> MethodRun:
    bundle = _load_pickle(spec["path"])
    nfe = int(bundle["nFE"])
    time_in_sub = int(bundle["time_in_sub_episodes"])
    warm_step = int(bundle.get("warm_start_step", time_in_sub * 10))
    dt = float(bundle.get("delta_t", 1.0))
    y_sp_phys = ysp_scaled_dev_to_phys(bundle)[:nfe]

    if spec["kind"] == "baseline":
        y = np.asarray(bundle["y_mpc"], float)
        u = np.asarray(bundle["u_mpc"], float)
        rewards_step = np.asarray(bundle.get("rewards_step", bundle.get("rewards_mpc", [])), float)
        avg_rewards = np.asarray(bundle.get("avg_rewards", bundle.get("avg_rewards_mpc", [])), float)
    else:
        y = np.asarray(bundle.get("y", bundle.get("y_rl")), float)
        u = np.asarray(bundle.get("u", bundle.get("u_rl")), float)
        rewards_step = np.asarray(bundle.get("rewards_step", []), float)
        avg_rewards = np.asarray(bundle.get("avg_rewards", []), float)

    y_eval = y[1 : nfe + 1]
    err_phys = y_eval - y_sp_phys
    delta_u_scaled = np.asarray(bundle.get("delta_u_storage", np.zeros((nfe, 2))), float)[:nfe]
    return MethodRun(
        label=label,
        spec=spec,
        bundle=bundle,
        nfe=nfe,
        time_in_sub=time_in_sub,
        warm_step=warm_step,
        dt=dt,
        y=y,
        u=u,
        y_sp_phys=y_sp_phys,
        err_phys=err_phys,
        rewards_step=rewards_step[:nfe],
        avg_rewards=avg_rewards,
        delta_u_scaled=delta_u_scaled,
    )


def window_masks(run: MethodRun) -> dict[str, np.ndarray]:
    idx = np.arange(run.nfe)
    return {
        "full": np.ones(run.nfe, dtype=bool),
        "postwarm": idx > run.warm_step,
        "tail20": idx >= max(0, run.nfe - 20 * run.time_in_sub),
    }


def performance_metrics(run: MethodRun) -> dict:
    masks = window_masks(run)
    tail = masks["tail20"]
    post = masks["postwarm"]
    e_tail = run.err_phys[tail]
    e_post = run.err_phys[post]
    rewards_tail = run.rewards_step[tail] if run.rewards_step.size else np.array([], dtype=float)
    rewards_post = run.rewards_step[post] if run.rewards_step.size else np.array([], dtype=float)
    du_tail = run.delta_u_scaled[tail]
    episode_tail_count = min(20, run.avg_rewards.size)
    return {
        "Method": run.label,
        "Bundle": run.spec["path"].relative_to(REPO_ROOT).as_posix(),
        "Mean reward": _nanmean(run.avg_rewards),
        "Post-warm reward": _nanmean(run.avg_rewards[run.warm_episode :]),
        "Tail-20 reward": _nanmean(run.avg_rewards[-episode_tail_count:]),
        "Tail step reward": _nanmean(rewards_tail),
        "Post-warm step reward": _nanmean(rewards_post),
        "Tail eta RMSE": float(np.sqrt(np.mean(e_tail[:, 0] ** 2))),
        "Tail T RMSE": float(np.sqrt(np.mean(e_tail[:, 1] ** 2))),
        "Tail eta MAE": float(np.mean(np.abs(e_tail[:, 0]))),
        "Tail T MAE": float(np.mean(np.abs(e_tail[:, 1]))),
        "Tail eta q95 abs": float(np.quantile(np.abs(e_tail[:, 0]), 0.95)),
        "Tail T q95 abs": float(np.quantile(np.abs(e_tail[:, 1]), 0.95)),
        "Post-warm eta RMSE": float(np.sqrt(np.mean(e_post[:, 0] ** 2))),
        "Post-warm T RMSE": float(np.sqrt(np.mean(e_post[:, 1] ** 2))),
        "Tail mean abs du scaled": float(np.mean(np.abs(du_tail))),
    }


def guard_reason_counts(run: MethodRun) -> dict[str, int]:
    active = np.asarray(run.bundle.get("residual_guard_active_log", []), int) == 1
    codes = np.asarray(run.bundle.get("residual_guard_reason_code_log", []), int)
    names = {v: k for k, v in run.bundle.get("residual_guard_reason_codes", {}).items()}
    out = {}
    if active.size and codes.size:
        for code in sorted(np.unique(codes[active]).tolist()):
            out[names.get(int(code), str(code))] = int(np.sum(codes[active] == code))
    return out


def safety_metrics(run: MethodRun) -> dict:
    if run.spec["kind"] == "baseline":
        return {
            "Method": run.label,
            "Ramp": "NA",
            "Post-warm cap clip": None,
            "Guard trigger active": None,
            "Guard accepted": None,
            "Guard objective worse": None,
            "Guard zero selected": None,
            "Headroom projection": None,
            "Zero fallback": None,
            "Shadow rho authority": None,
            "Shadow rho deadband": None,
            "Shadow rho eff mean": None,
            "SG policy selected": None,
            "SG supervisor selected": None,
        }

    post = window_masks(run)["postwarm"]
    guard_active = np.asarray(run.bundle["residual_guard_active_log"], int) == 1
    guard_triggered = np.asarray(run.bundle["residual_guard_triggered_log"], int) == 1
    guard_counts = guard_reason_counts(run)
    ramp = run.bundle.get("td3_authority_ramp", {})
    sg_source = run.bundle.get("sg_selected_source_log")
    if sg_source is not None:
        sg_source = np.asarray(sg_source, int)
        sg_policy = float(np.mean(sg_source[post] == 2))
        sg_supervisor = float(np.mean(sg_source[post] == 1))
    else:
        sg_policy = None
        sg_supervisor = None

    return {
        "Method": run.label,
        "Ramp": f"{ramp.get('start_cap', 'NA')} -> {ramp.get('end_cap', 'NA')}",
        "Post-warm cap clip": _fraction(np.asarray(run.bundle["residual_cap_projection_active_log"], int)[post]),
        "Guard trigger active": _fraction(guard_triggered[guard_active]) if np.any(guard_active) else None,
        "Guard accepted": guard_counts.get("requested_accepted", 0),
        "Guard objective worse": guard_counts.get("objective_worse", 0),
        "Guard zero selected": guard_counts.get("zero_selected", 0),
        "Headroom projection": _fraction(np.asarray(run.bundle["projection_due_to_headroom_log"], int)[post]),
        "Zero fallback": _fraction(np.asarray(run.bundle["residual_zero_fallback_reason_log"], int)[post] != 0),
        "Shadow rho authority": _fraction(np.asarray(run.bundle["shadow_rho_projection_due_to_authority_log"], int)[post]),
        "Shadow rho deadband": _fraction(np.asarray(run.bundle["shadow_rho_deadband_active_log"], int)[post]),
        "Shadow rho eff mean": _nanmean(np.asarray(run.bundle["shadow_rho_eff_log"], float)[post]),
        "SG policy selected": sg_policy,
        "SG supervisor selected": sg_supervisor,
    }


def episode_mean(values: np.ndarray, time_in_sub: int, nfe: int, reducer=np.nanmean) -> np.ndarray:
    values = np.asarray(values, float)
    n_ep = nfe // time_in_sub
    trimmed = values[: n_ep * time_in_sub]
    if trimmed.ndim == 1:
        return reducer(trimmed.reshape(n_ep, time_in_sub), axis=1)
    return reducer(trimmed.reshape(n_ep, time_in_sub, values.shape[1]), axis=1)


def episode_nanmax(values: np.ndarray, time_in_sub: int, nfe: int) -> np.ndarray:
    values = np.asarray(values, float)
    n_ep = nfe // time_in_sub
    trimmed = values[: n_ep * time_in_sub].reshape(n_ep, time_in_sub)
    out = np.full(n_ep, np.nan, dtype=float)
    finite_rows = np.any(np.isfinite(trimmed), axis=1)
    out[finite_rows] = np.nanmax(trimmed[finite_rows], axis=1)
    return out


def rolling_mean(values: np.ndarray, window=5) -> np.ndarray:
    values = np.asarray(values, float)
    if values.size < window:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def plot_rewards(runs: list[MethodRun]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for run in runs:
        ep = np.arange(1, run.avg_rewards.size + 1)
        color = run.spec["color"]
        ax.plot(ep, run.avg_rewards, color=color, alpha=0.28, linewidth=1.0)
        ax.plot(ep, rolling_mean(run.avg_rewards, window=7), color=color, linewidth=2.1, label=run.label)
    ax.axvline(10, color="0.4", linestyle="--", linewidth=1.0, label="warm-start end")
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward")
    ax.set_title("Polymer residual algorithm reward traces")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(FIG_DIR / "reward_curves.png", dpi=180)
    plt.close(fig)


def plot_tail_tracking(runs: list[MethodRun]) -> None:
    ref = runs[0]
    tail_start = max(0, ref.nfe - 20 * ref.time_in_sub)
    step_idx = np.arange(tail_start, ref.nfe)
    x = step_idx / float(ref.time_in_sub) + 1.0
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.6), sharex=True, constrained_layout=True)
    for out_idx, ax in enumerate(axes):
        ax.step(x, ref.y_sp_phys[tail_start : ref.nfe, out_idx], where="post", color="#D62728", linewidth=1.5, label="setpoint")
        for run in runs:
            y = run.y[1 : run.nfe + 1]
            ax.plot(x, y[tail_start : run.nfe, out_idx], color=run.spec["color"], linewidth=1.05, alpha=0.95, label=run.label)
        ax.set_ylabel(OUTPUT_LABELS[out_idx])
        ax.grid(alpha=0.2)
    axes[0].set_title("Tail-20 subepisode output tracking")
    axes[1].set_xlabel("Subepisode")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, ncol=3)
    fig.savefig(FIG_DIR / "tail_tracking_overlay.png", dpi=180)
    plt.close(fig)


def plot_tail_bars(perf_rows: list[dict]) -> None:
    methods = [row["Method"] for row in perf_rows]
    colors = [RUN_SPECS[m]["color"] for m in methods]
    x = np.arange(len(methods))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axes[0].bar(x - width / 2, [row["Tail eta RMSE"] for row in perf_rows], width, color=colors, alpha=0.88)
    axes[0].bar(x + width / 2, [row["Tail T RMSE"] for row in perf_rows], width, color=colors, alpha=0.48)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("RMSE in physical units")
    axes[0].set_title("Tail tracking RMSE")
    axes[0].legend(["eta", "T"], frameon=False)

    axes[1].bar(x, [row["Tail-20 reward"] for row in perf_rows], color=colors, alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("Average reward")
    axes[1].set_title("Tail-20 reward")
    axes[1].axhline(0.0, color="0.25", linewidth=0.9)
    fig.savefig(FIG_DIR / "tail_metric_bars.png", dpi=180)
    plt.close(fig)


def plot_residual_safety(runs: list[MethodRun]) -> None:
    residual_runs = [run for run in runs if run.spec["kind"] == "residual"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), sharex=True, constrained_layout=True)
    ax_cap, ax_clip, ax_guard, ax_shadow = axes.ravel()

    for run in residual_runs:
        ep = np.arange(1, run.episode_count + 1)
        color = run.spec["color"]
        residual_abs_coord = np.max(np.abs(np.asarray(run.bundle["delta_u_res_exec_log"], float)), axis=1)
        q95_res = episode_mean(
            residual_abs_coord,
            run.time_in_sub,
            run.nfe,
            reducer=lambda a, axis: np.nanquantile(a, 0.95, axis=axis),
        )
        cap = episode_nanmax(np.asarray(run.bundle["residual_active_cap_log"], float), run.time_in_sub, run.nfe)
        cap_clip = episode_mean(np.asarray(run.bundle["residual_cap_projection_active_log"], float), run.time_in_sub, run.nfe)
        guard_trig = episode_mean(np.asarray(run.bundle["residual_guard_triggered_log"], float), run.time_in_sub, run.nfe)
        shadow_auth = episode_mean(
            np.asarray(run.bundle["shadow_rho_projection_due_to_authority_log"], float), run.time_in_sub, run.nfe
        )
        ax_cap.plot(ep, q95_res, color=color, linewidth=1.8, label=f"{run.label} q95 abs coord")
        ax_cap.plot(ep, cap, color=color, linewidth=1.1, alpha=0.45, linestyle="--")
        ax_clip.plot(ep, cap_clip, color=color, linewidth=1.7, label=run.label)
        ax_guard.plot(ep, guard_trig, color=color, linewidth=1.7, label=run.label)
        ax_shadow.plot(ep, shadow_auth, color=color, linewidth=1.7, label=run.label)

    ax_cap.axvline(10, color="0.4", linestyle="--", linewidth=0.9)
    ax_clip.axvline(10, color="0.4", linestyle="--", linewidth=0.9)
    ax_guard.axvline(10, color="0.4", linestyle="--", linewidth=0.9)
    ax_shadow.axvline(10, color="0.4", linestyle="--", linewidth=0.9)
    ax_cap.set_ylabel("Scaled delta-u")
    ax_cap.set_title("Residual usage and ramp cap")
    ax_clip.set_ylabel("Fraction")
    ax_clip.set_title("Ramp-cap clipping")
    ax_guard.set_ylabel("Fraction")
    ax_guard.set_title("Early-release guard triggers")
    ax_shadow.set_ylabel("Fraction")
    ax_shadow.set_title("Shadow rho authority projection")
    for ax in axes.ravel():
        ax.set_xlabel("Subepisode")
        ax.grid(alpha=0.2)
    ax_cap.legend(frameon=False, fontsize=8)
    ax_clip.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "residual_safety_dashboard.png", dpi=180)
    plt.close(fig)


def plot_sg_gate(run: MethodRun) -> None:
    source = run.bundle.get("sg_selected_source_log")
    if source is None:
        return
    source = np.asarray(source, int)
    ep = np.arange(1, run.episode_count + 1)
    policy_frac = episode_mean((source == 2).astype(float), run.time_in_sub, run.nfe)
    supervisor_frac = episode_mean((source == 1).astype(float), run.time_in_sub, run.nfe)
    advantage = episode_mean(np.asarray(run.bundle["sg_advantage_log"], float), run.time_in_sub, run.nfe)
    qgap_policy = episode_mean(np.asarray(run.bundle["sg_q_gap_policy_log"], float), run.time_in_sub, run.nfe)
    qgap_supervisor = episode_mean(np.asarray(run.bundle["sg_q_gap_supervisor_log"], float), run.time_in_sub, run.nfe)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.3), sharex=True, constrained_layout=True)
    axes[0].stackplot(ep, supervisor_frac, policy_frac, labels=["supervisor", "policy"], colors=["#A0CBE8", "#54A24B"], alpha=0.88)
    axes[0].axvline(10, color="0.4", linestyle="--", linewidth=0.9)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Selection fraction")
    axes[0].set_title("Supervisor-gated TD3 candidate selection")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(ep, advantage, color="#54A24B", linewidth=1.8, label="policy minus supervisor score")
    axes[1].plot(ep, qgap_policy, color="#4C78A8", linewidth=1.2, alpha=0.75, label="policy critic gap")
    axes[1].plot(ep, qgap_supervisor, color="#F58518", linewidth=1.2, alpha=0.75, label="supervisor critic gap")
    axes[1].axhline(0.0, color="0.25", linewidth=0.9)
    axes[1].axvline(10, color="0.4", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("Subepisode")
    axes[1].set_ylabel("Score or gap")
    axes[1].legend(frameon=False, ncol=2)
    fig.savefig(FIG_DIR / "supervisor_gate_diagnostics.png", dpi=180)
    plt.close(fig)


def make_serializable(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, np.integer)):
                converted[key] = value.item()
            else:
                converted[key] = value
        out.append(converted)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = [load_method(label, spec) for label, spec in RUN_SPECS.items()]
    perf_rows = [performance_metrics(run) for run in runs]
    safety_rows = [safety_metrics(run) for run in runs]

    _write_csv(TABLE_DIR / "performance_summary.csv", make_serializable(perf_rows))
    _write_csv(TABLE_DIR / "safety_summary.csv", make_serializable(safety_rows))
    with (TABLE_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "performance": make_serializable(perf_rows),
                "safety": make_serializable(safety_rows),
                "guard_reason_counts": {
                    run.label: guard_reason_counts(run) for run in runs if run.spec["kind"] == "residual"
                },
            },
            handle,
            indent=2,
        )

    perf_md_rows = []
    for row in perf_rows:
        perf_md_rows.append(
            {
                "Method": row["Method"],
                "Mean reward": _fmt(row["Mean reward"], 3),
                "Post-warm reward": _fmt(row["Post-warm reward"], 3),
                "Tail-20 reward": _fmt(row["Tail-20 reward"], 3),
                "Tail eta RMSE": _fmt(row["Tail eta RMSE"], 4),
                "Tail T RMSE": _fmt(row["Tail T RMSE"], 4),
                "Tail eta MAE": _fmt(row["Tail eta MAE"], 4),
                "Tail T MAE": _fmt(row["Tail T MAE"], 4),
                "Tail mean abs du scaled": _fmt(row["Tail mean abs du scaled"], 4),
            }
        )
    _write_markdown_table(
        TABLE_DIR / "performance_summary.md",
        perf_md_rows,
        [
            "Method",
            "Mean reward",
            "Post-warm reward",
            "Tail-20 reward",
            "Tail eta RMSE",
            "Tail T RMSE",
            "Tail eta MAE",
            "Tail T MAE",
            "Tail mean abs du scaled",
        ],
    )

    safety_md_rows = []
    for row in safety_rows:
        safety_md_rows.append(
            {
                "Method": row["Method"],
                "Ramp": row["Ramp"],
                "Post-warm cap clip": _fmt_pct(row["Post-warm cap clip"]),
                "Guard trigger active": _fmt_pct(row["Guard trigger active"]),
                "Guard accepted": row["Guard accepted"] if row["Guard accepted"] is not None else "NA",
                "Guard objective worse": row["Guard objective worse"] if row["Guard objective worse"] is not None else "NA",
                "Guard zero selected": row["Guard zero selected"] if row["Guard zero selected"] is not None else "NA",
                "Headroom projection": _fmt_pct(row["Headroom projection"]),
                "Shadow rho authority": _fmt_pct(row["Shadow rho authority"]),
                "Shadow rho deadband": _fmt_pct(row["Shadow rho deadband"]),
                "SG policy selected": _fmt_pct(row["SG policy selected"]),
                "SG supervisor selected": _fmt_pct(row["SG supervisor selected"]),
            }
        )
    _write_markdown_table(
        TABLE_DIR / "safety_summary.md",
        safety_md_rows,
        [
            "Method",
            "Ramp",
            "Post-warm cap clip",
            "Guard trigger active",
            "Guard accepted",
            "Guard objective worse",
            "Guard zero selected",
            "Headroom projection",
            "Shadow rho authority",
            "Shadow rho deadband",
            "SG policy selected",
            "SG supervisor selected",
        ],
    )

    plot_rewards(runs)
    plot_tail_tracking(runs)
    plot_tail_bars(perf_rows)
    plot_residual_safety(runs)
    for run in runs:
        if run.bundle.get("supervisor_gated_td3_enabled"):
            plot_sg_gate(run)

    print(f"Wrote analysis assets to {FIG_DIR.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
