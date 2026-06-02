from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "markov_distillation_polymer_sg_td3_20260602"


RUNS = {
    "distillation_td3_full_no_safeguard": {
        "case": "distillation",
        "label": "Distillation Markov TD3-full no safeguard",
        "short_label": "Distill TD3 full",
        "rl_path": REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_current_reward_unified"
        / "20260601_214256"
        / "input_data.pkl",
        "compare_path": REPO_ROOT
        / "Distillation"
        / "Results"
        / "distillation_compare_markov_td3_disturb_fluctuation_td3_only_no_safeguard_current_reward"
        / "20260601_214316"
        / "input_data.pkl",
        "mpc_path": REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle",
        "output_names": ["x24 ethane", "T85"],
        "input_names": ["reflux", "reboiler"],
    },
    "polymer_sg_td3_markov": {
        "case": "polymer",
        "label": "Polymer Markov SG-TD3 critic-warm",
        "short_label": "Poly SG-TD3",
        "rl_path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb"
        / "20260601_215126"
        / "input_data.pkl",
        "compare_path": REPO_ROOT
        / "Polymer"
        / "Results"
        / "disturb_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow"
        / "20260601_215148"
        / "input_data.pkl",
        "mpc_path": REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle",
        "output_names": ["eta", "T"],
        "input_names": ["Qc", "Qm"],
    },
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def apply_min_max(data, min_val, max_val):
    return (np.asarray(data, float) - np.asarray(min_val, float)) / (np.asarray(max_val, float) - np.asarray(min_val, float))


def reverse_min_max(scaled_data, min_val, max_val):
    return np.asarray(scaled_data, float) * (np.asarray(max_val, float) - np.asarray(min_val, float)) + np.asarray(min_val, float)


def pick_array(bundle: dict, *keys: str) -> np.ndarray:
    for key in keys:
        if key in bundle and bundle[key] is not None:
            return np.asarray(bundle[key], float)
    raise KeyError(f"None of these keys are present: {keys}")


def y_to_line(y: np.ndarray, n_steps: int) -> np.ndarray:
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] >= n_steps + 1:
        return y[: n_steps + 1, :]
    if y.shape[0] == n_steps:
        return np.vstack([y, y[-1:, :]])
    if y.shape[0] == 0:
        raise ValueError("Output trajectory is empty.")
    pad = np.repeat(y[-1:, :], n_steps + 1 - y.shape[0], axis=0)
    return np.vstack([y, pad])


def u_to_step(u: np.ndarray, n_steps: int) -> np.ndarray:
    u = np.asarray(u, float)
    if u.ndim == 1:
        u = u[:, None]
    if u.shape[0] >= n_steps:
        return u[:n_steps, :]
    if u.shape[0] == 0:
        raise ValueError("Input trajectory is empty.")
    pad = np.repeat(u[-1:, :], n_steps - u.shape[0], axis=0)
    return np.vstack([u, pad])


def ysp_scaled_dev_to_phys(y_sp_scaled_dev, steady_states, data_min, data_max, n_inputs):
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    return reverse_min_max(np.asarray(y_sp_scaled_dev, float) + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def normalize_mpc_bundle(bundle: dict, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    y = pick_array(bundle, "y_mpc", "y", "y_rl")
    u = pick_array(bundle, "u_mpc", "u", "u_rl")
    return y_to_line(y, n_steps), u_to_step(u, n_steps)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_quantile(values, q) -> float:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def finite_fraction(mask) -> float:
    arr = np.asarray(mask)
    if arr.size == 0:
        return float("nan")
    finite = np.isfinite(arr.astype(float)) if arr.dtype.kind in "fc" else np.ones(arr.shape, dtype=bool)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(arr[finite]))


def episode_segments(y_sp: np.ndarray, episode_len: int) -> list[tuple[int, int, str]]:
    first = np.asarray(y_sp[:episode_len], float)
    if first.shape[0] == 0:
        return []
    changes = np.where(np.any(np.abs(np.diff(first, axis=0)) > 1.0e-12, axis=1))[0] + 1
    cuts = [0] + [int(v) for v in changes] + [int(episode_len)]
    return [(cuts[i], cuts[i + 1], f"SP{i + 1}") for i in range(len(cuts) - 1)]


def interval_error_metrics(
    *,
    y_line: np.ndarray,
    u_step: np.ndarray,
    y_sp_phys: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    steady_states: dict,
    start: int,
    end: int,
    output_names: list[str],
) -> tuple[list[dict], float]:
    end = min(int(end), u_step.shape[0], y_sp_phys.shape[0], y_line.shape[0] - 1)
    start = max(0, min(int(start), end))
    err = y_line[start + 1 : end + 1, :] - y_sp_phys[start:end, :]
    rows = []
    for idx, name in enumerate(output_names):
        e = err[:, idx]
        rows.append(
            {
                "output": name,
                "mae": float(np.mean(np.abs(e))) if e.size else float("nan"),
                "rmse": float(np.sqrt(np.mean(e**2))) if e.size else float("nan"),
                "max_abs": float(np.max(np.abs(e))) if e.size else float("nan"),
                "mean_signed": float(np.mean(e)) if e.size else float("nan"),
                "final_abs": float(abs(e[-1])) if e.size else float("nan"),
            }
        )

    n_inputs = u_step.shape[1]
    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    u_scaled = apply_min_max(u_step, data_min[:n_inputs], data_max[:n_inputs])
    du = np.zeros_like(u_scaled)
    du[0, :] = u_scaled[0, :] - ss_scaled_inputs
    if u_scaled.shape[0] > 1:
        du[1:, :] = u_scaled[1:, :] - u_scaled[:-1, :]
    du_seg = du[start:end, :]
    mean_abs_du = float(np.mean(np.abs(du_seg))) if du_seg.size else float("nan")
    return rows, mean_abs_du


def compute_block_rows(
    *,
    run_key: str,
    method: str,
    y_line: np.ndarray,
    y_sp_phys: np.ndarray,
    episode_ids: list[int],
    segments: list[tuple[int, int, str]],
    episode_len: int,
    output_names: list[str],
    window: str,
) -> list[dict]:
    rows = []
    for seg_start, seg_end, seg_name in segments:
        idxs = []
        for ep in episode_ids:
            start = int(ep * episode_len + seg_start)
            end = int(ep * episode_len + seg_end)
            idxs.append(np.arange(start, end))
        if not idxs:
            continue
        idx = np.concatenate(idxs)
        idx = idx[(idx >= 0) & (idx < y_sp_phys.shape[0]) & (idx + 1 < y_line.shape[0])]
        if idx.size == 0:
            continue
        err = y_line[idx + 1, :] - y_sp_phys[idx, :]
        for out_idx, output in enumerate(output_names):
            e = err[:, out_idx]
            rows.append(
                {
                    "run_key": run_key,
                    "window": window,
                    "method": method,
                    "block": seg_name,
                    "output": output,
                    "steps": int(e.size),
                    "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "max_abs": float(np.max(np.abs(e))),
                    "mean_signed": float(np.mean(e)),
                }
            )
    return rows


def source_fraction_rows(bundle: dict, run_key: str, window_name: str, start: int, end: int) -> list[dict]:
    rows = []
    source = np.asarray(bundle.get("rl_action_source_log", []), int).reshape(-1)
    names = {int(k): str(v) for k, v in dict(bundle.get("rl_action_source_names", {})).items()}
    if source.size:
        end = min(end, source.size)
        sub = source[start:end]
        for code in sorted(names):
            rows.append(
                {
                    "run_key": run_key,
                    "window": window_name,
                    "source": names[code],
                    "fraction": float(np.mean(sub == code)) if sub.size else float("nan"),
                }
            )

    sg_selected = np.asarray(bundle.get("sg_selected_source_log", []), int).reshape(-1)
    if sg_selected.size:
        end = min(end, sg_selected.size)
        sub = sg_selected[start:end]
        sg_names = {0: "sg_warm_start_or_none", 1: "sg_supervisor", 2: "sg_policy", 3: "sg_held", 4: "sg_fallback"}
        for code in sorted(sg_names):
            rows.append(
                {
                    "run_key": run_key,
                    "window": window_name,
                    "source": sg_names[code],
                    "fraction": float(np.mean(sub == code)) if sub.size else float("nan"),
                }
            )
    return rows


def diagnostic_rows(bundle: dict, run_key: str, window_name: str, start: int, end: int) -> list[dict]:
    rows = []

    def add(name: str, value) -> None:
        rows.append({"run_key": run_key, "window": window_name, "metric": name, "value": float(value)})

    for key in [
        "requested_prediction_score_log",
        "executed_prediction_score_log",
        "requested_gain_drift_log",
        "executed_gain_drift_log",
        "requested_cost_margin_log",
        "executed_cost_margin_log",
        "rl_requested_z_log",
        "requested_z_log",
        "z_executed_log",
        "u0_executed_minus_nominal_norm_log",
        "u_sequence_executed_minus_nominal_norm_log",
        "sg_advantage_log",
        "sg_q_gap_policy_log",
        "sg_q_gap_supervisor_log",
    ]:
        if key not in bundle or bundle[key] is None:
            continue
        arr = np.asarray(bundle[key], float)
        arr = arr[start : min(end, arr.shape[0])]
        if arr.ndim > 1:
            norm = np.linalg.norm(arr, axis=1)
            add(f"{key}_norm_mean", finite_mean(norm))
            add(f"{key}_norm_q95", finite_quantile(norm, 0.95))
            add(f"{key}_coord_abs_q95", finite_quantile(np.abs(arr).reshape(-1), 0.95))
        else:
            add(f"{key}_mean", finite_mean(arr))
            add(f"{key}_q05", finite_quantile(arr, 0.05))
            add(f"{key}_q95", finite_quantile(arr, 0.95))

    binary_specs = [
        ("requested_cost_guard_pass_log", "requested_cost_guard_pass_fraction"),
        ("executed_cost_guard_pass_log", "executed_cost_guard_pass_fraction"),
        ("requested_legacy_hard_gate_pass_log", "requested_legacy_hard_gate_pass_fraction"),
        ("rl_release_gate_pass_log", "bc_release_gate_pass_fraction"),
        ("rl_release_gate_blocked_log", "bc_release_gate_blocked_fraction"),
        ("shadow_td3_priority_allowed_log", "shadow_td3_priority_allowed_fraction"),
        ("shadow_nominal_fallback_eligible_log", "shadow_nominal_fallback_eligible_fraction"),
        ("shadow_z_safety_requested_projection_active_log", "shadow_z_projection_active_fraction"),
        ("shadow_z_safety_requested_coord_clip_active_log", "shadow_z_coord_clip_fraction"),
        ("shadow_z_safety_requested_vector_projection_active_log", "shadow_z_vector_projection_fraction"),
        ("z_safety_requested_projection_active_log", "live_z_projection_active_fraction"),
        ("z_safety_requested_coord_clip_active_log", "live_z_coord_clip_fraction"),
        ("z_safety_requested_vector_projection_active_log", "live_z_vector_projection_fraction"),
    ]
    for key, out_name in binary_specs:
        if key not in bundle or bundle[key] is None:
            continue
        arr = np.asarray(bundle[key]).reshape(-1)
        arr = arr[start : min(end, arr.shape[0])]
        valid = arr >= 0
        add(out_name, float(np.mean(arr[valid] == 1)) if np.any(valid) else float("nan"))

    return rows


def per_episode_diagnostics(bundle: dict, compare: dict, run_key: str, episode_len: int) -> list[dict]:
    rewards = np.asarray(compare["avg_rewards_rl"], float)
    rows = []
    for ep_idx, reward in enumerate(rewards):
        start = ep_idx * episode_len
        end = start + episode_len
        row = {
            "run_key": run_key,
            "episode": ep_idx + 1,
            "reward_rl": float(reward),
            "reward_mpc": float(compare["avg_rewards_mpc"][ep_idx]) if ep_idx < len(compare["avg_rewards_mpc"]) else float("nan"),
        }
        for key in [
            "requested_prediction_score_log",
            "executed_prediction_score_log",
            "requested_cost_guard_pass_log",
            "requested_legacy_hard_gate_pass_log",
            "rl_action_source_log",
            "rl_requested_z_log",
            "requested_z_log",
            "z_executed_log",
            "sg_advantage_log",
        ]:
            if key not in bundle or bundle[key] is None:
                continue
            arr = np.asarray(bundle[key])
            sub = arr[start : min(end, arr.shape[0])]
            if sub.size == 0:
                continue
            if key.endswith("_z_log") or key == "z_executed_log":
                sub = np.asarray(sub, float)
                row[f"{key}_norm_mean"] = finite_mean(np.linalg.norm(sub, axis=1))
            elif key == "rl_action_source_log":
                sub = np.asarray(sub, int)
                row["source_td3_policy_fraction"] = float(np.mean(sub == 2))
                row["source_sg_supervisor_fraction"] = float(np.mean((sub == 6) | (sub == 7) | (sub == 8)))
                row["source_nominal_fraction"] = float(np.mean(sub == 4))
            elif "pass" in key:
                sub = np.asarray(sub)
                valid = sub >= 0
                row[f"{key}_fraction"] = float(np.mean(sub[valid] == 1)) if np.any(valid) else float("nan")
            else:
                row[f"{key}_mean"] = finite_mean(sub.astype(float))
        rows.append(row)
    return rows


def plot_reward_curves(run_data: dict) -> Path:
    fig, axs = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=False)
    for ax, (run_key, data) in zip(axs, run_data.items()):
        compare = data["compare"]
        rl = np.asarray(compare["avg_rewards_rl"], float)
        mpc = np.asarray(compare["avg_rewards_mpc"], float)
        warm_ep = int(data["rl"]["warm_start_step"] // data["episode_len"])
        x = np.arange(1, len(rl) + 1)
        ax.plot(x, rl, label="RL/Markov", linewidth=2.0)
        ax.plot(x[: len(mpc)], mpc, label="OF-MPC", linewidth=2.0, linestyle="--")
        ax.axvspan(1, warm_ep, color="#bdbdbd", alpha=0.20, label="warm start" if run_key == next(iter(run_data)) else None)
        ax.axvspan(max(1, len(rl) - 19), len(rl), color="#80cdc1", alpha=0.20, label="tail 20" if run_key == next(iter(run_data)) else None)
        ax.set_title(data["spec"]["label"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average reward")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    path = OUT_DIR / "reward_curves.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_tracking(run_data: dict, run_key: str) -> Path:
    data = run_data[run_key]
    spec = data["spec"]
    ep_len = data["episode_len"]
    n_steps = data["n_steps"]
    start = n_steps - ep_len
    end = n_steps
    t_line = np.arange(0, ep_len + 1) * float(data["rl"]["delta_t"])
    t_step = t_line[:-1]
    y_sp = data["y_sp_phys"][start:end, :]
    y_rl = data["rl_y_line"][start : end + 1, :]
    y_mpc = data["mpc_y_line"][start : end + 1, :]

    fig, axs = plt.subplots(y_rl.shape[1], 1, figsize=(9.0, 5.8), sharex=True)
    if y_rl.shape[1] == 1:
        axs = [axs]
    for idx, ax in enumerate(axs):
        ax.plot(t_line, y_rl[:, idx], label="RL/Markov", linewidth=2.0)
        ax.plot(t_line, y_mpc[:, idx], label="OF-MPC", linestyle="--", linewidth=2.0)
        ax.step(t_step, y_sp[:, idx], where="post", label="setpoint", linestyle=":", linewidth=2.0)
        ax.set_ylabel(spec["output_names"][idx])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axs[-1].set_xlabel("time in final episode")
    fig.suptitle(f"Final episode tracking: {spec['label']}")
    fig.tight_layout()
    path = OUT_DIR / f"final_episode_tracking_{spec['case']}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tail_block_mae(block_rows: list[dict]) -> Path:
    tail_rows = [r for r in block_rows if r["window"] == "tail20"]
    fig, axs = plt.subplots(2, 1, figsize=(10.0, 7.2), sharex=False)
    for ax, case in zip(axs, ["distillation_td3_full_no_safeguard", "polymer_sg_td3_markov"]):
        rows = [r for r in tail_rows if r["run_key"] == case]
        labels = []
        rl_vals = []
        mpc_vals = []
        for block in sorted({r["block"] for r in rows}):
            for output in RUNS[case]["output_names"]:
                labels.append(f"{block} {output}")
                m_rl = next(r["mae"] for r in rows if r["block"] == block and r["output"] == output and r["method"] == "RL")
                m_mpc = next(r["mae"] for r in rows if r["block"] == block and r["output"] == output and r["method"] == "MPC")
                rl_vals.append(m_rl)
                mpc_vals.append(m_mpc)
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, rl_vals, width, label="RL/Markov")
        ax.bar(x + width / 2, mpc_vals, width, label="OF-MPC")
        ax.set_title(RUNS[case]["label"])
        ax.set_ylabel("Tail-20 MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    path = OUT_DIR / "tail20_blockwise_mae.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_action_source_summary(action_rows: list[dict]) -> Path:
    desired_sources = {
        "distillation_td3_full_no_safeguard": ["td3_accepted", "nominal_fallback", "ls_fallback", "sg_supervisor_mpc"],
        "polymer_sg_td3_markov": ["td3_accepted", "sg_supervisor_ls", "sg_supervisor_mpc", "sg_solver_fallback_supervisor"],
    }
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=False)
    for ax, run_key in zip(axs, desired_sources):
        rows = [r for r in action_rows if r["run_key"] == run_key and r["window"] in {"post_warm", "tail20"}]
        sources = desired_sources[run_key]
        x = np.arange(len(sources))
        width = 0.36
        for offset, window in [(-width / 2, "post_warm"), (width / 2, "tail20")]:
            vals = []
            for src in sources:
                match = [r for r in rows if r["window"] == window and r["source"] == src]
                vals.append(match[0]["fraction"] if match else 0.0)
            ax.bar(x + offset, vals, width, label=window)
        ax.set_title(RUNS[run_key]["label"])
        ax.set_ylabel("fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    path = OUT_DIR / "action_source_fractions.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distillation_episode_scatter(episode_rows: list[dict]) -> Path:
    rows = [r for r in episode_rows if r["run_key"] == "distillation_td3_full_no_safeguard"]
    rewards = np.asarray([r["reward_rl"] for r in rows], float)
    z_norm = np.asarray([r.get("z_executed_log_norm_mean", np.nan) for r in rows], float)
    score = np.asarray([r.get("requested_prediction_score_log_mean", np.nan) for r in rows], float)
    gate = np.asarray([r.get("requested_legacy_hard_gate_pass_log_fraction", np.nan) for r in rows], float)
    x = np.arange(1, len(rows) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.8))
    sc = axs[0].scatter(z_norm, rewards, c=x, cmap="viridis", s=28)
    axs[0].set_xlabel("mean executed z 2-norm")
    axs[0].set_ylabel("episode reward")
    axs[0].grid(True, alpha=0.25)
    axs[1].scatter(score, rewards, c=x, cmap="viridis", s=28)
    axs[1].set_xlabel("mean requested prediction score")
    axs[1].grid(True, alpha=0.25)
    axs[2].scatter(gate, rewards, c=x, cmap="viridis", s=28)
    axs[2].set_xlabel("legacy hard-gate pass fraction")
    axs[2].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axs, label="episode")
    fig.suptitle("Distillation no-safeguard episodes: reward versus candidate diagnostics")
    fig.tight_layout()
    path = OUT_DIR / "distillation_episode_reward_vs_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_data = {}
    reward_rows = []
    tracking_rows = []
    block_rows = []
    action_rows = []
    diag_rows = []
    episode_rows = []

    for run_key, spec in RUNS.items():
        rl = load_pickle(spec["rl_path"])
        compare = load_pickle(spec["compare_path"])
        mpc = load_pickle(spec["mpc_path"])

        n_steps = int(rl["nFE"])
        episode_len = int(rl["time_in_sub_episodes"])
        n_episodes = int(n_steps // episode_len)
        warm_ep = int(rl["warm_start_step"] // episode_len)
        n_inputs = int(rl.get("n_inputs", pick_array(rl, "u_rl", "u_step_full").shape[1]))
        data_min = np.asarray(rl["data_min"], float)
        data_max = np.asarray(rl["data_max"], float)
        y_sp = np.asarray(rl["y_sp"], float)
        y_sp_phys = ysp_scaled_dev_to_phys(y_sp, rl["steady_states"], data_min, data_max, n_inputs)
        rl_y_line = y_to_line(pick_array(rl, "y_rl", "y_line_full", "y"), n_steps)
        rl_u_step = u_to_step(pick_array(rl, "u_rl", "u_step_full", "u"), n_steps)
        mpc_y_line, mpc_u_step = normalize_mpc_bundle(mpc, n_steps)
        segments = episode_segments(y_sp, episode_len)

        run_data[run_key] = {
            "spec": spec,
            "rl": rl,
            "compare": compare,
            "mpc": mpc,
            "n_steps": n_steps,
            "episode_len": episode_len,
            "n_episodes": n_episodes,
            "warm_ep": warm_ep,
            "y_sp_phys": y_sp_phys,
            "rl_y_line": rl_y_line,
            "rl_u_step": rl_u_step,
            "mpc_y_line": mpc_y_line,
            "mpc_u_step": mpc_u_step,
            "segments": segments,
        }

        rewards_rl = np.asarray(compare["avg_rewards_rl"], float)
        rewards_mpc = np.asarray(compare["avg_rewards_mpc"], float)
        tail_count = min(20, len(rewards_rl))
        post_start_ep = min(warm_ep, len(rewards_rl))
        first_post_end = min(len(rewards_rl), warm_ep + 20)
        for method, rewards in [("RL", rewards_rl), ("MPC", rewards_mpc)]:
            post = rewards[post_start_ep:]
            first_post = rewards[post_start_ep:first_post_end]
            reward_rows.append(
                {
                    "run_key": run_key,
                    "case": spec["case"],
                    "method": method,
                    "episodes": int(len(rewards)),
                    "warm_start_episodes": int(warm_ep),
                    "mean_reward": float(np.mean(rewards)),
                    "warm_mean_reward": float(np.mean(rewards[:warm_ep])) if warm_ep else float("nan"),
                    "post_warm_mean_reward": float(np.mean(post)) if post.size else float("nan"),
                    "post_warm_worst_reward": float(np.min(post)) if post.size else float("nan"),
                    "post_warm_worst_episode": int(np.argmin(post) + warm_ep + 1) if post.size else -1,
                    "first20_post_warm_worst_reward": float(np.min(first_post)) if first_post.size else float("nan"),
                    "tail20_reward": float(np.mean(rewards[-tail_count:])) if tail_count else float("nan"),
                    "final_reward": float(rewards[-1]) if rewards.size else float("nan"),
                    "best_reward": float(np.max(rewards)) if rewards.size else float("nan"),
                }
            )

        for window, start, end in [
            ("tail20", max(0, (n_episodes - 20) * episode_len), n_episodes * episode_len),
            ("final_episode", max(0, (n_episodes - 1) * episode_len), n_episodes * episode_len),
            ("post_warm", warm_ep * episode_len, n_episodes * episode_len),
        ]:
            for method, y_line, u_step in [("RL", rl_y_line, rl_u_step), ("MPC", mpc_y_line, mpc_u_step)]:
                rows, mean_abs_du = interval_error_metrics(
                    y_line=y_line,
                    u_step=u_step,
                    y_sp_phys=y_sp_phys,
                    data_min=data_min,
                    data_max=data_max,
                    steady_states=rl["steady_states"],
                    start=start,
                    end=end,
                    output_names=spec["output_names"],
                )
                for row in rows:
                    row.update(
                        {
                            "run_key": run_key,
                            "case": spec["case"],
                            "window": window,
                            "method": method,
                            "mean_abs_scaled_du": mean_abs_du,
                        }
                    )
                    tracking_rows.append(row)

        tail_eps = list(range(max(0, n_episodes - 20), n_episodes))
        final_eps = [n_episodes - 1]
        for method, y_line in [("RL", rl_y_line), ("MPC", mpc_y_line)]:
            block_rows.extend(
                compute_block_rows(
                    run_key=run_key,
                    method=method,
                    y_line=y_line,
                    y_sp_phys=y_sp_phys,
                    episode_ids=tail_eps,
                    segments=segments,
                    episode_len=episode_len,
                    output_names=spec["output_names"],
                    window="tail20",
                )
            )
            block_rows.extend(
                compute_block_rows(
                    run_key=run_key,
                    method=method,
                    y_line=y_line,
                    y_sp_phys=y_sp_phys,
                    episode_ids=final_eps,
                    segments=segments,
                    episode_len=episode_len,
                    output_names=spec["output_names"],
                    window="final_episode",
                )
            )

        for window_name, start, end in [
            ("overall", 0, n_steps),
            ("post_warm", warm_ep * episode_len, n_steps),
            ("tail20", max(0, (n_episodes - 20) * episode_len), n_steps),
            ("first20_post_warm", warm_ep * episode_len, min(n_steps, (warm_ep + 20) * episode_len)),
        ]:
            action_rows.extend(source_fraction_rows(rl, run_key, window_name, start, end))
            diag_rows.extend(diagnostic_rows(rl, run_key, window_name, start, end))

        episode_rows.extend(per_episode_diagnostics(rl, compare, run_key, episode_len))

    figure_paths = {
        "reward_curves": rel(plot_reward_curves(run_data)),
        "distillation_final_tracking": rel(plot_final_tracking(run_data, "distillation_td3_full_no_safeguard")),
        "polymer_final_tracking": rel(plot_final_tracking(run_data, "polymer_sg_td3_markov")),
        "tail20_blockwise_mae": rel(plot_tail_block_mae(block_rows)),
        "action_source_fractions": rel(plot_action_source_summary(action_rows)),
        "distillation_episode_reward_vs_diagnostics": rel(plot_distillation_episode_scatter(episode_rows)),
    }

    write_csv(OUT_DIR / "reward_summary.csv", reward_rows)
    write_csv(OUT_DIR / "tracking_summary.csv", tracking_rows)
    write_csv(OUT_DIR / "blockwise_tracking_summary.csv", block_rows)
    write_csv(OUT_DIR / "action_source_summary.csv", action_rows)
    write_csv(OUT_DIR / "diagnostic_summary.csv", diag_rows)
    write_csv(OUT_DIR / "episode_diagnostics.csv", episode_rows)

    summary = {
        "source_paths": {
            run_key: {
                "rl_path": rel(spec["rl_path"]),
                "compare_path": rel(spec["compare_path"]),
                "mpc_path": rel(spec["mpc_path"]),
            }
            for run_key, spec in RUNS.items()
        },
        "figures": figure_paths,
        "tables": {
            "reward_summary": rel(OUT_DIR / "reward_summary.csv"),
            "tracking_summary": rel(OUT_DIR / "tracking_summary.csv"),
            "blockwise_tracking_summary": rel(OUT_DIR / "blockwise_tracking_summary.csv"),
            "action_source_summary": rel(OUT_DIR / "action_source_summary.csv"),
            "diagnostic_summary": rel(OUT_DIR / "diagnostic_summary.csv"),
            "episode_diagnostics": rel(OUT_DIR / "episode_diagnostics.csv"),
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
