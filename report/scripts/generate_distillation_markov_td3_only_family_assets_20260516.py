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
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_markov_td3_only_family_20260516"
BASELINE_BUNDLE = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
WINDOW = 10
TAIL_EPISODES = 20

RUN_SPECS = [
    {
        "key": "guarded",
        "label": "Guarded",
        "notebook": "distillation_RL_assisted_MPC_markov_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_unified",
        "compare_root": REPO_ROOT / "Distillation" / "Results" / "distillation_compare_markov_td3_disturb_fluctuation",
        "color": "#0B6E4F",
    },
    {
        "key": "relaxed",
        "label": "Relaxed",
        "notebook": "distillation_RL_assisted_MPC_markov_relaxed_acceptance_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_relaxed_acceptance_unified",
        "compare_root": REPO_ROOT / "Distillation" / "Results" / "distillation_compare_markov_td3_disturb_fluctuation_relaxed_acceptance",
        "color": "#1f77b4",
    },
    {
        "key": "ls_only",
        "label": "LS only",
        "notebook": "distillation_RL_assisted_MPC_markov_ls_only_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_ls_only_unified",
        "compare_root": REPO_ROOT / "Distillation" / "Results" / "distillation_compare_markov_td3_disturb_fluctuation_ls_only",
        "color": "#F59E0B",
    },
    {
        "key": "td3_without_ls",
        "label": "TD3 no LS",
        "notebook": "distillation_RL_assisted_MPC_markov_td3_without_ls_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_td3_without_ls_unified",
        "compare_root": REPO_ROOT / "Distillation" / "Results" / "distillation_compare_markov_td3_disturb_fluctuation_td3_without_ls",
        "color": "#C2410C",
    },
    {
        "key": "td3_only",
        "label": "TD3 only",
        "notebook": "distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_unified.ipynb",
        "root": REPO_ROOT / "Distillation" / "Results" / "distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified",
        "compare_root": REPO_ROOT / "Distillation" / "Results" / "distillation_compare_markov_td3_disturb_fluctuation_td3_only_no_safeguard",
        "color": "#7A1FA2",
    },
]


@dataclass
class RunData:
    key: str
    label: str
    color: str
    notebook: str
    run_dir: Path
    compare_dir: Path
    bundle: dict
    compare_bundle: dict
    stage_df: pd.DataFrame

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle["avg_rewards"], dtype=float)

    @property
    def rewards_step(self) -> np.ndarray:
        return np.asarray(self.bundle["rewards_step"], dtype=float)

    @property
    def n_episodes(self) -> int:
        return int(self.avg_rewards.size)

    @property
    def steps_per_episode(self) -> int:
        return int(self.bundle["time_in_sub_episodes"])

    @property
    def warm_start_step(self) -> int:
        return int(self.bundle.get("warm_start_step", 0))

    @property
    def warm_start_episodes(self) -> int:
        return int(self.warm_start_step // self.steps_per_episode)

    @property
    def n_inputs(self) -> int:
        return len(np.asarray(self.bundle["steady_states"]["ss_inputs"], dtype=float))

    @property
    def output_labels(self) -> list[str]:
        return list(self.bundle["system_metadata"]["output_labels"])

    @property
    def input_labels(self) -> list[str]:
        return list(self.bundle["system_metadata"]["input_labels"])

    @property
    def data_min(self) -> np.ndarray:
        return np.asarray(self.bundle["data_min"], dtype=float)

    @property
    def data_max(self) -> np.ndarray:
        return np.asarray(self.bundle["data_max"], dtype=float)

    @property
    def y_ss(self) -> np.ndarray:
        return np.asarray(self.bundle["steady_states"]["y_ss"], dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.bundle["y"][1:], dtype=float).reshape(self.n_episodes, self.steps_per_episode, -1)

    @property
    def u(self) -> np.ndarray:
        return np.asarray(self.bundle["u"], dtype=float).reshape(self.n_episodes, self.steps_per_episode, -1)

    @property
    def y_sp_phys(self) -> np.ndarray:
        y_range = self.data_max[self.n_inputs :] - self.data_min[self.n_inputs :]
        arr = self.y_ss.reshape(1, -1) + np.asarray(self.bundle["y_sp"], dtype=float) * y_range.reshape(1, -1)
        return arr.reshape(self.n_episodes, self.steps_per_episode, -1)

    @property
    def action_source(self) -> np.ndarray:
        return np.asarray(self.bundle["rl_action_source_log"], dtype=int).reshape(self.n_episodes, self.steps_per_episode)

    @property
    def accepted(self) -> np.ndarray:
        return np.asarray(self.bundle["accepted_log"], dtype=int).reshape(self.n_episodes, self.steps_per_episode)

    @property
    def executed_z(self) -> np.ndarray:
        return np.asarray(self.bundle["z_executed_log"], dtype=float).reshape(self.n_episodes, self.steps_per_episode, -1)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def latest_run_dir(root: Path) -> Path:
    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return candidates[-1]


def moving_average(values: np.ndarray, width: int = WINDOW) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < width:
        return arr.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(arr, kernel, mode="same")


def final_episode_block_slices(steps: int) -> tuple[slice, slice]:
    return slice(0, steps // 2), slice(steps // 2, steps)


def compute_band_phys(run: RunData) -> np.ndarray:
    params = run.bundle["reward_params"]
    k_rel = np.asarray(params["k_rel"], dtype=float).reshape(1, 1, -1)
    band_floor = np.asarray(params["band_floor_phys"], dtype=float).reshape(1, 1, -1)
    return np.maximum(k_rel * np.abs(run.y_sp_phys), band_floor)


def compute_step_reward_breakdown(y_phys: np.ndarray, u_phys: np.ndarray, y_sp_phys: np.ndarray, run: RunData) -> dict[str, np.ndarray]:
    params = run.bundle["reward_params"]
    k_rel = np.asarray(params["k_rel"], dtype=float)
    band_floor_phys = np.asarray(params["band_floor_phys"], dtype=float)
    q_diag = np.asarray(params["Q_diag"], dtype=float)
    r_diag = np.asarray(params["R_diag"], dtype=float)
    tau_frac = float(params["tau_frac"])
    gamma_out = float(params["gamma_out"])
    gamma_in = float(params["gamma_in"])
    beta = float(params["beta"])
    gate = str(params["gate"])
    lam_in = float(params["lam_in"])
    bonus_kind = str(params["bonus_kind"])
    bonus_k = float(params["bonus_k"])
    bonus_p = float(params["bonus_p"])
    bonus_c = float(params["bonus_c"])
    reward_scale = float(params["reward_scale"])

    dy = run.data_max[run.n_inputs :] - run.data_min[run.n_inputs :]
    du_span = run.data_max[: run.n_inputs] - run.data_min[: run.n_inputs]

    e_scaled = (y_phys - y_sp_phys) / dy.reshape(1, -1)
    du_scaled = np.diff(np.vstack([u_phys[0:1, :], u_phys]), axis=0) / du_span.reshape(1, -1)
    band_phys = np.maximum(k_rel.reshape(1, -1) * np.abs(y_sp_phys), band_floor_phys.reshape(1, -1))
    band_scaled = band_phys / dy.reshape(1, -1)

    tau_scaled = tau_frac * band_scaled
    abs_e = np.abs(e_scaled)
    sig_arg = (band_scaled - abs_e) / np.maximum(tau_scaled, 1.0e-12)
    s_i = 1.0 / (1.0 + np.exp(-np.clip(sig_arg, -60.0, 60.0)))

    if gate == "prod":
        w_in = np.prod(s_i, axis=1)
    elif gate == "mean":
        w_in = np.mean(s_i, axis=1)
    elif gate == "geom":
        w_in = np.prod(s_i, axis=1) ** (1.0 / s_i.shape[1])
    else:
        raise ValueError(f"Unsupported gate: {gate}")

    if bonus_kind == "linear":
        phi = 1.0 - np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
    elif bonus_kind == "quadratic":
        phi = (1.0 - np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)) ** 2
    elif bonus_kind == "exp":
        z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
        phi = (np.exp(-bonus_k * z) - np.exp(-bonus_k)) / (1.0 - np.exp(-bonus_k))
    elif bonus_kind == "power":
        z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
        phi = 1.0 - np.power(z, bonus_p)
    elif bonus_kind == "log":
        z = np.clip(abs_e / np.maximum(band_scaled, 1.0e-12), 0.0, 1.0)
        phi = np.log1p(bonus_c * (1.0 - z)) / np.log1p(bonus_c)
    else:
        raise ValueError(f"Unsupported bonus kind: {bonus_kind}")

    err_quad_vec = q_diag.reshape(1, -1) * (e_scaled**2)
    move_vec = r_diag.reshape(1, -1) * (du_scaled**2)
    slope_at_edge = 2.0 * q_diag.reshape(1, -1) * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    lin_out_vec = (1.0 - w_in).reshape(-1, 1) * gamma_out * slope_at_edge * overflow
    lin_in_vec = w_in.reshape(-1, 1) * gamma_in * slope_at_edge * inside_mag
    qb2 = q_diag.reshape(1, -1) * (band_scaled**2)
    bonus_vec = w_in.reshape(-1, 1) * beta * qb2 * phi

    err_quad = np.sum(err_quad_vec, axis=1)
    move = np.sum(move_vec, axis=1)
    lin_out = np.sum(lin_out_vec, axis=1)
    lin_in = np.sum(lin_in_vec, axis=1)
    reward = (-(err_quad + move + lin_out + lin_in) + np.sum(bonus_vec, axis=1)) * reward_scale

    return {
        "reward": reward,
        "w_in": w_in,
        "err_quad_vec": err_quad_vec,
        "move_vec": move_vec,
        "lin_out_vec": lin_out_vec,
        "lin_in_vec": lin_in_vec,
        "bonus_vec": bonus_vec,
        "band_phys": band_phys,
        "abs_error_phys": np.abs(y_phys - y_sp_phys),
    }


def tail_indexer(run: RunData) -> slice:
    return slice(max(0, run.n_episodes - TAIL_EPISODES), run.n_episodes)


def build_run_data() -> tuple[dict[str, RunData], dict]:
    runs: dict[str, RunData] = {}
    for spec in RUN_SPECS:
        run_dir = latest_run_dir(spec["root"])
        compare_dir = latest_run_dir(spec["compare_root"])
        runs[spec["key"]] = RunData(
            key=spec["key"],
            label=spec["label"],
            color=spec["color"],
            notebook=spec["notebook"],
            run_dir=run_dir,
            compare_dir=compare_dir,
            bundle=load_pickle(run_dir / "input_data.pkl"),
            compare_bundle=load_pickle(compare_dir / "input_data.pkl"),
            stage_df=pd.read_csv(run_dir / "markov_stage_diagnostics.csv"),
        )
    baseline = load_pickle(BASELINE_BUNDLE)
    return runs, baseline


def baseline_arrays(run: RunData, baseline_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(baseline_bundle["y"][1:], dtype=float).reshape(run.n_episodes, run.steps_per_episode, -1)
    u = np.asarray(baseline_bundle["u"], dtype=float).reshape(run.n_episodes, run.steps_per_episode, -1)
    return y, u


def canonical_mpc_reward_series(runs: dict[str, RunData], baseline_bundle: dict) -> np.ndarray:
    preferred = ["td3_only", "guarded", "relaxed", "ls_only", "td3_without_ls"]
    for key in preferred:
        if key in runs:
            values = np.asarray(runs[key].compare_bundle.get("avg_rewards_mpc", []), dtype=float)
            if values.size == runs[key].n_episodes:
                return values
    return np.asarray(baseline_bundle["avg_rewards"], dtype=float)


def summarize_run(run: RunData, baseline_bundle: dict, mpc_reward_series: np.ndarray) -> dict:
    y_mpc, u_mpc = baseline_arrays(run, baseline_bundle)
    band_phys = compute_band_phys(run)
    err = np.abs(run.y - run.y_sp_phys)
    err_mpc = np.abs(y_mpc - run.y_sp_phys)
    reward_mpc = np.asarray(mpc_reward_series, dtype=float)
    source = run.action_source
    td3_mask = source == 2
    ls_mask = (source == 3) | (source == 5)
    nominal_mask = (source == 0) | (source == 4)

    tail = tail_indexer(run)
    sp1, sp2 = final_episode_block_slices(run.steps_per_episode)

    final_stage = run.stage_df.iloc[(run.n_episodes - 1) * run.steps_per_episode : run.n_episodes * run.steps_per_episode].reset_index(drop=True)
    final_sp1 = final_stage.iloc[: run.steps_per_episode // 2]
    final_sp2 = final_stage.iloc[run.steps_per_episode // 2 :]

    summary = {
        "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
        "notebook": run.notebook,
        "reward_mean": float(np.mean(run.avg_rewards)),
        "reward_tail20": float(np.mean(run.avg_rewards[tail])),
        "reward_tail10": float(np.mean(run.avg_rewards[-10:])),
        "reward_final": float(run.avg_rewards[-1]),
        "reward_best": float(np.max(run.avg_rewards)),
        "reward_best_episode": int(np.argmax(run.avg_rewards) + 1),
        "reward_delta_vs_mpc_tail20": float(np.mean(run.avg_rewards[tail] - reward_mpc[tail])),
        "td3_fraction_all": float(np.mean(td3_mask)),
        "ls_fraction_all": float(np.mean(ls_mask)),
        "nominal_fraction_all": float(np.mean(nominal_mask)),
        "td3_fraction_tail20": float(np.mean(td3_mask[tail])),
        "ls_fraction_tail20": float(np.mean(ls_mask[tail])),
        "nominal_fraction_tail20": float(np.mean(nominal_mask[tail])),
        "final_sp1_temp_mae": float(np.mean(err[-1, sp1, 1])),
        "final_sp1_temp_mae_mpc": float(np.mean(err_mpc[-1, sp1, 1])),
        "final_sp2_temp_mae": float(np.mean(err[-1, sp2, 1])),
        "final_sp2_temp_mae_mpc": float(np.mean(err_mpc[-1, sp2, 1])),
        "final_sp1_comp_mae": float(np.mean(err[-1, sp1, 0])),
        "final_sp1_comp_mae_mpc": float(np.mean(err_mpc[-1, sp1, 0])),
        "final_sp2_comp_mae": float(np.mean(err[-1, sp2, 0])),
        "final_sp2_comp_mae_mpc": float(np.mean(err_mpc[-1, sp2, 0])),
        "tail20_sp1_temp_mae": float(np.mean(err[tail, sp1, 1])),
        "tail20_sp1_temp_mae_mpc": float(np.mean(err_mpc[tail, sp1, 1])),
        "tail20_sp2_temp_mae": float(np.mean(err[tail, sp2, 1])),
        "tail20_sp2_temp_mae_mpc": float(np.mean(err_mpc[tail, sp2, 1])),
        "tail20_sp1_comp_mae": float(np.mean(err[tail, sp1, 0])),
        "tail20_sp1_comp_mae_mpc": float(np.mean(err_mpc[tail, sp1, 0])),
        "tail20_sp2_comp_mae": float(np.mean(err[tail, sp2, 0])),
        "tail20_sp2_comp_mae_mpc": float(np.mean(err_mpc[tail, sp2, 0])),
        "tail20_executed_prediction_score": float(np.nanmean(run.stage_df.loc[run.stage_df["step"] > run.warm_start_step, "executed_prediction_score"].to_numpy(dtype=float)[-TAIL_EPISODES * run.steps_per_episode :])),
        "tail20_executed_gain_drift": float(np.nanmean(run.stage_df.loc[run.stage_df["step"] > run.warm_start_step, "executed_gain_drift"].to_numpy(dtype=float)[-TAIL_EPISODES * run.steps_per_episode :])),
        "tail20_executed_z_norm": float(np.nanmean(run.stage_df.loc[run.stage_df["step"] > run.warm_start_step, "executed_z_norm"].to_numpy(dtype=float)[-TAIL_EPISODES * run.steps_per_episode :])),
        "final_sp1_executed_prediction_score": float(np.nanmean(final_sp1["executed_prediction_score"])),
        "final_sp2_executed_prediction_score": float(np.nanmean(final_sp2["executed_prediction_score"])),
        "final_sp1_executed_gain_drift": float(np.nanmean(final_sp1["executed_gain_drift"])),
        "final_sp2_executed_gain_drift": float(np.nanmean(final_sp2["executed_gain_drift"])),
        "final_sp1_executed_z_norm": float(np.nanmean(final_sp1["executed_z_norm"])),
        "final_sp2_executed_z_norm": float(np.nanmean(final_sp2["executed_z_norm"])),
        "final_sp1_temp_band_mean": float(np.mean(band_phys[-1, sp1, 1])),
        "final_sp2_temp_band_mean": float(np.mean(band_phys[-1, sp2, 1])),
        "final_sp1_comp_band_mean": float(np.mean(band_phys[-1, sp1, 0])),
        "final_sp2_comp_band_mean": float(np.mean(band_phys[-1, sp2, 0])),
        "final_sp1_temp_inside_frac": float(np.mean(err[-1, sp1, 1] <= band_phys[-1, sp1, 1])),
        "final_sp2_temp_inside_frac": float(np.mean(err[-1, sp2, 1] <= band_phys[-1, sp2, 1])),
        "final_sp1_comp_inside_frac": float(np.mean(err[-1, sp1, 0] <= band_phys[-1, sp1, 0])),
        "final_sp2_comp_inside_frac": float(np.mean(err[-1, sp2, 0] <= band_phys[-1, sp2, 0])),
    }
    return summary


def baseline_summary(reference_run: RunData, baseline_bundle: dict, mpc_reward_series: np.ndarray) -> dict:
    y_mpc, _u_mpc = baseline_arrays(reference_run, baseline_bundle)
    err_mpc = np.abs(y_mpc - reference_run.y_sp_phys)
    reward_mpc = np.asarray(mpc_reward_series, dtype=float)
    tail = tail_indexer(reference_run)
    sp1, sp2 = final_episode_block_slices(reference_run.steps_per_episode)
    return {
        "reward_mean": float(np.mean(reward_mpc)),
        "reward_tail20": float(np.mean(reward_mpc[tail])),
        "reward_final": float(reward_mpc[-1]),
        "final_sp1_temp_mae": float(np.mean(err_mpc[-1, sp1, 1])),
        "final_sp2_temp_mae": float(np.mean(err_mpc[-1, sp2, 1])),
        "final_sp1_comp_mae": float(np.mean(err_mpc[-1, sp1, 0])),
        "final_sp2_comp_mae": float(np.mean(err_mpc[-1, sp2, 0])),
        "tail20_sp1_temp_mae": float(np.mean(err_mpc[tail, sp1, 1])),
        "tail20_sp2_temp_mae": float(np.mean(err_mpc[tail, sp2, 1])),
        "tail20_sp1_comp_mae": float(np.mean(err_mpc[tail, sp1, 0])),
        "tail20_sp2_comp_mae": float(np.mean(err_mpc[tail, sp2, 0])),
    }


def plot_reward_family_trends(runs: dict[str, RunData], mpc_reward_series: np.ndarray) -> Path:
    reference = runs["guarded"]
    episodes = np.arange(1, reference.n_episodes + 1)

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    ax.plot(episodes, mpc_reward_series, color="#4C4C4C", linewidth=2.0, linestyle=":", label="Disturbance MPC")
    for spec in RUN_SPECS:
        run = runs[spec["key"]]
        ax.plot(
            episodes,
            moving_average(run.avg_rewards, WINDOW),
            color=run.color,
            linewidth=2.6 if run.key == "td3_only" else 2.0,
            label=run.label,
        )
    ax.axvline(reference.warm_start_episodes + 0.5, color="0.3", linewidth=1.0, linestyle="--")
    ax.set_title("Distillation Markov family: TD3-only starts worse but finishes with the strongest late reward")
    ax.set_xlabel("Sub-episode")
    ax.set_ylabel("Average reward")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", ncol=3, fontsize=9)

    out = OUT_DIR / "fig_reward_family_trends.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_action_source_and_reward_summary(runs: dict[str, RunData], baseline_bundle: dict, summaries: dict[str, dict], baseline_stats: dict) -> Path:
    labels = [spec["label"] for spec in RUN_SPECS]
    x = np.arange(len(labels))
    tail_reward = [summaries[spec["key"]]["reward_tail20"] for spec in RUN_SPECS]
    final_reward = [summaries[spec["key"]]["reward_final"] for spec in RUN_SPECS]
    td3_frac = [summaries[spec["key"]]["td3_fraction_tail20"] for spec in RUN_SPECS]
    ls_frac = [summaries[spec["key"]]["ls_fraction_tail20"] for spec in RUN_SPECS]
    nominal_frac = [summaries[spec["key"]]["nominal_fraction_tail20"] for spec in RUN_SPECS]

    fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.8))

    width = 0.34
    axs[0].bar(x - width / 2.0, tail_reward, width, color="#7A1FA2", label="Tail-20 reward")
    axs[0].bar(x + width / 2.0, final_reward, width, color="#C084FC", label="Final reward")
    axs[0].axhline(baseline_stats["reward_tail20"], color="#4C4C4C", linewidth=1.2, linestyle=":", label="MPC tail-20")
    axs[0].axhline(baseline_stats["reward_final"], color="#9CA3AF", linewidth=1.2, linestyle="--", label="MPC final")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=15)
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Late reward is best in the TD3-only run")
    axs[0].grid(alpha=0.25, axis="y")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].legend(loc="best", fontsize=9)

    axs[1].bar(x, td3_frac, color="#7A1FA2", label="TD3")
    axs[1].bar(x, ls_frac, bottom=td3_frac, color="#1f77b4", label="LS")
    axs[1].bar(x, nominal_frac, bottom=np.asarray(td3_frac) + np.asarray(ls_frac), color="#9CA3AF", label="Nominal")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=15)
    axs[1].set_ylabel("Tail-20 fraction of steps")
    axs[1].set_ylim(0.0, 1.02)
    axs[1].set_title("Only the TD3-only notebook actually uses TD3 online")
    axs[1].grid(alpha=0.25, axis="y")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].legend(loc="best", fontsize=9)

    out = OUT_DIR / "fig_action_source_and_reward_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_blockwise_tracking_tradeoff(summaries: dict[str, dict], baseline_stats: dict) -> Path:
    labels = ["MPC"] + [spec["label"] for spec in RUN_SPECS]
    colors = ["#4C4C4C"] + [spec["color"] for spec in RUN_SPECS]

    def series(metric_key: str) -> list[float]:
        return [baseline_stats[metric_key]] + [summaries[spec["key"]][metric_key] for spec in RUN_SPECS]

    metrics = [
        ("final_sp1_temp_mae", "Final SP1 temperature MAE"),
        ("final_sp2_temp_mae", "Final SP2 temperature MAE"),
        ("final_sp1_comp_mae", "Final SP1 composition MAE"),
        ("final_sp2_comp_mae", "Final SP2 composition MAE"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14.0, 9.0))
    axs = axs.ravel()
    x = np.arange(len(labels))
    for ax, (metric_key, title) in zip(axs, metrics):
        vals = series(metric_key)
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("K")
    axs[1].set_ylabel("K")
    axs[2].set_ylabel("Fraction")
    axs[3].set_ylabel("Fraction")
    fig.suptitle("TD3-only trades first-setpoint temperature for stronger composition and second-setpoint tracking", y=1.01)

    out = OUT_DIR / "fig_blockwise_tracking_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_td3_only_final_episode(runs: dict[str, RunData], baseline_bundle: dict) -> Path:
    td3_only = runs["td3_only"]
    guarded = runs["guarded"]
    y_mpc, u_mpc = baseline_arrays(td3_only, baseline_bundle)
    steps = td3_only.steps_per_episode
    t = np.arange(steps)
    y_sp = td3_only.y_sp_phys[-1]

    fig, axs = plt.subplots(2, 2, figsize=(13.4, 8.2), sharex="col")
    for idx, ax in enumerate(axs[0]):
        ax.step(t, y_sp[:, idx], where="post", color="#111111", linewidth=1.3, linestyle="--", label="Setpoint")
        ax.plot(t, y_mpc[-1, :, idx], color="#4C4C4C", linewidth=1.8, linestyle=":", label="MPC")
        ax.plot(t, guarded.y[-1, :, idx], color=guarded.color, linewidth=1.7, linestyle="--", label="Guarded")
        ax.plot(t, td3_only.y[-1, :, idx], color=td3_only.color, linewidth=2.0, label="TD3 only")
        ax.axvline(steps // 2, color="0.55", linewidth=1.0, linestyle=":")
        ax.set_title(td3_only.output_labels[idx])
        ax.set_ylabel("Output")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)
    for idx, ax in enumerate(axs[1]):
        ax.step(t, u_mpc[-1, :, idx], where="post", color="#4C4C4C", linewidth=1.8, linestyle=":", label="MPC")
        ax.step(t, guarded.u[-1, :, idx], where="post", color=guarded.color, linewidth=1.7, linestyle="--", label="Guarded")
        ax.step(t, td3_only.u[-1, :, idx], where="post", color=td3_only.color, linewidth=2.0, label="TD3 only")
        ax.axvline(steps // 2, color="0.55", linewidth=1.0, linestyle=":")
        ax.set_title(td3_only.input_labels[idx])
        ax.set_ylabel("Input")
        ax.set_xlabel("Step within final sub-episode")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("Final episode: TD3-only misses first-setpoint temperature, then dominates the second block", y=1.01)
    out = OUT_DIR / "fig_td3_only_final_episode_vs_guarded_and_mpc.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_td3_only_reward_tradeoff(runs: dict[str, RunData], baseline_bundle: dict) -> Path:
    td3_only = runs["td3_only"]
    guarded = runs["guarded"]
    y_mpc, u_mpc = baseline_arrays(td3_only, baseline_bundle)
    rewards_guarded = guarded.rewards_step.reshape(guarded.n_episodes, guarded.steps_per_episode)
    rewards_td3 = td3_only.rewards_step.reshape(td3_only.n_episodes, td3_only.steps_per_episode)
    sp1, sp2 = final_episode_block_slices(td3_only.steps_per_episode)

    td3_breakdown = compute_step_reward_breakdown(td3_only.y[-1], td3_only.u[-1], td3_only.y_sp_phys[-1], td3_only)
    mpc_breakdown = compute_step_reward_breakdown(y_mpc[-1], u_mpc[-1], td3_only.y_sp_phys[-1], td3_only)
    band_phys = compute_band_phys(td3_only)
    err_td3 = np.abs(td3_only.y[-1] - td3_only.y_sp_phys[-1])
    err_mpc = np.abs(y_mpc[-1] - td3_only.y_sp_phys[-1])

    final_stage = td3_only.stage_df.iloc[(td3_only.n_episodes - 1) * td3_only.steps_per_episode : td3_only.n_episodes * td3_only.steps_per_episode].reset_index(drop=True)
    final_sp1 = final_stage.iloc[: td3_only.steps_per_episode // 2]
    final_sp2 = final_stage.iloc[td3_only.steps_per_episode // 2 :]

    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.4))
    axs = axs.ravel()

    reward_block_labels = ["SP1", "SP2"]
    reward_x = np.arange(2)
    width = 0.24
    td3_block_reward = [float(np.mean(rewards_td3[-1, sp1])), float(np.mean(rewards_td3[-1, sp2]))]
    guarded_block_reward = [float(np.mean(rewards_guarded[-1, sp1])), float(np.mean(rewards_guarded[-1, sp2]))]
    mpc_block_reward = [float(np.mean(mpc_breakdown["reward"][sp1])), float(np.mean(mpc_breakdown["reward"][sp2]))]
    axs[0].bar(reward_x - width, td3_block_reward, width, color=td3_only.color, label="TD3 only")
    axs[0].bar(reward_x, guarded_block_reward, width, color=guarded.color, label="Guarded")
    axs[0].bar(reward_x + width, mpc_block_reward, width, color="#4C4C4C", label="MPC")
    axs[0].set_xticks(reward_x)
    axs[0].set_xticklabels(reward_block_labels)
    axs[0].set_ylabel("Mean step reward")
    axs[0].set_title("Final episode reward is lost in SP1 and recovered strongly in SP2")
    axs[0].grid(alpha=0.25, axis="y")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].legend(loc="best", fontsize=9)

    band_x = np.arange(2)
    axs[1].bar(band_x - width / 2.0, [float(np.mean(err_td3[sp1, 1])), float(np.mean(err_td3[sp2, 1]))], width, color=td3_only.color, label="TD3 temp error")
    axs[1].bar(band_x + width / 2.0, [float(np.mean(err_mpc[sp1, 1])), float(np.mean(err_mpc[sp2, 1]))], width, color="#4C4C4C", label="MPC temp error")
    axs[1].plot(band_x, [float(np.mean(band_phys[-1, sp1, 1])), float(np.mean(band_phys[-1, sp2, 1]))], color="#DC2626", linewidth=2.0, marker="o", label="Reward band")
    axs[1].set_xticks(band_x)
    axs[1].set_xticklabels(reward_block_labels)
    axs[1].set_ylabel("Temperature absolute error / band (K)")
    axs[1].set_title("The first temperature block is outside the reward band")
    axs[1].grid(alpha=0.25, axis="y")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].legend(loc="best", fontsize=9)

    frac_labels = ["SP1 comp", "SP1 temp", "SP2 comp", "SP2 temp"]
    frac_x = np.arange(len(frac_labels))
    td3_frac = [
        float(np.mean(err_td3[sp1, 0] <= band_phys[-1, sp1, 0])),
        float(np.mean(err_td3[sp1, 1] <= band_phys[-1, sp1, 1])),
        float(np.mean(err_td3[sp2, 0] <= band_phys[-1, sp2, 0])),
        float(np.mean(err_td3[sp2, 1] <= band_phys[-1, sp2, 1])),
    ]
    mpc_frac = [
        float(np.mean(err_mpc[sp1, 0] <= band_phys[-1, sp1, 0])),
        float(np.mean(err_mpc[sp1, 1] <= band_phys[-1, sp1, 1])),
        float(np.mean(err_mpc[sp2, 0] <= band_phys[-1, sp2, 0])),
        float(np.mean(err_mpc[sp2, 1] <= band_phys[-1, sp2, 1])),
    ]
    axs[2].bar(frac_x - width / 2.0, td3_frac, width, color=td3_only.color, label="TD3 only")
    axs[2].bar(frac_x + width / 2.0, mpc_frac, width, color="#4C4C4C", label="MPC")
    axs[2].set_xticks(frac_x)
    axs[2].set_xticklabels(frac_labels, rotation=12)
    axs[2].set_ylabel("Fraction inside reward band")
    axs[2].set_ylim(0.0, 1.02)
    axs[2].set_title("TD3-only protects composition first, then temperature later")
    axs[2].grid(alpha=0.25, axis="y")
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[2].legend(loc="best", fontsize=9)

    diag_labels = ["Pred. score", "Gain drift", "||z||"]
    diag_x = np.arange(len(diag_labels))
    sp1_diag = [
        float(np.nanmean(final_sp1["executed_prediction_score"])),
        float(np.nanmean(final_sp1["executed_gain_drift"])),
        float(np.nanmean(final_sp1["executed_z_norm"])),
    ]
    sp2_diag = [
        float(np.nanmean(final_sp2["executed_prediction_score"])),
        float(np.nanmean(final_sp2["executed_gain_drift"])),
        float(np.nanmean(final_sp2["executed_z_norm"])),
    ]
    axs[3].bar(diag_x - width / 2.0, sp1_diag, width, color="#A21CAF", label="SP1")
    axs[3].bar(diag_x + width / 2.0, sp2_diag, width, color="#D8B4FE", label="SP2")
    axs[3].axhline(0.0, color="0.2", linewidth=1.0, linestyle=":")
    axs[3].set_xticks(diag_x)
    axs[3].set_xticklabels(diag_labels)
    axs[3].set_title("TD3-only uses larger, more negative-score actions in SP1")
    axs[3].grid(alpha=0.25, axis="y")
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)
    axs[3].legend(loc="best", fontsize=9)

    fig.suptitle("Why the TD3-only run misses first-setpoint temperature", y=1.01)
    out = OUT_DIR / "fig_td3_only_reward_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    del td3_breakdown
    del mpc_breakdown
    return out


def write_summary_files(runs: dict[str, RunData], summaries: dict[str, dict], baseline_stats: dict) -> list[Path]:
    rows = []
    for spec in RUN_SPECS:
        row = {"variant": spec["label"]}
        row.update(summaries[spec["key"]])
        rows.append(row)

    csv_path = OUT_DIR / "summary_metrics.csv"
    fieldnames = ["variant"] + sorted({key for row in rows for key in row.keys() if key != "variant"})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    td3_only = runs["td3_only"]
    y_mpc, u_mpc = baseline_arrays(td3_only, load_pickle(BASELINE_BUNDLE))
    sp1, sp2 = final_episode_block_slices(td3_only.steps_per_episode)
    td3_breakdown = compute_step_reward_breakdown(td3_only.y[-1], td3_only.u[-1], td3_only.y_sp_phys[-1], td3_only)
    mpc_breakdown = compute_step_reward_breakdown(y_mpc[-1], u_mpc[-1], td3_only.y_sp_phys[-1], td3_only)

    def block_breakdown(parts: dict[str, np.ndarray], block: slice) -> dict[str, float]:
        return {
            "reward_mean": float(np.mean(parts["reward"][block])),
            "w_in_mean": float(np.mean(parts["w_in"][block])),
            "err_quad_comp": float(np.mean(parts["err_quad_vec"][block, 0])),
            "err_quad_temp": float(np.mean(parts["err_quad_vec"][block, 1])),
            "move_reflux": float(np.mean(parts["move_vec"][block, 0])),
            "move_reboiler": float(np.mean(parts["move_vec"][block, 1])),
            "lin_out_comp": float(np.mean(parts["lin_out_vec"][block, 0])),
            "lin_out_temp": float(np.mean(parts["lin_out_vec"][block, 1])),
            "lin_in_comp": float(np.mean(parts["lin_in_vec"][block, 0])),
            "lin_in_temp": float(np.mean(parts["lin_in_vec"][block, 1])),
            "bonus_comp": float(np.mean(parts["bonus_vec"][block, 0])),
            "bonus_temp": float(np.mean(parts["bonus_vec"][block, 1])),
        }

    json_path = OUT_DIR / "summary.json"
    payload = {
        "baseline_bundle": str(BASELINE_BUNDLE.relative_to(REPO_ROOT)),
        "runs": {spec["key"]: str(runs[spec["key"]].run_dir.relative_to(REPO_ROOT)) for spec in RUN_SPECS},
        "notebooks": {spec["key"]: spec["notebook"] for spec in RUN_SPECS},
        "baseline_metrics": baseline_stats,
        "run_metrics": summaries,
        "td3_only_final_block_reward_breakdown": {
            "sp1": block_breakdown(td3_breakdown, sp1),
            "sp2": block_breakdown(td3_breakdown, sp2),
            "mpc_sp1": block_breakdown(mpc_breakdown, sp1),
            "mpc_sp2": block_breakdown(mpc_breakdown, sp2),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return [csv_path, json_path]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, baseline_bundle = build_run_data()
    mpc_reward_series = canonical_mpc_reward_series(runs, baseline_bundle)
    summaries = {key: summarize_run(run, baseline_bundle, mpc_reward_series) for key, run in runs.items()}
    baseline_stats = baseline_summary(runs["guarded"], baseline_bundle, mpc_reward_series)

    generated = [
        plot_reward_family_trends(runs, mpc_reward_series),
        plot_action_source_and_reward_summary(runs, baseline_bundle, summaries, baseline_stats),
        plot_blockwise_tracking_tradeoff(summaries, baseline_stats),
        plot_td3_only_final_episode(runs, baseline_bundle),
        plot_td3_only_reward_tradeoff(runs, baseline_bundle),
    ]
    generated.extend(write_summary_files(runs, summaries, baseline_stats))

    print("Generated:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
