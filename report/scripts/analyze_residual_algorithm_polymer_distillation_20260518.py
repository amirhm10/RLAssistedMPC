from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "residual_algorithm_polymer_distillation_20260518"

DIST_RUN = REPO_ROOT / "Distillation" / "Results" / "distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified" / "20260507_212833"
POLY_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_residual_disturb" / "20260501_000607"

LIVE_DISTILLATION_LOG = """
Sub_Episode: 1 | avg. reward: 8.175754939675388 | avg residual: [2.78904426e-09 1.64293182e-09]
Sub_Episode: 2 | avg. reward: 14.588388925422613 | avg residual: [ 6.47901403e-10 -4.24232676e-09]
Sub_Episode: 3 | avg. reward: 14.61353885430912 | avg residual: [1.10085673e-09 8.82807310e-10]
Sub_Episode: 4 | avg. reward: 14.641646436212728 | avg residual: [-1.45475333e-09  2.52839014e-09]
Sub_Episode: 5 | avg. reward: 14.670011463625908 | avg residual: [-1.97982621e-09 -2.27493513e-10]
Sub_Episode: 6 | avg. reward: 14.699633829182869 | avg residual: [ 7.98921575e-10 -3.18287855e-09]
Sub_Episode: 7 | avg. reward: 14.732365458787708 | avg residual: [-1.10921308e-09  1.22893406e-09]
Sub_Episode: 8 | avg. reward: 14.773747246446526 | avg residual: [-3.31217986e-09 -1.79791616e-09]
Sub_Episode: 9 | avg. reward: 14.81523864726871 | avg residual: [-2.62408583e-09  3.94037178e-10]
Sub_Episode: 10 | avg. reward: 14.859037702498833 | avg residual: [-2.75993265e-09 -1.28840950e-09]
Sub_Episode: 11 | avg. reward: 14.90098039943715 | avg residual: [-1.87658824e-09 -2.01544108e-09]
Sub_Episode: 12 | avg. reward: 14.94778471467249 | avg residual: [-2.26086804e-09 -1.28548104e-09]
Sub_Episode: 13 | avg. reward: 15.007134840731737 | avg residual: [ 2.84457107e-10 -2.99837388e-10]
Sub_Episode: 14 | avg. reward: 15.056602336902323 | avg residual: [-2.70841471e-10 -2.71730250e-09]
Sub_Episode: 15 | avg. reward: 15.432758530276454 | avg residual: [-2.00983574e-09 -5.98978521e-10]
Sub_Episode: 16 | avg. reward: 2.6813131678698774 | avg residual: [ 0.00010049 -0.00100234]
Sub_Episode: 17 | avg. reward: 2.2857440296616436 | avg residual: [ 0.00156213 -0.00223556]
Sub_Episode: 18 | avg. reward: 2.306153042285366 | avg residual: [ 0.00155983 -0.00223012]
Sub_Episode: 19 | avg. reward: 2.328659003351555 | avg residual: [ 0.0015575  -0.00222454]
Sub_Episode: 20 | avg. reward: 2.3528708055908525 | avg residual: [ 0.00155513 -0.00221887]
Sub_Episode: 21 | avg. reward: 3.130475815218773 | avg residual: [ 0.00151832 -0.00174903]
Sub_Episode: 22 | avg. reward: 4.590732257032838 | avg residual: [ 1.42467473e-03 -5.80246624e-05]
Sub_Episode: 23 | avg. reward: 4.948252680749001 | avg residual: [0.00145184 0.0002934 ]
Sub_Episode: 24 | avg. reward: 6.2804485832348975 | avg residual: [0.00147398 0.00119691]
Sub_Episode: 25 | avg. reward: 6.263260879700279 | avg residual: [0.00146615 0.00132872]
Sub_Episode: 26 | avg. reward: 6.297185040307515 | avg residual: [0.00146681 0.00131486]
Sub_Episode: 27 | avg. reward: 6.3152060122183755 | avg residual: [0.00146805 0.00123951]
Sub_Episode: 28 | avg. reward: 6.696430206661721 | avg residual: [0.00143972 0.00091325]
Sub_Episode: 29 | avg. reward: 5.7710193509594605 | avg residual: [ 0.00139914 -0.00040554]
Sub_Episode: 30 | avg. reward: 3.52300321129007 | avg residual: [ 0.00148483 -0.001023  ]
Sub_Episode: 31 | avg. reward: 3.385912856716576 | avg residual: [ 0.00150802 -0.00117344]
Sub_Episode: 32 | avg. reward: 3.5339957754628784 | avg residual: [ 0.00155843 -0.00034498]
Sub_Episode: 33 | avg. reward: 3.0924338648702876 | avg residual: [1.60586427e-03 9.82034129e-05]
Sub_Episode: 34 | avg. reward: 3.082474735079218 | avg residual: [0.00163619 0.00026293]
Sub_Episode: 35 | avg. reward: 3.0128329995774146 | avg residual: [0.00165095 0.00028004]
Sub_Episode: 36 | avg. reward: 2.749097569107341 | avg residual: [0.00167398 0.00021907]
Sub_Episode: 37 | avg. reward: 3.1241998911662416 | avg residual: [0.00165723 0.00048138]
Sub_Episode: 38 | avg. reward: 5.069530045374313 | avg residual: [0.00155505 0.0006607 ]
Sub_Episode: 39 | avg. reward: 4.38154477060934 | avg residual: [0.00163266 0.00062907]
Sub_Episode: 40 | avg. reward: 4.321827999198339 | avg residual: [0.00164215 0.00054156]
Sub_Episode: 41 | avg. reward: 4.252870644236306 | avg residual: [0.00165042 0.0004545 ]
Sub_Episode: 42 | avg. reward: 4.278195845934716 | avg residual: [0.0016514  0.00032652]
Sub_Episode: 43 | avg. reward: 2.8902358530522854 | avg residual: [1.73048251e-03 5.91743804e-05]
Sub_Episode: 44 | avg. reward: 4.573943786345139 | avg residual: [ 0.00156492 -0.00010032]
Sub_Episode: 45 | avg. reward: 4.092096394491689 | avg residual: [ 0.00148513 -0.00039095]
Sub_Episode: 46 | avg. reward: 4.185102472176542 | avg residual: [ 0.00116209 -0.00071685]
Sub_Episode: 47 | avg. reward: 6.430108587303119 | avg residual: [ 0.00076355 -0.00121913]
Sub_Episode: 48 | avg. reward: 7.020220760478534 | avg residual: [ 0.00039937 -0.00129405]
Sub_Episode: 49 | avg. reward: 3.9904181114067567 | avg residual: [ 0.0004292 -0.0011987]
Sub_Episode: 50 | avg. reward: 4.205611107620112 | avg residual: [ 0.00055689 -0.00097285]
"""


@dataclass
class ResidualRun:
    name: str
    run_dir: Path
    bundle: dict

    @property
    def avg_rewards(self) -> np.ndarray:
        return np.asarray(self.bundle["avg_rewards"], dtype=float)

    @property
    def steps_per_episode(self) -> int:
        return int(self.bundle["time_in_sub_episodes"])

    @property
    def warm_episode_count(self) -> int:
        return int(self.bundle["warm_start_step"] // max(1, self.steps_per_episode))


def load_run(name: str, run_dir: Path) -> ResidualRun:
    with (run_dir / "input_data.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    return ResidualRun(name=name, run_dir=run_dir, bundle=bundle)


def parse_live_log() -> dict[str, np.ndarray]:
    rows = []
    pattern = re.compile(
        r"Sub_Episode:\s+(\d+)\s+\|\s+avg\. reward:\s+([-+0-9.eE]+)\s+\|\s+avg residual:\s+\[([^\]]+)\]"
    )
    for match in pattern.finditer(LIVE_DISTILLATION_LOG):
        values = [float(item) for item in match.group(3).split()]
        rows.append((int(match.group(1)), float(match.group(2)), values[0], values[1]))
    arr = np.asarray(rows, dtype=float)
    return {
        "episode": arr[:, 0].astype(int),
        "reward": arr[:, 1],
        "residual": arr[:, 2:4],
        "residual_norm": np.linalg.norm(arr[:, 2:4], axis=1),
    }


def window_slice(run: ResidualRun, label: str) -> slice:
    n_ep = run.avg_rewards.size
    steps = run.steps_per_episode
    warm = int(run.bundle["warm_start_step"])
    if label == "warm":
        return slice(0, warm)
    if label == "release":
        return slice(warm, min(warm + 5 * steps, n_ep * steps))
    if label == "tail20":
        return slice(max(0, n_ep - 20) * steps, n_ep * steps)
    raise ValueError(label)


def summarize_run(run: ResidualRun) -> dict:
    out = {
        "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
        "reward_mean": float(np.nanmean(run.avg_rewards)),
        "reward_tail20": float(np.nanmean(run.avg_rewards[-20:])),
        "reward_final": float(run.avg_rewards[-1]),
        "state_mode": run.bundle.get("state_mode"),
        "authority_use_rho": bool(run.bundle.get("authority_use_rho")),
        "append_rho_to_state": bool(run.bundle.get("append_rho_to_state")),
        "authority_beta_res": np.asarray(run.bundle.get("authority_beta_res"), float).tolist(),
        "authority_du0_res": np.asarray(run.bundle.get("authority_du0_res"), float).tolist(),
        "authority_rho_floor": float(run.bundle.get("authority_rho_floor")),
        "rho_mapping_mode": run.bundle.get("rho_mapping_mode"),
        "behavioral_cloning_enabled": bool(run.bundle.get("behavioral_cloning_enabled", False)),
        "warm_episode_count": run.warm_episode_count,
    }
    for label in ("warm", "release", "tail20"):
        sl = window_slice(run, label)
        raw = np.asarray(run.bundle["delta_u_res_raw_log"], dtype=float)[sl]
        exe = np.asarray(run.bundle["delta_u_res_exec_log"], dtype=float)[sl]
        out[label] = {
            "rho_mean": float(np.nanmean(run.bundle["rho_log"][sl])),
            "rho_eff_mean": float(np.nanmean(run.bundle["rho_eff_log"][sl])),
            "projection_active_fraction": float(np.mean(run.bundle["projection_active_log"][sl])),
            "projection_authority_fraction": float(np.mean(run.bundle["projection_due_to_authority_log"][sl])),
            "projection_deadband_fraction": float(np.mean(run.bundle["projection_due_to_deadband_log"][sl])),
            "raw_residual_norm_mean": float(np.nanmean(np.linalg.norm(raw, axis=1))),
            "exec_residual_norm_mean": float(np.nanmean(np.linalg.norm(exe, axis=1))),
            "policy_executed_gap_mean": float(np.nanmean(run.bundle["policy_executed_gap_norm_log"][sl])),
        }
    return out


def plot_live_log(live: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True, constrained_layout=True)
    ep = live["episode"]
    axes[0].plot(ep, live["reward"], color="#4c78a8", linewidth=2.0)
    axes[0].axvline(15.5, color="0.35", linestyle="--", linewidth=1.1, label="first live residual authority")
    axes[0].set_ylabel("Average reward")
    axes[0].legend(frameon=False)
    axes[0].set_title("Distillation residual run: release shock in live log")
    axes[1].plot(ep, live["residual_norm"], color="#b84a62", linewidth=2.0, label="||avg residual||")
    axes[1].plot(ep, live["residual"][:, 0], color="#54a24b", linewidth=1.3, alpha=0.85, label="residual 1")
    axes[1].plot(ep, live["residual"][:, 1], color="#f58518", linewidth=1.3, alpha=0.85, label="residual 2")
    axes[1].axvline(15.5, color="0.35", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Subepisode")
    axes[1].set_ylabel("Average scaled residual")
    axes[1].legend(frameon=False, ncol=3)
    fig.savefig(OUT_DIR / "fig_distillation_live_residual_release_shock.png", dpi=180)
    plt.close(fig)


def plot_saved_diagnostics(runs: list[ResidualRun]) -> None:
    labels = [run.name for run in runs]
    windows = ["warm", "release", "tail20"]
    summaries = [summarize_run(run) for run in runs]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    x = np.arange(len(windows))
    width = 0.34
    for idx, summary in enumerate(summaries):
        offset = (-0.5 + idx) * width
        axes[0].bar(x + offset, [summary[w]["rho_eff_mean"] for w in windows], width, label=labels[idx])
        axes[1].bar(x + offset, [summary[w]["projection_authority_fraction"] for w in windows], width)
        axes[2].bar(x + offset, [summary[w]["exec_residual_norm_mean"] for w in windows], width)
    for ax, title, ylabel in [
        (axes[0], "rho authority is active", "Mean rho_eff"),
        (axes[1], "authority projection fraction", "Fraction"),
        (axes[2], "executed residual magnitude", "Mean norm"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(["warm", "release", "tail-20"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_saved_residual_rho_projection_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    colors = ["#4c78a8", "#54a24b"]
    for run, color in zip(runs, colors):
        ax.plot(np.arange(1, run.avg_rewards.size + 1), run.avg_rewards, label=run.name, color=color, linewidth=1.8)
    ax.set_xlabel("Subepisode")
    ax.set_ylabel("Average reward")
    ax.set_title("Saved residual runs use the same shared residual runner")
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_saved_residual_reward_traces.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dist = load_run("Distillation TD3 residual", DIST_RUN)
    poly = load_run("Polymer TD3 residual", POLY_RUN)
    live = parse_live_log()
    plot_live_log(live)
    plot_saved_diagnostics([dist, poly])

    live_summary = {
        "episode_count": int(live["episode"].size),
        "warm_freeze_reward_mean_ep1_15": float(np.mean(live["reward"][:15])),
        "release_reward_mean_ep16_20": float(np.mean(live["reward"][15:20])),
        "reward_drop_ep15_to_ep16": float(live["reward"][14] - live["reward"][15]),
        "residual_norm_mean_ep1_15": float(np.mean(live["residual_norm"][:15])),
        "residual_norm_mean_ep16_20": float(np.mean(live["residual_norm"][15:20])),
        "residual_norm_max_ep16_50": float(np.max(live["residual_norm"][15:])),
    }
    summary = {
        "live_distillation_so_far": live_summary,
        "saved_distillation": summarize_run(dist),
        "saved_polymer": summarize_run(poly),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
