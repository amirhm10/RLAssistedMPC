from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation.config import (
    DISTILLATION_RL_SETPOINTS_PHYS,
    RL_REWARD_DEFAULTS as DISTILLATION_REWARD_DEFAULTS,
)
from systems.polymer.config import (
    POLYMER_RL_SETPOINTS_PHYS,
    RL_REWARD_DEFAULTS as POLYMER_REWARD_DEFAULTS,
)
from utils.rewards import make_reward_fn_relative_QR

OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_reward_geometry_smoothing_20260510"
SUMMARY_CSV = OUT_DIR / "reward_smoothing_candidates.csv"
CURVE_CSV = OUT_DIR / "bonus_shape_samples.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"
FIG_PATH = OUT_DIR / "fig_reward_smoothing_options.png"

N_EPISODES = 200

DISTILLATION_SCALAR_MAY8 = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_matrix_td3_disturb_fluctuation_mismatch_unified"
    / "20260508_015834"
    / "input_data.pkl"
)
DISTILLATION_STRUCTURED_MAY8 = (
    REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_structured_matrix_td3_disturb_fluctuation_mismatch_unified"
    / "20260508_005027"
    / "input_data.pkl"
)
DISTILLATION_BASELINE = REPO_ROOT / "Distillation" / "Data" / "mpc_results_disturb_fluctuation.pickle"
POLYMER_BASELINE = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def reward_curve(z: np.ndarray, *, kind: str, k: float = 12.0, p: float = 0.6, c: float = 20.0) -> np.ndarray:
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    if kind == "linear":
        return 1.0 - z
    if kind == "quadratic":
        return (1.0 - z) ** 2
    if kind == "exp":
        return (np.exp(-k * z) - np.exp(-k)) / (1.0 - np.exp(-k))
    if kind == "power":
        return 1.0 - np.power(z, p)
    if kind == "log":
        return np.log1p(c * (1.0 - z)) / np.log1p(c)
    raise ValueError(f"Unknown bonus kind: {kind}")


def reconstruct_y_sp_phys(bundle: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(np.asarray(bundle["delta_u_storage"], float).shape[1])
    dy = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_ss = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = (y_ss - data_min[n_inputs:]) / dy
    y_sp_scaled = np.asarray(bundle["y_sp"], float)
    return (y_sp_scaled + y_ss_scaled.reshape(1, -1)) * dy.reshape(1, -1) + data_min[n_inputs:].reshape(1, -1)


def rescore_episode_means(bundle: dict, reward_overrides: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = int(np.asarray(bundle["delta_u_storage"], float).shape[1])
    reward_cfg = {k: v for k, v in DISTILLATION_REWARD_DEFAULTS.items()}
    reward_cfg.update(reward_overrides)
    _, reward_fn = make_reward_fn_relative_QR(data_min, data_max, n_inputs, **reward_cfg)

    e = np.asarray(bundle["delta_y_storage"], float)
    du = np.asarray(bundle["delta_u_storage"], float)
    y_sp_phys = reconstruct_y_sp_phys(bundle)
    rewards = np.array([reward_fn(e[i], du[i], y_sp_phys[i]) for i in range(e.shape[0])], float)
    steps_per_episode = int(e.shape[0] // N_EPISODES)
    return rewards.reshape(N_EPISODES, steps_per_episode).mean(axis=1)


def compute_geometry_rows(system_name: str, reward_cfg: dict, setpoints_phys: np.ndarray, baseline_path: Path) -> list[dict]:
    baseline_bundle = load_pickle(baseline_path)
    data_min = np.asarray(baseline_bundle["data_min"], float)
    data_max = np.asarray(baseline_bundle["data_max"], float)
    n_inputs = int(np.asarray(baseline_bundle["delta_u_storage"], float).shape[1])
    output_ranges = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    q_diag = np.asarray(reward_cfg["Q_diag"], float)
    k_rel = np.asarray(reward_cfg["k_rel"], float)
    band_floor = np.asarray(reward_cfg["band_floor_phys"], float)
    beta = float(reward_cfg["beta"])

    rows = []
    for idx, y_sp in enumerate(np.asarray(setpoints_phys, float), start=1):
        band_phys = np.maximum(k_rel * np.abs(y_sp), band_floor)
        band_scaled = band_phys / output_ranges
        slope_at_edge = 2.0 * q_diag * band_scaled
        bonus_prefactor = beta * q_diag * (band_scaled**2)
        rows.append(
            {
                "system": system_name,
                "setpoint": f"SP{idx}",
                "output1_edge_slope": float(slope_at_edge[0]),
                "output2_edge_slope": float(slope_at_edge[1]),
                "output1_bonus_prefactor": float(bonus_prefactor[0]),
                "output2_bonus_prefactor": float(bonus_prefactor[1]),
                "edge_ratio_out1_to_out2": float(slope_at_edge[0] / max(slope_at_edge[1], 1e-12)),
                "bonus_ratio_out1_to_out2": float(bonus_prefactor[0] / max(bonus_prefactor[1], 1e-12)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scalar_bundle = load_pickle(DISTILLATION_SCALAR_MAY8)
    structured_bundle = load_pickle(DISTILLATION_STRUCTURED_MAY8)
    baseline_bundle = load_pickle(DISTILLATION_BASELINE)

    candidates = [
        {
            "key": "current",
            "label": "Current",
            "plot_label": "Current",
            "reward_overrides": {},
            "note": "Current distillation relative-band reward.",
        },
        {
            "key": "q1_15000",
            "label": "Q1=15000",
            "plot_label": "Q1=15000",
            "reward_overrides": {"Q_diag": np.array([15000.0, 1500.0], float)},
            "note": "Conservative reweighting toward polymer-like output balance.",
        },
        {
            "key": "q1_10000",
            "label": "Q1=10000",
            "plot_label": "Q1=10000",
            "reward_overrides": {"Q_diag": np.array([10000.0, 1500.0], float)},
            "note": "Balanced reweighting that materially lowers composition dominance.",
        },
        {
            "key": "q1_5300",
            "label": "Q1=5300",
            "plot_label": "Q1=5300",
            "reward_overrides": {"Q_diag": np.array([5300.0, 1500.0], float)},
            "note": "Approximate SP1 edge-equalized anchor from the current geometry table.",
        },
        {
            "key": "q1_10000_beta3_power06",
            "label": "Q1=10000, beta=3, power",
            "plot_label": "Q1=10000\nbeta=3\npower",
            "reward_overrides": {
                "Q_diag": np.array([10000.0, 1500.0], float),
                "beta": 3.0,
                "bonus_kind": "power",
                "bonus_p": 0.6,
            },
            "note": "Same weight rebalance plus lower absolute bonus and smoother power bonus.",
        },
    ]

    polymer_geometry = compute_geometry_rows(
        "polymer",
        POLYMER_REWARD_DEFAULTS,
        np.asarray(POLYMER_RL_SETPOINTS_PHYS, float),
        POLYMER_BASELINE,
    )
    polymer_sp2 = next(row for row in polymer_geometry if row["setpoint"] == "SP2")

    baseline_avg = rescore_episode_means(baseline_bundle, {})
    post_live_slice = slice(15, None)  # episode 16 onward for warm_start=10, freeze=5
    tail_slice = slice(-10, None)

    candidate_rows = []
    for candidate in candidates:
        reward_cfg = {k: v for k, v in DISTILLATION_REWARD_DEFAULTS.items()}
        reward_cfg.update(candidate["reward_overrides"])
        geometry_rows = compute_geometry_rows(
            "distillation",
            reward_cfg,
            np.asarray(DISTILLATION_RL_SETPOINTS_PHYS, float),
            DISTILLATION_BASELINE,
        )
        sp1 = next(row for row in geometry_rows if row["setpoint"] == "SP1")
        sp2 = next(row for row in geometry_rows if row["setpoint"] == "SP2")

        scalar_avg = rescore_episode_means(scalar_bundle, candidate["reward_overrides"])
        structured_avg = rescore_episode_means(structured_bundle, candidate["reward_overrides"])
        scalar_delta = scalar_avg - baseline_avg
        structured_delta = structured_avg - baseline_avg

        candidate_rows.append(
            {
                "candidate_key": candidate["key"],
                "candidate_label": candidate["label"],
                "note": candidate["note"],
                "Q1": float(np.asarray(reward_cfg["Q_diag"], float)[0]),
                "Q2": float(np.asarray(reward_cfg["Q_diag"], float)[1]),
                "beta": float(reward_cfg["beta"]),
                "bonus_kind": str(reward_cfg["bonus_kind"]),
                "bonus_p": float(reward_cfg.get("bonus_p", np.nan)),
                "sp1_edge_ratio": float(sp1["edge_ratio_out1_to_out2"]),
                "sp2_edge_ratio": float(sp2["edge_ratio_out1_to_out2"]),
                "sp1_bonus_ratio": float(sp1["bonus_ratio_out1_to_out2"]),
                "sp2_bonus_ratio": float(sp2["bonus_ratio_out1_to_out2"]),
                "sp2_output1_edge_slope": float(sp2["output1_edge_slope"]),
                "sp2_output2_edge_slope": float(sp2["output2_edge_slope"]),
                "sp2_output1_bonus_prefactor": float(sp2["output1_bonus_prefactor"]),
                "sp2_output2_bonus_prefactor": float(sp2["output2_bonus_prefactor"]),
                "polymer_sp2_edge_ratio_reference": float(polymer_sp2["edge_ratio_out1_to_out2"]),
                "polymer_sp2_bonus_ratio_reference": float(polymer_sp2["bonus_ratio_out1_to_out2"]),
                "scalar_post_live_reward_delta": float(scalar_delta[post_live_slice].mean()),
                "scalar_tail10_reward_delta": float(scalar_delta[tail_slice].mean()),
                "scalar_last_reward_delta": float(scalar_delta[-1]),
                "structured_post_live_reward_delta": float(structured_delta[post_live_slice].mean()),
                "structured_tail10_reward_delta": float(structured_delta[tail_slice].mean()),
                "structured_last_reward_delta": float(structured_delta[-1]),
            }
        )

    z_grid = np.linspace(0.0, 1.0, 101)
    curve_rows = []
    curve_specs = [
        ("exp_k12", {"kind": "exp", "k": 12.0}),
        ("power_p0p6", {"kind": "power", "p": 0.6}),
        ("quadratic", {"kind": "quadratic"}),
    ]
    for key, kwargs in curve_specs:
        phi = reward_curve(z_grid, **kwargs)
        for z, value in zip(z_grid, phi):
            curve_rows.append({"curve_key": key, "z": float(z), "phi": float(value)})

    write_csv(SUMMARY_CSV, candidate_rows)
    write_csv(CURVE_CSV, curve_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    bar_candidates = [row for row in candidate_rows if row["candidate_key"] in {"current", "q1_15000", "q1_10000", "q1_5300"}]
    x = np.arange(len(bar_candidates))
    width = 0.36
    axes[0].bar(x - width / 2, [row["sp2_edge_ratio"] for row in bar_candidates], width, color="#c44e52", label="SP2 edge ratio")
    axes[0].bar(x + width / 2, [row["sp2_bonus_ratio"] for row in bar_candidates], width, color="#4c72b0", label="SP2 bonus ratio")
    axes[0].axhline(float(polymer_sp2["edge_ratio_out1_to_out2"]), color="#c44e52", linestyle="--", linewidth=1.2, label="Polymer SP2 edge ratio")
    axes[0].axhline(float(polymer_sp2["bonus_ratio_out1_to_out2"]), color="#4c72b0", linestyle=":", linewidth=1.2, label="Polymer SP2 bonus ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([row["candidate_label"] for row in bar_candidates], rotation=18, ha="right")
    axes[0].set_ylabel("Output-1 / output-2 ratio")
    axes[0].set_title("Reweighting Q1 shrinks the SP2 imbalance")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    curve_labels = {
        "exp_k12": "Current exp k=12",
        "power_p0p6": "Power p=0.6",
        "quadratic": "Quadratic",
    }
    curve_colors = {
        "exp_k12": "#c44e52",
        "power_p0p6": "#55a868",
        "quadratic": "#4c72b0",
    }
    for key, _kwargs in curve_specs:
        phi = np.array([row["phi"] for row in curve_rows if row["curve_key"] == key], float)
        axes[1].plot(z_grid, phi, linewidth=2.2, color=curve_colors[key], label=curve_labels[key])
    axes[1].set_xlabel("Normalized error z = |e| / band")
    axes[1].set_ylabel("Bonus shape phi(z)")
    axes[1].set_title("Smoother bonus shapes are less cliff-like")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    rescore_candidates = ["current", "q1_15000", "q1_10000", "q1_10000_beta3_power06"]
    rescore_rows = [next(row for row in candidate_rows if row["candidate_key"] == key) for key in rescore_candidates]
    x = np.arange(len(rescore_rows))
    axes[2].bar(x - width / 2, [row["scalar_post_live_reward_delta"] for row in rescore_rows], width, color="#dd8452", label="Scalar May 8")
    axes[2].bar(
        x + width / 2,
        [row["structured_post_live_reward_delta"] for row in rescore_rows],
        width,
        color="#8172b3",
        label="Structured May 8",
    )
    axes[2].axhline(0.0, color="0.35", linewidth=1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([row["candidate_label"] for row in rescore_rows], rotation=18, ha="right")
    axes[2].set_ylabel("Post-live reward delta vs MPC")
    axes[2].set_title("Fixed-trajectory rescoring narrows the deficit, but does not create a win")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Distillation reward smoothing options: rebalance Q1, soften the bonus, and reduce composition-dominant harshness")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_payload = {
        "figure": str(FIG_PATH.relative_to(REPO_ROOT)),
        "summary_csv": str(SUMMARY_CSV.relative_to(REPO_ROOT)),
        "curve_csv": str(CURVE_CSV.relative_to(REPO_ROOT)),
        "polymer_sp2_reference": polymer_sp2,
        "candidates": candidate_rows,
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Wrote summary CSV: {SUMMARY_CSV}")
    print(f"Wrote curve CSV: {CURVE_CSV}")
    print(f"Wrote summary JSON: {SUMMARY_JSON}")
    print(f"Wrote figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
