from __future__ import annotations

import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from Simulation.mpc import augment_state_space
from systems.polymer.config import POLYMER_RL_SETPOINTS_PHYS, RL_REWARD_DEFAULTS
from utils.structured_model_update import (
    build_block_scaled_model,
    build_structured_update_spec,
    map_normalized_action_to_multipliers,
)
REPORT_ROOT = REPO_ROOT / "report" / "polymer_wide_range_matrix_structured"
FIG_ROOT = REPORT_ROOT / "figures"
DATA_ROOT = REPORT_ROOT / "data"

BASELINE_PATH = REPO_ROOT / "Polymer" / "Data" / "mpc_results_dist.pickle"
POLYMER_SYS_DICT = REPO_ROOT / "Polymer" / "Data" / "system_dict.pickle"
DISTILLATION_SYS_DICT = REPO_ROOT / "Distillation" / "Data" / "system_dict_new.pickle"
DISTILLATION_BASELINE_PATH = REPO_ROOT / "Distillation" / "Results" / "distillation_baseline_disturb_fluctuation_unified" / "20260413_085608" / "input_data.pkl"
DISTILLATION_MATRIX_DISTURB_PATH = REPO_ROOT / "Distillation" / "Results" / "distillation_matrix_sac_disturb_fluctuation_standard_unified" / "20260415_104840" / "input_data.pkl"
DISTILLATION_STRUCTURED_DISTURB_PATH = REPO_ROOT / "Distillation" / "Results" / "distillation_structured_matrix_sac_disturb_fluctuation_standard_unified" / "20260415_120923" / "input_data.pkl"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    family: str
    variant: str
    label: str
    path: Path


RUN_SPECS = [
    RunSpec("baseline", "baseline", "baseline", "Baseline MPC", BASELINE_PATH),
    RunSpec("matrix_legacy", "matrix", "legacy", "Matrix legacy", REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260411_011134" / "input_data.pkl"),
    RunSpec("matrix_refresh", "matrix", "narrow_refresh", "Matrix narrow refresh", REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260420_234944" / "input_data.pkl"),
    RunSpec("matrix_wide", "matrix", "wide_refresh", "Matrix wide", REPO_ROOT / "Polymer" / "Results" / "td3_multipliers_disturb" / "20260421_011145" / "input_data.pkl"),
    RunSpec("structured_legacy", "structured", "legacy", "Structured legacy", REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260409_193654" / "input_data.pkl"),
    RunSpec("structured_refresh", "structured", "narrow_refresh", "Structured narrow refresh", REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260420_235100" / "input_data.pkl"),
    RunSpec("structured_wide", "structured", "wide_refresh", "Structured wide", REPO_ROOT / "Polymer" / "Results" / "td3_structured_matrices_disturb" / "20260421_013208" / "input_data.pkl"),
    RunSpec("reid_refresh", "reidentification", "refresh", "Reidentification refresh", REPO_ROOT / "Polymer" / "Results" / "td3_reidentification_disturb" / "20260420_234346" / "input_data.pkl"),
]


COLORS = {
    "baseline": "#3A3A3A",
    "legacy": "#9C755F",
    "narrow_refresh": "#4E79A7",
    "wide_refresh": "#F28E2B",
    "refresh": "#E15759",
}


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def y_array(bundle: dict) -> np.ndarray:
    return np.asarray(bundle.get("y", bundle.get("y_mpc")), float)[:-1]


def setpoint_phys(bundle: dict) -> np.ndarray:
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = len(np.asarray(bundle["steady_states"]["ss_inputs"], float))
    y_ss_phys = np.asarray(bundle["steady_states"]["y_ss"], float)
    y_ss_scaled = (y_ss_phys - data_min[n_inputs:]) / np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    y_sp_scaled_abs = np.asarray(bundle["y_sp"], float) + y_ss_scaled
    return data_min[n_inputs:] + y_sp_scaled_abs * (data_max[n_inputs:] - data_min[n_inputs:])


def test_slice(bundle: dict) -> slice:
    episode_len = int(bundle["time_in_sub_episodes"])
    return slice(-episode_len, None)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def image_lines(filename: str, alt: str) -> list[str]:
    return [f"![{alt}](./polymer_wide_range_matrix_structured/figures/{filename})", ""]


def fmt(value: float, digits: int = 4) -> str:
    arr = np.asarray(value)
    if arr.ndim > 0:
        raise ValueError("fmt expects a scalar")
    if np.isnan(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def pct_change(new_value: float, old_value: float) -> str:
    if not np.isfinite(new_value) or not np.isfinite(old_value) or abs(old_value) < 1e-12:
        return "n/a"
    return f"{100.0 * (new_value - old_value) / old_value:+.1f}%"


def disturbance_consistent(bundle: dict, baseline_bundle: dict) -> bool:
    d0 = baseline_bundle.get("disturbance_profile")
    d1 = bundle.get("disturbance_profile")
    if not isinstance(d0, dict) or not isinstance(d1, dict):
        return False
    if set(d0.keys()) != set(d1.keys()):
        return False
    return all(np.allclose(np.asarray(d0[key], float), np.asarray(d1[key], float)) for key in d0.keys())


def reward_components_for_test(bundle: dict) -> dict:
    dy = np.asarray(bundle["delta_y_storage"], float)[test_slice(bundle)]
    du = np.asarray(bundle["delta_u_storage"], float)[test_slice(bundle)]
    q = np.asarray(RL_REWARD_DEFAULTS["Q_diag"], float)
    scaled_mae_out = np.mean(np.abs(dy), axis=0)
    weighted_quad = np.mean((dy ** 2) * q.reshape(1, -1), axis=0)
    return {
        "scaled_mae_out1": float(scaled_mae_out[0]),
        "scaled_mae_out2": float(scaled_mae_out[1]),
        "weighted_quad_out1": float(weighted_quad[0]),
        "weighted_quad_out2": float(weighted_quad[1]),
        "u_move_test": float(np.mean(np.linalg.norm(du, axis=1))),
        "reward_test": float(np.asarray(bundle["avg_rewards"], float)[-1]),
    }


def merged_reward_params(bundle: dict) -> dict:
    params = dict(RL_REWARD_DEFAULTS)
    params.update(bundle.get("reward_params") or {})
    return {
        key: (np.asarray(value, float) if isinstance(value, (list, tuple, np.ndarray)) else value)
        for key, value in params.items()
    }


def reward_phi(z: np.ndarray, params: dict) -> np.ndarray:
    kind = str(params["bonus_kind"])
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    if kind == "linear":
        return 1.0 - z
    if kind == "quadratic":
        return (1.0 - z) ** 2
    if kind == "exp":
        k = float(params["bonus_k"])
        return (np.exp(-k * z) - np.exp(-k)) / (1.0 - np.exp(-k))
    if kind == "power":
        return 1.0 - np.power(z, float(params["bonus_p"]))
    if kind == "log":
        c = float(params["bonus_c"])
        return np.log1p(c * (1.0 - z)) / np.log1p(c)
    raise ValueError(f"unknown bonus kind: {kind}")


def reward_decomposition_for_slice(bundle: dict, sl: slice) -> dict:
    params = merged_reward_params(bundle)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = len(np.asarray(bundle["steady_states"]["ss_inputs"], float))
    y_sp_phys = setpoint_phys(bundle)[sl]
    e_scaled = np.asarray(bundle["delta_y_storage"], float)[sl]
    du_scaled = np.asarray(bundle["delta_u_storage"], float)[sl]

    dy_range = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)
    band_phys = np.maximum(
        np.abs(y_sp_phys) * np.asarray(params["k_rel"], float).reshape(1, -1),
        np.asarray(params["band_floor_phys"], float).reshape(1, -1),
    )
    band_scaled = band_phys / dy_range.reshape(1, -1)
    tau_scaled = float(params["tau_frac"]) * band_scaled
    abs_e = np.abs(e_scaled)

    x = np.clip((band_scaled - abs_e) / np.maximum(tau_scaled, 1e-12), -60.0, 60.0)
    s_i = 1.0 / (1.0 + np.exp(-x))
    gate = str(params["gate"]).lower()
    if gate == "prod":
        w_in = np.prod(s_i, axis=1, dtype=np.float64)
    elif gate == "mean":
        w_in = np.mean(s_i, axis=1)
    elif gate == "geom":
        w_in = np.prod(s_i, axis=1, dtype=np.float64) ** (1.0 / s_i.shape[1])
    else:
        raise ValueError("gate must be 'prod'|'mean'|'geom'")

    q_diag = np.asarray(params["Q_diag"], float).reshape(1, -1)
    r_diag = np.asarray(params["R_diag"], float).reshape(1, -1)
    lam_in = float(params["lam_in"])
    gamma_out = float(params["gamma_out"])
    gamma_in = float(params["gamma_in"])
    beta = float(params["beta"])
    reward_scale = float(params["reward_scale"])

    err_quad_vec = q_diag * (e_scaled ** 2)
    quad_coeff = ((1.0 - w_in) + lam_in * w_in).reshape(-1, 1)
    err_eff_vec = quad_coeff * err_quad_vec

    slope_at_edge = 2.0 * q_diag * band_scaled
    overflow = np.maximum(abs_e - band_scaled, 0.0)
    inside_mag = np.minimum(abs_e, band_scaled)
    linear_vec = ((1.0 - w_in).reshape(-1, 1) * gamma_out * slope_at_edge * overflow) + (
        w_in.reshape(-1, 1) * gamma_in * slope_at_edge * inside_mag
    )

    qb2 = q_diag * (band_scaled ** 2)
    z = abs_e / np.maximum(band_scaled, 1e-12)
    phi = reward_phi(z, params)
    bonus_vec = w_in.reshape(-1, 1) * beta * qb2 * phi
    move_vec = r_diag * (du_scaled ** 2)

    reward_terms = reward_scale * (
        -np.sum(err_eff_vec, axis=1)
        - np.sum(linear_vec, axis=1)
        - np.sum(move_vec, axis=1)
        + np.sum(bonus_vec, axis=1)
    )

    return {
        "band_scaled_mean": np.mean(band_scaled, axis=0),
        "w_in_mean": float(np.mean(w_in)),
        "quad_out_mean": reward_scale * np.mean(err_eff_vec, axis=0),
        "linear_out_mean": reward_scale * np.mean(linear_vec, axis=0),
        "bonus_out_mean": reward_scale * np.mean(bonus_vec, axis=0),
        "move_in_mean": reward_scale * np.mean(move_vec, axis=0),
        "reward_mean": float(np.mean(reward_terms)),
    }


def reward_geometry_table(bundle: dict) -> list[dict]:
    params = merged_reward_params(bundle)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = len(np.asarray(bundle["steady_states"]["ss_inputs"], float))
    dy_range = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    q_diag = np.asarray(params["Q_diag"], float)
    band_floor = np.asarray(params["band_floor_phys"], float)
    k_rel = np.asarray(params["k_rel"], float)
    reward_scale = float(params["reward_scale"])
    beta = float(params["beta"])

    rows = []
    for sp_idx, y_sp_phys in enumerate(np.asarray(POLYMER_RL_SETPOINTS_PHYS, float), start=1):
        band_phys = np.maximum(k_rel * np.abs(y_sp_phys), band_floor)
        band_scaled = band_phys / dy_range
        edge_slope = reward_scale * 2.0 * q_diag * band_scaled
        bonus_prefactor = reward_scale * beta * q_diag * (band_scaled ** 2)
        for out_idx in range(band_scaled.size):
            rows.append(
                {
                    "setpoint": f"SP{sp_idx}",
                    "output": f"out{out_idx + 1}",
                    "y_sp_phys": float(y_sp_phys[out_idx]),
                    "band_phys": float(band_phys[out_idx]),
                    "band_scaled": float(band_scaled[out_idx]),
                    "edge_slope": float(edge_slope[out_idx]),
                    "bonus_prefactor": float(bonus_prefactor[out_idx]),
                }
            )
    return rows


def reward_balance_targets(bundle: dict) -> dict:
    params = merged_reward_params(bundle)
    data_min = np.asarray(bundle["data_min"], float)
    data_max = np.asarray(bundle["data_max"], float)
    n_inputs = len(np.asarray(bundle["steady_states"]["ss_inputs"], float))
    dy_range = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    q_diag = np.asarray(params["Q_diag"], float)
    q2 = float(q_diag[1])
    bands = []
    for y_sp_phys in np.asarray(POLYMER_RL_SETPOINTS_PHYS, float):
        band_phys = np.maximum(
            np.asarray(params["k_rel"], float) * np.abs(y_sp_phys),
            np.asarray(params["band_floor_phys"], float),
        )
        bands.append(band_phys / dy_range)
    bands = np.asarray(bands, float)
    q1_equal_edge = q2 * (bands[:, 1] / bands[:, 0])
    q1_equal_bonus = q2 * (bands[:, 1] / bands[:, 0]) ** 2
    return {
        "current_q": q_diag.copy(),
        "q1_equal_edge_per_sp": q1_equal_edge,
        "q1_equal_bonus_per_sp": q1_equal_bonus,
        "q1_equal_edge_mean": float(np.mean(q1_equal_edge)),
        "q1_equal_bonus_mean": float(np.mean(q1_equal_bonus)),
    }


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    blocks = [B]
    Ak = np.eye(A.shape[0], dtype=float)
    for _ in range(1, A.shape[0]):
        Ak = Ak @ A
        blocks.append(Ak @ B)
    return np.hstack(blocks)


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    C = np.asarray(C, float)
    blocks = [C]
    Ak = np.eye(A.shape[0], dtype=float)
    for _ in range(1, A.shape[0]):
        Ak = Ak @ A
        blocks.append(C @ Ak)
    return np.vstack(blocks)


def matrix_admissibility_grid(A: np.ndarray, B: np.ndarray, C: np.ndarray, alpha_grid: np.ndarray) -> list[dict]:
    rows = []
    for alpha in np.asarray(alpha_grid, float):
        A_alpha = float(alpha) * np.asarray(A, float)
        obs = observability_matrix(A_alpha, C)
        ctrb = controllability_matrix(A_alpha, B)
        rows.append(
            {
                "alpha": float(alpha),
                "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A_alpha)))),
                "observability_rank": int(np.linalg.matrix_rank(obs)),
                "controllability_rank": int(np.linalg.matrix_rank(ctrb)),
                "observability_cond": float(np.linalg.cond(obs)),
                "controllability_cond": float(np.linalg.cond(ctrb)),
            }
        )
    return rows


def structured_frontier_grid(
    A_aug: np.ndarray,
    B_aug: np.ndarray,
    C: np.ndarray,
    n_outputs: int,
    upper_grid: np.ndarray,
    *,
    lower_bound: float = 0.85,
    n_samples: int = 2000,
    seed: int = 7,
) -> list[dict]:
    spec = build_structured_update_spec(
        A_aug=A_aug,
        B_aug=B_aug,
        n_outputs=n_outputs,
        update_family="block",
        range_profile="wide",
    )
    rng = np.random.default_rng(seed)
    rows = []
    for upper in np.asarray(upper_grid, float):
        low = np.full(spec["action_dim"], float(lower_bound), dtype=float)
        high = np.full(spec["action_dim"], float(upper), dtype=float)
        sr_vals = np.zeros(n_samples, dtype=float)
        obs_rank = np.zeros(n_samples, dtype=int)
        obs_cond = np.zeros(n_samples, dtype=float)
        for idx in range(n_samples):
            action = rng.uniform(-1.0, 1.0, size=spec["action_dim"])
            theta = map_normalized_action_to_multipliers(action, low, high)
            model = build_block_scaled_model(
                A_aug=A_aug,
                B_aug=B_aug,
                n_outputs=n_outputs,
                block_cfg=spec["block_cfg"],
                theta_A=theta[: spec["a_dim"]],
                theta_B=theta[spec["a_dim"] :],
            )
            obs = observability_matrix(model["A_phys"], C)
            sr_vals[idx] = float(model["spectral_radius"])
            obs_rank[idx] = int(np.linalg.matrix_rank(obs))
            obs_cond[idx] = float(np.linalg.cond(obs))
        rows.append(
            {
                "upper_bound": float(upper),
                "lower_bound": float(lower_bound),
                "unstable_frac": float(np.mean(sr_vals >= 1.0)),
                "near_unit_frac": float(np.mean(sr_vals >= 0.98)),
                "spectral_p95": float(np.percentile(sr_vals, 95)),
                "observability_rank_min": int(np.min(obs_rank)),
                "bad_observability_frac": float(np.mean(obs_rank < C.shape[1])),
                "observability_cond_median": float(np.percentile(obs_cond, 50)),
                "observability_cond_p95": float(np.percentile(obs_cond, 95)),
            }
        )
    return rows


def series_model_deviation(bundle: dict) -> np.ndarray:
    a = np.asarray(bundle["A_model_delta_ratio_log"], float)
    b = np.asarray(bundle["B_model_delta_ratio_log"], float)
    return 0.5 * (a + b)


def authority_gate(tracking_raw: np.ndarray, deadband: float = 1.0, k: float = 0.35) -> np.ndarray:
    tracking_raw = np.asarray(tracking_raw, float)
    gate = np.zeros_like(tracking_raw, dtype=float)
    active = tracking_raw > float(deadband)
    gate[active] = 1.0 - np.exp(-float(k) * (tracking_raw[active] - float(deadband)))
    return np.clip(gate, 0.0, 1.0)


def gate_diagnostics(bundle: dict, sl: slice) -> dict:
    tracking = np.max(np.abs(np.asarray(bundle["tracking_error_raw_log"], float)[sl]), axis=1)
    deviation = series_model_deviation(bundle)[sl]
    gate = authority_gate(tracking, deadband=1.0, k=0.35)
    bins = np.array([0.0, 1.0, 3.0, 10.0, 30.0, np.inf], float)
    labels = ["<=1", "1-3", "3-10", "10-30", ">30"]
    rows = []
    for low, high, label in zip(bins[:-1], bins[1:], labels):
        mask = (tracking >= low) & (tracking < high)
        rows.append(
            {
                "bin": label,
                "count": int(np.sum(mask)),
                "tracking_mean": float(np.mean(tracking[mask])) if np.any(mask) else np.nan,
                "deviation_mean": float(np.mean(deviation[mask])) if np.any(mask) else np.nan,
                "gated_deviation_mean": float(np.mean((gate * deviation)[mask])) if np.any(mask) else np.nan,
            }
        )
    return {
        "tracking": tracking,
        "deviation": deviation,
        "gate": gate,
        "rows": rows,
    }


def compute_generic_run_summary(label: str, bundle: dict) -> dict:
    y = y_array(bundle)
    sp = setpoint_phys(bundle)
    err = y - sp
    ts = test_slice(bundle)
    row = {
        "label": label,
        "run_mode": str(bundle.get("run_mode", "n/a")),
        "algorithm": str(bundle.get("algorithm", "n/a")),
        "range_profile": str(bundle.get("range_profile", "n/a")),
        "update_family": str(bundle.get("update_family", "n/a")),
        "test_phys_mae_mean": float(np.mean(np.abs(err[ts]))),
        "test_phys_mae_out1": float(np.mean(np.abs(err[ts, 0]))),
        "test_phys_mae_out2": float(np.mean(np.abs(err[ts, 1]))),
        "reward_test": float(np.asarray(bundle["avg_rewards"], float)[-1]),
        "time_in_sub_episodes": int(bundle["time_in_sub_episodes"]),
    }
    if bundle.get("alpha_log") is not None:
        alpha = np.asarray(bundle["alpha_log"], float)[ts]
        delta = np.asarray(bundle["delta_log"], float)[ts]
        row.update(
            {
                "alpha_mean_test": float(np.mean(alpha)),
                "alpha_p95_test": float(np.percentile(alpha, 95)),
                "alpha_max_test": float(np.max(alpha)),
                "delta1_mean_test": float(np.mean(delta[:, 0])),
                "delta2_mean_test": float(np.mean(delta[:, 1])),
                "delta1_p95_test": float(np.percentile(delta[:, 0], 95)),
                "delta2_p95_test": float(np.percentile(delta[:, 1], 95)),
            }
        )
    if bundle.get("mapped_multiplier_log") is not None:
        mm = np.asarray(bundle["mapped_multiplier_log"], float)[ts]
        sr = np.asarray(bundle["spectral_radius_log"], float)[ts]
        row.update(
            {
                "multiplier_mean_test": np.mean(mm, axis=0),
                "multiplier_p95_test": np.percentile(mm, 95, axis=0),
                "spectral_mean_test": float(np.mean(sr)),
                "spectral_p95_test": float(np.percentile(sr, 95)),
                "spectral_max_test": float(np.max(sr)),
            }
        )
    return row


def plot_cross_system_admissibility(
    polymer_matrix_rows: list[dict],
    polymer_structured_rows: list[dict],
    distillation_matrix_rows: list[dict],
    distillation_structured_rows: list[dict],
    polymer_alpha_max: float,
    distillation_alpha_max: float,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    axs[0].plot(
        [row["alpha"] for row in polymer_matrix_rows],
        [row["spectral_radius"] for row in polymer_matrix_rows],
        color="#F28E2B",
        lw=2.2,
        label="polymer",
    )
    axs[0].plot(
        [row["alpha"] for row in distillation_matrix_rows],
        [row["spectral_radius"] for row in distillation_matrix_rows],
        color="#4E79A7",
        lw=2.2,
        label="distillation",
    )
    axs[0].axhline(1.0, color="black", lw=1.0, ls=":")
    axs[0].axvline(polymer_alpha_max, color="#F28E2B", lw=1.0, ls="--", label=f"polymer alpha_max {polymer_alpha_max:.3f}")
    axs[0].axvline(distillation_alpha_max, color="#4E79A7", lw=1.0, ls="--", label=f"distillation alpha_max {distillation_alpha_max:.3f}")
    axs[0].set_title("Matrix spectral frontier by system")
    axs[0].set_xlabel("A multiplier alpha")
    axs[0].set_ylabel("Spectral radius")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.25)

    axs[1].plot(
        [row["upper_bound"] for row in polymer_structured_rows],
        [row["unstable_frac"] for row in polymer_structured_rows],
        color="#F28E2B",
        lw=2.2,
        marker="o",
        label="polymer",
    )
    axs[1].plot(
        [row["upper_bound"] for row in distillation_structured_rows],
        [row["unstable_frac"] for row in distillation_structured_rows],
        color="#4E79A7",
        lw=2.2,
        marker="s",
        label="distillation",
    )
    axs[1].set_title("Structured block unstable fraction by system")
    axs[1].set_xlabel("Common upper multiplier bound")
    axs[1].set_ylabel("Fraction of sampled models with spectral radius >= 1")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.25)

    fig.savefig(FIG_ROOT / "wide_range_cross_system_admissibility.png", dpi=220)
    plt.close(fig)


def plot_b_multiplier_design() -> None:
    delta = np.linspace(0.70, 1.30, 300)
    gain_ratio = delta
    required_move_ratio = 1.0 / delta
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    axs[0].plot(delta, gain_ratio, color="#4E79A7", lw=2.2)
    axs[0].axhline(1.0, color="black", lw=1.0, ls=":")
    axs[0].axvspan(0.85, 1.20, color="#F28E2B", alpha=0.15, label="polymer current band")
    axs[0].axvspan(0.95, 1.05, color="#59A14F", alpha=0.18, label="distillation current band")
    axs[0].set_title("B multiplier and predicted input gain")
    axs[0].set_xlabel("B multiplier delta")
    axs[0].set_ylabel("Predicted gain ratio")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.25)

    axs[1].plot(delta, required_move_ratio, color="#E15759", lw=2.2)
    axs[1].axhline(1.0, color="black", lw=1.0, ls=":")
    axs[1].axvspan(0.85, 1.20, color="#F28E2B", alpha=0.15, label="polymer current band")
    axs[1].axvspan(0.95, 1.05, color="#59A14F", alpha=0.18, label="distillation current band")
    axs[1].set_title("B multiplier and required move inflation")
    axs[1].set_xlabel("B multiplier delta")
    axs[1].set_ylabel("Approx. move ratio for same correction")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.25)

    fig.savefig(FIG_ROOT / "wide_range_b_multiplier_design.png", dpi=220)
    plt.close(fig)


def plot_practical_fixes_explainer(
    polymer_structured_rows: list[dict],
    distillation_structured_rows: list[dict],
) -> None:
    rng = np.random.default_rng(17)
    episodes = np.arange(1, 201)
    upper_schedule = np.piecewise(
        episodes.astype(float),
        [episodes <= 60, (episodes > 60) & (episodes <= 120), episodes > 120],
        [1.05, 1.10, 1.20],
    )

    white = np.clip(rng.normal(0.0, 0.45, size=160), -1.0, 1.0)
    ar = np.zeros(160, dtype=float)
    for idx in range(1, ar.size):
        ar[idx] = np.clip(0.92 * ar[idx - 1] + rng.normal(0.0, 0.12), -1.0, 1.0)

    uncertainty = np.linspace(0.0, 1.0, 200)
    excitation = np.clip((uncertainty - 0.25) / 0.55, 0.0, 1.0)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    axs[0, 0].step(episodes, upper_schedule, where="post", color="#4E79A7", lw=2.2)
    axs[0, 0].axvline(60, color="#999999", lw=1.0, ls=":")
    axs[0, 0].axvline(120, color="#999999", lw=1.0, ls=":")
    axs[0, 0].set_title("Progressive widening / continuation")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Upper multiplier bound")
    axs[0, 0].text(12, 1.057, "phase 1: narrow", fontsize=9)
    axs[0, 0].text(74, 1.107, "phase 2: intermediate", fontsize=9)
    axs[0, 0].text(136, 1.207, "phase 3: wide", fontsize=9)
    axs[0, 0].grid(alpha=0.25)

    axs[0, 1].plot(white, color="#E15759", lw=1.4, label="i.i.d. action noise")
    axs[0, 1].plot(ar, color="#59A14F", lw=1.8, label="AR(1) action noise")
    axs[0, 1].axhline(0.0, color="black", lw=1.0, ls=":")
    axs[0, 1].set_title("Smoother exploration")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Normalized action")
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(alpha=0.25)

    axs[1, 0].plot(uncertainty, excitation, color="#F28E2B", lw=2.2)
    axs[1, 0].fill_between(uncertainty, 0.0, excitation, color="#F28E2B", alpha=0.18)
    axs[1, 0].set_title("Data-aware excitation only when needed")
    axs[1, 0].set_xlabel("Model uncertainty / informativeness deficit")
    axs[1, 0].set_ylabel("Extra excitation authority")
    axs[1, 0].grid(alpha=0.25)

    axs[1, 1].plot(
        [row["upper_bound"] for row in polymer_structured_rows],
        [row["unstable_frac"] for row in polymer_structured_rows],
        color="#F28E2B",
        lw=2.2,
        marker="o",
        label="polymer",
    )
    axs[1, 1].plot(
        [row["upper_bound"] for row in distillation_structured_rows],
        [row["unstable_frac"] for row in distillation_structured_rows],
        color="#4E79A7",
        lw=2.2,
        marker="s",
        label="distillation",
    )
    axs[1, 1].axvline(1.05, color="#999999", lw=1.0, ls=":")
    axs[1, 1].axvline(1.20, color="#999999", lw=1.0, ls="--")
    axs[1, 1].set_title("Robust uncertainty-set shaping")
    axs[1, 1].set_xlabel("Common upper multiplier bound")
    axs[1, 1].set_ylabel("Unstable fraction")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(alpha=0.25)

    fig.savefig(FIG_ROOT / "wide_range_practical_fixes_explainer.png", dpi=220)
    plt.close(fig)


def plot_distillation_counterpart(distillation_runs: dict[str, dict]) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    order = [
        ("baseline", "Baseline fluctuation", "#3A3A3A", "--"),
        ("matrix", "Matrix SAC disturbance", "#4E79A7", "-"),
        ("structured", "Structured SAC disturbance", "#F28E2B", "-"),
    ]
    for ax_idx in range(2):
        for key, label, color, ls in order:
            bundle = distillation_runs[key]["bundle"]
            ts = test_slice(bundle)
            y = y_array(bundle)[ts]
            sp = setpoint_phys(bundle)[ts]
            x = np.linspace(0.0, 1.0, y.shape[0])
            axs[ax_idx].plot(x, y[:, ax_idx], color=color, lw=1.8, ls=ls, label=label)
            axs[ax_idx].plot(x, sp[:, ax_idx], color=color, lw=1.0, ls=":", alpha=0.65)
        axs[ax_idx].grid(alpha=0.25)
    axs[0].set_title("Distillation final test episode: current saved disturbance runs")
    axs[0].set_ylabel("Tray-24 ethane composition")
    axs[1].set_ylabel("Tray-85 temperature")
    axs[1].set_xlabel("Normalized test-episode time")
    axs[0].legend(fontsize=8)
    fig.savefig(FIG_ROOT / "wide_range_distillation_counterpart.png", dpi=220)
    plt.close(fig)


def compute_summary(spec: RunSpec, bundle: dict, baseline_bundle: dict, nominal_A_radius: float) -> dict:
    y = y_array(bundle)
    sp = setpoint_phys(bundle)
    err_phys = y - sp
    dy = np.asarray(bundle["delta_y_storage"], float)
    rewards = np.asarray(bundle["avg_rewards"], float)
    ts = test_slice(bundle)
    tail = slice(int(0.9 * len(err_phys)), len(err_phys))

    row = {
        "run_id": spec.run_id,
        "family": spec.family,
        "variant": spec.variant,
        "label": spec.label,
        "run_path": repo_rel(spec.path),
        "run_mode": str(bundle.get("run_mode", "n/a")),
        "state_mode": str(bundle.get("state_mode", "n/a")),
        "observer_update_alignment": str(bundle.get("observer_update_alignment", "n/a")),
        "base_state_norm_mode": str(bundle.get("base_state_norm_mode", "n/a")),
        "mismatch_feature_transform_mode": str(bundle.get("mismatch_feature_transform_mode", "n/a")),
        "range_profile": str(bundle.get("range_profile", "n/a")),
        "update_family": str(bundle.get("update_family", "n/a")),
        "disturbance_consistent": disturbance_consistent(bundle, baseline_bundle),
        "tail_phys_mae_mean": float(np.mean(np.abs(err_phys[tail]))),
        "test_phys_mae_mean": float(np.mean(np.abs(err_phys[ts]))),
        "test_phys_mae_out1": float(np.mean(np.abs(err_phys[ts, 0]))),
        "test_phys_mae_out2": float(np.mean(np.abs(err_phys[ts, 1]))),
        "test_scaled_mae_out1": float(np.mean(np.abs(dy[ts, 0]))),
        "test_scaled_mae_out2": float(np.mean(np.abs(dy[ts, 1]))),
        "test_u_move": float(np.mean(np.linalg.norm(np.asarray(bundle["delta_u_storage"], float)[ts], axis=1))),
        "reward_last": float(rewards[-1]),
        "reward_final10_mean": float(np.mean(rewards[-10:])),
        "reward_best": float(np.max(rewards)),
        "reward_trough": float(np.min(rewards)),
        "reward_trough_episode": int(np.argmin(rewards) + 1),
    }

    target_end10 = row["reward_final10_mean"]
    trough_idx = int(np.argmin(rewards))
    recover_idx = np.where(rewards[trough_idx:] >= target_end10)[0]
    row["reward_recover_episode"] = int(trough_idx + 1 + recover_idx[0]) if recover_idx.size else np.nan

    row.update(reward_components_for_test(bundle))
    reward_decomp = reward_decomposition_for_slice(bundle, ts)
    row.update(
        {
            "reward_gate_mean_test": reward_decomp["w_in_mean"],
            "reward_quad_out1": float(reward_decomp["quad_out_mean"][0]),
            "reward_quad_out2": float(reward_decomp["quad_out_mean"][1]),
            "reward_linear_out1": float(reward_decomp["linear_out_mean"][0]),
            "reward_linear_out2": float(reward_decomp["linear_out_mean"][1]),
            "reward_bonus_out1": float(reward_decomp["bonus_out_mean"][0]),
            "reward_bonus_out2": float(reward_decomp["bonus_out_mean"][1]),
            "reward_move_in1": float(reward_decomp["move_in_mean"][0]),
            "reward_move_in2": float(reward_decomp["move_in_mean"][1]),
        }
    )

    if bundle.get("alpha_log") is not None:
        alpha = np.asarray(bundle["alpha_log"], float)[ts]
        delta = np.asarray(bundle["delta_log"], float)[ts]
        derived_sr = nominal_A_radius * alpha
        row.update(
            {
                "alpha_mean_test": float(np.mean(alpha)),
                "alpha_p95_test": float(np.percentile(alpha, 95)),
                "alpha_max_test": float(np.max(alpha)),
                "delta1_mean_test": float(np.mean(delta[:, 0])),
                "delta2_mean_test": float(np.mean(delta[:, 1])),
                "delta1_p95_test": float(np.percentile(delta[:, 0], 95)),
                "delta2_p95_test": float(np.percentile(delta[:, 1], 95)),
                "A_model_delta_ratio_mean_test": float(np.mean(np.asarray(bundle["A_model_delta_ratio_log"], float)[ts])),
                "B_model_delta_ratio_mean_test": float(np.mean(np.asarray(bundle["B_model_delta_ratio_log"], float)[ts])),
                "derived_spectral_mean_test": float(np.mean(derived_sr)),
                "derived_spectral_p95_test": float(np.percentile(derived_sr, 95)),
                "derived_spectral_max_test": float(np.max(derived_sr)),
            }
        )

    if bundle.get("mapped_multiplier_log") is not None:
        mapped = np.asarray(bundle["mapped_multiplier_log"], float)[ts]
        row.update(
            {
                "mapped_multiplier_abs_mean_test": float(np.mean(np.abs(mapped))),
                "mapped_multiplier_abs_p95_test": float(np.percentile(np.abs(mapped), 95)),
            }
        )

    if bundle.get("spectral_radius_log") is not None:
        sr = np.asarray(bundle["spectral_radius_log"], float)[ts]
        row.update(
            {
                "spectral_mean_test": float(np.mean(sr)),
                "spectral_p95_test": float(np.percentile(sr, 95)),
                "spectral_max_test": float(np.max(sr)),
                "near_bound_mean_test": float(np.mean(np.asarray(bundle["near_bound_fraction_log"], float)[ts])),
                "action_saturation_mean_test": float(np.mean(np.asarray(bundle["action_saturation_fraction_log"], float)[ts])),
                "prediction_fallback_frac_test": float(np.mean(np.asarray(bundle.get("prediction_fallback_active_log", np.zeros(len(sr))), float)[ts])),
            }
        )

    if "id_candidate_valid_log" in bundle:
        residual_ratio_log = bundle.get("id_residual_ratio_full_log")
        row.update(
            {
                "candidate_valid_frac": float(np.mean(np.asarray(bundle["id_candidate_valid_log"], float))),
                "update_success_frac": float(np.mean(np.asarray(bundle["id_update_success_log"], float))),
                "condition_median": float(np.percentile(np.asarray(bundle["id_condition_number_log"], float), 50)),
                "condition_p95": float(np.percentile(np.asarray(bundle["id_condition_number_log"], float), 95)),
                "residual_ratio_median": float(np.percentile(np.asarray(residual_ratio_log, float), 50)) if residual_ratio_log is not None else np.nan,
                "residual_ratio_p95": float(np.percentile(np.asarray(residual_ratio_log, float), 95)) if residual_ratio_log is not None else np.nan,
            }
        )

    if bundle.get("tracking_error_raw_log") is not None and bundle.get("A_model_delta_ratio_log") is not None and bundle.get("B_model_delta_ratio_log") is not None:
        tracking_raw = np.max(np.abs(np.asarray(bundle["tracking_error_raw_log"], float)[ts]), axis=1)
        gate_diag = gate_diagnostics(bundle, ts)
        row.update(
            {
                "tracking_raw_p25_test": float(np.percentile(tracking_raw, 25)),
                "tracking_raw_p75_test": float(np.percentile(tracking_raw, 75)),
                "deviation_mean_low_tracking_test": float(np.mean(gate_diag["deviation"][tracking_raw <= 1.0])) if np.any(tracking_raw <= 1.0) else np.nan,
                "deviation_mean_high_tracking_test": float(np.mean(gate_diag["deviation"][tracking_raw > 10.0])) if np.any(tracking_raw > 10.0) else np.nan,
                "deviation_gt_0p1_frac_low_tracking_test": float(np.mean(gate_diag["deviation"][tracking_raw <= 1.0] > 0.1)) if np.any(tracking_raw <= 1.0) else np.nan,
                "gated_deviation_mean_test": float(np.mean(gate_diag["gate"] * gate_diag["deviation"])),
                "gate_mean_test": float(np.mean(gate_diag["gate"])),
                "low_tracking_frac_test": float(np.mean(tracking_raw <= 1.0)),
            }
        )

    return row


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_reward_curves(runs: dict[str, dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    families = [
        ("matrix", ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide"], "Matrix reward curves"),
        ("structured", ["baseline", "structured_legacy", "structured_refresh", "structured_wide"], "Structured reward curves"),
    ]
    for ax, (_, run_ids, title) in zip(axs, families):
        for run_id in run_ids:
            spec = runs[run_id]["spec"]
            rewards = np.asarray(runs[run_id]["bundle"]["avg_rewards"], float)
            variant_key = spec.variant
            color = COLORS["baseline"] if run_id == "baseline" else COLORS[variant_key]
            lw = 2.4 if "wide" in run_id else 1.8
            ls = "--" if run_id == "baseline" else "-"
            ax.plot(np.arange(1, len(rewards) + 1), rewards, color=color, lw=lw, ls=ls, label=spec.label)
            trough_idx = int(np.argmin(rewards))
            ax.scatter(trough_idx + 1, rewards[trough_idx], color=color, s=20)
        ax.set_title(title)
        ax.set_xlabel("Sub-episode")
        ax.set_ylabel("Average reward")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "wide_range_reward_curves.png", dpi=220)
    plt.close(fig)


def plot_final_test_outputs(runs: dict[str, dict]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    panels = [
        ("matrix", ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide"], 0, "Matrix family: final test episode"),
        ("structured", ["baseline", "structured_legacy", "structured_refresh", "structured_wide"], 1, "Structured family: final test episode"),
    ]
    for _, run_ids, col, title in panels:
        for run_id in run_ids:
            spec = runs[run_id]["spec"]
            bundle = runs[run_id]["bundle"]
            ts = test_slice(bundle)
            y = y_array(bundle)[ts]
            sp = setpoint_phys(bundle)[ts]
            color = COLORS["baseline"] if run_id == "baseline" else COLORS[spec.variant]
            lw = 2.4 if "wide" in run_id else 1.8
            ls = "--" if run_id == "baseline" else "-"
            axs[0, col].plot(y[:, 0], color=color, lw=lw, ls=ls, label=spec.label)
            axs[1, col].plot(y[:, 1], color=color, lw=lw, ls=ls, label=spec.label)
        axs[0, col].plot(sp[:, 0], color="#000000", lw=1.2, ls=":", label="setpoint")
        axs[1, col].plot(sp[:, 1], color="#000000", lw=1.2, ls=":", label="setpoint")
        axs[0, col].set_title(title)
        axs[1, col].set_xlabel("Final test step")
        axs[0, col].grid(alpha=0.25)
        axs[1, col].grid(alpha=0.25)
        axs[0, col].legend(fontsize=8)
    axs[0, 0].set_ylabel("eta")
    axs[1, 0].set_ylabel("T")
    fig.savefig(FIG_ROOT / "wide_range_final_test_outputs.png", dpi=220)
    plt.close(fig)


def plot_model_usage(runs: dict[str, dict], nominal_A_radius: float) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    matrix_refresh = runs["matrix_refresh"]["bundle"]
    matrix_wide = runs["matrix_wide"]["bundle"]
    ts_m = test_slice(matrix_wide)
    alpha_refresh = np.asarray(matrix_refresh["alpha_log"], float)[test_slice(matrix_refresh)]
    alpha_wide = np.asarray(matrix_wide["alpha_log"], float)[ts_m]
    delta_refresh = np.asarray(matrix_refresh["delta_log"], float)[test_slice(matrix_refresh)]
    delta_wide = np.asarray(matrix_wide["delta_log"], float)[ts_m]

    labels = ["alpha", "delta1", "delta2"]
    x = np.arange(len(labels))
    w = 0.35
    mean_refresh = np.array([np.mean(alpha_refresh), np.mean(delta_refresh[:, 0]), np.mean(delta_refresh[:, 1])], float)
    mean_wide = np.array([np.mean(alpha_wide), np.mean(delta_wide[:, 0]), np.mean(delta_wide[:, 1])], float)
    p95_refresh = np.array([np.percentile(alpha_refresh, 95), np.percentile(delta_refresh[:, 0], 95), np.percentile(delta_refresh[:, 1], 95)], float)
    p95_wide = np.array([np.percentile(alpha_wide, 95), np.percentile(delta_wide[:, 0], 95), np.percentile(delta_wide[:, 1], 95)], float)

    axs[0, 0].bar(x - w / 2, mean_refresh, width=w, color=COLORS["narrow_refresh"], label="narrow mean")
    axs[0, 0].bar(x + w / 2, mean_wide, width=w, color=COLORS["wide_refresh"], label="wide mean")
    axs[0, 0].scatter(x - w / 2, p95_refresh, color="#173F5F", marker="D", s=30, label="narrow p95")
    axs[0, 0].scatter(x + w / 2, p95_wide, color="#A23E02", marker="D", s=30, label="wide p95")
    axs[0, 0].axhline(1.0, color="black", lw=1.0, ls=":")
    axs[0, 0].set_title("Matrix final-test multiplier usage")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].legend(fontsize=8)

    sr_refresh = nominal_A_radius * alpha_refresh
    sr_wide = nominal_A_radius * alpha_wide
    axs[0, 1].hist(sr_refresh, bins=30, alpha=0.6, color=COLORS["narrow_refresh"], label="matrix narrow")
    axs[0, 1].hist(sr_wide, bins=30, alpha=0.6, color=COLORS["wide_refresh"], label="matrix wide")
    axs[0, 1].axvline(1.0, color="black", lw=1.0, ls=":")
    axs[0, 1].set_title("Matrix derived spectral radius in final test")
    axs[0, 1].set_xlabel("spectral radius")
    axs[0, 1].legend(fontsize=8)

    structured_refresh = runs["structured_refresh"]["bundle"]
    structured_wide = runs["structured_wide"]["bundle"]
    mm_refresh = np.asarray(structured_refresh["mapped_multiplier_log"], float)[test_slice(structured_refresh)]
    mm_wide = np.asarray(structured_wide["mapped_multiplier_log"], float)[test_slice(structured_wide)]
    dim_labels = [f"m{i+1}" for i in range(mm_wide.shape[1])]
    x2 = np.arange(mm_wide.shape[1])
    axs[1, 0].bar(x2 - w / 2, np.mean(mm_refresh, axis=0), width=w, color=COLORS["narrow_refresh"], label="narrow mean")
    axs[1, 0].bar(x2 + w / 2, np.mean(mm_wide, axis=0), width=w, color=COLORS["wide_refresh"], label="wide mean")
    axs[1, 0].scatter(x2 - w / 2, np.percentile(mm_refresh, 95, axis=0), color="#173F5F", marker="D", s=25, label="narrow p95")
    axs[1, 0].scatter(x2 + w / 2, np.percentile(mm_wide, 95, axis=0), color="#A23E02", marker="D", s=25, label="wide p95")
    axs[1, 0].axhline(1.0, color="black", lw=1.0, ls=":")
    axs[1, 0].set_title("Structured final-test multiplier usage")
    axs[1, 0].set_xticks(x2)
    axs[1, 0].set_xticklabels(dim_labels)
    axs[1, 0].legend(fontsize=8)

    sr_refresh_s = np.asarray(structured_refresh["spectral_radius_log"], float)[test_slice(structured_refresh)]
    sr_wide_s = np.asarray(structured_wide["spectral_radius_log"], float)[test_slice(structured_wide)]
    axs[1, 1].hist(sr_refresh_s, bins=30, alpha=0.6, color=COLORS["narrow_refresh"], label="structured narrow")
    axs[1, 1].hist(sr_wide_s, bins=30, alpha=0.6, color=COLORS["wide_refresh"], label="structured wide")
    axs[1, 1].axvline(1.0, color="black", lw=1.0, ls=":")
    axs[1, 1].set_title("Structured spectral radius in final test")
    axs[1, 1].set_xlabel("spectral radius")
    axs[1, 1].legend(fontsize=8)

    for ax in axs.flat:
        ax.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "wide_range_model_usage.png", dpi=220)
    plt.close(fig)


def plot_tradeoff(summary_map: dict[str, dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    run_ids = ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide", "structured_legacy", "structured_refresh", "structured_wide", "reid_refresh"]
    for run_id in run_ids:
        row = summary_map[run_id]
        color = COLORS["baseline"] if run_id == "baseline" else COLORS.get(row["variant"], "#888888")
        marker = "X" if "wide" in run_id else "o"
        axs[0].scatter(row["reward_test"], row["test_phys_mae_mean"], color=color, marker=marker, s=90)
        axs[0].text(row["reward_test"], row["test_phys_mae_mean"] + 0.002, row["label"], fontsize=8)
    axs[0].set_title("Final test reward vs final test physical MAE")
    axs[0].set_xlabel("Final test average reward")
    axs[0].set_ylabel("Final test physical MAE mean")
    axs[0].grid(alpha=0.25)

    labels = ["Matrix narrow", "Matrix wide", "Structured narrow", "Structured wide"]
    rows = [summary_map["matrix_refresh"], summary_map["matrix_wide"], summary_map["structured_refresh"], summary_map["structured_wide"]]
    x = np.arange(len(labels))
    w = 0.35
    out1 = [row["test_scaled_mae_out1"] for row in rows]
    out2 = [row["test_scaled_mae_out2"] for row in rows]
    axs[1].bar(x - w / 2, out1, width=w, color="#59A14F", label="output 1 scaled MAE")
    axs[1].bar(x + w / 2, out2, width=w, color="#E15759", label="output 2 scaled MAE")
    axs[1].set_title("Final test scaled MAE by output")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=15, ha="right")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.25)

    fig.savefig(FIG_ROOT / "wide_range_reward_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_reward_balance(reference_bundle: dict) -> None:
    geom_rows = reward_geometry_table(reference_bundle)
    targets = reward_balance_targets(reference_bundle)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    setpoints = ["SP1", "SP2"]
    x = np.arange(len(setpoints))
    w = 0.35
    edge_out1 = [row["edge_slope"] for row in geom_rows if row["output"] == "out1"]
    edge_out2 = [row["edge_slope"] for row in geom_rows if row["output"] == "out2"]
    axs[0].bar(x - w / 2, edge_out1, width=w, color="#4E79A7", label="output 1 edge slope")
    axs[0].bar(x + w / 2, edge_out2, width=w, color="#E15759", label="output 2 edge slope")
    axs[0].set_title("Reward slope at the band edge")
    axs[0].set_ylabel("Reward units per scaled error")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(setpoints)
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.25)

    labels = ["Current Q1", "Q1 for equal edge", "Q1 for equal bonus"]
    q_vals = [
        float(targets["current_q"][0]),
        float(targets["q1_equal_edge_mean"]),
        float(targets["q1_equal_bonus_mean"]),
    ]
    axs[1].bar(np.arange(len(labels)), q_vals, color=["#4E79A7", "#59A14F", "#F28E2B"])
    axs[1].axhline(float(targets["current_q"][1]), color="black", lw=1.0, ls=":", label="Q2 fixed")
    axs[1].set_title("Implied output-1 weight if output-2 weight stays at 90")
    axs[1].set_ylabel("Q1")
    axs[1].set_xticks(np.arange(len(labels)))
    axs[1].set_xticklabels(labels, rotation=15, ha="right")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.25)

    fig.savefig(FIG_ROOT / "wide_range_reward_balance.png", dpi=220)
    plt.close(fig)


def plot_admissibility_frontier(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    A_aug: np.ndarray,
    B_aug: np.ndarray,
    n_outputs: int,
) -> tuple[list[dict], list[dict]]:
    alpha_grid = np.linspace(0.70, 1.25, 111)
    matrix_rows = matrix_admissibility_grid(A, B, C, alpha_grid)
    structured_rows = structured_frontier_grid(
        A_aug=A_aug,
        B_aug=B_aug,
        C=C,
        n_outputs=n_outputs,
        upper_grid=np.array([1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20], float),
    )

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axs[0].plot(
        [row["alpha"] for row in matrix_rows],
        [row["spectral_radius"] for row in matrix_rows],
        color="#4E79A7",
        lw=2.2,
        label=r"$\rho(\alpha A)$",
    )
    axs[0].axhline(1.0, color="black", lw=1.0, ls=":")
    alpha_max = 1.0 / float(np.max(np.abs(np.linalg.eigvals(np.asarray(A, float)))))
    axs[0].axvline(alpha_max, color="#E15759", lw=1.2, ls="--", label=f"alpha_max stable = {alpha_max:.4f}")
    axs[0].set_title("Matrix family: spectral admissibility frontier")
    axs[0].set_xlabel("A multiplier alpha")
    axs[0].set_ylabel("Spectral radius")
    axs[0].text(
        0.705,
        1.13,
        "Observability rank = 7\nControllability rank = 7\nfor all alpha > 0 tested",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.25)

    uppers = [row["upper_bound"] for row in structured_rows]
    unstable = [row["unstable_frac"] for row in structured_rows]
    p95 = [row["spectral_p95"] for row in structured_rows]
    axs[1].plot(uppers, unstable, color="#F28E2B", lw=2.2, marker="o", label="unstable fraction")
    ax2 = axs[1].twinx()
    ax2.plot(uppers, p95, color="#59A14F", lw=2.0, marker="s", label="spectral p95")
    axs[1].axvline(1.05, color="#4E79A7", lw=1.0, ls=":", label="zero-unstable frontier ~ 1.05")
    axs[1].axvline(1.20, color="#E15759", lw=1.0, ls="--", label="current wide upper bound")
    axs[1].set_title("Structured block family: empirical frontier")
    axs[1].set_xlabel("Common upper multiplier bound")
    axs[1].set_ylabel("Fraction of sampled models with spectral radius >= 1")
    ax2.set_ylabel("Sampled spectral-radius p95")
    axs[1].text(
        1.03,
        0.48,
        "Sampled observability rank stays full.\nThe frontier is instability, not rank loss.",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    lines_left, labels_left = axs[1].get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    axs[1].legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="upper left")
    axs[1].grid(alpha=0.25)

    fig.savefig(FIG_ROOT / "wide_range_admissibility_frontier.png", dpi=220)
    plt.close(fig)
    return matrix_rows, structured_rows


def plot_authority_gate(runs: dict[str, dict]) -> dict[str, dict]:
    gate_map = {
        "matrix_wide": gate_diagnostics(runs["matrix_wide"]["bundle"], test_slice(runs["matrix_wide"]["bundle"])),
        "structured_wide": gate_diagnostics(runs["structured_wide"]["bundle"], test_slice(runs["structured_wide"]["bundle"])),
    }
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, (run_id, label) in zip(axs, [("matrix_wide", "Matrix wide"), ("structured_wide", "Structured wide")]):
        rows = gate_map[run_id]["rows"]
        labels = [row["bin"] for row in rows]
        x = np.arange(len(labels))
        current = [row["deviation_mean"] for row in rows]
        gated = [row["gated_deviation_mean"] for row in rows]
        ax.bar(x - 0.18, current, width=0.36, color="#F28E2B", label="current mean deviation")
        ax.bar(x + 0.18, gated, width=0.36, color="#59A14F", label="gated mean deviation")
        ax.set_title(f"{label}: multiplier deviation vs raw tracking")
        ax.set_xlabel(r"max |tracking_error_raw| bin")
        ax.set_ylabel("Mean model deviation ratio")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "wide_range_authority_gate.png", dpi=220)
    plt.close(fig)
    return gate_map


def write_markdown(
    rows: list[dict],
    nominal_A_radius: float,
    reference_bundle: dict,
    matrix_frontier_rows: list[dict],
    structured_frontier_rows: list[dict],
    gate_map: dict[str, dict],
    nominal_A_radius_dist: float,
    distillation_summary: dict[str, dict],
    distillation_matrix_frontier_rows: list[dict],
    distillation_structured_frontier_rows: list[dict],
) -> None:
    summary = {row["run_id"]: row for row in rows}
    matrix_legacy = summary["matrix_legacy"]
    matrix_refresh = summary["matrix_refresh"]
    matrix_wide = summary["matrix_wide"]
    structured_legacy = summary["structured_legacy"]
    structured_refresh = summary["structured_refresh"]
    structured_wide = summary["structured_wide"]
    reid_refresh = summary["reid_refresh"]
    baseline = summary["baseline"]
    reward_geom = reward_geometry_table(reference_bundle)
    reward_targets = reward_balance_targets(reference_bundle)
    alpha_max_stable = 1.0 / nominal_A_radius
    alpha_max_stable_dist = 1.0 / nominal_A_radius_dist
    matrix_frontier_selected = [min(matrix_frontier_rows, key=lambda row: abs(row["alpha"] - alpha)) for alpha in [0.85, 1.00, 1.20]]
    distillation_matrix_frontier_selected = [min(distillation_matrix_frontier_rows, key=lambda row: abs(row["alpha"] - alpha)) for alpha in [0.95, 1.00, 1.20]]

    lines: list[str] = [
        "# Polymer Wide-Range Matrix and Structured Report With Distillation Counterpart",
        "",
        "Date: 2026-04-21",
        "",
        "This report focuses on the latest widened-range polymer matrix and structured-matrix runs, then extends the same admissibility and design logic to the distillation column as a cross-system counterpart.",
        "",
        "The goal is to answer five questions with data from the saved runs and the shared polymer model:",
        "",
        "1. Why did the widened matrix/structured methods improve, and why did reidentification still fail?",
        "2. What reward is actually used in polymer, and how should it be changed if both outputs should matter more evenly?",
        "3. Is there a mathematical way to set the multiplier range, instead of widening blindly?",
        "4. Why do the wide runs first degrade and then recover, and how can residual-style ideas help?",
        "5. Is polymer reidentification still worth pursuing, or is it currently dominated by direct multiplier methods?",
        "",
        "## Run Set",
        "",
    ]

    config_rows = []
    for run_id in ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide", "structured_legacy", "structured_refresh", "structured_wide", "reid_refresh"]:
        row = summary[run_id]
        config_rows.append(
            [
                row["label"],
                f"`{row['run_path']}`",
                row["run_mode"],
                row["state_mode"],
                row["observer_update_alignment"],
                row["base_state_norm_mode"],
                row["mismatch_feature_transform_mode"],
                row["range_profile"],
                row["update_family"],
                "yes" if row["disturbance_consistent"] else "no",
            ]
        )
    lines += markdown_table(
        [
            "Run",
            "Saved bundle",
            "run_mode",
            "state_mode",
            "observer",
            "base_state_norm",
            "mismatch_transform",
            "range_profile",
            "update_family",
            "disturbance consistent",
        ],
        config_rows,
    )
    lines += [
        "",
        "All compared runs use the same polymer disturbance schedule. The saved `qi`, `qs`, and `ha` arrays in the result bundles are numerically identical to the baseline schedule, so the widened-range comparison is not confounded by different disturbances.",
        "",
        "## Main Comparison",
        "",
    ]

    perf_rows = []
    for run_id in ["baseline", "matrix_legacy", "matrix_refresh", "matrix_wide", "structured_legacy", "structured_refresh", "structured_wide", "reid_refresh"]:
        row = summary[run_id]
        perf_rows.append(
            [
                row["label"],
                fmt(row["tail_phys_mae_mean"]),
                fmt(row["test_phys_mae_mean"]),
                fmt(row["test_phys_mae_out1"]),
                fmt(row["test_phys_mae_out2"]),
                fmt(row["reward_test"]),
                fmt(row["reward_final10_mean"]),
                fmt(row["test_u_move"]),
            ]
        )
    lines += markdown_table(
        [
            "Run",
            "Tail phys MAE",
            "Final test phys MAE",
            "Final test out1 MAE",
            "Final test out2 MAE",
            "Final test reward",
            "Final-10 reward",
            "Final test input move",
        ],
        perf_rows,
    )
    lines += [
        "",
        f"- Matrix wide is the strongest run in this study. Final test physical MAE drops from `{fmt(matrix_refresh['test_phys_mae_mean'])}` in the previous narrow refresh to `{fmt(matrix_wide['test_phys_mae_mean'])}` ({pct_change(matrix_wide['test_phys_mae_mean'], matrix_refresh['test_phys_mae_mean'])}), beating both the legacy matrix run `{fmt(matrix_legacy['test_phys_mae_mean'])}` and baseline MPC `{fmt(baseline['test_phys_mae_mean'])}`.",
        f"- Structured wide is mixed. Its overall late-run tail MAE improves from `{fmt(structured_refresh['tail_phys_mae_mean'])}` to `{fmt(structured_wide['tail_phys_mae_mean'])}`, but the held-out final test gets worse: `{fmt(structured_refresh['test_phys_mae_mean'])}` to `{fmt(structured_wide['test_phys_mae_mean'])}` ({pct_change(structured_wide['test_phys_mae_mean'], structured_refresh['test_phys_mae_mean'])}).",
        f"- Reidentification still fails to convert the richer RL state into better control. Its final test MAE remains `{fmt(reid_refresh['test_phys_mae_mean'])}`, worse than matrix wide and worse than baseline MPC.",
        "",
    ]
    lines += image_lines("wide_range_reward_curves.png", "Reward curves for matrix and structured narrow versus wide runs")
    lines += [
        "The reward curves show the same pattern in both wide runs: a severe early deterioration followed by recovery to a better late-stage reward than the narrow runs. The matrix wide trough is deeper (`{}` at episode `{}`) than the structured wide trough (`{}` at episode `{}`), but both recover only much later, around episodes `{}` and `{}` respectively.".format(
            fmt(matrix_wide["reward_trough"]),
            matrix_wide["reward_trough_episode"],
            fmt(structured_wide["reward_trough"]),
            structured_wide["reward_trough_episode"],
            int(matrix_wide["reward_recover_episode"]),
            int(structured_wide["reward_recover_episode"]),
        ),
        "",
        "## Final Test Episode",
        "",
    ]
    lines += image_lines("wide_range_final_test_outputs.png", "Final held-out test episode outputs for matrix and structured families")
    lines += [
        f"Matrix wide improves both outputs in the held-out test episode: output 1 MAE falls from `{fmt(matrix_refresh['test_phys_mae_out1'])}` to `{fmt(matrix_wide['test_phys_mae_out1'])}`, and output 2 MAE falls from `{fmt(matrix_refresh['test_phys_mae_out2'])}` to `{fmt(matrix_wide['test_phys_mae_out2'])}`.",
        f"Structured wide does not. Output 1 improves from `{fmt(structured_refresh['test_phys_mae_out1'])}` to `{fmt(structured_wide['test_phys_mae_out1'])}`, but output 2 gets substantially worse: `{fmt(structured_refresh['test_phys_mae_out2'])}` to `{fmt(structured_wide['test_phys_mae_out2'])}`. This is why the final test mean degrades even though the overall late-run tail and reward look better.",
        "",
        "## Why Matrix Wide Worked And Reidentification Did Not",
        "",
        "The latest matrix result is successful because the controller is allowed to choose from a small, direct, always-realized model family. There is no identification gate between the action and the prediction model used by MPC. The action changes three smooth multipliers and the model correction is immediately available to the optimizer on every step.",
        "",
        f"In the final test episode, matrix wide mainly uses a damped `A` correction and a stronger first-input `B` correction: `alpha` mean is `{fmt(matrix_wide['alpha_mean_test'])}`, while `delta1` mean is `{fmt(matrix_wide['delta1_mean_test'])}` and its p95 reaches `{fmt(matrix_wide['delta1_p95_test'])}`. That is a coherent low-dimensional correction, not a noisy online identification problem.",
        "",
        f"Reidentification is fundamentally different. The policy may request strong blend authority, but the online identification engine almost never produces an admissible new model. In the latest polymer reidentification run, candidate-valid fraction is only `{fmt(reid_refresh['candidate_valid_frac'])}` and update-success fraction is only `{fmt(reid_refresh['update_success_frac'])}`. Median condition number is `{fmt(reid_refresh['condition_median'], digits=1)}` and p95 is `{fmt(reid_refresh['condition_p95'], digits=1)}`. So reidentification is bottlenecked by data informativity and numerical conditioning, not by the policy alone.",
        "",
        "This difference matches the literature. The direct multiplier methods behave like bounded parametric adaptation inside a fixed low-dimensional family. Reidentification, by contrast, needs persistently informative data and a numerically healthy regression problem. The adaptive MPC and identification literature repeatedly stresses persistence of excitation, targeted excitation, and dual control/experiment design as prerequisites for successful online model maintenance [Berberich2022] [Heirung2015] [Heirung2017] [Oshima2024].",
        "",
        "## The Polymer Reward Actually Used",
        "",
        "The RL notebooks do not use a plain `Q = [5, 1]` quadratic reward. They use the shared relative-band reward from `utils/rewards.py` with setpoint-dependent bands, inside-band gating, linear edge penalties, and an inside-band bonus.",
        "",
        "$$",
        "r_t = \\Big(-\\sum_i e^{\\mathrm{eff}}_{i,t} - \\sum_i \\ell_{i,t} - \\sum_j m_{j,t} + \\sum_i b_{i,t}\\Big) \\cdot \\texttt{reward\\_scale}",
        "$$",
        "",
        "where the output balance depends on the scaled band",
        "",
        "$$",
        "\\text{band}_{i,t} = \\frac{\\max(k_{\\mathrm{rel},i}|y^{sp}_{i,t}|,\\; \\text{band\\_floor}_{i})}{y^{max}_i - y^{min}_i},",
        "\\qquad",
        "\\text{slope at edge}_i = 2 Q_i \\cdot \\text{band}_{i,t}.",
        "$$",
        "",
        "So the relevant output balance is not just the raw `Q_diag`. It is `Q_diag` filtered through the setpoint-dependent band width.",
        "",
    ]

    lines += markdown_table(
        [
            "Setpoint",
            "Output",
            "Physical setpoint",
            "Band (phys)",
            "Band (scaled)",
            "Edge slope",
            "Bonus prefactor",
        ],
        [
            [
                row["setpoint"],
                row["output"],
                fmt(row["y_sp_phys"], digits=3),
                fmt(row["band_phys"], digits=4),
                fmt(row["band_scaled"], digits=5),
                fmt(row["edge_slope"], digits=4),
                fmt(row["bonus_prefactor"], digits=4),
            ]
            for row in reward_geom
        ],
    )
    lines += image_lines("wide_range_reward_balance.png", "Reward geometry and implied equalization targets")
    lines += [
        f"If `Q2` stays at `90`, then an edge-equalized `Q1` would be about `{fmt(reward_targets['q1_equal_edge_mean'], digits=1)}`, while a bonus-equalized `Q1` would be about `{fmt(reward_targets['q1_equal_bonus_mean'], digits=1)}`. The current `Q1 = {fmt(reward_targets['current_q'][0], digits=1)}` is therefore almost exactly the bonus-equalized value, not the edge-equalized value.",
        "That is the rigorous explanation for the \"even output\" issue. Inside the reward band, the outputs are treated much more evenly than the old report implied. But outside the band, output 1 still carries a much steeper correction slope, so the policy can gain reward by helping output 1 more aggressively even when output 2 gets worse.",
        "",
        "A practical reward fix depends on what you want to equalize:",
        "",
        f"- Equal band-edge urgency: move `Q1` toward about `{fmt(reward_targets['q1_equal_edge_mean'], digits=0)}` while keeping `Q2 = 90`.",
        f"- Keep the current inside-band bonus balance: leave `Q1` near about `{fmt(reward_targets['q1_equal_bonus_mean'], digits=0)}` and increase output-2 edge penalties separately.",
        "- Clean implementation: separate inside-band bonus weights from outside-band penalty weights instead of forcing one `Q_diag` to do both jobs.",
        "",
        "The saved runs already show this reward tradeoff in the final test episode:",
        "",
    ]

    lines += markdown_table(
        [
            "Run",
            "Final test scaled MAE out1",
            "Final test scaled MAE out2",
            "Reward quad out1",
            "Reward quad out2",
            "Reward linear out1",
            "Reward linear out2",
            "Reward bonus out1",
            "Reward bonus out2",
            "Final test input move",
            "Final test reward",
        ],
        [
            [
                row["label"],
                fmt(row["test_scaled_mae_out1"]),
                fmt(row["test_scaled_mae_out2"]),
                fmt(row["reward_quad_out1"]),
                fmt(row["reward_quad_out2"]),
                fmt(row["reward_linear_out1"]),
                fmt(row["reward_linear_out2"]),
                fmt(row["reward_bonus_out1"]),
                fmt(row["reward_bonus_out2"]),
                fmt(row["test_u_move"]),
                fmt(row["reward_test"]),
            ]
            for row in [matrix_refresh, matrix_wide, structured_refresh, structured_wide]
        ],
    )
    lines += image_lines("wide_range_reward_tradeoff.png", "Reward versus evaluation tradeoff for the wide matrix and structured runs")
    lines += [
        f"Structured wide improves output 1 sharply: final test scaled MAE drops from `{fmt(structured_refresh['test_scaled_mae_out1'])}` to `{fmt(structured_wide['test_scaled_mae_out1'])}`. But output 2 worsens from `{fmt(structured_refresh['test_scaled_mae_out2'])}` to `{fmt(structured_wide['test_scaled_mae_out2'])}`. Under the actual polymer reward, the output-1 quadratic and linear penalties both fall materially, while the output-2 penalties rise by less. The reward therefore improves even though the held-out mean physical error gets worse.",
        "",
        "## Mathematical Limits For The Multiplier Range",
        "",
        "For the plain matrix family, observability and controllability rank are not what limit the range. If the physical model is changed only through `A -> alpha A`, then",
        "",
        "$$",
        "\\mathcal{O}(\\alpha A, C) = \\begin{bmatrix} C \\\\ \\alpha C A \\\\ \\alpha^2 C A^2 \\\\ \\vdots \\end{bmatrix},",
        "\\qquad",
        "\\mathcal{C}(\\alpha A, B) = \\begin{bmatrix} B & \\alpha A B & \\alpha^2 A^2 B & \\cdots \\end{bmatrix}.",
        "$$",
        "",
        "For any `alpha > 0`, these are just row-wise or column-wise scalings of the nominal observability and controllability matrices, so the rank stays the same. That is exactly what the polymer model shows numerically:",
        "",
    ]
    lines += markdown_table(
        [
            "alpha",
            "rho(alpha A)",
            "obs rank",
            "ctrb rank",
            "obs cond",
            "ctrb cond",
        ],
        [
            [
                fmt(row["alpha"], digits=3),
                fmt(row["spectral_radius"], digits=4),
                str(row["observability_rank"]),
                str(row["controllability_rank"]),
                fmt(row["observability_cond"], digits=1),
                fmt(row["controllability_cond"], digits=1),
            ]
            for row in matrix_frontier_selected
        ],
    )
    lines += image_lines("wide_range_admissibility_frontier.png", "Matrix and structured admissibility frontier")
    lines += [
        f"The useful matrix bound comes instead from open-loop spectral admissibility. Since `rho(A_nom) = {fmt(nominal_A_radius)}`, requiring `rho(alpha A) < 1` gives `alpha < 1 / rho(A_nom) = {fmt(alpha_max_stable)}`. That is the clean analytical upper bound if every candidate prediction `A` must remain open-loop stable.",
        "The lower bound is different. For positive `alpha`, neither observability nor open-loop stability imposes a nontrivial lower bound. So there is no comparable analytical `alpha_min > 0` from these criteria alone. The practical lower bound comes from how much model-speed distortion you are willing to tolerate, not from rank loss.",
        "",
        "There are two practical lower-bound rules that do make sense:",
        "",
        "1. Trust-region rule: require the relative model change to stay small,",
        "",
        "$$",
        "\\frac{\\|\\alpha A - A\\|_F}{\\|A\\|_F} = |\\alpha - 1| \\le \\varepsilon_A",
        "\\quad \\Rightarrow \\quad",
        "\\alpha \\in [1-\\varepsilon_A,\\; 1+\\varepsilon_A].",
        "$$",
        "",
        "2. Time-scale rule: require the dominant spectral radius to stay above a chosen fraction of nominal,",
        "",
        "$$",
        "\\rho(\\alpha A) \\ge \\kappa \\rho(A) \\quad \\Rightarrow \\quad \\alpha \\ge \\kappa,",
        "$$",
        "",
        "where `kappa` is a modeling-choice floor such as `0.85` or `0.90`. This is not a stability necessity. It is a way to stop the learned prediction model from becoming unrealistically fast.",
        "",
        "For the structured block family, entrywise multipliers can in principle change the rank properties, so a sampled frontier is useful. On the polymer model, however, the sampled positive ranges still keep full observability rank. The frontier is again instability, not rank loss:",
        "",
    ]
    lines += markdown_table(
        [
            "Upper bound",
            "Unstable frac",
            "Near-unit frac",
            "Spectral p95",
            "Min obs rank",
            "Bad obs frac",
        ],
        [
            [
                fmt(row["upper_bound"], digits=2),
                fmt(row["unstable_frac"], digits=3),
                fmt(row["near_unit_frac"], digits=3),
                fmt(row["spectral_p95"], digits=4),
                str(row["observability_rank_min"]),
                fmt(row["bad_observability_frac"], digits=3),
            ]
            for row in structured_frontier_rows
        ],
    )
    lines += [
        "This gives a practical polymer rule. If you want a mostly stable structured family without frequent unstable prediction models, the common upper bound should stay near about `1.05`. By `1.08`, instability already appears. By `1.20`, roughly half the sampled block models are unstable.",
        "",
        "## What About The B Multipliers?",
        "",
        "The `B` side is different from `A`. In the matrix family the model update is `B(\\delta) = B \\operatorname{diag}(\\delta_1, \\delta_2)`. For any strictly positive `delta_j`, controllability rank is preserved for exactly the same reason as above: the controllability matrix is just column-scaled block by block. So, again, rank does not give a useful finite bound.",
        "",
        "What `delta_j` changes is the perceived input authority. If `G_ss = C(I-A)^{-1}B` is the steady-state gain, then",
        "",
        "$$",
        "G_{ss}(\\delta) = G_{ss}\\,\\operatorname{diag}(\\delta_1, \\delta_2).",
        "$$",
        "",
        "So the predicted output gain of input `j` scales linearly with `delta_j`, while the input move required to get the same correction scales approximately like `1 / delta_j`.",
        "",
    ]
    lines += image_lines("wide_range_b_multiplier_design.png", "B-multiplier gain and move-ratio explanation")
    lines += [
        "That means `B` selection should be done with actuator and trust considerations, not with pole placement logic. A practical lower-bound calculation is",
        "",
        "$$",
        "\\delta_{j,\\min} \\ge \\frac{\\Delta u^{nom}_{j,\\;p}}{h^{avail}_{j,\\;min}},",
        "$$",
        "",
        "where `Delta u^{nom}_{j,p}` is a representative nominal move level such as the 95th percentile under baseline MPC, and `h^{avail}_{j,min}` is the minimum available headroom for that input over the scenarios you care about. If the learned model is allowed to believe the input is too weak, the MPC will ask for moves that may not fit inside the real actuator margin.",
        "",
        "A practical upper-bound calculation is a gain-trust region in log space,",
        "",
        "$$",
        "|\\log \\delta_j| \\le \\varepsilon_B",
        "\\quad \\Longleftrightarrow \\quad",
        "\\delta_j \\in [e^{-\\varepsilon_B}, e^{\\varepsilon_B}],",
        "$$",
        "",
        "which is often a better way to think about input-gain uncertainty than a raw linear bound because equal multiplicative uncertainty is treated symmetrically above and below `1.0`.",
        "",
        "That gives a practical design rule for `B` multipliers:",
        "",
        "- Lower bound: choose `delta_min` so the required move inflation `1 / delta_min` still fits inside actuator headroom on representative disturbances.",
        "- Upper bound: choose `delta_max` so the predicted input gain increase stays inside the uncertainty set you are willing to trust, or inside a symmetric trust region such as `|log(delta_j)| <= eps_B`.",
        "",
    ]
    lines += markdown_table(
        [
            "delta",
            "Predicted gain ratio",
            "Required move ratio",
            "Interpretation",
        ],
        [
            ["0.75", "0.75x", fmt(1.0 / 0.75, digits=3) + "x", "aggressive low-gain assumption; large move inflation"],
            ["0.85", "0.85x", fmt(1.0 / 0.85, digits=3) + "x", "moderate low-gain assumption"],
            ["0.95", "0.95x", fmt(1.0 / 0.95, digits=3) + "x", "conservative low-gain assumption"],
            ["1.05", "1.05x", fmt(1.0 / 1.05, digits=3) + "x", "conservative high-gain assumption"],
            ["1.25", "1.25x", fmt(1.0 / 1.25, digits=3) + "x", "aggressive high-gain assumption"],
        ],
    )
    lines += [
        "",
        f"With the widened defaults requested in this update, both polymer and distillation now use `delta \\in [0.75, 1.25]` on the `B` side. That means up to about `{fmt(1.0/0.75, digits=3)}x` required-move inflation on the low side and up to `1.25x` predicted input gain on the high side. The systems differ only on the `A` side, where polymer is capped at `alpha <= {fmt(alpha_max_stable)}` and distillation at `alpha <= {fmt(alpha_max_stable_dist)}`.",
        "",
        "That is why the safest widening order is still `B` before `A`: `B` does not move the open-loop poles, but it does change how hard the MPC will push the actuators. So `B` should be bounded by input-headroom and validation logic, not by spectral radius.",
        "",
        "## Distillation Counterpart",
        "",
        "The same mathematics does not transfer one-for-one across systems. Distillation is much less fragile than polymer at the model level.",
        "",
    ]
    lines += image_lines("wide_range_cross_system_admissibility.png", "Cross-system admissibility comparison")
    lines += markdown_table(
        [
            "System",
            "rho(A_nom)",
            "alpha_max stable",
            "Structured unstable frac at 1.20",
            "Structured p95 spectral at 1.20",
        ],
        [
            [
                "Polymer",
                fmt(nominal_A_radius),
                fmt(alpha_max_stable),
                fmt(structured_frontier_rows[-1]["unstable_frac"], digits=3),
                fmt(structured_frontier_rows[-1]["spectral_p95"], digits=4),
            ],
            [
                "Distillation",
                fmt(nominal_A_radius_dist),
                fmt(alpha_max_stable_dist),
                fmt(distillation_structured_frontier_rows[-1]["unstable_frac"], digits=3),
                fmt(distillation_structured_frontier_rows[-1]["spectral_p95"], digits=4),
            ],
        ],
    )
    lines += [
        f"Polymer has `alpha_max_stable = {fmt(alpha_max_stable)}`, while distillation has `alpha_max_stable = {fmt(alpha_max_stable_dist)}`. At the same structured upper bound `1.20`, polymer's sampled unstable fraction is `{fmt(structured_frontier_rows[-1]['unstable_frac'], digits=3)}`, but distillation's is only `{fmt(distillation_structured_frontier_rows[-1]['unstable_frac'], digits=3)}`. So distillation is mathematically much more tolerant of multiplier widening than polymer.",
        "",
        "However, the currently saved disturbance runs do not yet show that widening alone solves distillation performance. The latest disturbance bundles available in this tree are baseline fluctuation, matrix SAC disturbance, and structured SAC disturbance:",
        "",
    ]
    lines += markdown_table(
        [
            "Run",
            "Final test phys MAE",
            "Final test out1 MAE",
            "Final test out2 MAE",
            "Final test reward",
            "Alpha / spectral usage",
        ],
        [
            [
                distillation_summary["baseline"]["label"],
                fmt(distillation_summary["baseline"]["test_phys_mae_mean"]),
                fmt(distillation_summary["baseline"]["test_phys_mae_out1"]),
                fmt(distillation_summary["baseline"]["test_phys_mae_out2"]),
                fmt(distillation_summary["baseline"]["reward_test"]),
                "n/a",
            ],
            [
                distillation_summary["matrix"]["label"],
                fmt(distillation_summary["matrix"]["test_phys_mae_mean"]),
                fmt(distillation_summary["matrix"]["test_phys_mae_out1"]),
                fmt(distillation_summary["matrix"]["test_phys_mae_out2"]),
                fmt(distillation_summary["matrix"]["reward_test"]),
                f"alpha mean {fmt(distillation_summary['matrix']['alpha_mean_test'])}, p95 {fmt(distillation_summary['matrix']['alpha_p95_test'])}",
            ],
            [
                distillation_summary["structured"]["label"],
                fmt(distillation_summary["structured"]["test_phys_mae_mean"]),
                fmt(distillation_summary["structured"]["test_phys_mae_out1"]),
                fmt(distillation_summary["structured"]["test_phys_mae_out2"]),
                fmt(distillation_summary["structured"]["reward_test"]),
                f"spectral mean {fmt(distillation_summary['structured']['spectral_mean_test'])}, p95 {fmt(distillation_summary['structured']['spectral_p95_test'])}",
            ],
        ],
    )
    lines += image_lines("wide_range_distillation_counterpart.png", "Distillation final-test counterpart")
    lines += [
        f"The current saved disturbance matrix run does not beat baseline (`{fmt(distillation_summary['matrix']['test_phys_mae_mean'])}` vs `{fmt(distillation_summary['baseline']['test_phys_mae_mean'])}`), and the current saved structured disturbance run is worse still (`{fmt(distillation_summary['structured']['test_phys_mae_mean'])}`). So the distillation section changes the conclusion in an important way: the model-level admissibility landscape is wider, but the currently saved RL policies are not exploiting it well.",
        "",
        "That is exactly why the cross-system figure matters. Polymer needs guards because the model family is fragile. Distillation does not need those guards for the same mathematical reason, but it still needs better policy learning and reward alignment before wider ranges will automatically help.",
        "",
        "## Why The Wide Runs First Degrade And Then Recover",
        "",
        "The early deterioration is consistent with a larger action/model-search space. Widening the multiplier ranges expands the set of prediction models the agent can induce. Early in training, the replay buffer is dominated by low-quality exploratory transitions from this larger space, so the policy gets worse before it learns which stronger corrections are actually useful. Once enough informative transitions accumulate, the policy recovers and starts exploiting the wider authority.",
        "",
        "The control literature and RL literature both suggest practical fixes:",
        "",
        "- Progressive widening / continuation: instead of jumping directly from narrow to wide bounds, increase the bounds in stages once reward, solver success, and held-out tracking have stabilized [Seo2025].",
        "- Smoother exploration: use temporally coherent exploration rather than step-to-step jagged action noise [Korenkevych2019].",
        "- Data-aware excitation only when needed: if model learning is part of the method, add excitation only when uncertainty is high or the data are not informative enough [Heirung2015] [Heirung2017] [Oshima2024].",
        "- Robust uncertainty-set shaping: treat the multiplier family as an uncertainty set and grow that set only while feasibility and worst-case behavior remain acceptable [Chen2024] [Limon2013] [Kothare1996].",
        "",
    ]
    lines += image_lines("wide_range_practical_fixes_explainer.png", "Literature-backed practical fixes explained")
    lines += [
        "The four panels in the figure are not copied from the papers. They are explanatory plots built from the mechanisms those papers discuss, translated into this project's setting.",
        "",
        "What each paper-backed idea means here in concrete terms:",
        "",
        "- Progressive widening / continuation: widen the allowed multiplier range in phases. In this repo, that means promoting `high_coef` and the structured range profile only after held-out MAE, fallback rate, and p95 multiplier saturation are acceptable. This directly addresses the early degradation seen in the wide polymer reward curves.",
        "- Smoother exploration: replace jagged per-step multiplier noise with temporally coherent exploration, so the replay buffer contains locally consistent trajectories rather than violent model jumps. In practical terms, use an autoregressive action-noise process or a slower parameter-noise refresh for matrix and structured agents.",
        "- Data-aware excitation only when needed: make extra exploration or model-learning authority a function of uncertainty or poor informativeness. In this repo, that can be the same mismatch-based gate used for residual-style authority, but applied to exploration amplitude or reidentification authority instead of directly to `u_res`.",
        "- Robust uncertainty-set shaping: use the admissibility frontier to decide whether a candidate range should even be trainable. The polymer structured frontier shows why a uniform `1.20` upper bound is too wide as a default uncertainty set, while the distillation frontier shows that the same number is not equally dangerous there.",
        "",
        "A practical implementation in this repository would be:",
        "",
        "1. Start from the current narrow run.",
        "2. Continue training with an intermediate range first, not with the full wide range.",
        "3. Use a smoothed exploration process for the multiplier action.",
        "4. Gate extra exploration or multiplier authority by mismatch magnitude when the trajectory is already near setpoint.",
        "5. Only promote to the next wider range if held-out MAE improves and p95 multiplier use is not already saturating.",
        "6. Stop widening when the held-out episode stops improving or the sampled spectral statistics cross the chosen admissibility limit.",
        "",
        "## Residual-Style Gating For Matrix And Structured Multipliers",
        "",
        "The saved wide runs show another practical issue. The policy keeps the model far from nominal even when the band-normalized raw tracking error is already small. That is exactly the situation where the residual method's deadband idea can help.",
        "",
        "Use the same raw mismatch feature already logged in mismatch mode:",
        "",
        "$$",
        "\\tau_t = \\max_i |\\mathrm{tracking\\_error\\_raw}_{i,t}|.",
        "$$",
        "",
        "Because `tracking_error_raw` is already normalized by the tracking scale, `tau_t \\le 1` means the outputs are inside the reward band. Then apply a residual-style gate to the multiplier deviation:",
        "",
        "$$",
        "g_t = \\begin{cases}",
        "0, & \\tau_t \\le 1, \\\\",
        "1 - \\exp(-k(\\tau_t - 1)), & \\tau_t > 1,",
        "\\end{cases}",
        "\\qquad",
        "m^{eff}_t = 1 + g_t (m^{rl}_t - 1).",
        "$$",
        "",
        "The same equation applies to structured multipliers `theta_t`. This keeps the policy free to make large corrections when tracking is bad, but collapses back toward the nominal model near setpoint.",
        "",
    ]
    lines += image_lines("wide_range_authority_gate.png", "Residual-style gate effect on multiplier deviation")
    gate_rows = []
    for label, diag in [("Matrix wide", gate_map["matrix_wide"]), ("Structured wide", gate_map["structured_wide"])]:
        for row in diag["rows"]:
            gate_rows.append(
                [
                    label,
                    row["bin"],
                    str(row["count"]),
                    fmt(row["deviation_mean"]),
                    fmt(row["gated_deviation_mean"]),
                ]
            )
    lines += markdown_table(
        [
            "Run",
            "Tracking bin",
            "Count",
            "Current mean deviation",
            "Gated mean deviation",
        ],
        gate_rows,
    )
    lines += [
        f"In the current wide matrix run, the mean deviation in the low-tracking bin is still about `{fmt(matrix_wide['deviation_mean_low_tracking_test'])}`, and `{fmt(100.0 * matrix_wide['deviation_gt_0p1_frac_low_tracking_test'], digits=1)}%` of low-tracking test steps still deviate from nominal by more than `0.1`. The structured-wide run is even more aggressive. The gate figure shows that a deadband plus exponential gate would cut that low-tracking authority sharply without changing the high-tracking regime nearly as much.",
        "",
        "This is the cleanest way to borrow the residual idea here. The same mismatch signal and the same authority logic can be reused, but the action being gated is the model deviation instead of a residual control move.",
        "",
        "## How Far Can The Range Be Widened Safely?",
        "",
        f"The nominal polymer physical `A` has spectral radius `{fmt(nominal_A_radius)}`. In the final test episode, matrix wide already reaches a derived p95 spectral radius of `{fmt(matrix_wide['derived_spectral_p95_test'])}` and max `{fmt(matrix_wide['derived_spectral_max_test'])}`. Structured wide is more aggressive still: mean spectral radius is `{fmt(structured_wide['spectral_mean_test'])}`, p95 is `{fmt(structured_wide['spectral_p95_test'])}`, max is `{fmt(structured_wide['spectral_max_test'])}`, and several structured multipliers hit the hard upper bound `1.2` at p95.",
        "",
        "So the answer is different for the two methods:",
        "",
        f"- Matrix: a slightly wider range may still be worth testing, but not symmetrically. The successful wide matrix policy mainly uses stronger `B` correction and slightly smaller `A`, so the safer next ablation is to widen `B` more than `A`, not to raise every bound uniformly.",
        f"- Structured: the current wide run already looks too aggressive for held-out evaluation. It pushes multiple grouped multipliers to `1.2`, keeps the test spectral radius above `1.0` on average, and doubles the final-test input movement from `{fmt(structured_refresh['test_u_move'])}` to `{fmt(structured_wide['test_u_move'])}`. Widening further before adding guards is not justified by these results.",
        "",
        "With a state-space model in the loop, the practical limit is not a single scalar bound. It is the largest uncertainty set for which the prediction model remains numerically admissible for MPC: stabilizable/detectable enough for the observer-controller pair, solver-feasible, and not so aggressive that held-out performance collapses. Robust MPC literature frames this as keeping the model family inside an uncertainty set where feasibility and robust performance can still be guaranteed or approximated [Chen2024] [Limon2013] [Kothare1996].",
        "",
        "For this project, a practical safe-widening recipe is:",
        "",
        "1. Keep the current solve fallback for structured wide.",
        "2. Add an explicit spectral-radius cap or smooth fade-back to nominal when the prediction model gets too aggressive.",
        "3. Widen bounds asymmetrically, favoring `B` before `A`.",
        "4. Use a staged widening schedule tied to held-out test MAE and solver/fallback statistics.",
        "5. Treat the empirical limit as reached when p95 multiplier use is already on the hard bound and held-out test performance no longer improves.",
        "",
        "## Model-Usage Diagnostics",
        "",
    ]
    lines += image_lines("wide_range_model_usage.png", "Wide-range multiplier usage and spectral-radius diagnostics")
    lines += [
        f"Matrix wide uses the expanded range in a targeted way. It does not simply saturate everything upward. Instead, it tends to push the first input gain higher while keeping `A` on average below nominal. Structured wide behaves differently: several grouped multipliers have p95 equal to the hard upper bound `1.2`, and the test spectral radius stays above `1.1` on average. That explains why structured wide can still improve reward while giving an over-aggressive held-out trajectory.",
        "",
        "## Is Reidentification Useless Now?",
        "",
        "For the current polymer implementation, the honest answer is: reidentification is currently dominated, not theoretically useless. The widened direct-multiplier methods are clearly more effective today because they avoid the identification bottleneck and exploit a low-dimensional model family that the MPC can use immediately.",
        "",
        f"On the evidence in these runs, the research priority should be matrix/structured continuation before polymer reidentification. Matrix wide reaches final-test MAE `{fmt(matrix_wide['test_phys_mae_mean'])}` while reidentification stays at `{fmt(reid_refresh['test_phys_mae_mean'])}` with candidate-valid fraction `{fmt(reid_refresh['candidate_valid_frac'])}`. Until the identification windowing, excitation, and candidate validation are redesigned, polymer reidentification is not competitive with direct multiplier supervision.",
        "",
        "That said, the literature does not support calling reidentification useless in general. It says the opposite: reidentification can work when the controller deliberately creates informative data and validates updates carefully [Berberich2022] [Heirung2015] [Heirung2017] [Oshima2024]. In this project, that means reidentification should be treated as a separate adaptive-control problem, not as a drop-in replacement for multiplier tuning.",
        "",
        "## Conclusions",
        "",
        f"- The latest matrix wide run is genuinely more successful than the previous matrix runs. Its final test MAE `{fmt(matrix_wide['test_phys_mae_mean'])}` beats the previous narrow refresh `{fmt(matrix_refresh['test_phys_mae_mean'])}`, the legacy matrix run `{fmt(matrix_legacy['test_phys_mae_mean'])}`, and baseline MPC `{fmt(baseline['test_phys_mae_mean'])}`.",
        f"- Structured wide is not a clean success. It improves late-run reward and aggregate tail MAE, but its held-out final test MAE degrades from `{fmt(structured_refresh['test_phys_mae_mean'])}` to `{fmt(structured_wide['test_phys_mae_mean'])}` because it over-optimizes the first output relative to the evaluation metric and becomes much more aggressive on the prediction model and control effort.",
        f"- The actual polymer reward is bonus-balanced more than edge-balanced. If both outputs should matter more evenly in evaluation, the most direct reward fix is to reduce `Q1` toward about `{fmt(reward_targets['q1_equal_edge_mean'], digits=0)}` or to separate inside-band bonus weights from outside-band penalty weights.",
        f"- For the matrix family, observability and controllability do not bound positive `alpha`; the meaningful analytical upper bound is `alpha < {fmt(alpha_max_stable)}` if all candidate `A` matrices must stay open-loop stable. There is no comparable analytical lower bound, so the lower side should be chosen by a trust-region or time-scale rule. For `B`, the useful bounds are about gain-trust and actuator headroom, not spectral stability.",
        f"- Distillation is mathematically much less fragile than polymer: `alpha_max_stable` is `{fmt(alpha_max_stable_dist)}` instead of `{fmt(alpha_max_stable)}`, and the structured unstable fraction at `1.20` is only `{fmt(distillation_structured_frontier_rows[-1]['unstable_frac'], digits=3)}` instead of `{fmt(structured_frontier_rows[-1]['unstable_frac'], digits=3)}`. But the currently saved disturbance RL runs still do not beat the distillation baseline, so wider admissibility alone is not enough.",
        "- The degradation-then-recovery pattern is expected after abrupt widening. The most practical fix is staged widening, smoother exploration, mismatch-gated authority, and uncertainty-set shaping before wider training ranges are accepted.",
        "- Polymer reidentification is currently dominated by the widened direct-multiplier methods and should be deprioritized until the identification layer itself is redesigned around informative-window generation and candidate validation.",
        "",
        "## Sources",
        "",
        "- [Berberich2022] Forward-looking persistent excitation in model predictive control.",
        "- [Heirung2015] MPC-based dual control with online experiment design.",
        "- [Heirung2017] Dual adaptive model predictive control.",
        "- [Oshima2024] Targeted excitation and re-identification methods for multivariate process and model predictive control.",
        "- [Korenkevych2019] Autoregressive Policies for Continuous Control Deep Reinforcement Learning.",
        "- [Seo2025] Continuous Control with Coarse-to-fine Reinforcement Learning.",
        "- [Chen2024] Robust model predictive control with polytopic model uncertainty through System Level Synthesis.",
        "- [Limon2013] Robust feedback model predictive control of constrained uncertain systems.",
        "- [Kothare1996] Robust constrained model predictive control using linear matrix inequalities.",
        "",
        "[Berberich2022]: https://doi.org/10.1016/j.automatica.2021.110033",
        "[Heirung2015]: https://doi.org/10.1016/j.jprocont.2015.04.012",
        "[Heirung2017]: https://doi.org/10.1016/j.automatica.2017.01.030",
        "[Oshima2024]: https://doi.org/10.1016/j.jprocont.2024.103190",
        "[Korenkevych2019]: https://www.ijcai.org/proceedings/2019/0382.pdf",
        "[Seo2025]: https://proceedings.mlr.press/v270/seo25a.html",
        "[Chen2024]: https://doi.org/10.1016/j.automatica.2023.111431",
        "[Limon2013]: https://doi.org/10.1016/j.jprocont.2012.08.003",
        "[Kothare1996]: https://doi.org/10.1016/0005-1098(96)00063-5",
    ]

    (REPO_ROOT / "report" / "polymer_wide_range_matrix_structured_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    with POLYMER_SYS_DICT.open("rb") as fh:
        sys_dict = pickle.load(fh)
    A_phys = np.asarray(sys_dict["A"], float)
    B_phys = np.asarray(sys_dict["B"], float)
    C_phys = np.asarray(sys_dict["C"], float)
    nominal_A_radius = float(np.max(np.abs(np.linalg.eigvals(A_phys))))
    A_aug, B_aug, _ = augment_state_space(A_phys, B_phys, C_phys)

    with DISTILLATION_SYS_DICT.open("rb") as fh:
        dist_sys = pickle.load(fh)
    A_dist = np.asarray(dist_sys["A"], float)
    B_dist = np.asarray(dist_sys["B"], float)
    C_dist = np.asarray(dist_sys["C"], float)
    A_aug_dist, B_aug_dist, _ = augment_state_space(A_dist, B_dist, C_dist)
    nominal_A_radius_dist = float(np.max(np.abs(np.linalg.eigvals(A_dist))))

    runs: dict[str, dict] = {}
    for spec in RUN_SPECS:
        runs[spec.run_id] = {"spec": spec, "bundle": load_pickle(spec.path)}

    distillation_runs = {
        "baseline": {"bundle": load_pickle(DISTILLATION_BASELINE_PATH)},
        "matrix": {"bundle": load_pickle(DISTILLATION_MATRIX_DISTURB_PATH)},
        "structured": {"bundle": load_pickle(DISTILLATION_STRUCTURED_DISTURB_PATH)},
    }

    baseline_bundle = runs["baseline"]["bundle"]
    rows = [compute_summary(spec=spec, bundle=runs[spec.run_id]["bundle"], baseline_bundle=baseline_bundle, nominal_A_radius=nominal_A_radius) for spec in RUN_SPECS]
    summary_map = {row["run_id"]: row for row in rows}
    distillation_summary = {
        "baseline": compute_generic_run_summary("Distillation baseline fluctuation", distillation_runs["baseline"]["bundle"]),
        "matrix": compute_generic_run_summary("Distillation matrix SAC disturbance", distillation_runs["matrix"]["bundle"]),
        "structured": compute_generic_run_summary("Distillation structured SAC disturbance", distillation_runs["structured"]["bundle"]),
    }

    save_csv(
        DATA_ROOT / "wide_range_summary.csv",
        rows,
        [
            "run_id",
            "family",
            "variant",
            "label",
            "run_path",
            "run_mode",
            "state_mode",
            "observer_update_alignment",
            "base_state_norm_mode",
            "mismatch_feature_transform_mode",
            "range_profile",
            "update_family",
            "disturbance_consistent",
            "tail_phys_mae_mean",
            "test_phys_mae_mean",
            "test_phys_mae_out1",
            "test_phys_mae_out2",
            "test_scaled_mae_out1",
            "test_scaled_mae_out2",
            "reward_test",
            "reward_final10_mean",
            "reward_trough",
            "reward_trough_episode",
            "reward_recover_episode",
            "test_u_move",
            "weighted_quad_out1",
            "weighted_quad_out2",
            "alpha_mean_test",
            "alpha_p95_test",
            "alpha_max_test",
            "delta1_mean_test",
            "delta2_mean_test",
            "delta1_p95_test",
            "delta2_p95_test",
            "A_model_delta_ratio_mean_test",
            "B_model_delta_ratio_mean_test",
            "derived_spectral_mean_test",
            "derived_spectral_p95_test",
            "derived_spectral_max_test",
            "spectral_mean_test",
            "spectral_p95_test",
            "spectral_max_test",
            "near_bound_mean_test",
            "action_saturation_mean_test",
            "prediction_fallback_frac_test",
            "candidate_valid_frac",
            "update_success_frac",
            "condition_median",
            "condition_p95",
            "residual_ratio_median",
            "residual_ratio_p95",
        ],
    )

    plot_reward_curves(runs)
    plot_final_test_outputs(runs)
    plot_model_usage(runs, nominal_A_radius=nominal_A_radius)
    plot_tradeoff(summary_map)
    plot_reward_balance(baseline_bundle)
    matrix_frontier_rows, structured_frontier_rows = plot_admissibility_frontier(
        A=A_phys,
        B=B_phys,
        C=C_phys,
        A_aug=A_aug,
        B_aug=B_aug,
        n_outputs=C_phys.shape[0],
    )
    distillation_matrix_frontier_rows = matrix_admissibility_grid(A_dist, B_dist, C_dist, np.linspace(0.70, 1.25, 111))
    distillation_structured_frontier_rows = structured_frontier_grid(
        A_aug=A_aug_dist,
        B_aug=B_aug_dist,
        C=C_dist,
        n_outputs=C_dist.shape[0],
        upper_grid=np.array([1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20], float),
    )
    gate_map = plot_authority_gate(runs)
    plot_cross_system_admissibility(
        matrix_frontier_rows,
        structured_frontier_rows,
        distillation_matrix_frontier_rows,
        distillation_structured_frontier_rows,
        1.0 / nominal_A_radius,
        1.0 / nominal_A_radius_dist,
    )
    plot_b_multiplier_design()
    plot_practical_fixes_explainer(structured_frontier_rows, distillation_structured_frontier_rows)
    plot_distillation_counterpart(distillation_runs)
    write_markdown(
        rows,
        nominal_A_radius=nominal_A_radius,
        reference_bundle=baseline_bundle,
        matrix_frontier_rows=matrix_frontier_rows,
        structured_frontier_rows=structured_frontier_rows,
        gate_map=gate_map,
        nominal_A_radius_dist=nominal_A_radius_dist,
        distillation_summary=distillation_summary,
        distillation_matrix_frontier_rows=distillation_matrix_frontier_rows,
        distillation_structured_frontier_rows=distillation_structured_frontier_rows,
    )


if __name__ == "__main__":
    main()
