from __future__ import annotations

import csv
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Simulation.mpc import augment_state_space

FIG_DIR = REPO_ROOT / "report" / "figures" / "distillation_transfer_20260428"
POLYMER_DATA = REPO_ROOT / "Polymer" / "Data"
DISTILLATION_DATA = REPO_ROOT / "Distillation" / "Data"
POLYMER_SENS_DIR = REPO_ROOT / "Polymer" / "Results" / "offline_multiplier_sensitivity"
DISTILLATION_SENS_DIR = REPO_ROOT / "Distillation" / "Results" / "offline_multiplier_sensitivity"


def load_system(root: Path):
    with open(root / "system_dict.pickle", "rb") as handle:
        system_dict = pickle.load(handle)
    A = np.asarray(system_dict["A"], dtype=float)
    B = np.asarray(system_dict["B"], dtype=float)
    C = np.asarray(system_dict["C"], dtype=float)
    A_aug, B_aug, C_aug = augment_state_space(A, B, C)
    return A, B, C, A_aug, B_aug, C_aug


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def markov_stack(A: np.ndarray, B: np.ndarray, C: np.ndarray, horizon: int) -> np.ndarray:
    blocks = []
    A_power = np.eye(A.shape[0], dtype=float)
    for _ in range(int(horizon)):
        blocks.append(C @ A_power @ B)
        A_power = A_power @ A
    return np.vstack(blocks)


def horizon_sum(A: np.ndarray, B: np.ndarray, C: np.ndarray, horizon: int) -> np.ndarray:
    total = np.zeros((C.shape[0], B.shape[1]), dtype=float)
    A_power = np.eye(A.shape[0], dtype=float)
    for _ in range(int(horizon)):
        total += C @ A_power @ B
        A_power = A_power @ A
    return total


def latest_matching_dir(root: Path, prefix: str) -> Path:
    candidates = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No directories found under {root} with prefix {prefix!r}.")
    return candidates[-1]


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float_map(rows, key_field: str, value_field: str):
    return {str(row[key_field]): float(row[value_field]) for row in rows}


def save_model_metrics(poly_metrics: dict[str, float], dist_metrics: dict[str, float]):
    path = FIG_DIR / "distillation_transfer_model_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "polymer", "distillation"])
        for key in poly_metrics:
            writer.writerow([key, poly_metrics[key], dist_metrics[key]])
    return path


def plot_local_model_metrics(poly_metrics: dict[str, float], dist_metrics: dict[str, float]):
    labels = [
        r"$\rho(A_{\mathrm{phys}})$",
        r"$\|G_N\|_F$",
        "Horizon gain cond.",
        "Input-2 / input-1 authority",
    ]
    poly_values = [
        poly_metrics["rho_A_phys"],
        poly_metrics["markov_fro"],
        poly_metrics["horizon_sum_condition"],
        poly_metrics["input2_input1_authority_ratio"],
    ]
    dist_values = [
        dist_metrics["rho_A_phys"],
        dist_metrics["markov_fro"],
        dist_metrics["horizon_sum_condition"],
        dist_metrics["input2_input1_authority_ratio"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, poly_values, width, label="Polymer")
    ax.bar(x + width / 2, dist_values, width, label="Distillation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Control-Relevant Local Model Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(poly_values):
        ax.text(idx - width / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(dist_values):
        ax.text(idx + width / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = FIG_DIR / "distillation_transfer_local_model_metrics.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_horizon_sum_heatmaps(poly_hsum: np.ndarray, dist_hsum: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, matrix, title in [
        (axes[0], poly_hsum, "Polymer horizon-sum gain"),
        (axes[1], dist_hsum, "Distillation horizon-sum gain"),
    ]:
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Input 1", "Input 2"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Output 1", "Output 2"])
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = FIG_DIR / "distillation_transfer_horizon_sum_heatmaps.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_scalar_sensitivity(poly_scalar_rows, dist_scalar_rows):
    labels = ["alpha", "B_col_1", "B_col_2"]
    poly_sg = as_float_map(poly_scalar_rows, "coordinate_label", "S_G")
    dist_sg = as_float_map(dist_scalar_rows, "coordinate_label", "S_G")
    poly_sr = as_float_map(poly_scalar_rows, "coordinate_label", "S_rho")
    dist_sr = as_float_map(dist_scalar_rows, "coordinate_label", "S_rho")

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, poly_vals, dist_vals, title in [
        (axes[0], [poly_sr[k] for k in labels], [dist_sr[k] for k in labels], r"Scalar sensitivity $S_{\rho}$"),
        (axes[1], [poly_sg[k] for k in labels], [dist_sg[k] for k in labels], r"Scalar sensitivity $S_G$"),
    ]:
        ax.bar(x - width / 2, poly_vals, width, label="Polymer")
        ax.bar(x + width / 2, dist_vals, width, label="Distillation")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    out_path = FIG_DIR / "distillation_transfer_scalar_sensitivity.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_caps_and_structured_sensitivity(dist_scalar_bounds, dist_struct_rows):
    labels = [row["coordinate_label"] for row in dist_scalar_bounds]
    current_low = np.array([float(row["current_low"]) for row in dist_scalar_bounds], dtype=float)
    current_high = np.array([float(row["current_high"]) for row in dist_scalar_bounds], dtype=float)
    suggested_low = np.array([float(row["suggested_low"]) for row in dist_scalar_bounds], dtype=float)
    suggested_high = np.array([float(row["suggested_high"]) for row in dist_scalar_bounds], dtype=float)

    struct_labels = [row["coordinate_label"] for row in dist_struct_rows]
    struct_sg = np.array([float(row["S_G"]) for row in dist_struct_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    y = np.arange(len(labels))
    axes[0].hlines(y, current_low, current_high, color="0.75", linewidth=6, label="Current range")
    axes[0].hlines(y, suggested_low, suggested_high, color="#1f77b4", linewidth=3, label="Suggested Step 2 range")
    axes[0].scatter(np.ones_like(y), y, color="black", s=20, zorder=3, label="Nominal = 1")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Distillation scalar Step 2 ranges")
    axes[0].set_xlabel("Multiplier")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", alpha=0.25)

    order = np.argsort(struct_sg)[::-1]
    axes[1].bar(np.arange(len(struct_labels)), struct_sg[order], color="#d62728")
    axes[1].set_xticks(np.arange(len(struct_labels)))
    axes[1].set_xticklabels([struct_labels[idx] for idx in order], rotation=30, ha="right")
    axes[1].set_title(r"Distillation structured sensitivity $S_G$")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path = FIG_DIR / "distillation_transfer_caps_and_structured_sensitivity.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    poly_A, poly_B, poly_C, poly_A_aug, poly_B_aug, poly_C_aug = load_system(POLYMER_DATA)
    dist_A, dist_B, dist_C, dist_A_aug, dist_B_aug, dist_C_aug = load_system(DISTILLATION_DATA)

    poly_horizon = 9
    dist_horizon = 6

    poly_markov = markov_stack(poly_A_aug, poly_B_aug, poly_C_aug, poly_horizon)
    dist_markov = markov_stack(dist_A_aug, dist_B_aug, dist_C_aug, dist_horizon)
    poly_hsum = horizon_sum(poly_A_aug, poly_B_aug, poly_C_aug, poly_horizon)
    dist_hsum = horizon_sum(dist_A_aug, dist_B_aug, dist_C_aug, dist_horizon)

    poly_svals = np.linalg.svd(poly_hsum, compute_uv=False)
    dist_svals = np.linalg.svd(dist_hsum, compute_uv=False)

    poly_metrics = {
        "rho_A_phys": spectral_radius(poly_A),
        "markov_fro": float(np.linalg.norm(poly_markov, ord="fro")),
        "horizon_sum_condition": float(poly_svals[0] / poly_svals[-1]),
        "input2_input1_authority_ratio": float(np.sum(np.abs(poly_hsum), axis=0)[1] / np.sum(np.abs(poly_hsum), axis=0)[0]),
    }
    dist_metrics = {
        "rho_A_phys": spectral_radius(dist_A),
        "markov_fro": float(np.linalg.norm(dist_markov, ord="fro")),
        "horizon_sum_condition": float(dist_svals[0] / dist_svals[-1]),
        "input2_input1_authority_ratio": float(np.sum(np.abs(dist_hsum), axis=0)[1] / np.sum(np.abs(dist_hsum), axis=0)[0]),
    }

    poly_scalar_dir = latest_matching_dir(POLYMER_SENS_DIR, "polymer_matrix_td3_disturb_")
    dist_scalar_dir = latest_matching_dir(DISTILLATION_SENS_DIR, "distillation_matrix_td3_disturb_fluctuation_")
    dist_struct_dir = latest_matching_dir(DISTILLATION_SENS_DIR, "distillation_structured_matrix_td3_disturb_fluctuation_")

    poly_scalar_rows = read_csv_rows(poly_scalar_dir / "sensitivity_by_coordinate.csv")
    dist_scalar_rows = read_csv_rows(dist_scalar_dir / "sensitivity_by_coordinate.csv")
    dist_scalar_bounds = read_csv_rows(dist_scalar_dir / "suggested_bounds.csv")
    dist_struct_rows = read_csv_rows(dist_struct_dir / "sensitivity_by_coordinate.csv")

    save_model_metrics(poly_metrics, dist_metrics)
    plot_local_model_metrics(poly_metrics, dist_metrics)
    plot_horizon_sum_heatmaps(poly_hsum, dist_hsum)
    plot_scalar_sensitivity(poly_scalar_rows, dist_scalar_rows)
    plot_caps_and_structured_sensitivity(dist_scalar_bounds, dist_struct_rows)


if __name__ == "__main__":
    main()
