from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.structured_model_update import build_band_scaled_model, build_block_scaled_model, validate_positive_bounds


def spectral_radius(A_phys):
    """Return the spectral radius of a physical-state transition matrix."""
    A_phys = np.asarray(A_phys, dtype=float)
    if A_phys.ndim != 2 or A_phys.shape[0] != A_phys.shape[1]:
        raise ValueError("A_phys must be a square 2D array.")
    eigvals = np.linalg.eigvals(A_phys)
    if eigvals.size == 0:
        return 0.0
    return float(np.max(np.abs(eigvals)))


def build_markov_matrix(A, B, C, predict_h):
    """Stack finite-horizon Markov blocks C A^k B for k=0..predict_h-1."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    predict_h = int(predict_h)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array.")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B must be 2D and share A's row dimension.")
    if C.ndim != 2 or C.shape[1] != A.shape[0]:
        raise ValueError("C must be 2D and share A's state dimension.")
    if predict_h <= 0:
        raise ValueError("predict_h must be positive.")

    blocks = []
    A_power = np.eye(A.shape[0], dtype=float)
    for _ in range(predict_h):
        blocks.append(C @ A_power @ B)
        A_power = A_power @ A
    return np.vstack(blocks)


def run_scalar_matrix_sensitivity(
    A_aug,
    B_aug,
    C_aug,
    low_bounds,
    high_bounds,
    predict_h,
    n_outputs=None,
    epsilon_log=0.02,
    n_random_samples=2_000,
    seed=42,
    rho_target=0.995,
    gain_threshold=0.25,
    system_name=None,
    method_family="matrix",
):
    """Run the offline rho and finite-horizon gain diagnostic for scalar multipliers."""
    A_aug = np.asarray(A_aug, dtype=float)
    B_aug = np.asarray(B_aug, dtype=float)
    C_aug = np.asarray(C_aug, dtype=float)
    low_bounds, high_bounds = validate_positive_bounds(low_bounds, high_bounds)
    n_outputs = _resolve_n_outputs(A_aug, C_aug, n_outputs)
    n_inputs = int(B_aug.shape[1])
    expected_dim = 1 + n_inputs
    if low_bounds.size != expected_dim or high_bounds.size != expected_dim:
        raise ValueError(f"Scalar matrix diagnostic expects {expected_dim} bounds.")

    labels = ["alpha"] + [f"B_col_{idx + 1}" for idx in range(n_inputs)]
    coord_types = ["A_scalar"] + ["B_col"] * n_inputs

    def build_candidate(theta):
        theta = np.asarray(theta, dtype=float)
        n_phys = A_aug.shape[0] - n_outputs
        A_new = A_aug.copy()
        B_new = B_aug.copy()
        A_new[:n_phys, :n_phys] = A_aug[:n_phys, :n_phys] * float(theta[0])
        B_new[:n_phys, :] = B_aug[:n_phys, :] * theta[1:].reshape(1, -1)
        return A_new, B_new, A_new[:n_phys, :n_phys]

    return _run_sensitivity_core(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        labels=labels,
        coord_types=coord_types,
        build_candidate=build_candidate,
        predict_h=predict_h,
        epsilon_log=epsilon_log,
        n_random_samples=n_random_samples,
        seed=seed,
        rho_target=rho_target,
        gain_threshold=gain_threshold,
        system_name=system_name,
        method_family=method_family,
    )


def run_structured_matrix_sensitivity(
    A_aug,
    B_aug,
    C_aug,
    structured_spec,
    predict_h,
    epsilon_log=0.02,
    n_random_samples=2_000,
    seed=42,
    rho_target=0.995,
    gain_threshold=0.25,
    system_name=None,
    method_family="structured_matrix",
):
    """Run the offline rho and finite-horizon gain diagnostic for structured multipliers."""
    A_aug = np.asarray(A_aug, dtype=float)
    B_aug = np.asarray(B_aug, dtype=float)
    C_aug = np.asarray(C_aug, dtype=float)
    labels = list(structured_spec["action_labels"])
    low_bounds, high_bounds = validate_positive_bounds(
        structured_spec["low_bounds"],
        structured_spec["high_bounds"],
    )
    n_outputs = int(structured_spec["n_outputs"])
    a_dim = int(structured_spec["a_dim"])
    update_family = str(structured_spec["update_family"]).strip().lower()

    coord_types = [_structured_coordinate_type(label) for label in labels]

    def build_candidate(theta):
        theta = np.asarray(theta, dtype=float)
        theta_a = theta[:a_dim]
        theta_b = theta[a_dim:]
        if update_family == "block":
            model = build_block_scaled_model(
                A_aug,
                B_aug,
                n_outputs,
                structured_spec.get("block_cfg"),
                theta_a,
                theta_b,
            )
        elif update_family == "band":
            model = build_band_scaled_model(
                A_aug,
                B_aug,
                n_outputs,
                structured_spec.get("band_cfg"),
                theta_a,
                theta_b,
            )
        else:
            raise ValueError("structured_spec update_family must be 'block' or 'band'.")
        return model["A_aug"], model["B_aug"], model["A_phys"]

    return _run_sensitivity_core(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        labels=labels,
        coord_types=coord_types,
        build_candidate=build_candidate,
        predict_h=predict_h,
        epsilon_log=epsilon_log,
        n_random_samples=n_random_samples,
        seed=seed,
        rho_target=rho_target,
        gain_threshold=gain_threshold,
        system_name=system_name,
        method_family=method_family,
        structured_update_family=update_family,
    )


def save_multiplier_sensitivity_outputs(result, output_dir, make_plots=True):
    """Save diagnostic tables and optional plots. The result object is not mutated."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "sensitivity_by_coordinate": output_dir / "sensitivity_by_coordinate.csv",
        "candidate_scan_summary": output_dir / "candidate_scan_summary.csv",
        "suggested_bounds": output_dir / "suggested_bounds.csv",
        "metadata": output_dir / "metadata.json",
    }
    _write_dict_rows(paths["sensitivity_by_coordinate"], result["sensitivity_by_coordinate"])
    _write_dict_rows(paths["candidate_scan_summary"], [result["candidate_scan_summary"]])
    _write_dict_rows(paths["suggested_bounds"], result["suggested_bounds"])
    paths["metadata"].write_text(json.dumps(result["metadata"], indent=2), encoding="utf-8")

    plot_paths = {}
    if make_plots:
        plot_paths = _save_plots(result, output_dir)
    paths.update(plot_paths)
    return {key: str(value) for key, value in paths.items()}


def timestamped_sensitivity_output_dir(base_results_dir, system_name, method_family, agent_kind, run_mode):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "_".join(
        str(part).strip().lower().replace(" ", "_")
        for part in (system_name, method_family, agent_kind, run_mode, timestamp)
        if part is not None and str(part).strip()
    )
    return Path(base_results_dir) / "offline_multiplier_sensitivity" / slug


def format_multiplier_sensitivity_summary(result, max_rows=12):
    """Return a compact text summary suitable for notebook printing."""
    summary = result["candidate_scan_summary"]
    rows = sorted(
        result["sensitivity_by_coordinate"],
        key=lambda row: max(float(row["S_rho"]), float(row["S_G"])),
        reverse=True,
    )[: int(max_rows)]

    lines = [
        "Offline multiplier sensitivity diagnostic",
        f"  system: {result['metadata'].get('system_name') or 'unknown'}",
        f"  method: {result['metadata'].get('method_family')}",
        f"  nominal rho: {float(summary['nominal_rho']):.6g}",
        f"  nominal gain drift: {float(summary['nominal_gain_ratio']):.6g}",
        f"  sampled candidates: {int(summary['sampled_count'])}",
        f"  unstable fraction: {float(summary['unstable_frac']):.4f}",
        f"  near-unit fraction: {float(summary['near_unit_frac']):.4f}",
        f"  gain ratio p95/max: {float(summary['gain_ratio_p95']):.4f} / {float(summary['gain_ratio_max']):.4f}",
        "  most sensitive coordinates:",
    ]
    for row in rows:
        lines.append(
            "    "
            f"{row['coordinate_label']} ({row['coordinate_type']}): "
            f"S_rho={float(row['S_rho']):.4g}, S_G={float(row['S_G']):.4g}, "
            f"rho-={float(row['rho_minus']):.6g}, rho+={float(row['rho_plus']):.6g}, "
            f"G-={float(row['gain_ratio_minus']):.4g}, G+={float(row['gain_ratio_plus']):.4g}"
        )
    return "\n".join(lines)


def _run_sensitivity_core(
    *,
    A_aug,
    B_aug,
    C_aug,
    low_bounds,
    high_bounds,
    labels,
    coord_types,
    build_candidate,
    predict_h,
    epsilon_log,
    n_random_samples,
    seed,
    rho_target,
    gain_threshold,
    system_name,
    method_family,
    structured_update_family=None,
):
    low_bounds, high_bounds = validate_positive_bounds(low_bounds, high_bounds)
    theta_nom = np.ones_like(low_bounds, dtype=float)
    if np.any(theta_nom < low_bounds) or np.any(theta_nom > high_bounds):
        raise ValueError("Nominal multiplier value 1.0 must lie inside every diagnostic bound.")
    epsilon_log = float(epsilon_log)
    n_random_samples = int(n_random_samples)
    rho_target = float(rho_target)
    gain_threshold = float(gain_threshold)
    if epsilon_log <= 0.0:
        raise ValueError("epsilon_log must be positive.")
    if n_random_samples < 0:
        raise ValueError("n_random_samples must be non-negative.")
    if len(labels) != low_bounds.size or len(coord_types) != low_bounds.size:
        raise ValueError("labels, coord_types, and bounds must have matching lengths.")

    A_nom, B_nom, A_phys_nom = build_candidate(theta_nom)
    G_nom = build_markov_matrix(A_nom, B_nom, C_aug, predict_h)
    rho_nom = spectral_radius(A_phys_nom)
    gain_nominal_ratio = _gain_ratio(G_nom, G_nom)

    sensitivity_rows = []
    for idx, (label, coord_type) in enumerate(zip(labels, coord_types)):
        theta_plus = theta_nom.copy()
        theta_minus = theta_nom.copy()
        plus_value = min(float(high_bounds[idx]), float(np.exp(epsilon_log)))
        minus_value = max(float(low_bounds[idx]), float(np.exp(-epsilon_log)))
        theta_plus[idx] = plus_value
        theta_minus[idx] = minus_value

        A_plus, B_plus, A_phys_plus = build_candidate(theta_plus)
        A_minus, B_minus, A_phys_minus = build_candidate(theta_minus)
        G_plus = build_markov_matrix(A_plus, B_plus, C_aug, predict_h)
        G_minus = build_markov_matrix(A_minus, B_minus, C_aug, predict_h)
        rho_plus = spectral_radius(A_phys_plus)
        rho_minus = spectral_radius(A_phys_minus)
        gain_plus = _gain_ratio(G_plus, G_nom)
        gain_minus = _gain_ratio(G_minus, G_nom)
        denom = max(abs(np.log(plus_value) - np.log(minus_value)), np.finfo(float).eps)
        s_rho = abs(rho_plus - rho_minus) / denom
        s_g = _relative_difference(G_plus, G_minus) / denom
        sensitivity_rows.append(
            {
                "coordinate_label": label,
                "coordinate_type": coord_type,
                "S_rho": float(s_rho),
                "S_G": float(s_g),
                "rho_minus": float(rho_minus),
                "rho_plus": float(rho_plus),
                "gain_ratio_minus": float(gain_minus),
                "gain_ratio_plus": float(gain_plus),
                "log_minus": float(np.log(minus_value)),
                "log_plus": float(np.log(plus_value)),
            }
        )

    sample_rows = _random_candidate_scan(
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        build_candidate=build_candidate,
        C_aug=C_aug,
        predict_h=predict_h,
        G_nom=G_nom,
        n_random_samples=n_random_samples,
        seed=seed,
    )
    candidate_summary = _summarize_candidate_scan(
        sample_rows=sample_rows,
        rho_target=rho_target,
        rho_nom=rho_nom,
        gain_nominal_ratio=gain_nominal_ratio,
        sensitivity_rows=sensitivity_rows,
    )
    suggested_bounds = _suggest_bounds(
        labels=labels,
        coord_types=coord_types,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        sensitivity_rows=sensitivity_rows,
        rho_nom=rho_nom,
        rho_target=rho_target,
        gain_threshold=gain_threshold,
    )
    metadata = {
        "system_name": system_name,
        "method_family": method_family,
        "structured_update_family": structured_update_family,
        "epsilon_log": epsilon_log,
        "n_random_samples": n_random_samples,
        "seed": int(seed),
        "rho_target": rho_target,
        "gain_threshold": gain_threshold,
        "apply_suggested_caps": False,
        "predict_h": int(predict_h),
        "coordinate_labels": list(labels),
    }
    return {
        "metadata": metadata,
        "sensitivity_by_coordinate": sensitivity_rows,
        "candidate_scan_summary": candidate_summary,
        "candidate_samples": sample_rows,
        "suggested_bounds": suggested_bounds,
    }


def _resolve_n_outputs(A_aug, C_aug, n_outputs):
    if n_outputs is None:
        n_outputs = C_aug.shape[0]
    n_outputs = int(n_outputs)
    if n_outputs <= 0 or n_outputs >= A_aug.shape[0]:
        raise ValueError("n_outputs must define a non-empty disturbance block.")
    return n_outputs


def _structured_coordinate_type(label):
    if str(label).startswith("A_block"):
        return "A_block"
    if str(label).startswith("A_off"):
        return "A_off"
    if str(label).startswith("A_band"):
        return "A_band"
    if str(label).startswith("B_col"):
        return "B_col"
    return "unknown"


def _relative_difference(left, right):
    denom = float(np.linalg.norm(right, ord="fro"))
    if denom <= np.finfo(float).eps:
        return float(np.linalg.norm(left - right, ord="fro"))
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def _gain_ratio(G_candidate, G_nom):
    return _relative_difference(G_candidate, G_nom)


def _random_candidate_scan(*, low_bounds, high_bounds, build_candidate, C_aug, predict_h, G_nom, n_random_samples, seed):
    if n_random_samples <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    log_low = np.log(low_bounds)
    log_high = np.log(high_bounds)
    theta_samples = np.exp(rng.uniform(log_low, log_high, size=(n_random_samples, low_bounds.size)))
    rows = []
    for idx, theta in enumerate(theta_samples):
        A_candidate, B_candidate, A_phys_candidate = build_candidate(theta)
        G_candidate = build_markov_matrix(A_candidate, B_candidate, C_aug, predict_h)
        rows.append(
            {
                "sample_index": int(idx),
                "rho": float(spectral_radius(A_phys_candidate)),
                "gain_ratio": float(_gain_ratio(G_candidate, G_nom)),
                "theta": theta.astype(float).tolist(),
            }
        )
    return rows


def _summarize_candidate_scan(*, sample_rows, rho_target, rho_nom, gain_nominal_ratio, sensitivity_rows):
    rho_values = np.asarray([row["rho"] for row in sample_rows], dtype=float)
    gain_values = np.asarray([row["gain_ratio"] for row in sample_rows], dtype=float)
    if rho_values.size == 0:
        rho_values = np.asarray([rho_nom], dtype=float)
        gain_values = np.asarray([gain_nominal_ratio], dtype=float)

    worst_rho = max(sensitivity_rows, key=lambda row: float(row["S_rho"])) if sensitivity_rows else {}
    worst_gain = max(sensitivity_rows, key=lambda row: float(row["S_G"])) if sensitivity_rows else {}
    return {
        "sampled_count": int(len(sample_rows)),
        "nominal_rho": float(rho_nom),
        "nominal_gain_ratio": float(gain_nominal_ratio),
        "unstable_frac": float(np.mean(rho_values >= 1.0)),
        "near_unit_frac": float(np.mean(rho_values >= float(rho_target))),
        "rho_p50": float(np.quantile(rho_values, 0.50)),
        "rho_p90": float(np.quantile(rho_values, 0.90)),
        "rho_p95": float(np.quantile(rho_values, 0.95)),
        "rho_max": float(np.max(rho_values)),
        "gain_ratio_p50": float(np.quantile(gain_values, 0.50)),
        "gain_ratio_p90": float(np.quantile(gain_values, 0.90)),
        "gain_ratio_p95": float(np.quantile(gain_values, 0.95)),
        "gain_ratio_max": float(np.max(gain_values)),
        "worst_rho_coordinate": worst_rho.get("coordinate_label", ""),
        "worst_gain_coordinate": worst_gain.get("coordinate_label", ""),
    }


def _suggest_bounds(
    *,
    labels,
    coord_types,
    low_bounds,
    high_bounds,
    sensitivity_rows,
    rho_nom,
    rho_target,
    gain_threshold,
):
    rows = []
    sensitivity_by_label = {row["coordinate_label"]: row for row in sensitivity_rows}
    rho_margin = max(0.0, float(rho_target) - float(rho_nom))
    eps = np.finfo(float).eps
    for idx, (label, coord_type) in enumerate(zip(labels, coord_types)):
        row = sensitivity_by_label[label]
        user_low_log = abs(float(np.log(low_bounds[idx])))
        user_high_log = abs(float(np.log(high_bounds[idx])))
        s_rho = max(float(row["S_rho"]), 0.0)
        s_g = max(float(row["S_G"]), 0.0)
        rho_limit = np.inf if coord_type == "B_col" else rho_margin / max(s_rho, eps)
        gain_limit = float(gain_threshold) / max(s_g, eps)

        low_step, low_reason = _minimum_limited_step(
            user_step=user_low_log,
            rho_step=rho_limit,
            gain_step=gain_limit,
        )
        high_step, high_reason = _minimum_limited_step(
            user_step=user_high_log,
            rho_step=rho_limit,
            gain_step=gain_limit,
        )
        rows.append(
            {
                "coordinate_label": label,
                "coordinate_type": coord_type,
                "current_low": float(low_bounds[idx]),
                "current_high": float(high_bounds[idx]),
                "suggested_low": float(np.exp(-low_step)),
                "suggested_high": float(np.exp(high_step)),
                "low_reason": low_reason,
                "high_reason": high_reason,
                "reason": _combine_reasons(low_reason, high_reason),
                "apply_suggested_caps": False,
            }
        )
    return rows


def _minimum_limited_step(*, user_step, rho_step, gain_step):
    options = [
        ("user-bound-limited", float(user_step)),
        ("rho-limited", float(rho_step)),
        ("gain-limited", float(gain_step)),
    ]
    finite_options = [(name, value) for name, value in options if np.isfinite(value)]
    if not finite_options:
        return float(user_step), "user-bound-limited"
    reason, step = min(finite_options, key=lambda item: item[1])
    return max(0.0, float(step)), reason


def _combine_reasons(low_reason, high_reason):
    if low_reason == high_reason:
        return low_reason
    return f"{low_reason};{high_reason}"


def _write_dict_rows(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(result, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    output_dir = Path(output_dir)
    paths = {}
    sensitivity_rows = result["sensitivity_by_coordinate"]
    sample_rows = result["candidate_samples"]
    bounds_rows = result["suggested_bounds"]

    labels = [row["coordinate_label"] for row in sensitivity_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.5))
    width = 0.38
    ax.bar(x - width / 2, [row["S_rho"] for row in sensitivity_rows], width=width, label="S_rho")
    ax.bar(x + width / 2, [row["S_G"] for row in sensitivity_rows], width=width, label="S_G")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("local log-space sensitivity")
    ax.legend()
    fig.tight_layout()
    paths["sensitivity_bar_plot"] = output_dir / "coordinate_sensitivity_bar.png"
    fig.savefig(paths["sensitivity_bar_plot"], dpi=160)
    plt.close(fig)

    if sample_rows:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.scatter(
            [row["rho"] for row in sample_rows],
            [row["gain_ratio"] for row in sample_rows],
            s=12,
            alpha=0.45,
        )
        ax.axvline(float(result["metadata"]["rho_target"]), color="tab:red", linestyle="--", linewidth=1.0)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_xlabel("spectral radius")
        ax.set_ylabel("finite-horizon gain ratio")
        fig.tight_layout()
        paths["rho_gain_scatter_plot"] = output_dir / "sampled_rho_vs_gain_ratio.png"
        fig.savefig(paths["rho_gain_scatter_plot"], dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    ax.plot(x, [row["current_low"] for row in bounds_rows], marker="o", label="current low")
    ax.plot(x, [row["current_high"] for row in bounds_rows], marker="o", label="current high")
    ax.plot(x, [row["suggested_low"] for row in bounds_rows], marker="x", linestyle="--", label="suggested low")
    ax.plot(x, [row["suggested_high"] for row in bounds_rows], marker="x", linestyle="--", label="suggested high")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("multiplier bound")
    ax.legend()
    fig.tight_layout()
    paths["suggested_bounds_plot"] = output_dir / "suggested_cap_comparison.png"
    fig.savefig(paths["suggested_bounds_plot"], dpi=160)
    plt.close(fig)
    return paths


__all__ = [
    "build_markov_matrix",
    "format_multiplier_sensitivity_summary",
    "run_scalar_matrix_sensitivity",
    "run_structured_matrix_sensitivity",
    "save_multiplier_sensitivity_outputs",
    "spectral_radius",
    "timestamped_sensitivity_output_dir",
]
